# =========================================================
#  server.py - MERGED FINAL VERSION (20 FRAMES -> pick 10 diverse)
# =========================================================

import os
import json
import base64
import shutil
import tempfile
import re
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import yt_dlp
import cv2
import ffmpeg
from dotenv import load_dotenv
import google.generativeai as genai

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import numpy as np
import math

# ---------------------------------------------------------
# LOAD ENVIRONMENT
# ---------------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Please set GOOGLE_API_KEY in .env")

# NOTE: if you later want a model with higher quota, change this line.
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


# ---------------------------------------------------------
# ADD FFMPEG PATH FOR WINDOWS
# ---------------------------------------------------------
FFMPEG_PATH = r"C:\Users\dhruv dhemare\AppData\Local\ffmpegio\ffmpeg-downloader\ffmpeg\bin"
if os.path.exists(FFMPEG_PATH):
    os.environ["PATH"] += os.pathsep + FFMPEG_PATH


# ---------------------------------------------------------
# REDUCE OPENCV LOGGING
# ---------------------------------------------------------
try:
    from cv2 import utils
    utils.logging.setLogLevel(utils.logging.ERROR)
except:
    pass


# ---------------------------------------------------------
# FASTAPI SETUP
# ---------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------------
class ImageRequest(BaseModel):
    image_base64: str
    filename: str = "image.jpg"

class LinkRequest(BaseModel):
    video_url: str


# ---------------------------------------------------------
# JSON CLEANER
# ---------------------------------------------------------
def clean_json_from_gemini(text: str) -> Dict[str, Any]:
    text = text.replace("```json", "").replace("```", "").strip()
    text = text.replace("“", "\"").replace("”", "\"")
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    try:
        return json.loads(text)
    except:
        return {"_raw_text": text}


# ---------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------
FULL_PROMPT = """
Identify the movie and describe the scene with full detail.

Return ONLY JSON:

{
  "movie": "Movie Title",
  "year": 2010,
  "characters": ["Character 1", "Character 2"],
  "actors": ["Actor 1", "Actor 2"],
  "scene_description": "2-4 detailed sentences describing what is happening.",
  "location_in_movie": "Where this scene appears in the story.",
  "confidence": 0.92
}

Rules:
- No markdown, only JSON.
- All fields MUST be filled (even if guessed).
"""

ENRICH_PROMPT = """
The previous prediction is incomplete. Fill ALL missing fields.
Even if unsure, guess based on movie knowledge.

Return ONLY JSON:

{
  "movie": "Movie Title",
  "year": 2010,
  "characters": ["Character 1", "Character 2"],
  "actors": ["Actor 1", "Actor 2"],
  "scene_description": "2-4 detailed sentences.",
  "location_in_movie": "Where scene appears.",
  "confidence": 0.92
}
"""


# ---------------------------------------------------------
# NEEDS ENRICHMENT ?
# ---------------------------------------------------------
def needs_enrichment(data: Dict[str, Any]) -> bool:
    required = [
        "movie", "year", "characters", "actors",
        "scene_description", "location_in_movie", "confidence"
    ]

    if "_raw_text" in data:
        return True

    for key in required:
        if key not in data:
            return True
        if data[key] in [None, "", [], {}, "N/A", 0]:
            return True

    return False


# ---------------------------------------------------------
# SAFE GEMINI CALL (retries if API returns retry_delay)
# ---------------------------------------------------------
def safe_gemini_call(image_bytes: bytes, prompt: str = FULL_PROMPT) -> Dict[str, Any]:
    """
    Calls Gemini and if a rate-limit / retry_delay is received,
    parse the delay and sleep, then retry.
    """
    while True:
        try:
            response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": image_bytes}])
            return clean_json_from_gemini(response.text)
        except Exception as e:
            # Look for "retry in Xs" pattern in exception text (best-effort)
            msg = str(e)
            match = re.search(r"retry in\s*([0-9]+(?:\.[0-9]+)?)s", msg)
            if match:
                wait_sec = float(match.group(1))
                # Safety margin
                wait_sec = max(wait_sec, 5.0)
                print(f"⚠ Gemini rate-limit detected. Waiting {wait_sec:.1f}s before retrying...")
                time.sleep(wait_sec + 1.0)
                continue
            # If not a retryable error, re-raise
            raise


# ---------------------------------------------------------
# GEMINI WRAPPERS (for compatibility)
# ---------------------------------------------------------
def call_gemini_on_bytes(image_bytes: bytes) -> Dict[str, Any]:
    return safe_gemini_call(image_bytes, FULL_PROMPT)


def enrich_gemini_result(image_bytes: bytes, movie_name: str) -> Dict[str, Any]:
    prompt_with_movie = ENRICH_PROMPT + f"\nMovie: {movie_name}"
    res = safe_gemini_call(image_bytes, prompt_with_movie)

    if needs_enrichment(res):
        res2 = safe_gemini_call(image_bytes, "FILL EVERYTHING. RETURN ONLY JSON.\n" + f"Movie: {movie_name}")
        return res2
    return res


# ---------------------------------------------------------
# FRAME → JPEG
# ---------------------------------------------------------
def frame_to_jpeg_bytes(frame) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return buf.tobytes()


# ---------------------------------------------------------
# DOWNLOAD VIDEO
# ---------------------------------------------------------
def download_video(url: str, dest: str) -> str:
    outpath = os.path.join(dest, "video.%(ext)s")

    opts = {
        "format": "mp4/best",
        "outtmpl": outpath,
        "quiet": True,
        "no_warnings": True
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = ydl.prepare_filename(info)

        if not os.path.exists(filepath):
            for f in os.listdir(dest):
                if f.startswith("video."):
                    return os.path.join(dest, f)
            raise FileNotFoundError("Video not downloaded.")

    return filepath


# ---------------------------------------------------------
# REENCODE VIDEO
# ---------------------------------------------------------
def reencode_video(input_path, output_path):
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, vcodec="libx264", crf=18,
                    preset="ultrafast", pix_fmt="yuv420p")
            .overwrite_output()
            .run(quiet=True)
        )
        return output_path

    except Exception as e:
        print("FFmpeg error:", e)
        raise HTTPException(500, "FFmpeg failed to repair video")


# ---------------------------------------------------------
# EXTRACT 20 FRAMES
# ---------------------------------------------------------
def extract_frames(video_path: str, dest_dir: str) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    positions = [i/20 for i in range(1, 20)]  # 20 points

    paths = []
    for idx, pos in enumerate(positions):
        frame_no = int(total * pos)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            continue

        bytes_jpg = frame_to_jpeg_bytes(frame)
        path = os.path.join(dest_dir, f"frame_{idx}.jpg")
        with open(path, "wb") as f:
            f.write(bytes_jpg)
        paths.append(path)

    cap.release()
    return paths


# ---------------------------------------------------------
# SELECT REPRESENTATIVE FRAMES (pick k from n using histogram diversity)
# ---------------------------------------------------------
def select_representative_frames(frame_paths: List[str], k: int = 10) -> List[str]:
    """
    Load each frame, compute a color histogram in HSV, and greedily pick k frames
    that maximize diversity (farthest from already selected frames).
    """
    imgs = []
    for p in frame_paths:
        img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            # fallback to cv2.imread if needed
            img = cv2.imread(p)
        if img is None:
            continue
        imgs.append((p, img))

    if len(imgs) <= k:
        return [p for p, _ in imgs]

    # compute histograms (HSV 3-channel concatenated)
    hists = []
    for _, img in imgs:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(h, h)
        hists.append(h.flatten())

    # greedy k-center-like selection
    selected_idx = [0]  # start with first frame
    remaining = set(range(len(hists))) - set(selected_idx)

    def hist_distance(a, b):
        # use Bhattacharyya distance (lower = similar), convert to dissimilarity
        return cv2.compareHist(a.astype('float32'), b.astype('float32'), cv2.HISTCMP_BHATTACHARYYA)

    while len(selected_idx) < k and remaining:
        best_candidate = None
        best_score = -math.inf

        for r in list(remaining):
            # compute distance to nearest selected
            dists = [hist_distance(hists[r], hists[s]) for s in selected_idx]
            # want the *most* dissimilar (largest distance). Bhattacharyya returns smaller for similar,
            # so convert by negative.
            min_dist = min(dists) if dists else 0.0
            # prefer larger min_dist
            score = min_dist
            if score > best_score:
                best_score = score
                best_candidate = r

        if best_candidate is None:
            break

        selected_idx.append(best_candidate)
        remaining.remove(best_candidate)

    # return corresponding paths
    return [imgs[i][0] for i in selected_idx]


# ---------------------------------------------------------
# PROCESS FRAME
# ---------------------------------------------------------
def process_frame(fpath: str) -> Dict[str, Any]:
    with open(fpath, "rb") as f:
        img = f.read()

    res = call_gemini_on_bytes(img)

    if needs_enrichment(res):
        res = enrich_gemini_result(img, res.get("movie", "Unknown"))

    res["_frame"] = os.path.basename(fpath)
    return res


# ---------------------------------------------------------
# PROCESS SELECTED FRAMES (up to 10) - parallel safe
# ---------------------------------------------------------
def process_selected_frames(frame_paths: List[str]) -> List[Dict[str, Any]]:
    """
    We process only up to 10 representative frames in parallel (free-tier safe).
    safe_gemini_call handles retry/backoff if necessary.
    """
    results = []
    with ThreadPoolExecutor(max_workers=min(2, len(frame_paths))) as exe:
        futures = [exe.submit(process_frame, f) for f in frame_paths]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results

# def process_frames_sequential(frame_paths: List[str]) -> List[Dict[str, Any]]:
#     """
#     Process ALL frames (20) sequentially.
#     Free-tier safe: 6.5 second wait per request.
#     Highest accuracy: no frames discarded or clustered.
#     """
#     results = []
#     for i, fpath in enumerate(frame_paths):
#         print(f"🔍 Processing frame {i+1}/{len(frame_paths)}: {fpath}")

#         with open(fpath, "rb") as f:
#             img = f.read()

#         # safest method - one by one
#         res = safe_gemini_call(img)

#         if needs_enrichment(res):
#             res = enrich_gemini_result(img, res.get("movie", "Unknown"))

#         res["_frame"] = os.path.basename(fpath)
#         results.append(res)

#         # respect Gemini free-tier (10 req/min = 1 every 6 sec)
#         if i < len(frame_paths) - 1:
#             time.sleep(0)

#     return results


# ---------------------------------------------------------
# AGGREGATE
# ---------------------------------------------------------
def aggregate_predictions(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    data = {}

    for p in preds:
        movie = p.get("movie", "Unknown")
        conf = float(p.get("confidence", 0))

        if movie not in data:
            data[movie] = {"sum": 0, "count": 0, "examples": []}

        data[movie]["sum"] += conf
        data[movie]["count"] += 1
        data[movie]["examples"].append(p)

    best = None
    best_score = -999

    for movie, stats in data.items():
        avg = stats["sum"] / stats["count"]
        score = avg * 0.7 + stats["count"] * 0.3

        if score > best_score:
            best_score = score
            best = {
                "movie": movie,
                "avg": avg,
                "count": stats["count"],
                "examples": stats["examples"],
            }

    return best


# =========================================================
# ROUTES
# =========================================================

@app.post("/identify")
async def identify_image(req: ImageRequest):
    img = base64.b64decode(req.image_base64)

    result = call_gemini_on_bytes(img)
    if needs_enrichment(result):
        result = enrich_gemini_result(img, result.get("movie", "Unknown"))

    return {"gemini": result}


@app.post("/identify/link")
async def identify_from_link(req: LinkRequest):

    temp = tempfile.mkdtemp(prefix="mf_")

    try:
        original = download_video(req.video_url, temp)
        repaired = os.path.join(temp, "fixed.mp4")
        video_path = reencode_video(original, repaired)

        frames = extract_frames(video_path, temp)
        if not frames:
            raise HTTPException(500, "Could not extract frames.")

        # Select top-10 representative frames from the 20 we extracted
        selected = select_representative_frames(frames, k=10)

        # Process only the selected frames (<=10 requests)
        preds = process_selected_frames(selected)
        # preds = process_frames_sequential(frames)

        best_group = aggregate_predictions(preds)

        # If aggregate empty (rare), fallback to raw preds
        if not best_group:
            return {"best_guess": {}, "raw_predictions": preds}

        first = best_group["examples"][0]

        clean_best = {
            "movie": first.get("movie", best_group["movie"]),
            "year": first.get("year"),
            "characters": first.get("characters", []),
            "actors": first.get("actors", []),
            "scene_description": first.get("scene_description", ""),
            "location_in_movie": first.get("location_in_movie", ""),
            "confidence": first.get("confidence", best_group["avg"]),
            "frames_used": best_group["count"]
        }

        return {
            "best_guess": clean_best,
            "raw_predictions": preds,
            "selected_frames": selected
        }

    finally:
        shutil.rmtree(temp, ignore_errors=True)


@app.get("/")
def root():
    return {"status": "server running"}
