# # =========================================================
# #  server.py - MERGED FINAL VERSION (20 FRAMES -> pick 10 diverse)
# # =========================================================

# import os
# import json
# import base64
# import shutil
# import tempfile
# import re
# from typing import List, Dict, Any

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware

# import yt_dlp
# import cv2
# import ffmpeg
# from dotenv import load_dotenv
# import google.generativeai as genai

# from concurrent.futures import ThreadPoolExecutor, as_completed
# import time
# import numpy as np
# import math

# # ---------------------------------------------------------
# # LOAD ENVIRONMENT
# # ---------------------------------------------------------
# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise RuntimeError("Please set GOOGLE_API_KEY in .env")

# # NOTE: if you later want a model with higher quota, change this line.
# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel("gemini-2.5-flash")


# # ---------------------------------------------------------
# # ADD FFMPEG PATH FOR WINDOWS
# # ---------------------------------------------------------
# FFMPEG_PATH = r"C:\Users\dhruv dhemare\AppData\Local\ffmpegio\ffmpeg-downloader\ffmpeg\bin"
# if os.path.exists(FFMPEG_PATH):
#     os.environ["PATH"] += os.pathsep + FFMPEG_PATH


# # ---------------------------------------------------------
# # REDUCE OPENCV LOGGING
# # ---------------------------------------------------------
# try:
#     from cv2 import utils
#     utils.logging.setLogLevel(utils.logging.ERROR)
# except:
#     pass


# # ---------------------------------------------------------
# # FASTAPI SETUP
# # ---------------------------------------------------------
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # ---------------------------------------------------------
# # REQUEST MODELS
# # ---------------------------------------------------------
# class ImageRequest(BaseModel):
#     image_base64: str
#     filename: str = "image.jpg"

# class LinkRequest(BaseModel):
#     video_url: str


# # ---------------------------------------------------------
# # JSON CLEANER
# # ---------------------------------------------------------
# def clean_json_from_gemini(text: str) -> Dict[str, Any]:
#     text = text.replace("```json", "").replace("```", "").strip()
#     text = text.replace("“", "\"").replace("”", "\"")
#     text = text.replace("‘", "'").replace("’", "'")
#     text = re.sub(r",\s*}", "}", text)
#     text = re.sub(r",\s*]", "]", text)

#     try:
#         return json.loads(text)
#     except:
#         return {"_raw_text": text}


# # ---------------------------------------------------------
# # PROMPTS
# # ---------------------------------------------------------
# FULL_PROMPT = """
# Identify the movie and describe the scene with full detail.

# Return ONLY JSON:

# {
#   "movie": "Movie Title",
#   "year": 2010,
#   "characters": ["Character 1", "Character 2"],
#   "actors": ["Actor 1", "Actor 2"],
#   "scene_description": "2-4 detailed sentences describing what is happening.",
#   "location_in_movie": "Where this scene appears in the story.",
#   "confidence": 0.92
# }

# Rules:
# - No markdown, only JSON.
# - All fields MUST be filled (even if guessed).
# """

# ENRICH_PROMPT = """
# The previous prediction is incomplete. Fill ALL missing fields.
# Even if unsure, guess based on movie knowledge.

# Return ONLY JSON:

# {
#   "movie": "Movie Title",
#   "year": 2010,
#   "characters": ["Character 1", "Character 2"],
#   "actors": ["Actor 1", "Actor 2"],
#   "scene_description": "2-4 detailed sentences.",
#   "location_in_movie": "Where scene appears.",
#   "confidence": 0.92
# }
# """


# # ---------------------------------------------------------
# # NEEDS ENRICHMENT ?
# # ---------------------------------------------------------
# def needs_enrichment(data: Dict[str, Any]) -> bool:
#     required = [
#         "movie", "year", "characters", "actors",
#         "scene_description", "location_in_movie", "confidence"
#     ]

#     if "_raw_text" in data:
#         return True

#     for key in required:
#         if key not in data:
#             return True
#         if data[key] in [None, "", [], {}, "N/A", 0]:
#             return True

#     return False


# # ---------------------------------------------------------
# # SAFE GEMINI CALL (retries if API returns retry_delay)
# # ---------------------------------------------------------
# def safe_gemini_call(image_bytes: bytes, prompt: str = FULL_PROMPT) -> Dict[str, Any]:
#     """
#     Calls Gemini and if a rate-limit / retry_delay is received,
#     parse the delay and sleep, then retry.
#     """
#     while True:
#         try:
#             response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": image_bytes}])
#             return clean_json_from_gemini(response.text)
#         except Exception as e:
#             # Look for "retry in Xs" pattern in exception text (best-effort)
#             msg = str(e)
#             match = re.search(r"retry in\s*([0-9]+(?:\.[0-9]+)?)s", msg)
#             if match:
#                 wait_sec = float(match.group(1))
#                 # Safety margin
#                 wait_sec = max(wait_sec, 5.0)
#                 print(f"⚠ Gemini rate-limit detected. Waiting {wait_sec:.1f}s before retrying...")
#                 time.sleep(wait_sec + 1.0)
#                 continue
#             # If not a retryable error, re-raise
#             raise


# # ---------------------------------------------------------
# # GEMINI WRAPPERS (for compatibility)
# # ---------------------------------------------------------
# def call_gemini_on_bytes(image_bytes: bytes) -> Dict[str, Any]:
#     return safe_gemini_call(image_bytes, FULL_PROMPT)


# def enrich_gemini_result(image_bytes: bytes, movie_name: str) -> Dict[str, Any]:
#     prompt_with_movie = ENRICH_PROMPT + f"\nMovie: {movie_name}"
#     res = safe_gemini_call(image_bytes, prompt_with_movie)

#     if needs_enrichment(res):
#         res2 = safe_gemini_call(image_bytes, "FILL EVERYTHING. RETURN ONLY JSON.\n" + f"Movie: {movie_name}")
#         return res2
#     return res


# # ---------------------------------------------------------
# # FRAME → JPEG
# # ---------------------------------------------------------
# def frame_to_jpeg_bytes(frame) -> bytes:
#     ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
#     if not ok:
#         raise RuntimeError("Failed to encode JPEG")
#     return buf.tobytes()


# # ---------------------------------------------------------
# # DOWNLOAD VIDEO
# # ---------------------------------------------------------
# def download_video(url: str, dest: str) -> str:
#     outpath = os.path.join(dest, "video.%(ext)s")

#     opts = {
#         "format": "mp4/best",
#         "outtmpl": outpath,
#         "quiet": True,
#         "no_warnings": True
#     }

#     with yt_dlp.YoutubeDL(opts) as ydl:
#         info = ydl.extract_info(url, download=True)
#         filepath = ydl.prepare_filename(info)

#         if not os.path.exists(filepath):
#             for f in os.listdir(dest):
#                 if f.startswith("video."):
#                     return os.path.join(dest, f)
#             raise FileNotFoundError("Video not downloaded.")

#     return filepath


# # ---------------------------------------------------------
# # REENCODE VIDEO
# # ---------------------------------------------------------
# def reencode_video(input_path, output_path):
#     try:
#         (
#             ffmpeg
#             .input(input_path)
#             .output(output_path, vcodec="libx264", crf=18,
#                     preset="ultrafast", pix_fmt="yuv420p")
#             .overwrite_output()
#             .run(quiet=True)
#         )
#         return output_path

#     except Exception as e:
#         print("FFmpeg error:", e)
#         raise HTTPException(500, "FFmpeg failed to repair video")


# # ---------------------------------------------------------
# # EXTRACT 20 FRAMES
# # ---------------------------------------------------------
# def extract_frames(video_path: str, dest_dir: str) -> List[str]:
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise RuntimeError("Could not open video.")

#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     positions = [i/20 for i in range(1, 20)]  # 20 points

#     paths = []
#     for idx, pos in enumerate(positions):
#         frame_no = int(total * pos)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
#         ok, frame = cap.read()
#         if not ok:
#             continue

#         bytes_jpg = frame_to_jpeg_bytes(frame)
#         path = os.path.join(dest_dir, f"frame_{idx}.jpg")
#         with open(path, "wb") as f:
#             f.write(bytes_jpg)
#         paths.append(path)

#     cap.release()
#     return paths


# # ---------------------------------------------------------
# # SELECT REPRESENTATIVE FRAMES (pick k from n using histogram diversity)
# # ---------------------------------------------------------
# def select_representative_frames(frame_paths: List[str], k: int = 10) -> List[str]:
#     """
#     Load each frame, compute a color histogram in HSV, and greedily pick k frames
#     that maximize diversity (farthest from already selected frames).
#     """
#     imgs = []
#     for p in frame_paths:
#         img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
#         if img is None:
#             # fallback to cv2.imread if needed
#             img = cv2.imread(p)
#         if img is None:
#             continue
#         imgs.append((p, img))

#     if len(imgs) <= k:
#         return [p for p, _ in imgs]

#     # compute histograms (HSV 3-channel concatenated)
#     hists = []
#     for _, img in imgs:
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
#         cv2.normalize(h, h)
#         hists.append(h.flatten())

#     # greedy k-center-like selection
#     selected_idx = [0]  # start with first frame
#     remaining = set(range(len(hists))) - set(selected_idx)

#     def hist_distance(a, b):
#         # use Bhattacharyya distance (lower = similar), convert to dissimilarity
#         return cv2.compareHist(a.astype('float32'), b.astype('float32'), cv2.HISTCMP_BHATTACHARYYA)

#     while len(selected_idx) < k and remaining:
#         best_candidate = None
#         best_score = -math.inf

#         for r in list(remaining):
#             # compute distance to nearest selected
#             dists = [hist_distance(hists[r], hists[s]) for s in selected_idx]
#             # want the *most* dissimilar (largest distance). Bhattacharyya returns smaller for similar,
#             # so convert by negative.
#             min_dist = min(dists) if dists else 0.0
#             # prefer larger min_dist
#             score = min_dist
#             if score > best_score:
#                 best_score = score
#                 best_candidate = r

#         if best_candidate is None:
#             break

#         selected_idx.append(best_candidate)
#         remaining.remove(best_candidate)

#     # return corresponding paths
#     return [imgs[i][0] for i in selected_idx]


# # ---------------------------------------------------------
# # PROCESS FRAME
# # ---------------------------------------------------------
# def process_frame(fpath: str) -> Dict[str, Any]:
#     with open(fpath, "rb") as f:
#         img = f.read()

#     res = call_gemini_on_bytes(img)

#     if needs_enrichment(res):
#         res = enrich_gemini_result(img, res.get("movie", "Unknown"))

#     res["_frame"] = os.path.basename(fpath)
#     return res


# # ---------------------------------------------------------
# # PROCESS SELECTED FRAMES (up to 10) - parallel safe
# # ---------------------------------------------------------
# def process_selected_frames(frame_paths: List[str]) -> List[Dict[str, Any]]:
#     """
#     We process only up to 10 representative frames in parallel (free-tier safe).
#     safe_gemini_call handles retry/backoff if necessary.
#     """
#     results = []
#     with ThreadPoolExecutor(max_workers=min(2, len(frame_paths))) as exe:
#         futures = [exe.submit(process_frame, f) for f in frame_paths]
#         for fut in as_completed(futures):
#             results.append(fut.result())
#     return results

# # def process_frames_sequential(frame_paths: List[str]) -> List[Dict[str, Any]]:
# #     """
# #     Process ALL frames (20) sequentially.
# #     Free-tier safe: 6.5 second wait per request.
# #     Highest accuracy: no frames discarded or clustered.
# #     """
# #     results = []
# #     for i, fpath in enumerate(frame_paths):
# #         print(f"🔍 Processing frame {i+1}/{len(frame_paths)}: {fpath}")

# #         with open(fpath, "rb") as f:
# #             img = f.read()

# #         # safest method - one by one
# #         res = safe_gemini_call(img)

# #         if needs_enrichment(res):
# #             res = enrich_gemini_result(img, res.get("movie", "Unknown"))

# #         res["_frame"] = os.path.basename(fpath)
# #         results.append(res)

# #         # respect Gemini free-tier (10 req/min = 1 every 6 sec)
# #         if i < len(frame_paths) - 1:
# #             time.sleep(0)

# #     return results


# # ---------------------------------------------------------
# # AGGREGATE
# # ---------------------------------------------------------
# def aggregate_predictions(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
#     data = {}

#     for p in preds:
#         movie = p.get("movie", "Unknown")
#         conf = float(p.get("confidence", 0))

#         if movie not in data:
#             data[movie] = {"sum": 0, "count": 0, "examples": []}

#         data[movie]["sum"] += conf
#         data[movie]["count"] += 1
#         data[movie]["examples"].append(p)

#     best = None
#     best_score = -999

#     for movie, stats in data.items():
#         avg = stats["sum"] / stats["count"]
#         score = avg * 0.7 + stats["count"] * 0.3

#         if score > best_score:
#             best_score = score
#             best = {
#                 "movie": movie,
#                 "avg": avg,
#                 "count": stats["count"],
#                 "examples": stats["examples"],
#             }

#     return best


# # =========================================================
# # ROUTES
# # =========================================================

# @app.post("/identify")
# async def identify_image(req: ImageRequest):
#     img = base64.b64decode(req.image_base64)

#     result = call_gemini_on_bytes(img)
#     if needs_enrichment(result):
#         result = enrich_gemini_result(img, result.get("movie", "Unknown"))

#     return {"gemini": result}


# @app.post("/identify/link")
# async def identify_from_link(req: LinkRequest):

#     temp = tempfile.mkdtemp(prefix="mf_")

#     try:
#         original = download_video(req.video_url, temp)
#         repaired = os.path.join(temp, "fixed.mp4")
#         video_path = reencode_video(original, repaired)

#         frames = extract_frames(video_path, temp)
#         if not frames:
#             raise HTTPException(500, "Could not extract frames.")

#         # Select top-10 representative frames from the 20 we extracted
#         selected = select_representative_frames(frames, k=10)

#         # Process only the selected frames (<=10 requests)
#         preds = process_selected_frames(selected)
#         # preds = process_frames_sequential(frames)

#         best_group = aggregate_predictions(preds)

#         # If aggregate empty (rare), fallback to raw preds
#         if not best_group:
#             return {"best_guess": {}, "raw_predictions": preds}

#         first = best_group["examples"][0]

#         clean_best = {
#             "movie": first.get("movie", best_group["movie"]),
#             "year": first.get("year"),
#             "characters": first.get("characters", []),
#             "actors": first.get("actors", []),
#             "scene_description": first.get("scene_description", ""),
#             "location_in_movie": first.get("location_in_movie", ""),
#             "confidence": first.get("confidence", best_group["avg"]),
#             "frames_used": best_group["count"]
#         }

#         return {
#             "best_guess": clean_best,
#             "raw_predictions": preds,
#             "selected_frames": selected
#         }

#     finally:
#         shutil.rmtree(temp, ignore_errors=True)


# @app.get("/")
# def root():
#     return {"status": "server running"}
































# server_high_accuracy_pipeline.py - High-accuracy movie-scene identification
# Implements scene-detection sampling, face-prioritized frame selection, multi-frame Gemini
# prompts, and robust fallbacks. Designed as a drop-in replacement for your previous server.py.

import os
import io
import json
import time
import math
import shutil
import base64
import tempfile
import threading
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import yt_dlp
import cv2
import numpy as np
import ffmpeg
from dotenv import load_dotenv

# PySceneDetect for scene boundary detection
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg

# Gemini client
import google.generativeai as genai

# ---------------------------------------------------------
# CONFIG & ENV
# ---------------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # switch to pro model via env for higher accuracy
if not GOOGLE_API_KEY:
    raise RuntimeError("Please set GOOGLE_API_KEY in your .env")

# configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

# FFmpeg path (optional). If you are on Windows and need this, set FFMPEG_PATH env var.
FFMPEG_PATH = os.getenv("FFMPEG_PATH")
if FFMPEG_PATH and os.path.exists(FFMPEG_PATH):
    os.environ["PATH"] += os.pathsep + FFMPEG_PATH

# Face detector files (OpenCV DNN 'res10' model) will be downloaded automatically if missing
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
FACE_PROTO = os.path.join(MODEL_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# fallback parameters
MAX_REP_FRAMES = 10
MIN_FACE_CONF = 0.5

# ---------------------------------------------------------
# FASTAPI
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
# UTIL: download face model if absent
# ---------------------------------------------------------
def ensure_face_model():
    """Ensure OpenCV DNN face detector files are present. If not, download them."""
    if os.path.exists(FACE_PROTO) and os.path.exists(FACE_MODEL):
        return

    import urllib.request

    proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

    print("Downloading face detector models (this may take a few seconds)...")
    urllib.request.urlretrieve(proto_url, FACE_PROTO)
    urllib.request.urlretrieve(model_url, FACE_MODEL)
    print("Face detector models saved to:", MODEL_DIR)


# ---------------------------------------------------------
# JSON cleaning helper (for any LLM output)
# ---------------------------------------------------------
def clean_json_from_gemini(text: str) -> Dict[str, Any]:
    text = text.replace("```json", "").replace("```", "").strip()
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace('\r\n', '\n')
    text = text.replace('\t', ' ')
    text = re_sub_json_cleanup(text)
    try:
        return json.loads(text)
    except Exception:
        # If strict parse fails, return raw for debugging
        return {"_raw_text": text}


def re_sub_json_cleanup(s: str) -> str:
    import re
    s = re.sub(r",\s*\}", "}", s)
    s = re.sub(r",\s*\]", "]", s)
    return s


# ---------------------------------------------------------
# VIDEO HELPERS
# ---------------------------------------------------------

def download_video(url: str, dest: str) -> str:
    outpath = os.path.join(dest, "video.%(ext)s")
    opts = {
        "format": "mp4/best",
        "outtmpl": outpath,
        "quiet": True,
        "no_warnings": True,
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


def reencode_video(input_path, output_path):
    try:
        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                vcodec="libx264",
                crf=18,
                preset="medium",
                pix_fmt="yuv420p"
            )
            .overwrite_output()
            .run(cmd=r"C:\Users\dhruv dhemare\AppData\Local\ffmpegio\ffmpeg-downloader\ffmpeg\bin\ffmpeg.exe")
        )
        return output_path

    except Exception as e:
        print("FFmpeg error:", e)
        raise HTTPException(500, f"FFmpeg failed to reencode video: {e}")


# ---------------------------------------------------------
# SCENE DETECTION: returns list of (start_frame, end_frame)
# ---------------------------------------------------------

def detect_scenes(video_path: str) -> List[Dict[str, int]]:
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())

    try:
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()

        if not scene_list:
            # fallback: return uniform segments
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            if total <= 0:
                return []
            step = max(1, total // 20)
            return [{"start": i, "end": min(i + step, total - 1)} for i in range(0, total, step)]

        # convert timecodes to frame indices
        scenes = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        for sc in scene_list:
            start_tc, end_tc = sc
            start_frame = int(start_tc.get_frames())
            end_frame = int(end_tc.get_frames())
            # clamp
            start_frame = max(0, min(total - 1, start_frame))
            end_frame = max(0, min(total - 1, end_frame))
            if end_frame <= start_frame:
                end_frame = min(total - 1, start_frame + 1)
            scenes.append({"start": start_frame, "end": end_frame})

        return scenes

    finally:
        try:
            video_manager.release()
        except Exception:
            pass


# ---------------------------------------------------------
# FRAME QUALITY METRICS: faces, sharpness, blur, brightness
# ---------------------------------------------------------

def compute_sharpness_gray(gray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def score_frame_by_face_and_sharpness(img) -> Dict[str, Any]:
    """Returns dict with face_count, max_face_area_ratio, sharpness, face_score composite."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = compute_sharpness_gray(gray)

    # face detection using DNN
    net = getattr(score_frame_by_face_and_sharpness, "net", None)
    if net is None:
        ensure_face_model()
        net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
        score_frame_by_face_and_sharpness.net = net

    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    face_count = 0
    max_area = 0
    for i in range(0, detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < MIN_FACE_CONF:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        area = max(0, x2 - x1) * max(0, y2 - y1)
        face_count += 1
        if area > max_area:
            max_area = area

    max_area_ratio = max_area / (w * h) if (w * h) > 0 else 0.0

    # composite score: prioritize presence and size of faces, then sharpness
    face_score = (face_count * 0.6) + (max_area_ratio * 3.0) + (min(sharpness / 1000.0, 1.0) * 0.4)

    return {
        "face_count": int(face_count),
        "max_area_ratio": float(max_area_ratio),
        "sharpness": float(sharpness),
        "face_score": float(face_score),
    }


# ---------------------------------------------------------
# SELECT REPRESENTATIVE FRAMES: scene-based + face-prioritized
# ---------------------------------------------------------

def extract_frame_at(video_path: str, frame_no: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_no}")
    return frame


def select_representative_frames_for_video(video_path: str, k: int = MAX_REP_FRAMES) -> List[str]:
    scenes = detect_scenes(video_path)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    candidates = []

    # For each scene choose the best frame by face_score (sample multiple frames inside the scene)
    for sc in scenes:
        start = sc["start"]
        end = sc["end"]
        if end <= start:
            end = start + 1
        # sample up to 5 frames inside the scene uniformly
        samples = min(5, max(1, end - start))
        sample_positions = [start + (end - start) * (i + 0.5) // samples for i in range(samples)]

        best_local = None
        best_score = -math.inf
        for sp in sample_positions:
            frame_no = int(max(0, min(total - 1, int(sp))))
            try:
                frame = extract_frame_at(video_path, frame_no)
            except Exception:
                continue
            metrics = score_frame_by_face_and_sharpness(frame)
            # also compute color histogram diversity metric for fallback
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            cand = {
                "frame_no": frame_no,
                "metrics": metrics,
                "hist": hist.flatten(),
            }
            if metrics["face_score"] > best_score:
                best_score = metrics["face_score"]
                best_local = cand

        if best_local:
            candidates.append(best_local)

    # If no scenes found or no face candidates, fallback to uniform sampling
    if len(candidates) == 0:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if total <= 0:
            return []
        positions = [int(total * i / 20) for i in range(1, 20)]
        for p in positions:
            try:
                frame = extract_frame_at(video_path, p)
            except Exception:
                continue
            metrics = score_frame_by_face_and_sharpness(frame)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            candidates.append({"frame_no": p, "metrics": metrics, "hist": hist.flatten()})

    # sort candidates by face_score desc, then sharpness
    candidates.sort(key=lambda x: (x["metrics"]["face_score"], x["metrics"]["sharpness"]), reverse=True)

    # take top k distinct frames
    selected = []
    used_frames = set()
    for c in candidates:
        fn = c["frame_no"]
        if fn in used_frames:
            continue
        selected.append(fn)
        used_frames.add(fn)
        if len(selected) >= k:
            break

    # write frames to temp files and return paths
    tempdir = tempfile.mkdtemp(prefix="mf_sel_")
    paths = []
    cap = cv2.VideoCapture(video_path)
    for fn in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ok, frame = cap.read()
        if not ok:
            continue
        p = os.path.join(tempdir, f"frame_{fn}.jpg")
        cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(p)
        paths.append(p)
    cap.release()

    return paths


# ---------------------------------------------------------
# GEMINI MULTI-FRAME CALL
# ---------------------------------------------------------

def gemini_multi_frame_identify(frame_paths: List[str]) -> Dict[str, Any]:
    """Send multiple frames together to Gemini and ask for a single JSON response."""
    
    prompt = (
        "You are a movie expert. You are given multiple image frames from the same short video clip. "
        "Identify the movie and describe the scene in detail. If unsure, give the best guess and a "
        "confidence from 0 to 1. Return ONLY a JSON object with the fields below. Fill all fields.\n\n"
        "{\n"
        "  \"movie\": \"Movie Title\",\n"
        "  \"year\": 0,\n"
        "  \"characters\": [\"Character 1\", \"Character 2\"],\n"
        "  \"actors\": [\"Actor 1\", \"Actor 2\"],\n"
        "  \"scene_description\": \"2-4 detailed sentences describing what is happening.\",\n"
        "  \"location_in_movie\": \"Where in the story this scene appears.\",\n"
        "  \"confidence\": 0.0\n"
        "}\n\n"
        "RULES:\n"
        "- No markdown, only JSON.\n"
        "- If uncertain, guess but put a low confidence."
    )

    # build request payload: prompt followed by image parts
    parts = [prompt]
    for p in frame_paths:
        with open(p, "rb") as f:
            parts.append({
                "mime_type": "image/jpeg",
                "data": f.read()
            })

    # call Gemini once with all frames
    resp = model.generate_content(parts)
    text = getattr(resp, "text", None) or str(resp)

    # try to parse JSON
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {"_raw_text": text}

    return parsed



# ---------------------------------------------------------
# ROUTES
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


@app.post("/identify")
async def identify_image(req: ImageRequest):
    img = base64.b64decode(req.image_base64)

    # Use the same strong prompt as your old FULL_PROMPT
    prompt = """
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

    # Call Gemini
    resp = model.generate_content([
        prompt,
        {"mime_type": "image/jpeg", "data": img}
    ])

    raw = getattr(resp, "text", None) or str(resp)

    # Parse JSON strictly
    result = clean_json_from_gemini(raw)

    # Enrich if needed (same logic as before)
    if needs_enrichment(result):
        enriched = model.generate_content([
            ENRICH_PROMPT,
            {"mime_type": "image/jpeg", "data": img}
        ])
        final_text = getattr(enriched, "text", None) or str(enriched)
        result = clean_json_from_gemini(final_text)

    return {"gemini": result}


@app.post("/identify/link")
async def identify_from_link(req: LinkRequest):
    temp = tempfile.mkdtemp(prefix="mf_")

    try:
        original = download_video(req.video_url, temp)
        repaired = os.path.join(temp, "fixed.mp4")
        video_path = reencode_video(original, repaired)

        # pick representative frames
        selected_frame_paths = select_representative_frames_for_video(video_path, k=MAX_REP_FRAMES)
        if not selected_frame_paths:
            raise HTTPException(500, "No frames selected from video.")

        # call Gemini with multiple frames at once
        gemini_result = gemini_multi_frame_identify(selected_frame_paths)

        # basic validation: ensure movie field exists, otherwise ask a short follow-up verification
        if isinstance(gemini_result, dict) and gemini_result.get("movie"):
            out = {
                "best_guess": gemini_result,
                "selected_frames": selected_frame_paths,
            }
            return out
        else:
            # fallback: ask Gemini to analyze again but give the selected frames and say "I couldn't parse your JSON, please reply in JSON only"
            verify_prompt = (
                "You previously returned a response that could not be parsed as JSON. "
                "You are given the same frames again. Please ONLY return a valid JSON object with the structure requested previously."
            )
            parts = [verify_prompt]
            for p in selected_frame_paths:
                with open(p, 'rb') as f:
                    parts.append({"mime_type": "image/jpeg", "data": f.read()})
            resp2 = model.generate_content(parts)
            text2 = getattr(resp2, 'text', None) or str(resp2)
            try:
                parsed2 = json.loads(text2)
            except Exception:
                parsed2 = {"_raw_text": text2}

            return {"best_guess": parsed2, "selected_frames": selected_frame_paths}

    finally:
        # DO NOT delete selected frames dir until caller has had a chance to fetch results in some systems;
        # we remove temp resources but keep selected frames list paths (they live inside a tempdir which we remove).
        shutil.rmtree(temp, ignore_errors=True)


@app.get("/")
def root():
    return {"status": "high-accuracy server running"}


# ---------------------------------------------------------
# USAGE NOTES (not executed):
# 1) Set environment variables in a .env file:
#    GOOGLE_API_KEY=your_key_here
#    GEMINI_MODEL=gemini-1.5-pro   # optional for higher accuracy
#    FFMPEG_PATH=C:\path\to\ffmpeg\bin   # optional on Windows
# 2) Install requirements:
#    pip install fastapi uvicorn python-dotenv yt-dlp opencv-python-headless scenedetect numpy google-generative-ai ffmpeg-python
# 3) Run:
#    uvicorn server_high_accuracy_pipeline:app --host 0.0.0.0 --port 8000
# ---------------------------------------------------------
