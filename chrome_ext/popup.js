console.log("Popup loaded");

function generateCard(gemini, omdb) {
  if (!gemini) return `<div class="card">No data</div>`;
  return `
    <div class="card">
      ${omdb?.Poster && omdb.Poster !== "N/A" ? `<img class="poster" src="${omdb.Poster}" />` : ""}
      <div class="title">${omdb?.Title || gemini.movie || 'Unknown'}</div>
      <div class="subtitle">${omdb?.Year || gemini.year || ""}</div>
      <div class="section-title">Scene Description</div>
      <div>${gemini.scene_description || gemini.raw || "No description available."}</div>
      <div class="section-title">Confidence</div>
      <div>${Math.round((gemini.confidence || 0) * 100)}%</div>
    </div>
  `;
}

function renderResult(movieResult) {
  const loading = document.getElementById("loading");
  const result = document.getElementById("result");

  loading.style.display = "none";
  result.innerHTML = generateCard(movieResult.gemini || movieResult.best_guess, movieResult.omdb || {});
}

function showStatus(text){
  const loading = document.getElementById("loading");
  loading.style.display = "block";
  loading.innerText = text;
}

async function identifyFromLink() {
  const url = document.getElementById("video-url").value.trim();
  if (!url) return alert("Please paste a video link");

  showStatus("Downloading video & extracting frames…");

  try {
    const res = await fetch("http://127.0.0.1:8000/identify/link", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ video_url: url })
    });

    if (!res.ok) throw new Error(await res.text());

    const data = await res.json();

    chrome.storage.local.set({ movieResult: data });
    renderResult(data);

    // Trigger toast on webpage
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (!tabs || tabs.length === 0) return;
      chrome.tabs.sendMessage(tabs[0].id, { 
        action: "showToast", 
        text: `Found: ${data.best_guess.movie}` 
      });
    });

  } catch (err) {
    console.error(err);
    showStatus("Error: " + err.message);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  // 🧹 Clear previous result on popup open
  chrome.storage.local.remove("movieResult");

  // Reset UI
  document.getElementById("loading").innerText = 
    "Right-click an image → Identify movie scene";

  document.getElementById("result").innerHTML = "";

  // Add event handler
  document.getElementById("identify-link").addEventListener("click", identifyFromLink);
});
