// --------------------------------------------------
// CONFIG
// --------------------------------------------------
const BACKEND_URL = "http://127.0.0.1:8000/identify";

console.log("Service Worker Loaded");

// --------------------------------------------------
// CREATE CONTEXT MENU
// --------------------------------------------------
chrome.runtime.onInstalled.addListener(() => {
  console.log("Extension installed → creating context menu");

  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: "identify-image",
      title: "Identify Movie Scene",
      contexts: ["image"]
    });

    console.log("Context menu created ✔");
  });
});

// --------------------------------------------------
// IMAGE → BASE64 CONVERSION
// --------------------------------------------------
async function fetchImageAsBase64(url) {
  const resp = await fetch(url);
  const blob = await resp.blob();

  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result.split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

// --------------------------------------------------
// RIGHT-CLICK HANDLER
// --------------------------------------------------
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "identify-image") return;

  console.log("Right-click triggered:", info.srcUrl);

  try {
    // --------------------------------------------------
    // STEP 1: Inject toast.js and show LOADER immediately
    // --------------------------------------------------
    chrome.scripting.executeScript(
      {
        target: { tabId: tab.id },
        files: ["toast.js"]
      },
      () => {
        chrome.tabs.sendMessage(tab.id, { action: "loader" }, () => {
          console.log("Loader toast triggered");
        });
      }
    );

    // --------------------------------------------------
    // STEP 2: Convert image → Base64
    // --------------------------------------------------
    const base64 = await fetchImageAsBase64(info.srcUrl);

    // --------------------------------------------------
    // STEP 3: Call backend
    // --------------------------------------------------
    const response = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: base64 })
    });

    const data = await response.json();
    console.log("Backend result:", data);

    const movie = data.best_guess || data.gemini || {};

    // --------------------------------------------------
    // STEP 4: Replace loader → Show final toast
    // --------------------------------------------------
    chrome.tabs.sendMessage(
      tab.id,
      { action: "showToast", movie },
      () => {
        if (chrome.runtime.lastError)
          console.warn("Toast delivery error:", chrome.runtime.lastError);
      }
    );

  } catch (err) {
    console.error("Identify error:", err);

    // Optional: show error toast
    chrome.tabs.sendMessage(tab.id, {
      action: "showToast",
      movie: {
        movie: "Error",
        scene_description: err.message,
        actors: [],
        characters: [],
        location_in_movie: "",
        confidence: 0
      }
    });
  }
});
