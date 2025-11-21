console.log("Toast content script LOADED!");

function showToast(movie) {
  console.log("Showing toast with movie:", movie);

  // Remove old toast
  const old = document.getElementById("movie-toast");
  if (old) old.remove();

  // Safe fallbacks
  const title = movie.movie || "Unknown Movie";
  const year = movie.year || "Unknown Year";
  const actors = movie.actors?.join(", ") || "Unknown Actors";
  const characters = movie.characters?.join(", ") || "Unknown Characters";
  const scene = movie.scene_description || "No description available.";
  const location = movie.location_in_movie || "Unknown location in movie";
  const confidence =
    movie.confidence != null
      ? Math.round(movie.confidence * 100) + "%"
      : "N/A";

  // Create toast
  const toast = document.createElement("div");
  toast.id = "movie-toast";

  toast.innerHTML = `
    <div class="toast-icon">🎬</div>

    <div class="toast-content">
      <div class="toast-title">${title} (${year})</div>

      <div class="toast-line"><strong>Actors:</strong> ${actors}</div>

      <div class="toast-line"><strong>Characters:</strong> ${characters}</div>

      <div class="toast-line"><strong>Scene:</strong> ${scene}</div>

      <div class="toast-line"><strong>Location:</strong> ${location}</div>

      <div class="toast-line"><strong>Confidence:</strong> ${confidence}</div>
    </div>
  `;

  // Netflix style
  Object.assign(toast.style, {
    position: "fixed",
    bottom: "-400px", // start hidden
    right: "20px",
    width: "360px",
    background: "linear-gradient(135deg, #E50914, #B20710)",
    color: "white",
    padding: "20px",
    borderRadius: "16px",
    fontFamily: "'Segoe UI', Roboto, sans-serif",
    fontSize: "14px",
    zIndex: 999999999,
    boxShadow: "0 15px 30px rgba(0,0,0,0.45)",
    opacity: 0,
    transition: "bottom 0.45s ease, opacity 0.4s ease",
    cursor: "pointer",
    lineHeight: "1.45"
  });

  // Add it
  document.body.appendChild(toast);

  // Slide up animation
  setTimeout(() => {
    toast.style.bottom = "20px";
    toast.style.opacity = "1";
  }, 80);

  // Hide after 7 sec
  setTimeout(() => {
    toast.style.bottom = "-400px";
    toast.style.opacity = "0";
  }, 7000);

  // Remove
  setTimeout(() => toast.remove(), 7600);

  // Click → open popup
  toast.addEventListener("click", () => {
    chrome.runtime.sendMessage({ action: "openPopup" });
  });
}

// Listen for SW messages
chrome.runtime.onMessage.addListener((msg) => {
  console.log("Toast.js received:", msg);

  if (msg.action === "showToast") {
    showToast(msg.movie);
  }
});
