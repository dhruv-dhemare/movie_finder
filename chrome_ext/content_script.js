// content_script.js
// Example: listen for a message from service worker to highlight the clicked image
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
if (msg.type === 'HIGHLIGHT_SRC' && msg.src) {
const imgs = [...document.images].filter(i=>i.src===msg.src);
imgs.forEach(img => {
img.style.outline = '4px solid #f39c12';
});
}
});