/*
 * GreenBuddy front-end logic (plain JavaScript, no framework).
 *
 * The page never decides answers itself: every question is sent to the
 * Flask API (POST /chat) and the reply is shown as plain text.
 */

// The page is served by Flask, so the API is on the same address.
const API_BASE = window.GREENBUDDY_API_BASE || "";
const REQUEST_TIMEOUT_MS = 15000;
const WELCOME = "Hi! I'm GreenBuddy 🌱 How can I help your plants thrive today?";

const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendButton = document.getElementById("send-btn");
const clearButton = document.getElementById("clear-chat");
const statusLabel = document.getElementById("server-status");
const quickChips = document.querySelectorAll(".quick-questions .chip");

let isWaiting = false;
let requestCounter = 0;       // lets us ignore a reply that arrives after "Clear chat"
let activeController = null;

/* ---------- Talking to the API ---------- */

class ChatError extends Error {}

async function askBackend(question) {
  const controller = new AbortController();
  activeController = controller;
  const timer = setTimeout(() => controller.abort("timeout"), REQUEST_TIMEOUT_MS);
  let response;
  try {
    response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, debug: true }),
      signal: controller.signal,
    });
  } catch (err) {
    if (controller.signal.aborted && controller.signal.reason === "timeout") {
      throw new ChatError("GreenBuddy took too long to reply. Please try again.");
    }
    throw new ChatError("Can't reach the GreenBuddy server. Is the backend running (python3 app.py)?");
  } finally {
    clearTimeout(timer);
  }

  let body;
  try {
    body = await response.json();
  } catch {
    throw new ChatError("The server sent an unexpected reply. Please try again.");
  }
  if (!response.ok) {
    throw new ChatError(body?.error?.message || `The server returned an error (${response.status}).`);
  }
  if (typeof body.answer !== "string" || typeof body.fallback !== "boolean") {
    throw new ChatError("The server sent an incomplete reply. Please try again.");
  }
  return body;
}

async function checkServer() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    const body = await response.json();
    setStatus(body.nlp_ready ? "online" : "offline",
      body.nlp_ready ? `Online · ${body.entries} answers loaded` : "NLP engine not ready");
  } catch {
    setStatus("offline", "Offline: start the backend");
  }
}

function setStatus(state, text) {
  statusLabel.className = `status ${state}`;
  statusLabel.textContent = text;
}

/* ---------- Building chat messages (always textContent, never innerHTML) ---------- */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Fixed markup written by us (not server data), so innerHTML is safe here.
const LEAF_ICON = '<svg viewBox="0 0 32 32"><path d="M16 27V15" stroke="#f5f1e6" stroke-width="2" stroke-linecap="round"/>'
  + '<path d="M16 18C10 18 7 14 7 8c6 0 9 4 9 10Z" fill="#c9c6a3"/><path d="M16 15c0-6 3-10 9-10 0 6-3 10-9 10Z" fill="#efe6d2"/></svg>';

function makeAvatar(role) {
  const avatar = el("div", "avatar");
  avatar.setAttribute("aria-hidden", "true");
  if (role === "user") avatar.textContent = "You";
  else avatar.innerHTML = LEAF_ICON;
  return avatar;
}

function addMessage(role, text) {
  const row = el("div", `message ${role}`);
  const avatar = makeAvatar(role);
  const bubble = el("div", "bubble");
  const label = el("span", "sr-only", role === "user" ? "You said: " : "GreenBuddy said: ");
  bubble.append(label, el("span", "text", text));
  row.append(avatar, bubble);
  chatLog.append(row);
  scrollChatToBottom();
  return bubble;
}

function addBotAnswer(data) {
  const bubble = addMessage("bot", "");
  const textNode = bubble.querySelector(".text");

  if (data.confidence === "medium") {
    bubble.insertBefore(el("p", "closest-note", "Here's the closest topic I found:"), textNode);
  }
  textNode.textContent = data.answer;

  if (!data.fallback) {
    const meta = el("div", "meta");
    if (data.topic_label) meta.append(el("span", "tag", data.topic_label));
    if (data.source === "floradb") meta.append(el("span", "tag dataset", "From FloraDB dataset"));
    bubble.append(meta);
  }

  if (Array.isArray(data.related_questions) && data.related_questions.length) {
    const related = el("div", "related");
    related.append(el("span", "sr-only", "Related questions:"));
    data.related_questions.forEach((q) => {
      const chip = el("button", "chip", q);
      chip.type = "button";
      chip.addEventListener("click", () => sendQuestion(q));
      related.append(chip);
    });
    bubble.append(related);
  }

  bubble.append(buildNlpDetails(data));
}

// "How did I find this?" panel: shows the real NLP result for the viva demo.
function buildNlpDetails(data) {
  const details = el("details", "nlp-details");
  details.append(el("summary", "", "How did I find this?"));
  const list = el("dl");
  const add = (term, value) => list.append(el("dt", "", term), el("dd", "", value));
  const pipeline = data.debug?.pipeline;
  if (pipeline) add("Processed words", pipeline.stems.join(" ") || "(none)");
  add(data.fallback ? "Nearest entry (not used)" : "Best match", data.debug?.top_matches?.[0]?.question || "none");
  add("Cosine similarity", data.similarity.toFixed(3));
  add("Threshold", data.debug ? String(data.debug.threshold) : "-");
  add("Result", data.fallback ? `Fallback (${data.fallback_reason.replaceAll("_", " ")})` : `Answered (${data.confidence} confidence)`);
  details.append(list);
  return details;
}

function addErrorMessage(message, question) {
  const bubble = addMessage("bot error", message);
  const retry = el("button", "retry-btn", "Retry");
  retry.type = "button";
  retry.addEventListener("click", () => {
    bubble.parentElement.remove();
    sendQuestion(question, { isRetry: true });
  });
  bubble.append(el("br"), retry);
}

function showTyping() {
  const row = el("div", "message bot typing-row");
  const avatar = makeAvatar("bot");
  const dots = el("div", "bubble typing");
  dots.append(el("span"), el("span"), el("span"), el("span", "sr-only", "GreenBuddy is typing"));
  row.append(avatar, dots);
  chatLog.append(row);
  scrollChatToBottom();
  return row;
}

function scrollChatToBottom() {
  chatLog.scrollTop = chatLog.scrollHeight; // scrolls only the chat box, not the page
}

function setWaiting(waiting) {
  isWaiting = waiting;
  sendButton.disabled = waiting;
  chatInput.disabled = waiting;
  document.querySelectorAll(".chip").forEach((chip) => { chip.disabled = waiting; });
  chatLog.setAttribute("aria-busy", String(waiting));
}

/* ---------- Main flow ---------- */

async function sendQuestion(rawQuestion, { isRetry = false } = {}) {
  const question = rawQuestion.trim();
  if (!question || isWaiting) return;

  if (!isRetry) addMessage("user", question);
  chatInput.value = "";
  setWaiting(true);
  const typing = showTyping();
  const myRequest = ++requestCounter;

  try {
    const data = await askBackend(question);
    if (myRequest !== requestCounter) return; // chat was cleared meanwhile
    typing.remove();
    addBotAnswer(data);
  } catch (err) {
    if (myRequest !== requestCounter) return;
    typing.remove();
    addErrorMessage(err instanceof ChatError ? err.message : "Something went wrong. Please try again.", question);
    checkServer();
  } finally {
    if (myRequest === requestCounter) {
      setWaiting(false);
      chatInput.focus();
    }
  }
}

function resetChat() {
  requestCounter++;
  if (activeController) activeController.abort();
  chatLog.replaceChildren();
  setWaiting(false);
  addMessage("bot", WELCOME);
  chatInput.focus();
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuestion(chatInput.value);
});
quickChips.forEach((chip) => chip.addEventListener("click", () => sendQuestion(chip.textContent)));
clearButton.addEventListener("click", resetChat);

/* ---------- Navigation ---------- */

const menuToggle = document.getElementById("menu-toggle");
const siteNav = document.getElementById("site-nav");
menuToggle.addEventListener("click", () => {
  const open = siteNav.classList.toggle("open");
  menuToggle.setAttribute("aria-expanded", String(open));
});
siteNav.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => {
  siteNav.classList.remove("open");
  menuToggle.setAttribute("aria-expanded", "false");
}));
document.querySelectorAll(".js-start-chat").forEach((link) => link.addEventListener("click", () => {
  setTimeout(() => chatInput.focus({ preventScroll: true }), 400);
}));

addMessage("bot", WELCOME);
checkServer();
