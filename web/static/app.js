"use strict";
const $ = (s, r = document) => r.querySelector(s);
const ARMS = ["base", "mappo"];
let ready = false, busy = false;

function syncButton() {
  const b = $("#run");
  b.disabled = busy || !ready;
  b.textContent = busy ? "Running…" : ready ? "Send" : b.textContent;
}

async function poll() {
  try {
    const s = await (await fetch("/api/status")).json();
    ready = s.phase === "ready";
    if (!ready) {
      $("#run").textContent = s.phase === "downloading" ? "Downloading model…" : s.phase === "error" ? "Model failed to load" : "Loading model…";
    }
    syncButton();
    setTimeout(poll, ready ? 8000 : 1500);
  } catch { setTimeout(poll, 2500); }
}

function show(arm, text, cls = "") {
  const pre = $(`.lane[data-arm="${arm}"] pre`);
  pre.className = cls;
  pre.textContent = text;
}

async function ask(arm, prompt) {
  show(arm, "…", "wait");
  try {
    const r = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, arm }),
    });
    const d = await r.json();
    if (r.status === 409) return show(arm, "Not available yet (adapter not trained).", "wait");
    if (!r.ok) return show(arm, d.detail || `Error ${r.status}`, "wait");
    show(arm, d.response, d.leaked ? "leak" : "");
  } catch { show(arm, "Could not reach the server.", "wait"); }
}

$("#form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const prompt = $("#prompt").value.trim();
  if (!prompt || busy || !ready) return;
  busy = true; syncButton();
  $("#lanes").hidden = false;
  ARMS.forEach(a => show(a, "", ""));
  for (const arm of ARMS) await ask(arm, prompt);
  busy = false; syncButton();
});

// Enter sends, Shift+Enter makes a new line
$("#prompt").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("#form").requestSubmit(); }
});

poll();
