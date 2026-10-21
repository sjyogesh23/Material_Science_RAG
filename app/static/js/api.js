// Thin API helpers + a manual SSE parser for fetch() streams.

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

export async function api(path, { method = "GET", body, signal } = {}) {
  const init = { method, signal, headers: { Accept: "application/json" } };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, init);
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError("The NewRAG server is not reachable. Start it with python run.py.", 0);
  }
  const text = await res.text();
  let data = null;
  if (text) { try { data = JSON.parse(text); } catch { data = text; } }
  if (!res.ok) {
    const detail = data && typeof data === "object" ? (data.detail || data.message || data.error) : data;
    throw new ApiError(typeof detail === "string" && detail ? detail : `${method} ${path} failed (${res.status})`, res.status);
  }
  return data;
}

/**
 * POST a JSON body and parse the text/event-stream response.
 * Frames are blank-line separated; each has `event:` and one or more `data:` lines (JSON).
 * Calls onEvent(name, data) for each frame. Resolves when the stream ends.
 */
export async function streamSSE(path, body, { signal, onEvent }) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    let msg = `Request failed (${res.status})`;
    try { const j = await res.json(); msg = j.detail || j.message || msg; } catch { /* ignore */ }
    throw new ApiError(typeof msg === "string" ? msg : JSON.stringify(msg), res.status);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  const flush = (frame) => {
    let event = "message";
    const data = [];
    for (const raw of frame.split("\n")) {
      const line = raw.endsWith("\r") ? raw.slice(0, -1) : raw;
      if (!line || line.startsWith(":")) continue;
      const i = line.indexOf(":");
      const field = i === -1 ? line : line.slice(0, i);
      let value = i === -1 ? "" : line.slice(i + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      if (field === "event") event = value;
      else if (field === "data") data.push(value);
    }
    if (!data.length) return;
    const payload = data.join("\n");
    let parsed;
    try { parsed = JSON.parse(payload); } catch { parsed = payload; }
    onEvent(event, parsed);
  };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      flush(frame);
    }
  }
  buf += decoder.decode();
  if (buf.trim()) flush(buf);
}

export function toast(message, { error = false, ms = 2600 } = {}) {
  const el = document.getElementById("toast");
  if (!el) return;
  el.textContent = message;
  el.classList.toggle("is-error", error);
  el.classList.add("is-on");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove("is-on"), ms);
}

export function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

export const fmtInt = (n) => (n == null || Number.isNaN(+n) ? "–" : Math.round(+n).toLocaleString("en-US"));
