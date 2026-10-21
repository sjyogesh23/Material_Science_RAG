// NewRAG chat page.
import { api, streamSSE, toast, escapeHtml, fmtInt } from "./api.js";
import { renderMarkdown, isOOC, OOC } from "./render.js";
import { buildPagesPanel, openSource, openImages, initCiteCard, sourcePages } from "./viewer.js";

const $ = (id) => document.getElementById(id);
const els = {
  sidebar: $("sidebar"), scrim: $("scrim"), open: $("sidebar-open"), close: $("sidebar-close"),
  sessions: $("sessions"), newChat: $("new-chat"), title: $("chat-title"),
  thread: $("thread"), messages: $("messages"), form: $("composer"), input: $("composer-input"),
  send: $("send-btn"), stage: $("stage"), stageLabel: $("stage-label"), banner: $("health-banner"),
  count: $("composer-count"), modes: $("mode-picker"), modeNote: $("mode-note"),
  menu: $("sess-menu"), memDlg: $("memory-dlg"), confirmDlg: $("confirm-dlg"),
};
const MAX_CHARS = 4000; // = MAX_MESSAGE_CHARS in app/api/main.py
const MODE_KEY = "newrag.mode";

// Used when GET /api/modes is unavailable (older backend): same ids as docs/FEATURES_V2.md §4.
const DEFAULT_MODES = [
  { id: "swift", name: "Swift", tagline: "Time over effect", description: "The fast 4B model with quick routing and fewer passages. Usually under 20 s." },
  { id: "deep", name: "Deep", tagline: "Effect over time", description: "A larger model that reasons first, reads more passages and checks its grounding. About 1–3 min per answer on this laptop." },
];

const state = {
  sessionId: new URLSearchParams(location.search).get("s"),
  sessions: [],
  controller: null,
  streaming: false,
  turn: 0,
  modes: DEFAULT_MODES,
  mode: readStoredMode() || "swift",
  activeLeaf: null,
  stage: null,
};

// article element → per-message state. Both roles carry the branch fields:
// { role, id, parentId, siblings: [ids], sibIndex, sibCount }
const msgData = new WeakMap();

const STAGES = {
  routing: "Reading the question…",
  retrieving: "Searching the textbook…",
  reranking: "Ranking passages…",
  thinking: "Reasoning (Deep mode)…",
  generating: "Writing the answer…",
  validating: "Checking citations…",
  compacting: "Updating session memory…", // after the answer: past the last tick, so all ticks show done
};
// tick position of each stage (thinking shares the "generating" tick)
const STAGE_ORDER = ["routing", "retrieving", "reranking", "generating", "validating", "compacting"];
const STAGE_TICK = { thinking: "generating" };

const EXAMPLES = [
  "What is the difference between an edge dislocation and a screw dislocation?",
  "Explain Bragg's law and what each term in it means.",
  "Tabulate the coordination number and packing factor of SC, BCC and FCC.",
  "Show me the figure of the materials tetrahedron.",
];

const ICON = {
  up: `<svg viewBox="0 0 16 16" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" aria-hidden="true"><path class="fill" d="M2.5 7h2.75v6.5H2.5z"/><path class="fill" d="M5.25 7l2.6-4.4c.95 0 1.65.85 1.45 1.75L8.9 6.4h3.5c.9 0 1.55.85 1.33 1.72l-1.05 4.2a1.5 1.5 0 0 1-1.45 1.18H5.25"/></svg>`,
  down: `<svg viewBox="0 0 16 16" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" aria-hidden="true"><g transform="matrix(1 0 0 -1 0 16)"><path class="fill" d="M2.5 7h2.75v6.5H2.5z"/><path class="fill" d="M5.25 7l2.6-4.4c.95 0 1.65.85 1.45 1.75L8.9 6.4h3.5c.9 0 1.55.85 1.33 1.72l-1.05 4.2a1.5 1.5 0 0 1-1.45 1.18H5.25"/></g></svg>`,
  regen: `<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M13.2 8.6A5.2 5.2 0 1 1 11.6 4"/><path d="M12.4 1.9v2.7H9.7"/></svg>`,
  edit: `<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" aria-hidden="true"><path d="M10.5 3l2.5 2.5L6 12.5H3.5V10z"/></svg>`,
  prev: `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" aria-hidden="true"><path d="M10 3.5L5.5 8l4.5 4.5"/></svg>`,
  next: `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" aria-hidden="true"><path d="M6 3.5L10.5 8 6 12.5"/></svg>`,
  kebab: `<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><circle cx="8" cy="3.5" r="1.35"/><circle cx="8" cy="8" r="1.35"/><circle cx="8" cy="12.5" r="1.35"/></svg>`,
};

let uid = 0;
const nextId = (p) => `${p}${(++uid).toString(36)}`;

/* ============ storage (best effort) ============ */
function readStoredMode() {
  try { return localStorage.getItem(MODE_KEY); } catch { return null; }
}
function storeMode(m) {
  try { localStorage.setItem(MODE_KEY, m); } catch { /* private mode / blocked storage */ }
}

/* ============ sessions sidebar ============ */

function dayBucket(iso) {
  const d = new Date(iso);
  if (Number.isNaN(+d)) return "Earlier";
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const that = new Date(d); that.setHours(0, 0, 0, 0);
  const diff = Math.round((today - that) / 86400000);
  return diff <= 0 ? "Today" : diff === 1 ? "Yesterday" : "Earlier";
}

async function loadSessions() {
  try {
    const data = await api("/api/sessions");
    state.sessions = Array.isArray(data) ? data : (data && data.sessions) || [];
  } catch (err) {
    state.sessions = [];
    if (err.status !== 0) toast(err.message, { error: true });
  }
  renderSessions();
}

function renderSessions() {
  // keep keyboard focus on the same row's control across a re-render
  const focused = document.activeElement && els.sessions.contains(document.activeElement)
    ? { sid: document.activeElement.closest(".sess")?.dataset.sid, kebab: document.activeElement.classList.contains("sess__more") }
    : null;
  if (menu.btn && els.sessions.contains(menu.btn)) closeMenu(false);
  const list = [...state.sessions].sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at));
  els.sessions.replaceChildren();
  if (!list.length) {
    els.sessions.innerHTML = `<p class="sessions__empty">No conversations yet. Your questions will be kept here.</p>`;
    return;
  }
  const groups = new Map([["Today", []], ["Yesterday", []], ["Earlier", []]]);
  for (const s of list) groups.get(dayBucket(s.updated_at)).push(s);
  for (const [label, items] of groups) {
    if (!items.length) continue;
    const g = document.createElement("div");
    g.className = "sessions__group";
    const h = document.createElement("h2");
    h.className = "sessions__label";
    h.textContent = label;
    const ul = document.createElement("ul");
    ul.setAttribute("role", "list");
    for (const s of items) ul.append(sessionItem(s));
    g.append(h, ul);
    els.sessions.append(g);
  }
  if (focused && focused.sid) focusSessionRow(focused.sid, focused.kebab);
}

function sessionRow(sid) {
  return [...els.sessions.querySelectorAll(".sess")].find((li) => li.dataset.sid === sid) || null;
}

function focusSessionRow(sid, kebab = true) {
  const li = sessionRow(sid);
  const el = li && li.querySelector(kebab ? ".sess__more" : ".sess__link");
  if (el) el.focus(); else els.newChat.focus();
}

function sessionItem(s) {
  const li = document.createElement("li");
  li.className = "sess";
  li.dataset.sid = s.session_id;
  const title = s.title || "Untitled conversation";
  const a = document.createElement("a");
  a.className = "sess__link";
  a.href = `/?s=${encodeURIComponent(s.session_id)}`;
  a.textContent = title;
  a.title = title;
  if (s.session_id === state.sessionId) a.setAttribute("aria-current", "page");
  a.addEventListener("click", (e) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
    e.preventDefault();
    selectSession(s.session_id);
  });
  const more = document.createElement("button");
  more.type = "button";
  more.className = "icon-btn sess__more";
  more.id = nextId("kebab-");
  more.setAttribute("aria-haspopup", "menu");
  more.setAttribute("aria-expanded", "false");
  more.setAttribute("aria-controls", "sess-menu");
  more.setAttribute("aria-label", `Options for “${title}”`);
  more.innerHTML = ICON.kebab;
  more.addEventListener("click", () => {
    if (menu.btn === more && !els.menu.hidden) closeMenu(true);
    else openMenu(more, s, "first");
  });
  more.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); openMenu(more, s, "first"); }
    else if (e.key === "ArrowUp") { e.preventDefault(); openMenu(more, s, "last"); }
  });
  li.append(a, more);
  return li;
}

/* ---- the ⋮ menu (WAI-ARIA menu button pattern) ---- */
const menu = { btn: null, session: null };

function menuItems() { return [...els.menu.querySelectorAll('[role="menuitem"]')]; }

function openMenu(btn, s, focus = "first") {
  if (menu.btn && menu.btn !== btn) menu.btn.setAttribute("aria-expanded", "false");
  menu.btn = btn;
  menu.session = s;
  els.menu.setAttribute("aria-labelledby", btn.id);
  els.menu.hidden = false;
  btn.setAttribute("aria-expanded", "true");
  btn.closest(".sess")?.classList.add("has-menu");
  positionMenu();
  const items = menuItems();
  (focus === "last" ? items[items.length - 1] : items[0]).focus();
}

function positionMenu() {
  if (!menu.btn) return;
  const r = menu.btn.getBoundingClientRect();
  const m = els.menu.getBoundingClientRect();
  let top = r.bottom + 4;
  if (top + m.height > innerHeight - 8) top = Math.max(8, r.top - m.height - 4);
  let left = r.right - m.width;
  left = Math.min(Math.max(8, left), innerWidth - m.width - 8);
  els.menu.style.top = `${Math.round(top)}px`;
  els.menu.style.left = `${Math.round(left)}px`;
}

function closeMenu(returnFocus = true) {
  if (els.menu.hidden) return;
  els.menu.hidden = true;
  const btn = menu.btn;
  if (btn) {
    btn.setAttribute("aria-expanded", "false");
    btn.closest(".sess")?.classList.remove("has-menu");
    if (returnFocus && document.contains(btn)) btn.focus();
  }
  menu.btn = null;
}

function initMenu() {
  els.menu.addEventListener("keydown", (e) => {
    const items = menuItems();
    const i = items.indexOf(document.activeElement);
    let to = null;
    if (e.key === "ArrowDown") to = items[(i + 1) % items.length];
    else if (e.key === "ArrowUp") to = items[(i - 1 + items.length) % items.length];
    else if (e.key === "Home") to = items[0];
    else if (e.key === "End") to = items[items.length - 1];
    else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); closeMenu(true); return; }
    else if (e.key === "Tab") { closeMenu(false); return; }
    else if (e.key.length === 1 && /\S/.test(e.key)) {
      const k = e.key.toLowerCase();
      const order = [...items.slice(i + 1), ...items.slice(0, i + 1)];
      to = order.find((it) => it.textContent.trim().toLowerCase().startsWith(k)) || null;
    }
    if (to) { e.preventDefault(); to.focus(); }
  });
  els.menu.addEventListener("click", (e) => {
    const item = e.target.closest('[role="menuitem"]');
    if (!item) return;
    const s = menu.session;
    const btn = menu.btn;
    const cmd = item.dataset.cmd;
    closeMenu(cmd !== "rename");
    if (!s) return;
    if (cmd === "rename") startRename(btn.closest(".sess"), s);
    else if (cmd === "memory") showMemory(s);
    else if (cmd === "delete") deleteSession(s);
  });
  document.addEventListener("pointerdown", (e) => {
    if (els.menu.hidden) return;
    if (els.menu.contains(e.target) || (menu.btn && menu.btn.contains(e.target))) return;
    closeMenu(false);
  });
  els.sessions.addEventListener("scroll", () => closeMenu(false), { passive: true });
  window.addEventListener("resize", () => closeMenu(false));
}

function startRename(li, s) {
  if (!li) return;
  const input = document.createElement("input");
  input.className = "input sess__rename";
  input.value = s.title || "";
  input.setAttribute("aria-label", "Conversation title");
  input.maxLength = 120;
  li.replaceChildren(input);
  input.focus();
  input.select();
  let done = false;
  const finish = async (save, refocus) => {
    if (done) return;
    done = true;
    const title = input.value.trim();
    if (save && title && title !== s.title) {
      try {
        await api(`/api/sessions/${encodeURIComponent(s.session_id)}`, { method: "PATCH", body: { title } });
        s.title = title;
        if (s.session_id === state.sessionId) setTitle(title);
        toast("Renamed");
      } catch (err) { toast(err.message, { error: true }); }
    }
    renderSessions();
    if (refocus) focusSessionRow(s.session_id, true);
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); finish(true, true); }
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); finish(false, true); }
  });
  input.addEventListener("blur", () => finish(true, false));
}

async function deleteSession(s) {
  const title = s.title || "Untitled conversation";
  const ok = await confirmDialog({
    title: "Delete this conversation?",
    body: `“${title}” and its session memory file will be removed. This cannot be undone.`,
    confirm: "Delete",
  });
  if (!ok) { focusSessionRow(s.session_id, true); return; }
  // where focus goes afterwards: the next row, else the previous, else "New chat"
  const rows = [...els.sessions.querySelectorAll(".sess")];
  const idx = rows.findIndex((li) => li.dataset.sid === s.session_id);
  const neighbour = rows[idx + 1] || rows[idx - 1];
  try {
    await api(`/api/sessions/${encodeURIComponent(s.session_id)}`, { method: "DELETE" });
    state.sessions = state.sessions.filter((x) => x.session_id !== s.session_id);
    if (s.session_id === state.sessionId) resetConversation(null);
    renderSessions();
    if (neighbour) focusSessionRow(neighbour.dataset.sid, true); else els.newChat.focus();
    toast("Deleted");
  } catch (err) { toast(err.message, { error: true }); }
}

/* ---- accessible confirm dialog (replaces window.confirm) ---- */
function confirmDialog({ title, body, confirm = "Delete" }) {
  const dlg = els.confirmDlg;
  $("confirm-title").textContent = title;
  $("confirm-body").textContent = body;
  $("confirm-ok").textContent = confirm;
  dlg.returnValue = "";
  return new Promise((resolve) => {
    dlg.addEventListener("close", () => resolve(dlg.returnValue === "ok"), { once: true });
    dlg.showModal();
    $("confirm-cancel").focus();
  });
}

/* ---- session memory dialog ---- */
const mem = { session: null, opener: null };

async function showMemory(s) {
  mem.session = s;
  mem.opener = sessionRow(s.session_id)?.querySelector(".sess__more") || null;
  $("memory-sub").textContent = s.title || "Untitled conversation";
  $("memory-body").innerHTML = `<p class="muted">Loading…</p>`;
  $("memory-meter-text").textContent = "";
  $("memory-path").textContent = "";
  $("memory-compact").disabled = true;
  if (!els.memDlg.open) els.memDlg.showModal();
  try {
    renderMemory(await api(`/api/sessions/${encodeURIComponent(s.session_id)}/memory`));
  } catch (err) {
    $("memory-body").innerHTML = `<p class="note note--error" role="alert">${escapeHtml(err.message)}</p>`;
  }
}

function renderMemory(data) {
  const md = (data && (data.markdown ?? data.content)) || "";
  const body = $("memory-body");
  if (md.trim()) renderMarkdown(body, md, { citations: false });
  else body.innerHTML = `<p class="muted">This conversation has no memory yet. It is written after the first answer.</p>`;
  const tokens = +(data && data.tokens) || 0;
  const budget = +(data && data.budget) || 0;
  const bar = $("memory-bar");
  const ratio = budget ? tokens / budget : 0;
  bar.setAttribute("aria-valuemax", String(budget || 1));
  bar.setAttribute("aria-valuenow", String(Math.min(tokens, budget || tokens)));
  bar.setAttribute("aria-valuetext", budget ? `${fmtInt(tokens)} of ${fmtInt(budget)} tokens` : `${fmtInt(tokens)} tokens`);
  bar.style.setProperty("--fill", `${Math.min(100, Math.round(ratio * 100))}%`);
  bar.classList.toggle("is-near", ratio >= 0.8 && ratio <= 1);
  bar.classList.toggle("is-over", ratio > 1);
  $("memory-meter-text").textContent = budget
    ? `${fmtInt(tokens)} / ${fmtInt(budget)} tokens · ${Math.round(ratio * 100)}% of budget${ratio > 1 ? " (compacts after the next answer)" : ""}`
    : `${fmtInt(tokens)} tokens`;
  $("memory-path").textContent = (data && data.path) || "";
  $("memory-path").title = (data && data.path) || "";
  $("memory-compact").disabled = false;
}

async function compactMemory() {
  const s = mem.session;
  if (!s) return;
  const btn = $("memory-compact");
  const before = +$("memory-bar").getAttribute("aria-valuenow") || 0;
  btn.disabled = true;
  btn.setAttribute("aria-busy", "true");
  const label = btn.lastChild;
  const oldText = label.textContent;
  label.textContent = " Compacting…";
  try {
    const res = await api(`/api/sessions/${encodeURIComponent(s.session_id)}/memory/compact`, { method: "POST", body: {} });
    const data = res && typeof res.markdown === "string" ? res : await api(`/api/sessions/${encodeURIComponent(s.session_id)}/memory`);
    renderMemory(data);
    const from = (res && res.before_tokens) ?? before;
    toast(`Memory compacted: ${fmtInt(from)} → ${fmtInt(data.tokens)} tokens`);
  } catch (err) {
    toast(err.message, { error: true });
    btn.disabled = false;
  } finally {
    btn.removeAttribute("aria-busy");
    label.textContent = oldText;
  }
}

function initDialogs() {
  els.memDlg.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", () => els.memDlg.close()));
  els.memDlg.addEventListener("click", (e) => { if (e.target === els.memDlg) els.memDlg.close(); });
  els.memDlg.addEventListener("close", () => {
    const sid = mem.session && mem.session.session_id;
    if (mem.opener && document.contains(mem.opener)) mem.opener.focus();
    else if (sid) focusSessionRow(sid, true);
  });
  $("memory-compact").addEventListener("click", compactMemory);
  $("confirm-cancel").addEventListener("click", () => els.confirmDlg.close("cancel"));
  $("confirm-ok").addEventListener("click", () => els.confirmDlg.close("ok"));
  els.confirmDlg.addEventListener("click", (e) => { if (e.target === els.confirmDlg) els.confirmDlg.close("cancel"); });
}

function setUrl(id) {
  const url = id ? `/?s=${encodeURIComponent(id)}` : "/";
  if (location.pathname + location.search !== url) history.pushState({ s: id }, "", url);
}

function setTitle(t) {
  els.title.textContent = t || "New conversation";
  document.title = t ? `${t} - NewRAG` : "NewRAG - Materials Science Notebook";
}

function resetConversation(id) {
  stopStream();
  state.sessionId = id;
  state.turn = 0;
  state.activeLeaf = null;
  els.messages.replaceChildren();
  setUrl(id);
  setTitle(null);
  if (!id) showEmpty();
  renderSessions();
}

async function selectSession(id) {
  closeSidebar();
  if (id === state.sessionId && els.messages.children.length) return;
  resetConversation(id);
  const s = state.sessions.find((x) => x.session_id === id);
  setTitle(s && s.title);
  els.messages.setAttribute("aria-busy", "true");
  try {
    const data = await api(`/api/sessions/${encodeURIComponent(id)}/messages`);
    renderPath(data);
    scrollToEnd(true);
  } catch (err) {
    if (err.status === 404) { toast("That conversation no longer exists.", { error: true }); resetConversation(null); }
    else toast(err.message, { error: true });
  } finally {
    els.messages.removeAttribute("aria-busy");
  }
}

/** Render a conversation path (GET messages / POST switch response) into the thread. */
function renderPath(data) {
  const rows = Array.isArray(data) ? data : (data && (data.messages || data.path)) || [];
  els.messages.replaceChildren();
  if (!rows.length) showEmpty();
  for (const m of rows) {
    if (m.role === "user") {
      const art = addUser(m.content);
      applyBranch(art, m);
      renderUserTools(art);
    } else if (m.role === "assistant") {
      const art = addAssistant();
      const d = msgData.get(art);
      d.content = m.content || "";
      d.sources = m.sources || [];
      d.images = m.images || [];
      d.ooc = !!m.out_of_context || isOOC(d.content);
      d.usage = m.usage || null;
      d.mode = m.mode || null;
      d.feedback = m.feedback && typeof m.feedback === "object" && m.feedback.rating ? m.feedback : null;
      applyBranch(art, m);
      if (m.decisions && typeof m.decisions === "object") { d.decision = m.decisions; renderDecision(art, m.decisions); }
      finalize(art);
    }
  }
  const last = rows[rows.length - 1];
  state.activeLeaf = last ? last.message_id ?? null : null;
}

function applyBranch(art, m) {
  const d = msgData.get(art);
  d.id = m.message_id ?? null;
  d.parentId = m.parent_id ?? null;
  const sib = Array.isArray(m.sibling_ids) && m.sibling_ids.length ? m.sibling_ids : (d.id != null ? [d.id] : []);
  d.siblings = sib;
  d.sibIndex = Number.isInteger(m.sibling_index) ? m.sibling_index : Math.max(0, sib.indexOf(d.id));
  d.sibCount = Number.isInteger(m.sibling_count) ? m.sibling_count : sib.length;
  if (d.id != null) art.dataset.mid = String(d.id);
}

async function newChat() {
  closeSidebar();
  if (state.streaming) stopStream();
  if (!state.sessionId && !els.messages.querySelector(".msg")) { els.input.focus(); return; }
  try {
    const s = await api("/api/sessions", { method: "POST", body: {} });
    if (s && s.session_id) {
      state.sessions.unshift({ session_id: s.session_id, title: s.title, updated_at: new Date().toISOString(), message_count: 0 });
      resetConversation(s.session_id);
      setTitle(s.title);
    } else resetConversation(null);
  } catch {
    resetConversation(null); // the first message will create the session
  }
  els.input.focus();
}

/* ============ mobile sidebar ============ */
function openSidebar() {
  els.sidebar.classList.add("is-open");
  els.scrim.hidden = false;
  els.open.setAttribute("aria-expanded", "true");
  els.newChat.focus();
}
function closeSidebar() {
  if (!els.sidebar.classList.contains("is-open")) return;
  els.sidebar.classList.remove("is-open");
  els.scrim.hidden = true;
  els.open.setAttribute("aria-expanded", "false");
}

/* ============ answer modes ============ */
function modeById(id) { return state.modes.find((m) => m.id === id) || null; }

async function loadModes() {
  try {
    const data = await api("/api/modes");
    const list = Array.isArray(data) ? data : (data && data.modes) || [];
    if (list.length && list.every((m) => m && m.id)) state.modes = list;
  } catch { /* keep the built-in defaults */ }
  if (!modeById(state.mode)) state.mode = state.modes[0].id;
  renderModes();
}

function renderModes() {
  const box = els.modes;
  box.replaceChildren();
  for (const m of state.modes) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "modes__opt";
    b.dataset.mode = m.id;
    b.setAttribute("role", "radio");
    const checked = m.id === state.mode;
    b.setAttribute("aria-checked", String(checked));
    b.tabIndex = checked ? 0 : -1;
    const nameId = `mode-name-${m.id}`, descId = `mode-desc-${m.id}`;
    b.setAttribute("aria-labelledby", nameId);
    b.setAttribute("aria-describedby", descId);
    b.innerHTML = `<span class="modes__glyph" aria-hidden="true">${m.id === "deep" ? "◆" : "◇"}</span><span id="${nameId}">${escapeHtml(m.name || m.id)}</span>
      <span class="sr-only" id="${descId}">${escapeHtml([m.tagline, m.description].filter(Boolean).join(". "))}</span>`;
    b.addEventListener("click", () => setMode(m.id, true));
    b.addEventListener("pointerenter", () => showModeTip(b, m));
    b.addEventListener("pointerleave", hideModeTip);
    b.addEventListener("focus", () => showModeTip(b, m));
    b.addEventListener("blur", hideModeTip);
    box.append(b);
  }
  updateModeNote();
}

function setMode(id, focus = false) {
  if (!modeById(id)) return;
  const changed = id !== state.mode;
  state.mode = id;
  storeMode(id);
  els.modes.querySelectorAll(".modes__opt").forEach((b) => {
    const on = b.dataset.mode === id;
    b.setAttribute("aria-checked", String(on));
    b.tabIndex = on ? 0 : -1;
    if (on && focus) b.focus();
  });
  if (changed) updateModeNote();
}

function updateModeNote() {
  const m = modeById(state.mode);
  const deep = state.mode === "deep";
  els.modeNote.classList.toggle("is-warn", deep);
  els.modeNote.textContent = deep
    ? "Slower: about 1–3 min per answer. Shows its reasoning."
    : (m && m.tagline) || "";
}

let tipEl = null;
function showModeTip(btn, m) {
  if (!tipEl) {
    tipEl = document.createElement("div");
    tipEl.className = "mode-tip";
    tipEl.setAttribute("role", "tooltip");
    tipEl.id = "mode-tip";
    document.body.append(tipEl);
  }
  tipEl.innerHTML = `<strong>${escapeHtml(m.name || m.id)}</strong>${m.tagline ? ` · <span class="mode-tip__tag">${escapeHtml(m.tagline)}</span>` : ""}${m.description ? `<span class="mode-tip__desc">${escapeHtml(m.description)}</span>` : ""}`;
  tipEl.hidden = false;
  const r = btn.getBoundingClientRect();
  const t = tipEl.getBoundingClientRect();
  const left = Math.min(Math.max(8, r.left), innerWidth - t.width - 8);
  tipEl.style.left = `${Math.round(left)}px`;
  tipEl.style.top = `${Math.round(Math.max(8, r.top - t.height - 8))}px`;
  tipEl.classList.add("is-on");
}
function hideModeTip() { if (tipEl) { tipEl.classList.remove("is-on"); tipEl.hidden = true; } }

function initModes() {
  els.modes.addEventListener("keydown", (e) => {
    const opts = [...els.modes.querySelectorAll(".modes__opt")];
    const i = opts.findIndex((b) => b.dataset.mode === state.mode);
    let j = null;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") j = (i + 1) % opts.length;
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") j = (i - 1 + opts.length) % opts.length;
    else if (e.key === "Home") j = 0;
    else if (e.key === "End") j = opts.length - 1;
    else if (e.key === "Escape") { hideModeTip(); return; }
    if (j == null) return;
    e.preventDefault();
    setMode(opts[j].dataset.mode, true);
  });
  renderModes();
  loadModes();
}

/* ============ messages ============ */

function showEmpty() {
  const div = document.createElement("div");
  div.className = "empty";
  div.innerHTML = `
    <svg class="empty__cell" viewBox="0 0 120 120" aria-hidden="true">
      <g class="cell-edges">
        <path d="M30 42 L70 30 L98 46 L58 58 Z"/>
        <path d="M30 42 V86 M70 30 V74 M98 46 V90 M58 58 V102"/>
        <path d="M30 86 L70 74 L98 90 L58 102 Z"/>
      </g>
      <g class="cell-atoms">
        <circle cx="30" cy="42" r="4"/><circle cx="70" cy="30" r="4"/><circle cx="98" cy="46" r="4"/><circle cx="58" cy="58" r="4"/>
        <circle cx="30" cy="86" r="4"/><circle cx="70" cy="74" r="4"/><circle cx="98" cy="90" r="4"/><circle cx="58" cy="102" r="4"/>
      </g>
      <circle class="cell-body" cx="64" cy="66" r="7"/>
    </svg>
    <h2 class="empty__title">Ask the textbook.</h2>
    <p class="empty__lede">Answers come only from <em>Materials Science and Engineering</em>, with every claim tied to the page it came from. Questions outside the book get a plain <code>${OOC}</code>.</p>
    <ul class="empty__examples" role="list">
      ${EXAMPLES.map((q) => `<li><button type="button" class="example">${escapeHtml(q)}</button></li>`).join("")}
    </ul>`;
  div.querySelectorAll(".example").forEach((b) => b.addEventListener("click", () => {
    els.input.value = b.textContent;
    autosize();
    updateCount();
    send();
  }));
  els.messages.append(div);
}

function clearEmpty() { els.messages.querySelector(".empty")?.remove(); }

function blankBranch(role) {
  return { role, id: null, parentId: null, siblings: [], sibIndex: 0, sibCount: 1 };
}

function addUser(text) {
  clearEmpty();
  const art = document.createElement("article");
  art.className = "msg msg--user";
  art.setAttribute("aria-label", "Your question");
  const b = document.createElement("div");
  b.className = "bubble";
  b.textContent = text;
  const tools = document.createElement("div");
  tools.className = "msg__tools";
  art.append(b, tools);
  msgData.set(art, { ...blankBranch("user"), content: text });
  els.messages.append(art);
  return art;
}

function sibNavHtml(kind) {
  return `<div class="sibnav" role="group" aria-label="${kind} versions" hidden>
      <button type="button" class="icon-btn sibnav__btn" data-act="sib-prev" data-branch aria-label="Previous version of this ${kind.toLowerCase()}">${ICON.prev}</button>
      <span class="sibnav__pos num" aria-live="polite"></span>
      <button type="button" class="icon-btn sibnav__btn" data-act="sib-next" data-branch aria-label="Next version of this ${kind.toLowerCase()}">${ICON.next}</button>
    </div>`;
}

function renderSibNav(art) {
  const d = msgData.get(art);
  const nav = art.querySelector(".sibnav");
  if (!nav) return;
  const show = d.id != null && d.sibCount > 1;
  nav.hidden = !show;
  if (!show) return;
  const pos = d.sibIndex + 1;
  nav.querySelector(".sibnav__pos").textContent = `${pos} / ${d.sibCount}`;
  nav.querySelector(".sibnav__pos").setAttribute("aria-label", `Version ${pos} of ${d.sibCount}`);
  const prev = nav.querySelector('[data-act="sib-prev"]');
  const next = nav.querySelector('[data-act="sib-next"]');
  prev.dataset.edge = d.sibIndex <= 0 ? "1" : "";
  next.dataset.edge = d.sibIndex >= d.sibCount - 1 ? "1" : "";
  prev.disabled = state.streaming || d.sibIndex <= 0;
  next.disabled = state.streaming || d.sibIndex >= d.sibCount - 1;
}

function renderUserTools(art) {
  const d = msgData.get(art);
  const tools = art.querySelector(".msg__tools");
  if (!tools.firstChild) {
    tools.innerHTML = `${sibNavHtml("Question")}
      <button type="button" class="foot-btn msg__edit" data-act="edit" data-branch aria-label="Edit question" hidden>${ICON.edit}<span>Edit</span></button>`;
    tools.querySelector('[data-act="edit"]').addEventListener("click", () => startEdit(art));
    tools.querySelector('[data-act="sib-prev"]').addEventListener("click", () => switchSibling(art, -1));
    tools.querySelector('[data-act="sib-next"]').addEventListener("click", () => switchSibling(art, 1));
  }
  const edit = tools.querySelector('[data-act="edit"]');
  edit.hidden = d.id == null;
  edit.disabled = state.streaming;
  renderSibNav(art);
}

function addAssistant() {
  clearEmpty();
  const art = document.createElement("article");
  art.className = "msg msg--bot";
  art.setAttribute("aria-label", "Answer");
  const id = nextId("m");
  art.innerHTML = `
    <div class="msg__decisions" hidden></div>
    <details class="reasoning" hidden>
      <summary class="reasoning__sum"><span class="reasoning__label">Reasoning</span><span class="reasoning__meta"></span></summary>
      <div class="reasoning__body" aria-label="Model reasoning, not part of the answer"></div>
    </details>
    <div class="msg__body prose"></div>
    <div class="msg__gallery" hidden></div>
    <footer class="msg__foot" hidden>
      <span class="mode-tag" hidden></span>
      <span class="msg__tokens num"></span>
      <button type="button" class="foot-btn" data-act="copy" aria-label="Copy answer">
        <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><rect x="5" y="5" width="8" height="8.5" rx="1.5"/><path d="M3 10.5V3.5A1.5 1.5 0 0 1 4.5 2H10"/></svg>
        <span>Copy</span>
      </button>
      <button type="button" class="foot-btn foot-btn--pages" data-act="pages" aria-expanded="false" aria-controls="${id}-pages" hidden>
        <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><path d="M4 2h6l3 3v9H4z"/><path d="M6.5 8.5h4M6.5 11h4"/></svg>
        <span class="foot-btn__label">Show pages</span>
      </button>
      <span class="msg__acts">
        <button type="button" class="foot-btn foot-btn--icon fb fb--up" data-act="like" aria-pressed="false" aria-label="Good answer" title="Good answer" hidden>${ICON.up}</button>
        <button type="button" class="foot-btn foot-btn--icon fb fb--down" data-act="dislike" aria-pressed="false" aria-label="Bad answer" title="Bad answer" aria-controls="${id}-fb" hidden>${ICON.down}</button>
        <button type="button" class="foot-btn" data-act="regen" data-branch aria-label="Regenerate answer" title="Regenerate (keeps this version)" hidden>${ICON.regen}<span>Regenerate</span></button>
        ${sibNavHtml("Answer")}
      </span>
    </footer>
    <form class="fb-pop" id="${id}-fb" role="group" aria-labelledby="${id}-fb-label" hidden>
      <label class="fb-pop__label" id="${id}-fb-label" for="${id}-fb-text">What was wrong with this answer? <span class="fb-pop__opt">Optional</span></label>
      <textarea class="textarea fb-pop__input" id="${id}-fb-text" rows="2" maxlength="1000" placeholder="e.g. a wrong value, a missing citation, it should not have refused"></textarea>
      <div class="fb-pop__actions">
        <button type="button" class="btn" data-act="fb-skip">Skip</button>
        <button type="submit" class="btn btn--primary">Send feedback</button>
      </div>
    </form>
    <div class="msg__pages" id="${id}-pages" hidden></div>`;
  msgData.set(art, { ...blankBranch("assistant"), content: "", sources: [], images: [], usage: null, ooc: false, decision: null, error: null, stopped: false, uncited: false, mode: null, feedback: null, thinking: "", reasoningTouched: false });
  art.querySelector('[data-act="copy"]').addEventListener("click", () => copyAnswer(art));
  art.querySelector('[data-act="pages"]').addEventListener("click", () => togglePages(art));
  art.querySelector('[data-act="like"]').addEventListener("click", () => onLike(art));
  art.querySelector('[data-act="dislike"]').addEventListener("click", () => onDislike(art));
  art.querySelector('[data-act="regen"]').addEventListener("click", () => regenerate(art));
  art.querySelector('[data-act="sib-prev"]').addEventListener("click", () => switchSibling(art, -1));
  art.querySelector('[data-act="sib-next"]').addEventListener("click", () => switchSibling(art, 1));
  art.querySelector(".reasoning__sum").addEventListener("click", () => { msgData.get(art).reasoningTouched = true; });
  const pop = art.querySelector(".fb-pop");
  pop.addEventListener("submit", (e) => { e.preventDefault(); sendFeedbackComment(art); });
  pop.querySelector('[data-act="fb-skip"]').addEventListener("click", () => closeFeedbackPop(art, true));
  pop.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); closeFeedbackPop(art, true); }
    else if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.target.tagName === "TEXTAREA") { e.preventDefault(); sendFeedbackComment(art); }
  });
  els.messages.append(art);
  return art;
}

async function copyAnswer(art) {
  const text = msgData.get(art).content; // the answer only, never the reasoning
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.append(ta); ta.select();
    try { document.execCommand("copy"); } catch { /* ignore */ }
    ta.remove();
  }
  toast("Answer copied");
}

function togglePages(art, focusN) {
  const btn = art.querySelector('[data-act="pages"]');
  const panel = art.querySelector(".msg__pages");
  const d = msgData.get(art);
  const expand = focusN != null ? true : btn.getAttribute("aria-expanded") !== "true";
  if (expand && !panel.firstChild) {
    panel.append(buildPagesPanel(d.sources, (s, i) => openSource(s, i)));
  }
  panel.hidden = !expand;
  btn.setAttribute("aria-expanded", String(expand));
  btn.querySelector(".foot-btn__label").textContent = expand ? "Hide pages" : "Show pages";
  if (focusN != null) {
    const sec = panel.querySelector(`.pages__source[data-n="${CSS.escape(String(focusN))}"]`);
    sec?.scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  }
}

function renderDecision(art, dec) {
  const box = art.querySelector(".msg__decisions");
  if (!dec || typeof dec !== "object") return;
  const chips = [];
  if (dec.intent && dec.intent !== "other") chips.push(dec.intent.replace(/_/g, " "));
  if (dec.answer_format && dec.answer_format !== "prose") chips.push(`as ${dec.answer_format}`);
  if (dec.is_cross_reference) chips.push("cross-reference");
  if (dec.needs_images) chips.push("with figures");
  if (!chips.length) return;
  box.innerHTML = chips.map((c) => `<span class="dchip">${escapeHtml(c)}</span>`).join("");
  box.hidden = false;
}

/* ---- reasoning (Deep mode `thinking` events) ---- */
function renderReasoning(art, { live = false } = {}) {
  const d = msgData.get(art);
  const det = art.querySelector(".reasoning");
  if (!d.thinking) { det.hidden = true; return; }
  det.hidden = false;
  det.classList.toggle("is-live", live);
  art.querySelector(".reasoning__body").textContent = d.thinking.trim();
  const words = d.thinking.trim().split(/\s+/).length;
  art.querySelector(".reasoning__label").textContent = live ? "Reasoning…" : "Reasoning";
  art.querySelector(".reasoning__meta").textContent = `${fmtInt(words)} words · not part of the answer`;
}

let rafId = 0;
const pending = new Set();
function scheduleRender(art) {
  pending.add(art);
  if (rafId) return;
  rafId = requestAnimationFrame(() => {
    rafId = 0;
    for (const a of pending) {
      const d = msgData.get(a);
      if (d.thinkingDirty) { d.thinkingDirty = false; renderReasoning(a, { live: !d.content }); }
      if (d.content) renderMarkdown(a.querySelector(".msg__body"), d.content, { streaming: true });
    }
    pending.clear();
    scrollToEnd();
  });
}

function finalize(art) {
  const d = msgData.get(art);
  const body = art.querySelector(".msg__body");
  pending.delete(art);
  body.classList.remove("is-streaming");
  body.removeAttribute("aria-busy");
  art.classList.remove("is-live");
  renderReasoning(art);

  if (d.ooc || isOOC(d.content)) {
    d.ooc = true;
    art.classList.add("msg--ooc");
    body.classList.remove("prose");
    body.innerHTML = `<div class="ooc" role="note"><p class="ooc__tag">${OOC}</p></div>`;
    art.querySelector(".msg__gallery").hidden = true;
    const foot = art.querySelector(".msg__foot");
    foot.hidden = false;
    foot.querySelector('[data-act="pages"]').hidden = true;
    foot.querySelector('[data-act="copy"]').hidden = true;
    setTokens(art);
    renderBotTools(art);
    return;
  }

  if (d.content) renderMarkdown(body, d.content);
  else if (!d.error) body.innerHTML = `<p class="muted">No answer was produced.</p>`;

  if (d.warning) body.insertAdjacentHTML("afterbegin", `<p class="note">${escapeHtml(d.warning)}</p>`);
  if (d.stopped) body.insertAdjacentHTML("beforeend", `<p class="note">Stopped before the answer finished.</p>`);
  if (d.error) body.insertAdjacentHTML("beforeend", `<p class="note note--error" role="alert">${escapeHtml(d.error)}</p>`);
  if (d.uncited) body.insertAdjacentHTML("beforeend", `<p class="note">This answer has no citations; check it against the textbook before relying on it.</p>`);

  // citations that point at a missing source become plain text-styled chips
  const nums = new Set((d.sources || []).map((s) => +s.n));
  body.querySelectorAll(".cite__chip").forEach((c) => {
    if (!nums.has(+c.dataset.n)) { c.disabled = true; c.classList.add("is-missing"); c.setAttribute("aria-label", `Source ${c.dataset.n} (not available)`); }
  });

  renderGallery(art);

  const foot = art.querySelector(".msg__foot");
  foot.hidden = false;
  const pagesBtn = foot.querySelector('[data-act="pages"]');
  const hasPages = (d.sources || []).some((s) => sourcePages(s).length);
  pagesBtn.hidden = !hasPages;
  if (hasPages) {
    const n = d.sources.length;
    pagesBtn.setAttribute("aria-label", `Show pages for ${n} cited source${n === 1 ? "" : "s"}`);
  }
  setTokens(art);
  renderBotTools(art);
}

/** Mode label, feedback toggles, Regenerate and version arrows of an answer. */
function renderBotTools(art) {
  const d = msgData.get(art);
  const tag = art.querySelector(".mode-tag");
  const m = d.mode && (modeById(d.mode) || { id: d.mode, name: d.mode });
  tag.hidden = !m;
  if (m) {
    tag.textContent = m.name || m.id;
    tag.dataset.mode = m.id;
    tag.title = `Answered in ${m.name || m.id} mode${m.tagline ? ` (${m.tagline.toLowerCase()})` : ""}`;
  }
  const hasId = d.id != null;
  art.querySelector('[data-act="like"]').hidden = !hasId;
  art.querySelector('[data-act="dislike"]').hidden = !hasId;
  const regen = art.querySelector('[data-act="regen"]');
  regen.hidden = !hasId;
  regen.disabled = state.streaming;
  renderFeedback(art);
  renderSibNav(art);
}

function setTokens(art) {
  const d = msgData.get(art);
  const t = art.querySelector(".msg__tokens");
  const u = d.usage;
  if (!u) { t.hidden = true; return; }
  const total = u.total ?? u.total_tokens ?? ((u.prompt_tokens || 0) + (u.completion_tokens || 0));
  t.hidden = false;
  t.textContent = `${fmtInt(total)} tokens`;
  const parts = [`${fmtInt(u.prompt_tokens)} prompt, ${fmtInt(u.completion_tokens)} completion`];
  if (u.by_purpose && typeof u.by_purpose === "object") {
    for (const [k, v] of Object.entries(u.by_purpose)) {
      const n = typeof v === "object" && v ? (v.total ?? v.total_tokens ?? ((v.prompt_tokens || 0) + (v.completion_tokens || 0))) : v;
      parts.push(`${k.replace(/_/g, " ")}: ${fmtInt(n)}`);
    }
  }
  if (u.timings_ms && typeof u.timings_ms === "object") {
    if (u.timings_ms.total != null) parts.push(`time: ${(u.timings_ms.total / 1000).toFixed(1)} s`);
    const ms = Object.entries(u.timings_ms).filter(([k]) => k !== "total").map(([k, v]) => `${k} ${fmtInt(v)} ms`);
    if (ms.length) parts.push(ms.join(", "));
  }
  t.title = parts.join("\n");
}

function renderGallery(art) {
  const d = msgData.get(art);
  const g = art.querySelector(".msg__gallery");
  const imgs = (d.images || []).filter((im) => im && im.url);
  if (!imgs.length) { g.hidden = true; return; }
  g.replaceChildren();
  imgs.forEach((im, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "fig";
    const label = im.figure_number || (im.page ? `Page ${im.page}` : "Figure");
    b.setAttribute("aria-label", `Open ${label}${im.caption ? `: ${im.caption}` : ""}`);
    b.innerHTML = `
      <span class="fig__frame"><img loading="lazy" decoding="async" alt="" src="${escapeHtml(im.url)}"></span>
      <span class="fig__cap"><span class="fig__num">${escapeHtml(label)}</span>${im.caption ? ` ${escapeHtml(stripFigNum(im.caption, im.figure_number))}` : ""}</span>`;
    b.addEventListener("click", () => openImages(imgs, i));
    g.append(b);
  });
  g.hidden = false;
}

function stripFigNum(caption, num) {
  if (!num) return caption;
  return caption.replace(new RegExp(`^\\s*${num.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*[:.\\-–]?\\s*`, "i"), "");
}

function scrollToEnd(force = false) {
  const t = els.thread;
  const near = t.scrollHeight - t.scrollTop - t.clientHeight < 160;
  if (force || near) t.scrollTop = t.scrollHeight;
}

/* ============ feedback ============ */
function renderFeedback(art) {
  const d = msgData.get(art);
  const r = d.feedback ? d.feedback.rating : 0;
  const up = art.querySelector('[data-act="like"]');
  const down = art.querySelector('[data-act="dislike"]');
  up.setAttribute("aria-pressed", String(r === 1));
  down.setAttribute("aria-pressed", String(r === -1));
  down.title = r === -1 && d.feedback.comment ? `Bad answer: “${d.feedback.comment}”` : "Bad answer";
}

async function postFeedback(art, rating, comment) {
  const d = msgData.get(art);
  const prev = d.feedback;
  d.feedback = rating ? { rating, comment: comment !== undefined ? comment : (prev && prev.rating === rating ? prev.comment : null) } : null;
  renderFeedback(art);
  const body = { rating };
  if (comment !== undefined) body.comment = comment;
  try {
    await api(`/api/messages/${encodeURIComponent(d.id)}/feedback`, { method: "POST", body });
    return true;
  } catch (err) {
    d.feedback = prev;
    renderFeedback(art);
    toast(err.message, { error: true });
    return false;
  }
}

async function onLike(art) {
  const d = msgData.get(art);
  if (d.id == null) return;
  closeFeedbackPop(art, false);
  const on = !(d.feedback && d.feedback.rating === 1);
  if (await postFeedback(art, on ? 1 : 0)) toast(on ? "Marked as a good answer" : "Feedback removed");
}

async function onDislike(art) {
  const d = msgData.get(art);
  if (d.id == null) return;
  const on = !(d.feedback && d.feedback.rating === -1);
  if (!(await postFeedback(art, on ? -1 : 0))) return;
  if (on) openFeedbackPop(art);
  else { closeFeedbackPop(art, false); toast("Feedback removed"); }
}

function openFeedbackPop(art) {
  const d = msgData.get(art);
  const pop = art.querySelector(".fb-pop");
  const ta = pop.querySelector("textarea");
  ta.value = (d.feedback && d.feedback.comment) || "";
  pop.hidden = false;
  art.querySelector('[data-act="dislike"]').setAttribute("aria-expanded", "true");
  ta.focus();
  pop.scrollIntoView({ block: "nearest" });
}

function closeFeedbackPop(art, refocus) {
  const pop = art.querySelector(".fb-pop");
  if (pop.hidden) return;
  pop.hidden = true;
  const down = art.querySelector('[data-act="dislike"]');
  down.removeAttribute("aria-expanded");
  if (refocus) down.focus();
}

async function sendFeedbackComment(art) {
  const d = msgData.get(art);
  const text = art.querySelector(".fb-pop textarea").value.trim();
  if (!text) { closeFeedbackPop(art, true); toast("Marked as a bad answer"); return; }
  if (await postFeedback(art, -1, text)) {
    closeFeedbackPop(art, true);
    toast("Thanks, the comment was saved");
  } else if (d.feedback == null) closeFeedbackPop(art, true);
}

/* ============ stage indicator ============ */
let stageShownAt = 0, stageHideT = 0;
function setStage(stage, detail) {
  clearTimeout(stageHideT);
  state.stage = stage;
  if (!stage) {
    // keep it visible for at least 400 ms to avoid flicker
    const wait = Math.max(0, 400 - (performance.now() - stageShownAt));
    stageHideT = setTimeout(() => { els.stage.hidden = true; }, wait);
    return;
  }
  if (els.stage.hidden) { els.stage.hidden = false; stageShownAt = performance.now(); }
  const idx = STAGE_ORDER.indexOf(STAGE_TICK[stage] || stage);
  els.stage.querySelectorAll("li").forEach((li, i) => {
    li.classList.toggle("is-done", idx > -1 && i < idx);
    li.classList.toggle("is-now", i === idx);
  });
  els.stageLabel.textContent = (typeof detail === "string" && detail) ? `${STAGES[stage] ? STAGES[stage].replace("…", "") + ": " : ""}${detail}` : (STAGES[stage] || `${stage}…`);
}

/* ============ branching: edit / regenerate / switch ============ */

/** Remove `art` and every message after it (the old branch tail) from the view. */
function removeFrom(art) {
  let n = art;
  while (n) { const next = n.nextElementSibling; n.remove(); n = next; }
}

function syncBranchButtons() {
  els.messages.querySelectorAll(".msg").forEach((art) => {
    const d = msgData.get(art);
    if (!d) return;
    if (d.role === "user") renderUserTools(art);
    else if (art.querySelector(".msg__foot:not([hidden])")) {
      art.querySelector('[data-act="regen"]').disabled = state.streaming;
      renderSibNav(art);
    }
  });
}

function startEdit(art) {
  if (state.streaming || art.classList.contains("is-editing")) return;
  const d = msgData.get(art);
  if (d.id == null) return;
  art.classList.add("is-editing");
  const bubble = art.querySelector(".bubble");
  const tools = art.querySelector(".msg__tools");
  bubble.hidden = true;
  tools.hidden = true;
  const fid = nextId("edit-");
  const form = document.createElement("form");
  form.className = "edit";
  form.innerHTML = `
    <label class="sr-only" for="${fid}">Edit your question</label>
    <textarea class="textarea edit__input" id="${fid}" rows="2" maxlength="${MAX_CHARS}" aria-describedby="${fid}-hint"></textarea>
    <div class="edit__bar">
      <p class="edit__hint" id="${fid}-hint">Saving starts a new branch from here. The current one stays reachable with the version arrows.</p>
      <button type="button" class="btn" data-act="cancel">Cancel</button>
      <button type="submit" class="btn btn--primary">Save &amp; submit</button>
    </div>`;
  art.append(form);
  const ta = form.querySelector("textarea");
  ta.value = d.content;
  const size = () => { ta.style.height = "auto"; ta.style.height = `${Math.min(ta.scrollHeight, 260)}px`; };
  ta.addEventListener("input", size);
  size();
  ta.focus();
  ta.setSelectionRange(ta.value.length, ta.value.length);
  const cancel = () => {
    form.remove();
    bubble.hidden = false;
    tools.hidden = false;
    art.classList.remove("is-editing");
    art.querySelector('[data-act="edit"]')?.focus();
  };
  form.querySelector('[data-act="cancel"]').addEventListener("click", cancel);
  form.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); cancel(); }
    else if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.target === ta) { e.preventDefault(); form.requestSubmit(); }
  });
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = ta.value.trim();
    if (!text) { ta.focus(); toast("The question cannot be empty.", { error: true }); return; }
    if (state.streaming) { toast("Wait for the current answer to finish.", { error: true }); return; }
    submitEdit(art, text);
  });
}

function submitEdit(oldArt, text) {
  const old = msgData.get(oldArt);
  removeFrom(oldArt);
  const userArt = addUser(text);
  const ud = msgData.get(userArt);
  ud.parentId = old.parentId;
  renderUserTools(userArt);
  const art = addAssistant();
  runTurn({ message: text, edit_of: old.id }, { userArt, art, userSiblings: old.siblings });
}

function regenerate(botArt) {
  if (state.streaming) return;
  const old = msgData.get(botArt);
  if (old.id == null) return;
  removeFrom(botArt);
  const art = addAssistant();
  msgData.get(art).parentId = old.parentId;
  runTurn({ regenerate_of: old.id }, { userArt: null, art, botSiblings: old.siblings });
}

async function switchSibling(art, dir) {
  if (state.streaming || !state.sessionId) return;
  const d = msgData.get(art);
  const target = d.siblings[d.sibIndex + dir];
  if (target == null) return;
  // keep the switched message at the same place on screen
  const offset = art.getBoundingClientRect().top;
  els.messages.setAttribute("aria-busy", "true");
  try {
    const data = await api(`/api/sessions/${encodeURIComponent(state.sessionId)}/switch`, { method: "POST", body: { message_id: target } });
    renderPath(data);
    const na = els.messages.querySelector(`.msg[data-mid="${CSS.escape(String(target))}"]`);
    if (na) {
      els.thread.scrollTop += na.getBoundingClientRect().top - offset;
      const btn = na.querySelector(`[data-act="${dir < 0 ? "sib-prev" : "sib-next"}"]`);
      const other = na.querySelector(`[data-act="${dir < 0 ? "sib-next" : "sib-prev"}"]`);
      (btn && !btn.disabled ? btn : other)?.focus();
    }
  } catch (err) {
    toast(err.message, { error: true });
  } finally {
    els.messages.removeAttribute("aria-busy");
  }
}

/* ============ sending ============ */

function setStreaming(on) {
  state.streaming = on;
  els.send.classList.toggle("is-stop", on);
  els.send.setAttribute("aria-label", on ? "Stop generating" : "Send question");
  els.send.type = on ? "button" : "submit";
  syncBranchButtons();
}

function stopStream() {
  if (state.controller) state.controller.abort();
}

async function send() {
  const text = els.input.value.trim();
  if (!text || state.streaming) return;
  els.input.value = "";
  autosize();
  updateCount();
  const userArt = addUser(text);
  renderUserTools(userArt);
  const art = addAssistant();
  runTurn({ message: text }, { userArt, art });
}

/**
 * Stream one answer. `extra` = request fields (message / edit_of / regenerate_of).
 * ctx.userArt = the new user message (null for a regenerate); ctx.userSiblings / ctx.botSiblings = the sibling
 * ids of the edited question / regenerated answer, so the new message's version arrows are right at once.
 */
async function runTurn(extra, ctx) {
  const { userArt, art } = ctx;
  const d = msgData.get(art);
  const mode = state.mode;
  d.mode = mode;
  const body = art.querySelector(".msg__body");
  body.classList.add("is-streaming");
  body.setAttribute("aria-busy", "true");
  art.classList.add("is-live");
  scrollToEnd(true);

  const controller = new AbortController();
  state.controller = controller;
  setStreaming(true);
  setStage("routing");
  let finished = false; // `done` received: the answer is complete (a late `session` event may still follow)
  let unlocked = false;
  const unlock = () => {
    // finish this answer in the UI (idempotent); the stream itself may stay open for the generated title
    if (unlocked) return;
    unlocked = true;
    if (state.controller === controller) { state.controller = null; setStreaming(false); setStage(null); }
    finalize(art);
    if (userArt && document.contains(userArt)) renderUserTools(userArt);
    scrollToEnd();
    state.turn++;
  };
  const setUserId = (id) => {
    if (!userArt || id == null) return;
    const ud = msgData.get(userArt);
    if (ud.id === id) return;
    ud.id = id;
    ud.siblings = [...(ctx.userSiblings || []).filter((x) => x !== id), id];
    ud.sibIndex = ud.siblings.length - 1;
    ud.sibCount = ud.siblings.length;
    userArt.dataset.mid = String(id);
  };

  const onEvent = (event, data) => {
    switch (event) {
      case "session":
        if (data && data.session_id) {
          const known = state.sessions.find((s) => s.session_id === data.session_id);
          if (finished) {
            // late event after `done`: only the generated title; never switch the visible conversation
            if (known && data.title) known.title = data.title;
            if (data.session_id === state.sessionId && data.title) setTitle(data.title);
            renderSessions();
            break;
          }
          const isNew = data.session_id !== state.sessionId;
          state.sessionId = data.session_id;
          setUrl(data.session_id);
          if (data.title) setTitle(data.title);
          if (known) { if (data.title) known.title = data.title; }
          else if (isNew) {
            state.sessions.unshift({ session_id: data.session_id, title: data.title, updated_at: new Date().toISOString() });
          }
          if (data.active_leaf_id != null) {
            state.activeLeaf = data.active_leaf_id;
            setUserId(data.active_leaf_id); // the new question is the active leaf until the answer lands
          }
          renderSessions();
        }
        break;
      case "status":
        if (data && data.stage && !finished) setStage(data.stage, data.detail);
        break;
      case "warning":
        d.warning = (data && data.message) || "";
        if (d.warning) toast(d.warning, { error: true });
        break;
      case "decision":
        d.decision = data;
        renderDecision(art, data);
        break;
      case "thinking": {
        const t = typeof data === "string" ? data : (data && (data.t ?? data.text ?? data.content)) || "";
        if (!t) break;
        d.thinking += t;
        d.thinkingDirty = true;
        const det = art.querySelector(".reasoning");
        if (det.hidden && !d.reasoningTouched) det.open = true; // show it streaming; folds once the answer starts
        if (state.stage !== "thinking" && !d.content) setStage("thinking");
        scheduleRender(art);
        break;
      }
      case "token": {
        const t = typeof data === "string" ? data : (data && data.t) || "";
        if (!t) break;
        if (!d.content && d.thinking) {
          if (!d.reasoningTouched) art.querySelector(".reasoning").open = false;
          d.thinkingDirty = true;
        }
        d.content += t;
        if (!els.stage.hidden && state.stage !== "generating" && state.stage !== "validating") setStage("generating");
        scheduleRender(art);
        break;
      }
      case "answer":
        // validated + renumbered text replaces what was streamed
        if (data && typeof data.content === "string") { d.content = data.content; scheduleRender(art); }
        if (data && data.uncited) d.uncited = true;
        break;
      case "sources":
        d.sources = Array.isArray(data) ? data : (data && data.sources) || [];
        break;
      case "images":
        d.images = Array.isArray(data) ? data : (data && data.images) || [];
        break;
      case "usage":
        d.usage = data;
        break;
      case "error":
        d.error = (data && (data.message || data.detail)) || "Something went wrong while answering.";
        break;
      case "done":
        finished = true;
        if (data && data.out_of_context) d.ooc = true;
        if (data && data.uncited) d.uncited = true;
        if (data && typeof data.content === "string" && data.content && !d.error) d.content = data.content;
        if (data && data.mode) d.mode = data.mode;
        if (data) setUserId(data.user_message_id);
        if (data && data.message_id != null) {
          d.id = data.message_id;
          d.parentId = data.parent_id ?? data.user_message_id ?? d.parentId;
          d.siblings = [...(ctx.botSiblings || []).filter((x) => x !== d.id), d.id];
          d.sibIndex = d.siblings.length - 1;
          d.sibCount = d.siblings.length;
          art.dataset.mid = String(d.id);
          state.activeLeaf = d.id;
        }
        unlock(); // the UI unlocks now; a generated title may still arrive as a late `session` event
        break;
      default:
        break;
    }
  };

  const req = { session_id: state.sessionId || undefined, mode, ...extra };
  try {
    await streamSSE("/api/chat", req, { signal: controller.signal, onEvent });
    if (!finished && !d.error && !d.content) d.error = "The answer stream ended unexpectedly.";
  } catch (err) {
    if (finished) { /* closed after the answer was complete (new chat / navigation): nothing lost */ }
    else if (err.name === "AbortError") d.stopped = true;
    else d.error = err.message || "Could not reach the server.";
  } finally {
    if (state.controller === controller) state.controller = null;
    unlock();
    loadSessions();
  }
}

/* ============ composer ============ */
function autosize() {
  const ta = els.input;
  ta.style.height = "auto";
  ta.style.height = `${Math.min(ta.scrollHeight, 220)}px`;
}

function updateCount() {
  if (!els.count) return;
  const n = els.input.value.length;
  els.count.textContent = `${fmtInt(n)} / ${fmtInt(MAX_CHARS)}`;
  els.count.classList.toggle("is-near", n >= MAX_CHARS * 0.9 && n < MAX_CHARS);
  els.count.classList.toggle("is-full", n >= MAX_CHARS);
  els.count.setAttribute("aria-label", n >= MAX_CHARS ? `Character limit of ${MAX_CHARS} reached` : `${n} of ${MAX_CHARS} characters`);
}

/* ============ health ============ */
function flag(v) {
  if (v && typeof v === "object") return v.ok !== false && v.status !== "down";
  return v !== false && v !== 0 && v !== "down" && v != null;
}

async function checkHealth() {
  let h;
  const problems = [];
  try {
    h = await api("/api/health");
  } catch {
    problems.push(`<li>The NewRAG server is not answering. Start everything with <code>python run.py</code> in <code>E:\\NewRAG</code>, then reload this page.</li>`);
  }
  if (h) {
    if (!flag(h.postgres)) problems.push(`<li><strong>Postgres is down.</strong> Stop the app and run <code>python run.py</code> again; it starts the embedded database.</li>`);
    if (!flag(h.ollama)) problems.push(`<li><strong>Ollama is not running.</strong> Start it with <code>ollama serve</code> (or rerun <code>python run.py</code>), then reload.</li>`);
    else if (h.chat_model_present === false) problems.push(`<li><strong>The chat model is missing.</strong> Run <code>ollama pull qwen3:4b-instruct-2507-q4_K_M</code>.</li>`);
    if (flag(h.postgres) && (h.chunks === 0 || h.chunks === "0")) problems.push(`<li><strong>The textbook is not indexed yet.</strong> Run <code>python -m app.ingest</code> (or <code>python run.py --ingest</code>).</li>`);
  }
  if (problems.length) {
    els.banner.innerHTML = `<div><strong>Answers are unavailable right now.</strong><ul>${problems.join("")}</ul></div>
      <button type="button" class="btn banner__retry">Check again</button>`;
    els.banner.querySelector(".banner__retry").addEventListener("click", checkHealth);
    els.banner.hidden = false;
  } else {
    els.banner.hidden = true;
  }
}

/* ============ wiring ============ */
function init() {
  els.form.addEventListener("submit", (e) => { e.preventDefault(); send(); });
  els.send.addEventListener("click", (e) => { if (state.streaming) { e.preventDefault(); stopStream(); } });
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
  });
  els.input.addEventListener("input", () => { autosize(); updateCount(); });
  updateCount();
  els.newChat.addEventListener("click", newChat);
  els.open.addEventListener("click", openSidebar);
  els.close.addEventListener("click", closeSidebar);
  els.scrim.addEventListener("click", closeSidebar);
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || e.defaultPrevented) return;
    if (document.querySelector("dialog[open]") || !els.menu.hidden) return;
    closeSidebar();
  });
  window.addEventListener("popstate", () => {
    const id = new URLSearchParams(location.search).get("s");
    if (id) selectSession(id); else resetConversation(null);
  });
  initMenu();
  initDialogs();
  initModes();

  initCiteCard(
    els.messages,
    (chip) => { const art = chip.closest(".msg"); return art && msgData.get(art) ? msgData.get(art).sources : []; },
    (source, chip) => {
      const art = chip.closest(".msg");
      if (art) togglePages(art, source.n);
      openSource(source, 0);
    },
  );

  const ask = new URLSearchParams(location.search).get("q"); // deep link: /?q=question asks it
  loadSessions().then(async () => {
    if (state.sessionId) await selectSession(state.sessionId);
    else showEmpty();
    if (ask) {
      history.replaceState(null, "", state.sessionId ? `/?s=${encodeURIComponent(state.sessionId)}` : "/");
      els.input.value = ask;
      autosize();
      send();
    }
  });
  checkHealth();
  if (matchMedia("(pointer: fine)").matches) els.input.focus();
}

init();
