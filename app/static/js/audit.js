// NewRAG audit page: token usage (interactive charts), answer feedback, architecture, settings.
import { api, toast, escapeHtml, fmtInt } from "./api.js";
import { bars, donut, legend, PALETTE, NEUTRAL } from "./charts.js";
import { renderMarkdown } from "./render.js";
import { initArch } from "./arch.js";

const $ = (id) => document.getElementById(id);
const PURPOSES = ["router", "answer", "memory_compaction", "title", "grounding_check", "feedback_review", "embed_query", "rerank", "ingest_embed"];
const PURPOSE_LABEL = {
  router: "Router", answer: "Answer", memory_compaction: "Memory compaction", title: "Title",
  grounding_check: "Grounding check", feedback_review: "Feedback review", embed_query: "Query embedding",
  rerank: "Rerank", ingest_embed: "Ingest embedding",
};
const humanPurpose = (p) => PURPOSE_LABEL[p] || String(p || "unknown").replace(/_/g, " ");
const MODE_LABEL = { swift: "Swift", deep: "Deep", none: "No mode" };
const MODE_COLOR = { swift: PALETTE[2], deep: PALETTE[6], none: NEUTRAL };
const humanMode = (m) => MODE_LABEL[m] || String(m);
const TOK_SERIES = [
  { key: "p", label: "Prompt", color: "var(--series-prompt)" },
  { key: "c", label: "Completion", color: "var(--series-completion)" },
];
const FB_SERIES = [
  { key: "likes", label: "Likes", color: PALETTE[0] },
  { key: "dislikes", label: "Dislikes", color: PALETTE[1] },
];

/* ================= helpers ================= */
function asRows(x, keyName) {
  if (!x) return [];
  if (Array.isArray(x)) return x;
  if (typeof x === "object") {
    return Object.entries(x).map(([k, v]) => (v && typeof v === "object" ? { [keyName]: k, ...v } : { [keyName]: k, total_tokens: v }));
  }
  return [];
}
function norm(r) {
  const p = +(r.prompt_tokens ?? r.prompt ?? 0) || 0;
  const c = +(r.completion_tokens ?? r.completion ?? 0) || 0;
  const total = +(r.total_tokens ?? r.total ?? r.tokens ?? (p + c)) || 0;
  const calls = +(r.calls ?? r.count ?? r.n ?? 0) || 0;
  let avg = r.avg_ms ?? r.avg_duration_ms ?? r.mean_ms ?? r.avg_latency_ms;
  if (avg == null && r.duration_ms != null && calls) avg = r.duration_ms / calls;
  const failed = +(r.failed ?? r.errors ?? r.error_count ?? 0) || 0;
  return { ...r, p, c, total, calls, avg: avg == null ? null : +avg, failed };
}
const pick = (o, ...keys) => { for (const k of keys) if (o && o[k] != null) return o[k]; return null; };
const utcDay = (iso) => { const d = new Date(iso); return Number.isNaN(+d) ? "" : d.toISOString().slice(0, 10); };
const dayLabel = (key, opts = { month: "short", day: "numeric" }) => new Date(`${key}T00:00:00Z`).toLocaleDateString("en-US", { ...opts, timeZone: "UTC" });
const fmtMs = (ms) => (ms == null ? "–" : ms >= 1000 ? `${(ms / 1000).toFixed(ms >= 10000 ? 0 : 1)} s` : `${fmtInt(ms)} ms`);
function callMode(c) {
  let m = c.mode;
  if (m == null) {
    let meta = c.meta;
    if (typeof meta === "string") { try { meta = JSON.parse(meta); } catch { meta = null; } }
    m = meta && meta.mode;
  }
  return m || "none";
}
function lastDays(n) {
  const out = [];
  const t = new Date();
  for (let i = n - 1; i >= 0; i--) {
    const d = new Date(Date.UTC(t.getUTCFullYear(), t.getUTCMonth(), t.getUTCDate() - i));
    out.push(d.toISOString().slice(0, 10));
  }
  return out;
}
/** Styled confirmation dialog; resolves true/false. */
export function confirmDialog(title, text, okLabel = "Confirm") {
  const dlg = $("confirm-dlg");
  if (!dlg || typeof dlg.showModal !== "function") return Promise.resolve(window.confirm(text));
  if (dlg.open) dlg.close("cancel");
  $("confirm-title").textContent = title;
  $("confirm-text").textContent = text;
  $("confirm-ok").textContent = okLabel;
  dlg.returnValue = "";
  dlg.showModal();
  $("confirm-cancel").focus();
  return new Promise((resolve) => {
    dlg.addEventListener("close", () => resolve(dlg.returnValue === "ok"), { once: true });
  });
}

/* ================= usage ================= */
const U = {
  days: 30,
  summary: null,
  calls: [],
  sort: { key: "created_at", dir: -1 },
  filter: { day: null, purpose: "", model: null, mode: "", lat: null },
  tokHidden: new Set(),
  modeHidden: new Set(),
  charts: {},
};
const LAT_BUCKETS = [
  { key: "b0", label: "<100 ms", lo: 0, hi: 100 }, { key: "b1", label: "100–250 ms", lo: 100, hi: 250 },
  { key: "b2", label: "250–500 ms", lo: 250, hi: 500 }, { key: "b3", label: "0.5–1 s", lo: 500, hi: 1000 },
  { key: "b4", label: "1–2 s", lo: 1000, hi: 2000 }, { key: "b5", label: "2–5 s", lo: 2000, hi: 5000 },
  { key: "b6", label: "5–10 s", lo: 5000, hi: 10000 }, { key: "b7", label: "10–30 s", lo: 10000, hi: 30000 },
  { key: "b8", label: "≥30 s", lo: 30000, hi: Infinity },
];
LAT_BUCKETS.forEach((b) => { b.short = b.label.replace(" ms", "").replace(" s", "s"); });
const bucketOf = (ms) => (ms == null ? null : LAT_BUCKETS.find((b) => ms >= b.lo && ms < b.hi)?.key);

async function loadUsage() {
  $("usage-error").hidden = true;
  $("usage").classList.add("is-loading");
  try {
    const [summary, callData] = await Promise.all([
      api(`/api/audit/summary?days=${encodeURIComponent(U.days)}`),
      api(`/api/audit/calls?limit=2000`),
    ]);
    U.summary = summary || {};
    U.calls = (Array.isArray(callData) ? callData : pick(callData, "calls", "rows", "items") || []).map((c) => {
      const n = norm(c);
      return { ...n, mode: callMode(c), day: utcDay(c.created_at), ms: c.duration_ms == null ? null : +c.duration_ms };
    });
    renderUsage();
  } catch (err) {
    $("usage-error").textContent = `Could not load usage: ${err.message}`;
    $("usage-error").hidden = false;
  } finally {
    $("usage").classList.remove("is-loading");
  }
}

function inRange(c) {
  const since = Date.now() - U.days * 86400e3;
  return +new Date(c.created_at) >= since;
}

function renderUsage() {
  const s = U.summary;
  const byPurpose = asRows(pick(s, "by_purpose", "purposes"), "purpose").map(norm);
  const byModel = asRows(pick(s, "by_model", "models"), "model").map(norm);
  const byDay = asRows(pick(s, "by_day", "daily", "per_day", "days"), "day").map(norm)
    .map((r) => ({ ...r, day: String(r.day ?? r.date ?? "").slice(0, 10) })).filter((r) => r.day);
  const bySession = asRows(pick(s, "by_session", "per_session", "sessions"), "session_id").map(norm);
  let byMode = asRows(pick(s, "by_mode", "modes"), "mode").map(norm);
  if (!byMode.length) { // derive from the call log
    const m = new Map();
    for (const c of U.calls.filter(inRange)) {
      const a = m.get(c.mode) || { mode: c.mode, p: 0, c: 0, total: 0, calls: 0, ms: 0 };
      a.p += c.p; a.c += c.c; a.total += c.total; a.calls += 1; a.ms += c.ms || 0;
      m.set(c.mode, a);
    }
    byMode = [...m.values()].map((a) => ({ ...a, avg: a.calls ? a.ms / a.calls : null }));
  }
  U.byMode = byMode.map((r) => ({ ...r, mode: r.mode || "none" }));

  let t = pick(s, "totals", "total", "overall");
  t = t && typeof t === "object" ? norm(t) : null;
  if (!t || (!t.total && !t.calls)) {
    const agg = byPurpose.reduce((a, r) => ({ p: a.p + r.p, c: a.c + r.c, total: a.total + r.total, calls: a.calls + r.calls, ms: a.ms + (r.avg || 0) * r.calls, failed: a.failed + r.failed }), { p: 0, c: 0, total: 0, calls: 0, ms: 0, failed: 0 });
    t = { ...agg, avg: agg.calls ? agg.ms / agg.calls : null };
  }
  renderKpis(t);
  U.byDay = byDay; U.byPurpose = byPurpose; U.byModel = byModel;
  renderUsageCharts();
  renderSessions(bySession);
  renderPurposeFilter();
  renderCalls();
}

function renderKpis(t) {
  const set = (k, v) => { const el = document.querySelector(`[data-k="${k}"]`); if (el) el.textContent = v; };
  set("total", fmtInt(t.total));
  set("total-sub", U.days === 1 ? "over the last 24 hours" : `over the last ${U.days} days`);
  set("prompt", fmtInt(t.p));
  set("completion", fmtInt(t.c));
  const sum = t.p + t.c || 1;
  document.querySelector('[data-k="split-p"]').style.flexGrow = String(t.p / sum);
  const sc = document.querySelector('[data-k="split-c"]');
  sc.style.flexGrow = String(t.c / sum);
  sc.hidden = !t.c;
  set("calls", fmtInt(t.calls));
  set("calls-sub", t.failed ? `${fmtInt(t.failed)} failed` : "none failed");
  set("latency", fmtMs(t.avg));
}

function toggleTok(key) {
  if (U.tokHidden.has(key)) U.tokHidden.delete(key);
  else if (U.tokHidden.size < TOK_SERIES.length - 1) U.tokHidden.add(key);
  renderUsageCharts();
}

function renderUsageCharts() {
  const hidden = U.tokHidden;
  for (const id of ["day-legend", "purpose-legend", "model-legend"]) legend($(id), { series: TOK_SERIES, hidden, onToggle: toggleTok });
  const tokVal = (rows, key) => (k, sk) => { const r = rows.find((x) => String(x[key]) === String(k)); return r ? (sk === "p" ? r.p || (r.c ? 0 : r.total) : r.c) : 0; };
  const tokFoot = (r) => (r ? `${fmtInt(r.total)} total · ${fmtInt(r.calls)} calls · avg ${fmtMs(r.avg)}` : "");

  // ---- tokens per day
  const days = lastDays(Math.max(2, Math.min(U.days, 90)));
  const dayMap = new Map(U.byDay.map((r) => [r.day, r]));
  for (const r of U.byDay) if (!days.includes(r.day) && r.day >= days[0]) days.push(r.day);
  U.charts.day = bars($("day-chart"), {
    categories: days.map((d) => ({ key: d, label: dayLabel(d, { weekday: "short", month: "short", day: "numeric" }), short: dayLabel(d) })),
    series: TOK_SERIES, hidden,
    value: tokVal(U.byDay, "day"),
    selected: U.filter.day, height: 230,
    tipTitle: (c) => c.label, tipFoot: (c) => `${tokFoot(dayMap.get(c.key)) || "No calls"} · click to filter`,
    onSelect: (k) => setFilter("day", U.filter.day === k ? null : k),
    ariaLabel: "Tokens per day, prompt and completion stacked",
  });
  $("day-table").tBodies[0].innerHTML = days.map((d) => dayMap.get(d)).filter(Boolean).reverse().map((d) =>
    `<tr><td class="num">${escapeHtml(d.day)}</td><td class="r num">${fmtInt(d.p)}</td><td class="r num">${fmtInt(d.c)}</td><td class="r num">${fmtInt(d.total)}</td><td class="r num">${fmtInt(d.calls)}</td></tr>`).join("")
    || `<tr><td colspan="5" class="muted">No calls in this range.</td></tr>`;

  // ---- by purpose
  const purposes = [...U.byPurpose].sort((a, b) => b.total - a.total || PURPOSES.indexOf(a.purpose) - PURPOSES.indexOf(b.purpose));
  U.charts.purpose = bars($("purpose-chart"), {
    orientation: "h", categories: purposes.map((r) => ({ key: r.purpose, label: humanPurpose(r.purpose) })),
    series: TOK_SERIES, hidden, value: tokVal(purposes, "purpose"), labelWidth: 140,
    total: (c) => { const r = purposes.find((x) => x.purpose === c.key); return hidden.size ? (hidden.has("p") ? r.c : r.p || r.total) : r.total; },
    selected: U.filter.purpose || null,
    tipFoot: (c) => `${tokFoot(purposes.find((x) => x.purpose === c.key))} · click to filter`,
    onSelect: (k) => setFilter("purpose", U.filter.purpose === k ? "" : k),
    ariaLabel: "Tokens by purpose", emptyText: "No calls in this range.",
  });

  // ---- by model
  const models = [...U.byModel].sort((a, b) => b.total - a.total);
  U.charts.model = bars($("model-chart"), {
    orientation: "h", categories: models.map((r) => ({ key: r.model || "unknown", label: r.model || "unknown" })),
    series: TOK_SERIES, hidden, value: tokVal(models.map((r) => ({ ...r, model: r.model || "unknown" })), "model"), labelWidth: 200,
    total: (c) => { const r = models.find((x) => (x.model || "unknown") === c.key); return hidden.size ? (hidden.has("p") ? r.c : r.p || r.total) : r.total; },
    selected: U.filter.model,
    tipFoot: (c) => `${tokFoot(models.find((x) => (x.model || "unknown") === c.key))} · click to filter`,
    onSelect: (k) => setFilter("model", U.filter.model === k ? null : k),
    ariaLabel: "Tokens by model", emptyText: "No calls in this range.",
  });

  // ---- by mode (chart + small table)
  const order = ["swift", "deep", "none"];
  const modes = [...U.byMode].sort((a, b) => (order.indexOf(a.mode) + 9) % 9 - (order.indexOf(b.mode) + 9) % 9);
  const box = $("mode-cards");
  if (!box.querySelector(".chart")) box.innerHTML = `<div class="chart" id="mode-chart"></div><table class="tbl tbl--mini" id="mode-table"><thead><tr><th scope="col">Mode</th><th scope="col" class="r">Calls</th><th scope="col" class="r">Tokens</th><th scope="col" class="r">Avg latency</th></tr></thead><tbody></tbody></table>`;
  U.charts.mode = bars($("mode-chart"), {
    orientation: "h", categories: modes.map((r) => ({ key: r.mode, label: humanMode(r.mode) })),
    series: TOK_SERIES, hidden, value: tokVal(modes, "mode"), labelWidth: 90, rowHeight: 30,
    total: (c) => { const r = modes.find((x) => x.mode === c.key); return hidden.size ? (hidden.has("p") ? r.c : r.p || r.total) : r.total; },
    selected: U.filter.mode || null,
    tipFoot: (c) => `${tokFoot(modes.find((x) => x.mode === c.key))} · click to filter`,
    onSelect: (k) => setFilter("mode", U.filter.mode === k ? "" : k),
    ariaLabel: "Tokens by answer mode", emptyText: "No calls in this range.",
  });
  $("mode-table").tBodies[0].innerHTML = modes.map((r) => `<tr><th scope="row"><span class="mode-dot" style="background:${MODE_COLOR[r.mode] || NEUTRAL}"></span>${escapeHtml(humanMode(r.mode))}</th><td class="r num">${fmtInt(r.calls)}</td><td class="r num">${fmtInt(r.total)}</td><td class="r num">${fmtMs(r.avg)}</td></tr>`).join("")
    || `<tr><td colspan="4" class="muted">No calls in this range.</td></tr>`;

  // ---- latency distribution (from the call log, stacked by mode)
  const modeKeys = [...new Set(["swift", "deep", ...U.calls.map((c) => c.mode)])].sort((a, b) => (order.indexOf(a) + 9) % 9 - (order.indexOf(b) + 9) % 9);
  const modeSeries = modeKeys.map((m) => ({ key: m, label: humanMode(m), color: MODE_COLOR[m] || NEUTRAL }));
  legend($("lat-legend"), { series: modeSeries, hidden: U.modeHidden, onToggle: (k) => { if (U.modeHidden.has(k)) U.modeHidden.delete(k); else if (U.modeHidden.size < modeSeries.length - 1) U.modeHidden.add(k); renderUsageCharts(); } });
  const latCalls = U.calls.filter(inRange).filter((c) => (!U.filter.purpose || c.purpose === U.filter.purpose));
  const counts = {};
  for (const c of latCalls) { const b = bucketOf(c.ms); if (!b) continue; counts[b] = counts[b] || {}; counts[b][c.mode] = (counts[b][c.mode] || 0) + 1; }
  U.charts.lat = bars($("lat-chart"), {
    categories: LAT_BUCKETS, series: modeSeries, hidden: U.modeHidden, height: 200, maxBar: 44, xLabelWidth: 40,
    value: (k, m) => (counts[k] && counts[k][m]) || 0, fmt: (n) => `${fmtInt(n)} calls`,
    selected: U.filter.lat, tipFoot: () => `${U.filter.purpose ? `${humanPurpose(U.filter.purpose)} calls · ` : ""}click to filter`,
    onSelect: (k) => setFilter("lat", U.filter.lat === k ? null : k),
    ariaLabel: "Calls per latency bucket, stacked by mode",
  });
}

function renderSessions(rows) {
  const tb = $("session-table").tBodies[0];
  const list = rows.sort((a, b) => b.total - a.total).slice(0, 10);
  tb.innerHTML = list.length ? list.map((r) => {
    const id = r.session_id || "";
    const label = r.title || (id ? `${String(id).slice(0, 8)}…` : "No session");
    return `<tr><td class="ellip"><a href="/?s=${encodeURIComponent(id)}">${escapeHtml(label)}</a></td><td class="r num">${fmtInt(r.total)}</td><td class="r num">${fmtInt(r.calls)}</td><td class="r num">${r.avg == null ? "–" : fmtInt(r.avg)}</td></tr>`;
  }).join("") : `<tr><td colspan="4" class="muted">No conversations in this range.</td></tr>`;
}

function setFilter(k, v) {
  U.filter[k] = v;
  if (k === "purpose") $("f-purpose").value = v || "";
  if (k === "mode") $("f-mode").value = v || "";
  renderUsageCharts();
  renderCalls();
  if (v) $("calls-panel").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderPurposeFilter() {
  const sel = $("f-purpose");
  const cur = U.filter.purpose;
  const seen = [...new Set([...PURPOSES, ...U.calls.map((c) => c.purpose).filter(Boolean)])];
  sel.innerHTML = `<option value="">All purposes</option>` + seen.map((p) => `<option value="${escapeHtml(p)}">${escapeHtml(humanPurpose(p))}</option>`).join("");
  sel.value = cur || "";
}

function fmtTime(iso) {
  const d = new Date(iso);
  if (Number.isNaN(+d)) return escapeHtml(iso || "");
  const today = new Date().toDateString() === d.toDateString();
  return today
    ? d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    : `${d.toLocaleDateString("en-US", { month: "short", day: "numeric" })}, ${d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })}`;
}

function renderChips() {
  const f = U.filter;
  const chips = [];
  if (f.day) chips.push(["day", `Day: ${dayLabel(f.day)}`]);
  if (f.purpose) chips.push(["purpose", `Purpose: ${humanPurpose(f.purpose)}`]);
  if (f.model) chips.push(["model", `Model: ${f.model}`]);
  if (f.mode) chips.push(["mode", `Mode: ${humanMode(f.mode)}`]);
  if (f.lat) chips.push(["lat", `Latency: ${LAT_BUCKETS.find((b) => b.key === f.lat).label}`]);
  const box = $("active-filters");
  box.innerHTML = chips.map(([k, t]) => `<button type="button" class="chip" data-clear="${k}" aria-label="Remove filter ${escapeHtml(t)}">${escapeHtml(t)}<span aria-hidden="true">×</span></button>`).join("")
    + (chips.length > 1 ? `<button type="button" class="chip chip--ghost" data-clear="*">Clear all</button>` : "");
  box.hidden = !chips.length;
  box.querySelectorAll("[data-clear]").forEach((b) => b.addEventListener("click", () => {
    const k = b.dataset.clear;
    if (k === "*") { U.filter = { day: null, purpose: "", model: null, mode: "", lat: null }; $("f-purpose").value = ""; $("f-mode").value = ""; renderUsageCharts(); renderCalls(); }
    else setFilter(k, k === "purpose" || k === "mode" ? "" : null);
  }));
}

function renderCalls() {
  renderChips();
  const q = $("f-text").value.trim().toLowerCase();
  const f = U.filter;
  const failedOnly = $("f-failed").checked;
  const lat = f.lat && LAT_BUCKETS.find((b) => b.key === f.lat);
  let rows = U.calls.filter((c) =>
    (!f.purpose || c.purpose === f.purpose) &&
    (!f.day || c.day === f.day) &&
    (!f.model || (c.model || "unknown") === f.model) &&
    (!f.mode || c.mode === f.mode) &&
    (!lat || (c.ms != null && c.ms >= lat.lo && c.ms < lat.hi && inRange(c))) &&
    (!failedOnly || c.ok === false) &&
    (!q || [c.session_id, c.model, c.purpose, c.provider, c.error, c.mode].some((v) => v && String(v).toLowerCase().includes(q))));
  const { key, dir } = U.sort;
  const val = (c) => {
    if (key === "prompt_tokens") return c.p;
    if (key === "completion_tokens") return c.c;
    if (key === "total_tokens") return c.total;
    if (key === "duration_ms") return c.ms ?? -1;
    if (key === "ok") return c.ok === false ? 0 : 1;
    if (key === "created_at") return +new Date(c.created_at) || 0;
    return String(c[key] ?? "");
  };
  rows = rows.sort((a, b) => { const x = val(a), y = val(b); return (x > y ? 1 : x < y ? -1 : 0) * dir; });
  const shown = rows.slice(0, 500);
  $("calls-table").tBodies[0].innerHTML = shown.length ? shown.map((c) => `
    <tr class="${c.ok === false ? "is-failed" : ""}">
      <td class="num nowrap">${fmtTime(c.created_at)}</td>
      <td class="nowrap">${c.session_id ? `<a href="/?s=${encodeURIComponent(c.session_id)}" title="${escapeHtml(c.session_id)}">${escapeHtml(String(c.session_id).slice(0, 8))}</a>` : `<span class="muted">none</span>`}</td>
      <td class="nowrap">${escapeHtml(humanPurpose(c.purpose))}</td>
      <td class="nowrap">${c.mode === "none" ? `<span class="muted">–</span>` : `<span class="mode-dot" style="background:${MODE_COLOR[c.mode] || NEUTRAL}"></span>${escapeHtml(humanMode(c.mode))}`}</td>
      <td class="ellip" title="${escapeHtml(c.model || "")}">${escapeHtml(c.model || "")}</td>
      <td class="r num">${fmtInt(c.p)}</td>
      <td class="r num">${fmtInt(c.c)}</td>
      <td class="r num">${fmtInt(c.total)}</td>
      <td class="r num">${c.ms == null ? "–" : fmtInt(c.ms)}</td>
      <td>${c.ok === false ? `<span class="bad" title="${escapeHtml(c.error || "Failed")}">Failed</span>` : `<span class="good">OK</span>`}</td>
    </tr>`).join("") : `<tr><td colspan="10" class="muted">No calls match these filters.</td></tr>`;
  $("calls-foot").textContent = `${fmtInt(rows.length)} of ${fmtInt(U.calls.length)} most recent calls${rows.length > shown.length ? ` (first ${shown.length} shown)` : ""}`;
  document.querySelectorAll("#calls-table th[data-sort]").forEach((th) => {
    th.setAttribute("aria-sort", th.dataset.sort === key ? (dir === 1 ? "ascending" : "descending") : "none");
  });
}

/* ================= feedback ================= */
const F = { summary: null, items: [], filter: { day: null, mode: null, intent: null, issue: null }, hidden: new Set(), loaded: false };
const fbDay = (i) => utcDay(i.updated_at || i.created_at);
const ISSUE_LABEL = { wrong_fact: "Wrong fact", missing_citation: "Missing citation", bad_format: "Bad format", refused_wrongly: "Refused wrongly", incomplete: "Incomplete", other: "Other" };

async function loadFeedback() {
  $("fb-error").hidden = true;
  $("feedback").classList.add("is-loading");
  try {
    const [summary, list] = await Promise.all([api("/api/feedback/summary"), api("/api/feedback?rating=-1&limit=200")]);
    F.summary = summary || {};
    F.items = Array.isArray(list) ? list : pick(list, "items", "feedback", "rows") || [];
    F.loaded = true;
    renderFeedback();
  } catch (err) {
    $("fb-error").textContent = err.status === 404 ? "Feedback endpoints are not available on this server yet." : `Could not load feedback: ${err.message}`;
    $("fb-error").hidden = false;
  } finally {
    $("feedback").classList.remove("is-loading");
  }
}

function fbCounts(r) { return { likes: +(r.likes ?? r.up ?? 0) || 0, dislikes: +(r.dislikes ?? r.down ?? 0) || 0 }; }

function renderFeedback() {
  const s = F.summary;
  const items = F.items;
  const tot = s.totals && typeof s.totals === "object" ? s.totals : s;
  const likes = +(tot.likes ?? 0), dislikes = +(tot.dislikes ?? items.length);
  const unreviewed = s.unreviewed_dislikes != null ? +s.unreviewed_dislikes : items.filter((i) => !i.review).length;
  const reviewed = s.reviewed ?? Math.max(0, dislikes - unreviewed);
  const set = (k, v) => { const el = document.querySelector(`[data-f="${k}"]`); if (el) el.textContent = v; };
  set("likes", fmtInt(likes));
  set("dislikes", fmtInt(dislikes));
  const total = likes + dislikes;
  set("ratio-sub", total ? `${fmtInt(total)} rated answers${tot.rated_share != null ? ` · ${Math.round(tot.rated_share * 100)}% of all answers` : ""}` : "No ratings yet");
  set("reviewed", fmtInt(reviewed));
  set("reviewed-sub", unreviewed ? `${fmtInt(unreviewed)} waiting for review` : dislikes ? "all reviewed" : "");
  donut($("fb-gauge"), {
    size: 84, thickness: 9,
    segments: [{ key: "likes", label: "Likes", value: likes, color: FB_SERIES[0].color }, { key: "dislikes", label: "Dislikes", value: dislikes, color: FB_SERIES[1].color }],
    center: total ? `${Math.round((likes / total) * 100)}%` : "–", ariaLabel: "Share of liked answers",
  });

  legend($("fb-day-legend"), { series: FB_SERIES, hidden: F.hidden, onToggle: (k) => { if (F.hidden.has(k)) F.hidden.delete(k); else if (!F.hidden.size) F.hidden.add(k); renderFeedback(); } });
  const byDay = asRows(pick(s, "by_day", "daily"), "day").map((r) => ({ day: String(r.day ?? r.date).slice(0, 10), ...fbCounts(r) }));
  const days = lastDays(Math.max(7, Math.min(+(s.days || 30), 90)));
  const dm = new Map(byDay.map((r) => [r.day, r]));
  bars($("fb-day-chart"), {
    categories: days.map((d) => ({ key: d, label: dayLabel(d, { weekday: "short", month: "short", day: "numeric" }), short: dayLabel(d) })),
    series: FB_SERIES, hidden: F.hidden, negative: new Set(["dislikes"]), height: 200,
    value: (k, sk) => (dm.get(k) || {})[sk] || 0, selected: F.filter.day,
    tipFoot: () => "Click to list that day's dislikes",
    onSelect: (k) => setFbFilter("day", F.filter.day === k ? null : k),
    ariaLabel: "Likes and dislikes per day",
  });
  const breakdown = (el, rows, key, labelFn) => {
    const data = rows.map((r) => ({ key: String(r[key] ?? "none"), ...fbCounts(r) })).sort((a, b) => (b.likes + b.dislikes) - (a.likes + a.dislikes));
    bars(el, {
      orientation: "h", categories: data.map((r) => ({ key: r.key, label: labelFn(r.key) })), series: FB_SERIES, labelWidth: 120,
      value: (k, sk) => (data.find((r) => r.key === k) || {})[sk] || 0,
      total: (c) => { const r = data.find((x) => x.key === c.key); return r.likes + r.dislikes; },
      totalFmt: (n, c) => { const r = data.find((x) => x.key === c.key); return `${n} · ${n ? Math.round((r.likes / n) * 100) : 0}% liked`; },
      selected: F.filter[key], tipFoot: () => "Click to filter the disliked answers",
      onSelect: (k) => setFbFilter(key, F.filter[key] === k ? null : k),
      ariaLabel: `Feedback by ${key}`, emptyText: "No ratings yet.",
    });
  };
  breakdown($("fb-mode-chart"), asRows(pick(s, "by_mode"), "mode"), "mode", humanMode);
  breakdown($("fb-intent-chart"), asRows(pick(s, "by_intent"), "intent"), "intent", (k) => k.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase()));
  let issues = asRows(pick(s, "by_issue_type"), "issue_type").map((r) => ({ key: String(r.issue_type ?? "other"), n: +(r.n ?? r.count ?? 0) || 0 }));
  if (!issues.length) {
    const m = new Map();
    for (const i of F.items) if (i.review) { const k = i.review.issue_type || "other"; m.set(k, (m.get(k) || 0) + 1); }
    issues = [...m].map(([key, n]) => ({ key, n }));
  }
  issues.sort((a, b) => b.n - a.n);
  bars($("fb-issue-chart"), {
    orientation: "h", categories: issues.map((r) => ({ key: r.key, label: ISSUE_LABEL[r.key] || r.key })), labelWidth: 130,
    series: [{ key: "n", label: "Reviewed dislikes", color: PALETTE[1] }], value: (k) => (issues.find((r) => r.key === k) || {}).n || 0,
    totalFmt: (n) => String(n), selected: F.filter.issue, tipFoot: () => "Click to filter the disliked answers",
    onSelect: (k) => setFbFilter("issue", F.filter.issue === k ? null : k),
    ariaLabel: "Reviewed dislikes by issue type", emptyText: "No reviews yet. Use Review dislikes.",
  });
  renderFbList();
}

function setFbFilter(k, v) {
  F.filter[k] = v;
  renderFeedback();
  if (v) $("fb-list").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderFbList() {
  const f = F.filter;
  const unrev = $("fb-unreviewed").checked;
  const rows = F.items.filter((i) => (!f.day || fbDay(i) === f.day) && (!f.mode || (i.mode || "none") === f.mode)
    && (!f.intent || (i.intent || "unknown") === f.intent) && (!f.issue || (i.review && (i.review.issue_type || "other") === f.issue)) && (!unrev || !i.review));
  $("fb-count").textContent = `${fmtInt(rows.length)}${rows.length !== F.items.length ? ` of ${fmtInt(F.items.length)}` : ""}`;
  const chips = [];
  if (f.day) chips.push(["day", `Day: ${dayLabel(f.day)}`]);
  if (f.mode) chips.push(["mode", `Mode: ${humanMode(f.mode)}`]);
  if (f.intent) chips.push(["intent", `Intent: ${f.intent}`]);
  if (f.issue) chips.push(["issue", `Issue: ${ISSUE_LABEL[f.issue] || f.issue}`]);
  const cb = $("fb-filters");
  cb.innerHTML = chips.map(([k, t]) => `<button type="button" class="chip" data-clear="${k}" aria-label="Remove filter ${escapeHtml(t)}">${escapeHtml(t)}<span aria-hidden="true">×</span></button>`).join("");
  cb.hidden = !chips.length;
  cb.querySelectorAll("[data-clear]").forEach((b) => b.addEventListener("click", () => setFbFilter(b.dataset.clear, null)));

  const list = $("fb-list");
  if (!rows.length) { list.innerHTML = `<li class="muted fb-empty">${F.items.length ? "No disliked answers match these filters." : "No disliked answers. Readers have not disliked anything yet."}</li>`; return; }
  list.innerHTML = "";
  for (const it of rows) {
    const li = document.createElement("li");
    li.className = "fb";
    li.dataset.id = it.message_id;
    const when = new Date(it.updated_at || it.created_at);
    const sources = Array.isArray(it.sources) ? it.sources : [];
    li.innerHTML = `
      <div class="fb__meta">
        <span class="num">${escapeHtml(Number.isNaN(+when) ? "" : when.toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }))}</span>
        ${it.mode ? `<span class="tag"><span class="mode-dot" style="background:${MODE_COLOR[it.mode] || NEUTRAL}"></span>${escapeHtml(humanMode(it.mode))}</span>` : ""}
        ${it.intent ? `<span class="tag">${escapeHtml(it.intent)}</span>` : ""}
        ${it.out_of_context ? `<span class="tag">#OutofContext</span>` : ""}
        ${it.session_id ? `<a class="fb__open" href="/?s=${encodeURIComponent(it.session_id)}" title="${escapeHtml(it.session_title || "")}">Open conversation</a>` : ""}
      </div>
      <p class="fb__q"><span class="fb__lbl">Question</span>${escapeHtml(it.question || "(question not recorded)")}</p>
      ${it.comment ? `<p class="fb__comment"><span class="fb__lbl">Reader comment</span>${escapeHtml(it.comment)}</p>` : ""}
      <div class="fb__answer"><span class="fb__lbl">Answer</span><div class="fb__prose prose"></div></div>
      <button type="button" class="linkish fb__more" aria-expanded="false">Show full answer</button>
      ${sources.length ? `<details class="fb__sources"><summary>${sources.length} source${sources.length === 1 ? "" : "s"}</summary><ol>${sources.map((s) => `<li value="${escapeHtml(s.n ?? "")}"><span class="fb__src-h">${escapeHtml(s.header || s.doc_title || s.chunk_id || "")}</span> <span class="muted num">p. ${escapeHtml((s.pages || []).join(", "))}${s.score != null ? ` · score ${(+s.score).toFixed(2)}` : ""}</span></li>`).join("")}</ol></details>` : `<p class="muted fb__nosrc">No sources were shown with this answer.</p>`}
      <div class="fb__review"></div>`;
    renderMarkdown(li.querySelector(".fb__prose"), it.answer || "");
    const more = li.querySelector(".fb__more");
    more.addEventListener("click", () => {
      const open = li.classList.toggle("is-open");
      more.setAttribute("aria-expanded", String(open));
      more.textContent = open ? "Show less" : "Show full answer";
    });
    renderReview(li.querySelector(".fb__review"), it);
    list.append(li);
    requestAnimationFrame(() => { const p = li.querySelector(".fb__prose"); more.hidden = p.scrollHeight <= p.clientHeight + 4 && !li.classList.contains("is-open"); });
  }
}

function renderReview(box, it) {
  const r = it.review ? { model: it.review_model, created_at: it.reviewed_at, ...it.review } : null;
  if (!r) {
    box.innerHTML = `<div class="review review--none"><span class="muted">Not reviewed yet.</span> <button type="button" class="btn btn--sm" data-review-one>Review this answer</button></div>`;
    box.querySelector("[data-review-one]").addEventListener("click", (e) => runReview([it.message_id], e.currentTarget));
    return;
  }
  const issue = r.issue_type || "other";
  box.innerHTML = `<div class="review">
      <p class="review__head"><span class="issue issue--${escapeHtml(issue)}">${escapeHtml(ISSUE_LABEL[issue] || issue)}</span>
        <span class="muted">LLM review${r.model ? ` · ${escapeHtml(r.model)}` : ""}${r.created_at ? ` · ${escapeHtml(new Date(r.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric" }))}` : ""}</span></p>
      <p class="review__exp">${escapeHtml(r.explanation || "")}</p>
      ${r.suggested_fix ? `<p class="review__fix"><span class="fb__lbl">Suggested fix</span>${escapeHtml(r.suggested_fix)}</p>` : ""}
    </div>`;
}

async function runReview(ids, btn) {
  const main = $("fb-review");
  const buttons = [main, btn].filter(Boolean);
  const labels = buttons.map((b) => b.textContent);
  buttons.forEach((b) => { b.disabled = true; b.textContent = "Reviewing…"; });
  try {
    const res = await api("/api/feedback/review", { method: "POST", body: ids ? { message_ids: ids } : {} });
    const reviews = Array.isArray(res) ? res : pick(res, "reviews", "items") || [];
    let n = 0, failed = 0;
    for (const rv of reviews) {
      if (!rv.review && (rv.error || rv.review === null)) { failed++; continue; }
      const it = F.items.find((i) => String(i.message_id) === String(rv.message_id));
      if (!it) continue;
      const review = rv.review && typeof rv.review === "object" ? { ...rv.review } : { ...rv };
      if (rv.model && !review.model) review.model = rv.model;
      if (rv.created_at && !review.created_at) review.created_at = rv.created_at;
      it.review = review;
      n++;
    }
    toast(n ? `Reviewed ${n} disliked answer${n === 1 ? "" : "s"}${failed ? ` · ${failed} failed` : ""}` : failed ? `Review failed for ${failed} answer${failed === 1 ? "" : "s"}` : "Nothing new to review", { error: !n && failed > 0 });
    try { F.summary = (await api("/api/feedback/summary")) || F.summary; } catch { /* keep */ }
    renderFeedback();
  } catch (err) {
    toast(`Review failed: ${err.message}`, { error: true });
  } finally {
    buttons.forEach((b, i) => { if (b.isConnected) { b.disabled = false; b.textContent = labels[i]; } });
  }
}

/* ================= settings ================= */
let settingsOriginal = {};
let settingLimits = {};
const SETTING_HINTS = {
  chat_model: "Ollama model used to write answers.",
  router_model: "Ollama model for the routing decision.",
  num_ctx: "Context window in tokens passed to Ollama.",
  temperature: "0 is deterministic; keep low for faithful answers.",
  max_answer_tokens: "Upper bound on answer length.",
  top_k_dense: "Candidates from vector search.",
  top_k_bm25: "Candidates from keyword search.",
  rrf_k: "Reciprocal-rank-fusion constant.",
  rerank_top_n: "Candidates sent to the reranker.",
  final_k: "Passages given to the model.",
  ooc_threshold: "Below this best rerank score the reply is #OutofContext.",
  ooc_soft_threshold: "Used when the router also judges the question out of scope.",
  memory_budget_tokens: "Session memory is compacted above this size.",
  memory_keep_turns: "Recent turns kept verbatim after compaction.",
  grounding_check: "Run an extra model pass to verify the answer against its sources.",
  embed_device: "cpu or cuda.", rerank_device: "cpu or cuda.",
};
const GROUPS = [
  ["Models", (k) => /model|ollama|device/.test(k)],
  ["Generation", (k) => /num_ctx|temperature|answer|grounding/.test(k)],
  ["Retrieval", (k) => /top_k|rrf|rerank|final_k|bm25|dense|route/.test(k)],
  ["Out-of-context thresholds", (k) => /threshold|ooc/.test(k)],
  ["Memory", (k) => /memory/.test(k)],
  ["Other", () => true],
];
const SETTING_LABELS = {
  num_ctx: "Context size", top_k_dense: "Dense top-k", top_k_bm25: "BM25 top-k", rrf_k: "RRF k",
  rerank_top_n: "Rerank top-n", final_k: "Passages per answer", ooc_threshold: "Hard threshold",
  ooc_soft_threshold: "Soft threshold", max_answer_tokens: "Max answer tokens",
};
const humanKey = (k) => SETTING_LABELS[k] || k.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

async function loadSettings() {
  $("settings-error").hidden = true;
  try {
    const data = await api("/api/settings");
    try { settingLimits = (await api("/api/settings/limits")) || {}; } catch { settingLimits = {}; }
    const s = data && typeof data.settings === "object" && !Array.isArray(data.settings) ? data.settings : data;
    settingsOriginal = s || {};
    renderSettings(settingsOriginal);
  } catch (err) {
    $("settings-fields").innerHTML = "";
    $("settings-error").textContent = `Could not load settings: ${err.message}`;
    $("settings-error").hidden = false;
  }
}

function fieldFor(key, value) {
  const id = `s-${key}`;
  const hint = SETTING_HINTS[key] ? `<p class="hint" id="${id}-h">${escapeHtml(SETTING_HINTS[key])}</p>` : "";
  const desc = SETTING_HINTS[key] ? ` aria-describedby="${id}-h"` : "";
  const label = `<label for="${id}">${escapeHtml(humanKey(key))} <span class="key">${escapeHtml(key)}</span></label>`;
  if (typeof value === "boolean") {
    return `<div class="field field--bool"><label class="switch" for="${id}"><input type="checkbox" id="${id}" data-key="${escapeHtml(key)}" data-type="bool"${value ? " checked" : ""}${desc}><span class="switch__track" aria-hidden="true"></span><span>${escapeHtml(humanKey(key))} <span class="key">${escapeHtml(key)}</span></span></label>${hint}</div>`;
  }
  if (typeof value === "number") {
    const isInt = Number.isInteger(value) && !/threshold|temperature|score/.test(key);
    const lim = settingLimits.ranges && settingLimits.ranges[key];
    const mm = Array.isArray(lim) ? ` min="${escapeHtml(lim[0])}" max="${escapeHtml(lim[1])}"` : "";
    return `<div class="field">${label}<input class="input num" type="number" inputmode="${isInt ? "numeric" : "decimal"}" step="${isInt ? "1" : "any"}"${mm} id="${id}" data-key="${escapeHtml(key)}" data-type="${isInt ? "int" : "float"}" value="${escapeHtml(value)}"${desc}>${hint}<p class="field-error" id="${id}-e" role="alert" hidden></p></div>`;
  }
  if (value === null || typeof value === "string") {
    const choices = settingLimits.choices && settingLimits.choices[key];
    if (Array.isArray(choices) && choices.length) {
      return `<div class="field">${label}<select class="select" id="${id}" data-key="${escapeHtml(key)}" data-type="${value === null ? "nullable" : "str"}"${desc}>${choices.map((c) => `<option${c === value ? " selected" : ""}>${escapeHtml(c)}</option>`).join("")}</select>${hint}</div>`;
    }
    return `<div class="field">${label}<input class="input" type="text" id="${id}" data-key="${escapeHtml(key)}" data-type="${value === null ? "nullable" : "str"}" value="${escapeHtml(value ?? "")}" spellcheck="false"${desc}>${hint}</div>`;
  }
  return `<div class="field field--wide">${label}<textarea class="textarea mono" rows="4" id="${id}" data-key="${escapeHtml(key)}" data-type="json" spellcheck="false"${desc}>${escapeHtml(JSON.stringify(value, null, 2))}</textarea>${hint}<p class="field-error" id="${id}-e" role="alert" hidden></p></div>`;
}

function renderSettings(s) {
  const keys = Object.keys(s);
  if (!keys.length) { $("settings-fields").innerHTML = `<p class="muted">The server returned no editable settings.</p>`; return; }
  const used = new Set();
  let html = "";
  for (const [name, test] of GROUPS) {
    const ks = keys.filter((k) => !used.has(k) && test(k));
    if (!ks.length) continue;
    ks.forEach((k) => used.add(k));
    html += `<fieldset class="group"><legend>${escapeHtml(name)}</legend><div class="group__fields">${ks.map((k) => fieldFor(k, s[k])).join("")}</div></fieldset>`;
  }
  $("settings-fields").innerHTML = html;
  $("settings-msg").textContent = "";
}

function fieldError(el, text) {
  el.setAttribute("aria-invalid", text ? "true" : "false");
  const box = document.getElementById(`${el.id}-e`);
  if (box) { box.textContent = text || ""; box.hidden = !text; }
}

function readNumber(el, k, isInt) {
  const raw = el.value.trim();
  if (raw === "" || (el.validity && el.validity.badInput)) return { error: `${k} needs a number` };
  const v = Number(raw);
  if (!Number.isFinite(v)) return { error: `${k} must be a number` };
  if (isInt && !Number.isInteger(v)) return { error: `${k} must be a whole number` };
  const lim = settingLimits.ranges && settingLimits.ranges[k];
  if (Array.isArray(lim) && (v < lim[0] || v > lim[1])) return { error: `${k} must be between ${lim[0]} and ${lim[1]}` };
  return { value: v };
}

function readSettings() {
  const out = {};
  const errors = [];
  document.querySelectorAll("#settings-fields [data-key]").forEach((el) => {
    const k = el.dataset.key;
    let v;
    let err = "";
    switch (el.dataset.type) {
      case "bool": v = el.checked; break;
      case "int":
      case "float": {
        const r = readNumber(el, k, el.dataset.type === "int");
        if (r.error) err = r.error; else v = r.value;
        break;
      }
      case "json": try { v = JSON.parse(el.value); } catch { err = `${k} is not valid JSON`; } break;
      case "nullable": v = el.value === "" ? null : el.value; break;
      default: v = el.value;
    }
    fieldError(el, err);
    if (err) { errors.push(err); return; }
    if (JSON.stringify(v) !== JSON.stringify(settingsOriginal[k])) out[k] = v;
  });
  return { changes: out, errors };
}

function markServerErrors(message) {
  document.querySelectorAll("#settings-fields [data-key]").forEach((el) => {
    const k = el.dataset.key;
    const part = String(message || "").split(/;\s*/).find((p) => new RegExp(`(^|[^a-z_])${k}([^a-z_]|$)`).test(p.replace(/^Invalid value:\s*/, "")));
    if (part) fieldError(el, part.replace(/^Invalid value:\s*/, ""));
  });
}

async function saveSettings(e) {
  e.preventDefault();
  const msg = $("settings-msg");
  const { changes, errors } = readSettings();
  if (errors.length) {
    msg.className = "form-msg is-error";
    msg.textContent = `Not saved: ${errors.length === 1 ? errors[0] : `${errors.length} fields need fixing`}.`;
    document.querySelector('#settings-fields [aria-invalid="true"]')?.focus();
    return;
  }
  if (!Object.keys(changes).length) { msg.className = "form-msg"; msg.textContent = "Nothing changed."; return; }
  const btn = $("settings-save");
  btn.disabled = true; btn.textContent = "Saving…";
  try {
    const res = await api("/api/settings", { method: "PUT", body: changes });
    const s = res && typeof res === "object" ? (res.settings && typeof res.settings === "object" ? res.settings : res) : null;
    settingsOriginal = s && Object.keys(s).length > 1 ? s : { ...settingsOriginal, ...changes };
    renderSettings(settingsOriginal);
    msg.className = "form-msg is-ok";
    msg.textContent = `Saved ${Object.keys(changes).length} setting${Object.keys(changes).length === 1 ? "" : "s"}.`;
  } catch (err) {
    msg.className = "form-msg is-error";
    msg.textContent = `Not saved: ${err.message}`;
    markServerErrors(err.message);
  } finally {
    btn.disabled = false; btn.textContent = "Save settings";
  }
}

/* ================= wiring ================= */
const TABS = ["usage", "feedback", "architecture", "settings"];
let arch = null;
const loaded = new Set();

function showTab() {
  let h = (location.hash || "#usage").slice(1);
  if (h === "keys") h = "settings"; // old bookmark: the API-keys section was removed
  if (!TABS.includes(h)) h = "usage";
  for (const t of TABS) $(t).hidden = t !== h;
  document.querySelectorAll(".tab").forEach((a) => {
    if (a.getAttribute("href") === `#${h}`) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  if (!loaded.has(h)) {
    loaded.add(h);
    if (h === "usage") loadUsage();
    if (h === "feedback") loadFeedback();
    if (h === "architecture") { arch = arch || initArch({ confirmDialog }); arch.load(); }
    if (h === "settings") loadSettings();
  }
  window.scrollTo(0, 0);
}

function init() {
  $("days").addEventListener("click", (e) => {
    const b = e.target.closest("[data-days]");
    if (!b) return;
    U.days = +b.dataset.days;
    $("days").querySelectorAll("[data-days]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    U.filter.day = null; U.filter.lat = null;
    loadUsage();
  });
  $("refresh").addEventListener("click", loadUsage);
  $("f-text").addEventListener("input", renderCalls);
  $("f-purpose").addEventListener("change", (e) => setFilter("purpose", e.target.value));
  $("f-mode").addEventListener("change", (e) => setFilter("mode", e.target.value));
  $("f-failed").addEventListener("change", renderCalls);
  document.querySelectorAll("#calls-table th[data-sort] button").forEach((b) => b.addEventListener("click", () => {
    const key = b.parentElement.dataset.sort;
    U.sort = U.sort.key === key ? { key, dir: -U.sort.dir } : { key, dir: /tokens|duration|created/.test(key) ? -1 : 1 };
    renderCalls();
  }));
  $("fb-refresh").addEventListener("click", loadFeedback);
  $("fb-review").addEventListener("click", () => runReview(null, null));
  $("fb-unreviewed").addEventListener("change", renderFbList);
  $("settings-form").addEventListener("submit", saveSettings);
  $("settings-reset").addEventListener("click", () => { renderSettings(settingsOriginal); $("settings-msg").textContent = "Changes discarded."; });

  window.addEventListener("hashchange", showTab);
  showTab();
}


init();
