// Evaluation dashboard: runs, scenario x metric heatmap, run comparison, item drill-down, background jobs.
import { api, toast, escapeHtml } from "./api.js";
import { renderMarkdown, OOC } from "./render.js";
import { showTip as cvTip, hideTip as cvHide, legend as cvLegend, PALETTE, niceMax } from "./charts.js";

const $ = (id) => document.getElementById(id);
const esc = escapeHtml;

const state = {
  meta: null, runs: [], sel: null, ticked: [], view: "scores",
  data: null,          // {run, aggregate}
  cmp: null,           // /compare payload (a = baseline, b = selected)
  items: [], selItem: null, poll: null, activeRow: null,
  cmpItems: [],        // items of the compared run (A) for the latency overlay
  confSel: null,       // {matrix, cell}
  avgMode: "micro", latKey: "latency_ms", latRange: null, zoom: null,
  hidden: { scen: new Set(), metric: new Set(), lat: new Set(), scatter: new Set() },
};

const LIVE = new Set(["created", "running", "judging", "cancelling"]);
const SCENARIO_LABEL = (s) => s === "__all__" ? "All scenarios" : s.replace(/_/g, " ").replace(/\bocc\b/, "OOC")
  .replace(/^./, (c) => c.toUpperCase()).replace("Out of context", "Out of context");

// ---------------------------------------------------------------- colour
const SEQ = ["#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#cde2fb"];   // validated (dark surface)
const DIV = { neg: "#E66B52", mid: "#383835", pos: "#5598E7" };
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const mix = (a, b, t) => { const x = hex(a), y = hex(b); return `rgb(${x.map((v, i) => Math.round(v + (y[i] - v) * t)).join(" ")})`; };
const lum = (rgb) => { const [r, g, b] = rgb.match(/\d+/g).map(Number); return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255; };
function seqColor(t) {
  t = Math.max(0, Math.min(1, t));
  const p = t * (SEQ.length - 1), i = Math.min(SEQ.length - 2, Math.floor(p));
  return mix(SEQ[i], SEQ[i + 1], p - i);
}
function divColor(t) { // t in [-1, 1]; positive = better
  t = Math.max(-1, Math.min(1, t));
  return t >= 0 ? mix(DIV.mid, DIV.pos, t) : mix(DIV.mid, DIV.neg, -t);
}

// ---------------------------------------------------------------- formatting
const f1 = (v) => (v == null ? "–" : (+v).toFixed(1));
const f2 = (v) => (v == null ? "–" : (+v).toFixed(2));
const pc = (v) => (v == null ? "–" : `${Math.round(v * 100)}%`);
const secs = (ms) => (ms == null ? "–" : `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)} s`);
const sign = (v, fmt) => (v == null ? "" : `${v > 0 ? "+" : v < 0 ? "−" : "±"}${fmt(Math.abs(v))}`);
const when = (iso) => { if (!iso) return "–"; const d = new Date(iso); return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); };

// ---------------------------------------------------------------- runs
async function loadRuns() {
  try {
    state.runs = await api("/api/eval/runs");
    $("runs-error").hidden = true;
  } catch (e) {
    $("runs-error").textContent = e.message; $("runs-error").hidden = false; return;
  }
  renderRuns();
  renderJob();
  if (state.sel == null) {
    const want = +new URLSearchParams(location.search).get("run");
    const first = state.runs.find((r) => r.run_id === want) || state.runs.find((r) => r.summary?.overall?.n) || state.runs[0];
    if (first) {
      await selectRun(first.run_id);
      const qs = new URLSearchParams(location.search);
      const cmp = +qs.get("cmp");
      if (cmp && cmp !== first.run_id && state.runs.some((r) => r.run_id === cmp)) { state.ticked = [cmp, first.run_id]; await updateCompare(); }
      const it = qs.get("item");
      if (it) showItem(it);
    }
  }
}

function renderRuns() {
  const tb = $("runs-table").tBodies[0];
  if (!state.runs.length) {
    tb.innerHTML = `<tr><td colspan="12" class="muted">No runs yet. Start one with the button above or <code>python -m eval.run --label baseline</code>.</td></tr>`;
    return;
  }
  tb.innerHTML = state.runs.map((r) => {
    const o = r.summary?.overall || {};
    const prog = r.progress || {};
    const live = LIVE.has(r.status);
    const st = live && prog.total ? `${r.status} ${prog.done ?? 0}/${prog.total}` : r.status;
    const ticked = state.ticked.includes(r.run_id);
    const role = state.cmp && ticked ? (state.cmp.a.run.run_id === r.run_id ? "A" : "B") : "";
    return `<tr data-id="${r.run_id}" class="${r.run_id === state.sel ? "is-selected" : ""}">
      <td><input type="checkbox" aria-label="Compare run ${esc(r.label)}" ${ticked ? "checked" : ""} data-tick="${r.run_id}"></td>
      <td><span class="run-label">${esc(r.label)}</span><span class="run-id">#${r.run_id}</span>${role ? `<span class="cmp-tag">${role}</span>` : ""}</td>
      <td><span class="badge badge--${esc(r.status)}">${esc(st)}</span></td>
      <td class="r">${o.n ?? 0}/${r.n_items ?? "–"}</td>
      <td class="r">${o.judged ?? 0}</td>
      <td class="r">${o.judged ? f1(o.score) : `<span class="muted" title="Not judged yet">${f1(o.score)}*</span>`}</td>
      <td class="r">${pc(o.pass_rate)}</td>
      <td class="r">${f2(r.summary?.ooc_f1)}</td>
      <td class="r">${secs(r.summary?.latency_p50)}</td>
      <td>${r.judge_model ? `${esc(r.judge_model)}${r.judge_agreement != null ? ` <span class="muted">· cal ${pc(r.judge_agreement)}</span>` : ""}` : `<span class="muted">–</span>`}</td>
      <td class="nowrap">${when(r.created_at)}</td>
      <td class="nowrap">${!live && o.n ? `<button type="button" class="btn btn--sm" data-judge="${r.run_id}">${o.judged ? "Re-judge" : "Judge"}</button>` : ""}
        ${live ? "" : `<button type="button" class="icon-btn" data-del="${r.run_id}" aria-label="Delete run ${esc(r.label)}"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.6 8.5h5.8l.6-8.5"/></svg></button>`}</td>
    </tr>`;
  }).join("");
  $("runs-foot").textContent = state.ticked.length === 2 ? "Comparing the two ticked runs (A = older, B = newer)." : "Tick two runs to compare them.";
}

async function selectRun(id) {
  state.sel = id;
  state.selItem = null;
  renderRuns();
  try {
    state.data = await api(`/api/eval/runs/${id}`);
  } catch (e) { toast(e.message, { error: true }); return; }
  if (state.cmp && state.cmp.b.run.run_id !== id) { state.cmp = null; state.cmpItems = []; state.ticked = []; renderRuns(); }
  state.confSel = null; state.latRange = null; state.zoom = null;
  renderMatrix();
  await loadItems();
}

async function updateCompare() {
  if (state.ticked.length !== 2) {
    state.cmp = null; state.cmpItems = [];
    if (state.view === "delta") state.view = "scores";
    renderMatrix(); renderItems(); renderRuns();
    return;
  }
  const [a, b] = [...state.ticked].sort((x, y) => x - y);
  try {
    state.cmp = await api(`/api/eval/compare?a=${a}&b=${b}`);
  } catch (e) { toast(e.message, { error: true }); return; }
  state.view = "delta";
  try { state.cmpItems = await api(`/api/eval/runs/${a}/items`); } catch { state.cmpItems = []; }
  if (state.sel !== b) { await selectRun(b); } else { renderMatrix(); renderItems(); }
  renderRuns();
}

// ---------------------------------------------------------------- matrix
function renderMatrix() {
  const d = state.data;
  const seg = $("view-seg");
  seg.querySelector('[data-view="delta"]').disabled = !state.cmp;
  seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.view === state.view)));
  if (!d) return;
  const { run, aggregate: agg } = d;
  const o = agg.overall;
  const cmpRun = state.cmp?.a?.run;
  $("matrix-lede").innerHTML = `Run <b>${esc(run.label)}</b> #${run.run_id} · ${o.n} items · ${o.judged} judged` +
    (run.config?.settings?.chat_model ? ` · answerer <code>${esc(run.config.settings.chat_model)}</code>` : "") +
    (state.view === "delta" && cmpRun ? ` · showing change from <b>${esc(cmpRun.label)}</b> #${cmpRun.run_id}` : "") +
    (o.judged < o.n ? ` · <span class="warn">scores are provisional until the run is judged</span>` : "");
  renderKpis(agg);
  renderHeat(agg);
  renderJudgeInfo(run, agg);
  renderDefs(agg);
  renderConfusion(agg);
  renderAverages(agg);
  renderCharts();
}

function kpiDelta(path, fmt, higherBetter = true) {
  const h = state.cmp?.delta?.headline?.[path];
  if (state.view !== "delta" || !h || h.delta == null) return "";
  const good = higherBetter ? h.delta > 0 : h.delta < 0;
  return `<span class="kpi__delta ${h.delta === 0 ? "" : good ? "up" : "down"}">${sign(h.delta, fmt)}</span>`;
}

function renderKpis(agg) {
  const k = (n) => document.querySelector(`[data-k="${n}"]`);
  const o = agg.overall, ooc = agg.ooc, lat = agg.latency_ms, r = agg.retrieval || {};
  k("score").innerHTML = `${f1(o.score)}${kpiDelta("overall.score", f1)}`;
  k("score-sub").textContent = `0–100, scenario-balanced · item mean ${f1(o.score_items)}`;
  k("pass").innerHTML = `${pc(o.pass_rate)}${kpiDelta("overall.pass_rate", pc)}`;
  k("pass-sub").textContent = `${o.errors} errors`;
  k("ooc").innerHTML = `${f2(ooc.f1)}${kpiDelta("ooc.f1", f2)}`;
  k("ooc-sub").textContent = `precision ${f2(ooc.precision)} · recall ${f2(ooc.recall)} · exact ${pc(ooc.exact_match_rate)}`;
  k("hit").innerHTML = `${pc(r.hit_at_k)}${kpiDelta("retrieval.hit_at_k", pc)}`;
  k("hit-sub").textContent = `MRR ${f2(r.mrr)} · page hit ${pc(r.page_hit)} · cited gold ${pc(r.cited_gold)}`;
  k("lat").innerHTML = `${secs(lat.p50)} <span class="kpi__slash">/</span> ${secs(lat.p95)}${kpiDelta("latency_ms.p50", secs, false)}`;
  k("lat-sub").textContent = `first token p50 ${secs(lat.ttft_p50)} · ${agg.tokens.mean ?? "–"} tokens/answer`;
}

function columns(agg) {
  const used = (key) => [...agg.scenarios, "__all__"].some((s) => agg.matrix[s]?.[key]);
  return {
    scores: agg.metrics.filter((m) => used(m.key)),
    rates: agg.rates.filter((m) => used(m.key)),
  };
}

function renderHeat(agg) {
  const delta = state.view === "delta" && state.cmp;
  const cols = columns(agg);
  const dcells = delta ? state.cmp.delta.cells : null;
  const aAgg = delta ? state.cmp.a.aggregate : null;
  $("heat-title").textContent = delta ? "Scenario × metric - change (B − A)" : "Scenario × metric";
  $("heat-legend").innerHTML = delta
    ? `<span>worse</span><span class="scale__bar" style="background:linear-gradient(90deg,${DIV.neg},${DIV.mid},${DIV.pos})"></span><span>better</span>`
    : `<span>1</span><span class="scale__bar" style="background:linear-gradient(90deg,${SEQ.join(",")})"></span><span>5 · 100%</span>`;
  const head1 = `<tr><th></th><th class="grp" colspan="${cols.scores.length}">Judge scores (1–5)</th><th class="sep"></th>
    <th class="grp" colspan="${cols.rates.length}">Deterministic</th><th class="sep"></th><th class="grp" colspan="3">Scenario</th></tr>`;
  const head2 = `<tr><th scope="col" class="sr-only">Scenario</th>${cols.scores.map((m) => `<th scope="col" title="${esc(m.desc)}">${esc(m.label)}</th>`).join("")}
    <th class="sep"></th>${cols.rates.map((m) => `<th scope="col" title="${esc(m.desc)}">${esc(m.label)}</th>`).join("")}
    <th class="sep"></th><th scope="col" title="Weighted item score 0–100">Score</th><th scope="col" title="Share of items passing">Pass</th><th scope="col">n</th></tr>`;
  const rows = [...agg.scenarios, "__all__"].map((s) => {
    const row = agg.matrix[s] || {};
    const bs = s === "__all__" ? { score: agg.overall.score, pass_rate: agg.overall.pass_rate, n: agg.overall.n } : agg.by_scenario[s] || {};
    const cell = (m, kind) => {
      const c = row[m.key];
      if (!c) return `<td class="na">–</td>`;
      let bg, txt, t;
      if (delta) {
        const dv = dcells?.[s]?.[m.key];
        if (dv == null) return `<td class="na" data-s="${s}" data-m="${m.key}">n/a</td>`;
        const better = kind === "rate" && m.higher_better === false ? -dv : dv;
        t = kind === "score" ? better / 2 : better / 0.5;
        bg = divColor(t);
        txt = kind === "score" ? sign(dv, f2) : sign(dv, (x) => `${Math.round(x * 100)}`);
      } else {
        t = kind === "score" ? (c.mean - 1) / 4 : (m.higher_better === false ? 1 - c.mean : c.mean);
        bg = seqColor(t);
        txt = kind === "score" ? f2(c.mean) : pc(c.mean);
      }
      const ink = lum(bg) > 0.5 ? "ink-dark" : "ink-light";
      return `<td class="cell ${ink}" style="background:${bg}" data-s="${s}" data-m="${m.key}" data-kind="${kind}">${txt}</td>`;
    };
    const scoreCell = () => {
      if (bs.score == null) return `<td class="na">–</td>`;
      if (delta) {
        const a = s === "__all__" ? aAgg.overall.score : aAgg.by_scenario[s]?.score;
        if (a == null) return `<td class="na">n/a</td>`;
        const dv = bs.score - a, bg = divColor(dv / 25);
        return `<td class="cell ${lum(bg) > 0.5 ? "ink-dark" : "ink-light"}" style="background:${bg}" data-s="${s}" data-m="__score">${sign(dv, f1)}</td>`;
      }
      const bg = seqColor(bs.score / 100);
      return `<td class="cell ${lum(bg) > 0.5 ? "ink-dark" : "ink-light"}" style="background:${bg}" data-s="${s}" data-m="__score">${f1(bs.score)}</td>`;
    };
    return `<tr class="${s === "__all__" ? "is-total" : ""} ${state.activeRow === s ? "is-active" : ""}">
      <th scope="row"><button type="button" data-row="${s}">${esc(SCENARIO_LABEL(s))}</button></th>
      ${cols.scores.map((m) => cell(m, "score")).join("")}<td class="sep"></td>
      ${cols.rates.map((m) => cell(m, "rate")).join("")}<td class="sep"></td>
      ${scoreCell()}<td class="meta">${pc(bs.pass_rate)}</td><td class="meta">${bs.n ?? ""}</td></tr>`;
  }).join("");
  $("heat").innerHTML = `<thead>${head1}${head2}</thead><tbody>${rows}</tbody>`;
}

function showTip(td, ev) {
  const agg = state.data?.aggregate;
  if (!agg) return;
  const s = td.dataset.s, key = td.dataset.m;
  const tip = $("tip");
  const mdef = [...agg.metrics, ...agg.rates].find((m) => m.key === key);
  const c = agg.matrix[s]?.[key];
  let body = `<p class="tip__d">${esc(SCENARIO_LABEL(s))}</p><p>${esc(mdef ? mdef.label : "Item score")}<b>${esc(td.textContent)}</b></p>`;
  if (c) {
    body += `<p>Items<b>${c.n}</b></p>`;
    if (c.pass_rate != null) body += `<p>Scored 4 or 5<b>${pc(c.pass_rate)}</b></p>`;
  }
  if (state.view === "delta" && state.cmp && key !== "__score") {
    const a = state.cmp.a.aggregate.matrix[s]?.[key]?.mean, b = c?.mean;
    const fmt = td.dataset.kind === "rate" ? pc : f2;
    body += `<p>A → B<b>${fmt(a)} → ${fmt(b)}</b></p>`;
  }
  if (mdef?.desc) body += `<p class="tip__t">${esc(mdef.desc)}</p>`;
  tip.innerHTML = body;
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - 14;
  tip.style.transform = `translate(${x}px, ${y}px)`;
}

function renderJudgeInfo(run, agg) {
  const j = run.judge || {};
  const el = $("judge-info");
  if (!j.model) { el.innerHTML = `<p class="muted">Not judged yet. Use “Judge” in the runs table or <code>python -m eval.judge --run ${run.run_id}</code>.</p>`; }
  else {
    const mem = j.memory_loaded || {};
    const cal = j.calibration;
    el.innerHTML = `<p>Model <b>${esc(j.model)}</b>${j.same_as_answerer ? ` <span class="warn">(same as the answerer)</span>` : ""}</p>
      <p>${esc(j.reason || "")}</p>
      <p>Free RAM after loading <b>${mem.ram_available_gb ?? "–"} GB</b> · free VRAM <b>${mem.vram_free_gb ?? "–"} GB</b> · load ${j.load_s ?? "–"} s</p>
      ${(mem.ollama_loaded || []).map((m) => `<p>Resident: <b>${esc(m.name)}</b> ${m.size_gb} GB (${m.vram_gb} GB on GPU)</p>`).join("")}
      ${cal ? `<p>Calibration agreement <b>${pc(cal.agreement)}</b> (${cal.n_checks} checks)</p>
        <table><thead><tr><th>Reference</th><th>Label</th><th>Mean</th><th>OK</th></tr></thead><tbody>
        ${cal.cases.map((c) => `<tr><td>${esc(c.id)} · ${esc(c.item_id)}</td><td>${esc(c.label)}</td><td>${f2(c.mean)}</td><td class="${c.ok ? "good" : "bad"}">${c.ok ? "yes" : "no"}</td></tr>`).join("")}
        ${cal.pairs.map((p) => `<tr><td colspan="3">${esc(p.good)} ranked above ${esc(p.bad)}</td><td class="${p.ok ? "good" : "bad"}">${p.ok ? "yes" : "no"}</td></tr>`).join("")}
        </tbody></table>` : ""}
      ${j.judged != null ? `<p>Judged ${j.judged} · errors ${j.errors}</p>` : ""}`;
  }
  const p = agg.paraphrase || {};
  $("robust-info").innerHTML = `<p>Paraphrase / typo consistency <b>${pc(p.consistency)}</b></p>
    ${(p.pairs || []).map((x) => `<p>${esc(x.item)} vs ${esc(x.original)}: ${x.same_ooc ? "same scope decision" : `<span class="warn">different scope decision</span>`}${x.correctness_delta != null ? ` · correctness ${sign(x.correctness_delta, (v) => v.toFixed(0))}` : ""}</p>`).join("")}
    <p>Exact “${OOC}” when refusing <b>${pc(agg.ooc.exact_match_rate)}</b> · false refusals <b>${agg.ooc.fp}</b> · missed refusals <b>${agg.ooc.fn}</b></p>
    <p>Gold passage cited <b>${pc(agg.retrieval?.cited_gold)}</b> · recall@k <b>${pc(agg.retrieval?.recall_at_k)}</b></p>`;
}

function renderDefs(agg) {
  $("defs").innerHTML = [...agg.metrics, ...agg.rates].map((m) => `<dt>${esc(m.label)}</dt><dd>${esc(m.desc)}${m.weight ? ` <span class="muted">(weight ${m.weight})</span>` : ""}</dd>`).join("");
}

// ---------------------------------------------------------------- classification checks (confusion matrices)
// Colour carries "correct vs error" (blue / orange, validated categorical slots 1-2), always doubled by a text label.
const C_OK = PALETTE[0], C_ERR = PALETTE[1], C_A = PALETTE[1], C_B = PALETTE[0];
const CELL_META = {
  tp: { ok: true, abbr: "TP", exp: "pos", pred: "pos" },
  fn: { ok: false, abbr: "FN", exp: "pos", pred: "neg" },
  fp: { ok: false, abbr: "FP", exp: "neg", pred: "pos" },
  tn: { ok: true, abbr: "TN", exp: "neg", pred: "neg" },
};
const trunc = (s, n) => { s = String(s ?? ""); return s.length > n ? `${s.slice(0, Math.max(1, n - 1))}…` : s; };
const dTxt = (b, a, fmt = f2) => (b == null || a == null ? "–" : sign(b - a, fmt));
const dCls = (b, a, higher = true) => (b == null || a == null || b === a ? "" : (b > a) === higher ? "up" : "down");
const cellBg = (ok, t) => mix("#1D2328", ok ? C_OK : C_ERR, 0.14 + 0.72 * Math.max(0, Math.min(1, t)));
const clsName = (c, side) => (side === "pos" ? c.positive : c.negative);

function renderConfusion(agg) {
  const conf = agg.confusion;
  const el = $("conf-grid");
  if (!conf) { el.innerHTML = `<p class="muted">Not available for this run.</p>`; return; }
  const aConf = state.cmp ? state.cmp.a.aggregate.confusion : null;
  $("conf-hint").textContent = `Expected (rows) vs what the system did (columns). Select a cell to list its questions.${aConf ? ` “A:” counts and Δ F1 refer to run #${state.cmp.a.run.run_id}.` : ""}`;
  el.innerHTML = Object.values(conf).map((c) => confWidget(c, aConf?.[c.key])).join("");
  if (state.confSel) renderConfItems();
}

function confWidget(c, a) {
  const rowTot = { pos: c.tp + c.fn, neg: c.fp + c.tn };
  const cell = (k) => {
    const m = CELL_META[k], n = c[k], tot = rowTot[m.exp], t = tot ? n / tot : 0;
    const bg = cellBg(m.ok, t), ink = lum(bg) > 0.5 ? "ink-dark" : "ink-light";
    const sel = state.confSel?.matrix === c.key && state.confSel?.cell === k;
    return `<button type="button" class="cm__cell ${ink}${sel ? " is-sel" : ""}${m.ok ? "" : " is-err"}" style="background:${bg}"
      data-cm="${c.key}" data-cell="${k}" aria-pressed="${sel}"
      aria-label="${m.abbr}, ${m.ok ? "correct" : "error"}: ${n} questions expected ${esc(clsName(c, m.exp))}, predicted ${esc(clsName(c, m.pred))}">
      <span class="cm__n num">${n}</span>
      <span class="cm__k">${m.abbr} · ${m.ok ? "correct" : "error"}</span>
      <span class="cm__r num">${tot ? pc(t) : "–"} of row</span>
      ${a ? `<span class="cm__a num">A: ${a[k]}</span>` : ""}</button>`;
  };
  const row = (name, r, aR, strong) => `<tr${strong ? ` class="is-avg"` : ""}><th scope="row">${esc(name)}</th>
    <td class="r">${f2(r.precision)}</td><td class="r">${f2(r.recall)}</td><td class="r">${f2(r.f1)}</td><td class="r">${r.support}</td>
    ${a ? `<td class="r ${dCls(r.f1, aR?.f1)}">${dTxt(r.f1, aR?.f1)}</td>` : ""}</tr>`;
  const aCls = (name) => a?.classes?.find((x) => x.class === name);
  return `<article class="cm" aria-labelledby="cm-${c.key}-h">
    <h3 class="cm__title" id="cm-${c.key}-h">${esc(c.title)}</h3>
    <p class="cm__sub">${c.n} questions · accuracy <b>${pc(c.accuracy)}</b>${a ? ` <span class="${dCls(c.accuracy, a.accuracy)}">(${dTxt(c.accuracy, a.accuracy, (v) => `${Math.round(v * 100)} pp`)})</span>` : ""}</p>
    <div class="cm__grid">
      <span class="cm__corner">expected ↓ · predicted →</span>
      <span class="cm__colh">${esc(c.positive)}</span><span class="cm__colh">${esc(c.negative)}</span>
      <span class="cm__rowh">${esc(c.positive)}</span>${cell("tp")}${cell("fn")}
      <span class="cm__rowh">${esc(c.negative)}</span>${cell("fp")}${cell("tn")}
    </div>
    <table class="tbl cm__tbl"><thead><tr><th scope="col">Class</th><th scope="col" class="r">Prec.</th><th scope="col" class="r">Recall</th>
      <th scope="col" class="r">F1</th><th scope="col" class="r" title="Gold items of the class">Supp.</th>${a ? `<th scope="col" class="r">Δ F1</th>` : ""}</tr></thead>
    <tbody>${c.classes.map((x) => row(x.class, x, aCls(x.class))).join("")}
      ${row("micro avg", c.micro, a?.micro, true)}${row("macro avg", c.macro, a?.macro, true)}${row("weighted avg", c.weighted, a?.weighted, true)}</tbody></table>
    <p class="cm__rule"><span>Expected</span> ${esc(c.expected_rule)}<br><span>Predicted</span> ${esc(c.predicted_rule)}</p>
  </article>`;
}

function confTip(btn, ev) {
  const c = state.data?.aggregate?.confusion?.[btn.dataset.cm];
  if (!c) return;
  const k = btn.dataset.cell, m = CELL_META[k], n = c[k];
  const rowTot = m.exp === "pos" ? c.tp + c.fn : c.fp + c.tn;
  const colTot = m.pred === "pos" ? c.tp + c.fp : c.fn + c.tn;
  const a = state.cmp?.a?.aggregate?.confusion?.[c.key];
  cvTip(ev, {
    title: `${m.abbr} · expected ${clsName(c, m.exp)}, predicted ${clsName(c, m.pred)}`,
    rows: [{ label: "Questions", value: String(n), color: m.ok ? C_OK : C_ERR },
      { label: `Share of expected ${clsName(c, m.exp)}`, value: rowTot ? pc(n / rowTot) : "–" },
      { label: `Share of predicted ${clsName(c, m.pred)}`, value: colTot ? pc(n / colTot) : "–" },
      ...(a ? [{ label: `Run A (#${state.cmp.a.run.run_id})`, value: String(a[k]) }] : [])],
    foot: n ? "Select to list the questions" : "",
  });
}

function renderConfItems() {
  const box = $("conf-items");
  const sel = state.confSel;
  const c = sel && state.data?.aggregate?.confusion?.[sel.matrix];
  if (!c) { box.hidden = true; box.innerHTML = ""; return; }
  const m = CELL_META[sel.cell];
  const list = c.cells[sel.cell] || [];
  const byId = Object.fromEntries(state.items.map((i) => [i.item_id, i]));
  box.hidden = false;
  box.innerHTML = `<div class="panel__bar"><h3 class="conf-items__t">${esc(c.title)} · <b>${m.abbr}</b> - expected ${esc(clsName(c, m.exp))}, predicted ${esc(clsName(c, m.pred))} · ${list.length} question${list.length === 1 ? "" : "s"}</h3>
      <button type="button" class="btn btn--sm" id="conf-close">Close</button></div>
    ${list.length ? `<div class="tbl-scroll conf-items__scroll"><table class="tbl tbl--items"><thead><tr><th scope="col">Item</th><th scope="col">Question</th><th scope="col" class="r">Score</th><th scope="col">Result</th></tr></thead><tbody>
      ${list.map((x) => { const i = byId[x.item_id] || {}; return `<tr data-item="${esc(x.item_id)}" tabindex="0">
        <td><span class="iid">${esc(x.item_id)}</span><span class="sc">${esc(SCENARIO_LABEL(x.scenario))}</span></td>
        <td class="q" title="${esc(i.question || "")}">${esc(i.question || "")}</td><td class="r num">${f1(i.score)}</td>
        <td>${!i.judged && i.pass ? `<span class="pf pf--pending">not judged</span>` : i.pass ? `<span class="pf pf--pass">pass</span>` : `<span class="pf pf--fail">fail</span>`}${i.n_images ? ` <span class="muted">${i.n_images} fig.</span>` : ""}</td></tr>`; }).join("")}
      </tbody></table></div>` : `<p class="muted">No questions in this cell.</p>`}`;
}

// ---------------------------------------------------------------- micro vs macro
function renderAverages(agg) {
  const av = agg.averages, t = $("avg-table");
  if (!av) { t.innerHTML = `<tbody><tr><td class="muted">Not available.</td></tr></tbody>`; return; }
  const aAv = state.cmp ? state.cmp.a.aggregate.averages : null;
  const aM = Object.fromEntries((aAv?.metrics || []).map((m) => [m.key, m]));
  const pp = (v) => `${(v * 100).toFixed(1)} pp`;
  const fmtOf = (kind) => (kind === "score100" ? [f1, f1] : kind === "score" ? [f2, f2] : [pc, pp]);
  const row = (r) => {
    const [fv, fd] = fmtOf(r.kind), hb = r.higher_better !== false;
    return `<tr><th scope="row">${esc(r.label)}</th><td class="r">${fv(r.micro)}</td><td class="r">${fv(r.macro)}</td>
      <td class="r muted">${r.gap == null ? "–" : sign(r.gap, fd)}</td><td class="r">${r.n_items}</td><td class="r">${r.n_scenarios}</td>
      ${aAv ? `<td class="r sep-l">${fv(r.a?.micro)}</td><td class="r">${fv(r.a?.macro)}</td>
        <td class="r ${dCls(r.micro, r.a?.micro, hb)}">${dTxt(r.micro, r.a?.micro, fd)}</td><td class="r ${dCls(r.macro, r.a?.macro, hb)}">${dTxt(r.macro, r.a?.macro, fd)}</td>` : ""}</tr>`;
  };
  const span = aAv ? 10 : 6;
  const grp = (label) => `<tr class="grp-row"><th colspan="${span}" scope="colgroup">${esc(label)}</th></tr>`;
  const scores = av.metrics.filter((m) => m.kind === "score"), rates = av.metrics.filter((m) => m.kind === "rate");
  t.innerHTML = `<thead><tr><th scope="col">Metric</th><th scope="col" class="r">Micro</th><th scope="col" class="r">Macro</th>
      <th scope="col" class="r" title="Micro minus macro">Gap</th><th scope="col" class="r">Items</th><th scope="col" class="r">Scenarios</th>
      ${aAv ? `<th scope="col" class="r sep-l">A micro</th><th scope="col" class="r">A macro</th><th scope="col" class="r">Δ micro</th><th scope="col" class="r">Δ macro</th>` : ""}</tr></thead>
    <tbody>${grp("Headline")}${row({ label: "Item score (0–100)", kind: "score100", ...av.score, a: aAv?.score })}
      ${row({ label: "Pass rate", kind: "rate", ...av.pass_rate, a: aAv?.pass_rate })}
      ${scores.length ? grp("Judge scores (1–5)") + scores.map((m) => row({ ...m, a: aM[m.key] })).join("") : ""}
      ${rates.length ? grp("Deterministic rates") + rates.map((m) => row({ ...m, a: aM[m.key] })).join("") : ""}</tbody>`;
}

// ---------------------------------------------------------------- charts (inline SVG; shared tooltip + legend from charts.js)
const RO = new ResizeObserver((ents) => {
  for (const e of ents) {
    const el = e.target, w = Math.floor(el.clientWidth);
    if (el._evDraw && w && Math.abs(w - (el._evW || 0)) > 4) { el._evW = w; el._evDraw(); }
  }
});
function mountChart(el, fn) {
  el._evDraw = fn;
  if (!el._evObs) { RO.observe(el); el._evObs = true; }
  el._evW = Math.floor(el.clientWidth);
  fn();
}
function emptyChart(el, text) { el.innerHTML = `<p class="cv-empty">${esc(text)}</p>`; }
function wireChart(el, tipFor, onSelect) {
  el.querySelectorAll("[data-hit]").forEach((h) => {
    const id = h.dataset.hit, g = h.closest("g");
    const on = (e) => { g?.classList.add("is-hover"); cvTip(e, tipFor(id)); };
    const off = () => { g?.classList.remove("is-hover"); cvHide(); };
    h.addEventListener("pointermove", on);
    h.addEventListener("pointerleave", off);
    h.addEventListener("focus", on);
    h.addEventListener("blur", off);
    if (onSelect) {
      h.addEventListener("click", () => { cvHide(); onSelect(id); });
      h.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); cvHide(); onSelect(id); } });
    }
  });
}

/** Horizontal grouped bars: o = {cats:[{key,label}], series:[{key,label,color}], hidden:Set, value(cat, ser), lo, hi,
 *  ticks, fmt, right(cat) -> text, rightHead, tip(cat) -> {title, rows, foot}, onSelect, selected, aria, empty} */
function hbars(el, o) {
  mountChart(el, () => {
    const W = Math.max(280, Math.floor(el.clientWidth) || 600);
    const ser = o.series.filter((s) => !o.hidden?.has(s.key));
    if (!o.cats.length || !ser.length) return emptyChart(el, o.empty || "No data.");
    const barH = 10, gap = 2, padY = 7;
    const rowH = ser.length * barH + (ser.length - 1) * gap + padY * 2;
    const LW = Math.min(o.labelWidth || 160, Math.round(W * 0.36)), RW = o.right ? 54 : 6, VW = 38, T = 4, B = 22;
    const iw = W - LW - RW - VW, H = T + o.cats.length * rowH + B;
    const xp = (v) => LW + ((Math.max(o.lo, Math.min(o.hi, v)) - o.lo) / (o.hi - o.lo)) * iw;
    let svg = `<svg class="cv" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(o.aria)}">`;
    for (const t of o.ticks) svg += `<line class="cv-grid" x1="${xp(t)}" x2="${xp(t)}" y1="${T}" y2="${H - B}"/><text class="cv-ax" x="${xp(t)}" y="${H - 6}" text-anchor="middle">${esc(String(t))}</text>`;
    if (o.right) svg += `<text class="cv-ax" x="${W - 4}" y="${H - 6}" text-anchor="end">${esc(o.rightHead || "Δ")}</text>`;
    o.cats.forEach((c, i) => {
      const y = T + i * rowH, sel = o.selected != null && o.selected === c.key;
      svg += `<g class="cv-row${sel ? " is-sel" : ""}" data-g="${esc(c.key)}">`;
      if (i % 2 === 1) svg += `<rect class="ev-band" x="0" y="${y}" width="${W}" height="${rowH}"/>`;
      svg += `<text class="cv-lab" x="${LW - 8}" y="${y + rowH / 2 + 4}" text-anchor="end">${esc(trunc(c.label, Math.floor((LW - 10) / 6.4)))}</text>`;
      ser.forEach((s, si) => {
        const v = o.value(c.key, s.key), by = y + padY + si * (barH + gap);
        if (v == null) { svg += `<text class="cv-ax" x="${LW + 4}" y="${by + barH - 1}">n/a</text>`; return; }
        const w = Math.max(1.5, xp(v) - LW);
        svg += `<rect class="cv-bar" x="${LW}" y="${by}" width="${w.toFixed(1)}" height="${barH}" rx="2" fill="${s.color}"/>`;
        if (si === 0) svg += `<text class="cv-val" x="${(LW + w + 5).toFixed(1)}" y="${by + barH - 1}">${esc(o.fmt(v))}</text>`;
      });
      if (o.right) { const r = o.right(c.key); if (r) svg += `<text class="cv-val ev-delta" x="${W - 4}" y="${y + rowH / 2 + 4}" text-anchor="end">${esc(r)}</text>`; }
      svg += `<rect class="cv-hit" data-hit="${esc(c.key)}" x="0" y="${y}" width="${W}" height="${rowH}" tabindex="0"${o.onSelect ? ` role="button" aria-pressed="${sel}"` : ""} aria-label="${esc(c.label)}"/></g>`;
    });
    svg += `<line class="cv-base" x1="${LW}" x2="${LW}" y1="${T}" y2="${H - B}"/></svg>`;
    el.innerHTML = svg;
    wireChart(el, o.tip, o.onSelect);
  });
}

/** Vertical grouped bars for a histogram: o = {bins:[{key,label,lo,hi}], series, hidden, value(bin, ser), tip, onSelect, xTitle, aria} */
function histo(el, o) {
  mountChart(el, () => {
    const W = Math.max(280, Math.floor(el.clientWidth) || 600);
    const ser = o.series.filter((s) => !o.hidden?.has(s.key));
    if (!o.bins.length || !ser.length) return emptyChart(el, o.empty || "No data.");
    const H = 230, L = 36, R = 10, T = 10, B = 40, iw = W - L - R, ih = H - T - B;
    let max = Math.max(1, ...o.bins.flatMap((b) => ser.map((s) => o.value(b.key, s.key) || 0)));
    max = max <= 4 ? max : niceMax(max);
    const yp = (v) => T + ih - (v / max) * ih;
    const bw = iw / o.bins.length, inner = Math.max(3, Math.min(bw - 4, ser.length * 20)), barW = (inner - (ser.length - 1) * 2) / ser.length;
    let svg = `<svg class="cv" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(o.aria)}">`;
    const ticks = Number.isInteger(max / 2) ? [0, max / 2, max] : [0, max];
    for (const t of ticks) svg += `<line class="cv-grid" x1="${L}" x2="${W - R}" y1="${yp(t)}" y2="${yp(t)}"/><text class="cv-ax" x="${L - 6}" y="${yp(t) + 4}" text-anchor="end">${t}</text>`;
    const every = Math.ceil(o.bins.length / Math.max(2, Math.floor(iw / 34)));
    o.bins.forEach((b, i) => {
      const x0 = L + i * bw + (bw - inner) / 2, sel = o.selected === b.key;
      svg += `<g class="cv-col${sel ? " is-sel" : ""}" data-g="${esc(b.key)}">`;
      if (sel) svg += `<rect class="cv-selband" x="${L + i * bw}" y="${T}" width="${bw}" height="${ih}"/>`;
      ser.forEach((s, si) => {
        const v = o.value(b.key, s.key) || 0;
        if (!v) return;
        svg += `<rect class="cv-bar" x="${(x0 + si * (barW + 2)).toFixed(1)}" y="${yp(v).toFixed(1)}" width="${barW.toFixed(1)}" height="${(T + ih - yp(v)).toFixed(1)}" rx="2" fill="${s.color}"/>`;
      });
      if (i % every === 0) svg += `<text class="cv-ax" x="${L + i * bw}" y="${T + ih + 15}" text-anchor="middle">${esc(String(b.lo))}</text>`;
      svg += `<rect class="cv-hit" data-hit="${esc(b.key)}" x="${L + i * bw}" y="${T}" width="${bw}" height="${ih}" tabindex="0"${o.onSelect ? ` role="button" aria-pressed="${sel}"` : ""} aria-label="${esc(b.label)}"/></g>`;
    });
    const last = o.bins[o.bins.length - 1];
    svg += `<text class="cv-ax" x="${L + iw}" y="${T + ih + 15}" text-anchor="middle">${esc(String(last.hi))}</text>`;
    svg += `<line class="cv-base" x1="${L}" x2="${W - R}" y1="${T + ih}" y2="${T + ih}"/>`;
    svg += `<text class="cv-ax cv-ax--t" x="${L + iw / 2}" y="${H - 4}" text-anchor="middle">${esc(o.xTitle)}</text></svg>`;
    el.innerHTML = svg;
    wireChart(el, o.tip, o.onSelect);
  });
}

/** Scatter with shapes per series and drag-to-zoom on x: o = {points:[{id,x,y,series,label,rows}], series:[{key,label,color,shape}],
 *  hidden, domain:[x0,x1], yRef, onSelect(id), onZoom(lo,hi), xTitle, yTitle, aria} */
function scatterZoom(el, o) {
  mountChart(el, () => {
    const W = Math.max(280, Math.floor(el.clientWidth) || 600);
    const H = 270, L = 46, R = 12, T = 10, B = 40, iw = W - L - R, ih = H - T - B;
    const [x0, x1] = o.domain;
    const shape = Object.fromEntries(o.series.map((s) => [s.key, s]));
    const pts = o.points.filter((p) => !o.hidden?.has(p.series) && p.x >= x0 && p.x <= x1);
    const xp = (x) => L + ((x - x0) / (x1 - x0 || 1)) * iw, yp = (y) => T + ih - (y / 100) * ih;
    let svg = `<svg class="cv ev-scatter" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(o.aria)}">`;
    for (const t of [0, 25, 50, 75, 100]) svg += `<line class="cv-grid" x1="${L}" x2="${W - R}" y1="${yp(t)}" y2="${yp(t)}"/><text class="cv-ax" x="${L - 6}" y="${yp(t) + 4}" text-anchor="end">${t}</text>`;
    for (let k = 0; k <= 4; k++) { const t = x0 + ((x1 - x0) * k) / 4; svg += `<text class="cv-ax" x="${xp(t)}" y="${T + ih + 15}" text-anchor="middle">${+t.toFixed(t < 10 ? 1 : 0)}</text>`; }
    if (o.yRef != null) svg += `<line class="ev-ref" x1="${L}" x2="${W - R}" y1="${yp(o.yRef)}" y2="${yp(o.yRef)}"/><text class="cv-ax" x="${W - R - 2}" y="${yp(o.yRef) - 4}" text-anchor="end">pass needs ≥ ${o.yRef}</text>`;
    svg += `<line class="cv-base" x1="${L}" x2="${W - R}" y1="${T + ih}" y2="${T + ih}"/>`;
    svg += `<text class="cv-ax cv-ax--t" x="${L + iw / 2}" y="${H - 4}" text-anchor="middle">${esc(o.xTitle)}</text>`;
    svg += `<text class="cv-ax cv-ax--t" x="12" y="${T + ih / 2}" text-anchor="middle" transform="rotate(-90 12 ${T + ih / 2})">${esc(o.yTitle)}</text>`;
    svg += `<rect class="ev-brush-area" x="${L}" y="${T}" width="${iw}" height="${ih}"/><rect class="ev-brush" x="0" y="${T}" width="0" height="${ih}" visibility="hidden"/>`;
    pts.forEach((p) => {
      const cx = xp(p.x), cy = yp(p.y), s = shape[p.series] || o.series[0];
      const mark = s.shape === "tri"
        ? `<path class="cv-pt" d="M${cx.toFixed(1)},${(cy - 6).toFixed(1)}L${(cx + 5.5).toFixed(1)},${(cy + 4).toFixed(1)}L${(cx - 5.5).toFixed(1)},${(cy + 4).toFixed(1)}Z" fill="${s.color}"/>`
        : `<circle class="cv-pt" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="4.5" fill="${s.color}"/>`;
      svg += `<g data-g="${esc(p.id)}"${p.id === o.selected ? ` class="is-sel"` : ""}>${mark}<circle class="cv-hit" data-hit="${esc(p.id)}" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="10" tabindex="0" role="button" aria-label="${esc(p.label)}"/></g>`;
    });
    svg += `</svg>`;
    el.innerHTML = svg;
    if (!pts.length) el.insertAdjacentHTML("beforeend", `<p class="cv-empty ev-empty-over">${esc(o.empty || "No points in this range.")}</p>`);
    const byId = Object.fromEntries(pts.map((p) => [p.id, p]));
    wireChart(el, (id) => ({ title: byId[id].label, rows: byId[id].rows, foot: "Select to open the question" }), o.onSelect);
    // drag to zoom
    const root = el.querySelector("svg"), area = root.querySelector(".ev-brush-area"), band = root.querySelector(".ev-brush");
    const toX = (e) => { const b = root.getBoundingClientRect(); return Math.max(L, Math.min(L + iw, ((e.clientX - b.left) / b.width) * W)); };
    let start = null;
    area.addEventListener("pointerdown", (e) => { start = toX(e); area.setPointerCapture(e.pointerId); band.setAttribute("x", start); band.setAttribute("width", 0); band.setAttribute("visibility", "visible"); });
    area.addEventListener("pointermove", (e) => { if (start == null) return; const x = toX(e); band.setAttribute("x", Math.min(start, x)); band.setAttribute("width", Math.abs(x - start)); });
    area.addEventListener("pointerup", (e) => {
      if (start == null) return;
      const x = toX(e), a = Math.min(start, x), b = Math.max(start, x);
      start = null; band.setAttribute("visibility", "hidden");
      if (b - a > 8) { const v = (px) => x0 + ((px - L) / iw) * (x1 - x0); o.onZoom(+v(a).toFixed(1), +v(b).toFixed(1)); }
    });
  });
}

function renderCharts() {
  const d = state.data;
  if (!d) return;
  const agg = d.aggregate, run = d.run;
  const aAgg = state.cmp ? state.cmp.a.aggregate : null, aRun = state.cmp?.a?.run;
  const bName = `${run.label} #${run.run_id}`, aName = aRun ? `${aRun.label} #${aRun.run_id}` : null;
  const runSeries = [{ key: "b", label: bName, color: C_B }].concat(aRun ? [{ key: "a", label: aName, color: C_A }] : []);
  $("charts-lede").textContent = aRun
    ? `Selected run ${bName} (blue) against ${aName} (orange). Hover for values; select a bar or point to drill down; legend entries toggle a series.`
    : `Selected run ${bName}. Tick a second run in the table to overlay it. Hover for values; select a bar or point to drill down.`;
  const toggler = (name, redraw) => (key) => { const h = state.hidden[name]; h.has(key) ? h.delete(key) : h.add(key); redraw(); };

  // 1. score by scenario
  const scen = () => {
    cvLegend($("lg-scen"), { series: runSeries, hidden: state.hidden.scen, onToggle: toggler("scen", scen) });
    const val = (s, k) => (k === "b" ? agg.by_scenario[s]?.score : aAgg?.by_scenario?.[s]?.score) ?? null;
    hbars($("ch-scen"), {
      cats: agg.scenarios.map((s) => ({ key: s, label: SCENARIO_LABEL(s) })), series: runSeries, hidden: state.hidden.scen,
      value: val, lo: 0, hi: 100, ticks: [0, 25, 50, 75, 100], fmt: f1, selected: state.activeRow,
      right: aRun ? (s) => dTxt(val(s, "b"), val(s, "a"), f1) : null, rightHead: "Δ vs A",
      empty: agg.overall.judged ? "No scores." : "Not judged yet: scores appear once the run is judged.",
      aria: "Item score by scenario",
      tip: (s) => ({
        title: SCENARIO_LABEL(s),
        rows: [{ label: bName, value: f1(val(s, "b")), color: C_B },
          ...(aRun ? [{ label: aName, value: f1(val(s, "a")), color: C_A }, { label: "Change", value: dTxt(val(s, "b"), val(s, "a"), f1) }] : []),
          { label: "Pass rate", value: pc(agg.by_scenario[s]?.pass_rate) }, { label: "Questions", value: String(agg.by_scenario[s]?.n ?? "–") }],
        foot: "Select to list its questions",
      }),
      onSelect: (s) => filterScenario(s),
    });
  };
  if (agg.overall.judged || aAgg?.overall?.judged) scen();
  else { cvLegend($("lg-scen"), { series: [] }); emptyChart($("ch-scen"), "Not judged yet: scores appear once the run is judged."); }

  // 2. judge metrics, micro or macro
  const metr = () => {
    document.querySelectorAll("#metric-seg button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.avg === state.avgMode)));
    cvLegend($("lg-metric"), { series: runSeries, hidden: state.hidden.metric, onToggle: toggler("metric", metr) });
    const mb = Object.fromEntries((agg.averages?.metrics || []).filter((m) => m.kind === "score").map((m) => [m.key, m]));
    const ma = Object.fromEntries((aAgg?.averages?.metrics || []).filter((m) => m.kind === "score").map((m) => [m.key, m]));
    const keys = (agg.metrics || []).map((m) => m.key).filter((k) => mb[k] || ma[k]);
    const lab = Object.fromEntries((agg.metrics || []).map((m) => [m.key, m.label]));
    const val = (k, s) => (s === "b" ? mb[k] : ma[k])?.[state.avgMode] ?? null;
    hbars($("ch-metric"), {
      cats: keys.map((k) => ({ key: k, label: lab[k] || k })), series: runSeries, hidden: state.hidden.metric,
      value: val, lo: 1, hi: 5, ticks: [1, 2, 3, 4, 5], fmt: f2, labelWidth: 120,
      right: aRun ? (k) => dTxt(val(k, "b"), val(k, "a"), f2) : null, rightHead: "Δ vs A",
      empty: "No judge scores yet.", aria: `Judge metric means (${state.avgMode})`,
      tip: (k) => ({
        title: `${lab[k] || k} · ${state.avgMode}`,
        rows: [{ label: bName, value: f2(val(k, "b")), color: C_B },
          ...(aRun ? [{ label: aName, value: f2(val(k, "a")), color: C_A }] : []),
          { label: "Micro / macro (selected run)", value: `${f2(mb[k]?.micro)} / ${f2(mb[k]?.macro)}` },
          { label: "Items · scenarios", value: mb[k] ? `${mb[k].n_items} · ${mb[k].n_scenarios}` : "–" }],
      }),
    });
  };
  metr();

  // 3. latency histogram
  const lat = () => {
    const key = state.latKey;
    document.querySelectorAll("#lat-seg button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.lat === key)));
    cvLegend($("lg-lat"), { series: runSeries, hidden: state.hidden.lat, onToggle: toggler("lat", lat) });
    const vals = { b: state.items.map((i) => i[key]).filter((v) => v > 0).map((v) => v / 1000),
      a: aRun ? state.cmpItems.map((i) => i[key]).filter((v) => v > 0).map((v) => v / 1000) : [] };
    const all = [...vals.b, ...(state.hidden.lat.has("a") ? [] : vals.a)];
    const mx = all.length ? Math.max(...all) : 0;
    const step = [0.5, 1, 2, 2.5, 5, 10, 15, 20, 30, 60].find((s) => mx / s <= 14) || 120;
    const nb = Math.max(1, Math.ceil((mx + 1e-9) / step));
    const bins = Array.from({ length: nb }, (_, i) => ({ key: String(i), lo: +(i * step).toFixed(1), hi: +((i + 1) * step).toFixed(1) }))
      .map((b) => ({ ...b, label: `${b.lo}–${b.hi} s` }));
    const count = (k, s) => vals[s].filter((v) => v >= bins[+k].lo && v < bins[+k].hi).length;
    const selKey = state.latRange && state.latRange.key === key ? bins.find((b) => b.lo === state.latRange.lo)?.key : null;
    histo($("ch-lat"), {
      bins, series: runSeries, hidden: state.hidden.lat, value: count, selected: selKey,
      xTitle: key === "ttft_ms" ? "time to first token (s)" : "total latency (s)",
      empty: "No latency data.", aria: "Latency distribution",
      tip: (k) => ({ title: bins[+k].label, rows: runSeries.filter((s) => !state.hidden.lat.has(s.key)).map((s) => ({ label: s.label, value: `${count(k, s.key)} questions`, color: s.color })), foot: "Select to filter the items list" }),
      onSelect: (k) => {
        const b = bins[+k];
        state.latRange = state.latRange && state.latRange.key === key && state.latRange.lo === b.lo ? null : { key, lo: b.lo, hi: b.hi };
        lat(); renderItems();
        if (state.latRange) document.getElementById("items").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
      },
    });
    const pq = (v, q) => { const s = [...v].sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))] : null; };
    $("lat-foot").textContent = `Questions per ${step}-second bin · median ${pq(vals.b, 0.5)?.toFixed(1) ?? "–"} s${aRun ? ` (A ${pq(vals.a, 0.5)?.toFixed(1) ?? "–"} s)` : ""}. Select a bin to filter the items list.`;
  };
  lat();

  // 4. score vs latency (selected run)
  const scat = () => {
    const ser = [{ key: "pass", label: "Pass", color: C_OK, shape: "dot" }, { key: "fail", label: "Fail", color: C_ERR, shape: "tri" }];
    cvLegend($("lg-scatter"), { series: ser, hidden: state.hidden.scatter, onToggle: toggler("scatter", scat) });
    const dItems = state.cmp?.delta?.items;
    const pts = state.items.filter((i) => i.judged && i.score != null && i.latency_ms).map((i) => ({
      id: i.item_id, x: i.latency_ms / 1000, y: i.score, series: i.pass ? "pass" : "fail",
      label: `${i.item_id} · ${SCENARIO_LABEL(i.scenario)}`,
      rows: [{ label: "Score", value: f1(i.score), color: i.pass ? C_OK : C_ERR }, { label: "Latency", value: secs(i.latency_ms) },
        { label: "Result", value: i.pass ? "pass" : "fail" },
        ...(dItems?.[i.item_id]?.delta != null ? [{ label: "Change vs A", value: sign(dItems[i.item_id].delta, f1) }] : []),
        { label: "Question", value: trunc(i.question, 60) }],
    }));
    const mx = pts.length ? Math.max(...pts.map((p) => p.x)) : 1;
    const xStep = mx > 120 ? 20 : mx > 60 ? 10 : 5;
    const domain = state.zoom || [0, Math.max(xStep, Math.ceil(mx / xStep) * xStep)];
    $("sc-reset").hidden = !state.zoom;
    const nIn = pts.filter((p) => p.x >= domain[0] && p.x <= domain[1]).length;
    $("scatter-foot").textContent = `${nIn} of ${pts.length} judged questions${state.zoom ? ` between ${domain[0]} and ${domain[1]} s` : ""}${state.items.length - pts.length ? ` (${state.items.length - pts.length} without a score)` : ""}. Drag across the chart to zoom into a latency range; select a point to open the question.`;
    if (!pts.length) { emptyChart($("ch-scatter"), "No judged questions yet."); return; }
    scatterZoom($("ch-scatter"), {
      points: pts, series: ser, hidden: state.hidden.scatter, domain, yRef: 60, selected: state.selItem,
      xTitle: "total latency (s)", yTitle: "item score", aria: "Item score against latency",
      onSelect: (id) => { showItem(id); document.getElementById("items").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" }); },
      onZoom: (lo, hi) => { state.zoom = [lo, hi]; scat(); },
    });
  };
  state._redrawScatter = scat;
  scat();
}

function filterScenario(s) {
  state.activeRow = s;
  $("f-scenario").value = s === "__all__" ? "" : s;
  if (state.data) renderHeat(state.data.aggregate);
  renderItems();
  renderCharts();
  document.getElementById("items").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
}

// ---------------------------------------------------------------- items
async function loadItems() {
  if (state.sel == null) return;
  try { state.items = await api(`/api/eval/runs/${state.sel}/items`); }
  catch (e) { toast(e.message, { error: true }); return; }
  const sel = $("f-scenario"), cur = sel.value;
  const scen = [...new Set(state.items.map((i) => i.scenario))];
  sel.innerHTML = `<option value="">All scenarios</option>` + scen.map((s) => `<option value="${s}">${esc(SCENARIO_LABEL(s))}</option>`).join("");
  sel.value = scen.includes(cur) ? cur : "";
  renderItems();
  renderCharts();
  if (state.confSel) renderConfItems();
}

function renderItems() {
  const sc = $("f-scenario").value, q = $("f-text").value.trim().toLowerCase(), failed = $("f-failed").checked;
  const dItems = state.view === "delta" && state.cmp ? state.cmp.delta.items : null;
  $("items-delta-h").hidden = !dItems;
  const lr = state.latRange;
  const rows = state.items.filter((i) => (!sc || i.scenario === sc) && (!failed || !i.pass) &&
    (!lr || (i[lr.key] != null && i[lr.key] / 1000 >= lr.lo && i[lr.key] / 1000 < lr.hi)) &&
    (!q || i.item_id.toLowerCase().includes(q) || (i.question || "").toLowerCase().includes(q)));
  $("items-lede").innerHTML = esc(`${rows.length} of ${state.items.length} questions${sc ? ` in ${SCENARIO_LABEL(sc)}` : ""}${failed ? ", failed only" : ""}`) +
    (lr ? ` · ${esc(lr.key === "ttft_ms" ? "first token" : "latency")} ${lr.lo}–${lr.hi} s <button type="button" class="btn btn--sm" id="lat-clear">Clear latency filter</button>` : "") +
    ". Select one to see the answer, its sources and the judge's reasoning.";
  $("items-table").tBodies[0].innerHTML = rows.map((i) => {
    const res = !i.judged && i.pass ? `<span class="pf pf--pending">not judged</span>` : i.pass ? `<span class="pf pf--pass">pass</span>` : `<span class="pf pf--fail">fail</span>`;
    const dv = dItems?.[i.item_id]?.delta;
    return `<tr data-item="${esc(i.item_id)}" class="${i.item_id === state.selItem ? "is-selected" : ""}">
      <td><span class="iid">${esc(i.item_id)}</span><span class="sc">${esc(SCENARIO_LABEL(i.scenario))}</span></td>
      <td class="q" title="${esc(i.question)}">${esc(i.question)}</td>
      <td class="r num">${f1(i.score)}</td>
      ${dItems ? `<td class="r num ${dv > 0 ? "up" : dv < 0 ? "down" : ""}">${dv == null ? "–" : sign(dv, f1)}</td>` : ""}
      <td>${res}${i.out_of_context ? ` <span class="muted">OOC</span>` : ""}${i.status === "error" ? ` <span class="bad">error</span>` : ""}</td>
      <td class="r num">${i.latency_ms ? (i.latency_ms / 1000).toFixed(0) : "–"}</td></tr>`;
  }).join("") || `<tr><td colspan="6" class="muted">No items match.</td></tr>`;
}

const SCORE_ORDER = () => (state.meta?.metrics || []).map((m) => m.key);

async function showItem(id) {
  state.selItem = id;
  renderItems();
  const el = $("detail");
  el.innerHTML = `<p class="muted">Loading…</p>`;
  let r;
  try { r = await api(`/api/eval/runs/${state.sel}/items/${encodeURIComponent(id)}`); }
  catch (e) { el.innerHTML = `<p class="warn">${esc(e.message)}</p>`; return; }
  const it = r.item || {};
  const g = it.gold || {};
  const det = r.det_metrics || {};
  const turns = r.turns || [];
  const labels = Object.fromEntries([...(state.meta?.metrics || []), ...(state.meta?.rates || [])].map((m) => [m.key, m.label]));
  const judge = r.judge || {};
  const scoreKeys = SCORE_ORDER().filter((k) => judge[k] || (k === "refusal_correctness" && det.refusal_correctness != null));
  const scoreRow = (k) => {
    const v = k === "refusal_correctness" ? { score: det.refusal_correctness, justification: det.pred_ooc === det.expected_ooc ? "Scope decision matches the expected behaviour." : "Wrong scope decision.", auto: true } : judge[k];
    const bg = v.score != null ? seqColor((v.score - 1) / 4) : "transparent";
    return `<tr><th>${esc(labels[k] || k)}${v.auto ? `<span class="auto">auto</span>` : ""}</th>
      <td class="s ${lum(bg) > 0.5 ? "ink-dark" : "ink-light"}" style="background:${bg};color:${lum(bg) > 0.5 ? "#10161C" : "#F3F6F8"}">${v.score ?? "–"}</td>
      <td class="j">${esc(v.justification || "")}</td></tr>`;
  };
  const tm = r.timings || {};
  el.innerHTML = `
    <h2>${esc(it.id || id)} <span class="muted">· ${esc(SCENARIO_LABEL(r.scenario))}</span></h2>
    <div class="detail__meta">
      <span class="badge ${r.pass ? "badge--done" : "badge--failed"}">${r.pass ? "pass" : "fail"}</span>
      <span class="badge">score ${f1(r.score)}</span>
      <span class="badge">expected: ${esc(it.expected || "?")}</span>
      <span class="badge">${esc(it.difficulty || "medium")}</span>
      ${it.format ? `<span class="badge">format: ${esc(it.format)}</span>` : ""}
      ${r.judge_model ? `<span class="badge">judge ${esc(r.judge_model)}</span>` : ""}
    </div>
    ${r.error ? `<p class="warn">Error: ${esc(r.error)}</p>` : ""}
    <h3>Conversation</h3>
    <div id="d-turns"></div>
    <h3>Answer</h3>
    <div class="answer-box ${r.out_of_context ? "is-ooc" : ""}"><div class="prose" id="d-answer"></div></div>
    ${r.uncited ? `<p class="warn">The pipeline flagged this answer as uncited.</p>` : ""}
    ${(r.images || []).length ? `<h3>Images returned</h3><div class="figs">${r.images.map((im) => `<figure><img src="${esc(im.url)}" alt="${esc(im.caption || im.figure_number || "figure")}" loading="lazy"><figcaption>${esc(im.figure_number || "")} ${esc(im.caption || "")}</figcaption></figure>`).join("")}</div>` : ""}
    <h3>Judge</h3>
    ${scoreKeys.length ? `<table class="scores"><tbody>${scoreKeys.map(scoreRow).join("")}</tbody></table>` : `<p class="muted">Not judged yet.</p>`}
    ${judge._error ? `<p class="warn">${esc(judge._error.justification)}</p>` : ""}
    <h3>Gold</h3>
    ${(g.facts || []).length ? `<ul class="facts">${g.facts.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>` : `<p class="muted">Not covered by the textbook: the answer must be exactly ${OOC}.</p>`}
    ${it.notes ? `<p class="hint">Grader notes: ${esc(it.notes)}</p>` : ""}
    ${(g.pages || []).length ? `<p class="hint">Gold pages ${g.pages.join(", ")} · ${(g.chunk_ids || []).length} gold passages</p>` : ""}
    <h3>Sources (${(r.sources || []).length})</h3>
    <div id="d-sources">${(r.sources || []).map((s) => `
      <details class="src" id="src-${s.n}">
        <summary><span class="n">${s.n}</span><span>${esc((s.section_path || []).slice(-1)[0] || s.header || s.chunk_id)}</span>
          <span class="p">${(g.chunk_ids || []).includes(s.chunk_id) ? "gold · " : ""}p. ${(s.pages || []).join(", ")}</span></summary>
        <div class="src__body"><div class="prose" data-src="${s.n}"></div></div>
      </details>`).join("") || `<p class="muted">None.</p>`}</div>
    <h3>Measurements</h3>
    <dl class="kv">
      <dt>Latency</dt><dd>${secs(r.latency_ms)} total · first token ${secs(r.ttft_ms)}</dd>
      <dt>Stages</dt><dd>${Object.entries(tm).filter(([k]) => k !== "total").map(([k, v]) => `${esc(k)} ${(v / 1000).toFixed(1)} s`).join(" · ") || "–"}</dd>
      <dt>Tokens</dt><dd>${r.usage?.prompt_tokens ?? "–"} prompt · ${r.usage?.completion_tokens ?? "–"} completion</dd>
      <dt>Citations</dt><dd>coverage ${pc(det.citation_coverage)} · invalid ${pc(det.invalid_citation_rate)} · ${det.n_citations ?? 0} markers</dd>
      <dt>Retrieval</dt><dd>hit@k ${det.hit_at_k == null ? "–" : det.hit_at_k ? "yes" : "no"} · MRR ${f2(det.mrr)} · page hit ${det.page_hit == null ? "–" : det.page_hit ? "yes" : "no"}</dd>
      ${det.image_returned != null ? `<dt>Image</dt><dd>${det.image_returned ? "returned" : "none"}${det.image_correct != null ? ` · ${det.image_correct ? "gold figure" : "not the gold figure"}` : ""}</dd>` : ""}
      ${det.must_not_ok != null ? `<dt>Forbidden text</dt><dd class="${det.must_not_ok ? "good" : "bad"}">${det.must_not_ok ? "none" : esc((det.must_not_hits || []).join(", "))}</dd>` : ""}
      <dt>Route</dt><dd>${esc(r.decision?.intent || "–")} · ${esc(r.decision?.answer_format || "")}${r.decision?.in_scope === false ? " · router: out of scope" : ""}</dd>
      <dt>Session</dt><dd class="mono-ish">${esc(r.session_id || "–")}</dd>
    </dl>`;

  // conversation turns (earlier answers rendered too)
  const tEl = $("d-turns");
  turns.forEach((t, i) => {
    const last = i === turns.length - 1;
    const u = document.createElement("div");
    u.className = "turn turn--user";
    u.innerHTML = `<span class="turn__who">User${turns.length > 1 ? ` · turn ${i + 1}` : ""}</span>${esc(t.question)}` +
      (t.decision?.standalone_query && t.decision.standalone_query.trim().toLowerCase() !== (t.question || "").trim().toLowerCase()
        ? `<p class="standalone">Rewritten as <code>${esc(t.decision.standalone_query)}</code></p>` : "");
    tEl.append(u);
    if (!last) {
      const a = document.createElement("div");
      a.className = "turn turn--prev";
      a.innerHTML = `<span class="turn__who">Assistant</span><div class="prose"></div>`;
      renderMarkdown(a.querySelector(".prose"), t.answer || "");
      tEl.append(a);
    }
  });
  if (!turns.length) tEl.innerHTML = `<div class="turn turn--user">${esc(r.question || "")}</div>`;

  const ans = $("d-answer");
  if ((r.answer || "").trim() === OOC) ans.textContent = OOC;
  else renderMarkdown(ans, r.answer || "");
  const valid = new Set((r.sources || []).map((s) => s.n));
  ans.querySelectorAll(".cite__chip").forEach((b) => { if (!valid.has(+b.dataset.n)) { b.classList.add("is-missing"); b.title = "No such source"; } });
  (r.sources || []).forEach((s) => {
    const body = el.querySelector(`[data-src="${s.n}"]`);
    if (body) renderMarkdown(body, s.content || s.snippet || "");
  });
}

// ---------------------------------------------------------------- jobs
function renderJob() {
  const live = state.runs.find((r) => LIVE.has(r.status) || r.job?.running);
  const panel = $("job");
  if (!live) {
    panel.hidden = true;
    if (state.poll) { clearInterval(state.poll); state.poll = null; if (state.sel != null) selectRun(state.sel); }
    return;
  }
  panel.hidden = false;
  const p = live.progress || {};
  const total = p.total || live.n_items || 0, done = p.done || 0;
  $("job-title").textContent = `${live.status === "judging" ? "Judging" : live.status === "cancelling" ? "Stopping" : "Answering"} · ${live.label} #${live.run_id}`;
  $("job-bar").style.width = total ? `${(100 * done) / total}%` : "0";
  $("job-meta").textContent = `${done} / ${total}${p.current ? ` · now ${p.current}` : ""}${live.job && !live.job.running && live.job.returncode ? ` · process exited with code ${live.job.returncode}` : ""}`;
  $("job-cancel").dataset.run = live.run_id;
  api(`/api/eval/runs/${live.run_id}/log?tail=40`).then((l) => {
    const pre = $("job-log");
    const atEnd = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 4;
    pre.textContent = (l.lines || []).join("\n");
    if (atEnd) pre.scrollTop = pre.scrollHeight;
  }).catch(() => {});
  if (!state.poll) state.poll = setInterval(loadRuns, 4000);
}

async function startRun(ev) {
  ev.preventDefault();
  const label = $("nr-label").value.trim();
  if (!label) { $("nr-label").setAttribute("aria-invalid", "true"); $("nr-msg").textContent = "Give the run a label."; $("nr-label").focus(); return; }
  $("nr-label").removeAttribute("aria-invalid");
  const scenarios = [...document.querySelectorAll("#nr-scenarios input:checked")].map((i) => i.value);
  const limit = +$("nr-limit").value || null;
  $("nr-submit").disabled = true;
  try {
    const r = await api("/api/eval/runs", { method: "POST", body: { label, scenarios: scenarios.length ? scenarios : null, limit, judge: $("nr-judge").checked } });
    toast(`Run #${r.run_id} started`);
    $("new-run").hidden = true; $("new-run-toggle").setAttribute("aria-expanded", "false");
    $("nr-msg").textContent = "";
    await loadRuns();
  } catch (e) { $("nr-msg").textContent = e.message; $("nr-msg").className = "form-msg is-error"; }
  finally { $("nr-submit").disabled = false; }
}

// ---------------------------------------------------------------- wiring
function init() {
  api("/api/eval/meta").then((m) => {
    state.meta = m;
    $("nr-scenarios").innerHTML = m.scenarios.map((s) => `<label class="chip"><input type="checkbox" value="${s}"><span>${esc(SCENARIO_LABEL(s))} · ${m.bank.by_scenario[s] || 0}</span></label>`).join("");
  }).catch((e) => toast(e.message, { error: true }));

  $("refresh").addEventListener("click", () => { loadRuns(); if (state.sel != null) selectRun(state.sel); });
  $("new-run-toggle").addEventListener("click", () => {
    const f = $("new-run"); f.hidden = !f.hidden;
    $("new-run-toggle").setAttribute("aria-expanded", String(!f.hidden));
    if (!f.hidden) $("nr-label").focus();
  });
  $("nr-cancel").addEventListener("click", () => { $("new-run").hidden = true; $("new-run-toggle").setAttribute("aria-expanded", "false"); });
  $("new-run").addEventListener("submit", startRun);
  $("job-cancel").addEventListener("click", async () => {
    try { await api(`/api/eval/runs/${$("job-cancel").dataset.run}/cancel`, { method: "POST" }); toast("Stopping after the current item"); loadRuns(); }
    catch (e) { toast(e.message, { error: true }); }
  });

  $("runs-table").addEventListener("click", async (ev) => {
    const tick = ev.target.closest("[data-tick]");
    if (tick) {
      const id = +tick.dataset.tick;
      state.ticked = tick.checked ? [...state.ticked.filter((x) => x !== id), id].slice(-2) : state.ticked.filter((x) => x !== id);
      await updateCompare();
      return;
    }
    const j = ev.target.closest("[data-judge]");
    if (j) {
      const id = +j.dataset.judge;
      try { await api(`/api/eval/runs/${id}/judge${j.textContent.trim() === "Re-judge" ? "?rejudge=true" : ""}`, { method: "POST" }); toast(`Judging run #${id}`); loadRuns(); }
      catch (e) { toast(e.message, { error: true }); }
      return;
    }
    const del = ev.target.closest("[data-del]");
    if (del) {
      const id = +del.dataset.del;
      if (!confirm(`Delete run #${id} and all its results?`)) return;
      try { await api(`/api/eval/runs/${id}`, { method: "DELETE" }); toast(`Deleted run #${id}`); if (state.sel === id) state.sel = null; state.ticked = state.ticked.filter((x) => x !== id); state.cmp = null; loadRuns(); }
      catch (e) { toast(e.message, { error: true }); }
      return;
    }
    const tr = ev.target.closest("tr[data-id]");
    if (tr) selectRun(+tr.dataset.id);
  });

  $("view-seg").addEventListener("click", (ev) => {
    const b = ev.target.closest("button[data-view]");
    if (!b || b.disabled) return;
    state.view = b.dataset.view;
    renderMatrix(); renderItems();
  });
  const heat = $("heat");
  heat.addEventListener("mouseover", (ev) => { const td = ev.target.closest("td.cell"); if (td) showTip(td, ev); });
  heat.addEventListener("mousemove", (ev) => { const td = ev.target.closest("td.cell"); if (td) showTip(td, ev); else $("tip").hidden = true; });
  heat.addEventListener("mouseleave", () => { $("tip").hidden = true; });
  heat.addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-row]") || ev.target.closest("td.cell");
    if (!b) return;
    filterScenario(b.dataset.row || b.dataset.s);
  });

  // classification checks: cell tooltips + click -> item list -> item drill-down
  const cg = $("conf-grid");
  cg.addEventListener("pointermove", (ev) => { const b = ev.target.closest(".cm__cell"); if (b) confTip(b, ev); else cvHide(); });
  cg.addEventListener("pointerleave", () => cvHide());
  cg.addEventListener("focusin", (ev) => { const b = ev.target.closest(".cm__cell"); if (b) confTip(b, ev); });
  cg.addEventListener("focusout", () => cvHide());
  cg.addEventListener("click", (ev) => {
    const b = ev.target.closest(".cm__cell");
    if (!b) return;
    cvHide();
    const same = state.confSel?.matrix === b.dataset.cm && state.confSel?.cell === b.dataset.cell;
    state.confSel = same ? null : { matrix: b.dataset.cm, cell: b.dataset.cell };
    renderConfusion(state.data.aggregate);
    if (!state.confSel) renderConfItems();
    else $("conf-items").scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  });
  const ci = $("conf-items");
  const openFromList = (tr) => { showItem(tr.dataset.item); document.getElementById("items").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" }); };
  ci.addEventListener("click", (ev) => {
    if (ev.target.closest("#conf-close")) { state.confSel = null; renderConfusion(state.data.aggregate); renderConfItems(); return; }
    const tr = ev.target.closest("tr[data-item]");
    if (tr) openFromList(tr);
  });
  ci.addEventListener("keydown", (ev) => { const tr = ev.target.closest("tr[data-item]"); if (tr && (ev.key === "Enter" || ev.key === " ")) { ev.preventDefault(); openFromList(tr); } });

  // chart controls
  $("metric-seg").addEventListener("click", (ev) => { const b = ev.target.closest("button[data-avg]"); if (!b) return; state.avgMode = b.dataset.avg; renderCharts(); });
  $("lat-seg").addEventListener("click", (ev) => { const b = ev.target.closest("button[data-lat]"); if (!b) return; state.latKey = b.dataset.lat; renderCharts(); });
  $("sc-reset").addEventListener("click", () => { state.zoom = null; renderCharts(); });
  $("items-lede").addEventListener("click", (ev) => { if (ev.target.closest("#lat-clear")) { state.latRange = null; renderItems(); renderCharts(); } });

  ["f-scenario", "f-failed"].forEach((id) => $(id).addEventListener("change", renderItems));
  $("f-text").addEventListener("input", renderItems);
  $("items-table").addEventListener("click", (ev) => { const tr = ev.target.closest("tr[data-item]"); if (tr) showItem(tr.dataset.item); });
  $("detail").addEventListener("click", (ev) => {
    const chip = ev.target.closest(".cite__chip");
    if (!chip) return;
    const d = document.getElementById(`src-${chip.dataset.n}`);
    if (!d) return;
    d.open = true;
    d.scrollIntoView({ block: "nearest" });
    d.classList.add("is-flash");
    setTimeout(() => d.classList.remove("is-flash"), 1200);
  });

  const markTab = () => {
    const h = location.hash || "#runs";
    document.querySelectorAll(".tab").forEach((a) => (a.getAttribute("href") === h ? a.setAttribute("aria-current", "location") : a.removeAttribute("aria-current")));
  };
  window.addEventListener("hashchange", markTab);
  markTab();
  loadRuns();
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
else init();
