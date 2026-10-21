// Architecture tab: pipeline flow diagram (inline SVG, top -> bottom) from GET /api/arch,
// plus editors for the versioned components behind each step (GET/POST /api/components...).
import { api, toast, escapeHtml, fmtInt } from "./api.js";
import { showTip, hideTip } from "./charts.js";

const $ = (id) => document.getElementById(id);
const KINDS = {
  io: { label: "Input / output", color: "#95A1AB" },
  guardrail: { label: "Guardrail", color: "#d55181" },
  memory: { label: "Memory", color: "#9085e9" },
  llm: { label: "LLM call", color: "#D6B25E" },
  model: { label: "Local model", color: "#c98500" },
  retrieval: { label: "Retrieval", color: "#3987e5" },
  prompt: { label: "Prompt assembly", color: "#199e70" },
  postprocess: { label: "Post-processing", color: "#6da7ec" },
};
KINDS.guard = KINDS.guardrail;
KINDS.build = KINDS.prompt;
const COMPONENT_INFO = {
  answer_system_prompt: "System prompt for the answer model (rules, citation format, #OutofContext policy).",
  router_prompt: "Instructions for the Jev-style routing decision (structured output).",
  form_notes: "NOTE lines added to the prompt for each detected question form.",
  table_rules: "Extra rules used when the answer format is a table.",
  compaction_prompt: "How session memory is summarised when it exceeds its budget.",
  title_prompt: "How a conversation title is generated.",
  grounding_prompt: "Grounding check: scores how well the answer is supported by its sources.",
  retrieval_params: "Retrieval knobs: candidate counts, RRF k, rerank depth, passages per answer, thresholds.",
  modes: "Answer-mode profiles (Swift / Deep): model, routing, retrieval depth, token limits.",
};
const humanComp = (c) => c.replace(/_/g, " ").replace(/^./, (x) => x.toUpperCase());
const shortNum = (n) => (n >= 1e6 ? `${+(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${+(n / 1e3).toFixed(1)}k` : String(Math.round(n)));
const fmtMs = (ms) => (ms == null ? "–" : ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);
// stats arrive as numbers; guard against strings from other backends
const fmtDate = (iso) => { const d = new Date(iso); return Number.isNaN(+d) ? "" : d.toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); };

export function initArch({ confirmDialog }) {
  const A = { graph: null, selected: null, ro: null, comps: new Map(), dirty: new Set() };

  async function load() {
    $("arch-error").hidden = true;
    $("architecture").classList.add("is-loading");
    try {
      const mode = $("arch-mode") ? $("arch-mode").value : "";
      const g = await api(`/api/arch?days=${encodeURIComponent($("arch-days").value)}${mode ? `&mode=${encodeURIComponent(mode)}` : ""}`);
      A.graph = normGraph(g || {});
      draw();
      const other = Object.entries(A.graph.other || {}).filter(([, v]) => v && v.calls);
      $("arch-other").textContent = other.length ? `Also logged outside the answer path: ${other.map(([p, v]) => `${p} (${fmtInt(v.calls)} calls, ${fmtMs(v.avg_ms)} avg)`).join(" · ")}` : "";
      $("arch-other").hidden = !other.length;
      if (A.selected && A.graph.nodes.some((n) => n.id === A.selected)) select(A.selected, { keepPanel: true });
    } catch (err) {
      $("arch-error").textContent = err.status === 404 ? "The architecture endpoint is not available on this server yet." : `Could not load the architecture: ${err.message}`;
      $("arch-error").hidden = false;
    } finally {
      $("architecture").classList.remove("is-loading");
    }
  }

  function normGraph(g) {
    const nodes = (g.nodes || []).map((n) => ({
      id: String(n.id), label: n.label || n.name || String(n.id), kind: n.kind || n.type || "build",
      group: n.group || null, description: n.description || n.detail || "",
      components: (n.components || n.component_ids || []).map(String), stats: n.stats || null,
      purposes: n.purposes || (n.stats && n.stats.purposes) || [], versions: n.component_versions || {},
      rank: Number.isFinite(+n.rank) && n.rank !== null && n.rank !== undefined ? +n.rank : null,
      lane: Number.isFinite(+n.lane) ? +n.lane : 0,
    }));
    const ids = new Set(nodes.map((n) => n.id));
    const edges = (g.edges || []).map((e) => ({ from: String(e.from ?? e.source), to: String(e.to ?? e.target), label: e.label || "", kind: e.kind || "main" }))
      .filter((e) => ids.has(e.from) && ids.has(e.to));
    return { nodes, edges, versions: g.components || {}, other: g.other_purposes || {} };
  }

  /* ---------- layout: layered DAG (back edges removed), long edges routed in side lanes ---------- */
  function layout(W) {
    const { nodes, edges } = A.graph;
    const idx = new Map(nodes.map((n, i) => [n.id, i]));
    const out = new Map(nodes.map((n) => [n.id, []]));
    edges.forEach((e) => out.get(e.from).push(e));
    // layers: the server's `rank` hints when every node has one, else longest path over a DFS-acyclic graph
    const ranked = nodes.length && nodes.every((n) => n.rank != null);
    const back = new Set();
    const layer = new Map();
    if (ranked) {
      const ranks = [...new Set(nodes.map((n) => n.rank))].sort((a, b) => a - b);
      nodes.forEach((n) => layer.set(n.id, ranks.indexOf(n.rank)));
      edges.forEach((e) => { if (layer.get(e.to) <= layer.get(e.from)) back.add(e); });
    } else {
      const state = new Map();
      const dfs = (id) => {
        state.set(id, 1);
        for (const e of out.get(id)) {
          const s = state.get(e.to);
          if (s === 1) back.add(e);
          else if (!s) dfs(e.to);
        }
        state.set(id, 2);
      };
      nodes.forEach((n) => { if (!state.get(n.id)) dfs(n.id); });
      nodes.forEach((n) => layer.set(n.id, 0));
      const f0 = edges.filter((e) => !back.has(e));
      for (let pass = 0; pass < nodes.length; pass++) {
        let changed = false;
        for (const e of f0) if (layer.get(e.to) < layer.get(e.from) + 1) { layer.set(e.to, layer.get(e.from) + 1); changed = true; }
        if (!changed) break;
      }
    }
    const fwd = edges.filter((e) => !back.has(e));
    const L = Math.max(0, ...layer.values()) + 1;
    const layers = Array.from({ length: L }, () => []);
    nodes.forEach((n) => layers[layer.get(n.id)].push(n));
    layers.forEach((ls) => ls.sort((a, b) => (a.lane - b.lane) || (idx.get(a.id) - idx.get(b.id))));
    const useLanes = nodes.some((n) => n.lane);

    const laneW = 22;
    const place = (rightLanes) => {
      const padL = 22 + Math.max(1, back.size) * laneW, padR = 22 + Math.max(1, rightLanes) * laneW;
      const maxPer = Math.max(2, ...layers.map((l) => l.length));
      const gapX = 14;
      const nodeW = Math.max(150, Math.min(300, (W - padL - padR - gapX * (maxPer - 1)) / maxPer));
      const nodeH = 58, gapY = 34, groupHead = 20;
      const pos = new Map();
      const mid = padL + (W - padL - padR) / 2;
      let y = 14;
      const groupFirst = new Map();
      layers.forEach((ls, li) => {
        const startsGroup = ls.some((n) => n.group && !groupFirst.has(n.group));
        if (startsGroup) { ls.forEach((n) => { if (n.group && !groupFirst.has(n.group)) groupFirst.set(n.group, li); }); y += groupHead; }
        const lanesDistinct = new Set(ls.map((n) => n.lane)).size === ls.length;
        if (ls.length === 1 && !ls[0].lane) {
          const w = Math.min(330, nodeW * 2 + gapX);
          pos.set(ls[0].id, { x: mid - w / 2, y, w, h: nodeH, layer: li });
        } else if (useLanes && lanesDistinct) {
          ls.forEach((n) => pos.set(n.id, { x: mid + n.lane * (nodeW / 2 + gapX / 2) - nodeW / 2, y, w: nodeW, h: nodeH, layer: li }));
        } else {
          const total = ls.length * nodeW + (ls.length - 1) * gapX;
          const x0 = mid - total / 2;
          ls.forEach((n, i) => pos.set(n.id, { x: x0 + i * (nodeW + gapX), y, w: nodeW, h: nodeH, layer: li }));
        }
        y += nodeH + gapY;
        const next = layers[li + 1] || [];
        if (ls.some((n) => n.group && !next.some((m) => m.group === n.group))) y += 8;
      });
      // a skip edge is drawn directly when its straight path crosses no node; otherwise it takes a right-side lane
      const lane = new Set();
      for (const e of fwd) {
        const a = pos.get(e.from), b = pos.get(e.to);
        if (b.layer - a.layer <= 1) continue;
        const sx = a.x + a.w / 2, tx = b.x + b.w / 2;
        const x0 = Math.min(sx, tx) - 8, x1 = Math.max(sx, tx) + 8;
        const blocked = nodes.some((n) => { const p = pos.get(n.id); return p.layer > a.layer && p.layer < b.layer && p.x < x1 && p.x + p.w > x0; });
        if (blocked) lane.add(e);
      }
      return { pos, padL, padR, H: y - gapY + 18, groupHead, lane };
    };
    let r = place(fwd.filter((e) => layer.get(e.to) - layer.get(e.from) > 1).length);
    const r2 = place(r.lane.size);
    if (r2.lane.size <= r.lane.size) r = r2;
    return { pos: r.pos, back, fwd, layer, H: r.H, padL: r.padL, padR: r.padR, laneW, groupHead: r.groupHead, sideLane: r.lane };
  }

  function draw() {
    const box = $("arch-svg");
    const W = Math.max(320, Math.floor(box.clientWidth || 640));
    if (!A.graph.nodes.length) { box.innerHTML = `<p class="muted">The server returned an empty graph.</p>`; return; }
    const lay = layout(W);
    const { pos, back, fwd, layer, H } = lay;
    let s = `<svg class="flow" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="group" aria-label="Pipeline flow diagram">`;
    s += `<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="flow__head"/></marker>
      <marker id="arr-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="flow__head flow__head--back"/></marker></defs>`;
    // group frames
    const groups = new Map();
    A.graph.nodes.forEach((n) => { if (n.group) { const p = pos.get(n.id); const g = groups.get(n.group) || { x0: 1e9, y0: 1e9, x1: -1e9, y1: -1e9 }; g.x0 = Math.min(g.x0, p.x); g.y0 = Math.min(g.y0, p.y); g.x1 = Math.max(g.x1, p.x + p.w); g.y1 = Math.max(g.y1, p.y + p.h); groups.set(n.group, g); } });
    for (const [name, g] of groups) {
      s += `<rect class="flow__group" x="${g.x0 - 10}" y="${g.y0 - lay.groupHead - 2}" width="${g.x1 - g.x0 + 20}" height="${g.y1 - g.y0 + lay.groupHead + 12}" rx="10"/>`;
      s += `<text class="flow__group-t" x="${g.x0 - 2}" y="${g.y0 - 8}">${escapeHtml(name.toUpperCase())}</text>`;
    }
    // edges: short side-lane edges take the inner lanes
    const labels = [];
    const spanOf = (e) => Math.abs(layer.get(e.to) - layer.get(e.from));
    const between = (a, b) => A.graph.nodes.map((n) => pos.get(n.id)).filter((p) => p.layer >= Math.min(a.layer, b.layer) && p.layer <= Math.max(a.layer, b.layer));
    const laneOf = new Map();
    [...back].sort((x, y) => spanOf(x) - spanOf(y)).forEach((e, i) => laneOf.set(e, i));
    [...lay.sideLane].sort((x, y) => spanOf(x) - spanOf(y)).forEach((e, i) => laneOf.set(e, i));
    const vLabel = (e, x, y0, y1, side) => {
      const run = Math.abs(y1 - y0) - 14;
      const max = Math.max(4, Math.floor(run / 6.3));
      const t = e.label.length > max ? `${e.label.slice(0, max - 1)}…` : e.label;
      const ly = (y0 + y1) / 2;
      return `<text class="flow__elabel${side === "l" ? " is-back" : ""}" x="${x}" y="${ly}" text-anchor="middle" transform="rotate(${side === "l" ? -90 : 90} ${x} ${ly})"><title>${escapeHtml(e.label)}</title>${escapeHtml(t)}</text>`;
    };
    for (const e of [...fwd, ...back]) {
      const a = pos.get(e.from), b = pos.get(e.to);
      const isBack = back.has(e);
      let d;
      if (isBack) {
        const sx = a.x, sy = a.y + a.h / 2, tx = b.x, ty = b.y + b.h / 2;
        const gx = Math.min(...between(a, b).map((p) => p.x)) - 18 - laneOf.get(e) * lay.laneW;
        d = `M${sx},${sy} H${gx + 6} Q${gx},${sy} ${gx},${sy - 6} V${ty + 6} Q${gx},${ty} ${gx + 6},${ty} H${tx - 1}`;
        if (e.label) labels.push(vLabel(e, gx - 6, sy, ty, "l"));
      } else if (lay.sideLane.has(e)) {
        const sx = a.x + a.w, sy = a.y + a.h / 2, tx = b.x + b.w, ty = b.y + b.h / 2;
        const gx = Math.max(...between(a, b).map((p) => p.x + p.w)) + 18 + laneOf.get(e) * lay.laneW;
        d = `M${sx},${sy} H${gx - 6} Q${gx},${sy} ${gx},${sy + 6} V${ty - 6} Q${gx},${ty} ${gx - 6},${ty} H${tx + 1}`;
        if (e.label) labels.push(vLabel(e, gx + 6, sy, ty, "r"));
      } else {
        const sx = a.x + a.w / 2, sy = a.y + a.h, tx = b.x + b.w / 2, ty = b.y;
        const my = (sy + ty) / 2;
        d = Math.abs(sx - tx) < 1 ? `M${sx},${sy} V${ty - 1}` : `M${sx},${sy} C${sx},${my} ${tx},${my} ${tx},${ty - 1}`;
        if (e.label) {
          const lx = (sx + tx) / 2 + 8, ly = spanOf(e) > 1 ? my : Math.min(my + 4, ty - 4);
          const max = Math.max(6, Math.floor((W - lx - 8) / 6.3));
          labels.push(`<text class="flow__elabel" x="${lx}" y="${ly}"><title>${escapeHtml(e.label)}</title>${escapeHtml(e.label.length > max ? `${e.label.slice(0, max - 1)}…` : e.label)}</text>`);
        }
      }
      const cls = isBack ? " flow__edge--back" : e.kind === "alt" ? " flow__edge--alt" : "";
      s += `<path class="flow__edge${cls}" d="${d}" marker-end="url(#${isBack ? "arr-b" : "arr"})" data-from="${escapeHtml(e.from)}" data-to="${escapeHtml(e.to)}"><title>${escapeHtml(`${e.from} → ${e.to}${e.label ? `: ${e.label}` : ""}`)}</title></path>`;
    }
    // nodes
    for (const n of A.graph.nodes) {
      const p = pos.get(n.id);
      const k = KINDS[n.kind] || KINDS.build;
      const st = n.stats;
      const hasCalls = st && st.calls != null;
      const statTxt = hasCalls ? `${fmtInt(st.calls)} calls · ${fmtMs(st.avg_ms)} · ${shortNum(+st.tokens || 0)} tok` : "";
      const maxChars = Math.floor((p.w - 26) / 7.2);
      const lab = n.label.length > maxChars ? `${n.label.slice(0, maxChars - 1)}…` : n.label;
      const sub = n.description || "";
      const subMax = Math.floor((p.w - 26) / 6.1);
      const nComp = n.components.length;
      s += `<g class="flow__node${A.selected === n.id ? " is-sel" : ""}" data-node="${escapeHtml(n.id)}" tabindex="0" role="button" aria-pressed="${A.selected === n.id}"
          aria-label="${escapeHtml(n.label)}${statTxt ? `, ${escapeHtml(statTxt)}` : ""}${nComp ? `, ${nComp} editable component${nComp > 1 ? "s" : ""}` : ""}">
        <rect class="flow__box" x="${p.x}" y="${p.y}" width="${p.w}" height="${p.h}" rx="7"/>
        <rect x="${p.x}" y="${p.y + 8}" width="3" height="${p.h - 16}" rx="1.5" fill="${k.color}"/>
        <text class="flow__label" x="${p.x + 14}" y="${p.y + 21}">${escapeHtml(lab)}</text>
        ${hasCalls ? `<text class="flow__stat" x="${p.x + 14}" y="${p.y + 42}">${escapeHtml(statTxt)}</text>` : `<text class="flow__sub" x="${p.x + 14}" y="${p.y + 42}">${escapeHtml(sub.length > subMax ? `${sub.slice(0, subMax - 1)}…` : sub)}</text>`}
        ${nComp ? `<g class="flow__badge"><rect x="${p.x + p.w - 30}" y="${p.y + 8}" width="22" height="16" rx="8"/><text x="${p.x + p.w - 19}" y="${p.y + 20}" text-anchor="middle">${nComp}</text></g>` : ""}
      </g>`;
    }
    s += labels.join("") + `</svg>`;
    box.innerHTML = s;
    box.querySelectorAll(".flow__node").forEach((g) => {
      const id = g.dataset.node;
      const n = A.graph.nodes.find((x) => x.id === id);
      const tipFor = (e) => showTip(e, {
        title: n.label,
        rows: [
          ...(n.stats && n.stats.calls != null ? [{ label: "Calls", value: fmtInt(n.stats.calls) }, { label: "Avg latency", value: fmtMs(n.stats.avg_ms) }, { label: "Tokens", value: fmtInt(n.stats.tokens) }] : []),
          ...(n.components.length ? [{ label: "Components", value: String(n.components.length) }] : []),
        ],
        foot: n.description,
      });
      g.addEventListener("pointermove", tipFor);
      g.addEventListener("pointerleave", hideTip);
      g.addEventListener("click", () => { hideTip(); select(id); });
      g.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(id); }
        if (e.key === "ArrowDown" || e.key === "ArrowUp") {
          e.preventDefault();
          const all = [...box.querySelectorAll(".flow__node")];
          const i = all.indexOf(g) + (e.key === "ArrowDown" ? 1 : -1);
          if (all[i]) all[i].focus();
        }
      });
      g.addEventListener("mouseenter", () => highlightEdges(id, true));
      g.addEventListener("mouseleave", () => highlightEdges(id, false));
    });
    // legend
    const kinds = [...new Set(A.graph.nodes.map((n) => n.kind))];
    $("arch-legend").innerHTML = kinds.map((k) => `<span class="arch__key"><span style="background:${(KINDS[k] || KINDS.build).color}"></span>${escapeHtml((KINDS[k] || { label: k }).label)}</span>`).join("")
      + `<span class="arch__key arch__key--edge"><svg width="22" height="8" aria-hidden="true"><path d="M1 4H20" class="flow__edge"/></svg>main path</span><span class="arch__key arch__key--edge"><svg width="22" height="8" aria-hidden="true"><path d="M1 4H20" class="flow__edge flow__edge--alt"/></svg>alternative path</span><span class="arch__key arch__key--back"><svg width="22" height="8" aria-hidden="true"><path d="M1 4H20" class="flow__edge flow__edge--back"/></svg>loop back</span><span class="arch__key"><span class="arch__badge">n</span>editable components</span>`;
  }

  function highlightEdges(id, on) {
    $("arch-svg").querySelectorAll(".flow__edge").forEach((p) => {
      if (p.dataset.from === id || p.dataset.to === id) p.classList.toggle("is-hl", on);
    });
  }

  /* ---------- side panel ---------- */
  async function select(id, { keepPanel = false } = {}) {
    if (!keepPanel && A.selected !== id && A.dirty.size) {
      const ok = await confirmDialog("Discard unsaved edits?", `You have unsaved changes in ${[...A.dirty].map(humanComp).join(", ")}. Switching steps discards them.`, "Discard");
      if (!ok) return;
    }
    if (!keepPanel) A.dirty.clear();
    A.selected = id;
    $("arch-svg").querySelectorAll(".flow__node").forEach((g) => {
      const on = g.dataset.node === id;
      g.classList.toggle("is-sel", on);
      g.setAttribute("aria-pressed", String(on));
    });
    const n = A.graph.nodes.find((x) => x.id === id);
    if (!n) return;
    if (keepPanel && $("arch-side").dataset.node === id) { renderStats(n); return; }
    const side = $("arch-side");
    side.dataset.node = id;
    const k = KINDS[n.kind] || KINDS.build;
    side.innerHTML = `
      <div class="side__head">
        <p class="side__kind"><span style="background:${k.color}"></span>${escapeHtml(k.label)}</p>
        <h2 class="side__title">${escapeHtml(n.label)}</h2>
        ${n.description ? `<p class="side__desc">${escapeHtml(n.description)}</p>` : ""}
        <dl class="side__stats" id="side-stats"></dl>
      </div>
      <div id="side-comps">${n.components.length ? n.components.map((c) => `<section class="comp" data-comp="${escapeHtml(c)}"><p class="muted">Loading ${escapeHtml(humanComp(c))}…</p></section>`).join("")
        : `<p class="muted side__none">This step has no editable components. Its behaviour is fixed in code${n.stats ? "" : " and it makes no model calls"}.</p>`}</div>`;
    renderStats(n);
    for (const c of n.components) loadComponent(c);
    if (matchMedia("(max-width: 1100px)").matches) side.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderStats(n) {
    const st = n.stats;
    const dl = $("side-stats");
    if (!dl) return;
    dl.innerHTML = st && st.calls != null
      ? `<div><dt>Calls</dt><dd class="num">${fmtInt(st.calls)}</dd></div><div><dt>Avg latency</dt><dd class="num">${fmtMs(st.avg_ms)}</dd></div>`
        + (st.recent_avg_ms != null ? `<div><dt>Last 20 avg</dt><dd class="num">${fmtMs(st.recent_avg_ms)}</dd></div>` : "")
        + (st.p95_ms != null ? `<div><dt>p95</dt><dd class="num">${fmtMs(st.p95_ms)}</dd></div>` : "")
        + `<div><dt>Tokens</dt><dd class="num">${fmtInt(st.tokens)}</dd></div>`
        + (st.errors ? `<div><dt>Errors</dt><dd class="num bad">${fmtInt(st.errors)}</dd></div>` : "")
        + (n.purposes.length ? `<div class="side__purp"><dt>Logged as</dt><dd>${n.purposes.map((p) => `<code>${escapeHtml(p)}</code>`).join(" ")}</dd></div>` : "")
        + (st.last_at ? `<div class="side__purp"><dt>Last call</dt><dd class="num">${escapeHtml(fmtDate(st.last_at))}</dd></div>` : "")
      : "";
    dl.hidden = !dl.innerHTML;
  }

  /* ---------- components ---------- */
  function shapeOf(content, kind) {
    if (typeof content === "string") return kind === "json" ? { kind: "json" } : { kind: "text" };
    if (content && typeof content === "object" && !Array.isArray(content)) {
      const keys = Object.keys(content);
      if (keys.length === 1 && ["text", "prompt", "template"].includes(keys[0]) && typeof content[keys[0]] === "string") return { kind: "text", wrap: keys[0] };
      if (keys.length && keys.every((k) => typeof content[k] === "string")) return { kind: "textmap" };
      if (keys.length && keys.every((k) => typeof content[k] === "number" || typeof content[k] === "boolean")) return { kind: "params" };
    }
    return { kind: "json" };
  }
  const textOf = (content) => (typeof content === "string" ? content : JSON.stringify(content, null, 2));

  async function loadComponent(c) {
    const sec = document.querySelector(`#side-comps .comp[data-comp="${CSS.escape(c)}"]`);
    if (!sec) return;
    try {
      const data = await api(`/api/components/${encodeURIComponent(c)}`);
      const active = data.active || {};
      const versions = (data.versions || data.history || []).slice().sort((a, b) => b.version - a.version);
      const st = { name: c, active, versions, view: null, mode: null, label: data.label, info: data.description, kind: data.kind, def: data.default };
      A.comps.set(c, st);
      renderComponent(sec, st);
    } catch (err) {
      sec.innerHTML = `<h3 class="comp__title">${escapeHtml(humanComp(c))}</h3><p class="load-error">${escapeHtml(err.status === 404 ? "Component endpoints are not available on this server yet." : `Could not load: ${err.message}`)}</p>`;
    }
  }

  function renderComponent(sec, st) {
    const c = st.name;
    const shape = shapeOf(st.active.content, st.kind);
    st.shape = shape;
    sec.classList.remove("is-dirty");
    st.mode = st.mode || (shape.kind === "params" || shape.kind === "textmap" ? "form" : "raw");
    const av = st.active.version;
    const id = `cmp-${c}`;
    sec.innerHTML = `
      <div class="comp__head">
        <h3 class="comp__title">${escapeHtml(st.label || humanComp(c))} <code class="comp__id">${escapeHtml(c)}</code></h3>
        <span class="vbadge vbadge--active">v${escapeHtml(av ?? "?")} active</span>
      </div>
      <p class="comp__info">${escapeHtml(st.info || COMPONENT_INFO[c] || "")}</p>
      ${shape.kind === "params" || shape.kind === "textmap" ? `<div class="seg seg--sm" role="group" aria-label="Editor view"><button type="button" data-view="form" aria-pressed="${st.mode === "form"}">Form</button><button type="button" data-view="raw" aria-pressed="${st.mode === "raw"}">JSON</button></div>` : ""}
      <div class="comp__editor"></div>
      <p class="field-error comp__err" role="alert" hidden></p>
      <div class="comp__save">
        <label class="sr-only" for="${id}-note">Version note</label>
        <input class="input" id="${id}-note" placeholder="What changed? (version note)" maxlength="200" autocomplete="off">
        <button type="button" class="btn btn--primary" data-save>Save as new version</button>
        <button type="button" class="btn" data-revert disabled>Revert edits</button>
        ${st.def != null && JSON.stringify(st.def) !== JSON.stringify(st.active.content) ? `<button type="button" class="btn" data-default title="Put the built-in code default into the editor">Load code default</button>` : ""}
      </div>
      <details class="comp__hist" open>
        <summary>Version history <span class="muted num">(${st.versions.length})</span></summary>
        <ol class="vlist">${st.versions.map((v) => `
          <li class="vrow${v.is_active ? " is-active" : ""}${st.view === v.version ? " is-view" : ""}">
            <span class="vrow__v num">v${escapeHtml(v.version)}</span>
            <span class="vrow__note">${escapeHtml(v.note || "No note")}<span class="vrow__date num">${escapeHtml(fmtDate(v.created_at))}${v.created_by ? ` · ${escapeHtml(v.created_by)}` : ""}</span></span>
            ${v.is_active ? `<span class="vbadge vbadge--active">Active</span>` : ""}
            <span class="vrow__acts">
              <button type="button" class="btn btn--sm" data-viewv="${v.version}">View</button>
              ${v.is_active ? "" : `<button type="button" class="btn btn--sm" data-diff="${v.version}">Diff</button><button type="button" class="btn btn--sm" data-activate="${v.version}">Activate</button>`}
            </span>
          </li>`).join("")}</ol>
      </details>
      <div class="comp__view" hidden></div>`;
    renderEditor(sec, st, st.active.content);
    sec.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => {
      const cur = readEditor(sec, st);
      if (cur.error) { showErr(sec, cur.error); return; }
      st.mode = b.dataset.view;
      sec.querySelectorAll("[data-view]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      renderEditor(sec, st, cur.value);
      if (isDirty(st, cur.value)) markDirty(sec, st, true);
    }));
    sec.querySelector("[data-save]").addEventListener("click", () => save(sec, st));
    sec.querySelector("[data-revert]").addEventListener("click", () => { renderEditor(sec, st, st.active.content); markDirty(sec, st, false); showErr(sec, ""); });
    sec.querySelector("[data-default]")?.addEventListener("click", () => {
      renderEditor(sec, st, st.def);
      markDirty(sec, st, isDirty(st, st.def));
      sec.querySelector(`#cmp-${CSS.escape(st.name)}-note`).value = "Back to code default";
    });
    sec.querySelectorAll("[data-viewv]").forEach((b) => b.addEventListener("click", () => viewVersion(sec, st, +b.dataset.viewv)));
    sec.querySelectorAll("[data-diff]").forEach((b) => b.addEventListener("click", () => showDiff(sec, st, +b.dataset.diff)));
    sec.querySelectorAll("[data-activate]").forEach((b) => b.addEventListener("click", () => activate(sec, st, +b.dataset.activate)));
  }

  function paramRule(k) {
    if (/threshold|score|temperature|_min_|min_score/.test(k)) return { min: 0, max: 1, int: false, hint: "0 – 1" };
    if (/top_k|_k$|^k$|top_n|_n$|tokens|turns|budget|ctx/.test(k)) return { min: k === "rrf_k" ? 1 : 0, max: 100000, int: true, hint: "whole number" };
    return { min: null, max: null, int: false, hint: "" };
  }

  function renderEditor(sec, st, content) {
    const ed = sec.querySelector(".comp__editor");
    const c = st.name;
    const shape = st.shape;
    if (shape.kind === "text") {
      const text = shape.wrap ? (content && content[shape.wrap]) ?? "" : typeof content === "string" ? content : textOf(content);
      ed.innerHTML = `<label class="sr-only" for="ed-${c}">${escapeHtml(humanComp(c))}</label><textarea class="textarea mono comp__text" id="ed-${c}" spellcheck="false" rows="${Math.min(18, Math.max(5, text.split("\n").length + 1))}">${escapeHtml(text)}</textarea><p class="hint comp__count num"></p>`;
      const ta = ed.querySelector("textarea");
      const count = () => { ed.querySelector(".comp__count").textContent = `${fmtInt(ta.value.length)} characters · ${fmtInt(ta.value.split("\n").length)} lines`; };
      count();
      ta.addEventListener("input", () => { count(); onEdit(sec, st); });
    } else if (shape.kind === "params" && st.mode === "form") {
      ed.innerHTML = `<div class="pform">${Object.entries(content || {}).map(([k, v]) => {
        const r = paramRule(k);
        const fid = `p-${c}-${k}`;
        if (typeof v === "boolean") return `<label class="switch pform__bool" for="${fid}"><input type="checkbox" id="${fid}" data-pk="${escapeHtml(k)}" data-pt="bool"${v ? " checked" : ""}><span class="switch__track" aria-hidden="true"></span><span>${escapeHtml(k)}</span></label>`;
        return `<div class="field"><label for="${fid}"><code>${escapeHtml(k)}</code>${r.hint ? ` <span class="muted">${escapeHtml(r.hint)}</span>` : ""}</label><input class="input num" type="number" id="${fid}" data-pk="${escapeHtml(k)}" data-pt="num" step="${r.int ? 1 : "any"}"${r.min != null ? ` min="${r.min}"` : ""}${r.max != null ? ` max="${r.max}"` : ""} value="${escapeHtml(v)}"><p class="field-error" id="${fid}-e" role="alert" hidden></p></div>`;
      }).join("")}</div>`;
      ed.querySelectorAll("input").forEach((i) => i.addEventListener("input", () => { validateParam(i); onEdit(sec, st); }));
    } else if (shape.kind === "textmap" && st.mode === "form") {
      ed.innerHTML = `<div class="tmap">${Object.entries(content || {}).map(([k, v]) => `<div class="field"><label for="t-${c}-${k}"><code>${escapeHtml(k)}</code></label><textarea class="textarea mono" id="t-${c}-${k}" data-tk="${escapeHtml(k)}" rows="${Math.min(8, Math.max(2, Math.ceil(String(v).length / 70)))}" spellcheck="false">${escapeHtml(v)}</textarea></div>`).join("")}</div>
        <p class="hint">To add or remove keys, switch to the JSON view.</p>`;
      ed.querySelectorAll("textarea").forEach((t) => t.addEventListener("input", () => onEdit(sec, st)));
    } else {
      const text = textOf(content);
      ed.innerHTML = `<label class="sr-only" for="ed-${c}">${escapeHtml(humanComp(c))} JSON</label><textarea class="textarea mono comp__text comp__json" id="ed-${c}" spellcheck="false" rows="${Math.min(22, Math.max(6, text.split("\n").length + 1))}">${escapeHtml(text)}</textarea><p class="hint comp__jsonmsg" aria-live="polite">Valid JSON.</p>`;
      const ta = ed.querySelector("textarea");
      ta.addEventListener("input", () => {
        const r = parseJson(ta.value, c);
        const m = ed.querySelector(".comp__jsonmsg");
        m.textContent = r.error || "Valid JSON.";
        m.classList.toggle("is-error", !!r.error);
        ta.setAttribute("aria-invalid", r.error ? "true" : "false");
        onEdit(sec, st);
      });
    }
  }

  function validateParam(i) {
    if (i.dataset.pt !== "num") return "";
    const k = i.dataset.pk, r = paramRule(k);
    const raw = i.value.trim();
    let err = "";
    const v = Number(raw);
    if (raw === "" || !Number.isFinite(v)) err = `${k} needs a number`;
    else if (r.int && !Number.isInteger(v)) err = `${k} must be a whole number`;
    else if (r.min != null && v < r.min) err = `${k} must be ≥ ${r.min}`;
    else if (r.max != null && v > r.max) err = `${k} must be ≤ ${r.max}`;
    i.setAttribute("aria-invalid", err ? "true" : "false");
    const e = document.getElementById(`${i.id}-e`);
    if (e) { e.textContent = err; e.hidden = !err; }
    return err;
  }

  function parseJson(text, c) {
    let v;
    try { v = JSON.parse(text); } catch (e) {
      const m = /position (\d+)/.exec(e.message);
      let where = "";
      if (m) { const before = text.slice(0, +m[1]); where = ` (line ${before.split("\n").length}, column ${before.length - before.lastIndexOf("\n")})`; }
      return { error: `Invalid JSON${where}: ${e.message.replace(/^JSON\.parse: /, "")}` };
    }
    if (c === "modes") {
      const entries = Array.isArray(v) ? v.map((m) => [m && m.id, m]) : v && typeof v === "object" ? Object.entries(v) : null;
      if (!entries || !entries.length) return { error: "modes must be an object (or list) of mode profiles." };
      for (const [id, m] of entries) {
        if (!id) return { error: "Every mode needs an id." };
        if (!m || typeof m !== "object") return { error: `Mode ${id} must be an object.` };
        const s = m.settings ?? m;
        if (typeof s !== "object") return { error: `Mode ${id}: settings must be an object.` };
      }
    }
    if (c === "retrieval_params") {
      if (!v || typeof v !== "object" || Array.isArray(v)) return { error: "retrieval_params must be an object." };
      for (const [k, x] of Object.entries(v)) {
        if (typeof x === "number") { const r = paramRule(k); if ((r.min != null && x < r.min) || (r.max != null && x > r.max) || (r.int && !Number.isInteger(x))) return { error: `${k} is out of range (${r.hint}).` }; }
      }
    }
    return { value: v };
  }

  function readEditor(sec, st) {
    const shape = st.shape, ed = sec.querySelector(".comp__editor");
    if (shape.kind === "text") {
      const t = ed.querySelector("textarea").value;
      if (!t.trim()) return { error: "The prompt cannot be empty." };
      return { value: shape.wrap ? { ...st.active.content, [shape.wrap]: t } : t };
    }
    if (shape.kind === "params" && st.mode === "form") {
      const out = { ...(st.active.content || {}) };
      const errs = [];
      ed.querySelectorAll("[data-pk]").forEach((i) => {
        if (i.dataset.pt === "bool") out[i.dataset.pk] = i.checked;
        else { const e = validateParam(i); if (e) errs.push(e); else out[i.dataset.pk] = Number(i.value); }
      });
      return errs.length ? { error: errs.length === 1 ? errs[0] : `${errs.length} values need fixing.` } : { value: out };
    }
    if (shape.kind === "textmap" && st.mode === "form") {
      const out = {};
      ed.querySelectorAll("[data-tk]").forEach((t) => { out[t.dataset.tk] = t.value; });
      return { value: out };
    }
    return parseJson(ed.querySelector("textarea").value, st.name);
  }

  const isDirty = (st, v) => JSON.stringify(v) !== JSON.stringify(st.active.content);
  function onEdit(sec, st) {
    const r = readEditor(sec, st);
    markDirty(sec, st, r.error ? true : isDirty(st, r.value));
    if (!r.error) showErr(sec, "");
  }
  function markDirty(sec, st, on) {
    if (on) A.dirty.add(st.name); else A.dirty.delete(st.name);
    sec.classList.toggle("is-dirty", on);
    sec.querySelector("[data-revert]").disabled = !on;
  }
  function showErr(sec, text) { const e = sec.querySelector(".comp__err"); e.textContent = text; e.hidden = !text; }

  async function save(sec, st) {
    const r = readEditor(sec, st);
    if (r.error) { showErr(sec, `Not saved: ${r.error}`); return; }
    if (!isDirty(st, r.value)) { showErr(sec, "Nothing changed - edit the content first."); return; }
    const note = sec.querySelector(`#cmp-${CSS.escape(st.name)}-note`).value.trim();
    const btn = sec.querySelector("[data-save]");
    btn.disabled = true; btn.textContent = "Saving…";
    try {
      const res = await api(`/api/components/${encodeURIComponent(st.name)}`, { method: "POST", body: { content: r.value, note } });
      A.dirty.delete(st.name);
      toast(`Saved ${humanComp(st.name)} as v${res && res.version != null ? res.version : "?"} (now active)`);
      st.view = null;
      await loadComponent(st.name);
      refreshVersions();
    } catch (err) {
      showErr(sec, `Not saved: ${err.message}`);
      btn.disabled = false; btn.textContent = "Save as new version";
    }
  }

  async function viewVersion(sec, st, v) {
    const box = sec.querySelector(".comp__view");
    try {
      const data = v === st.active.version ? st.active : await api(`/api/components/${encodeURIComponent(st.name)}/versions/${v}`);
      st.view = v;
      sec.querySelectorAll(".vrow").forEach((r) => r.classList.toggle("is-view", r.querySelector("[data-viewv]")?.dataset.viewv === String(v)));
      box.innerHTML = `<div class="comp__viewbar"><strong>v${escapeHtml(v)}</strong> <span class="muted">${escapeHtml(data.note || "")}</span>
          <span class="comp__viewacts">${data.is_active ? "" : `<button type="button" class="btn btn--sm" data-load>Load into editor</button>`}<button type="button" class="icon-btn" data-close aria-label="Close version view"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg></button></span></div>
        <pre class="comp__pre">${escapeHtml(textOf(data.content))}</pre>`;
      box.hidden = false;
      box.querySelector("[data-close]").addEventListener("click", () => { box.hidden = true; st.view = null; sec.querySelectorAll(".vrow.is-view").forEach((r) => r.classList.remove("is-view")); });
      box.querySelector("[data-load]")?.addEventListener("click", () => {
        renderEditor(sec, st, data.content);
        markDirty(sec, st, isDirty(st, data.content));
        sec.querySelector(`#cmp-${CSS.escape(st.name)}-note`).value = `Based on v${v}`;
        toast(`Loaded v${v} into the editor - save to make it a new version, or use Activate`);
      });
      box.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) { toast(`Could not load v${v}: ${err.message}`, { error: true }); }
  }

  async function showDiff(sec, st, v) {
    const box = sec.querySelector(".comp__view");
    const a = v, b = st.active.version;
    try {
      const res = await api(`/api/components/${encodeURIComponent(st.name)}/diff?a=${a}&b=${b}`);
      const text = typeof res === "string" ? res : Array.isArray(res && res.lines) ? res.lines.join("\n") : (res && (res.diff ?? res.unified)) || "";
      const lines = text.replace(/\n$/, "").split("\n");
      let add = 0, del = 0;
      const html = lines.map((l) => {
        let cls = "";
        if (l.startsWith("+++") || l.startsWith("---")) cls = "d-file";
        else if (l.startsWith("@@")) cls = "d-hunk";
        else if (l.startsWith("+")) { cls = "d-add"; add++; }
        else if (l.startsWith("-")) { cls = "d-del"; del++; }
        return `<span class="${cls}">${escapeHtml(l) || " "}</span>`;
      }).join("");
      box.innerHTML = `<div class="comp__viewbar"><strong>Diff v${escapeHtml(a)} → v${escapeHtml(b)} (active)</strong> <span class="num"><span class="d-addc">+${add}</span> <span class="d-delc">−${del}</span></span>
        <span class="comp__viewacts"><button type="button" class="icon-btn" data-close aria-label="Close diff"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg></button></span></div>
        ${text.trim() && !(res && res.identical) ? `<pre class="comp__pre comp__diff">${html}</pre>` : `<p class="muted">The two versions are identical.</p>`}`;
      box.hidden = false;
      box.querySelector("[data-close]").addEventListener("click", () => { box.hidden = true; });
      box.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) { toast(`Could not diff: ${err.message}`, { error: true }); }
  }

  async function activate(sec, st, v) {
    const dir = v < st.active.version ? "back" : "forward";
    const ok = await confirmDialog(`Activate v${v} of ${humanComp(st.name)}?`,
      `This switches ${dir} from v${st.active.version} to v${v}. The next question uses it immediately; v${st.active.version} stays in the history so you can switch back.${A.dirty.has(st.name) ? " Your unsaved edits to this component are discarded." : ""}`, `Activate v${v}`);
    if (!ok) return;
    try {
      await api(`/api/components/${encodeURIComponent(st.name)}/versions/${v}/activate`, { method: "POST" });
      A.dirty.delete(st.name);
      toast(`${humanComp(st.name)}: v${v} is now active`);
      st.view = null;
      await loadComponent(st.name);
      refreshVersions();
    } catch (err) { toast(`Could not activate: ${err.message}`, { error: true }); }
  }

  async function refreshVersions() {
    try { const g = await api(`/api/arch?days=${encodeURIComponent($("arch-days").value)}`); A.graph.versions = (g && g.components) || A.graph.versions; } catch { /* stats refresh is optional */ }
  }

  // wiring
  $("arch-refresh").addEventListener("click", load);
  $("arch-days").addEventListener("change", load);
  $("arch-mode")?.addEventListener("change", load);
  let lastW = 0;
  A.ro = new ResizeObserver(() => { const w = Math.floor($("arch-svg").clientWidth); if (A.graph && w && Math.abs(w - lastW) > 8) { lastW = w; draw(); } });
  A.ro.observe($("arch-svg"));
  window.addEventListener("beforeunload", (e) => { if (A.dirty.size) { e.preventDefault(); e.returnValue = ""; } });

  return { load, select };
}
