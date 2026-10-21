// NewRAG charts - tiny dependency-free inline-SVG charts with one shared tooltip.
// Used by the audit page (and available to the eval page). Styles: /static/css/charts.css.
//
// API (every function takes a container element and an options object, renders into it,
// re-renders on resize, and returns {update(newOpts)}):
//   bars(el, {categories:[{key,label}], series:[{key,label,color}], value:(catKey, seriesKey)=>num,
//             orientation:"v"|"h", stacked:true, hidden:Set, selected:catKey|null, onSelect:(catKey)=>void,
//             negative:Set(seriesKey drawn below zero), fmt, tipTitle:(cat)=>str, tipFoot:(cat)=>str,
//             height, ariaLabel, total:(cat)=>num (h: direct label), labelWidth})
//   line(el, {x:[{key,label}], series:[{key,label,color,values:[...]}], hidden, fmt, yMax, height, ariaLabel, onSelect})
//   scatter(el, {points:[{id,x,y,label,color}], xLabel, yLabel, xFmt, yFmt, logX, yMax, onSelect, height, ariaLabel})
//   donut(el, {segments:[{key,label,value,color}], center, sub, size, ariaLabel})
//   heatmap(el, {rows:[{key,label}], cols:[{key,label}], value:(r,c)=>num, fmt, onSelect:(r,c)=>void, ariaLabel})
//   legend(el, {series:[{key,label,color,mark:"rect"|"line"}], hidden:Set, onToggle:(key)=>void})
// Colours: PALETTE (categorical, fixed order, validated on #1D2328 - never cycle past 8).

export const PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
export const NEUTRAL = "#6B7782";

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const shortNum = (n) => {
  const a = Math.abs(n);
  return a >= 1e6 ? `${+(n / 1e6).toFixed(1)}M` : a >= 1e3 ? `${+(n / 1e3).toFixed(1)}k` : String(Math.round(n * 100) / 100);
};
export const fmtNum = (n) => (n == null || Number.isNaN(+n) ? "–" : Math.round(+n).toLocaleString("en-US"));

export function niceMax(v) {
  if (!(v > 0)) return 1;
  const e = 10 ** Math.floor(Math.log10(v));
  return [1, 2, 2.5, 5, 10].map((m) => m * e).find((m) => m >= v * 0.999);
}

/* ---------------- shared tooltip ---------------- */
let tipEl = null;
function tip() {
  if (!tipEl) {
    tipEl = document.createElement("div");
    tipEl.className = "cv-tip";
    tipEl.setAttribute("role", "tooltip");
    tipEl.hidden = true;
    document.body.appendChild(tipEl);
  }
  return tipEl;
}
/** at: MouseEvent (pointer) or Element (keyboard focus). rows: [{label, value, color}] */
export function showTip(at, { title = "", rows = [], foot = "" }) {
  const t = tip();
  t.replaceChildren();
  if (title) { const p = document.createElement("p"); p.className = "cv-tip__t"; p.textContent = title; t.append(p); }
  for (const r of rows) {
    const p = document.createElement("p");
    p.className = "cv-tip__r";
    if (r.color) { const k = document.createElement("span"); k.className = "cv-tip__k"; k.style.background = r.color; p.append(k); }
    const l = document.createElement("span"); l.className = "cv-tip__l"; l.textContent = r.label ?? "";
    const v = document.createElement("b"); v.textContent = r.value ?? "";
    p.append(l, v);
    t.append(p);
  }
  if (foot) { const p = document.createElement("p"); p.className = "cv-tip__f"; p.textContent = foot; t.append(p); }
  t.hidden = false;
  let cx, cy;
  if (at && "clientX" in at && at.clientX != null && at.type !== "focus") { cx = at.clientX; cy = at.clientY; }
  else { const b = (at.target || at).getBoundingClientRect(); cx = b.left + b.width / 2; cy = b.top; }
  const tw = t.offsetWidth, th = t.offsetHeight;
  let x = cx + 14, y = cy - th - 12;
  if (x + tw > innerWidth - 8) x = cx - tw - 14;
  if (x < 8) x = 8;
  if (y < 8) y = cy + 16;
  t.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
}
export function hideTip() { if (tipEl) tipEl.hidden = true; }

/* ---------------- mounting / resize ---------------- */
function mount(el, opts, draw) {
  const state = { opts };
  const render = () => {
    const w = Math.max(240, Math.floor(el.clientWidth || el.getBoundingClientRect().width || 600));
    state.w = w;
    draw(el, state.opts, w);
  };
  if (el._cvRO) el._cvRO.disconnect();
  let lastW = 0;
  el._cvRO = new ResizeObserver(() => {
    const w = Math.floor(el.clientWidth);
    if (w && Math.abs(w - lastW) > 4) { lastW = w; render(); }
  });
  el._cvRO.observe(el);
  render();
  lastW = Math.floor(el.clientWidth);
  return { update(o) { state.opts = { ...state.opts, ...o }; render(); } };
}

function wireHits(el, svg, { onTip, onSelect }) {
  svg.querySelectorAll("[data-hit]").forEach((h) => {
    const id = h.dataset.hit;
    const on = (e) => {
      svg.querySelectorAll(".is-hover").forEach((x) => x.classList.remove("is-hover"));
      svg.querySelectorAll(`[data-g="${CSS.escape(id)}"]`).forEach((x) => x.classList.add("is-hover"));
      showTip(e, onTip(id));
    };
    const off = () => { svg.querySelectorAll(".is-hover").forEach((x) => x.classList.remove("is-hover")); hideTip(); };
    h.addEventListener("pointermove", on);
    h.addEventListener("pointerleave", off);
    h.addEventListener("focus", on);
    h.addEventListener("blur", off);
    if (onSelect) {
      h.addEventListener("click", () => { hideTip(); onSelect(id); });
      h.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); hideTip(); onSelect(id); } });
    }
  });
}

function emptyMsg(el, text = "No data in this range.") {
  el.innerHTML = `<p class="cv-empty">${esc(text)}</p>`;
}

/* ---------------- bars (vertical or horizontal, stacked or single, optional negative series) ---------------- */
export function bars(el, opts) { return mount(el, opts, drawBars); }

function drawBars(el, o, W) {
  const cats = o.categories || [];
  const hidden = o.hidden || new Set();
  const series = (o.series || []).filter((s) => !hidden.has(s.key));
  const neg = o.negative || new Set();
  const fmt = o.fmt || fmtNum;
  if (!cats.length) return emptyMsg(el, o.emptyText);
  const v = (c, s) => Math.max(0, +o.value(c.key, s.key) || 0);
  const up = cats.map((c) => series.filter((s) => !neg.has(s.key)).reduce((a, s) => a + v(c, s), 0));
  const dn = cats.map((c) => series.filter((s) => neg.has(s.key)).reduce((a, s) => a + v(c, s), 0));
  const clickable = !!o.onSelect;
  const label = esc(o.ariaLabel || "Bar chart");
  const tipFor = (key) => {
    const c = cats.find((x) => String(x.key) === key);
    const rows = (o.series || []).filter((s) => !hidden.has(s.key)).slice().reverse()
      .map((s) => ({ label: s.label, value: fmt(o.value(c.key, s.key) || 0), color: s.color }));
    return { title: o.tipTitle ? o.tipTitle(c) : c.label, rows, foot: o.tipFoot ? o.tipFoot(c) : (clickable ? "Click to filter" : "") };
  };

  let svg;
  if (o.orientation === "h") {
    const rowH = o.rowHeight || 26, barH = Math.min(14, rowH - 10);
    const LW = Math.min(o.labelWidth || 150, W * 0.38), RW = 64, T = 4;
    const H = T + cats.length * rowH + 4;
    const iw = W - LW - RW;
    const max = niceMax(Math.max(1, ...up)) || 1;
    svg = `<svg class="cv cv--h" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${label}">`;
    cats.forEach((c, i) => {
      const y = T + i * rowH;
      const sel = o.selected != null && String(o.selected) === String(c.key);
      const dim = o.selected != null && !sel;
      svg += `<g class="cv-row${sel ? " is-sel" : ""}${dim ? " is-dim" : ""}" data-g="${esc(c.key)}">`;
      svg += `<text class="cv-lab" x="${LW - 8}" y="${y + rowH / 2 + 4}" text-anchor="end">${esc(trunc(c.label, Math.floor((LW - 10) / 6.6)))}</text>`;
      let x = LW;
      series.filter((s) => !neg.has(s.key)).forEach((s, si, arr) => {
        const w = (v(c, s) / max) * iw;
        if (w <= 0) return;
        const gap = si < arr.length - 1 ? 1 : 0;
        svg += `<rect class="cv-bar" x="${x.toFixed(1)}" y="${y + (rowH - barH) / 2}" width="${Math.max(1, w - gap).toFixed(1)}" height="${barH}" rx="2" fill="${s.color}"/>`;
        x += w;
      });
      const tot = o.total ? o.total(c) : up[i];
      svg += `<text class="cv-val" x="${(x + 6).toFixed(1)}" y="${y + rowH / 2 + 4}">${esc(o.totalFmt ? o.totalFmt(tot, c) : shortNum(tot))}</text>`;
      svg += `<rect class="cv-hit" data-hit="${esc(c.key)}" x="0" y="${y}" width="${W}" height="${rowH}" tabindex="0"${clickable ? ` role="button" aria-pressed="${sel}"` : ""} aria-label="${esc(c.label)}: ${esc(fmt(tot))}"/>`;
      svg += `</g>`;
    });
    svg += `<line class="cv-base" x1="${LW}" x2="${LW}" y1="${T}" y2="${H - 4}"/></svg>`;
  } else {
    const H = o.height || 220, L = 46, R = 8, T = 10, B = 26;
    const iw = W - L - R, ih = H - T - B;
    const maxUp = niceMax(Math.max(0, ...up)), maxDn = dn.some((x) => x > 0) ? niceMax(Math.max(...dn)) : 0;
    const span = maxUp + maxDn || 1;
    const zero = T + ih * (maxUp / span);
    const yv = (val) => (val / span) * ih;
    const bw = iw / cats.length;
    const barW = Math.max(2, Math.min(o.maxBar || 30, bw - (bw > 8 ? 3 : 1)));
    svg = `<svg class="cv" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${label}">`;
    const dnPx = ih * (maxDn / span);
    const ticks = [0, maxUp / 2, maxUp].concat(maxDn ? (dnPx < 40 ? [-maxDn] : [-maxDn / 2, -maxDn]) : []);
    for (const tv of ticks) {
      const ty = zero - yv(tv);
      svg += `<line class="cv-grid" x1="${L}" x2="${W - R}" y1="${ty}" y2="${ty}"/>`;
      svg += `<text class="cv-ax" x="${L - 6}" y="${ty + 4}" text-anchor="end">${esc(shortNum(Math.abs(tv)))}</text>`;
    }
    const maxLabels = Math.max(2, Math.floor(iw / (o.xLabelWidth || 58)));
    const every = Math.ceil(cats.length / maxLabels);
    cats.forEach((c, i) => {
      const x = L + i * bw + (bw - barW) / 2;
      const sel = o.selected != null && String(o.selected) === String(c.key);
      const dim = o.selected != null && !sel;
      svg += `<g class="cv-col${sel ? " is-sel" : ""}${dim ? " is-dim" : ""}" data-g="${esc(c.key)}">`;
      if (sel) svg += `<rect class="cv-selband" x="${L + i * bw}" y="${T}" width="${bw}" height="${ih}"/>`;
      let yUp = zero, yDn = zero;
      const ups = series.filter((s) => !neg.has(s.key)), dns = series.filter((s) => neg.has(s.key));
      ups.forEach((s, si) => {
        const h = yv(v(c, s));
        if (h <= 0) return;
        const g = si > 0 && yUp < zero ? 1.5 : 0;
        svg += `<rect class="cv-bar" x="${x.toFixed(1)}" y="${(yUp - h).toFixed(1)}" width="${barW.toFixed(1)}" height="${Math.max(0.5, h - g).toFixed(1)}" rx="${Math.min(2, barW / 3)}" fill="${s.color}"/>`;
        yUp -= h;
      });
      dns.forEach((s, si) => {
        const h = yv(v(c, s));
        if (h <= 0) return;
        const g = si > 0 && yDn > zero ? 1.5 : 0;
        svg += `<rect class="cv-bar" x="${x.toFixed(1)}" y="${(yDn + g).toFixed(1)}" width="${barW.toFixed(1)}" height="${Math.max(0.5, h - g).toFixed(1)}" rx="${Math.min(2, barW / 3)}" fill="${s.color}"/>`;
        yDn += h;
      });
      const tot = up[i] + dn[i];
      svg += `<rect class="cv-hit" data-hit="${esc(c.key)}" x="${(L + i * bw).toFixed(1)}" y="${T}" width="${bw.toFixed(1)}" height="${ih}" tabindex="${cats.length <= 40 ? 0 : -1}"${clickable ? ` role="button" aria-pressed="${sel}"` : ""} aria-label="${esc(c.label)}: ${esc(fmt(tot))}"/>`;
      if (i % every === 0 || (i === cats.length - 1 && cats.length - 1 - Math.floor(i / every) * every >= every / 2)) {
        svg += `<text class="cv-ax${sel ? " is-sel" : ""}" x="${(L + i * bw + bw / 2).toFixed(1)}" y="${H - 8}" text-anchor="middle">${esc(c.short || c.label)}</text>`;
      }
      svg += `</g>`;
    });
    svg += `<line class="cv-base" x1="${L}" x2="${W - R}" y1="${zero}" y2="${zero}"/></svg>`;
  }
  el.innerHTML = svg;
  wireHits(el, el.querySelector("svg"), { onTip: tipFor, onSelect: o.onSelect });
}

function trunc(s, n) { s = String(s ?? ""); return s.length > n ? `${s.slice(0, Math.max(1, n - 1))}…` : s; }

/* ---------------- line (crosshair snaps to nearest x) ---------------- */
export function line(el, opts) { return mount(el, opts, drawLine); }

function drawLine(el, o, W) {
  const xs = o.x || [];
  const hidden = o.hidden || new Set();
  const series = (o.series || []).filter((s) => !hidden.has(s.key));
  const fmt = o.fmt || fmtNum;
  if (!xs.length) return emptyMsg(el, o.emptyText);
  const H = o.height || 200, L = 46, R = 12, T = 10, B = 26;
  const iw = W - L - R, ih = H - T - B;
  const vals = series.flatMap((s) => s.values.filter((x) => x != null));
  const max = o.yMax ?? niceMax(Math.max(0, ...vals));
  const xp = (i) => L + (xs.length === 1 ? iw / 2 : (i / (xs.length - 1)) * iw);
  const yp = (v) => T + ih - (v / max) * ih;
  let svg = `<svg class="cv" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(o.ariaLabel || "Line chart")}">`;
  for (const tv of [0, max / 2, max]) {
    svg += `<line class="cv-grid" x1="${L}" x2="${W - R}" y1="${yp(tv)}" y2="${yp(tv)}"/><text class="cv-ax" x="${L - 6}" y="${yp(tv) + 4}" text-anchor="end">${esc(o.axFmt ? o.axFmt(tv) : shortNum(tv))}</text>`;
  }
  const every = Math.ceil(xs.length / Math.max(2, Math.floor(iw / 62)));
  xs.forEach((x, i) => { if (i % every === 0) svg += `<text class="cv-ax" x="${xp(i)}" y="${H - 8}" text-anchor="middle">${esc(x.short || x.label)}</text>`; });
  for (const s of series) {
    let d = "", pen = false;
    s.values.forEach((v, i) => {
      if (v == null) { pen = false; return; }
      d += `${pen ? "L" : "M"}${xp(i).toFixed(1)},${yp(v).toFixed(1)}`;
      pen = true;
    });
    svg += `<path class="cv-line" d="${d}" stroke="${s.color}"/>`;
    s.values.forEach((v, i) => { if (v != null && (xs.length <= 40)) svg += `<circle class="cv-dot" cx="${xp(i).toFixed(1)}" cy="${yp(v).toFixed(1)}" r="3" fill="${s.color}"/>`; });
  }
  svg += `<line class="cv-cross" x1="0" x2="0" y1="${T}" y2="${T + ih}" visibility="hidden"/>`;
  svg += `<line class="cv-base" x1="${L}" x2="${W - R}" y1="${T + ih}" y2="${T + ih}"/>`;
  svg += `<rect class="cv-hit cv-hit--area" x="${L - 6}" y="${T}" width="${iw + 12}" height="${ih}" tabindex="0" aria-label="${esc(o.ariaLabel || "Line chart")} - use arrow keys"/></svg>`;
  el.innerHTML = svg;
  const root = el.querySelector("svg"), area = root.querySelector(".cv-hit--area"), cross = root.querySelector(".cv-cross");
  let cur = -1;
  const at = (i, e) => {
    cur = i;
    cross.setAttribute("x1", xp(i)); cross.setAttribute("x2", xp(i)); cross.setAttribute("visibility", "visible");
    showTip(e, { title: xs[i].label, rows: series.map((s) => ({ label: s.label, value: s.values[i] == null ? "–" : fmt(s.values[i]), color: s.color })), foot: o.onSelect ? "Click to filter" : "" });
  };
  const idx = (e) => {
    const b = root.getBoundingClientRect();
    const px = ((e.clientX - b.left) / b.width) * W;
    return Math.max(0, Math.min(xs.length - 1, Math.round(xs.length === 1 ? 0 : ((px - L) / iw) * (xs.length - 1))));
  };
  area.addEventListener("pointermove", (e) => at(idx(e), e));
  area.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  area.addEventListener("blur", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  area.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); at(Math.max(0, Math.min(xs.length - 1, (cur < 0 ? 0 : cur + (e.key === "ArrowRight" ? 1 : -1)))), area); }
    if ((e.key === "Enter" || e.key === " ") && o.onSelect && cur >= 0) { e.preventDefault(); o.onSelect(xs[cur].key); }
  });
  if (o.onSelect) area.addEventListener("click", (e) => { hideTip(); o.onSelect(xs[idx(e)].key); });
}

/* ---------------- scatter (24px hit targets) ---------------- */
export function scatter(el, opts) { return mount(el, opts, drawScatter); }

function drawScatter(el, o, W) {
  const pts = (o.points || []).filter((p) => p.x != null && p.y != null && !(o.hidden && o.hidden.has(p.series)));
  if (!pts.length) return emptyMsg(el, o.emptyText);
  const H = o.height || 260, L = 50, R = 14, T = 12, B = 40;
  const iw = W - L - R, ih = H - T - B;
  const xFmt = o.xFmt || shortNum, yFmt = o.yFmt || shortNum;
  const xsr = pts.map((p) => p.x);
  let x0 = o.xMin ?? 0, x1 = o.xMax ?? niceMax(Math.max(...xsr));
  const log = o.logX && Math.min(...xsr) > 0;
  if (log) { x0 = 10 ** Math.floor(Math.log10(Math.min(...xsr))); x1 = 10 ** Math.ceil(Math.log10(Math.max(...xsr))); }
  const y1 = o.yMax ?? niceMax(Math.max(...pts.map((p) => p.y))), y0 = o.yMin ?? 0;
  const xp = (x) => L + (log ? (Math.log10(x) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0) || 1) : (x - x0) / (x1 - x0 || 1)) * iw;
  const yp = (y) => T + ih - ((y - y0) / (y1 - y0 || 1)) * ih;
  let svg = `<svg class="cv" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(o.ariaLabel || "Scatter plot")}">`;
  for (const t of [y0, (y0 + y1) / 2, y1]) svg += `<line class="cv-grid" x1="${L}" x2="${W - R}" y1="${yp(t)}" y2="${yp(t)}"/><text class="cv-ax" x="${L - 6}" y="${yp(t) + 4}" text-anchor="end">${esc(yFmt(t))}</text>`;
  const xt = log ? Array.from({ length: Math.round(Math.log10(x1 / x0)) + 1 }, (_, i) => x0 * 10 ** i) : [x0, (x0 + x1) / 2, x1];
  for (const t of xt) svg += `<text class="cv-ax" x="${xp(t)}" y="${T + ih + 16}" text-anchor="middle">${esc(xFmt(t))}</text>`;
  svg += `<line class="cv-base" x1="${L}" x2="${W - R}" y1="${T + ih}" y2="${T + ih}"/>`;
  if (o.xLabel) svg += `<text class="cv-ax cv-ax--t" x="${L + iw / 2}" y="${H - 4}" text-anchor="middle">${esc(o.xLabel)}</text>`;
  if (o.yLabel) svg += `<text class="cv-ax cv-ax--t" x="12" y="${T + ih / 2}" text-anchor="middle" transform="rotate(-90 12 ${T + ih / 2})">${esc(o.yLabel)}</text>`;
  pts.forEach((p, i) => {
    const cx = xp(p.x).toFixed(1), cy = yp(p.y).toFixed(1);
    svg += `<g data-g="p${i}"><circle class="cv-pt" cx="${cx}" cy="${cy}" r="4.5" fill="${p.color || PALETTE[0]}"/>`;
    svg += `<circle class="cv-hit" data-hit="p${i}" cx="${cx}" cy="${cy}" r="12" tabindex="${pts.length <= 60 ? 0 : -1}"${o.onSelect ? ' role="button"' : ""} aria-label="${esc(p.label)}"/></g>`;
  });
  svg += `</svg>`;
  el.innerHTML = svg;
  wireHits(el, el.querySelector("svg"), {
    onTip: (id) => { const p = pts[+id.slice(1)]; return { title: p.label, rows: p.rows || [{ label: o.xLabel || "x", value: xFmt(p.x) }, { label: o.yLabel || "y", value: yFmt(p.y) }], foot: o.onSelect ? "Click for details" : "" }; },
    onSelect: o.onSelect ? (id) => o.onSelect(pts[+id.slice(1)].id) : null,
  });
}

/* ---------------- donut / gauge ---------------- */
export function donut(el, opts) { return mount(el, opts, drawDonut); }

function drawDonut(el, o) {
  const segs = (o.segments || []).filter((s) => s.value > 0);
  const total = segs.reduce((a, s) => a + s.value, 0);
  const S = o.size || 120, r = S / 2 - 8, sw = o.thickness || 12, C = 2 * Math.PI * r;
  let svg = `<svg class="cv cv--donut" viewBox="0 0 ${S} ${S}" width="${S}" height="${S}" role="img" aria-label="${esc(o.ariaLabel || "Donut")}">`;
  svg += `<circle cx="${S / 2}" cy="${S / 2}" r="${r}" fill="none" class="cv-track" stroke-width="${sw}"/>`;
  let off = 0;
  const gap = segs.length > 1 ? 2 : 0;
  segs.forEach((s, i) => {
    const len = (s.value / total) * C;
    svg += `<g data-g="s${i}"><circle class="cv-seg" cx="${S / 2}" cy="${S / 2}" r="${r}" fill="none" stroke="${s.color}" stroke-width="${sw}"
      stroke-dasharray="${Math.max(0.5, len - gap).toFixed(2)} ${(C - Math.max(0.5, len - gap)).toFixed(2)}" stroke-dashoffset="${(-off).toFixed(2)}" transform="rotate(-90 ${S / 2} ${S / 2})"/>
      <circle class="cv-hit" data-hit="s${i}" cx="${S / 2}" cy="${S / 2}" r="${r}" fill="none" stroke="transparent" stroke-width="${sw + 10}" stroke-dasharray="${len.toFixed(2)} ${(C - len).toFixed(2)}" stroke-dashoffset="${(-off).toFixed(2)}" transform="rotate(-90 ${S / 2} ${S / 2})" tabindex="0" aria-label="${esc(s.label)}: ${esc(fmtNum(s.value))}"/></g>`;
    off += len;
  });
  if (o.center != null) svg += `<text class="cv-center" x="${S / 2}" y="${S / 2 + (o.sub ? 2 : 6)}" text-anchor="middle">${esc(o.center)}</text>`;
  if (o.sub) svg += `<text class="cv-center-sub" x="${S / 2}" y="${S / 2 + 18}" text-anchor="middle">${esc(o.sub)}</text>`;
  svg += `</svg>`;
  el.innerHTML = svg;
  wireHits(el, el.querySelector("svg"), {
    onTip: (id) => { const s = segs[+id.slice(1)]; return { title: s.label, rows: [{ label: "Count", value: fmtNum(s.value), color: s.color }, { label: "Share", value: `${Math.round((s.value / total) * 100)}%` }] }; },
  });
}

/* ---------------- heatmap (sequential single hue) ---------------- */
export function heatmap(el, opts) { return mount(el, opts, drawHeat); }

const SEQ = ["#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4"];
function drawHeat(el, o, W) {
  const rows = o.rows || [], cols = o.cols || [];
  if (!rows.length || !cols.length) return emptyMsg(el, o.emptyText);
  const fmt = o.fmt || fmtNum;
  const LW = Math.min(140, W * 0.3), T = 22, cw = (W - LW - 4) / cols.length, ch = 30;
  const H = T + rows.length * ch + 4;
  const all = rows.flatMap((r) => cols.map((c) => +o.value(r.key, c.key) || 0));
  const max = o.max ?? Math.max(1, ...all);
  let svg = `<svg class="cv" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(o.ariaLabel || "Heatmap")}">`;
  cols.forEach((c, j) => { svg += `<text class="cv-ax" x="${LW + j * cw + cw / 2}" y="14" text-anchor="middle">${esc(trunc(c.label, Math.floor(cw / 6.5)))}</text>`; });
  rows.forEach((r, i) => {
    svg += `<text class="cv-lab" x="${LW - 8}" y="${T + i * ch + ch / 2 + 4}" text-anchor="end">${esc(trunc(r.label, 20))}</text>`;
    cols.forEach((c, j) => {
      const v = +o.value(r.key, c.key) || 0;
      const k = v <= 0 ? -1 : Math.min(SEQ.length - 1, Math.floor((v / max) * (SEQ.length - 1)));
      const id = `${i}:${j}`;
      svg += `<g data-g="${id}"><rect class="cv-cell" x="${LW + j * cw + 1}" y="${T + i * ch + 1}" width="${cw - 2}" height="${ch - 2}" rx="3" fill="${k < 0 ? "var(--cv-empty)" : SEQ[SEQ.length - 1 - k]}"/>
        <text class="cv-cellv${k > 4 ? " is-dark" : ""}" x="${LW + j * cw + cw / 2}" y="${T + i * ch + ch / 2 + 4}" text-anchor="middle">${v ? esc(fmt(v)) : ""}</text>
        <rect class="cv-hit" data-hit="${id}" x="${LW + j * cw}" y="${T + i * ch}" width="${cw}" height="${ch}" tabindex="0"${o.onSelect ? ' role="button"' : ""} aria-label="${esc(r.label)} / ${esc(c.label)}: ${esc(fmt(v))}"/></g>`;
    });
  });
  svg += `</svg>`;
  el.innerHTML = svg;
  wireHits(el, el.querySelector("svg"), {
    onTip: (id) => { const [i, j] = id.split(":").map(Number); return { title: `${rows[i].label} → ${cols[j].label}`, rows: [{ label: "Value", value: fmt(+o.value(rows[i].key, cols[j].key) || 0) }], foot: o.onSelect ? "Click to list items" : "" }; },
    onSelect: o.onSelect ? (id) => { const [i, j] = id.split(":").map(Number); o.onSelect(rows[i].key, cols[j].key); } : null,
  });
}

/* ---------------- legend with toggles ---------------- */
export function legend(el, { series, hidden = new Set(), onToggle }) {
  el.classList.add("cv-legend");
  el.replaceChildren();
  for (const s of series) {
    const b = document.createElement(onToggle ? "button" : "span");
    b.className = "cv-legend__item";
    if (onToggle) {
      b.type = "button";
      b.setAttribute("aria-pressed", String(!hidden.has(s.key)));
      b.title = hidden.has(s.key) ? `Show ${s.label}` : `Hide ${s.label}`;
      b.addEventListener("click", () => onToggle(s.key));
    }
    const k = document.createElement("span");
    k.className = `cv-legend__k${s.mark === "line" ? " cv-legend__k--line" : ""}`;
    k.style.background = s.color;
    const t = document.createElement("span");
    t.textContent = s.label;
    b.append(k, t);
    el.append(b);
  }
}
