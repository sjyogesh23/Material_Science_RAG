// Source pages with highlight overlays, the lightbox viewer, and the citation hover card.
import { escapeHtml } from "./api.js";

const metaCache = new Map();

async function pageSize(source, page) {
  const ps = source.page_size || {};
  const known = ps[String(page)] || ps[page];
  if (Array.isArray(known) && known.length === 2) return known;
  const key = `${source.doc_id}/${page}`;
  if (!metaCache.has(key)) {
    metaCache.set(key, fetch(`/api/page/${encodeURIComponent(source.doc_id)}/${page}/meta`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => (j && j.width ? [j.width, j.height] : null))
      .catch(() => null));
  }
  return metaCache.get(key);
}

/** Pages to show for a source: highlight pages first, else its page list. */
export function sourcePages(source) {
  const hl = (source.highlights || []).map((h) => h.page);
  const pages = hl.length ? hl : (source.pages || []);
  return [...new Set(pages)].filter((p) => Number.isFinite(+p)).map(Number);
}

function bboxesFor(source, page) {
  return (source.highlights || []).filter((h) => +h.page === +page).flatMap((h) => h.bboxes || []);
}

export function shortSection(source) {
  const sp = source.section_path || [];
  if (sp.length) return sp[sp.length - 1];
  const h = source.header || "";
  const parts = h.split(/\s*[|>]\s*/);
  return parts[parts.length - 1] || h;
}

export function pageLabel(pages) {
  const p = [...new Set(pages || [])].sort((a, b) => a - b);
  if (!p.length) return "";
  if (p.length === 1) return `p. ${p[0]}`;
  const contiguous = p.every((v, i) => i === 0 || v === p[i - 1] + 1);
  return contiguous ? `pp. ${p[0]}–${p[p.length - 1]}` : `pp. ${p.join(", ")}`;
}

/** A page image in a relative box with absolutely positioned highlight rectangles. */
export function pageBox(source, page, { dpi = 110, lazy = true } = {}) {
  const box = document.createElement("div");
  box.className = "page-box";
  const img = document.createElement("img");
  img.alt = `Page ${page} of ${source.doc_title || "the textbook"}`;
  img.decoding = "async";
  if (lazy) img.loading = "lazy";
  img.src = `/api/page/${encodeURIComponent(source.doc_id)}/${page}.png?dpi=${dpi}`;
  img.addEventListener("error", () => box.classList.add("is-broken"), { once: true });
  img.addEventListener("load", () => {
    if (img.naturalWidth) box.style.aspectRatio = `${img.naturalWidth} / ${img.naturalHeight}`;
  }, { once: true });
  box.append(img);
  const boxes = bboxesFor(source, page);
  if (boxes.length) {
    pageSize(source, page).then((size) => {
      const apply = (w, h) => {
        if (!(img.complete && img.naturalWidth)) box.style.aspectRatio = `${w} / ${h}`;
        for (const [x0, y0, x1, y1] of boxes) {
          const r = document.createElement("span");
          r.className = "hl";
          r.style.left = `${(x0 / w) * 100}%`;
          r.style.top = `${(y0 / h) * 100}%`;
          r.style.width = `${((x1 - x0) / w) * 100}%`;
          r.style.height = `${((y1 - y0) / h) * 100}%`;
          box.append(r);
        }
      };
      if (size) apply(size[0], size[1]);
      else if (img.complete && img.naturalWidth) apply(img.naturalWidth * 72 / dpi, img.naturalHeight * 72 / dpi);
      else img.addEventListener("load", () => apply(img.naturalWidth * 72 / dpi, img.naturalHeight * 72 / dpi), { once: true });
    });
  }
  return box;
}

/** The "Show pages" panel content. */
export function buildPagesPanel(sources, onOpen) {
  const wrap = document.createElement("div");
  wrap.className = "pages";
  for (const s of sources) {
    const sec = document.createElement("section");
    sec.className = "pages__source";
    sec.id = "";
    sec.dataset.n = s.n;
    const pages = sourcePages(s);
    sec.innerHTML = `
      <header class="pages__head">
        <span class="pages__n num" aria-hidden="true">${escapeHtml(s.n)}</span>
        <div class="pages__meta">
          <p class="pages__section">${escapeHtml(shortSection(s))}</p>
          <p class="pages__sub">${escapeHtml(s.doc_title || "")}${s.doc_title ? ", " : ""}<span class="num">${escapeHtml(pageLabel(s.pages && s.pages.length ? s.pages : pages))}</span></p>
        </div>
      </header>`;
    const strip = document.createElement("div");
    strip.className = "pages__strip";
    pages.forEach((p, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "pages__thumb";
      btn.setAttribute("aria-label", `Open page ${p} of source ${s.n} in the viewer`);
      btn.append(pageBox(s, p));
      const cap = document.createElement("span");
      cap.className = "pages__cap num";
      cap.textContent = `Page ${p}`;
      btn.append(cap);
      btn.addEventListener("click", () => onOpen(s, i));
      strip.append(btn);
    });
    if (!pages.length) {
      const p = document.createElement("p");
      p.className = "pages__none";
      p.textContent = "No page location recorded for this source.";
      strip.append(p);
    }
    sec.append(strip);
    wrap.append(sec);
  }
  return wrap;
}

/* ---------------- lightbox ---------------- */
const viewer = {
  dlg: null, body: null, title: null, caption: null, count: null, prev: null, next: null,
  items: [], index: 0, opener: null,
};

function initViewer() {
  if (viewer.dlg) return;
  viewer.dlg = document.getElementById("viewer");
  viewer.body = document.getElementById("viewer-body");
  viewer.title = document.getElementById("viewer-title");
  viewer.caption = document.getElementById("viewer-caption");
  viewer.count = document.getElementById("viewer-count");
  viewer.prev = document.getElementById("viewer-prev");
  viewer.next = document.getElementById("viewer-next");
  document.getElementById("viewer-close").addEventListener("click", () => viewer.dlg.close());
  viewer.prev.addEventListener("click", () => show(viewer.index - 1));
  viewer.next.addEventListener("click", () => show(viewer.index + 1));
  viewer.dlg.addEventListener("click", (e) => { if (e.target === viewer.dlg) viewer.dlg.close(); });
  viewer.dlg.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { e.preventDefault(); show(viewer.index - 1); }
    if (e.key === "ArrowRight") { e.preventDefault(); show(viewer.index + 1); }
  });
  viewer.dlg.addEventListener("close", () => {
    if (viewer.dlg.open) return; // reopened before the close event was delivered
    viewer.body.replaceChildren();
    if (viewer.opener && document.contains(viewer.opener)) viewer.opener.focus();
  });
}

function show(i) {
  if (!viewer.items.length) return;
  viewer.index = (i + viewer.items.length) % viewer.items.length;
  const it = viewer.items[viewer.index];
  viewer.title.textContent = it.title;
  viewer.caption.textContent = it.caption || "";
  viewer.caption.hidden = !it.caption;
  viewer.body.replaceChildren(it.render());
  viewer.body.scrollTop = 0;
  const multi = viewer.items.length > 1;
  viewer.prev.hidden = viewer.next.hidden = viewer.count.hidden = !multi;
  viewer.count.textContent = `${viewer.index + 1} / ${viewer.items.length}`;
}

function open(items, index = 0) {
  initViewer();
  viewer.opener = document.activeElement;
  viewer.items = items;
  if (!viewer.dlg.open) viewer.dlg.showModal();
  show(index);
}

export function openSource(source, pageIndex = 0) {
  const pages = sourcePages(source);
  if (!pages.length) return;
  const items = pages.map((p) => ({
    title: `[${source.n}] ${shortSection(source)}, page ${p}`,
    caption: source.snippet ? `“${source.snippet.trim()}”` : "",
    render: () => {
      const b = pageBox(source, p, { dpi: 160, lazy: false });
      b.classList.add("page-box--large");
      return b;
    },
  }));
  open(items, pageIndex);
}

export function openImages(images, index = 0) {
  const items = images.map((im) => ({
    title: [im.figure_number, im.page ? `page ${im.page}` : ""].filter(Boolean).join(", ") || "Figure",
    caption: im.caption || im.description || "",
    render: () => {
      const img = document.createElement("img");
      img.className = "viewer__figure";
      img.src = im.url;
      img.alt = im.caption || im.figure_number || "Figure from the textbook";
      return img;
    },
  }));
  open(items, index);
}

/* ---------------- citation hover card ---------------- */
let card, hideTimer, cardFor = null, shownAt = 0;

export function initCiteCard(root, getSources, onOpen) {
  card = document.getElementById("cite-card");
  const chipOf = (e) => e.target.closest && e.target.closest(".cite__chip");
  const showFor = (chip) => {
    clearTimeout(hideTimer);
    const sources = getSources(chip) || [];
    const s = sources.find((x) => +x.n === +chip.dataset.n);
    if (!s) {
      card.innerHTML = `<p class="cite-card__sub">Source ${escapeHtml(chip.dataset.n)} arrives when the answer finishes.</p>`;
    } else {
      card.innerHTML = `
        <p class="cite-card__title">${escapeHtml(s.doc_title || "Source")}</p>
        <p class="cite-card__section">${escapeHtml(shortSection(s))}</p>
        <p class="cite-card__sub num">${escapeHtml(pageLabel(s.pages || sourcePages(s)))}${s.chunk_type && s.chunk_type !== "text" ? `, ${escapeHtml(s.chunk_type)}` : ""}</p>
        ${s.snippet ? `<p class="cite-card__snippet">${escapeHtml(s.snippet)}</p>` : ""}
        <p class="cite-card__hint">Click to open the page</p>`;
    }
    card.hidden = false;
    chip.setAttribute("aria-describedby", "cite-card");
    cardFor = chip;
    const r = chip.getBoundingClientRect();
    const cw = card.offsetWidth, ch = card.offsetHeight;
    let left = Math.min(Math.max(8, r.left + r.width / 2 - cw / 2), window.innerWidth - cw - 8);
    let top = r.top - ch - 8;
    if (top < 8) top = r.bottom + 8;
    card.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
    card.classList.add("is-on");
    shownAt = performance.now();
  };
  const hide = () => {
    hideTimer = setTimeout(() => {
      card.classList.remove("is-on");
      card.hidden = true;
      if (cardFor) cardFor.removeAttribute("aria-describedby");
      cardFor = null;
    }, 80);
  };
  root.addEventListener("mouseover", (e) => { const c = chipOf(e); if (c) showFor(c); });
  root.addEventListener("mouseout", (e) => { if (chipOf(e)) hide(); });
  root.addEventListener("focusin", (e) => { const c = chipOf(e); if (c) showFor(c); });
  root.addEventListener("focusout", (e) => { if (chipOf(e)) hide(); });
  root.addEventListener("click", (e) => {
    const c = chipOf(e);
    if (!c) return;
    const s = (getSources(c) || []).find((x) => +x.n === +c.dataset.n);
    hide();
    if (s) onOpen(s, c);
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && cardFor) hide(); });
  window.addEventListener("scroll", () => { if (cardFor && performance.now() - shownAt > 250) hide(); }, true);
}
