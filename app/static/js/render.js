// Markdown → sanitized HTML → KaTeX auto-render → highlight.js → citation chips.
import { escapeHtml } from "./api.js";

export const OOC = "#OutofContext";

let configured = false;
function configure() {
  if (configured || !window.marked) return;
  window.marked.setOptions({ gfm: true, breaks: false });
  configured = true;
}

// Math is lifted out before marked runs (so `_` and `\\` survive), then put back as
// escaped text with its delimiters for KaTeX auto-render to typeset.
const MATH_RE = /\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$(?!\s)[^$\n]+?(?<!\s)\$|\$ [^$\n]+? \$/g; // 2nd: LLM-style "$ x $" (space-padded)
const CODE_RE = /(```[\s\S]*?(?:```|$)|`[^`\n]+`)/g;

function protectMath(md) {
  const store = [];
  const out = md.split(CODE_RE).map((part, i) => {
    if (i % 2 === 1) return part; // code segment: leave untouched
    return part.replace(MATH_RE, (m) => {
      store.push(m);
      return `\uE000M${store.length - 1}\uE001`;
    });
  }).join("");
  return { md: out, store };
}

function restoreMath(html, store) {
  return html.replace(/\uE000M(\d+)\uE001/g, (_, i) => escapeHtml(store[+i]));
}

const PURIFY_CFG = { ADD_ATTR: ["target"], FORBID_TAGS: ["style", "form", "input"], FORBID_ATTR: ["style"] };

/** Render markdown into `el`. `streaming` = cheaper pass (no highlight); `citations: false` keeps [n] as text. */
export function renderMarkdown(el, md, { streaming = false, citations = true } = {}) {
  configure();
  const { md: safe, store } = protectMath(md || "");
  let html;
  if (window.marked) {
    html = window.marked.parse(safe);
  } else {
    html = `<p>${escapeHtml(safe).replace(/\n/g, "<br>")}</p>`;
  }
  html = window.DOMPurify ? window.DOMPurify.sanitize(html, PURIFY_CFG) : escapeHtml(html);
  el.innerHTML = restoreMath(html, store);

  // links open in a new tab
  el.querySelectorAll("a[href]").forEach((a) => { a.target = "_blank"; a.rel = "noopener noreferrer"; });
  // wide tables scroll inside their own box
  el.querySelectorAll("table").forEach((t) => {
    if (t.parentElement.classList.contains("table-wrap")) return;
    const wrap = document.createElement("div");
    wrap.className = "table-wrap";
    wrap.tabIndex = 0;
    wrap.setAttribute("role", "region");
    wrap.setAttribute("aria-label", "Table");
    t.replaceWith(wrap);
    wrap.appendChild(t);
  });

  if (window.renderMathInElement) {
    try {
      window.renderMathInElement(el, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false },
          { left: "$", right: "$", display: false },
        ],
        throwOnError: false,
        ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
      });
    } catch { /* leave raw TeX visible */ }
  }
  if (!streaming && window.hljs) {
    el.querySelectorAll("pre code").forEach((c) => { try { window.hljs.highlightElement(c); } catch { /* ignore */ } });
  }
  if (citations) linkCitations(el);
}

// [1]  [1, 3]  [1,2,3]  [2–4]
const CITE_RE = /\[(\d{1,2}(?:\s*(?:,|–|-)\s*\d{1,2})*)\]/g;

function expand(spec) {
  const out = [];
  for (const part of spec.split(",")) {
    const m = part.trim().match(/^(\d+)\s*[–-]\s*(\d+)$/);
    if (m) {
      const a = +m[1], b = +m[2];
      if (b >= a && b - a < 20) for (let i = a; i <= b; i++) out.push(i);
    } else if (part.trim()) out.push(+part.trim());
  }
  return out;
}

/** Replace [n] markers in text nodes with citation chip buttons. */
export function linkCitations(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const p = node.parentElement;
      if (!p || p.closest("pre, code, .katex, .cite")) return NodeFilter.FILTER_REJECT;
      CITE_RE.lastIndex = 0;
      return CITE_RE.test(node.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
    },
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) {
    const text = node.nodeValue;
    const frag = document.createDocumentFragment();
    let last = 0;
    CITE_RE.lastIndex = 0;
    let m;
    while ((m = CITE_RE.exec(text))) {
      frag.append(text.slice(last, m.index));
      const group = document.createElement("span");
      group.className = "cite";
      for (const n of expand(m[1])) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "cite__chip num";
        b.dataset.n = String(n);
        b.textContent = String(n);
        b.setAttribute("aria-label", `Source ${n}`);
        group.append(b);
      }
      frag.append(group);
      last = m.index + m[0].length;
    }
    frag.append(text.slice(last));
    node.replaceWith(frag);
  }
}

export function isOOC(text) {
  return (text || "").trim() === OOC;
}
