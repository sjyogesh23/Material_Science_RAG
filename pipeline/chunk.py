#!/usr/bin/env python3
"""Turn extracted JSON (extract.py output) into embedding-ready chunks.

Strategy (structure-aware, not fixed-size):
  * text      - paragraphs / lists / display formulas / footnotes grouped *within one section*
                up to CHUNK_TARGET_TOKENS (hard cap CHUNK_MAX_TOKENS), with a sentence-level overlap.
                A new heading always starts a new chunk; oversized paragraphs are split by sentence.
  * figure    - one chunk per figure / chart: caption + VLM description + chart axes/findings + text in image
  * table     - one chunk per table (caption + markdown); long tables are split by rows, header repeated
  * reference - bibliography entries grouped, tagged so retrieval can filter them out
  * skipped   - running headers/footers, watermarks, links, the book's contents / list-of-figures pages

Every chunk has two texts:
  * ``text``    - what gets embedded: "<document title> | <section path>" header + content
  * ``content`` - the raw content for showing to the LLM / user
and provenance: pages, section path, element ids, bbox + line numbers per element, image/table files.

Usage:
    python pipeline/chunk.py                       # all JSON under extracted/
    python pipeline/chunk.py --files "extracted/textbooks/**/*.json"
Output:  chunks/chunks.jsonl  (+ chunks/stats.json)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent  # project root (this file lives in pipeline/)
load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
TARGET = int(os.getenv("CHUNK_TARGET_TOKENS", "400"))
MAX = int(os.getenv("CHUNK_MAX_TOKENS", "600"))
OVERLAP = int(os.getenv("CHUNK_OVERLAP_TOKENS", "60"))
MIN = int(os.getenv("CHUNK_MIN_TOKENS", "40"))

SKIP_TYPES = {"header", "footer", "watermark", "watermark_text", "link", "annotation", "logo",
              "table_of_contents", "form", "key_value"}
SKIP_SECTIONS = re.compile(r"^(table of )?contents$|^list of (figures|tables|symbols)|^index$", re.I)
TEXTY = {"text", "list", "formula", "footnote", "quote", "code_block", "sidebar", "form_field"}
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[$])")


class Tok:
    def __init__(self, model: str):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model)

    def n(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def tail(self, text: str, tokens: int) -> str:
        """Last ~`tokens` tokens of text, snapped to a sentence start (used as overlap)."""
        sents = SENT_SPLIT.split(text)
        out = []
        for s in reversed(sents):
            if self.n(" ".join([s] + out)) > tokens:
                break
            out.insert(0, s)
        return " ".join(out)


# ----------------------------------------------------------------------------- element -> text
def formula_text(e: dict, with_defs: bool = True) -> str:
    reliable = e.get("latex") and (e["metadata"].get("confidence") or 1) >= 0.6
    body = f"$${e['latex']}$$" if reliable else (e.get("plain_text") or e.get("latex") or "")
    if e.get("equation_number"):
        body += f" {e['equation_number']}"
    defs = []
    for v in e.get("variables") or []:
        if not v.get("meaning"):
            continue
        sym = v["symbol"].replace("{", "").replace("}", "")
        # drop inherited meanings (r_o / r_m copying plain r) produced by older extractions
        if "_" in sym and any(w["symbol"] == sym.split("_")[0] and w.get("meaning") == v["meaning"]
                              for w in e.get("variables") or []):
            continue
        defs.append(f"{v['symbol']}: {v['meaning']}")
    if defs and with_defs:
        body += " where " + "; ".join(defs)
    return body


def element_text(e: dict, group_ids: set | None = None) -> str:
    t = e["type"]
    if t == "formula":
        if e.get("formula_type") == "inline":
            return ""
        # the "where ..." paragraph is already in this chunk -> don't repeat the definitions
        defined_in = e["metadata"].get("variables_defined_in")
        return formula_text(e, with_defs=not (group_ids and defined_in in group_ids))
    if t == "list":
        return "\n".join(("  " * (i.get("level", 1) - 1)) + f"- {i['text']}" for i in e.get("items", []))
    if t == "quote":
        return f"\"{e.get('content', '')}\"" + (f" - {e['attribution']}" if e.get("attribution") else "")
    if t == "code_block":
        return f"```{e.get('language') or ''}\n{e.get('content', '')}\n```"
    if t == "footnote":
        return f"[Footnote {e.get('marker') or ''}] {e.get('content', '')}".strip()
    if t == "form_field":
        return f"{e.get('label') or e.get('name') or 'field'}: {e.get('value')}"
    return e.get("content") or ""


def figure_text(e: dict) -> str:
    cap = (e.get("caption") or {}).get("text")
    parts = []
    if cap:
        parts.append(cap)
    if e.get("description"):
        parts.append(f"Figure description: {e['description']}")
    if e["type"] == "chart":
        for k, label in (("x_axis", "X axis"), ("y_axis", "Y axis")):
            ax = e.get(k)
            if isinstance(ax, dict) and ax.get("label"):
                parts.append(f"{label}: {ax['label']}" + (f" ({ax['unit']})" if ax.get("unit") else ""))
        if e.get("series"):
            names = [s.get("name") for s in e["series"] if isinstance(s, dict) and s.get("name")]
            if names:
                parts.append("Series: " + ", ".join(names))
        if e.get("key_findings"):
            parts.append("Key findings: " + "; ".join(map(str, e["key_findings"])))
    texts = [t["text"] for t in e.get("text_in_image") or [] if len(t["text"]) > 1]
    if texts:
        joined = " | ".join(texts)
        parts.append("Text in figure: " + (joined[:600] + "…" if len(joined) > 600 else joined))
    cues = e.get("micrograph_cues") or {}
    if cues.get("scale_bars"):
        parts.append("Scale bar: " + ", ".join(cues["scale_bars"]))
    return "\n".join(parts)


def loc(e: dict) -> dict:
    m = e["metadata"]
    return {k: v for k, v in {"element_id": e["id"], "type": e["type"], "page": m.get("page"), "bbox": m.get("bbox"),
                              "line_start": m.get("line_start"), "line_end": m.get("line_end")}.items() if v is not None}


def merge_entities(els: list[dict]) -> dict:
    out: dict[str, list] = {}
    for e in els:
        for k, vals in (e.get("entities") or {}).items():
            if k == "quantities":
                out.setdefault(k, []).extend(q["text"] for q in vals)
            else:
                out.setdefault(k, []).extend(vals)
    return {k: list(dict.fromkeys(v))[:40] for k, v in out.items()}


# ----------------------------------------------------------------------------- chunker
class Chunker:
    def __init__(self, tok: Tok):
        self.tok = tok

    def run(self, doc: dict, json_path: Path) -> list[dict]:
        d = doc["document"]
        self.doc = d
        self.base = json_path.parent.relative_to(ROOT).as_posix()
        self.title = d.get("title") or d.get("file_name")
        self.chunks: list[dict] = []
        self._carry: list[dict] = []
        els = [e for p in doc["pages"] for e in p["elements"]]
        els.sort(key=lambda e: e["metadata"].get("global_order", 0))
        captions_used = {(e.get("caption") or {}).get("linked_element_id") if isinstance(e.get("caption"), dict)
                         else e.get("caption_element_id") for e in els}

        buf: list[dict] = []
        cur_section: tuple = ()
        for e in els:
            m = e["metadata"]
            sec = tuple(m.get("section_path") or [])
            if e["type"] in SKIP_TYPES or (sec and SKIP_SECTIONS.search(sec[0])):
                continue
            if e.get("category") == "caption" and (m.get("caption_for") or e["id"] in captions_used):
                continue  # emitted with its figure / table
            if e.get("category") in ("heading", "subheading", "title"):
                self._flush_text(buf, cur_section)
                buf, cur_section = [], tuple(sec)
                continue
            if sec != cur_section and e["type"] in TEXTY:
                self._flush_text(buf, cur_section)
                buf, cur_section = [], sec
            if e["type"] in ("image", "chart", "signature", "stamp", "barcode"):
                self._figure(e)
            elif e["type"] == "table":
                self._table(e)
            elif e.get("category") in ("reference", "reference_list"):
                self._references(e)
            elif e["type"] in TEXTY or e["type"] == "text":
                if element_text(e).strip():
                    buf.append(e)
        self._flush_text(buf, cur_section, final=True)
        for i, c in enumerate(self.chunks):
            c["chunk_id"] = f"{d['id']}:c{i:05d}"
            c["chunk_index"] = i
        for i, c in enumerate(self.chunks):
            c["prev_chunk_id"] = self.chunks[i - 1]["chunk_id"] if i else None
            c["next_chunk_id"] = self.chunks[i + 1]["chunk_id"] if i + 1 < len(self.chunks) else None
        return self.chunks

    # ------------------------------------------------------------ builders
    def _header(self, section: tuple | list) -> str:
        path = " > ".join(section) if section else ""
        return f"{self.title}" + (f" | {path}" if path else "")

    def _emit(self, ctype: str, content: str, els: list[dict], section, **extra):
        content = content.strip()
        if not content:
            return
        header = self._header(section)
        text = f"{header}\n\n{content}"
        pages = sorted({p for e in els for p in e["metadata"].get("pages", [e["metadata"]["page"]])})
        roles = Counter(e.get("section_role") for e in els if e.get("section_role"))
        cats = Counter(e.get("category") or e["type"] for e in els)
        self.chunks.append({
            "chunk_id": None, "chunk_type": ctype, "text": text, "content": content, "header": header,
            "token_count": self.tok.n(text), "char_count": len(text),
            "doc_id": self.doc["id"], "doc_title": self.title, "file_name": self.doc["file_name"],
            "doc_category": self.doc.get("category"), "source": self.doc.get("source"),
            "section_path": list(section), "section_role": roles.most_common(1)[0][0] if roles else None,
            "categories": dict(cats), "pages": pages, "page_start": pages[0] if pages else None,
            "page_end": pages[-1] if pages else None, "element_ids": [e["id"] for e in els],
            "locations": [loc(e) for e in els], "entities": merge_entities(els),
            "language": next((e["metadata"].get("language") for e in els if e["metadata"].get("language")), self.doc.get("language")),
            **extra,
        })

    def _flush_text(self, buf: list[dict], section, final: bool = False):
        # tiny sections (cover-page lines, one-sentence subsections) are carried into the next chunk
        buf = self._carry + buf
        self._carry = []
        if not buf:
            return
        ids = {e["id"] for e in buf}
        if not final and sum(self.tok.n(element_text(e, ids)) for e in buf) < MIN:
            self._carry = buf
            return
        header_tokens = self.tok.n(self._header(section)) + 2
        budget, cap = TARGET - header_tokens, MAX - header_tokens
        pieces: list[tuple[str, dict]] = []
        for e in buf:  # split oversized paragraphs by sentence
            t = element_text(e, ids).strip()
            if self.tok.n(t) <= cap:
                pieces.append((t, e))
                continue
            cur = ""
            for s in SENT_SPLIT.split(t):
                if cur and self.tok.n(cur + " " + s) > budget:
                    pieces.append((cur, e))
                    cur = s
                else:
                    cur = f"{cur} {s}".strip()
            if cur:
                pieces.append((cur, e))

        groups: list[list[tuple[str, dict]]] = [[]]
        size = 0
        for t, e in pieces:
            n = self.tok.n(t)
            if groups[-1] and size + n > budget:
                groups.append([])
                size = 0
            groups[-1].append((t, e))
            size += n
        # fold a tiny trailing group into the previous one when it still fits under the hard cap
        if len(groups) > 1 and sum(self.tok.n(t) for t, _ in groups[-1]) < MIN:
            if sum(self.tok.n(t) for g in groups[-2:] for t, _ in g) <= cap:
                groups[-2].extend(groups.pop())

        prev_text = ""
        for g in groups:
            body = "\n\n".join(t for t, _ in g)
            overlap = self.tok.tail(prev_text, OVERLAP) if prev_text and OVERLAP else ""
            content = (f"… {overlap}\n\n{body}" if overlap else body)
            els = list({id(e): e for _, e in g}.values())
            formulas = [{"element_id": e["id"], "latex": e.get("latex"), "equation_number": e.get("equation_number")}
                        for e in els if e["type"] == "formula" and e.get("formula_type") != "inline"]
            self._emit("text", content, els, section, overlap_chars=len(overlap),
                       formulas=formulas or None)
            prev_text = body

    def _figure(self, e: dict):
        content = figure_text(e)
        if not content or (not (e.get("caption") or {}).get("text") and not e.get("description")):
            return  # decorative / unlabelled graphics carry no retrievable meaning
        cap = e.get("caption") or {}
        self._emit("figure", content, [e], e["metadata"].get("section_path") or [],
                   figure={"element_id": e["id"], "figure_number": cap.get("figure_number"),
                           "image_file": f"{self.base}/{e['image_file']}" if e.get("image_file") else None,
                           "image_type": e.get("image_type") or e.get("category"),
                           "chart_type": e.get("chart_type"), "values_estimated": e.get("values_estimated")})

    def _table(self, e: dict):
        cap = e.get("caption") or ""
        md = e.get("markdown") or e.get("csv") or ""
        if e.get("is_empty") or not md.strip():
            return
        lines = md.strip().splitlines()
        head, rows = (lines[:2], lines[2:]) if len(lines) > 2 and set(lines[1].replace("|", "").strip()) <= set("-: ") else ([], lines)
        section = e["metadata"].get("section_path") or []
        meta = {"element_id": e["id"], "table_number": e.get("table_number"), "num_rows": e.get("num_rows"),
                "num_columns": e.get("num_columns"), "headers": e.get("headers"),
                "image_file": f"{self.base}/{e['image_file']}" if e.get("image_file") else None}
        prefix = "\n".join(filter(None, [cap, f"Notes: {e['notes']}" if e.get("notes") else None]))
        budget = TARGET - self.tok.n(self._header(section) + prefix + "\n".join(head)) - 4
        part, parts = [], []
        for r in rows:
            if part and self.tok.n("\n".join(part + [r])) > budget:
                parts.append(part)
                part = []
            part.append(r)
        if part:
            parts.append(part)
        for i, p in enumerate(parts):
            content = "\n".join(filter(None, [prefix, "\n".join(head + p)]))
            self._emit("table", content, [e], section, table=dict(meta, part=i + 1, parts=len(parts)))

    def _references(self, e: dict):
        section = e["metadata"].get("section_path") or []
        items = ([f"{i.get('marker') or ''} {i['text']}".strip() for i in e.get("items", [])]
                 if e["type"] == "list" else [e.get("content", "")])
        part = []
        for it in items:
            if part and self.tok.n("\n".join(part + [it])) > TARGET:
                self._emit("reference", "\n".join(part), [e], section)
                part = []
            part.append(it)
        if part:
            self._emit("reference", "\n".join(part), [e], section)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", help="extracted JSON files / globs (default: all under extracted/)")
    ap.add_argument("--out", default=str(ROOT / "chunks"))
    a = ap.parse_args()

    files = ([Path(p) for pat in a.files for p in glob.glob(pat, recursive=True)] if a.files else
             [p for p in (ROOT / "extracted").rglob("*.json") if p.name != "index.json"])
    files = sorted({f.resolve() for f in files if f.suffix == ".json"})
    if not files:
        sys.exit("no extracted JSON found - run extract.py first")

    tok = Tok(EMBED_MODEL)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_chunks, per_doc = [], {}
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        chunks = Chunker(tok).run(doc, f)
        all_chunks += chunks
        per_doc[doc["document"]["file_name"]] = {"chunks": len(chunks),
                                                 "by_type": dict(Counter(c["chunk_type"] for c in chunks))}
        print(f"{f.name}: {len(chunks)} chunks {per_doc[doc['document']['file_name']]['by_type']}")

    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    toks = sorted(c["token_count"] for c in all_chunks)
    stats = {"embed_model": EMBED_MODEL, "target_tokens": TARGET, "max_tokens": MAX, "overlap_tokens": OVERLAP,
             "total_chunks": len(all_chunks), "by_type": dict(Counter(c["chunk_type"] for c in all_chunks)),
             "tokens": {"min": toks[0], "median": toks[len(toks) // 2], "p95": toks[int(len(toks) * 0.95)],
                        "max": toks[-1], "total": sum(toks)} if toks else None,
             "documents": per_doc}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("total_chunks", "by_type", "tokens")}, indent=2))
    print(f"-> {out_dir / 'chunks.jsonl'}")


if __name__ == "__main__":
    main()
