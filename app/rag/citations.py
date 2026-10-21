"""Citation validation / renumbering + Source / Image payloads for the UI (PDF highlights, figures).

``validate(answer, hits)``:
  * keeps only ``[n]`` that point to an existing source (math ``$...$`` / code spans are never touched, so an
    interval like ``$[0, 1]$`` is safe); ``[Source 2]``, ``[2][3]``, ``[2,3]``, ``[2-4]`` are normalised
  * renumbers by first appearance (1..m) in the text and in the returned list; adjacent groups are merged
    into one ``[1, 2]``; sources never cited are dropped
  * zero citations: the best source is appended as ``[1]`` only if the answer overlaps it, else the returned
    list carries ``uncited=True`` (UI shows a warning)
"""
from __future__ import annotations

import logging
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.llm.prompts import OOC

log = logging.getLogger(__name__)

GROUP = r"\[\s*(?:[Ss]ources?\s*)?\d+(?:\s*(?:,|;|–|-|and)\s*(?:[Ss]ources?\s*)?\d+)*\s*\](?!\()"
RUN_RE = re.compile(rf"[ \t]?{GROUP}(?:\s*{GROUP})*")
NUM_RE = re.compile(r"(\d+)\s*(?:(?:–|-)\s*(\d+))?")
MASK_RE = re.compile(r"\$\$.+?\$\$|\$[^$\n]+?\$|```.+?```|`[^`\n]+`", re.S)
WORD_RE = re.compile(r"[a-z][a-z0-9-]{3,}")
STOP = set("this that with from have been were which their there these those into about than then they them "
           "also such when where what will would could should because between each other more most some only "
           "very used using answer source sources".split())


class SourceList(list):
    """list[Source] + ``uncited`` flag (keeps ``clean, used = validate(...)`` unpacking)."""
    uncited: bool = False


# ----------------------------------------------------------------------------- page sizes
_pdf_paths: dict[str, Path] = {}
_lock = threading.Lock()


def register_pdf_path(doc_id: str, path: str | Path | None) -> None:
    """Called by the pipeline with ``documents.pdf_path`` so page sizes can be read from the PDF."""
    if doc_id and path:
        _pdf_paths[doc_id] = Path(path)


def _find_pdf(doc_id: str, file_name: str | None) -> Path | None:
    if doc_id in _pdf_paths:
        p = _pdf_paths[doc_id]
        if not p.is_absolute():
            from app.llm.ollama_client import settings
            p = Path(settings().root) / p
        return p
    if not file_name:
        return None
    from app.llm.ollama_client import settings
    for p in (Path(settings().root) / "data").rglob(file_name):
        _pdf_paths[doc_id] = p
        return p
    return None


@lru_cache(maxsize=16)
def _pdf_page_sizes(path: str) -> dict[int, tuple[float, float]]:
    import pymupdf

    with pymupdf.open(path) as doc:
        return {i + 1: (round(p.rect.width, 2), round(p.rect.height, 2)) for i, p in enumerate(doc)}


def page_sizes(doc_id: str, pages: list[int], payload: dict | None = None) -> dict[str, list[float]]:
    payload = payload or {}
    known = payload.get("page_sizes") or {}
    if (ps := payload.get("page_size")) and len(pages) == 1 and isinstance(ps, (list, tuple)):
        known = {str(pages[0]): list(ps)}
    out = {str(p): list(known[str(p)]) for p in pages if str(p) in known}
    missing = [p for p in pages if str(p) not in out]
    if missing:
        path = _find_pdf(doc_id, payload.get("file_name"))
        try:
            with _lock:
                sizes = _pdf_page_sizes(str(path)) if path and path.exists() else {}
        except Exception as e:  # noqa: BLE001
            log.warning("page sizes unavailable for %s: %s", doc_id, e)
            sizes = {}
        for p in missing:
            out[str(p)] = list(sizes.get(p, (612.0, 792.0)))  # US-letter fallback
    return out


# ----------------------------------------------------------------------------- payloads
def _g(hit: Any, name: str, default: Any = None) -> Any:
    return hit.get(name, default) if isinstance(hit, dict) else getattr(hit, name, default)


def highlights(payload: dict) -> list[dict]:
    by_page: dict[int, list[list[float]]] = {}
    for loc in payload.get("locations") or []:
        if loc.get("page") and loc.get("bbox"):
            by_page.setdefault(int(loc["page"]), []).append([round(float(v), 2) for v in loc["bbox"]])
    return [{"page": p, "bboxes": b} for p, b in sorted(by_page.items())]


def snippet(text: str, n: int = 320) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…"


def to_source(hit: Any, n: int) -> dict:
    payload = _g(hit, "payload") or {}
    pages = list(_g(hit, "pages") or payload.get("pages") or [])
    score = _g(hit, "rerank_score")
    fig = payload.get("figure")
    if fig:
        fig = {**fig, "url": f"/api/figure/{_g(hit, 'chunk_id')}"}
    return {
        "n": n, "chunk_id": _g(hit, "chunk_id"), "doc_id": _g(hit, "doc_id"),
        "doc_title": payload.get("doc_title") or "", "chunk_type": _g(hit, "chunk_type"),
        "header": _g(hit, "header") or "", "section_path": list(_g(hit, "section_path") or []), "pages": pages,
        "snippet": snippet(_g(hit, "content") or ""),
        "score": round(float(score if score is not None else _g(hit, "rrf_score", 0.0) or 0.0), 4),
        "highlights": highlights(payload),
        "page_size": page_sizes(_g(hit, "doc_id"), pages, payload),
        "figure": fig, "via": _g(hit, "via", "hybrid"),
    }


def figure_number(hit: Any) -> str | None:
    payload = _g(hit, "payload") or {}
    num = (payload.get("figure") or {}).get("figure_number")
    if num:
        return num
    m = re.match(r"\s*(Fig(?:ure)?\.?\s*\d+[-.–]?\d*)", _g(hit, "content") or "")
    return m.group(1) if m else None


def to_image(hit: Any) -> dict:
    content = _g(hit, "content") or ""
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    caption = lines[0] if lines else ""
    desc = next((ln.split(":", 1)[1].strip() for ln in lines if ln.lower().startswith("figure description:")), None)
    pages = list(_g(hit, "pages") or [])
    return {"chunk_id": _g(hit, "chunk_id"), "url": f"/api/figure/{_g(hit, 'chunk_id')}", "caption": caption,
            "figure_number": figure_number(hit), "page": pages[0] if pages else None, "description": desc}


# ----------------------------------------------------------------------------- validation
def _words(text: str) -> set[str]:
    return {w for w in WORD_RE.findall(text.lower()) if w not in STOP}


def _overlap(answer: str, source_text: str) -> float:
    a = _words(answer)
    return len(a & _words(source_text)) / len(a) if a else 0.0


def _nums(group_run: str) -> list[int]:
    out: list[int] = []
    for g in re.findall(GROUP, group_run):
        for a, b in NUM_RE.findall(re.sub(r"[Ss]ources?", "", g)):
            lo, hi = int(a), int(b or a)
            out += list(range(lo, hi + 1)) if 0 < hi - lo <= 10 else [lo] + ([hi] if b else [])
    return out


def validate(answer: str, sources: list[Any]) -> tuple[str, SourceList]:
    text = (answer or "").strip()
    used = SourceList()
    if not text or text.startswith(OOC):
        return (OOC if text.startswith(OOC) else text), used

    masks: list[str] = []

    def _mask(m: re.Match) -> str:
        masks.append(m.group(0))
        return f"\x00{len(masks) - 1}\x00"

    body = MASK_RE.sub(_mask, text)
    mapping: dict[int, int] = {}

    def _renumber(m: re.Match) -> str:
        new: list[int] = []
        for old in _nums(m.group(0)):
            if 1 <= old <= len(sources):
                if old not in mapping:
                    mapping[old] = len(mapping) + 1
                if mapping[old] not in new:
                    new.append(mapping[old])
        if not new:
            return ""
        return " [" + ", ".join(map(str, sorted(new))) + "]"

    body = RUN_RE.sub(_renumber, body)
    body = re.sub(r" +([.,;:!?)])", r"\1", body)
    body = re.sub(r"\x00(\d+)\x00", lambda m: masks[int(m.group(1))], body)
    body = re.sub(r"(?m)^ \[", "[", body)

    if mapping:
        for old, new in sorted(mapping.items(), key=lambda kv: kv[1]):
            used.append(to_source(sources[old - 1], new))
        return body, used

    if sources and _overlap(body, _g(sources[0], "content") or "") >= 0.45:
        used.append(to_source(sources[0], 1))
        return body.rstrip() + " [1]", used
    used.uncited = True
    return body, used
