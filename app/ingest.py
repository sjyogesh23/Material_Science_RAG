"""Load chunks (chunk.py output) + the extracted element JSON into Postgres.

Per document:
  * upsert ``documents`` (absolute PDF path, page counts, extraction metadata)
  * upsert ``chunks`` (full chunk record in ``payload``); embed ``text`` (no prompt) only for chunks whose
    ``content_hash`` = sha256(embed identity + text) changed or that have no embedding yet (``--force`` = all).
    The embed identity is ``ollama:<ollama_embed_model>`` or ``<embed_model>`` (``settings.embed_backend``), so
    switching backend re-embeds everything - documents and queries always come from the same model.
    Ollama backend: batches of ``embed_batch_size`` texts to the Ollama server (GPU if Ollama has room, unless
    ``ingest_device=cpu``). sentence_transformers backend: cuda if >= ``ingest_min_free_vram_gb`` VRAM is free.
  * delete chunks of the document that no longer exist in chunks.jsonl
  * rebuild ``element_chunks`` (element id -> chunk) and ``chunk_links`` (chunk -> chunk):
      equation        element ``cross_references[*].target_id`` / ``refers_to_equation`` relationships
      variables       formula -> paragraph defining its variables (``variables_defined_in``)
      caption         figure/table <-> caption element (``has_caption``, ``caption.linked_element_id``,
                      ``metadata.caption_for``) when they landed in different chunks
      mentions_figure / mentions_table / mentions_equation
                      "Figure 6-3", "Table 2-4", "Eq. 7.86" written in a text chunk -> that chunk
      figure / context
                      text chunk -> figure chunk of the same section on the same page (and back);
                      lets ``Retriever.figures_for`` find the figures belonging to a text hit
  * one ``llm_calls`` row (purpose ``ingest_embed``, provider ``local``)

Usage:
    python -m app.ingest                       # chunks/chunks.jsonl
    python -m app.ingest --chunks other.jsonl --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from app import db
from app.config import ROOT, Settings, get_settings

Link = tuple[str, str, str, str | None]     # (src_chunk_id, dst_chunk_id, kind, label)

MENTION_RE = re.compile(r"\b(Fig(?:ure)?s?\.?|Tables?|Eq(?:uation)?s?\.?)\s*\(?(\d+(?:[-.]\d+)+)\)?", re.I)


# ----------------------------------------------------------------------------- inputs
def read_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def find_extracted(settings: Settings, doc_id: str, file_name: str) -> Path | None:
    """Locate the extracted JSON of a document (via extracted/index.json, else by folder name)."""
    index = settings.extracted_dir / "index.json"
    if index.exists():
        for entry in json.loads(index.read_text(encoding="utf-8")).values():
            if entry.get("doc_id") == doc_id and entry.get("output"):
                p = ROOT / entry["output"]
                if p.exists():
                    return p
    stem = re.sub(r"\W+", "_", Path(file_name).stem).strip("_")
    hits = list(settings.extracted_dir.glob(f"**/{stem}/{stem}.json"))
    return hits[0] if hits else None


def content_hash(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\n{text}".encode("utf-8")).hexdigest()


def _num(label: str) -> str:
    """Normalise "1-1" / "1.1" / "(7.86)" to "1.1" / "7.86" so figure/equation numbers compare equal."""
    return re.sub(r"[-–.]+", ".", label.strip("() ").strip())


# ----------------------------------------------------------------------------- links
def build_links(chunks: list[dict], extracted: dict | None) -> list[Link]:
    e2c = {eid: c["chunk_id"] for c in chunks for eid in c.get("element_ids") or []}
    links: dict[tuple[str, str, str], str | None] = {}

    def add(src: str | None, dst: str | None, kind: str, label: str | None = None) -> None:
        if src and dst and src != dst:
            links.setdefault((src, dst, kind), label)

    elements: dict[str, dict] = {}
    if extracted:
        elements = {e["id"]: e for p in extracted.get("pages", []) for e in p.get("elements", [])}
        # element-level cross references (equation / figure / table references with resolved targets)
        for eid, el in elements.items():
            for ref in el.get("cross_references") or []:
                if isinstance(ref, dict) and ref.get("target_id"):
                    add(e2c.get(eid), e2c.get(ref["target_id"]), ref.get("kind") or "ref", ref.get("label"))
            cap = el.get("caption")
            if isinstance(cap, dict) and cap.get("linked_element_id"):
                add(e2c.get(eid), e2c.get(cap["linked_element_id"]), "caption")
                add(e2c.get(cap["linked_element_id"]), e2c.get(eid), "caption")
            cap_for = (el.get("metadata") or {}).get("caption_for")
            if cap_for:
                add(e2c.get(eid), e2c.get(cap_for), "caption")
                add(e2c.get(cap_for), e2c.get(eid), "caption")
        for rel in extracted.get("relationships") or []:
            src, dst, typ = e2c.get(rel.get("from")), e2c.get(rel.get("to")), rel.get("type")
            if typ == "refers_to_equation":
                label = (elements.get(rel["to"]) or {}).get("equation_number")
                add(src, dst, "equation", _num(label) if label else None)
            elif typ == "variables_defined_in":
                label = (elements.get(rel["from"]) or {}).get("equation_number")
                add(src, dst, "variables", _num(label) if label else None)
            elif typ == "has_caption":
                add(src, dst, "caption")
                add(dst, src, "caption")

    # "Figure 6-3" / "Table 2-4" / "Eq. 7.86" mentions inside text chunks
    fig_by_num = {_num(re.sub(r"(?i)^fig(ure)?\.?\s*", "", c["figure"]["figure_number"])): c["chunk_id"]
                  for c in chunks if c.get("figure") and c["figure"].get("figure_number")}
    tab_by_num = {_num(c["table"]["table_number"]): c["chunk_id"]
                  for c in chunks if c.get("table") and c["table"].get("table_number")}
    eq_by_num: dict[str, str] = {}
    for c in chunks:
        for f in c.get("formulas") or []:
            if f.get("equation_number"):
                eq_by_num.setdefault(_num(f["equation_number"]), c["chunk_id"])
    for c in chunks:
        if c["chunk_type"] != "text":
            continue
        for m in MENTION_RE.finditer(c.get("content") or ""):
            word, num = m.group(1).lower(), _num(m.group(2))
            if word.startswith("fig"):
                add(c["chunk_id"], fig_by_num.get(num), "mentions_figure", m.group(0))
            elif word.startswith("tab"):
                add(c["chunk_id"], tab_by_num.get(num), "mentions_table", m.group(0))
            else:
                add(c["chunk_id"], eq_by_num.get(num), "mentions_equation", m.group(0))

    # figure/table <-> surrounding text of the same section on the same page
    texts_by_section: dict[tuple, list[dict]] = defaultdict(list)
    for c in chunks:
        if c["chunk_type"] == "text":
            texts_by_section[(c["doc_id"], tuple(c.get("section_path") or []))].append(c)
    for c in chunks:
        if c["chunk_type"] not in ("figure", "table"):
            continue
        label = (c.get("figure") or {}).get("figure_number") or (c.get("table") or {}).get("table_number")
        same = texts_by_section.get((c["doc_id"], tuple(c.get("section_path") or [])), [])
        pages = set(c.get("pages") or [])
        ctx = [t for t in same if pages & set(t.get("pages") or [])]
        if not ctx and same:            # no text on that page: nearest text chunk of the section
            ctx = [min(same, key=lambda t: abs((t.get("chunk_index") or 0) - (c.get("chunk_index") or 0)))]
        for t in ctx:
            add(t["chunk_id"], c["chunk_id"], c["chunk_type"], label)   # text -> figure|table
            add(c["chunk_id"], t["chunk_id"], "context", label)          # figure|table -> text
    return [(s, d, k, lbl) for (s, d, k), lbl in links.items()]


# ----------------------------------------------------------------------------- embedding
def embed_texts(settings: Settings, texts: list[str]) -> tuple[list[Any], str]:
    if settings.embed_backend == "ollama":
        return _embed_texts_ollama(settings, texts)
    from app.retrieval.embedder import load_model, pick_ingest_device

    device = pick_ingest_device(settings)
    model = load_model(settings.embed_model, device)
    vecs = model.encode(texts, batch_size=settings.embed_batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=len(texts) > 32)
    del model
    if device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()
    return [v.astype("float32") for v in vecs], device


def _embed_texts_ollama(settings: Settings, texts: list[str]) -> tuple[list[Any], str]:
    """Documents (no prompt) through Ollama, in batches. Uses the query-time options (same num_ctx / GPU
    placement, so the model is not reloaded between ingest and queries) unless ingest_device=cuda, which lets
    Ollama put the embedder on the GPU (faster bulk embedding, reloaded on CPU at the next query)."""
    from app.retrieval.embedder import ollama_client, ollama_embed

    client = ollama_client(settings)
    num_gpu = -1 if settings.ingest_device.lower() == "cuda" else None
    out: list[Any] = []
    bs = max(1, settings.embed_batch_size)
    for i, batch in enumerate(_batches(texts, bs)):
        vecs, _ = ollama_embed(settings, batch, num_gpu=num_gpu, client=client)
        out.extend(vecs)
        if len(texts) > 32 and (i + 1) % 5 == 0:
            print(f"  embedded {min(len(texts), (i + 1) * bs)}/{len(texts)}", flush=True)
    where = "ollama" if (num_gpu == -1 or settings.ollama_embed_num_gpu != 0) else "ollama:cpu"
    return out, where


# ----------------------------------------------------------------------------- ingest
def _batches(items: list, n: int) -> Iterable[list]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def ingest_document(conn, settings: Settings, doc_id: str, chunks: list[dict], force: bool) -> dict:
    first = chunks[0]
    extracted_path = find_extracted(settings, doc_id, first.get("file_name") or "")
    extracted = json.loads(extracted_path.read_text(encoding="utf-8")) if extracted_path else None
    meta_doc = dict((extracted or {}).get("document") or {})
    meta_doc.pop("fonts", None)
    rel_pdf = meta_doc.get("relative_path") or f"data/{first.get('doc_category', '')}/{first.get('file_name')}"
    pdf_path = str((ROOT / rel_pdf).resolve())
    total_pages = meta_doc.get("total_pages")
    extracted_pages = sum(1 for p in (extracted or {}).get("pages") or [] if p.get("elements")) or None
    meta = {"document": meta_doc, "extracted_json": str(extracted_path) if extracted_path else None,
            "summary": (extracted or {}).get("summary")}

    conn.execute(
        "INSERT INTO documents(doc_id, title, file_name, pdf_path, total_pages, extracted_pages, meta, ingested_at)"
        " VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb, now()) ON CONFLICT (doc_id) DO UPDATE SET title=EXCLUDED.title,"
        " file_name=EXCLUDED.file_name, pdf_path=EXCLUDED.pdf_path, total_pages=EXCLUDED.total_pages,"
        " extracted_pages=EXCLUDED.extracted_pages, meta=EXCLUDED.meta, ingested_at=now()",
        (doc_id, first.get("doc_title"), first.get("file_name"), pdf_path, total_pages, extracted_pages,
         json.dumps(meta, default=str)))

    # which chunks need (re-)embedding?
    existing = {r[0]: (r[1], r[2]) for r in conn.execute(
        "SELECT chunk_id, content_hash, embedding IS NOT NULL FROM chunks WHERE doc_id = %s", (doc_id,))}
    from app.retrieval.embedder import embed_identity

    hashes = {c["chunk_id"]: content_hash(embed_identity(settings), c["text"]) for c in chunks}
    to_embed = [c for c in chunks
                if force or existing.get(c["chunk_id"]) != (hashes[c["chunk_id"]], True)]
    vectors: dict[str, Any] = {}
    device, embed_ms = None, 0
    if to_embed:
        t0 = time.perf_counter()
        vecs, device = embed_texts(settings, [c["text"] for c in to_embed])
        embed_ms = int((time.perf_counter() - t0) * 1000)
        vectors = {c["chunk_id"]: v for c, v in zip(to_embed, vecs)}

    sql = ("INSERT INTO chunks(chunk_id, doc_id, chunk_index, chunk_type, text, content, header, section_path,"
           " section_role, pages, page_start, page_end, token_count, content_hash, payload, embedding)"
           " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)"
           " ON CONFLICT (chunk_id) DO UPDATE SET doc_id=EXCLUDED.doc_id, chunk_index=EXCLUDED.chunk_index,"
           " chunk_type=EXCLUDED.chunk_type, text=EXCLUDED.text, content=EXCLUDED.content, header=EXCLUDED.header,"
           " section_path=EXCLUDED.section_path, section_role=EXCLUDED.section_role, pages=EXCLUDED.pages,"
           " page_start=EXCLUDED.page_start, page_end=EXCLUDED.page_end, token_count=EXCLUDED.token_count,"
           " content_hash=EXCLUDED.content_hash, payload=EXCLUDED.payload,"
           " embedding=COALESCE(EXCLUDED.embedding, chunks.embedding)")
    rows = [(c["chunk_id"], doc_id, c.get("chunk_index"), c["chunk_type"], c["text"], c.get("content"),
             c.get("header"), c.get("section_path") or [], c.get("section_role"), c.get("pages") or [],
             c.get("page_start"), c.get("page_end"), c.get("token_count"), hashes[c["chunk_id"]],
             json.dumps(c, ensure_ascii=False), vectors.get(c["chunk_id"])) for c in chunks]
    with conn.cursor() as cur:
        for batch in _batches(rows, 100):
            cur.executemany(sql, batch)

    ids = [c["chunk_id"] for c in chunks]
    deleted = conn.execute("DELETE FROM chunks WHERE doc_id = %s AND NOT (chunk_id = ANY(%s))",
                           (doc_id, ids)).rowcount

    # element -> chunk map
    conn.execute("DELETE FROM element_chunks WHERE doc_id = %s", (doc_id,))
    ec_rows = [(doc_id, eid, c["chunk_id"]) for c in chunks for eid in dict.fromkeys(c.get("element_ids") or [])]
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO element_chunks(doc_id, element_id, chunk_id) VALUES (%s,%s,%s)"
                        " ON CONFLICT DO NOTHING", ec_rows)

    # chunk -> chunk links
    conn.execute("DELETE FROM chunk_links WHERE src_chunk_id = ANY(%s) OR dst_chunk_id = ANY(%s)", (ids, ids))
    links = build_links(chunks, extracted)
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO chunk_links(src_chunk_id, dst_chunk_id, kind, label) VALUES (%s,%s,%s,%s)"
                        " ON CONFLICT DO NOTHING", links)
    by_kind: dict[str, int] = defaultdict(int)
    for _, _, kind, _ in links:
        by_kind[kind] += 1

    return {"doc_id": doc_id, "chunks": len(chunks), "embedded": len(to_embed),
            "skipped": len(chunks) - len(to_embed), "deleted": deleted, "element_chunks": len(ec_rows),
            "chunk_links": len(links), "links_by_kind": dict(sorted(by_kind.items())), "device": device,
            "embed_ms": embed_ms, "embed_tokens": sum(int(c.get("token_count") or 0) for c in to_embed),
            "extracted_json": str(extracted_path) if extracted_path else None}


def run(chunks_path: Path | None = None, force: bool = False) -> list[dict]:
    settings = get_settings()
    path = chunks_path or settings.chunks_path
    chunks = read_chunks(path)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c)
    db.init_db()
    results = []
    for doc_id, doc_chunks in by_doc.items():
        doc_chunks.sort(key=lambda c: c.get("chunk_index") or 0)
        with db.get_pool().connection() as conn:          # one transaction per document
            res = ingest_document(conn, settings, doc_id, doc_chunks, force)
        results.append(res)
        from app.retrieval.embedder import embed_identity

        db.log_llm_call("ingest_embed", provider="local", model=embed_identity(settings),
                        prompt_tokens=res["embed_tokens"], duration_ms=res["embed_ms"],
                        meta={k: res[k] for k in ("doc_id", "chunks", "embedded", "skipped", "deleted",
                                                  "chunk_links", "device")})
    with db.get_pool().connection() as conn:
        conn.execute("ANALYZE chunks")
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest chunks.jsonl (+ extracted JSON) into Postgres")
    ap.add_argument("--chunks", type=Path, default=None, help="chunks JSONL (default: settings.chunks_path)")
    ap.add_argument("--force", action="store_true", help="re-embed every chunk")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    path = args.chunks if args.chunks is None or args.chunks.is_absolute() else ROOT / args.chunks
    for res in run(path, args.force):
        print(json.dumps(res, indent=2))
    print(f"done in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
