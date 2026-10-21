"""Hybrid retriever: dense (pgvector cosine) + BM25 -> reciprocal-rank fusion -> Qwen3 rerank.

    search(query, sub_queries=[...])
        for q in [query, *sub_queries]: dense top_k_dense + bm25 top_k_bm25
        RRF over all ranked lists (score = sum 1 / (rrf_k + rank))
        rerank the top rerank_top_n against `query` (P(yes) 0..1) -> top k sorted by rerank score
    expand_cross_refs(hits)   follow chunk_links (equation refs, variable definitions, mentions ...)
    figures_for(query, hits)  figure chunks among the hits, linked from them, or found by a figure-only search
    coverage_for(sub_queries) best 1-2 text/table chunks PER sub-query (comparisons must cover every entity)
    section_text_for(ids)     same-section text explaining a figure/table (chunk_links figure/context)
    figure_by_number(label)   figure/table chunk by its number ("Figure 2-7")

Timings of the last search are kept in ``Retriever.last_timings`` (ms per stage). Every search logs
``embed_query`` and ``rerank`` rows in llm_calls (provider ``local``, prompt_tokens = tokens processed).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any

from app import db
from app.config import Settings, get_settings
from app.retrieval.bm25 import BM25Index
from app.retrieval.embedder import OllamaEmbedder, QueryEmbedder, get_query_embedder
from app.retrieval.reranker import OllamaReranker, Reranker, get_reranker

log = logging.getLogger(__name__)

_COLUMNS = "chunk_id, doc_id, chunk_type, header, content, text, section_path, pages, payload"
# chunk_links kinds followed by expand_cross_refs (figure/table/context links are used by figures_for)
CROSS_REF_KINDS = ("equation", "variables", "caption", "mentions_equation", "mentions_table",
                   "mentions_figure", "ref")
_KIND_PRIORITY = {k: i for i, k in enumerate(CROSS_REF_KINDS)}


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    chunk_type: str
    header: str
    content: str
    text: str
    section_path: list[str]
    pages: list[int]
    payload: dict = field(default_factory=dict)
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None      # sigmoid prob 0..1
    via: str = "hybrid"                    # "hybrid" | "cross_ref" | "neighbor" | "figure_search"


def _hit_from_row(row: tuple) -> Hit:
    return Hit(chunk_id=row[0], doc_id=row[1], chunk_type=row[2], header=row[3] or "", content=row[4] or "",
               text=row[5] or "", section_path=list(row[6] or []), pages=list(row[7] or []),
               payload=row[8] or {})


class Retriever:
    def __init__(self, settings: Settings | None = None):
        self._settings = settings
        self._bm25: BM25Index | None = None
        self._lock = threading.Lock()
        self.last_timings: dict[str, float] = {}

    # ------------------------------------------------------------------ lazy components
    @property
    def settings(self) -> Settings:
        return self._settings or get_settings()     # follows reload_settings() unless pinned

    @property
    def embedder(self) -> QueryEmbedder | OllamaEmbedder:
        return get_query_embedder(self.settings)

    @property
    def reranker(self) -> Reranker | OllamaReranker:
        return get_reranker(self.settings)

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            with self._lock:
                if self._bm25 is None:
                    self._bm25 = BM25Index().build_from_db()
        return self._bm25

    def reload_bm25(self) -> None:
        idx = BM25Index().build_from_db()
        with self._lock:
            self._bm25 = idx

    def warmup(self, preload_chat: bool = True) -> dict[str, float]:
        """Load BM25 + embedder + reranker now (so the first question is not slow); returns seconds per part.

        With Ollama backends the chat model is loaded FIRST (same num_ctx as the LLM client): Ollama sizes the
        GPU split of a model when it loads it, so loading the small reranker first would push chat layers to
        the CPU. Afterwards the reranker gets the VRAM that is still free (it fits next to qwen3:4b on 4 GB).
        """
        s, out = self.settings, {}
        t0 = time.perf_counter()
        _ = self.bm25
        out["bm25_s"] = time.perf_counter() - t0
        if preload_chat and "ollama" in (s.embed_backend, s.rerank_backend):
            t0 = time.perf_counter()
            try:
                from app.retrieval.embedder import ollama_client

                ollama_client(s).generate(model=s.chat_model, prompt="", keep_alive=s.ollama_keep_alive,
                                          options={"num_ctx": s.num_ctx})
            except Exception as exc:  # pragma: no cover - chat model missing / server down
                log.warning("chat model preload failed: %s", exc)
            out["chat_preload_s"] = time.perf_counter() - t0
        for name, comp in (("embedder", self.embedder), ("reranker", self.reranker)):
            t0 = time.perf_counter()
            comp.warmup()
            out[f"{name}_s"] = time.perf_counter() - t0
        log.info("retriever warm: %s", {k: round(v, 2) for k, v in out.items()})
        return out

    # ------------------------------------------------------------------ stages
    def _dense(self, vec: Any, k: int, chunk_types: list[str] | None) -> list[str]:
        sql = "SELECT chunk_id FROM chunks WHERE embedding IS NOT NULL"
        params: list[Any] = []
        if chunk_types:
            sql += " AND chunk_type = ANY(%s)"
            params.append(chunk_types)
        sql += " ORDER BY embedding <=> %s LIMIT %s"
        params += [vec, k]
        with db.get_pool().connection() as conn:
            return [r[0] for r in conn.execute(sql, params).fetchall()]

    def _fetch(self, chunk_ids: list[str]) -> dict[str, Hit]:
        if not chunk_ids:
            return {}
        with db.get_pool().connection() as conn:
            rows = conn.execute(f"SELECT {_COLUMNS} FROM chunks WHERE chunk_id = ANY(%s)", (chunk_ids,)).fetchall()
        return {r[0]: _hit_from_row(r) for r in rows}

    def _rerank(self, query: str, hits: list[Hit], session_id: Any = None, message_id: int | None = None) -> None:
        todo = [h for h in hits if h.rerank_score is None]
        if not todo:
            return
        rr = self.reranker
        docs = [h.text or h.content for h in todo]
        t0 = time.perf_counter()
        scores = rr.score(query, docs)
        ms = (time.perf_counter() - t0) * 1000
        self.last_timings["rerank_ms"] = self.last_timings.get("rerank_ms", 0.0) + ms
        for h, s in zip(todo, scores):
            h.rerank_score = s
        db.log_llm_call("rerank", model=rr.model_name, prompt_tokens=rr.count_tokens(query, docs),
                        duration_ms=int(ms), session_id=session_id, message_id=message_id,
                        meta={"pairs": len(docs), "device": rr.device, "query": query[:200]})

    # ------------------------------------------------------------------ public API
    def search(self, query: str, *, k: int | None = None, chunk_types: list[str] | None = None,
               sub_queries: list[str] | None = None, session_id: Any = None,
               message_id: int | None = None, rerank_top_n: int | None = None) -> list[Hit]:
        """See module docstring. ``rerank_top_n`` overrides settings.rerank_top_n (cost knob)."""
        s = self.settings
        k = k or s.final_k
        queries = list(dict.fromkeys(q.strip() for q in [query, *(sub_queries or [])] if q and q.strip()))
        self.last_timings = {}
        if not queries:
            return []
        t_all = time.perf_counter()

        emb = self.embedder
        t0 = time.perf_counter()
        vecs = emb.encode_many(queries)
        embed_ms = (time.perf_counter() - t0) * 1000
        db.log_llm_call("embed_query", model=emb.model_name, prompt_tokens=emb.prompt_tokens(queries),
                        duration_ms=int(embed_ms), session_id=session_id, message_id=message_id,
                        meta={"queries": len(queries), "device": emb.device})

        rrf: dict[str, float] = {}
        dense_rank: dict[str, int] = {}
        bm25_rank: dict[str, int] = {}
        dense_ms = bm25_ms = 0.0
        for q, vec in zip(queries, vecs):
            t0 = time.perf_counter()
            dense = self._dense(vec, s.top_k_dense, chunk_types)
            dense_ms += (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            sparse = [cid for cid, _ in self.bm25.search(q, s.top_k_bm25, chunk_types)]
            bm25_ms += (time.perf_counter() - t0) * 1000
            for ranked, best in ((dense, dense_rank), (sparse, bm25_rank)):
                for rank, cid in enumerate(ranked, 1):
                    rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (s.rrf_k + rank)
                    best[cid] = min(best.get(cid, rank), rank)

        n_rerank = self.reranker.max_candidates(rerank_top_n or s.rerank_top_n)
        top = sorted(rrf, key=rrf.__getitem__, reverse=True)[:n_rerank]
        rows = self._fetch(top)
        hits: list[Hit] = []
        for cid in top:
            h = rows.get(cid)
            if h is None:
                continue
            h.dense_rank, h.bm25_rank, h.rrf_score = dense_rank.get(cid), bm25_rank.get(cid), rrf[cid]
            hits.append(h)
        self.last_timings.update(embed_ms=embed_ms, dense_ms=dense_ms, bm25_ms=bm25_ms)
        self._rerank(query, hits, session_id, message_id)
        hits.sort(key=lambda h: (h.rerank_score or 0.0, h.rrf_score), reverse=True)
        self.last_timings["total_ms"] = (time.perf_counter() - t_all) * 1000
        return hits[:k]

    def links_for(self, chunk_ids: list[str], kinds: tuple[str, ...] | None = None) -> list[tuple[str, str, str, str | None]]:
        """chunk_links rows (src, dst, kind, label) leaving the given chunks."""
        if not chunk_ids:
            return []
        sql = "SELECT src_chunk_id, dst_chunk_id, kind, label FROM chunk_links WHERE src_chunk_id = ANY(%s)"
        params: list[Any] = [chunk_ids]
        if kinds:
            sql += " AND kind = ANY(%s)"
            params.append(list(kinds))
        with db.get_pool().connection() as conn:
            return [tuple(r) for r in conn.execute(sql, params).fetchall()]

    def expand_cross_refs(self, hits: list[Hit], max_extra: int = 3) -> list[Hit]:
        """hits + up to ``max_extra`` chunks they reference (via="cross_ref", rerank_score=None).

        Linked chunks inherit nothing from their source except ``payload["_link"]`` =
        {"from": src_chunk_id, "kind": kind, "label": label}; order = source hit rank, then link kind.
        """
        if not hits or max_extra <= 0:
            return list(hits)
        have = {h.chunk_id for h in hits}
        order = {h.chunk_id: i for i, h in enumerate(hits)}
        links = [l for l in self.links_for(list(have), CROSS_REF_KINDS) if l[1] not in have]
        links.sort(key=lambda l: (order[l[0]], _KIND_PRIORITY.get(l[2], 99)))
        picked: dict[str, tuple] = {}
        for l in links:
            if l[1] not in picked:
                picked[l[1]] = l
            if len(picked) >= max_extra:
                break
        rows = self._fetch(list(picked))
        extra = []
        for dst, (src, _, kind, label) in picked.items():
            h = rows.get(dst)
            if h is None:
                continue
            h.via = "cross_ref"
            h.payload = {**h.payload, "_link": {"from": src, "kind": kind, "label": label}}
            extra.append(h)
        return list(hits) + extra

    def figures_for(self, query: str, hits: list[Hit], k: int = 3, *, min_score: float | None = None,
                    session_id: Any = None, message_id: int | None = None) -> list[Hit]:
        """Best figure chunks for the answer: figures among ``hits``, figures linked from the hits
        (same section + page, or "Figure X-Y" mentions), and a figure-only hybrid search; all reranked
        against ``query`` and filtered by ``min_score`` (default settings.figure_min_score)."""
        s = self.settings
        floor = s.figure_min_score if min_score is None else min_score
        cands: dict[str, Hit] = {h.chunk_id: replace(h) for h in hits if h.chunk_type == "figure"}
        linked = [l[1] for l in self.links_for([h.chunk_id for h in hits], ("figure", "mentions_figure"))]
        for cid, h in self._fetch([c for c in linked if c not in cands]).items():
            h.via = "cross_ref"
            cands[cid] = h
        saved = dict(self.last_timings)
        # figure-only search; rerank just a few candidates (reranking is the expensive stage on CPU)
        for h in self.search(query, k=k, chunk_types=["figure"], session_id=session_id, message_id=message_id,
                             rerank_top_n=max(k, 6)):
            if h.chunk_id not in cands:
                h.via = "figure_search"
                cands[h.chunk_id] = h
        self.last_timings = {**saved, "figure_search_ms": self.last_timings.get("total_ms", 0.0)}
        pool = list(cands.values())
        self._rerank(query, pool, session_id, message_id)
        pool = [h for h in pool if (h.rerank_score or 0.0) >= floor]
        pool.sort(key=lambda h: h.rerank_score or 0.0, reverse=True)
        return pool[:k]


    # ------------------------------------------------------------------ coverage / section text
    def _same_section_text(self, hit: Hit, exclude: set[str]) -> Hit | None:
        """Next text/table chunk of the same section (or one of its subsections) after ``hit`` - e.g. the
        "atomic packing factor ... 0.68" passage that follows the BCC introduction."""
        if not hit.section_path:
            return None
        n = len(hit.section_path)
        with db.get_pool().connection() as conn:
            row = conn.execute("SELECT chunk_index FROM chunks WHERE chunk_id = %s", (hit.chunk_id,)).fetchone()
            if row is None or row[0] is None:
                return None
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM chunks WHERE doc_id = %s AND chunk_type IN ('text', 'table') "
                "AND section_path[1:%s] = %s AND chunk_id <> ALL(%s) AND chunk_index > %s "
                "ORDER BY chunk_index LIMIT 1",
                (hit.doc_id, n, hit.section_path, list(exclude), row[0])).fetchall()
        return _hit_from_row(rows[0]) if rows else None

    def coverage_for(self, sub_queries: list[str], *, per_query: int = 2, min_score: float = 0.3,
                     session_id: Any = None, message_id: int | None = None) -> list[Hit]:
        """Guarantee every sub-query (one per compared entity / connected topic) contributes its own evidence:
        for each, the best text/table chunk reranked against the SUB-query (so "BCC" is not drowned by the FCC
        passages that best match the whole question) plus the next passage of the same section (or the second
        hit). Hits carry via="sub_query" and payload["_sub_query"]; order = sub-query order, deduplicated."""
        out: list[Hit] = []
        seen: set[str] = set()
        saved = dict(self.last_timings)
        for sq in [q.strip() for q in sub_queries if q and q.strip()]:
            hits = self.search(sq, k=3, chunk_types=["text", "table"], session_id=session_id,
                               message_id=message_id, rerank_top_n=6)
            hits = [h for h in hits if (h.rerank_score or 0.0) >= min_score]
            if not hits:
                continue
            best = hits[0]
            picked = [best]
            if per_query > 1:
                nxt = self._same_section_text(best, seen | {best.chunk_id})
                if nxt is None and len(hits) > 1 and (hits[1].rerank_score or 0.0) >= 0.5 * (best.rerank_score or 0):
                    nxt = hits[1]
                if nxt is not None:
                    picked.append(nxt)
            for h in picked[:per_query]:
                if h.chunk_id in seen:
                    continue
                seen.add(h.chunk_id)
                h.via = "sub_query"
                h.payload = {**h.payload, "_sub_query": sq}
                out.append(h)
        self.last_timings = saved
        return out

    def section_text_for(self, chunk_ids: list[str], query: str | None = None, k: int = 2,
                         session_id: Any = None, message_id: int | None = None) -> list[Hit]:
        """Text chunks of the same section that explain the given figure/table chunks (chunk_links: figure ->
        text ``context``, text -> figure ``figure``/``mentions_figure``, same for tables). Reranked against
        ``query`` if given; via="section_text", payload["_link"] = {"from": figure chunk, "kind": ...}."""
        if not chunk_ids:
            return []
        with db.get_pool().connection() as conn:
            rows = conn.execute(
                "SELECT dst_chunk_id, src_chunk_id, kind FROM chunk_links WHERE src_chunk_id = ANY(%s) AND kind = 'context' "
                "UNION ALL SELECT src_chunk_id, dst_chunk_id, kind FROM chunk_links WHERE dst_chunk_id = ANY(%s) "
                "AND kind IN ('figure', 'table', 'mentions_figure', 'mentions_table')",
                (chunk_ids, chunk_ids)).fetchall()
        order = {c: i for i, c in enumerate(chunk_ids)}
        link: dict[str, tuple[str, str]] = {}
        for text_id, fig_id, kind in sorted(rows, key=lambda r: order.get(r[1], 99)):
            if text_id not in order:
                link.setdefault(text_id, (fig_id, kind))
        hits = [h for h in self._fetch(list(link)).values() if h.chunk_type in ("text", "table")]
        if query and hits:
            self._rerank(query, hits, session_id, message_id)
            hits.sort(key=lambda h: (order.get(link[h.chunk_id][0], 99), -(h.rerank_score or 0.0)))
        else:
            hits.sort(key=lambda h: order.get(link[h.chunk_id][0], 99))
        for h in hits:
            h.via = "section_text"
            h.payload = {**h.payload, "_link": {"from": link[h.chunk_id][0], "kind": link[h.chunk_id][1]}}
        return hits[:k]

    def figure_by_number(self, label: str) -> list[Hit]:
        """Figure/table chunks whose number matches ``label`` ("Figure 2-7", "fig 2.7", "Table 3-1")."""
        m = re.search(r"(fig(?:ure)?|table)\.?\s*(\d+)\s*[-.\u2013]?\s*(\d+)?", label, re.I)
        if not m:
            return []
        kind = "table" if m.group(1).lower().startswith("table") else "figure"
        num = m.group(2) + (f"[-.\u2013]{m.group(3)}" if m.group(3) else "")
        with db.get_pool().connection() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM chunks WHERE chunk_type = %s AND "
                f"COALESCE(payload->%s->>'{kind}_number', '') ~* %s ORDER BY chunk_index LIMIT 2",
                (kind, kind, rf"(^|\D){num}($|\D)")).fetchall()
        return [_hit_from_row(r) for r in rows]


_retriever: Retriever | None = None
_rlock = threading.Lock()


def get_retriever() -> Retriever:
    """Process-wide singleton."""
    global _retriever
    with _rlock:
        if _retriever is None:
            _retriever = Retriever()
        return _retriever
