"""Retrieval tests: tokenizer + link building (pure), then DB/model integration (skipped if not ingested).

Run:  .venv/Scripts/python -m pytest tests/test_retrieval.py -q
The integration part uses the configured backends - by default the Ollama server (qwen3-embedding on CPU,
qwen3-reranker on GPU/CPU), so the test process itself stays torch-free (~0.3 GB RSS).
"""
from __future__ import annotations

import pytest

from app.ingest import _num, build_links
from app.retrieval.bm25 import BM25Index, tokenize


# ----------------------------------------------------------------------------- tokenizer
def test_tokenizer_keeps_formulas_alloys_numbers_whole():
    toks = tokenize("Cementite Fe3C, Al2O3 and 304L steel at a 0.2% offset")
    for t in ("fe3c", "al2o3", "304l", "0.2"):
        assert t in toks
    assert {"fe", "c", "al", "o"} <= set(toks)          # element tokens emitted for formulas
    assert "cementit" in toks and "steel" in toks       # stemmed
    assert "and" not in toks and "a" not in toks        # stopwords


def test_tokenizer_acronyms_not_split_into_elements():
    toks = tokenize("FCC versus BCC crystals, NaCl")
    assert "fcc" in toks and "bcc" in toks and "f" not in toks and "b" not in toks
    assert {"nacl", "na", "cl"} <= set(toks)
    assert "crystal" in toks


def test_bm25_ranks_keyword_match_first():
    idx = BM25Index()
    idx.build([("a", "text", "Creep at high temperature in metals"),
               ("b", "text", "Bragg's law describes X-ray diffraction"),
               ("c", "figure", "Figure: stress strain curve of a steel")])
    assert idx.search("bragg diffraction")[0][0] == "b"
    assert idx.search("stress-strain curve", chunk_types=["figure"])[0][0] == "c"
    assert idx.search("creep", chunk_types=["figure"]) == []
    assert idx.search("the of and") == []


# ----------------------------------------------------------------------------- settings / helpers
def test_settings_have_backend_and_grounding_fields():
    from app.config import EDITABLE_FIELDS, Settings

    s = Settings()
    assert s.grounding_check is False and "grounding_check" in EDITABLE_FIELDS
    assert s.embed_backend in ("ollama", "sentence_transformers")
    assert s.rerank_backend in ("ollama", "sentence_transformers")
    assert s.ollama_rerank_num_ctx >= s.rerank_max_length


def test_embed_identity_depends_on_backend():
    from app.config import Settings
    from app.retrieval.embedder import embed_identity

    assert embed_identity(Settings(embed_backend="ollama")).startswith("ollama:")
    assert embed_identity(Settings(embed_backend="sentence_transformers")) == Settings().embed_model


def test_truncate_keeps_short_text_and_cuts_long_text():
    from app.retrieval.reranker import _truncate

    assert _truncate("short text", 100) == "short text"
    long = "diffusion " * 2000
    cut = _truncate(long, 50)
    from app.tokens import count_tokens
    assert len(cut) < len(long) and count_tokens(cut) <= 52


# ----------------------------------------------------------------------------- links
def test_number_normalisation():
    assert _num("1-1") == _num("1.1") == _num("(1.1)") == "1.1"


def test_build_links_mentions_and_figure_context():
    base = {"doc_id": "d", "section_path": ["Ch 1", "1.1"], "formulas": None}
    chunks = [
        {**base, "chunk_id": "t1", "chunk_type": "text", "chunk_index": 0, "pages": [3],
         "element_ids": ["p3_e1"], "content": "As shown in Figure 1-2 and Eq. 1.1 ..."},
        {**base, "chunk_id": "f1", "chunk_type": "figure", "chunk_index": 1, "pages": [3],
         "element_ids": ["p3_e2"], "content": "Figure 1-2: x", "figure": {"figure_number": "Figure 1-2"}},
        {**base, "chunk_id": "t2", "chunk_type": "text", "chunk_index": 2, "pages": [4],
         "element_ids": ["p4_e1"], "content": "eq", "formulas": [{"equation_number": "(1.1)"}]},
    ]
    extracted = {"pages": [{"elements": [{"id": "p3_e1", "cross_references": [
        {"kind": "equation", "label": "1.1", "target_id": "p4_e1"}]}]}], "relationships": []}
    links = {(s, d, k) for s, d, k, _ in build_links(chunks, extracted)}
    assert ("t1", "f1", "mentions_figure") in links
    assert ("t1", "t2", "mentions_equation") in links
    assert ("t1", "t2", "equation") in links
    assert ("t1", "f1", "figure") in links and ("f1", "t1", "context") in links
    assert not any(s == d for s, d, _ in links)


# ----------------------------------------------------------------------------- integration
@pytest.fixture(scope="module")
def retriever():
    from app import db
    try:
        db.init_db()
        with db.get_pool().connection() as conn:
            n = conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"database unavailable: {exc}")
    if n == 0:
        pytest.skip("no ingested chunks - run `python -m app.ingest` first")
    from app.retrieval.hybrid import Retriever
    return Retriever()


def test_db_counts(retriever):
    from app import db
    with db.get_pool().connection() as conn:
        assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 375
        assert conn.execute("SELECT count(*) FROM element_chunks").fetchone()[0] > 1000
        assert conn.execute("SELECT count(*) FROM chunk_links").fetchone()[0] > 100
        assert conn.execute("SELECT pdf_path FROM documents").fetchone()[0].endswith(".pdf")


def test_search_in_scope(retriever):
    hits = retriever.search("What is Bragg's law?")
    assert 0 < len(hits) <= retriever.settings.final_k
    assert hits[0].rerank_score is not None and hits[0].rerank_score > 0.5
    assert any("bragg" in (h.content + h.header).lower() for h in hits[:3])
    scores = [h.rerank_score for h in hits]
    assert scores == sorted(scores, reverse=True)
    for k in ("embed_ms", "dense_ms", "bm25_ms", "rerank_ms"):
        assert k in retriever.last_timings


@pytest.mark.parametrize("q", ["What is the capital of France?", "What is the GDP of India"])
def test_search_out_of_scope_scores_low(retriever, q):
    hits = retriever.search(q)
    assert max((h.rerank_score or 0) for h in hits) < retriever.settings.ooc_threshold


@pytest.mark.parametrize("q", ["difference between FCC and BCC", "how does creep happen at high temperature",
                               "explain Fick's first law of diffusion"])
def test_search_in_scope_scores_high(retriever, q):
    hits = retriever.search(q)
    assert hits[0].rerank_score > 0.8


def test_query_embedding_is_unit_1024(retriever):
    import numpy as np

    v = retriever.embedder.encode("What is Bragg's law?")
    assert v.shape == (1024,) and abs(float(np.linalg.norm(v)) - 1.0) < 1e-3


def test_retrieval_latency(retriever):
    retriever.search("What is Bragg's law?")                      # warm
    retriever.search("explain Fick's first law of diffusion")
    assert retriever.last_timings["total_ms"] < 8000             # target ~1 s on GPU reranker, <=~8 s on CPU


def test_chunk_types_filter(retriever):
    hits = retriever.search("stress strain curve", chunk_types=["figure"], k=3)
    assert hits and all(h.chunk_type == "figure" for h in hits)


def test_figures_for(retriever):
    q = "show me the stress strain curve"
    figs = retriever.figures_for(q, retriever.search(q), k=3)
    assert figs and all(f.chunk_type == "figure" for f in figs)
    assert "stress" in figs[0].content.lower()


def test_expand_cross_refs(retriever):
    from app import db
    with db.get_pool().connection() as conn:
        src = conn.execute("SELECT src_chunk_id FROM chunk_links WHERE kind = 'variables' LIMIT 1").fetchone()[0]
    seed = list(retriever._fetch([src]).values())
    out = retriever.expand_cross_refs(seed, max_extra=3)
    extra = [h for h in out if h.via == "cross_ref"]
    assert extra and extra[0].payload["_link"]["from"] == src
    assert len({h.chunk_id for h in out}) == len(out)
