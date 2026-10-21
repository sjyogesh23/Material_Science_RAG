"""Retrieval latency / quality / memory benchmark (run while the chat model is loaded in Ollama).

    .venv/Scripts/python scripts/bench_retrieval.py [--top 5] [--runs 2] [query ...]

Prints per-query top hits (rerank score, chunk type, page, header), per-stage timings, the Ollama placement of
each model (`ollama ps`), and the process' peak RSS / whether torch was imported.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUERIES = ["What is Bragg's law?", "difference between FCC and BCC", "how does creep happen at high temperature",
           "show me the stress strain curve", "What is the capital of France?", "What is the GDP of India",
           "explain Fick's first law of diffusion"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="*")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--runs", type=int, default=2, help="timed passes over the query set (the best is reported)")
    args = ap.parse_args()

    import psutil

    from app import db
    from app.config import get_settings
    from app.retrieval.hybrid import Retriever

    proc = psutil.Process()
    db.init_db()
    s = get_settings()
    print(f"embed={s.embed_backend}:{s.ollama_embed_model if s.embed_backend == 'ollama' else s.embed_model}  "
          f"rerank={s.rerank_backend}:{s.ollama_rerank_model if s.rerank_backend == 'ollama' else s.rerank_model}  "
          f"top_n={s.rerank_top_n} (cpu {s.rerank_top_n_cpu}) max_len={s.rerank_max_length}  "
          f"ooc={s.ooc_threshold}/{s.ooc_soft_threshold}")
    r = Retriever()
    t0 = time.perf_counter()
    warm = r.warmup()
    print(f"warmup {time.perf_counter() - t0:.1f}s {({k: round(v, 2) for k, v in warm.items()})}  "
          f"rss={proc.memory_info().rss / 1e9:.2f} GB")
    if s.rerank_backend == "ollama" or s.embed_backend == "ollama":
        from app.retrieval.embedder import ollama_client

        for m in ollama_client(s).ps().models:
            share = (m.size_vram or 0) / max(1, m.size or 1)
            print(f"  ollama ps: {m.name:34s} {m.size / 1e9:4.1f} GB  {share:4.0%} GPU")

    best: dict[str, dict] = {}
    for run in range(args.runs):
        for q in args.queries or QUERIES:
            hits = r.search(q, k=args.top)
            t = dict(r.last_timings)
            if q not in best or t["total_ms"] < best[q]["t"]["total_ms"]:
                best[q] = {"t": t, "hits": hits}
    print()
    for q, b in best.items():
        t, hits = b["t"], b["hits"]
        mx = max((h.rerank_score or 0) for h in hits) if hits else 0.0
        verdict = "OOC" if mx < s.ooc_threshold else "in"
        print(f"Q: {q}   [max={mx:.4f} -> {verdict}]  total={t['total_ms']:.0f} ms  (embed {t['embed_ms']:.0f}, "
              f"dense {t['dense_ms']:.0f}, bm25 {t['bm25_ms']:.0f}, rerank {t['rerank_ms']:.0f})")
        for h in hits:
            pg = h.pages[0] if h.pages else "-"
            print(f"    {h.rerank_score:.4f}  {h.chunk_type:6s} p{pg:<4} {h.header[:70]}")
    print(f"\npeak RSS {proc.memory_info().peak_wset / 1e9:.2f} GB, current {proc.memory_info().rss / 1e9:.2f} GB, "
          f"torch imported: {'torch' in sys.modules}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
