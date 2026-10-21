"""Run the question bank through the real pipeline (app.rag.pipeline.answer) and store the results.

    python -m eval.run --label baseline [--scenarios a,b] [--ids fd-01,fe-01] [--limit N] [--notes "..."]
    python -m eval.run --run-id 7        # fill a run row pre-created by the API ("Start run" button)

* one fresh chat session per item (multi-turn items share one session); sessions are archived afterwards so
  they do not clutter the chat sidebar, and their memory files are deleted
* every passage the retriever returns is recorded (a recording proxy around the real Retriever) so retrieval
  hit@k can be scored against the gold chunks
* results -> eval_results (+ deterministic metrics); judging is a separate step: ``python -m eval.judge``
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
import uuid
from typing import Any

log = logging.getLogger("eval.run")
TURN_TIMEOUT_S = float(os.getenv("EVAL_TURN_TIMEOUT_S", "420"))


# ----------------------------------------------------------------------------- recording retriever
class RecordingRetriever:
    """Wraps the real Retriever and logs every hit list it returns (stage = search | cross_ref | figures)."""

    def __init__(self, inner: Any):
        self._inner = inner
        self.log: list[dict] = []

    def reset(self) -> None:
        self.log = []

    def _record(self, stage: str, hits: list) -> None:
        for h in hits or []:
            self.log.append({"stage": stage, "chunk_id": getattr(h, "chunk_id", None),
                             "chunk_type": getattr(h, "chunk_type", None), "pages": list(getattr(h, "pages", []) or []),
                             "rerank_score": getattr(h, "rerank_score", None), "via": getattr(h, "via", None),
                             "header": (getattr(h, "header", "") or "")[:160]})

    def search(self, *a, **kw):
        hits = self._inner.search(*a, **kw)
        self._record("search", hits)
        return hits

    def expand_cross_refs(self, *a, **kw):
        hits = self._inner.expand_cross_refs(*a, **kw)
        self._record("cross_ref", hits)
        return hits

    def figures_for(self, *a, **kw):
        hits = self._inner.figures_for(*a, **kw)
        self._record("figures", hits)
        return hits

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# ----------------------------------------------------------------------------- one turn / one item
def _slim_source(s: dict) -> dict:
    return {k: s.get(k) for k in ("n", "chunk_id", "chunk_type", "header", "section_path", "pages", "snippet", "score",
                                  "figure")}


async def run_turn(session_id: str, text: str) -> dict:
    from app.rag.pipeline import answer
    t0 = time.perf_counter()
    rec: dict[str, Any] = {"question": text, "answer": "", "decision": None, "sources": [], "images": [],
                           "usage": {}, "timings": {}, "out_of_context": False, "uncited": False, "error": None,
                           "ttft_ms": None}
    streamed: list[str] = []
    agen = answer(session_id, text)
    try:
        async with asyncio.timeout(TURN_TIMEOUT_S):
            async for ev in agen:
                name, data = ev.get("event"), ev.get("data")
                if name == "token":
                    if rec["ttft_ms"] is None:
                        rec["ttft_ms"] = int((time.perf_counter() - t0) * 1000)
                    streamed.append((data or {}).get("t", ""))
                elif name == "decision":
                    rec["decision"] = data
                elif name == "answer":
                    rec["answer"] = (data or {}).get("content", "")
                    rec["uncited"] = bool((data or {}).get("uncited"))
                elif name == "sources":
                    rec["sources"] = [_slim_source(s) for s in (data or [])]
                elif name == "images":
                    rec["images"] = data or []
                elif name == "usage":
                    rec["usage"] = {k: v for k, v in (data or {}).items() if k != "timings_ms"}
                    rec["timings"] = (data or {}).get("timings_ms") or {}
                elif name == "error":
                    rec["error"] = (data or {}).get("message")
                elif name == "done":
                    rec["out_of_context"] = bool((data or {}).get("out_of_context"))
                    if not rec["answer"]:
                        rec["answer"] = (data or {}).get("content") or "".join(streamed)
    except TimeoutError:
        rec["error"] = f"timeout after {TURN_TIMEOUT_S:.0f} s"
    finally:
        await agen.aclose()
    if not rec["answer"]:
        rec["answer"] = "".join(streamed)
    rec["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    return rec


def _archive_session(sid: str, item_id: str) -> None:
    from eval import db as edb
    try:
        edb.execute("UPDATE sessions SET archived = true, title = %s WHERE session_id = %s",
                    (f"[eval] {item_id}", sid))
    except Exception as e:  # noqa: BLE001
        log.warning("could not archive session %s: %s", sid, e)
    try:
        from app.memory.session_memory import SessionMemory
        SessionMemory.path_for(sid).unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


async def run_item(item: dict, recorder: RecordingRetriever | None, final_k: int) -> dict:
    from eval.metrics import item_metrics
    from eval.questions import user_turns
    sid = str(uuid.uuid4())
    turns = []
    for text in user_turns(item):
        if recorder:
            recorder.reset()
        t = await run_turn(sid, text)
        t["retrieved"] = list(recorder.log) if recorder else []
        turns.append(t)
    last = turns[-1]
    rec = {
        "item_id": item["id"], "scenario": item["scenario"], "session_id": sid,
        "status": "error" if any(t["error"] for t in turns) else "ok",
        "question": last["question"], "answer": last["answer"], "decision": last["decision"],
        "sources": last["sources"], "images": last["images"], "retrieved": last["retrieved"],
        "out_of_context": last["out_of_context"], "uncited": last["uncited"], "timings": last["timings"],
        "usage": last["usage"], "latency_ms": last["latency_ms"], "ttft_ms": last["ttft_ms"],
        "error": "; ".join(t["error"] for t in turns if t["error"]) or None,
        "turns": [{k: v for k, v in t.items() if k != "retrieved"} | {"n_retrieved": len(t["retrieved"])}
                  for t in turns],
    }
    rec["det_metrics"] = item_metrics(item, rec, final_k=final_k)
    _archive_session(sid, item["id"])
    return rec


# ----------------------------------------------------------------------------- run
def settings_snapshot() -> dict:
    from app.config import get_settings
    s = get_settings()
    keys = ["chat_model", "router_model", "num_ctx", "temperature", "max_answer_tokens", "embed_backend",
            "ollama_embed_model", "rerank_backend", "ollama_rerank_model", "top_k_dense", "top_k_bm25", "rrf_k",
            "rerank_top_n", "final_k", "ooc_threshold", "ooc_soft_threshold", "grounding_check",
            "memory_budget_tokens"]
    return {k: getattr(s, k, None) for k in keys}


async def run(label: str | None = None, *, run_id: int | None = None, scenarios: list[str] | None = None,
              ids: list[str] | None = None, limit: int | None = None, notes: str | None = None,
              record_retrieval: bool = True) -> int:
    from app import db
    from app.config import reload_settings
    from app.rag import pipeline
    from eval import db as edb
    from eval import questions

    await asyncio.to_thread(db.init_db)
    s = reload_settings()
    await asyncio.to_thread(edb.ensure_schema)
    bank = questions.load()
    h = questions.bank_hash(bank)
    edb.upsert_items(bank, h)
    if run_id is not None:
        row = edb.get_run(run_id)
        if row is None:
            raise SystemExit(f"run {run_id} not found")
        cfg = row.get("config") or {}
        scenarios = scenarios or cfg.get("scenarios")
        ids = ids or cfg.get("ids")
        limit = limit or cfg.get("limit")
        label = row["label"]
    chosen = questions.select(bank, scenarios, ids, limit)
    if not chosen:
        raise SystemExit("no items selected")
    cfg = {"settings": settings_snapshot(), "scenarios": scenarios, "ids": ids, "limit": limit}
    if run_id is None:
        run_id = edb.create_run(label or "run", cfg, h, len(chosen))
    edb.update_run(run_id, status="running", started_at="now", bank_hash=h, n_items=len(chosen), config=cfg,
                   progress={"phase": "answering", "done": 0, "total": len(chosen), "pid": os.getpid()})
    if notes:
        edb.update_run(run_id, notes=notes)

    recorder = None
    if record_retrieval:
        from app.retrieval.hybrid import get_retriever
        recorder = RecordingRetriever(get_retriever())
        pipeline.set_retriever_factory(lambda: recorder)
    print(f"[eval] run {run_id} '{label}': {len(chosen)} items (bank {h})", flush=True)
    t_run = time.perf_counter()
    status = "answered"
    try:
        for i, item in enumerate(chosen, 1):
            st = edb.one("SELECT status FROM eval_runs WHERE run_id=%s", (run_id,))
            if st and st[0] in ("cancelling", "cancelled"):
                status = "cancelled"
                print("[eval] cancelled", flush=True)
                break
            edb.update_run(run_id, progress={"current": item["id"], "done": i - 1})
            t0 = time.perf_counter()
            try:
                rec = await run_item(item, recorder, s.final_k)
            except Exception as e:  # noqa: BLE001 - keep going; record the failure
                log.exception("item %s failed", item["id"])
                rec = {"item_id": item["id"], "scenario": item["scenario"], "status": "error",
                       "question": questions.user_turns(item)[-1], "answer": "", "error": f"{type(e).__name__}: {e}"}
                from eval.metrics import item_metrics
                rec["det_metrics"] = item_metrics(item, rec, final_k=s.final_k)
            edb.save_result(run_id, rec)
            dm = rec.get("det_metrics") or {}
            flag = "OK " if dm.get("refusal_correctness") == 5 else "BAD"
            ans = (rec.get("answer") or rec.get("error") or "").replace("\n", " ")[:90]
            print(f"[eval] {i:3d}/{len(chosen)} {item['id']:7s} {flag} {time.perf_counter() - t0:6.1f}s  {ans}",
                  flush=True)
            edb.update_run(run_id, progress={"done": i})
    except (KeyboardInterrupt, asyncio.CancelledError):
        status = "cancelled"
        raise
    except Exception as e:  # noqa: BLE001
        status = "failed"
        edb.update_run(run_id, notes=f"failed: {type(e).__name__}: {e}"[:1000])
        raise
    finally:
        pipeline.set_retriever_factory(None)
        edb.update_run(run_id, status=status, finished_at="now",
                       progress={"phase": status, "elapsed_s": round(time.perf_counter() - t_run, 1)})
    print(f"[eval] run {run_id} {status} in {time.perf_counter() - t_run:.0f} s. "
          f"Judge it with: python -m eval.judge --run {run_id}", flush=True)
    return run_id


def _csv(v: str | None) -> list[str] | None:
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the NewRAG evaluation question bank through the pipeline.")
    ap.add_argument("--label", help="run label, e.g. baseline")
    ap.add_argument("--run-id", type=int, help="fill an existing (API-created) run row")
    ap.add_argument("--scenarios", help="comma-separated scenario filter")
    ap.add_argument("--ids", help="comma-separated item ids")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--notes")
    ap.add_argument("--judge", action="store_true", help="run eval.judge right after answering")
    args = ap.parse_args(argv)
    if not args.label and args.run_id is None:
        ap.error("--label or --run-id is required")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rid = asyncio.run(run(args.label, run_id=args.run_id, scenarios=_csv(args.scenarios), ids=_csv(args.ids),
                          limit=args.limit, notes=args.notes))
    if args.judge:
        from eval import judge
        return judge.main(["--run", str(rid)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
