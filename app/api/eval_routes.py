"""Evaluation API (/api/eval/...): runs, scenario x metric matrix, run comparison, per-item drill-down, and
background jobs that launch ``python -m eval.run`` / ``python -m eval.judge`` as separate processes.

Mounted from app/api/main.py with ``app.include_router(eval_router)``; the page itself is GET /eval.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import ROOT

log = logging.getLogger("newrag.eval")
router = APIRouter(prefix="/api/eval", tags=["evaluation"])
LOG_DIR = ROOT / "logs"

_jobs: dict[int, dict[str, Any]] = {}       # run_id -> {proc, kind, log, started}
_jobs_lock = threading.Lock()


def _edb():
    from eval import db as edb
    edb.ensure_schema()
    return edb


def _bank() -> dict[str, dict]:
    from eval import questions
    return {it["id"]: it for it in questions.load()}


def _run_or_404(run_id: int) -> dict:
    run = _edb().get_run(run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return run


def _job_state(run_id: int) -> dict | None:
    with _jobs_lock:
        j = _jobs.get(run_id)
        if not j:
            return None
        rc = j["proc"].poll()
        return {"kind": j["kind"], "running": rc is None, "returncode": rc, "log": str(j["log"]),
                "started": j["started"]}


def _any_job_running() -> int | None:
    with _jobs_lock:
        for rid, j in _jobs.items():
            if j["proc"].poll() is None:
                return rid
    return None


def _spawn(run_id: int, kind: str, args: list[str]) -> dict:
    LOG_DIR.mkdir(exist_ok=True)
    logf = LOG_DIR / f"eval_{kind}_{run_id}.log"
    fh = open(logf, "a", encoding="utf-8")  # noqa: SIM115 - handed to the child process
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen([sys.executable, "-u", "-m", *args], cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                            creationflags=flags)
    with _jobs_lock:
        _jobs[run_id] = {"proc": proc, "kind": kind, "log": logf, "started": time.time(), "fh": fh}
    return {"run_id": run_id, "kind": kind, "pid": proc.pid, "log": str(logf)}


# ----------------------------------------------------------------------------- meta
@router.get("/meta")
def meta():
    from eval import metrics as M
    from eval import questions
    from eval.judge import CANDIDATES
    items = questions.load()
    return {"metrics": M.METRICS, "rates": M.RATES, "scenarios": questions.SCENARIOS, "bank": questions.stats(items),
            "judge_candidates": [c[0] for c in CANDIDATES], "active_job": _any_job_running()}


@router.get("/questions")
def question_bank(scenario: str | None = None):
    items = list(_bank().values())
    return [it for it in items if not scenario or it["scenario"] == scenario]


# ----------------------------------------------------------------------------- runs
@router.get("/runs")
def list_runs():
    from eval import metrics as M
    edb = _edb()
    runs = edb.rows("SELECT run_id, label, status, created_at, started_at, finished_at, n_items, bank_hash, progress, "
                    "judge->>'model' AS judge_model, (judge->'calibration'->>'agreement')::float AS judge_agreement, "
                    "config->'settings'->>'chat_model' AS chat_model, notes FROM eval_runs ORDER BY run_id DESC")
    if runs:
        ids = [r["run_id"] for r in runs]
        res = edb.rows("SELECT run_id, item_id, scenario, status, judge, det_metrics, latency_ms, usage "
                       "FROM eval_results WHERE run_id = ANY(%s)", (ids,))
        by: dict[int, list] = {}
        for r in res:
            by.setdefault(r["run_id"], []).append(r)
        bank = _bank()
        for run in runs:
            agg = M.aggregate(by.get(run["run_id"], []), bank, extras=False)
            run["summary"] = {"overall": agg["overall"], "ooc_f1": agg["ooc"]["f1"], "latency_p50": agg["latency_ms"]["p50"]}
            run["job"] = _job_state(run["run_id"])
    return runs


@router.get("/runs/{run_id}")
def get_run(run_id: int):
    from eval import metrics as M
    run = _run_or_404(run_id)
    agg = M.aggregate(_edb().results(run_id), _bank())
    run["job"] = _job_state(run_id)
    return {"run": run, "aggregate": agg}


@router.get("/runs/{run_id}/confusion")
def run_confusion(run_id: int, matrix: str | None = None, cell: str | None = None):
    """Confusion matrices (ooc / images / table) with per-class and micro / macro / weighted P, R, F1.
    ``?matrix=ooc&cell=fp`` returns only that cell's items (with question, answer preview, score, pass)."""
    from eval import metrics as M
    _run_or_404(run_id)
    res = _edb().results(run_id)
    conf = M.confusion(res, _bank())
    if matrix is None:
        return conf
    if matrix not in conf:
        raise HTTPException(404, f"Unknown matrix {matrix!r}; one of {sorted(conf)}")
    if cell is None:
        return conf[matrix]
    if cell not in ("tp", "fp", "fn", "tn"):
        raise HTTPException(422, "cell must be tp, fp, fn or tn")
    ids = {c["item_id"] for c in conf[matrix]["cells"][cell]}
    return [{"item_id": r["item_id"], "scenario": r["scenario"], "question": r["question"],
             "answer_preview": (r.get("answer") or "")[:220], "n_images": len(r.get("images") or []),
             "score": M.final_score(r), "pass": M.item_pass(r), "latency_ms": r.get("latency_ms")}
            for r in res if r["item_id"] in ids]


@router.get("/runs/{run_id}/aggregates")
def run_aggregates(run_id: int):
    """Micro (item-weighted) vs macro (scenario-weighted) means of score, pass rate and every metric."""
    from eval import metrics as M
    _run_or_404(run_id)
    return M.averages(_edb().results(run_id))


@router.get("/runs/{run_id}/items")
def run_items(run_id: int, scenario: str | None = None, failed: bool = False):
    from eval import metrics as M
    _run_or_404(run_id)
    bank = _bank()
    out = []
    for r in _edb().results(run_id):
        if scenario and r["scenario"] != scenario:
            continue
        sc = M.scores_for(r)
        passed = M.item_pass(r)
        if failed and passed:
            continue
        it = bank.get(r["item_id"], {})
        out.append({"item_id": r["item_id"], "scenario": r["scenario"], "status": r["status"],
                    "question": r["question"], "expected": it.get("expected"), "difficulty": it.get("difficulty"),
                    "answer_preview": (r.get("answer") or "")[:220], "out_of_context": r["out_of_context"],
                    "scores": sc, "score": M.final_score(r), "pass": passed, "judged": M.is_judged(r),
                    "latency_ms": r.get("latency_ms"), "ttft_ms": r.get("ttft_ms"),
                    "tokens": (r.get("usage") or {}).get("total"), "n_images": len(r.get("images") or []),
                    "det": r.get("det_metrics") or {}})
    return out


@router.get("/runs/{run_id}/items/{item_id}")
def run_item(run_id: int, item_id: str):
    from eval import metrics as M
    edb = _edb()
    rows = edb.rows("SELECT * FROM eval_results WHERE run_id=%s AND item_id=%s", (run_id, item_id))
    if not rows:
        raise HTTPException(404, "Result not found")
    r = rows[0]
    ids = [s.get("chunk_id") for s in r.get("sources") or []]
    texts = edb.chunk_texts([i for i in ids if i])
    for s in r.get("sources") or []:
        s["content"] = (texts.get(s.get("chunk_id")) or {}).get("content")
    r["item"] = _bank().get(item_id)
    r["scores"] = M.scores_for(r)
    r["score"] = M.final_score(r)
    r["pass"] = M.item_pass(r)
    return r


@router.get("/compare")
def compare(a: int, b: int):
    from eval import metrics as M
    bank = _bank()
    ra, rb = _run_or_404(a), _run_or_404(b)
    aa = M.aggregate(_edb().results(a), bank)
    ab = M.aggregate(_edb().results(b), bank)
    return {"a": {"run": ra, "aggregate": aa}, "b": {"run": rb, "aggregate": ab}, "delta": M.compare(aa, ab)}


class RunIn(BaseModel):
    label: str = Field(min_length=1, max_length=80)
    scenarios: list[str] | None = None
    limit: int | None = Field(default=None, ge=1, le=1000)
    judge: bool = False
    notes: str | None = Field(default=None, max_length=500)


@router.post("/runs")
def start_run(body: RunIn):
    from eval import questions
    busy = _any_job_running()
    if busy is not None:
        raise HTTPException(409, f"An evaluation job is already running (run {busy}).")
    items = questions.select(questions.load(), body.scenarios or None, None, body.limit)
    if not items:
        raise HTTPException(422, "No questions match the selection.")
    edb = _edb()
    cfg = {"scenarios": body.scenarios or None, "limit": body.limit, "launched_from": "ui", "judge": body.judge}
    run_id = edb.create_run(" ".join(body.label.split()), cfg, questions.bank_hash(questions.load()), len(items))
    if body.notes:
        edb.update_run(run_id, notes=body.notes)
    args = ["eval.run", "--run-id", str(run_id)] + (["--judge"] if body.judge else [])
    return _spawn(run_id, "run", args)


@router.post("/runs/{run_id}/judge")
def start_judge(run_id: int, rejudge: bool = False, model: str | None = None):
    _run_or_404(run_id)
    busy = _any_job_running()
    if busy is not None:
        raise HTTPException(409, f"An evaluation job is already running (run {busy}).")
    args = ["eval.judge", "--run", str(run_id)] + (["--rejudge"] if rejudge else []) + (["--model", model] if model else [])
    return _spawn(run_id, "judge", args)


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: int):
    _run_or_404(run_id)
    _edb().update_run(run_id, status="cancelling")
    return {"ok": True}


@router.get("/runs/{run_id}/log")
def run_log(run_id: int, tail: int = 60):
    tail = max(1, min(int(tail), 1000))
    files = sorted(LOG_DIR.glob(f"eval_*_{run_id}.log"), key=lambda p: p.stat().st_mtime)
    if not files:
        return {"lines": []}
    lines = Path(files[-1]).read_text(encoding="utf-8", errors="replace").splitlines()
    return {"file": str(files[-1]), "lines": lines[-tail:]}


@router.delete("/runs/{run_id}")
def delete_run(run_id: int):
    if (j := _job_state(run_id)) and j["running"]:
        raise HTTPException(409, "Run is in progress; cancel it first.")
    n = _edb().execute("DELETE FROM eval_runs WHERE run_id=%s", (run_id,))
    if not n:
        raise HTTPException(404, "Run not found")
    return {"ok": True}
