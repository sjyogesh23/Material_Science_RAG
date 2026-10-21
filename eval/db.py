"""Postgres helpers for the evaluation tables (sync pool from app.db; schema applied idempotently)."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from app import db

SCHEMA = Path(__file__).with_name("schema.sql")
_LOCK_ID = 7_142_077            # pg_advisory_lock key for the eval schema
_lock = threading.Lock()
_ready = False


def ensure_schema() -> None:
    global _ready
    if _ready:
        return
    with _lock:
        if _ready:
            return
        with db.get_pool().connection() as conn:
            conn.execute("SELECT pg_advisory_lock(%s)", (_LOCK_ID,))
            try:
                conn.execute(SCHEMA.read_text(encoding="utf-8"))
                conn.commit()
            finally:
                conn.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_ID,))
                conn.commit()
        _ready = True


def jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, uuid.UUID):
        return str(v)
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, Path):
        return str(v)
    return v


def execute(sql: str, params: tuple | list = ()) -> int:
    ensure_schema()
    with db.get_pool().connection() as conn:
        return conn.execute(sql, params).rowcount


def one(sql: str, params: tuple | list = ()) -> tuple | None:
    ensure_schema()
    with db.get_pool().connection() as conn:
        return conn.execute(sql, params).fetchone()


def rows(sql: str, params: tuple | list = ()) -> list[dict]:
    ensure_schema()
    with db.get_pool().connection() as conn:
        cur = conn.execute(sql, params)
        cols = [c.name for c in cur.description]
        return [jsonable(dict(zip(cols, r))) for r in cur.fetchall()]


def dumps(v: Any) -> str:
    return json.dumps(jsonable(v), ensure_ascii=False, default=str)


# ----------------------------------------------------------------------------- runs
def create_run(label: str, config: dict, bank_hash: str, n_items: int) -> int:
    row = one("INSERT INTO eval_runs(label, status, config, bank_hash, n_items, progress) "
              "VALUES (%s,'created',%s::jsonb,%s,%s,%s::jsonb) RETURNING run_id",
              (label, dumps(config), bank_hash, n_items, dumps({"phase": "created", "done": 0, "total": n_items})))
    return int(row[0])


def update_run(run_id: int, **fields: Any) -> None:
    sets, params = [], []
    for k, v in fields.items():
        if k in ("config", "progress", "judge"):
            sets.append(f"{k} = COALESCE({k}, '{{}}'::jsonb) || %s::jsonb")
            params.append(dumps(v))
        elif k in ("started_at", "finished_at") and v == "now":
            sets.append(f"{k} = now()")
        else:
            sets.append(f"{k} = %s")
            params.append(v)
    if sets:
        execute(f"UPDATE eval_runs SET {', '.join(sets)} WHERE run_id = %s", (*params, run_id))


def get_run(run_id: int) -> dict | None:
    r = rows("SELECT * FROM eval_runs WHERE run_id = %s", (run_id,))
    return r[0] if r else None


def resolve_run(ref: str | int) -> dict | None:
    """Run id, or the newest run with that label."""
    s = str(ref)
    if s.isdigit():
        return get_run(int(s))
    r = rows("SELECT * FROM eval_runs WHERE label = %s ORDER BY run_id DESC LIMIT 1", (s,))
    return r[0] if r else None


def upsert_items(items: list[dict], bank_hash: str) -> None:
    ensure_schema()
    with db.get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO eval_items(item_id, scenario, payload, bank_hash, updated_at) VALUES (%s,%s,%s::jsonb,%s,now()) "
                "ON CONFLICT (item_id) DO UPDATE SET scenario=EXCLUDED.scenario, payload=EXCLUDED.payload, "
                "bank_hash=EXCLUDED.bank_hash, updated_at=now()",
                [(it["id"], it["scenario"], dumps(it), bank_hash) for it in items])


def save_result(run_id: int, rec: dict) -> None:
    cols = ["run_id", "item_id", "scenario", "status", "session_id", "question", "answer", "turns", "decision",
            "sources", "images", "retrieved", "out_of_context", "uncited", "timings", "usage", "latency_ms",
            "ttft_ms", "error", "det_metrics"]
    jcols = {"turns", "decision", "sources", "images", "retrieved", "timings", "usage", "det_metrics"}
    vals = []
    for c in cols:
        v = run_id if c == "run_id" else rec.get(c)
        vals.append(dumps(v if v is not None else ({} if c in ("timings", "usage", "det_metrics") else
                                                    [] if c in ("turns", "sources", "images", "retrieved") else None))
                    if c in jcols else v)
    ph = ", ".join(f"%s::jsonb" if c in jcols else "%s" for c in cols)
    upd = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols[2:])
    execute(f"INSERT INTO eval_results({', '.join(cols)}) VALUES ({ph}) "
            f"ON CONFLICT (run_id, item_id) DO UPDATE SET {upd}, judge='{{}}'::jsonb, judge_model=NULL, judged_at=NULL",
            vals)


def save_judgement(run_id: int, item_id: str, judge: dict, model: str) -> None:
    execute("UPDATE eval_results SET judge=%s::jsonb, judge_model=%s, judged_at=now() WHERE run_id=%s AND item_id=%s",
            (dumps(judge), model, run_id, item_id))


def results(run_id: int) -> list[dict]:
    return rows("SELECT * FROM eval_results WHERE run_id=%s ORDER BY scenario, item_id", (run_id,))


def chunk_texts(chunk_ids: list[str]) -> dict[str, dict]:
    if not chunk_ids:
        return {}
    out = {}
    for r in rows("SELECT chunk_id, chunk_type, header, content, section_path, pages FROM chunks WHERE chunk_id = ANY(%s)",
                  (list(chunk_ids),)):
        out[r["chunk_id"]] = r
    return out
