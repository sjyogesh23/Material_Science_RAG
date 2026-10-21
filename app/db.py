"""Embedded Postgres (pgserver) + psycopg connection pools + schema migration.

* ``start_postgres()`` boots (or re-attaches to) the embedded PG 16 server in ``settings.pg_data_dir``.
  Several processes may call it: pgserver re-uses a server that is already running on that directory.
  ``srv.psql()`` is never used (it breaks on paths with spaces) - everything goes through psycopg.
* ``init_db()`` = start + ``CREATE EXTENSION vector`` + ``schema.sql`` + open the sync pool.
  The extension must exist before a pool whose ``configure`` hook calls ``register_vector`` opens.
* ``get_async_pool()`` returns an ``AsyncConnectionPool`` created closed; open it inside the event loop
  with ``await open_async_pool()``. NOTE (Windows): psycopg async needs a *Selector* event loop
  (``asyncio.WindowsSelectorEventLoopPolicy``), not the default Proactor loop.
* ``log_llm_call()`` writes one ``llm_calls`` row; used for the local purposes (embed_query, rerank,
  ingest_embed). Failures to log never break the caller.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import psycopg
from pgvector.psycopg import register_vector, register_vector_async
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from app.config import get_settings

log = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_MIGRATION_LOCK_ID = 7_142_001          # pg_advisory_lock key: serialises concurrent schema runs

_lock = threading.Lock()
_server: Any = None                     # pgserver handle (kept alive for the process lifetime)
_uri: str | None = None
_pool: ConnectionPool | None = None
_async_pool: AsyncConnectionPool | None = None


# ----------------------------------------------------------------------------- server
def start_postgres() -> str:
    """Return a connection URI; boots the embedded server unless DATABASE_URL is set."""
    global _server, _uri
    if _uri:
        return _uri
    s = get_settings()
    if s.database_url:
        _uri = s.database_url
        return _uri
    import pgserver

    s.pg_data_dir.mkdir(parents=True, exist_ok=True)
    _server = pgserver.get_server(str(s.pg_data_dir), cleanup_mode="stop")
    _uri = _server.get_uri()
    return _uri


def _migrate(uri: str) -> None:
    with psycopg.connect(uri, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK_ID,))
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK_ID,))


# ----------------------------------------------------------------------------- pools
def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)
    conn.execute("SET hnsw.ef_search = %s" % int(get_settings().hnsw_ef_search))
    conn.commit()                       # pool requires the connection to be idle after configure


async def _configure_async(conn: psycopg.AsyncConnection) -> None:
    await register_vector_async(conn)
    await conn.execute("SET hnsw.ef_search = %s" % int(get_settings().hnsw_ef_search))
    await conn.commit()


def init_db() -> None:
    """Start Postgres (if needed), create the extension + schema, open the sync pool. Idempotent."""
    global _pool
    with _lock:
        if _pool is not None:
            return
        uri = start_postgres()
        _migrate(uri)
        _pool = ConnectionPool(uri, min_size=1, max_size=10, configure=_configure, open=True,
                               name="newrag")
        _pool.wait(timeout=30)


def get_pool() -> ConnectionPool:
    if _pool is None:
        init_db()
    assert _pool is not None
    return _pool


def peek_pool() -> ConnectionPool | None:
    """The sync pool if already open, else None (never starts anything)."""
    return _pool


def get_async_pool() -> AsyncConnectionPool:
    """Async pool (created closed). Call ``await open_async_pool()`` once inside the event loop."""
    global _async_pool
    if _async_pool is None:
        init_db()                       # make sure server + extension + schema exist
        _async_pool = AsyncConnectionPool(start_postgres(), min_size=1, max_size=10,
                                          configure=_configure_async, open=False, name="newrag-async")
    return _async_pool


async def open_async_pool() -> AsyncConnectionPool:
    pool = get_async_pool()
    await pool.open(wait=True, timeout=30)   # no-op if already open
    return pool


def close_pools() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


# ----------------------------------------------------------------------------- audit helper
def log_llm_call(purpose: str, *, provider: str = "local", model: str = "", prompt_tokens: int = 0,
                 completion_tokens: int = 0, duration_ms: int = 0, ok: bool = True,
                 error: str | None = None, meta: dict | None = None, session_id: Any = None,
                 message_id: int | None = None) -> None:
    """Insert one llm_calls row (best effort - logging must never break retrieval). The call is also added to
    the current per-answer usage scope (``app.llm.ollama_client``) so the chat ``usage`` event counts it."""
    try:
        from app.llm.ollama_client import Usage, add_scope_usage, call_meta
        meta = call_meta(meta)
        add_scope_usage(Usage(int(prompt_tokens), int(completion_tokens), int(duration_ms), model, purpose, ok=ok))
    except Exception as exc:  # pragma: no cover
        log.debug("usage scope append failed: %s", exc)
    try:
        with get_pool().connection() as conn:
            conn.execute(
                "INSERT INTO llm_calls(session_id, message_id, purpose, provider, model, prompt_tokens,"
                " completion_tokens, total_tokens, duration_ms, ok, error, meta)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)",
                (str(session_id) if session_id else None, message_id, purpose, provider, model,
                 int(prompt_tokens), int(completion_tokens), int(prompt_tokens) + int(completion_tokens),
                 int(duration_ms), ok, error, json.dumps(meta or {}, default=str)))
    except Exception as exc:  # pragma: no cover - audit is best effort
        log.warning("llm_calls insert failed (%s): %s", purpose, exc)


# ----------------------------------------------------------------------------- conversation tree (FEATURES_V2 §2)
# A session's messages form a tree through ``parent_id``; the UI shows one path (root -> ``active_leaf_id``).
# Siblings = same (parent_id, role). Empty assistant rows are placeholders of stopped / unfinished answers: they
# are kept in the tree but never listed as siblings or shown.
MESSAGE_COLS = ("m.message_id, m.parent_id, m.role, m.content, m.standalone_query, m.decisions, m.sources, m.images,"
                " m.out_of_context, m.usage, m.mode, m.created_at, f.rating AS feedback_rating,"
                " f.comment AS feedback_comment")


def is_placeholder(row: dict) -> bool:
    return row.get("role") == "assistant" and not (row.get("content") or "").strip()


def load_tree(conn: psycopg.Connection, session_id: str) -> tuple[dict[int, dict], int | None]:
    """All messages of a session ({id: row}, ordered by id) + the session's active leaf id. Repairs v1 rows
    (mode IS NULL, no parent, not the first message): they get the previous message as parent - rows written by
    a still-running v1 server after the startup backfill."""
    cur = conn.execute(f"SELECT {MESSAGE_COLS} FROM messages m LEFT JOIN message_feedback f USING (message_id) "
                       "WHERE m.session_id = %s ORDER BY m.message_id", (session_id,))
    cols = [c.name for c in cur.description]
    rows = {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}
    prev = None
    for mid, r in rows.items():
        if r["parent_id"] is None and r["mode"] is None and prev is not None:
            r["parent_id"] = prev
            conn.execute("UPDATE messages SET parent_id = %s WHERE message_id = %s AND parent_id IS NULL", (prev, mid))
        prev = mid
    row = conn.execute("SELECT active_leaf_id FROM sessions WHERE session_id = %s", (session_id,)).fetchone()
    leaf = row[0] if row else None
    if rows:
        newest = max(rows)
        # no / stale leaf, or v1 rows appended after it (v1 server): the newest message
        if leaf not in rows or (rows[newest]["mode"] is None and newest > leaf):
            leaf = newest
    else:
        leaf = None
    return rows, leaf


def path_to(rows: dict[int, dict], leaf_id: int | None) -> list[dict]:
    """Rows from the root to ``leaf_id`` (inclusive)."""
    out, seen = [], set()
    mid = leaf_id
    while mid is not None and mid in rows and mid not in seen:
        seen.add(mid)
        out.append(rows[mid])
        mid = rows[mid]["parent_id"]
    out.reverse()
    return out


def children(rows: dict[int, dict], parent_id: int | None) -> list[dict]:
    return [r for r in rows.values() if r["parent_id"] == parent_id]


def newest_leaf(rows: dict[int, dict], message_id: int) -> int:
    """Follow the newest (non-placeholder if possible) child down from ``message_id`` to a leaf."""
    mid, seen = message_id, set()
    while mid not in seen:
        seen.add(mid)
        kids = children(rows, mid)
        if not kids:
            break
        real = [k for k in kids if not is_placeholder(k)] or kids
        mid = max(k["message_id"] for k in real)
    return mid


def siblings(rows: dict[int, dict], row: dict) -> list[int]:
    """Ids of the row's siblings (incl. itself), oldest first; placeholders excluded (except the row itself)."""
    return sorted(r["message_id"] for r in rows.values()
                  if r["parent_id"] == row["parent_id"] and r["role"] == row["role"]
                  and (r["message_id"] == row["message_id"] or not is_placeholder(r)))


def set_active_leaf(conn: psycopg.Connection, session_id: str, leaf_id: int | None) -> None:
    conn.execute("UPDATE sessions SET active_leaf_id = %s, updated_at = now() WHERE session_id = %s",
                 (leaf_id, session_id))
