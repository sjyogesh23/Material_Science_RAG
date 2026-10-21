"""FastAPI app: SSE chat (modes, branches), sessions, PDF pages, figures, token audit, runtime settings,
feedback (app/api/feedback_routes.py), eval (eval_routes) and architecture / components (arch_routes).

All Postgres access goes through the *sync* psycopg pool (``app.db.get_pool()``) from plain ``def`` endpoints
(FastAPI runs them in its thread pool) or ``run_in_threadpool`` - psycopg's async pool does not work on the
Windows Proactor event loop that uvicorn uses.

Run: ``python run.py`` (starts Postgres + Ollama first) or ``uvicorn app.api.main:app``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import db
from app.config import CHOICES, EDITABLE_FIELDS, RANGES, ROOT, get_settings, reload_settings, save_overrides

log = logging.getLogger("newrag.api")
STATIC = ROOT / "app" / "static"
MAX_MESSAGE_CHARS = 4000

# ----------------------------------------------------------------------------- readiness
_ready: dict[str, Any] = {"db": False, "tokenizer": False, "retriever": False, "warmup_error": None,
                          "warmup_s": None}


def _warmup() -> None:
    """Tokenizer + retriever (BM25 index, embedder, reranker) in a background thread so the server is
    reachable immediately; /api/health reports ``retriever_ready`` when done."""
    t0 = time.perf_counter()
    try:
        from app.tokens import count_tokens
        count_tokens("x")
        _ready["tokenizer"] = True
        if os.getenv("NEWRAG_SKIP_WARMUP", "").lower() in ("1", "true", "yes"):
            return
        from app.retrieval.hybrid import get_retriever
        r = get_retriever()
        if hasattr(r, "warmup"):
            r.warmup()
        _ready["retriever"] = True
    except Exception as e:  # noqa: BLE001 - warmup is best effort; the first request loads lazily
        log.exception("warmup failed")
        _ready["warmup_error"] = f"{type(e).__name__}: {e}"
    finally:
        _ready["warmup_s"] = round(time.perf_counter() - t0, 1)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        await run_in_threadpool(db.init_db)
        reload_settings()                       # apply app_settings overrides now that the pool is open
        _ready["db"] = True
    except Exception as e:  # noqa: BLE001 - keep serving; /api/health shows postgres down
        log.exception("database init failed: %s", e)
    threading.Thread(target=_warmup, name="newrag-warmup", daemon=True).start()
    yield
    db.close_pools()


app = FastAPI(title="NewRAG", version="1.0",
              description="Offline materials-science RAG over the Steimel textbook (Ollama + Postgres).",
              lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
from app.api.eval_routes import router as eval_router  # noqa: E402 - evaluation API (/api/eval/...)
app.include_router(eval_router)
from app.api.arch_routes import router as arch_router  # noqa: E402 - architecture + versioned components (/api/arch, /api/components)
app.include_router(arch_router)
from app.api.feedback_routes import router as feedback_router  # noqa: E402 - likes / dislikes + reviews (/api/feedback)
app.include_router(feedback_router)


# ----------------------------------------------------------------------------- helpers
def _q(sql: str, params: tuple | list = (), fetch: str = "all") -> Any:
    try:
        with db.get_pool().connection() as conn:
            cur = conn.execute(sql, params)
            if fetch == "one":
                return cur.fetchone()
            if fetch == "all":
                return cur.fetchall()
            return cur.rowcount
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ in ("OperationalError", "PoolTimeout", "PoolClosed"):
            raise HTTPException(503, f"Postgres unavailable: {e}") from e
        raise


def _dict_rows(sql: str, params: tuple | list = ()) -> list[dict]:
    with db.get_pool().connection() as conn:
        cur = conn.execute(sql, params)
        cols = [c.name for c in cur.description]
        return [_jsonable(dict(zip(cols, r))) for r in cur.fetchall()]


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, uuid.UUID):
        return str(v)
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if hasattr(v, "item") and callable(v.item):        # numpy scalars
        try:
            return v.item()
        except Exception:  # noqa: BLE001
            return str(v)
    return v


def _uuid(sid: str) -> str:
    try:
        return str(uuid.UUID(sid))
    except (ValueError, TypeError):
        raise HTTPException(404, "Session not found") from None


# ----------------------------------------------------------------------------- pages
@app.get("/", include_in_schema=False)
def chat_page():
    return FileResponse(STATIC / "chat.html")


@app.get("/audit", include_in_schema=False)
def audit_page():
    return FileResponse(STATIC / "audit.html")


@app.get("/eval", include_in_schema=False)
def eval_page():
    return FileResponse(STATIC / "eval.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    for name in ("favicon.ico", "favicon.svg"):
        if (STATIC / name).exists():
            return FileResponse(STATIC / name)
    return Response(status_code=204)


# ----------------------------------------------------------------------------- health
@app.get("/api/health")
async def health():
    s = get_settings()
    out: dict[str, Any] = {"postgres": False, "ollama": False, "chat_model_present": False, "chunks": 0,
                           "embedder_loaded": _ready["retriever"], "retriever_ready": _ready["retriever"],
                           "tokenizer_ready": _ready["tokenizer"], "warmup_error": _ready["warmup_error"],
                           "warmup_s": _ready["warmup_s"], "chat_model": s.chat_model,
                           "ollama_host": s.ollama_host}

    def _pg() -> int:
        with db.get_pool().connection() as conn:
            return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

    try:
        out["chunks"] = await asyncio.wait_for(run_in_threadpool(_pg), timeout=5)
        out["postgres"] = True
    except Exception as e:  # noqa: BLE001
        out["postgres_error"] = f"{type(e).__name__}: {e}"[:300]
    try:
        from app.llm.ollama_client import get_llm
        h = await asyncio.wait_for(get_llm().health(), timeout=5)
        out["ollama"] = bool(h.get("ok"))
        out["chat_model_present"] = bool(h.get("chat_model_present"))
        out["models"] = h.get("models", [])
        if h.get("error"):
            out["ollama_error"] = str(h["error"])[:300]
    except Exception as e:  # noqa: BLE001
        out["ollama_error"] = f"{type(e).__name__}: {e}"[:300]
    try:
        from app.retrieval.embedder import embedder_loaded
        out["embedder_loaded"] = bool(embedder_loaded())
        from app.retrieval.reranker import reranker_loaded
        out["reranker_loaded"] = bool(reranker_loaded())
    except Exception:  # noqa: BLE001 - older retrieval modules: keep the warmup flag
        pass
    out["embed_backend"] = getattr(s, "embed_backend", None)
    out["rerank_backend"] = getattr(s, "rerank_backend", None)
    out["ok"] = bool(out["postgres"] and out["ollama"] and out["chat_model_present"] and out["chunks"])
    return out


# ----------------------------------------------------------------------------- sessions
class SessionPatch(BaseModel):
    title: str = Field(min_length=1, max_length=200)


_SESSION_COLS = ("session_id, COALESCE(title, 'New conversation') AS title, created_at, updated_at, "
                 "message_count")


@app.get("/api/sessions")
def list_sessions():
    return _dict_rows(f"SELECT {_SESSION_COLS} FROM sessions WHERE NOT COALESCE(archived, false) "
                      "ORDER BY updated_at DESC NULLS LAST LIMIT 500")


@app.post("/api/sessions")
def create_session():
    from app.memory.session_memory import SessionMemory
    sid = str(uuid.uuid4())
    _q("INSERT INTO sessions(session_id, title, memory_path) VALUES (%s, NULL, %s)",
       (sid, str(SessionMemory.path_for(sid))), fetch="none")
    return {"session_id": sid, "title": "New conversation"}


@app.patch("/api/sessions/{sid}")
def rename_session(sid: str, body: SessionPatch):
    sid = _uuid(sid)
    title = " ".join(body.title.split())[:200]
    rows = _dict_rows(f"UPDATE sessions SET title=%s, updated_at=now() WHERE session_id=%s RETURNING {_SESSION_COLS}",
                      (title, sid))
    if not rows:
        raise HTTPException(404, "Session not found")
    try:  # keep the memory file's title in sync (best effort)
        from app.memory.session_memory import SessionMemory
        mem = SessionMemory.load(sid)
        if mem.path.exists():
            mem.title = title
            mem.save()
    except Exception as e:  # noqa: BLE001
        log.warning("memory title update failed: %s", e)
    return rows[0]


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    sid = _uuid(sid)
    row = _q("DELETE FROM sessions WHERE session_id=%s RETURNING memory_path", (sid,), fetch="one")
    if row is None:
        raise HTTPException(404, "Session not found")
    from app.memory.session_memory import SessionMemory
    for p in {SessionMemory.path_for(sid), Path(row[0]) if row[0] else None} - {None}:
        try:
            p.unlink(missing_ok=True)
        except OSError as e:
            log.warning("could not delete memory file %s: %s", p, e)
    return {"ok": True}


def _session_exists(sid: str) -> bool:
    return _q("SELECT 1 FROM sessions WHERE session_id=%s", (sid,), fetch="one") is not None


def _path_items(rows: dict[int, dict], leaf: int | None) -> list[dict]:
    """The active path (root -> leaf) as API items with sibling info; placeholders of stopped answers skipped."""
    out = []
    for r in db.path_to(rows, leaf):
        if db.is_placeholder(r):
            continue  # placeholder of an answer that was stopped / never finished
        sib = db.siblings(rows, r)
        item = {k: r.get(k) for k in ("message_id", "parent_id", "role", "content", "standalone_query", "decisions",
                                      "usage", "mode", "created_at")}
        item["sources"] = r.get("sources") or []
        item["images"] = r.get("images") or []
        item["out_of_context"] = bool(r.get("out_of_context"))
        item["feedback"] = ({"rating": r["feedback_rating"], "comment": r.get("feedback_comment")}
                            if r.get("feedback_rating") is not None else None)
        item["sibling_ids"] = sib
        item["sibling_index"] = sib.index(r["message_id"]) if r["message_id"] in sib else 0
        item["sibling_count"] = len(sib)
        out.append(_jsonable(item))
    return out


def _session_path(sid: str) -> list[dict]:
    with db.get_pool().connection() as conn:
        rows, leaf = db.load_tree(conn, sid)
    return _path_items(rows, leaf)


@app.get("/api/sessions/{sid}/messages")
def session_messages(sid: str):
    """Active path only: [{message_id, parent_id, role, content, sources, images, decisions, usage, out_of_context,
    feedback, mode, created_at, sibling_ids, sibling_index, sibling_count}]."""
    sid = _uuid(sid)
    if not _session_exists(sid):
        raise HTTPException(404, "Session not found")
    return _session_path(sid)


class SwitchIn(BaseModel):
    message_id: int


async def _rebuild_memory_for_path(sid: str, rows: dict[int, dict], leaf: int | None) -> None:
    """Rewrite the session memory file so it describes the path root -> leaf (compacting if over budget)."""
    from app.llm.ollama_client import get_llm
    from app.memory.session_memory import SessionMemory
    from app.rag.pipeline import path_turns
    mem = SessionMemory.load(sid)
    if mem.rebuild_from_path(path_turns(db.path_to(rows, leaf))):
        await mem.compact(get_llm())
    mem.save()


@app.post("/api/sessions/{sid}/switch")
async def switch_branch(sid: str, body: SwitchIn):
    """Make the newest leaf under ``message_id`` the active leaf (e.g. after picking a sibling) and rewrite the
    session memory from the new path. -> {messages: [items shaped like GET .../messages], active_leaf_id}."""
    sid = _uuid(sid)
    from app.rag.pipeline import session_lock

    def _load():
        if not _session_exists(sid):
            raise HTTPException(404, "Session not found")
        with db.get_pool().connection() as conn:
            return db.load_tree(conn, sid)

    async with session_lock(sid):          # never while an answer in this session is being written
        rows, old_leaf = await run_in_threadpool(_load)
        if body.message_id not in rows:
            raise HTTPException(404, "Message not found in this session")
        leaf = db.newest_leaf(rows, body.message_id)
        if leaf != old_leaf:
            def _save():
                with db.get_pool().connection() as conn:
                    db.set_active_leaf(conn, sid, leaf)
            await run_in_threadpool(_save)
        try:
            await _rebuild_memory_for_path(sid, rows, leaf)
        except Exception as e:  # noqa: BLE001 - the switch itself succeeded
            log.warning("memory rebuild after switch failed: %s", e)
    return {"messages": _path_items(rows, leaf), "active_leaf_id": leaf}


@app.get("/api/sessions/{sid}/memory")
def session_memory(sid: str):
    sid = _uuid(sid)
    if not _session_exists(sid):
        raise HTTPException(404, "Session not found")
    from app.memory.session_memory import SessionMemory
    mem = SessionMemory.load(sid)
    md = mem.path.read_text(encoding="utf-8") if mem.path.exists() else mem.to_markdown()
    return {"markdown": md, "tokens": mem.token_count(), "budget": get_settings().memory_budget_tokens,
            "path": str(mem.path)}


@app.post("/api/sessions/{sid}/memory/compact")
async def compact_session_memory(sid: str):
    """Fold all but the last memory_keep_turns turns into the summary now (LLM; heuristic fallback).
    -> {markdown, tokens, budget, before_tokens, path, compacted}"""
    sid = _uuid(sid)
    if not await run_in_threadpool(_session_exists, sid):
        raise HTTPException(404, "Session not found")
    from app.llm.ollama_client import get_llm
    from app.memory.session_memory import SessionMemory
    from app.rag.pipeline import session_lock
    async with session_lock(sid):
        mem = SessionMemory.load(sid)
        before = mem.token_count()
        changed = False
        if mem.turns or mem.summary:
            changed = await mem.compact(get_llm())
            mem.save()
    md = mem.path.read_text(encoding="utf-8") if mem.path.exists() else mem.to_markdown()
    return {"compacted": bool(changed), "before_tokens": before, "markdown": md, "tokens": mem.token_count(),
            "budget": get_settings().memory_budget_tokens, "path": str(mem.path)}


# ----------------------------------------------------------------------------- modes
@app.get("/api/modes")
def list_modes():
    """[{id, name, tagline, description, settings, warning?}] (active ``modes`` component or code defaults)."""
    from app.config import get_modes
    return [_jsonable(m) for m in get_modes().values()]


# ----------------------------------------------------------------------------- chat (SSE)
def truncate_message(msg: str, limit: int = MAX_MESSAGE_CHARS) -> tuple[str, int]:
    """Keep the TAIL of an over-long message (the question is usually last, after pasted context), cut at a
    word boundary. Returns (kept text, number of dropped characters)."""
    if len(msg) <= limit:
        return msg, 0
    tail = msg[-limit:]
    cut = re.search(r"\s", tail[:200])
    if cut and msg[-limit - 1:-limit].strip():   # started mid-word: drop the partial word
        tail = tail[cut.end():]
    tail = tail.lstrip()
    return tail, len(msg) - len(tail)


class ChatIn(BaseModel):
    session_id: str | None = None
    message: str | None = None
    mode: str | None = None               # swift (default) | deep
    parent_id: int | None = None          # attach the new user message here (default: the active leaf)
    edit_of: int | None = None            # new sibling of this user message (same parent), with ``message``
    regenerate_of: int | None = None      # new sibling answer of this assistant message (no ``message``)


def _mode_ids() -> list[str]:
    from app.config import get_modes
    return list(get_modes())


def _sse(event: str, data: Any) -> ServerSentEvent:
    return ServerSentEvent(event=event, raw_data=json.dumps(_jsonable(data), ensure_ascii=False,
                                                            default=str))


@app.post("/api/chat", response_class=EventSourceResponse)
async def chat(body: ChatIn, request: Request):
    msg = (body.message or "").strip()
    if body.mode is not None and body.mode.lower().strip() not in _mode_ids():
        yield _sse("error", {"message": f"Unknown mode {body.mode!r} (use one of: {', '.join(_mode_ids())})."})
        yield _sse("done", {"message_id": None, "out_of_context": False})
        return
    if body.regenerate_of is not None:
        msg = ""                          # the stored question is answered again
    elif not msg:
        yield _sse("error", {"message": "Empty question."})
        yield _sse("done", {"message_id": None, "out_of_context": False})
        return
    if len(msg) > MAX_MESSAGE_CHARS:
        msg, dropped = truncate_message(msg)
        yield _sse("warning", {"code": "message_truncated", "dropped_chars": dropped, "kept_chars": len(msg),
                               "message": f"Your message was {dropped + len(msg):,} characters; only the last "
                                          f"{len(msg):,} were used (limit {MAX_MESSAGE_CHARS:,})."})
    from app.rag.pipeline import answer

    kwargs = {k: v for k, v in (("mode", body.mode), ("parent_id", body.parent_id), ("edit_of", body.edit_of),
                                ("regenerate_of", body.regenerate_of)) if v is not None}
    agen = answer(body.session_id or None, msg, **kwargs) if kwargs else answer(body.session_id or None, msg)
    try:
        async for ev in agen:
            if await request.is_disconnected():
                log.info("client disconnected - stopping generation")
                break
            yield _sse(ev.get("event", "message"), ev.get("data"))
    finally:
        await agen.aclose()   # closes the Ollama stream if the client went away mid-answer


# ----------------------------------------------------------------------------- PDF pages
_pdf_cache: dict[str, Path] = {}


def _pdf_for(doc_id: str) -> Path:
    if doc_id in _pdf_cache:
        return _pdf_cache[doc_id]
    row = _q("SELECT pdf_path, file_name FROM documents WHERE doc_id=%s", (doc_id,), fetch="one")
    if row is None:
        raise HTTPException(404, "Unknown document")
    pdf_path, file_name = row
    cands = []
    if pdf_path:
        p = Path(pdf_path)
        cands.append(p if p.is_absolute() else ROOT / p)
    if file_name:
        cands += list((ROOT / "data").rglob(file_name))
    for p in cands:
        if p.exists():
            _pdf_cache[doc_id] = p
            return p
    raise HTTPException(404, "PDF file not found on disk")


def _safe_id(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:80]


@app.get("/api/page/{doc_id}/{page}.png")
def page_png(doc_id: str, page: int, dpi: int = 110):
    dpi = max(50, min(int(dpi), 200))
    cache_dir = Path(get_settings().page_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{_safe_id(doc_id)}_p{page}_{dpi}.png"
    headers = {"Cache-Control": "public, max-age=604800"}
    if out.exists():
        return FileResponse(out, media_type="image/png", headers=headers)
    pdf = _pdf_for(doc_id)
    import pymupdf
    with pymupdf.open(pdf) as doc:
        if not 1 <= page <= doc.page_count:
            raise HTTPException(404, "No such page")
        png = doc[page - 1].get_pixmap(dpi=dpi).tobytes("png")
    tmp = out.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_bytes(png)
    try:
        os.replace(tmp, out)
    except OSError:
        tmp.unlink(missing_ok=True)
    return Response(png, media_type="image/png", headers=headers)


@app.get("/api/page/{doc_id}/{page}/meta")
def page_meta(doc_id: str, page: int):
    pdf = _pdf_for(doc_id)
    import pymupdf
    with pymupdf.open(pdf) as doc:
        if not 1 <= page <= doc.page_count:
            raise HTTPException(404, "No such page")
        r = doc[page - 1].rect
        return {"width": round(r.width, 2), "height": round(r.height, 2), "page_count": doc.page_count}


# ----------------------------------------------------------------------------- figures
_MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
          ".gif": "image/gif", ".svg": "image/svg+xml"}


@app.get("/api/figure/{chunk_id}")
def figure(chunk_id: str):
    row = _q("SELECT payload->'figure'->>'image_file' FROM chunks WHERE chunk_id=%s", (chunk_id,), fetch="one")
    if row is None or not row[0]:
        raise HTTPException(404, "No figure for this chunk")
    p = Path(row[0])
    path = (p if p.is_absolute() else ROOT / p).resolve()
    if ROOT.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404, "Figure file missing")
    return FileResponse(path, media_type=_MEDIA.get(path.suffix.lower(), "application/octet-stream"),
                        headers={"Cache-Control": "public, max-age=604800"})


# ----------------------------------------------------------------------------- audit
_AGG = ("count(*)::int AS calls, COALESCE(sum(prompt_tokens),0)::bigint AS prompt_tokens, "
        "COALESCE(sum(completion_tokens),0)::bigint AS completion_tokens, "
        "COALESCE(sum(total_tokens),0)::bigint AS total_tokens, round(avg(duration_ms)::numeric, 1) AS avg_ms, "
        "count(*) FILTER (WHERE NOT ok)::int AS failed")


@app.get("/api/audit/summary")
def audit_summary(days: int = 30):
    days = max(1, min(int(days), 3650))
    where = "WHERE c.created_at >= now() - make_interval(days => %s)"
    p = (days,)
    totals = _dict_rows(f"SELECT {_AGG} FROM llm_calls c {where}", p)[0]
    by_purpose = _dict_rows(f"SELECT purpose, {_AGG} FROM llm_calls c {where} GROUP BY purpose "
                            "ORDER BY total_tokens DESC", p)
    by_model = _dict_rows(f"SELECT model, provider, {_AGG} FROM llm_calls c {where} GROUP BY model, provider "
                          "ORDER BY total_tokens DESC", p)
    by_day = _dict_rows(f"SELECT to_char(c.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD') AS day, {_AGG} "
                        f"FROM llm_calls c {where} GROUP BY 1 ORDER BY 1", p)
    by_session = _dict_rows(
        f"SELECT c.session_id, s.title, {_AGG} FROM llm_calls c LEFT JOIN sessions s USING (session_id) "
        f"{where} AND c.session_id IS NOT NULL GROUP BY c.session_id, s.title ORDER BY total_tokens DESC LIMIT 200", p)
    return {"days": days, "totals": totals, "by_purpose": by_purpose, "by_model": by_model, "by_day": by_day,
            "by_session": by_session}


@app.get("/api/audit/calls")
def audit_calls(limit: int = 200, session_id: str | None = None, purpose: str | None = None):
    limit = max(1, min(int(limit), 5000))
    conds, params = [], []
    if session_id:
        conds.append("session_id = %s")
        params.append(_uuid(session_id))
    if purpose:
        conds.append("purpose = %s")
        params.append(purpose)
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    return _dict_rows("SELECT id, session_id, message_id, purpose, provider, model, prompt_tokens, completion_tokens,"
                      f" total_tokens, duration_ms, ok, error, meta, created_at FROM llm_calls {where} "
                      "ORDER BY created_at DESC, id DESC LIMIT %s", (*params, limit))


# ----------------------------------------------------------------------------- settings
def _editable(s=None) -> dict[str, Any]:
    s = s or get_settings()
    return {k: _jsonable(getattr(s, k)) for k in EDITABLE_FIELDS if hasattr(s, k)}


@app.get("/api/settings")
def get_settings_api():
    return _editable()


@app.get("/api/settings/limits")
def settings_limits():
    """Allowed ranges / choices of the editable settings (the audit UI validates inline with them)."""
    return {"ranges": {k: list(v) for k, v in RANGES.items()}, "choices": {k: list(v) for k, v in CHOICES.items()}}


@app.put("/api/settings")
def put_settings(changes: dict[str, Any]):
    if not isinstance(changes, dict) or not changes:
        raise HTTPException(422, "Body must be a non-empty JSON object of setting: value")
    unknown = [k for k in changes if k.lower() not in EDITABLE_FIELDS]
    if unknown:
        raise HTTPException(422, f"Unknown or read-only setting(s): {', '.join(unknown)}")
    try:
        s = save_overrides(changes)
    except (TypeError, ValueError) as e:
        raise HTTPException(422, f"Invalid value: {e}") from e
    except KeyError as e:
        raise HTTPException(422, str(e)) from e
    return _editable(s)


# ----------------------------------------------------------------------------- errors
@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):  # pragma: no cover - last resort
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)


__all__ = ["app"]
