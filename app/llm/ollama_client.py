"""Ollama LLM wrapper: every call (ok or failed) is recorded as one ``llm_calls`` row.

* ``complete()`` - one non-streaming chat call (optionally structured output via ``format=<json schema>``)
* ``stream()``   - yields content tokens; usage is recorded at the end and exposed as ``last_usage``
* ``usage_scope()`` - context manager collecting every call made inside it (the pipeline uses it to emit
  the per-message ``usage`` event without threading a collector through every function)

``think=False`` except ``stream(think=True)`` (Deep mode answers); options = {temperature, num_ctx} from settings, merged with per-call options.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

from ollama import AsyncClient

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------- settings / tokens / db
@dataclass
class _FallbackSettings:
    """Used only if app.config is unavailable (keeps this package testable on its own)."""
    root: Path = ROOT
    ollama_host: str = os.getenv("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
    chat_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    router_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    num_ctx: int = 8192
    temperature: float = 0.1
    max_answer_tokens: int = 900
    final_k: int = 6
    ooc_threshold: float = 0.20
    ooc_soft_threshold: float = 0.45
    sessions_dir: Path = ROOT / "sessions"
    page_cache_dir: Path = ROOT / "cache" / "pages"
    memory_budget_tokens: int = 1500
    memory_keep_turns: int = 6
    grounding_check: bool = False


def settings() -> Any:
    try:
        from app.config import get_settings
        return get_settings()
    except Exception as e:  # noqa: BLE001 - config owned by another module; degrade, don't crash
        log.debug("app.config unavailable (%s); using fallback settings", e)
        return _FallbackSettings()


def setting(name: str, default: Any) -> Any:
    return getattr(settings(), name, default)


def count_tokens(text: str) -> int:
    try:
        from app.tokens import count_tokens as _ct
        return _ct(text)
    except Exception:  # noqa: BLE001
        return math.ceil(len(text) / 3.6)  # ~chars/token for English + LaTeX with the Qwen tokenizer


_db_failed_at = -1e9


def _db_run(sql: str, params: tuple, fetch: str | None) -> Any:
    from app import db
    with db.get_pool().connection() as conn:
        cur = conn.execute(sql, params)
        if fetch == "one":
            return cur.fetchone()
        if fetch == "all":
            return cur.fetchall()
        return None


async def db_execute(sql: str, params: tuple = (), fetch: str | None = None) -> Any:
    """Run one statement on the shared *sync* pool in a worker thread (works on any event loop, incl. the
    Windows Proactor loop that async psycopg rejects). Returns None - with a warning, retried after 30 s -
    if Postgres is unavailable; persistence must never break answering."""
    global _db_failed_at
    if time.monotonic() - _db_failed_at < 30:
        return None
    try:
        return await asyncio.to_thread(_db_run, sql, params, fetch)
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ in ("OperationalError", "PoolTimeout", "PoolClosed", "ImportError",
                                "ModuleNotFoundError", "FileNotFoundError"):
            _db_failed_at = time.monotonic()  # DB down: stop trying for a while
        log.warning("Postgres write failed, not persisted: %s: %s", type(e).__name__, str(e)[:300])
        return None


# ----------------------------------------------------------------------------- usage
@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    duration_ms: int
    model: str
    purpose: str = ""
    ok: bool = True


_scope: contextvars.ContextVar[list[Usage] | None] = contextvars.ContextVar("llm_usage_scope", default=None)


@contextmanager
def usage_scope() -> Iterator[list[Usage]]:
    calls: list[Usage] = []
    token = _scope.set(calls)
    try:
        yield calls
    finally:
        _scope.reset(token)


def add_scope_usage(u: Usage) -> None:
    """Append a usage record made outside this client (retrieval: ``embed_query`` / ``rerank`` via
    ``app.db.log_llm_call``) to the current usage scope, so the per-answer ``usage`` event includes it.
    Works from ``asyncio.to_thread`` workers: they run in a copy of the context that shares the same list."""
    if (calls := _scope.get()) is not None:
        calls.append(u)


def begin_usage_scope() -> list[Usage]:
    """Like usage_scope() but without a reset - for async generators whose steps may resume in another
    context (a fresh list per call; the scope ends with the task)."""
    calls: list[Usage] = []
    _scope.set(calls)
    return calls


# extra keys merged into every llm_calls.meta written in this context (answer mode, component versions, ...)
_call_meta: contextvars.ContextVar[dict | None] = contextvars.ContextVar("llm_call_meta", default=None)


def set_call_meta(meta: dict | None) -> None:
    """Replace the per-context llm_calls.meta extras (the pipeline sets {"mode": ...} per answer)."""
    _call_meta.set(dict(meta) if meta else None)


def update_call_meta(**kv: Any) -> None:
    """Add keys to the per-context llm_calls.meta extras (e.g. component versions used)."""
    _call_meta.set({**(_call_meta.get() or {}), **kv})


def call_meta(meta: dict | None = None) -> dict:
    """The context extras merged with ``meta`` (``meta`` wins) + ``components`` = active component versions
    (app.versioning, so eval runs can be compared per prompt version)."""
    out = {**(_call_meta.get() or {}), **(meta or {})}
    if "components" not in out:
        try:
            from app import versioning
            out["components"] = versioning.active_versions()
        except Exception:  # noqa: BLE001 - versioning unavailable
            pass
    return out


class ThinkingToken(str):
    """A piece of the model's thinking text (``stream(think=True)``); never part of the answer."""


def summarize_usage(calls: list[Usage]) -> dict:
    by: dict[str, dict] = {}
    for u in calls:
        b = by.setdefault(u.purpose, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "duration_ms": 0})
        b["calls"] += 1
        b["prompt_tokens"] += u.prompt_tokens
        b["completion_tokens"] += u.completion_tokens
        b["duration_ms"] += u.duration_ms
    p = sum(u.prompt_tokens for u in calls)
    c = sum(u.completion_tokens for u in calls)
    return {"by_purpose": by, "prompt_tokens": p, "completion_tokens": c, "total": p + c}


# ----------------------------------------------------------------------------- client
def _transient(e: Exception) -> bool:
    msg = str(e).lower()
    return (getattr(e, "status_code", 0) in (500, 503) or "terminated" in msg or "allocate" in msg
            or "connect" in msg or "timed out" in msg)


_pending: set[asyncio.Task] = set()   # usage records of stopped streams still being written


class LLM:
    def __init__(self, host: str | None = None):
        self.host = host or settings().ollama_host
        self.client = AsyncClient(host=self.host)
        self.last_usage: Usage | None = None
        self.last_logprobs: list | None = None
        self.last_done_reason: str | None = None
        self.last_thinking_chars: int = 0

    def _options(self, options: dict | None) -> dict:
        s = settings()
        return {"temperature": s.temperature, "num_ctx": s.num_ctx, **(options or {})}

    async def _record(self, u: Usage, *, session_id=None, message_id=None, error: str | None = None,
                      meta: dict | None = None) -> None:
        self.last_usage = u
        if (calls := _scope.get()) is not None:
            calls.append(u)
        await db_execute(
            "INSERT INTO llm_calls(session_id, message_id, purpose, provider, model, prompt_tokens,"
            " completion_tokens, total_tokens, duration_ms, ok, error, meta)"
            " VALUES (%s,%s,%s,'ollama',%s,%s,%s,%s,%s,%s,%s,%s::jsonb)",
            (str(session_id) if session_id else None, message_id, u.purpose, u.model, u.prompt_tokens,
             u.completion_tokens, u.prompt_tokens + u.completion_tokens, u.duration_ms, u.ok, error,
             json.dumps(call_meta(meta), default=str)))

    async def complete(self, messages: list[dict], *, purpose: str, model: str | None = None,
                       format: dict | None = None, options: dict | None = None, session_id=None,
                       message_id=None, logprobs: bool = False) -> tuple[str, Usage]:
        model = model or settings().chat_model
        t0 = time.perf_counter()
        try:
            for attempt in range(2):  # one retry: model (re)load can fail transiently under RAM pressure
                try:
                    r = await self.client.chat(model=model, messages=messages, format=format, think=False,
                                               options=self._options(options), stream=False,
                                               logprobs=logprobs or None, top_logprobs=None)
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt or not _transient(e):
                        raise
                    log.warning("Ollama call failed (%s), retrying once", str(e)[:120])
                    await asyncio.sleep(2)
        except Exception as e:
            u = Usage(0, 0, int((time.perf_counter() - t0) * 1000), model, purpose, ok=False)
            await self._record(u, session_id=session_id, message_id=message_id, error=str(e)[:500])
            raise
        u = Usage(r.prompt_eval_count or 0, r.eval_count or 0,
                  int((r.total_duration or 0) / 1e6) or int((time.perf_counter() - t0) * 1000), model, purpose)
        self.last_logprobs = r.logprobs
        await self._record(u, session_id=session_id, message_id=message_id,
                           meta={"structured": format is not None})
        return r.message.content or "", u

    async def stream(self, messages: list[dict], *, purpose: str, model: str | None = None,
                     options: dict | None = None, session_id=None, message_id=None,
                     think: bool = False) -> AsyncIterator[str]:
        """Yields content tokens. If the consumer stops early (aclose / cancel - the user pressed Stop) Ollama
        never sends its final counts, so the usage row is estimated: completion = streamed chunks (Ollama
        streams one token per chunk), prompt = tokenizer count of the messages; meta.partial = true.
        ``think=True``: the model's thinking is yielded as ``ThinkingToken`` pieces (consumers must keep them out
        of the answer); ``last_done_reason`` / ``last_thinking_chars`` describe the finished stream."""
        model = model or settings().chat_model
        t0 = time.perf_counter()
        prompt = completion = dur = 0
        streamed = 0
        thinking_chars = 0
        finished = False
        error: str | None = None
        self.last_done_reason = None
        try:
            async for chunk in await self.client.chat(model=model, messages=messages, think=bool(think),
                                                      options=self._options(options), stream=True):
                if chunk.done:
                    prompt, completion = chunk.prompt_eval_count or 0, chunk.eval_count or 0
                    dur = int((chunk.total_duration or 0) / 1e6)
                    finished = True
                    self.last_done_reason = getattr(chunk, "done_reason", None)
                if chunk.message and think and getattr(chunk.message, "thinking", None):
                    thinking_chars += len(chunk.message.thinking)
                    yield ThinkingToken(chunk.message.thinking)
                if chunk.message and chunk.message.content:
                    streamed += 1
                    yield chunk.message.content
        except Exception as e:
            error = str(e)[:500]
            raise
        finally:
            self.last_thinking_chars = thinking_chars
            meta: dict = {"think": True, "thinking_chars": thinking_chars} if think else {}
            if not finished:  # stopped early: estimate what was used so far
                completion = completion or streamed
                if not prompt:
                    try:
                        prompt = count_tokens("\n".join(str(m.get("content", "")) for m in messages))
                    except Exception:  # noqa: BLE001
                        prompt = 0
                meta = {**meta, "partial": True, "estimated": True, "streamed_chunks": streamed}
            u = Usage(prompt, completion, dur or int((time.perf_counter() - t0) * 1000), model, purpose,
                      ok=error is None)
            if finished:
                await self._record(u, session_id=session_id, message_id=message_id, error=error, meta=meta)
            else:
                # stopped: the consumer's task is usually being cancelled (client disconnect; anyio re-raises the
                # cancellation at every await), so record in a separate task that survives it
                task = asyncio.ensure_future(self._record(u, session_id=session_id, message_id=message_id,
                                                          error=error, meta=meta))
                _pending.add(task)
                task.add_done_callback(_pending.discard)
                try:
                    await asyncio.shield(task)
                except BaseException:  # noqa: BLE001 - cancelled while waiting: the task still finishes
                    pass

    async def health(self) -> dict:
        try:
            r = await self.client.list()
            models = [m.model for m in r.models]
            want = settings().chat_model
            return {"ok": True, "models": models,
                    "chat_model_present": any(m == want or m.split(":")[0] == want for m in models)}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "models": [], "chat_model_present": False, "error": str(e)}


_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None or _llm.host != settings().ollama_host:
        _llm = LLM()
    return _llm


__all__ = ["LLM", "Usage", "ThinkingToken", "set_call_meta", "update_call_meta", "call_meta", "get_llm", "usage_scope", "begin_usage_scope", "add_scope_usage", "summarize_usage",
           "settings", "setting",
           "count_tokens", "db_execute"]
