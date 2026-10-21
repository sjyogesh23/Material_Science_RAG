"""answer() orchestration -> async event stream (consumed by the SSE endpoint).

load memory -> route (standalone query, format, images, cross-ref) -> greeting? canned reply ->
retrieve (+ sub-queries) -> expand cross-refs (+ figures) -> out-of-context gate -> prompt within num_ctx ->
stream answer -> citations.validate -> [grounding check] -> images -> persist + memory (+ compaction) ->
usage -> done -> [session title].

Events (contract): session, status, decision, token, sources, images, usage, done | error.
Extensions: ``answer`` {content, uncited} after streaming (the validated/renumbered text that replaces the
streamed tokens), ``usage.timings_ms``, ``done.content``; a second ``session`` event carries the generated
title of a new session. The title is generated in a background task AFTER ``done`` (the UI unlocks at
``done``), delivered if the stream is still open and always persisted.

Concurrency: requests for the same session are serialised with a per-session asyncio.Lock (memory file +
turn numbering would otherwise lose turns); a queued request first emits ``status`` {stage: "waiting"}.

Routing fast path: when there is nothing to resolve from memory and no comparison / cross-reference, the
question is retrieved first and, if the reranker settles the scope (best score >= fast_route_min_score or
< ooc_threshold), keyword rules decide the rest and the 3-5 s LLM router call is skipped.

v2 (docs/FEATURES_V2.md):
  * Branching: every message has a ``parent_id``; the answer is attached to ``parent_id`` (default: the session's
    active leaf), ``edit_of`` (new sibling of a user message) or ``regenerate_of`` (new sibling of an assistant
    message). The memory used for the answer is rebuilt from the path whenever it is not the active branch
    (``SessionMemory.rebuild_from_path``); the new assistant message becomes the active leaf (sessions row).
    ``session`` adds {active_leaf_id = the (new) user message id, user_message_id, parent_id = that user message's
    parent, mode}; ``done`` adds {user_message_id, parent_id = user_message_id (the answer's parent), mode,
    active_leaf_id = the answer id}.
  * Modes (swift | deep, ``config.get_modes()``): the profile's Settings fields override the settings for this
    answer (``config.request_overrides``); ``think`` streams the model's thinking as ``thinking`` {t} events (never
    stored / remembered); ``grounding_check`` scores the answer 1-5 and ``self_revision`` revises it once when the
    score is below ``revision_min_grounding``. The mode is stored in messages.mode, decisions.mode and
    llm_calls.meta.mode.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import aclosing, asynccontextmanager
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable

from app.config import DEFAULT_MODE, get_mode, request_overrides, reset_request_overrides

from app.llm import prompts
from app.llm.decisions import (FIG_REF_RE, GREETING_RE, RouteDecision, Score, Text, decide, enumerated_items, fast_route,
                               heuristic_title,
                               is_jailbreak_only, route, router_needed, strip_injection, topic_query)
from app.llm.ollama_client import (ThinkingToken, begin_usage_scope, count_tokens, db_execute, get_llm,
                                   set_call_meta, settings, summarize_usage, update_call_meta)
from app.memory.session_memory import SessionMemory
from app.rag import citations, guardrails

log = logging.getLogger(__name__)

OOC_PREFIX = "#outof"
TITLE_WAIT_S = 60.0          # how long an open stream waits for the background title before closing
_retriever_factory: Callable[[], Any] | None = None
_pdf_paths_loaded = False
_total_pages: int | None = None     # pages of the (largest) textbook, for "cite page 999" checks


def set_retriever_factory(factory: Callable[[], Any] | None) -> None:
    """Tests / tools can inject a stub retriever (same interface as app.retrieval.hybrid.Retriever)."""
    global _retriever_factory
    _retriever_factory = factory


def _retriever() -> Any:
    if _retriever_factory is not None:
        return _retriever_factory()
    from app.retrieval.hybrid import get_retriever
    return get_retriever()


def _ev(event: str, data: Any) -> dict:
    return {"event": event, "data": data}


def _g(hit: Any, name: str, default: Any = None) -> Any:
    return getattr(hit, name, default)


# ----------------------------------------------------------------------------- per-session serialisation
_locks: dict[str, list] = {}    # session_id -> [asyncio.Lock, number of holders + waiters]


@asynccontextmanager
async def session_lock(sid: str):
    """Serialise everything that reads-modifies-writes one session (memory file, message counts). Entries are
    dropped when unused, so locks never outlive the event loop they were used on."""
    entry = _locks.get(sid)
    if entry is None:
        entry = _locks[sid] = [asyncio.Lock(), 0]
    entry[1] += 1
    try:
        async with entry[0]:
            yield
    finally:
        entry[1] -= 1
        if entry[1] <= 0 and _locks.get(sid) is entry:
            del _locks[sid]


def session_busy(sid: str) -> bool:
    entry = _locks.get(sid)
    return bool(entry and entry[0].locked())


# ----------------------------------------------------------------------------- persistence (best effort)
async def _ensure_session(session_id: str, first_question: str) -> tuple[str, bool]:
    row = await db_execute("SELECT title, message_count FROM sessions WHERE session_id = %s", (session_id,),
                           fetch="one")
    mem_path = str(SessionMemory.path_for(session_id))
    if row is None:
        title = heuristic_title(first_question)
        await db_execute("INSERT INTO sessions(session_id, title, memory_path) VALUES (%s,%s,%s) "
                         "ON CONFLICT (session_id) DO NOTHING", (session_id, title, mem_path))
        return title, True
    title, count = row
    return (title or heuristic_title(first_question)), (count or 0) == 0 or not title or title == "New chat"


async def _insert_message(session_id: str, role: str, content: str, parent_id: int | None = None,
                          mode: str | None = None) -> int | None:
    row = await db_execute("INSERT INTO messages(session_id, role, content, parent_id, mode) VALUES (%s,%s,%s,%s,%s) "
                           "RETURNING message_id", (session_id, role, content, parent_id, mode or DEFAULT_MODE),
                           fetch="one")
    return row[0] if row else None


async def _finish_message(message_id: int | None, session_id: str, content: str, decision: RouteDecision,
                          sources: list, images: list, ooc: bool, added: int = 2) -> None:
    if message_id is not None:
        await db_execute(
            "UPDATE messages SET content=%s, standalone_query=%s, decisions=%s::jsonb, sources=%s::jsonb,"
            " images=%s::jsonb, out_of_context=%s WHERE message_id=%s",
            (content, decision.standalone_query, json.dumps(decision.model_dump()), json.dumps(sources),
             json.dumps(images), ooc, message_id))
    await db_execute("UPDATE sessions SET message_count = COALESCE(message_count,0) + %s, updated_at = now() "
                     "WHERE session_id = %s", (added, session_id))


# ----------------------------------------------------------------------------- branches
class BranchError(ValueError):
    """Invalid parent_id / edit_of / regenerate_of (unknown message, wrong role, other session)."""


def _load_tree_sync(sid: str) -> tuple[dict[int, dict], int | None]:
    from app import db
    with db.get_pool().connection() as conn:
        return db.load_tree(conn, sid)


async def _load_tree(sid: str) -> tuple[dict[int, dict], int | None]:
    try:
        return await asyncio.to_thread(_load_tree_sync, sid)
    except Exception as e:  # noqa: BLE001 - DB down: answer without history (as v1 did)
        log.warning("could not load the conversation tree: %s", e)
        return {}, None


async def session_of_message(message_id: int) -> str | None:
    row = await db_execute("SELECT session_id FROM messages WHERE message_id = %s", (int(message_id),), fetch="one")
    return str(row[0]) if row else None


def path_turns(path: list[dict]) -> list[dict]:
    """Completed turns on a path (root -> leaf rows) for SessionMemory.rebuild_from_path: a user message followed
    by its non-empty assistant answer; greetings and failed answers are skipped (they were never remembered)."""
    out = []
    for i, row in enumerate(path):
        if row.get("role") != "user" or i + 1 >= len(path):
            continue
        a = path[i + 1]
        content = (a.get("content") or "").strip()
        if a.get("role") != "assistant" or not content or content.startswith("[error]"):
            continue
        dec = a.get("decisions") or {}
        if dec.get("intent") == "greeting":
            continue
        ooc = bool(a.get("out_of_context")) or content.startswith(prompts.OOC)
        ents = [] if ooc else _memory_entities(SimpleNamespace(entities=list(dec.get("entities") or [])),
                                                list(a.get("images") or []))
        out.append({"user": row.get("content") or "", "assistant": content, "message_id": a["message_id"],
                    "entities": ents})
    return out


def resolve_branch(rows: dict[int, dict], leaf: int | None, *, parent_id: int | None = None,
                   edit_of: int | None = None, regenerate_of: int | None = None) -> dict:
    """Where the new turn goes. -> {parent (of the new user message, or of the user message being re-answered),
    user_id (existing user message when regenerating, else None), question (regenerate only)}."""
    if regenerate_of is not None:
        a = rows.get(int(regenerate_of))
        if a is None or a["role"] != "assistant":
            raise BranchError(f"regenerate_of={regenerate_of} is not an assistant message of this conversation")
        u = rows.get(a["parent_id"]) if a["parent_id"] is not None else None
        if u is None or u["role"] != "user":
            raise BranchError(f"message {regenerate_of} has no question to regenerate")
        return {"parent": u["parent_id"], "user_id": u["message_id"], "question": u["content"] or ""}
    if edit_of is not None:
        u = rows.get(int(edit_of))
        if u is None or u["role"] != "user":
            raise BranchError(f"edit_of={edit_of} is not a user message of this conversation")
        return {"parent": u["parent_id"], "user_id": None, "question": None}
    if parent_id is not None:
        if int(parent_id) not in rows:
            raise BranchError(f"parent_id={parent_id} is not a message of this conversation")
        return {"parent": int(parent_id), "user_id": None, "question": None}
    return {"parent": leaf, "user_id": None, "question": None}


async def _set_leaf(sid: str, leaf: int | None) -> None:
    await db_execute("UPDATE sessions SET active_leaf_id=%s, updated_at=now() WHERE session_id=%s", (leaf, sid))


async def _save_usage(message_id: int | None, usage: dict) -> None:
    if message_id is not None:
        await db_execute("UPDATE messages SET usage=%s::jsonb WHERE message_id=%s",
                         (json.dumps(usage, default=str), message_id))


async def _load_pdf_paths() -> None:
    global _pdf_paths_loaded, _total_pages
    if _pdf_paths_loaded:
        return
    rows = await db_execute("SELECT doc_id, pdf_path, total_pages FROM documents", fetch="all")
    for doc_id, path, pages in rows or []:
        citations.register_pdf_path(doc_id, path)
        if pages:
            _total_pages = max(_total_pages or 0, int(pages))
    _pdf_paths_loaded = rows is not None


# ----------------------------------------------------------------------------- prompt
def _source_block(n: int, hit: Any, max_tokens: int) -> str:
    sec = " > ".join((_g(hit, "section_path") or [])[-2:])
    pages = _g(hit, "pages") or []
    kind = _g(hit, "chunk_type") or "text"
    head = f"[{n}] ({kind}; {sec}; page {', '.join(map(str, pages[:3]))})"
    body = (_g(hit, "content") or "").strip()
    if count_tokens(body) > max_tokens:  # trim long chunks to their share of the budget
        body = body[: max_tokens * 3].rsplit(" ", 1)[0] + " …"
    return f"{head}\n{body}"


def _explains_notes(used: list) -> list[str]:
    """Deterministic pointer for "which section explains Figure X": tell the model which numbered passage is
    the figure's explaining text and what its section is called (the 4B model otherwise cites the figure)."""
    pos = {_g(h, "chunk_id"): i + 1 for i, h in enumerate(used)}
    notes = []
    for i, h in enumerate(used, 1):
        if _g(h, "via") != "section_text":
            continue
        src = ((_g(h, "payload") or {}).get("_link") or {}).get("from")
        if src in pos:
            sec = ((_g(h, "section_path") or [""])[-1] or "").strip(" :")
            notes.append(f'[{i}] is the text passage of section "{sec}" that explains the figure/table in '
                         f'[{pos[src]}]; if asked which section explains it, name that section and cite [{i}].')
    return notes


def build_messages(question: str, decision: RouteDecision, hits: list, memory: SessionMemory,
                   extra_notes: list[str] | None = None) -> tuple[list[dict], list]:
    """System (rules + format + memory) + user (numbered sources + question + guardrail notes), sized to fit
    num_ctx minus the reserved answer tokens. Returns (messages, hits actually numbered in the prompt)."""
    s = settings()
    mem_md = memory.render(min(s.memory_budget_tokens, s.num_ctx // 5)) if memory.turns or memory.summary else ""
    system = prompts.answer_system(decision.answer_format, mem_md, decision.needs_images)
    fixed = (count_tokens(system) + count_tokens(prompts.answer_user(question, "", decision.standalone_query,
                                                                     extra_notes)) + 64 + 200)
    budget = s.num_ctx - s.max_answer_tokens - fixed
    blocks, used = [], []
    per_source = max(150, budget // max(1, len(hits)))
    spent = 0
    for hit in hits:
        block = _source_block(len(used) + 1, hit, per_source)
        cost = count_tokens(block) + 4
        if spent + cost > budget:
            break
        blocks.append(block)
        used.append(hit)
        spent += cost
    user = prompts.answer_user(question, "\n\n".join(blocks), decision.standalone_query,
                               _explains_notes(used) + list(extra_notes or []))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], used


def _merge(*lists: list) -> list:
    seen, out = set(), []
    for lst in lists:
        for h in lst or []:
            if h.chunk_id not in seen:
                seen.add(h.chunk_id)
                out.append(h)
    return out


def _ref_num(text: str | None) -> str:
    """'Figure 2-7' / 'fig. 2.7' / '2–7' -> '2.7' (comparable figure / table numbers)."""
    m = re.search(r"(\d+)(?:\s*[-.–]\s*(\d+))?", text or "")
    return "" if not m else m.group(1) + (f".{m.group(2)}" if m.group(2) else "")


def _figure_number(hit: Any) -> str:
    p = _g(hit, "payload") or {}
    return _ref_num(((p.get("figure") or {}).get("figure_number")) or ((p.get("table") or {}).get("table_number")))


def _figure_targets(question: str, decision: RouteDecision, hits: list, figs: list, retriever: Any) -> list:
    """Figure/table chunks the question is about: those whose number is mentioned, else (figure request that
    asks for the explaining section) the best figure. [] if the question is not about a figure/table."""
    refs = list(FIG_REF_RE.finditer(question))
    if not refs and not (decision.intent == "figure_request" and decision.is_cross_reference):
        return []
    pool = [h for h in _merge(hits, figs) if _g(h, "chunk_type") in ("figure", "table")]
    wanted = {_ref_num(m.group(2)) for m in refs}
    targets = [h for h in pool if _figure_number(h) and _figure_number(h) in wanted]
    if refs and not targets and hasattr(retriever, "figure_by_number"):
        for m in refs:
            targets += retriever.figure_by_number(m.group(0))
    if not targets and not refs:
        targets = pool[:1]
    return _merge(targets)[:2]


def _focus_query(question: str, long_chars: int = 600) -> str:
    """Long messages (pasted notes + a question at the end) retrieve badly as a whole: route / retrieve on the
    last question sentence instead (the full message still goes into the answer prompt)."""
    if len(question) <= long_chars:
        return question
    qs = [m.group(0).strip() for m in re.finditer(r"[^.?!\n]{8,400}\?", question)]
    if qs:
        return qs[-1]
    tail = question[-400:]
    return tail[tail.find(" ") + 1:].strip() if " " in tail else tail


def _prompt_question(question: str) -> str:
    """The question as shown to the answer model: injected role / style / rule-change instructions, fake sources
    and output-control sentences removed (the 4B model follows "answer as a pirate" even when told not to), the
    science part kept. See decisions.strip_injection."""
    cleaned, removed = strip_injection(question)
    if not removed:
        return question
    return cleaned if len(cleaned.split()) >= 2 else question


async def _make_title(question: str, sid: str, mid: int | None) -> str:
    try:
        text, _ = await get_llm().complete(
            [{"role": "system", "content": prompts.TITLE_SYSTEM}, {"role": "user", "content": question}],
            purpose="title", options={"temperature": 0, "num_predict": 16}, session_id=sid, message_id=mid)
        t = (text.strip().splitlines() or [""])[0]
        t = re.sub(r"[*#`]", "", t).strip().strip("\"'“”").rstrip(".!?:").strip()
        words = t.split()
        if 0 < len(words) <= 8:
            return " ".join(words[:6])
    except Exception as e:  # noqa: BLE001
        log.warning("title generation failed: %s", e)
    return heuristic_title(question)


async def _title_job(question: str, sid: str, mid: int | None, old_title: str) -> str | None:
    """Background (after ``done``): generate + persist the session title. Only replaces the provisional title,
    so a rename made meanwhile wins."""
    try:
        title = await _make_title(question, sid, mid)
        await db_execute("UPDATE sessions SET title=%s WHERE session_id=%s "
                         "AND (title IS NULL OR title IN (%s, 'New chat', 'New conversation'))",
                         (title, sid, old_title))
        async with session_lock(sid):   # never interleave with an answer that is rewriting the memory file
            mem = SessionMemory.load(sid)
            if mem.path.exists() and (not mem.title or mem.title == old_title):
                mem.title = title
                mem.save()
        return title
    except Exception as e:  # noqa: BLE001
        log.warning("background title failed: %s", e)
        return None


def _memory_entities(decision: RouteDecision, images: list[dict]) -> list[str]:
    figs = []
    for im in images[:2]:  # figures shown -> "that figure" can be resolved next turn (first shown = newest)
        cap, num = im.get("caption") or "", im.get("figure_number")
        if num:
            cap = cap.rstrip(". ")
            figs.append(cap if cap.lower().startswith(num.lower()) else f"{num}: {cap}".rstrip(": "))
    return (figs + [e for e in decision.entities if not any(e.lower() in f.lower() for f in figs)])[:4]


# ----------------------------------------------------------------------------- main entry
def _mode_overrides(profile: dict) -> dict:
    """Settings for one answer: the mode profile's settings on top of the global settings (which already include
    the active retrieval_params component, see config.get_settings)."""
    return dict(profile.get("settings") or {})


async def answer(session_id: str | None, user_message: str | None = None, *, mode: str | None = None,
                 parent_id: int | None = None, edit_of: int | None = None,
                 regenerate_of: int | None = None) -> AsyncIterator[dict]:
    """Answer one message as an event stream. v1 callers (eval runner) pass (session_id, text) only: the turn is
    appended to the active branch in Swift mode."""
    profile = get_mode(mode)
    mode_id = str(profile.get("id") or DEFAULT_MODE)
    if session_id is None and (edit_of or regenerate_of or parent_id):
        session_id = await session_of_message(edit_of or regenerate_of or parent_id)
        if session_id is None:
            yield _ev("error", {"message": "Unknown message id."})
            yield _ev("done", {"message_id": None, "out_of_context": False})
            return
    try:
        sid = str(uuid.UUID(str(session_id))) if session_id else str(uuid.uuid4())
    except ValueError:
        sid = str(uuid.uuid4())
    ctx: dict[str, Any] = {"mode": mode_id, "profile": profile,
                           "branch": {"parent_id": parent_id, "edit_of": edit_of, "regenerate_of": regenerate_of}}
    token = request_overrides(_mode_overrides(profile))
    set_call_meta({"mode": mode_id})
    try:
        from app import versioning
        update_call_meta(components=versioning.active_versions())
    except Exception:  # noqa: BLE001
        pass
    try:
        if session_busy(sid):
            yield _ev("status", {"stage": "waiting", "detail": "waiting for the previous answer in this conversation"})
        async with session_lock(sid):
            async with aclosing(_answer(sid, user_message or "", ctx)) as inner:
                async for ev in inner:
                    yield ev
        task: asyncio.Task | None = ctx.get("title_task")
        if task is not None:
            try:  # shield: a disconnect cancels this wait, not the title generation / persistence
                title = await asyncio.wait_for(asyncio.shield(task), timeout=TITLE_WAIT_S)
            except Exception:  # noqa: BLE001 - timeout etc.: the title is still persisted by the task
                title = None
            if title:
                yield _ev("session", {"session_id": sid, "title": title})
    finally:
        reset_request_overrides(token)
        set_call_meta(None)


async def _answer(sid: str, user_message: str, ctx: dict) -> AsyncIterator[dict]:
    s = settings()
    calls = begin_usage_scope()
    profile: dict = ctx.get("profile") or {}
    mode_id: str = ctx.get("mode") or DEFAULT_MODE
    mset: dict = profile.get("settings") or {}
    timings: dict[str, int] = {}
    t_all = time.perf_counter()
    mark = time.perf_counter()

    def lap(stage: str) -> None:
        nonlocal mark
        now = time.perf_counter()
        timings[stage] = timings.get(stage, 0) + int((now - mark) * 1000)
        mark = now

    question = (user_message or "").strip()
    message_id: int | None = None
    br = {k: v for k, v in (ctx.get("branch") or {}).items() if v is not None}
    rows, leaf = await _load_tree(sid)
    try:
        target = resolve_branch(rows, leaf, **br)
    except BranchError as e:
        yield _ev("error", {"message": str(e)})
        yield _ev("done", {"message_id": None, "out_of_context": False})
        return
    if target["question"] is not None:          # regenerate: answer the stored question again
        question = target["question"].strip()
    if not question:
        yield _ev("error", {"message": "Empty question."})
        yield _ev("done", {"message_id": None, "out_of_context": False})
        return
    rq = _focus_query(question)          # what routing / retrieval look at (the question itself if short)
    decision = RouteDecision(standalone_query=rq, mode=mode_id)
    ctx["added"] = 1 if target["user_id"] is not None else 2
    try:
        title, is_new = await _ensure_session(sid, question)
        memory = SessionMemory.load(sid)
        if not memory.title:
            memory.title = title
        ctx_parent = target["parent"]
        if ctx_parent != leaf:
            # not a continuation of the active branch: the context is the path root -> new message's parent only
            from app import db as _db
            turns = path_turns(_db.path_to(rows, ctx_parent)) if ctx_parent is not None else []
            if memory.rebuild_from_path(turns):
                yield _ev("status", {"stage": "compacting", "detail": "rebuilding the memory of this branch"})
                await memory.compact(get_llm())
            memory.save()
        if target["user_id"] is None:
            user_mid = await _insert_message(sid, "user", question, ctx_parent, mode_id)
        else:
            user_mid = target["user_id"]
        message_id = await _insert_message(sid, "assistant", "", user_mid, mode_id)
        if message_id is not None:
            await _set_leaf(sid, message_id)
        # done: parent_id = the answer's parent (the user message); session: active_leaf_id = the user message
        # (the UI shows it at once), parent_id = the user message's parent (where its siblings hang)
        ctx["done_extra"] = {"user_message_id": user_mid, "parent_id": user_mid, "mode": mode_id,
                             "active_leaf_id": message_id}
        yield _ev("session", {"session_id": sid, "title": title, "active_leaf_id": user_mid,
                              "user_message_id": user_mid, "parent_id": ctx_parent, "mode": mode_id})
        await _load_pdf_paths()
        lap("setup")

        # ---- route (fast path: retrieve first, skip the LLM router when the scores settle it)
        yield _ev("status", {"stage": "routing", "detail": "understanding the question"})
        retriever = _retriever()
        memory_ctx = memory.router_context()
        pre_hits: list | None = None
        routed: RouteDecision | None = None
        if (getattr(s, "fast_route", True) and not GREETING_RE.match(rq)
                and not router_needed(rq, memory_ctx)):
            tq = topic_query(rq)         # retrieve the TOPIC (analogy / citation / injection wrappers removed)
            yield _ev("status", {"stage": "retrieving", "detail": tq})
            pre_hits = await asyncio.to_thread(retriever.search, tq, sub_queries=None,
                                               session_id=sid, message_id=message_id)
            lap("retrieval")
            routed = fast_route(rq, pre_hits, min_score=getattr(s, "fast_route_min_score", 0.9),
                                ooc_below=s.ooc_threshold)
        if routed is None:
            routed = await route(rq, memory_ctx, session_id=sid, message_id=message_id)
            lap("routing")
        decision = routed
        decision.mode = mode_id
        yield _ev("decision", decision.model_dump())

        if decision.intent == "greeting":
            from app.llm.decisions import THANKS_RE
            reply = prompts.THANKS_REPLY if THANKS_RE.match(question) else prompts.GREETING_REPLY
            yield _ev("token", {"t": reply})
            yield _ev("answer", {"content": reply, "uncited": False})
            yield _ev("sources", [])
            yield _ev("images", [])
            await _finish_message(message_id, sid, reply, decision, [], [], False, added=ctx.get('added', 2))
            async for ev in _wrap_up(sid, question, reply, decision, [], memory, is_new, message_id, calls,
                                     timings, t_all, lap, ctx, title, ooc=False, remember=False):
                yield ev
            return

        # ---- retrieve
        q = topic_query(decision.standalone_query or rq)
        if pre_hits is not None and (decision.standalone_query or rq) == rq and not decision.sub_queries:
            hits = pre_hits
        else:
            yield _ev("status", {"stage": "retrieving", "detail": q + (f" (+{len(decision.sub_queries)} sub-queries)"
                                                                       if decision.sub_queries else "")})
            hits = await asyncio.to_thread(retriever.search, q, sub_queries=decision.sub_queries or None,
                                           session_id=sid, message_id=message_id)
            lap("retrieval")
        top = max([h.rerank_score or 0.0 for h in hits], default=0.0)
        yield _ev("status", {"stage": "reranking", "detail": f"{len(hits)} passages, best score {top:.2f}"})

        # ---- out-of-context gate (before any further retrieval work)
        missing = (await asyncio.to_thread(guardrails.missing_refs, question, retriever)
                   if FIG_REF_RE.search(question) else [])
        reason = f"best score {top:.2f}"
        ooc = (not hits or top < s.ooc_threshold or (not decision.in_scope and top < s.ooc_soft_threshold))
        if "jailbreak" in decision.rules or is_jailbreak_only(question):   # nothing but an injection request
            ooc, reason = True, "injection only"
        elif guardrails.only_missing_ref(question, missing):
            ooc, reason = True, f"{', '.join(missing)} does not exist"
        if ooc:
            async for ev in _emit_ooc(sid, question, decision, memory, is_new, message_id, calls, timings,
                                      t_all, lap, ctx, title, reason=reason):
                yield ev
            return

        extra = await asyncio.to_thread(retriever.expand_cross_refs, hits, 3 if decision.is_cross_reference else 1)
        extra = [h for h in extra if h.chunk_id not in {x.chunk_id for x in hits}]
        figs: list = []
        if decision.needs_images:
            figs = await asyncio.to_thread(lambda: retriever.figures_for(q, hits, 3, session_id=sid,
                                                                         message_id=message_id))
        # comparisons / cross-references: every sub-query (entity) gets its own best evidence
        cover: list = []
        cover_q = decision.sub_queries or (enumerated_items(question)
                                           if decision.answer_format == "table" or decision.intent == "comparison"
                                           else [])
        if cover_q and hasattr(retriever, "coverage_for") and not (
                FIG_REF_RE.search(question) and decision.intent == "figure_request"):
            cover = await asyncio.to_thread(lambda: retriever.coverage_for(
                cover_q, per_query=2 if len(cover_q) <= 3 else 1, session_id=sid, message_id=message_id))
        # "what does Figure X show and which section explains it": the figure + its explaining text
        lead: list = []
        targets = await asyncio.to_thread(_figure_targets, question, decision, hits, figs, retriever)
        if targets:
            sec_text = []
            if hasattr(retriever, "section_text_for"):
                sec_text = await asyncio.to_thread(lambda: retriever.section_text_for(
                    [t.chunk_id for t in targets], q, 2, session_id=sid, message_id=message_id))
            for t in targets:
                lead.append(t)
                lead += [x for x in sec_text if ((x.payload or {}).get("_link") or {}).get("from") == t.chunk_id]
            lead = _merge(lead, sec_text)
        lap("expansion")

        # weak passages invite spurious citations: keep those near the best score (at least 2)
        floor = max(0.5 * s.ooc_threshold, 0.25 * top)
        hits = [h for i, h in enumerate(hits) if i < 2 or (h.rerank_score or 0.0) >= floor]
        # the figure asked about (+ its text) first, then the strongest passages, one per compared entity,
        # requested figures, the rest
        ordered = _merge(lead, hits[:2], cover, figs[:2], hits[2:], extra, figs[2:])
        ordered = ordered[: max(s.final_k + 4, len(lead) + len(cover) + 2)]
        shown = decision.model_copy(update={"standalone_query": _prompt_question(decision.standalone_query or rq)})
        pages = {p for h in ordered for p in (_g(h, "pages") or [])}
        notes = guardrails.form_notes(decision, question, len(ordered), pages, _total_pages, missing)
        messages, prompt_hits = build_messages(_prompt_question(question), shown, ordered, memory, notes)
        lap("prompt")

        # ---- generate (hold back the first characters to catch "#OutofContext")
        yield _ev("status", {"stage": "generating", "detail": f"{len(prompt_hits)} sources"})
        out: dict[str, Any] = {}
        think = bool(mset.get("think"))
        think_budget = int(mset.get("think_budget_tokens") or 0)
        async for ev in _generate(messages, sid, message_id, out, think=think, think_budget=think_budget):
            yield ev
        raw = out["raw"]
        lap("generation")
        if (raw.lower().startswith(OOC_PREFIX) or not raw) and getattr(s, "refusal_retry", True) and \
                guardrails.should_retry_refusal(decision, top, getattr(s, "refusal_retry_min_score", 0.85),
                                                missing, question):
            # both signals are needed to refuse: retrieval of the topic is strong and the router kept it in
            # scope, so the refusal is about the question's FORM -> one retry that rules #OutofContext out
            yield _ev("status", {"stage": "generating", "detail": "retrying: sources cover this topic"})
            decision.rules.append("refusal_retry")
            retry = [messages[0], {"role": "user", "content": messages[1]["content"] + "\nNOTE: " + prompts.RETRY_NOTE}]
            out = {}
            async for ev in _generate(retry, sid, message_id, out, think=think, think_budget=think_budget):
                yield ev
            raw = out["raw"]
            lap("generation")
        if raw.lower().startswith(OOC_PREFIX) or not raw:
            async for ev in _emit_ooc(sid, question, decision, memory, is_new, message_id, calls, timings,
                                      t_all, lap, ctx, title, reason="model declined", streamed=False):
                yield ev
            return

        # ---- output guardrails: prompt leaks, injected canaries
        raw, leaked = guardrails.strip_prompt_leak(raw)
        if leaked:
            decision.rules.append("prompt_leak_removed")
        _, removed = strip_injection(question)
        if removed:
            raw = guardrails.strip_injected_tokens(raw, removed, " ".join(_g(h, "content") or "" for h in prompt_hits))
        if not raw.strip() or (leaked and not re.search(r"\[\d", raw)):
            async for ev in _emit_ooc(sid, question, decision, memory, is_new, message_id, calls, timings,
                                      t_all, lap, ctx, title, reason="prompt leak", streamed=True):
                yield ev
            return

        # ---- validate citations
        yield _ev("status", {"stage": "validating", "detail": "checking citations"})
        clean, used = citations.validate(raw, prompt_hits)
        if getattr(s, "grounding_check", False) and used:
            yield _ev("status", {"stage": "validating", "detail": "checking the answer against its sources"})
            g = await _grounding_score(clean, used, sid, message_id, prompt_hits)
            decision.grounding = {"score": g["score"], "unsupported": g["unsupported"], "revised": False}
            min_score = int(mset.get("revision_min_grounding") or 3)
            if g["ok"] and g["score"] < min_score:
                if mset.get("self_revision"):
                    yield _ev("status", {"stage": "revising",
                                         "detail": f"grounding {g['score']}/5: revising the answer once"})
                    revised = await _revise(messages, raw, g, sid, message_id)
                    if revised and not _acceptable_revision(clean, revised):
                        decision.rules.append("revision_rejected")
                        revised = None
                    if revised:
                        clean2, used2 = citations.validate(revised, prompt_hits)
                        if used2:
                            clean, used = clean2, used2
                            decision.grounding["revised"] = True
                            decision.rules.append("self_revised")
                else:
                    clean = clean + _grounding_caution()
            lap("grounding")
        images = _images(decision, used, prompt_hits, figs)
        lap("validation")
        yield _ev("answer", {"content": clean, "uncited": bool(getattr(used, "uncited", False)),
                             "grounding": decision.grounding or None})
        yield _ev("sources", list(used))
        yield _ev("images", images)
        await _finish_message(message_id, sid, clean, decision, list(used), images, False, added=ctx.get('added', 2))
        async for ev in _wrap_up(sid, question, clean, decision, images, memory, is_new, message_id, calls,
                                 timings, t_all, lap, ctx, title, ooc=False):
            yield ev
    except Exception as e:  # noqa: BLE001
        log.exception("pipeline failed")
        msg = f"{type(e).__name__}: {e}"
        yield _ev("error", {"message": msg})
        if message_id is not None:
            await db_execute("UPDATE messages SET content=%s, decisions=%s::jsonb WHERE message_id=%s",
                             (f"[error] {msg[:500]}", json.dumps(decision.model_dump()), message_id))
        usage = {**summarize_usage(calls), "timings_ms": dict(timings),
                 "mode": ctx.get("mode") or DEFAULT_MODE}
        await _save_usage(message_id, usage)
        yield _ev("usage", usage)
        yield _ev("done", {"message_id": message_id, "out_of_context": False, **(ctx.get("done_extra") or {})})


async def _generate(messages: list[dict], sid: str, message_id: int | None, out: dict, *, think: bool = False,
                    think_budget: int = 0) -> AsyncIterator[dict]:
    """Stream one answer attempt as ``token`` events, holding back the first characters so that a refusal
    ("#OutofContext") is never shown; out["raw"] = the full text (a trailing #OutofContext removed).
    ``think``: the model's thinking is streamed as ``thinking`` events and never enters ``raw``; if the thinking
    used up the whole token budget (no answer text), the answer is generated once more without thinking."""
    async for ev in _generate_once(messages, sid, message_id, out, think=think, think_budget=think_budget):
        yield ev
    if think and not out["raw"] and out.get("done_reason") == "length":
        yield _ev("status", {"stage": "generating", "detail": "thinking used the whole budget: answering directly"})
        async for ev in _generate_once(messages, sid, message_id, out, think=False, think_budget=0):
            yield ev


async def _generate_once(messages: list[dict], sid: str, message_id: int | None, out: dict, *, think: bool,
                         think_budget: int) -> AsyncIterator[dict]:
    s = settings()
    parts: list[str] = []
    held, released = "", False
    llm = get_llm()
    num_predict = s.max_answer_tokens + (max(0, think_budget) if think else 0)
    async with aclosing(llm.stream(messages, purpose="answer", options={"num_predict": num_predict},
                                   session_id=sid, message_id=message_id, think=think)) as stream:
        async for tok in stream:
            if isinstance(tok, ThinkingToken):
                yield _ev("thinking", {"t": str(tok)})
                continue
            parts.append(tok)
            if released:
                yield _ev("token", {"t": tok})
                continue
            held += tok
            probe = held.lstrip().lower()
            if len(probe) >= len(OOC_PREFIX) or not OOC_PREFIX.startswith(probe[: len(OOC_PREFIX)]):
                if probe.startswith(OOC_PREFIX):
                    continue  # swallow: will be normalised to exactly #OutofContext
                released = True
                yield _ev("token", {"t": held})
    if not released and held and not held.lstrip().lower().startswith(OOC_PREFIX):
        yield _ev("token", {"t": held})
    out["done_reason"] = getattr(llm, "last_done_reason", None)
    raw = "".join(parts).strip()
    out["raw"] = re.sub(r"\s*#OutofContext\W*$", "", raw, flags=re.I)  # sometimes appended to a real answer


def _images(decision: RouteDecision, used: list[dict], prompt_hits: list, figs: list) -> list[dict]:
    """Images only when the user asked to see something (decision.needs_images): the figures the answer cites, else
    the best searched figure(s) (only those close to the best figure score, so unrelated figures are not shown).
    A text question whose answer merely cites a figure description gets no image."""
    if not decision.needs_images:
        return []
    by_id = {h.chunk_id: h for h in [*prompt_hits, *figs]}
    chosen = [by_id[src["chunk_id"]] for src in used if src.get("chunk_type") == "figure" and src["chunk_id"] in by_id]
    if decision.needs_images and not chosen and figs:
        best = max(f.rerank_score or 0.0 for f in figs)
        chosen = [f for f in figs if (f.rerank_score or 0.0) >= max(0.1, 0.6 * best)][:2]
    seen, out = set(), []
    for h in chosen:
        if h.chunk_id not in seen and h.chunk_type == "figure":
            seen.add(h.chunk_id)
            out.append(citations.to_image(h))
    return out[:3]


def _grounding_question() -> str:
    try:
        return prompts.grounding_prompts()[0]
    except Exception:  # noqa: BLE001 - older prompts module
        return prompts.GROUNDING_QUESTION


def _grounding_caution() -> str:
    try:
        return prompts.grounding_prompts()[1]
    except Exception:  # noqa: BLE001
        return prompts.GROUNDING_CAUTION


GROUNDING_LEVELS = ["1", "2", "3", "4", "5"]
REVISION_NOTE = ("REVISION: a check of your answer against the SOURCES flagged these statements as not supported by "
                 "the SOURCES{detail}. Return the same answer with ONLY those statements corrected (using the SOURCES, "
                 "with [n]) or removed. Keep every other sentence, the structure, the requested format and the "
                 "citations as they are. The answer must still answer the question: never replace it with a refusal "
                 "or with a statement that the sources do not explain the topic. Add no preamble and do not mention "
                 "this revision.")
REFUSAL_LIKE_RE = re.compile(r"\b(sources?|textbook|passages?)\s+(do(es)?\s+not|don'?t|never)\s+(explain|describe|"
                             r"discuss|state|mention|cover|address|provide|contain)", re.I)


def _acceptable_revision(original: str, revised: str) -> bool:
    """A revision may drop unsupported sentences, not the answer: reject it when it is much shorter than the
    original or reads like a refusal that the original was not."""
    if len(revised) < 0.6 * len(original):
        return False
    return not (REFUSAL_LIKE_RE.search(revised[:400]) and not REFUSAL_LIKE_RE.search(original[:400]))


def _grounding_sources(used: list[dict], prompt_hits: list | None, budget_tokens: int) -> str:
    """The FULL text of every cited passage (the 320-char UI snippets made most claims look unsupported), each
    trimmed to its share of the budget."""
    by_id = {_g(h, "chunk_id"): h for h in prompt_hits or []}
    per = max(200, budget_tokens // max(1, len(used)))
    blocks = []
    for u in used:
        hit = by_id.get(u.get("chunk_id"))
        body = ((_g(hit, "content") if hit is not None else None) or u.get("snippet") or "").strip()
        if count_tokens(body) > per:
            body = body[: per * 3].rsplit(" ", 1)[0] + " …"
        blocks.append(f"[{u['n']}] {body}")
    return "\n\n".join(blocks)


async def _grounding_score(answer_text: str, used: list[dict], sid: str, mid: int | None,
                           prompt_hits: list | None = None) -> dict:
    """Grounding of the answer in its cited sources, 1 (unsupported) .. 5 (every statement supported)."""
    s = settings()
    budget = max(1500, s.num_ctx - count_tokens(answer_text) - 1200)
    src = _grounding_sources(used, prompt_hits, budget)
    r = await decide(f"SOURCES:\n{src}\n\nANSWER:\n{answer_text}",
                     [Score(name="grounded", levels=GROUNDING_LEVELS, default="5", question=(
                         _grounding_question() + " Score 1-5: 5 = every statement supported, 4 = minor "
                         "unsupported details, 3 = some unsupported claims, 2 = mostly unsupported, 1 = unsupported "
                         "or contradicts the sources.")),
                      Text(name="unsupported", max_chars=400, question=(
                          "The statements of the ANSWER that the SOURCES do not support (short, separated by ';'), "
                          "or an empty string."))],
                     purpose="grounding_check", session_id=sid, message_id=mid)
    return {"ok": bool(r.get("_ok")), "score": int(r["grounded"]), "unsupported": (r.get("unsupported") or "").strip(),
            "confidence": r.get("grounded_confidence", 0.0)}


async def _revise(messages: list[dict], raw: str, g: dict, sid: str, mid: int | None) -> str | None:
    """One self-revision pass (no thinking). None if it failed / refused."""
    detail = f" ({g['unsupported']})" if g.get("unsupported") else ""
    msgs = [*messages, {"role": "assistant", "content": raw},
            {"role": "user", "content": REVISION_NOTE.format(detail=detail)}]
    try:
        text, _ = await get_llm().complete(msgs, purpose="answer_revision",
                                           options={"num_predict": settings().max_answer_tokens},
                                           session_id=sid, message_id=mid)
    except Exception as e:  # noqa: BLE001
        log.warning("self-revision failed: %s", e)
        return None
    text = re.sub(r"\s*#OutofContext\W*$", "", (text or "").strip(), flags=re.I)
    if not text or text.lower().startswith(OOC_PREFIX):
        return None
    return text


async def _emit_ooc(sid, question, decision, memory, is_new, message_id, calls, timings, t_all, lap, ctx, title, *,
                    reason: str, streamed: bool = False) -> AsyncIterator[dict]:
    yield _ev("status", {"stage": "validating", "detail": f"out of context ({reason})"})
    if not streamed:
        yield _ev("token", {"t": prompts.OOC})
    yield _ev("answer", {"content": prompts.OOC, "uncited": False})
    yield _ev("sources", [])
    yield _ev("images", [])
    await _finish_message(message_id, sid, prompts.OOC, decision, [], [], True, added=ctx.get('added', 2))
    async for ev in _wrap_up(sid, question, prompts.OOC, decision, [], memory, is_new, message_id, calls,
                             timings, t_all, lap, ctx, title, ooc=True):
        yield ev


async def _wrap_up(sid, question, final_text, decision, images, memory: SessionMemory, is_new, message_id,
                   calls, timings, t_all, lap, ctx, title, *, ooc: bool, remember: bool = True) -> AsyncIterator[dict]:
    s = settings()
    if remember:
        memory.add_turn(question, final_text, [] if ooc else _memory_entities(decision, images),
                        message_id=message_id)
        if memory.needs_compaction(s.memory_budget_tokens):
            yield _ev("status", {"stage": "compacting", "detail": "summarising older turns"})
            await memory.compact(get_llm())
    memory.save()
    await db_execute("UPDATE sessions SET memory_path=%s WHERE session_id=%s", (str(memory.path), sid))
    lap("memory")
    timings["total"] = int((time.perf_counter() - t_all) * 1000)
    usage = {**summarize_usage(calls), "timings_ms": dict(timings),
                 "mode": ctx.get("mode") or DEFAULT_MODE}
    await _save_usage(message_id, usage)
    if is_new and not ooc and decision.intent != "greeting":
        # the title call (0.5-40 s when the GPU is busy) runs after `done`, off the critical path
        ctx["title_task"] = asyncio.create_task(_title_job(question, sid, message_id, title))
    yield _ev("usage", usage)
    yield _ev("done", {"message_id": message_id, "out_of_context": ooc, "content": final_text,
                       **(ctx.get("done_extra") or {})})
