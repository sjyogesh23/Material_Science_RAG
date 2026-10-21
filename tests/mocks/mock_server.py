"""Mock NewRAG backend for frontend development.

Implements the API/SSE contract in docs/ARCHITECTURE.md with fake data so the UI can be
exercised without Postgres, Ollama or the retrieval stack.

    .venv/Scripts/python tests/mocks/mock_server.py [--port 8765]

Then open http://127.0.0.1:8765/ (chat) and /audit.
v2 (docs/FEATURES_V2.md 1-4): message tree (parent_id, siblings, switch), feedback, modes (deep streams
`thinking` events), session memory + compaction.
Triggers in the chat message:
    "france"  -> #OutofContext reply (no sources)
    "hello"   -> canned greeting (no citations)
    "fail"    -> error event mid-stream
Env MOCK_DOWN=ollama|postgres|model|chunks makes /api/health report that problem.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]  # project root (tests/mocks/mock_server.py)
STATIC = ROOT / "app" / "static"
PDF = ROOT / "data" / "textbooks" / "Materials Science and Engineering.pdf"
CHUNKS = ROOT / "chunks" / "chunks.jsonl"
DOC_ID = "doc_f49a86e05ec4"

app = FastAPI(title="NewRAG mock")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------- data from the real chunks
def load_chunks() -> dict[str, dict]:
    out = {}
    if CHUNKS.exists():
        with CHUNKS.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                out[r["chunk_id"]] = r
    return out


CHUNK = load_chunks()
_page_cache: dict[tuple[int, int], bytes] = {}
_size_cache: dict[int, tuple[float, float]] = {}


def page_size(page: int) -> tuple[float, float]:
    if page not in _size_cache:
        try:
            import pymupdf  # type: ignore
            with pymupdf.open(PDF) as doc:
                r = doc[page - 1].rect
                _size_cache[page] = (round(r.width, 2), round(r.height, 2))
        except Exception:
            _size_cache[page] = (612.0, 792.0)
    return _size_cache[page]


def make_source(n: int, chunk_id: str, score: float) -> dict:
    c = CHUNK.get(chunk_id)
    if not c:  # fallback if chunks.jsonl missing
        return {"n": n, "chunk_id": chunk_id, "doc_id": DOC_ID, "doc_title": "Materials Science and Engineering",
                "chunk_type": "text", "header": "Chapter 3 > 3.4 Dislocation Defects (1D)",
                "section_path": ["Chapter 3: Defects in Crystalline Materials", "3.4 Dislocation Defects (1D)"],
                "pages": [58], "snippet": "A dislocation is a linear or one dimensional line defect.", "score": score,
                "highlights": [{"page": 58, "bboxes": [[72, 180, 540, 256]]}], "page_size": {"58": [612, 792]}, "figure": None}
    hl: dict[int, list] = {}
    for loc in c.get("locations") or []:
        if loc.get("bbox"):
            hl.setdefault(loc["page"], []).append(loc["bbox"])
    pages = c.get("pages") or []
    return {
        "n": n, "chunk_id": chunk_id, "doc_id": c["doc_id"], "doc_title": c["doc_title"],
        "chunk_type": c["chunk_type"], "header": c["header"], "section_path": c["section_path"],
        "pages": pages, "snippet": c["content"][:260].rsplit(" ", 1)[0] + "…", "score": score,
        "highlights": [{"page": p, "bboxes": b} for p, b in hl.items()],
        "page_size": {str(p): list(page_size(p)) for p in pages},
        "figure": c.get("figure"),
    }


def make_image(chunk_id: str) -> dict | None:
    c = CHUNK.get(chunk_id)
    if not c or not c.get("figure"):
        return None
    cap = c["content"].split("\n", 1)[0]
    return {"chunk_id": chunk_id, "url": f"/api/figure/{chunk_id}", "caption": cap,
            "figure_number": c["figure"].get("figure_number"), "page": c["page_start"],
            "description": c["content"].split("Figure description:", 1)[-1].strip()[:300]}


ANSWER = r"""A **dislocation** is a line defect: a one-dimensional region of the crystal around which atoms are misaligned [1]. Two limiting types are distinguished by how the Burgers vector $\vec{b}$ relates to the dislocation line $\vec{t}$ [1, 2].

| Type | Burgers vector vs. line | Motion |
|---|---|---|
| Edge | $\vec{b} \perp \vec{t}$ | glides in the plane containing $\vec{b}$ and $\vec{t}$ [1] |
| Screw | $\vec{b} \parallel \vec{t}$ | can cross-slip onto any plane containing the line [2] |

The elastic strain energy stored per unit length scales with the square of the Burgers vector [1]:

$$E \approx \alpha\, G\, b^{2}, \qquad \alpha \approx 0.5\text{–}1$$

so dislocations with the shortest $b$ (close-packed directions) are favoured [2]. Grain boundaries, by contrast, are 2D defects that also raise the surface energy $\gamma$ of the material [2].

```python
# shear modulus G of Fe ≈ 82 GPa, b ≈ 0.248 nm
E = 0.5 * 82e9 * (0.248e-9) ** 2   # J/m
```
"""

GREETING = "Hello! Ask me anything from *Materials Science and Engineering*: crystal structures, defects, diffusion, phase diagrams or mechanical properties."


# ---------------------------------------------------------------- in-memory state (v2: message tree)
SESSIONS: dict[str, dict] = {}
MESSAGES: dict[str, list[dict]] = {}     # every message of a session (all branches), creation order
MSG: dict[int, dict] = {}                # message_id -> message
MEMORY_COMPACTED: dict[str, bool] = {}
_msg_id = 1000

MODES = [
    {"id": "swift", "name": "Swift", "tagline": "Time over effect",
     "description": "qwen3:4b-instruct, fast routing, 5 passages, up to 700 answer tokens. Usually 5-20 s.",
     "settings": {"chat_model": "qwen3:4b-instruct-2507-q4_K_M", "fast_route": True, "rerank_top_n": 12,
                  "final_k": 5, "max_answer_tokens": 700, "grounding_check": False, "thinking": False}},
    {"id": "deep", "name": "Deep", "tagline": "Effect over time",
     "description": "qwen3.5:9b with reasoning, full router, 8 passages, grounding check and one self-revision. "
                    "About 1-3 min per answer on this laptop.",
     "settings": {"chat_model": "qwen3.5:9b", "fast_route": False, "rerank_top_n": 30, "final_k": 8,
                  "max_answer_tokens": 1400, "grounding_check": True, "thinking": True}},
]


def new_session(title: str | None = None, when: datetime | None = None) -> dict:
    sid = str(uuid.uuid4())
    t = when or now()
    SESSIONS[sid] = {"session_id": sid, "title": title or "New conversation", "created_at": t.isoformat(),
                     "updated_at": t.isoformat(), "message_count": 0, "active_leaf_id": None}
    MESSAGES[sid] = []
    return SESSIONS[sid]


def add_msg(sid: str, role: str, content: str, parent_id="default", **kw) -> dict:
    """parent_id="default" = append to the session's active leaf."""
    global _msg_id
    _msg_id += 1
    if parent_id == "default":
        parent_id = SESSIONS[sid]["active_leaf_id"]
    m = {"message_id": _msg_id, "session_id": sid, "parent_id": parent_id, "role": role, "content": content,
         "sources": kw.get("sources") or [], "images": kw.get("images") or [],
         "decisions": kw.get("decisions"), "usage": kw.get("usage"),
         "out_of_context": kw.get("out_of_context", False), "feedback": None,
         "mode": kw.get("mode") if role == "assistant" else None, "created_at": now().isoformat()}
    MESSAGES[sid].append(m)
    MSG[m["message_id"]] = m
    SESSIONS[sid]["active_leaf_id"] = m["message_id"]
    SESSIONS[sid]["message_count"] += 1
    SESSIONS[sid]["updated_at"] = now().isoformat()
    return m


def siblings(m: dict) -> list[int]:
    return [x["message_id"] for x in MESSAGES[m["session_id"]]
            if x["parent_id"] == m["parent_id"] and x["role"] == m["role"]]


def path(sid: str) -> list[dict]:
    out, cur = [], SESSIONS[sid]["active_leaf_id"]
    while cur is not None:
        m = MSG[cur]
        out.append(m)
        cur = m["parent_id"]
    out.reverse()
    rows = []
    for m in out:
        sib = siblings(m)
        rows.append({**{k: v for k, v in m.items() if k != "session_id"},
                     "sibling_ids": sib, "sibling_index": sib.index(m["message_id"]), "sibling_count": len(sib)})
    return rows


def newest_leaf(mid: int) -> int:
    m = MSG[mid]
    kids = [x for x in MESSAGES[m["session_id"]] if x["parent_id"] == mid]
    if not kids:
        return mid
    leaves = [newest_leaf(k["message_id"]) for k in kids]
    return max(leaves, key=lambda i: (MSG[i]["created_at"], i))


def seed() -> None:
    s = new_session("Edge vs screw dislocations", now() - timedelta(hours=2))
    sid = s["session_id"]
    add_msg(sid, "user", "What is the difference between edge and screw dislocations?")
    add_msg(sid, "assistant", ANSWER, mode="swift",
            sources=[make_source(1, "doc_f49a86e05ec4:c00104", 0.91), make_source(2, "doc_f49a86e05ec4:c00107", 0.74)],
            images=[i for i in [make_image("doc_f49a86e05ec4:c00005")] if i],
            usage={"prompt_tokens": 12950, "completion_tokens": 500, "total": 13450})
    s2 = new_session("Capital of France", now() - timedelta(days=1, hours=3))
    add_msg(s2["session_id"], "user", "What is the capital of France?")
    add_msg(s2["session_id"], "assistant", "#OutofContext", out_of_context=True, mode="swift")
    new_session("Bragg's law derivation", now() - timedelta(days=6))


seed()

SETTINGS = {
    "chat_model": "qwen3:4b-instruct-2507-q4_K_M", "router_model": "qwen3:4b-instruct-2507-q4_K_M",
    "num_ctx": 8192, "temperature": 0.1, "max_answer_tokens": 900,
    "embed_model": "Qwen/Qwen3-Embedding-0.6B", "embed_device": "cpu",
    "rerank_model": "Qwen/Qwen3-Reranker-0.6B", "rerank_device": "cpu",
    "top_k_dense": 40, "top_k_bm25": 40, "rrf_k": 60, "rerank_top_n": 30, "final_k": 6,
    "ooc_threshold": 0.2, "ooc_soft_threshold": 0.45,
    "memory_budget_tokens": 1500, "memory_keep_turns": 6, "grounding_check": False,
}
KEYS: dict[str, dict] = {"HF_TOKEN": {"value": "hf_exampleexampleexample1234", "note": "Model downloads", "updated_at": now().isoformat()}}


def mask(v: str) -> str:
    return (v[:3] + "•" * 8 + v[-4:]) if len(v) > 8 else "•" * 8


def fake_calls() -> list[dict]:
    rnd = random.Random(7)
    rows = []
    sids = list(SESSIONS)
    purposes = [("router", "ollama", 520, 90, 900), ("answer", "ollama", 2600, 380, 7400),
                ("title", "ollama", 180, 12, 400), ("memory_compaction", "ollama", 1600, 260, 3800),
                ("grounding_check", "ollama", 1900, 40, 1600), ("embed_query", "local", 24, 0, 80),
                ("rerank", "local", 9800, 0, 2100)]
    for i in range(230):
        t = now() - timedelta(minutes=rnd.randint(1, 60 * 24 * 28))
        p, prov, pt, ct, ms = rnd.choice(purposes)
        ptok = int(pt * rnd.uniform(0.6, 1.4))
        ctok = int(ct * rnd.uniform(0.5, 1.5))
        ok = rnd.random() > 0.03
        model = {"embed_query": SETTINGS["embed_model"], "rerank": SETTINGS["rerank_model"]}.get(p, SETTINGS["chat_model"])
        rows.append({"id": 5000 - i, "session_id": rnd.choice(sids), "message_id": None, "purpose": p,
                     "provider": prov, "model": model, "prompt_tokens": ptok, "completion_tokens": ctok,
                     "total_tokens": ptok + ctok, "duration_ms": int(ms * rnd.uniform(0.6, 1.6)), "ok": ok,
                     "error": None if ok else "ReadTimeout: Ollama did not answer within 120 s", "meta": {},
                     "created_at": t.isoformat()})
    rows.append({"id": 1, "session_id": None, "message_id": None, "purpose": "ingest_embed", "provider": "local",
                 "model": SETTINGS["embed_model"], "prompt_tokens": 112_400, "completion_tokens": 0,
                 "total_tokens": 112_400, "duration_ms": 94_000, "ok": True, "error": None, "meta": {},
                 "created_at": (now() - timedelta(days=12)).isoformat()})
    return sorted(rows, key=lambda r: r["created_at"], reverse=True)


CALLS = fake_calls()


# ---------------------------------------------------------------- pages
@app.get("/")
def chat_page():
    return FileResponse(STATIC / "chat.html")


@app.get("/audit")
def audit_page():
    return FileResponse(STATIC / "audit.html")


@app.get("/api/health")
def health():
    down = os.environ.get("MOCK_DOWN", "")
    return {"postgres": down != "postgres", "ollama": down != "ollama", "chat_model_present": down != "model",
            "chunks": 0 if down == "chunks" else len(CHUNK) or 375, "embedder_loaded": True}


# ---------------------------------------------------------------- sessions
@app.get("/api/sessions")
def list_sessions():
    return sorted(SESSIONS.values(), key=lambda s: s["updated_at"], reverse=True)


@app.post("/api/sessions")
def create_session():
    s = new_session()
    return {"session_id": s["session_id"], "title": s["title"]}


@app.patch("/api/sessions/{sid}")
async def rename_session(sid: str, request: Request):
    if sid not in SESSIONS:
        raise HTTPException(404, "Session not found")
    body = await request.json()
    SESSIONS[sid]["title"] = str(body.get("title", "")).strip() or SESSIONS[sid]["title"]
    return SESSIONS[sid]


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    if sid not in SESSIONS:
        raise HTTPException(404, "Session not found")
    SESSIONS.pop(sid)
    MESSAGES.pop(sid, None)
    return {"ok": True}


@app.get("/api/sessions/{sid}/messages")
def session_messages(sid: str):
    if sid not in SESSIONS:
        raise HTTPException(404, "Session not found")
    return path(sid)


@app.post("/api/sessions/{sid}/switch")
async def switch_branch(sid: str, request: Request):
    if sid not in SESSIONS:
        raise HTTPException(404, "Session not found")
    body = await request.json()
    mid = body.get("message_id")
    if mid not in MSG or MSG[mid]["session_id"] != sid:
        raise HTTPException(404, "Message not found in this session")
    SESSIONS[sid]["active_leaf_id"] = newest_leaf(mid)
    MEMORY_COMPACTED[sid] = False
    return path(sid)


def memory_markdown(sid: str) -> str:
    turns = path(sid)
    lines = [f"# Session memory: {SESSIONS[sid]['title']}", "", "## Summary", ""]
    if MEMORY_COMPACTED.get(sid):
        n = len([t for t in turns if t["role"] == "user"])
        lines.append(f"The student asked {n} question(s) about line defects; answers compared **edge** and "
                     "**screw** dislocations using the Burgers vector $\\vec b$ [1].")
        lines += ["", "## Entities", "", "- dislocation, Burgers vector", ""]
        return "\n".join(lines)
    lines.append("_No summary yet: all turns fit in the budget._")
    lines += ["", "## Recent turns", ""]
    for t in turns:
        who = "User" if t["role"] == "user" else "Assistant"
        text = t["content"].replace("\n", " ")
        lines.append(f"- **{who}:** {text[:220]}{'…' if len(text) > 220 else ''}")
    lines += ["", "## Entities", "", "- dislocation (edge, screw)", "- Burgers vector", "- grain boundary", ""]
    return "\n".join(lines)


def memory_payload(sid: str) -> dict:
    md = memory_markdown(sid)
    return {"markdown": md, "tokens": int(len(md.split()) * 1.35), "budget": SETTINGS["memory_budget_tokens"],
            "path": f"E:/NewRAG/memory/sessions/{sid}.md"}


@app.get("/api/sessions/{sid}/memory")
def session_memory(sid: str):
    if sid not in SESSIONS:
        raise HTTPException(404, "Session not found")
    return memory_payload(sid)


@app.post("/api/sessions/{sid}/memory/compact")
async def compact_memory(sid: str):
    if sid not in SESSIONS:
        raise HTTPException(404, "Session not found")
    before = memory_payload(sid)["tokens"]
    await asyncio.sleep(0.6)
    MEMORY_COMPACTED[sid] = True
    return {**memory_payload(sid), "before_tokens": before}


@app.get("/api/modes")
def list_modes():
    return MODES


@app.post("/api/messages/{mid}/feedback")
async def feedback(mid: int, request: Request):
    if mid not in MSG:
        raise HTTPException(404, "Message not found")
    m = MSG[mid]
    if m["role"] != "assistant":
        raise HTTPException(422, "Feedback is only for answers")
    body = await request.json()
    rating = body.get("rating")
    if rating not in (-1, 0, 1):
        raise HTTPException(422, "rating must be -1, 0 or 1")
    comment = body.get("comment")
    comment = (str(comment).strip()[:1000] or None) if comment else None
    m["feedback"] = None if rating == 0 else {"rating": rating, "comment": comment}
    return {"message_id": mid, "rating": rating}


# ---------------------------------------------------------------- chat (SSE)
def ev(event: str, data) -> ServerSentEvent:
    return ServerSentEvent(event=event, data=data)


class ChatIn(BaseModel):
    session_id: str | None = None
    message: str | None = None
    mode: str | None = None
    parent_id: int | None = None
    edit_of: int | None = None
    regenerate_of: int | None = None


THINKING = ("The student asks about {q!r}. The retrieved passages [1] and [2] cover dislocations. "
            "I should define a dislocation first, then contrast edge and screw types by the angle between the "
            "Burgers vector and the line direction, then give the strain-energy relation E ~ G b^2. "
            "A table makes the comparison easy to scan. Check: every claim needs a citation; page 58 has the "
            "definition. I will not mention Hall-Petch since it is not in the textbook.")
_regen_count = 0


@app.post("/api/chat", response_class=EventSourceResponse)
async def chat(body: ChatIn, request: Request):
    global _regen_count
    sid = body.session_id
    mode = body.mode if body.mode in ("swift", "deep") else "swift"
    if not sid or sid not in SESSIONS:
        sid = new_session()["session_id"]
    s = SESSIONS[sid]
    variant = ""
    if body.regenerate_of is not None:
        old = MSG.get(body.regenerate_of)
        if not old or old["role"] != "assistant" or old["session_id"] != sid:
            yield ev("error", {"message": "Cannot regenerate: answer not found."})
            yield ev("done", {"message_id": None, "out_of_context": False})
            return
        user = MSG[old["parent_id"]]
        msg = user["content"]
        _regen_count += 1
        variant = f"_Regenerated answer ({_regen_count})._ "
    elif body.edit_of is not None:
        old = MSG.get(body.edit_of)
        if not old or old["role"] != "user" or old["session_id"] != sid:
            yield ev("error", {"message": "Cannot edit: message not found."})
            yield ev("done", {"message_id": None, "out_of_context": False})
            return
        msg = (body.message or "").strip()
        user = add_msg(sid, "user", msg, parent_id=old["parent_id"])
    else:
        msg = (body.message or "").strip()
        parent = body.parent_id if body.parent_id in MSG else "default"
        user = add_msg(sid, "user", msg, parent_id=parent)
    if s["title"] == "New conversation":
        s["title"] = (msg[:48] + "…") if len(msg) > 48 else msg
    yield ev("session", {"session_id": sid, "title": s["title"], "active_leaf_id": user["message_id"]})

    low = msg.lower()
    ooc = "france" in low
    greeting = low.startswith(("hello", "hi ", "hey"))
    yield ev("status", {"stage": "routing", "detail": None})
    await asyncio.sleep(0.4)
    decision = {"in_scope": not ooc, "in_scope_confidence": 0.12 if ooc else 0.93,
                "intent": "greeting" if greeting else "comparison", "needs_images": "figure" in low or not ooc,
                "answer_format": "table" if "table" in low else "prose", "is_cross_reference": False,
                "standalone_query": msg, "sub_queries": []}
    yield ev("decision", decision)
    by_purpose = {"router": {"prompt_tokens": 540, "completion_tokens": 88}}
    if not greeting:
        yield ev("status", {"stage": "retrieving", "detail": None})
        await asyncio.sleep(0.5)
        yield ev("status", {"stage": "reranking", "detail": "30 candidates" if mode == "deep" else "12 candidates"})
        await asyncio.sleep(0.5)
        by_purpose["embed_query"] = {"prompt_tokens": 22, "completion_tokens": 0}
        by_purpose["rerank"] = {"prompt_tokens": 9600, "completion_tokens": 0}

    if ooc:
        text, sources, images = "#OutofContext", [], []
    elif greeting:
        text, sources, images = GREETING, [], []
    else:
        text = variant + ANSWER
        sources = [make_source(1, "doc_f49a86e05ec4:c00104", 0.91), make_source(2, "doc_f49a86e05ec4:c00107", 0.74)]
        images = [i for i in (make_image("doc_f49a86e05ec4:c00005"), make_image("doc_f49a86e05ec4:c00008")) if i]

    yield ev("status", {"stage": "generating", "detail": None})
    if mode == "deep" and not ooc:
        for w in THINKING.format(q=msg[:60]).split(" "):
            if await request.is_disconnected():
                return
            yield ev("thinking", {"t": w + " "})
            await asyncio.sleep(0.03)
    pieces = [text] if ooc else [w + " " for w in text.split(" ")]
    if not ooc:
        pieces[-1] = pieces[-1].rstrip(" ")
    for i, p in enumerate(pieces):
        if await request.is_disconnected():
            return
        if "fail" in low and i == 25:
            yield ev("error", {"message": "Ollama stopped responding (mock error)."})
            yield ev("done", {"message_id": None, "user_message_id": user["message_id"], "out_of_context": False})
            return
        yield ev("token", {"t": p})
        await asyncio.sleep(0.025)
    if not ooc and not greeting:
        yield ev("status", {"stage": "validating", "detail": None})
        await asyncio.sleep(0.3)
        by_purpose["answer"] = {"prompt_tokens": 2810, "completion_tokens": 412}
    yield ev("sources", sources)
    yield ev("images", images)
    pt = sum(v["prompt_tokens"] for v in by_purpose.values())
    ct = sum(v["completion_tokens"] for v in by_purpose.values())
    usage = {"by_purpose": by_purpose, "prompt_tokens": pt, "completion_tokens": ct, "total": pt + ct}
    yield ev("usage", usage)
    m = add_msg(sid, "assistant", text, parent_id=user["message_id"], sources=sources, images=images,
                out_of_context=ooc, mode=mode, decisions=decision, usage=usage)
    yield ev("done", {"message_id": m["message_id"], "user_message_id": user["message_id"],
                      "parent_id": user["message_id"], "out_of_context": ooc, "mode": mode})


# ---------------------------------------------------------------- pages & figures
@app.get("/api/page/{doc_id}/{page}.png")
def page_png(doc_id: str, page: int, dpi: int = 110):
    dpi = max(50, min(dpi, 220))
    key = (page, dpi)
    if key not in _page_cache:
        try:
            import pymupdf  # type: ignore
            with pymupdf.open(PDF) as doc:
                if not 1 <= page <= doc.page_count:
                    raise HTTPException(404, "No such page")
                _page_cache[key] = doc[page - 1].get_pixmap(dpi=dpi).tobytes("png")
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover
            raise HTTPException(500, f"Render failed: {exc}")
    return Response(_page_cache[key], media_type="image/png", headers={"Cache-Control": "max-age=86400"})


@app.get("/api/page/{doc_id}/{page}/meta")
def page_meta(doc_id: str, page: int):
    w, h = page_size(page)
    return {"width": w, "height": h}


@app.get("/api/figure/{chunk_id}")
def figure(chunk_id: str):
    c = CHUNK.get(chunk_id)
    if not c or not c.get("figure"):
        raise HTTPException(404, "No figure")
    path = ROOT / c["figure"]["image_file"]
    if not path.exists():
        raise HTTPException(404, "Figure file missing")
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------- audit
def agg(rows, key):
    out: dict = {}
    for r in rows:
        k = key(r)
        a = out.setdefault(k, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "ms": 0, "failed": 0})
        a["calls"] += 1
        a["prompt_tokens"] += r["prompt_tokens"]
        a["completion_tokens"] += r["completion_tokens"]
        a["total_tokens"] += r["total_tokens"]
        a["ms"] += r["duration_ms"]
        a["failed"] += 0 if r["ok"] else 1
    for a in out.values():
        a["avg_ms"] = round(a.pop("ms") / a["calls"], 1) if a["calls"] else None
    return out


@app.get("/api/audit/summary")
def audit_summary(days: int = 30):
    since = now() - timedelta(days=days)
    rows = [r for r in CALLS if datetime.fromisoformat(r["created_at"]) >= since]
    tot = agg(rows, lambda r: "all").get("all", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "avg_ms": None, "failed": 0})
    by_session = agg([r for r in rows if r["session_id"]], lambda r: r["session_id"])
    return {
        "days": days,
        "totals": tot,
        "by_purpose": [{"purpose": k, **v} for k, v in agg(rows, lambda r: r["purpose"]).items()],
        "by_model": [{"model": k, **v} for k, v in agg(rows, lambda r: r["model"]).items()],
        "by_day": sorted(({"day": k, **v} for k, v in agg(rows, lambda r: r["created_at"][:10]).items()), key=lambda x: x["day"]),
        "by_session": [{"session_id": k, "title": SESSIONS.get(k, {}).get("title"), **v} for k, v in by_session.items()],
    }


@app.get("/api/audit/calls")
def audit_calls(limit: int = 200, session_id: str | None = None):
    rows = [r for r in CALLS if not session_id or r["session_id"] == session_id]
    return rows[:limit]


# ---------------------------------------------------------------- settings & keys
@app.get("/api/settings")
def get_settings():
    return SETTINGS


@app.put("/api/settings")
async def put_settings(request: Request):
    body = await request.json()
    unknown = [k for k in body if k not in SETTINGS]
    if unknown:
        raise HTTPException(422, f"Unknown setting(s): {', '.join(unknown)}")
    SETTINGS.update(body)
    return SETTINGS


@app.get("/api/keys")
def get_keys():
    return [{"name": k, "value": mask(v["value"]), "note": v["note"], "updated_at": v["updated_at"]} for k, v in KEYS.items()]


@app.put("/api/keys/{name}")
async def put_key(name: str, request: Request):
    body = await request.json()
    KEYS[name] = {"value": str(body.get("value", "")), "note": str(body.get("note", "")), "updated_at": now().isoformat()}
    return {"name": name, "value": mask(KEYS[name]["value"]), "note": KEYS[name]["note"]}


@app.delete("/api/keys/{name}")
def delete_key(name: str):
    if name not in KEYS:
        raise HTTPException(404, "Key not found")
    KEYS.pop(name)
    return {"ok": True}


# ---------------------------------------------------------------- self-test harness
HARNESS = r"""<!doctype html><html><head><meta charset="utf-8"><title>UI self-test</title>
<style>body{background:#000;color:#fff;font:14px monospace;margin:0} iframe{border:0;display:block;background:#15191D}
#out{white-space:pre-wrap;padding:8px}</style></head><body>
<iframe id="f" width="__W__" height="__H__"></iframe><pre id="out">running</pre>
<script>
const out = {errors: [], checks: {}};
const f = document.getElementById("f");
const wait = (ms) => fetch("/__sleep?ms=" + ms).then(r => r.text());  // real-time wait (survives virtual time)
const until = async (fn, ms = 30000) => { for (let i = 0; i < ms / 200; i++) { try { if (fn()) return true; } catch {} await wait(200); } return false; };
function hook(w) {
  w.addEventListener("error", e => out.errors.push("error: " + e.message + " @" + (e.filename||"") + ":" + e.lineno));
  w.addEventListener("unhandledrejection", e => out.errors.push("rejection: " + (e.reason && e.reason.stack || e.reason)));
  const ce = w.console.error.bind(w.console); w.console.error = (...a) => { out.errors.push("console.error: " + a.join(" ")); ce(...a); };
}
async function load(url) {
  f.src = url;
  await new Promise(r => f.onload = r);
  hook(f.contentWindow);
  return f.contentDocument;
}
(async () => {
  const mode = "__MODE__";
  let d;
  if (mode === "answer") {
    d = await load("/?q=" + encodeURIComponent("Compare edge and screw dislocations"));
    out.checks.answer_done = await until(() => d.querySelector(".msg--bot .msg__foot:not([hidden])"));
    out.checks.citation_chips = d.querySelectorAll(".msg--bot .cite__chip:not(.is-missing)").length;
    out.checks.katex_nodes = d.querySelectorAll(".msg--bot .katex").length;
    out.checks.katex_errors = d.querySelectorAll(".katex-error").length;
    out.checks.tables = d.querySelectorAll(".msg--bot table").length;
    out.checks.hljs = d.querySelectorAll(".msg--bot pre code.hljs").length;
    out.checks.gallery_figs = d.querySelectorAll(".msg__gallery .fig").length;
    out.checks.tokens_label = (d.querySelector(".msg__tokens") || {}).textContent;
    out.checks.decision_chips = [...d.querySelectorAll(".dchip")].map(x => x.textContent);
    out.checks.url_has_session = f.contentWindow.location.search;
    const chip = d.querySelector(".cite__chip");
    chip.dispatchEvent(new f.contentWindow.FocusEvent("focusin", {bubbles: true}));
    await wait(50);
    out.checks.card_visible = !d.getElementById("cite-card").hidden && d.getElementById("cite-card").textContent.includes("Dislocation");
    const btn = d.querySelector('[data-act="pages"]');
    btn.click();
    out.checks.pages_expanded = btn.getAttribute("aria-expanded");
    out.checks.pages_imgs_loaded = await until(() => { const im = [...d.querySelectorAll(".msg__pages img")]; return im.length && im.every(i => i.complete && i.naturalWidth > 0); }, 15000);
    out.checks.page_boxes = d.querySelectorAll(".msg__pages .page-box").length;
    out.checks.highlight_rects = d.querySelectorAll(".msg__pages .hl").length;
    const hl = d.querySelector(".msg__pages .hl"); out.checks.first_rect = hl && hl.getAttribute("style");
    chip.click();
    await wait(300);
    out.checks.viewer_open = d.getElementById("viewer").open;
    out.checks.viewer_title = d.getElementById("viewer-title").textContent;
    out.checks.viewer_rects = d.querySelectorAll("#viewer .hl").length;
    d.getElementById("viewer").close();
    d.querySelector(".fig").click(); await wait(200);
    out.checks.figure_viewer = d.getElementById("viewer").open && !!d.querySelector("#viewer .viewer__figure");
    await until(() => d.querySelector("#viewer .viewer__figure").naturalWidth > 0, 5000);
    d.getElementById("viewer").close();
  } else if (mode === "ooc") {
    d = await load("/?q=" + encodeURIComponent("What is the capital of France?"));
    out.checks.done = await until(() => d.querySelector(".msg--bot .msg__foot:not([hidden])"));
    out.checks.ooc_notice = !!d.querySelector(".ooc");
    out.checks.ooc_text = (d.querySelector(".ooc__tag") || {}).textContent;
    out.checks.pages_btn_hidden = d.querySelector('.msg--bot [data-act="pages"]').hidden;
    out.checks.chips = d.querySelectorAll(".msg--bot .cite__chip").length;
  } else if (mode === "stop") {
    d = await load("/?q=" + encodeURIComponent("Explain dislocations in detail"));
    await until(() => (d.querySelector(".msg--bot .msg__body") || {}).textContent?.length > 40);
    out.checks.stop_button = d.getElementById("send-btn").getAttribute("aria-label");
    out.checks.stage_visible = !d.getElementById("stage").hidden;
    d.getElementById("send-btn").click();
    out.checks.stopped = await until(() => d.querySelector(".msg--bot .note"));
    out.checks.note = (d.querySelector(".msg--bot .note") || {}).textContent;
  } else if (mode === "error") {
    d = await load("/?q=" + encodeURIComponent("please fail now"));
    out.checks.error_note = await until(() => d.querySelector(".note--error"));
    out.checks.text = (d.querySelector(".note--error") || {}).textContent;
  } else if (mode === "audit") {
    d = await load("/audit");
    out.checks.kpi_total = await until(() => d.querySelector('[data-k="total"]').textContent !== "–");
    out.checks.purpose_rows = d.querySelectorAll("#purpose-table tbody tr").length;
    out.checks.chart_bars = d.querySelectorAll("#day-chart rect.b-p").length;
    out.checks.calls_rows = d.querySelectorAll("#calls-table tbody tr").length;
    d.querySelector('#calls-table th[data-sort="total_tokens"] button').click();
    out.checks.sorted = d.querySelector('#calls-table th[data-sort="total_tokens"]').getAttribute("aria-sort");
    d.getElementById("f-failed").click();
    out.checks.failed_rows = d.querySelectorAll("#calls-table tbody tr.is-failed").length + "/" + d.querySelectorAll("#calls-table tbody tr").length;
    out.checks.settings_fields = await until(() => d.querySelectorAll("#settings-fields [data-key]").length > 5) && d.querySelectorAll("#settings-fields [data-key]").length;
    const t = d.getElementById("s-temperature"); t.value = "0.2";
    d.getElementById("settings-save").click();
    await until(() => d.getElementById("settings-msg").textContent.startsWith("Saved") || d.getElementById("settings-msg").textContent.startsWith("Not"));
    out.checks.settings_msg = d.getElementById("settings-msg").textContent;
    out.checks.keys_rows = await until(() => d.querySelector("#keys-table [data-del]")) && d.querySelectorAll("#keys-table tbody tr").length;
    d.getElementById("k-name").value = "TEST_KEY"; d.getElementById("k-value").value = "secret-value-123"; d.getElementById("k-note").value = "harness";
    d.getElementById("k-toggle").click();
    out.checks.toggle_type = d.getElementById("k-value").type;
    d.getElementById("k-save").click();
    await until(() => d.querySelectorAll("#keys-table [data-del]").length === 2);
    out.checks.keys_after_add = d.querySelectorAll("#keys-table [data-del]").length;
    out.checks.masked = [...d.querySelectorAll("#keys-table .masked")].map(x => x.textContent);
    f.contentWindow.confirm = () => true;
    d.querySelector('#keys-table [data-del="TEST_KEY"]').click();
    await until(() => d.querySelectorAll("#keys-table [data-del]").length === 1);
    out.checks.keys_after_delete = d.querySelectorAll("#keys-table [data-del]").length;
  } else {
    d = await load(mode);
    await until(() => d.querySelector(".msg--bot .msg__foot:not([hidden]), .empty, .kpi"), 10000);
    const act = "__ACT__";
    if (act === "pages" || act === "viewer") {
      d.querySelector('[data-act="pages"]').click();
      await until(() => [...d.querySelectorAll(".msg__pages img")].every(i => i.complete && i.naturalWidth));
      d.querySelector(".msg__pages").scrollIntoView();
    }
    if (act === "viewer") { d.querySelector(".pages__thumb").click(); await wait(1500); }
    if (act === "card") { const c = d.querySelectorAll(".cite__chip")[1]; c.focus(); await wait(300); }
    if (act === "menu") { d.getElementById("sidebar-open").click(); await wait(500); }
    await wait(800);
  }
  document.getElementById("out").textContent = "RESULT " + JSON.stringify(out, null, 1);
})().catch(e => { out.errors.push("harness: " + e.stack); document.getElementById("out").textContent = "RESULT " + JSON.stringify(out, null, 1); });
</script></body></html>"""


@app.get("/__sleep")
async def test_sleep(ms: int = 100):
    await asyncio.sleep(min(ms, 5000) / 1000)
    return "ok"


@app.get("/__test")
def test_harness(mode: str = "answer", w: int = 1280, h: int = 860, act: str = ""):
    from fastapi.responses import HTMLResponse
    html = HARNESS.replace("__MODE__", mode.replace('"', "")).replace("__W__", str(w)).replace("__H__", str(h)).replace("__ACT__", act.replace('"', ""))
    return HTMLResponse(html)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
