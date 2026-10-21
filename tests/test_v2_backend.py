"""FEATURES_V2 backend: branching (edit / regenerate / switch + branch-correct memory), feedback (+ LLM review),
memory compaction endpoint, answer modes (thinking events, grounding + self-revision), API-keys removal.

Pipeline runs with a stub retriever + fake LLM against the real embedded Postgres (sessions are deleted after
each test); nothing here calls Ollama.

    .venv\\Scripts\\python -m pytest tests/test_v2_backend.py -v
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("NEWRAG_SKIP_WARMUP", "1")

from app import config, db  # noqa: E402
from app.llm import decisions  # noqa: E402
from app.llm.decisions import RouteDecision  # noqa: E402
from app.llm.ollama_client import ThinkingToken, Usage  # noqa: E402
from app.memory.session_memory import SessionMemory  # noqa: E402
from app.rag import pipeline  # noqa: E402
from app.retrieval.hybrid import Hit  # noqa: E402


# ============================================================================ pure units
def _row(mid, parent, role, content="x", mode="swift"):
    return {"message_id": mid, "parent_id": parent, "role": role, "content": content, "mode": mode,
            "decisions": {}, "images": [], "out_of_context": False}


def _tree():
    # 1 u -> 2 a -> 3 u -> 4 a ; edit of 3 = 5 u -> 6 a ; regenerate of 6 = 7 a ; stopped placeholder 8 under 5
    rows = [_row(1, None, "user", "Q1"), _row(2, 1, "assistant", "A1"), _row(3, 2, "user", "Q2"),
            _row(4, 3, "assistant", "A2"), _row(5, 2, "user", "Q2 edited"), _row(6, 5, "assistant", "A2e"),
            _row(7, 5, "assistant", "A2e regenerated"), _row(8, 5, "assistant", "")]
    return {r["message_id"]: r for r in rows}


def test_tree_helpers():
    rows = _tree()
    assert [r["message_id"] for r in db.path_to(rows, 4)] == [1, 2, 3, 4]
    assert [r["message_id"] for r in db.path_to(rows, 7)] == [1, 2, 5, 7]
    assert db.siblings(rows, rows[3]) == [3, 5]
    assert db.siblings(rows, rows[6]) == [6, 7]            # the empty placeholder 8 is not a sibling
    assert db.newest_leaf(rows, 3) == 4
    assert db.newest_leaf(rows, 5) == 7                    # newest non-placeholder child
    assert db.newest_leaf(rows, 1) == 7                    # follows the newest branch (5 > 3)


def test_resolve_branch():
    rows = _tree()
    assert pipeline.resolve_branch(rows, 7) == {"parent": 7, "user_id": None, "question": None}
    assert pipeline.resolve_branch(rows, 7, edit_of=3)["parent"] == 2
    r = pipeline.resolve_branch(rows, 7, regenerate_of=4)
    assert r == {"parent": 2, "user_id": 3, "question": "Q2"}
    assert pipeline.resolve_branch(rows, 7, parent_id=2)["parent"] == 2
    for bad in ({"edit_of": 4}, {"regenerate_of": 3}, {"parent_id": 999}, {"edit_of": 999}):
        with pytest.raises(pipeline.BranchError):
            pipeline.resolve_branch(rows, 7, **bad)


def test_path_turns_skip_placeholders_errors_greetings():
    rows = _tree()
    rows[2]["decisions"] = {"entities": ["Body-centred cubic"]}
    turns = pipeline.path_turns(db.path_to(rows, 4))
    assert [(t["user"], t["message_id"]) for t in turns] == [("Q1", 2), ("Q2", 4)]
    assert turns[0]["entities"] == ["Body-centred cubic"]
    rows[4]["content"] = "[error] boom"
    rows[2]["decisions"] = {"intent": "greeting"}
    assert pipeline.path_turns(db.path_to(rows, 4)) == []


def test_memory_path_ids_roundtrip_and_rebuild(tmp_path):
    mem = SessionMemory("s1", tmp_path / "s1.md", title="t")
    mem.add_turn("What is BCC?", "BCC is a cubic structure [1].", ["BCC"], message_id=2)
    mem.add_turn("And its APF?", "0.68 [2].", ["APF"], message_id=4)
    mem.save()
    again = SessionMemory.load("s1", tmp_path)
    assert again.path_ids == [2, 4] and len(again.turns) == 2
    assert "<!-- path: 2,4 -->" in (tmp_path / "s1.md").read_text(encoding="utf-8")

    # summary covers turn 1 (id 2): kept when the new path shares it, dropped otherwise
    again.summary, again.turns = "User asked about BCC.", again.turns[1:]
    same_prefix = [{"user": "What is BCC?", "assistant": "BCC ...", "message_id": 2},
                   {"user": "And its density?", "assistant": "rho = ... [1]", "message_id": 9, "entities": ["density"]}]
    assert again.rebuild_from_path(same_prefix) is False
    assert again.summary == "User asked about BCC." and [t.n for t in again.turns] == [2]
    assert again.turns[0].user == "And its density?" and again.path_ids == [2, 9]
    assert again.entities[0].name == "density" and all(e.name != "APF" for e in again.entities)

    other = [{"user": "What is FCC?", "assistant": "FCC ... [1]", "message_id": 11}]
    again.rebuild_from_path(other)
    assert again.summary == "" and [t.user for t in again.turns] == ["What is FCC?"]
    assert all(e.turn <= 0 or e.name != "BCC" for e in again.entities)
    again.rebuild_from_path([])
    assert again.turns == [] and again.path_ids == []


def test_v1_memory_file_rebuild_by_text(tmp_path):
    mem = SessionMemory("v1", tmp_path / "v1.md")
    mem.add_turn("Q1", "A1", [])                           # no ids: a v1 file
    mem.add_turn("Q2", "A2", [])
    assert mem.path_ids == []
    mem.rebuild_from_path([{"user": "Q1", "assistant": "A1", "message_id": 5},
                           {"user": "Q2 edited", "assistant": "A2e", "message_id": 7}])
    assert [t.user for t in mem.turns] == ["Q1", "Q2 edited"] and mem.path_ids == [5, 7]


def test_modes_profiles_and_request_overrides():
    modes = config.get_modes()
    assert set(modes) >= {"swift", "deep"}
    sw, dp = modes["swift"]["settings"], modes["deep"]["settings"]
    assert (sw["rerank_top_n"], sw["final_k"], sw["max_answer_tokens"], sw["fast_route"], sw["think"]) == \
        (20, 6, 900, True, False)
    assert dp["chat_model"] == "qwen3.5:9b" and dp["think"] and dp["grounding_check"] and not dp["fast_route"]
    assert (dp["rerank_top_n"], dp["final_k"], dp["max_answer_tokens"]) == (30, 8, 1400)
    assert config.get_mode(None)["id"] == "swift" and config.get_mode("nope")["id"] == "swift"
    base = config.get_settings()
    tok = config.request_overrides({"final_k": 3, "think": True, "unknown": 1})
    try:
        assert config.get_settings().final_k == 3 and config.get_settings() is not base
    finally:
        config.reset_request_overrides(tok)
    assert config.get_settings() is base


# ============================================================================ pipeline + Postgres (fake LLM)
def hit(cid, content, score=0.99, kind="text"):
    return Hit(chunk_id=cid, doc_id="doc_test", chunk_type=kind, header="Ch 2 > 2.5", content=content, text=content,
               section_path=["Ch 2", "2.5 BCC"], pages=[40], payload={}, rerank_score=score)


class StubRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, **kw):
        return list(self.hits)

    def expand_cross_refs(self, hits, max_extra=3):
        return []

    def figures_for(self, query, hits, k=3, **kw):
        return []


class FakeLLM:
    def __init__(self, answers=None, thinking=None, revision=None):
        self.answers = list(answers or [])
        self.thinking = thinking
        self.revision = revision
        self.prompts: list[list[dict]] = []
        self.stream_kw: list[dict] = []
        self.last_usage = self.last_logprobs = self.last_done_reason = None

    async def stream(self, messages, *, purpose, **kw):
        self.prompts.append(messages)
        self.stream_kw.append(kw)
        if kw.get("think") and self.thinking:
            for piece in self.thinking:
                yield ThinkingToken(piece)
        text = self.answers.pop(0) if self.answers else "BCC has an APF of 0.68 [1]."
        for tok in re.findall(r"\S+\s*", text):
            yield tok
        self.last_done_reason = "stop"

    async def complete(self, messages, *, purpose, **kw):
        if purpose == "answer_revision":
            return self.revision or "", Usage(5, 3, 1, "fake", purpose)
        return ("Crystal chat" if purpose == "title" else "summary of earlier turns"), Usage(5, 3, 1, "fake", purpose)


def _db_ok() -> bool:
    try:
        db.init_db()
        return True
    except Exception:  # noqa: BLE001
        return False


needs_db = pytest.mark.skipif(not _db_ok(), reason="embedded Postgres unavailable")


@pytest.fixture
def live_db(monkeypatch, tmp_path):
    s = config.get_settings()
    monkeypatch.setattr(s, "sessions_dir", tmp_path)
    llm = FakeLLM()
    monkeypatch.setattr(pipeline, "get_llm", lambda: llm)
    import app.llm.ollama_client as oc
    monkeypatch.setattr(oc, "get_llm", lambda: llm)
    pipeline.set_retriever_factory(lambda: StubRetriever([hit("c1", "BCC packs atoms with an APF of 0.68."),
                                                          hit("c2", "FCC has an APF of 0.74.")]))

    async def fake_route(question, memory_md, **ids):
        return RouteDecision(standalone_query=question, entities=["BCC"])
    monkeypatch.setattr(pipeline, "route", fake_route)
    sessions: list[str] = []
    yield {"llm": llm, "sessions": sessions, "dir": tmp_path}
    pipeline.set_retriever_factory(None)
    with db.get_pool().connection() as conn:
        for sid in sessions:
            conn.execute("DELETE FROM sessions WHERE session_id = %s", (sid,))


def ask(sid, text=None, **kw):
    async def go():
        return [ev async for ev in pipeline.answer(sid, text, **kw)]
    return asyncio.run(go())


def ev(evs, name):
    return next(e["data"] for e in evs if e["event"] == name)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.api.main import app
    with TestClient(app) as c:
        yield c


@needs_db
def test_branching_edit_regenerate_switch_and_memory(live_db, client):
    sid = str(uuid.uuid4())
    live_db["sessions"].append(sid)
    e1 = ask(sid, "What is the APF of BCC?")
    s1, d1 = ev(e1, "session"), ev(e1, "done")
    assert s1["parent_id"] is None and s1["mode"] == "swift" and s1["active_leaf_id"] == s1["user_message_id"]
    assert d1["user_message_id"] == s1["user_message_id"] and d1["parent_id"] == d1["user_message_id"]
    assert d1["active_leaf_id"] == d1["message_id"] and d1["mode"] == "swift"
    e2 = ask(sid, "Which structure has the APF of 0.74?")
    d2 = ev(e2, "done")
    assert ev(e2, "session")["parent_id"] == d1["message_id"]

    path = client.get(f"/api/sessions/{sid}/messages").json()
    assert [m["role"] for m in path] == ["user", "assistant"] * 2
    assert set(path[0]) >= {"message_id", "parent_id", "role", "content", "sources", "images", "decisions", "usage",
                            "out_of_context", "feedback", "mode", "created_at", "sibling_ids", "sibling_index",
                            "sibling_count"}
    assert path[2]["parent_id"] == path[1]["message_id"] and path[3]["mode"] == "swift"

    # edit the second question -> new sibling user message, answered with ONLY turn 1 as memory
    q2_id = path[2]["message_id"]
    e3 = ask(sid, "What is the APF of FCC?", edit_of=q2_id)
    d3 = ev(e3, "done")
    assert ev(e3, "session")["parent_id"] == path[1]["message_id"]
    sys_prompt = live_db["llm"].prompts[-1][0]["content"]
    assert "What is the APF of BCC?" in sys_prompt and "Which structure has the APF" not in sys_prompt
    path = client.get(f"/api/sessions/{sid}/messages").json()
    assert path[2]["content"] == "What is the APF of FCC?"
    assert path[2]["sibling_count"] == 2 and path[2]["sibling_index"] == 1 and path[2]["sibling_ids"][0] == q2_id
    mem = (live_db["dir"] / f"{sid}.md").read_text(encoding="utf-8")
    assert "What is the APF of FCC?" in mem and "Which structure has the APF" not in mem

    # regenerate the newest answer -> sibling assistant message, no new user message
    e4 = ask(sid, None, regenerate_of=d3["message_id"])
    d4 = ev(e4, "done")
    assert d4["user_message_id"] == d3["user_message_id"]
    path = client.get(f"/api/sessions/{sid}/messages").json()
    assert path[3]["message_id"] == d4["message_id"] and path[3]["sibling_count"] == 2
    assert len(path) == 4
    mem = SessionMemory.load(sid, live_db["dir"])
    assert [t.user for t in mem.turns] == ["What is the APF of BCC?", "What is the APF of FCC?"]
    assert mem.path_ids == [d1["message_id"], d4["message_id"]]

    # switch back to the original second question -> its answer is the leaf; memory follows the path
    r = client.post(f"/api/sessions/{sid}/switch", json={"message_id": q2_id})
    assert r.status_code == 200
    path = r.json()["messages"]
    assert path[-1]["message_id"] == d2["message_id"] and path[2]["sibling_index"] == 0
    assert client.get(f"/api/sessions/{sid}/messages").json() == path
    mem = SessionMemory.load(sid, live_db["dir"])
    assert [t.user for t in mem.turns] == ["What is the APF of BCC?", "Which structure has the APF of 0.74?"]
    # default continuation now attaches to the switched branch
    assert ev(ask(sid, "Thanks, and what about HCP?"), "session")["parent_id"] == d2["message_id"]
    assert client.post(f"/api/sessions/{sid}/switch", json={"message_id": 10 ** 12}).status_code == 404

    # invalid branch ids -> error + done (nothing stored)
    bad = ask(sid, "x", edit_of=d2["message_id"])
    assert [e["event"] for e in bad] == ["error", "done"]


@needs_db
def test_feedback_endpoints_and_review(live_db, client, monkeypatch):
    sid = str(uuid.uuid4())
    live_db["sessions"].append(sid)
    d = ev(ask(sid, "What is the APF of BCC?"), "done")
    mid, uid = d["message_id"], d["user_message_id"]
    assert client.post(f"/api/messages/{mid}/feedback", json={"rating": 1}).json() == {"message_id": mid, "rating": 1}
    assert client.get(f"/api/sessions/{sid}/messages").json()[1]["feedback"] == {"rating": 1, "comment": None}
    r = client.post(f"/api/messages/{mid}/feedback", json={"rating": -1, "comment": "wrong number"})
    assert r.json()["rating"] == -1
    assert client.post(f"/api/messages/{uid}/feedback", json={"rating": 1}).status_code == 422
    assert client.post(f"/api/messages/{mid}/feedback", json={"rating": 5}).status_code == 422
    assert client.post("/api/messages/999999999999/feedback", json={"rating": 1}).status_code == 404

    items = client.get("/api/feedback?rating=-1&limit=500").json()
    mine = next(i for i in items if i["message_id"] == mid)
    assert mine["question"] == "What is the APF of BCC?" and mine["comment"] == "wrong number"
    assert mine["mode"] == "swift" and isinstance(mine["sources"], list) and mine["review"] is None

    seen = {}

    async def fake_review(question, answer, sources, comment=None, **ids):
        seen.update(question=question, comment=comment, ids=ids)
        return {"issue_type": "wrong_fact", "explanation": "APF misquoted", "suggested_fix": "cite the table",
                "confidence": 0.9, "_ok": True}
    monkeypatch.setattr(decisions, "review_feedback", fake_review)
    out = client.post("/api/feedback/review", json={"message_ids": [mid]}).json()
    assert out["reviewed"] == 1 and out["reviews"][0]["review"]["issue_type"] == "wrong_fact"
    assert seen["question"] == "What is the APF of BCC?" and seen["ids"]["message_id"] == mid
    mine = next(i for i in client.get("/api/feedback?rating=-1").json() if i["message_id"] == mid)
    assert mine["review"]["suggested_fix"] == "cite the table"

    summ = client.get("/api/feedback/summary?days=7").json()
    assert set(summ) >= {"totals", "by_day", "by_mode", "by_intent", "by_issue_type", "unreviewed_dislikes"}
    assert summ["totals"]["dislikes"] >= 1 and any(m["mode"] == "swift" for m in summ["by_mode"])
    assert client.post(f"/api/messages/{mid}/feedback", json={"rating": 0}).json()["rating"] == 0
    assert client.get(f"/api/sessions/{sid}/messages").json()[1]["feedback"] is None


@needs_db
def test_memory_compact_endpoint(live_db, client):
    sid = str(uuid.uuid4())
    live_db["sessions"].append(sid)
    s = config.get_settings()
    for i in range(s.memory_keep_turns + 2):
        ask(sid, f"Question number {i} about BCC packing?")
    r = client.post(f"/api/sessions/{sid}/memory/compact").json()
    assert r["compacted"] is True and "summary of earlier turns" in r["markdown"]
    assert set(r) >= {"markdown", "tokens", "budget", "path", "before_tokens"}
    assert client.post(f"/api/sessions/{uuid.uuid4()}/memory/compact").status_code == 404


@needs_db
def test_deep_mode_thinking_grounding_revision(live_db, monkeypatch):
    sid = str(uuid.uuid4())
    live_db["sessions"].append(sid)
    llm = live_db["llm"]
    llm.thinking = ["Let me check ", "the sources..."]
    llm.answers = ["BCC has an APF of 0.70 and is very common [1]."]
    # long enough to pass _acceptable_revision (revisions < 60 % of the original length are rejected)
    llm.revision = "BCC has an atomic packing factor (APF) of 0.68 [1]."
    calls = []

    async def fake_decide(state, questions, **kw):
        calls.append(kw.get("purpose"))
        return {"grounded": "2", "unsupported": "APF 0.70; very common", "grounded_confidence": 0.8, "_ok": True}
    monkeypatch.setattr(pipeline, "decide", fake_decide)
    evs = ask(sid, "What is the APF of BCC?", mode="deep")
    names = [e["event"] for e in evs]
    assert "thinking" in names and "error" not in names, evs
    thinking = "".join(e["data"]["t"] for e in evs if e["event"] == "thinking")
    assert thinking == "Let me check the sources..."
    tokens = "".join(e["data"]["t"] for e in evs if e["event"] == "token")
    assert "Let me check" not in tokens
    assert llm.stream_kw[-1]["think"] is True and llm.stream_kw[-1]["options"]["num_predict"] == 1400 + 1500
    final = ev(evs, "answer")["content"]
    assert "0.68" in final and "Let me check" not in final
    assert calls == ["grounding_check"]
    assert ev(evs, "decision")["mode"] == "deep"
    with db.get_pool().connection() as conn:
        mode, dec = conn.execute("SELECT mode, decisions FROM messages WHERE message_id=%s",
                                 (ev(evs, "done")["message_id"],)).fetchone()
    assert mode == "deep" and dec["grounding"] == {"score": 2, "unsupported": "APF 0.70; very common", "revised": True}
    assert "self_revised" in dec["rules"]
    mem = (live_db["dir"] / f"{sid}.md").read_text(encoding="utf-8")
    assert "Let me check" not in mem and "0.68" in mem


@needs_db
def test_swift_mode_meta_in_llm_calls(live_db):
    sid = str(uuid.uuid4())
    live_db["sessions"].append(sid)
    from app.llm.ollama_client import call_meta, set_call_meta
    set_call_meta({"mode": "deep"})
    try:
        assert call_meta({"x": 1})["mode"] == "deep" and call_meta({"x": 1})["x"] == 1
    finally:
        set_call_meta(None)
    d = ev(ask(sid, "What is the APF of BCC?"), "done")
    assert d["mode"] == "swift"
    usage = ev(ask(sid, "And FCC?"), "usage")
    assert usage["mode"] == "swift"


def test_modes_endpoint_and_keys_removed(client):
    modes = client.get("/api/modes").json()
    assert [m["id"] for m in modes][:2] == ["swift", "deep"]
    assert set(modes[0]) >= {"id", "name", "tagline", "description", "settings"}
    assert client.get("/api/keys").status_code in (404, 405)
    assert client.put("/api/keys/x", json={"value": "y"}).status_code in (404, 405)


def test_chat_rejects_unknown_mode(client):
    with client.stream("POST", "/api/chat", json={"message": "hi", "mode": "turbo"}) as r:
        body = r.read().decode("utf-8")
    assert "event: error" in body and "Unknown mode" in body
