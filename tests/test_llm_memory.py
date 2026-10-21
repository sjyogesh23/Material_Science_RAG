"""Tests for the decision layer, routing rules, session memory and Ollama client.

Unit tests use a fake LLM; tests marked ``live`` call the local Ollama model and are skipped if it is down.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm import decisions, prompts  # noqa: E402
from app.llm.decisions import Choice, Noul, Score, Text, TextList, decide, route  # noqa: E402
from app.llm.ollama_client import Usage, get_llm, summarize_usage, usage_scope  # noqa: E402
from app.memory.session_memory import SessionMemory  # noqa: E402


def run(coro):
    return asyncio.run(coro)


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.last_logprobs = None

    async def complete(self, messages, *, purpose, model=None, format=None, options=None, logprobs=False, **ids):
        self.calls.append({"messages": messages, "purpose": purpose, "format": format, "options": options})
        r = self.replies.pop(0) if self.replies else ""
        if isinstance(r, Exception):
            raise r
        return (r if isinstance(r, str) else json.dumps(r)), Usage(10, 5, 1, "fake", purpose)


def route_reply(**kw):
    base = {"standalone_query": "", "in_scope": True, "intent": "definition", "answer_format": "prose",
            "needs_images": False, "is_cross_reference": False, "sub_queries": [], "entities": []}
    base.update(kw)
    return base


# ----------------------------------------------------------------------------- decide()
QS = [Choice(name="color", question="?", options=["red", "blue"], default="blue"),
      Noul(name="ok", question="?"), Score(name="lvl", question="?", levels=["low", "mid", "high"]),
      Text(name="q", question="?"), TextList(name="subs", question="?", max_items=2)]


def test_schema_is_enum_constrained():
    schema = decisions.build_model(QS).model_json_schema()
    props = schema["properties"]
    assert props["color"]["enum"] == ["red", "blue"]
    assert props["lvl"]["enum"] == ["low", "mid", "high"]
    assert props["ok"]["type"] == "boolean"
    assert props["subs"]["maxItems"] == 2
    assert set(schema["required"]) == {"color", "ok", "lvl", "q", "subs"}


def test_decide_valid_first_try():
    llm = FakeLLM([{"color": "red", "ok": True, "lvl": "high", "q": "x", "subs": ["a"]}])
    r = run(decide("state", QS, llm=llm))
    assert r["color"] == "red" and r["ok"] is True and r["lvl"] == "high" and r["subs"] == ["a"]
    assert 0 <= r["color_confidence"] <= 1 and r["_ok"]
    assert llm.calls[0]["format"]["properties"]["color"]["enum"] == ["red", "blue"]
    assert llm.calls[0]["options"]["temperature"] == 0


def test_decide_retries_once_then_succeeds():
    llm = FakeLLM(['{"color": "green"}', {"color": "blue", "ok": False, "lvl": "low", "q": "", "subs": []}])
    r = run(decide("s", QS, llm=llm))
    assert r["_ok"] and r["color"] == "blue" and len(llm.calls) == 2
    assert "Invalid" in llm.calls[1]["messages"][-1]["content"]


def test_decide_safe_defaults():
    llm = FakeLLM(["garbage", "still garbage"])
    r = run(decide("s", QS, llm=llm))
    assert not r["_ok"]
    assert r["color"] == "blue" and r["ok"] is False and r["lvl"] == "low" and r["subs"] == []
    assert r["color_confidence"] == 0.0


def test_decide_llm_down_defaults():
    r = run(decide("s", QS, llm=FakeLLM([ConnectionError("down")])))
    assert not r["_ok"] and r["color"] == "blue"


def test_confidence_from_logprobs():
    text = '{"color": "red", "ok": true}'
    toks = ['{"', 'color', '":', ' "', 'red', '",', ' "', 'ok', '":', ' true', '}']
    assert "".join(toks) == text
    lps = [SimpleNamespace(token=t, logprob=-0.0) for t in toks]
    lps[4].logprob = -0.5108  # p(red) = 0.6
    lps[9].logprob = -0.1054  # p(true) = 0.9
    c = decisions._confidences(text, lps, QS[:2])
    assert abs(c["color"] - 0.6) < 1e-3 and abs(c["ok"] - 0.9) < 1e-3


# ----------------------------------------------------------------------------- route() rules
def test_greeting_needs_no_llm():
    llm = FakeLLM([])
    d = run(route("Hello there!", "", llm=llm))
    assert d.intent == "greeting" and llm.calls == []
    assert run(route("thanks a lot", "", llm=llm)).intent == "greeting"


def test_table_and_image_rules_override():
    llm = FakeLLM([route_reply(intent="comparison"), route_reply(intent="figure_request")])
    d = run(route("Compare FCC and BCC in a table", "", llm=llm))
    assert d.answer_format == "table" and "table" in d.rules
    assert d.standalone_query == "Compare FCC and BCC in a table"
    d = run(route("Show me the stress strain curve figure", "", llm=llm))
    assert d.needs_images and d.answer_format == "prose"


def test_table_number_is_not_table_format():
    d = run(route("What does Table 3-1 list about slip systems?", "", llm=FakeLLM([route_reply()])))
    assert d.answer_format == "prose"


def test_cross_reference_rule_and_subqueries():
    llm = FakeLLM([route_reply(intent="explanation", sub_queries=["Hall-Petch relation", "grain boundaries",
                                                                  "defects", "extra"])])
    d = run(route("How does the Hall-Petch relation relate to grain boundaries from the defects chapter?", "",
                  llm=llm))
    assert d.is_cross_reference and len(d.sub_queries) == 3


def test_subqueries_dropped_when_not_cross_reference():
    d = run(route("What is the definition of a vacancy defect?", "",
                  llm=FakeLLM([route_reply(sub_queries=["vacancy"])])))
    assert d.sub_queries == []


def test_pronoun_uses_llm_rewrite_and_fallback():
    mem = "Entities in focus (newest first):\n- Bragg's law - What is Bragg's law? (turn 1)\nRecent turns:"
    d = run(route("What is its unit?", mem, llm=FakeLLM([route_reply(standalone_query="What is the unit of the "
                                                                                       "wavelength in Bragg's law?")])))
    assert "Bragg" in d.standalone_query
    d = run(route("What is its unit?", mem, llm=FakeLLM([route_reply(standalone_query="What is its unit?")])))
    assert d.standalone_query == "What is its unit? (about Bragg's law)" and "anaphora_fallback" in d.rules


def test_self_contained_reference_not_anchored_on_memory():
    mem = "Entities in focus (newest first):\n- Figure 7-1: Stress Strain Curve - Show me (turn 4)"
    q = "What does Figure 2-7 show and which section explains it?"
    d = run(route(q, mem, llm=FakeLLM([route_reply(standalone_query=q, intent="cross_reference")])))
    assert d.standalone_query == q and d.is_cross_reference and d.needs_images


def test_standalone_question_kept_verbatim():
    d = run(route("Explain the iron carbon phase diagram eutectoid point",
                  "", llm=FakeLLM([route_reply(standalone_query="Explain eutectoid")])))
    assert d.standalone_query == "Explain the iron carbon phase diagram eutectoid point"


def test_route_llm_failure_defaults_in_scope():
    d = run(route("What is a dislocation in a crystal?", "", llm=FakeLLM(["bad", "bad"])))
    assert d.in_scope and not d.llm_ok and d.standalone_query.startswith("What is a dislocation")


# ----------------------------------------------------------------------------- memory
def make_mem(tmp_path, n_turns=3) -> SessionMemory:
    m = SessionMemory.load("s1", sessions_dir=tmp_path)
    m.title = "Bragg chat"
    for i in range(1, n_turns + 1):
        m.add_turn(f"Question {i} about Bragg's law?",
                   f"Answer {i}: $n\\lambda = 2d\\sin\\theta$ relates spacing [1, 2]. More text [3].",
                   [f"Entity {i}"])
    return m


def test_memory_roundtrip(tmp_path):
    m = make_mem(tmp_path)
    m.add_turn("What is the capital of France?", prompts.OOC, [])
    m.summary = "Earlier we discussed X-rays."
    m.save()
    md = (tmp_path / "s1.md").read_text(encoding="utf-8")
    assert md.startswith("# Session s1\n_Title: Bragg chat_\n## Summary")
    assert "### Turn 4 - Assistant\n#OutofContext" in md
    assert "[1, 2]" not in md  # citation markers stripped
    assert "- Entity 3 - Question 3 about Bragg's law? (turn 3)" in md
    m2 = SessionMemory.load("s1", sessions_dir=tmp_path)
    assert m2.title == m.title and m2.summary == m.summary
    assert [(t.n, t.user, t.assistant) for t in m2.turns] == [(t.n, t.user, t.assistant) for t in m.turns]
    assert [e.name for e in m2.entities] == ["Entity 3", "Entity 2", "Entity 1"]
    assert m2.next_turn == 5


def test_entities_dedup_and_newest_first(tmp_path):
    m = make_mem(tmp_path)
    m.add_turn("again", "ok", ["entity 1"])
    assert m.entities[0].name == "entity 1" and m.entities[0].turn == 4
    assert len([e for e in m.entities if e.name.lower() == "entity 1"]) == 1


def test_render_respects_budget_newest_first(tmp_path):
    m = make_mem(tmp_path, 6)
    full = m.token_count()
    small = m.render(full // 2)
    assert "### Turn 6" in small and "### Turn 1 " not in small
    assert "## Entities in focus" in small
    from app.llm.ollama_client import count_tokens
    assert count_tokens(small) <= full // 2 + 5
    assert m.render(10_000) == m.to_markdown()


def test_router_context(tmp_path):
    ctx = make_mem(tmp_path).router_context()
    assert "Entity 3" in ctx and "Turn 3 user" in ctx and "Turn 1 user" not in ctx


def test_compaction_with_llm(tmp_path):
    m = make_mem(tmp_path, 8)
    llm = FakeLLM(["User asked about Bragg's law repeatedly."])
    before = m.token_count()
    assert m.needs_compaction(budget=before - 1)
    assert run(m.compact(llm, budget=10_000, keep_turns=3))
    assert [t.n for t in m.turns] == [6, 7, 8]
    assert m.summary == "User asked about Bragg's law repeatedly."
    assert llm.calls[0]["purpose"] == "memory_compaction"
    assert m.token_count() < before


def test_compaction_tiny_budget_and_llm_failure(tmp_path):
    m = make_mem(tmp_path, 5)
    assert run(m.compact(FakeLLM([RuntimeError("ollama down")]), budget=120, keep_turns=6))
    assert len(m.turns) == 1 and m.turns[0].n == 5
    assert "Turn 1: asked" in m.summary  # heuristic summary


def test_usage_scope_and_summary():
    with usage_scope() as calls:
        calls.append(Usage(100, 20, 500, "m", "router"))
        calls.append(Usage(1000, 200, 5000, "m", "answer"))
    s = summarize_usage(calls)
    assert s["total"] == 1320 and s["by_purpose"]["answer"]["completion_tokens"] == 200


# ----------------------------------------------------------------------------- live Ollama
def _ollama_up() -> bool:
    try:
        return run(get_llm().health())["chat_model_present"]
    except Exception:
        return False


live = pytest.mark.skipif(not _ollama_up(), reason="Ollama / chat model not available")


@live
def test_live_route_pronoun_and_scope():
    async def go():
        get_llm.__globals__["_llm"] = None  # fresh client bound to this event loop
        mem = ("Entities in focus (newest first):\n- Bragg's law - What is Bragg's law? (turn 1)\n"
               "Recent turns:\nTurn 1 user: What is Bragg's law?\nTurn 1 assistant: Bragg's law, "
               "$n\\lambda = 2d\\sin\\theta$, gives the condition for constructive interference of X-rays.")
        d1 = await route("What is its formula and what does each symbol mean?", mem)
        d2 = await route("What is the capital of France?", "")
        return d1, d2
    d1, d2 = run(go())
    assert "bragg" in d1.standalone_query.lower()
    assert d1.answer_format == "equation"
    assert d2.in_scope is False


@live
def test_live_decide_score():
    async def go():
        get_llm.__globals__["_llm"] = None
        return await decide("SOURCES: [1] FCC has 4 atoms per unit cell.\nANSWER: FCC has 4 atoms per cell [1].",
                            [Score(name="grounded", question=prompts.GROUNDING_QUESTION,
                                   levels=["unsupported", "partially", "fully"])], purpose="grounding_check")
    r = run(go())
    assert r["_ok"] and r["grounded"] == "fully" and 0 < r["grounded_confidence"] <= 1
