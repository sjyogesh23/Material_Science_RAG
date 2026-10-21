"""Versioned components + architecture API (FEATURES_V2 §6).

Unit tests run without a database; API tests use the FastAPI TestClient against the real embedded Postgres.
They never change what the live app uses: new versions are created with ``activate: false`` (or with content
identical to the active version), authored ``pytest``, and deleted again with the previous active versions
restored.

    .venv\\Scripts\\python -m pytest tests/test_versioning.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("NEWRAG_SKIP_WARMUP", "1")

from app import versioning  # noqa: E402
from app.llm import prompts  # noqa: E402

AUTHOR = "pytest"


# ----------------------------------------------------------------------------- unit (no DB needed)
def test_prompt_names_resolve_to_code_defaults():
    d = prompts.DEFAULT_COMPONENTS
    assert prompts.ANSWER_SYSTEM == d["answer_system_prompt"] and "#OutofContext" in prompts.ANSWER_SYSTEM
    assert prompts.ROUTER_SYSTEM == d["router_prompt"]
    assert prompts.FORMAT_RULES["table"] == d["table_rules"] and set(prompts.FORMAT_RULES) == {
        "prose", "table", "steps", "equation"}
    assert set(prompts.FORM_NOTES) == set(prompts.FORM_NOTE_KEYS)
    assert "{k}" in prompts.CITE_COUNT_NOTE and "{page}" in prompts.PAGE_NOTE_MISSING
    assert "do NOT reply #OutofContext" in prompts.RETRY_NOTE
    assert "{max_words}" in prompts.COMPACTION_SYSTEM and "{turns}" in prompts.COMPACTION_USER
    assert prompts.TITLE_SYSTEM.startswith("Write a short title")
    assert prompts.GROUNDING_CAUTION.startswith("\n\n> Caution")
    assert prompts.compaction_prompts() == (prompts.COMPACTION_SYSTEM, prompts.COMPACTION_USER)
    with pytest.raises(AttributeError):
        prompts.NOT_A_PROMPT  # noqa: B018


def test_answer_system_uses_active_components(monkeypatch):
    monkeypatch.setattr(versioning, "get_component",
                        lambda n: "CUSTOM #OutofContext RULES" if n == "answer_system_prompt" else
                        ("TABLE RULE" if n == "table_rules" else versioning.code_default(n)))
    s = prompts.answer_system("table", "", False)
    assert s.startswith("CUSTOM #OutofContext RULES\n\nTABLE RULE")
    assert prompts.ANSWER_SYSTEM == "CUSTOM #OutofContext RULES"


def test_bad_component_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setattr(versioning, "get_component", lambda n: {"bogus": 1} if n == "router_prompt" else "  ")
    assert prompts.router_prompt() == prompts.DEFAULT_COMPONENTS["router_prompt"]
    assert prompts.title_prompt() == prompts.DEFAULT_COMPONENTS["title_prompt"]
    assert prompts.form_notes() == prompts.DEFAULT_COMPONENTS["form_notes"]

    def boom(n):
        raise RuntimeError("db down")
    monkeypatch.setattr(versioning, "get_component", boom)
    assert prompts.ROUTER_SYSTEM == prompts.DEFAULT_COMPONENTS["router_prompt"]


def test_code_defaults_cover_components():
    d = versioning.code_defaults()
    for name in ("answer_system_prompt", "router_prompt", "form_notes", "table_rules", "compaction_prompt",
                 "title_prompt", "grounding_prompt", "retrieval_params"):
        assert name in d
    rp = d["retrieval_params"]
    assert rp["rrf_k"] == 60 and rp["final_k"] == 6 and rp["ooc_threshold"] == 0.25
    assert set(rp) == set(versioning.RETRIEVAL_KEYS)


def test_validate_text_components():
    assert versioning.validate("title_prompt", "Name it.") == "Name it."
    for bad in ("", "   ", 5, None, {"a": "b"}):
        with pytest.raises(versioning.ComponentError):
            versioning.validate("router_prompt", bad)
    with pytest.raises(versioning.ComponentError, match="OutofContext"):
        versioning.validate("answer_system_prompt", "Answer from the sources.")
    with pytest.raises(KeyError):
        versioning.validate("nope", "x")


def test_validate_template_placeholders():
    notes = dict(prompts.DEFAULT_COMPONENTS["form_notes"])
    assert versioning.validate("form_notes", notes) == notes
    with pytest.raises(versioning.ComponentError, match="unknown placeholder"):
        versioning.validate("form_notes", {**notes, "page_missing": "page {pg} missing"})
    with pytest.raises(versioning.ComponentError, match="missing key"):
        versioning.validate("form_notes", {k: v for k, v in notes.items() if k != "quote"})
    with pytest.raises(versioning.ComponentError, match="unknown key"):
        versioning.validate("form_notes", {**notes, "extra": "x"})
    with pytest.raises(versioning.ComponentError):
        versioning.validate("compaction_prompt", {"system": "at most {max_words words", "user": "{summary} {turns}"})
    # untemplated notes may contain literal braces (they are never .format()-ed)
    assert versioning.validate("form_notes", {**notes, "quote": "Quote {exactly}."})["quote"] == "Quote {exactly}."
    ok = versioning.validate("compaction_prompt", {"system": "Max {max_words} words.", "user": "{turns}"})
    assert ok["user"] == "{turns}"


def test_validate_retrieval_params():
    assert versioning.validate("retrieval_params", {"final_k": 8, "ooc_threshold": 0.3}) == {
        "final_k": 8, "ooc_threshold": 0.3}
    assert versioning.validate("retrieval_params", {"rrf_k": 60.0}) == {"rrf_k": 60}
    for bad, msg in (({"final_k": 0}, "between"), ({"rrf_k": 1.5}, "whole"), ({"top_k": 3}, "unknown key"),
                     ({"ooc_threshold": 0.6}, "must not exceed"), ({"final_k": "x"}, "number"), ({}, "non-empty"),
                     ({"final_k": True}, "true/false")):
        with pytest.raises(versioning.ComponentError, match=msg):
            versioning.validate("retrieval_params", bad)


def test_validate_modes_against_default(monkeypatch):
    default = {"swift": {"name": "Swift", "settings": {"final_k": 5, "grounding_check": False}},
               "deep": {"name": "Deep", "settings": {"final_k": 8, "grounding_check": True}}}
    monkeypatch.setattr(versioning, "mode_defaults", lambda: default)
    assert versioning.validate("modes", default) == default
    with pytest.raises(versioning.ComponentError, match="between"):
        versioning.validate("modes", {**default, "deep": {"name": "Deep", "settings": {"final_k": 500}}})
    with pytest.raises(versioning.ComponentError, match="missing mode"):
        versioning.validate("modes", {"swift": default["swift"]})
    with pytest.raises(versioning.ComponentError, match="unknown key"):
        versioning.validate("modes", {**default, "turbo": {}})


def test_retrieval_overrides_only_differences(monkeypatch):
    base = versioning.retrieval_defaults()
    monkeypatch.setattr(versioning, "get_component", lambda n: {**base, "final_k": 9})
    assert versioning.retrieval_overrides() == {"final_k": 9}
    monkeypatch.setattr(versioning, "get_component", lambda n: dict(base))
    assert versioning.retrieval_overrides() == {}


def test_get_component_without_db_is_default(monkeypatch):
    from app import db
    monkeypatch.setattr(db, "peek_pool", lambda: None)
    assert versioning.get_component("router_prompt") == prompts.DEFAULT_COMPONENTS["router_prompt"]
    assert versioning.active_versions() == {}
    with pytest.raises(KeyError):
        versioning.get_component("nope")


# ----------------------------------------------------------------------------- API (embedded Postgres)
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app import db
    from app.api.main import app
    with TestClient(app) as c:
        before = {r["component"]: r["active_version"] for r in c.get("/api/components").json()}
        yield c
        with db.get_pool().connection() as conn:     # restore active versions, drop test rows
            for comp, v in before.items():
                if v is not None:
                    conn.execute("UPDATE component_versions SET is_active = (version = %s) WHERE component = %s",
                                 (v, comp))
            conn.execute("DELETE FROM component_versions WHERE created_by = %s", (AUTHOR,))
        versioning._invalidate()


def test_components_seeded(client):
    comps = {c["component"]: c for c in client.get("/api/components").json()}
    for name in ("answer_system_prompt", "router_prompt", "form_notes", "table_rules", "compaction_prompt",
                 "title_prompt", "grounding_prompt", "retrieval_params"):
        assert comps[name]["available"] and comps[name]["active_version"] >= 1, name
    h = client.get("/api/components/router_prompt").json()
    v1 = client.get("/api/components/router_prompt/versions/1").json()
    assert v1["created_by"] == "code" and v1["note"] == "seeded from code defaults"
    assert h["active"]["content"] == prompts.ROUTER_SYSTEM
    assert h["default"] == prompts.DEFAULT_COMPONENTS["router_prompt"]
    assert h["versions"][-1]["version"] == 1
    # runtime cache agrees with the DB
    assert versioning.active_versions()["router_prompt"] == h["active"]["version"]


def test_create_inactive_version_and_diff(client):
    cur = client.get("/api/components/title_prompt").json()["active"]
    new_text = cur["content"] + "\nUse Title Case."
    r = client.post("/api/components/title_prompt",
                    json={"content": new_text, "note": "test", "author": AUTHOR, "activate": False})
    assert r.status_code == 200, r.text
    row = r.json()
    assert row["is_active"] is False and row["version"] > cur["version"] and row["created_by"] == AUTHOR
    assert client.get("/api/components/title_prompt").json()["active"]["version"] == cur["version"]
    d = client.get(f"/api/components/title_prompt/diff?a={cur['version']}&b={row['version']}").json()
    assert d["added"] == 1 and d["removed"] == 0 and "+Use Title Case." in d["diff"] and not d["identical"]
    assert d["diff"].startswith(f"--- title_prompt@v{cur['version']}")
    # json components diff on pretty JSON
    rp = client.get("/api/components/retrieval_params").json()["active"]
    r2 = client.post("/api/components/retrieval_params",
                     json={"content": {**rp["content"], "final_k": 7}, "author": AUTHOR, "activate": False}).json()
    d2 = client.get(f"/api/components/retrieval_params/diff?a={rp['version']}&b={r2['version']}").json()
    assert '+  "final_k": 7' in d2["diff"] and d2["removed"] == 1


def test_activate_identical_version_switches_active(client):
    cur = client.get("/api/components/title_prompt").json()["active"]
    r = client.post("/api/components/title_prompt", json={"content": cur["content"], "author": AUTHOR})
    new = r.json()
    assert new["is_active"] is True
    assert versioning.active_versions()["title_prompt"] == new["version"]     # invalidated immediately
    assert prompts.TITLE_SYSTEM == cur["content"]
    back = client.post(f"/api/components/title_prompt/versions/{cur['version']}/activate").json()
    assert back["is_active"] is True and back["version"] == cur["version"]
    h = client.get("/api/components/title_prompt").json()
    assert sum(v["is_active"] for v in h["versions"]) == 1 and h["active"]["version"] == cur["version"]
    assert versioning.active_versions()["title_prompt"] == cur["version"]


def test_component_errors(client):
    assert client.get("/api/components/nope").status_code == 404
    assert client.get("/api/components/title_prompt/versions/99999").status_code == 404
    assert client.post("/api/components/title_prompt/versions/99999/activate").status_code == 404
    assert client.get("/api/components/title_prompt/diff?a=1&b=99999").status_code == 404
    r = client.post("/api/components/title_prompt", json={"content": "  ", "author": AUTHOR})
    assert r.status_code == 422 and "non-empty" in r.json()["detail"]
    r = client.post("/api/components/retrieval_params", json={"content": {"final_k": 0}, "author": AUTHOR})
    assert r.status_code == 422 and "final_k" in r.json()["detail"]
    r = client.post("/api/components/compaction_prompt",
                    json={"content": {"system": "{oops}", "user": "{turns}"}, "author": AUTHOR})
    assert r.status_code == 422 and "placeholder" in r.json()["detail"]
    assert client.post("/api/components/nope", json={"content": "x"}).status_code == 404


def test_arch_graph(client):
    g = client.get("/api/arch?days=30").json()
    ids = [n["id"] for n in g["nodes"]]
    for nid in ("input_guard", "memory", "router", "retrieval_embed", "retrieval_dense", "retrieval_bm25", "rrf",
                "rerank", "coverage", "ooc_gate", "prompt_builder", "llm", "output_guard", "citations",
                "memory_update", "ui"):
        assert nid in ids
    assert len(ids) == len(set(ids))
    for e in g["edges"]:
        assert e["from"] in ids and e["to"] in ids
    by = {n["id"]: n for n in g["nodes"]}
    assert "router_prompt" in by["router"]["components"] and by["router"]["purposes"] == ["router"]
    assert by["router"]["component_versions"]["router_prompt"] == g["components"]["router_prompt"]
    assert by["memory"]["stats"] is None
    st = by["llm"]["stats"]
    assert set(st) >= {"calls", "avg_ms", "recent_avg_ms", "p95_ms", "tokens", "errors", "by_purpose"}
    for comp in (c for n in g["nodes"] for c in n["components"]):
        assert comp in versioning.COMPONENTS
    assert client.get("/api/arch?days=0").status_code == 422
    assert client.get("/api/arch?mode=swift").status_code == 200
