"""API tests (FastAPI TestClient against the real embedded Postgres; the chat test also needs Ollama and
loads the retrieval models on CPU).

    .venv\\Scripts\\python -m pytest tests/test_api.py -v
"""
from __future__ import annotations

import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("NEWRAG_SKIP_WARMUP", "1")      # models load lazily (only the chat test needs them)

from fastapi.testclient import TestClient  # noqa: E402

from app.api.main import app  # noqa: E402
from app.config import get_settings  # noqa: E402

DOC_ID = "doc_f49a86e05ec4"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen(f"{get_settings().ollama_host}/api/tags", timeout=2)
        return True
    except Exception:  # noqa: BLE001
        return False


def _png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def parse_sse(text: str) -> list[tuple[str, object]]:
    out = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        event, data = "message", []
        for line in frame.split("\n"):
            if not line or line.startswith(":"):
                continue
            k, _, v = line.partition(":")
            v = v[1:] if v.startswith(" ") else v
            if k == "event":
                event = v
            elif k == "data":
                data.append(v)
        if data:
            out.append((event, json.loads("\n".join(data))))
    return out


# ----------------------------------------------------------------------------- basics
def test_static_pages(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert client.get("/audit").status_code == 200
    assert client.get("/static/js/chat.js").status_code == 200
    assert client.get("/docs").status_code == 200


def test_health(client):
    h = client.get("/api/health").json()
    for k in ("postgres", "ollama", "chat_model_present", "chunks", "embedder_loaded", "retriever_ready"):
        assert k in h
    assert h["postgres"] is True
    assert h["chunks"] > 0


# ----------------------------------------------------------------------------- sessions
def test_sessions_crud(client):
    r = client.post("/api/sessions")
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert r.json()["title"]
    try:
        ids = [s["session_id"] for s in client.get("/api/sessions").json()]
        assert sid in ids
        row = next(s for s in client.get("/api/sessions").json() if s["session_id"] == sid)
        assert set(row) >= {"session_id", "title", "updated_at", "message_count"}

        r = client.patch(f"/api/sessions/{sid}", json={"title": "Renamed by pytest"})
        assert r.status_code == 200 and r.json()["title"] == "Renamed by pytest"
        assert client.get(f"/api/sessions/{sid}/messages").json() == []
        mem = client.get(f"/api/sessions/{sid}/memory").json()
        assert set(mem) >= {"markdown", "tokens", "budget"}
        assert client.patch(f"/api/sessions/{sid}", json={"title": ""}).status_code == 422
    finally:
        assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.delete(f"/api/sessions/{sid}").status_code == 404
    assert client.get(f"/api/sessions/{sid}/messages").status_code == 404
    assert client.get("/api/sessions/not-a-uuid/messages").status_code == 404


# ----------------------------------------------------------------------------- settings
def test_settings_roundtrip(client):
    s = client.get("/api/settings").json()
    assert "chat_model" in s and "ooc_threshold" in s and "database_url" not in s
    orig = s["final_k"]
    try:
        r = client.put("/api/settings", json={"final_k": orig + 1})
        assert r.status_code == 200, r.text
        assert r.json()["final_k"] == orig + 1
        assert client.get("/api/settings").json()["final_k"] == orig + 1
        assert get_settings().final_k == orig + 1
    finally:
        assert client.put("/api/settings", json={"final_k": orig}).json()["final_k"] == orig
    assert client.put("/api/settings", json={"pg_data_dir": "x"}).status_code == 422
    assert client.put("/api/settings", json={"final_k": "lots"}).status_code == 422


# ----------------------------------------------------------------------------- pages / figures
def test_page_png_and_meta(client):
    meta = client.get(f"/api/page/{DOC_ID}/58/meta").json()
    assert meta["width"] > 100 and meta["height"] > 100
    r = client.get(f"/api/page/{DOC_ID}/58.png?dpi=72")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    w, h = _png_size(r.content)
    assert abs(w - meta["width"]) <= 2 and abs(h - meta["height"]) <= 2
    r = client.get(f"/api/page/{DOC_ID}/58.png?dpi=5000")          # clamped to 200 dpi
    w, _ = _png_size(r.content)
    assert abs(w - meta["width"] * 200 / 72) <= 3
    assert (Path(get_settings().page_cache_dir) / f"{DOC_ID}_p58_200.png").exists()
    assert client.get(f"/api/page/{DOC_ID}/9999.png").status_code == 404
    assert client.get("/api/page/doc_nope/1.png").status_code == 404


def test_figure(client):
    from app import db
    with db.get_pool().connection() as conn:
        row = conn.execute("SELECT chunk_id FROM chunks WHERE chunk_type='figure' "
                           "AND payload->'figure'->>'image_file' IS NOT NULL LIMIT 1").fetchone()
    assert row, "no figure chunk in the DB"
    r = client.get(f"/api/figure/{row[0]}")
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/")
    assert len(r.content) > 500
    assert client.get("/api/figure/doc_f49a86e05ec4:nope").status_code == 404


# ----------------------------------------------------------------------------- audit
def test_audit(client):
    s = client.get("/api/audit/summary?days=30").json()
    assert set(s) >= {"days", "totals", "by_purpose", "by_model", "by_day", "by_session"}
    assert set(s["totals"]) >= {"calls", "prompt_tokens", "completion_tokens", "total_tokens", "avg_ms", "failed"}
    calls = client.get("/api/audit/calls?limit=5").json()
    assert isinstance(calls, list) and len(calls) <= 5
    if calls:
        assert set(calls[0]) >= {"id", "purpose", "provider", "model", "prompt_tokens", "completion_tokens",
                                 "total_tokens", "duration_ms", "ok", "created_at"}


# ----------------------------------------------------------------------------- chat (SSE)
@pytest.mark.skipif(not _ollama_up(), reason="Ollama is not running")
def test_chat_out_of_context(client):
    with client.stream("POST", "/api/chat", json={"message": "What is the capital of France?"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(r.read().decode("utf-8"))
    names = [e for e, _ in events]
    assert names[0] == "session" and names[-1] == "done", names
    assert "error" not in names, events
    data = {e: d for e, d in events}
    sid = data["session"]["session_id"]
    try:
        tokens = "".join(d["t"] for e, d in events if e == "token")
        assert tokens == "#OutofContext"
        if "answer" in data:
            assert data["answer"]["content"] == "#OutofContext"
        assert data["sources"] == []
        assert data["images"] == []
        assert data["done"]["out_of_context"] is True
        assert "usage" in data
        msgs = client.get(f"/api/sessions/{sid}/messages").json()
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[1]["content"] == "#OutofContext" and msgs[1]["out_of_context"] is True
        assert msgs[1]["sources"] == []
    finally:
        client.delete(f"/api/sessions/{sid}")


def test_chat_empty_message(client):
    with client.stream("POST", "/api/chat", json={"message": "   "}) as r:
        events = parse_sse(r.read().decode("utf-8"))
    assert [e for e, _ in events] == ["error", "done"]
