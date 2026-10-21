"""Mock for the v2 audit page (feedback, architecture, versioned components, mode-split usage).

Extends tests/mocks/mock_server.py with the docs/FEATURES_V2.md §1/§5/§6 endpoints:

    .venv/Scripts/python tests/mocks/mock_audit_v2.py [--port 8766]

Then open http://127.0.0.1:8766/audit . Self-test: /__test_audit?mode=usage|feedback|arch|settings
(drive with headless Chrome --dump-dom; result in the "RESULT {...}" line).
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
from datetime import timedelta
from pathlib import Path

import uvicorn
from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mock_server as base  # noqa: E402

app = base.app
now = base.now
rnd = random.Random(11)

# ------------------------------------------------------------------ calls: add mode + feedback_review
for c in base.CALLS:
    if c["purpose"] in ("router", "answer", "grounding_check", "embed_query", "rerank"):
        c["meta"] = {"mode": "deep" if rnd.random() < 0.3 else "swift"}
        if c["meta"]["mode"] == "deep" and c["purpose"] == "answer":
            c["model"] = "qwen3.5:9b"
            c["duration_ms"] = int(c["duration_ms"] * 6)
for i in range(6):
    t = now() - timedelta(days=rnd.randint(0, 9), minutes=rnd.randint(0, 900))
    base.CALLS.append({"id": 9000 + i, "session_id": None, "message_id": None, "purpose": "feedback_review",
                       "provider": "ollama", "model": "qwen3:4b-instruct-2507-q4_K_M", "prompt_tokens": 1900,
                       "completion_tokens": 160, "total_tokens": 2060, "duration_ms": 5200, "ok": True, "error": None,
                       "meta": {}, "created_at": t.isoformat()})
base.CALLS.sort(key=lambda r: r["created_at"], reverse=True)

# replace /api/audit/summary to also return by_mode
app.router.routes = [r for r in app.router.routes if getattr(r, "path", "") != "/api/audit/summary"]


@app.get("/api/audit/summary")
def audit_summary_v2(days: int = 30):
    out = base.audit_summary(days)
    since = now() - timedelta(days=days)
    from datetime import datetime
    rows = [r for r in base.CALLS if datetime.fromisoformat(r["created_at"]) >= since]
    out["by_mode"] = [{"mode": k, **v} for k, v in base.agg(rows, lambda r: (r.get("meta") or {}).get("mode") or "none").items()]
    return out


# ------------------------------------------------------------------ feedback
QUESTIONS = [
    ("What is the difference between edge and screw dislocations?", "answer", "comparison"),
    ("Explain Fick's second law with an example.", "answer", "explanation"),
    ("What does Figure 4.2 show?", "answer", "figure"),
    ("Give me a table of the 7 crystal systems.", "answer", "table"),
    ("Why is FCC more ductile than BCC?", "answer", "explanation"),
    ("What is the Hall-Petch relation?", "ooc", "definition"),
    ("Define vacancy.", "answer", "definition"),
    ("Compare annealing and quenching", "answer", "comparison"),
]
FEEDBACK: list[dict] = []
REVIEWS: dict[int, dict] = {}
mid = 70000
for d in range(24):
    for _ in range(rnd.randint(0, 4)):
        mid += 1
        q, kind, intent = rnd.choice(QUESTIONS)
        rating = -1 if rnd.random() < 0.32 else 1
        ans = "#OutofContext" if kind == "ooc" else base.ANSWER
        src = [] if kind == "ooc" else [base.make_source(1, "doc_f49a86e05ec4:c00104", 0.91), base.make_source(2, "doc_f49a86e05ec4:c00107", 0.74)]
        FEEDBACK.append({"message_id": mid, "session_id": next(iter(base.SESSIONS)), "rating": rating,
                         "comment": ("The table cites the wrong source." if rating < 0 and rnd.random() < 0.5 else None),
                         "created_at": (now() - timedelta(days=d, minutes=rnd.randint(0, 600))).isoformat(),
                         "question": q, "answer": ans, "sources": src, "mode": "deep" if rnd.random() < 0.3 else "swift",
                         "intent": intent})
FEEDBACK.sort(key=lambda r: r["created_at"], reverse=True)
first_dislike = next(f for f in FEEDBACK if f["rating"] < 0)
REVIEWS[first_dislike["message_id"]] = {"issue_type": "missing_citation", "explanation": "The second paragraph states the strain-energy formula without citing [1].",
                                        "suggested_fix": "Cite [1] after the formula; strengthen rule 'every factual sentence cites'.",
                                        "model": "qwen3:4b-instruct-2507-q4_K_M", "created_at": now().isoformat()}


def fb_out(f):
    r = REVIEWS.get(f["message_id"])
    review = {k: r[k] for k in ("issue_type", "explanation", "suggested_fix")} if r else None
    return {**f, "updated_at": f["created_at"], "out_of_context": f["answer"] == "#OutofContext", "session_title": "Mock session",
            "review": review, "review_model": r and r["model"], "reviewed_at": r and r["created_at"]}


@app.get("/api/feedback")
def feedback_list(rating: int | None = None, limit: int = 50):
    rows = [f for f in FEEDBACK if rating is None or f["rating"] == rating]
    return [fb_out(f) for f in rows[:limit]]


@app.get("/api/feedback/summary")
def feedback_summary(days: int = 30):
    since = now() - timedelta(days=days)
    from datetime import datetime
    rows = [f for f in FEEDBACK if datetime.fromisoformat(f["created_at"]) >= since]

    def grp(key):
        out: dict = {}
        for f in rows:
            a = out.setdefault(key(f), {"likes": 0, "dislikes": 0})
            a["likes" if f["rating"] > 0 else "dislikes"] += 1
        return out
    likes = sum(1 for f in rows if f["rating"] > 0)
    dislikes = len(rows) - likes
    issues: dict = {}
    for f in rows:
        if f["rating"] < 0 and f["message_id"] in REVIEWS:
            k = REVIEWS[f["message_id"]]["issue_type"]
            issues[k] = issues.get(k, 0) + 1
    return {"days": days, "totals": {"likes": likes, "dislikes": dislikes, "total": len(rows),
                                     "ratio": round(likes / len(rows), 4) if rows else None, "answers": len(rows) * 3,
                                     "rated_share": 0.3333},
            "unreviewed_dislikes": sum(1 for f in FEEDBACK if f["rating"] < 0 and f["message_id"] not in REVIEWS),
            "by_issue_type": [{"issue_type": k, "n": v} for k, v in sorted(issues.items(), key=lambda x: -x[1])],
            "by_day": sorted(({"day": k, **v} for k, v in grp(lambda f: f["created_at"][:10]).items()), key=lambda x: x["day"]),
            "by_mode": [{"mode": k, **v} for k, v in grp(lambda f: f["mode"]).items()],
            "by_intent": [{"intent": k, **v} for k, v in grp(lambda f: f["intent"]).items()]}


ISSUES = ["wrong_fact", "missing_citation", "bad_format", "refused_wrongly", "incomplete", "other"]


@app.post("/api/feedback/review")
async def feedback_review(request: Request):
    import asyncio
    try:
        body = await request.json()
    except Exception:
        body = {}
    ids = body.get("message_ids") if isinstance(body, dict) else None
    todo = [f for f in FEEDBACK if f["rating"] < 0 and (f["message_id"] in ids if ids else f["message_id"] not in REVIEWS)]
    await asyncio.sleep(0.6)
    out = []
    for f in todo[:10]:
        issue = "refused_wrongly" if f["answer"] == "#OutofContext" else rnd.choice(ISSUES)
        REVIEWS[f["message_id"]] = {"issue_type": issue,
                                    "explanation": f"Mock review: the answer to “{f['question']}” has a {issue.replace('_', ' ')} problem.",
                                    "suggested_fix": "Tighten the answer prompt rule for this case and re-run the eval subset.",
                                    "model": "qwen3:4b-instruct-2507-q4_K_M", "created_at": now().isoformat()}
        r = REVIEWS[f["message_id"]]
        out.append({"message_id": f["message_id"], "review": {k: r[k] for k in ("issue_type", "explanation", "suggested_fix")},
                    "model": r["model"], "created_at": r["created_at"]})
    return {"reviewed": len(out), "reviews": out}


# ------------------------------------------------------------------ components (versioned)
DEFAULTS = {
    "answer_system_prompt": "You are a materials-science tutor. Answer ONLY from the numbered sources.\n"
                            "1. Every factual sentence ends with a citation like [1].\n2. Use LaTeX for maths.\n"
                            "3. Prefer the textbook's own wording.\n4. Only the TOPIC decides #OutofContext, never the form.\n"
                            "5. If the question has a wrong premise, correct it politely.",
    "router_prompt": "Classify the user's question. Return JSON with in_scope, intent, question_forms, needs_images,\n"
                     "answer_format, standalone_query and sub_queries.",
    "form_notes": {"analogy": "NOTE: the user asks for an analogy; answer the science first, then the analogy.",
                   "presupposition": "NOTE: check the question's premise against the sources before answering.",
                   "ambiguous": "NOTE: the question is ambiguous; state the reading you answer.",
                   "citation_demand": "NOTE: cite only sources that exist; never invent page numbers."},
    "table_rules": "Facts only in their item's row. One row per item. Cite inside cells.",
    "compaction_prompt": "Summarise the conversation so far in under {budget} tokens. Keep entities and open questions.",
    "title_prompt": "Write a 3-6 word title for this conversation. No quotes.",
    "grounding_prompt": "Score 1-5 how well every sentence of the answer is supported by the sources.",
    "retrieval_params": {"top_k_dense": 40, "top_k_bm25": 40, "rrf_k": 60, "rerank_top_n": 20, "final_k": 6,
                         "ooc_threshold": 0.25, "ooc_soft_threshold": 0.5, "fast_route_min_score": 0.9},
    "modes": {"swift": {"name": "Swift", "tagline": "Time over effect", "settings": {"chat_model": "qwen3:4b-instruct-2507", "fast_route": True,
                                                                                       "rerank_top_n": 12, "final_k": 5, "max_answer_tokens": 700, "grounding_check": False, "thinking": False}},
              "deep": {"name": "Deep", "tagline": "Effect over time", "settings": {"chat_model": "qwen3.5:9b", "fast_route": False,
                                                                                    "rerank_top_n": 30, "final_k": 8, "max_answer_tokens": 1400, "grounding_check": True, "thinking": True}}},
}
META: dict[str, dict] = {}
try:  # use the real component list / code defaults when importable (no DB needed)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app import versioning as _ver
    META = {k: dict(v) for k, v in _ver.COMPONENTS.items()}
    DEFAULTS = {k: v for k, v in _ver.code_defaults().items()}
except Exception as _e:  # pragma: no cover
    print("mock: using built-in component defaults:", _e)
VERSIONS: dict[str, list[dict]] = {}
for name, content in DEFAULTS.items():
    VERSIONS[name] = [{"component": name, "version": 1, "content": content, "note": "seeded from code defaults",
                       "created_at": (now() - timedelta(days=2)).isoformat(), "created_by": "code", "is_active": True}]
VERSIONS["answer_system_prompt"].append({"component": "answer_system_prompt", "version": 2,
                                         "content": DEFAULTS["answer_system_prompt"] + "\nExtra rule: figures are cited with the figure's own [n].",
                                         "note": "Round 1: figure citations", "created_at": (now() - timedelta(hours=5)).isoformat(),
                                         "created_by": "ui", "is_active": False})
VERSIONS["answer_system_prompt"][0]["is_active"] = False
VERSIONS["answer_system_prompt"][1]["is_active"] = True


def comp_or_404(c):
    if c not in VERSIONS:
        raise HTTPException(404, f"Unknown component {c}")
    return VERSIONS[c]


def active(c):
    return next(v for v in VERSIONS[c] if v["is_active"])


def meta(v):
    return {k: v[k] for k in ("version", "note", "created_at", "created_by", "is_active")}


@app.get("/api/components")
def components():
    return [{"component": c, "active_version": active(c)["version"], "versions": len(v)} for c, v in VERSIONS.items()]


@app.get("/api/components/{c}")
def component(c: str):
    vs = comp_or_404(c)
    return {"component": c, **META.get(c, {}), "active": active(c), "default": DEFAULTS.get(c),
            "versions": [meta(v) for v in sorted(vs, key=lambda v: -v["version"])]}


@app.get("/api/components/{c}/versions/{v}")
def component_version(c: str, v: int):
    for x in comp_or_404(c):
        if x["version"] == v:
            return x
    raise HTTPException(404, "No such version")


@app.post("/api/components/{c}")
async def component_save(c: str, request: Request):
    vs = comp_or_404(c)
    body = await request.json()
    content = body.get("content")
    if content is None or content == "":
        raise HTTPException(422, "content must not be empty")
    if c == "retrieval_params":
        bad = [k for k, x in content.items() if not isinstance(x, (int, float)) or x < 0]
        if bad:
            raise HTTPException(422, f"retrieval_params: {', '.join(bad)} must be non-negative numbers")
    for x in vs:
        x["is_active"] = False
    new = {"component": c, "version": max(x["version"] for x in vs) + 1, "content": content, "note": body.get("note") or "",
           "created_at": now().isoformat(), "created_by": "ui", "is_active": True}
    vs.append(new)
    return new


@app.post("/api/components/{c}/versions/{v}/activate")
def component_activate(c: str, v: int):
    vs = comp_or_404(c)
    if not any(x["version"] == v for x in vs):
        raise HTTPException(404, "No such version")
    for x in vs:
        x["is_active"] = x["version"] == v
    return {"component": c, "active_version": v}


def as_text(content):
    return content if isinstance(content, str) else json.dumps(content, indent=2, ensure_ascii=False)


@app.get("/api/components/{c}/diff")
def component_diff(c: str, a: int, b: int):
    vs = {x["version"]: x for x in comp_or_404(c)}
    if a not in vs or b not in vs:
        raise HTTPException(404, "No such version")
    lines = difflib.unified_diff(as_text(vs[a]["content"]).splitlines(), as_text(vs[b]["content"]).splitlines(),
                                 fromfile=f"{c} v{a}", tofile=f"{c} v{b}", lineterm="")
    lines = list(lines)
    return {"component": c, "a": a, "b": b, "diff": "\n".join(lines), "added": sum(1 for x in lines if x.startswith("+") and not x.startswith("+++")),
            "removed": sum(1 for x in lines if x.startswith("-") and not x.startswith("---")), "identical": not lines}


# ------------------------------------------------------------------ architecture graph
NODES = [
    ("user", "User message", "io", None, "Question text, mode and parent message", [], []),
    ("guard_in", "Injection strip", "guard", None, "strip_injection(), guardrails-in, >4000-char tail", [], []),
    ("memory", "Session memory", "memory", None, "Recent turns + entities + summary (markdown file)", [], []),
    ("router", "Jev-style router", "llm", None, "in_scope · intent · question_forms · needs_images · answer_format · standalone_query · sub_queries",
     ["router_prompt"], ["router"]),
    ("embed", "Embed query", "retrieval", "Retrieval", "qwen3-embedding:0.6b, Instruct prefix", [], ["embed_query"]),
    ("dense", "Dense · pgvector", "retrieval", "Retrieval", "cosine top-k", ["retrieval_params"], []),
    ("bm25", "BM25 · bm25s", "retrieval", "Retrieval", "keyword top-k", ["retrieval_params"], []),
    ("rrf", "RRF fusion", "retrieval", "Retrieval", "k = rrf_k", ["retrieval_params"], []),
    ("rerank", "Rerank", "retrieval", "Retrieval", "qwen3-reranker yes/no logprobs", ["retrieval_params"], ["rerank"]),
    ("expand", "Coverage · cross-refs · figures", "retrieval", "Retrieval", "per-sub-query hits, chunk_links, section text", [], []),
    ("ooc", "OOC gate", "guard", None, "best rerank vs ooc_threshold / soft threshold", ["retrieval_params"], []),
    ("prompt", "Prompt builder", "build", None, "system prompt + form notes + memory + numbered sources",
     ["answer_system_prompt", "form_notes", "table_rules"], []),
    ("llm", "LLM answer", "llm", None, "Swift: qwen3:4b · Deep: qwen3.5:9b + thinking", ["modes"], ["answer"]),
    ("guard_out", "Output guardrails", "guard", None, "refusal retry · injection canaries · citation validation · grounding",
     ["grounding_prompt"], ["grounding_check"]),
    ("cite", "Citations + highlights", "build", None, "[n] → sources, page bboxes, figures", [], []),
    ("mem_update", "Memory update", "memory", None, "append turn, compact over budget, title", ["compaction_prompt", "title_prompt"],
     ["memory_compaction", "title"]),
    ("ui", "Chat UI", "io", None, "SSE stream: tokens, sources, images, usage", [], []),
]
EDGES = [("user", "guard_in"), ("guard_in", "memory"), ("memory", "router"), ("router", "embed", "standalone_query"),
         ("embed", "dense"), ("embed", "bm25"), ("dense", "rrf"), ("bm25", "rrf"), ("rrf", "rerank"),
         ("rerank", "expand"), ("expand", "ooc"), ("router", "ooc", "in_scope"), ("ooc", "prompt", "in scope"), ("ooc", "ui", "#OutofContext"),
         ("prompt", "llm"), ("llm", "guard_out"), ("guard_out", "cite"), ("cite", "mem_update"), ("mem_update", "ui"),
         ("mem_update", "memory", "next turn")]


try:
    from app.api.arch_routes import EDGES as REAL_EDGES, NODES as REAL_NODES
except Exception as _e:  # pragma: no cover
    REAL_NODES = REAL_EDGES = None
    print("mock: using built-in graph:", _e)


@app.get("/api/arch")
def arch(days: int = 7, mode: str | None = None):
    if REAL_NODES:
        from datetime import datetime
        since = now() - timedelta(days=days)
        rows = [r for r in base.CALLS if datetime.fromisoformat(r["created_at"]) >= since
                and (not mode or (r.get("meta") or {}).get("mode") == mode)]
        nodes = []
        for n in REAL_NODES:
            rs = [r for r in rows if r["purpose"] in n["purposes"]]
            stats = None
            if n["purposes"]:
                ms = sorted(r["duration_ms"] for r in rs)
                stats = {"calls": len(rs), "avg_ms": round(sum(ms) / len(ms), 1) if ms else None,
                         "recent_avg_ms": round(sum(ms[-20:]) / len(ms[-20:]), 1) if ms else None,
                         "p95_ms": ms[int(len(ms) * 0.95) - 1] if ms else None,
                         "prompt_tokens": sum(r["prompt_tokens"] for r in rs), "completion_tokens": sum(r["completion_tokens"] for r in rs),
                         "tokens": sum(r["total_tokens"] for r in rs), "errors": sum(1 for r in rs if not r["ok"]),
                         "last_at": max((r["created_at"] for r in rs), default=None)}
            nodes.append({**n, "component_versions": {c: active(c)["version"] for c in n["components"] if c in VERSIONS}, "stats": stats})
        mapped = {p for n in REAL_NODES for p in n["purposes"]}
        other = {}
        for r in rows:
            if r["purpose"] not in mapped:
                o = other.setdefault(r["purpose"], {"calls": 0, "ms": 0})
                o["calls"] += 1
                o["ms"] += r["duration_ms"]
        return {"days": days, "mode": mode, "nodes": nodes, "edges": REAL_EDGES,
                "components": {c: active(c)["version"] for c in VERSIONS},
                "other_purposes": {k: {"calls": v["calls"], "avg_ms": round(v["ms"] / v["calls"], 1)} for k, v in other.items()}}
    return arch_builtin(days)


def arch_builtin(days: int = 30):
    from datetime import datetime
    since = now() - timedelta(days=days)
    rows = [r for r in base.CALLS if datetime.fromisoformat(r["created_at"]) >= since]
    nodes = []
    for nid, label, kind, group, desc, comps, purposes in NODES:
        rs = [r for r in rows if r["purpose"] in purposes]
        stats = None
        if purposes:
            stats = {"calls": len(rs), "avg_ms": round(sum(r["duration_ms"] for r in rs) / len(rs), 1) if rs else None,
                     "tokens": sum(r["total_tokens"] for r in rs), "purposes": purposes}
        nodes.append({"id": nid, "label": label, "kind": kind, "group": group, "description": desc,
                      "components": comps, "stats": stats})
    return {"days": days, "nodes": nodes,
            "edges": [{"from": e[0], "to": e[1], **({"label": e[2]} if len(e) > 2 else {})} for e in EDGES],
            "components": {c: active(c)["version"] for c in VERSIONS}}


# ------------------------------------------------------------------ self-test harness
HARNESS = (Path(__file__).resolve().parent / "mock_audit_v2_harness.html")


LOG: dict = {}


@app.post("/__log")
async def test_log(request: Request):
    LOG["last"] = await request.json()
    return "ok"


@app.get("/__log")
def test_log_get():
    return LOG.get("last")


@app.get("/__test_audit")
def test_audit(mode: str = "usage", w: int = 1366, h: int = 900, act: str = ""):
    html = HARNESS.read_text(encoding="utf-8")
    html = html.replace("__MODE__", mode.replace('"', "")).replace("__W__", str(w)).replace("__H__", str(h)).replace("__ACT__", act.replace('"', ""))
    return HTMLResponse(html)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
