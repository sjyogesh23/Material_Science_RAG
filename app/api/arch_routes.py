"""Architecture graph + versioned components API (FEATURES_V2 §6).

Mounted from app/api/main.py with ``app.include_router(arch_router)``. Endpoints are plain ``def`` (sync psycopg
pool in FastAPI's thread pool, like main.py).

    GET  /api/arch?days=7&mode=                       graph (nodes, edges, per-node llm_calls stats, components)
    GET  /api/components                              all components (active version, counts)
    GET  /api/components/{c}                          active version + history list + code default
    GET  /api/components/{c}/versions/{v}             one version
    POST /api/components/{c}  {content, note?, author?, activate?=true}   -> new (active) version
    POST /api/components/{c}/versions/{v}/activate    -> that version becomes active
    GET  /api/components/{c}/diff?a=&b=               unified diff a -> b (defaults: b = active, a = b - 1)
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app import versioning

log = logging.getLogger("newrag.arch")
router = APIRouter(prefix="/api", tags=["architecture"])

# ----------------------------------------------------------------------------- graph
# rank = row (top -> bottom), lane = column hint (0 = main column, -1 / +1 = side branches).
NODES: list[dict[str, Any]] = [
    {"id": "user_message", "label": "User message", "kind": "io", "rank": 0, "lane": 0,
     "description": "Chat request (message, mode, parent / edit / regenerate). Over-long messages keep the tail.",
     "components": [], "purposes": []},
    {"id": "input_guard", "label": "Input guardrails", "kind": "guardrail", "rank": 1, "lane": 0,
     "description": "Injection stripping (fake sources, 'ignore your rules', prompt-leak requests), jailbreak-only "
                    "messages -> #OutofContext, question-form detection.",
     "components": [], "purposes": []},
    {"id": "memory", "label": "Session memory", "kind": "memory", "rank": 2, "lane": 0,
     "description": "Markdown memory of the active branch (summary + recent turns + entities) for pronoun "
                    "resolution - never a source of facts.",
     "components": [], "purposes": []},
    {"id": "router", "label": "Jev-style router", "kind": "llm", "rank": 3, "lane": 0,
     "description": "Typed decisions: in_scope, intent, question_forms, needs_images, answer_format, "
                    "standalone_query, sub_queries. Skipped by the fast route when retrieval already decides.",
     "components": ["router_prompt", "modes"], "purposes": ["router"]},
    {"id": "retrieval_embed", "label": "Query embedding", "kind": "model", "rank": 4, "lane": -1,
     "description": "qwen3-embedding:0.6b (Ollama, CPU) with the query instruction prefix.",
     "components": [], "purposes": ["embed_query"]},
    {"id": "retrieval_dense", "label": "Dense search (pgvector)", "kind": "retrieval", "rank": 5, "lane": -1,
     "description": "HNSW cosine search over chunk embeddings (top_k_dense).",
     "components": ["retrieval_params"], "purposes": []},
    {"id": "retrieval_bm25", "label": "BM25 search", "kind": "retrieval", "rank": 5, "lane": 1,
     "description": "bm25s keyword search (top_k_bm25).",
     "components": ["retrieval_params"], "purposes": []},
    {"id": "rrf", "label": "RRF fusion", "kind": "retrieval", "rank": 6, "lane": 0,
     "description": "Reciprocal rank fusion of dense + BM25 lists (rrf_k).",
     "components": ["retrieval_params"], "purposes": []},
    {"id": "rerank", "label": "Reranker", "kind": "model", "rank": 7, "lane": 0,
     "description": "qwen3-reranker:0.6b yes/no logprob scores of the top rerank_top_n candidates.",
     "components": ["retrieval_params"], "purposes": ["rerank"]},
    {"id": "coverage", "label": "Coverage & cross-refs", "kind": "retrieval", "rank": 8, "lane": 0,
     "description": "Per-sub-query / per-item coverage, linked section text for figures, figure selection "
                    "(figure_min_score), final_k sources.",
     "components": ["retrieval_params"], "purposes": []},
    {"id": "ooc_gate", "label": "Out-of-context gate", "kind": "guardrail", "rank": 9, "lane": 0,
     "description": "best rerank < ooc_threshold, or router out-of-scope and < ooc_soft_threshold -> #OutofContext.",
     "components": ["retrieval_params"], "purposes": []},
    {"id": "prompt_builder", "label": "Prompt builder", "kind": "prompt", "rank": 10, "lane": 0,
     "description": "System prompt + FORMAT rule + form notes + session memory + numbered SOURCES.",
     "components": ["answer_system_prompt", "table_rules", "form_notes"], "purposes": []},
    {"id": "llm", "label": "Answer LLM", "kind": "llm", "rank": 11, "lane": 0,
     "description": "Streams the cited answer (Swift: qwen3:4b-instruct; Deep: qwen3.5:9b with thinking).",
     "components": ["modes"], "purposes": ["answer"]},
    {"id": "output_guard", "label": "Output guardrails", "kind": "guardrail", "rank": 12, "lane": 0,
     "description": "Refusal retry, prompt-leak / injected-canary removal, grounding check (+ caution).",
     "components": ["form_notes", "grounding_prompt"], "purposes": ["grounding_check"]},
    {"id": "citations", "label": "Citations & highlights", "kind": "postprocess", "rank": 13, "lane": 0,
     "description": "Validates [n] markers, maps them to sources, page bboxes for highlighting, figure images.",
     "components": [], "purposes": []},
    {"id": "memory_update", "label": "Memory update", "kind": "memory", "rank": 14, "lane": 0,
     "description": "Appends the turn, compacts the memory when over budget, names new sessions.",
     "components": ["compaction_prompt", "title_prompt"], "purposes": ["memory_compaction", "title"]},
    {"id": "ui", "label": "Chat UI", "kind": "io", "rank": 15, "lane": 0,
     "description": "SSE events: status, token, sources, images, answer, usage, done.",
     "components": [], "purposes": []},
]
EDGES: list[dict[str, Any]] = [
    {"from": "user_message", "to": "input_guard", "kind": "main"},
    {"from": "input_guard", "to": "memory", "kind": "main"},
    {"from": "input_guard", "to": "ui", "kind": "alt", "label": "jailbreak only -> #OutofContext"},
    {"from": "memory", "to": "router", "kind": "main"},
    {"from": "memory", "to": "retrieval_embed", "kind": "alt", "label": "fast route: retrieve first"},
    {"from": "router", "to": "retrieval_embed", "kind": "main", "label": "standalone + sub-queries"},
    {"from": "router", "to": "retrieval_bm25", "kind": "main"},
    {"from": "retrieval_embed", "to": "retrieval_dense", "kind": "main"},
    {"from": "retrieval_dense", "to": "rrf", "kind": "main"},
    {"from": "retrieval_bm25", "to": "rrf", "kind": "main"},
    {"from": "rrf", "to": "rerank", "kind": "main"},
    {"from": "rerank", "to": "coverage", "kind": "main"},
    {"from": "rerank", "to": "router", "kind": "alt", "label": "fast route decides (score >= 0.90 / < OOC)"},
    {"from": "coverage", "to": "ooc_gate", "kind": "main"},
    {"from": "ooc_gate", "to": "prompt_builder", "kind": "main", "label": "in scope"},
    {"from": "ooc_gate", "to": "ui", "kind": "alt", "label": "#OutofContext"},
    {"from": "prompt_builder", "to": "llm", "kind": "main"},
    {"from": "llm", "to": "output_guard", "kind": "main"},
    {"from": "output_guard", "to": "llm", "kind": "alt", "label": "refusal retry / self-revision"},
    {"from": "output_guard", "to": "citations", "kind": "main"},
    {"from": "citations", "to": "memory_update", "kind": "main"},
    {"from": "memory_update", "to": "ui", "kind": "main"},
]


def _purpose_stats(days: int, mode: str | None) -> dict[str, dict[str, Any]]:
    from app import db
    where = "created_at >= now() - make_interval(days => %s)"
    params: list[Any] = [int(days)]
    if mode:
        where += " AND meta->>'mode' = %s"
        params.append(mode)
    sql = f"""
        WITH c AS (
            SELECT purpose, duration_ms, prompt_tokens, completion_tokens, ok, created_at,
                   row_number() OVER (PARTITION BY purpose ORDER BY created_at DESC) AS rn
            FROM llm_calls WHERE {where})
        SELECT purpose, count(*), avg(duration_ms),
               percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms),
               percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms),
               avg(duration_ms) FILTER (WHERE rn <= 20),
               coalesce(sum(prompt_tokens), 0), coalesce(sum(completion_tokens), 0),
               count(*) FILTER (WHERE NOT coalesce(ok, true)), max(created_at)
        FROM c GROUP BY purpose"""
    with db.get_pool().connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    def num(x: Any) -> float | None:
        return round(float(x), 1) if x is not None else None

    return {r[0] or "unknown": {"calls": int(r[1]), "avg_ms": num(r[2]), "p50_ms": num(r[3]), "p95_ms": num(r[4]),
                                "recent_avg_ms": num(r[5]), "prompt_tokens": int(r[6]),
                                "completion_tokens": int(r[7]), "tokens": int(r[6]) + int(r[7]),
                                "errors": int(r[8]), "last_at": r[9].isoformat() if r[9] else None}
            for r in rows}


def _node_stats(purposes: list[str], by_purpose: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    hits = [by_purpose[p] for p in purposes if p in by_purpose]
    if not purposes:
        return None
    calls = sum(h["calls"] for h in hits)

    def wavg(key: str) -> float | None:
        vals = [(h[key], h["calls"]) for h in hits if h[key] is not None]
        n = sum(c for _, c in vals)
        return round(sum(v * c for v, c in vals) / n, 1) if n else None

    last = max((h["last_at"] for h in hits if h["last_at"]), default=None)
    return {"calls": calls, "avg_ms": wavg("avg_ms"), "recent_avg_ms": wavg("recent_avg_ms"),
            "p95_ms": max((h["p95_ms"] for h in hits if h["p95_ms"] is not None), default=None),
            "prompt_tokens": sum(h["prompt_tokens"] for h in hits),
            "completion_tokens": sum(h["completion_tokens"] for h in hits),
            "tokens": sum(h["tokens"] for h in hits), "errors": sum(h["errors"] for h in hits),
            "last_at": last, "by_purpose": {p: by_purpose[p] for p in purposes if p in by_purpose}}


@router.get("/arch")
def get_arch(days: int = Query(7, ge=1, le=3650), mode: str | None = None) -> dict[str, Any]:
    try:
        by_purpose = _purpose_stats(days, mode)
        active = {c["component"]: c["active_version"] for c in versioning.list_components()}
    except Exception as e:  # noqa: BLE001
        log.exception("arch stats failed")
        raise HTTPException(503, f"Postgres unavailable: {e}") from e
    mapped = {p for n in NODES for p in n["purposes"]}
    nodes = [{**n, "component_versions": {c: active.get(c) for c in n["components"]},
              "stats": _node_stats(n["purposes"], by_purpose)} for n in NODES]
    return {"days": days, "mode": mode, "nodes": nodes, "edges": EDGES, "components": active,
            "other_purposes": {p: s for p, s in by_purpose.items() if p not in mapped}}


# ----------------------------------------------------------------------------- components
class ComponentIn(BaseModel):
    content: Any = Field(..., description="text components: a string; json components: an object")
    note: str | None = Field(None, max_length=500)
    author: str | None = Field(None, max_length=64)
    activate: bool = True


def _known(c: str) -> None:
    if c not in versioning.COMPONENTS:
        raise HTTPException(404, f"unknown component {c!r}; known: {', '.join(versioning.COMPONENTS)}")


@router.get("/components")
def list_components() -> list[dict[str, Any]]:
    return versioning.list_components()


@router.get("/components/{c}")
def get_component(c: str) -> dict[str, Any]:
    _known(c)
    return versioning.get_history(c)


@router.get("/components/{c}/versions/{v}")
def get_version(c: str, v: int) -> dict[str, Any]:
    _known(c)
    row = versioning.get_version(c, v)
    if row is None:
        raise HTTPException(404, f"{c} has no version {v}")
    return row


@router.post("/components/{c}")
def create_version(c: str, body: ComponentIn) -> dict[str, Any]:
    _known(c)
    try:
        return versioning.create_version(c, body.content, body.note, created_by=(body.author or "ui").strip() or "ui",
                                         activate=body.activate)
    except versioning.ComponentError as e:
        raise HTTPException(422, str(e)) from e


@router.post("/components/{c}/versions/{v}/activate")
def activate_version(c: str, v: int) -> dict[str, Any]:
    _known(c)
    row = versioning.activate(c, v)
    if row is None:
        raise HTTPException(404, f"{c} has no version {v}")
    return row


@router.get("/components/{c}/diff")
def diff_versions(c: str, a: int | None = Query(None, ge=1), b: int | None = Query(None, ge=1)) -> dict[str, Any]:
    _known(c)
    d = versioning.diff(c, a, b)
    if d is None:
        raise HTTPException(404, f"{c}: version not found")
    return d
