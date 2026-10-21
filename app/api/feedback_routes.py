"""Feedback API (FEATURES_V2 §1): like / dislike per assistant message, KPIs, and LLM reviews of dislikes.

    POST /api/messages/{id}/feedback {rating: 1|-1|0, comment?}  -> {message_id, rating}      (0 = remove)
    GET  /api/feedback?rating=-1&limit=&mode=&days=               -> [{message_id, session_id, rating, comment,
         question, answer, sources, mode, intent, out_of_context, review, created_at, updated_at, answered_at}]
    POST /api/feedback/review {message_ids?: [..], limit?}        -> {reviewed, reviews: [{message_id, review, model,
         created_at}]}   (without ids: every disliked answer that has no review yet, oldest first)
    GET  /api/feedback/summary?days=                              -> {totals, by_day, by_mode, by_intent,
         by_issue_type, unreviewed_dislikes}

A review is one structured-output call (purpose ``feedback_review``, app.llm.decisions.review_feedback):
{issue_type: wrong_fact|missing_citation|bad_format|refused_wrongly|incomplete|other, explanation, suggested_fix,
confidence}, stored in ``feedback_reviews`` (one row per message, replaced on re-review).
Sync DB access in plain ``def`` endpoints (FastAPI thread pool), like app.api.main.
"""
from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import db

log = logging.getLogger("newrag.feedback")
router = APIRouter()
MAX_REVIEWS_PER_CALL = 25


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
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _rows(sql: str, params: tuple | list = ()) -> list[dict]:
    try:
        with db.get_pool().connection() as conn:
            cur = conn.execute(sql, params)
            if cur.description is None:
                return []
            cols = [c.name for c in cur.description]
            return [_jsonable(dict(zip(cols, r))) for r in cur.fetchall()]
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ in ("OperationalError", "PoolTimeout", "PoolClosed"):
            raise HTTPException(503, f"Postgres unavailable: {e}") from e
        raise


# ----------------------------------------------------------------------------- rate
class FeedbackIn(BaseModel):
    rating: int = Field(description="1 = like, -1 = dislike, 0 = remove")
    comment: str | None = Field(default=None, max_length=2000)


@router.post("/api/messages/{message_id}/feedback")
def set_feedback(message_id: int, body: FeedbackIn):
    if body.rating not in (-1, 0, 1):
        raise HTTPException(422, "rating must be 1, -1 or 0")
    msg = _rows("SELECT role, content FROM messages WHERE message_id = %s", (message_id,))
    if not msg:
        raise HTTPException(404, "Message not found")
    if msg[0]["role"] != "assistant":
        raise HTTPException(422, "Only assistant answers can be rated")
    if body.rating == 0:
        _rows("DELETE FROM message_feedback WHERE message_id = %s", (message_id,))
        _rows("DELETE FROM feedback_reviews WHERE message_id = %s", (message_id,))
        return {"message_id": message_id, "rating": 0}
    comment = (body.comment or "").strip() or None
    old = _rows("SELECT rating FROM message_feedback WHERE message_id = %s", (message_id,))
    _rows("INSERT INTO message_feedback(message_id, rating, comment) VALUES (%s, %s, %s) "
          "ON CONFLICT (message_id) DO UPDATE SET rating = EXCLUDED.rating, comment = EXCLUDED.comment, "
          "updated_at = now()", (message_id, body.rating, comment))
    if old and old[0]["rating"] != body.rating:      # a review of an earlier dislike no longer applies
        _rows("DELETE FROM feedback_reviews WHERE message_id = %s", (message_id,))
    return {"message_id": message_id, "rating": body.rating}


# ----------------------------------------------------------------------------- list
_LIST_SQL = """
SELECT f.message_id, a.session_id, f.rating, f.comment, u.content AS question, a.content AS answer, a.sources,
       a.images, COALESCE(a.mode, 'swift') AS mode, a.decisions->>'intent' AS intent, a.out_of_context,
       r.review, r.model AS review_model, r.created_at AS reviewed_at, f.created_at, f.updated_at,
       a.created_at AS answered_at, s.title AS session_title
FROM message_feedback f
JOIN messages a ON a.message_id = f.message_id
LEFT JOIN messages u ON u.message_id = a.parent_id AND u.role = 'user'
LEFT JOIN sessions s ON s.session_id = a.session_id
LEFT JOIN feedback_reviews r ON r.message_id = f.message_id
"""


@router.get("/api/feedback")
def list_feedback(rating: int | None = None, limit: int = 200, mode: str | None = None, days: int | None = None,
                  reviewed: bool | None = None):
    limit = max(1, min(int(limit), 2000))
    conds, params = [], []
    if rating is not None:
        if rating not in (-1, 1):
            raise HTTPException(422, "rating must be 1 or -1")
        conds.append("f.rating = %s")
        params.append(rating)
    if mode:
        conds.append("COALESCE(a.mode, 'swift') = %s")
        params.append(mode)
    if days:
        conds.append("f.updated_at >= now() - make_interval(days => %s)")
        params.append(max(1, min(int(days), 3650)))
    if reviewed is not None:
        conds.append("r.message_id IS NOT NULL" if reviewed else "r.message_id IS NULL")
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    rows = _rows(f"{_LIST_SQL} {where} ORDER BY f.updated_at DESC LIMIT %s", (*params, limit))
    for r in rows:
        r["sources"] = r["sources"] or []
        r["images"] = r["images"] or []
    return rows


# ----------------------------------------------------------------------------- review
class ReviewIn(BaseModel):
    message_ids: list[int] | None = None
    limit: int = Field(default=10, ge=1, le=MAX_REVIEWS_PER_CALL)


@router.post("/api/feedback/review")
async def review_dislikes(body: ReviewIn | None = None):
    """LLM review of disliked answers. Without ``message_ids``: the oldest ``limit`` unreviewed dislikes. With ids:
    those (disliked) messages, re-reviewed even if a review exists. Sequential (one model call each)."""
    from fastapi.concurrency import run_in_threadpool

    from app.config import get_settings
    from app.llm.decisions import review_feedback

    body = body or ReviewIn()
    if body.message_ids:
        ids = [int(i) for i in body.message_ids][:MAX_REVIEWS_PER_CALL]
        items = await run_in_threadpool(_rows, f"{_LIST_SQL} WHERE f.rating = -1 AND f.message_id = ANY(%s) "
                                                "ORDER BY f.updated_at", (ids,))
    else:
        items = await run_in_threadpool(_rows, f"{_LIST_SQL} WHERE f.rating = -1 AND r.message_id IS NULL "
                                                "ORDER BY f.updated_at LIMIT %s", (body.limit,))
    model = get_settings().router_model
    out = []
    for it in items:
        rv = await review_feedback(it.get("question") or "", it.get("answer") or "", it.get("sources") or [],
                                   it.get("comment"), session_id=it.get("session_id"), message_id=it["message_id"])
        ok = rv.pop("_ok", False)
        if not ok:
            log.warning("feedback review of %s failed (defaults returned) - not stored", it["message_id"])
            out.append({"message_id": it["message_id"], "review": None, "model": model, "error": "review failed"})
            continue
        saved = await run_in_threadpool(
            _rows, "INSERT INTO feedback_reviews(message_id, review, model) VALUES (%s, %s::jsonb, %s) "
                   "ON CONFLICT (message_id) DO UPDATE SET review = EXCLUDED.review, model = EXCLUDED.model, "
                   "created_at = now() RETURNING message_id, review, model, created_at",
            (it["message_id"], json.dumps(rv), model))
        out.append(saved[0] if saved else {"message_id": it["message_id"], "review": rv, "model": model})
    return {"reviewed": sum(1 for o in out if o.get("review")), "reviews": out}


# ----------------------------------------------------------------------------- summary
_KPI = ("count(*) FILTER (WHERE f.rating = 1)::int AS likes, count(*) FILTER (WHERE f.rating = -1)::int AS dislikes, "
        "round((count(*) FILTER (WHERE f.rating = 1))::numeric / NULLIF(count(*), 0), 4) AS ratio")


@router.get("/api/feedback/summary")
def feedback_summary(days: int = 30):
    days = max(1, min(int(days), 3650))
    base = ("FROM message_feedback f JOIN messages a ON a.message_id = f.message_id "
            "WHERE f.updated_at >= now() - make_interval(days => %s)")
    p = (days,)
    totals = _rows(f"SELECT {_KPI}, count(*)::int AS total {base}", p)[0]
    answers = _rows("SELECT count(*)::int AS n FROM messages WHERE role = 'assistant' AND content <> '' "
                    "AND created_at >= now() - make_interval(days => %s)", p)[0]["n"]
    totals["answers"] = answers
    totals["rated_share"] = round(totals["total"] / answers, 4) if answers else None
    by_day = _rows(f"SELECT to_char(f.updated_at AT TIME ZONE 'UTC', 'YYYY-MM-DD') AS day, {_KPI} {base} "
                   "GROUP BY 1 ORDER BY 1", p)
    by_mode = _rows(f"SELECT COALESCE(a.mode, 'swift') AS mode, {_KPI} {base} GROUP BY 1 ORDER BY 1", p)
    by_intent = _rows(f"SELECT COALESCE(a.decisions->>'intent', 'unknown') AS intent, {_KPI} {base} "
                      "GROUP BY 1 ORDER BY count(*) DESC", p)
    by_issue = _rows("SELECT r.review->>'issue_type' AS issue_type, count(*)::int AS n FROM feedback_reviews r "
                     "JOIN message_feedback f ON f.message_id = r.message_id AND f.rating = -1 "
                     "WHERE f.updated_at >= now() - make_interval(days => %s) GROUP BY 1 ORDER BY 2 DESC", p)
    unreviewed = _rows("SELECT count(*)::int AS n FROM message_feedback f LEFT JOIN feedback_reviews r USING "
                       "(message_id) WHERE f.rating = -1 AND r.message_id IS NULL", ())[0]["n"]
    return {"days": days, "totals": totals, "by_day": by_day, "by_mode": by_mode, "by_intent": by_intent,
            "by_issue_type": by_issue, "unreviewed_dislikes": unreviewed}
