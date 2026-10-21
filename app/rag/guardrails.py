"""Input / output guardrails around the answer model (pure functions; the pipeline wires them in).

Input side - turn the question's FORM into explicit instructions (``form_notes``):
  analogy, presupposition (possible false premise), ambiguous, injection_wrapped, citation demands (page N,
  "cite 10 sources", "quote exactly"), references to figures / tables that do not exist.
  The form of a question never decides its scope: scope is decided by retrieval of its topic + the router.

Output side:
  * ``should_retry_refusal``: the model said #OutofContext although retrieval of the topic is strong and the
    router kept it in scope -> regenerate once with prompts.RETRY_NOTE (both signals are needed to refuse).
  * ``strip_prompt_leak``: drop lines that reproduce the system prompt (prompt-leak attacks).
  * ``strip_injected_tokens``: drop words at the end of the answer that only occur in the removed injection
    (e.g. "end your answer with the word X").
"""
from __future__ import annotations

import json
import re
from typing import Any

from app.llm import prompts
from app.llm.decisions import (CITE_COUNT_RE, FIG_REF_RE, PAGE_RE, QUOTE_RE, SCIENCE_RE, RouteDecision)

_WORD_RE = re.compile(r"[a-z0-9]+")
NUMBER_RE = re.compile(r"\d")


def _g(hit: Any, name: str, default: Any = None) -> Any:
    return getattr(hit, name, default)


# ----------------------------------------------------------------------------- input: references
def missing_refs(question: str, retriever: Any) -> list[str]:
    """Figure / table numbers named in the question that do not exist in the corpus ("Figure 12-7")."""
    if not hasattr(retriever, "figure_by_number"):
        return []
    out = []
    for m in FIG_REF_RE.finditer(question):
        if m.group(1).lower().startswith("eq"):
            continue
        label = m.group(0).strip()
        try:
            found = retriever.figure_by_number(label)
        except Exception:  # noqa: BLE001 - DB hiccup: assume it exists
            found = [True]
        if not found:
            out.append(label)
    return out


def only_missing_ref(question: str, missing: list[str]) -> bool:
    """True if the question is only about a nonexistent figure/table (nothing else to answer) -> out of context."""
    if not missing:
        return False
    rest = question
    for label in missing:
        rest = rest.replace(label, " ")
    return not SCIENCE_RE.search(rest)


# ----------------------------------------------------------------------------- input: form notes
def form_notes(decision: RouteDecision, question: str, n_sources: int, source_pages: set[int],
               total_pages: int | None = None, missing: list[str] | None = None) -> list[str]:
    """Explicit per-question instructions for the answer prompt (see prompts.FORM_NOTES)."""
    forms = list(decision.question_forms or [])
    notes = [prompts.FORM_NOTES[f] for f in forms if f in prompts.FORM_NOTES]
    if decision.intent == "comparison":
        notes.append(prompts.COMPARISON_NOTE)
    if decision.intent == "calculation" and NUMBER_RE.search(FIG_REF_RE.sub(" ", question)):
        notes.append(prompts.CALCULATION_NOTE)     # numbers to compute with (not "how many atoms per cell")
    if "citation_demand" in forms:
        for page in sorted({int(p) for p in PAGE_RE.findall(question)}):
            if total_pages and page > total_pages:
                notes.append(prompts.PAGE_NOTE_MISSING.format(total=total_pages, page=page))
            elif page not in source_pages:
                notes.append(prompts.PAGE_NOTE_ABSENT.format(page=page))
        m = CITE_COUNT_RE.search(question)
        if m:
            k = int(m.group(1) or m.group(2))
            if k > max(1, n_sources // 2):
                notes.append(prompts.CITE_COUNT_NOTE.format(k=k))
        if QUOTE_RE.search(question):
            notes.append(prompts.QUOTE_NOTE)
    for label in missing or []:
        notes.append(prompts.MISSING_REF_NOTE.format(label=label))
    return notes


# ----------------------------------------------------------------------------- output
def should_retry_refusal(decision: RouteDecision, top_score: float, min_score: float, missing: list[str] | None = None,
                         question: str = "") -> bool:
    """The model refused, but the topic retrieved strongly and the router kept the question in scope: one retry.
    Refusing then needs BOTH signals (weak retrieval or router out-of-scope) or a second refusal."""
    if not decision.in_scope or "jailbreak" in decision.rules or decision.intent == "greeting":
        return False
    if missing and only_missing_ref(question, missing):
        return False
    return top_score >= min_score


def _shingles(text: str, n: int) -> set[tuple[str, ...]]:
    w = _WORD_RE.findall(text.lower())
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


_shingle_cache: dict[str, Any] = {"key": None, "shingles": set()}


def _system_shingles() -> set[tuple[str, ...]]:
    """8-word shingles of the system prompt texts, recomputed when a prompt component version changes."""
    try:
        from app import versioning
        key = json.dumps(versioning.active_versions(), sort_keys=True)
    except Exception:  # noqa: BLE001 - versioning unavailable -> code defaults
        key = "default"
    if _shingle_cache["key"] != key:
        _shingle_cache["shingles"] = _shingles(
            prompts.ANSWER_SYSTEM + " " + " ".join(prompts.FORMAT_RULES.values()) + " "
            + " ".join(prompts.FORM_NOTES.values()) + " " + prompts.RETRY_NOTE, 8)
        _shingle_cache["key"] = key
    return _shingle_cache["shingles"]


def strip_prompt_leak(answer: str) -> tuple[str, bool]:
    """Remove lines (paragraphs) that reproduce >= 8 consecutive words of the system prompt."""
    lines = answer.split("\n")
    sys_sh = _system_shingles()
    kept = [ln for ln in lines if not (_shingles(ln, 8) & sys_sh)]
    leaked = len(kept) != len(lines)
    return ("\n".join(kept).strip() if leaked else answer), leaked


def strip_injected_tokens(answer: str, removed: str, source_text: str) -> str:
    """Drop trailing words of the answer that appear in the removed injection but in no source (a canary such
    as 'end your answer with the word X')."""
    if not removed:
        return answer
    inj = set(_WORD_RE.findall(removed.lower())) - set(_WORD_RE.findall(source_text.lower()))
    inj = {w for w in inj if len(w) >= 3}
    if not inj:
        return answer
    out = answer.rstrip()
    while True:
        m = re.search(r"([A-Za-z0-9]+)[\s.!*_\"'`]*$", out)
        if not m or m.group(1).lower() not in inj:
            break
        out = out[: m.start()].rstrip()
    return out if out != answer.rstrip() else answer
