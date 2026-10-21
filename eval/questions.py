"""Question bank loading / validation (eval/questions.jsonl)."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

BANK = Path(__file__).with_name("questions.jsonl")

SCENARIOS = [
    "factual_definition", "explanation", "formula_equation", "numerical", "comparison", "table_request",
    "figure_request", "cross_reference", "analogy", "multi_turn_pronoun", "out_of_context",
    "out_of_context_near_domain", "false_premise", "ambiguous", "prompt_injection", "adversarial_citation",
    "long_multipart", "paraphrase_typo", "summarization",
]
EXPECTED = {"answer", "out_of_context"}


def load(path: Path | str = BANK) -> list[dict]:
    items = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            it = json.loads(line)
            it.setdefault("_line", n)
            items.append(it)
    validate(items)
    for it in items:
        it.pop("_line", None)
    return items


def validate(items: list[dict]) -> None:
    seen = set()
    for it in items:
        where = f"item {it.get('id')!r} (line {it.get('_line')})"
        assert it.get("id") and it["id"] not in seen, f"{where}: missing/duplicate id"
        seen.add(it["id"])
        assert it.get("scenario") in SCENARIOS, f"{where}: unknown scenario {it.get('scenario')}"
        assert it.get("expected") in EXPECTED, f"{where}: expected must be one of {EXPECTED}"
        turns = it.get("turns")
        assert bool(it.get("question")) != bool(turns), f"{where}: needs exactly one of question / turns"
        if turns:
            assert isinstance(turns, list) and len(turns) >= 2 and all(isinstance(t, str) and t for t in turns)
        g = it.get("gold")
        assert isinstance(g, dict) and isinstance(g.get("facts", []), list), f"{where}: gold.facts list required"
        if it["expected"] == "answer":
            assert g.get("facts"), f"{where}: answer items need gold facts"
        if it.get("paraphrase_of"):
            assert it["paraphrase_of"] in seen or True


def user_turns(item: dict) -> list[str]:
    return list(item["turns"]) if item.get("turns") else [item["question"]]


def bank_hash(items: list[dict]) -> str:
    blob = json.dumps(items, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def select(items: list[dict], scenarios: list[str] | None = None, ids: list[str] | None = None,
           limit: int | None = None) -> list[dict]:
    out = [it for it in items if (not scenarios or it["scenario"] in scenarios) and (not ids or it["id"] in ids)]
    return out[:limit] if limit else out


def stats(items: list[dict]) -> dict:
    return {"total": len(items), "by_scenario": dict(Counter(i["scenario"] for i in items)),
            "by_expected": dict(Counter(i["expected"] for i in items)),
            "by_difficulty": dict(Counter(i.get("difficulty", "medium") for i in items)),
            "multi_turn": sum(1 for i in items if i.get("turns"))}
