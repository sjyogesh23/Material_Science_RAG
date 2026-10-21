#!/usr/bin/env python3
"""Compact session memory files (sessions/<id>.md) that exceed the token budget.

Usage:
    python scripts/compact_memory.py --all                 # every session over memory_budget_tokens
    python scripts/compact_memory.py --session <id>        # one session (only if over budget)
    python scripts/compact_memory.py --session <id> --force [--budget 300] [--keep 2]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm.ollama_client import get_llm, settings  # noqa: E402
from app.memory.session_memory import SessionMemory  # noqa: E402


async def run(ids: list[str], force: bool, budget: int | None, keep: int | None) -> int:
    s = settings()
    budget = budget or s.memory_budget_tokens
    llm = get_llm()
    changed = 0
    for sid in ids:
        mem = SessionMemory.load(sid)
        before = mem.token_count()
        if not force and before <= budget:
            print(f"{sid}: {before} tokens <= budget {budget}, skipped")
            continue
        did = await mem.compact(llm, budget=budget, keep_turns=keep)
        if did:
            mem.save()
            changed += 1
        print(f"{sid}: {before} -> {mem.token_count()} tokens ({'compacted' if did else 'nothing to fold'}; "
              f"{len(mem.turns)} turns kept, budget {budget})")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session", help="session id")
    g.add_argument("--all", action="store_true", help="all sessions in sessions_dir")
    ap.add_argument("--force", action="store_true", help="compact even if under budget")
    ap.add_argument("--budget", type=int, help="override memory_budget_tokens")
    ap.add_argument("--keep", type=int, help="override memory_keep_turns")
    a = ap.parse_args()
    sdir = Path(settings().sessions_dir)
    ids = [a.session] if a.session else sorted(p.stem for p in sdir.glob("*.md"))
    if a.session and not SessionMemory.path_for(a.session).exists():
        sys.exit(f"no memory file for session {a.session} in {sdir}")
    n = asyncio.run(run(ids, a.force, a.budget, a.keep))
    print(f"done: {n}/{len(ids)} session(s) compacted")


if __name__ == "__main__":
    main()
