"""Per-session markdown memory (``sessions/<session_id>.md``) + LLM compaction.

The file is human-readable and is the single source of conversational state for the LLM:
  * ``render(budget)``      -> injected into the answer system prompt (newest turns first until the budget is hit)
  * ``router_context()``    -> compact view for the router (entities in focus + last turns) to resolve pronouns
  * ``compact(llm)``        -> folds all but the last ``memory_keep_turns`` turns into the rolling summary

Assistant answers are stored without the sources block and without ``[n]`` markers (numbering is per message,
meaningless later), trimmed to ``ANSWER_CHARS``; ``#OutofContext`` is kept verbatim.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.llm import prompts
from app.llm.ollama_client import count_tokens, settings

log = logging.getLogger(__name__)

ANSWER_CHARS = 900
MAX_ENTITIES = 8
DASH = "-"
# readers also accept the em dash (U+2014) that older session files used as the separator
SEP = r"(?:-|\u2014)"
TURN_RE = re.compile(rf"^### Turn (\d+) {SEP} (User|Assistant)\s*$")
SECTION_RE = re.compile(r"^## (Summary|Entities in focus|Recent turns)\s*$")
ENTITY_RE = re.compile(rf"^- (.+?) {SEP} (.*?)(?: \(turn (\d+)\))?\s*$")
CITE_RE = re.compile(r"\s?\[\d+(?:\s*[,–-]\s*\d+)*\]")
PATH_RE = re.compile(r"^<!-- path: ([\d,\s]*) -->\s*$")
SOURCES_BLOCK_RE = re.compile(r"\n+(?:\*\*)?(?:Sources|References)(?:\*\*)?:?\s*\n.*\Z", re.S | re.I)


@dataclass
class Turn:
    n: int
    user: str
    assistant: str


@dataclass
class Entity:
    name: str
    note: str
    turn: int


def clean_answer_for_memory(text: str) -> str:
    t = text.strip()
    if t.startswith(prompts.OOC):
        return prompts.OOC
    t = SOURCES_BLOCK_RE.sub("", t)
    t = CITE_RE.sub("", t).strip()
    t = re.sub(r"\n{3,}", "\n\n", t)
    if len(t) > ANSWER_CHARS:
        t = t[:ANSWER_CHARS].rsplit(" ", 1)[0] + " …"
    return t


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$\\(\[])")


def first_sentences(text: str, n: int) -> str:
    """The first ``n`` sentences of ``text`` (LaTeX-safe enough for memory notes); adds " …" if cut."""
    t = " ".join(text.split())
    parts = _SENT_RE.split(t)
    return t if len(parts) <= n else " ".join(parts[:n]).rstrip() + " …"


def _one_line(text: str, n: int) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 1].rsplit(" ", 1)[0] + "…"


@dataclass
class SessionMemory:
    session_id: str
    path: Path
    title: str = ""
    summary: str = ""
    entities: list[Entity] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    # assistant message ids of EVERY turn this memory covers (summarised ones first), in path order; written as an
    # HTML comment so branch switches can tell which part of the summary is still valid. [] = unknown (v1 file).
    path_ids: list[int] = field(default_factory=list)

    # ------------------------------------------------------------------ io
    @classmethod
    def path_for(cls, session_id: Any, sessions_dir: Path | None = None) -> Path:
        d = Path(sessions_dir or settings().sessions_dir)
        return d / f"{session_id}.md"

    @classmethod
    def load(cls, session_id: Any, sessions_dir: Path | None = None) -> "SessionMemory":
        path = cls.path_for(session_id, sessions_dir)
        mem = cls(str(session_id), path)
        if path.exists():
            mem._parse(path.read_text(encoding="utf-8"))
        return mem

    def _parse(self, md: str) -> None:
        section, cur, buf = None, None, []

        def flush():
            if cur is not None:
                n, role = cur
                body = "\n".join(buf).strip()
                if role == "User":
                    self.turns.append(Turn(n, body, ""))
                elif self.turns and self.turns[-1].n == n:
                    self.turns[-1].assistant = body
                else:
                    self.turns.append(Turn(n, "", body))

        summary_lines: list[str] = []
        for line in md.splitlines():
            if m := re.match(r"^_Title: (.*)_\s*$", line):
                self.title = m.group(1).strip()
                continue
            if section is None and (m := PATH_RE.match(line)):
                self.path_ids = [int(x) for x in re.findall(r"\d+", m.group(1))]
                continue
            if m := SECTION_RE.match(line):
                flush()
                cur, buf, section = None, [], m.group(1)
                continue
            if section == "Summary":
                summary_lines.append(line)
            elif section == "Entities in focus":
                if m := ENTITY_RE.match(line):
                    self.entities.append(Entity(m.group(1), m.group(2), int(m.group(3) or 0)))
            elif section == "Recent turns":
                if m := TURN_RE.match(line):
                    flush()
                    cur, buf = (int(m.group(1)), m.group(2)), []
                elif cur is not None:
                    buf.append(line)
        flush()
        self.summary = "\n".join(summary_lines).strip()

    def to_markdown(self, turns: list[Turn] | None = None, entities: bool = True) -> str:
        out = [f"# Session {self.session_id}", f"_Title: {self.title}_"]
        if self.path_ids:
            out.append(f"<!-- path: {','.join(str(i) for i in self.path_ids)} -->")
        out += ["## Summary", self.summary, "## Entities in focus"]
        if entities:
            out += [f"- {e.name} {DASH} {e.note} (turn {e.turn})" for e in self.entities]
        out.append("## Recent turns")
        for t in self.turns if turns is None else turns:
            out += [f"### Turn {t.n} {DASH} User", t.user, f"### Turn {t.n} {DASH} Assistant", t.assistant]
        return "\n".join(out).strip() + "\n"

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".md.tmp")
        tmp.write_text(self.to_markdown(), encoding="utf-8")
        tmp.replace(self.path)

    # ------------------------------------------------------------------ updates
    @property
    def next_turn(self) -> int:
        return (self.turns[-1].n + 1) if self.turns else 1

    def add_turn(self, user: str, assistant: str, entities: list[str], message_id: int | None = None) -> None:
        n = self.next_turn
        if self.path_ids or message_id:
            if not self.path_ids and (self.turns or self.summary):
                pass                            # v1 file: earlier ids unknown -> keep the path unknown
            else:
                self.path_ids.append(int(message_id or 0))
        self.turns.append(Turn(n, user.strip(), clean_answer_for_memory(assistant)))
        self._add_entities(user, entities, n)

    def _add_entities(self, user: str, entities: list[str], n: int) -> None:
        note = _one_line(user, 70)
        for name in reversed([e.strip() for e in entities or [] if e and e.strip()]):
            self.entities = [e for e in self.entities if e.name.lower() != name.lower()]
            self.entities.insert(0, Entity(_one_line(name, 60), note, n))
        self.entities = self.entities[:MAX_ENTITIES]

    def rebuild_from_path(self, turns: list[Any]) -> bool:
        """Make this memory describe a (possibly different) branch. ``turns`` = the path's completed turns in
        order, each a dict / object with ``user``, ``assistant``, ``message_id`` (assistant id) and optional
        ``entities``. The summary is kept only if every turn it folded in is on the common prefix of the old and
        the new path; Recent turns (and the entities of turns after the common prefix) are rebuilt from the path.
        Returns True if the result is over budget - the caller should then ``await compact(llm)``."""
        def get(t: Any, k: str, default: Any = None) -> Any:
            return t.get(k, default) if isinstance(t, dict) else getattr(t, k, default)

        new_ids = [int(get(t, "message_id") or 0) for t in turns]
        n_recent = len(self.turns)
        if self.path_ids and len(self.path_ids) >= n_recent:          # ids known: exact common prefix
            summarised = len(self.path_ids) - n_recent
            common = 0
            for a, b in zip(self.path_ids, new_ids):
                if not a or a != b:
                    break
                common += 1
        else:                                                         # v1 file: compare the visible turns' text
            summarised = (self.turns[0].n - 1) if self.turns else (10 ** 6 if self.summary else 0)
            common = summarised if len(turns) >= summarised else 0
            for i, old in enumerate(self.turns):
                j = summarised + i
                if common != j or j >= len(turns) or " ".join(str(get(turns[j], "user", "")).split()) != " ".join(old.user.split()):
                    break
                common += 1
        keep_summary = bool(self.summary) and summarised <= common
        start = summarised if keep_summary else 0
        if not keep_summary:
            self.summary = ""
        self.entities = [e for e in self.entities if 0 < e.turn <= common]
        self.turns = []
        for i, t in enumerate(turns):
            if i < start:
                continue
            n = i + 1
            user = str(get(t, "user", "") or "").strip()
            self.turns.append(Turn(n, user, clean_answer_for_memory(str(get(t, "assistant", "") or ""))))
            if i >= common:
                self._add_entities(user, list(get(t, "entities", None) or []), n)
        self.path_ids = new_ids if all(new_ids) else []
        return self.needs_compaction()

    # ------------------------------------------------------------------ views
    def token_count(self) -> int:
        return count_tokens(self.to_markdown())

    def needs_compaction(self, budget: int | None = None) -> bool:
        return self.token_count() > (budget or settings().memory_budget_tokens)

    def render(self, budget_tokens: int) -> str:
        """Header + summary + entities + as many of the newest turns as fit (kept in chronological order)."""
        base = count_tokens(self.to_markdown(turns=[]))
        if base > budget_tokens:  # degenerate: drop entities, then cut the summary
            md = self.to_markdown(turns=[], entities=False)
            return md[: max(0, budget_tokens) * 3]
        chosen: list[Turn] = []
        used = base
        for t in reversed(self.turns):
            cost = count_tokens(f"### Turn {t.n} {DASH} User\n{t.user}\n### Turn {t.n} {DASH} Assistant\n{t.assistant}\n")
            if used + cost > budget_tokens:
                break
            chosen.insert(0, t)
            used += cost
        return self.to_markdown(turns=chosen)

    def router_context(self, max_turns: int = 2, answer_chars: int = 280) -> str:
        if not self.turns and not self.summary:
            return ""
        lines = []
        if self.summary:
            lines.append("Summary: " + _one_line(self.summary, 400))
        if self.entities:
            lines.append("Entities in focus (newest first):")
            lines += [f"- {e.name} {DASH} {e.note} (turn {e.turn})" for e in self.entities[:5]]
        lines.append("Recent turns:")
        for t in self.turns[-max_turns:]:
            lines.append(f"Turn {t.n} user: {t.user}")
            lines.append(f"Turn {t.n} assistant: {_one_line(t.assistant, answer_chars)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ compaction
    async def compact(self, llm, *, budget: int | None = None, keep_turns: int | None = None) -> bool:
        """Summarise all but the last ``keep_turns`` turns (fewer if needed to fit ``budget``). Returns True if
        anything was folded into the summary."""
        s = settings()
        budget = budget or s.memory_budget_tokens
        keep = min(keep_turns or s.memory_keep_turns, len(self.turns))
        summary_allowance = max(40, min(250, budget // 5))
        while keep > 1 and count_tokens(self.to_markdown(turns=self.turns[-keep:])) + summary_allowance > budget:
            keep -= 1
        old, kept = self.turns[: len(self.turns) - keep], self.turns[len(self.turns) - keep:]
        if not old:
            return self.trim_to_budget(budget)
        max_words = max(30, int(summary_allowance * 0.7))
        turns_txt = "\n".join(f"Turn {t.n} user: {t.user}\nTurn {t.n} assistant: {_one_line(t.assistant, 500)}"
                              for t in old)
        try:
            text, _ = await llm.complete(
                [{"role": "system", "content": prompts.COMPACTION_SYSTEM.format(max_words=max_words)},
                 {"role": "user", "content": prompts.COMPACTION_USER.format(summary=self.summary or "(none)",
                                                                            turns=turns_txt)}],
                purpose="memory_compaction", options={"temperature": 0, "num_predict": int(max_words * 2)},
                session_id=self.session_id)
            new = text.strip()
            if not new:
                raise ValueError("empty summary")
        except Exception as e:  # noqa: BLE001 - heuristic fallback keeps the memory bounded anyway
            log.warning("memory compaction LLM failed (%s); using heuristic summary", e)
            new = " ".join(filter(None, [self.summary] + [
                f"Turn {t.n}: asked \"{_one_line(t.user, 80)}\"" +
                (" (out of context)." if t.assistant == prompts.OOC else ".") for t in old]))
        words = new.split()
        self.summary = " ".join(words[: max_words + 20])
        self.turns = kept
        self.trim_to_budget(budget)
        return True

    def trim_to_budget(self, budget: int) -> bool:
        """Still over ``budget`` after summarising? Shorten the kept turns (oldest first): assistant answers to
        their first sentences (3 -> 2 -> 1), then to ~200 / ~80 characters, then long user messages. Returns True
        if anything was shortened."""
        if self.token_count() <= budget:
            return False
        changed = False

        def shorten(fn) -> bool:
            nonlocal changed
            for t in self.turns:
                if self.token_count() <= budget:
                    return True
                if t.assistant == prompts.OOC:
                    continue
                new = fn(t)
                if new != t.assistant:
                    t.assistant, changed = new, True
            return self.token_count() <= budget

        for n in (3, 2, 1):
            if shorten(lambda t, n=n: first_sentences(t.assistant, n)):
                return changed
        for chars in (200, 80):
            if shorten(lambda t, c=chars: _one_line(t.assistant, c)):
                return changed
        for t in self.turns:
            if self.token_count() <= budget:
                return changed
            if len(t.user) > 160:
                t.user, changed = _one_line(t.user, 160), True
        return changed   # best effort: the summary (already bounded) and the entities in focus are kept
