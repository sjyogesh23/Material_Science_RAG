"""Local "Jev-style" decision layer: typed questions -> ONE structured-output Ollama call -> typed answers.

Jev (TypeSafe AI) is cloud-only, so its model is reproduced locally:
  * questions are typed (Choice / Noul / Score, plus Text / TextList for free-form slots such as a rewritten
    query); a pydantic model is built from them and its JSON schema is passed as Ollama ``format`` so the
    grammar forces valid, enum-constrained output; temperature 0
  * confidence = probability of the value's first generated token (Ollama logprobs), i.e. how sure the model
    was when it committed to the answer - not a self-reported number
  * validated with pydantic, one retry with the validation error, then safe defaults (confidence 0)

``route()`` uses it to classify a chat message (scope, intent, format, images, cross-reference, standalone
query) and then applies cheap keyword rules that override the model where wording is unambiguous.
"""
from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, ValidationError, create_model

from app.llm import prompts
from app.llm.ollama_client import LLM, get_llm, settings

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- typed questions
class Choice(BaseModel):
    name: str
    question: str
    options: list[str]
    default: str | None = None


class Noul(BaseModel):  # yes / no -> bool
    name: str
    question: str
    default: bool = False


class Score(BaseModel):  # ordinal levels, lowest first
    name: str
    question: str
    levels: list[str]
    default: str | None = None


class Text(BaseModel):  # free-form string slot (extension of the Jev types)
    name: str
    question: str
    default: str = ""
    max_chars: int = 300


class TextList(BaseModel):  # short list of strings (extension of the Jev types)
    name: str
    question: str
    max_items: int = 3
    default: list[str] = Field(default_factory=list)


Question = Union[Choice, Noul, Score, Text, TextList]


def _field(q: Question) -> tuple[Any, Any]:
    if isinstance(q, Choice):
        return Literal[tuple(q.options)], ...
    if isinstance(q, Score):
        return Literal[tuple(q.levels)], ...
    if isinstance(q, Noul):
        return bool, ...
    if isinstance(q, Text):
        return str, Field(..., max_length=q.max_chars)
    return list[str], Field(..., max_length=q.max_items)


def build_model(questions: list[Question]) -> type[BaseModel]:
    return create_model("Decision", **{q.name: _field(q) for q in questions})


def default_value(q: Question) -> Any:
    if isinstance(q, Choice):
        return q.default if q.default in q.options else q.options[0]
    if isinstance(q, Score):
        return q.default if q.default in q.levels else q.levels[0]
    return q.default if not isinstance(q, TextList) else list(q.default)


def _describe(q: Question) -> str:
    if isinstance(q, Choice):
        kind = "one of " + " | ".join(q.options)
    elif isinstance(q, Score):
        kind = "level, lowest to highest: " + " < ".join(q.levels)
    elif isinstance(q, Noul):
        kind = "true/false"
    elif isinstance(q, Text):
        kind = "string"
    else:
        kind = f"list of at most {q.max_items} short strings"
    return f'- "{q.name}" ({kind}): {q.question}'


def _confidences(text: str, logprobs: list | None, questions: list[Question]) -> dict[str, float]:
    """Per-field confidence from token logprobs: first value token for categorical answers, geometric mean
    over the value's tokens for free text."""
    if not logprobs:
        return {}
    toks = [lp.token for lp in logprobs]
    starts, pos = [], 0
    for t in toks:
        starts.append(pos)
        pos += len(t)
    joined = "".join(toks)
    out: dict[str, float] = {}
    dec = json.JSONDecoder()
    for q in questions:
        m = re.search(rf'"{re.escape(q.name)}"\s*:\s*', joined)
        if not m:
            continue
        v0 = m.end()
        try:
            _, v1 = dec.raw_decode(joined, v0)
        except ValueError:
            continue
        if isinstance(q, (Choice, Score)) and joined[v0:v0 + 1] == '"':
            v0 += 1
        idx = [i for i, s in enumerate(starts) if s < v1 and s + len(toks[i]) > v0]
        if not idx:
            continue
        if isinstance(q, (Text, TextList)):
            out[q.name] = math.exp(sum(logprobs[i].logprob for i in idx) / len(idx))
        else:
            out[q.name] = math.exp(logprobs[idx[0]].logprob)
    return {k: round(v, 4) for k, v in out.items()}


def _lenient(data: Any, questions: list[Question]) -> Any:
    """Trim over-long lists / strings instead of failing (the grammar normally prevents them anyway)."""
    if isinstance(data, dict):
        for q in questions:
            v = data.get(q.name)
            if isinstance(q, TextList) and isinstance(v, list):
                data[q.name] = [str(x) for x in v if str(x).strip()][: q.max_items]
            elif isinstance(q, Text) and isinstance(v, str):
                data[q.name] = v[: q.max_chars]
    return data


async def decide(state: str, questions: list[Question], *, purpose: str = "router", system: str | None = None,
                 llm: LLM | None = None, **ids) -> dict[str, Any]:
    """-> {name: value, f"{name}_confidence": 0..1}. Never raises: falls back to defaults (confidence 0)."""
    llm = llm or get_llm()
    model_cls = build_model(questions)
    schema = model_cls.model_json_schema()
    user = ("STATE:\n" + state.strip() + "\n\nQUESTIONS (answer each key):\n" +
            "\n".join(_describe(q) for q in questions) +
            "\n\nReturn one compact single-line JSON object with exactly these keys.")
    messages = [{"role": "system", "content": system or "You answer typed questions as JSON. Be literal and exact."},
                {"role": "user", "content": user}]
    s = settings()
    for attempt in range(2):
        text = ""
        try:
            text, _ = await llm.complete(messages, purpose=purpose, model=getattr(s, "router_model", None),
                                         format=schema, options={"temperature": 0, "num_predict": 300},
                                         logprobs=True, **ids)
            parsed = model_cls.model_validate(_lenient(json.loads(text), questions))
        except (ValidationError, ValueError) as e:
            log.warning("decision attempt %d invalid: %s", attempt + 1, str(e)[:200])
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"Invalid: {str(e)[:300]}. Return valid JSON only."}]
            continue
        except Exception as e:  # noqa: BLE001 - Ollama down etc.
            log.warning("decision call failed: %s", e)
            break
        conf = _confidences(text, llm.last_logprobs, questions)
        out: dict[str, Any] = {}
        for q in questions:
            out[q.name] = getattr(parsed, q.name)
            out[f"{q.name}_confidence"] = conf.get(q.name, 0.5)
        out["_ok"] = True
        return out
    out = {}
    for q in questions:
        out[q.name] = default_value(q)
        out[f"{q.name}_confidence"] = 0.0
    out["_ok"] = False
    return out


# ----------------------------------------------------------------------------- routing
Intent = Literal["greeting", "definition", "explanation", "comparison", "calculation", "cross_reference",
                 "figure_request", "table_request", "summary", "other"]
Fmt = Literal["prose", "table", "steps", "equation"]
INTENTS = list(Intent.__args__)
FORMATS = list(Fmt.__args__)


class RouteDecision(BaseModel):
    in_scope: bool = True
    in_scope_confidence: float = 0.0
    intent: Intent = "other"
    needs_images: bool = False
    answer_format: Fmt = "prose"
    is_cross_reference: bool = False
    standalone_query: str = ""
    sub_queries: list[str] = Field(default_factory=list)
    # extensions (not in the v1 contract, harmless for consumers)
    entities: list[str] = Field(default_factory=list)   # focus terms -> session memory
    # unusual question FORMS (independent of scope): analogy | presupposition | ambiguous | injection_wrapped |
    # citation_demand. They turn into explicit answer instructions and never make a question out of scope.
    question_forms: list[str] = Field(default_factory=list)
    confidence: dict[str, float] = Field(default_factory=dict)
    rules: list[str] = Field(default_factory=list)       # keyword overrides that fired
    llm_ok: bool = True
    # v2: answer mode (swift | deep) and the grounding check result {score 1-5, unsupported, revised}
    mode: str | None = None
    grounding: dict[str, Any] = Field(default_factory=dict)


GREETING_RE = re.compile(
    r"^\s*(hi+|hello+|hey+|hiya|howdy|greetings|yo|good\s+(morning|afternoon|evening|day)|thanks?|thank\s+you"
    r"|thx|ty|cheers|ok(ay)?|cool|great|nice|bye|goodbye|see\s+you|who\s+are\s+you|what\s+can\s+you\s+do"
    r"|how\s+are\s+you)\b[\s,!.?]*(there|bot|assistant|so\s+much|very\s+much|a\s+lot|again)?[\s,!.?:)]*$", re.I)
THANKS_RE = re.compile(r"^\s*(thanks?|thank\s+you|thx|ty|cheers)\b", re.I)
TABLE_RE = re.compile(r"\b(tabulate|tabular|in\s+a\s+table|as\s+a\s+table|table\s+(comparing|of|with|that)"
                      r"|make\s+a\s+table|side[-\s]by[-\s]side)\b|\btable\b(?!\s*\d)", re.I)
# the user wants to SEE something: a visual noun, or "show me / draw / visualise". Bare "curve" / "show" are not
# enough ("describe the regions of a stress-strain curve" is a text question) - images are shown only on request.
IMAGE_RE = re.compile(r"\b(figures?|fig\.|diagrams?|images?|pictures?|plots?|graphs?|sketch\w*|illustration\w*"
                      r"|schematics?|draw\w*|visuali[sz]\w*|show\s+(me|us)|(can|could)\s+you\s+show"
                      r"|^\s*show\s+(the|a|an))\b", re.I)
EQUATION_RE = re.compile(r"\b(formula|equation|expression\s+for|mathematical(ly)?|derive)\b", re.I)
COMPARE_RE = re.compile(r"\b(compare|comparison|versus|vs\.?|differ(ence|ences|ent)?\s+(between|from))\b", re.I)
CROSSREF_RE = re.compile(
    r"\b(relat(e|es|ed|ion|ionship)\s+(to|with|between)|connect(ion|ed)?\s+(to|with|between)|link(ed)?\s+to"
    r"|which\s+(section|chapter|part)|what\s+(section|chapter)|where\s+in\s+the\s+(book|text|chapter)"
    r"|from\s+the\s+[\w\s-]{2,30}\s+chapter|in\s+the\s+[\w\s-]{2,30}\s+chapter|(section|chapter)\s+\d"
    r"|(figure|fig\.|table|equation|eq\.)\s*\d+[-.]\d+.*\b(section|chapter|explain\w*)\b)", re.I)
ANAPHORA_RE = re.compile(
    r"\b(it|its|it's|itself|this|that|these|those|they|them|their|previous|former|latter|above|same|"
    r"aforementioned|earlier|again|one)\b", re.I)
REF_NUM_RE = re.compile(r"\b(fig(ure)?|table|eq(uation)?|section|chapter)\.?\s*\d", re.I)
# a figure / table / equation NUMBER ("Figure 2-7", "Table 3.1", "Eq. 7-4") -> its explaining text is added
FIG_REF_RE = re.compile(r"\b(fig(?:ure)?|table|eq(?:uation)?)\.?\s*(\d+(?:\s*[-.\u2013]\s*\d+)?)", re.I)
# prompt-injection / jailbreak phrasing (role, rules or style changes)
JAILBREAK_RE = re.compile(
    r"\b(ignore|disregard|forget|override|bypass)\b[^.?!]{0,40}?\b(instructions?|rules?|prompts?|guidelines?"
    r"|restrictions?|constraints?|everything|all|above|previous|system)\b[^.?!:]*"
    r"|\b((you\s+are\s+now|pretend\s+(to\s+be|you\s+are|you're)|act\s+as)(\s+an?)?(\s+\w+)?|from\s+now\s+on"
    r"|role-?\s?play|jailbreak|\bDAN\b"
    r"|developer\s+mode|(reveal|print|show|repeat)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions)"
    r"|(answer|respond|reply|talk|speak|write)\s+(only\s+)?(as|like)\s+an?\s+\w+|in\s+the\s+style\s+of\s+\w+"
    r"|tell\s+me\s+a\s+joke|write\s+(me\s+)?a\s+(poem|song|story))", re.I)
# syllabus vocabulary: a jailbreak that still contains one of these is answered normally (minus the injection)
SCIENCE_RE = re.compile(
    r"\b(materials?|crystal\w*|lattice|unit\s+cell|atoms?|atomic|bond\w*|fcc|bcc|hcp|cubic|hexagonal|miller"
    r"|bragg|diffract\w*|x-?rays?|defects?|vacanc\w+|interstitial\w*|dislocations?|grains?|diffusion|fick"
    r"|stress\w*|strain\w*|elastic\w*|plastic\w*|yield|tensile|hardness|ductil\w+|brittle|toughness|fracture"
    r"|fatigue|creep|modulus|phases?|eutectic|eutectoid|peritectic|alloys?|steels?|iron|carbon|cementite|ferrite"
    r"|austenite|martensite|pearlite|bainite|heat\s+treat\w*|quench\w*|anneal\w*|temper\w*|metals?|metallic"
    r"|ceramics?|polymers?|composites?|semiconduct\w+|conduct\w+|dielectric|magnet\w*|thermal|optical|band\s+gap"
    r"|electrons?|density|packing|coordination|slip|recrystalli\w+|solid\s+solution|precipitat\w+|corrosion"
    r"|glass|amorphous|ionic|covalent|van\s+der\s+waals|young'?s|poisson|hooke|hall-?petch|burgers|schmid)\b",
    re.I)
RELATE_RE = re.compile(r"\b(relat\w*|connect\w*|link\w*|between|compare\w*|versus|vs\.?)\b", re.I)
DEFINE_RE = re.compile(r"^\s*(what\s+(is|are)\b|define\b|definition\b|what\s+does\s+\S+\s+mean|meaning\s+of\b)", re.I)
CALC_RE = re.compile(r"\b(calculate|compute|determine\s+the|how\s+(much|many)|estimate|solve|work\s+out)\b", re.I)
SUMMARY_RE = re.compile(r"\b(summari[sz]e|summary|overview|key\s+points|outline)\b", re.I)
ENTITY_LINE_RE = re.compile(r"^- (.+?) (?:-|\u2014) .*\(turn \d+\)[ \t]*$", re.M)

# ---- question forms (how a question is asked, not what it is about)
ANALOGY_RE = re.compile(
    r"\b(analog(y|ies)|metaphor|everyday\s+(example|terms|life|objects?|things?)|in\s+(simple|plain|layman'?s?)\s+terms"
    r"|like\s+i'?m\s+(five|5|a\s+child|a\s+kid)|eli5|real[-\s]life\s+example)\b", re.I)
# a presupposition the answer must check against the sources ("Since X, why ...", "why does X ...")
PRESUPPOSITION_RE = re.compile(
    r"^\s*(since|given\s+that|because|as\s+we\s+know)\b"
    r"|\bwhy\s+(is|are|does|do|did|can|can'?t|cannot|doesn'?t|don'?t|isn'?t|aren'?t|would|will|must|has|have)\b"
    r"|\bhow\s+come\b", re.I)
# a short question whose subject is a bare symbol or a vague "the X" (several meanings possible)
AMBIGUOUS_RE = re.compile(
    r"^\s*(what\s+(is|are|does)|what'?s|explain|define|describe|tell\s+me\s+about)\s+"
    r"(the\s+[\w.\-]+(\s+(thing|one|stuff))?|[A-Za-z]{1,2}\d?|that\s+\w+)\s*(mean)?\s*[?.!]*\s*$", re.I)
PAGE_RE = re.compile(r"\b(?:on\s+)?pages?\s+(\d{1,4})\b", re.I)
CITE_COUNT_RE = re.compile(r"\b(?:cite|use|give|list)\s+(?:at\s+least\s+|exactly\s+|minimum\s+)?(\d{1,3})\s+"
                           r"(?:different\s+|separate\s+|distinct\s+)?(?:sources|citations|references)\b"
                           r"|\bat\s+least\s+(\d{1,3})\s+(?:different\s+)?(?:sources|citations|references)\b", re.I)
QUOTE_RE = re.compile(r"\bquote\s+(exactly|verbatim|word\s+for\s+word)\b|\bverbatim\b", re.I)
# whole-sentence injections: fake sources, "pretend the textbook says", output-control and prompt-leak requests.
# (JAILBREAK_RE above removes role/style clauses and keeps the rest of the sentence; these drop the sentence.)
INJECT_SENTENCE_RE = re.compile(
    r"</?\s*(sources?|system|context|instructions?|documents?)\s*>"
    r"|\bnew\s+(source|context|document|passage)\b|\bsource\s*\[\d+\]\s*:"
    r"|\b(pretend|imagine|assume|suppose)\s+(that\s+)?(the\s+)?(textbook|book|sources?|text)\s+(says?|states?|claims?|said)\b"
    r"|\b(end|finish|start|begin|conclude|sign)\s+(off\s+)?(your|the|each)\s+(answer|response|reply|output|message)\s+with\b"
    r"|\b(do\s+not|don'?t|never|stop)\s+(cite|citing|use\s+citations|include\s+citations)\b"
    r"|\b(please\s+)?cite\s+\[\d+\]"
    r"|\bsystem\s+(override|prompt|message)\b"
    r"|\b(reveal|print|show|repeat|output|dump)\b[^.?!]{0,30}\b(system\s+prompt|your\s+(instructions|rules|prompt))\b",
    re.I)
SENT_SPLIT_RE = re.compile(r"(?<=[.?!])\s+|\n+")
ANALOGY_STRIP_RE = re.compile(
    r"^\s*(please\s+)?use\s+an?\s+((simple|everyday|familiar|real[-\s]life)\s+)*(analogy|metaphor)\s+to\s+"
    r"|\s*,?\s*\b(using|with|via|through|by\s+means\s+of)\s+(an?\s+)?((simple|everyday|familiar|real[-\s]life)\s+)*"
    r"(analog(y|ies)|metaphor)\b", re.I)


_LIST_RE = re.compile(r"\(([^()]{3,200})\)|\b(?:comparing|compare|between|of|for)\s+([^:;?.()]{3,200}?)(?=\s*(?::|;|\?|\.|$|\bwith\b|\bshowing\b|\bin\s+terms\b))",
                      re.I)


def enumerated_items(question: str, max_items: int = 4) -> list[str]:
    """Items the user lists explicitly ("(eutectic, eutectoid, peritectic)", "comparing simple cubic, BCC and
    FCC:") -> one coverage sub-query each, so every table row gets its own evidence. [] if fewer than 3 items."""
    for m in _LIST_RE.finditer(question):
        body = m.group(1) or m.group(2) or ""
        parts = [p.strip(" .") for p in re.split(r",\s*(?:and\s+|or\s+)?|\s+(?:and|or|vs\.?|versus)\s+", body)]
        parts = [re.sub(r"^(the|a|an)\s+", "", p, flags=re.I) for p in parts if p]
        if len(parts) >= 3 and all(1 <= len(p.split()) <= 4 for p in parts):
            return parts[:max_items]
    return []


def strip_injection(question: str) -> tuple[str, str]:
    """-> (question without injected instructions, removed text). Sentences that are pure injections (fake
    sources, "pretend the textbook says ...", "end your answer with X", prompt-leak requests) are dropped whole;
    role / style / rule-change clauses (JAILBREAK_RE) are cut and the rest of their sentence is kept when it
    still asks something. The removed text is returned so callers can check that none of it leaks back."""
    kept, removed = [], []
    for sent in [s for s in SENT_SPLIT_RE.split(question.strip()) if s and s.strip()]:
        if INJECT_SENTENCE_RE.search(sent):
            removed.append(sent.strip())
            continue
        if JAILBREAK_RE.search(sent):
            rest = re.sub(r"\s+([,.;:?!])", r"\1", JAILBREAK_RE.sub(" ", sent))
            rest = re.sub(r"^[^\w]*(and|then|but|also|so)?\b[^\w]*", "", " ".join(rest.split()), flags=re.I)
            rest = re.sub(r"([?!])[.,;:\s]+$", r"\1", rest).strip(" ,;:")
            removed.append(sent.strip())
            if len(rest.split()) >= 2 and (SCIENCE_RE.search(rest) or "?" in rest):
                kept.append(rest)
            continue
        kept.append(sent.strip())
    return " ".join(kept).strip(), " ".join(removed).strip()


def question_forms(question: str) -> list[str]:
    """How the question is asked (several may apply). Pure rules: cheap, and the fast path has no LLM router."""
    q = question.strip()
    forms = []
    if ANALOGY_RE.search(q):
        forms.append("analogy")
    if PRESUPPOSITION_RE.search(q):
        forms.append("presupposition")
    if AMBIGUOUS_RE.match(q) and len(q.split()) <= 6:
        forms.append("ambiguous")
    if JAILBREAK_RE.search(q) or INJECT_SENTENCE_RE.search(q):
        forms.append("injection_wrapped")
    if PAGE_RE.search(q) or CITE_COUNT_RE.search(q) or QUOTE_RE.search(q):
        forms.append("citation_demand")
    return forms


def topic_query(question: str) -> str:
    """The TOPIC of a question for retrieval: injected instructions, analogy wrappers and citation demands
    ("cite page 999", "cite at least 10 sources") removed, so the reranker scores the subject itself."""
    q, _ = strip_injection(question)
    q = q or question
    q = ANALOGY_STRIP_RE.sub(" ", q)
    q = re.sub(r"\bpages?\s+\d{1,4}\s+of\s+(the\s+)?", "the ", q, flags=re.I)
    q = " ".join(s for s in SENT_SPLIT_RE.split(q) if s.strip() and not re.match(
        r"^\s*(please\s+)?(cite|quote|give|list)\b[^?]*\b(page|sources|citations|references)\b", s, re.I))
    q = re.sub(r"\s*,?\s*\band\s+(cite|quote)\b[^.?!]*", "", q, flags=re.I)
    q = PAGE_RE.sub(" ", q)
    q = re.sub(r"\s+([,.;:?!])", r"\1", " ".join(q.split())).strip(" ,;:")
    if q.rstrip(".?!") == question.strip().rstrip(".?!"):
        return question.strip()
    if q and q[-1] not in ".?!":
        q += "?" if re.match(r"^(what|why|how|which|who|when|where|is|are|does|do|can)\b", q, re.I) else "."
    return q if len(q.split()) >= 2 else question.strip()


def _rule_overrides(question: str, d: RouteDecision) -> RouteDecision:
    q = question
    if TABLE_RE.search(q) and d.answer_format != "table":
        d.answer_format = "table"
        d.rules.append("table")
    wants_image = bool(IMAGE_RE.search(q))
    if wants_image and not d.needs_images:
        d.needs_images = True
        d.rules.append("needs_images")
    elif d.needs_images and not wants_image:
        d.needs_images = False          # the router over-asks for figures; images only when the user asks
        d.rules.append("no_images")
    if d.answer_format == "table" and not TABLE_RE.search(q):
        d.answer_format = "prose"  # model over-uses tables (and pads them with unsupported rows); only when asked
        d.rules.append("no_table")
    if EQUATION_RE.search(q) and d.answer_format == "prose":
        d.answer_format = "equation"
        d.rules.append("equation")
    elif d.answer_format == "equation" and not EQUATION_RE.search(q) and d.intent != "calculation":
        d.answer_format = "prose"  # "what is X?" deserves an explanation, not just the bare equation
        d.rules.append("no_equation")
    if COMPARE_RE.search(q) and d.intent in ("other", "explanation", "definition"):
        d.intent = "comparison"
        d.rules.append("comparison")
    if CROSSREF_RE.search(q) and not d.is_cross_reference:
        d.is_cross_reference = True
        d.rules.append("cross_reference")
    if d.answer_format == "table" and d.intent == "other":
        d.intent = "table_request"
    if d.needs_images and d.intent == "other":
        d.intent = "figure_request"
    return d


def _clean_subqueries(d: RouteDecision) -> list[str]:
    if not (d.is_cross_reference or d.intent in ("comparison", "cross_reference")
            or (d.answer_format == "table" and len(d.sub_queries) >= 2)):
        return []
    out, seen = [], {d.standalone_query.strip().lower()}
    # the small model often leaves sub_queries empty: fall back to the focus entities (one topic each)
    cands = d.sub_queries if len(d.sub_queries) >= 2 else d.sub_queries + d.entities
    for s in cands:
        s = s.strip()
        if len(s) > 2 and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out[:4 if d.answer_format == "table" else 3]


def _router_questions() -> list[Question]:
    return [
        Text(name="standalone_query", max_chars=240, question=(
            "Rewrite the NEW MESSAGE as one self-contained question. Replace pronouns and references "
            "(it, its, this, that figure, the previous one, compare it) with the concrete subject taken from the "
            "memory (entities in focus / recent turns). If it is already self-contained, copy it unchanged.")),
        Noul(name="in_scope", default=True, question=(
            "Is the standalone question about materials science / engineering (the textbook topics), or a "
            "follow-up about them?")),
        Choice(name="intent", options=INTENTS, default="other", question="What does the user want?"),
        Choice(name="answer_format", options=FORMATS, default="prose", question=(
            "Best answer format: table (user asks for a table or a multi-property comparison), steps "
            "(procedure / calculation), equation (asks for a formula or equation), otherwise prose.")),
        Noul(name="needs_images", question="Does the user want to see a figure, diagram, image, plot or curve?"),
        Noul(name="is_cross_reference", question=(
            "Does the question connect different topics, chapters or sections, or ask which section explains a "
            "figure/table?")),
        TextList(name="sub_queries", max_items=4, question=(
            "Only if it connects, compares or lists several topics or items (e.g. 'the four types of X'): one short "
            "search query per topic / item, naming it (max 4); otherwise an empty list.")),
        TextList(name="entities", max_items=3, question=(
            "The 1-3 key technical terms (or figure/table numbers) the standalone question is about.")),
    ]


def is_jailbreak_only(question: str) -> bool:
    """Injection / role-play phrasing with no syllabus content left once it is removed -> out of scope."""
    if not (JAILBREAK_RE.search(question) or INJECT_SENTENCE_RE.search(question)):
        return False
    rest, _ = strip_injection(question)
    return not SCIENCE_RE.search(JAILBREAK_RE.sub(" ", rest))


def _apply_jailbreak(question: str, d: RouteDecision) -> RouteDecision:
    d.question_forms = question_forms(question)
    if is_jailbreak_only(question):
        d.in_scope, d.intent = False, "other"
        d.sub_queries, d.entities = [], []
        d.rules.append("jailbreak")
    elif "injection_wrapped" in d.question_forms:
        d.rules.append("injection_ignored")
    return d


def router_needed(question: str, memory_md: str) -> bool:
    """True if only the LLM router can decide: a reference to resolve from memory (pronouns / ellipsis), or a
    comparison / cross-reference that needs sub-queries. Otherwise keyword rules + reranker scores suffice
    (see fast_route) and the 3-5 s router call is skipped."""
    q = question.strip()
    if memory_md.strip() and (ANAPHORA_RE.search(q) or len(q.split()) < 5) and not REF_NUM_RE.search(q):
        return True
    if COMPARE_RE.search(q):
        return True
    if TABLE_RE.search(q) and not enumerated_items(q):
        return True                     # "a table of the four X": the router names the items (one search each)
    if CROSSREF_RE.search(q):
        # "what does Figure X show / which section explains it" is handled deterministically (figure + its linked
        # section text); only real topic connections need the router's sub-queries
        return not (FIG_REF_RE.search(q) and not RELATE_RE.search(q))
    return False


def fast_route(question: str, hits: list, *, min_score: float = 0.9, ooc_below: float = 0.25) -> RouteDecision | None:
    """Rule-based RouteDecision when the reranker already settles scope: best score >= ``min_score`` (clearly
    in the textbook) or < ``ooc_below`` (clearly not). None if ambiguous -> the caller uses the LLM router."""
    q = question.strip()
    top_hit = max(hits, key=lambda h: getattr(h, "rerank_score", None) or 0.0, default=None)
    top = (getattr(top_hit, "rerank_score", None) or 0.0) if top_hit is not None else 0.0
    if ooc_below <= top < min_score:
        return None
    in_scope = top >= min_score
    if CALC_RE.search(q):
        intent, fmt = "calculation", "steps"
    elif SUMMARY_RE.search(q):
        intent, fmt = "summary", "prose"
    elif DEFINE_RE.search(q):
        intent, fmt = "definition", "prose"
    else:
        intent, fmt = "explanation", "prose"
    entities: list[str] = []
    for m in FIG_REF_RE.finditer(q):
        entities.append(f"{m.group(1).capitalize()} {re.sub(r'\s+', '', m.group(2))}")
    if in_scope and top_hit is not None:
        sec = [s for s in (getattr(top_hit, "section_path", None) or []) if s]
        if sec:
            name = re.sub(r"^\s*[\d.]+\s*", "", sec[-1]).strip(" :")
            if name and len(name) <= 60:
                entities.append(name)
    d = RouteDecision(in_scope=in_scope, in_scope_confidence=round(top if in_scope else 1 - top, 4), intent=intent,
                      answer_format=fmt, standalone_query=q, entities=entities[:3] if in_scope else [],
                      confidence={"in_scope": round(top, 4)}, rules=["fast_route"], llm_ok=True)
    d = _rule_overrides(q, d)
    d.sub_queries = []
    return _apply_jailbreak(q, d)


def _greeting(question: str) -> RouteDecision:
    return RouteDecision(in_scope=True, in_scope_confidence=1.0, intent="greeting", standalone_query=question.strip(),
                         rules=["greeting"], confidence={"intent": 1.0})


async def route(question: str, memory_md: str, *, llm: LLM | None = None, **ids) -> RouteDecision:
    question = question.strip()
    if GREETING_RE.match(question):
        return _greeting(question)
    words = len(question.split())
    has_ref = bool(ANAPHORA_RE.search(question)) or words < 5
    rewrite = bool(memory_md.strip()) and has_ref   # otherwise the standalone query is the question itself
    state = (f"MEMORY:\n{memory_md.strip()}\n\nNEW MESSAGE:\n{question}" if memory_md.strip()
             else f"NEW MESSAGE:\n{question}")
    qs = _router_questions() if rewrite else [q for q in _router_questions() if q.name != "standalone_query"]
    r = await decide(state, qs, purpose="router", system=prompts.ROUTER_SYSTEM, llm=llm, **ids)
    r.setdefault("standalone_query", question)
    r.setdefault("standalone_query_confidence", 1.0)
    d = RouteDecision(
        in_scope=r["in_scope"], in_scope_confidence=r["in_scope_confidence"], intent=r["intent"],
        needs_images=r["needs_images"], answer_format=r["answer_format"],
        is_cross_reference=r["is_cross_reference"] or r["intent"] == "cross_reference",
        standalone_query=(r["standalone_query"] or "").strip(), sub_queries=list(r["sub_queries"] or []),
        entities=[e.strip() for e in (r["entities"] or []) if e.strip()][:3],
        confidence={k[:-11]: v for k, v in r.items() if k.endswith("_confidence")}, llm_ok=r["_ok"])
    if not r["_ok"]:
        d.in_scope, d.intent = True, "other"  # let the retrieval score gate decide
    # standalone query: keep the user's own words unless there is something to resolve
    if not d.standalone_query or (not has_ref and words >= 5):
        d.standalone_query = question
    elif (has_ref and memory_md.strip() and d.standalone_query.lower() == question.lower()
          and words <= 7 and not REF_NUM_RE.search(question)):
        focus = ENTITY_LINE_RE.findall(memory_md)
        if focus:  # model failed to resolve the reference: anchor it on the newest entity in focus
            d.standalone_query = f"{question} (about {focus[0]})"
            d.rules.append("anaphora_fallback")
    d = _rule_overrides(question, d)
    if d.intent == "greeting" and not GREETING_RE.match(question):
        d.intent = "other"  # a real question that merely starts politely
    d.sub_queries = _clean_subqueries(d)
    return _apply_jailbreak(question, d)


# ----------------------------------------------------------------------------- feedback review (FEATURES_V2 §1)
ISSUE_TYPES = ["wrong_fact", "missing_citation", "bad_format", "refused_wrongly", "incomplete", "other"]
FEEDBACK_REVIEW_SYSTEM = (
    "You review answers of a textbook Q&A assistant (materials science) that a user DISLIKED. The assistant must "
    "answer only from its numbered SOURCES, cite [n] after every fact, use the requested format, and answer "
    "exactly #OutofContext only for questions outside the textbook. Decide the most likely reason for the dislike "
    "from the question, the answer and the sources. Be concrete and brief.")


async def review_feedback(question: str, answer: str, sources: list[dict] | None, comment: str | None = None,
                          *, llm: LLM | None = None, **ids) -> dict[str, Any]:
    """Structured review of one disliked answer -> {issue_type, explanation, suggested_fix, confidence, _ok}."""
    src = "\n".join(f"[{s.get('n', i + 1)}] (page {', '.join(map(str, (s.get('pages') or [])[:3]))}) "
                    f"{' '.join(str(s.get('snippet') or s.get('content') or '').split())[:500]}"
                    for i, s in enumerate((sources or [])[:8])) or "(no sources were cited)"
    state = (f"QUESTION:\n{question.strip()[:2000]}\n\nANSWER (disliked):\n{answer.strip()[:3000]}\n\n"
             f"SOURCES CITED:\n{src}" + (f"\n\nUSER COMMENT:\n{comment.strip()[:500]}" if comment else ""))
    r = await decide(state, [
        Choice(name="issue_type", options=ISSUE_TYPES, default="other", question=(
            "Main problem: wrong_fact (a statement contradicts or is not in the sources), missing_citation (facts "
            "without [n]), bad_format (ignored the requested format / unreadable), refused_wrongly (#OutofContext or "
            "a refusal although the topic is in the textbook), incomplete (misses part of the question), other.")),
        Text(name="explanation", max_chars=400, question="One or two sentences: what exactly is wrong."),
        Text(name="suggested_fix", max_chars=400, question=(
            "One or two sentences: what the system (prompt, retrieval, routing) should change to avoid it.")),
    ], purpose="feedback_review", system=FEEDBACK_REVIEW_SYSTEM, llm=llm, **ids)
    return {"issue_type": r["issue_type"], "explanation": r["explanation"], "suggested_fix": r["suggested_fix"],
            "confidence": r.get("issue_type_confidence", 0.0), "_ok": r["_ok"]}


def heuristic_title(question: str) -> str:
    q = re.sub(r"[?!.]+$", "", question.strip())
    q = re.sub(r"^(please\s+)?(can|could|would)\s+you\s+(please\s+)?|^(tell me|explain|describe)\s+", "", q, flags=re.I)
    words = q.split()
    t = " ".join(words[:6])
    return (t[:1].upper() + t[1:]) if t else "New chat"
