"""Formula post-processing: LaTeX clean-up, validation, MathML, equation numbers, variables."""
from __future__ import annotations

import re
from collections import Counter

try:
    from latex2mathml.converter import convert as _latex_to_mathml
except ImportError:  # pragma: no cover
    _latex_to_mathml = None

GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "varepsilon", "ϵ": "epsilon", "ζ": "zeta",
    "η": "eta", "θ": "theta", "ϑ": "vartheta", "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu", "µ": "mu",
    "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau", "υ": "upsilon", "φ": "phi",
    "ϕ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega", "Γ": "Gamma", "Δ": "Delta", "Θ": "Theta", "Λ": "Lambda",
    "Ξ": "Xi", "Π": "Pi", "Σ": "Sigma", "Φ": "Phi", "Ψ": "Psi", "Ω": "Omega",
}
_GREEK_CMDS = set(GREEK.values()) | {"epsilon", "varphi", "varrho", "varsigma", "varpi"}
_NOT_VARS = {"frac", "left", "right", "sqrt", "sum", "int", "oint", "prod", "lim", "partial", "nabla", "cdot",
             "times", "mathrm", "mathbf", "mathit", "text", "operatorname", "vec", "hat", "bar", "dot", "ddot",
             "tilde", "overline", "underline", "begin", "end", "quad", "qquad", "exp", "log", "ln", "sin", "cos",
             "tan", "sinh", "cosh", "tanh", "max", "min", "infty", "approx", "equiv", "propto", "leq", "geq",
             "neq", "pm", "mp", "to", "rightarrow", "leftarrow", "Big", "big", "Bigg", "bigg", "mathcal", "boldsymbol",
             "langle", "rangle", "intertext", "displaystyle", "textstyle", "hbar", "ell", "prime", "circ", "mathsf",
             "mathbb", "operatorname*", "det", "tr", "Tr", "d"}

EQ_NUM_RE = re.compile(r"\(\s*((?:[A-Z]\.?)?\d+(?:[.\-]\d+)*[a-z]?)\s*\)\s*[.,;]?\s*$")
TAG_RE = re.compile(r"\\tag\*?\{([^}]*)\}")


def compact_latex(latex: str) -> str:
    """Remove the token spacing Docling emits ("d r _ { i }" -> "dr_{i}") without changing meaning."""
    s = latex.strip()
    s = re.sub(r"(?<!\\)\s+(?=[_^{}])", "", s)  # space before _ ^ { }
    s = re.sub(r"(?<=[_^{])\s+", "", s)          # space after _ ^ {
    s = re.sub(r"(?<=})\s+(?=[^\\\s])", "", s)   # space after } (unless a command follows)
    prev = None
    while prev != s:  # join single-char tokens: "1 0 0" -> "100", but keep "\alpha b"
        prev = s
        s = re.sub(r"(?<![\\a-zA-Z])([a-zA-Z0-9]) (?=[a-zA-Z0-9(])", r"\1", s)
    s = re.sub(r"\s*([=+<>,])\s*", r" \1 ", s)
    s = re.sub(r" ,", ",", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def to_mathml(latex: str) -> str | None:
    if not _latex_to_mathml or not latex:
        return None
    try:
        return _latex_to_mathml(latex)
    except Exception:  # noqa: BLE001
        return None


def _alnum_bag(s: str) -> Counter:
    s = re.sub(r"\\[a-zA-Z]+", " ", s)  # drop latex commands
    return Counter(ch for ch in s if ch.isalnum())


def agreement(latex: str, text_layer: str) -> float | None:
    """Overlap between the characters of the LaTeX and of the PDF text layer (0..1)."""
    if not text_layer or not latex:
        return None
    # map unicode greek in the text layer to their names so they line up with \alpha etc.
    tl = "".join(GREEK.get(ch, ch) if ch in GREEK else ch for ch in text_layer)
    lt = re.sub(r"\\(" + "|".join(sorted(_GREEK_CMDS, key=len, reverse=True)) + r")(?![a-zA-Z])", lambda m: m.group(1), latex)
    a, b = _alnum_bag(lt), _alnum_bag(tl)
    if not a or not b:
        return None
    inter = sum((a & b).values())
    return round(2 * inter / (sum(a.values()) + sum(b.values())), 3)


def quality_flags(latex: str, text_layer: str | None) -> list[str]:
    flags = []
    if re.search(r"\\intertext|\\text\s*\{\s*(?:\w\s){3,}", latex):
        flags.append("contains_spurious_text")
    toks = latex.split()
    if len(toks) > 30 and len(set(toks)) / len(toks) < 0.2:
        flags.append("repetitive_output")
    if text_layer and len(latex) > 8 * max(10, len(text_layer)):
        flags.append("much_longer_than_text_layer")
    if latex.count("{") != latex.count("}"):
        flags.append("unbalanced_braces")
    return flags


TRAILING_NUM_RE = re.compile(r"(?:\\quad|\\qquad|\\ |~|\s)*\(\s*((?:[A-Z]\s?\.?\s?)?\d(?:\s?\d)*(?:\s?\.\s?\d(?:\s?\d)*)*\s?[a-z]?)\s*\)\s*[.,]?\s*$")


def clean_latex(raw: str, orig: str | None) -> tuple[str, str | None, list[str]]:
    """Trim hallucinated tails from formula-model output.

    The model sometimes emits the right equation followed by ``\\\\ \\intertext{...}`` and a repeat, or
    junk after a blank line. Candidates are scored against the PDF text layer and the best one is kept.
    Returns (latex, equation_number_from_latex, notes).
    """
    notes: list[str] = []
    cands = [raw]
    for marker in (r"\intertext", r"\\ \\", r"\\"):
        idx = raw.find(marker)
        if idx > 0:
            cands.append(raw[:idx])
    best, best_score = raw, agreement(raw, orig or "") or 0.0
    for c in cands[1:]:
        sc = agreement(c, orig or "") or 0.0
        if sc > best_score + 0.02:
            best, best_score = c, sc
    if best is not raw:
        notes.append("truncated_spurious_tail")
    best = re.sub(r"(?:\\\\\s*)+$", "", best.strip()).strip()
    if "&" in best and not re.search(r"\\begin\{", best):
        best = best.replace("&", "")
        notes.append("removed_alignment_marks")
    eq = None
    best = re.sub(r"(?:\\[ ,;:!]|\\quad|\\qquad|\s)+$", "", best)
    m = TRAILING_NUM_RE.search(best)
    if m and len(best) - len(m.group(0)) > 3:
        eq = "(" + re.sub(r"\s+", "", m.group(1)) + ")"
        best = best[: m.start()].rstrip(" ,\\")
    best = re.sub(r"(?:\\[ ,;:!]|\\quad|\\qquad|\s)+$", "", best)
    return compact_latex(best), eq, notes


def equation_number(latex: str, orig: str | None) -> str | None:
    m = TAG_RE.search(latex or "")
    if m:
        return f"({m.group(1)})"
    m = EQ_NUM_RE.search(orig or "")
    return f"({m.group(1)})" if m else None


def symbols(latex: str) -> list[str]:
    """Candidate variable symbols appearing in the LaTeX (in order of appearance)."""
    out, seen = [], set()
    pattern = re.compile(r"\\([a-zA-Z]+)(?:\s*_\s*(\{[^{}]*\}|[a-zA-Z0-9]))?|(?<![\\a-zA-Z])([a-zA-Z])(?![a-zA-Z])(?:\s*_\s*(\{[^{}]*\}|[a-zA-Z0-9]))?")
    for m in pattern.finditer(latex or ""):
        cmd, sub1, letter, sub2 = m.groups()
        if cmd:
            if cmd not in _GREEK_CMDS:
                continue
            sym = "\\" + cmd + ("_" + sub1.replace(" ", "") if sub1 else "")
        else:
            if letter == "d" and not sub2:
                continue
            sym = letter + ("_" + sub2.replace(" ", "") if sub2 else "")
        key = re.sub(r"_\{?([a-z]{1,4})\}?$", "", sym) if re.search(r"_\{?[ijkln]{1,4}\}?$", sym) else sym
        if key not in seen:
            seen.add(key)
            out.append(sym)
    return out[:30]


_WHERE_RE = re.compile(r"\b(?:where|in which|here)\b[,:]?\s+(.*)", re.I | re.S)
_DEF_RE = re.compile(
    r"^\s*(?:and\s+)?([^\s,;]{1,12})\s+(?:is|are|denotes|denote|represents|represent|stands for|is defined as|=)\s+(?:the\s+|a\s+|an\s+)?(.{3,120})$",
    re.I)


def parse_where_clause(text: str) -> list[dict]:
    """'where σ is the stress, ε the strain and E is Young's modulus' -> [{symbol, meaning}, ...]."""
    m = _WHERE_RE.search(text or "")
    if not m:
        return []
    body = m.group(1)
    body = re.split(r"(?<=[a-z0-9)])\.\s+[A-Z]", body)[0]  # stop at sentence end
    parts = re.split(r";|,\s*(?=\S{1,12}\s+(?:is|are|denotes|represents|the)\b)|\s+and\s+(?=\S{1,12}\s+(?:is|are|denotes|represents|the)\b)", body)
    out = []
    for p in parts:
        p = p.strip(" .")
        d = _DEF_RE.match(p) or re.match(r"^\s*(?:and\s+)?([^\s,;]{1,12})\s+(?:the\s+)(.{3,120})$", p, re.I)
        if d:
            out.append({"symbol": d.group(1).strip(), "meaning": d.group(2).strip(" .,")})
    return out[:20]


def symbol_to_latex(sym_text: str) -> str:
    return "".join(("\\" + GREEK[ch]) if ch in GREEK else ch for ch in sym_text)


def build_variables(latex: str, where_defs: list[dict]) -> list[dict]:
    syms = symbols(latex)
    defs = {symbol_to_latex(d["symbol"]).replace(" ", ""): d["meaning"] for d in where_defs}
    out = []
    for s in syms:
        base = s.replace("{", "").replace("}", "")
        # exact match only: r_o must not inherit the meaning of plain r
        meaning = defs.get(s) or defs.get(base) or (defs.get(base.split("_")[0]) if "_" not in base else None)
        out.append({"symbol": s, "meaning": meaning})
    # definitions whose symbol we did not spot in the LaTeX are still useful
    known = {v["symbol"].replace("{", "").replace("}", "").split("_")[0] for v in out}
    for k, v in defs.items():
        if k.split("_")[0] not in known:
            out.append({"symbol": k, "meaning": v})
    return out
