"""BM25 keyword index over the chunks table (bm25s) with a scientific tokenizer.

Tokenizer rules:
  * tokens = decimal numbers (``0.2``, ``1.5``) or runs of letters/digits (``Fe3C``, ``304L``, ``fcc``)
  * everything is lower-cased; pure-alpha tokens are Snowball-stemmed (PyStemmer) after stopword removal
  * tokens containing digits (formulas, alloy grades, numbers) are kept whole, never stemmed
  * chemical formulas (``Fe3C``, ``Al2O3``, ``SiO2``, ``NaCl``) additionally emit their element symbols
    (``fe``, ``c`` / ``al``, ``o``) so "iron carbide" style queries mentioning ``Fe`` still match
The index is built in memory from the DB (375 chunks -> milliseconds) and rebuilt by ``reload()``.
"""
from __future__ import annotations

import re
import threading

import bm25s
import numpy as np
import Stemmer

from app import db

ELEMENTS = frozenset("""H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge
As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er
Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr""".split())

STOPWORDS = frozenset("""a about above after again against all am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for from further had has
have having he her here hers herself him himself his how i if in into is it its itself just me more most my
myself no nor not now of off on once only or other our ours ourselves out over own same she should so some
such than that the their theirs them themselves then there these they this those through to too under until
up very was we were what when where which while who whom why will with would you your yours yourself
yourselves also may might must shall us via per e g ie eg etc""".split())

_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)+|[^\W_]+", re.UNICODE)
_FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\d*(?:\.\d+)?)+$")
_ELEMENT_RE = re.compile(r"[A-Z][a-z]?")

_stemmer = Stemmer.Stemmer("english")


def _formula_elements(tok: str) -> list[str] | None:
    """Element symbols if ``tok`` (original case) looks like a chemical formula, else None."""
    if not _FORMULA_RE.match(tok):
        return None
    symbols = _ELEMENT_RE.findall(tok)
    if not all(s in ELEMENTS for s in symbols):
        return None
    if not any(ch.isdigit() for ch in tok) and (len(symbols) < 2 or tok.isupper()):
        return None      # NaCl / SiC yes; single symbols ("In") and acronyms ("FCC", "BCC") no
    return [s.lower() for s in symbols]


def tokenize(text: str) -> list[str]:
    """Scientific tokenizer (see module docstring)."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text or ""):
        low = raw.lower()
        if any(ch.isdigit() for ch in raw):
            out.append(low.replace(",", "."))              # keep Fe3C / 304L / 0.2 whole
            elems = _formula_elements(raw)
            if elems:
                out.extend(elems)
            continue
        elems = _formula_elements(raw)                      # digit-free formulas, e.g. NaCl, SiC
        if elems:
            out.append(low)
            out.extend(elems)
            continue
        if low in STOPWORDS:
            continue
        out.append(_stemmer.stemWord(low))
    return out


class BM25Index:
    """In-memory BM25 over ``header + content`` of every chunk."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.chunk_ids: list[str] = []
        self.chunk_types: np.ndarray = np.array([], dtype=object)
        self.retriever: bm25s.BM25 | None = None

    def build_from_db(self) -> "BM25Index":
        with db.get_pool().connection() as conn:
            rows = conn.execute(
                "SELECT chunk_id, chunk_type, coalesce(header,'') || E'\\n' || coalesce(content,'') "
                "FROM chunks ORDER BY chunk_id").fetchall()
        self.build([(r[0], r[1], r[2]) for r in rows])
        return self

    def build(self, docs: list[tuple[str, str, str]]) -> None:
        """docs = [(chunk_id, chunk_type, text)]."""
        retriever = None
        if docs:
            retriever = bm25s.BM25()
            retriever.index([tokenize(t) or ["_empty_"] for _, _, t in docs], show_progress=False)
        with self._lock:
            self.chunk_ids = [d[0] for d in docs]
            self.chunk_types = np.array([d[1] for d in docs], dtype=object)
            self.retriever = retriever

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def search(self, query: str, k: int = 40, chunk_types: list[str] | None = None) -> list[tuple[str, float]]:
        """Top-k (chunk_id, bm25 score) with score > 0, optionally restricted to chunk types."""
        q = tokenize(query)
        with self._lock:
            if self.retriever is None or not q:
                return []
            scores = np.asarray(self.retriever.get_scores(q), dtype=np.float32)
            ids, types = self.chunk_ids, self.chunk_types
        if chunk_types:
            scores = np.where(np.isin(types, chunk_types), scores, 0.0)
        order = np.argsort(-scores)[:k]
        return [(ids[i], float(scores[i])) for i in order if scores[i] > 0]
