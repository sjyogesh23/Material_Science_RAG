"""Heuristic bibliography-entry parser (APA / ACS / physics / IEEE styles)."""
from __future__ import annotations

import re

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+[^\s\"<>.,;)\]])", re.I)
ARXIV_RE = re.compile(r"arXiv[:\s]*(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+/\d{7})", re.I)
URL_RE = re.compile(r"https?://\S+[^\s.,;)\]]")
YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)[a-z]?\b")
MARKER_RE = re.compile(r"^\s*(?:\[(\d{1,4})\]|\((\d{1,4})\)|(\d{1,4})[.)]\s)")
# "Journal Name 12(3), 45-60" | "Phys. Rev. B 97, 014105 (2018)" | "vol. 12, no. 3, pp. 45-60"
VOL_RE = re.compile(r"(?:vol\.?\s*)?\b(\d{1,4})\s*(?:\((\d{1,4}(?:[-–]\d{1,4})?)\)|,\s*no\.?\s*(\d{1,4}))?\s*[,:]\s*(?:pp?\.?\s*)?([A-Z]?\d{1,6}(?:\s*[-–]\s*[A-Z]?\d{1,6})?)\b")
# physics style: "Phys. Rev. B 88, 134102 (2013)"
PHYS_RE = re.compile(r"((?:[A-Z][A-Za-z]*\.?\s?){1,6}(?:[A-Z])?)\s+(\d{1,4}),\s*([A-Z]?\d{1,7})\s*\(((?:19|20)\d{2})\)")
CITATION_RE = re.compile(r"\[(\d{1,4}(?:\s*[-–,]\s*\d{1,4})*)\]")


def parse_reference(text: str) -> dict:
    t = " ".join(text.split())
    out: dict = {"raw": t}
    m = MARKER_RE.match(t)
    if m:
        out["marker"] = next(g for g in m.groups() if g)
        t = t[m.end():].strip()
    if (d := DOI_RE.search(t)):
        out["doi"] = d.group(1)
    if (a := ARXIV_RE.search(t)):
        out["arxiv_id"] = a.group(1)
    if (u := URL_RE.search(t)):
        out["url"] = u.group(0)
    years = YEAR_RE.findall(t)
    if years:
        out["year"] = int(years[-1] if re.search(r"\((19|20)\d{2}\)\s*\.?$", t) else years[0])

    # quoted title (IEEE / ACS) or the sentence after "(year)." (APA)
    title = None
    q = re.search(r"[\"“](.{10,300}?)[\"”]", t)
    if q:
        title = q.group(1).strip(" ,.")
    else:
        apa = re.search(r"\((?:19|20)\d{2}[a-z]?\)\.\s*(.{10,300}?)\.\s", t)
        if apa:
            title = apa.group(1)
    if title:
        out["title"] = title

    # authors: the leading segment before the year / title
    cut = len(t)
    phys = PHYS_RE.search(t)
    if phys and "title" not in out:
        out["journal"], out["volume"], out["pages"] = phys.group(1).strip(" ,"), phys.group(2), phys.group(3)
        out["year"] = int(phys.group(4))
    for pat in (r"\((?:19|20)\d{2}", r"[\"“]") + ((re.escape(phys.group(0)),) if phys else ()):
        mm = re.search(pat, t)
        if mm:
            cut = min(cut, mm.start())
    head = t[:cut].strip(" ,.")
    if head and len(head) < 400:
        authors = [a.strip(" .") for a in re.split(r",\s*(?:and\s+|&\s*)?|\s+and\s+|;\s*|\s&\s", head) if a.strip(" .")]
        # merge "Smith, J." style pairs: a surname followed by initials
        merged = []
        for a in authors:
            if merged and re.fullmatch(r"(?:[A-Z]\.\s?-?){1,4}", a + ".") and " " not in merged[-1]:
                merged[-1] = f"{merged[-1]}, {a}."
            else:
                merged.append(a)
        if merged and all(len(a) < 60 for a in merged):
            out["authors"] = merged[:50]
            out["et_al"] = bool(re.search(r"et\s+al", head))

    if "journal" not in out and (v := VOL_RE.search(t[cut:] if cut < len(t) else t)):
        out["volume"] = v.group(1)
        if v.group(2) or v.group(3):
            out["issue"] = v.group(2) or v.group(3)
        out["pages"] = re.sub(r"\s+", "", v.group(4))
        jseg = t[cut:][: v.start()] if cut < len(t) else t[: v.start()]
        jseg = re.sub(r"^.*?[\"“].*?[\"”]\s*,?", "", jseg)  # drop quoted title
        jseg = re.sub(r"^\s*\((?:19|20)\d{2}[a-z]?\)\.?", "", jseg)
        if title:
            jseg = jseg.replace(title, "")
        journal = jseg.strip(" ,.:;")
        if 2 < len(journal) < 150:
            out["journal"] = journal.split(". ")[-1].strip(" ,.") if ". " in journal and len(journal) > 60 else journal
    out["parser"] = "heuristic"
    return out


def expand_citation(s: str) -> list[str]:
    """'3-5, 8' -> ['3','4','5','8']"""
    out = []
    for part in re.split(r"\s*,\s*", s):
        r = re.split(r"\s*[-–]\s*", part)
        if len(r) == 2 and r[0].isdigit() and r[1].isdigit() and int(r[1]) - int(r[0]) < 50:
            out += [str(i) for i in range(int(r[0]), int(r[1]) + 1)]
        elif part.isdigit():
            out.append(part)
    return out
