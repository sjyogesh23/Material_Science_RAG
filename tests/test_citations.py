"""Unit tests for app.rag.citations (no DB / LLM needed)."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag import citations  # noqa: E402
from app.rag.citations import validate  # noqa: E402


@dataclass
class FakeHit:  # same attribute names as app.retrieval.hybrid.Hit
    chunk_id: str
    content: str
    chunk_type: str = "text"
    doc_id: str = "doc_test"
    header: str = "Book | Ch 1 > 1.1"
    text: str = ""
    section_path: list[str] = field(default_factory=lambda: ["Ch 1", "1.1"])
    pages: list[int] = field(default_factory=lambda: [3])
    payload: dict = field(default_factory=dict)
    rerank_score: float | None = 0.9
    rrf_score: float = 0.01
    via: str = "hybrid"


def hits(n: int = 4) -> list[FakeHit]:
    out = []
    for i in range(1, n + 1):
        out.append(FakeHit(f"c{i}", f"content number {i} about crystals lattice planes diffraction",
                           payload={"doc_title": "Book", "page_sizes": {"3": [612, 792]},
                                    "locations": [{"page": 3, "bbox": [10, 20, 30, 40]},
                                                  {"page": 3, "bbox": [10, 50, 30, 60]}]}))
    return out


def test_renumber_by_first_appearance_and_drop_uncited():
    clean, used = validate("Bragg's law relates spacing [3]. Lattice planes diffract [1]. Again [3].", hits())
    assert clean == "Bragg's law relates spacing [1]. Lattice planes diffract [2]. Again [1]."
    assert [s["n"] for s in used] == [1, 2]
    assert [s["chunk_id"] for s in used] == ["c3", "c1"]
    assert not used.uncited


def test_invalid_numbers_removed_and_groups_merged():
    clean, used = validate("A fact [2][9]. Another [4, 2]. Range [1-2]. Bogus [7].", hits())
    assert clean == "A fact [1]. Another [1, 2]. Range [1, 3]. Bogus."
    assert [s["chunk_id"] for s in used] == ["c2", "c4", "c1"]


def test_source_word_and_table_cells():
    ans = "| Structure | Packing |\n|---|---|\n| FCC | 0.74 [Source 2] |\n| BCC | 0.68 [3] |"
    clean, used = validate(ans, hits())
    assert "| FCC | 0.74 [1] |" in clean and "| BCC | 0.68 [2] |" in clean
    assert len(used) == 2


def test_math_is_not_touched():
    clean, _ = validate("The interval $x \\in [0, 1]$ holds [2]. $$n\\lambda = 2d\\sin\\theta$$ [1]", hits())
    assert "$x \\in [0, 1]$" in clean
    assert clean.endswith("$$n\\lambda = 2d\\sin\\theta$$ [2]")


def test_markdown_links_untouched():
    clean, used = validate("See [1](http://x) and fact [1].", hits())
    assert "[1](http://x)" in clean
    assert len(used) == 1


def test_out_of_context_passthrough():
    clean, used = validate("  #OutofContext  ", hits())
    assert clean == "#OutofContext" and used == []


def test_zero_citations_overlap_appends_best():
    h = hits()
    clean, used = validate("Content about crystals lattice planes diffraction number", h)
    assert clean.endswith("[1]") and used[0]["chunk_id"] == "c1"


def test_zero_citations_no_overlap_flags_uncited():
    clean, used = validate("Completely unrelated sentence regarding football tournaments.", hits())
    assert used == [] and used.uncited is True
    assert "[" not in clean


def test_source_payload_highlights_and_page_size():
    _, used = validate("x [1]", hits())
    s = used[0]
    assert s["highlights"] == [{"page": 3, "bboxes": [[10, 20, 30, 40], [10, 50, 30, 60]]}]
    assert s["page_size"] == {"3": [612, 792]}
    assert set(s) >= {"n", "chunk_id", "doc_id", "doc_title", "chunk_type", "header", "section_path", "pages",
                      "snippet", "score", "highlights", "page_size", "figure"}


def test_page_size_from_pdf(tmp_path):
    import pymupdf

    pdf = tmp_path / "t.pdf"
    doc = pymupdf.open()
    doc.new_page(width=500, height=700)
    doc.save(pdf)
    citations.register_pdf_path("doc_pdf", pdf)
    assert citations.page_sizes("doc_pdf", [1]) == {"1": [500.0, 700.0]}


def test_image_payload():
    h = FakeHit("f1", "Figure 1-6: Bragg Diffraction\nFigure description: A schematic of planes.",
                chunk_type="figure", pages=[22],
                payload={"figure": {"figure_number": "Figure 1-6", "image_file": "x.png"}})
    im = citations.to_image(h)
    assert im == {"chunk_id": "f1", "url": "/api/figure/f1", "caption": "Figure 1-6: Bragg Diffraction",
                  "figure_number": "Figure 1-6", "page": 22, "description": "A schematic of planes."}
