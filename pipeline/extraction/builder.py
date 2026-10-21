"""Builds the final per-PDF JSON from Docling (semantics) + PyMuPDF (native layer).

Phase A  - convert page batches with Docling and turn every item into an element
           (crops, OCR, colours, tables, LaTeX ...).
Phase B  - document-wide passes: running headers/footers, title/subtitle, heading
           levels, section paths, categories, line numbers, columns, captions,
           footnotes, cross-references, citations, table continuation, links,
           annotations, form fields, watermarks, sidebars, entities, relationships.
"""
from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, pstdev

from . import formulas as F
from .config import EXTRACTOR_NAME, EXTRACTOR_VERSION, Settings
from .domain import extract_entities, merge_entity_summary
from .images import (decode_codes, dominant_colors, image_stats, merge_text_sources, micrograph_cues,
                     ocr_image)
from .native import NativeDoc, NativePage, read_native
from .references import CITATION_RE, expand_citation, parse_reference
from .tables import build_table
from .utils import (area, center, contains_point, normalize_ws, overlap_ratio, r2, rbbox, sha256_file,
                    union, word_count)

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- patterns
HEADING_NUM_RE = re.compile(r"^\s*((?:\d+|[IVXLC]+|[A-Z])(?:\.\d+)*)\.?\s+\S")
CAPTION_NUM_RE = re.compile(r"^\s*(Fig(?:ure)?s?\.?|Table|Tab\.|Scheme|Chart|Plate|Graph|Map|Box|Algorithm|Listing)\s*([A-Z]?\d+(?:[.\-]\d+)*[a-z]?|[IVX]+)", re.I)
CAPTION_START_RE = re.compile(CAPTION_NUM_RE.pattern + r"\s*[.:|\u2014–]\s", re.I)
FIG_REF_RE =re.compile(r"\b(?:Fig(?:ure)?s?\.?)\s*([A-Z]?\d+(?:[.\-]\d+)*)(?:\s*\(?([a-h])\)?)?", re.I)
TAB_REF_RE = re.compile(r"\bTables?\.?\s*([A-Z]?\d+(?:[.\-]\d+)*|[IVX]+)\b")
EQ_REF_RE = re.compile(r"\b(?:Eqs?\.?|Equations?)\s*\(?\s*([A-Z]?\d+(?:\.\d+)*[a-z]?)\s*\)?", re.I)
SEC_REF_RE = re.compile(r"\b(?:Sec(?:tion)?s?\.?|§)\s*(\d+(?:\.\d+)*)", re.I)
FOOTNOTE_MARK_RE = re.compile(r"^\s*(\d{1,2}|[*†‡§¶#a-z])\s*[).]?\s+\S")
QUOTE_RE = re.compile(r"^\s*[\"“«„](.{15,}?)[\"”»]\s*(?:[\u2014–-]{1,2}\s*(.{2,80}))?\s*$", re.S)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
AFFIL_RE = re.compile(r"\b(Universit|Institut|Department|Dept\.|Laborator|College|School of|Facult|Cent(?:er|re)\b|"
                      r"Academy|Corporation|Inc\.|GmbH|Ltd|National|Research|Recherche|Service|CNRS|CEA|CSIC|"
                      r"Fraunhofer|Max[- ]Planck|Helmholtz|USA|U\.S\.A|United States|France|Germany|China|Japan|"
                      r"India|Korea|Italy|Spain|United Kingdom|UK|Canada|Australia|Switzerland|Netherlands|Sweden)\w*",
                      re.I)
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+[^\s\"<>.,;)\]])")
KEYWORDS_RE = re.compile(r"^\s*(?:Key\s?words|Index Terms)\s*[:\u2014–-]\s*(.+)$", re.I | re.S)
PAGE_NUM_RE = re.compile(r"^\s*(?:page\s*)?(\d{1,4}|[ivxlc]{1,6})(?:\s*(?:of|/)\s*\d{1,4})?\s*$", re.I)
ROLE_PREFIX_RE = re.compile(
    r"^\s*(Example|Worked Example|Problem|Exercise|Question|Solution|Answer|Definition|Theorem|Lemma|Corollary|Proof|"
    r"Note|Tip|Warning|Caution|Important|Remark|Key Points?|Summary|Learning Objectives?|Objectives|Case Study|"
    r"Activity|Experiment|Procedure|Safety)(?:\s+[A-Z]?\d+(?:\.\d+)*[a-z]?)?\s*[:.\u2014–]", re.I)
ROLE_NAMES = {"worked example": "example", "key point": "key_points", "key points": "key_points",
              "learning objective": "learning_objectives", "learning objectives": "learning_objectives",
              "objectives": "learning_objectives", "case study": "case_study"}
SECTION_ROLES = [
    ("abstract", r"^abstract|^summary$"), ("introduction", r"introduction|^background|^overview|^motivation"),
    ("related_work", r"related work|literature review|state of the art|prior work"),
    ("methods", r"method|experimental|materials and|procedure|synthesis|fabrication|preparation|simulation details|computational details|approach|set-?up"),
    ("theory", r"theory|theoretical|model(?:ing|ling)?\b|formulation|framework"),
    ("results", r"result|findings|characteri[sz]ation|performance|evaluation|analysis"),
    ("discussion", r"discussion"), ("conclusion", r"conclusion|concluding|outlook|future work|summary and"),
    ("acknowledgements", r"acknowledg"), ("references", r"^references?$|bibliography|works cited|literature cited|^citations?$"),
    ("appendix", r"appendix|supplementary|supporting information"),
    ("author_info", r"author contributions?|competing interests?|conflicts? of interest|data availability|funding|code availability|declaration"),
    ("nomenclature", r"nomenclature|abbreviations|list of symbols|glossary|notation"),
    ("problems", r"problems|exercises|questions|homework|assignments?"),
]

TEXT_LABEL_CATEGORY = {"title": "title", "section_header": "heading", "text": "description", "paragraph": "description",
                       "caption": "caption", "reference": "reference", "handwritten_text": "handwritten",
                       "grading_scale": "grading_scale", "marker": "marker", "list_item": "list_item"}
PICTURE_TYPE_BY_CLASS = {"signature": "signature", "stamp": "stamp", "logo": "logo", "qr_code": "barcode",
                         "bar_code": "barcode"}
CHART_CLASSES = {"bar_chart", "line_chart", "pie_chart", "scatter_plot", "scatter_chart", "box_plot", "other_chart",
                 "stacked_bar_chart", "heatmap", "flow_chart"}


def _gc() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _rss_gb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e9
    except Exception:  # noqa: BLE001
        return 0.0


def _ram_fraction() -> float:
    try:
        import psutil

        return psutil.virtual_memory().percent / 100
    except Exception:  # noqa: BLE001
        return 0.0


def _nan(v):
    try:
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)
    except (TypeError, ValueError):
        return None


# =================================================================================== Docling
class DoclingRunner:
    """Owns the Docling converter (models are loaded once and reused across PDFs)."""

    def __init__(self, s: Settings):
        self.s = s
        self._conv = None

    @property
    def converter(self):
        if self._conv is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (AcceleratorOptions, HeadingHierarchyOptions,
                                                            PdfPipelineOptions, TableFormerMode)
            from docling.document_converter import DocumentConverter, PdfFormatOption

            o = PdfPipelineOptions()
            o.accelerator_options = AcceleratorOptions(device=self.s.resolved_device, num_threads=self.s.num_threads)
            o.do_ocr = self.s.enable_ocr
            o.ocr_options.force_full_page_ocr = self.s.force_full_page_ocr
            o.do_table_structure = True
            o.table_structure_options.mode = TableFormerMode.ACCURATE
            o.table_structure_options.do_cell_matching = True
            o.do_formula_enrichment = self.s.enable_formula_latex
            o.do_code_enrichment = self.s.enable_code_enrichment
            o.do_picture_classification = self.s.enable_picture_classification
            o.do_picture_description = False  # done by our own VLM stage (structured JSON, better model)
            o.generate_page_images = True
            o.generate_picture_images = True
            o.images_scale = self.s.images_scale
            o.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
            self._conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=o)})
        return self._conv

    def convert(self, path: Path, first: int, last: int):
        return self.converter.convert(str(path), page_range=(first, last), raises_on_error=False)

    @staticmethod
    def release(res) -> None:
        """Free a conversion result: close the PDF backend, drop page images, collect garbage."""
        try:
            backend = getattr(getattr(res, "input", None), "_backend", None)
            if backend is not None:
                backend.unload()
        except Exception:  # noqa: BLE001
            pass
        try:
            for p in getattr(res, "pages", []) or []:
                for attr in ("_image_cache", "_backend"):
                    if hasattr(p, attr):
                        setattr(p, attr, {} if attr == "_image_cache" else None)
        except Exception:  # noqa: BLE001
            pass
        _gc()

    def reset(self) -> None:
        self._conv = None
        _gc()

    def docling_version(self) -> str:
        import importlib.metadata as md

        return md.version("docling")


# =================================================================================== builder
class DocumentBuilder:
    def __init__(self, s: Settings, runner: DoclingRunner):
        self.s = s
        self.runner = runner

    # ------------------------------------------------------------------ public
    def build(self, pdf: Path, out_json: Path, source_info: dict | None = None) -> dict:
        t0 = time.time()
        self.out_dir = out_json.parent
        self.assets = {k: self.out_dir / k for k in ("images", "tables", "formulas")}
        for p in self.assets.values():
            p.mkdir(parents=True, exist_ok=True)
        self.warnings: list[str] = []
        self.elements: list[dict] = []
        self.page_counters: Counter = Counter()
        self.page_conf: dict[int, dict] = {}
        self.docling_pages: dict[int, dict] = {}
        self.markdown_parts: list[str] = []
        self.ref_map: dict[tuple[int, str], str] = {}  # (batch, docling self_ref) -> element id
        self.pending_links: list[tuple[str, str, int, str]] = []  # (from_id, kind, batch, target_ref)

        sha = sha256_file(pdf)
        self.doc_id = "doc_" + sha[:12]
        self.file_name = pdf.name
        self.native: NativeDoc = read_native(pdf)
        n_pages = self.native.page_count

        if self.native.encrypted:
            self.warnings.append("PDF is encrypted; extraction may be incomplete")

        # ---------------- Phase A: Docling in page batches
        bs = max(1, self.s.page_batch_size)
        for batch_idx, first in enumerate(range(1, n_pages + 1, bs)):
            last = min(n_pages, first + bs - 1)
            log.info("  docling pages %d-%d / %d  (RAM %.1f GB)", first, last, n_pages, _rss_gb())
            res = None
            for attempt in (1, 2):
                try:
                    res = self.runner.convert(pdf, first, last)
                    break
                except Exception as e:  # noqa: BLE001
                    log.warning("docling failed on pages %d-%d (attempt %d): %r", first, last, attempt, e)
                    self.runner.reset()  # drop converter + caches, then retry once
                    if attempt == 2:
                        self.warnings.append(f"Docling failed on pages {first}-{last}: {e!r}")
            if res is None:
                continue
            if str(res.status).endswith("FAILURE"):
                self.warnings.append(f"Docling status {res.status} for pages {first}-{last}")
            for err in getattr(res, "errors", []) or []:
                self.warnings.append(f"Docling: {getattr(err, 'error_message', err)}")
            self._collect_confidence(res)
            if res.document is not None:
                self._process_docling_doc(res.document, batch_idx)
                try:
                    self.markdown_parts.append(res.document.export_to_markdown())
                except Exception:  # noqa: BLE001
                    pass
            self.runner.release(res)
            del res
            # RAM grows across batches on long books (page images / backend caches); rebuild the
            # converter before it can exhaust memory
            if _ram_fraction() > 0.80:
                log.info("  RAM at %.0f%% - rebuilding Docling converter", _ram_fraction() * 100)
                self.runner.reset()

        # ---------------- Phase B: document-wide passes
        self._mark_running_furniture(n_pages)
        self._detect_title_and_subtitle()
        self._assign_heading_levels()
        self._assign_sections_and_categories()
        self._add_native_elements()
        self._layout_pass()
        self._link_captions_footnotes_and_refs()
        self._formula_variables()
        self._table_continuations()
        self._entities()

        doc_json = self._assemble(pdf, sha, source_info or {}, time.time() - t0)
        md_path = out_json.with_suffix(".md")
        md_path.write_text("\n\n".join(self.markdown_parts), encoding="utf-8")
        return doc_json

    # ---------------------------------------------------------------- helpers
    def _new_id(self, page: int) -> str:
        self.page_counters[page] += 1
        return f"p{page}_e{self.page_counters[page]}"

    def _page_h(self, page: int) -> float:
        dp = self.docling_pages.get(page)
        return dp["height"] if dp else self.native.pages[page].height

    def _bbox(self, bb, page: int) -> list[float]:
        """docling BoundingBox -> top-left [x0, y0, x1, y1] in PDF points."""
        from docling_core.types.doc import CoordOrigin

        if bb.coord_origin == CoordOrigin.BOTTOMLEFT:
            bb = bb.to_top_left_origin(page_height=self._page_h(page))
        return rbbox([bb.l, bb.t, bb.r, bb.b])

    def _prov(self, item) -> tuple[int | None, list[float] | None, list[dict]]:
        provs = getattr(item, "prov", None) or []
        if not provs:
            return None, None, []
        regions = [{"page": p.page_no, "bbox": self._bbox(p.bbox, p.page_no),
                    "charspan": list(p.charspan) if getattr(p, "charspan", None) else None} for p in provs]
        return regions[0]["page"], regions[0]["bbox"], regions

    def _collect_confidence(self, res):
        conf = getattr(res, "confidence", None)
        if conf is None:
            return
        for pno, pc in (getattr(conf, "pages", {}) or {}).items():
            self.page_conf[int(pno)] = {
                "parse_score": _nan(getattr(pc, "parse_score", None)), "layout_score": _nan(getattr(pc, "layout_score", None)),
                "table_score": _nan(getattr(pc, "table_score", None)), "ocr_score": _nan(getattr(pc, "ocr_score", None)),
                "mean_grade": str(getattr(pc, "mean_grade", "")).split(".")[-1].lower() or None,
            }

    def _native_lines_in(self, page: int, bbox) -> list[int]:
        np_ = self.native.pages.get(page)
        if not np_ or bbox is None:
            return []
        idx = [i for i, ln in enumerate(np_.lines) if contains_point(bbox, center(ln.bbox), pad=2)]
        return sorted(idx, key=lambda i: (np_.lines[i].bbox[1], np_.lines[i].bbox[0]))

    def _font_info(self, page: int, line_ids: list[int]) -> dict | None:
        np_ = self.native.pages.get(page)
        if not np_ or not line_ids:
            return None
        spans = [np_.spans[s] for i in line_ids for s in np_.lines[i].span_ids]
        if not spans:
            return None
        w = Counter()
        for sp in spans:
            w[(sp.font, sp.size, sp.color)] += len(sp.text.strip())
        (name, size, color), _ = w.most_common(1)[0]
        total = sum(len(sp.text.strip()) for sp in spans) or 1
        frac = lambda pred: sum(len(sp.text.strip()) for sp in spans if pred(sp)) / total  # noqa: E731
        return {"name": name, "size": size, "bold": frac(lambda sp: sp.bold) > 0.6,
                "italic": frac(lambda sp: sp.italic) > 0.6, "underline": None,
                "monospace": frac(lambda sp: sp.monospace) > 0.6, "serif": frac(lambda sp: sp.serif) > 0.5,
                "color": color, "fonts_used": sorted({sp.font for sp in spans})[:6],
                "sizes_used": sorted({sp.size for sp in spans})[:6],
                "has_superscript": any(sp.superscript for sp in spans)}

    def _base(self, etype: str, page: int, bbox, regions, batch: int, item=None, **extra) -> dict:
        eid = self._new_id(page)
        el = {"id": eid, "type": etype, **extra,
              "metadata": {"doc_id": self.doc_id, "file_name": self.file_name, "page": page,
                           "pages": sorted({r["page"] for r in regions}) if regions else [page],
                           "bbox": bbox, "regions": regions if regions and len(regions) > 1 else None}}
        if item is not None:
            el["_ref"] = (batch, item.self_ref)
            self.ref_map[(batch, item.self_ref)] = eid
            layer = getattr(item, "content_layer", None)
            el["metadata"]["content_layer"] = getattr(layer, "value", str(layer)) if layer else None
        el["_lines"] = self._native_lines_in(page, bbox)
        self.elements.append(el)
        return el

    def _save_crop(self, item, doc, kind: str, eid: str):
        try:
            img = item.get_image(doc)
        except Exception as e:  # noqa: BLE001
            log.debug("crop failed %s: %s", eid, e)
            img = None
        if img is None:
            return None, None
        path = self.assets[kind] / f"{eid}.png"
        img.save(path)
        return img, f"{kind}/{eid}.png"

    # ===================================================================== Phase A
    def _process_docling_doc(self, doc, batch: int):
        from docling_core.types.doc import (CodeItem, ContentLayer, DocItemLabel, FormulaItem, GroupItem,
                                            GroupLabel, ListItem, PictureItem, TableItem, TextItem)

        for pno, pg in doc.pages.items():
            self.docling_pages[int(pno)] = {"width": r2(pg.size.width), "height": r2(pg.size.height)}

        consumed: set[str] = set()
        layers = {ContentLayer.BODY, ContentLayer.FURNITURE, ContentLayer.BACKGROUND}
        for item, level in doc.iterate_items(with_groups=True, included_content_layers=layers):
            if item.self_ref in consumed:
                continue
            try:
                if isinstance(item, GroupItem):
                    if item.label in (GroupLabel.LIST, GroupLabel.ORDERED_LIST):
                        self._list_element(item, doc, batch, consumed)
                    elif item.label == GroupLabel.INLINE:
                        self._inline_element(item, doc, batch, consumed)
                    continue
                if isinstance(item, TableItem):
                    self._table_element(item, doc, batch)
                elif isinstance(item, PictureItem):
                    self._picture_element(item, doc, batch, consumed)
                elif isinstance(item, CodeItem):
                    self._code_element(item, doc, batch)
                elif isinstance(item, FormulaItem) or getattr(item.label, "value", "") == "formula":
                    self._formula_element(item, doc, batch, inline=False)
                elif isinstance(item, ListItem):  # orphan list item
                    self._text_element(item, doc, batch)
                elif isinstance(item, TextItem):
                    self._text_element(item, doc, batch)
                else:
                    self._generic_element(item, doc, batch)
            except Exception as e:  # noqa: BLE001
                log.exception("element failed: %s", item.self_ref)
                self.warnings.append(f"Failed to process {item.self_ref} (batch {batch}): {e}")

        # caption / footnote links resolved inside this batch's namespace
        for from_id, kind, b, target_ref in self.pending_links:
            if b == batch and (b, target_ref) in self.ref_map:
                tgt = self._by_id(self.ref_map[(b, target_ref)])
                src = self._by_id(from_id)
                if tgt is None or src is None:
                    continue
                if kind == "caption":
                    tgt.setdefault("metadata", {})["caption_for"] = from_id
                    tgt["category"] = "caption"
                    src.setdefault("_captions", []).append(tgt["id"])
                elif kind == "footnote":
                    tgt["metadata"]["footnote_for"] = from_id
                    src.setdefault("_footnotes", []).append(tgt["id"])

    def _by_id(self, eid: str) -> dict | None:
        if not hasattr(self, "_id_index") or len(self._id_index) != len(self.elements):
            self._id_index = {e["id"]: e for e in self.elements}
        return self._id_index.get(eid)

    # ---------------------------------------------------------------- text
    def _text_element(self, item, doc, batch):
        label = item.label.value
        page, bbox, regions = self._prov(item)
        if page is None:
            return
        text = normalize_ws(item.text)
        if label in ("page_header", "page_footer"):
            el = self._base("header" if label == "page_header" else "footer", page, bbox, regions, batch, item,
                            content=text)
            m = PAGE_NUM_RE.match(text)
            if m:
                el["page_number_label"] = m.group(1)
            return
        if label == "footnote":
            el = self._base("footnote", page, bbox, regions, batch, item, content=text)
            m = FOOTNOTE_MARK_RE.match(text)
            el["marker"] = m.group(1) if m else None
            return
        if label in ("checkbox_selected", "checkbox_unselected"):
            self._base("form_field", page, bbox, regions, batch, item, field_type="checkbox", label=text or None,
                       value=label == "checkbox_selected", source="layout_model")
            return
        if label == "empty_value":
            return

        category = TEXT_LABEL_CATEGORY.get(label, "description")
        if category == "description" and CAPTION_START_RE.match(text) and word_count(text) < 150:
            category, label = "caption", "caption"  # "Table 1. ..." labelled as plain text by the layout model
        el = self._base("text", page, bbox, regions, batch, item, category=category, content=text)
        if text != normalize_ws(getattr(item, "orig", text) or text):
            el["content_original"] = normalize_ws(item.orig)
        if label == "section_header":
            el["heading_level"] = getattr(item, "level", 1) or 1
            el["_docling_level"] = el["heading_level"]
        if label == "list_item":
            el["marker"] = getattr(item, "marker", None)
        fmt = getattr(item, "formatting", None)
        if fmt is not None:
            el["formatting"] = {k: v for k, v in fmt.model_dump().items() if v not in (None, False)}
            if "script" in el["formatting"]:
                el["formatting"]["script"] = str(el["formatting"]["script"]).split(".")[-1].lower()
        if getattr(item, "hyperlink", None):
            el["hyperlink"] = str(item.hyperlink)
        if label == "caption":
            m = CAPTION_NUM_RE.match(text)
            if m:
                el["caption_label"] = f"{m.group(1).rstrip('.').capitalize()} {m.group(2)}"
                el["_caption_kind"] = "table" if m.group(1).lower().startswith("tab") else "figure"
                el["_caption_num"] = m.group(2)

    def _inline_element(self, group, doc, batch, consumed):
        """Paragraph with inline formulas: merge into one text element + inline formula elements."""
        from docling_core.types.doc import FormulaItem

        parts, formulas, provs, cursor = [], [], [], 0
        for ref in group.children:
            child = ref.resolve(doc)
            consumed.add(child.self_ref)
            if getattr(child, "prov", None):
                provs.append(child)
            if isinstance(child, FormulaItem):
                latex = F.compact_latex(child.text) if child.text else normalize_ws(child.orig)
                piece = f"${latex}$"
                formulas.append((child, cursor + (1 if parts else 0), latex))
            else:
                piece = normalize_ws(getattr(child, "text", ""))
            if not piece:
                continue
            if parts:
                cursor += 1
            parts.append(piece)
            cursor += len(piece)
        if not provs:
            return
        content = " ".join(parts)
        page, bbox, regions = self._prov(provs[0])
        boxes = [self._bbox(p.bbox, p.page_no) for c in provs for p in c.prov if p.page_no == page]
        bbox = rbbox(union(boxes)) if boxes else bbox
        el = self._base("text", page, bbox, regions, batch, group, category="description", content=content,
                        content_plain=" ".join(normalize_ws(getattr(c, "orig", "") or getattr(c, "text", ""))
                                               for c in (ref.resolve(doc) for ref in group.children)))
        ids = []
        for child, start, latex in formulas:
            f = self._formula_element(child, doc, batch, inline=True)
            if f:
                f["metadata"]["inline_in_element"] = el["id"]
                f["metadata"]["char_offset_start"] = start
                f["metadata"]["char_offset_end"] = start + len(latex) + 2
                ids.append(f["id"])
        if ids:
            el["contains_inline_formulas"] = ids

    def _list_element(self, group, doc, batch, consumed, level: int = 1, parent_items=None):
        from docling_core.types.doc import GroupItem, GroupLabel, ListItem

        items, provs = [], []

        def walk(g, lvl):
            for ref in g.children:
                child = ref.resolve(doc)
                consumed.add(child.self_ref)
                if isinstance(child, GroupItem) and child.label in (GroupLabel.LIST, GroupLabel.ORDERED_LIST):
                    walk(child, lvl + 1)
                elif isinstance(child, ListItem) or hasattr(child, "text"):
                    if getattr(child, "prov", None):
                        provs.append(child)
                    items.append({"index": len(items) + 1, "text": normalize_ws(getattr(child, "text", "")),
                                  "marker": getattr(child, "marker", None) or None, "level": lvl,
                                  "page": child.prov[0].page_no if getattr(child, "prov", None) else None})
                    for sub in getattr(child, "children", []) or []:  # nested lists hang under items
                        sc = sub.resolve(doc)
                        if isinstance(sc, GroupItem):
                            consumed.add(sc.self_ref)
                            walk(sc, lvl + 1)

        walk(group, level)
        if not provs:
            return
        page, _, _ = self._prov(provs[0])
        regions = [{"page": p.page_no, "bbox": self._bbox(p.bbox, p.page_no)} for c in provs for p in c.prov]
        first_page_boxes = [r["bbox"] for r in regions if r["page"] == page]
        bbox = rbbox(union(first_page_boxes))
        markers = [i["marker"] or "" for i in items]
        enumerated = group.label == GroupLabel.ORDERED_LIST or sum(bool(re.match(r"^\(?\d+|^\(?[a-z][.)]|^[ivx]+[.)]", m)) for m in markers) > len(markers) / 2
        el = self._base("list", page, bbox, regions, batch, group,
                        list_type="numbered" if enumerated else "bulleted", items=items,
                        content="\n".join(("  " * (i["level"] - 1)) + f"{i['marker'] or '-'} {i['text']}" for i in items))
        # lines of all items (first page)
        lines = set()
        for b in first_page_boxes:
            lines.update(self._native_lines_in(page, b))
        el["_lines"] = sorted(lines, key=lambda i: (self.native.pages[page].lines[i].bbox[1],
                                                    self.native.pages[page].lines[i].bbox[0]))

    # ------------------------------------------------------------ formula
    def _formula_element(self, item, doc, batch, inline: bool):
        page, bbox, regions = self._prov(item)
        if page is None:
            return None
        orig = normalize_ws(getattr(item, "orig", "") or "")
        raw_latex = (item.text or "").strip()
        has_latex = bool(raw_latex) and raw_latex != orig
        latex, eq_from_latex, notes = F.clean_latex(raw_latex, orig) if has_latex else (None, None, [])
        el = self._base("formula", page, bbox, regions, batch, item,
                        formula_type="inline" if inline else "display",
                        latex=latex, latex_raw=raw_latex if has_latex and latex != F.compact_latex(raw_latex) else None,
                        plain_text=orig or None)
        if notes:
            el["cleanup_applied"] = notes
        if latex:
            mathml = F.to_mathml(latex)
            agree = F.agreement(latex, orig)
            flags = F.quality_flags(latex, orig)
            el["mathml"] = mathml
            el["latex_valid"] = mathml is not None
            el["text_layer_agreement"] = agree
            if flags:
                el["quality_flags"] = flags
            conf = agree if agree is not None else 0.7
            if flags:
                conf = min(conf, 0.4)
            el["metadata"]["confidence"] = round(conf, 3)
            if conf < 0.6:
                self.warnings.append(f"Formula {el['id']} LaTeX may be unreliable (agreement={agree}, flags={flags})")
        else:
            el["latex_valid"] = False
            if self.s.enable_formula_latex:
                self.warnings.append(f"Formula {el['id']} has no LaTeX (model returned nothing)")
        el["equation_number"] = F.equation_number(raw_latex, orig) or eq_from_latex
        if self.s.save_formula_images and not inline:
            _, rel = self._save_crop(item, doc, "formulas", el["id"])
            el["image_file"] = rel
        return el

    # --------------------------------------------------------------- code
    def _code_element(self, item, doc, batch):
        page, bbox, regions = self._prov(item)
        if page is None:
            return
        lang = getattr(item, "code_language", None)
        lang = getattr(lang, "value", lang)
        self._base("code_block", page, bbox, regions, batch, item, language=(lang or None),
                   content=item.text, line_count=len((item.text or "").splitlines()))
        self._register_float_links(item, batch)

    # -------------------------------------------------------------- table
    def _table_element(self, item, doc, batch):
        page, bbox, regions = self._prov(item)
        if page is None:
            return
        label = item.label.value
        tbl = build_table(item, doc, self._bbox)
        caption = normalize_ws(item.caption_text(doc)) if hasattr(item, "caption_text") else ""
        etype = "table_of_contents" if label == "document_index" else "table"
        el = self._base(etype, page, bbox, regions, batch, item, title=None, caption=caption or None, **tbl)
        m = CAPTION_NUM_RE.match(caption or "")
        if m:
            el["table_number"] = m.group(2)
            el["title"] = normalize_ws(caption[m.end():]).lstrip(":.- ") or None
        el["notes"] = None
        el["metadata"]["confidence"] = (self.page_conf.get(page) or {}).get("table_score")
        if tbl["is_empty"]:
            self.warnings.append(f"Table {el['id']} has no recognised cells")
        if self.s.save_table_images:
            _, rel = self._save_crop(item, doc, "tables", el["id"])
            el["image_file"] = rel
        self._register_float_links(item, batch, el)

    def _register_float_links(self, item, batch, el=None):
        eid = self.ref_map.get((batch, item.self_ref))
        for ref in getattr(item, "captions", []) or []:
            self.pending_links.append((eid, "caption", batch, ref.cref))
        for ref in getattr(item, "footnotes", []) or []:
            self.pending_links.append((eid, "footnote", batch, ref.cref))

    # ------------------------------------------------------------ picture
    def _picture_element(self, item, doc, batch, consumed):
        page, bbox, regions = self._prov(item)
        if page is None:
            return
        # classification
        cls_label, cls_conf, top_k = None, None, []
        meta = getattr(item, "meta", None)
        if meta is not None and getattr(meta, "classification", None) is not None:
            preds = sorted(meta.classification.predictions, key=lambda p: -p.confidence)
            top_k = [{"label": p.class_name, "confidence": round(p.confidence, 4)} for p in preds[:3]]
            if preds:
                cls_label, cls_conf = preds[0].class_name, round(preds[0].confidence, 4)
        if item.label.value == "chart" and not cls_label:
            cls_label = "other_chart"
        if cls_label in CHART_CLASSES and cls_label != "flow_chart":
            etype = "chart"
        else:
            etype = PICTURE_TYPE_BY_CLASS.get(cls_label, "image")

        caption = normalize_ws(item.caption_text(doc)) if hasattr(item, "caption_text") else ""
        page_area = (self.native.pages[page].width * self.native.pages[page].height) or 1
        area_frac = area(bbox) / page_area
        el = self._base(etype, page, bbox, regions, batch, item, category=cls_label or "unknown")
        if etype == "chart":
            el["chart_type"] = (cls_label or "other_chart").replace("_chart", "").replace("_plot", "")

        img, rel = self._save_crop(item, doc, "images", el["id"])
        el["image_file"] = rel
        el["format"] = "png" if rel else None
        if img is not None:
            el["width_px"], el["height_px"] = img.size
            el["dpi"] = int(72 * self.s.images_scale)
        el["classification"] = {"label": cls_label, "confidence": cls_conf, "top_k": top_k,
                                "model": "DocumentFigureClassifier-v2.5"} if cls_label else None
        el["description"] = None          # filled by the VLM stage
        el["short_caption"] = None
        el["objects_detected"] = None

        # text inside the figure: vector text from the PDF (exact) + OCR for raster content
        pdf_texts = []
        sx = (img.size[0] / max(1e-6, bbox[2] - bbox[0])) if img is not None else 1
        sy = (img.size[1] / max(1e-6, bbox[3] - bbox[1])) if img is not None else 1
        natives = [ni for ni in self.native.pages[page].images if overlap_ratio(ni["bbox"], bbox) > 0.5]
        child_source = "docling_ocr_or_text_layer" if natives else "pdf_text_layer"
        for ref in item.children or []:
            child = ref.resolve(doc)
            consumed.add(child.self_ref)
            clabel = getattr(getattr(child, "label", None), "value", "")
            if clabel in ("caption", "footnote"):  # captions are their own elements, not text in the image
                self._text_element(child, doc, batch)
                continue
            t = normalize_ws(getattr(child, "text", ""))
            if not t or not getattr(child, "prov", None) or (caption and t == caption):
                continue
            cb = self._bbox(child.prov[0].bbox, child.prov[0].page_no)
            pdf_texts.append({"text": t, "bbox_in_image": [int((cb[0] - bbox[0]) * sx), int((cb[1] - bbox[1]) * sy),
                                                          int((cb[2] - bbox[0]) * sx), int((cb[3] - bbox[1]) * sy)],
                              "bbox": cb, "confidence": 1.0, "origin": child_source})
        ocr = ocr_image(img) if (img is not None and self.s.enable_image_ocr and area_frac >= self.s.min_image_area_fraction / 2) else []
        texts = merge_text_sources(pdf_texts, ocr)
        el["text_in_image"] = texts
        el["has_text"] = bool(texts)
        if img is not None:
            el["dominant_colors"] = dominant_colors(img)
            el["image_stats"] = image_stats(img)
            codes = decode_codes(img)
            if codes:
                el["codes"] = codes
                if etype == "barcode" or cls_label in ("qr_code", "bar_code"):
                    el["code_type"] = codes[0]["code_type"]
                    el["decoded_value"] = codes[0]["decoded_value"]
        cues = micrograph_cues([t["text"] for t in texts])
        if cues:
            el["micrograph_cues"] = cues

        el["caption"] = {"text": caption, "figure_number": None, "linked_element_id": None} if caption else None
        if caption:
            m = CAPTION_NUM_RE.match(caption)
            if m:
                el["caption"]["figure_number"] = f"{m.group(1).rstrip('.').capitalize()} {m.group(2)}"
                el["_fig_num"] = m.group(2)
        # native raster image backing this figure (resolution, colourspace, transparency)
        el["image_properties"] = {
            "is_vector_graphic": not natives,
            "embedded_images": [{k: v for k, v in ni.items() if k != "bbox"} for ni in natives][:5] or None,
            "has_transparency": any(ni.get("has_transparency") for ni in natives),
            "area_fraction_of_page": round(area_frac, 4),
        }
        el["metadata"].update({
            "is_inline": area_frac < 0.01 and (bbox[3] - bbox[1]) < 30,
            "is_background": area_frac > 0.85,
            "is_logo": cls_label == "logo", "rotation": self.native.pages[page].rotation,
            "describe_with_vlm": bool(rel) and area_frac >= self.s.min_image_area_fraction,
        })
        self._register_float_links(item, batch, el)

    def _generic_element(self, item, doc, batch):
        page, bbox, regions = self._prov(item)
        if page is None:
            return
        label = item.label.value
        etype = {"key_value_region": "key_value", "form": "form", "field_region": "form", "field_item": "form_field",
                 "field_key": "form_field", "field_value": "form_field", "field_heading": "text",
                 "field_hint": "text"}.get(label, label)
        text = normalize_ws(getattr(item, "text", "") or "")
        if not text:
            np_ = self.native.pages.get(page)
            if np_:
                text = normalize_ws(" ".join(np_.lines[i].text for i in self._native_lines_in(page, bbox)))
        self._base(etype, page, bbox, regions, batch, item, docling_label=label, content=text or None)

    # ===================================================================== Phase B
    def _mark_running_furniture(self, n_pages):
        """Headers/footers that repeat across pages (digits normalised, e.g. 'Page 3 of 12')."""
        norm = lambda t: re.sub(r"\d+", "#", (t or "").lower()).strip()  # noqa: E731
        groups = defaultdict(list)
        for e in self.elements:
            if e["type"] in ("header", "footer"):
                groups[(e["type"], norm(e.get("content")))].append(e)
        for (_, _), els in groups.items():
            pages = sorted({e["metadata"]["page"] for e in els})
            for e in els:
                e["metadata"]["region"] = "top" if e["type"] == "header" else "bottom"
                e["metadata"]["repeats_on_pages"] = "all" if len(pages) == n_pages and n_pages > 1 else pages

        # body text that is really a running header (e.g. journal name on every page, top 7 % of page)
        pos_groups = defaultdict(list)
        for e in self.elements:
            if e["type"] == "text" and e.get("category") == "description" and e["metadata"]["bbox"]:
                h = self.native.pages[e["metadata"]["page"]].height
                y = e["metadata"]["bbox"][1] / h
                if y < 0.07 or y > 0.93:
                    pos_groups[norm(e["content"])].append(e)
        for key, els in pos_groups.items():
            pages = {e["metadata"]["page"] for e in els}
            if len(pages) >= max(3, n_pages * 0.3) and len(key) < 120:
                for e in els:
                    y = e["metadata"]["bbox"][1] / self.native.pages[e["metadata"]["page"]].height
                    e["type"] = "header" if y < 0.5 else "footer"
                    e.pop("category", None)
                    e["metadata"]["repeats_on_pages"] = sorted(pages)
                    e["metadata"]["detected_by"] = "repetition_heuristic"

    def _font_size(self, e) -> float | None:
        fi = self._font_info(e["metadata"]["page"], e.get("_lines", []))
        return fi["size"] if fi else None

    def _detect_title_and_subtitle(self):
        first_pages = [e for e in self.elements if e["metadata"]["page"] <= 2 and e["type"] == "text"]
        if not first_pages:
            return
        titles = [e for e in first_pages if e.get("category") == "title"]
        if not titles:
            cands = [e for e in first_pages if e["metadata"]["page"] == 1 and e.get("category") in ("heading", "description")
                     and 2 <= word_count(e["content"]) <= 40]
            sized = [(self._font_size(e) or 0, -i, e) for i, e in enumerate(cands)]
            body_sizes = [self._font_size(e) or 0 for e in self.elements[:200] if e.get("category") == "description"]
            body = median([s for s in body_sizes if s]) if any(body_sizes) else 10
            if sized:
                size, _, best = max(sized)
                if size >= body * 1.25 or best.get("category") == "heading":
                    best["category"] = "title"
                    best.pop("heading_level", None)
                    best["metadata"]["detected_by"] = "largest_font_on_first_page"
                    titles = [best]
        if not titles:
            return
        title = titles[0]
        tsize = self._font_size(title) or 0
        # subtitle: the element right after the title, larger than body text, not an author line
        idx = self.elements.index(title)
        for nxt in self.elements[idx + 1: idx + 3]:
            if nxt["type"] != "text" or nxt["metadata"]["page"] != title["metadata"]["page"]:
                break
            s = self._font_size(nxt) or 0
            txt = nxt["content"]
            if (nxt.get("category") in ("heading", "description") and 2 <= word_count(txt) <= 30
                    and s and tsize and s < tsize and s >= 11 and not EMAIL_RE.search(txt)
                    and not AFFIL_RE.search(txt) and not re.match(r"^abstract", txt, re.I)
                    and not re.search(r"\b[A-Z]\.\s?[A-Z][a-z]+", txt)):
                nxt["category"] = "subtitle"
                nxt.pop("heading_level", None)
                nxt["metadata"]["parent_id_hint"] = title["id"]
            break

    def _merge_split_headings(self):
        """'Chapter 3' / 'Part II' followed by its title as a separate heading -> one heading."""
        drop = set()
        for a, b in zip(self.elements, self.elements[1:]):
            if (a.get("category") in ("heading", "subheading") and b.get("category") in ("heading", "subheading")
                    and a["metadata"]["page"] == b["metadata"]["page"] and a["id"] not in drop
                    and re.fullmatch(r"(chapter|part|unit|module|appendix)\s+([\dIVXLC]+|[A-Z])\.?", a["content"].strip(), re.I)):
                a["content"] = f"{a['content'].strip()}: {b['content'].strip()}"
                a["metadata"]["bbox"] = rbbox(union([a["metadata"]["bbox"], b["metadata"]["bbox"]]))
                a["_lines"] = sorted(set(a.get("_lines", [])) | set(b.get("_lines", [])))
                a["metadata"]["merged_from"] = [a["id"], b["id"]]
                drop.add(b["id"])
        # junk headings: OCR fragments from decorative chapter-title images (mostly 1-2 letter tokens)
        for e in self.elements:
            if e.get("category") in ("heading", "subheading"):
                toks = re.findall(r"[A-Za-z]+", e["content"])
                if len(toks) >= 3 and sum(len(t) <= 2 for t in toks) / len(toks) > 0.5:
                    e["category"] = "description"
                    e["metadata"]["demoted_heading"] = "ocr_fragment"
                    e.pop("heading_level", None)
        self.elements = [e for e in self.elements if e["id"] not in drop]
        self._id_index = {e["id"]: e for e in self.elements}

    def _assign_heading_levels(self):
        self._merge_split_headings()
        heads = [e for e in self.elements if e.get("category") == "heading"]
        if not heads:
            return
        sizes = sorted({round(self._font_size(e) or 0) for e in heads if self._font_size(e)}, reverse=True)
        size_rank = {s: i + 1 for i, s in enumerate(sizes[:6])}
        for e in heads:
            txt = e["content"]
            m = HEADING_NUM_RE.match(txt)
            level, how = None, None
            if m and re.match(r"^\d+(?:\.\d+)*$", m.group(1)):
                level, how = m.group(1).count(".") + 1, "numbering"
                e["section_number"] = m.group(1)
            elif m and re.match(r"^[IVXLC]+$", m.group(1)):
                level, how = 1, "numbering"
                e["section_number"] = m.group(1)
            elif re.match(r"^(chapter|part)\b", txt, re.I):
                level, how = 1, "chapter_keyword"
            else:
                fs = self._font_size(e)
                if fs and len(size_rank) > 1:
                    level, how = size_rank.get(round(fs), len(size_rank)), "font_size_rank"
                else:
                    level, how = e.get("_docling_level", 1), "docling"
            e["heading_level"] = max(1, min(6, level))
            e["metadata"]["heading_level_source"] = how
            if e["heading_level"] >= 2:
                e["category"] = "subheading"

    def _assign_sections_and_categories(self):
        stack: list[tuple[int, str, str]] = []  # (level, title, id)
        title_el = next((e for e in self.elements if e.get("category") == "title"), None)
        seen_body_heading = False
        role = None
        for e in self.elements:
            cat = e.get("category")
            if e["type"] in ("header", "footer"):
                e["metadata"]["section_path"] = []
                continue
            if cat in ("heading", "subheading"):
                lvl = e["heading_level"]
                while stack and stack[-1][0] >= lvl:
                    stack.pop()
                e["metadata"]["parent_id"] = stack[-1][2] if stack else (title_el["id"] if title_el else None)
                stack.append((lvl, e["content"], e["id"]))
                e["metadata"]["section_path"] = [s[1] for s in stack]
                seen_body_heading = True
                top = re.sub(r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)\.?\s+", "", stack[0][1]).strip().lower() if stack else ""
                role = next((r for r, rx in SECTION_ROLES if re.search(rx, top, re.I)), None)
                cur = re.sub(r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)\.?\s+", "", e["content"]).strip().lower()
                sub_role = next((r for r, rx in SECTION_ROLES if re.search(rx, cur, re.I)), None)
                if sub_role in ("references", "acknowledgements", "abstract", "appendix"):
                    role = sub_role
                e["section_role"] = role
                continue
            e["metadata"]["section_path"] = [s[1] for s in stack]
            e["metadata"]["parent_id"] = stack[-1][2] if stack else (title_el["id"] if title_el and e is not title_el else None)
            if role:
                e["section_role"] = role
            if e["type"] != "text":
                continue
            txt = e["content"]
            # ---- refine text categories
            if cat == "description":
                if (km := KEYWORDS_RE.match(txt)):
                    e["category"] = "keywords"
                    e["keywords"] = [k.strip(" .") for k in re.split(r"[;,·•]", km.group(1)) if k.strip(" .")]
                elif role == "references":
                    e["category"] = "reference"
                elif role == "abstract":
                    e["category"] = "abstract"
                elif role == "acknowledgements":
                    e["category"] = "acknowledgement"
                elif re.match(r"^abstract\b[\s.:\u2014-]*", txt, re.I) and e["metadata"]["page"] <= 2:
                    e["category"] = "abstract"
                elif not seen_body_heading and e["metadata"]["page"] <= 2 and e is not title_el:
                    if EMAIL_RE.search(txt) or "orcid" in txt.lower():
                        e["category"] = "author_info"
                    elif AFFIL_RE.search(txt) and word_count(txt) < 60:
                        e["category"] = "affiliation"
                    elif word_count(txt) < 40 and len(re.findall(r"\b[A-Z][a-z]+", txt)) >= 2 and (
                            txt.count(",") >= 1 or " and " in txt) and not txt.endswith("."):
                        e["category"] = "authors"
                if e["category"] == "description":
                    if (q := QUOTE_RE.match(txt)):
                        e["type"], e["content"] = "quote", q.group(1).strip()
                        e["attribution"] = q.group(2).strip() if q.group(2) else None
                        e.pop("category", None)
                    elif (rp := ROLE_PREFIX_RE.match(txt)):
                        name = rp.group(1).lower()
                        e["category"] = ROLE_NAMES.get(name, name.replace(" ", "_"))
            if e.get("category") == "reference":
                e["parsed_reference"] = parse_reference(txt)
                if (m := e["parsed_reference"].get("marker")):
                    e["marker"] = m
            elif e.get("category") == "list_item" and role == "references":
                e["category"] = "reference"
                e["parsed_reference"] = parse_reference(txt)
        # lists inside references section -> parsed references per item
        for e in self.elements:
            if e["type"] == "list" and e.get("section_role") == "references":
                for it in e["items"]:
                    it["parsed_reference"] = parse_reference(((it.get("marker") or "") + " " + it["text"]).strip())
                e["category"] = "reference_list"

    # ---------------------------------------------------------------- native-only elements
    def _add_native_elements(self):
        pages_with_wm = defaultdict(list)

        def boxes_on(e, pno):  # an element may continue over several pages (e.g. a reference list)
            regs = e["metadata"].get("regions") or []
            out = [r["bbox"] for r in regs if r["page"] == pno]
            if not out and e["metadata"]["page"] == pno and e["metadata"]["bbox"]:
                out = [e["metadata"]["bbox"]]
            return out

        def best_overlap(b, e, pno):
            return max((overlap_ratio(b, x) for x in boxes_on(e, pno)), default=0.0)

        for pno, np_ in self.native.pages.items():
            text_els = [e for e in self.elements if pno in e["metadata"].get("pages", [e["metadata"]["page"]])
                        and e["type"] in ("text", "list", "footnote", "quote")]
            # hyperlinks
            for ln in np_.links:
                host = max(text_els, key=lambda e: best_overlap(ln["bbox"], e, pno), default=None)
                hid = host["id"] if host and best_overlap(ln["bbox"], host, pno) > 0.3 else None
                link = {"text": ln["text"] or None, "url": ln["url"], "link_type": ln["link_type"],
                        "target_page": ln["target_page"]}
                if hid:
                    host["has_hyperlinks"] = True
                    host.setdefault("hyperlinks", []).append(link)
                # links inside a paragraph / list stay attached to it (host["hyperlinks"]);
                # only links that sit outside any text block become standalone elements
                if not hid:
                    el = self._native_el("link", pno, ln["bbox"], **link)
                    el["metadata"]["in_element_id"] = hid
                    if hid:
                        link["link_id"] = el["id"]
            # annotations
            for an in np_.annotations:
                linked = max(self.elements, key=lambda e: overlap_ratio(an["bbox"], e["metadata"]["bbox"] or [0, 0, 0, 0]) if e["metadata"]["page"] == pno else -1, default=None)
                el = self._native_el("annotation", pno, an["bbox"], **{k: v for k, v in an.items() if k != "bbox"})
                if linked and linked["metadata"]["page"] == pno and overlap_ratio(an["bbox"], linked["metadata"]["bbox"] or [0, 0, 0, 0]) > 0:
                    el["metadata"]["linked_element_id"] = linked["id"]
            # form fields
            for w in np_.widgets:
                self._native_el("form_field", pno, w["bbox"], source="pdf_acroform", **{k: v for k, v in w.items() if k != "bbox"})
            for wm in np_.watermark_candidates:
                pages_with_wm[wm["text"].lower()].append((pno, wm))
            # sidebars / call-out boxes
            for bi, box in enumerate(np_.boxes):
                if (box["bbox"][2] - box["bbox"][0]) < 0.15 * np_.width:
                    continue
                if any(e["type"] in ("table", "image", "chart") and overlap_ratio(box["bbox"], e["metadata"]["bbox"] or [0, 0, 0, 0]) > 0.4
                       for e in self.elements if e["metadata"]["page"] == pno):
                    continue
                members = [e for e in text_els if e["metadata"]["bbox"] and overlap_ratio(e["metadata"]["bbox"], box["bbox"]) > 0.8]
                if not members or area(box["bbox"]) > 0.6 * np_.width * np_.height:
                    for m in members:
                        m["metadata"]["background_color"] = box["background_color"]
                    continue
                first = members[0]
                title = first["content"] if first.get("category") in ("heading", "subheading") or word_count(first.get("content", "")) <= 6 else None
                el = self._native_el("sidebar", pno, box["bbox"], title=title,
                                     content="\n".join(m.get("content", "") for m in members),
                                     member_ids=[m["id"] for m in members])
                el["metadata"]["background_color"] = box["background_color"]
                for m in members:
                    m["metadata"]["in_sidebar"] = el["id"]
                    m["metadata"]["background_color"] = box["background_color"]
        # watermarks: one element per distinct text, listing all pages
        n = self.native.page_count
        for key, hits in pages_with_wm.items():
            pages = sorted({p for p, _ in hits})
            wm = hits[0][1]
            if len(key) < 2 or (len(pages) < max(2, 0.5 * n) and abs(wm["rotation"]) < 10 and wm["opacity"] >= 0.8):
                continue
            el = self._native_el("watermark", pages[0], wm["bbox"], content=wm["text"])
            el["metadata"].update({"applies_to_pages": "all" if len(pages) == n else pages, "opacity": wm["opacity"],
                                   "rotation": wm["rotation"], "font_size": wm["font_size"], "color": wm["color"]})
            for e in self.elements:  # Docling may have turned the watermark into body text
                if e["type"] == "text" and normalize_ws(e.get("content", "")).lower() == key:
                    e["type"] = "watermark_text"
                    e["metadata"]["watermark_id"] = el["id"]

    def _native_el(self, etype, page, bbox, **fields) -> dict:
        eid = self._new_id(page)
        el = {"id": eid, "type": etype, **fields,
              "metadata": {"doc_id": self.doc_id, "file_name": self.file_name, "page": page, "pages": [page],
                           "bbox": rbbox(bbox), "source": "pdf_native"}}
        el["_lines"] = []
        self.elements.append(el)
        return el

    # -------------------------------------------------------------------- layout
    def _layout_pass(self):
        by_page = defaultdict(list)
        for i, e in enumerate(self.elements):
            e["_order"] = i
            by_page[e["metadata"]["page"]].append(e)
        self.page_layouts = {}
        global_order = 0
        for pno in sorted(by_page):
            els = by_page[pno]
            np_ = self.native.pages.get(pno)
            W = np_.width if np_ else self.docling_pages.get(pno, {}).get("width", 612)
            H = np_.height if np_ else self.docling_pages.get(pno, {}).get("height", 792)
            # reading order: running header first, body in Docling order, footer/footnotes last
            rank = {"header": 0, "footer": 3, "watermark": 4, "link": 5, "annotation": 5, "form_field": 2}
            els.sort(key=lambda e: (rank.get(e["type"], 1 if e["type"] != "footnote" else 2), e["_order"]))

            # columns: merge x-intervals of narrow body blocks
            body = [e for e in els if e["type"] in ("text", "list", "formula", "table", "image", "chart", "code_block", "quote")
                    and e["metadata"]["bbox"] and e["metadata"].get("content_layer") != "furniture"]
            # centred blocks (title, author lines) straddle the gutter, so they don't define columns
            narrow = [b for b in (e["metadata"]["bbox"] for e in body)
                      if (b[2] - b[0]) < 0.45 * W and abs((b[0] + b[2]) / 2 - W / 2) > 0.08 * W]
            bands = []
            for b in sorted(narrow, key=lambda b: b[0]):
                if bands and b[0] < bands[-1][1] - 5:
                    bands[-1][1] = max(bands[-1][1], b[2])
                else:
                    bands.append([b[0], b[2]])
            # a real column holds several blocks or a tall block (filters stray narrow headings/labels)
            bands = [bd for bd in bands
                     if sum(1 for b in narrow if bd[0] - 1 <= b[0] <= bd[1]) >= 2
                     or sum(b[3] - b[1] for b in narrow if bd[0] - 1 <= b[0] <= bd[1]) > 0.15 * H] or bands[:1]
            n_cols = max(1, min(4, len(bands)))
            self.page_layouts[pno] = {"columns": n_cols, "orientation": "landscape" if W > H else "portrait",
                                      "column_bands": [[r2(a), r2(b)] for a, b in bands] if n_cols > 1 else None}

            line_no, para_idx, ro = 0, 0, 0
            claimed: set[int] = set()
            for e in els:
                md = e["metadata"]
                bb = md.get("bbox")
                md["reading_order"] = ro
                md["global_order"] = global_order
                ro += 1
                global_order += 1
                if not bb:
                    continue
                md["bbox_normalized"] = [r2(bb[0] / W), r2(bb[1] / H), r2(bb[2] / W), r2(bb[3] / H)]
                md["page_width"], md["page_height"] = W, H
                cx, cy = center(bb)
                width = bb[2] - bb[0]
                md["side"] = ("full_width" if width > 0.6 * W else "left" if cx < 0.4 * W else
                              "right" if cx > 0.6 * W else "center")
                md["region"] = md.get("region") or ("top" if cy < H / 3 else "bottom" if cy > 2 * H / 3 else "middle")
                if n_cols > 1 and width < 0.6 * W:
                    md["column"] = next((i + 1 for i, (a, b) in enumerate(bands) if a - 5 <= cx <= b + 5), None)
                else:
                    md["column"] = None if n_cols > 1 else 1
                # line numbers (reading-order line count on the page)
                lines = [i for i in e.get("_lines", []) if i not in claimed]
                if lines:
                    claimed.update(lines)
                    md["line_start"], md["line_end"] = line_no + 1, line_no + len(lines)
                    md["line_count"] = len(lines)
                    line_no += len(lines)
                elif e["type"] in ("text", "list", "formula", "code_block", "quote", "footnote", "header", "footer") and (np_ is None or not np_.lines):
                    est = max(1, round((bb[3] - bb[1]) / 12))
                    md["line_start"], md["line_end"], md["line_count"] = line_no + 1, line_no + est, est
                    md["line_numbers_estimated"] = True
                    line_no += est
                if e["type"] in ("text", "list", "formula", "code_block", "quote") and e.get("formula_type") != "inline":
                    para_idx += 1
                    md["paragraph_index"] = para_idx
                # typography
                if e["type"] in ("text", "list", "footnote", "header", "footer", "quote", "code_block", "caption"):
                    fi = self._font_info(pno, e.get("_lines", []))
                    if fi:
                        md["font"] = fi
                    md["alignment"] = self._alignment(pno, e.get("_lines", []), bb, W)
                    if e.get("content"):
                        md["word_count"] = word_count(e["content"])
                        md["char_count"] = len(e["content"])
                # confidence: page-level layout score unless the element has its own
                if md.get("confidence") is None:
                    pc = self.page_conf.get(pno) or {}
                    md["confidence"] = pc.get("layout_score")
                    md["confidence_source"] = "docling_page_layout_score" if pc.get("layout_score") is not None else None
            # ids in reading order
            for e in els:
                e.pop("_order", None)
        # keep the global list in final reading order
        self.elements.sort(key=lambda e: e["metadata"].get("global_order", 0))
        self._id_index = {e["id"]: e for e in self.elements}

    def _alignment(self, pno, line_ids, bb, W) -> str | None:
        np_ = self.native.pages.get(pno)
        if not np_ or not line_ids:
            return None
        # merge fragments on the same baseline (PyMuPDF splits visual lines at inline math / font changes)
        visual: list[list[float]] = []
        for b in sorted((np_.lines[i].bbox for i in line_ids), key=lambda b: ((b[1] + b[3]) / 2, b[0])):
            if visual and abs((b[1] + b[3]) / 2 - (visual[-1][1] + visual[-1][3]) / 2) < 3:
                v = visual[-1]
                visual[-1] = [min(v[0], b[0]), min(v[1], b[1]), max(v[2], b[2]), max(v[3], b[3])]
            else:
                visual.append(list(b))
        lines = visual
        if len(lines) == 1:
            l = lines[0]
            mid = (l[0] + l[2]) / 2
            return "center" if abs(mid - W / 2) < 8 and (l[2] - l[0]) < 0.7 * W else "left"
        # first line may be indented, last line is usually ragged
        lefts = [l[0] for l in (lines[1:] if len(lines) > 2 else lines)]
        rights = [l[2] for l in lines[:-1]]
        mids = [(l[0] + l[2]) / 2 for l in lines]
        sl, sm = pstdev(lefts), pstdev(mids)
        sr = pstdev(rights) if len(rights) > 1 else None
        if sl < 2 and sr is not None and sr < 2:
            return "justified"
        if sl < 2:
            return "left"
        if sr is not None and sr < 2:
            return "right"
        if sm < 3:
            return "center"
        return "left"

    # ------------------------------------------------------------- linking
    def _orphan_captions(self):
        """Captions Docling did not attach (e.g. 'Table 1.' printed on the previous page): attach by
        proximity to the nearest following (else preceding) uncaptioned table / figure within one page."""
        for i, c in enumerate(self.elements):
            if c.get("category") != "caption" or c["metadata"].get("caption_for") or not c.get("_caption_kind"):
                continue
            kinds = ("table",) if c["_caption_kind"] == "table" else ("image", "chart")
            pg = c["metadata"]["page"]

            def free(e):
                if e["type"] not in kinds or abs(e["metadata"]["page"] - pg) > 1:
                    return False
                return not (e.get("caption") if e["type"] == "table" else (e.get("caption") or {}).get("text"))

            after = next((e for e in self.elements[i + 1:] if free(e)), None)
            before = next((e for e in reversed(self.elements[:i]) if free(e)), None)
            tgt = after or before
            if not tgt:
                continue
            c["metadata"]["caption_for"] = tgt["id"]
            c["metadata"]["caption_linked_by"] = "proximity"
            if tgt["type"] == "table":
                tgt["caption"], tgt["caption_element_id"] = c["content"], c["id"]
                tgt["table_number"] = c["_caption_num"]
                m = CAPTION_NUM_RE.match(c["content"])
                tgt["title"] = normalize_ws(c["content"][m.end():]).lstrip(":.- ") or None
            else:
                tgt["caption"] = {"text": c["content"], "figure_number": c.get("caption_label"), "linked_element_id": c["id"]}
                tgt["_fig_num"] = c["_caption_num"]

    def _link_captions_footnotes_and_refs(self):
        self._orphan_captions()
        figs, tabs, eqs, secs, refs = {}, {}, {}, {}, {}
        for e in self.elements:
            if e["type"] in ("image", "chart") and e.get("_fig_num"):
                figs.setdefault(e["_fig_num"], e["id"])
            if e["type"] == "table" and e.get("table_number"):
                tabs.setdefault(e["table_number"], e["id"])
            if e["type"] == "formula" and e.get("equation_number"):
                eqs.setdefault(e["equation_number"].strip("()"), e["id"])
            if e.get("section_number"):
                secs.setdefault(e["section_number"], e["id"])
            if e.get("category") == "reference" and e.get("marker"):
                refs.setdefault(e["marker"], e["id"])
            if e["type"] == "list" and e.get("category") == "reference_list":
                for it in e["items"]:
                    mk = (it.get("parsed_reference") or {}).get("marker")
                    if mk:
                        refs.setdefault(mk, f"{e['id']}#item{it['index']}")
        # captions declared by Docling -> caption elements; numbered captions without docling link
        for e in self.elements:
            for cid in e.pop("_captions", []) or []:
                c = self._by_id(cid)
                if c and e.get("caption") and isinstance(e["caption"], dict):
                    e["caption"]["linked_element_id"] = cid
                elif c and e["type"] == "table":
                    e["caption_element_id"] = cid
            for fid in e.pop("_footnotes", []) or []:
                e.setdefault("footnote_ids", []).append(fid)
            if e["type"] == "table" and e.get("footnote_ids"):
                e["notes"] = " ".join(filter(None, ((self._by_id(f) or {}).get("content") for f in e["footnote_ids"]))) or None
        for e in self.elements:
            if e.get("category") == "caption" and not e["metadata"].get("caption_for") and e.get("_caption_num"):
                tgt = (tabs if e.get("_caption_kind") == "table" else figs).get(e["_caption_num"])
                if tgt:
                    e["metadata"]["caption_for"] = tgt
        # footnotes: find the superscript marker on the same page
        for e in self.elements:
            if e["type"] != "footnote" or not e.get("marker"):
                continue
            pno = e["metadata"]["page"]
            np_ = self.native.pages.get(pno)
            if not np_:
                continue
            sup = [s for s in np_.spans if s.superscript and s.text.strip() == e["marker"]]
            for s in sup:
                host = next((x for x in self.elements if x["metadata"]["page"] == pno and x["type"] in ("text", "list", "table")
                             and x["id"] != e["id"] and x["metadata"]["bbox"] and contains_point(x["metadata"]["bbox"], center(s.bbox))), None)
                if host:
                    e["metadata"]["references_element_id"] = host["id"]
                    host.setdefault("footnote_ids", []).append(e["id"])
                    break
        # in-text cross references
        for e in self.elements:
            txt = e.get("content") if e["type"] in ("text", "list", "quote", "footnote", "sidebar") else None
            if not txt or e.get("category") in ("reference", "caption"):
                continue
            xr = []
            for m in FIG_REF_RE.finditer(txt):
                xr.append({"kind": "figure", "label": m.group(1) + (m.group(2) or ""), "text": m.group(0), "target_id": figs.get(m.group(1))})
            for m in TAB_REF_RE.finditer(txt):
                xr.append({"kind": "table", "label": m.group(1), "text": m.group(0), "target_id": tabs.get(m.group(1))})
            for m in EQ_REF_RE.finditer(txt):
                xr.append({"kind": "equation", "label": m.group(1), "text": m.group(0), "target_id": eqs.get(m.group(1))})
            for m in SEC_REF_RE.finditer(txt):
                xr.append({"kind": "section", "label": m.group(1), "text": m.group(0), "target_id": secs.get(m.group(1))})
            if xr:
                e["cross_references"] = xr
            cites = []
            for m in CITATION_RE.finditer(txt):
                for n in expand_citation(m.group(1)):
                    cites.append({"marker": n, "target_id": refs.get(n)})
            if cites:
                e["citations"] = cites

    def _formula_variables(self):
        for i, e in enumerate(self.elements):
            if e["type"] != "formula" or not e.get("latex"):
                continue
            defs = []
            # "where ..." usually follows the display formula (same or next text block)
            for nxt in self.elements[i + 1: i + 3]:
                if nxt["type"] == "text" and nxt.get("content"):
                    defs = F.parse_where_clause(nxt["content"])
                    if defs:
                        e["metadata"]["variables_defined_in"] = nxt["id"]
                        break
                if nxt["type"] not in ("formula",):
                    break
            e["variables"] = F.build_variables(e["latex"], defs)
            # what text mentions this equation?
            if e.get("equation_number"):
                e["metadata"]["referenced_in_text"] = [x["id"] for x in self.elements
                                                       if any(c.get("target_id") == e["id"] for c in x.get("cross_references", []))]

    def _table_continuations(self):
        tables = [e for e in self.elements if e["type"] == "table"]
        for prev, cur in zip(tables, tables[1:]):
            p, c = prev["metadata"], cur["metadata"]
            if (c["page"] == p["page"] + 1 and cur["num_columns"] == prev["num_columns"] and not cur.get("caption")
                    and c["bbox"] and p["bbox"] and c["bbox"][1] < 0.3 * c.get("page_height", 800)
                    and p["bbox"][3] > 0.6 * p.get("page_height", 800)):
                cur["continues_from_previous_page"] = True
                cur["continuation_of"] = prev["id"]
                prev["continues_on_next_page"] = True
                if not cur["has_header_row"] or cur["headers"] == prev["headers"]:
                    cur["inherited_headers"] = prev["headers"]
        for t in tables:
            t.setdefault("continues_from_previous_page", False)
            t.setdefault("continues_on_next_page", False)

    def _entities(self):
        for e in self.elements:
            if e["type"] in ("text", "list", "quote", "footnote", "sidebar", "code_block") and e.get("content"):
                txt = e["content"]
            elif e["type"] == "table":
                txt = " ".join(filter(None, [e.get("caption"), e.get("csv")]))
            elif e["type"] in ("image", "chart"):
                txt = " ".join(filter(None, [(e.get("caption") or {}).get("text")] + [t["text"] for t in e.get("text_in_image", [])]))
            else:
                continue
            ents = extract_entities(txt)
            if ents:
                e["entities"] = ents
            if len(txt) >= 200 and e["type"] in ("text", "list", "quote"):
                e["metadata"]["language"] = _lang(txt)

    # ================================================================== assemble
    def _assemble(self, pdf: Path, sha: str, source: dict, elapsed: float) -> dict:
        n = self.native.page_count
        full_text = " ".join(e.get("content", "") for e in self.elements if e["type"] == "text")[:20000]
        doc_lang = _lang(full_text) if full_text else None

        pages_out = []
        by_page = defaultdict(list)
        for e in self.elements:
            for k in [k for k in e if k.startswith("_")]:
                e.pop(k)
            md = e["metadata"]
            if e["type"] in ("text", "list", "quote"):
                md.setdefault("language", doc_lang)
            md = {k: v for k, v in md.items() if v is not None}
            e["metadata"] = md
            by_page[md["page"]].append(e)
        for pno in range(1, n + 1):
            np_: NativePage = self.native.pages[pno]
            pc = self.page_conf.get(pno)
            pages_out.append({
                "page_number": pno, "page_label": np_.label,
                "width": np_.width, "height": np_.height, "unit": "pt", "rotation": np_.rotation,
                "layout": self.page_layouts.get(pno, {"columns": 1, "orientation": "landscape" if np_.width > np_.height else "portrait"}),
                "is_scanned": np_.is_scanned, "ocr_used": bool(self.s.enable_ocr and (np_.is_scanned or self.s.force_full_page_ocr)),
                "char_count": np_.char_count, "native_line_count": len(np_.lines),
                "num_embedded_images": len(np_.images), "num_vector_drawings": np_.num_drawings,
                "image_coverage": round(np_.image_coverage, 3),
                "confidence": pc, "processed_by_docling": pno in self.docling_pages,
                "elements": by_page.get(pno, []),
            })

        rels = []
        for e in self.elements:
            md = e["metadata"]
            if md.get("caption_for"):
                rels.append({"from": md["caption_for"], "to": e["id"], "type": "has_caption"})
            if md.get("footnote_for"):
                rels.append({"from": e["id"], "to": md["footnote_for"], "type": "footnote_of"})
            if md.get("references_element_id"):
                rels.append({"from": e["id"], "to": md["references_element_id"], "type": "footnote_of"})
            for fid in e.get("contains_inline_formulas", []):
                rels.append({"from": e["id"], "to": fid, "type": "contains_inline_formula"})
            if md.get("linked_element_id") and e["type"] == "annotation":
                rels.append({"from": e["id"], "to": md["linked_element_id"], "type": "annotates"})
            if md.get("in_element_id") and e["type"] == "link":
                rels.append({"from": md["in_element_id"], "to": e["id"], "type": "has_link"})
            if e.get("continuation_of"):
                rels.append({"from": e["continuation_of"], "to": e["id"], "type": "continues_on"})
            if md.get("variables_defined_in"):
                rels.append({"from": e["id"], "to": md["variables_defined_in"], "type": "variables_defined_in"})
            if md.get("in_sidebar"):
                rels.append({"from": md["in_sidebar"], "to": e["id"], "type": "contains"})
            if md.get("parent_id"):
                rels.append({"from": md["parent_id"], "to": e["id"], "type": "parent_of"})
            for xr in e.get("cross_references", []):
                if xr.get("target_id"):
                    rels.append({"from": e["id"], "to": xr["target_id"], "type": f"refers_to_{xr['kind']}"})
            for c in e.get("citations", []):
                if c.get("target_id"):
                    rels.append({"from": e["id"], "to": c["target_id"], "type": "cites"})

        counts = Counter(e["type"] for e in self.elements)
        cats = Counter(e.get("category") for e in self.elements if e.get("category"))
        title = next((e["content"] for e in self.elements if e.get("category") == "title"), None)
        subtitle = next((e["content"] for e in self.elements if e.get("category") == "subtitle"), None)
        abstract = " ".join(e["content"] for e in self.elements if e.get("category") == "abstract") or None
        keywords = next((e["keywords"] for e in self.elements if e.get("category") == "keywords"), None)
        first_text = " ".join(e.get("content", "") for e in self.elements if e["metadata"]["page"] <= 2)
        doi = DOI_RE.search(first_text)
        heading_toc = [{"title": e["content"], "page": e["metadata"]["page"], "level": e["heading_level"],
                        "element_id": e["id"]} for e in self.elements if e.get("category") in ("heading", "subheading")]
        # PDF bookmarks are often missing or junk (e.g. a temp file name); prefer detected headings then
        native_ok = len(self.native.toc) >= max(3, len(heading_toc) // 4) and not any(
            re.search(r"\.pdf\b|^tmp\.", t["title"], re.I) for t in self.native.toc)
        toc = self.native.toc if native_ok else heading_toc
        props = dict(self.native.metadata)
        if not props.get("keywords") and keywords:
            props["keywords"] = keywords
        all_ents = [e["entities"] for e in self.elements if e.get("entities")]
        fonts = Counter()
        for np_ in self.native.pages.values():
            for sp in np_.spans:
                fonts[sp.font] += len(sp.text)

        return {
            "schema_version": "1.0",
            "document": {
                "id": self.doc_id, "file_name": pdf.name, "relative_path": source.get("relative_path"),
                "category": source.get("category"), "file_size_bytes": pdf.stat().st_size, "sha256": sha,
                "total_pages": n, "language": doc_lang,
                "is_scanned": sum(p.is_scanned for p in self.native.pages.values()) > n / 2,
                "scanned_pages": [p for p, np_ in self.native.pages.items() if np_.is_scanned],
                "ocr_used": self.s.enable_ocr and any(p.is_scanned for p in self.native.pages.values()) or self.s.force_full_page_ocr,
                "encrypted": self.native.encrypted, "pdf_version": self.native.pdf_version,
                "is_form": self.native.is_form_pdf, "has_xmp_metadata": self.native.has_xmp,
                "embedded_files": self.native.embedded_files or None,
                "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "extractor": {"name": EXTRACTOR_NAME, "version": EXTRACTOR_VERSION,
                              "docling_version": self.runner.docling_version(),
                              "models": {"layout": "docling-layout-heron", "tables": "TableFormer (accurate)",
                                         "formulas": "CodeFormulaV2" if self.s.enable_formula_latex else None,
                                         "picture_classifier": "DocumentFigureClassifier-v2.5" if self.s.enable_picture_classification else None,
                                         "ocr": "RapidOCR (PP-OCR, onnxruntime)",
                                         "vlm": None}},
                # PDF metadata title wins when present: the first-page title is often a repository
                # cover sheet ("Scholarly Commons", "arXiv", ...)
                "title": props.get("title") if props.get("title") and len(props["title"]) > 4
                and not re.search(r"\.(pdf|docx?|tex)$|^untitled|^microsoft", props["title"], re.I) else title,
                "detected_title": title, "subtitle": subtitle, "doi": doi.group(1) if doi else None,
                "abstract": abstract, "keywords": keywords or props.get("keywords") or None,
                "properties": props,
                "source": {k: v for k, v in source.items() if k not in ("relative_path", "category")} or None,
                "table_of_contents": toc,
                "fonts": [{"name": f, "chars": c} for f, c in fonts.most_common(15)],
                "domain_summary": merge_entity_summary(all_ents),
                "page_confidence_mean": _nan(sum(v["layout_score"] or 0 for v in self.page_conf.values()) / len(self.page_conf)) if self.page_conf else None,
            },
            "pages": pages_out,
            "relationships": rels,
            "summary": {
                "total_elements": len(self.elements), "counts_by_type": dict(counts.most_common()),
                "counts_by_category": dict(cats.most_common()),
                "figures_pending_description": sum(1 for e in self.elements if e["metadata"].get("describe_with_vlm")),
                "warnings": self.warnings, "processing_time_ms": int(elapsed * 1000),
            },
        }


def _lang(text: str) -> str | None:
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(text[:3000])
    except Exception:  # noqa: BLE001
        return None
