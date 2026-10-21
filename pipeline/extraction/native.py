"""Native PDF layer via PyMuPDF.

Docling gives us *semantics* (what is a heading, table, formula ...). PyMuPDF gives
us the exact *typography and geometry* the PDF itself stores: fonts, colours,
individual lines, hyperlinks, annotations, form fields, embedded image properties,
shaded boxes (sidebars) and rotated/transparent text (watermarks).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from .utils import area, color_int_to_hex, color_tuple_to_hex, r2, rbbox

log = logging.getLogger(__name__)

# PyMuPDF span flag bits
F_SUPERSCRIPT, F_ITALIC, F_SERIF, F_MONO, F_BOLD = 1, 2, 4, 8, 16

LINK_KINDS = {pymupdf.LINK_URI: "external", pymupdf.LINK_GOTO: "internal",
              pymupdf.LINK_GOTOR: "external_pdf", pymupdf.LINK_LAUNCH: "launch",
              pymupdf.LINK_NAMED: "named"}


@dataclass
class Span:
    bbox: list[float]
    text: str
    font: str
    size: float
    flags: int
    color: str | None
    alpha: int
    line_idx: int

    @property
    def bold(self) -> bool:
        return bool(self.flags & F_BOLD) or bool(re.search(r"bold|black|heavy|semibold", self.font, re.I))

    @property
    def italic(self) -> bool:
        return bool(self.flags & F_ITALIC) or bool(re.search(r"italic|oblique", self.font, re.I))

    @property
    def monospace(self) -> bool:
        return bool(self.flags & F_MONO) or bool(re.search(r"mono|courier|consol", self.font, re.I))

    @property
    def serif(self) -> bool:
        return bool(self.flags & F_SERIF)

    @property
    def superscript(self) -> bool:
        return bool(self.flags & F_SUPERSCRIPT)


@dataclass
class Line:
    bbox: list[float]
    text: str
    direction: tuple[float, float]
    span_ids: list[int]


@dataclass
class NativePage:
    number: int  # 1-based
    width: float
    height: float
    rotation: int
    label: str | None
    spans: list[Span] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)
    annotations: list[dict] = field(default_factory=list)
    widgets: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    boxes: list[dict] = field(default_factory=list)  # filled rectangles (sidebars, highlight boxes)
    watermark_candidates: list[dict] = field(default_factory=list)
    num_drawings: int = 0
    char_count: int = 0

    @property
    def image_coverage(self) -> float:
        page_area = self.width * self.height
        return min(1.0, sum(area(i["bbox"]) for i in self.images) / page_area) if page_area else 0.0

    @property
    def is_scanned(self) -> bool:
        return self.char_count < 30 and self.image_coverage > 0.5


@dataclass
class NativeDoc:
    path: Path
    page_count: int
    metadata: dict
    pdf_version: str | None
    encrypted: bool
    is_form_pdf: bool
    toc: list[dict]
    embedded_files: list[str]
    has_xmp: bool
    pages: dict[int, NativePage]


def _pdf_date(s: str | None) -> str | None:
    """'D:20240206154945+00'00'' -> '2024-02-06T15:49:45+00:00'"""
    if not s:
        return None
    m = re.match(r"D?:?(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?([Zz+\-])?(\d{2})?'?(\d{2})?", s)
    if not m:
        return s
    y, mo, d, h, mi, sec, tz, tzh, tzm = m.groups()
    out = f"{y}-{mo or '01'}-{d or '01'}"
    if h:
        out += f"T{h}:{mi or '00'}:{sec or '00'}"
        if tz in ("Z", "z"):
            out += "Z"
        elif tz in ("+", "-") and tzh:
            out += f"{tz}{tzh}:{tzm or '00'}"
    return out


def _read_page(doc: pymupdf.Document, page: pymupdf.Page) -> NativePage:
    rect = page.rect
    try:
        label = page.get_label() or None
    except Exception:  # noqa: BLE001
        label = None
    np_ = NativePage(number=page.number + 1, width=r2(rect.width), height=r2(rect.height),
                     rotation=page.rotation, label=label)

    # ---- text: spans and lines --------------------------------------------
    td = page.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)
    for block in td.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            ids = []
            for s in line.get("spans", []):
                txt = s.get("text", "")
                if not txt.strip():
                    continue
                np_.spans.append(Span(bbox=rbbox(s["bbox"]), text=txt, font=s.get("font", ""),
                                      size=r2(s.get("size", 0)), flags=s.get("flags", 0),
                                      color=color_int_to_hex(s.get("color")),
                                      alpha=s.get("alpha", 255), line_idx=len(np_.lines)))
                ids.append(len(np_.spans) - 1)
                np_.char_count += len(txt.strip())
            if ids:
                ltxt = "".join(np_.spans[i].text for i in ids)
                np_.lines.append(Line(bbox=rbbox(line["bbox"]), text=ltxt,
                                      direction=tuple(round(v, 3) for v in line.get("dir", (1, 0))),
                                      span_ids=ids))

    # ---- watermark candidates: rotated / large / translucent text --------
    for ln in np_.lines:
        spans = [np_.spans[i] for i in ln.span_ids]
        max_size = max(s.size for s in spans)
        rotated = abs(ln.direction[1]) > 0.2
        translucent = any(s.alpha < 200 for s in spans)
        if (rotated and max_size >= 18) or (translucent and max_size >= 14) or max_size >= 60:
            ang = 0.0
            if rotated:
                import math
                ang = round(math.degrees(math.atan2(-ln.direction[1], ln.direction[0])), 1)
            np_.watermark_candidates.append({
                "text": ln.text.strip(), "bbox": ln.bbox, "rotation": ang, "font_size": max_size,
                "opacity": round(min(s.alpha for s in spans) / 255, 2), "color": spans[0].color})

    # ---- hyperlinks -------------------------------------------------------
    for ln in page.get_links():
        r = ln.get("from")
        if r is None:
            continue
        txt = page.get_textbox(r).strip()
        kind = LINK_KINDS.get(ln.get("kind"), "other")
        uri = ln.get("uri")
        if kind == "external" and uri and uri.lower().startswith("mailto:"):
            kind = "email"
        np_.links.append({"bbox": rbbox(r), "text": " ".join(txt.split()), "link_type": kind, "url": uri,
                          "target_page": (ln["page"] + 1) if ln.get("page", -1) is not None and ln.get("page", -1) >= 0 else None})

    # ---- annotations (comments, highlights, stamps ...) -------------------
    for a in page.annots() or []:
        try:
            atype = a.type[1]
            if atype in ("Link", "Widget"):
                continue
            info = a.info or {}
            marked = ""
            if atype in ("Highlight", "Underline", "StrikeOut", "Squiggly"):
                marked = page.get_textbox(a.rect).strip()
            np_.annotations.append({
                "annotation_type": atype.lower(), "bbox": rbbox(a.rect),
                "author": info.get("title") or None, "content": info.get("content") or None,
                "subject": info.get("subject") or None,
                "created": _pdf_date(info.get("creationDate")), "modified": _pdf_date(info.get("modDate")),
                "color": color_tuple_to_hex((a.colors or {}).get("stroke")),
                "marked_text": " ".join(marked.split()) or None,
            })
        except Exception as e:  # noqa: BLE001
            log.debug("annotation read failed p%s: %s", np_.number, e)

    # ---- form fields ------------------------------------------------------
    for w in page.widgets() or []:
        try:
            val = w.field_value
            if w.field_type in (pymupdf.PDF_WIDGET_TYPE_CHECKBOX, pymupdf.PDF_WIDGET_TYPE_RADIOBUTTON):
                val = val not in (False, "Off", "", None)
            np_.widgets.append({
                "field_type": (w.field_type_string or "").lower(), "name": w.field_name,
                "label": w.field_label or None, "value": val, "bbox": rbbox(w.rect),
                "read_only": bool(w.field_flags & 1), "required": bool(w.field_flags & 2),
                "options": list(w.choice_values) if w.choice_values else None,
            })
        except Exception as e:  # noqa: BLE001
            log.debug("widget read failed p%s: %s", np_.number, e)

    # ---- embedded raster images ------------------------------------------
    try:
        for info in page.get_image_info(xrefs=True):
            xref = info.get("xref", 0)
            has_alpha = False
            if xref:
                try:
                    has_alpha = doc.xref_get_key(xref, "SMask")[0] != "null"
                except Exception:  # noqa: BLE001
                    pass
            np_.images.append({
                "bbox": rbbox(info["bbox"]), "xref": xref, "native_width_px": info.get("width"),
                "native_height_px": info.get("height"), "bits_per_component": info.get("bpc"),
                "colorspace": {1: "gray", 3: "rgb", 4: "cmyk"}.get(info.get("colorspace"), info.get("cs-name")),
                "x_dpi": info.get("xres"), "y_dpi": info.get("yres"), "has_transparency": has_alpha,
            })
    except Exception as e:  # noqa: BLE001
        log.debug("image info failed p%s: %s", np_.number, e)

    # ---- filled rectangles (sidebars / call-out boxes) ---------------------
    try:
        drawings = page.get_drawings()
        np_.num_drawings = len(drawings)
        page_area = rect.width * rect.height
        for d in drawings[:3000]:
            fill = d.get("fill")
            if not fill or d.get("type") not in ("f", "fs"):
                continue
            r = d.get("rect")
            if r is None:
                continue
            a = r.width * r.height
            hexc = color_tuple_to_hex(fill)
            # ignore tiny marks, full-page backgrounds and pure white fills
            if a < 0.01 * page_area or a > 0.9 * page_area or hexc in ("#FFFFFF", None):
                continue
            np_.boxes.append({"bbox": rbbox(r), "background_color": hexc,
                              "opacity": d.get("fill_opacity")})
    except Exception as e:  # noqa: BLE001
        log.debug("drawings failed p%s: %s", np_.number, e)

    return np_


def read_native(path: Path) -> NativeDoc:
    doc = pymupdf.open(path)
    try:
        meta = doc.metadata or {}
        toc = [{"title": t[1].strip(), "page": t[2], "level": t[0]} for t in doc.get_toc(simple=True)]
        try:
            emb = [doc.embfile_info(i).get("filename") for i in range(doc.embfile_count())]
        except Exception:  # noqa: BLE001
            emb = []
        try:
            has_xmp = bool(doc.get_xml_metadata())
        except Exception:  # noqa: BLE001
            has_xmp = False
        fmt = meta.get("format") or ""
        pages = {}
        for page in doc:
            try:
                pages[page.number + 1] = _read_page(doc, page)
            except Exception as e:  # noqa: BLE001
                log.warning("native read failed on page %s: %s", page.number + 1, e)
                pages[page.number + 1] = NativePage(page.number + 1, r2(page.rect.width), r2(page.rect.height),
                                                    page.rotation, None)
        return NativeDoc(
            path=path, page_count=doc.page_count,
            metadata={
                "title": meta.get("title") or None, "author": meta.get("author") or None,
                "subject": meta.get("subject") or None,
                "keywords": [k.strip() for k in re.split(r"[;,]", meta.get("keywords") or "") if k.strip()],
                "creator": meta.get("creator") or None, "producer": meta.get("producer") or None,
                "created_date": _pdf_date(meta.get("creationDate")),
                "modified_date": _pdf_date(meta.get("modDate")),
                "trapped": meta.get("trapped") or None,
            },
            pdf_version=fmt.replace("PDF", "").strip() or None,
            encrypted=bool(doc.is_encrypted or meta.get("encryption")),
            is_form_pdf=bool(doc.is_form_pdf),
            toc=toc, embedded_files=emb, has_xmp=has_xmp, pages=pages,
        )
    finally:
        doc.close()
