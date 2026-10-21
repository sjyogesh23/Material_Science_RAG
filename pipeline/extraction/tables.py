"""Table structure -> rich JSON (headers, row dicts, typed cells, merged cells, CSV/Markdown/HTML)."""
from __future__ import annotations

import csv
import io
import re
from collections import Counter

from .domain import FORMULA_RE, _is_formula

_NUM = r"[-−+]?\d+(?:[.,]\d+)*(?:[eE][-−]?\d+)?"
UNIT_IN_HEADER_RE = re.compile(r"[\(\[]\s*([^()\[\]]{1,20})\s*[\)\]]\s*$")
TOTAL_ROW_RE = re.compile(r"^\s*(total|sum|overall|mean|average|avg\.?|std\.?|s\.d\.|median|subtotal)\b", re.I)


def _to_float(s: str) -> float | None:
    s = s.replace("−", "-").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def classify_cell(text: str) -> dict:
    """Infer the data type of a cell and, when numeric, its value / unit / uncertainty."""
    t = (text or "").strip()
    if not t or t in {"-", "–", "\u2014", "n/a", "N/A", "NA", "nd", "n.d.", "/"}:
        return {"data_type": "empty" if not t else "missing"}
    if t.lower() in {"yes", "no", "true", "false", "y", "n", "✓", "✔", "✗", "✘", "x", "•", "●", "+"}:
        # in tables a plain "x" / "•" / "+" is a tick mark (= yes); only the dedicated cross glyphs mean no
        return {"data_type": "boolean", "value": t.lower() in {"yes", "true", "y", "✓", "✔", "x", "•", "●", "+"}}
    m = re.fullmatch(rf"({_NUM})\s*%", t)
    if m:
        return {"data_type": "percentage", "value": _to_float(m.group(1)), "unit": "%"}
    m = re.fullmatch(rf"({_NUM})\s*(?:±|\+/-|\+-)\s*({_NUM})\s*(\S{{0,12}})", t)
    if m:
        d = {"data_type": "number_with_uncertainty", "value": _to_float(m.group(1)),
             "uncertainty": _to_float(m.group(2))}
        if m.group(3):
            d["unit"] = m.group(3)
        return d
    m = re.fullmatch(rf"({_NUM})\s*(?:–|-|\u2014|to|~)\s*({_NUM})\s*(\S{{0,12}})", t)
    if m and not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", t):
        d = {"data_type": "range", "value": _to_float(m.group(1)), "value_max": _to_float(m.group(2))}
        if m.group(3):
            d["unit"] = m.group(3)
        return d
    m = re.fullmatch(rf"({_NUM})", t)
    if m:
        v = _to_float(t)
        return {"data_type": "integer" if v is not None and float(v).is_integer() and "." not in t else "number", "value": v}
    m = re.fullmatch(rf"({_NUM})\s*([A-Za-zµμΩ°Å%/·.\-\d²³]{{1,15}})", t)
    if m and not re.search(r"[A-Za-z]{6,}", m.group(2)):
        return {"data_type": "number_with_unit", "value": _to_float(m.group(1)), "unit": m.group(2)}
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}", t):
        return {"data_type": "date"}
    if FORMULA_RE.fullmatch(t) and _is_formula(t):
        return {"data_type": "chemical_formula"}
    return {"data_type": "text"}


def _unique(names: list[str]) -> list[str]:
    seen: Counter = Counter()
    out = []
    for n in names:
        seen[n] += 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


def build_table(item, doc, bbox_fn) -> dict:
    """Convert a Docling TableItem into the table JSON block.

    ``bbox_fn(bbox, page_no)`` converts a docling BoundingBox to a top-left [x0,y0,x1,y1] list.
    """
    data = item.data
    grid = data.grid
    n_rows, n_cols = data.num_rows, data.num_cols
    page_no = item.prov[0].page_no if item.prov else None

    # --- header rows: leading rows whose non-empty cells are all column headers
    header_rows = 0
    for row in grid:
        cells = [c for c in row if c.text.strip()]
        if cells and all(c.column_header for c in cells):
            header_rows += 1
        else:
            break
    has_header_row = header_rows > 0
    has_header_column = any(row and row[0].row_header for row in grid)

    headers = []
    for j in range(n_cols):
        parts = []
        for i in range(header_rows):
            txt = grid[i][j].text.strip() if j < len(grid[i]) else ""
            if txt and (not parts or parts[-1] != txt):
                parts.append(txt)
        headers.append(" / ".join(parts) if parts else f"col_{j + 1}")
    headers = _unique(headers)

    rows, footer_rows, section_rows = [], [], []
    for i in range(header_rows, n_rows):
        row = grid[i]
        texts = [c.text.strip() for c in row]
        if not any(texts):
            continue
        idx = i - header_rows + 1
        non_empty = {c.text.strip() for c in row if c.text.strip()}
        if any(c.row_section for c in row) or (len(non_empty) == 1 and n_cols > 2 and row[0].col_span >= n_cols - 1):
            section_rows.append({"row_index": idx, "text": next(iter(non_empty))})
            continue
        r = {"row_index": idx, "cells": {headers[j]: texts[j] for j in range(min(n_cols, len(texts)))}}
        if has_header_column and texts and texts[0]:
            r["row_header"] = texts[0]
        if section_rows:
            r["section"] = section_rows[-1]["text"]
        rows.append(r)
        if texts and TOTAL_ROW_RE.match(texts[0]):
            footer_rows.append({"row_index": idx, "type": TOTAL_ROW_RE.match(texts[0]).group(1).lower().rstrip(".")})

    # --- typed cells
    cells, merged, col_types = [], [], [Counter() for _ in range(n_cols)]
    for c in data.table_cells:
        info = classify_cell(c.text)
        r0, c0 = c.start_row_offset_idx, c.start_col_offset_idx
        cell = {"row": r0, "column": c0, "row_span": c.row_span, "col_span": c.col_span, "text": c.text.strip(),
                "is_header": bool(c.column_header), "is_row_header": bool(c.row_header),
                "is_section": bool(c.row_section), **info}
        if c.bbox is not None and page_no is not None:
            cell["bbox"] = bbox_fn(c.bbox, page_no)
        if not c.column_header and 0 <= c0 < n_cols:
            col_types[c0][info["data_type"]] += 1
            cell["column_name"] = headers[c0]
        cells.append(cell)
        if c.row_span > 1 or c.col_span > 1:
            merged.append({"row": r0, "column": c0, "row_span": c.row_span, "col_span": c.col_span, "text": c.text.strip()})

    columns = []
    for j, h in enumerate(headers):
        types = col_types[j]
        types.pop("empty", None)
        dom = types.most_common(1)[0][0] if types else "empty"
        unit = None
        m = UNIT_IN_HEADER_RE.search(h)
        if m and len(m.group(1)) <= 15:
            unit = m.group(1)
        else:
            units = Counter(cl.get("unit") for cl in cells if cl.get("column") == j and cl.get("unit"))
            unit = units.most_common(1)[0][0] if units else None
        columns.append({"index": j, "name": h, "data_type": dom, "unit": unit})

    # --- serialisations
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for row in grid:
        w.writerow([c.text.strip() for c in row])
    try:
        md = item.export_to_markdown(doc=doc)
    except Exception:  # noqa: BLE001
        md = None
    try:
        html = item.export_to_html(doc=doc)
    except Exception:  # noqa: BLE001
        html = None

    return {
        "num_rows": n_rows, "num_columns": n_cols, "num_header_rows": header_rows,
        "num_data_rows": len(rows), "has_header_row": has_header_row, "has_header_column": has_header_column,
        "headers": headers, "columns": columns, "rows": rows, "section_rows": section_rows,
        "footer_rows": footer_rows, "cells": cells, "merged_cells": merged,
        "is_empty": not any(c.text.strip() for c in data.table_cells),
        "csv": buf.getvalue(), "markdown": md, "html": html,
    }
