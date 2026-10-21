"""Small geometry / text helpers shared by the extraction modules."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, Sequence

BBox = Sequence[float]  # [x0, y0, x1, y1], top-left origin, PDF points


def r2(x: float) -> float:
    return round(float(x), 2)


def rbbox(b: BBox) -> list[float]:
    return [r2(v) for v in b]


def area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: BBox, b: BBox) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def overlap_ratio(inner: BBox, outer: BBox) -> float:
    """Fraction of `inner` covered by `outer`."""
    a = area(inner)
    return intersection(inner, outer) / a if a > 0 else 0.0


def center(b: BBox) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def contains_point(b: BBox, p: tuple[float, float], pad: float = 1.0) -> bool:
    return b[0] - pad <= p[0] <= b[2] + pad and b[1] - pad <= p[1] <= b[3] + pad


def union(boxes: Iterable[BBox]) -> list[float] | None:
    boxes = list(boxes)
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def slug(s: str, n: int = 80) -> str:
    return re.sub(r"[^\w.-]+", "_", s).strip("_")[:n] or "doc"


def color_int_to_hex(c: int | None) -> str | None:
    if c is None:
        return None
    return f"#{(c >> 16) & 255:02X}{(c >> 8) & 255:02X}{c & 255:02X}"


def color_tuple_to_hex(c: Sequence[float] | None) -> str | None:
    if not c:
        return None
    if len(c) == 1:  # gray
        c = (c[0], c[0], c[0])
    if len(c) == 4:  # CMYK -> RGB (naive)
        cc, m, y, k = c
        c = ((1 - cc) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
    return "#" + "".join(f"{int(round(max(0, min(1, v)) * 255)):02X}" for v in c[:3])


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def word_count(s: str) -> int:
    return len(re.findall(r"\w+", s or ""))
