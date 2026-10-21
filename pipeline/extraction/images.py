"""Image-level analysis that does not need a VLM: OCR, colours, QR/barcodes, micrograph cues."""
from __future__ import annotations

import logging
import re
from functools import lru_cache

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

SCALE_BAR_RE = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s?(nm|µm|μm|um|mm|cm|Å|pm|m)\s*$", re.I)
MAGNIFICATION_RE = re.compile(r"(?:^|\s|mag\.?\s?=?\s?)([x×X]\s?\d[\d,.]*\s?[kK]?|\d[\d,.]*\s?[kK]?\s?[x×X])(?:\s|$)")
PANEL_RE = re.compile(r"^\s*\(?([a-hA-H])\)?\s*$")
INSTRUMENT_RE = re.compile(r"\b(EHT|HV|WD|Mag|Spot|Det|Signal|kV|Pressure|Tilt|HFW|dwell|Date|Time)\b\s*[=:]?", re.I)


@lru_cache(maxsize=1)
def _ocr_engine():
    from rapidocr import RapidOCR

    return RapidOCR(params={"Global.log_level": "error"})


def ocr_image(img: Image.Image, min_conf: float = 0.5) -> list[dict]:
    """Detect + recognise text in an image. Returns [{text, bbox_in_image, confidence}]."""
    try:
        res = _ocr_engine()(np.array(img.convert("RGB")))
    except Exception as e:  # noqa: BLE001
        log.warning("OCR failed: %s", e)
        return []
    if res is None or res.boxes is None:
        return []
    out = []
    for box, txt, score in zip(res.boxes, res.txts, res.scores):
        if not txt or not txt.strip() or float(score) < min_conf:
            continue
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        out.append({"text": txt.strip(), "bbox_in_image": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                    "confidence": round(float(score), 3)})
    return out


def dominant_colors(img: Image.Image, k: int = 5) -> list[dict]:
    """Top-k colours with their share of the image (median-cut quantisation)."""
    try:
        small = img.convert("RGB").copy()
        small.thumbnail((200, 200))
        q = small.quantize(colors=k, method=Image.Quantize.MEDIANCUT)
        pal = q.getpalette()[: k * 3]
        counts = sorted(q.getcolors(), reverse=True)
        total = sum(c for c, _ in counts) or 1
        return [{"hex": "#{:02X}{:02X}{:02X}".format(*pal[i * 3: i * 3 + 3]), "fraction": round(c / total, 3)}
                for c, i in counts[:k]]
    except Exception as e:  # noqa: BLE001
        log.debug("colour quantisation failed: %s", e)
        return []


def image_stats(img: Image.Image) -> dict:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    sat = (arr.max(axis=2) - arr.min(axis=2)).mean()
    return {"is_grayscale": bool(sat < 8), "mean_brightness": round(float(arr.mean()) / 255, 3)}


def decode_codes(img: Image.Image) -> list[dict]:
    """QR codes and 1-D barcodes via OpenCV (no extra system libraries needed)."""
    out = []
    try:
        import cv2

        arr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        ok, texts, pts, _ = cv2.QRCodeDetector().detectAndDecodeMulti(arr)
        if ok:
            for t, p in zip(texts, pts):
                if t:
                    out.append({"code_type": "QR", "decoded_value": t,
                                "bbox_in_image": [int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max())]})
        if hasattr(cv2, "barcode"):
            res = cv2.barcode.BarcodeDetector().detectAndDecodeWithType(arr)
            if res and res[0]:
                for t, typ in zip(res[1], res[2]):
                    if t:
                        out.append({"code_type": typ or "barcode", "decoded_value": t})
    except Exception as e:  # noqa: BLE001
        log.debug("code decoding failed: %s", e)
    return out


def micrograph_cues(texts: list[str]) -> dict:
    """Scale bars, magnifications, panel labels and SEM/TEM info-bar fields found in image text."""
    cues: dict = {}
    scale = [t for t in texts if SCALE_BAR_RE.match(t)]
    if scale:
        cues["scale_bars"] = scale
    mags = [m.group(1).strip() for t in texts for m in MAGNIFICATION_RE.finditer(t)]
    if mags:
        cues["magnifications"] = mags
    panels = [PANEL_RE.match(t).group(1).lower() for t in texts if PANEL_RE.match(t)]
    if len(panels) >= 2:
        cues["panel_labels"] = sorted(set(panels))
    instr = [t for t in texts if INSTRUMENT_RE.search(t) and re.search(r"\d", t)]
    if instr:
        cues["instrument_info"] = instr
    return cues


def merge_text_sources(pdf_texts: list[dict], ocr_texts: list[dict]) -> list[dict]:
    """Combine vector text (exact, from the PDF) with OCR text (raster), dropping OCR duplicates."""
    out = [dict({k: v for k, v in t.items() if k != "origin"}, source=t.get("origin", "pdf_text_layer")) for t in pdf_texts]
    seen = {re.sub(r"\W+", "", t["text"]).lower() for t in pdf_texts}
    for t in ocr_texts:
        key = re.sub(r"\W+", "", t["text"]).lower()
        if key and key not in seen and not any(key in s for s in seen if len(key) > 3):
            out.append(dict(t, source="ocr"))
            seen.add(key)
    return out
