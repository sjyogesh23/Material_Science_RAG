"""Stage 2: add VLM descriptions to figures of already-extracted documents.

Runs after Docling has released the GPU so that the VLM gets the whole 4 GB.
It is resumable: figures that already have a description are skipped.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from PIL import Image

from .vlm import VLM

log = logging.getLogger(__name__)

CHART_KEYS = ("title", "x_axis", "y_axis", "series", "legend", "key_findings")


def _apply(e: dict, r: dict, vlm: VLM) -> None:
    e["description"] = r.get("description")
    e["short_caption"] = r.get("short_caption")
    objs = r.get("objects")
    if isinstance(objs, list):
        e["objects_detected"] = [o for o in objs if isinstance(o, dict) and o.get("label")][:25]
    if r.get("image_type"):
        e["image_type"] = r["image_type"]
    if r.get("materials"):
        e["materials_shown"] = [m for m in r["materials"] if isinstance(m, str) and m.strip()][:15] or None
    if r.get("technique") not in (None, "", "null"):
        e["technique"] = r["technique"]
    if r.get("_prompt_kind") == "chart":
        for k in CHART_KEYS:
            if r.get(k) not in (None, "", [], {}):
                e[k if k != "title" else "chart_title"] = r[k]
        if r.get("chart_type"):
            e["chart_type_vlm"] = r["chart_type"]
        e["values_estimated"] = True  # axis ranges / trends read by a VLM, not exact data
    e["vlm"] = {"model": vlm.model_name, "backend": vlm.backend, "parse_error": bool(r.get("parse_error"))}


VLM_FIELDS = ("description", "short_caption", "objects_detected", "image_type", "materials_shown", "technique",
              "chart_title", "x_axis", "y_axis", "series", "legend", "key_findings", "chart_type_vlm",
              "values_estimated", "vlm")


def carry_over_descriptions(old_json: Path, new_doc: dict) -> int:
    """Copy VLM output from a previous extraction onto the same figures (same page + bbox) of a
    fresh one, so re-parsing a PDF does not throw away slow-to-compute descriptions."""
    if not old_json.exists():
        return 0
    try:
        old = json.loads(old_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0

    def key(e):
        b = e.get("metadata", {}).get("bbox") or [0, 0, 0, 0]
        return e["metadata"].get("page"), tuple(round(v / 5) for v in b)  # ~5 pt tolerance

    prev = {key(e): e for p in old.get("pages", []) for e in p["elements"] if e.get("description")}
    n = 0
    for p in new_doc["pages"]:
        for e in p["elements"]:
            src = prev.get(key(e)) if e.get("metadata", {}).get("describe_with_vlm") else None
            if src:
                for f in VLM_FIELDS:
                    if f in src:
                        e[f] = src[f]
                n += 1
    if n:
        new_doc["document"]["extractor"]["models"]["vlm"] = old["document"]["extractor"]["models"].get("vlm")
        new_doc["summary"]["figures_pending_description"] = sum(
            1 for p in new_doc["pages"] for e in p["elements"]
            if e.get("metadata", {}).get("describe_with_vlm") and not e.get("description"))
    return n


def describe_document(json_path: Path, vlm: VLM, force: bool = False) -> int:
    from tqdm import tqdm

    data = json.loads(json_path.read_text(encoding="utf-8"))
    done = 0
    todo = [e for page in data["pages"] for e in page["elements"]
            if e.get("metadata", {}).get("describe_with_vlm") and e.get("image_file")
            and (force or not e.get("description"))]
    for e in tqdm(todo, desc=f"  figures in {json_path.stem[:30]}", unit="fig", leave=False):
        img_path = json_path.parent / e["image_file"]
        if not img_path.exists():
            continue
        with Image.open(img_path) as img:
            r = vlm.describe(img, e.get("category"), (e.get("caption") or {}).get("text"))
        if "error" in r:
            data["summary"]["warnings"].append(f"VLM failed for {e['id']}: {r['error']}")
            continue
        _apply(e, r, vlm)
        done += 1
        if done % 10 == 0:  # checkpoint, so an interruption loses at most 10 figures
            _save(json_path, data, vlm)
    _save(json_path, data, vlm)
    return done


def _save(json_path: Path, data: dict, vlm: VLM) -> None:
    data["document"]["extractor"]["models"]["vlm"] = vlm.model_name
    data["summary"]["figures_pending_description"] = sum(
        1 for p in data["pages"] for e in p["elements"]
        if e.get("metadata", {}).get("describe_with_vlm") and not e.get("description"))
    tmp = json_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, json_path)
