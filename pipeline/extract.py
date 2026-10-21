#!/usr/bin/env python3
"""Extract structured JSON from every PDF under ./data for the materials-science RAG.

For each PDF it produces   extracted/<category>/<pdf_stem>/
    <pdf_stem>.json   text (title/subtitle/heading/description/abstract/...), lists, tables (cell-level JSON),
                      formulas (LaTeX + MathML), images & charts (VLM description, OCR text-in-image,
                      colours, classification), footnotes, captions, links, annotations, form fields,
                      watermarks, sidebars, references, cross-references, materials entities -- each with
                      page / line / paragraph / column / side / bbox / section metadata
    <pdf_stem>.md     Docling markdown export (handy for eyeballing)
    images/ tables/ formulas/   PNG crops referenced from the JSON

Two stages (run both by default):
    parse     Docling + PyMuPDF + OCR  (GPU: layout, tables, formula->LaTeX, figure classifier)
    describe  vision-language model writes figure descriptions (GPU is freed from Docling first)

Usage:
    python pipeline/extract.py                          # all PDFs, both stages
    python pipeline/extract.py --stage parse            # skip VLM descriptions
    python pipeline/extract.py --files "data/preprints/*.pdf" --force
    python pipeline/extract.py --limit 2 --log-level DEBUG
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tqdm import tqdm  # noqa: E402

from extraction.config import EXTRACTOR_VERSION, ROOT, Settings  # noqa: E402
from extraction.utils import sha256_file, slug  # noqa: E402

log = logging.getLogger("extract")


def setup_logging(level: str) -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    logging.basicConfig(level=logging.WARNING, format=fmt,
                        handlers=[logging.FileHandler(ROOT / "logs" / "extract.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    for name in ("extract", "extraction"):
        logging.getLogger(name).setLevel(level)
    try:  # rapidocr resets its own logger to INFO on import, so import it before silencing
        import rapidocr.utils.log  # noqa: F401
    except ImportError:
        pass
    for noisy in ("docling", "RapidOCR", "rapidocr", "transformers", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    # Docling re-initialises RapidOCR with log_level=info; a filter survives those level resets
    logging.getLogger("RapidOCR").addFilter(lambda r: r.levelno >= logging.ERROR)


def discover(s: Settings, patterns: list[str] | None) -> list[Path]:
    if patterns:
        files = {Path(p).resolve() for pat in patterns for p in glob.glob(pat, recursive=True)}
    else:
        files = {p.resolve() for p in s.data_dir.rglob("*") if p.suffix.lower() == ".pdf"}
    return sorted(f for f in files if f.is_file())


def load_manifest(s: Settings) -> dict[str, dict]:
    """data/manifest.json written by scripts/scrap.py -> provenance (URL, license, title) per file."""
    mp = s.data_dir / "manifest.json"
    if not mp.exists():
        return {}
    try:
        items = json.loads(mp.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    out = {}
    for m in items:
        p = Path(m.get("local_path", ""))
        p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
        out[str(p).lower()] = m
    return out


def output_path(s: Settings, pdf: Path) -> tuple[Path, str]:
    try:
        rel = pdf.relative_to(s.data_dir.resolve())
    except ValueError:
        rel = Path(pdf.name)
    category = rel.parent.as_posix() if rel.parent != Path(".") else "uncategorized"
    stem = slug(pdf.stem)
    return s.output_dir / category / stem / f"{stem}.json", category


def up_to_date(out_json: Path, sha: str) -> bool:
    if not out_json.exists():
        return False
    try:
        d = json.loads(out_json.read_text(encoding="utf-8"))
        return d["document"]["sha256"] == sha and d["document"]["extractor"]["version"] == EXTRACTOR_VERSION
    except Exception:  # noqa: BLE001
        return False


def stage_parse(s: Settings, pdfs: list[Path], force: bool, index: dict) -> list[Path]:
    from extraction.builder import DoclingRunner, DocumentBuilder
    from extraction.describe import carry_over_descriptions

    runner = DoclingRunner(s)
    builder = DocumentBuilder(s, runner)
    manifest = load_manifest(s)
    outputs = []
    for pdf in tqdm(pdfs, desc="parse", unit="pdf"):
        out_json, category = output_path(s, pdf)
        sha = sha256_file(pdf)
        rel = pdf.relative_to(ROOT).as_posix() if pdf.is_relative_to(ROOT) else str(pdf)
        if not force and up_to_date(out_json, sha):
            log.info("skip (up to date): %s", rel)
            outputs.append(out_json)
            continue
        out_json.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        log.info("parsing %s", rel)
        src = dict(manifest.get(str(pdf).lower(), {}))
        src.pop("local_path", None)
        src.pop("sha256", None)
        src.update({"relative_path": rel, "category": category})
        try:
            doc = builder.build(pdf, out_json, src)
        except Exception as e:  # noqa: BLE001
            log.exception("FAILED %s", rel)
            index[rel] = {"status": "failed", "error": str(e)}
            continue
        kept = carry_over_descriptions(out_json, doc)
        if kept:
            log.info("  kept %d existing figure descriptions", kept)
        tmp = out_json.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, out_json)
        sm = doc["summary"]
        index[rel] = {"status": "parsed", "output": out_json.relative_to(ROOT).as_posix(),
                      "doc_id": doc["document"]["id"], "pages": doc["document"]["total_pages"],
                      "elements": sm["total_elements"], "counts_by_type": sm["counts_by_type"],
                      "warnings": len(sm["warnings"]), "seconds": round(time.time() - t0, 1)}
        log.info("  -> %s (%d elements, %d warnings, %.0fs)", out_json.name, sm["total_elements"],
                 len(sm["warnings"]), time.time() - t0)
        outputs.append(out_json)
    # free GPU memory held by Docling models before the VLM stage
    del builder, runner
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return outputs


def stage_describe(s: Settings, jsons: list[Path], force: bool, index: dict) -> None:
    from extraction.describe import describe_document
    from extraction.vlm import VLM

    pending = []
    for j in jsons:
        if not j.exists():
            continue
        d = json.loads(j.read_text(encoding="utf-8"))
        if force or d["summary"].get("figures_pending_description", 0) > 0:
            pending.append(j)
    if not pending:
        log.info("describe: nothing to do")
        return
    vlm = VLM(s)
    try:
        for j in tqdm(pending, desc="describe", unit="pdf"):
            t0 = time.time()
            n = describe_document(j, vlm, force=force)
            log.info("described %d figures in %s (%.0fs)", n, j.name, time.time() - t0)
            for v in index.values():
                if v.get("output") == j.relative_to(ROOT).as_posix():
                    v["figures_described"] = n
    finally:
        vlm.unload()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", help="folder with PDFs (default: DATA_DIR from .env)")
    ap.add_argument("--out-dir", help="output folder (default: OUTPUT_DIR from .env)")
    ap.add_argument("--files", nargs="+", help="explicit PDF paths / glob patterns")
    ap.add_argument("--stage", choices=["all", "parse", "describe"], default="all")
    ap.add_argument("--force", action="store_true", help="re-extract even if output is up to date")
    ap.add_argument("--limit", type=int, help="process at most N PDFs")
    ap.add_argument("--no-vlm", action="store_true", help="same as --stage parse")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level.upper())
    s = Settings()
    if args.data_dir:
        s.data_dir = Path(args.data_dir).resolve()
    if args.out_dir:
        s.output_dir = Path(args.out_dir).resolve()
    stage = "parse" if args.no_vlm else args.stage
    if not s.enable_picture_description and stage == "all":
        stage = "parse"

    pdfs = discover(s, args.files)
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        log.error("no PDFs found in %s (run scripts/scrap.py first?)", s.data_dir)
        sys.exit(1)
    log.info("%d PDF(s) | device=%s | stage=%s | out=%s", len(pdfs), s.resolved_device, stage, s.output_dir)
    s.output_dir.mkdir(parents=True, exist_ok=True)

    index_path = s.output_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    try:
        if stage in ("all", "parse"):
            jsons = stage_parse(s, pdfs, args.force, index)
        else:
            jsons = [output_path(s, p)[0] for p in pdfs]
        if stage in ("all", "describe"):
            stage_describe(s, jsons, args.force and stage == "describe", index)
    except KeyboardInterrupt:
        log.warning("interrupted - progress so far is saved")
    finally:
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("done. index: %s", index_path)


if __name__ == "__main__":
    main()
