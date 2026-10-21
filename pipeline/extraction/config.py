"""Settings loaded from .env (with sane defaults)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]  # project root (pipeline/extraction/config.py)
load_dotenv(ROOT / ".env")

EXTRACTOR_NAME = "materials-rag-extractor"
EXTRACTOR_VERSION = "1.0.0"


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None or v == "" else v.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v else default


def _float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v else default


def _str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v else default


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: ROOT / _str("DATA_DIR", "data"))
    output_dir: Path = field(default_factory=lambda: ROOT / _str("OUTPUT_DIR", "extracted"))

    device: str = field(default_factory=lambda: _str("DEVICE", "auto"))
    num_threads: int = field(default_factory=lambda: _int("NUM_THREADS", 4))

    page_batch_size: int = field(default_factory=lambda: _int("PAGE_BATCH_SIZE", 20))
    images_scale: float = field(default_factory=lambda: _float("IMAGES_SCALE", 2.0))
    enable_ocr: bool = field(default_factory=lambda: _bool("ENABLE_OCR", True))
    force_full_page_ocr: bool = field(default_factory=lambda: _bool("FORCE_FULL_PAGE_OCR", False))
    enable_formula_latex: bool = field(default_factory=lambda: _bool("ENABLE_FORMULA_LATEX", True))
    enable_code_enrichment: bool = field(default_factory=lambda: _bool("ENABLE_CODE_ENRICHMENT", True))
    enable_picture_classification: bool = field(default_factory=lambda: _bool("ENABLE_PICTURE_CLASSIFICATION", True))
    save_formula_images: bool = field(default_factory=lambda: _bool("SAVE_FORMULA_IMAGES", True))
    save_table_images: bool = field(default_factory=lambda: _bool("SAVE_TABLE_IMAGES", True))

    enable_picture_description: bool = field(default_factory=lambda: _bool("ENABLE_PICTURE_DESCRIPTION", True))
    enable_image_ocr: bool = field(default_factory=lambda: _bool("ENABLE_IMAGE_OCR", True))
    min_image_area_fraction: float = field(default_factory=lambda: _float("MIN_IMAGE_AREA_FRACTION", 0.01))

    vlm_backend: str = field(default_factory=lambda: _str("VLM_BACKEND", "local"))
    vlm_local_model: str = field(default_factory=lambda: _str("VLM_LOCAL_MODEL", "Qwen/Qwen3.5-0.8B"))
    vlm_load_4bit: bool = field(default_factory=lambda: _bool("VLM_LOAD_4BIT", False))
    vlm_max_image_side: int = field(default_factory=lambda: _int("VLM_MAX_IMAGE_SIDE", 896))
    vlm_max_new_tokens: int = field(default_factory=lambda: _int("VLM_MAX_NEW_TOKENS", 400))
    vlm_api_url: str = field(default_factory=lambda: _str("VLM_API_URL", "http://localhost:11434/v1/chat/completions"))
    vlm_api_model: str = field(default_factory=lambda: _str("VLM_API_MODEL", "qwen3.5:4b"))
    vlm_api_key: str = field(default_factory=lambda: _str("VLM_API_KEY", ""))
    vlm_timeout: int = field(default_factory=lambda: _int("VLM_TIMEOUT", 180))

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
