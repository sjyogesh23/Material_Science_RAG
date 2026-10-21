"""Token counting with the Qwen3 tokenizer (shared vocabulary of the embedder, reranker and chat LLM).

The tokenizer is loaded lazily from the embed model's HF snapshot (no network needed once cached).
If it cannot be loaded, a ~4 chars/token heuristic is used so callers never fail.
"""
from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Any

from app.config import get_settings

log = logging.getLogger(__name__)
_lock = threading.Lock()
_tokenizer: Any = None
_failed = False


class _FastTokenizer:
    """tokenizer.json via the ``tokenizers`` library. ``transformers.AutoTokenizer`` would import torch
    (~2 GB commit on Windows) just to count tokens; this keeps the API process torch-free."""

    def __init__(self, repo: str):
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        try:
            path = hf_hub_download(repo, "tokenizer.json", local_files_only=True)
        except Exception:
            path = hf_hub_download(repo, "tokenizer.json")
        self._tok = Tokenizer.from_file(path)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)


def get_tokenizer() -> Any | None:
    global _tokenizer, _failed
    if _tokenizer is None and not _failed:
        with _lock:
            if _tokenizer is None and not _failed:
                try:
                    _tokenizer = _FastTokenizer(get_settings().embed_model)
                except Exception as exc:  # pragma: no cover - offline without cache
                    log.warning("Qwen3 tokenizer unavailable, using heuristic: %s", exc)
                    _failed = True
    return _tokenizer


@lru_cache(maxsize=4096)
def _count_cached(text: str) -> int:
    tok = get_tokenizer()
    if tok is None:
        return max(1, len(text) // 4) if text else 0
    return len(tok.encode(text, add_special_tokens=False))


def count_tokens(text: str | None) -> int:
    """Number of Qwen3 tokens in ``text`` (no special tokens)."""
    if not text:
        return 0
    if len(text) > 20_000:                  # don't pin huge strings in the LRU cache
        tok = get_tokenizer()
        return len(tok.encode(text, add_special_tokens=False)) if tok else len(text) // 4
    return _count_cached(text)
