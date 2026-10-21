"""Qwen3-Reranker -> relevance probability P(yes) in 0..1, with two backends (``settings.rerank_backend``).

``ollama`` (default)
    ``qwen3-reranker:0.6b-q8_0`` - a Q8_0 GGUF of Qwen/Qwen3-Reranker-0.6B built locally by
    ``scripts/build_reranker_gguf.py`` (the public ``dengcao/Qwen3-Reranker-0.6B`` upload has all-zero weights
    and Ollama 0.40 cannot import safetensors without MLX). Each (query, document) pair is sent to
    ``/api/generate`` as the model card's raw chat prompt with ``num_predict=1, logprobs, top_logprobs=20``;
    score = softmax([logprob("no"), logprob("yes")])[yes] - the model card's formula. When one of the two
    tokens is not in the top-20 its logprob is taken as (lowest listed - 1), so the score is an *upper bound*
    for clearly irrelevant pairs (≈0.05-0.10 instead of ≈0.001) - still far below in-scope scores (≥0.95).
    The runner needs only ~0.8 GB and, with the chat model loaded, still fits 100 % on the RTX 3050 in the
    VRAM Ollama leaves free (~20 pairs/s); on CPU llama.cpp does ~330 tok/s (~1 s/pair), so when Ollama
    places it (partly) on CPU ``rerank_top_n_cpu`` caps the candidate count.
``sentence_transformers``
    in-process ``CrossEncoder`` (torch). CPU fp32 costs ~2.3 GB RAM + ~2.2 GB for torch and ~1.2 s/pair;
    ``rerank_quantize="int8"`` loads bf16 and dynamically quantises the decoder Linear layers to int8
    layer by layer (peak RAM ≈ bf16 model), threads = physical cores, length-sorted batches.

Both backends expose ``score(query, docs)``, ``count_tokens``, ``model_name``, ``device``, ``max_candidates()``.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from app.config import Settings, get_settings
from app.retrieval.embedder import ollama_client, resolve_device, set_cpu_threads, torch_dtype

log = logging.getLogger(__name__)

SYSTEM = ('Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
          'Note that the answer can only be "yes" or "no".')
PREFIX = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
TEMPLATE_TOKENS = 70          # prefix + suffix + "<Instruct>: / <Query>: / <Document>: " labels (approx.)


def _truncate(text: str, max_tokens: int) -> str:
    """Cut ``text`` to ~max_tokens Qwen3 tokens (tokenizer if available, else 4 chars/token)."""
    if max_tokens <= 0:
        return ""
    if len(text) <= max_tokens * 3:            # cheap path: certainly short enough
        return text
    try:
        from app.tokens import get_tokenizer

        tok = get_tokenizer()
    except Exception:  # pragma: no cover
        tok = None
    if tok is None:
        return text[: max_tokens * 4]
    ids = tok.encode(text, add_special_tokens=False)
    return text if len(ids) <= max_tokens else tok.decode(ids[:max_tokens])


def _count(text: str) -> int:
    try:
        from app.tokens import count_tokens

        return count_tokens(text)
    except Exception:  # pragma: no cover
        return len(text) // 4


# ----------------------------------------------------------------------------- ollama backend
class OllamaReranker:
    backend = "ollama"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.ollama_rerank_model
        self.client = ollama_client(settings)
        self.device = "ollama"
        self.load_seconds = 0.0
        self._placement: tuple[float, str] = (0.0, "unknown")     # (checked_at, "gpu"|"cpu"|"partial"|"unknown")
        self._pool = ThreadPoolExecutor(max(1, settings.rerank_parallel), thread_name_prefix="rerank")
        self._last_tokens = 0

    def key(self) -> tuple:
        s = self.settings
        return ("ollama", s.ollama_rerank_model, s.ollama_rerank_num_gpu, s.ollama_rerank_num_ctx, s.ollama_host,
                s.rerank_parallel)

    def _options(self) -> dict:
        s = self.settings
        opts: dict[str, Any] = {"num_predict": 1, "temperature": 0, "num_ctx": s.ollama_rerank_num_ctx}
        if s.ollama_rerank_num_gpu >= 0:
            opts["num_gpu"] = s.ollama_rerank_num_gpu
        return opts

    # -- placement -----------------------------------------------------------------------------------------
    def placement(self, max_age: float = 30.0) -> str:
        """Where Ollama runs the reranker right now: gpu | partial | cpu | unknown (cached ``max_age`` s)."""
        t, where = self._placement
        if time.monotonic() - t < max_age:
            return where
        where = "unknown"
        try:
            for m in self.client.ps().models:
                if m.model == self.model_name or m.name == self.model_name:
                    vram, size = int(m.size_vram or 0), int(m.size or 0)
                    where = "gpu" if size and vram >= size else ("cpu" if vram == 0 else "partial")
        except Exception:  # pragma: no cover
            pass
        self._placement = (time.monotonic(), where)
        self.device = f"ollama:{where}"
        return where

    def max_candidates(self, wanted: int) -> int:
        """Cap the candidate count when the runner is not fully on the GPU (CPU llama.cpp ≈ 1 s/pair)."""
        if self.placement() in ("cpu", "partial"):
            return min(wanted, self.settings.rerank_top_n_cpu)
        return wanted

    # -- scoring -------------------------------------------------------------------------------------------
    def _prompt(self, instruction: str, query: str, doc: str) -> str:
        budget = self.settings.rerank_max_length - TEMPLATE_TOKENS - _count(instruction) - _count(query)
        doc = _truncate(doc, max(32, budget))
        return f"{PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}{SUFFIX}"

    def _score_one(self, prompt: str) -> tuple[float, int]:
        r = self.client.generate(model=self.model_name, prompt=prompt, raw=True, logprobs=True, top_logprobs=20,
                                 options=self._options(), keep_alive=self.settings.ollama_keep_alive)
        top = {t.token: t.logprob for t in (r.logprobs[0].top_logprobs if r.logprobs else [])}
        if not top:                                     # pragma: no cover - logprobs unsupported
            return (1.0 if (r.response or "").strip().lower().startswith("yes") else 0.0), int(r.prompt_eval_count or 0)
        floor = min(top.values()) - 1.0
        y, n = top.get("yes", floor), top.get("no", floor)
        return 1.0 / (1.0 + math.exp(n - y)), int(r.prompt_eval_count or 0)

    def warmup(self) -> None:
        t0 = time.perf_counter()
        self._score_one(self._prompt(self.settings.rerank_instruction, "warmup", "warmup"))
        self.load_seconds = time.perf_counter() - t0
        self._placement = (0.0, "unknown")
        log.info("ollama reranker %s ready in %.1fs (%s)", self.model_name, self.load_seconds, self.placement())

    def score(self, query: str, docs: list[str], instruction: str | None = None) -> list[float]:
        if not docs:
            return []
        ins = instruction or self.settings.rerank_instruction
        prompts = [self._prompt(ins, query, d) for d in docs]
        out = list(self._pool.map(self._score_one, prompts))
        self._last_tokens = sum(t for _, t in out)
        return [p for p, _ in out]

    def count_tokens(self, query: str, docs: list[str]) -> int:
        """Tokens processed by the last ``score`` call (Ollama prompt_eval_count)."""
        return self._last_tokens


# ----------------------------------------------------------------------------- torch backend
def _quantize_int8(cross_encoder: Any) -> None:
    """Dynamic int8 quantisation of every decoder layer's Linear modules, one layer at a time."""
    import gc

    import torch

    decoder = next(m for m in cross_encoder.modules()
                   if isinstance(getattr(m, "layers", None), torch.nn.ModuleList))
    for layer in decoder.layers:
        layer.float()
        torch.ao.quantization.quantize_dynamic(layer, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        gc.collect()
    cross_encoder.float()        # embeddings / norms / tied head back to fp32 for the int8 kernels


class Reranker:
    """sentence-transformers cross-encoder (torch)."""

    backend = "sentence_transformers"

    def __init__(self, settings: Settings | None = None):
        import torch
        from sentence_transformers import CrossEncoder

        self.settings = settings or get_settings()
        self.model_name = self.settings.rerank_model
        self.requested_device = self.settings.rerank_device
        self.device = resolve_device(self.requested_device, self.settings.auto_min_free_vram_gb)
        quant = self.settings.rerank_quantize if self.device == "cpu" else "none"
        set_cpu_threads(self.settings)
        dtype = torch.bfloat16 if quant == "int8" else torch_dtype(self.device, self.settings.cpu_dtype)
        t0 = time.perf_counter()
        self.model: Any = CrossEncoder(self.model_name, device=self.device,
                                       max_length=self.settings.rerank_max_length,
                                       model_kwargs={"torch_dtype": dtype})
        if quant == "int8":
            _quantize_int8(self.model)
            self.device = "cpu:int8"
        self.load_seconds = time.perf_counter() - t0
        self._sigmoid = torch.nn.Sigmoid()
        self._score_lock = threading.Lock()
        log.info("reranker %s on %s loaded in %.1fs", self.model_name, self.device, self.load_seconds)

    def key(self) -> tuple:
        s = self.settings
        return ("sentence_transformers", s.rerank_model, s.rerank_device, s.rerank_quantize, s.rerank_max_length)

    @classmethod
    def get(cls, settings: Settings | None = None) -> "Reranker | OllamaReranker":
        """Backwards-compatible alias of ``get_reranker``."""
        return get_reranker(settings)

    @classmethod
    def loaded(cls) -> bool:
        return _instance is not None

    def max_candidates(self, wanted: int) -> int:
        return wanted if self.device.startswith("cuda") else min(wanted, self.settings.rerank_top_n_cpu)

    def warmup(self) -> None:
        self.score("warmup", ["warmup"])

    def score(self, query: str, docs: list[str], instruction: str | None = None) -> list[float]:
        if not docs:
            return []
        import torch

        order = sorted(range(len(docs)), key=lambda i: len(docs[i]))      # less padding per batch
        with self._score_lock, torch.inference_mode():
            probs = self.model.predict([(query, docs[i]) for i in order],
                                       prompt=instruction or self.settings.rerank_instruction,
                                       activation_fn=self._sigmoid,
                                       batch_size=self.settings.rerank_batch_size,
                                       show_progress_bar=False, convert_to_numpy=True)
        flat = np.asarray(probs, dtype=np.float32).reshape(-1)
        out = [0.0] * len(docs)
        for pos, i in enumerate(order):
            out[i] = float(flat[pos])
        return out

    def count_tokens(self, query: str, docs: list[str]) -> int:
        """Approximate tokens processed (query + doc + ~60 template tokens per pair, capped at max_length)."""
        tok = self.model.tokenizer
        q = len(tok.encode(query, add_special_tokens=False))
        cap = self.settings.rerank_max_length
        return sum(min(cap, q + len(tok.encode(d, add_special_tokens=False)) + 60) for d in docs)


# ----------------------------------------------------------------------------- singleton
_instance: "Reranker | OllamaReranker | None" = None
_lock = threading.Lock()


def _wanted_key(s: Settings) -> tuple:
    if s.rerank_backend == "ollama":
        return ("ollama", s.ollama_rerank_model, s.ollama_rerank_num_gpu, s.ollama_rerank_num_ctx, s.ollama_host,
                s.rerank_parallel)
    return ("sentence_transformers", s.rerank_model, s.rerank_device, s.rerank_quantize, s.rerank_max_length)


def get_reranker(settings: Settings | None = None) -> "Reranker | OllamaReranker":
    """Process-wide singleton for the configured backend (rebuilt when the relevant settings change)."""
    global _instance
    s = settings or get_settings()
    with _lock:
        if _instance is None or _instance.key() != _wanted_key(s):
            old, _instance = _instance, None
            del old
            _instance = OllamaReranker(s) if s.rerank_backend == "ollama" else Reranker(s)
        elif _instance.settings is not s:
            _instance.settings = s           # pick up non-structural changes (instruction, max_length ...)
        return _instance


def reranker_loaded() -> bool:
    return _instance is not None
