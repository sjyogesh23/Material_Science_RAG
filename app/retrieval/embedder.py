"""Qwen3-Embedding wrapper with two interchangeable backends (``settings.embed_backend``).

* ``ollama`` (default): ``qwen3-embedding:0.6b`` (Q8_0 GGUF of the same Qwen3-Embedding-0.6B weights) served by
  the local Ollama server. No torch in the API process (saves ~2.2 GB for torch + ~2.3 GB for the fp32 model)
  and ~0.1 s per query on CPU. ``ollama_embed_num_gpu=0`` keeps it on CPU so the GPU stays with the chat LLM.
  Measured against sentence-transformers fp32: cosine(query vectors) >= 0.998 with the same instruction prompt,
  i.e. Ollama does NOT add the prompt itself - we prepend it like the model card says.
* ``sentence_transformers``: the original in-process torch model (``settings.embed_model``).

Queries use the instruction prompt ``"Instruct: {task}\\nQuery: "`` (asymmetric retrieval, model card);
documents are embedded without a prompt. Vectors are L2-normalised (cosine == dot product).
Documents and queries MUST be embedded by the same backend: ``embed_identity()`` goes into the chunk
``content_hash`` so switching backend makes ``python -m app.ingest`` re-embed everything.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from app.config import Settings, get_settings

log = logging.getLogger(__name__)


def query_prompt(settings: Settings) -> str:
    return f"Instruct: {settings.query_instruction}\nQuery: "


def embed_identity(settings: Settings) -> str:
    """Model identity stored (hashed) with every chunk embedding."""
    if settings.embed_backend == "ollama":
        return f"ollama:{settings.ollama_embed_model}"
    return settings.embed_model


# ----------------------------------------------------------------------------- torch helpers
def pick_ingest_device(settings: Settings) -> str:
    """cuda only if requested/auto and enough VRAM is free (the LLM may be loaded), else cpu."""
    want = settings.ingest_device.lower()
    if want == "cpu":
        return "cpu"
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        free, _total = torch.cuda.mem_get_info()
        if want == "cuda" or free >= settings.ingest_min_free_vram_gb * 1024**3:
            return "cuda"
        log.info("only %.2f GB VRAM free -> embedding on CPU", free / 1024**3)
    except Exception:  # pragma: no cover
        pass
    return "cpu"


def resolve_device(want: str, min_free_gb: float) -> str:
    """'auto' -> cuda if available with >= min_free_gb VRAM free (the LLM may be loaded), else cpu."""
    want = (want or "cpu").lower()
    if want != "auto":
        return want
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= min_free_gb * 1024**3:
            return "cuda"
    except Exception:  # pragma: no cover
        pass
    return "cpu"


def torch_dtype(device: str, cpu_dtype: str = "float32") -> Any:
    """float16 on cuda, ``cpu_dtype`` (default float32) on CPU."""
    import torch

    return torch.float16 if device.startswith("cuda") else getattr(torch, cpu_dtype)


def set_cpu_threads(settings: Settings) -> None:
    """torch intra-op threads = physical cores (hyper-threads only add contention for GEMMs)."""
    try:
        import torch

        n = settings.torch_threads
        if n <= 0:
            import psutil

            n = psutil.cpu_count(logical=False) or 4
        torch.set_num_threads(n)
    except Exception:  # pragma: no cover
        pass


def load_model(model_name: str, device: str, cpu_dtype: str = "float32") -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device,
                               model_kwargs={"torch_dtype": torch_dtype(device, cpu_dtype)})


# ----------------------------------------------------------------------------- ollama backend
def ollama_embed_options(settings: Settings, num_gpu: int | None = None) -> dict:
    n = settings.ollama_embed_num_gpu if num_gpu is None else num_gpu
    opts: dict[str, Any] = {"num_ctx": settings.ollama_embed_num_ctx}
    if n >= 0:
        opts["num_gpu"] = n
    return opts


def ollama_client(settings: Settings) -> Any:
    import ollama

    return ollama.Client(host=settings.ollama_host, timeout=settings.ollama_timeout)


def ollama_embed(settings: Settings, texts: list[str], *, num_gpu: int | None = None,
                 client: Any = None) -> tuple[np.ndarray, int]:
    """Embed ``texts`` (already prompted) with Ollama -> (float32 (n, 1024) unit vectors, prompt tokens)."""
    client = client or ollama_client(settings)
    r = client.embed(model=settings.ollama_embed_model, input=texts, truncate=True,
                     options=ollama_embed_options(settings, num_gpu), keep_alive=settings.ollama_keep_alive)
    vecs = np.asarray(r.embeddings, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-12)
    return vecs, int(r.prompt_eval_count or 0)


class OllamaEmbedder:
    """Query encoder backed by the Ollama server (no torch). Same interface as ``QueryEmbedder``."""

    backend = "ollama"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.ollama_embed_model
        self.device = "ollama:" + ("cpu" if settings.ollama_embed_num_gpu == 0 else "auto")
        self.client = ollama_client(settings)
        self.load_seconds = 0.0
        self._last_tokens = 0

    def key(self) -> tuple:
        s = self.settings
        return ("ollama", s.ollama_embed_model, s.ollama_embed_num_gpu, s.ollama_host)

    def warmup(self) -> None:
        t0 = time.perf_counter()
        self.encode("warmup")
        self.load_seconds = time.perf_counter() - t0
        log.info("ollama embedder %s ready in %.1fs", self.model_name, self.load_seconds)

    def encode(self, query: str) -> np.ndarray:
        return self.encode_many([query])[0]

    def encode_many(self, queries: list[str]) -> np.ndarray:
        p = query_prompt(self.settings)
        vecs, self._last_tokens = ollama_embed(self.settings, [p + q for q in queries], client=self.client)
        return vecs

    def prompt_tokens(self, queries: list[str]) -> int:
        """Tokens processed by the last ``encode_many`` (Ollama's prompt_eval_count)."""
        return self._last_tokens


# ----------------------------------------------------------------------------- torch backend
class QueryEmbedder:
    """sentence-transformers query encoder. ``encode(q)`` -> float32 (1024,) unit vector."""

    backend = "sentence_transformers"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.model_name = self.settings.embed_model
        self.requested_device = self.settings.embed_device
        self.device = resolve_device(self.requested_device, self.settings.auto_min_free_vram_gb)
        set_cpu_threads(self.settings)
        t0 = time.perf_counter()
        self.model = load_model(self.model_name, self.device, self.settings.cpu_dtype)
        self.load_seconds = time.perf_counter() - t0
        self._encode_lock = threading.Lock()   # torch CPU inference is not re-entrant-friendly
        log.info("embedder %s on %s loaded in %.1fs", self.model_name, self.device, self.load_seconds)

    def key(self) -> tuple:
        return ("sentence_transformers", self.settings.embed_model, self.settings.embed_device)

    @classmethod
    def get(cls, settings: Settings | None = None) -> "QueryEmbedder | OllamaEmbedder":
        """Backwards-compatible alias of ``get_query_embedder``."""
        return get_query_embedder(settings)

    @classmethod
    def loaded(cls) -> bool:
        return _instance is not None

    def warmup(self) -> None:
        self.encode("warmup")

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text, add_special_tokens=True))

    def encode(self, query: str) -> np.ndarray:
        return self.encode_many([query])[0]

    def encode_many(self, queries: list[str]) -> np.ndarray:
        with self._encode_lock:
            vecs = self.model.encode(queries, prompt=query_prompt(self.settings), normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def prompt_tokens(self, queries: list[str]) -> int:
        """Tokens processed for these queries (prompt included) - for llm_calls accounting."""
        p = query_prompt(self.settings)
        return sum(self.count_tokens(p + q) for q in queries)


# ----------------------------------------------------------------------------- singleton
_instance: "QueryEmbedder | OllamaEmbedder | None" = None
_lock = threading.Lock()


def _wanted_key(s: Settings) -> tuple:
    if s.embed_backend == "ollama":
        return ("ollama", s.ollama_embed_model, s.ollama_embed_num_gpu, s.ollama_host)
    return ("sentence_transformers", s.embed_model, s.embed_device)


def get_query_embedder(settings: Settings | None = None) -> "QueryEmbedder | OllamaEmbedder":
    """Process-wide singleton for the configured backend (rebuilt when the relevant settings change)."""
    global _instance
    s = settings or get_settings()
    with _lock:
        if _instance is None or _instance.key() != _wanted_key(s):
            old, _instance = _instance, None
            del old
            _instance = OllamaEmbedder(s) if s.embed_backend == "ollama" else QueryEmbedder(s)
        elif _instance.settings is not s:
            _instance.settings = s           # pick up non-structural changes (query_instruction ...)
        return _instance


def embedder_loaded() -> bool:
    return _instance is not None
