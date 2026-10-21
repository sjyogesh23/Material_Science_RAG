"""Application settings.

Resolution order (later wins):
  1. dataclass defaults below
  2. environment / ``.env`` (key = field name upper-cased, e.g. ``OOC_THRESHOLD=0.2``)
  3. ``app_settings`` rows in Postgres (key = field name lower-cased, value = jsonb) - runtime overrides
     edited through ``PUT /api/settings``. They are only applied once the DB pool is open (config never
     starts Postgres by itself, to avoid an import cycle with ``app.db``).

``get_settings()`` is cached; call ``reload_settings()`` after changing ``app_settings`` (or ``.env``).
"""
from __future__ import annotations

import contextvars
import dataclasses
import json
import os
import threading
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

DEFAULT_CHAT_MODEL = "qwen3:4b-instruct-2507-q4_K_M"


@dataclass
class Settings:
    # --- paths / services ---------------------------------------------------------------------------
    root: Path = ROOT
    pg_data_dir: Path = ROOT / "pgdata"
    database_url: str | None = None          # if None -> db.start_postgres() (embedded pgserver)
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    ollama_host: str = "http://127.0.0.1:11434"
    # --- LLM ------------------------------------------------------------------------------------------
    chat_model: str = DEFAULT_CHAT_MODEL
    router_model: str = DEFAULT_CHAT_MODEL
    num_ctx: int = 8192
    temperature: float = 0.1
    max_answer_tokens: int = 900
    # --- embeddings / reranking -------------------------------------------------------------------------
    # Backends: "ollama" (default - models served by the Ollama server, no torch in the API process) or
    # "sentence_transformers" (in-process torch models embed_model / rerank_model). Chunks must be embedded with
    # the same backend as queries: switching embed_backend requires `python -m app.ingest` (auto re-embeds).
    embed_backend: str = "ollama"
    ollama_embed_model: str = "qwen3-embedding:0.6b"
    ollama_embed_num_gpu: int = 0             # 0 = CPU (~0.1 s/query; keeps the GPU for the chat LLM), -1 = Ollama decides
    ollama_embed_num_ctx: int = 2048          # chunks are <= ~700 tokens; smaller ctx = less RAM
    rerank_backend: str = "ollama"
    ollama_rerank_model: str = "qwen3-reranker:0.6b-q8_0"   # built by scripts/build_reranker_gguf.py
    ollama_rerank_num_gpu: int = -1           # -1 = Ollama decides (fits on the GPU next to the chat model), 0 = CPU
    ollama_rerank_num_ctx: int = 1024         # >= rerank_max_length
    ollama_keep_alive: str = "2h"             # keep embedder/reranker resident between questions
    ollama_timeout: float = 120.0
    rerank_parallel: int = 1                  # concurrent rerank requests (Ollama serialises them anyway)
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embed_device: str = "cpu"                 # query-time device: cpu | cuda | auto (GPU is reserved for the LLM)
    rerank_model: str = "Qwen/Qwen3-Reranker-0.6B"
    rerank_device: str = "cpu"                # cpu | cuda | auto (cuda only if auto_min_free_vram_gb is free)
    # --- retrieval ----------------------------------------------------------------------------------------
    top_k_dense: int = 40
    top_k_bm25: int = 40
    rrf_k: int = 60
    rerank_top_n: int = 20                    # candidates reranked when the reranker runs on the GPU (~45 ms each)
    rerank_top_n_cpu: int = 5                 # cap when the reranker runs on CPU (~1 s each)
    final_k: int = 6
    ooc_threshold: float = 0.25               # max rerank prob below this -> #OutofContext
    ooc_soft_threshold: float = 0.50          # router says out-of-scope AND max prob < this -> #OutofContext
    grounding_check: bool = False             # pipeline: extra LLM pass checking the answer against its sources
    fast_route: bool = True                   # skip the LLM router when rules + reranker scores already decide
    fast_route_min_score: float = 0.90        # ... i.e. best rerank score >= this (clearly in the textbook) or < ooc_threshold
    refusal_retry: bool = True                # model refused although the topic retrieved strongly -> retry once
    refusal_retry_min_score: float = 0.85     # ... strong = best rerank score >= this (and router in scope)
    # --- sessions / memory ----------------------------------------------------------------------------------
    sessions_dir: Path = ROOT / "sessions"
    page_cache_dir: Path = ROOT / "cache" / "pages"
    memory_budget_tokens: int = 1500
    memory_keep_turns: int = 6
    # --- extras (not in the contract's minimal list, used by ingest / retrieval) ---------------------------
    chunks_path: Path = ROOT / "chunks" / "chunks.jsonl"
    extracted_dir: Path = ROOT / "extracted"
    ingest_device: str = "auto"               # auto = cuda if >= ingest_min_free_vram_gb free, else cpu
    ingest_min_free_vram_gb: float = 2.5
    embed_batch_size: int = 8
    query_instruction: str = ("Given a materials science question, retrieve passages, figures, tables or "
                              "formulas that answer it")
    rerank_instruction: str = ("Given a materials science question, judge whether the passage, figure "
                               "description or table from the textbook answers it")
    cpu_dtype: str = "float32"                # dtype of models on CPU (bfloat16 halves RAM but is ~5x slower here)
    auto_min_free_vram_gb: float = 2.0        # "auto" devices pick cuda only with this much VRAM free at load
    rerank_max_length: int = 512              # tokens per (query, document) pair fed to the reranker
    rerank_batch_size: int = 4
    rerank_quantize: str = "none"             # sentence_transformers backend on CPU: none | int8 (dynamic quantisation)
    torch_threads: int = 0                    # torch intra-op threads for the torch backends (0 = physical cores)
    figure_min_score: float = 0.10            # figures_for() drops figures the reranker scores below this
    hnsw_ef_search: int = 100

    def to_public_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (paths as strings) - used by GET /api/settings."""
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(self).items()}


# Fields that may be overridden at runtime through app_settings (paths / DB location are not).
EDITABLE_FIELDS: tuple[str, ...] = (
    "chat_model", "router_model", "num_ctx", "temperature", "max_answer_tokens",
    "embed_model", "embed_device", "rerank_model", "rerank_device",
    "top_k_dense", "top_k_bm25", "rrf_k", "rerank_top_n", "final_k",
    "ooc_threshold", "ooc_soft_threshold", "memory_budget_tokens", "memory_keep_turns",
    "query_instruction", "rerank_instruction", "figure_min_score", "auto_min_free_vram_gb", "rerank_max_length",
    "grounding_check", "embed_backend", "ollama_embed_model", "rerank_backend", "ollama_rerank_model",
    "ollama_rerank_num_gpu", "rerank_top_n_cpu", "rerank_quantize", "fast_route", "fast_route_min_score",
    "refusal_retry", "refusal_retry_min_score",
)

# Allowed ranges of the numeric editable settings (inclusive). PUT /api/settings rejects values outside them
# with 422 - an out-of-range value (num_ctx=10, temperature=-1, ooc_threshold=5 ...) silently breaks every chat.
RANGES: dict[str, tuple[float, float]] = {
    "num_ctx": (1024, 32768),
    "temperature": (0.0, 2.0),
    "max_answer_tokens": (64, 4096),
    "top_k_dense": (1, 500),
    "top_k_bm25": (1, 500),
    "rrf_k": (1, 1000),
    "rerank_top_n": (1, 200),
    "rerank_top_n_cpu": (1, 200),
    "final_k": (1, 50),
    "ooc_threshold": (0.0, 1.0),
    "ooc_soft_threshold": (0.0, 1.0),
    "figure_min_score": (0.0, 1.0),
    "fast_route_min_score": (0.0, 1.0),
    "refusal_retry_min_score": (0.0, 1.0),
    "memory_budget_tokens": (100, 32768),
    "memory_keep_turns": (1, 100),
    "auto_min_free_vram_gb": (0.0, 128.0),
    "rerank_max_length": (64, 8192),
    "ollama_rerank_num_gpu": (-1, 1000),
}
# String settings restricted to a fixed vocabulary.
CHOICES: dict[str, tuple[str, ...]] = {
    "embed_backend": ("ollama", "sentence_transformers"),
    "rerank_backend": ("ollama", "sentence_transformers"),
    "embed_device": ("cpu", "cuda", "auto"),
    "rerank_device": ("cpu", "cuda", "auto"),
    "rerank_quantize": ("none", "int8"),
}
NON_EMPTY: tuple[str, ...] = ("chat_model", "router_model", "embed_model", "rerank_model", "ollama_embed_model",
                              "ollama_rerank_model")

# ----------------------------------------------------------------------------- answer modes (FEATURES_V2 §4)
# Code defaults of the versioned component ``modes`` (app.versioning seeds version 1 from this dict; the pipeline
# reads ``app.versioning.get_component("modes")`` and falls back to it). ``settings`` keys that are Settings fields
# override them for one answer (``request_overrides``); the others steer the pipeline:
#   think (bool)                  - Ollama thinking for the answer call (streamed as ``thinking`` events, never stored)
#   think_budget_tokens (int)     - extra num_predict for the thinking text
#   self_revision (bool)          - one revision pass when the grounding score (1-5) < revision_min_grounding
DEEP_CHAT_MODEL = "qwen3.5:9b"
MODE_PROFILES: dict[str, dict[str, Any]] = {
    "swift": {
        "id": "swift", "name": "Swift", "tagline": "Time over effect",
        "description": ("Fast answers from the 4B model: the LLM router is skipped when the reranker already settles "
                        "the question, fewer passages are reranked, no grounding check. Usually 3-15 s."),
        "settings": {"chat_model": DEFAULT_CHAT_MODEL, "router_model": DEFAULT_CHAT_MODEL, "fast_route": True,
                     "rerank_top_n": 20, "final_k": 6, "max_answer_tokens": 900, "grounding_check": False,
                     "think": False, "self_revision": False},
    },
    "deep": {
        "id": "deep", "name": "Deep", "tagline": "Effect over time",
        "description": ("Careful answers from the 9B model with thinking: the router always runs, more passages are "
                        "reranked and used, the answer is checked against its sources and revised once if it is "
                        "not well grounded. Much slower on this laptop (about 2-5 minutes)."),
        "warning": "Deep is slower: about 2-5 minutes per answer on this laptop.",
        "settings": {"chat_model": DEEP_CHAT_MODEL, "router_model": DEEP_CHAT_MODEL, "fast_route": False,
                     "rerank_top_n": 30, "final_k": 8, "max_answer_tokens": 1400, "num_ctx": 12288,
                     "grounding_check": True, "think": True, "think_budget_tokens": 1500,
                     "self_revision": True, "revision_min_grounding": 3},
    },
}
DEFAULT_MODE = "swift"

_HINTS = typing.get_type_hints(Settings)
_lock = threading.Lock()
_cached: Settings | None = None
# per-request settings (answer mode): a copy of the global settings with the mode's overrides, visible to everything
# running in the same context (incl. asyncio.to_thread workers, which copy the context)
_request_settings: contextvars.ContextVar[Settings | None] = contextvars.ContextVar("newrag_request_settings",
                                                                                    default=None)


def _coerce(name: str, value: Any) -> Any:
    """Convert an env string / jsonb value to the field's declared type."""
    hint = _HINTS[name]
    if value is None:
        return None
    base = hint
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        base = args[0] if args else str
        if isinstance(value, str) and value.strip() == "":
            return None
    if base is bool:
        return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")
    if base is int:
        return int(value)
    if base is float:
        return float(value)
    if base is Path:
        p = Path(value)
        return p if p.is_absolute() else ROOT / p
    return str(value)


def _from_env() -> Settings:
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(Settings):
        raw = os.getenv(f.name.upper())
        if raw is not None and f.name != "root":
            kwargs[f.name] = _coerce(f.name, raw)
    return Settings(**kwargs)


def _db_overrides() -> dict[str, Any]:
    """app_settings rows, only if the DB pool is already open (never starts Postgres)."""
    try:
        from app import db
        pool = db.peek_pool()
        if pool is None:
            return {}
        with pool.connection() as conn:
            rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
    except Exception:
        return {}
    out: dict[str, Any] = {}
    for key, value in rows:
        key = str(key).lower()
        if key in EDITABLE_FIELDS:
            try:
                out[key] = _coerce(key, value)
            except (TypeError, ValueError):
                pass
    return out


def get_settings() -> Settings:
    override = _request_settings.get()
    if override is not None:
        return override
    return _global_settings()


def _global_settings() -> Settings:
    global _cached
    with _lock:
        if _cached is not None:
            return _cached
        s = _from_env()
        for k, v in _db_overrides().items():
            setattr(s, k, v)
    # active retrieval_params component (app.versioning) on top of .env / app_settings; outside the lock because
    # versioning may open pool connections whose configure hook calls get_settings()
    try:
        from app import versioning
        for k, v in versioning.retrieval_overrides().items():
            if k in _HINTS:
                setattr(s, k, _coerce(k, v))
    except Exception:  # noqa: BLE001 - versioning unavailable / DB down -> plain settings
        pass
    with _lock:
        if _cached is None:
            _cached = s
        return _cached


def reload_settings() -> Settings:
    """Drop the cache and re-read .env + app_settings."""
    global _cached
    with _lock:
        _cached = None
    load_dotenv(ROOT / ".env", override=True)
    return _global_settings()


_FIELD_NAMES = {f.name for f in dataclasses.fields(Settings)}


def request_overrides(changes: dict[str, Any] | None) -> contextvars.Token:
    """Make ``get_settings()`` return the global settings + ``changes`` (Settings fields only; unknown keys and
    None values are ignored) in the current context. Returns a token for ``reset_request_overrides``."""
    base = _global_settings()
    clean = {}
    for k, v in (changes or {}).items():
        if k in _FIELD_NAMES and k != "root" and v is not None:
            try:
                clean[k] = _coerce(k, v)
            except (TypeError, ValueError):
                pass
    return _request_settings.set(dataclasses.replace(base, **clean) if clean else base)


def _normalise_modes(content: Any) -> dict[str, dict[str, Any]]:
    """{id: profile} from a ``modes`` component (dict keyed by id, or a list of profiles)."""
    items = content.values() if isinstance(content, dict) else content if isinstance(content, list) else []
    out: dict[str, dict[str, Any]] = {}
    for p in items:
        if isinstance(p, dict) and p.get("id") and isinstance(p.get("settings", {}), dict):
            out[str(p["id"])] = {**p, "settings": dict(p.get("settings") or {})}
    return out


def get_modes() -> dict[str, dict[str, Any]]:
    """Answer-mode profiles: the active version of the versioned component ``modes`` if available, else
    MODE_PROFILES. Profiles missing from the component fall back to the code defaults."""
    modes = {k: {**v, "settings": dict(v["settings"])} for k, v in MODE_PROFILES.items()}
    try:
        from app import versioning  # versioned components (optional at import time)
        content = versioning.get_component("modes")
    except Exception:  # noqa: BLE001 - ImportError / DB down / unknown component -> code defaults
        content = None
    if content:
        modes.update(_normalise_modes(content))
    return modes


def get_mode(mode_id: str | None) -> dict[str, Any]:
    modes = get_modes()
    return modes.get((mode_id or DEFAULT_MODE).lower().strip()) or modes.get(DEFAULT_MODE) or MODE_PROFILES[DEFAULT_MODE]


def reset_request_overrides(token: contextvars.Token | None) -> None:
    if token is None:
        return
    try:
        _request_settings.reset(token)
    except ValueError:      # created in another context (async generator resumed elsewhere)
        _request_settings.set(None)


def validate_value(name: str, value: Any) -> Any:
    """Coerce + range-check one editable setting. Raises ValueError with a readable message."""
    name = name.lower()
    if name not in EDITABLE_FIELDS:
        raise KeyError(f"setting {name!r} is not editable")
    hint = _HINTS[name]
    nullable = type(None) in typing.get_args(hint)
    if value is None or (isinstance(value, str) and not value.strip() and hint in (int, float)):
        if nullable:
            return None
        raise ValueError(f"{name} cannot be empty")  # e.g. num_ctx=null would break every chat
    if hint in (int, float):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a number, not true/false")
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number (got {value!r})") from None
        if num != num or num in (float("inf"), float("-inf")):
            raise ValueError(f"{name} must be a finite number")
        if hint is int and not num.is_integer():
            raise ValueError(f"{name} must be a whole number (got {value!r})")
        value = int(num) if hint is int else num
        if name in RANGES:
            lo, hi = RANGES[name]
            if not lo <= value <= hi:
                fmt = (lambda x: str(int(x))) if hint is int else (lambda x: f"{x:g}")
                raise ValueError(f"{name} must be between {fmt(lo)} and {fmt(hi)} (got {value!r})")
        return value
    value = _coerce(name, value)
    if name in CHOICES and value not in CHOICES[name]:
        raise ValueError(f"{name} must be one of {', '.join(CHOICES[name])} (got {value!r})")
    if name in NON_EMPTY and not str(value).strip():
        raise ValueError(f"{name} cannot be empty")
    return value


def save_overrides(changes: dict[str, Any]) -> Settings:
    """Validate + persist runtime overrides into app_settings, then reload. Unknown keys raise KeyError,
    invalid / out-of-range values raise ValueError (nothing is saved then)."""
    from app import db

    clean: dict[str, Any] = {}
    errors: list[str] = []
    for k, v in changes.items():
        k = k.lower()
        if k not in EDITABLE_FIELDS:
            raise KeyError(f"setting {k!r} is not editable")
        try:
            clean[k] = validate_value(k, v)
        except ValueError as e:
            errors.append(str(e))
    if not errors:  # cross-field: the answer budget must leave room for the prompt
        cur = get_settings()
        num_ctx = clean.get("num_ctx", cur.num_ctx)
        max_ans = clean.get("max_answer_tokens", cur.max_answer_tokens)
        if max_ans > num_ctx - 512:
            errors.append(f"max_answer_tokens ({max_ans}) must be at most num_ctx - 512 ({num_ctx - 512})")
        lo = clean.get("ooc_threshold", cur.ooc_threshold)
        hi = clean.get("ooc_soft_threshold", cur.ooc_soft_threshold)
        if lo > hi:
            errors.append(f"ooc_threshold ({lo:g}) must not exceed ooc_soft_threshold ({hi:g})")
    if errors:
        raise ValueError("; ".join(errors))
    with db.get_pool().connection() as conn:
        for k, v in clean.items():
            conn.execute(
                "INSERT INTO app_settings(key, value, updated_at) VALUES (%s, %s::jsonb, now()) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                (k, json.dumps(v)))
    return reload_settings()
