"""Versioned components (FEATURES_V2 §6): prompts, retrieval parameters and mode profiles stored as versions in
the ``component_versions`` table, one active version per component.

* ``get_component(name)`` -> active content. In-process cache, refreshed after ``CACHE_TTL_S`` seconds (so an
  eval process sees a version activated in the API process within ~5 s) and invalidated immediately on
  create / activate in this process. Never starts Postgres: without an open pool (``db.peek_pool()``) or on any
  DB error it returns the CODE DEFAULT, so behaviour is identical to the pre-versioning code.
* Version 1 of every component is seeded from the code defaults (``created_by='code'``). When a code default
  later changes, a new ``code`` version is added; it becomes active only if the active version is itself a
  ``code`` version (a version chosen in the UI is never overridden by a code change).
* ``active_versions()`` -> ``{component: version}`` for llm_calls.meta; ``retrieval_overrides()`` -> the active
  ``retrieval_params`` keys whose value differs from the Settings dataclass default (empty for the seed).
"""
from __future__ import annotations

import copy
import dataclasses
import difflib
import json
import logging
import string
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SQL_PATH = Path(__file__).with_name("versioning.sql")
CACHE_TTL_S = 5.0
_LOCK_ID = 7_142_002                 # pg_advisory_xact_lock key for schema + seeding
MAX_TEXT_CHARS = 20_000

# retrieval knobs exposed as the `retrieval_params` component (all are Settings fields)
RETRIEVAL_KEYS: tuple[str, ...] = (
    "top_k_dense", "top_k_bm25", "rrf_k", "rerank_top_n", "rerank_top_n_cpu", "final_k",
    "ooc_threshold", "ooc_soft_threshold", "fast_route_min_score", "refusal_retry_min_score", "figure_min_score",
)

# component -> metadata. kind: text (a string) | json (an object). Order = UI order.
COMPONENTS: dict[str, dict[str, Any]] = {
    "answer_system_prompt": {"label": "Answer system prompt", "kind": "text",
                             "description": "Rules of the answering LLM (grounding, citations, #OutofContext)."},
    "router_prompt": {"label": "Router prompt", "kind": "text",
                      "description": "System prompt of the Jev-style router (scope, intent, standalone query)."},
    "form_notes": {"label": "Form notes", "kind": "json",
                   "description": "Per-question-form NOTE lines (analogy, presupposition, ambiguous, injection) "
                                  "and the page / citation-count / quote / missing-reference / refusal-retry notes. "
                                  "Templates keep their {placeholders}."},
    "table_rules": {"label": "Table format rules", "kind": "text",
                    "description": "FORMAT line used when the router picks answer_format = table."},
    "compaction_prompt": {"label": "Memory compaction prompt", "kind": "json",
                          "description": "System ({max_words}) and user ({summary}, {turns}) templates of the "
                                         "session-memory compaction call."},
    "title_prompt": {"label": "Title prompt", "kind": "text",
                     "description": "System prompt that names a new chat session."},
    "grounding_prompt": {"label": "Grounding check", "kind": "json",
                         "description": "Question asked by the grounding check and the caution appended to weakly "
                                        "grounded answers."},
    "retrieval_params": {"label": "Retrieval parameters", "kind": "json",
                         "description": "top-k, RRF k, rerank_top_n, final_k and score thresholds."},
    "modes": {"label": "Answer modes", "kind": "json",
              "description": "Swift / Deep mode profiles (models, budgets, router, grounding)."},
}

_lock = threading.RLock()
_cache: dict[str, tuple[int, Any]] = {}      # component -> (version, content) of the ACTIVE version
_cache_t = 0.0
_schema_ok = False


class ComponentError(ValueError):
    """Invalid component content (message is shown to the user)."""


# ----------------------------------------------------------------------------- code defaults
def _settings_defaults() -> dict[str, Any]:
    from app.config import Settings
    return {f.name: f.default for f in dataclasses.fields(Settings)}


def retrieval_defaults() -> dict[str, Any]:
    d = _settings_defaults()
    return {k: d[k] for k in RETRIEVAL_KEYS}


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


def mode_defaults() -> Any | None:
    """``app.config.MODE_PROFILES`` (added by the modes feature) as plain JSON, or None if it does not exist yet."""
    try:
        from app import config
        prof = getattr(config, "MODE_PROFILES", None)
    except Exception:  # noqa: BLE001
        return None
    if not prof:
        return None
    if dataclasses.is_dataclass(prof):
        prof = dataclasses.asdict(prof)
    elif isinstance(prof, dict):
        prof = {k: (dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v) for k, v in prof.items()}
    return _jsonable(prof)


def code_default(name: str) -> Any | None:
    if name not in COMPONENTS:
        raise KeyError(name)
    if name == "retrieval_params":
        return retrieval_defaults()
    if name == "modes":
        return mode_defaults()
    from app.llm import prompts
    return copy.deepcopy(prompts.DEFAULT_COMPONENTS[name])


def code_defaults() -> dict[str, Any]:
    out = {}
    for name in COMPONENTS:
        d = code_default(name)
        if d is not None:
            out[name] = d
    return out


# ----------------------------------------------------------------------------- schema + seed
def _seed(conn) -> list[str]:
    """Insert a `code` version for every component whose code default is new/changed (caller holds the lock)."""
    seeded = []
    for name, default in code_defaults().items():
        default = _jsonable(default)
        rows = conn.execute(
            "SELECT version, content, created_by, is_active FROM component_versions WHERE component = %s "
            "ORDER BY version DESC", (name,)).fetchall()
        last_code = next((r for r in rows if r[2] == "code"), None)
        if last_code is not None and last_code[1] == default:
            continue
        active = next((r for r in rows if r[3]), None)
        version = (rows[0][0] + 1) if rows else 1
        make_active = active is None or active[2] == "code"
        if make_active and active is not None:
            conn.execute("UPDATE component_versions SET is_active = false WHERE component = %s AND is_active",
                         (name,))
        conn.execute(
            "INSERT INTO component_versions(component, version, content, note, created_by, is_active) "
            "VALUES (%s, %s, %s::jsonb, %s, 'code', %s)",
            (name, version, json.dumps(default),
             "seeded from code defaults" if version == 1 else "code default changed", make_active))
        seeded.append(f"{name}@{version}")
    if seeded:
        log.info("component versions seeded: %s", ", ".join(seeded))
    return seeded


def ensure_schema(pool=None, *, force: bool = False) -> None:
    """Create the table (versioning.sql) and seed code defaults. Idempotent, safe across processes."""
    global _schema_ok
    if _schema_ok and not force:
        return
    if pool is None:
        from app import db
        pool = db.get_pool()
    with _lock:
        if _schema_ok and not force:
            return
        with pool.connection() as conn:
            with conn.transaction():
                conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_ID,))
                conn.execute(SQL_PATH.read_text(encoding="utf-8"))
                _seed(conn)
        _schema_ok = True
        _invalidate()


def _invalidate() -> None:
    global _cache_t
    _cache_t = 0.0


def _refresh(pool) -> None:
    global _cache, _cache_t
    ensure_schema(pool)
    with pool.connection() as conn:
        rows = conn.execute("SELECT component, version, content FROM component_versions WHERE is_active").fetchall()
    cache = {r[0]: (int(r[1]), r[2]) for r in rows}
    if any(n not in cache for n in code_defaults()):      # e.g. MODE_PROFILES appeared after start-up
        ensure_schema(pool, force=True)
        with pool.connection() as conn:
            rows = conn.execute(
                "SELECT component, version, content FROM component_versions WHERE is_active").fetchall()
        cache = {r[0]: (int(r[1]), r[2]) for r in rows}
    _cache = cache
    _cache_t = time.monotonic()


def _active(name: str | None = None) -> dict[str, tuple[int, Any]]:
    """Active (version, content) per component from the cache; {} without an open DB pool / on DB errors."""
    global _cache_t
    from app import db
    pool = db.peek_pool()
    if pool is None:
        return {}
    with _lock:
        if time.monotonic() - _cache_t > CACHE_TTL_S:
            try:
                _refresh(pool)
            except Exception as e:  # noqa: BLE001 - fall back to the (possibly stale) cache / code defaults
                _cache_t = time.monotonic()      # don't hammer a broken DB on every prompt lookup
                log.warning("component cache refresh failed: %s", e)
                return dict(_cache)
        return dict(_cache)


# ----------------------------------------------------------------------------- runtime API
def get_component(name: str) -> Any:
    """Active content of a component (code default without DB). Raises KeyError for unknown names."""
    if name not in COMPONENTS:
        raise KeyError(f"unknown component {name!r}")
    hit = _active().get(name)
    if hit is not None:
        return copy.deepcopy(hit[1])
    return code_default(name)


def active_versions() -> dict[str, int]:
    """{component: active version} - attach to llm_calls.meta (key ``components``). {} without DB."""
    return {k: v[0] for k, v in sorted(_active().items())}


def retrieval_overrides() -> dict[str, Any]:
    """Active retrieval_params entries that differ from the Settings dataclass defaults ({} for the seed).
    Merge on top of get_settings() (after .env / app_settings) - see the report for the call site."""
    active = get_component("retrieval_params") or {}
    base = retrieval_defaults()
    return {k: v for k, v in active.items() if k in base and v != base[k]}


# ----------------------------------------------------------------------------- validation
def _fields(template: str) -> set[str]:
    try:
        return {f for _, f, _, _ in string.Formatter().parse(template) if f is not None}
    except ValueError as e:
        raise ComponentError(f"invalid template braces: {e}") from None


def _check_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ComponentError(f"{where} must be a non-empty string")
    if len(value) > MAX_TEXT_CHARS:
        raise ComponentError(f"{where} is too long ({len(value)} > {MAX_TEXT_CHARS} characters)")
    return value


def _check_like(default: Any, value: Any, where: str) -> Any:
    """Structure check against the code default: dict keys must exist in the default, scalar types must match,
    editable Settings keys are range-checked with config.validate_value."""
    from app.config import EDITABLE_FIELDS, validate_value
    if isinstance(default, dict):
        if not isinstance(value, dict):
            raise ComponentError(f"{where} must be an object")
        unknown = sorted(set(value) - set(default))
        if unknown:
            raise ComponentError(f"{where}: unknown key(s) {', '.join(unknown)}")
        out = {}
        for k, v in value.items():
            path = f"{where}.{k}"
            if k in EDITABLE_FIELDS and not isinstance(default[k], (dict, list)):
                try:
                    out[k] = validate_value(k, v)
                except (KeyError, ValueError) as e:
                    raise ComponentError(f"{path}: {e}") from None
            else:
                out[k] = _check_like(default[k], v, path)
        return out
    if isinstance(default, list):
        if not isinstance(value, list):
            raise ComponentError(f"{where} must be a list")
        if default:
            return [_check_like(default[0], v, f"{where}[{i}]") for i, v in enumerate(value)]
        return value
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ComponentError(f"{where} must be true or false")
        return value
    if isinstance(default, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ComponentError(f"{where} must be a number")
        if isinstance(default, int) and not float(value).is_integer():
            raise ComponentError(f"{where} must be a whole number")
        return int(value) if isinstance(default, int) else float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ComponentError(f"{where} must be a string")
        return value
    return value                        # default None: anything JSON


def validate(name: str, content: Any) -> Any:
    """-> cleaned content, or raises ComponentError / KeyError (unknown component)."""
    meta = COMPONENTS.get(name)
    if meta is None:
        raise KeyError(f"unknown component {name!r}")
    default = code_default(name)
    if meta["kind"] == "text":
        text = _check_text(content, name)
        if name == "answer_system_prompt" and "#OutofContext" not in text:
            raise ComponentError("answer_system_prompt must keep the #OutofContext rule")
        return text
    if name == "retrieval_params":
        if not isinstance(content, dict) or not content:
            raise ComponentError("retrieval_params must be a non-empty object")
        clean = _check_like(default, content, name)
        merged = {**default, **clean}
        if merged["ooc_threshold"] > merged["ooc_soft_threshold"]:
            raise ComponentError(f"ooc_threshold ({merged['ooc_threshold']:g}) must not exceed "
                                 f"ooc_soft_threshold ({merged['ooc_soft_threshold']:g})")
        return clean
    if name == "modes":
        if not isinstance(content, (dict, list)) or not content:
            raise ComponentError("modes must be a non-empty object")
        if default is None:
            return content
        clean = _check_like(default, content, name)
        if isinstance(default, dict) and set(clean) != set(default):
            missing = sorted(set(default) - set(clean))
            raise ComponentError(f"modes: missing mode(s) {', '.join(missing)}")
        return clean
    # json prompt templates: same keys as the default, non-empty strings, placeholders a subset of the default's
    if not isinstance(content, dict):
        raise ComponentError(f"{name} must be an object with keys {', '.join(default)}")
    unknown, missing = sorted(set(content) - set(default)), sorted(set(default) - set(content))
    if unknown:
        raise ComponentError(f"{name}: unknown key(s) {', '.join(unknown)}")
    if missing:
        raise ComponentError(f"{name}: missing key(s) {', '.join(missing)}")
    out = {}
    for k, v in content.items():
        text = _check_text(v, f"{name}.{k}")
        allowed = _fields(default[k])
        if allowed:                     # only templates that are .format()-ed need valid placeholders
            extra = sorted(_fields(text) - allowed)
            if extra:
                raise ComponentError(f"{name}.{k}: unknown placeholder(s) {', '.join('{' + e + '}' for e in extra)}"
                                     f"; allowed: {', '.join('{' + a + '}' for a in sorted(allowed))}")
            try:
                text.format(**{a: "x" for a in allowed})
            except (IndexError, KeyError, ValueError) as e:
                raise ComponentError(f"{name}.{k}: invalid template ({e})") from None
        out[k] = text
    return out


# ----------------------------------------------------------------------------- CRUD (used by arch_routes)
def _pool():
    from app import db
    pool = db.get_pool()
    ensure_schema(pool)
    return pool


def _row(r) -> dict[str, Any]:
    return {"component": r[0], "version": int(r[1]), "content": r[2], "note": r[3],
            "created_at": r[4].isoformat() if r[4] else None, "created_by": r[5], "is_active": bool(r[6])}


_COLS = "component, version, content, note, created_at, created_by, is_active"


def list_components() -> list[dict[str, Any]]:
    with _pool().connection() as conn:
        rows = conn.execute(
            "SELECT component, count(*), max(version), max(created_at), "
            "max(version) FILTER (WHERE is_active) FROM component_versions GROUP BY component").fetchall()
    stats = {r[0]: r for r in rows}
    out = []
    for name, meta in COMPONENTS.items():
        r = stats.get(name)
        out.append({"component": name, **meta, "active_version": r[4] if r else None,
                    "latest_version": r[2] if r else None, "version_count": int(r[1]) if r else 0,
                    "updated_at": r[3].isoformat() if r and r[3] else None,
                    "available": r is not None})
    return out


def get_history(name: str) -> dict[str, Any]:
    if name not in COMPONENTS:
        raise KeyError(name)
    with _pool().connection() as conn:
        rows = conn.execute(f"SELECT {_COLS} FROM component_versions WHERE component = %s ORDER BY version DESC",
                            (name,)).fetchall()
    versions = [_row(r) for r in rows]
    active = next((v for v in versions if v["is_active"]), None)
    return {"component": name, **COMPONENTS[name], "active": active, "default": code_default(name),
            "versions": [{k: v[k] for k in ("version", "note", "created_at", "created_by", "is_active")}
                         for v in versions]}


def get_version(name: str, version: int) -> dict[str, Any] | None:
    if name not in COMPONENTS:
        raise KeyError(name)
    with _pool().connection() as conn:
        r = conn.execute(f"SELECT {_COLS} FROM component_versions WHERE component = %s AND version = %s",
                         (name, int(version))).fetchone()
    return _row(r) if r else None


def _after_change(name: str) -> None:
    _invalidate()
    if name in ("retrieval_params", "modes"):
        try:
            from app.config import reload_settings
            reload_settings()           # lets a get_settings() that merges retrieval_overrides() pick it up
        except Exception:  # noqa: BLE001
            pass


def create_version(name: str, content: Any, note: str | None = None, *, created_by: str = "ui",
                   activate: bool = True) -> dict[str, Any]:
    clean = validate(name, content)
    with _pool().connection() as conn:
        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_ID,))
            v = conn.execute("SELECT coalesce(max(version), 0) + 1 FROM component_versions WHERE component = %s",
                             (name,)).fetchone()[0]
            if activate:
                conn.execute("UPDATE component_versions SET is_active = false WHERE component = %s AND is_active",
                             (name,))
            r = conn.execute(
                f"INSERT INTO component_versions(component, version, content, note, created_by, is_active) "
                f"VALUES (%s, %s, %s::jsonb, %s, %s, %s) RETURNING {_COLS}",
                (name, v, json.dumps(clean), (note or "").strip() or None, created_by or "ui", activate)).fetchone()
    _after_change(name)
    return _row(r)


def activate(name: str, version: int) -> dict[str, Any] | None:
    if name not in COMPONENTS:
        raise KeyError(name)
    with _pool().connection() as conn:
        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_ID,))
            exists = conn.execute("SELECT 1 FROM component_versions WHERE component = %s AND version = %s",
                                  (name, int(version))).fetchone()
            if not exists:
                return None
            conn.execute("UPDATE component_versions SET is_active = false WHERE component = %s AND is_active",
                         (name,))
            r = conn.execute(f"UPDATE component_versions SET is_active = true WHERE component = %s AND version = %s "
                             f"RETURNING {_COLS}", (name, int(version))).fetchone()
    _after_change(name)
    return _row(r)


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content if content.endswith("\n") else content + "\n"
    return json.dumps(content, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def diff(name: str, a: int | None = None, b: int | None = None) -> dict[str, Any] | None:
    """Unified diff of version a -> b (b defaults to the active version, a to b - 1)."""
    hist = get_history(name)
    if b is None:
        b = hist["active"]["version"] if hist["active"] else None
    if b is None:
        return None
    if a is None:
        a = max(1, int(b) - 1)
    va, vb = get_version(name, a), get_version(name, b)
    if va is None or vb is None:
        return None
    lines = list(difflib.unified_diff(_as_text(va["content"]).splitlines(keepends=True),
                                      _as_text(vb["content"]).splitlines(keepends=True),
                                      fromfile=f"{name}@v{a}", tofile=f"{name}@v{b}"))
    added = sum(1 for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in lines if ln.startswith("-") and not ln.startswith("---"))
    return {"component": name, "a": int(a), "b": int(b), "diff": "".join(lines), "added": added,
            "removed": removed, "identical": not lines}
