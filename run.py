"""One-command launcher for NewRAG.

    .venv\\Scripts\\python run.py [--port 8000] [--no-browser] [--ingest]

1. starts the embedded Postgres (pgserver, data in pgdata/) and applies the schema
2. makes sure Ollama answers at settings.ollama_host (else starts ``ollama serve`` detached)
3. pulls the chat / router (and optional embedding) models if they are missing
4. ingests chunks/chunks.jsonl if the chunks table is empty (or always with --ingest)
5. runs uvicorn on settings.api_host:api_port (next free port if busy), prints the links and opens the chat
   in the browser once /api/health answers.  Ctrl+C stops uvicorn and Postgres (Ollama keeps running).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

log = logging.getLogger("newrag.run")


def say(msg: str) -> None:
    print(f"[newrag] {msg}", flush=True)


# ----------------------------------------------------------------------------- http helpers
def _get_json(url: str, timeout: float = 3.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8") or "{}")
    except (urllib.error.URLError, OSError, ValueError):
        return None


# ----------------------------------------------------------------------------- postgres
def start_postgres() -> str:
    from app import db
    say("starting embedded Postgres ...")
    t0 = time.perf_counter()
    uri = db.start_postgres()
    db.init_db()
    say(f"Postgres ready in {time.perf_counter() - t0:.1f}s")
    return uri


def _prune_pg_handles(srv) -> None:
    """pgserver only stops Postgres when the last *registered* process exits, but processes that were killed
    (or crashed) never unregister, so dead pids pile up in pgdata/.handle_pids.json and Postgres is never
    stopped. Drop pids that are no longer live Python processes."""
    try:
        import psutil
        lst = srv.global_process_id_list
        me = os.getpid()
        alive = []
        for pid in lst.get():
            if pid == me:
                alive.append(pid)
                continue
            try:
                if "python" in psutil.Process(pid).name().lower():
                    alive.append(pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        lst.put(alive)
    except Exception as e:  # noqa: BLE001
        log.debug("could not prune pgserver handles: %s", e)


def stop_postgres() -> None:
    from app import db
    try:
        db.close_pools()
    except Exception as e:  # noqa: BLE001
        log.debug("close_pools: %s", e)
    srv = getattr(db, "_server", None)
    if srv is not None:
        _prune_pg_handles(srv)
        try:
            srv._cleanup()        # stops the server if this is the last process holding it
            info = getattr(srv, "_postmaster_info", None)
            running = bool(info and info.process.is_running())
            say("Postgres left running (another NewRAG process still uses it)" if running else "Postgres stopped")
        except Exception as e:  # noqa: BLE001
            say(f"Postgres stop failed: {e}")


def chunk_count() -> int:
    from app import db
    with db.get_pool().connection() as conn:
        return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]


# ----------------------------------------------------------------------------- ollama
def ollama_exe() -> Path | None:
    cands = [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"]
    from shutil import which
    if w := which("ollama"):
        cands.append(Path(w))
    return next((p for p in cands if p.is_file()), None)


def ensure_ollama(host: str, timeout_s: float = 60) -> bool:
    if _get_json(f"{host}/api/tags") is not None:
        say(f"Ollama is running at {host}")
        return True
    exe = ollama_exe()
    if exe is None:
        say("Ollama is not running and ollama.exe was not found - install it from ollama.com")
        return False
    say(f"starting {exe} serve (detached) ...")
    env = dict(os.environ)
    hostport = host.split("://", 1)[-1].rstrip("/")
    env.setdefault("OLLAMA_HOST", hostport)
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    with open(logs / "ollama.log", "ab") as fh:
        subprocess.Popen([str(exe), "serve"], stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         env=env, creationflags=flags, close_fds=True)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _get_json(f"{host}/api/tags", timeout=2) is not None:
            say("Ollama is up")
            return True
        time.sleep(1)
    say(f"Ollama did not answer within {timeout_s:.0f}s (see logs/ollama.log)")
    return False


def _has_model(models: list[str], want: str) -> bool:
    if ":" not in want:
        want_l = want + ":latest"
        return any(m in (want, want_l) for m in models)
    return want in models


def ensure_models(host: str, names: list[str]) -> None:
    tags = _get_json(f"{host}/api/tags") or {}
    have = [m.get("model") or m.get("name") for m in tags.get("models", [])]
    for name in dict.fromkeys(n for n in names if n):
        if _has_model(have, name):
            continue
        say(f"pulling Ollama model {name} (one-time download) ...")
        try:
            import ollama
            last = ""
            for p in ollama.Client(host=host).pull(name, stream=True):
                status = getattr(p, "status", "") or ""
                total, done = getattr(p, "total", None), getattr(p, "completed", None)
                line = f"{status} {done / total * 100:5.1f}%" if total and done else status
                if line != last:
                    print(f"\r    {line:<60}", end="", flush=True)
                    last = line
            print()
            say(f"{name} ready")
        except Exception as e:  # noqa: BLE001
            print()
            say(f"pull of {name} failed: {e}")


def ensure_reranker(host: str, name: str) -> None:
    """The Ollama reranker is built locally (not on the registry) by scripts/build_reranker_gguf.py."""
    tags = _get_json(f"{host}/api/tags") or {}
    if _has_model([m.get("model") or m.get("name") for m in tags.get("models", [])], name):
        return
    script = ROOT / "scripts" / "build_reranker_gguf.py"
    if not script.exists():
        say(f"reranker model {name} is missing in Ollama and {script} was not found - retrieval will fail")
        return
    say(f"reranker model {name} is missing - building it now with scripts/build_reranker_gguf.py")
    say("  (one-time: downloads Qwen/Qwen3-Reranker-0.6B from Hugging Face, quantises to Q8_0, `ollama create`)")
    r = subprocess.run([sys.executable, str(script), "--name", name], cwd=ROOT)
    if r.returncode == 0:
        say(f"{name} ready")
    else:
        say(f"reranker build failed (exit {r.returncode}); run it manually: "
            f".venv\\Scripts\\python scripts\\build_reranker_gguf.py --name {name}")


# ----------------------------------------------------------------------------- ingest
def run_ingest(force_flag: bool) -> None:
    say("ingesting chunks/chunks.jsonl into Postgres (embeddings) ...")
    # separate process: the ingest embedder's RAM/VRAM is released before the server loads its own models
    cmd = [sys.executable, "-m", "app.ingest"]
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        say(f"ingest failed (exit {r.returncode}) - the server will start anyway")


# ----------------------------------------------------------------------------- port / links
def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def pick_port(host: str, port: int, tries: int = 30) -> int:
    for p in range(port, port + tries):
        if port_free(host, p):
            if p != port:
                say(f"port {port} is busy - using {p}")
            return p
    raise SystemExit(f"no free port in {port}-{port + tries - 1}")


def print_links(base: str, pg_uri: str, ollama: str) -> None:
    rows = [("Chat UI", f"{base}/"), ("Token audit", f"{base}/audit"), ("API docs", f"{base}/docs"),
            ("Health", f"{base}/api/health"), ("Postgres URI", pg_uri), ("Ollama URL", ollama)]
    w1 = max(len(a) for a, _ in rows)
    w2 = max(len(b) for _, b in rows)
    line = "+" + "-" * (w1 + 2) + "+" + "-" * (w2 + 2) + "+"
    print("\n" + line)
    print(f"| {'NewRAG is running'.ljust(w1 + w2 + 3)} |")
    print(line)
    for a, b in rows:
        print(f"| {a.ljust(w1)} | {b.ljust(w2)} |")
    print(line)
    print("  Stop: press Ctrl+C in this window (stops the web server and Postgres; Ollama keeps running).\n",
          flush=True)


def wait_healthy_then(base: str, pg_uri: str, ollama: str, open_browser: bool, server) -> None:
    deadline = time.monotonic() + 120
    h = None
    while time.monotonic() < deadline and not server.should_exit:
        if server.started:
            h = _get_json(f"{base}/api/health", timeout=10)
            if h is not None:
                break
        time.sleep(0.3)
    if h is None:
        return
    print_links(base, pg_uri, ollama)
    problems = [k for k in ("postgres", "ollama", "chat_model_present") if not h.get(k)]
    if not h.get("chunks"):
        problems.append("chunks (run with --ingest)")
    if problems:
        say("health warnings: " + ", ".join(problems))
    else:
        say(f"health ok: {h.get('chunks')} chunks, model {h.get('chat_model')}; "
            "retriever models warming up in the background")
    if open_browser:
        webbrowser.open(f"{base}/")


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Start NewRAG (Postgres + Ollama + API/UI)")
    ap.add_argument("--port", type=int, default=None, help="HTTP port (default settings.api_port = 8000)")
    ap.add_argument("--host", default=None, help="bind address (default settings.api_host)")
    ap.add_argument("--no-browser", action="store_true", help="don't open the chat in the browser")
    ap.add_argument("--ingest", action="store_true", help="run ingest even if chunks already exist")
    ap.add_argument("--log-level", default="warning", help="uvicorn log level (default warning)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "pgserver", "httpcore", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from app.config import get_settings, reload_settings

    s = get_settings()
    try:
        pg_uri = start_postgres()
    except Exception as e:  # noqa: BLE001
        say(f"Postgres failed to start: {e}")
        return 1
    s = reload_settings()             # pick up app_settings overrides
    try:
        ollama_ok = ensure_ollama(s.ollama_host)
        if ollama_ok:
            want = [s.chat_model, s.router_model]
            rr = getattr(s, "ollama_rerank_model", None) if getattr(s, "rerank_backend", "") == "ollama" else None
            if getattr(s, "embed_backend", "") == "ollama" or rr:
                # also the reranker build's template (same architecture + tokenizer)
                want.append(getattr(s, "ollama_embed_model", None) or "qwen3-embedding:0.6b")
            ensure_models(s.ollama_host, want)
            if rr:
                ensure_reranker(s.ollama_host, rr)
        n = chunk_count()
        if args.ingest or n == 0:
            run_ingest(args.ingest)
            say(f"chunks in Postgres: {chunk_count()}")
        else:
            say(f"{n} chunks already indexed (use --ingest to re-run ingest)")

        import uvicorn

        host = args.host or s.api_host
        port = pick_port(host, args.port or s.api_port)
        os.environ["NEWRAG_API_PORT"] = str(port)
        base = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"
        config = uvicorn.Config("app.api.main:app", host=host, port=port, log_level=args.log_level,
                                access_log=False, timeout_graceful_shutdown=5)
        server = uvicorn.Server(config)
        threading.Thread(target=wait_healthy_then, args=(base, pg_uri, s.ollama_host, not args.no_browser, server),
                         daemon=True).start()
        say(f"starting web server on {base} ...")
        # uvicorn shuts down gracefully on SIGINT/SIGTERM/SIGBREAK, then re-raises the signal with the handler
        # that was installed before it: make that a KeyboardInterrupt so `finally` still stops Postgres
        # (the default SIGBREAK/SIGTERM handlers would kill the process first).
        def _interrupt(sig, frame):
            raise KeyboardInterrupt
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), _interrupt)
        server.run()                  # blocks; Ctrl+C -> graceful shutdown
    except KeyboardInterrupt:
        pass
    finally:
        say("shutting down ...")
        stop_postgres()
    say("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
