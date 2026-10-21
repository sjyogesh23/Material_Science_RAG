#!/usr/bin/env python3
"""Live progress bar for a running `extract.py` describe stage (works even on runs started
without a progress bar). It peeks at the process with py-spy - read-only, does not slow it down.

Usage (project root, venv active):
    python scripts/progress.py              # auto-finds the running extract.py
    python scripts/progress.py --pid 13036
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYSPY = Path(sys.executable).parent / "py-spy.exe"


def find_pid() -> int | None:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                          "Where-Object { $_.CommandLine -like '*extract.py*' } | "
                          "Sort-Object WorkingSetSize -Descending | Select-Object -First 1 -ExpandProperty ProcessId"],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else None


def sample(pid: int) -> dict | None:
    r = subprocess.run([str(PYSPY), "dump", "--pid", str(pid), "--locals"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    txt = r.stdout
    m = re.search(r"describe_document.*?Locals:(.*?)(?:\n\s{4}\w|\Z)", txt, re.S)
    if not m:  # parsing stage: read the latest "docling pages X-Y / N" line from the log
        last = None
        log = ROOT / "logs" / "extract.log"
        if log.exists():
            with open(log, encoding="utf-8", errors="ignore") as f:
                f.seek(max(0, log.stat().st_size - 200_000))
                for ln in f:
                    if (mm := re.search(r"docling pages (\d+)-(\d+) / (\d+)", ln)):
                        last = mm
        if last:
            return {"stage": "parse", "page": int(last.group(2)), "pages": int(last.group(3))}
        return {"stage": "starting"}
    loc = m.group(1)
    done = re.search(r"\bdone: (\d+)", loc)
    page = re.search(r'"page_number": (\d+)', loc) or re.search(r'"page": (\d+)', loc)
    path = re.search(r'"file_name": "([^"]+)"', loc)
    return {"stage": "describe", "done": int(done.group(1)) if done else 0,
            "page": int(page.group(1)) if page else None, "file": path.group(1) if path else "?"}


def pending_total() -> int:
    """Figures still to describe across all outputs (as of their last save)."""
    total = 0
    for j in (ROOT / "extracted").rglob("*.json"):
        if j.name == "index.json":
            continue
        try:
            total += json.loads(j.read_text(encoding="utf-8"))["summary"].get("figures_pending_description", 0)
        except Exception:  # noqa: BLE001
            pass
    return total


def bar(frac: float, width: int = 40) -> str:
    n = int(round(frac * width))
    return "█" * n + "░" * (width - n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int)
    ap.add_argument("--total", type=int, help="figures to describe (default: read from saved JSON)")
    ap.add_argument("--every", type=int, default=15, help="seconds between checks")
    a = ap.parse_args()
    if not PYSPY.exists():
        sys.exit("py-spy missing: pip install py-spy")
    pid = a.pid or find_pid()
    if not pid:
        sys.exit("no running extract.py found")
    total = a.total or pending_total()
    hist: list[tuple[float, int]] = []
    print(f"watching extract.py (pid {pid}), {total} figures to describe - Ctrl+C to quit\n")
    while True:
        s = sample(pid)
        if s is None:
            print("\nprocess finished (or exited).")
            break
        if s["stage"] == "parse":
            frac = s["page"] / s["pages"]
            print(f"\r PARSING   {bar(frac)} page {s['page']}/{s['pages']} ({frac:.0%})            ",
                  end="", flush=True)
        elif s["stage"] != "describe":
            print(f"\r stage: {s['stage']:<40}", end="", flush=True)
        else:
            now, done = time.time(), s["done"]
            hist.append((now, done))
            hist = [h for h in hist if now - h[0] < 900]  # rate over the last 15 min
            rate = (hist[-1][1] - hist[0][1]) / (hist[-1][0] - hist[0][0]) if len(hist) > 1 and hist[-1][0] > hist[0][0] else 0
            left = max(0, total - done)
            eta = time.strftime("%H:%M", time.localtime(now + left / rate)) if rate > 0 else "measuring..."
            spf = f"{1 / rate:.0f}s/fig" if rate > 0 else "--"
            frac = min(1.0, done / total) if total else 0
            print(f"\r {bar(frac)} {done}/{total} ({frac:.0%}) | page {s['page']} | {spf} | ETA {eta}   ",
                  end="", flush=True)
        time.sleep(a.every)


if __name__ == "__main__":
    main()
