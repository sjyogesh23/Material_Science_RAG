"""Print the evaluation matrix of a run (and optionally the deltas against another run).

    python -m eval.report --run baseline
    python -m eval.report --run tuned --compare baseline
    python -m eval.report --run 3 --json > matrix.json
"""
from __future__ import annotations

import argparse
import json
import sys

from eval import db as edb
from eval import metrics as M
from eval import questions


def build(run_ref: str) -> tuple[dict, dict]:
    from app import db
    db.init_db()
    run = edb.resolve_run(run_ref)
    if run is None:
        raise SystemExit(f"run {run_ref!r} not found")
    bank = {it["id"]: it for it in questions.load()}
    return run, M.aggregate(edb.results(run["run_id"]), bank)


def _fmt(v, nd=2):
    return "  -  " if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _p(v) -> str:
    return "  -  " if v is None else f"{v:.3f}"


def print_confusion(agg: dict, base: dict | None = None) -> None:
    """Confusion matrices with per-class and micro / macro / weighted P, R, F1 (base values in brackets)."""
    for key, c in (agg.get("confusion") or {}).items():
        b = (base or {}).get("confusion", {}).get(key)
        print(f"\nConfusion: {c['title']}  (n={c['n']}, accuracy {_p(c['accuracy'])}"
              + (f" [{_p(b['accuracy'])}]" if b else "") + ")")
        print(f"  expected: {c['expected_rule']}\n  predicted: {c['predicted_rule']}")
        pos, neg = c["positive"], c["negative"]
        w = max(len(pos), len(neg), 14) + 2
        print(" " * 18 + f"pred {pos}".rjust(w) + f"pred {neg}".rjust(w + 5))
        print(f"  {'exp ' + pos:<16}" + f"TP {c['tp']}".rjust(w) + f"FN {c['fn']}".rjust(w + 5))
        print(f"  {'exp ' + neg:<16}" + f"FP {c['fp']}".rjust(w) + f"TN {c['tn']}".rjust(w + 5))
        print(f"  {'class':<16}{'precision':>10}{'recall':>10}{'F1':>10}{'support':>9}")
        rows = [(cl["class"], cl) for cl in c["classes"]] + [(k, c[k]) for k in ("micro", "macro", "weighted")]
        for name, r in rows:
            line = f"  {name:<16}{_p(r['precision']):>10}{_p(r['recall']):>10}{_p(r['f1']):>10}{r['support']:>9}"
            if b:
                br = next((x for x in b["classes"] if x["class"] == name), None) if name not in ("micro", "macro", "weighted") else b[name]
                if br:
                    line += f"   [F1 {_p(br['f1'])}]"
            print(line)


def print_averages(agg: dict, base: dict | None = None) -> None:
    av = agg.get("averages")
    if not av:
        return
    bav = (base or {}).get("averages") or {}
    bm = {m["key"]: m for m in bav.get("metrics", [])}
    print("\nMicro (item-weighted) vs macro (scenario-weighted) means" + ("  [compared run]" if base else ""))
    print(f"  {'metric':<26}{'micro':>9}{'macro':>9}{'gap':>8}{'items':>7}{'scen':>6}")

    def line(label, m, bmx):
        s = f"  {label:<26}{_fmt(m['micro'], 3):>9}{_fmt(m['macro'], 3):>9}{_fmt(m['gap'], 3):>8}{m['n_items']:>7}{m['n_scenarios']:>6}"
        if bmx:
            s += f"   [{_fmt(bmx['micro'], 3)} / {_fmt(bmx['macro'], 3)}]"
        print(s)
    line("item score (0-100)", av["score"], bav.get("score"))
    line("pass rate", av["pass_rate"], bav.get("pass_rate"))
    for m in av["metrics"]:
        line(m["label"] + (" (1-5)" if m["kind"] == "score" else ""), m, bm.get(m["key"]))


def print_matrix(run: dict, agg: dict, cmp: dict | None = None, base: dict | None = None) -> None:
    cols = [m["key"] for m in M.METRICS if any(m["key"] in agg["matrix"].get(s, {}) for s in agg["scenarios"])]
    rates = ["ooc_correct", "citation_coverage", "hit_at_k"]
    o = agg["overall"]
    print(f"\nRun {run['run_id']} '{run['label']}' status={run['status']}  items={o['n']} judged={o['judged']} "
          f"errors={o['errors']}")
    print(f"Overall score {o['score']} (scenario-balanced, 0-100)   item-mean {o['score_items']}   pass rate {o['pass_rate']}")
    ooc = agg["ooc"]
    print(f"OOC  P={ooc['precision']} R={ooc['recall']} F1={ooc['f1']} exact={ooc['exact_match_rate']}  "
          f"(tp {ooc['tp']} fp {ooc['fp']} fn {ooc['fn']} tn {ooc['tn']})")
    lat = agg["latency_ms"]
    print(f"Latency p50 {lat['p50']} ms  p95 {lat['p95']} ms  TTFT p50 {lat['ttft_p50']} ms   tokens/answer {agg['tokens']['mean']}")
    print(f"Retrieval {agg['retrieval']}   paraphrase consistency {agg['paraphrase']['consistency']}\n")
    w = 26
    head = "scenario".ljust(w) + "".join(c[:10].rjust(11) for c in cols + rates) + "   score  pass"
    print(head)
    print("-" * len(head))
    for s in agg["scenarios"] + ["__all__"]:
        row = agg["matrix"].get(s, {})
        line = (s if s != "__all__" else "ALL").ljust(w)
        for c in cols:
            v = row.get(c, {}).get("mean")
            d = (cmp or {}).get("cells", {}).get(s, {}).get(c) if cmp else None
            line += (_fmt(v) + (f"{d:+.1f}" if d is not None else "")).rjust(11)
        for c in rates:
            v = row.get(c, {}).get("mean")
            line += _fmt(v).rjust(11)
        bs = agg["by_scenario"].get(s, {})
        line += f"   {_fmt(bs.get('score'), 1):>5}  {_fmt(bs.get('pass_rate'), 2):>4}" if bs else ""
        print(line)
    print_confusion(agg, base)
    print_averages(agg, base)
    if cmp:
        print("\nHeadline deltas (compared run -> this run):")
        for k, v in cmp["headline"].items():
            print(f"  {k:24s} {_fmt(v['a'])} -> {_fmt(v['b'])}  ({_fmt(v['delta'])})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--compare", help="baseline run id/label to diff against")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run, agg = build(a.run)
    cmp = base = None
    if a.compare:
        _, base = build(a.compare)
        cmp = M.compare(base, agg)
    if a.json:
        print(json.dumps({"run": run, "aggregate": agg, "compare": cmp}, default=str, indent=1))
    else:
        print_matrix(run, agg, cmp, base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
