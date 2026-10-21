"""Dev helper: dump items of a run with score / pass / judge justifications (python -m eval._dump_items RUN [--all] [--ids a,b] [--scen s])."""
from __future__ import annotations

import argparse
import json
import sys

from eval import db as edb
from eval import metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ids", default="")
    ap.add_argument("--scen", default="")
    ap.add_argument("--max-answer", type=int, default=1500)
    ap.add_argument("--images", action="store_true", help="only list image FP items")
    a = ap.parse_args()
    run = edb.one("SELECT run_id FROM eval_runs WHERE label=%s OR run_id::text=%s ORDER BY run_id DESC LIMIT 1",
                  (a.run, a.run))
    rid = run[0]
    rows = edb.rows("SELECT r.*, i.payload FROM eval_results r JOIN eval_items i USING(item_id) WHERE run_id=%s "
                    "ORDER BY item_id", (rid,))
    ids = {x for x in a.ids.split(",") if x}
    for r in rows:
        if ids and r["item_id"] not in ids:
            continue
        if a.scen and r["scenario"] not in a.scen.split(","):
            continue
        item = r["payload"]
        if a.images:
            exp = item.get("format") == "image" or r["scenario"] == "figure_request"
            if r["images"] and not exp:
                d = r.get("decision") or {}
                print(r["item_id"], r["scenario"], "| needs_images=", d.get("needs_images"), "| rules=",
                      d.get("rules"), "| imgs=", [im.get("figure_number") for im in r["images"]], "|",
                      (r["question"] or "")[:100])
            continue
        sc = metrics.final_score(r)
        try:
            ok = metrics.item_pass(r)
        except Exception:
            ok = None
        jd = r["judge"] or {}
        low = any(isinstance(v, dict) and (v.get("score") or 5) < 5 for v in jd.values())
        if not a.all and not ids and not low:
            continue
        print("=" * 100)
        print(r["item_id"], r["scenario"], "score", sc, "pass", ok, "latency", r["latency_ms"])
        print("Q:", r["question"])
        print("GOLD:", json.dumps(item.get("gold", {}).get("facts"), ensure_ascii=False)[:800])
        if item.get("notes"):
            print("NOTES:", item["notes"])
        d = r.get("decision") or {}
        print("DECISION:", {k: d.get(k) for k in ("in_scope", "intent", "answer_format", "needs_images",
                                                   "question_forms", "rules", "sub_queries")})
        print("ANSWER:", (r["answer"] or "")[: a.max_answer])
        print("SOURCES:", [(s.get("n"), s.get("chunk_type"), (s.get("section_path") or [""])[-1][:40], s.get("pages"))
                           for s in r["sources"] or []])
        for k, v in jd.items():
            if isinstance(v, dict) and (v.get("score") or 5) < 5:
                print(f"  JUDGE {k}={v.get('score')}: {v.get('justification')}")


if __name__ == "__main__":
    sys.exit(main())
