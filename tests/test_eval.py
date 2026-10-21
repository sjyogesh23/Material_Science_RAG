"""Evaluation package tests (no Ollama needed; the DB/API tests use the embedded Postgres).

    .venv\\Scripts\\python -m pytest tests/test_eval.py -v
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NEWRAG_SKIP_WARMUP", "1")

from eval import judge as J  # noqa: E402
from eval import metrics as M  # noqa: E402
from eval import questions as Q  # noqa: E402

CHUNKS = {json.loads(l)["chunk_id"]: json.loads(l)
          for l in (ROOT / "chunks" / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}


# ----------------------------------------------------------------------------- question bank
@pytest.fixture(scope="module")
def bank():
    return Q.load()


def test_bank_size_and_coverage(bank):
    st = Q.stats(bank)
    assert 70 <= st["total"] <= 100
    for s in Q.SCENARIOS:
        assert st["by_scenario"].get(s, 0) >= 4, s
    assert st["by_expected"]["out_of_context"] >= 10
    assert st["multi_turn"] >= 5


def test_gold_references_exist(bank):
    for it in bank:
        g = it["gold"]
        for cid in g["chunk_ids"] + g.get("figure_chunk_ids", []):
            assert cid in CHUNKS, (it["id"], cid)
        chunk_pages = {p for c in g["chunk_ids"] for p in CHUNKS[c]["pages"]}
        assert set(g["pages"]) <= chunk_pages, it["id"]
        if it["expected"] == "answer":
            assert g["facts"] and g["chunk_ids"], it["id"]
        if it.get("paraphrase_of"):
            assert any(o["id"] == it["paraphrase_of"] for o in bank)


def test_near_domain_topics_absent_from_book(bank):
    text = " ".join(c["content"].lower() for c in CHUNKS.values())
    for term in ["weibull", "charpy", "vickers", "pilling", "czochralski"]:
        assert term not in text, term
    assert {i["id"] for i in bank if i["scenario"] == "out_of_context_near_domain"}


def test_figure_items_have_gold_figures(bank):
    for it in bank:
        if it["scenario"] == "figure_request":
            figs = it["gold"]["figure_chunk_ids"]
            assert figs and all(CHUNKS[f]["chunk_type"] == "figure" for f in figs)


def test_select_and_hash(bank):
    sel = Q.select(bank, ["out_of_context"], None, 2)
    assert len(sel) == 2 and all(i["scenario"] == "out_of_context" for i in sel)
    assert Q.bank_hash(bank) == Q.bank_hash(json.loads(json.dumps(bank)))


# ----------------------------------------------------------------------------- deterministic metrics
def test_citation_parsing():
    assert M.cited_numbers("a [1] b [2, 3] c [4-6]") == [1, 2, 3, 4, 5, 6]
    units = M.answer_units("Bragg's law relates angle to spacing [1]. It needs no citation here though.\n\n"
                           "| A | B |\n|---|---|\n| BCC | 8 [2] |\n| FCC | 12 |\n\n- n: integer order [1]")
    assert len(units) == 5
    assert sum(1 for u in units if M.CITE_RE.search(u)) == 3


def _item(expected="answer", scenario="factual_definition", **kw):
    return {"id": "x", "scenario": scenario, "question": "q", "expected": expected,
            "gold": {"facts": ["f"], "chunk_ids": ["doc:c1"], "pages": [5]}, **kw}


def test_item_metrics_answer():
    rec = {"answer": "Fact one is stated [1]. Fact two has a bad cite [7]. No cite in this sentence at all.",
           "out_of_context": False, "status": "ok",
           "sources": [{"n": 1, "chunk_id": "doc:c1", "pages": [5]}],
           "retrieved": [{"stage": "search", "chunk_id": "doc:c9", "pages": [9]},
                         {"stage": "search", "chunk_id": "doc:c1", "pages": [5]}]}
    m = M.item_metrics(_item(), rec, final_k=6)
    assert m["refusal_correctness"] == 5 and m["ooc_correct"]
    assert m["citation_coverage"] == pytest.approx(2 / 3)
    assert m["invalid_citation_rate"] == pytest.approx(0.5)
    assert m["hit_at_k"] and m["mrr"] == pytest.approx(0.5) and m["page_hit"] and m["cited_gold"]


def test_item_metrics_refusals():
    ooc = {"answer": "#OutofContext", "out_of_context": True, "status": "ok", "sources": []}
    assert M.item_metrics(_item("out_of_context", "out_of_context"), ooc)["refusal_correctness"] == 5
    false_ref = M.item_metrics(_item(), ooc)
    assert false_ref["refusal_correctness"] == 1 and not false_ref["ooc_correct"]
    answered = {"answer": "Canberra [1].", "out_of_context": False, "status": "ok", "sources": [{"n": 1}]}
    assert M.item_metrics(_item("out_of_context", "out_of_context"), answered)["refusal_correctness"] == 1
    sloppy = {"answer": "#OutofContext - sorry", "out_of_context": True, "status": "ok"}
    assert M.item_metrics(_item("out_of_context", "out_of_context"), sloppy)["refusal_correctness"] == 1


def test_must_not_and_images():
    it = _item(scenario="figure_request", must_not_contain=["PWNED"], format="image")
    it["gold"]["figure_chunk_ids"] = ["doc:f1"]
    rec = {"answer": "Here [1]. PWNED", "status": "ok", "sources": [{"n": 1}], "images": [{"chunk_id": "doc:f1"}]}
    m = M.item_metrics(it, rec)
    assert m["must_not_ok"] is False and m["image_returned"] and m["image_correct"]


def _res(item_id, scenario, pred_ooc, exp_ooc, judge=None, lat=1000):
    det = {"pred_ooc": pred_ooc, "expected_ooc": exp_ooc, "ooc_correct": pred_ooc == exp_ooc,
           "refusal_correctness": 5 if pred_ooc == exp_ooc else 1, "exact_ooc": True if pred_ooc else None}
    return {"item_id": item_id, "scenario": scenario, "status": "ok", "det_metrics": det, "judge": judge or {},
            "latency_ms": lat, "usage": {"total": 100}}


def test_aggregate_and_compare():
    good = {"correctness": {"score": 5}, "groundedness": {"score": 4}}
    bad = {"correctness": {"score": 2}, "groundedness": {"score": 1}}
    a = [_res("a1", "factual_definition", False, False, good), _res("a2", "factual_definition", True, False, bad),
         _res("o1", "out_of_context", True, True, {"_meta": {}}), _res("o2", "out_of_context", False, True, bad)]
    agg = M.aggregate(a)
    ooc = agg["ooc"]
    assert (ooc["tp"], ooc["fp"], ooc["fn"], ooc["tn"]) == (1, 1, 1, 1)
    assert ooc["precision"] == 0.5 and ooc["recall"] == 0.5 and ooc["f1"] == 0.5
    assert agg["matrix"]["factual_definition"]["correctness"]["mean"] == 3.5
    assert agg["overall"]["judged"] == 4
    assert agg["latency_ms"]["p50"] == 1000
    b = [_res("a1", "factual_definition", False, False, good), _res("a2", "factual_definition", False, False, good),
         _res("o1", "out_of_context", True, True, {"_meta": {}}), _res("o2", "out_of_context", True, True, {"_meta": {}})]
    cmp = M.compare(agg, M.aggregate(b))
    assert cmp["cells"]["factual_definition"]["correctness"] == pytest.approx(1.5)
    assert cmp["headline"]["ooc.f1"]["delta"] > 0
    assert cmp["items"]["a2"]["delta"] > 0


def test_unjudged_scores_are_none():
    agg = M.aggregate([_res("a1", "factual_definition", False, False)])
    assert agg["overall"]["score"] is None and agg["overall"]["judged"] == 0
    assert M.final_score(_res("a1", "factual_definition", False, False)) is None


def test_item_score_weighting():
    assert M.item_score({"correctness": 5, "groundedness": 5}) == 100
    assert M.item_score({"correctness": 1, "groundedness": 1}) == 0
    assert M.item_score({}) is None


# ----------------------------------------------------------------------------- judge (no model)
def test_applicable_metrics():
    it = _item(scenario="prompt_injection")
    rec = {"answer": "Bragg [1].", "status": "ok", "det_metrics": {"pred_ooc": False}}
    ms, auto = J.applicable_metrics(it, rec)
    assert "injection_resistance" in ms and "correctness" in ms and not auto
    ooc = {"answer": "#OutofContext", "status": "ok", "det_metrics": {"pred_ooc": True}}
    ms, auto = J.applicable_metrics(_item("out_of_context", "prompt_injection"), ooc)
    assert ms == [] and auto["injection_resistance"]["score"] == 5
    ms, auto = J.applicable_metrics(_item(scenario="analogy"), ooc)
    assert ms == [] and auto["correctness"]["score"] == 1 and auto["analogy_quality"]["score"] == 1
    ms, _ = J.applicable_metrics(_item(scenario="table_request"), rec)
    assert "format_compliance" in ms
    ms, _ = J.applicable_metrics(_item("out_of_context", "out_of_context_near_domain"), rec)
    assert ms == ["groundedness", "citation_accuracy"]


def test_schema_and_validation():
    sch = J.build_schema(["correctness", "groundedness"])
    assert sch["required"] == ["correctness", "groundedness"]
    assert sch["properties"]["correctness"]["properties"]["score"]["enum"] == [1, 2, 3, 4, 5]
    ok = J._validate({"correctness": {"score": 4, "justification": "fine"},
                      "groundedness": {"score": "5", "justification": "x"}}, ["correctness", "groundedness"])
    assert ok["groundedness"]["score"] == 5
    with pytest.raises(ValueError):
        J._validate({"correctness": {"score": 9, "justification": ""}}, ["correctness"])
    with pytest.raises(ValueError):
        J._validate({}, ["correctness"])


def test_build_prompt_contains_everything(bank):
    item = next(i for i in bank if i["id"] == "mt-02")
    cid = item["gold"]["chunk_ids"][0]
    rec = {"question": item["turns"][-1], "answer": "Frenkel is a vacancy-interstitial pair [1].",
           "decision": {"standalone_query": "How is a Frenkel defect different from a Schottky defect?"},
           "turns": [{"question": item["turns"][0], "answer": "A Schottky defect is ... [1]"}, {"question": item["turns"][1]}],
           "sources": [{"n": 1, "chunk_id": cid, "chunk_type": "text", "section_path": ["3.2"], "pages": [56]}],
           "retrieved": []}
    texts = {cid: {"content": CHUNKS[cid]["content"], "section_path": ["3.2"]}}
    p = J.build_prompt(item, rec, ["correctness", "pronoun_resolution"], source_texts=texts)
    for needle in ["GOLD FACTS", "Frenkel", "STANDALONE QUERY", "EARLIER TURNS", "[1] (text", "pronoun_resolution",
                   "ANSWER TO GRADE"]:
        assert needle in p, needle
    from app.tokens import count_tokens
    assert count_tokens(J.SYSTEM + p) < J.NUM_CTX - J.NUM_PREDICT


def test_choose_model_respects_budget():
    installed = {"qwen3.5:9b": {}, "qwen3:8b": {}}
    roomy = {"ram_available_gb": 8.0, "vram_free_gb": 3.8}
    assert J.choose_model(None, roomy, installed)[0] == "qwen3.5:9b"
    tight = {"ram_available_gb": 4.4, "vram_free_gb": 3.5}
    assert J.choose_model(None, tight, installed)[0] == "qwen3:8b"
    with pytest.raises(SystemExit):
        J.choose_model(None, {"ram_available_gb": 1.0, "vram_free_gb": 0.5}, installed)
    assert J.choose_model(None, {"ram_available_gb": 1.0, "vram_free_gb": 0.5}, installed, force=True)[0] == "qwen3.5:9b"


def test_calibration_file_is_valid(bank):
    ids = {i["id"] for i in bank}
    cases = [json.loads(l) for l in J.CALIBRATION.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(cases) >= 5
    assert {c["label"] for c in cases} == {"good", "bad"}
    for c in cases:
        assert c["item_id"] in ids and all(s in CHUNKS for s in c["source_chunk_ids"]) and c["expect"]


# ----------------------------------------------------------------------------- runner (stub pipeline)
class _Hit:
    def __init__(self, cid, pages):
        self.chunk_id, self.pages, self.chunk_type, self.rerank_score, self.via, self.header = cid, pages, "text", .9, "hybrid", "h"


def test_recording_retriever():
    from eval.run import RecordingRetriever

    class Inner:
        name = "inner"

        def search(self, q, **kw):
            return [_Hit("c1", [1]), _Hit("c2", [2])]

        def expand_cross_refs(self, hits, n):
            return [_Hit("c3", [3])]

    r = RecordingRetriever(Inner())
    r.search("q", sub_queries=None)
    r.expand_cross_refs([], 1)
    assert [x["stage"] for x in r.log] == ["search", "search", "cross_ref"]
    assert r.name == "inner"
    r.reset()
    assert r.log == []


def test_run_turn_collects_events(monkeypatch):
    from app.rag import pipeline
    from eval.run import run_turn

    async def fake_answer(sid, text):
        yield {"event": "session", "data": {"session_id": sid}}
        yield {"event": "decision", "data": {"standalone_query": text, "intent": "definition"}}
        yield {"event": "token", "data": {"t": "Bragg "}}
        yield {"event": "token", "data": {"t": "law [1]."}}
        yield {"event": "answer", "data": {"content": "Bragg law [1].", "uncited": False}}
        yield {"event": "sources", "data": [{"n": 1, "chunk_id": "c1", "pages": [22], "snippet": "s", "extra": 1}]}
        yield {"event": "images", "data": []}
        yield {"event": "usage", "data": {"total": 50, "prompt_tokens": 40, "completion_tokens": 10,
                                          "timings_ms": {"routing": 5, "total": 9}}}
        yield {"event": "done", "data": {"message_id": 1, "out_of_context": False}}

    monkeypatch.setattr(pipeline, "answer", fake_answer)
    rec = asyncio.run(run_turn("00000000-0000-0000-0000-000000000000", "What is Bragg's law?"))
    assert rec["answer"] == "Bragg law [1]." and rec["ttft_ms"] is not None
    assert rec["sources"][0]["chunk_id"] == "c1" and "extra" not in rec["sources"][0]
    assert rec["timings"]["routing"] == 5 and rec["usage"]["total"] == 50 and rec["error"] is None


# ----------------------------------------------------------------------------- DB + API
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app.api.main import app
    with TestClient(app) as c:
        yield c


def test_db_roundtrip_and_api(client, bank):
    from eval import db as edb
    edb.ensure_schema()
    edb.ensure_schema()   # idempotent
    item = next(i for i in bank if i["id"] == "oc-01")
    rid = edb.create_run("pytest-eval", {"settings": {}}, Q.bank_hash(bank), 1)
    try:
        rec = {"item_id": "oc-01", "scenario": item["scenario"], "status": "ok", "question": item["question"],
               "answer": "#OutofContext", "out_of_context": True, "turns": [{"question": item["question"]}]}
        rec["det_metrics"] = M.item_metrics(item, rec)
        edb.save_result(rid, rec)
        edb.save_judgement(rid, "oc-01", {"_meta": {"metrics": []}}, "test-model")
        r = client.get(f"/api/eval/runs/{rid}")
        assert r.status_code == 200
        agg = r.json()["aggregate"]
        assert agg["overall"]["n"] == 1 and agg["ooc"]["tp"] == 1
        items = client.get(f"/api/eval/runs/{rid}/items").json()
        assert items[0]["item_id"] == "oc-01" and items[0]["pass"] is True and items[0]["judged"] is True
        d = client.get(f"/api/eval/runs/{rid}/items/oc-01").json()
        assert d["item"]["id"] == "oc-01"
        assert client.get(f"/api/eval/compare?a={rid}&b={rid}").status_code == 200
        assert any(x["run_id"] == rid for x in client.get("/api/eval/runs").json())
        assert client.get("/api/eval/meta").json()["bank"]["total"] == len(bank)
        assert client.get("/eval").status_code == 200
        assert client.get(f"/api/eval/runs/{rid}/items/nope").status_code == 404
    finally:
        assert client.delete(f"/api/eval/runs/{rid}").status_code == 200
    assert client.get(f"/api/eval/runs/{rid}").status_code == 404


# ----------------------------------------------------------------------------- v2: confusion matrices + micro/macro
def _rows(tp=0, fp=0, fn=0, tn=0):
    out = []
    for k, n, e, p in (("tp", tp, True, True), ("fp", fp, False, True), ("fn", fn, True, False), ("tn", tn, False, False)):
        out += [(f"{k}{i}", "s", e, p) for i in range(n)]
    return out


def test_binary_report_hand_computed():
    # TP=1 FP=3 FN=1 TN=5 (n=10)
    r = M.binary_report(_rows(1, 3, 1, 5), "pos", "neg")
    assert (r["tp"], r["fp"], r["fn"], r["tn"], r["n"]) == (1, 3, 1, 5, 10)
    pos, neg = r["classes"]
    assert pos["precision"] == pytest.approx(1 / 4, abs=1e-4) and pos["recall"] == pytest.approx(1 / 2, abs=1e-4)
    assert pos["f1"] == pytest.approx(1 / 3, abs=1e-4) and pos["support"] == 2 and pos["predicted"] == 4
    assert neg["precision"] == pytest.approx(5 / 6, abs=1e-4) and neg["recall"] == pytest.approx(5 / 8, abs=1e-4)
    assert neg["f1"] == pytest.approx(5 / 7, abs=1e-4) and neg["support"] == 8
    # micro = pooled = accuracy for single-label binary data
    assert r["accuracy"] == pytest.approx(0.6)
    assert r["micro"]["precision"] == r["micro"]["recall"] == r["micro"]["f1"] == pytest.approx(0.6)
    # macro = unweighted class mean; weighted = support-weighted
    assert r["macro"]["precision"] == pytest.approx((1 / 4 + 5 / 6) / 2, abs=1e-4)
    assert r["macro"]["recall"] == pytest.approx((1 / 2 + 5 / 8) / 2, abs=1e-4)
    assert r["macro"]["f1"] == pytest.approx((1 / 3 + 5 / 7) / 2, abs=1e-4)
    assert r["weighted"]["f1"] == pytest.approx((2 * (1 / 3) + 8 * (5 / 7)) / 10, abs=1e-4)
    assert r["rates"]["fpr"] == pytest.approx(3 / 8) and r["rates"]["tpr"] == pytest.approx(0.5)
    assert [c["item_id"] for c in r["cells"]["fp"]] == ["fp0", "fp1", "fp2"]


def test_binary_report_absent_class_and_empty():
    # no positive gold items, two false alarms: positive precision 0, recall undefined; macro uses present classes
    r = M.binary_report(_rows(0, 2, 0, 3))
    pos, neg = r["classes"]
    assert pos["precision"] == 0 and pos["recall"] is None and pos["f1"] is None and pos["support"] == 0
    assert neg["recall"] == pytest.approx(0.6) and neg["f1"] == pytest.approx(0.75)
    assert r["macro"]["f1"] == pytest.approx(0.75) and r["micro"]["f1"] == pytest.approx(0.6)
    # a class with support but never predicted counts as precision 0 in the macro mean
    r2 = M.binary_report(_rows(0, 0, 2, 2))
    assert r2["classes"][0]["precision"] is None and r2["macro"]["precision"] == pytest.approx((0 + 0.5) / 2)
    e = M.binary_report([])
    assert e["n"] == 0 and e["accuracy"] is None and e["macro"]["f1"] is None


def test_has_markdown_table():
    assert M.has_markdown_table("Intro\n\n| A | B |\n|---|---|\n| 1 | 2 |\n")
    assert M.has_markdown_table("| A | B |\n| :--- | ---: |\n| 1 | 2 |")
    assert not M.has_markdown_table("a | b without a separator row\n| x |")
    assert not M.has_markdown_table(None)


def test_confusion_three_matrices():
    items = {"oc": {"id": "oc", "scenario": "out_of_context", "expected": "out_of_context"},
             "tb": {"id": "tb", "scenario": "table_request", "expected": "answer", "format": "table"},
             "fg": {"id": "fg", "scenario": "figure_request", "expected": "answer", "format": "image"},
             "fe": {"id": "fe", "scenario": "factual_definition", "expected": "answer"}}
    res = [
        {"item_id": "oc", "scenario": "out_of_context", "answer": "#OutofContext", "out_of_context": True, "images": []},
        {"item_id": "tb", "scenario": "table_request", "answer": "| a | b |\n|---|---|\n| 1 | 2 [1] |", "images": [{"chunk_id": "x"}]},
        {"item_id": "fg", "scenario": "figure_request", "answer": "#OutofContext", "out_of_context": True, "images": []},
        {"item_id": "fe", "scenario": "factual_definition", "answer": "Plain [1].", "images": [{"chunk_id": "y"}]},
    ]
    for r in res:
        r["det_metrics"] = M.item_metrics(items[r["item_id"]], r)
    c = M.confusion(res, items)
    assert set(c) == {"ooc", "images", "table"}
    o = c["ooc"]
    assert (o["tp"], o["fp"], o["fn"], o["tn"]) == (1, 1, 0, 2)          # fg is a false refusal
    assert o["cells"]["fp"] == [{"item_id": "fg", "scenario": "figure_request"}]
    im = c["images"]
    assert (im["tp"], im["fp"], im["fn"], im["tn"]) == (0, 2, 1, 1)      # figure wanted but refused; two unasked images
    t = c["table"]
    assert (t["tp"], t["fp"], t["fn"], t["tn"]) == (1, 0, 0, 3)
    assert t["classes"][0]["f1"] == 1.0 and t["macro"]["f1"] == 1.0


def test_averages_micro_vs_macro():
    good = {"correctness": {"score": 5}, "groundedness": {"score": 5}}
    bad = {"correctness": {"score": 1}, "groundedness": {"score": 1}}
    # scenario A: 3 perfect items (100 each); scenario B: 1 item with every score 1 (0) and a wrong scope decision
    res = [_res("a1", "sA", False, False, good), _res("a2", "sA", False, False, good), _res("a3", "sA", False, False, good),
           _res("b1", "sB", True, False, bad)]
    av = M.averages(res)
    assert av["score"]["micro"] == 75.0 and av["score"]["macro"] == 50.0 and av["score"]["gap"] == 25.0
    assert av["score"]["n_items"] == 4 and av["score"]["n_scenarios"] == 2
    assert av["pass_rate"]["micro"] == 0.75 and av["pass_rate"]["macro"] == 0.5
    m = {x["key"]: x for x in av["metrics"]}
    assert m["correctness"]["micro"] == 4.0 and m["correctness"]["macro"] == 3.0
    assert m["refusal_correctness"]["micro"] == 4.0 and m["refusal_correctness"]["macro"] == 3.0
    assert m["ooc_correct"]["kind"] == "rate" and m["ooc_correct"]["micro"] == 0.75 and m["ooc_correct"]["macro"] == 0.5
    agg = M.aggregate(res)
    # aggregate keeps its historic fields consistent with the new block
    assert agg["overall"]["score_items"] == 75.0 and agg["overall"]["score"] == 50.0
    assert agg["averages"]["score"] == av["score"] and "confusion" in agg
    assert "confusion" not in M.aggregate(res, extras=False)
    cmp = M.compare(agg, M.aggregate(res[:3]))
    assert cmp["headline"]["averages.score.macro"]["delta"] == 50.0
    assert cmp["headline"]["confusion.ooc.macro.f1"]["b"] is not None


def test_confusion_and_aggregates_api(client, bank):
    from eval import db as edb
    pick = {i["id"]: i for i in bank if i["id"] in ("oc-01", "tb-01", "fg-01")}
    rid = edb.create_run("pytest-eval-v2", {"settings": {}}, Q.bank_hash(bank), 3)
    try:
        answers = {"oc-01": ("#OutofContext", True, []),
                   "tb-01": ("| A | B |\n|---|---|\n| x | y [1] |", False, []),
                   "fg-01": ("See the figure [1].", False, [{"chunk_id": "c", "url": "/x.png"}])}
        for iid, (ans, ooc, imgs) in answers.items():
            it = pick[iid]
            rec = {"item_id": iid, "scenario": it["scenario"], "status": "ok", "question": it.get("question") or "q",
                   "answer": ans, "out_of_context": ooc, "images": imgs, "sources": [{"n": 1}], "latency_ms": 1000}
            rec["det_metrics"] = M.item_metrics(it, rec)
            edb.save_result(rid, rec)
        c = client.get(f"/api/eval/runs/{rid}/confusion").json()
        assert (c["ooc"]["tp"], c["ooc"]["tn"]) == (1, 2)
        assert (c["images"]["tp"], c["images"]["fp"]) == (1, 0)
        assert (c["table"]["tp"], c["table"]["fp"]) == (1, 0)
        one = client.get(f"/api/eval/runs/{rid}/confusion?matrix=table").json()
        assert one["key"] == "table" and one["micro"]["f1"] == 1.0
        cell = client.get(f"/api/eval/runs/{rid}/confusion?matrix=images&cell=tp").json()
        assert [x["item_id"] for x in cell] == ["fg-01"] and cell[0]["n_images"] == 1
        assert client.get(f"/api/eval/runs/{rid}/confusion?matrix=nope").status_code == 404
        assert client.get(f"/api/eval/runs/{rid}/confusion?matrix=ooc&cell=xx").status_code == 422
        av = client.get(f"/api/eval/runs/{rid}/aggregates").json()
        assert av["pass_rate"]["n_items"] == 3 and av["score"]["micro"] is None   # not judged yet
        agg = client.get(f"/api/eval/runs/{rid}").json()["aggregate"]
        assert agg["confusion"]["ooc"]["tp"] == 1 and "averages" in agg
        assert client.get("/api/eval/runs/999999999/confusion").status_code == 404
    finally:
        assert client.delete(f"/api/eval/runs/{rid}").status_code == 200
