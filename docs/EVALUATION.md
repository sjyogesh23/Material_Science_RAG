# Evaluation: LLM-as-judge + evaluation matrix

Fully offline evaluation of the NewRAG pipeline. A question bank runs through the **real** pipeline
(`app.rag.pipeline.answer`). A **different, stronger local model** (Ollama) then grades each answer against gold
facts taken from the textbook. Results are stored in Postgres and shown at **`/eval`**.

```
eval/questions.jsonl ──► python -m eval.run ──► eval_results (answers, sources, retrieval, timings, det. metrics)
                                                   │
eval/calibration.jsonl ─► python -m eval.judge ────┘ (1-5 scores + justifications per metric)
                                                   │
                         python -m eval.report / GET /eval ◄── scenario × metric matrix, run comparison
```

## Files
| Path | What |
|---|---|
| `eval/questions.jsonl` | 90 items, 19 scenarios; gold facts written from the chunks, with gold `chunk_ids` / `pages` |
| `eval/calibration.jsonl` | 5 hand-written reference answers (2 good, 3 bad) used to check the judge |
| `eval/questions.py` | load / validate / select / hash the bank |
| `eval/run.py` | runner (CLI + `run()`); recording retriever proxy; archives the eval sessions |
| `eval/judge.py` | judge-model selection (RAM-aware), unloading, rubric prompts, structured outputs, calibration |
| `eval/metrics.py` | deterministic metrics, item score, aggregation into the matrix, run comparison |
| `eval/report.py` | prints the matrix in the terminal (`--compare` for deltas, `--json`) |
| `eval/schema.sql`, `eval/db.py` | `eval_runs`, `eval_items`, `eval_results` (idempotent, applied automatically) |
| `app/api/eval_routes.py` | `/api/eval/...` router (mounted in `app/api/main.py`) |
| `app/static/eval.html`, `js/eval.js`, `css/eval.css` | dashboard (vendored libs only; reuses `render.js`) |
| `tests/test_eval.py` | 26 tests (bank integrity, metrics incl. hand-computed confusion / micro-macro cases, judge plumbing, runner, DB + API) |

## Question bank
Item fields: `id`, `scenario`, `question` **or** `turns[]` (multi-turn; all turns share one session and the
last turn is graded), `expected` (`answer` | `out_of_context`), `format` (`table`/`equation`/`steps`/`image`),
`gold.facts[]`, `gold.chunk_ids[]`, `gold.pages[]`, `gold.figure_chunk_ids[]`, `difficulty`, optional `notes`
(guidance for the grader), `must_not_contain[]` (injection canaries), `paraphrase_of`, `expected_standalone_terms`.

| Scenario | n | What it probes |
|---|---|---|
| factual_definition | 5 | short definitions (tetrahedron, CN, Kirkendall, allotropy, Schottky) |
| explanation | 5 | why-questions (vacancies at T > 0, yield stress gap, heterogeneous nucleation, …) |
| formula_equation | 5 | equation + symbol meanings (Bragg, Fick I, Paris, Nernst, r*) |
| numerical | 5 | calculations from textbook formulas (Fe lattice parameter from Bragg, diffusion depth, hoop stress, grain growth t ∝ R², dislocation spacing) |
| comparison | 5 | edge vs screw, FCC vs HCP, Maxwell vs Kelvin-Voigt, spinodal vs nucleation, NH vs Coble |
| table_request | 5 | tabulated answers (crystal structures, bond energies, creep table, strengthening, invariant reactions) |
| figure_request | 5 | must return the figure (gold figure chunk ids) |
| cross_reference | 5 | multi-chapter links (ρ ↔ work hardening, Arrhenius in three chapters, Fig 7-13 ↔ §7.4.1, …) |
| analogy | 5 | everyday analogy that stays grounded and cited |
| multi_turn_pronoun | 5 | 2-3 turn conversations ("it", "that figure", "what about BCC?"), incl. an in-scope → OOC switch |
| out_of_context | 5 | clearly unrelated (geography, code, sport, cooking, literature) |
| out_of_context_near_domain | 5 | materials topics **absent** from the extracted book (Weibull, Charpy, Vickers, Pilling-Bedworth, Czochralski; verified by grep in `test_near_domain_topics_absent_from_book`) |
| false_premise | 5 | wrong assumption to correct (BCC APF > FCC, FCC→BCC on heating, hoop = ½ longitudinal, …) |
| ambiguous | 4 | "What is D?", "What is the critical radius?", "Explain the transition.", "What's the 0.74 thing?" |
| prompt_injection | 5 | instruction override, prompt leak, PWNED canary, fake source [9], "pretend the textbook says" |
| adversarial_citation | 4 | "cite page 999", non-existent Figure 12-7, "cite ≥10 sources", equation/section/page lookup |
| long_multipart | 4 | 4-6 sub-questions in one message |
| paraphrase_typo | 4 | typo/synonym rewrites of fe-01, fd-03, an-02, co-01 (consistency is measured against the originals) |
| summarization | 4 | summaries of §2.9.3, §8.8, §3.5, §9.1 |

Totals: 90 items (76 answer, 14 out_of_context), 5 multi-turn.

## Metrics
**Judge metrics (1-5, rubric anchors in `eval/judge.py:RUBRICS`)**: correctness, completeness, groundedness
(every claim supported by the source texts the system saw, given to the judge in full), citation_accuracy
(each `[n]` supports its sentence), relevance (also rewards clarifying ambiguous questions), format_compliance,
clarity, plus scenario-specific ones: injection_resistance, false_premise_handling, pronoun_resolution (the judge sees
the system's standalone query), analogy_quality. Metrics are only requested when they apply (see
`applicable_metrics`). Some cases get automatic scores (marked `auto`): a correct refusal needs no judge call, and a
false refusal scores correctness = completeness = 1.

**Deterministic** (`eval/metrics.py`):
- `refusal_correctness`: 5 when the scope decision matches `expected` **and** a refusal is exactly `#OutofContext`, otherwise 1.
- OOC precision / recall / F1 (positive class = out of context), exact-match rate of refusals.
- `citation_coverage`: share of answer sentences / table rows carrying ≥ 1 `[n]`. `invalid_citation_rate`: share of `[n]` pointing to no returned source.
- Retrieval vs gold: `hit_at_k` (gold chunk in the top `final_k` reranked passages), `mrr`, `recall_at_k`, `page_hit`, `hit_any` (incl. cross-ref expansions), `cited_gold`.
- `image_returned` / `image_correct` for figure requests; `must_not_ok` (injection canaries); `format_ok_det` (table / LaTeX present).
- Latency (total, first token), p50/p95; tokens per answer (all LLM calls of the turn).
- Paraphrase consistency: same scope decision, and judge correctness within ±1 of the original item.

**Item score** = weighted mean of the available 1-5 scores mapped to 0-100 (weights: correctness .20,
groundedness .20, refusal .15, completeness .10, citation .10, each special metric .10, relevance/format/clarity .05).
An item **passes** when it has no error, the refusal is correct, no canary leaked, every critical metric
(correctness, groundedness, injection, false premise, pronouns) is ≥ 3 and the item score is ≥ 60.
**Overall score** = mean of the scenario scores (scenario-balanced), shown next to the plain item mean. Scores stay
empty until a run is judged.

**Confusion matrices** (`metrics.confusion`, `metrics.binary_report`; computed from stored results, no re-judging):
| Matrix | Positive class = expected when | Predicted positive when |
|---|---|---|
| `ooc` | `item.expected == "out_of_context"` | answer is `#OutofContext` (stored `det_metrics.pred_ooc`) |
| `images` | `item.format == "image"` or scenario `figure_request` | at least one image was returned with the answer |
| `table` | `item.format == "table"` | the answer contains a markdown table (pipe row followed by a `|---|` separator; refusals never count) |

Each gives TP/FP/FN/TN plus the item ids per cell, accuracy, TPR/FPR/FNR/TNR, and **both classes** scored with
precision / recall / F1 / support (the negative class uses TN as its hits). Undefined ratios (0/0) are `null`.
Averages: **micro** = pooled over both classes (correct / items; for a single-label yes/no decision micro P = R = F1 =
accuracy); **macro** = unweighted mean over the classes present in the gold labels (support > 0; a present class that
is never predicted counts as precision 0); **weighted** = the same weighted by support. Macro is the honest number for
the rare positive classes (14 OOC, 5 figure, 5 table items of 90).

**Micro vs macro aggregation** (`metrics.averages`): for the item score (0-100), the pass rate, every 1-5 metric and
every deterministic rate, micro = mean over all items that have the value (item-weighted) and macro = mean of the
per-scenario means (each scenario counts once), plus gap = micro − macro and the item / scenario counts.
`overall.score` (macro) and `overall.score_items` (micro) are the same two numbers for the item score; the headline
pass rate is macro. With the full bank (19 scenarios of 4-5 items) the two are within ~0.3 points; they diverge on
subsets or when a scenario is much larger.

**Matrix** = scenario × metric means (scores 1-5; rates in %), plus each scenario's score, pass rate and n.
**Comparison** = cell-wise B − A, headline deltas (score, pass rate, OOC F1/P/R, latency, hit@k, consistency) and
per-item deltas.

## Judge model
Candidates, strongest first: `qwen3.5:9b` (pulled; 6.55 GB on disk incl. vision projector, Q4_K_M), then
`gemma3:12b-it-qat` and `qwen3:8b` (not installed: C: has about 5.8 GB free). `qwen3:4b` is the last resort and is
flagged as the same model as the answerer. Selection: unload every resident Ollama model (`keep_alive=0`), measure
free RAM (psutil) and VRAM (nvidia-smi), and pick the first installed candidate whose estimated footprint
(qwen3.5:9b ≈ 7.3 GB) is at most free RAM + 0.9 × free VRAM − 1.0 GB. `--force` overrides the check, and `--model`
picks a model explicitly. Calls use `think=false` (qwen3.5 can think), temperature 0, seed 7, `num_ctx` 8192, and
`format=<JSON schema>` with one `{justification, score∈1..5}` object per metric. Each call is logged in `llm_calls`
(purpose `eval_judge`). The RAM before and after loading is stored in `eval_runs.judge`. When the run finishes, the
judge model is unloaded so the app reloads its own models.

**Calibration**: before judging, the 5 reference answers are graded. Checks: good answers have a mean ≥ 4, bad
answers ≤ 2.5, the injection-complying answer has injection_resistance ≤ 2, and every good answer outranks the bad
answer for the same item. `agreement` = passed checks / all checks (7 checks). It is shown in the runs table and in
the Judge panel.

## Commands
```powershell
# answer (≈ 30 s/question with qwen3:4b → ~50-60 min for all 90 items)
.venv\Scripts\python -m eval.run --label baseline
.venv\Scripts\python -m eval.run --label quick --scenarios out_of_context,prompt_injection --limit 5
# judge (stop other RAM-heavy work first; unloads the chat/reranker/embedder models)
.venv\Scripts\python -m eval.judge --run baseline
.venv\Scripts\python -m eval.judge --run baseline --dry-run --ids fe-01   # inspect prompts, no model
.venv\Scripts\python -m eval.judge --calibrate-only
# matrix in the terminal / diff two runs
.venv\Scripts\python -m eval.report --run baseline
.venv\Scripts\python -m eval.report --run tuned --compare baseline
.venv\Scripts\python -m pytest tests/test_eval.py -q
```
UI: `http://127.0.0.1:8000/eval` (deep links: `?run=<id>`, `&cmp=<other id>`, `&item=<item id>`). "Start run" and
"Judge" spawn the same CLIs as background processes (one job at a time; logs in `logs/eval_<kind>_<run>.log`). "Stop
after current item" sets the status to `cancelling`.

## API (`/api/eval`)
`GET /meta`, `GET /questions`, `GET /runs`, `GET /runs/{id}` (run + aggregate), `GET /runs/{id}/items?scenario=&failed=`,
`GET /runs/{id}/items/{item}` (with full source texts), `GET /runs/{id}/confusion[?matrix=ooc|images|table[&cell=tp|fp|fn|tn]]`
(all three matrices; one matrix; or one cell's items with question, answer preview, score, pass), `GET /runs/{id}/aggregates`
(micro vs macro block), `GET /compare?a=&b=`, `POST /runs` {label, scenarios?, limit?, judge?},
`POST /runs/{id}/judge?rejudge=&model=`, `POST /runs/{id}/cancel`, `GET /runs/{id}/log`, `DELETE /runs/{id}`.
`GET /runs/{id}` and `/compare` aggregates also carry `confusion` and `averages`; compare headline deltas include
`averages.score.micro|macro`, `averages.pass_rate.micro|macro` and `confusion.{ooc,images,table}.macro.f1`.
`eval.report` prints the three matrices (per-class + micro/macro/weighted, compared run's F1 in brackets) and the
micro/macro table.

## Dashboard (`/eval`)
Runs table (tick two to compare) · KPI strip · scenario × metric heatmap (scores / change view) · **Classification
checks**: the three 2×2 matrices (count, TP/FP/FN/TN + correct/error label, share of row; blue = correct, orange =
error, colour always doubled by text), hover for row/column shares and run A's count, select a cell to list its
questions, select a question for the drill-down; per-class + micro/macro/weighted table with Δ F1 vs the compared run ·
**Micro vs macro** table (with A values and deltas when comparing) · **Charts** section (inline SVG, shared tooltip and
toggle legend from `js/charts.js`, validated categorical slots blue = selected run, orange = compared run): score by
scenario (grouped bars + Δ column, select → items filter), judge metrics (grouped bars from 1, micro/macro switch),
latency histogram (total / first-token switch, select a bin → items latency filter), score vs latency scatter
(pass = circle, fail = triangle, pass line at 60, drag to zoom an x range, select a point → drill-down) · Items
list + drill-down. Screenshots: `logs/qa/v2/eval_*.png`.

## Notes and limitations
- Eval sessions are real chat sessions. After each item they are archived (title `[eval] <id>`) and their memory
  file is deleted, so the chat sidebar stays clean. Their `llm_calls` rows do show up in the token audit.
- The judge shares the textbook's view of the facts. Where the book is unusual (e.g. "Rankine is the most
  conservative criterion", Δσ_y = κ/d_g), the gold follows the book.
- The numerical gold assumes 44.7° is 2θ (Fe a ≈ 2.87 Å). The book does not state the answer.
- Ambiguous items have no single right answer, so they rely on the relevance rubric. Expect more judge noise there.
- One local judge (not an ensemble), and calibration uses only 5 references. Treat differences under about 0.3 on
  the 1-5 scale as noise.
- Judging and the chat app compete for RAM and VRAM. Do not judge while someone is using the chat.

## Score history
| Run | Label | Overall | Pass | OOC P / R | Notes |
|---|---|---|---|---|---|
| 48 | baseline | 83.3 (micro 83.5) | 0.776 (micro 0.778) | 0.59 / 0.93 | 9 false refusals, all emitted by the answer model despite rerank >= 0.95. OOC macro F1 0.826; figure-returned macro F1 0.557 (25 unasked figures); table macro F1 0.794 (6 unasked tables) |
| - | r1-dev-1 (23 items: analogy, false_premise, adversarial_citation, ambiguous, prompt_injection) | 97.0 | 0.95 | 1.0 / 1.0 | analogy 100, false_premise 100, injection 100, adversarial 97, ambiguous 88.2 |
| 52 | round1 (83 of 90 items answered) | 92.4 | 0.875 | 1.0 / 1.0 | weakest: comparison 73.5, false_premise 80, analogy 82.1, ambiguous 82.7, figure 86.7, table 88.6 |

Round-1 changes: question-form guardrails (`app/rag/guardrails.py`, `decisions.question_forms`,
`topic_query`, `strip_injection`, `enumerated_items`), prompt rules 4/5 rewritten (form never decides scope),
refusal retry (`refusal_retry`, `refusal_retry_min_score`), prompt-leak / canary output guards.
