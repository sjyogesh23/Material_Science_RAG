-- NewRAG evaluation tables (idempotent; applied by eval.db.ensure_schema()).
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id       bigserial PRIMARY KEY,
    label        text NOT NULL,
    status       text NOT NULL DEFAULT 'created',   -- created | running | answered | judging | done | failed | cancelled
    created_at   timestamptz NOT NULL DEFAULT now(),
    started_at   timestamptz,
    finished_at  timestamptz,
    bank_hash    text,
    n_items      int DEFAULT 0,
    config       jsonb DEFAULT '{}'::jsonb,         -- settings snapshot (models, thresholds), CLI filters
    progress     jsonb DEFAULT '{}'::jsonb,         -- {phase, done, total, current, pid}
    judge        jsonb DEFAULT '{}'::jsonb,         -- {model, ram, calibration, started_at, finished_at}
    notes        text
);
CREATE INDEX IF NOT EXISTS eval_runs_label_idx ON eval_runs(label);

CREATE TABLE IF NOT EXISTS eval_items (
    item_id      text PRIMARY KEY,
    scenario     text NOT NULL,
    payload      jsonb NOT NULL,                    -- the full question-bank row
    bank_hash    text,
    updated_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_results (
    run_id        bigint NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
    item_id       text NOT NULL,
    scenario      text NOT NULL,
    status        text NOT NULL DEFAULT 'pending',  -- pending | ok | error
    session_id    uuid,
    question      text,                             -- final user turn
    answer        text,                             -- final assistant answer
    turns         jsonb DEFAULT '[]'::jsonb,        -- every turn: question, answer, decision, sources, images, ...
    decision      jsonb,
    sources       jsonb DEFAULT '[]'::jsonb,
    images        jsonb DEFAULT '[]'::jsonb,
    retrieved     jsonb DEFAULT '[]'::jsonb,        -- every passage the retriever returned (final turn)
    out_of_context boolean DEFAULT false,
    uncited       boolean DEFAULT false,
    timings       jsonb DEFAULT '{}'::jsonb,
    usage         jsonb DEFAULT '{}'::jsonb,
    latency_ms    int,
    ttft_ms       int,
    error         text,
    det_metrics   jsonb DEFAULT '{}'::jsonb,        -- deterministic metrics (eval.metrics.item_metrics)
    judge         jsonb DEFAULT '{}'::jsonb,        -- {metric: {score, justification}}
    judge_model   text,
    judged_at     timestamptz,
    created_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, item_id)
);
CREATE INDEX IF NOT EXISTS eval_results_scenario_idx ON eval_results(run_id, scenario);
