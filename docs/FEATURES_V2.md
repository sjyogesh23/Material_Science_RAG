# Features v2 - contract (2026-10-19)

Second feature set: feedback, branching, session menu, answer modes, richer audit/eval pages and
versioned pipeline components. All earlier behaviour is kept, except the API-keys feature, which was removed.

## 1. Feedback (like / dislike) - audited
- Table `message_feedback(message_id bigint PK REFERENCES messages ON DELETE CASCADE, rating smallint
  CHECK (rating IN (-1,1)), comment text, created_at, updated_at)`.
- `POST /api/messages/{id}/feedback {rating: 1|-1|0, comment?: str}` (0 = remove) → `{message_id, rating}`.
- Messages API returns `feedback: {rating, comment} | null` per assistant message.
- Audit: feedback KPIs (likes, dislikes, ratio by day / by mode / by intent), list of disliked answers
  with question + answer + sources, and **"Review dislikes"**: an LLM review (purpose `feedback_review`,
  structured output {issue_type: wrong_fact|missing_citation|bad_format|refused_wrongly|incomplete|other,
  explanation, suggested_fix}) stored in `feedback_reviews(message_id, review jsonb, model, created_at)`.
  Endpoints: `GET /api/feedback?rating=-1&limit=`, `POST /api/feedback/review {message_ids?: [..]}` (reviews
  unreviewed dislikes, returns reviews), `GET /api/feedback/summary`.

## 2. Branching conversations (edit / regenerate / go back)
- `messages` gains `parent_id bigint NULL REFERENCES messages(message_id) ON DELETE CASCADE` (previous
  message on the path; NULL for the first user message) and `mode text`. Existing rows: backfill parent_id
  as the previous message of the session by created_at (idempotent migration).
- `sessions` gains `active_leaf_id bigint NULL`.
- Siblings = messages with the same (session_id, parent_id, role). A user edit = new user message with the
  SAME parent as the edited one; a regenerate = new assistant message with the same parent (the user msg).
- `POST /api/chat` body: `{session_id?, message?, mode?: "swift"|"deep", parent_id?: int,
  edit_of?: int, regenerate_of?: int}`
  - default parent = session.active_leaf_id;
  - `edit_of=<user msg id>` + `message` → new sibling user message (parent = edited.parent_id), then answer;
  - `regenerate_of=<assistant msg id>` → new sibling assistant answer to the same user message (no message text).
  - The answer's context (session memory + pronoun resolution) uses ONLY the path root → new leaf.
  - Emits existing events; `session` event adds `{active_leaf_id}`; `done` adds `{message_id, user_message_id,
    parent_id}`.
- `GET /api/sessions/{id}/messages` → active path only, each item
  `{message_id, parent_id, role, content, sources, images, decisions, usage, out_of_context, feedback, mode,
    created_at, sibling_ids: [..], sibling_index: int, sibling_count: int}`.
- `POST /api/sessions/{id}/switch {message_id}` → sets active leaf to the newest leaf under that message,
  rewrites the session memory file from the new path, returns the new path (same shape as messages).
- Session memory for a branch: `SessionMemory.rebuild_from_path(turns)` (keeps the summary only for the
  common prefix; otherwise rebuilds Recent turns + entities from the path and compacts if over budget).

## 3. Session ⋮ menu
- Sidebar row = title + a single ⋮ (kebab) button (aria-haspopup="menu"). Menu items: **Rename**,
  **Show session memory**, **Delete**. Memory opens a dialog rendering `GET /api/sessions/{id}/memory`
  (markdown, tokens / budget, a "Compact now" button → `POST /api/sessions/{id}/memory/compact`).
  Keyboard accessible (Enter/Space opens, arrows move, Esc closes).

## 4. Answer modes (user-selectable in the composer)
- `GET /api/modes` → `[{id, name, tagline, description, settings}]`:
  - `swift` - **Swift**: time over effect. Chat model qwen3:4b-instruct-2507, fast_route on, rerank_top_n
    20, final_k 6, max_answer_tokens 900 (round-1 tuned values; 12/5/700 until round 2), no grounding check, no thinking.
  - `deep` - **Deep**: effect over time. Chat model `qwen3.5:9b` (already pulled for the judge) with
    thinking ON (thinking text streamed as `thinking` events, not stored in the answer), router always on,
    rerank_top_n 30, final_k 8, max_answer_tokens 1400, grounding_check on, one self-revision pass when the
    grounding score < 3 (round 2: was < 4; the revision may only fix flagged sentences, never refuse). Warn in the UI that Deep is slower (≈1–3 min on this laptop).
- Mode stored per message (`messages.mode`) and in llm_calls.meta; audit charts split by mode.
- Mode profiles are versioned config (component `modes`, see §6).

## 5. Audit & eval UIs
- Remove the API-keys feature entirely (UI section, `/api/keys*` endpoints, tests). Keep settings.
- Charts must be interactive (hover tooltips, click-to-filter, range selector for days, legend toggles),
  inline SVG, no external chart lib (vendored-only rule), dataviz principles (colour-blind safe).
- Eval additions:
  - Confusion matrices: (a) OOC classifier (expected answer/out_of_context vs predicted); (b) needs_images
    (figure expected vs images returned); (c) answer_format (expected table vs produced table) - clickable
    cells list the items.
  - Per-class precision / recall / F1 with **micro** and **macro** averages (and support); also score
    aggregation micro (item-weighted mean) vs macro (scenario-weighted mean) shown side by side.
  - Charts: scenario score bars (run vs compare run), metric radar or grouped bars, latency distribution,
    score-vs-latency scatter per item (click → item drill-down).

## 6. Architecture view + versioned components (in the audit page, tab "Architecture")
- Flow diagram (inline SVG, laid out top→bottom) of the real pipeline:
  User message → injection strip / guardrails-in → session memory → Jev-style router (decisions: in_scope,
  intent, question_forms, needs_images, answer_format, standalone_query, sub_queries) → retrieval
  (embed → dense pgvector ∥ BM25 → RRF → rerank → coverage / cross-refs / figures) → OOC gate →
  prompt builder (system prompt + form notes + memory + sources) → LLM (mode: Swift/Deep) →
  output guardrails (refusal retry, injection canaries, citation validation) → citations + highlights →
  memory update / compaction → UI. Each node shows live stats from llm_calls (calls, avg ms, tokens) and
  is clickable.
- Clicking a node opens an editor for its versioned components:
  `answer_system_prompt`, `router_prompt`, `form_notes`, `table_rules`, `compaction_prompt`,
  `title_prompt`, `grounding_prompt`, `retrieval_params` (top-k, rrf_k, rerank_top_n, final_k, thresholds),
  `modes`.
- Table `component_versions(id bigserial PK, component text, version int, content jsonb, note text,
  created_at, created_by text default 'ui', is_active bool, UNIQUE(component, version))`.
  Version 1 of each component is seeded from the current code defaults (idempotent).
- Endpoints: `GET /api/arch` (graph: nodes, edges, per-node stats, component ids),
  `GET /api/components`, `GET /api/components/{c}` (active + history list),
  `GET /api/components/{c}/versions/{v}`, `POST /api/components/{c}` {content, note} → new active version,
  `POST /api/components/{c}/versions/{v}/activate`, `GET /api/components/{c}/diff?a=&b=` (unified diff).
- Runtime: `app/versioning.py` `get_component(name) -> content` (cached, invalidated on change) used by
  prompts.py / pipeline / config instead of hard-coded constants (code defaults remain the fallback).
  llm_calls.meta records the component versions used, so eval runs can be compared per version.
