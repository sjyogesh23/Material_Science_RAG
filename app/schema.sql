-- NewRAG schema (idempotent: safe to run on every start). Requires `CREATE EXTENSION vector` first.

CREATE TABLE IF NOT EXISTS documents (
    doc_id          text PRIMARY KEY,
    title           text,
    file_name       text,
    pdf_path        text,                       -- absolute path of the source PDF
    total_pages     int,
    extracted_pages int,
    meta            jsonb,
    ingested_at     timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      text PRIMARY KEY,
    doc_id        text REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index   int,
    chunk_type    text,                         -- text | figure | table
    text          text,                         -- embedded text ("<doc title> | <section path>\n\n<content>")
    content       text,                         -- raw content shown to the LLM / user
    header        text,
    section_path  text[],
    section_role  text,
    pages         int[],
    page_start    int,
    page_end      int,
    token_count   int,
    content_hash  text,                         -- sha256(embed_model + text): re-embed only on change
    payload       jsonb,                        -- full chunk record (locations, figure, table, entities, formulas ...)
    embedding     vector(1024),
    tsv           tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc_id ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_chunk_type ON chunks (chunk_type);

-- extracted element id -> chunk that contains it
CREATE TABLE IF NOT EXISTS element_chunks (
    doc_id     text,
    element_id text,
    chunk_id   text,
    PRIMARY KEY (doc_id, element_id)
);
CREATE INDEX IF NOT EXISTS element_chunks_chunk ON element_chunks (chunk_id);

-- chunk -> chunk links (equation refs, variable definitions, figure/table mentions, figure context ...)
CREATE TABLE IF NOT EXISTS chunk_links (
    src_chunk_id text,
    dst_chunk_id text,
    kind         text,
    label        text,
    PRIMARY KEY (src_chunk_id, dst_chunk_id, kind)
);
CREATE INDEX IF NOT EXISTS chunk_links_dst ON chunk_links (dst_chunk_id);

CREATE TABLE IF NOT EXISTS sessions (
    session_id    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    title         text,
    created_at    timestamptz DEFAULT now(),
    updated_at    timestamptz DEFAULT now(),
    memory_path   text,
    message_count int DEFAULT 0,
    archived      bool DEFAULT false
);

CREATE TABLE IF NOT EXISTS messages (
    message_id       bigserial PRIMARY KEY,
    session_id       uuid REFERENCES sessions(session_id) ON DELETE CASCADE,
    role             text,
    content          text,
    standalone_query text,
    decisions        jsonb,
    sources          jsonb,
    images           jsonb,
    out_of_context   bool DEFAULT false,
    created_at       timestamptz DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_session ON messages (session_id, message_id);
-- per-answer token usage ({by_purpose, prompt_tokens, completion_tokens, total, timings_ms}); added after v1
ALTER TABLE messages ADD COLUMN IF NOT EXISTS usage jsonb;

CREATE TABLE IF NOT EXISTS llm_calls (
    id                bigserial PRIMARY KEY,
    session_id        uuid NULL,
    message_id        bigint NULL,
    purpose           text,     -- router | answer | memory_compaction | title | grounding_check | embed_query | rerank | ingest_embed
    provider          text,     -- ollama | local
    model             text,
    prompt_tokens     int,
    completion_tokens int,
    total_tokens      int,
    duration_ms       int,
    ok                bool DEFAULT true,
    error             text,
    meta              jsonb,
    created_at        timestamptz DEFAULT now()
);
CREATE INDEX IF NOT EXISTS llm_calls_created ON llm_calls (created_at);
CREATE INDEX IF NOT EXISTS llm_calls_session ON llm_calls (session_id);

CREATE TABLE IF NOT EXISTS app_settings (
    key        text PRIMARY KEY,
    value      jsonb,
    updated_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS api_keys (
    name       text PRIMARY KEY,
    value      text,
    note       text,
    updated_at timestamptz DEFAULT now()
);

-- ============================================================================ v2 (FEATURES_V2 §1, §2, §4)
-- Branching conversations: every message points at the previous message on its path (NULL = first user message
-- of a branch root). Siblings = same (session_id, parent_id, role). messages.mode = answer mode (swift | deep);
-- rows written by v1 code have mode NULL, which is also the marker for the parent_id backfill below.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS parent_id bigint NULL REFERENCES messages(message_id) ON DELETE CASCADE;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS mode text;
CREATE INDEX IF NOT EXISTS messages_parent ON messages (session_id, parent_id);
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS active_leaf_id bigint NULL;

-- Backfill (idempotent): a v1 message (mode IS NULL) without a parent gets the previous message of its session.
UPDATE messages m SET parent_id = p.prev
FROM (SELECT message_id, lag(message_id) OVER (PARTITION BY session_id ORDER BY message_id) AS prev
      FROM messages) p
WHERE m.message_id = p.message_id AND m.parent_id IS NULL AND m.mode IS NULL AND p.prev IS NOT NULL;
-- sessions without an active leaf: their newest message
UPDATE sessions s SET active_leaf_id = x.leaf
FROM (SELECT session_id, max(message_id) AS leaf FROM messages GROUP BY session_id) x
WHERE s.session_id = x.session_id AND s.active_leaf_id IS NULL;

-- Feedback (like / dislike) per assistant message + LLM reviews of dislikes
CREATE TABLE IF NOT EXISTS message_feedback (
    message_id bigint PRIMARY KEY REFERENCES messages(message_id) ON DELETE CASCADE,
    rating     smallint NOT NULL CHECK (rating IN (-1, 1)),
    comment    text,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);
CREATE TABLE IF NOT EXISTS feedback_reviews (
    message_id bigint PRIMARY KEY REFERENCES messages(message_id) ON DELETE CASCADE,
    review     jsonb,      -- {issue_type, explanation, suggested_fix, confidence}
    model      text,
    created_at timestamptz DEFAULT now()
);
