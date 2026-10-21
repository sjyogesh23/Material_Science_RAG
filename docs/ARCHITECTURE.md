# NewRAG - Architecture contract (v1)

Offline materials-science RAG chatbot over the Steimel textbook. **This file is the contract between
modules**: the names, signatures and data shapes each module implements.

## Hard constraints
- 100 % offline at runtime. No cloud APIs. LLM = **Ollama** (`http://127.0.0.1:11434`).
- Windows 11, Python 3.12 venv at `.venv`, RTX 3050 **4 GB VRAM**, 15.6 GB RAM.
  All models are served by Ollama (chat on GPU+CPU, reranker on GPU, embedder on CPU); the API process does not import torch.
- Postgres for everything (embedded via `pgserver`, PG 16.2 + pgvector 0.6.2). Never call `srv.psql()`
  (breaks on paths with spaces) - use psycopg only. Data dir: `E:\NewRAG\pgdata`.
- Frontend works offline: all JS/CSS/fonts vendored under `app/static/vendor/` (no CDN at runtime).

## Existing inputs (already produced)
- `chunks/chunks.jsonl` - 375 chunks (241 text / 130 figure / 4 table) from `chunk.py`.
  Fields: chunk_id, chunk_type, text (embed text = "<doc title> | <section path>\n\n<content>"), content,
  header, token_count, doc_id, doc_title, file_name, doc_category, source, section_path, section_role,
  categories, pages, page_start, page_end, element_ids, locations[{element_id,type,page,bbox,line_start,line_end}],
  entities, formulas, figure{element_id, figure_number, image_file, image_type, chart_type, values_estimated},
  table{...}, prev_chunk_id, next_chunk_id, language.
- `extracted/textbooks/Materials_Science_and_Engineering/Materials_Science_and_Engineering.json` - full
  element-level extraction (elements have `cross_references[{kind,label,target_id}]`, `citations`).
- PDF: `data/textbooks/Materials Science and Engineering.pdf` (doc_id `doc_f49a86e05ec4`, 211 pages; pages
  181-211 not extracted). bbox = PDF points, top-left origin.
- Models (updated after perf fix): embeddings Ollama `qwen3-embedding:0.6b` (1024-d, CPU; queries prefixed
  with `"Instruct: {task}\nQuery: "`), reranker Ollama `qwen3-reranker:0.6b-q8_0` (built locally by
  `scripts/build_reranker_gguf.py`, scored via /api/generate yes/no logprobs), LLM
  `qwen3:4b-instruct-2507-q4_K_M` (Ollama).

## Package layout
```
run.py                      one-command launcher (pg + ollama + ingest-if-needed + uvicorn) → prints links
app/
  __init__.py
  config.py                 Settings (from .env) + runtime overrides (DB app_settings)
  db.py                     embedded Postgres + psycopg pool + migrations
  schema.sql
  ingest.py                 chunks.jsonl (+ extracted JSON) → Postgres (embeddings, links); CLI
  tokens.py                 count_tokens(text) with the Qwen3 tokenizer
  retrieval/
    __init__.py
    embedder.py             QueryEmbedder (singleton)
    reranker.py             Reranker (singleton, Qwen3-Reranker, sigmoid 0-1 scores)
    bm25.py                 BM25Index (bm25s, scientific tokenizer, built from DB)
    hybrid.py               Retriever (dense+bm25 → RRF → rerank → cross-ref expansion)
  llm/
    __init__.py
    ollama_client.py        LLM wrapper, records every call in llm_calls
    decisions.py            local "Jev-style" decision layer (typed questions → typed answers)
    prompts.py              all system prompts / templates
  memory/
    __init__.py
    session_memory.py       per-session markdown memory file + compaction
  rag/
    __init__.py
    pipeline.py             answer() orchestration → async event stream
    citations.py            citation validation / renumbering / highlight payloads
  api/
    __init__.py
    main.py                 FastAPI app (SSE chat, sessions, pages, figures, audit, settings, keys)
  static/
    chat.html  audit.html  css/  js/  vendor/  fonts/
scripts/compact_memory.py   CLI: compact session memory files that exceed the budget
sessions/                   <session_id>.md memory files (git-ignored)
cache/pages/                rendered PDF page PNGs (git-ignored)
tests/                      pytest + e2e question set
```

## config.py
```python
@dataclass
class Settings:  # values from .env, then overridden by app_settings rows (key = field name, lowercase)
    root: Path; pg_data_dir: Path            # ROOT/"pgdata"
    database_url: str | None                 # env DATABASE_URL; if None → db.start_postgres()
    api_host: str = "127.0.0.1"; api_port: int = 8000
    ollama_host: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen3:4b-instruct-2507-q4_K_M"; router_model: str = <same>
    num_ctx: int = 8192; temperature: float = 0.1; max_answer_tokens: int = 900
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"; embed_device: str = "cpu"
    rerank_model: str = "Qwen/Qwen3-Reranker-0.6B"; rerank_device: str = "cpu"
    top_k_dense: int = 40; top_k_bm25: int = 40; rrf_k: int = 60
    rerank_top_n: int = 30; final_k: int = 6
    ooc_threshold: float = 0.20      # max rerank prob below this → #OutofContext
    ooc_soft_threshold: float = 0.45 # if router says out-of-scope AND max prob < this → #OutofContext
    sessions_dir: Path; page_cache_dir: Path
    memory_budget_tokens: int = 1500; memory_keep_turns: int = 6
def get_settings() -> Settings          # cached; reload_settings() after PUT /api/settings
```

## db.py
```python
def start_postgres() -> str                       # pgserver.get_server(pg_data_dir, cleanup_mode="stop"); returns URI
def init_db() -> None                             # start (if needed), CREATE EXTENSION vector, run schema.sql, open pool
def get_pool() -> psycopg_pool.ConnectionPool     # sync pool, configure=register_vector
def get_async_pool() -> psycopg_pool.AsyncConnectionPool   # configure=register_vector_async
```

## schema.sql (idempotent)
```sql
documents(doc_id text PK, title text, file_name text, pdf_path text, total_pages int, extracted_pages int,
          meta jsonb, ingested_at timestamptz default now())
chunks(chunk_id text PK, doc_id text REFERENCES documents ON DELETE CASCADE, chunk_index int, chunk_type text,
       text text, content text, header text, section_path text[], section_role text, pages int[],
       page_start int, page_end int, token_count int, content_hash text, payload jsonb,
       embedding vector(1024), tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED)
  -- indexes: hnsw (embedding vector_cosine_ops), gin(tsv), (doc_id), (chunk_type)
element_chunks(doc_id text, element_id text, chunk_id text, PRIMARY KEY(doc_id, element_id))
chunk_links(src_chunk_id text, dst_chunk_id text, kind text, label text, PRIMARY KEY(src_chunk_id,dst_chunk_id,kind))
sessions(session_id uuid PK default gen_random_uuid(), title text, created_at timestamptz, updated_at timestamptz,
         memory_path text, message_count int default 0, archived bool default false)
messages(message_id bigserial PK, session_id uuid REFERENCES sessions ON DELETE CASCADE, role text,
         content text, standalone_query text, decisions jsonb, sources jsonb, images jsonb,
         out_of_context bool default false, created_at timestamptz default now(),
         usage jsonb)   -- per-answer usage event payload (ALTER TABLE ... ADD COLUMN IF NOT EXISTS)
llm_calls(id bigserial PK, session_id uuid NULL, message_id bigint NULL, purpose text, provider text,
          model text, prompt_tokens int, completion_tokens int, total_tokens int, duration_ms int,
          ok bool default true, error text, meta jsonb, created_at timestamptz default now())
app_settings(key text PK, value jsonb, updated_at timestamptz default now())
api_keys(name text PK, value text, note text, updated_at timestamptz default now())
```
`purpose` vocabulary: `router`, `answer`, `memory_compaction`, `title`, `grounding_check`,
`embed_query` (provider `local`), `rerank` (provider `local`), `ingest_embed` (provider `local`).

## ingest.py  (`python -m app.ingest [--chunks chunks/chunks.jsonl] [--force]`)
Upsert documents + chunks; embed `text` (documents: no prompt) with embed_model on **cuda if available**
(ingest is offline batch), skip unchanged `content_hash`. Fill `element_chunks` from chunk.element_ids.
Fill `chunk_links` from extracted JSON element `cross_references[*].target_id` (map target element →
chunk via element_chunks; also figure/table caption links). Record one `llm_calls` row (purpose
`ingest_embed`). Delete chunks of the doc that no longer exist.

## retrieval
```python
@dataclass
class Hit:
    chunk_id: str; doc_id: str; chunk_type: str; header: str; content: str; text: str
    section_path: list[str]; pages: list[int]; payload: dict
    dense_rank: int | None = None; bm25_rank: int | None = None
    rrf_score: float = 0.0; rerank_score: float | None = None   # sigmoid prob 0..1
    via: str = "hybrid"        # "hybrid" | "cross_ref" | "neighbor" | "figure_search"

class Retriever:
    def __init__(self, settings=None): ...            # lazy-loads embedder, reranker, bm25 from DB
    def search(self, query: str, *, k: int | None = None, chunk_types: list[str] | None = None,
               sub_queries: list[str] | None = None, session_id=None, message_id=None) -> list[Hit]
        # dense(pgvector cosine, top_k_dense) + bm25(top_k_bm25) for query and each sub_query →
        # RRF(k=rrf_k) → rerank top rerank_top_n against `query` → return top k (final_k) sorted by rerank
    def expand_cross_refs(self, hits: list[Hit], max_extra: int = 3) -> list[Hit]   # follow chunk_links
    def figures_for(self, query: str, hits: list[Hit], k: int = 3) -> list[Hit]     # figure chunks: linked or searched
    def coverage_for(self, sub_queries, per_query=2) -> list[Hit]      # best text/table hits PER sub-query (via="sub_query")
    def section_text_for(self, chunk_ids, query=None, k=2) -> list[Hit] # text explaining a figure/table (via="section_text")
    def figure_by_number(self, label) -> list[Hit]                       # "Figure 2-7" -> its chunk
    def reload_bm25(self) -> None
def get_retriever() -> Retriever    # process-wide singleton
```
BM25 tokenizer keeps chemical formulas / alloys / numbers whole (`Fe3C`, `Al2O3`, `304L`, `0.2`), lowercases,
Snowball-stems pure-alpha tokens, English stopwords. Log `embed_query` and `rerank` usage rows
(prompt_tokens = tokens processed).

## llm/ollama_client.py
```python
@dataclass
class Usage: prompt_tokens: int; completion_tokens: int; duration_ms: int; model: str
class LLM:
    async def complete(self, messages: list[dict], *, purpose: str, model: str | None = None,
                       format: dict | None = None, options: dict | None = None,
                       session_id=None, message_id=None) -> tuple[str, Usage]
    async def stream(self, messages, *, purpose, model=None, options=None, session_id=None,
                     message_id=None) -> AsyncIterator[str]   # yields tokens; usage recorded at end,
                                                               # available as self.last_usage
    async def health(self) -> dict    # {"ok": bool, "models": [...], "chat_model_present": bool}
def get_llm() -> LLM
```
Always `think=False`, `options={"temperature": settings.temperature, "num_ctx": settings.num_ctx}` merged.
Every call → `llm_calls` row (prompt_eval_count / eval_count from Ollama). Errors recorded with ok=false.

## llm/decisions.py - local Jev-style decision layer
Jev (TypeSafe AI) is cloud-only (SDK needs Python ≥3.14 + API key), so we reproduce its model locally:
typed questions → one structured-output call → typed answers with confidence.
```python
class Choice(BaseModel): name: str; question: str; options: list[str]
class Noul(BaseModel):   name: str; question: str            # yes/no → bool + confidence
class Score(BaseModel):  name: str; question: str; levels: list[str]
async def decide(state: str, questions: list[Choice|Noul|Score], *, purpose="router", **ids) -> dict[str, Any]
     # → {name: value, f"{name}_confidence": float 0..1}; schema built dynamically, validated with pydantic,
     #   one retry, then safe defaults.

class RouteDecision(BaseModel):
    in_scope: bool; in_scope_confidence: float
    intent: Literal["greeting","definition","explanation","comparison","calculation","cross_reference",
                    "figure_request","table_request","summary","other"]
    needs_images: bool; answer_format: Literal["prose","table","steps","equation"]
    is_cross_reference: bool
    standalone_query: str      # pronouns/ellipsis resolved using session memory
    sub_queries: list[str]     # ≤3, only for cross_reference / comparison
async def route(question: str, memory_md: str, **ids) -> RouteDecision
```
Cheap rules may override (e.g. "table"/"tabulate" → answer_format=table; "figure/diagram/show me/image"
→ needs_images).

## memory/session_memory.py
File `sessions/<session_id>.md`:
```
# Session <id>
_Title: ..._
## Summary
<rolling LLM summary of older turns; empty at start>
## Entities in focus
- <entity> - <one-line note> (turn N)
## Recent turns
### Turn N - User
...
### Turn N - Assistant
... (answer text without the sources block; "#OutofContext" kept verbatim)
```
```python
class SessionMemory:
    @classmethod
    def load(cls, session_id) -> "SessionMemory"
    def add_turn(self, user: str, assistant: str, entities: list[str]) -> None
    def render(self, budget_tokens: int) -> str      # for the system prompt; newest turns first to fit
    def token_count(self) -> int
    def needs_compaction(self) -> bool               # > settings.memory_budget_tokens
    async def compact(self, llm) -> None             # summarise all but last memory_keep_turns turns
    def save(self) -> None
```
`scripts/compact_memory.py [--session ID | --all] [--force]` runs `compact()` from the CLI.

## rag/pipeline.py
```python
async def answer(session_id: str, user_message: str) -> AsyncIterator[dict]   # {"event": str, "data": Any}
```
Additions (QA fixes): `status` {stage:"waiting"} first if another answer of the same session is running
(per-session lock); `warning` {code:"message_truncated", ...} from /api/chat when a message > 4000 chars was cut
to its tail; the title `session` event now comes AFTER `done` (background task). Routing fast path: see
decisions.router_needed / fast_route.
Event sequence (SSE `event:` names): `session` {session_id, title} → `status` {stage, detail} (repeated:
routing, retrieving, reranking, generating, validating) → `decision` RouteDecision → `token` {t} (many) →
`sources` [Source] → `images` [Image] → `usage` {by_purpose, prompt_tokens, completion_tokens, total} →
`done` {message_id, out_of_context}. On error: `error` {message} then `done`.

Steps: load memory → route (standalone_query) → greeting? short canned reply (no citations) → retrieve
(sub_queries when cross-ref) → expand_cross_refs → OOC gate (thresholds above) → if OOC: emit exactly
`#OutofContext` as the only token, no sources → build prompt within num_ctx budget (system + memory +
numbered sources + question; reserve max_answer_tokens) → stream answer → citations.validate →
figures (needs_images or figure chunks cited) → persist message, update memory (+compaction) → usage.

```python
Source = {"n": int, "chunk_id": str, "doc_id": str, "doc_title": str, "chunk_type": str, "header": str,
          "section_path": list[str], "pages": list[int], "snippet": str, "score": float,
          "highlights": [{"page": int, "bboxes": [[x0,y0,x1,y1], ...]}],   # PDF points, top-left
          "page_size": {"<page>": [w, h]}, "figure": {...} | None}
Image  = {"chunk_id": str, "url": "/api/figure/<chunk_id>", "caption": str, "figure_number": str|None,
          "page": int, "description": str|None}
```

## rag/citations.py
`validate(answer: str, sources: list[Hit]) -> (clean_answer, used_sources)` - keep only `[n]` that exist,
renumber by first appearance (1..m) in text and in the returned list, collapse `[1][2]`→`[1, 2]` style
consistently, drop sources never cited. If an in-scope answer has zero citations → append the best
source as [1] only if the answer text overlaps it; otherwise flag `uncited=True` (UI shows warning).

## Prompting rules (llm/prompts.py)
- Answer ONLY from the numbered sources; every factual sentence ends with ≥1 citation `[n]`.
- If the sources do not contain the answer, or the question is outside the textbook's materials-science
  syllabus → reply with exactly `#OutofContext` and nothing else.
- Never invent numbers, equations, figure numbers, page numbers. Keep LaTeX as `$...$` / `$$...$$`.
- Use the session memory only to resolve references ("it", "that figure", "the previous one"), never as a
  source of facts. Format per decision (table → markdown table with citations in cells).

## api/main.py (FastAPI 0.146, built-in SSE `fastapi.sse.EventSourceResponse`)
```
GET  /                         chat.html          GET /audit   audit.html     /static/*  files
GET  /api/health               {postgres, ollama, chat_model_present, chunks, embedder_loaded}
POST /api/chat                 body {session_id?: str, message: str} → SSE stream (events above)
GET  /api/sessions             [{session_id, title, updated_at, message_count}]
POST /api/sessions             → {session_id, title}
PATCH /api/sessions/{id}       {title}            DELETE /api/sessions/{id}
GET  /api/sessions/{id}/messages   [{message_id, role, content, sources, images, out_of_context, created_at}]
GET  /api/sessions/{id}/memory     {markdown, tokens, budget}
GET  /api/page/{doc_id}/{page}.png?dpi=110   rendered page PNG (cached in cache/pages)
GET  /api/page/{doc_id}/{page}/meta          {width, height}
GET  /api/figure/{chunk_id}                  figure PNG
GET  /api/audit/summary?days=30              totals by purpose/model/day + per session
GET  /api/audit/calls?limit=200&session_id=  recent llm_calls rows
GET  /api/settings / PUT /api/settings       editable runtime settings (models, thresholds, num_ctx...)
GET  /api/keys (values masked) / PUT /api/keys/{name} {value, note} / DELETE /api/keys/{name}
```

## run.py
1. start embedded Postgres (pgserver) and init schema; 2. make sure Ollama answers on 11434 (else start
`ollama serve` from `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`), pull chat model if missing; 3. if
chunks table empty (or `--ingest`) run ingest; 4. run uvicorn (127.0.0.1:8000, next free port if busy);
5. print a links table: Chat UI, Audit UI, API docs, Health, Postgres URI, Ollama URL. Ctrl+C stops everything.
