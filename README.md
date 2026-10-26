# OmniResearch

A local-first, agentic research platform for technical documents. Upload research papers, ask a complex question, and
OmniResearch plans an investigation, searches your PDFs (text, tables, figure captions, OCR'd scans) and optionally the
live web, compares and cross-checks evidence, and writes a structured report whose citations open the exact PDF page or
web source — with every claim checked mechanically and (optionally) by a verifier model.

**Central rule:** provider implementations may change; capability interfaces and agent logic stay stable. Every AI
capability — chat, embeddings, reranking, OCR, vision, web search, webpage extraction — goes through one provider registry
and router, and each task can be routed to a different provider from the UI.

---

## Contents
- [Quick start (local)](#quick-start-local)
- [Quick start (Docker Compose)](#quick-start-docker-compose)
- [Using notebooks](#using-notebooks)
- [Configuring AI providers](#configuring-ai-providers)
- [Architecture](#architecture)
- [How an investigation works](#how-an-investigation-works)
- [Embedding indexes and safe reindexing](#embedding-indexes-and-safe-reindexing)
- [Security model](#security-model)
- [Testing](#testing)
- [Results from real runs](#results-from-real-runs)
- [Implementation status and known limitations](#implementation-status-and-known-limitations)

---

## Quick start (local)

Requirements: Python **3.11** (Docling/PaddleOCR wheels are not available for 3.13+ on all platforms), Node 20+.

```bash
# Backend
cd backend
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt          # Windows: .venv\Scripts\pip
.venv/bin/pip install -r requirements-ml.txt       # optional but recommended: Docling, PaddleOCR (PP-OCRv5), fastembed

# Frontend (build once; FastAPI serves frontend/dist)
cd ../frontend
npm install
npm run build

# Run
cd ../backend
.venv/bin/uvicorn app.main:app --port 8000          # open http://127.0.0.1:8000
```

For frontend development use `npm run dev` in `frontend/` (http://localhost:5173, proxies `/api` to port 8000).

On Windows set `PYTHONUTF8=1` in the shell that runs the backend.

**No credentials are needed to start.** On first start OmniResearch creates local, credential-free providers:

| Task | Default provider |
|---|---|
| Text embeddings | FastEmbed `BAAI/bge-small-en-v1.5` on CPU (falls back to dependency-free hashing embeddings if fastembed is not installed) |
| Reranking | FastEmbed cross-encoder `ms-marco-MiniLM-L-6-v2` → local lexical reranker as fallback |
| OCR | PaddleOCR PP-OCRv5 mobile models on CPU (if installed) |
| Webpage extraction | Built-in safe fetcher + HTML/PDF extractor |

You can upload and search documents immediately. **Research investigations need one chat model** assigned to the
`synthesis` task (Settings → Connect AI). Other chat tasks (planning, decomposition, rewriting, verification) inherit
the synthesis route until you assign them separately. Web research needs a search provider (Tavily, Brave, Exa,
SearXNG or a custom API) on the `web_search` task.

## Quick start (Docker Compose)

```bash
cp .env.example .env          # set OMNI_ADMIN_TOKEN (required: the API is not on loopback inside a container)
docker compose up -d --build  # app on http://localhost:8000, Qdrant server as a sidecar
docker compose --profile searxng up -d   # optional self-hosted metasearch (edit deploy/searxng/settings.yml first)
```

- `INSTALL_ML=false` in `.env` builds a slim image without Docling/PaddleOCR/fastembed (PyMuPDF parser, API-based models).
- Model weights are cached in the `omni-models` volume; data (SQLite, uploads, caches, checkpoints) in `omni-data`.
- In the UI, enter the admin token under Settings to manage providers.
- Declarative provider bootstrap: copy `providers.example.json` to `providers.json`, uncomment its mount in
  `docker-compose.yml`, and set `OMNI_PROVIDERS_FILE=/config/providers.json`. Secrets are referenced by env-var name.

All settings are environment variables prefixed `OMNI_`; see [.env.example](.env.example).

## Using notebooks

The app opens on **Notebooks** (NotebookLM-style). Each notebook is a set of PDFs with its own chat, notes and reports;
the top menu is just *Notebooks* and *Settings*.

- **Sources** (left): add sources three ways — upload PDFs, **paste links**, or **search the web**. A PDF link is
  downloaded and processed like an upload; a webpage's main text is captured into a PDF snapshot (title, address and
  capture time on page 1), so citations keep working if the page changes. Pages that need a login or JavaScript are
  reported, never saved empty. Web search has two modes: *Fast* (one search, pick results) and *Deep* (the AI plans
  several searches and recommends the best sources with a reason each); nothing is added until you pick it. Watch
  parsing/OCR progress, retry failures, open a source to read its pages. The checkboxes decide which sources are used.
- **Chat** (middle): an empty notebook shows an AI summary of the sources and three suggested questions. Answers come
  only from the checked sources, with numbered citation chips — hover for the passage, click to open the cited page with
  the region highlighted. Unknown citations and numbers that don't appear in the cited passages are flagged. Answers can
  be saved as notes or copied with their source list. The chat box has a **⚡ Fast / 🔬 Deep research** switch and a
  **🌐 Web** button: Fast gives a quick cited answer (with Web, a quick web search is added; web citations open the
  page); Deep research runs the full multi-step investigation and shows its progress as a card in the chat.
- **Studio** (right): finished deep-research reports (with mind map and follow-up chat). One-click outputs — **Summary, Briefing doc,
  Study guide, FAQ, Timeline, Mind map** — are written from passages sampled across every selected source; each shows a
  trust bar (passages used, statements without a citation, numbers not found in the cited passages, unknown citations).
  *Notes* holds saved answers and your own notes.
- Under 1000 px wide the three panels become tabs. Old `/research` and `/documents` links redirect into notebooks;
  document collections from earlier versions become notebooks automatically.

## Configuring AI providers

**Settings → Connect AI** is the whole setup: pick a service (NVIDIA NIM, OpenRouter, OpenAI, Anthropic, Groq,
Together, DeepSeek, Mistral, ModelScope, Ollama, or a **custom OpenAI-compatible endpoint**), paste the API key, and
click *Find models*. The app lists the service's models immediately and streams a live test of the best candidates
(many catalogs list models an account cannot use; the test also checks that a model returns valid structured
output). Recommended models are pre-selected for each job — writing reports, planning, reading figures, **ranking
search results** (scored by the connected chat model; thinking is switched off for this, ~1–2 s) and **reading scanned
pages** (the connected vision model) — and *Connect* sets everything up. Picking a cloud embedder rebuilds the search
index automatically. Web search services (Tavily, Brave, Exa) need only a key; web research turns on once one is
connected.

**Settings → Local models** installs free CPU models from the website: PaddleOCR PP-OCRv5 language packs for scanned
pages, fastembed embedders and cross-encoder rankers — from a built-in list or by **searching Hugging Face** (ONNX
models are configured automatically). Install shows download progress; *Use* switches the job to that model.

Under the hood the same registry and adapters below are used; Connect only fills in their configuration.

### Integration modes

| Mode | Adapters | Capabilities |
|---|---|---|
| **A. OpenAI-compatible** | `openai_compatible` — NVIDIA NIM, OpenRouter, ModelScope API-Inference, vLLM, SGLang, Ollama (`/v1`), LM Studio, OpenAI | chat, embeddings, vision, image embeddings (experimental) |
| **B. Native** | `anthropic` (Messages API), `rerank_api` (dialects: Cohere, Jina, Voyage, NVIDIA NIM ranking, HF TEI), `embedding_api` (TEI, Cohere v2, Ollama native), `paddleocr_serving`, `tavily`, `brave`, `exa`, `searxng` | per adapter |
| **C. Custom HTTP** | `custom_http` — request templates + response mappings, validated on save; nothing is executed | chat, embeddings, reranking, web search |
| **Local** | `fastembed`, `fastembed_rerank`, `local_hashing`, `local_lexical_rerank`, `paddleocr_local`, `builtin_web_extractor` | per adapter |

Optional OpenAI features are explicit flags, not assumptions: `supports_json_mode`, `supports_json_schema`,
`supports_system_role`, `embedding_dimensions`, `embedding_input_type_field` (e.g. `input_type` for NIM retrieval
embedders). A provider only advertises a capability when the model for it is configured (a chat-only endpoint is never
offered as an embedder or vision model).

Examples (all configurable in the UI, no code changes):

```jsonc
// NVIDIA NIM chat
{"adapter": "openai_compatible", "config": {"base_url": "https://integrate.api.nvidia.com/v1", "chat_model": "meta/llama-3.3-70b-instruct", "supports_json_mode": true}}
// NIM retrieval embeddings (asymmetric query/passage)
{"adapter": "openai_compatible", "config": {"base_url": "https://integrate.api.nvidia.com/v1", "embedding_model": "nvidia/nv-embedqa-e5-v5", "embedding_input_type_field": "input_type"}}
// Local vLLM / Ollama (add the host to OMNI_OUTBOUND_ALLOW_PRIVATE_HOSTS if it is not localhost)
{"adapter": "openai_compatible", "config": {"base_url": "http://localhost:11434/v1", "chat_model": "qwen2.5:14b"}}
// Hugging Face TEI reranker
{"adapter": "rerank_api", "config": {"dialect": "tei", "base_url": "http://localhost:8080"}}
// Custom in-house API (Mode C)
{"adapter": "custom_http", "config": {
  "base_url": "https://llm.internal.example/api", "auth": {"type": "header", "header_name": "X-Token"},
  "chat": {"path": "generate", "model": "house-7b",
           "body_template": {"model": "{{model}}", "prompt": "{{prompt}}", "system_prompt": "{{system}}"},
           "response_mapping": {"text": "output.text", "input_tokens": "meta.tokens.in"}}}}
```

Template variables per capability: chat `messages prompt system model temperature max_tokens`; embedding
`texts text input_type model`; reranking `query documents top_n model`; web search `query max_results recency_days model`.
Response paths support `a.b[0].c` and `items[*].field`. Paths are static, credentials are injected only through `auth`.

### Routing, fallback and resilience

Each task (`planning`, `decomposition`, `rewriting`, `synthesis`, `verification`, `text_embedding`, `image_embedding`,
`reranking`, `ocr`, `vision`, `web_search`, `web_extraction`) has an ordered provider list, a fallback toggle, and
optional `model`/`temperature`/`max_tokens` overrides. The router applies per-provider timeouts, bounded retries with
exponential backoff + full jitter (honouring `Retry-After`), concurrency limits, circuit breakers, and normalized errors
(`auth`, `rate_limit`, `timeout`, `unavailable`, `bad_request`, `invalid_response`, `blocked`, …). Fallback only goes to
providers that support the task's capability, and for embeddings only to providers producing the *identical* vector
space (see below). Every call is recorded with latency, outcome, tokens, cost (when pricing is configured) and whether it
was a fallback; the Health tab shows p50/p95 latency, failure rates, circuit state and recent failures.

## Architecture

```
backend/app
├── providers/          capability interfaces (base.py), registry, router, resilience, adapters per capability
│   ├── chat/ embeddings/ reranking/ ocr/ web_search/ webpage_extraction/ custom/
├── agents/             LangGraph state, nodes, tools, graph — depend only on the router + services
├── services/
│   ├── parsing/        Docling parser (primary), PyMuPDF fallback
│   ├── ingestion/      validation, selective OCR, caching, jobs
│   ├── chunking/       structure-aware chunker
│   ├── retrieval/      Qdrant store, index manager (compatibility + migrations), BM25, hybrid RRF + rerank
│   ├── web/            web research tools
│   ├── citations/      mechanical + semantic validation
│   ├── research/       evidence model, report schema/rendering, job service
├── repositories/       SQLite with versioned migrations
├── observability/      JSON logs with secret redaction, call metrics
├── core/               settings container (DI), SSRF guard, safe fetcher, secret encryption
└── api/                FastAPI routes
frontend/               React + TypeScript (Vite)
backend/tests/data/     synthetic test PDFs and their generator (make_corpus.py)
```

Grep check for the architectural rule: no module under `agents/`, `services/` or `api/` imports a vendor SDK or names a
vendor endpoint; vendor wire formats exist only inside `providers/*` adapters.

### Ingestion

1. Validate (PDF magic, readable, not encrypted, size/page limits), hash (SHA-256 → idempotent re-uploads), store atomically.
2. Parse with **Docling** (layout + TableFormer table structure; Docling's own OCR is disabled so OCR stays provider-independent).
   If Docling is not installed or fails, PyMuPDF is used and the document is labelled `parser=pymupdf`.
3. Detect pages needing OCR (no/garbled text layer); render only those pages; send them to the OCR route
   (PaddleOCR PP-OCRv5 on CPU by default); cache by page-image hash + OCR engine; deduplicate against parsed text.
4. Chunk by structure: section paths, tables as their own chunks (header row repeated when split), figure captions with
   page + bounding box, OCR text tagged `ocr:<engine>`.
5. Index into the active vector index; BM25 is rebuilt lazily. Every chunk keeps document id, filename, page(s),
   section path, element type, extraction method, chunk id, bbox and source element id.

## How an investigation works

LangGraph `StateGraph` (typed, checkpointed to SQLite so interrupted jobs resume after a restart):

```
analyze_question → plan_research → decompose → route_sources ─┬→ retrieve_local ─┐
                                                              └→ web_research ───┴→ rerank_evidence → inspect_visuals
→ assess_sufficiency ─(gaps, ≤ OMNI_AGENT_MAX_ITERATIONS)→ follow_up (query rewriting) → parallel retrieval again
                     └→ compare_sources (contradictions) → draft_report → validate_citations → finalize
```

- **Tools** (fixed allowlist, no shell): `search_documents` (hybrid, with table/figure focus), `web_search`,
  `fetch_webpage`, `extract_webpage_content`, `get_web_source_metadata`, `compare_web_sources`,
  `validate_web_citations`, page/figure rendering + vision inspection.
- **Web**: used when the plan says the question needs current/external information, when there are no local documents,
  when the sufficiency check finds gaps that need the web, or when requested. Only URLs returned by the search provider
  are fetched; page text is never mined for links or instructions. Snippet-only evidence is flagged.
- **Figures**: if a vision provider is configured, figure regions are rendered and described; observations are stored
  separately from source text and labelled as model interpretations. Without one, the report says figures were located
  by caption but not visually inspected.
- **Report**: executive summary, scope, findings, comparison tables (with experimental conditions), contradictions,
  limitations, evidence gaps, conclusions, follow-ups, and sources split into uploaded documents vs web — each claim is
  either `source` (must cite evidence) or `interpretation`.
- **Citation validation**, kept separate:
  - *Mechanical* (deterministic): cited evidence exists; document/page or fetched URL exists; the excerpt occurs
    verbatim on the cited page(s)/page; every number in the claim occurs in the cited evidence.
  - *Semantic* (verification route): supported / partially / not supported / cannot judge — reported as a model
    judgment, not proof. Unverified claims get one bounded rewrite attempt, then are marked `⚠ unverified` in the report.
- Cancellation, bounded loops (iterations, sub-queries, evidence cap; the follow-up loop also stops as soon as a round
  finds no new evidence), per-job token/cost accounting persisted with the call log, and a live event timeline.
- **Reading the report**: clicking a local citation opens a side panel with the PDF page and the cited region highlighted;
  web citations open the source. A **mind map** tab shows question → searches, findings → claims (coloured by
  verification status) → cited sources, plus contradictions and open questions; it is built from the report without
  extra model calls. **Follow-up chat** answers questions about a finished report from its evidence plus a fresh
  document search, with the same citation checks (unknown ids and numbers missing from the cited evidence are flagged).

## Embedding indexes and safe reindexing

- Each Qdrant collection records the full embedding fingerprint: provider identity and endpoint, model, dimensions,
  normalization, distance, chunking version and preprocessing. Vectors from different spaces are never mixed.
- The active index is always written and queried **with its pinned provider**. Changing the text-embedding route shows
  "reindex recommended" but does not change what the active index is fed.
- Changing the pinned provider's endpoint/model (even with the same model name) makes the index *incompatible*: dense
  retrieval is skipped (BM25 still works) until you **reindex** or explicitly **confirm compatibility** (recorded in the
  index history). `compatibility_key` lets you declare replicas of the same model as interchangeable for fallback.
- Reindex: new collection → bounded batches with retries → progress/errors recorded → validation (counts + self-retrieval
  check) → switch. The previous index is kept for **rollback** until you **confirm** the migration. Interrupted
  reindex jobs resume on startup; new uploads during a migration are written to both indexes. Missing vectors can be
  backfilled. Chat or reranker changes never trigger re-embedding.

## Security model

- **Credentials**: Fernet-encrypted at rest (`OMNI_SECRET_KEY` or a generated key file), never returned by the API,
  registered with the log redactor; logs also redact bearer tokens and common key formats. httpx request logging is muted.
- **Admin endpoints** (providers, routes, settings, reindexing, local models): require `OMNI_ADMIN_TOKEN`
  (constant-time comparison), or loopback-only access when no token is set.
- **SSRF / outbound control**: http(s) only, no credentials in URLs; provider endpoints may reach private addresses only
  for hosts listed in `OMNI_OUTBOUND_ALLOW_PRIVATE_HOSTS`; optional provider host allowlist and blocklist. Provider calls
  never follow redirects (credentials can't be forwarded). Agent web fetches re-validate every redirect hop, re-check
  the connected peer IP after connect (DNS-rebinding mitigation), cap size and time, send no cookies, honour robots.txt
  and never retry 401/402/403/451 responses.
- **Custom adapters**: JSON templates with whitelisted variables, static paths, validated header names; no code execution.
- **Prompt injection**: retrieved content is wrapped in `<source>` fences with an explicit untrusted-data instruction;
  fence tags inside documents/webpages and metadata are neutralized; models only emit structured queries — they cannot
  call tools, choose URLs, or change validation outcomes.
- **Uploads**: PDF validation, size/page limits, atomic temp-file writes, server-generated file names.
- **Privacy**: the deep-research panel lists which configured providers will receive document content before you start; local
  parsing/OCR/embeddings/reranking need no network.

## Testing

```bash
cd backend
PYTHONUTF8=1 .venv/bin/python -m pytest -q                    # unit + integration (mocked providers)
OMNI_ML_TESTS=1 .venv/bin/python -m pytest -q -k real_docling  # real Docling + PaddleOCR on the synthetic corpus
OMNI_LIVE_TESTS=1 OMNI_LIVE_CHAT_BASE_URL=... OMNI_LIVE_CHAT_MODEL=... OMNI_LIVE_CHAT_KEY=... pytest -m live

cd ../frontend && npm test
```

Automated tests use deterministic mock adapters (`backend/tests/fakes.py`) and HTTP mocks. They cover: adapter
normalization across NIM/OpenRouter-style, Anthropic, five rerank dialects, TEI/Cohere/Ollama embeddings and four search
engines; the custom adapter (and rejection of unsafe templates); routing per task, inheritance, retries, fallback,
circuit breaking, capability-safe substitution; embedding fallback restricted to identical spaces; credential encryption
and redaction; SSRF blocking; ingestion provenance, selective OCR, OCR caching and OCR fallback; hybrid retrieval and
reranking; document deletion; embedding migration (switch, rollback, confirm, failed reindex not switching, endpoint
change requiring confirmation); full LangGraph investigations (fabricated claims and invented table numbers flagged,
contradictions, web evidence, prompt-injection resistance, provider swap without code changes, LLM fallback,
cancellation, insufficient evidence); and the HTTP API including auth.

## Results from real runs

The automated tests use a small **synthetic** corpus generated by `backend/tests/data/make_corpus.py` (every document
says on page 1 that its values are invented for testing): a quantization paper with a results table, a KV-cache paper
with a chart, and a report whose second page is a scanned image and which contradicts the second paper (8x vs 23x).

### Investigations with NVIDIA NIM (24 Oct 2026)

The comparison question from the spec was run twice over the synthetic corpus with NIM-hosted models (synthesis and
verification: `nvidia/nemotron-3-ultra-550b-a55b`; planning, decomposition, rewriting and vision:
`nvidia/nemotron-3.5-super-vl-120b-a12b`; fallback `minimaxai/minimax-m3.1-flash`), with local embeddings, reranking and OCR:

| | First run | After fixes |
|---|---|---|
| Stages that fell back to defaults | 6 | 0 |
| Cross-source comparison | failed | 7 dimensions, 1 contradiction (23x vs 8x) found |
| Claims verified / partially / unverified (+ interpretation) | 9 / 7 / 6 (+6) | 13 / 11 / 0 (+7) |
| Semantic verification | unavailable | ran on all source claims |

The first run exposed real-provider issues that mocks did not, all now fixed and covered by regression tests:
reasoning models spend hidden tokens against `max_tokens` (structured calls now get ≥4096 tokens and retry with a larger
budget on truncation); web-tagged sub-queries were dropped when web search was off; follow-up rounds continued after
finding nothing new; per-job usage was lost when the server restarted. During the second run the dev server restarted
four times and the investigation resumed from its checkpoint each time. Practical notes: NIM's model list includes many
models not deployed for a given account (they return 404 — use "Test connection"), and NIM retrieval embedders require
`embedding_input_type_field: "input_type"`. A full investigation took ~9 minutes of model time.

## Implementation status and known limitations

Implemented and tested: everything described above, with the caveats below.

Not yet implemented / limitations:
- **PDF export** of reports (Markdown export only).
- **Visual page retrieval (ColPali-style)**: the `ImageEmbeddingProvider` interface, `image_embedding` route and an
  experimental OpenAI-compatible image-embedding mode exist, but no page-image index is built yet. Figures are found via
  captions and inspected with a vision model when one is configured.
- **Live third-party APIs**: NVIDIA NIM chat, vision and embeddings were exercised for real. OpenRouter, ModelScope,
  Anthropic, Tavily, Brave, Exa, Cohere, Jina and Voyage were tested only against mocked responses that follow their
  documented formats. Use `pytest -m live` with your credentials to verify. Live web research has not been run.
- **Audio overviews** are not implemented.
- The LLM reranker makes one model call per search; a local cross-encoder (Local models) is cheaper if latency or cost matters.
- OCR'd text has no bounding boxes, so the citation panel shows the page without a highlighted region for OCR evidence.
- A follow-up chat request cannot be cancelled server-side once sent; the model call still completes and is billed.
- `npm audit` reports advisories in dev tooling (vitest) and react-router 6; the fix requires react-router 7.
- **Docker image** has not been built on the development machine (Docker daemon not running there); `docker compose
  config` validates. The local (non-Docker) setup is the tested path.
- PaddleOCR is slow on CPU without oneDNN (`enable_mkldnn` is off by default because paddlepaddle 3.3's CPU build
  crashes with it on; enable it if your build works). Docling's first run downloads layout models (~1–2 min).
- PaddleOCR returns text lines, not table structure, for scanned tables.
- The built-in extractor does not run JavaScript. DNS rebinding is mitigated by a post-connect peer check, but the
  request is sent before that check (GET, no credentials).
- BM25 is an in-process index rebuilt when chunks change — fine for thousands of chunks, not millions.
- Embedded Qdrant mode is single-process; use the Qdrant server (Compose) for multi-worker deployments.
- Semantic verification quality depends entirely on the configured verifier model; it is reported as a model judgment.
- The test corpus is synthetic and small; results on it show the software works, not research quality on real papers.
