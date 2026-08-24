# Software Architecture & Technical Specification

> **Burmese Banking RAG Service** — Grounded Knowledge Retrieval Layer
> This document describes the **current implemented architecture** of the RAG service.

---

## 1. System Overview & Core Features

### 1.1 High-Level Goal
The application provides a grounded, bilingual (Burmese-first) banking support assistant that answers user questions using approved banking documents rather than relying on generic model knowledge.

The system must:
1. Ingest approved internal documents and FAQs.
2. Extract and normalize Burmese text.
3. Build a searchable knowledge base.
4. Retrieve the most relevant evidence for each question.
5. Answer using only retrieved and approved sources.
6. Avoid hallucination when evidence is insufficient.

### 1.2 Scope of This Repository
This repository is the **RAG & Knowledge Retrieval Layer** only. It does **not** include:
- Frontend / TTS / Voice
- Gemini/LLM answer generation (consumes this service's JSON contract)
- User authentication / RBAC (handled by the external gateway)
- PostgreSQL metadata store (optional, future)

---

## 2. AI Architecture & Model Selection Strategy

### 2.1 Implemented Model Stack
| Layer | Technology | Status |
| :--- | :--- | :--- |
| **Embedding Model** | Qwen3 Embedding (GGUF, CPU via `llama-cpp-python`) | ✅ Implemented |
| **Vector DB** | ChromaDB (persistent, cosine similarity) | ✅ Implemented |
| **Retrieval** | Hybrid: BM25 (Burmese n-grams) + vector search + cosine reranker | ✅ Implemented |
| **Generation Model** | Gemini API (external, consumes this service's JSON) | 🔜 External / future |
| **Orchestration** | FastAPI service (synchronous) | ✅ Implemented |
| **Async Workers** | Celery / Redis | 🔜 Future |

### 2.2 ML Pipeline Architecture

**API Entrypoint:** FastAPI (`app/main.py`)
- **Retrieval:** HybridRetriever (BM25 + vector + SemanticReranker)
- **Context Contract:** LLMContextBuilder produces the JSON consumed by the LLM layer
- **Decoupled LLM Layer:** Gemini API (external) consumes the retrieval JSON contract

---

## 3. System Architecture & Folder Structure

### 3.1 Current Implemented Structure

```
burmese_bank_rag_service/
├── core/
│   └── config.py              # RAGSettings + service config (RAG_API_TOKEN, SERVICE_NAME, SERVICE_VERSION)
├── app/
│   ├── main.py                # FastAPI entrypoint (lifespan, router registration)
│   ├── api/
│   │   ├── deps.py            # X-RAG-Service-Token auth dependency
│   │   ├── schemas.py         # Pydantic request/response contract models
│   │   └── v1/
│   │       └── routes.py      # /health, /retrieve, /ingest, /ingest-file
│   └── services/
│       └── rag/
│           ├── normalizer.py          # Burmese text normalization (Unicode NFC)
│           ├── ingestion_parser.py    # Markdown parsing + PII masking
│           ├── chunker.py             # Header-aware, hierarchy-preserving chunking
│           ├── embedder_service.py    # Qwen3 GGUF embeddings (CPU)
│           ├── vector_store.py        # ChromaDB persistent store
│           ├── hybrid_retriever.py    # BM25 + vector + reranker
│           ├── context_builder.py     # LLM JSON contract builder
│           ├── factory.py             # Singleton service factory
│           ├── exceptions.py          # Custom RAG exceptions
│           ├── ingestion_service.py   # Programmatic ingestion (API)
│           ├── retrieval_service.py   # Programmatic retrieval (API)
│           └── main.py                # CLI pipeline
├── knowledge/*.md             # Approved banking knowledge base
├── scripts/                   # ingest_all, ingest_single, search, delete_doc, eval scripts
├── tests/                     # unit + integration (incl. test_api.py)
├── evaluation/                # eval_dataset.jsonl, base_chunks.json
├── RAG_API_GUIDELINES.md      # Team integration guide
├── requirements.txt
└── .env
```

### 3.2 Data Flow

```
[User Query]
    │
    ▼
[FastAPI: POST /api/v1/retrieve]
    │
    ▼
[HybridRetriever]
    ├── BM25 (Burmese n-grams) ──┐
    ├── Vector search (ChromaDB) ─┤── RRF Fusion
    └── SemanticReranker ────────┘
    │
    ▼
[LLMContextBuilder]
    │  → JSON contract (contexts, citations, instructions)
    ▼
[Downstream LLM / Generation Layer (external)]
```

---

## 4. Data Storage

### 4.1 Vector Store (ChromaDB)
- **Collection:** `burmese_banking_knowledge`
- **Persistence:** `data/chroma_db` (local, persistent)
- **Metric:** cosine similarity
- **Chunk metadata:** `doc_id`, `doc_name`, `section_title`, `page_number`, `chunk_index`, `header_path`, `parent_section`, `pii_masked`, etc.
- **Text prefix:** chunks stored with `passage:` prefix for retrieval clarity

### 4.2 Relational Metadata (PostgreSQL)
- **Not yet implemented** — planned for future phases (documents, chunks, conversations, audit logs).

---

## 5. API Endpoints Overview (`/api/v1`)

### Implemented Endpoints

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service status, component readiness, document count |
| `POST` | `/api/v1/retrieve` | **Main contract** — returns retrieval JSON for the LLM team |
| `POST` | `/api/v1/ingest` | Ingest markdown content (add/update knowledge) |
| `POST` | `/api/v1/ingest-file` | Ingest a markdown file from the server filesystem (admin/internal) |
| `DELETE` | `/api/v1/documents/{doc_name}` | Delete all chunks for a document by filename |

### Authentication
- **Header:** `X-RAG-Service-Token`
- **Config:** `RAG_API_TOKEN` in `.env`
- **Behavior:** empty token = auth disabled (dev); set token = 401/403 enforcement

### The JSON Contract (for the LLM Team)

`POST /api/v1/retrieve` returns the exact payload consumed by the LLM/generation layer. The full field-by-field reference and examples are documented in **`RAG_API_GUIDELINES.md`**.

**Key fields:**
- `query` — original user query
- `language` — `"my"` or `"en"`
- `has_context` — whether relevant evidence was found
- `confidence` — `"high"` or `"low"`
- `contexts[]` — ranked chunks (`rank`, `chunk_id`, `question`, `text`, `source`, `retrieval_score`)
- `citations[]` — source attribution (`chunk_id`, `source`, `section`)
- `answer` — refusal message when `has_context=false`
- `instructions` — flags the LLM must honor

**LLM integration rules:**
- `has_context: true` → answer ONLY from `contexts[].text`
- `has_context: false` → return `answer` verbatim, never improvise
- Honor all `instructions` flags

---

## 6. Non-Functional & Security Requirements

### Implemented
- **Internal service token auth** (`X-RAG-Service-Token`) — replaces planned JWT/RBAC for this service
- **Input validation** — Pydantic models (query length, top_k bounds, filename rules)
- **PII masking** — NRC, card numbers, account numbers, phone numbers redacted before embedding
- **Refusal logic** — built-in when retrieval evidence is insufficient (`has_context=false`)
- **Idempotent ingestion** — delete-then-add on re-ingest

### Planned / Future
- **RBAC** — handled by external gateway
- **Structured logging (structlog)** — currently standard logging
- **Observability (Prometheus/Grafana)** — future
- **Async workers (Celery/Redis)** — future

---

## 7. Implementation Status & Roadmap

### Completed
- ✅ **Phase 4: Text Normalization & Chunking** — Burmese cleaning, header-aware chunking, metadata enrichment
- ✅ **Phase 5: Qwen3 Embeddings & ChromaDB** — vector generation, ChromaDB indexing, top-k retrieval
- ✅ **Phase 6 (partial): Retrieval API** — FastAPI layer, `/retrieve` contract, `/ingest`, `/health`
- ✅ **Phase 7 (partial): Evaluation** — `evaluate_retrieval.py`, `eval_dataset.jsonl`

### Pending / Future
- 🔜 **Phase 3: PDF/OCR ingestion** — PyMuPDF, PaddleOCR (currently Markdown-only)
- 🔜 **Phase 6 (rest): Gemini orchestration** — LLM layer consumes the JSON contract (external)
- 🔜 **Phase 7 (rest): Groundness validation, citation injection** — partially via `context_builder`
- 🔜 **Phase 8: Production Hardening** — observability, rate limiting, CI/CD
- ✅ **Docker Deployment** — multi-stage `Dockerfile` with pre-downloaded Qwen3 GGUF model, `docker-compose.yml` with persistent volumes (chroma_data, knowledge, logs), non-root user, health checks

---

## 8. Testing

- **Unit tests:** `tests/unit/` — parser, chunker, hybrid retriever
- **Integration tests:** `tests/integration/` — vector store, search, embedder, API (`test_api.py`)
- **Run:** `python -m pytest -q` → 31 passed, 1 skipped
- **API tests cover:** contract shape, refusal path, token auth, input validation