###  `docs/ARCHITECTURE.md`

```markdown
# Software Architecture & Technical Specification

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

---

## 2. AI Architecture & Model Selection Strategy

### 2.1 Final Model Recommendation
* **MVP Phase Strategy:**
  * **Embedding Model:** Qwen3 Embedding (Strong multilingual & Burmese support).
  * **Vector DB:** ChromaDB (Simple, fast setup for semantic search).
  * **Generation Model:** Gemini API (Production-grade answer quality with minimal operational maintenance).
  * **Orchestration:** FastAPI service with async background indexing via Celery/Redis.
* **Production Scale Strategy:**
  * **Hybrid Retrieval:** Combine vector search (Qwen3) + keyword search (BM25) with a re-ranker model for legal/banking precision.

### 2.2 Trade-off Analysis Summary
| Model Type | Accuracy | Latency | Hosting / Cost | Ease of Integration | Best Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen3 + Local LLM** | High with tuning | Medium | Low–Medium over time | Medium | Controlled enterprise deployment |
| **Proprietary API Stack** | High out-of-the-box | Low | Medium–High recurring | High | Fast MVP and experimentation |
| **Hybrid Domain-Tuned System** | Very High | Medium | Medium–High | Medium | Long-term production system |

### 2.3 ML Pipeline Architecture

**API Entrypoint:** FastAPI
* **Async Workers:** Celery / RQ for CPU/GPU intensive tasks (OCR, Chunking, Embedding generation).
* **Decoupled LLM Layer:** Gemini API wrapped inside an abstract service interface to enable seamless model swapping.

---

## 3. System Architecture & Folder Structure

### 3.1 Layered Modular Design
The project uses a clean, layered modular architecture separating presentation, domain, infrastructure, and persistence logic

### Recommended Project Structure
burmese_bank_rag_service/
├── README.md
├── ARCHITECTURE.md
├── app/
│   ├── api/
│   │   ├── deps.py
│   │   └── v1/
│   │       ├── auth.py
│   │       ├── documents.py      # Trigger document ingestion
│   │       ├── chat.py           # LLM orchestration endpoint
│   │       └── search.py         # Direct knowledge vector search endpoint
│   ├── core/
│   │   ├── config.py             # Chroma, Qwen, & RAG configurations
│   │   ├── logging.py
│   │   ├── security.py
│   │   └── exceptions.py
│   ├── db/
│   │   ├── base.py
│   │   ├── session.py
│   │   └── migrations/
│   ├── models/
│   │   ├── user.py
│   │   ├── document.py
│   │   ├── chunk.py
│   │   ├── conversation.py
│   │   └── message.py
│   ├── schemas/
│   │   ├── auth.py
│   │   ├── document.py
│   │   ├── chat.py
│   │   └── response.py
│   ├── services/
│   │   ├── rag/                          # <--- CORE RAG SUBSYSTEM
│   │   │   ├── __init__.py
│   │   │   ├── parsers.py                # PyMuPDF, PaddleOCR, docx parsers
│   │   │   ├── normalizer.py             # Unicode NFC cleaning
│   │   │   ├── embedding_service.py      # Qwen3 HuggingFace model wrapper
│   │   │   ├── vector_store.py           # ChromaDB client & metadata indexing
│   │   │   ├── ingestion_service.py      # LlamaIndex chunking & indexing
│   │   │   └── retrieval_service.py      # Similarity query engine & thresholding
│   │   ├── auth_service.py
│   │   ├── document_service.py
│   │   ├── llm_service.py
│   │   └── conversation_service.py
│   ├── workers/
│   │   ├── celery_app.py
│   │   ├── ingest_tasks.py               # Async document parsing/indexing
│   │   └── embedding_tasks.py
│   ├── utils/
│   │   ├── text_cleaning.py              # Burmese text cleaning utilities
│   │   ├── burmese_normalizer.py         # TTS number/currency normalizer
│   │   └── validators.py
│   ├── main.py
│   └── startup.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
│   └── ARCHITECTURE.md
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml

### Overview of Project structure
burmese_bank_rag_service/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── documents.py      # Doc Ingestion Endpoint
│   │       └── search.py         # Direct Search Endpoint
│   ├── core/
│   │   └── config.py             # App & Chroma Configuration
│   ├── services/
│   │   └── rag/
│   │       ├── __init__.py
│   │       ├── parsers.py        # PDF, Docx, PaddleOCR
│   │       ├── normalizer.py     # Burmese Text Cleaning
│   │       ├── vector_store.py   # ChromaDB & Qwen3 setup
│   │       ├── ingestion_service.py
│   │       └── retrieval_service.py
├── docs/
│   └── ARCHITECTURE.md
└── requirements.txt


## 4. Database Schema & Data Modeling

The system isolates relational metadata (PostgreSQL), vector indices (ChromaDB), and transient state (Redis).

### 4.1 Core Entities (PostgreSQL)
* **Users:** `id`, `email`, `password_hash`, `role` (admin/user/auditor), `is_active`, timestamps.
* **Documents:** `id`, `title`, `source_type`, `file_path`, `version`, `status`, `uploaded_by`, `checksum_hash`.
* **Document Sections:** `id`, `document_id`, `section_name`, `page_number`, `raw_text`.
* **Chunks:** `id`, `document_id`, `section_id`, `chunk_text`, `chunk_index`, `page_number`, `metadata_json`.
* **Conversations & Messages:** `id`, `user_id`, session titles, sender type (`user`/`assistant`), `retrieved_context_ids`.
* **Retrieval & Audit Logs:** `query_text`, `top_k`, `retrieved_chunk_ids`, `latency_ms`, action details.

---

## 5. API Endpoints Overview (`/api/v1`)

### Authentication & Users
* `POST /api/v1/auth/register` - User registration.
* `POST /api/v1/auth/login` - Authenticate & receive JWT access/refresh tokens.

### Document Management
* `POST /api/v1/documents/upload` - File ingestion (PDF/DOCX/PPTX/Image).
* `POST /api/v1/documents/{id}/ingest` - Trigger OCR, extraction, and vector indexing.
* `GET /api/v1/documents` - List document metadata and approval status.

### Retrieval & Search
* `POST /api/v1/search/retrieve` - Semantic vector retrieval (returns top-k chunks with scores).
* `POST /api/v1/search/hybrid` - Combined vector + keyword search.

### Chat & Conversation
* `POST /api/v1/chat/sessions` - Create chat session.
* `POST /api/v1/chat/sessions/{id}/messages` - Submit prompt, perform retrieval, and generate grounded answer via Gemini.

---

## 6. Non-Functional & Security Requirements

* **Role-Based Access Control (RBAC):** Admin-only controls for document indexing and system metrics.
* **Input Validation & Sanitization:** Strict file-type restrictions, payload validation via Pydantic, and defense against SQL Injection/XSS.
* **Structured Logging & Tracing:** Centralized `structlog` output using JSON format with correlation trace IDs across request pipelines.
* **Error Handling & Fallback:** Custom exception middleware returning standardized error responses. Built-in refusal logic when retrieval evidence score is below threshold.

---

## 7. Phased Implementation Roadmap

* **Phase 1: Foundation & Project Setup** — Folder structure, Docker setup, FastAPI base, PostgreSQL & Redis configs.
* **Phase 2: Authentication & User Management** — User schema, JWT access flow, RBAC authorization handlers.
* **Phase 3: Document Ingestion & Extraction** — Upload pipeline, file validators, PaddleOCR integration, metadata tracking.
* **Phase 4: Text Normalization & Chunking** — Burmese text cleaning, chunking strategies, metadata-enriched chunk DB insertion.
* **Phase 5: Qwen3 Embeddings & ChromaDB** — Vector generation pipeline, ChromaDB collection indexing, top-k retrieval API.
* **Phase 6: Conversation & Gemini Orchestration** — Context prompt builder, Gemini LLM connection, chat memory, refusal guardrails.
* **Phase 7: Evaluation & Quality Safeguards** — Retrieval benchmarking, groundness validation, citation metadata injection.
* **Phase 8: Production Hardening** — Observability (Prometheus/Grafana), rate limiting, query optimization, and CI/CD pipelines.
