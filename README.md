# AI-Powered Burmese Voice Support Copilot — RAG Service

> **Sub-Repository Focus:** Core RAG Engine & Knowledge Retrieval Layer

---

## 📌 Project Overview

This repository contains the **RAG (Retrieval-Augmented Generation) Engine and Knowledge Retrieval Layer** for the **AI-Powered Burmese Voice Support Copilot for Banking** internship project.

The primary goal of the overall project is to deliver a secure, fast, and consistent customer support solution that responds to banking queries in both **natural Burmese text and friendly Burmese voice**.

### 🎯 Purpose of This Repository

This service is **NOT a general-purpose conversational chatbot**. It is the **Grounded Knowledge Service**, responsible for:

1. **Document Ingestion & Parsing** — processing official bank policies, product guides, loan procedures, and FAQs (Markdown currently; PDF/OCR planned).
2. **Semantic Representation** — converting Burmese and mixed Burmese-English text chunks into rich vector representations using the **Qwen3 Embedding** model.
3. **High-Speed Vector Retrieval** — storing and querying vector embeddings within **ChromaDB** using semantic similarity.
4. **Context Provision for LLM** — providing exact, relevant, and source-attributed context chunks to the LLM (Gemini) layer, ensuring all responses are **100% grounded in approved bank knowledge**.

---

## 🏗 System Architecture

```text
[User Text/Voice Question]
          │
          ▼
[FastAPI Gateway & Security Layer]  (external)
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 📍 THIS REPOSITORY: RAG & Knowledge Retrieval Layer    │
│                                                        │
│  POST /api/v1/retrieve  →  JSON context contract       │
│  POST /api/v1/ingest    →  knowledge base updates      │
│  GET  /api/v1/health    →  service status              │
│                                                        │
│  Pipeline: Parse → Normalize → Chunk → Embed → Store   │
│            → HybridRetrieve → ContextBuilder           │
└─────────────────────────┬──────────────────────────────┘
                          │  [Relevant Knowledge Chunks + Metadata]
                          ▼
[Grounded LLM Generation Layer (Gemini)]  (external)
                          │
                          ▼
[Burmese Text Normalization & TTS]  (external)
```

---

## 🛠 Tech Stack

| Layer | Technology |
| :--- | :--- |
| **API Framework** | FastAPI (Python 3.11+) |
| **Vector Storage** | ChromaDB (persistent, cosine similarity) |
| **Embedding Model** | Qwen3 Embedding (GGUF, CPU via `llama-cpp-python`) |
| **Retrieval** | Hybrid: BM25 (Burmese n-grams) + vector search + cosine reranker |
| **Generation Model** | Google Gemini (external, consumes this service's JSON) |
| **Document Parsing** | Markdown (PDF/OCR via PyMuPDF/PaddleOCR planned) |
| **Testing** | pytest |

---

## 🚀 Key Features

- **Multilingual Document Ingestion:** Markdown knowledge base with Burmese text support.
- **Text Preprocessing & Chunking:** Burmese text normalization, header-aware chunking, metadata enrichment, PII masking.
- **Hybrid & Semantic Retrieval:** Qwen3 vector search + BM25 keyword search + cosine reranker for high-precision context retrieval.
- **Grounded JSON Contract:** `POST /api/v1/retrieve` returns the exact context payload consumed by the LLM layer, with built-in refusal when evidence is insufficient.
- **Internal Service Auth:** `X-RAG-Service-Token` header for secure service-to-service calls.

---

## 🏁 Quickstart & Setup (Development)

### Prerequisites
- Python 3.11+
- Qwen3 GGUF model (auto-downloads on first run via HuggingFace)

### Environment Setup
Create a `.env` file in the root directory:
```env
CHROMA_PERSIST_DIR=./data/chroma_db
COLLECTION_NAME=burmese_banking_knowledge
EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=3
SIMILARITY_THRESHOLD=0.3
RAG_API_TOKEN=your_shared_secret_here   # empty = auth disabled (dev)
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Ingest the Knowledge Base
```bash
python scripts/ingest_all.py
```

### Run the API Service
```bash
uvicorn app.main:app --reload
```
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/api/v1/health`

### 🐳 Run with Docker (Recommended)

The service is containerized with the Qwen3 model **pre-downloaded** into the image, so it starts instantly.

```bash
# Build and start the service
docker compose up --build

# Or run in the background
docker compose up --build -d
```

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/v1/health`

**What's included in the Docker image:**
- Python 3.11 + all dependencies from `requirements.txt`
- Qwen3 GGUF embedding model (pre-downloaded at build time)
- ChromaDB (embedded, persistent via Docker volume)
- The full RAG pipeline (parser, chunker, retriever, context builder)

**Volumes:**
- `chroma_data` — persists the vector store across container restarts
- `./knowledge` — mounted so ingested documents persist and can be edited
- `./logs` — mounted for log access

**Stop the service:**
```bash
docker compose down
```

---

## 🔌 API Endpoints

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service status, component readiness, document count |
| `POST` | `/api/v1/retrieve` | **Main contract** — returns retrieval JSON for the LLM team |
| `POST` | `/api/v1/ingest` | Ingest markdown content (add/update knowledge) |
| `POST` | `/api/v1/ingest-file` | Ingest a markdown file from the server filesystem (admin/internal) |
| `DELETE` | `/api/v1/documents/{doc_name}` | Delete all chunks for a document by filename |

**Authentication:** All endpoints require the `X-RAG-Service-Token` header when `RAG_API_TOKEN` is set.

> 📖 **Full API documentation for team members:** See [`RAG_API_GUIDELINES.md`](./RAG_API_GUIDELINES.md)

---

## 📁 Project Structure

```
burmese_bank_rag_service/
├── core/
│   └── config.py              # RAGSettings + service config
├── app/
│   ├── main.py                # FastAPI entrypoint
│   ├── api/
│   │   ├── deps.py            # X-RAG-Service-Token auth
│   │   ├── schemas.py         # Pydantic contract models
│   │   └── v1/routes.py       # /health, /retrieve, /ingest, /ingest-file
│   └── services/rag/          # Core RAG pipeline
├── knowledge/*.md             # Approved banking knowledge base
├── scripts/                   # ingest_all, ingest_single, search, delete_doc, eval
├── tests/                     # unit + integration tests
├── evaluation/                # eval_dataset.jsonl, base_chunks.json
├── RAG_API_GUIDELINES.md      # Team integration guide
├── ARCHITECTURE.md            # Architecture spec
└── requirements.txt
```

---

## 🧪 Testing

```bash
python -m pytest -q
```
Current result: **31 passed, 1 skipped**.

- **Unit tests:** `tests/unit/` — parser, chunker, hybrid retriever
- **Integration tests:** `tests/integration/` — vector store, search, embedder, API (`test_api.py`)

---

## 📚 Documentation

- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — Software architecture & technical specification
- [`RAG_API_GUIDELINES.md`](./RAG_API_GUIDELINES.md) — API integration guide for other teams
- [`ROADMAP.md`](./ROADMAP.md) — Development roadmap & standards

---

## 🔒 Security Notes

- **Internal service only** — do not expose directly to end users; route through your gateway.
- **Token required** in production (`RAG_API_TOKEN` in `.env`).
- **PII masking** is applied automatically in the pipeline (NRC, card numbers, account numbers, phone numbers redacted before embedding).
- **Markdown only** for ingestion in the current version (PDF/OCR is a future phase).