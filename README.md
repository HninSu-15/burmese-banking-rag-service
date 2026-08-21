# AI-Powered Burmese Voice Support Copilot for Banking
> **Sub-Repository Focus:** Core RAG Engine & Knowledge Retrieval Layer

---

## 📌 Project Overview

This repository contains the **RAG (Retrieval-Augmented Generation) Engine and Knowledge Retrieval Layer** for the **AI-Powered Burmese Voice Support Copilot for Banking** internship project. 

The primary goal of the overall project is to deliver a secure, fast, and consistent customer support solution that responds to banking queries in both **natural Burmese text and friendly Burmese voice**. 

### 🎯 Purpose of This Repository

This specific service is **NOT a general-purpose conversational chatbot**. In enterprise banking applications, generic chatbots are prone to hallucinations, tone errors, and unreliable information. To prevent this, our core architecture relies on a strict **Retrieval-Augmented Generation (RAG)** approach[cite: 1].

This repository serves as the **Grounded Knowledge Service**, responsible for:
1. **Document Ingestion & Parsing:** Processing official bank policies, product guides, loan procedures, and FAQs (including OCR for scanned documents).
2. **Semantic Representation:** Converting Burmese and mixed Burmese-English text chunks into rich vector representations using the **Qwen3 Embedding** model[cite: 1].
3. **High-Speed Vector Retrieval:** Storing and querying vector embeddings within **ChromaDB** using semantic similarity[cite: 1].
4. **Context Provision for LLM:** Providing exact, relevant, and source-attributed context chunks to the LLM (Gemini) layer, ensuring all responses are **100% grounded in approved bank knowledge**.
---
## 🏗 System Architecture & Scope

The complete end-to-end Voice Copilot workflow involves a multi-stage pipeline:

```text
[User Text/Voice Question]
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ FastAPI Gateway & Security Layer                       │
│ (Request Validation & Sensitive-Data Masking)          │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ 📍 THIS REPOSITORY: RAG & Knowledge Retrieval Layer    │
│                                                        │
│  1. Document Parsing & OCR (PaddleOCR / PyMuPDF)       │
│  2. Burmese Text Cleaning & Unicode Normalization      │
│  3. Sentence-Aware Chunking & Metadata Tagging         │
│  4. Qwen3 Embedding Generation                         │
│  5. ChromaDB Semantic Vector Search                    │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼ [Relevant Knowledge Chunks + Metadata]
┌────────────────────────────────────────────────────────┐
│ Grounded LLM Generation Layer (Gemini)                 │
│ (Generates precise Burmese response or Refusal)        │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ Burmese Text Normalization & Text-to-Speech (TTS)      │
└────────────────────────────────────────────────────────┘
## 🛠 Tech Stack Overview

| Layer | Technology |
| :--- | :--- |
| **Frontend** | React + Vite + TypeScript |
| **Backend Framework** | FastAPI (Python 3.11+) |
| **Relational Database** | PostgreSQL |
| **Vector Storage** | ChromaDB |
| **Embedding Model** | Qwen3 Embedding |
| **Generation Model** | Google Gemini (via thin service interface) |
| **Cache & Async Queue** | Redis + Celery / RQ |
| **Document Parsing & OCR** | PyMuPDF, pdfplumber, PaddleOCR |
| **Observability** | structlog (JSON), Prometheus, Grafana |
| **Deployment** | Docker & Docker Compose |

---

## 🚀 Key Features

* **Multilingual Document Ingestion:** Supports PDF, DOCX, PPTX, Images, and FAQs with Burmese OCR support (PaddleOCR).
* **Text Preprocessing & Chunking:** Burmese text normalization, section/heading-aware chunking, and metadata enrichment.
* **Hybrid & Semantic Retrieval:** Qwen3 vector search coupled with ChromaDB for high-precision context retrieval.
* **Grounded Conversational AI:** Session-managed chat using Gemini that strictly answers using retrieved context and refuses when evidence is insufficient.
* **Enterprise Security & Admin Capabilities:** Role-Based Access Control (RBAC), JWT authentication, async reindexing, and complete audit logging.

---


## 🏁 Quickstart & Setup (Development)

### Prerequisites
* Docker & Docker Compose
* Python 3.11+
* PostgreSQL & Redis instances

### Environment Setup
Create a `.env` file in the root directory:
```bash
DATABASE_URL=postgresql://user:password@localhost:5432/burmese_bank_db
REDIS_URL=redis://localhost:6379/0
CHROMADB_HOST=localhost
CHROMADB_PORT=8000
SECRET_KEY=your_production_secret_key


## 2. Scope and Responsibility

This service is responsible for the following:

1. Document ingestion
   - PDF, DOCX, PPTX, images, and FAQ sources
   - manual and automated document intake
   - source and version tracking

2. Text extraction and OCR
   - standard PDF text extraction
   - OCR for scanned PDFs and images using PaddleOCR
   - extraction from document pages and sections

3. Preprocessing and normalization
   - Burmese text normalization
   - whitespace cleanup
   - metadata enrichment
   - section and page awareness

4. Chunking strategy
   - split large documents into smaller semantic chunks
   - preserve source metadata
   - maintain document version and page information

5. Embedding and indexing
   - encode chunks with Qwen3 Embedding
   - store embeddings and metadata in ChromaDB
   - support semantic similarity search for user queries

6. Retrieval service
   - embed incoming queries
   - search vector store for closest matches
   - return top-k relevant chunks with metadata and similarity scores
   - expose clean retrieval payloads for upstream systems

7. Knowledge grounding support
   - ensure downstream generation uses approved and traceable evidence
   - reject unsupported responses at retrieval layer when no relevant results are found

## 3. High-Level Architecture

```text
Approved Banking Documents
   ↓
Text Extraction / OCR
   ↓
Burmese Text Cleaning
   ↓
Chunking + Metadata Enrichment
   ↓
Qwen3 Embedding
   ↓
ChromaDB Vector Storage

User Query
   ↓
Query Embedding
   ↓
Vector Similarity Search
   ↓
Top Relevant Chunks
   ↓
Retrieved Context for Downstream LLM / App Layer

## Expected Retrieval Output

```json
{
  "text": "If an ATM card is lost, contact the bank immediately...",
  "document": "ATM_Card_Policy.pdf",
  "page": 5,
  "section" :"..."
  "score": 0.92
}
```

## Summary

* This repository is the grounding and retrieval layer for a Burmese banking support system. It handles the document lifecycle, text extraction, chunking, embedding, vector storage, and semantic retrieval required to provide trusted context to downstream systems.