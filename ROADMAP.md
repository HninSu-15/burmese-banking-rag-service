##  Security and Operational Standards
This repository should follow production-grade standards:

secure environment variable managementJWT or service-level auth for admin and internal APIs
strict validation for document uploadsrole-based access for document management and admin actionsstructured logging for retrieval and indexing workflowsobservability for search latency and ingestion failures version-controlled document indexing and reindex processes
audit trail for uploads, updates, and reindex events

##  Development Roadmap

* Phase 1: Foundation
initialize FastAPI project
configure Postgres, Redis, and environment settings
define project structure and base schemas
create health-check and status endpoints

* Phase 2: Document Ingestion
implement upload flow for PDFs and documents
persist metadata and file records
validate supported file types

* Phase 3: Extraction and OCR
parse PDF and DOCX text
integrate PaddleOCR for scanned images
store raw extracted content
+-----------------------+
                                  | Raw Banking Documents |
                                  | (PDF, Scanned Images) |
                                  +-----------+-----------+
                                              |
                                              v
                              +---------------+---------------+
                              |    File Type & Layout Detector|
                              +---------------+---------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
             [Native Digital PDF]                               [Scanned PDF / Image]
                     |                                                 |
                     v                                                 v
           +---------+---------+                             +---------+---------+
           | PyMuPDF Extractor |                             | PaddleOCR Engine  |
           +---------+---------+                             +---------+---------+
                     |                                                 |
                     +------------------------+------------------------+
                                              |
                                              v
                              +---------------+---------------+
                              | Clean & Normalization Pipeline|
                              |  (Unicode / Burmese Refining) |
                              +---------------+---------------+
                                              |
                                              v
                              +---------------+---------------+
                              | Structuring & Metadata Tagging|
                              +---------------+---------------+
                                              |
                                              v
                                  (Pass to Module 4: Chunking)


[ Document Folder ]
       │
       ├── File 1 ──► [ Parse & Clean ] ──► [ Section Chunk ] ──► [ Embedding ] ──► [ ChromaDB Upsert ]
       ├── File 2 ──► [ Parse & Clean ] ──► [ Section Chunk ] ──► [ Embedding ] ──► [ ChromaDB Upsert ]
       └── ...

* Phase 4: Cleaning and Chunking
normalize Burmese text
split documents into chunks
preserve page and section metadata


* Phase 5: Embedding and Indexing
embed each chunk using Qwen3
store vector payloads in ChromaDB
test retrieval quality on sample banking questions

ChromaDB Schema:
├── Collection (1) → burmese_bank_knowledge
│   └── Documents (N) → Chunks
│       ├── id (string)
│       ├── embedding (List[float]) → 1024 numbers
│       ├── document (string) → Original text
│       └── metadata (dict) → doc_id, section_title, etc.
└── Storage → chroma.sqlite3 + data.parquet

* Phase 6: Retrieval API
create query endpoint
return top-k matching chunks
include source metadata and scores
User Query
    ↓
Embedding → Query Vector
    ↓
ChromaDB Search (Cosine Similarity)
    ↓
Top-K Results (အနီးဆုံး Chunks)
    ↓
Response (Answer)

* Phase 7: Hardening and Production Readiness
add logs and monitoring
add retries and async task handling
improve indexing performance
ensure secure document access and admin controls

### Standard JSON Output Payload Schema
{
  "query": "ATM ကတ် ပျောက်သွားရင် ဘာလုပ်ရမလဲ။",
  "has_context": true,
  "retrieved_count": 2,
  "max_score": 0.885,
  "contexts": [
    {
      "chunk_id": "chunk_9f823a",
      "text": "ATM ကတ် ပျောက်ဆုံးပါက သက်ဆိုင်ရာ ဘဏ်သို့ ချက်ချင်း ဖုန်းဆက်၍ ကတ်အား ပိတ်ဆို့ (Block) ရပါမည်။ ဘဏ်ခွဲသို့ လူကိုယ်တိုင် သွားရောက်ပါက မှတ်ပုံတင် မူရင်း ယူဆောင်လာရပါမည်။",
      "score": 0.885,
      "metadata": {
        "doc_id": "doc_102",
        "doc_name": "ATM_Services_FAQ_v1.pdf",
        "page_label": "3",
        "section_title": "Card Loss Procedure",
        "department": "Retail Banking",
        "last_updated": "2026-01-15"
      }
    },
    {
      "chunk_id": "chunk_9f823b",
      "text": "ကတ် အသစ်ပြန်လည် လျှောက်ထားပါက လျှောက်ထားခ ကျပ် ၅,၀၀၀ ကျသင့်မည် ဖြစ်ပါသည်။",
      "score": 0.762,
      "metadata": {
        "doc_id": "doc_102",
        "doc_name": "ATM_Services_FAQ_v1.pdf",
        "page_label": "4",
        "section_title": "Service Fees",
        "department": "Retail Banking",
        "last_updated": "2026-01-15"
      }
    }
  ]
}

## current folder structure 
project/
├── core/
│   ├── __init__.py
│   └── config.py              # ← ခင်ဗျားရဲ့ မူရင်းဖိုင်
├── app/
│   └── services/
│       └── rag/
│           ├── __init__.py
│           ├── normalizer.py
│           ├── ingestion_parser.py
│           ├── chunker.py
│           ├── embedding_service.py   # ← config ကိုသုံးမယ်
│           ├── vector_store.py        # ← config ကိုသုံးမယ်
│           └── factory.py             # ← config ကိုသုံးမယ်
├── main.py                            # ← အသုံးပြုပုံ
└── .env                               # ← Environment variables