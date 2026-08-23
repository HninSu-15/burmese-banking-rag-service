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

Query
  ↓
1. Hybrid Search (BM25 + Vector) → Top N (ဥပမာ - 10) ကိုယူတယ်
  ↓
2. Semantic Reranker (Cosine Similarity)
   → Query Vector နဲ့ Chunk Vector တွေကို တိုက်ရိုက် Cosine တွက်တယ်
  ↓
3. Semantic Gate (Dense Similarity Gate)
   → အမြင့်ဆုံး Score က Threshold (ဥပမာ - 0.5) အောက်ရင် Reject လုပ်တယ်
  ↓
4. Rerank လုပ်ထားတဲ့ Score အတိုင်း ပြန်စီပြီး Top 3 ပြန်ပေးတယ်


* Phase 7: Hardening and Production Readiness
add logs and monitoring
add retries and async task handling
improve indexing performance
ensure secure document access and admin controls

### Standard JSON Output Payload Schema
{
  "query": "ကတ်အသစ် ထုတ်ယူရာတွင် မည်သည့် စာရွက်စာတမ်းများ ယူဆောင်လာရမည်နည်း",
  "language": "my",
  "contexts": [
    {
      "rank": 1,
      "chunk_id": "card_replacement_policy_md_sec_2",
      "question": "ကတ်အသစ် ထုတ်ယူရာတွင် မည်သည့် စာရွက်စာတမ်းများ ယူဆောင်လာရမည်နည်း",
      "text": "ဘဏ်ခွဲတွင် ကတ်အသစ် ထုတ်ယူရာတွင် NRC မူရင်း၊ Passport နှင့် Stay Permit၊ Power of Attorney တို့ ယူဆောင်လာရပါမည်။",
      "source": {
        "doc_name": "card_replacement_policy.md",
        "section": "Required Verification Documents",
        "page_number": 1
      },
      "retrieval_score": 0.1639
    }
  ],
  "instructions": {
    "answer_only_from_context": true,
    "answer_language": "my",
    "include_citations": true,
    "return_json_only": true,
    "do_not_invent_information": true
  }
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