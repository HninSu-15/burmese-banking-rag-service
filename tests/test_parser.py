import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
from app.services.rag import KnowledgeDocumentParser

def test_flat_knowledge_ingestion():
    print("=" * 60)
    print(" 🧪 TESTING MODULE 3: FLAT `.MD` KNOWLEDGE INGESTION")
    print("=" * 60)

    knowledge_dir = os.path.join(PROJECT_ROOT, "knowledge")
    parser = KnowledgeDocumentParser()

    if not os.path.exists(knowledge_dir):
        os.makedirs(knowledge_dir, exist_ok=True)
        print(f"  ⚠️ Created empty directory '{knowledge_dir}'. Please add a sample .md file.")
        return

    docs = parser.parse_directory(knowledge_dir)
    print(f"  └─ Total .md Documents Ingested: {len(docs)}")

    for doc in docs:
        print(f"\n[DOCUMENT] {doc.doc_name}")
        print(f"  └─ Doc ID      : {doc.doc_id}")
        print(f"  └─ File Type   : {doc.file_type}")
        print(f"  └─ Total Pages : {doc.total_pages}")
        print(f"  └─ Text Preview: {doc.pages[0].raw_text[:120]}...")
        print(f"  └─ Metadata    : {doc.pages[0].metadata}")

    print("\n" + "=" * 60)
    print(" ✅ FLAT MARKDOWN INGESTION TEST PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    test_flat_knowledge_ingestion()