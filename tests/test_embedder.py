import sys
from pathlib import Path

# Set up project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.rag.embedder_service import get_embedder_service

def test_embedder():
    print("Initializing Embedder Service...")
    embedder = get_embedder_service()

    # 1. Test Query Embedding
    query = "ATM ကတ် ပျောက်သွားရင် ဘာလုပ်ရမလဲ"
    query_vector = embedder.get_query_embedding(query)
    print(f"Query Vector Length: {len(query_vector)}")
    print(f"Sample values: {query_vector[:5]}")

    # 2. Test Document Chunk Embedding
    doc_chunk = "ATM ကတ် ပျောက်ဆုံးပါက ဘဏ်၏ Call Center သို့ ချက်ချင်း ဖုန်းဆက်၍ ကတ်ကို ပိတ်ရမည်။"
    doc_vector = embedder.get_text_embedding(doc_chunk)
    print(f"Doc Chunk Vector Length: {len(doc_vector)}")

if __name__ == "__main__":
    test_embedder()