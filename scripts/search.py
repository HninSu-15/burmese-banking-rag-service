# scripts/search.py
"""
Search script for RAG knowledge base.
Usage: python scripts/search.py "သင့်ရဲ့မေးခွန်း"
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import settings
from app.services.rag.vector_store import get_vector_store_service


def search(query: str, top_k: int = None, threshold: float = None):
    """Search and display results with threshold filtering."""
    
    # Use config values if not provided
    if top_k is None:
        top_k = settings.TOP_K
    if threshold is None:
        threshold = settings.SIMILARITY_THRESHOLD
    
    print(f"\n🔍 Query: {query}")
    print(f"📊 Top-K: {top_k}, Threshold: {threshold}")
    print("=" * 60)
    
    try:
        # Get store
        store = get_vector_store_service(
            persist_dir=settings.CHROMA_PERSIST_DIR,
            collection_name=settings.COLLECTION_NAME
        )
        
        # Debug: Check if store has documents
        doc_count = store.count_documents()
        print(f"📊 Total documents in Vector Store: {doc_count}")
        
        if doc_count == 0:
            print("❌ No documents found. Please run ingest_all.py first.")
            return
        
        # Search with threshold
        results = store.retrieve(
            query=query,
            top_k=top_k,
            threshold=threshold
        )
        
        if not results:
            print(f"❌ သက်ဆိုင်သော အချက်အလက် ရှာမတွေ့ပါ။ (Threshold: {threshold})")
            print("   Please try a different query or lower the threshold.")
            return
        
        # Display results
        for i, r in enumerate(results, 1):
            score = r.get('relevance_score', 0)
            metadata = r.get('metadata', {})
            text = r.get('text', '')
            
            print(f"\n📌 Result {i} (Score: {score:.4f})")
            print(f"   Document: {metadata.get('doc_name', 'Unknown')}")
            print(f"   Section: {metadata.get('section_title', 'Unknown')}")
            print(f"\n   📝 {text}")
            print("-" * 60)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/search.py 'သင့်ရဲ့မေးခွန်း'")
        print("Example: python scripts/search.py 'ကတ်အသစ် ထုတ်ယူရာတွင် မည်သည့် စာရွက်စာတမ်းများ ယူဆောင်လာရမည်နည်း'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    search(query)