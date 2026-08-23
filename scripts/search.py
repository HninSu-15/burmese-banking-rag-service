# scripts/search.py
"""
Search script for RAG knowledge base.
Usage: python scripts/search.py "သင့်ရဲ့မေးခွန်း"
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from core.config import settings
from app.services.rag.hybrid_retriever import HybridRetriever
from app.services.rag.hybrid_retriever import SemanticReranker
from app.services.rag.context_builder import LLMContextBuilder
from app.services.rag.embedder_service import get_embedder_service
from app.services.rag.vector_store import get_vector_store_service


def search(query: str, top_k: int = None, threshold: float = None):
    """Search and display results with threshold filtering."""
    
    # Use config values if not provided
    if top_k is None:
        top_k = settings.TOP_K
    if threshold is None:
        threshold = settings.SIMILARITY_THRESHOLD
    
    try:
        # Get store
        store = get_vector_store_service(
            persist_dir=settings.CHROMA_PERSIST_DIR,
            collection_name=settings.COLLECTION_NAME
        )
        
        # Debug: Check if store has documents
        doc_count = store.count_documents()
        if doc_count == 0:
            print(json.dumps(
                LLMContextBuilder(max_contexts=top_k).build(query, []),
                ensure_ascii=False,
                indent=2,
            ))
            return
        
        # Search with hybrid BM25 + vector retrieval.
        retriever = HybridRetriever(
            vector_store=store,
            candidate_k=max(10, top_k * 3),
            reranker=SemanticReranker(get_embedder_service()),
        )
        results = retriever.retrieve(query=query, top_k=top_k)
        
        if not results:
            print(json.dumps(
                LLMContextBuilder(max_contexts=top_k).build(query, []),
                ensure_ascii=False,
                indent=2,
            ))
            return

        print(json.dumps(LLMContextBuilder(max_contexts=top_k).build(query, results), ensure_ascii=False, indent=2))
        return
            
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