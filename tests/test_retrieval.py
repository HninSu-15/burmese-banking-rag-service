# Retrieval Test
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from app.services.rag.vector_store import get_vector_store_service
from app.services.rag.embedder_service import get_embedder_service

store = get_vector_store_service()
embedder = get_embedder_service()

query = "ATM ကတ်ပျောက်ရင် ဘာလုပ်ရမလဲ"
query_vector = embedder.get_query_embedding(query)

results = store.retrieve(query_vector, top_k=3)

for r in results:
    print(f"Score: {r['score']:.3f}")
    print(f"Header Path: {r['metadata']['header_path']}")
    print(f"Text: {r['text'][:100]}...")
    print("---")