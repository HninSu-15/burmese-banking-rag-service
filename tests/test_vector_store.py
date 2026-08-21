import sys
from pathlib import Path

# Set up project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llama_index.core.schema import TextNode
from app.services.rag.embedder_service import get_embedder_service
from app.services.rag.vector_store import get_vector_store_service

if __name__ == "__main__":
    print("--- Testing Vector Store Service ---")
    
    # 1. Initialize Services
    embedder = get_embedder_service()
    vector_service = get_vector_store_service()
    
    # 2. Dummy Test Node Creation
    sample_text = "ATM ကတ် ပျောက်ဆုံးပါက ဘဏ်သို့ ချက်ချင်း ဖုန်းဆက်၍ ကတ်ပိတ်ရပါမည်။"
    text_vector = embedder.get_text_embedding(sample_text)
    
    node = TextNode(text=sample_text, embedding=text_vector)
    
    # 3. Add node to Vector DB
    index = vector_service.add_nodes(nodes=[node], embed_model=embedder)
    
    # 4. Verify output
    doc_count = vector_service.count_documents()
    print(f"Success! Total documents in Persistent Vector Store: {doc_count}")