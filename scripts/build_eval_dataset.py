# scripts/build_eval_dataset.py
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # scripts/ → project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import json
from pathlib import Path
from app.services.rag.vector_store import get_vector_store_service
from core.config import settings

def export_chunks_for_eval():
    store = get_vector_store_service(
        persist_dir=settings.CHROMA_PERSIST_DIR,
        collection_name=settings.COLLECTION_NAME
    )
    
    all_data = store._chroma_collection.get()
    
    if not all_data or not all_data['ids']:
        print("❌ No chunks found in ChromaDB!")
        return
    
    eval_data = []
    
    for i, chunk_id in enumerate(all_data['ids']):
        metadata = all_data['metadatas'][i] if all_data['metadatas'] else {}
        text = all_data['documents'][i] if all_data['documents'] else ""
        
        section_title = metadata.get('section_title', '')
        if "Q:" not in section_title:
            continue  
        
        eval_data.append({
            "chunk_id": chunk_id,
            "doc_name": metadata.get('doc_name', 'Unknown'),
            "section_title": section_title,
            "text_preview": text[:200],  
            "test_queries": []  
        })
    
    output_path = Path(__file__).parent.parent / "evaluation" / "base_chunks.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Exported {len(eval_data)} FAQ chunks to {output_path}")
    print("📝 Now manually add 'test_queries' for each chunk in this file.")

if __name__ == "__main__":
    export_chunks_for_eval()