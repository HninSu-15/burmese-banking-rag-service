# scripts/ingest_single.py
"""
Ingest a single markdown file into ChromaDB.
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.rag.factory import RAGServiceFactory
from app.services.rag.chunker import SectionDocumentChunker
from llama_index.core.schema import TextNode
from core.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = PROJECT_ROOT / "data" / "chroma_db"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def ingest_single(file_path: str):
    """Ingest a single markdown file."""
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    if file_path.suffix != ".md":
        print(f"❌ Only .md files supported: {file_path}")
        return
    
    print("=" * 60)
    print(f"📄 Ingesting: {file_path.name}")
    print("=" * 60)
    
    try:
        # Get services
        parser = RAGServiceFactory.get_parser()
        embedder = RAGServiceFactory.get_embedder()
        
        from app.services.rag.vector_store import get_vector_store_service
        store = get_vector_store_service(
            persist_dir=str(DATA_DIR),
            collection_name=settings.COLLECTION_NAME
        )
        
        chunker = SectionDocumentChunker()
        
        # 1. Parse
        print("📖 Parsing document...")
        doc = parser.parse_markdown(str(file_path))
        print(f"   ✅ Parsed: {doc.doc_name}")
        print(f"   📄 Pages: {doc.total_pages}")
        print(f"   📝 Text length: {len(doc.pages[0].raw_text)} chars")
        
        # 2. Chunk
        print("✂️ Chunking document...")
        chunks = chunker.chunk_document(doc)
        print(f"   📦 Chunks: {len(chunks)}")
        
        # 3. Convert to nodes and embed
        print("🧠 Generating embeddings...")
        nodes = []
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            if not text:
                text = chunk.get('raw_text', '')
            
            # Get embedding
            vector = embedder.get_text_embedding(text)
            
            # ✅ FIX: Store text with passage: prefix in ChromaDB
            node = TextNode(text=f"passage: {text}", embedding=vector)
            
            if 'metadata' in chunk:
                node.metadata = chunk['metadata']
            
            nodes.append(node)
            if (i + 1) % 5 == 0:
                print(f"   🔄 Processed {i + 1}/{len(chunks)} chunks")
        
        # 4. Store
        print("💾 Storing in ChromaDB...")
        store.add_nodes(nodes)
        
        # 5. Summary
        print("\n" + "=" * 60)
        print("✅ INGESTION COMPLETE!")
        print(f"   File: {file_path.name}")
        print(f"   Chunks: {len(nodes)}")
        print(f"   Vector Store: {store.count_documents()} documents")
        print(f"   Data Location: {DATA_DIR}")
        print("=" * 60)
        
        # 6. Verify passage: prefix in ChromaDB
        print("\n🔍 Verifying passage: prefix in ChromaDB...")
        results = store._chroma_collection.get(limit=1)
        if results and results['documents']:
            doc_preview = results['documents'][0][:150]
            print(f"   Document preview: {doc_preview}")
            if doc_preview.startswith('passage:'):
                print("   ✅ 'passage:' prefix found in document!")
            else:
                print("   ❌ 'passage:' prefix NOT found in document!")
                print("   ⚠️ Please check TextNode text field")
        
    except Exception as e:
        logger.error(f"Failed to ingest {file_path.name}: {e}")
        print(f"❌ Failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_single.py <path_to_md_file>")
        print("Example: python scripts/ingest_single.py ./knowledge/ATM_Services_FAQ.md")
        sys.exit(1)
    
    file_path = sys.argv[1]
    ingest_single(file_path)