# scripts/ingest_all.py
"""
Ingest all markdown files from knowledge/ folder into ChromaDB.
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # scripts/ → project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.rag.factory import RAGServiceFactory
from app.services.rag.exceptions import RAGError
from app.services.rag.chunker import SectionDocumentChunker
from llama_index.core.schema import TextNode
from core.config import settings

# ============================================================
# FIX: Data directory ကို project root အောက်မှာ သတ်မှတ်ပါ
# ============================================================
# ❌ ဒါကိုဖယ်ပါ (Relative Path ဖြစ်နေတယ်)
# DATA_DIR = Path(settings.CHROMA_PERSIST_DIR)

# ✅ ဒီလိုပြင်ပါ (Project Root ကိုသုံးမယ်)
DATA_DIR = PROJECT_ROOT / "data" / "chroma_db"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logs_dir = PROJECT_ROOT / "logs"
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "ingestion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def ingest_all():
    """Ingest all documents from knowledge/ folder."""
    
    knowledge_dir = PROJECT_ROOT / "knowledge"
    
    if not knowledge_dir.exists():
        logger.error(f"Knowledge directory not found: {knowledge_dir}")
        print(f"❌ Please create '{knowledge_dir}' and add .md files.")
        return
    
    md_files = list(knowledge_dir.glob("*.md"))
    
    if not md_files:
        logger.warning(f"No .md files found in {knowledge_dir}")
        print(f"⚠️ No .md files found in {knowledge_dir}")
        return
    
    print(f"\n📁 Found {len(md_files)} .md files to ingest")
    print(f"💾 Data Directory: {DATA_DIR}")
    print("=" * 50)
    
    try:
        # Get services
        parser = RAGServiceFactory.get_parser()
        embedder = RAGServiceFactory.get_embedder()
        
        # ============================================================
        # FIX: Vector Store ကို project root data နဲ့ သတ်မှတ်ပါ
        # ============================================================
        from app.services.rag.vector_store import get_vector_store_service
        store = get_vector_store_service(
            persist_dir=str(DATA_DIR),
            collection_name=settings.COLLECTION_NAME
        )
        
        chunker = SectionDocumentChunker()
        
        total_chunks = 0
        success_count = 0
        failed_files = []
        
        for i, file_path in enumerate(md_files, 1):
            print(f"\n[{i}/{len(md_files)}] Processing: {file_path.name}")
            
            try:
                # 1. Parse
                doc = parser.parse_markdown(str(file_path))
                
                # 2. Chunk
                chunks = chunker.chunk_document(doc)
                print(f"   📦 Chunks: {len(chunks)}")
                
                if len(chunks) <= 1:
                    logger.warning(f"Only {len(chunks)} chunk generated for {file_path.name}. Check chunker logic.")
                    print(f"   ⚠️ Warning: Only {len(chunks)} chunk generated!")
                
                # 3. Convert to nodes and embed
                nodes = []
                for chunk in chunks:
                    # Get text
                    text = chunk.get('text', '')
                    if not text:
                        text = chunk.get('raw_text', '')
                    
                    # Get embedding
                    vector = embedder.get_text_embedding(text)
                    
                    # ============================================================
                    # FIX: Store text with passage: prefix in ChromaDB
                    # ============================================================
                    node = TextNode(text=f"passage: {text}", embedding=vector)
                    
                    # Add metadata if available
                    if 'metadata' in chunk:
                        node.metadata = chunk['metadata']
                    elif 'section_title' in chunk:
                        # Direct metadata fields
                        node.metadata = {
                            'doc_id': doc.doc_id,
                            'doc_name': doc.doc_name,
                            'section_title': chunk.get('section_title', ''),
                            'section_level': chunk.get('section_level', 0),
                            'parent_section': chunk.get('parent_section', ''),
                            'header_path': chunk.get('header_path', '')
                        }
                    
                    nodes.append(node)
                
                # 4. Store in ChromaDB
                store.add_nodes(nodes)
                
                total_chunks += len(nodes)
                success_count += 1
                print(f"   ✅ Ingested: {len(nodes)} chunks")
                
            except Exception as e:
                logger.error(f"Failed to ingest {file_path.name}: {e}")
                failed_files.append(file_path.name)
                print(f"   ❌ Failed: {e}")
        
        # Summary
        print("\n" + "=" * 50)
        print(f"✅ INGESTION COMPLETE!")
        print(f"   Total files: {len(md_files)}")
        print(f"   Success: {success_count}")
        print(f"   Failed: {len(failed_files)}")
        if failed_files:
            print(f"   Failed files: {', '.join(failed_files)}")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Vector Store: {store.count_documents()} documents")
        print(f"   Data Location: {DATA_DIR}")
        print("=" * 50)
        
    except RAGError as e:
        logger.error(f"RAG Error: {e}")
        print(f"❌ Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("🚀 RAG Knowledge Base Ingestion")
    print("=" * 50)
    print(f"📂 Project Root: {PROJECT_ROOT}")
    print(f"📁 Knowledge Dir: {PROJECT_ROOT / 'knowledge'}")
    print(f"💾 Data Dir: {DATA_DIR}")
    print("=" * 50)
    
    ingest_all()