"""
Main RAG Pipeline
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from app.services.rag import RAGServiceFactory
from app.services.rag.exceptions import RAGError
from core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ingest_knowledge_base(knowledge_dir: str = "./knowledge") -> dict:
    """Ingest all documents from knowledge directory."""
    try:
        logger.info(f"Starting ingestion from: {knowledge_dir}")
        
        parser = RAGServiceFactory.get_parser()
        embedder = RAGServiceFactory.get_embedder()
        store = RAGServiceFactory.get_store()
        
        # Parse documents
        docs = parser.parse_directory(knowledge_dir)
        logger.info(f"Parsed {len(docs)} documents")
        
        if not docs:
            return {"chunks": 0, "docs": 0, "failed": []}
        
        # Chunk and ingest
        # ... (chunking logic)
        
        return {"chunks": total_chunks, "docs": len(docs), "failed": []}
        
    except RAGError as e:
        logger.error(f"Ingestion failed: {e}")
        return {"chunks": 0, "docs": 0, "failed": [str(e)]}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"chunks": 0, "docs": 0, "failed": [str(e)]}


def search_knowledge_base(query: str, top_k: int = 3) -> list:
    """Search the knowledge base."""
    try:
        store = RAGServiceFactory.get_store()
        return store.retrieve(query, top_k)
    except RAGError as e:
        logger.error(f"Search failed: {e}")
        return []


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    parser.add_argument(
        "--ingest",
        type=str,
        help="Ingest documents from directory",
        default=None
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search query",
        default=None
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of results to return"
    )
    
    args = parser.parse_args()
    
    if args.ingest:
        stats = ingest_knowledge_base(args.ingest)
        print(f"✅ Ingested {stats['chunks']} chunks from {stats['docs']} docs")
        if stats['failed']:
            print(f"⚠️ Failed: {stats['failed']}")
    
    elif args.search:
        results = search_knowledge_base(args.search, args.top_k)
        print(f"\n🔍 Results for: '{args.search}'")
        print("=" * 50)
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] Score: {r.get('score', 'N/A'):.3f}")
            print(f"   Document: {r['metadata'].get('doc_name', 'Unknown')}")
            print(f"   Preview: {r['text'][:200]}...")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()