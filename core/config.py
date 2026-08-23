# core/config.py
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class RAGSettings:
    """Central configuration for RAG Knowledge Retrieval Pipeline."""
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR", 
        str(PROJECT_ROOT / "data" / "chroma_db")  # ← Absolute path
    )
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "burmese_banking_knowledge")
    
    # Embedding Model Configuration
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding")
    
    # Chunking Strategy Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # Retrieval Configuration
    TOP_K: int = int(os.getenv("TOP_K", "3"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    SEMANTIC_DOMAIN_THRESHOLD: float = float(os.getenv("SEMANTIC_DOMAIN_THRESHOLD", "0.55"))
    RERANKER_THRESHOLD: float = float(os.getenv("RERANKER_THRESHOLD", "0.2"))

# Global Settings Instance
settings = RAGSettings()