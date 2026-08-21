# app/services/rag/factory.py
"""
Service Factory
"""

import logging
from typing import Optional, Dict, Any

from app.services.rag.exceptions import (
    RAGError,
    EmbeddingError,
    VectorStoreError,
    ParserError,
    ConfigError
)
from app.services.rag.embedder_service import get_embedder_service
from app.services.rag.vector_store import get_vector_store_service
from app.services.rag.ingestion_parser import KnowledgeDocumentParser
from core.config import settings

logger = logging.getLogger(__name__)


class RAGServiceFactory:
    """Singleton factory for RAG services with error handling."""
    
    _embedder = None
    _store = None
    _parser = None
    _initialized = False
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize all services at once."""
        if cls._initialized:
            return
        
        logger.info("Initializing RAG Service Factory...")
        try:
            cls.get_embedder()
            cls.get_store()
            cls.get_parser()
            cls._initialized = True
            logger.info("RAG Service Factory initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Service Factory: {e}")
            raise RAGError(f"Factory initialization failed: {e}")
    
    @classmethod
    def get_embedder(cls) -> Any:
        """Get or create Embedder service."""
        if cls._embedder is None:
            try:
                logger.info("Initializing Embedder Service...")
                cls._embedder = get_embedder_service()
                logger.info("Embedder Service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Embedder: {e}")
                raise EmbeddingError(f"Embedder initialization failed: {e}")
        return cls._embedder
    
    @classmethod
    def get_store(cls, persist_dir: Optional[str] = None) -> Any:
        """
        Get or create Vector Store service.
        
        Args:
            persist_dir: Override persist directory (optional)
                         If not provided, uses settings.CHROMA_PERSIST_DIR
        """
        if cls._store is None:
            try:
                logger.info("Initializing Vector Store...")
                # Use config if persist_dir not provided
                if persist_dir is None:
                    persist_dir = settings.CHROMA_PERSIST_DIR
                
                cls._store = get_vector_store_service(
                    persist_dir=persist_dir,
                    collection_name=settings.COLLECTION_NAME
                )
                logger.info(f"Vector Store initialized successfully at: {persist_dir}")
            except Exception as e:
                logger.error(f"Failed to initialize Vector Store: {e}")
                raise VectorStoreError(f"Vector Store initialization failed: {e}")
        return cls._store
    
    @classmethod
    def get_parser(cls) -> KnowledgeDocumentParser:
        """Get or create Document Parser service."""
        if cls._parser is None:
            try:
                logger.info("Initializing Document Parser...")
                cls._parser = KnowledgeDocumentParser(enable_pii_masking=True)
                logger.info("Document Parser initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Parser: {e}")
                raise ParserError(f"Parser initialization failed: {e}")
        return cls._parser
    
    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        """Check all services are working."""
        return {
            "initialized": cls._initialized,
            "embedder": cls._embedder is not None,
            "vector_store": cls._store is not None,
            "parser": cls._parser is not None,
            "status": "healthy" if cls._initialized else "not_initialized"
        }
    
    @classmethod
    def reset(cls) -> None:
        """Reset all services (for testing)."""
        cls._embedder = None
        cls._store = None
        cls._parser = None
        cls._initialized = False
        logger.info("RAG Service Factory reset")