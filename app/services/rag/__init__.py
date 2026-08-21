# app/services/rag/__init__.py
"""
RAG Service Package
"""

# Core modules
from app.services.rag.normalizer import BurmeseTextNormalizer
from app.services.rag.ingestion_parser import (
    KnowledgeDocumentParser, ParsedDocument, ExtractedPage, BurmesePIIMasker
)
from app.services.rag.chunker import SectionDocumentChunker
#from app.services.rag.retriever import HybridRetriever

# Services
from app.services.rag.embedder_service import get_embedder_service
from app.services.rag.vector_store import get_vector_store_service

# Factory
from app.services.rag.factory import RAGServiceFactory  

# Exceptions
from app.services.rag.exceptions import *

# Version
__version__ = "1.0.0"

__all__ = [
    # Normalizer
    "BurmeseTextNormalizer",
    
    # Parser
    "KnowledgeDocumentParser",
    "ParsedDocument",
    "ExtractedPage",
    "BurmesePIIMasker",
    
    # Chunker
    "SectionDocumentChunker",
    
    # Services
    "get_embedder_service",
    "get_vector_store_service",
    #"HybridRetriever",
    
    # Factory
    "RAGServiceFactory",  
    
    # Exceptions
    "RAGError",
    "EmbeddingError",
    "VectorStoreError",
    "ChunkingError",
    "ParserError",
    "ConfigError"
]