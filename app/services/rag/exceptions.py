"""
Custom Exceptions with Error Codes
"""

from typing import Optional


class RAGError(Exception):
    """Base RAG exception."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code or "RAG-000"
        super().__init__(self.message)


class EmbeddingError(RAGError):
    """Embedding service errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, error_code or "RAG-001")


class VectorStoreError(RAGError):
    """Vector store errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, error_code or "RAG-002")


class ChunkingError(RAGError):
    """Chunking errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, error_code or "RAG-003")


class ParserError(RAGError):
    """Parser errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, error_code or "RAG-004")


class ConfigError(RAGError):
    """Configuration errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, error_code or "RAG-005")


class DocumentNotFoundError(RAGError):
    """Document not found errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, error_code or "RAG-006")