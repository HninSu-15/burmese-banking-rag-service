# app/services/rag/embedder_service.py
import os
import ssl
import logging
from typing import List, Optional, Any
from pathlib import Path

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = ""
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

from pydantic import Field, PrivateAttr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from app.services.rag.normalizer import BurmeseTextNormalizer
from app.services.rag.ingestion_parser import BurmesePIIMasker
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

# Configure Logger
logger = logging.getLogger(__name__)


class Qwen3GGUFEmbedder(BaseEmbedding):
    """
    Production-grade LlamaIndex Custom Embedding Wrapper for Qwen3-Embedding GGUF
    running on CPU using llama-cpp-python.
    """

    model_path: str = Field(
        default="",
        description="Local path to the .gguf model file."
    )
    repo_id: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B-GGUF",  
        description="HuggingFace repository ID if auto-downloading."
    )
    filename: str = Field(
        default="Qwen3-Embedding-0.6B-Q8_0.gguf",  
        description="GGUF filename inside HuggingFace repository."
    )
    n_ctx: int = Field(
        default=512,
        description="Context window length. 512 tokens is optimal for CPU inference."
    )
    n_threads: int = Field(
        default=os.cpu_count() or 4,
        description="Number of CPU threads to use."
    )

    # Private internal llama.cpp engine instance
    _model: Llama = PrivateAttr()

    def __init__(
        self,
        model_path: Optional[str] = None,
        repo_id: str = "Qwen/Qwen3-Embedding-0.6B-GGUF",  
        filename: str = "Qwen3-Embedding-0.6B-Q8_0.gguf", 
        n_ctx: int = 512,
        n_threads: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Qwen3 GGUF Embedder service.
        """
        super().__init__(
            model_path=model_path or "",
            repo_id=repo_id,
            filename=filename,
            n_ctx=n_ctx,
            n_threads=n_threads or (os.cpu_count() or 4),
            **kwargs,
        )

        # ============================================================
        # Initialize Normalizer and PII Masker
        # ============================================================
        self._normalizer = BurmeseTextNormalizer()
        self._pii_masker = BurmesePIIMasker()

        # Download model if model_path is not explicitly provided or doesn't exist
        resolved_path = self._resolve_model_path(self.model_path, self.repo_id, self.filename)
        self.model_path = resolved_path

        logger.info(f"Loading Qwen3 GGUF Embedding model from: {self.model_path}")

        # Load the C++ model into memory
        try:
            self._model = Llama(
                model_path=self.model_path,
                embedding=True,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )
            logger.info("Qwen3 Embedding Model loaded successfully on CPU.")
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {str(e)}")
            raise RuntimeError(f"Error initializing llama-cpp model: {e}")

    @staticmethod
    def _resolve_model_path(path_str: str, repo_id: str, filename: str) -> str:
        """
        Resolves model file path or downloads it automatically from HuggingFace cache.
        """
        if path_str and Path(path_str).exists():
            return path_str

        logger.info(f"Model file not found locally. Downloading {filename} from HuggingFace ({repo_id})...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
            )
            return downloaded_path
        except Exception as e:
            logger.error(f"Failed to download model from HuggingFace: {str(e)}")
            raise FileNotFoundError(f"Could not locate or download GGUF model: {e}")

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Generates embedding for document chunks (Passages).
        """
        if not text or not text.strip():
            # Return zero vector fallback for empty strings
            return [0.0] * 1024

        # ============================================================
        # Step 1: Normalize text
        # ============================================================
        cleaned_text = self._normalizer.clean(text)

        # ============================================================
        # Step 2: Mask PII
        # ============================================================
        masked_text = self._pii_masker.mask(cleaned_text)

        # ============================================================
        # Step 3: Apply Qwen3 instruction prefix formatting
        # ============================================================
        formatted_text = f"passage: {masked_text}"
        response = self._model.create_embedding(formatted_text)
        embedding = response["data"][0]["embedding"]
        return embedding

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Generates embedding for user search queries.
        """
        if not query or not query.strip():
            return [0.0] * 1024

        # ============================================================
        # Step 1: Normalize query
        # ============================================================
        cleaned_query = self._normalizer.clean(query)

        # ============================================================
        # Step 2: Mask PII in query
        # ============================================================
        masked_query = self._pii_masker.mask(cleaned_query)

        # ============================================================
        # Step 3: Apply Qwen3 instruction prefix formatting for search queries
        # ============================================================
        formatted_query = f"query: {masked_query}"
        response = self._model.create_embedding(formatted_query)
        embedding = response["data"][0]["embedding"]
        return embedding

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """
        Async wrapper for document embedding generation.
        """
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        Async wrapper for query embedding generation.
        """
        return self._get_query_embedding(query)


# Singleton Instance Factory for FastAPI Dependency Injection
_embedder_instance: Optional[Qwen3GGUFEmbedder] = None


def get_embedder_service(
    model_path: Optional[str] = None,
    n_ctx: int = 512,
) -> Qwen3GGUFEmbedder:
    """
    Returns a singleton instance of Qwen3GGUFEmbedder to prevent
    re-loading the GGUF model binary into memory multiple times.
    """
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Qwen3GGUFEmbedder(
            model_path=model_path,
            n_ctx=n_ctx,
        )
    return _embedder_instance