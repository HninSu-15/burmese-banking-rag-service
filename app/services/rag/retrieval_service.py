# app/services/rag/retrieval_service.py
"""Programmatic retrieval logic used by the /api/v1/retrieve endpoint.

Wraps the HybridRetriever + SemanticReranker + LLMContextBuilder pipeline
so callers receive the exact JSON contract consumed by the LLM team.
"""

import logging
from typing import Dict, Any

from app.services.rag.vector_store import get_vector_store_service
from app.services.rag.embedder_service import get_embedder_service
from app.services.rag.hybrid_retriever import HybridRetriever, SemanticReranker
from app.services.rag.context_builder import LLMContextBuilder
from core.config import settings

logger = logging.getLogger(__name__)


def retrieve_context(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Run hybrid retrieval and return the LLM context contract JSON."""
    if top_k < 1:
        top_k = settings.TOP_K

    store = get_vector_store_service(
        persist_dir=settings.CHROMA_PERSIST_DIR,
        collection_name=settings.COLLECTION_NAME,
    )
    builder = LLMContextBuilder(max_contexts=top_k)

    if store.count_documents() == 0:
        logger.info("Retrieve called but the vector store is empty; returning refusal.")
        return builder.build(query, [])

    retriever = HybridRetriever(
        vector_store=store,
        candidate_k=max(10, top_k * 3),
        reranker=SemanticReranker(get_embedder_service()),
    )
    results = retriever.retrieve(query=query, top_k=top_k)

    payload = builder.build(query, results)
    logger.info(
        "Retrieve completed: query_preview=%r has_context=%s results=%d",
        query[:80],
        payload.get("has_context"),
        len(payload.get("contexts", [])),
    )
    return payload