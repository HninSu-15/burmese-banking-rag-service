# app/api/v1/routes.py
"""RAG service API v1 routes."""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import require_service_token
from app.api.schemas import (
    DeleteResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from app.services.rag.factory import RAGServiceFactory
from app.services.rag.ingestion_service import ingest_markdown_content, ingest_markdown_file
from app.services.rag.retrieval_service import retrieve_context
from app.services.rag.vector_store import get_vector_store_service
from core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_service_token)])


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Return service status and component readiness."""
    factory_health = RAGServiceFactory.health_check()
    documents = 0
    try:
        store = get_vector_store_service(
            persist_dir=settings.CHROMA_PERSIST_DIR,
            collection_name=settings.COLLECTION_NAME,
        )
        documents = store.count_documents()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Health: vector store unavailable: %s", exc)

    status_label = "healthy" if factory_health["initialized"] else "not_initialized"
    return HealthResponse(
        service=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        status=status_label,
        initialized=factory_health["initialized"],
        embedder=factory_health["embedder"],
        vector_store=factory_health["vector_store"],
        parser=factory_health["parser"],
        documents=documents,
    )


@router.post("/retrieve", response_model=RetrieveResponse, tags=["retrieval"])
def retrieve(payload: RetrieveRequest) -> Dict[str, Any]:
    """Return the LLM context contract for a query.

    This is the primary integration point for the LLM/generation layer.
    """
    try:
        return retrieve_context(query=payload.query, top_k=payload.top_k)
    except Exception as exc:
        logger.exception("Retrieval failed for query %r", payload.query[:80])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {exc}",
        ) from exc


@router.post("/ingest", response_model=IngestResponse, tags=["rag"])
def ingest(payload: IngestRequest) -> IngestResponse:
    """Ingest a markdown document from raw content (idempotent)."""
    try:
        result = ingest_markdown_content(
            file_name=payload.file_name,
            content=payload.content,
        )
        return IngestResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingestion failed for %s", payload.file_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc


@router.post("/ingest-file", response_model=IngestResponse, tags=["rag"])
def ingest_file(file_path: str) -> IngestResponse:
    """Ingest a markdown file from the server filesystem (admin/internal use)."""
    try:
        result = ingest_markdown_file(file_path)
        return IngestResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("File ingestion failed for %s", file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc


@router.delete("/documents/{doc_name}", response_model=DeleteResponse, tags=["rag"])
def delete_document(doc_name: str) -> DeleteResponse:
    """Delete all chunks belonging to a document by its filename.

    Example: DELETE /api/v1/documents/ATM_Services_FAQ.md
    """
    try:
        store = get_vector_store_service(
            persist_dir=settings.CHROMA_PERSIST_DIR,
            collection_name=settings.COLLECTION_NAME,
        )
        deleted_chunks = store.delete_document(doc_name)
        if deleted_chunks == 0:
            return DeleteResponse(
                doc_name=doc_name,
                success=False,
                deleted_chunks=0,
                message=f"No chunks found for document: {doc_name}",
            )
        return DeleteResponse(
            doc_name=doc_name,
            success=True,
            deleted_chunks=deleted_chunks,
            message=f"Deleted {deleted_chunks} chunks",
        )
    except Exception as exc:
        logger.exception("Delete failed for %s", doc_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Delete failed: {exc}",
        ) from exc
