# app/services/rag/ingestion_service.py
"""Programmatic ingestion logic used by the /api/v1/ingest endpoint."""

import logging
from pathlib import Path
from typing import Dict, Any

from llama_index.core.schema import TextNode

from app.services.rag.factory import RAGServiceFactory
from app.services.rag.chunker import SectionDocumentChunker
from app.services.rag.vector_store import get_vector_store_service
from core.config import settings, PROJECT_ROOT

logger = logging.getLogger(__name__)


def ingest_markdown_content(file_name: str, content: str) -> Dict[str, Any]:
    """Parse, chunk, embed, and store markdown content.

    The content is written to the project ``knowledge/`` directory so the
    existing parser/chunker pipeline can process it unchanged.

    Returns a dict with ``file_name``, ``success``, ``chunks``,
    ``document_count`` and ``deleted_existing`` keys.
    """
    if not file_name.endswith(".md"):
        raise ValueError("only .md files are supported")

    knowledge_dir = PROJECT_ROOT / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    target_path = knowledge_dir / file_name
    target_path.write_text(content, encoding="utf-8")

    return ingest_markdown_file(str(target_path))


def ingest_markdown_file(file_path: str) -> Dict[str, Any]:
    """Run the full ingestion pipeline for a single markdown file on disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix != ".md":
        raise ValueError(f"Only .md files supported: {path}")

    parser = RAGServiceFactory.get_parser()
    embedder = RAGServiceFactory.get_embedder()
    store = get_vector_store_service(
        persist_dir=settings.CHROMA_PERSIST_DIR,
        collection_name=settings.COLLECTION_NAME,
    )
    chunker = SectionDocumentChunker()

    doc = parser.parse_markdown(str(path))

    # Idempotency: delete existing chunks for this document first.
    deleted_existing = store.delete_document(doc.doc_name)

    chunks = chunker.chunk_document(doc)
    nodes = []
    for chunk in chunks:
        text = chunk.get("text") or chunk.get("raw_text") or ""
        if not text:
            continue
        vector = embedder.get_text_embedding(text)
        node = TextNode(text=f"passage: {text}", id_=chunk["chunk_id"], embedding=vector)
        if "metadata" in chunk:
            node.metadata = chunk["metadata"]
        nodes.append(node)

    store.add_nodes(nodes)

    result = {
        "file_name": path.name,
        "success": True,
        "chunks": len(nodes),
        "document_count": store.count_documents(),
        "deleted_existing": deleted_existing,
    }
    logger.info(
        "Ingested %s: %d chunks (deleted %d existing), total docs %d",
        path.name,
        len(nodes),
        deleted_existing,
        result["document_count"],
    )
    return result