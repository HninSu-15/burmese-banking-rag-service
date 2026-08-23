"""Inspect generated chunks for a Markdown knowledge document."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.rag.chunker import SectionDocumentChunker
from app.services.rag.ingestion_parser import KnowledgeDocumentParser
from core.config import settings


def inspect_chunks(file_path: str, verify_store: bool = False) -> None:
    """Parse and print every generated chunk without embedding or indexing."""
    document_path = Path(file_path)
    if not document_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    document = KnowledgeDocumentParser().parse_markdown(str(document_path))
    chunks = SectionDocumentChunker().chunk_document(document)

    print(f"Document: {document.doc_name}")
    print(f"Pages: {document.total_pages}")
    print(f"Chunks: {len(chunks)}")
    print("=" * 80)

    for index, chunk in enumerate(chunks, start=1):
        print(f"Chunk {index}: {chunk['chunk_id']}")
        print(f"Section: {chunk['section_title']}")
        print("Text:")
        print(chunk["text"])
        print("-" * 80)

    if verify_store:
        from app.services.rag.vector_store import get_vector_store_service

        store = get_vector_store_service(
            persist_dir=settings.CHROMA_PERSIST_DIR,
            collection_name=settings.COLLECTION_NAME,
        )
        stored = store._chroma_collection.get(
            where={"doc_name": document.doc_name}
        )
        stored_texts = {
            text.removeprefix("passage: ")
            for text in (stored.get("documents") or [])
        }
        missing = [
            chunk["chunk_id"]
            for chunk in chunks
            if chunk["text"] not in stored_texts
        ]

        print("Chroma verification:")
        print(f"Stored chunks for {document.doc_name}: {len(stored_texts)}")
        if missing:
            print(f"Missing chunk IDs: {', '.join(missing)}")
            raise SystemExit(1)
        print("All generated chunks exist in Chroma.")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != "--verify-store"):
        print("Usage: python scripts/inspect_chunks.py <path_to_md_file> [--verify-store]")
        sys.exit(1)

    inspect_chunks(sys.argv[1], verify_store=len(sys.argv) == 3)