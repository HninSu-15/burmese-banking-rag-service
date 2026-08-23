import os

import pytest

pytest.importorskip("llama_cpp")
pytest.importorskip("chromadb")
pytest.importorskip("llama_index.vector_stores.chroma")

from app.services.rag.embedder_service import get_embedder_service


@pytest.mark.skipif(
    os.getenv("RUN_REAL_EMBEDDER_TESTS") != "1",
    reason="Set RUN_REAL_EMBEDDER_TESTS=1 to run the model-backed integration test",
)
def test_embedder_returns_valid_query_and_document_vectors():
    embedder = get_embedder_service()
    query_vector = embedder.get_query_embedding("ATM card replacement")
    document_vector = embedder.get_text_embedding("ATM card replacement policy")

    assert query_vector
    assert document_vector
    assert len(query_vector) == len(document_vector)
    assert all(isinstance(value, (int, float)) for value in query_vector)
    assert all(isinstance(value, (int, float)) for value in document_vector)