import pytest

pytest.importorskip("llama_index.vector_stores.chroma")

from llama_index.core.schema import TextNode

from app.services.rag import vector_store as vector_store_module
from app.services.rag.vector_store import ChromaVectorStoreService
import app.services.rag.embedder_service as embedder_module

class FakeEmbedder:
    def get_query_embedding(self, query):
        return [1.0, 0.0, 0.0]


def test_retrieve_returns_expected_result_schema(tmp_path, monkeypatch):
    fake_embedder = FakeEmbedder()
    monkeypatch.setattr(embedder_module, "get_embedder_service", lambda: fake_embedder)
    store = ChromaVectorStoreService(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="search_collection",
    )
    node = TextNode(
        text="ATM card instructions",
        id_="search-chunk-1",
        embedding=[1.0, 0.0, 0.0],
    )
    node.metadata = {"doc_id": "atm-policy", "section_title": "ATM Services"}
    store.add_nodes([node])

    results = store.retrieve("ATM card", top_k=1, threshold=0.0)

    assert len(results) == 1
    result = results[0]
    assert {"relevance_score", "chunk_id", "metadata"}.issubset(result)
    assert isinstance(result["chunk_id"], str)
    assert isinstance(result["metadata"], dict)
    assert isinstance(result["relevance_score"], float)
    assert result["relevance_score"] >= 0.0