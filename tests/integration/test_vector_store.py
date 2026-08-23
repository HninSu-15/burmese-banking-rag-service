from llama_index.core.schema import TextNode

from app.services.rag import vector_store as vector_store_module
from app.services.rag.vector_store import ChromaVectorStoreService
import app.services.rag.embedder_service as embedder_module  


class FakeEmbedder:
    def get_text_embedding(self, text):
        return [1.0, 0.0, 0.0] if "ATM" in text else [0.0, 1.0, 0.0]

    def get_query_embedding(self, query):
        return [1.0, 0.0, 0.0] if "ATM" in query else [0.0, 1.0, 0.0]


def test_vector_store_uses_temporary_persistence_and_retrieves_metadata(tmp_path, monkeypatch):
    fake_embedder = FakeEmbedder()
    monkeypatch.setattr(embedder_module, "get_embedder_service", lambda: fake_embedder)
    store = ChromaVectorStoreService(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_collection",
    )
    node = TextNode(
        text="ATM card replacement policy",
        id_="atm-card-1",
        embedding=fake_embedder.get_text_embedding("ATM card replacement policy"),
    )
    node.metadata = {"doc_id": "atm-policy", "section_title": "Card Replacement"}

    store.add_nodes([node])

    assert store.count_documents() == 1
    results = store.retrieve("ATM card", top_k=1, threshold=0.0)

    assert len(results) == 1
    result = results[0]
    assert result["chunk_id"] == "atm-card-1"
    assert result["metadata"]["doc_id"] == "atm-policy"
    assert result["metadata"]["section_title"] == "Card Replacement"
    assert result["relevance_score"] == 1.0


def test_vector_store_filters_results_by_metadata(tmp_path, monkeypatch):
    fake_embedder = FakeEmbedder()
    monkeypatch.setattr(embedder_module, "get_embedder_service", lambda: fake_embedder)
    store = ChromaVectorStoreService(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="filter_collection",
    )
    nodes = []
    for chunk_id, document_id in [("atm-1", "atm-policy"), ("branch-1", "branch-policy")]:
        node = TextNode(
            text=f"{document_id} information",
            id_=chunk_id,
            embedding=fake_embedder.get_text_embedding(document_id),
        )
        node.metadata = {"doc_id": document_id}
        nodes.append(node)

    store.add_nodes(nodes)

    results = store.retrieve(
        "ATM question",
        top_k=2,
        threshold=0.0,
        filter_metadata={"doc_id": "branch-policy"},
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "branch-1"