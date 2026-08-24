"""Integration tests for the RAG service HTTP API (FastAPI layer)."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from app.main import app
import app.services.rag.retrieval_service as retrieval_module
import app.api.v1.routes as routes_module
import core.config as config_module


class FakeEmbedder:
    def get_query_embedding(self, query):
        return [1.0, 0.0, 0.0]


class FakeVectorStore:
    def __init__(self, documents=0, results=None):
        self._documents = documents
        self._results = results or []

    def count_documents(self):
        return self._documents

    def retrieve(self, query, top_k=3, threshold=0.0):
        return self._results[:top_k]


class FakeHybridRetriever:
    def __init__(self, **kwargs):
        self.store = kwargs.get("vector_store")

    def retrieve(self, query, top_k=3):
        return self.store.retrieve(query, top_k=top_k) if self.store else []


def _monkeypatch(monkeypatch, store):
    monkeypatch.setattr(retrieval_module, "get_vector_store_service", lambda **kw: store)
    monkeypatch.setattr(routes_module, "get_vector_store_service", lambda **kw: store)
    monkeypatch.setattr(retrieval_module, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(retrieval_module, "SemanticReranker", lambda *a, **k: None)
    monkeypatch.setattr(retrieval_module, "get_embedder_service", lambda: FakeEmbedder())
    monkeypatch.setattr(config_module.settings, "RAG_API_TOKEN", "")


def test_retrieve_empty_returns_refusal(monkeypatch):
    _monkeypatch(monkeypatch, FakeVectorStore(documents=0))
    response = TestClient(app).post("/api/v1/retrieve", json={"query": "something"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["has_context"] is False
    assert payload["confidence"] == "low"
    assert payload["contexts"] == []
    assert payload["answer"]


def test_retrieve_returns_contract(monkeypatch):
    results = [
        {
            "chunk_id": "atm_md_sec_0",
            "text": "ATM policy text",
            "metadata": {
                "doc_name": "ATM_Services_FAQ.md",
                "section_title": "ATM Card Loss",
                "page_number": 1,
            },
            "final_score": 0.8,
        }
    ]
    _monkeypatch(monkeypatch, FakeVectorStore(documents=5, results=results))
    response = TestClient(app).post(
        "/api/v1/retrieve", json={"query": "ATM card lost", "top_k": 3}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["has_context"] is True
    assert payload["language"] in ("my", "en")
    assert payload["confidence"] in ("high", "low")
    assert len(payload["contexts"]) == 1
    ctx = payload["contexts"][0]
    assert ctx["chunk_id"] == "atm_md_sec_0"
    assert ctx["source"]["doc_name"] == "ATM_Services_FAQ.md"
    assert payload["citations"][0]["source"] == "ATM_Services_FAQ.md"
    assert payload["instructions"]["answer_only_from_context"] is True


def test_retrieve_requires_token(monkeypatch):
    _monkeypatch(monkeypatch, FakeVectorStore(documents=0))
    monkeypatch.setattr(config_module.settings, "RAG_API_TOKEN", "secret-token")
    response = TestClient(app).post("/api/v1/retrieve", json={"query": "hello"})
    assert response.status_code == 401


def test_retrieve_validates_query(monkeypatch):
    _monkeypatch(monkeypatch, FakeVectorStore(documents=0))
    response = TestClient(app).post("/api/v1/retrieve", json={"query": ""})
    assert response.status_code == 422


def test_ingest_rejects_non_markdown(monkeypatch):
    _monkeypatch(monkeypatch, FakeVectorStore(documents=0))
    response = TestClient(app).post(
        "/api/v1/ingest", json={"file_name": "bad.txt", "content": "# x"}
    )
    assert response.status_code == 422


def test_delete_document(monkeypatch):
    store = FakeVectorStore(documents=5)
    store.delete_document = lambda doc_name: 3
    _monkeypatch(monkeypatch, store)
    response = TestClient(app).delete("/api/v1/documents/ATM_Services_FAQ.md")
    assert response.status_code == 200
    payload = response.json()
    assert payload["doc_name"] == "ATM_Services_FAQ.md"
    assert payload["success"] is True
    assert payload["deleted_chunks"] == 3


def test_delete_document_not_found(monkeypatch):
    store = FakeVectorStore(documents=5)
    store.delete_document = lambda doc_name: 0
    _monkeypatch(monkeypatch, store)
    response = TestClient(app).delete("/api/v1/documents/missing.md")
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is False
    assert payload["deleted_chunks"] == 0
