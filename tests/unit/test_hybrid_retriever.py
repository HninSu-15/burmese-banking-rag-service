import pytest

from app.services.rag.hybrid_retriever import BurmeseBM25Retriever, HybridRetriever, SemanticReranker
from app.services.rag.context_builder import LLMContextBuilder


class FakeCollection:
    def __init__(self, documents=None):
        self.documents = documents or [
            {
                "id": "fee",
                "text": "passage: ကတ်အသစ် အစားထိုး ဝန်ဆောင်ခ ကျပ် ၅,၀၀၀။",
                "metadata": {"section_title": "Fees"},
            },
            {
                "id": "documents",
                "text": "passage: ကတ်အသစ် ထုတ်ယူရာတွင် စာရွက်စာတမ်းများ ယူဆောင်လာရပါမည်။ NRC မူရင်း၊ Passport နှင့် Stay Permit၊ Power of Attorney။",
                "metadata": {"section_title": "Documents"},
            },
        ]

    def get(self, include=None):
        return {
            "ids": [document["id"] for document in self.documents],
            "documents": [document["text"] for document in self.documents],
            "metadatas": [document["metadata"] for document in self.documents],
        }


def test_bm25_refreshes_when_collection_chunks_change():
    collection = FakeCollection([
        {
            "id": "fee",
            "text": "passage: ကတ်အစားထိုး ဝန်ဆောင်ခ။",
            "metadata": {"section_title": "Fees"},
        },
    ])
    retriever = BurmeseBM25Retriever(collection)

    collection.documents.append({
        "id": "documents",
        "text": "passage: Passport နှင့် NRC စာရွက်စာတမ်းများ လိုအပ်ပါသည်။",
        "metadata": {"section_title": "Documents"},
    })

    results = retriever.retrieve("Passport စာရွက်စာတမ်း", top_k=2)

    assert "documents" in {result["chunk_id"] for result in results}


class FakeVectorStore:
    _chroma_collection = FakeCollection()

    def retrieve(self, query, top_k, threshold):
        return [
            {
                "chunk_id": "fee",
                "text": "fee",
                "metadata": {"section_title": "Fees"},
                "relevance_score": 0.9,
            },
            {
                "chunk_id": "documents",
                "text": "documents",
                "metadata": {"section_title": "Documents"},
                "relevance_score": 0.8,
            },
        ][:top_k]


def test_hybrid_retriever_fuses_bm25_and_vector_ranks():
    retriever = HybridRetriever(FakeVectorStore(), candidate_k=2)

    results = retriever.retrieve(
        "ကတ်အသစ် ထုတ်ယူရာတွင် မည်သည့် စာရွက်စာတမ်းများ ယူဆောင်လာရမည်နည်း",
        top_k=2,
    )

    assert results[0]["chunk_id"] == "documents"
    assert results[0]["retrieval_sources"] == ["vector", "bm25"]
    assert results[0]["rrf_score"] > results[1]["rrf_score"]


@pytest.mark.parametrize(
    "query",
    [
        "ကတ်အသစ် ထုတ်ယူဖို့ ဘာစာရွက်စာတမ်းတွေ ယူလာရမလဲ",
        "ကတ်အသစ်ရဖို့ NRC မူရင်းနဲ့ ဘာအထောက်အထားတွေ လိုအပ်လဲ",
        "ကတ်သစ် လက်ခံယူတဲ့အခါ Passport နဲ့ ကိုယ်စားလှယ်လွှဲစာ လိုအပ်ပါသလား",
    ],
)
def test_hybrid_retriever_ranks_documents_faq_first_for_paraphrases(query):
    retriever = HybridRetriever(FakeVectorStore(), candidate_k=2)

    results = retriever.retrieve(query, top_k=2)

    assert results[0]["chunk_id"] == "documents"


def test_query_expansion_adds_burmese_faq_variants():
    expanded = BurmeseBM25Retriever.expand_query(
        "ကတ်သစ်ရဖို့ အထောက်အထားတွေ ယူလာရမလဲ"
    )

    assert "ကတ်အသစ်" in expanded
    assert "စာရွက်စာတမ်း" in expanded
    assert "ယူဆောင်လာ" in expanded
    assert "ကတ်အသစ် ထုတ်ယူရာတွင်" in expanded
    assert "မည်သည့် စာရွက်စာတမ်းများ" in expanded


def test_hybrid_retriever_rejects_non_banking_query(monkeypatch):

    retriever = HybridRetriever(FakeVectorStore(), candidate_k=2)
    monkeypatch.setattr(retriever, "passes_semantic_domain_gate", lambda query: False)

    assert retriever.retrieve("အပြင်သွားရအောငါ", top_k=3) == []


def test_banking_domain_detector_accepts_burmese_bank_query():

    assert HybridRetriever.is_banking_query("ATM ကတ်ပျောက်ရင် ဘာလုပ်ရမလဲ")


class FakeEmbedder:
    def get_query_embedding(self, text):
        return [1.0, 0.0] if "documents" in text or "Passport" in text else [0.0, 1.0]

    def get_text_embedding(self, text):
        return [1.0, 0.0] if "documents" in text or "Passport" in text else [0.0, 1.0]


def test_semantic_reranker_orders_candidates_by_embedding_similarity():
    reranker = SemanticReranker(FakeEmbedder(), threshold=0.5)
    results = reranker.rerank("Passport documents", [
        {"chunk_id": "fee", "text": "fee", "metadata": {}},
        {"chunk_id": "documents", "text": "Passport documents", "metadata": {}},
    ])

    assert results[0]["chunk_id"] == "documents"
    assert results[0]["rerank_score"] == 1.0


def test_hybrid_retriever_blends_reranker_with_rrf():
    retriever = HybridRetriever(
        FakeVectorStore(),
        candidate_k=2,
        reranker=SemanticReranker(FakeEmbedder(), threshold=0.5),
    )

    results = retriever.retrieve(
        "ကတ်အသစ် ထုတ်ယူရာတွင် စာရွက်စာတမ်း Passport",
        top_k=2,
    )

    assert results[0]["chunk_id"] == "documents"
    assert "final_score" in results[0]


def test_context_builder_sends_only_strong_primary_context():
    payload = LLMContextBuilder().build("ATM card question", [
        {
            "chunk_id": "primary",
            "text": "passage: Primary answer",
            "metadata": {"question": "Primary question", "doc_name": "atm.md", "section_title": "Primary", "page_number": 2},
            "final_score": 0.9,
        },
        {
            "chunk_id": "unrelated",
            "text": "passage: Unrelated answer",
            "metadata": {"question": "Other question", "doc_name": "loan.md"},
            "final_score": 0.2,
        },
    ])

    assert len(payload["contexts"]) == 1
    assert payload["contexts"][0]["rank"] == 1
    assert payload["contexts"][0]["text"] == "Primary answer"
    assert payload["contexts"][0]["source"]["page_number"] == 2
    assert payload["has_context"] is True
    assert payload["confidence"] == "high"
    assert payload["citations"][0]["chunk_id"] == "primary"


def test_context_builder_keeps_close_complementary_contexts():
    results = [
        {"chunk_id": "one", "text": "one", "metadata": {}, "rrf_score": 0.9},
        {"chunk_id": "two", "text": "two", "metadata": {}, "rrf_score": 0.8},
    ]

    assert len(LLMContextBuilder().build("bank question", results)["contexts"]) == 2


def test_context_builder_returns_structured_unavailable_response():
    payload = LLMContextBuilder().build("အပြင်သွားရအောငါ", [])

    assert payload["has_context"] is False
    assert payload["confidence"] == "low"
    assert payload["contexts"] == []
    assert payload["citations"] == []
    assert payload["answer"]