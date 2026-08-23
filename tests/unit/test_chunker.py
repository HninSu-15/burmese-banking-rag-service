

from app.services.rag.chunker import SectionDocumentChunker
from app.services.rag.hybrid_retriever import BurmeseBM25Retriever
from app.services.rag.ingestion_parser import ExtractedPage, ParsedDocument


def make_document(text: str) -> ParsedDocument:
    return ParsedDocument(
        doc_id="test_doc_001",
        doc_name="test.md",
        file_path="/path/to/test.md",
        file_type="markdown",
        total_pages=1,
        pages=[ExtractedPage(page_number=1, raw_text=text)],
    )


def test_chunker_preserves_sections_and_metadata():
    document = make_document(
        """
## Introduction to RAG
This is the introduction section explaining RAG architecture.

### Key Components
The main components include ingestion, retrieval, and generation.

### Benefits
RAG provides accurate and contextual responses.

## Implementation Details
Here are the implementation specifics for production.
"""
    )

    chunks = SectionDocumentChunker(min_chunk_size=30).chunk_document(document)

    assert len(chunks) == 4
    assert [chunk["chunk_id"] for chunk in chunks] == [
        "test_doc_001_sec_0",
        "test_doc_001_sec_1",
        "test_doc_001_sec_2",
        "test_doc_001_sec_3",
    ]

    first_metadata = chunks[0]["metadata"]
    assert first_metadata["doc_id"] == "test_doc_001"
    assert first_metadata["doc_name"] == "test.md"
    assert first_metadata["section_title"] == "Introduction to RAG"
    assert first_metadata["section_level"] == 2
    assert first_metadata["parent_section"] == "Introduction to RAG"
    assert first_metadata["header_path"] == "Introduction to RAG"
    assert first_metadata["total_chunks_in_doc"] == len(chunks)

    key_components = chunks[1]
    assert key_components["section_title"] == "Key Components"
    assert key_components["parent_section"] == "Introduction to RAG"
    assert key_components["header_path"] == "Introduction to RAG > Key Components"
    assert "## Introduction to RAG" in key_components["text"]
    assert "### Key Components" in key_components["text"]


def test_chunker_excludes_header_only_sections():
    document = make_document(
        """
# Main Topic
## Empty Section
## Populated Section
This section contains enough useful content.
"""
    )

    chunks = SectionDocumentChunker(min_chunk_size=10).chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0]["section_title"] == "Populated Section"
    assert any(line.strip() and not line.strip().startswith("#") for line in chunks[0]["text"].splitlines())


def test_chunker_processes_all_pages_and_assigns_global_indexes():
    document = ParsedDocument(
        doc_id="multi_page_doc",
        doc_name="multi_page.md",
        file_path="/path/to/multi_page.md",
        file_type="markdown",
        total_pages=2,
        pages=[
            ExtractedPage(page_number=1, raw_text="# Page One\nContent from page one."),
            ExtractedPage(page_number=2, raw_text="# Page Two\nContent from page two."),
        ],
    )

    chunks = SectionDocumentChunker(min_chunk_size=10).chunk_document(document)

    assert len(chunks) == 2
    assert [chunk["chunk_id"] for chunk in chunks] == [
        "multi_page_doc_sec_0",
        "multi_page_doc_sec_1",
    ]
    assert [chunk["metadata"]["page_number"] for chunk in chunks] == [1, 2]
    assert [chunk["metadata"]["chunk_index"] for chunk in chunks] == [0, 1]
    assert all(chunk["metadata"]["total_chunks_in_doc"] == 2 for chunk in chunks)


def test_chunker_keeps_required_documents_faq_question_and_answer_together():
    question = "### Q: ကတ်အသစ် ထုတ်ယူရာတွင် မည်သည့် စာရွက်စာတမ်းများ ယူဆောင်လာရမည်နည်း။"
    answer = """ဘဏ်ခွဲတွင် ကတ်အသစ် ထုတ်ယူရာတွင် အောက်ပါ စာရွက်စာတမ်းများ ယူဆောင်လာရပါမည်-
1. နိုင်ငံသား စိစစ်ရေး ကတ်ပြား (NRC) မူရင်း။
2. Passport မူရင်း နှင့် Stay Permit။
3. တရားဝင် ကိုယ်စားလှယ်လွှဲစာ (Power of Attorney) နှင့် ကိုယ်စားလှယ်၏ NRC မူရင်း။"""
    document = make_document(f"# Card Policy\n## Required Documents\n{question}\n{answer}\n\n### Q: နောက်မေးခွန်း\nနောက်အဖြေ။")

    chunks = SectionDocumentChunker(min_chunk_size=10).chunk_document(document)

    target_chunks = [chunk for chunk in chunks if chunk["section_title"] == question[4:]]
    assert len(target_chunks) == 1
    target_text = target_chunks[0]["text"]
    assert question in target_text
    assert "NRC" in target_text
    assert "Passport" in target_text
    assert "Stay Permit" in target_text
    assert "Power of Attorney" in target_text
    assert target_chunks[0]["metadata"]["question"] == question.removeprefix("### Q: ")


class FakeCollection:
    def __init__(self, documents):
        self.documents = documents

    def get(self, include=None):
        return {
            "ids": [document["id"] for document in self.documents],
            "documents": [document["text"] for document in self.documents],
            "metadatas": [document["metadata"] for document in self.documents],
        }


def test_burmese_bm25_ranks_required_documents_faq_first():
    collection = FakeCollection([
        {
            "id": "fee",
            "text": "passage: ကတ်အသစ် အစားထိုး ဝန်ဆောင်ခ ကျပ် ၅,၀၀၀။",
            "metadata": {"doc_name": "card_replacement_policy.md"},
        },
        {
            "id": "documents",
            "text": "passage: ကတ်အသစ် ထုတ်ယူရာတွင် စာရွက်စာတမ်းများ ယူဆောင်လာရပါမည်။ NRC မူရင်း၊ Passport နှင့် Stay Permit၊ Power of Attorney။",
            "metadata": {"doc_name": "card_replacement_policy.md"},
        },
    ])

    results = BurmeseBM25Retriever(collection).retrieve(
        "ကတ်အသစ် ထုတ်ယူရာတွင် မည်သည့် စာရွက်စာတမ်းများ ယူဆောင်လာရမည်နည်း",
        top_k=2,
    )

    assert results[0]["chunk_id"] == "documents"
    assert results[0]["bm25_score"] > results[1]["bm25_score"]