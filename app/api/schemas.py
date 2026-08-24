# app/api/schemas.py
"""
Pydantic request/response models for the RAG service API.
These mirror the LLMContextBuilder JSON contract consumed by the LLM team.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    """Payload for the /api/v1/retrieve endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User query in Burmese or English.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of contexts to return (1-10).",
    )

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()


class IngestRequest(BaseModel):
    """Payload for the /api/v1/ingest endpoint (markdown content)."""

    file_name: str = Field(
        ...,
        description="Markdown filename, e.g. ATM_Services_FAQ.md",
    )
    content: str = Field(
        ...,
        description="Full UTF-8 markdown content to ingest.",
    )

    @field_validator("file_name")
    @classmethod
    def _validate_markdown_name(cls, value: str) -> str:
        if not value.endswith(".md"):
            raise ValueError("only .md files are supported")
        if "/" in value or "\\" in value or ".." in value:
            raise ValueError("file_name must be a plain filename, not a path")
        return value


# ---------------------------------------------------------------------------
# Response Models (Mirror LLMContextBuilder output)
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    chunk_id: str
    source: str
    section: str


class ContextSource(BaseModel):
    doc_name: str
    section: str
    page_number: int


class ContextItem(BaseModel):
    rank: int
    chunk_id: str
    question: str
    text: str
    source: ContextSource
    retrieval_score: float


class Instructions(BaseModel):
    answer_only_from_context: bool = True
    answer_language: Literal["my", "en"]
    include_citations: bool = True
    return_json_only: bool = True
    do_not_invent_information: bool = True


class RetrieveResponse(BaseModel):
    """Exact JSON contract delivered to the LLM/generation team."""

    query: str
    language: Literal["my", "en"]
    has_context: bool
    confidence: Literal["high", "low"]
    contexts: List[ContextItem] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    answer: str = (
        "တောင်းပန်ပါတယ်။ သက်ဆိုင်ရာ ဘဏ်ဝန်ဆောင်မှုအချက်အလက်ကို မတွေ့ရှိပါ။"
    )
    instructions: Optional[Instructions] = None


class HealthResponse(BaseModel):
    service: str
    version: str
    status: Literal["healthy", "not_initialized", "degraded"]
    initialized: bool
    embedder: bool
    vector_store: bool
    parser: bool
    documents: int


class IngestResponse(BaseModel):
    file_name: str
    success: bool
    chunks: int
    document_count: int
    deleted_existing: int
    message: Optional[str] = None


class DeleteResponse(BaseModel):
    doc_name: str
    success: bool
    deleted_chunks: int
    message: Optional[str] = None
