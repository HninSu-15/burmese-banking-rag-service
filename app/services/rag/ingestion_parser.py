# app/services/rag/ingestion_parser.py
import os
import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# Imports from the RAG service package
from app.services.rag.normalizer import BurmeseTextNormalizer


# =====================================================================
# 1. PII Protection Layer
# =====================================================================

class BurmesePIIMasker:
    """
    Detects and redacts sensitive personal identifiable information (PII) 
    from Burmese and mixed English text before vector embedding.
    """

    def __init__(self):
        # Regex for Burmese NRC Numbers (e.g., 12/မဂဒ(နိုင်)123456 or 12/MAGADA(N)123456)
        self.nrc_pattern = re.compile(
            r'(?:[၁-၉1-9]{1,2}\s*/\s*[\u1000-\u102A a-zA-Z]+\s*\((?:နိုင်|ဧည့်|ပြု|N|P|A)\)\s*[၀-၉0-9]{6})',
            re.IGNORECASE
        )
        # Regex for 16-digit Credit/Debit Card Numbers
        self.card_pattern = re.compile(r'\b(?:\d[ -]*?){13,16}\b')
        
        # Regex for 10-digit Bank Account Numbers
        self.account_pattern = re.compile(r'\b\d{10,12}\b')
        
        # Regex for Phone Numbers (Burmese + English format)
        self.phone_pattern = re.compile(r'(?:\+?959|09)\s*[-]?\s*[0-9၀-၉]{7,9}')

    def mask(self, text: str) -> str:
        if not text:
            return ""
        
        # Mask sensitive banking details
        text = self.nrc_pattern.sub("[REDACTED_NRC]", text)
        text = self.card_pattern.sub("[REDACTED_CARD]", text)
        text = self.account_pattern.sub("[REDACTED_ACCOUNT]", text)
        text = self.phone_pattern.sub("[REDACTED_PHONE]", text)
        
        return text


# =====================================================================
# 2. Data Contracts (Pydantic Schemas)
# =====================================================================

class ExtractedPage(BaseModel):
    """Represents a single parsed document section/page."""
    page_number: int
    raw_text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Represents a fully parsed document with multiple pages."""
    doc_id: str
    doc_name: str
    file_path: str
    file_type: str
    total_pages: int
    pages: List[ExtractedPage]
    
    # ... other methods ...
    
    def get_all_text(self) -> str:
        """Get all text from all pages."""
        return "\n\n".join(p.raw_text for p in self.pages)


# =====================================================================
# 3. Flexible Document Parser Service
# =====================================================================

class KnowledgeDocumentParser:
    """
    Modular parser for the RAG Ingestion Layer.
    Handles text extraction, PII masking, and schema construction.
    Note: Normalization is now handled in embedder_service.py
    """

    def __init__(self, enable_pii_masking: bool = True):
        self.pii_masker = BurmesePIIMasker()
        self.enable_pii_masking = enable_pii_masking

    def _process_text(self, text: str) -> str:
        """Applies optional PII masking only. Normalization is handled in embedder."""
        if self.enable_pii_masking:
            text = self.pii_masker.mask(text)
        return text

    def parse_markdown(self, file_path: str) -> ParsedDocument:
        """Parses a Markdown file and redacts PII. Keeps raw text for chunking."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Markdown file not found at: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Process PII masking (preserve raw_text for chunking)
        processed_text = self._process_text(raw_text)
        doc_name = os.path.basename(file_path)
        
        # Create a clean, ChromaDB-compatible unique doc_id
        doc_id = re.sub(r'[^a-zA-Z0-9_]', '_', doc_name.lower())

        page = ExtractedPage(
            page_number=1,
            raw_text=raw_text,  # ← Keep original raw_text for chunking
            metadata={
                "doc_id": doc_id,
                "doc_name": doc_name,
                "page_number": 1,
                "source_type": "markdown",
                "pii_masked": self.enable_pii_masking,
                "processed_text": processed_text  # Store processed text in metadata
            }
        )

        return ParsedDocument(
            doc_id=doc_id,
            doc_name=doc_name,
            file_path=file_path,
            file_type="markdown",
            total_pages=1,
            pages=[page]
        )

    def parse_file(self, file_path: str) -> ParsedDocument:
        """Auto-detects file extension and routes to the appropriate parser."""
        if file_path.endswith(".md"):
            return self.parse_markdown(file_path)
        elif file_path.endswith(".pdf"):
            raise NotImplementedError("PDF ingestion is scheduled for Phase 2. Use .md for now.")
        else:
            raise ValueError(f"Unsupported file format: {file_path}. Only .md files are supported.")

    def parse_directory(self, dir_path: str) -> List[ParsedDocument]:
        """Scans a flat directory and ingests all supported knowledge files."""
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Knowledge directory not found at: {dir_path}")

        parsed_docs: List[ParsedDocument] = []
        for file in os.listdir(dir_path):
            if file.endswith(".md"):
                full_path = os.path.join(dir_path, file)
                parsed_docs.append(self.parse_markdown(full_path))

        return parsed_docs