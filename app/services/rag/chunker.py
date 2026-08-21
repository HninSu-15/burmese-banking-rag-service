# app/services/rag/chunker.py
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from core.config import settings

# Import ParsedDocument
from app.services.rag.ingestion_parser import ParsedDocument

class ChunkMetadata(BaseModel):
    """
    Metadata Structure for each chunk to be stored in ChromaDB
    """
    doc_id: str
    doc_name: str
    file_path: str
    file_type: str = ".md"
    page_number: int = 1
    chunk_index: int
    section_title: str
    section_level: int
    parent_section: str
    header_path: str
    pii_masked: bool = True
    total_chunks_in_doc: int = 0


class SectionDocumentChunker:
    """
    Header-Aware & Hierarchy-Preserving Markdown Chunker
    Prepends parent headers context into sub-chunks to maximize Qwen3 embedding quality.
    """

    def __init__(
        self,
        min_chunk_size: int = 10,
        max_chunk_size: int = settings.CHUNK_SIZE,  
        preserve_header_hierarchy: bool = True,
        include_header_path: bool = True
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.preserve_header_hierarchy = preserve_header_hierarchy
        self.include_header_path = include_header_path
        
        # Matches #, ##, ###, #### or Q: / ### Q:
        self.header_pattern = re.compile(
            r'^(?P<level>#{1,4})\s+(?P<title>.+)$|^(?P<q_prefix>###\s+)?(?P<is_q>Q:|\bQ\b[:\s]).*$',
            re.MULTILINE
        )

    def chunk_document(self, doc: ParsedDocument) -> List[Dict[str, Any]]:
        """
        Main entry point: Chunk a ParsedDocument.
        
        Args:
            doc: ParsedDocument from ingestion_parser
            
        Returns:
            List of chunk dictionaries
        """
        if not doc or not doc.pages:
            return []
        
        # Get text from first page (or combine all pages)
        raw_text = doc.pages[0].raw_text if doc.pages else ""
        page_number = doc.pages[0].page_number if doc.pages else 1
        
        # Call internal chunking method
        return self._chunk_raw_text(
            raw_text=raw_text,
            doc_id=doc.doc_id,
            doc_name=doc.doc_name,
            file_path=doc.file_path,
            page_number=page_number
        )

    def _chunk_raw_text(
        self,
        raw_text: str,
        doc_id: str,
        doc_name: str,
        file_path: str,
        page_number: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Internal method that does the actual chunking.
        Creates chunks from headers with their content.
        """
        lines = raw_text.splitlines()
        
        # Structure to hold parsed sections
        sections = []
        current_header_stack = {}  # {level: title}
        current_lines = []
        current_title = doc_name
        current_level = 1

        for line in lines:
            header_info = self._extract_header_info(line)

            if header_info:
                # Save accumulated content before new header
                if current_lines:
                    text_block = "\n".join(current_lines).strip()
                    if text_block:
                        # Build parent section and header path from stack
                        parent_path = [
                            current_header_stack[lvl] 
                            for lvl in sorted(current_header_stack.keys()) 
                            if lvl < current_level
                        ]
                        
                        parent_section = parent_path[-1] if parent_path else current_title
                        full_header_path = " > ".join(
                            [current_header_stack[lvl] for lvl in sorted(current_header_stack.keys())]
                        ) if current_header_stack else current_title

                        # ============================================================
                        # FIX: Skip header-only chunks (no content)
                        # ============================================================
                        if len(text_block) >= self.min_chunk_size:
                            # Check if this is a header-only chunk
                            is_header_only = True
                            for line_check in text_block.splitlines():
                                line_stripped = line_check.strip()
                                if line_stripped and not line_stripped.startswith('#'):
                                    is_header_only = False
                                    break
                            
                            # Only add if not header-only
                            if not is_header_only:
                                sections.append({
                                    "title": current_title,
                                    "level": current_level,
                                    "parent_section": parent_section,
                                    "header_path": full_header_path,
                                    "context_headers": [
                                        (lvl, current_header_stack[lvl]) 
                                        for lvl in sorted(current_header_stack.keys()) 
                                        if lvl < current_level
                                    ],
                                    "body_text": text_block
                                })
                    
                    current_lines = []

                # Update hierarchy tracking
                current_level = header_info["level"]
                current_title = header_info["title"]
                
                # Clear lower/equal level headers from stack
                current_header_stack = {
                    lvl: t for lvl, t in current_header_stack.items() if lvl < current_level
                }
                current_header_stack[current_level] = current_title
                
                # Always add header line for context
                current_lines.append(line)
            
            else:
                # Non-header line - add to current section
                current_lines.append(line)

        # Flush final section
        if current_lines:
            text_block = "\n".join(current_lines).strip()
            if text_block and len(text_block) >= self.min_chunk_size:
                parent_path = [
                    current_header_stack[lvl] 
                    for lvl in sorted(current_header_stack.keys()) 
                    if lvl < current_level
                ]
                parent_section = parent_path[-1] if parent_path else current_title
                full_header_path = " > ".join(
                    [current_header_stack[lvl] for lvl in sorted(current_header_stack.keys())]
                ) if current_header_stack else current_title

                # ============================================================
                # FIX: Skip header-only chunks (no content)
                # ============================================================
                is_header_only = True
                for line_check in text_block.splitlines():
                    line_stripped = line_check.strip()
                    if line_stripped and not line_stripped.startswith('#'):
                        is_header_only = False
                        break
                
                if not is_header_only:
                    sections.append({
                        "title": current_title,
                        "level": current_level,
                        "parent_section": parent_section,
                        "header_path": full_header_path,
                        "context_headers": [
                            (lvl, current_header_stack[lvl]) 
                            for lvl in sorted(current_header_stack.keys()) 
                            if lvl < current_level
                        ],
                        "body_text": text_block
                    })

        # If no sections were created, create one from the whole text
        if not sections and raw_text.strip():
            sections.append({
                "title": doc_name,
                "level": 1,
                "parent_section": doc_name,
                "header_path": doc_name,
                "context_headers": [],
                "body_text": raw_text.strip()
            })

        # Construct final Chunk Payload with Prepend Headers Logic
        final_chunks = []
        total_chunks = len(sections)

        for index, sec in enumerate(sections):
            # Prepend Parent Headers to body_text if they exist
            prepended_prefix = ""
            if sec["context_headers"] and self.preserve_header_hierarchy:
                header_lines = [f"{'#' * lvl} {title}" for lvl, title in sec["context_headers"]]
                prepended_prefix = "\n".join(header_lines) + "\n\n"

            # Check if the section body already starts with its own header
            full_chunk_text = sec["body_text"]
            
            # Don't prepend if body already has the header
            if prepended_prefix and not full_chunk_text.startswith("#"):
                # Check if title is already in the body
                title_in_body = f"### {sec['title']}" in full_chunk_text or f"## {sec['title']}" in full_chunk_text
                if not title_in_body:
                    full_chunk_text = f"{prepended_prefix}### {sec['title']}\n{full_chunk_text}"
                else:
                    full_chunk_text = f"{prepended_prefix}{full_chunk_text}"
            elif prepended_prefix:
                full_chunk_text = f"{prepended_prefix}{full_chunk_text}"

            # Create Metadata Object
            meta = ChunkMetadata(
                doc_id=doc_id,
                doc_name=doc_name,
                file_path=file_path,
                file_type="." + doc_name.split(".")[-1] if "." in doc_name else ".md",
                page_number=page_number,
                chunk_index=index,
                section_title=sec["title"],
                section_level=sec["level"],
                parent_section=sec["parent_section"],
                header_path=sec["header_path"] if self.include_header_path else "",
                pii_masked=True,
                total_chunks_in_doc=total_chunks
            )

            chunk_payload = {
                "chunk_id": f"{doc_id}_sec_{index}",
                "text": full_chunk_text,
                "section_title": sec["title"],
                "section_level": sec["level"],
                "parent_section": sec["parent_section"],
                "header_path": sec["header_path"] if self.include_header_path else "",
                "metadata": meta.model_dump()
            }
            final_chunks.append(chunk_payload)

        return final_chunks

    def _extract_header_info(self, line: str) -> Optional[Dict[str, Any]]:
        """Extract header level and title from a line."""
        line_clean = line.strip()
        if not line_clean:
            return None

        # Check for standard markdown headers (#, ##, ###)
        md_match = re.match(r'^(#{1,4})\s+(.+)$', line_clean)
        if md_match:
            level = len(md_match.group(1))
            title = md_match.group(2).strip()
            return {"level": level, "title": title}

        # Check for Question pattern without markdown hash (e.g., Q: ...)
        if line_clean.startswith("Q:") or line_clean.startswith("Q :"):
            return {"level": 3, "title": line_clean}

        return None
    