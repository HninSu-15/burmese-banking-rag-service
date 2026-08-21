# tests/test_chunker.py
"""
Test Module 4: Header-Aware Chunking with Real Documents
"""

import sys
from pathlib import Path

# Set up project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
from app.services.rag.chunker import SectionDocumentChunker
from app.services.rag.ingestion_parser import ParsedDocument, ExtractedPage


def test_chunker_with_sample():
    """Quick test with sample data."""
    
    print("=" * 60)
    print(" 🧪 QUICK TEST: CHUNKER WITH SAMPLE DATA")
    print("=" * 60)

    sample_text = """
## Introduction to RAG
This is the introduction section explaining RAG architecture.

### Key Components
The main components include ingestion, retrieval, and generation.

### Benefits
RAG provides accurate and contextual responses.

## Implementation Details
Here are the implementation specifics for production.
"""

    page = ExtractedPage(
        page_number=1,
        raw_text=sample_text,
        metadata={"pii_masked": True}
    )

    doc = ParsedDocument(
        doc_id="test_doc_001",
        doc_name="test.md",
        file_path="/path/to/test.md",
        file_type="markdown",
        total_pages=1,
        pages=[page]
    )

    chunker = SectionDocumentChunker(
        min_chunk_size=30,
        max_chunk_size=1500,
        preserve_header_hierarchy=True,
        include_header_path=True
    )

    chunks = chunker.chunk_document(doc)

    print(f"\n✅ Generated {len(chunks)} chunks")
    
    # Display chunks
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get('metadata', {})
        print(f"\n   ─── Chunk {i} ───")
        print(f"   📌 CHUNK ID     : {chunk.get('chunk_id', 'N/A')}")
        print(f"   📝 TEXT PREVIEW : {chunk.get('text', '')[:80]}...")
        print(f"   📊 METADATA:")
        print(f"      ├─ doc_id            : {metadata.get('doc_id', 'N/A')}")
        print(f"      ├─ doc_name          : {metadata.get('doc_name', 'N/A')}")
        print(f"      ├─ section_title     : {metadata.get('section_title', 'Untitled')}")
        print(f"      ├─ section_level     : {metadata.get('section_level', 0)}")
        print(f"      ├─ parent_section    : {metadata.get('parent_section', 'None')}")
        print(f"      ├─ header_path       : {metadata.get('header_path', 'N/A')}")
        print(f"      └─ total_chunks      : {metadata.get('total_chunks_in_doc', 0)}")

    print("\n" + "=" * 60)
    print(" ✅ SAMPLE TEST COMPLETE!")
    print("=" * 60)


def inspect_real_md_chunking(target_filename: str = "ATM_Services_FAQ.md"):
    """Test chunker with real document from knowledge folder."""
    
    print("=" * 75)
    print(f" 🧪 INSPECTING CHUNK RESULTS: knowledge/{target_filename}")
    print("=" * 75)

    knowledge_dir = os.path.join(PROJECT_ROOT, "knowledge")
    target_file_path = os.path.join(knowledge_dir, target_filename)

    # 1. Verify file existence
    if not os.path.exists(target_file_path):
        print(f"❌ Error: File '{target_filename}' not found in {knowledge_dir}")
        return

    # 2. Read raw file contents
    with open(target_file_path, "r", encoding="utf-8") as f:
        file_content = f.read()

    # 3. Construct ExtractedPage Object (Pydantic Model)
    extracted_page = ExtractedPage(
        page_number=1,
        raw_text=file_content,
        metadata={"pii_masked": True}
    )

    doc_id = f"doc_{Path(target_filename).stem}"
    parsed_doc = ParsedDocument(
        doc_id=doc_id,
        doc_name=target_filename,
        file_path=target_file_path,
        file_type=".md",
        total_pages=1,
        pages=[extracted_page]
    )

    # 4. Initialize Chunker
    chunker = SectionDocumentChunker(
        min_chunk_size=30,
        max_chunk_size=1500,
        preserve_header_hierarchy=True,
        include_header_path=True
    )

    # 5. Execute Chunking
    chunks = chunker.chunk_document(parsed_doc)

    print(f"\n📊 TOTAL CHUNKS GENERATED FOR '{target_filename}': {len(chunks)}\n")

    # 6. Check for header-only chunks (warn if found)
    header_only_chunks = []
    for idx, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        # Check if chunk contains only headers (no content after headers)
        lines = text.splitlines()
        content_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        if len(content_lines) == 0:
            header_only_chunks.append(idx + 1)

    if header_only_chunks:
        print(f"⚠️  WARNING: Found header-only chunks: {header_only_chunks}")
        print("   These chunks have no content and should be skipped.\n")
    else:
        print("✅ All chunks have content (no header-only chunks found).\n")

    # 7. Display Chunks and Metadata Payloads
    for idx, chunk in enumerate(chunks):
        print("─" * 75)
        print(f"📌 CHUNK #{idx + 1} | ID: {chunk['chunk_id']}")
        print(f" ├─ Title         : {chunk.get('section_title', 'Untitled')}")
        print(f" ├─ Level         : H{chunk.get('section_level', 0)}")
        print(f" ├─ Parent Section: {chunk.get('parent_section', 'None')}")
        print(f" ├─ Header Path   : {chunk.get('header_path', 'N/A')}")
        print(f" ├─ Text Length   : {len(chunk.get('text', ''))} characters")
        
        # Show content preview
        text_preview = chunk.get('text', '')
        if text_preview:
            preview_lines = text_preview.splitlines()[:5]
            print(" └─ Chunk Raw Text Preview:")
            print("    ---------------------------------------------------------")
            for line in preview_lines:
                print(f"    | {line[:80]}")
            if len(text_preview.splitlines()) > 5:
                print(f"    | ... and {len(text_preview.splitlines()) - 5} more lines")
            print("    ---------------------------------------------------------")
        
        print(" 🏷️ CHROMA METADATA PAYLOAD:")
        print(json.dumps(chunk.get('metadata', {}), indent=6, ensure_ascii=False))
        print()

    # 8. Summary Statistics
    print("=" * 75)
    print(" 📊 CHUNKING SUMMARY")
    print("=" * 75)
    print(f"   Total Chunks      : {len(chunks)}")
    
    if chunks:
        chunk_sizes = [len(chunk.get('text', '')) for chunk in chunks]
        print(f"   Min Size          : {min(chunk_sizes)} chars")
        print(f"   Max Size          : {max(chunk_sizes)} chars")
        print(f"   Avg Size          : {sum(chunk_sizes) / len(chunk_sizes):.0f} chars")
        
        # Count chunks by level
        levels = {}
        for chunk in chunks:
            level = chunk.get('section_level', 0)
            levels[level] = levels.get(level, 0) + 1
        print(f"\n   📊 Chunks by Level:")
        for level, count in sorted(levels.items()):
            print(f"      H{level}: {count} chunks")

    print("=" * 75)
    print(f" ✅ CHUNK INSPECTION COMPLETED FOR {target_filename}")
    print("=" * 75)


def test_all_documents():
    """Test chunker with all documents in knowledge folder."""
    
    print("=" * 75)
    print(" 🧪 TESTING ALL DOCUMENTS IN KNOWLEDGE FOLDER")
    print("=" * 75)

    knowledge_dir = os.path.join(PROJECT_ROOT, "knowledge")
    
    if not os.path.exists(knowledge_dir):
        print(f"⚠️ Knowledge directory not found: {knowledge_dir}")
        return

    md_files = [f for f in os.listdir(knowledge_dir) if f.endswith('.md')]
    
    if not md_files:
        print(f"⚠️ No .md files found in {knowledge_dir}")
        return

    print(f"\n📄 Found {len(md_files)} .md files\n")

    for md_file in md_files:
        print("-" * 50)
        inspect_real_md_chunking(md_file)
        print()


if __name__ == "__main__":
    # Run sample test first
    test_chunker_with_sample()
    
    print("\n" + "=" * 75)
    print(" NOW TESTING WITH REAL DOCUMENTS...")
    print("=" * 75)
    
    # Then run real document tests
    target_file = "ATM_Services_FAQ.md"
    inspect_real_md_chunking(target_file)
    
    # Optional: Test all documents
    # test_all_documents()