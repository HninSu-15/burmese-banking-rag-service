# tests/test_search.py
"""
Test Module 5: Search/Retrieval from ChromaDB
"""

import sys
from pathlib import Path

# Set up project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
from app.services.rag.vector_store import get_vector_store_service
from app.services.rag.embedder_service import get_embedder_service


def test_single_search():
    """Test a single search query."""
    
    print("=" * 60)
    print(" 🔍 TESTING SEARCH/RETRIEVAL")
    print("=" * 60)
    
    # 1. Initialize services
    print("\n📦 Initializing services...")
    embedder = get_embedder_service()
    store = get_vector_store_service()
    
    # 2. Check documents count
    doc_count = store.count_documents()
    print(f"📊 Total documents in Vector Store: {doc_count}")
    
    if doc_count == 0:
        print("❌ No documents found. Please run ingest_all.py first.")
        return
    
    # 3. Define test queries
    test_queries = [
        "ATM ကတ်ပျောက်ရင် ဘာလုပ်ရမလဲ",
        "ကတ်အသစ်ထုတ်ရင် ဘယ်လောက်ကျမလဲ",
        "တစ်နေ့ဘယ်လောက်ထုတ်လို့ရလဲ",
        "ATM စက်ထဲကတ်ညပ်ရင်",
        "PIN နံပါတ်မေ့ရင် ဘာလုပ်ရမလဲ"
    ]
    
    # 4. Search for each query
    print("\n" + "=" * 60)
    print(" 🔎 SEARCH RESULTS")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n📌 Query: {query}")
        print("-" * 50)
        
        try:
            results = store.retrieve(query, top_k=3)
            
            if not results:
                print("   ❌ No results found")
                continue
            
            # Display results
            for i, result in enumerate(results, 1):
                score = result.get('relevance_score', 0)
                metadata = result.get('metadata', {})
                text = result.get('text', '')
                
                print(f"\n   [{i}] Score: {score:.4f}")
                print(f"       Document: {metadata.get('doc_name', 'Unknown')}")
                print(f"       Section: {metadata.get('section_title', 'Unknown')[:60]}...")
                print(f"       Header Path: {metadata.get('header_path', 'N/A')[:60]}...")
                print(f"       Preview: {text[:150]}...")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print(" ✅ SEARCH TEST COMPLETE!")
    print("=" * 60)


def test_compare_queries():
    """Compare results between similar and different queries."""
    
    print("\n" + "=" * 60)
    print(" 📊 COMPARING QUERY RESULTS")
    print("=" * 60)
    
    store = get_vector_store_service()
    
    # Similar queries (same topic)
    similar_queries = [
        "ATM ကတ်ပျောက်ရင်",
        "ကတ်ပျောက်သွားရင် ဘာလုပ်ရမလဲ",
        "ATM card lost"
    ]
    
    # Different queries
    different_queries = [
        "PIN နံပါတ်",
        "ငွေထုတ်ယူမှု",
        "ဘဏ်ခွဲ"
    ]
    
    # Get first result title for each query
    def get_top_result(query):
        results = store.retrieve(query, top_k=1)
        if results:
            return results[0]['metadata'].get('section_title', 'Unknown')
        return None
    
    print("\n🔍 Similar Queries (Should return similar sections):")
    print("-" * 50)
    for q in similar_queries:
        title = get_top_result(q)
        print(f"   '{q}' → {title}")
    
    print("\n🔍 Different Queries (Should return different sections):")
    print("-" * 50)
    for q in different_queries:
        title = get_top_result(q)
        print(f"   '{q}' → {title}")
    
    print("\n" + "=" * 60)
    print(" ✅ COMPARISON COMPLETE!")
    print("=" * 60)


def test_metadata_filtering():
    """Test search with metadata filtering."""
    
    print("\n" + "=" * 60)
    print(" 🏷️ TESTING METADATA FILTERING")
    print("=" * 60)
    
    store = get_vector_store_service()
    
    query = "ATM ကတ်"
    
    # Filter by doc_id
    print(f"\n🔍 Query: '{query}'")
    print("-" * 50)
    
    # Without filter
    print("\n📌 Without filter:")
    results = store.retrieve(query, top_k=2)
    for i, r in enumerate(results, 1):
        print(f"   [{i}] {r['metadata'].get('doc_name', 'Unknown')} - Score: {r.get('relevance_score', 0):.4f}")
    
    # With filter - by doc_id
    print("\n📌 With filter (doc_id='doc_ATM_Services_FAQ'):")
    filter_metadata = {"doc_id": "doc_ATM_Services_FAQ"}
    results = store.retrieve(query, top_k=2, filter_metadata=filter_metadata)
    for i, r in enumerate(results, 1):
        print(f"   [{i}] {r['metadata'].get('doc_name', 'Unknown')} - Score: {r.get('relevance_score', 0):.4f}")
    
    # With filter - by section level
    print("\n📌 With filter (section_level=3):")
    filter_metadata = {"section_level": 3}
    results = store.retrieve(query, top_k=3, filter_metadata=filter_metadata)
    for i, r in enumerate(results, 1):
        print(f"   [{i}] {r['metadata'].get('section_title', 'Unknown')[:40]}... - Score: {r.get('relevance_score', 0):.4f}")
    
    print("\n" + "=" * 60)
    print(" ✅ METADATA FILTERING COMPLETE!")
    print("=" * 60)


def test_performance():
    """Test search performance."""
    
    print("\n" + "=" * 60)
    print(" ⏱️ TESTING SEARCH PERFORMANCE")
    print("=" * 60)
    
    import time
    
    store = get_vector_store_service()
    
    queries = [
        "ATM ကတ်",
        "ငွေထုတ်ယူ",
        "ဘဏ်ခွဲ",
        "ကတ်အသစ်"
    ]
    
    times = []
    
    print(f"\n📊 Running {len(queries)} queries...")
    print("-" * 50)
    
    for query in queries:
        start = time.time()
        results = store.retrieve(query, top_k=3)
        elapsed = time.time() - start
        
        times.append(elapsed)
        print(f"   '{query[:20]}...' → {elapsed:.3f}s ({len(results)} results)")
    
    avg_time = sum(times) / len(times)
    print(f"\n📈 Average search time: {avg_time:.3f}s")
    print(f"   Total documents: {store.count_documents()}")
    
    if avg_time < 1.0:
        print("   ✅ Performance is good (under 1 second)")
    elif avg_time < 3.0:
        print("   ⚠️ Performance is acceptable (under 3 seconds)")
    else:
        print("   ❌ Performance is slow (over 3 seconds)")
    
    print("\n" + "=" * 60)
    print(" ✅ PERFORMANCE TEST COMPLETE!")
    print("=" * 60)


def test_embedder_services():
    """Test embedder service for query processing."""
    
    print("\n" + "=" * 60)
    print(" 🧠 TESTING EMBEDDER SERVICE")
    print("=" * 60)
    
    embedder = get_embedder_service()
    
    test_texts = [
        "ATM ကတ်ပျောက်ရင် ဘဏ်ကိုဆက်ပါ",
        "ကတ်ပျောက်ရင် ဘဏ်ကိုဖုန်းဆက်ပါ",
        "ဒီနေ့ ရာသီဥတု ကောင်းတယ်"
    ]
    
    print("\n📊 Testing Query Embeddings:")
    print("-" * 50)
    
    import numpy as np
    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    embeddings = []
    for text in test_texts:
        vector = embedder.get_query_embedding(text)
        embeddings.append(vector)
        print(f"   '{text[:30]}...' → Vector length: {len(vector)}")
    
    print("\n📊 Similarity between texts:")
    print("-" * 50)
    
    # Similar texts
    sim = cosine_similarity(embeddings[0], embeddings[1])
    print(f"   Text 1 & 2 (similar): {sim:.4f}")
    
    # Different texts
    sim = cosine_similarity(embeddings[0], embeddings[2])
    print(f"   Text 1 & 3 (different): {sim:.4f}")
    
    print("\n" + "=" * 60)
    print(" ✅ EMBEDDER SERVICE TEST COMPLETE!")
    print("=" * 60)


def interactive_search():
    """Interactive search mode."""
    
    print("\n" + "=" * 60)
    print(" 💬 INTERACTIVE SEARCH MODE")
    print("=" * 60)
    print("Type your query (or 'quit' to exit)")
    print("-" * 50)
    
    store = get_vector_store_service()
    embedder = get_embedder_service()
    
    while True:
        query = input("\n🔍 Query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not query:
            continue
        
        results = store.retrieve(query, top_k=3)
        
        print(f"\n📊 Results for: '{query}'")
        print("-" * 50)
        
        if not results:
            print("   ❌ No results found")
            continue
        
        for i, result in enumerate(results, 1):
            score = result.get('relevance_score', 0)
            metadata = result.get('metadata', {})
            text = result.get('text', '')
            
            print(f"\n   [{i}] Score: {score:.4f}")
            print(f"       Document: {metadata.get('doc_name', 'Unknown')}")
            print(f"       Section: {metadata.get('section_title', 'Unknown')[:60]}...")
            print(f"       Preview: {text[:200]}...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Search/Retrieval")
    parser.add_argument("--query", type=str, help="Single query to search")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results")
    parser.add_argument("--interactive", action="store_true", help="Interactive search mode")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if args.query:
        # Single query search
        store = get_vector_store_service()
        results = store.retrieve(args.query, args.top_k)
        
        print(f"\n🔍 Results for: '{args.query}'")
        print("=" * 50)
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] Score: {r.get('relevance_score', 0):.4f}")
            print(f"    Document: {r['metadata'].get('doc_name', 'Unknown')}")
            print(f"    Section: {r['metadata'].get('section_title', 'Unknown')[:60]}...")
            print(f"    Preview: {r['text'][:200]}...")
    
    elif args.interactive:
        interactive_search()
    
    elif args.all:
        test_single_search()
        test_compare_queries()
        test_metadata_filtering()
        test_performance()
        test_embedder_services()
    
    else:
        # Default: run single search test
        test_single_search()
        test_compare_queries()
        test_metadata_filtering()
        test_performance()
        test_embedder_services()