# scripts/evaluate_retrieval.py
"""
Evaluate retrieval performance using Recall@1 and MRR
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.rag.hybrid_retriever import HybridRetriever
from app.services.rag.vector_store import get_vector_store_service
from core.config import settings


def load_eval_dataset(file_path: Path):
    """Load evaluation dataset from JSONL file."""
    if not file_path.exists():
        print(f"❌ Evaluation dataset not found: {file_path}")
        print("   Please run scripts/finalize_eval_dataset.py first.")
        sys.exit(1)
    
    test_cases = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_cases.append(json.loads(line))
    return test_cases


def evaluate_retrieval():
    """Main evaluation function."""
    
    # 1. Load dataset
    dataset_path = PROJECT_ROOT / "evaluation" / "eval_dataset.jsonl"
    test_cases = load_eval_dataset(dataset_path)
    
    if not test_cases:
        print("❌ No test cases found in evaluation dataset.")
        return
    
    print(f"📊 Loaded {len(test_cases)} test cases")
    print("=" * 50)
    
    # 2. Initialize Vector Store and Hybrid Retriever
    print("🔧 Initializing Hybrid Retriever...")
    
    store = get_vector_store_service(
        persist_dir=settings.CHROMA_PERSIST_DIR,
        collection_name=settings.COLLECTION_NAME
    )
    
    retriever = HybridRetriever(vector_store=store)
    
    # 3. Evaluate each test case
    correct_at_1 = 0
    reciprocal_ranks = []
    results = []
    
    for i, test in enumerate(test_cases, 1):
        query = test.get('query', '')
        ground_truth_ids = test.get('ground_truth_chunk_ids', [])
        
        if not query or not ground_truth_ids:
            continue
        
        # ✅ FIX: Use `retrieve` instead of `search`
        retrieved = retriever.retrieve(query, top_k=5)
        retrieved_ids = [r.get('chunk_id', '') for r in retrieved]
        
        # Find rank of ground truth
        found_rank = None
        for rank, retrieved_id in enumerate(retrieved_ids, 1):
            if retrieved_id in ground_truth_ids:
                found_rank = rank
                break
        
        # Update metrics
        if found_rank == 1:
            correct_at_1 += 1
        if found_rank:
            reciprocal_ranks.append(1 / found_rank)
        
        results.append({
            'query': query,
            'ground_truth_ids': ground_truth_ids,
            'retrieved_ids': retrieved_ids,
            'found_rank': found_rank
        })
    
    # 4. Calculate metrics
    total = len(test_cases)
    recall_1 = correct_at_1 / total if total > 0 else 0
    mrr = sum(reciprocal_ranks) / total if total > 0 else 0
    
    # 5. Display results
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    print(f"✅ Total test cases: {total}")
    print(f"✅ Recall@1: {recall_1:.2%} ({correct_at_1}/{total})")
    print(f"✅ MRR: {mrr:.4f}")
    print("=" * 50)
    
    # 6. Show failures for debugging
    failures = [r for r in results if r['found_rank'] is None or r['found_rank'] > 1]
    if failures:
        print(f"\n⚠️ {len(failures)} test cases had rank > 1 or not found:")
        for failure in failures[:5]:
            print(f"  - Query: {failure['query'][:50]}...")
            print(f"    Ground truth: {failure['ground_truth_ids']}")
            print(f"    Retrieved: {failure['retrieved_ids'][:3]}")
            print()


if __name__ == "__main__":
    evaluate_retrieval()