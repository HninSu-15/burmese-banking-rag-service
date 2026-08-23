# scripts/finalize_eval_dataset.py
import json
from pathlib import Path

def create_final_dataset():
    input_path = Path(__file__).parent.parent / "evaluation" / "base_chunks.json"
    
    with open(input_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    final_dataset = []
    for chunk in chunks:
        for query in chunk.get('test_queries', []):
            if query.strip():
                final_dataset.append({
                    "query": query.strip(),
                    "ground_truth_chunk_ids": [chunk['chunk_id']]  
                })
    
    output_path = Path(__file__).parent.parent / "evaluation" / "eval_dataset.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in final_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✅ Final evaluation dataset created: {output_path}")
    print(f"📊 Total test cases: {len(final_dataset)}")

if __name__ == "__main__":
    create_final_dataset()