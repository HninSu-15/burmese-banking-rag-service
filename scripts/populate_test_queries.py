# scripts/populate_test_queries.py
"""
Auto-populate test_queries for evaluation dataset
"""

import json
import re
from pathlib import Path

def generate_queries(section_title: str) -> list:
    """Generate natural Burmese paraphrase queries from section title."""
    
    # "Q: " ကိုဖယ်ပါ
    title = section_title.replace("Q: ", "").strip()
    
    # ဗလာဖြစ်နေရင် ဘာမှမပြန်ပါနဲ့
    if not title or title == "":
        return []
    
    queries = []
    
    # 1. မူရင်းမေးခွန်း
    queries.append(title)
    
    # 2. အတိုချုံးပုံစံ (ဘာလုပ်ရမလဲ → ဘယ်လိုလုပ်မလဲ)
    short = title.replace("ဘာလုပ်ရမလဲ", "ဘယ်လိုလုပ်ရမလဲ")
    if short != title:
        queries.append(short)
    
    # 3. ပြောဆိုပုံအမျိုးမျိုး
    if "ပျောက်ဆုံး" in title:
        queries.append(title.replace("ပျောက်ဆုံးသွားပါက", "ပျောက်သွားရင်"))
        queries.append(title.replace("ပျောက်ဆုံး", "ပျောက်"))
    
    if "ထုတ်ယူ" in title:
        queries.append(title.replace("ထုတ်ယူ", "ထုတ်ဖို့"))
        queries.append(title.replace("ထုတ်ယူနိုင်", "ထုတ်လို့ရ"))
    
    if "ကတ်အသစ်" in title:
        queries.append(title.replace("ကတ်အသစ်", "ကတ်သစ်"))
    
    if "စာရွက်စာတမ်း" in title:
        queries.append(title.replace("စာရွက်စာတမ်း", "အထောက်အထား"))
        queries.append(title.replace("စာရွက်စာတမ်းများ", "စာရွက်တွေ"))
    
    # 4. မေးခွန်းအဆုံးသတ် ထည့်ခြင်း
    if not title.endswith("?") and not title.endswith("။"):
        queries.append(title + " လား")
    
    # 5. Unique ဖြစ်အောင်လုပ်ပြီး ပြန်ပါ
    unique_queries = list(dict.fromkeys(queries))
    
    # ရလဒ် ၅ ခုထက်ပိုရင် ပထမ ၅ ခုပဲယူပါ
    return unique_queries[:5]

def main():
    base_path = Path(__file__).parent.parent / "evaluation" / "base_chunks.json"
    
    if not base_path.exists():
        print(f"❌ File not found: {base_path}")
        return
    
    with open(base_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    updated = False
    for chunk in data:
        # test_queries ဗလာဖြစ်နေရင် ဖြည့်ပါ
        if not chunk.get('test_queries') or len(chunk['test_queries']) == 0:
            section_title = chunk.get('section_title', '')
            if section_title and "Q:" in section_title:
                chunk['test_queries'] = generate_queries(section_title)
                updated = True
                print(f"✅ Added {len(chunk['test_queries'])} queries for: {section_title[:40]}...")
            else:
                print(f"⚠️ Skipping (no valid Q:): {chunk.get('chunk_id', 'N/A')}")
    
    if updated:
        with open(base_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Updated {base_path} with test queries.")
    else:
        print("✅ All chunks already have test queries.")

if __name__ == "__main__":
    main()