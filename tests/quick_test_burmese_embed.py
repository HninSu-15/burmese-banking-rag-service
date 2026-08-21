# quick_test.py
import sys
from pathlib import Path

# Set up project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from app.services.rag.embedder_service import get_embedder_service
import numpy as np

embedder = get_embedder_service()

texts = [
    "ATM ကတ် ပျောက်ရင် ဘဏ်ကိုဆက်ပါ။",
    "ကတ်ပျောက်ရင် ဘဏ်ကိုဖုန်းဆက်ပါ။",
    "ဒီနေ့ ရာသီဥတု ကောင်းတယ်။"
]

vecs = [embedder.get_text_embedding(t) for t in texts]

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("တူတဲ့စာသား (၁၊၂):", cos_sim(vecs[0], vecs[1]))  # မြင့်သင့်တယ်
print("မတူတဲ့စာသား (၁၊၃):", cos_sim(vecs[0], vecs[2]))  # နိမ့်သင့်တယ်