# Python ထဲမှာ

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # scripts/ → project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from app.services.rag.vector_store import get_vector_store_service

store = get_vector_store_service()
store.delete_document("doc_ATM_Services_FAQ")  # Document ID