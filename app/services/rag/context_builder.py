"""Build a filtered, stable context contract for the answer-generation service."""

import re
from typing import Any, Dict, List

from app.services.rag.normalizer import BurmeseTextNormalizer


class LLMContextBuilder:
    """Convert ranked retrieval results into the LLM input JSON contract."""

    def __init__(self, max_contexts: int = 3, min_score_ratio: float = 0.85) -> None:
        if max_contexts < 1 or not 0 < min_score_ratio <= 1:
            raise ValueError("invalid context builder configuration")
        self.max_contexts = max_contexts
        self.min_score_ratio = min_score_ratio

    def build(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return filtered contexts and instructions for the LLM team member."""
        selected = self._select_contexts(results)
        return {
            "query": query,
            "language": self._detect_language(query),
            "has_context": bool(selected),
            "confidence": "high" if selected else "low",
            "contexts": [self._format_context(rank, result) for rank, result in enumerate(selected, 1)],
            "citations": [
                {
                    "chunk_id": result.get("chunk_id", ""),
                    "source": (result.get("metadata") or {}).get("doc_name", "Unknown"),
                    "section": (result.get("metadata") or {}).get("section_title", "Unknown"),
                }
                for result in selected
            ],
            "answer": "" if selected else "တောင်းပန်ပါတယ်။ သက်ဆိုင်ရာ ဘဏ်ဝန်ဆောင်မှုအချက်အလက်ကို မတွေ့ရှိပါ။",
            "instructions": {
                "answer_only_from_context": True,
                "answer_language": self._detect_language(query),
                "include_citations": True,
                "return_json_only": True,
                "do_not_invent_information": True,
            },
        }

    def _select_contexts(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return []

        best_score = self._score(results[0])
        if best_score <= 0:
            return results[:1]

        return [
            result
            for result in results[: self.max_contexts]
            if self._score(result) >= best_score * self.min_score_ratio
        ]

    @staticmethod
    def _score(result: Dict[str, Any]) -> float:
        return float(
            result.get("final_score", result.get("rerank_score", result.get("rrf_score", 0.0)))
        )

    @staticmethod
    def _format_context(rank: int, result: Dict[str, Any]) -> Dict[str, Any]:
        metadata = result.get("metadata") or {}
        question = metadata.get("question") or metadata.get("section_title", "")
        question = re.sub(r"^Q\s*:\s*", "", question).strip()
        return {
            "rank": rank,
            "chunk_id": result.get("chunk_id", ""),
            "question": question,
            "text": (result.get("text") or "").removeprefix("passage: "),
            "source": {
                "doc_name": metadata.get("doc_name", "Unknown"),
                "section": metadata.get("section_title", "Unknown"),
                "page_number": metadata.get("page_number", 1),
            },
            "retrieval_score": LLMContextBuilder._score(result),
        }

    @staticmethod
    def _detect_language(query: str) -> str:
        cleaned = BurmeseTextNormalizer.clean(query)
        burmese_characters = sum("\u1000" <= character <= "\u109f" for character in cleaned)
        return "my" if burmese_characters else "en"