"""Lexical retrieval over the chunks stored in ChromaDB."""

import re
import math
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from app.services.rag.normalizer import BurmeseTextNormalizer
from core.config import settings


class SemanticReranker:
	"""Rerank candidates with cosine similarity from the configured embedder."""

	def __init__(self, embedder: Any, threshold: float = settings.RERANKER_THRESHOLD) -> None:
		if not 0 <= threshold <= 1:
			raise ValueError("reranker threshold must be between 0 and 1")
		self.embedder = embedder
		self.threshold = threshold

	@staticmethod
	def _cosine(left: List[float], right: List[float]) -> float:
		dot = sum(a * b for a, b in zip(left, right))
		left_norm = math.sqrt(sum(value * value for value in left))
		right_norm = math.sqrt(sum(value * value for value in right))
		if not left_norm or not right_norm:
			return 0.0
		return dot / (left_norm * right_norm)

	def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		if not candidates:
			return []
		query_vector = self.embedder.get_query_embedding(query)
		ranked = []
		for candidate in candidates:
			text_vector = self.embedder.get_text_embedding(candidate.get("text", ""))
			cosine_score = self._cosine(query_vector, text_vector)
			rerank_score = (cosine_score + 1.0) / 2.0
			if rerank_score >= self.threshold:
				ranked.append({**candidate, "rerank_score": rerank_score})
		return sorted(ranked, key=lambda result: result["rerank_score"], reverse=True)


class BurmeseBM25Retriever:
	"""Retrieve Chroma chunks using Burmese-friendly character n-grams."""

	def __init__(self, collection: Any, ngram_size: int = 2) -> None:
		if ngram_size < 1:
			raise ValueError("ngram_size must be at least 1")

		self.collection = collection
		self.ngram_size = ngram_size
		self._records: List[Dict[str, Any]] = []
		self._index: BM25Okapi | None = None
		self._collection_signature: tuple[str, ...] = ()
		self.refresh()

	def refresh(self) -> None:
		"""Rebuild the index from the current Chroma collection contents."""
		stored = self.collection.get(include=["documents", "metadatas"])
		documents = stored.get("documents") or []
		ids = stored.get("ids") or []
		metadatas = stored.get("metadatas") or []

		self._records = []
		tokenized_documents = []
		for index, document in enumerate(documents):
			text = (document or "").removeprefix("passage: ")
			metadata = metadatas[index] or {}
			question = metadata.get("question", "")
			searchable_text = " ".join([question] * 3 + [text]) if question else text
			tokenized_documents.append(self._tokenize(searchable_text))
			self._records.append({
				"chunk_id": ids[index],
				"text": text,
				"metadata": metadata,
			})

		self._index = BM25Okapi(tokenized_documents) if tokenized_documents else None
		self._collection_signature = tuple(sorted(ids))

	def _ensure_fresh(self) -> None:
		"""Refresh the index only when the Chroma collection contents change."""
		stored = self.collection.get()
		current_signature = tuple(sorted(stored.get("ids") or []))
		if current_signature != self._collection_signature:
			self.refresh()

	def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
		"""Return the highest-scoring chunks for a normalized query."""
		if not query or not query.strip() or top_k <= 0:
			return []

		self._ensure_fresh()
		if self._index is None:
			return []

		query_tokens = self._tokenize(self.expand_query(query))
		scores = self._index.get_scores(query_tokens)
		query_token_set = set(query_tokens)
		adjusted_scores = []
		for index, score in enumerate(scores):
			question = self._records[index]["metadata"].get("question", "")
			question_tokens = set(self._tokenize(question))
			question_overlap = (
				len(query_token_set & question_tokens) / len(query_token_set)
				if query_token_set
				else 0.0
			)
			adjusted_scores.append(float(score) + question_overlap * 10.0)
		ranked_indexes = sorted(
			range(len(adjusted_scores)),
			key=lambda index: adjusted_scores[index],
			reverse=True,
		)[:top_k]

		return [
			{
				**self._records[index],
				"bm25_score": adjusted_scores[index],
			}
			for index in ranked_indexes
		]

	@staticmethod
	def expand_query(query: str) -> str:
		"""Add common Burmese wording variants used in the knowledge base."""
		variants = {
			"ကတ်သစ်": "ကတ်အသစ်",
			"အထောက်အထား": "စာရွက်စာတမ်း",
			"ယူလာ": "ယူဆောင်လာ",
			"လက်ခံယူ": "ထုတ်ယူ",
			"ရဖို့": "ကတ်အသစ် ထုတ်ယူရာတွင်",
			"လိုအပ်လဲ": "လိုအပ်သနည်း",
		}
		expanded = query
		for source, replacement in variants.items():
			if source in query:
				expanded = f"{expanded} {replacement}"
		if "ကတ်" in query and any(
			term in query for term in ("NRC", "Passport", "အထောက်အထား", "စာရွက်စာတမ်း")
		):
			expanded = (
				f"{expanded} ကတ်အသစ် ထုတ်ယူရာတွင် မည်သည့် စာရွက်စာတမ်းများ "
				"ယူဆောင်လာရမည်နည်း"
			)
		return expanded

	def _tokenize(self, text: str) -> List[str]:
		cleaned = BurmeseTextNormalizer.clean(text).lower()
		compact = re.sub(r"\s+", "", cleaned)
		if len(compact) <= self.ngram_size:
			return [compact] if compact else []
		return [
			compact[index:index + self.ngram_size]
			for index in range(len(compact) - self.ngram_size + 1)
		]


class HybridRetriever:
	"""Fuse Burmese BM25 and Chroma vector rankings with RRF."""

	BANKING_TERMS = (
		"ဘဏ်", "ဘဏ်ခွဲ", "ကတ်", "ကဒ်", "ATM", "CRM", "PIN", "NRC",
		"အကောင့်", "စာရင်း", "ချေးငွေ", "အတိုး", "အပ်ငွေ", "အထောက်အထား",
		"စာရွက်စာတမ်း", "လွှဲ", "ငွေ", "ထုတ်ယူ", "အပ်နှံ", "Mobile Banking",
		"Fixed Deposit", "Passport", "KPay", "WavePay", "ဝန်ဆောင်ခ",
	)

	def __init__(
		self,
		vector_store: Any,
		candidate_k: int = 10,
		rrf_k: int = 60,
		bm25_weight: float = 10.0,
		bm25_min_score_ratio: float = 0.8,
		semantic_domain_threshold: float = settings.SEMANTIC_DOMAIN_THRESHOLD,
		reranker: Any = None,
	) -> None:
		if (
			candidate_k <= 0
			or rrf_k <= 0
			or bm25_weight <= 0
			or not 0 < bm25_min_score_ratio <= 1
			or not 0 <= semantic_domain_threshold <= 1
		):
			raise ValueError("invalid hybrid retrieval configuration")

		self.vector_store = vector_store
		self.candidate_k = candidate_k
		self.rrf_k = rrf_k
		self.bm25_weight = bm25_weight
		self.bm25_min_score_ratio = bm25_min_score_ratio
		self.semantic_domain_threshold = semantic_domain_threshold
		self.reranker = reranker
		self.bm25 = BurmeseBM25Retriever(vector_store._chroma_collection)

	def refresh(self) -> None:
		"""Refresh lexical data after documents are ingested or deleted."""
		self.bm25.refresh()

	@classmethod
	def is_banking_query(cls, query: str) -> bool:
		"""Return whether the query contains a known banking-domain signal."""
		cleaned = BurmeseTextNormalizer.clean(query).casefold()
		return any(term.casefold() in cleaned for term in cls.BANKING_TERMS)

	def passes_semantic_domain_gate(self, query: str) -> bool:
		"""Accept keyword-free queries only when dense search finds bank knowledge."""
		results = self.vector_store.retrieve(query=query, top_k=1, threshold=0.0)
		if not results:
			return False
		score = results[0].get("relevance_score")
		return score is not None and score >= self.semantic_domain_threshold

	def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
		"""Return final results ranked by Reciprocal Rank Fusion."""
		if not query or not query.strip() or top_k <= 0:
			return []
		if not self.is_banking_query(query) and not self.passes_semantic_domain_gate(query):
			return []

		candidate_k = max(self.candidate_k, top_k)
		expanded_query = BurmeseBM25Retriever.expand_query(query)
		vector_results = self.vector_store.retrieve(
			query=expanded_query,
			top_k=candidate_k,
			threshold=0.0,
		)
		bm25_results = self.bm25.retrieve(query, top_k=candidate_k)
		best_bm25_score = bm25_results[0]["bm25_score"] if bm25_results else 0.0
		if best_bm25_score > 0:
			bm25_results = [
				result
				for result in bm25_results
				if result["bm25_score"] >= best_bm25_score * self.bm25_min_score_ratio
			]

		fused: Dict[str, Dict[str, Any]] = {}
		for source, results in (("vector", vector_results), ("bm25", bm25_results)):
			for rank, result in enumerate(results, start=1):
				chunk_id = result["chunk_id"]
				entry = fused.setdefault(chunk_id, {**result})
				weight = self.bm25_weight if source == "bm25" else 1.0
				entry["rrf_score"] = entry.get("rrf_score", 0.0) + weight / (self.rrf_k + rank)
				entry.setdefault("retrieval_sources", []).append(source)
				entry[f"{source}_rank"] = rank

		ranked = sorted(
			fused.values(),
			key=lambda result: result["rrf_score"],
			reverse=True,
		)
		if not ranked:
			return []
		if self.reranker is not None:
			ranked = self.reranker.rerank(query, ranked)
			if not ranked:
				return []
			for result in ranked:
				result["final_score"] = result["rrf_score"] + 0.05 * result["rerank_score"]
			ranked.sort(key=lambda result: result["final_score"], reverse=True)

		best = ranked[0]
		if not best.get("bm25_rank") and not best.get("vector_rank"):
			return []
		return ranked[:top_k]
