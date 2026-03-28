"""Hybrid retrieval combining keyword and semantic search."""

from __future__ import annotations

import logging
from typing import Any

from src.retrieval.keyword import keyword_search
from src.retrieval.semantic import semantic_search


LOGGER = logging.getLogger(__name__)


def _normalize_scores(results: list[dict[str, Any]], score_key: str) -> dict[str, float]:
    """Normalize retrieval scores into the 0..1 range by chunk."""
    if not results:
        return {}

    raw_scores = [float(result[score_key]) for result in results]
    max_score = max(raw_scores)
    if max_score <= 0:
        return {str(result["chunk_id"]): 0.0 for result in results}

    return {
        str(result["chunk_id"]): float(result[score_key]) / max_score
        for result in results
    }


def hybrid_search(
    query: str,
    chunks: list[dict[str, Any]],
    model: Any,
    faiss_index: Any,
    metadata: list[dict[str, Any]],
    top_k: int = 5,
    keyword_weight: float = 0.4,
    semantic_weight: float = 0.6,
) -> list[dict[str, Any]]:
    """Combine keyword and semantic retrieval into a single ranked result list."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if keyword_weight < 0 or semantic_weight < 0:
        raise ValueError("Search weights must be non-negative.")

    cleaned_query = query.strip()
    if not cleaned_query:
        LOGGER.info("hybrid_search_completed", extra={"query": query, "match_count": 0})
        return []

    keyword_results = keyword_search(cleaned_query, chunks, top_k=top_k)
    semantic_results = semantic_search(cleaned_query, model, faiss_index, metadata, top_k=top_k)

    keyword_scores = _normalize_scores(keyword_results, "score")
    semantic_scores = _normalize_scores(semantic_results, "score")

    keyword_by_chunk = {str(result["chunk_id"]): result for result in keyword_results}
    semantic_by_chunk = {str(result["chunk_id"]): result for result in semantic_results}
    all_chunk_ids = set(keyword_by_chunk) | set(semantic_by_chunk)

    combined_results: list[dict[str, Any]] = []
    for chunk_id in all_chunk_ids:
        keyword_result = keyword_by_chunk.get(chunk_id)
        semantic_result = semantic_by_chunk.get(chunk_id)
        keyword_score = keyword_scores.get(chunk_id, 0.0)
        semantic_score = semantic_scores.get(chunk_id, 0.0)
        combined_score = (keyword_score * keyword_weight) + (semantic_score * semantic_weight)

        combined_results.append(
            {
                "chunk_id": chunk_id,
                "keyword_score": keyword_score,
                "semantic_score": semantic_score,
                "combined_score": combined_score,
                "matched_terms": list(keyword_result.get("matched_terms", [])) if keyword_result else [],
                "matched_sentences": list(semantic_result.get("matched_sentences", [])) if semantic_result else [],
                "snippet": (
                    str(semantic_result.get("snippet", "")).strip()
                    if semantic_result and str(semantic_result.get("snippet", "")).strip()
                    else str(keyword_result.get("snippet", "")).strip() if keyword_result else ""
                ),
            }
        )

    ranked_results = sorted(
        combined_results,
        key=lambda item: (-float(item["combined_score"]), -float(item["semantic_score"]), -float(item["keyword_score"]), str(item["chunk_id"])),
    )
    final_results = ranked_results[:top_k]

    LOGGER.info(
        "hybrid_search_completed",
        extra={"query": query, "match_count": len(final_results), "top_k": top_k},
    )
    return final_results
