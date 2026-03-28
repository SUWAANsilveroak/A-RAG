"""Cross-encoder reranking for retrieval results."""

from __future__ import annotations

import logging
from typing import Any


LOGGER = logging.getLogger(__name__)
_RERANK_MODELS: dict[str, Any] = {}
_FALLBACK_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _load_reranker_model(model_name: str) -> Any:
    """Load and cache the cross-encoder reranker model."""
    cached_model = _RERANK_MODELS.get(model_name)
    if cached_model is not None:
        return cached_model

    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except ImportError as error:
        raise RuntimeError("Reranking requires the optional 'sentence-transformers' package.") from error

    try:
        model = CrossEncoder(model_name)
    except Exception as primary_error:
        if model_name == _FALLBACK_RERANK_MODEL:
            raise RuntimeError(f"Failed to load reranker model: {model_name}") from primary_error

        LOGGER.warning(
            "reranker_model_load_failed",
            extra={"requested_model": model_name, "fallback_model": _FALLBACK_RERANK_MODEL},
        )
        try:
            model = CrossEncoder(_FALLBACK_RERANK_MODEL)
            model_name = _FALLBACK_RERANK_MODEL
        except Exception as fallback_error:
            raise RuntimeError(
                f"Failed to load reranker models: {model_name}, {_FALLBACK_RERANK_MODEL}"
            ) from fallback_error

    _RERANK_MODELS[model_name] = model
    return model


def _normalize_rerank_scores(scores: list[float]) -> list[float]:
    """Normalize reranker outputs into the 0..1 range."""
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 if max_score > 0 else 0.0 for _ in scores]

    scale = max_score - min_score
    return [(score - min_score) / scale for score in scores]


def rerank_results(
    query: str,
    retrieval_results: list[dict[str, Any]],
    model_name: str = "BAAI/bge-reranker-base",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Rerank retrieval results with a cross-encoder over query-snippet pairs."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    cleaned_query = query.strip()
    if not cleaned_query:
        LOGGER.info("rerank_completed", extra={"query": query, "match_count": 0})
        return []

    if not retrieval_results:
        LOGGER.info("rerank_completed", extra={"query": query, "match_count": 0})
        return []

    valid_results: list[dict[str, Any]] = []
    for result in retrieval_results:
        snippet = str(result.get("snippet", "")).strip()
        chunk_id = str(result.get("chunk_id", "")).strip()
        if not chunk_id or not snippet:
            continue
        valid_results.append(result)

    if not valid_results:
        LOGGER.info("rerank_completed", extra={"query": query, "match_count": 0})
        return []

    model = _load_reranker_model(model_name)
    pairs = [(cleaned_query, str(result["snippet"]).strip()) for result in valid_results]
    raw_scores = model.predict(pairs)
    rerank_scores = [float(score) for score in raw_scores]
    normalized_scores = _normalize_rerank_scores(rerank_scores)

    reranked_results: list[dict[str, Any]] = []
    for result, rerank_score in zip(valid_results, normalized_scores, strict=True):
        retrieval_score = float(result.get("combined_score", result.get("score", 0.0)))
        final_score = (retrieval_score * 0.4) + (rerank_score * 0.6)
        reranked_results.append(
            {
                "chunk_id": str(result["chunk_id"]),
                "snippet": str(result["snippet"]).strip(),
                "retrieval_score": retrieval_score,
                "rerank_score": rerank_score,
                "final_score": final_score,
            }
        )

    ranked_results = sorted(
        reranked_results,
        key=lambda item: (-float(item["final_score"]), -float(item["rerank_score"]), str(item["chunk_id"])),
    )
    final_results = ranked_results[:top_k]

    LOGGER.info(
        "rerank_completed",
        extra={"query": query, "match_count": len(final_results), "top_k": top_k},
    )
    return final_results
