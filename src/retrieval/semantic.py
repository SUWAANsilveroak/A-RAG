"""Semantic retrieval over sentence embeddings stored in FAISS."""

from __future__ import annotations

import logging
from typing import Any


LOGGER = logging.getLogger(__name__)


def _get_numpy_module() -> Any:
    """Import NumPy lazily for FAISS query vector preparation."""
    try:
        import numpy as np  # type: ignore
    except ImportError as error:
        raise RuntimeError("Semantic search requires the optional 'numpy' package.") from error
    return np


def _prepare_query_vector(model: Any, query: str) -> Any:
    """Encode a query and convert it into a float32 matrix for FAISS."""
    np = _get_numpy_module()

    raw_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=False,
        show_progress_bar=False,
    )
    if not raw_embedding:
        raise ValueError("Query embedding generation returned no vectors.")

    query_matrix = np.asarray(raw_embedding, dtype=np.float32)
    if query_matrix.ndim != 2:
        raise ValueError("Query embedding must be a 2D vector matrix.")
    return query_matrix


def _aggregate_hits(
    scores: list[float],
    indices: list[int],
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate sentence-level hits into chunk-level semantic results."""
    chunk_scores: dict[str, float] = {}
    chunk_sentences: dict[str, list[str]] = {}
    chunk_best_sentence: dict[str, tuple[float, str]] = {}

    for score, index in zip(scores, indices, strict=True):
        if index < 0 or index >= len(metadata):
            continue

        record = metadata[index]
        chunk_id = str(record["chunk_id"])
        sentence_text = str(record["text"])

        chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + float(score)

        sentences = chunk_sentences.setdefault(chunk_id, [])
        if sentence_text not in sentences:
            sentences.append(sentence_text)

        best_score, _ = chunk_best_sentence.get(chunk_id, (float("-inf"), ""))
        if float(score) > best_score:
            chunk_best_sentence[chunk_id] = (float(score), sentence_text)

    aggregated_results: list[dict[str, Any]] = []
    for chunk_id, score in chunk_scores.items():
        aggregated_results.append(
            {
                "chunk_id": chunk_id,
                "score": score,
                "matched_sentences": chunk_sentences[chunk_id],
                "snippet": chunk_best_sentence[chunk_id][1],
            }
        )

    return aggregated_results


def semantic_search(
    query: str,
    model: Any,
    faiss_index: Any,
    metadata: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search semantically similar sentences, then aggregate them by chunk."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    cleaned_query = query.strip()
    if not cleaned_query:
        LOGGER.info("semantic_search_completed", extra={"query": query, "match_count": 0})
        return []

    if not metadata:
        LOGGER.info("semantic_search_completed", extra={"query": query, "match_count": 0})
        return []

    if getattr(faiss_index, "ntotal", 0) <= 0:
        LOGGER.info("semantic_search_completed", extra={"query": query, "match_count": 0})
        return []

    try:
        query_vector = _prepare_query_vector(model, cleaned_query)
    except Exception as error:
        LOGGER.warning("semantic_search_query_embedding_failed", extra={"query": query, "error": str(error)})
        return []

    search_k = min(top_k, len(metadata), int(getattr(faiss_index, "ntotal", 0)))
    scores_matrix, indices_matrix = faiss_index.search(query_vector, search_k)
    scores = [float(score) for score in scores_matrix[0]]
    indices = [int(index) for index in indices_matrix[0]]

    aggregated_results = _aggregate_hits(scores, indices, metadata)
    ranked_results = sorted(aggregated_results, key=lambda item: (-float(item["score"]), str(item["chunk_id"])))
    final_results = ranked_results[:top_k]

    LOGGER.info(
        "semantic_search_completed",
        extra={"query": query, "match_count": len(final_results), "top_k": top_k},
    )
    return final_results
