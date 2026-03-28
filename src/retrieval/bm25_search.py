"""BM25 lexical retrieval for chunk ranking."""

from __future__ import annotations

import logging
import re
from typing import Any


LOGGER = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"\b[\w-]+\b", flags=re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _get_bm25_class() -> Any:
    """Import BM25 lazily so the rest of the package remains usable without it."""
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
    except ImportError as error:
        raise RuntimeError("BM25 search requires the optional 'rank-bm25' package.") from error
    return BM25Okapi


def _tokenize_text(text: str) -> list[str]:
    """Tokenize text with lowercase lexical tokens."""
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _build_snippet(text: str, query_tokens: list[str], max_length: int = 200) -> str:
    """Return the first relevant sentence when possible, else a short prefix."""
    if not text.strip():
        return ""

    sentences = [sentence.strip() for sentence in _SENTENCE_RE.split(text.strip()) if sentence.strip()]
    for sentence in sentences:
        lowered_sentence = sentence.lower()
        if any(token in lowered_sentence for token in query_tokens):
            if len(sentence) <= max_length:
                return sentence
            return sentence[: max_length - 3].rstrip() + "..."

    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3].rstrip() + "..."


def build_bm25_index(chunks: list[dict[str, Any]]) -> Any:
    """Build a BM25 index from chunk text."""
    BM25Okapi = _get_bm25_class()

    corpus_tokens: list[list[str]] = []
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        corpus_tokens.append(_tokenize_text(text))

    bm25_index = BM25Okapi(corpus_tokens)
    LOGGER.info("bm25_index_built", extra={"chunk_count": len(chunks)})
    return bm25_index


def bm25_search(
    query: str,
    bm25_index: Any,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Run BM25 search over chunks and return top-ranked lexical matches."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    query_tokens = _tokenize_text(query)
    if not query_tokens:
        LOGGER.info("bm25_search_completed", extra={"query": query, "match_count": 0})
        return []

    if not chunks:
        LOGGER.info("bm25_search_completed", extra={"query": query, "match_count": 0})
        return []

    scores = bm25_index.get_scores(query_tokens)
    results: list[dict[str, Any]] = []

    for chunk, score in zip(chunks, scores, strict=True):
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        chunk_text = str(chunk.get("text", "")).strip()
        if not chunk_id or not chunk_text:
            continue

        numeric_score = float(score)
        if numeric_score <= 0.0:
            continue

        results.append(
            {
                "chunk_id": chunk_id,
                "score": numeric_score,
                "snippet": _build_snippet(chunk_text, query_tokens),
            }
        )

    ranked_results = sorted(results, key=lambda item: (-float(item["score"]), str(item["chunk_id"])))
    final_results = ranked_results[:top_k]

    LOGGER.info(
        "bm25_search_completed",
        extra={"query": query, "match_count": len(final_results), "top_k": top_k},
    )
    return final_results
