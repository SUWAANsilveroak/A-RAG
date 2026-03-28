"""Exact lexical keyword search for chunk retrieval."""

from __future__ import annotations

import logging
import re
from typing import Any


LOGGER = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"\b[\w-]+\b", flags=re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "with",
}


def _normalize_query(query: str) -> list[str]:
    """Normalize a query into unique keyword tokens."""
    normalized_tokens = [token.lower() for token in _TOKEN_RE.findall(query)]
    deduplicated_tokens: list[str] = []
    seen_tokens: set[str] = set()

    for token in normalized_tokens:
        if token in _STOP_WORDS:
            continue
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        deduplicated_tokens.append(token)

    return deduplicated_tokens


def _count_keyword_occurrences(text: str, keyword: str) -> int:
    """Count whole-word case-insensitive matches for a keyword."""
    pattern = re.compile(rf"\b{re.escape(keyword)}\b", flags=re.IGNORECASE)
    return len(pattern.findall(text))


def _score_chunk(text: str, keywords: list[str]) -> tuple[float, list[str]]:
    """Calculate weighted lexical score and matched terms for one chunk."""
    score = 0.0
    matched_terms: list[str] = []

    for keyword in keywords:
        frequency = _count_keyword_occurrences(text, keyword)
        if frequency <= 0:
            continue
        score += float(frequency * len(keyword))
        matched_terms.append(keyword)

    return score, matched_terms


def _build_snippet(text: str, matched_terms: list[str], max_length: int = 180) -> str:
    """Return a concise excerpt centered on a matched sentence when possible."""
    if not text.strip():
        return ""

    sentences = [sentence.strip() for sentence in _SENTENCE_RE.split(text.strip()) if sentence.strip()]
    lowered_terms = [term.lower() for term in matched_terms]

    for sentence in sentences:
        lowered_sentence = sentence.lower()
        if any(term in lowered_sentence for term in lowered_terms):
            if len(sentence) <= max_length:
                return sentence
            return sentence[: max_length - 3].rstrip() + "..."

    compact_text = re.sub(r"\s+", " ", text).strip()
    if len(compact_text) <= max_length:
        return compact_text
    return compact_text[: max_length - 3].rstrip() + "..."


def keyword_search(query: str, chunks: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    """Run exact lexical retrieval over chunk text and return ranked matches."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    keywords = _normalize_query(query)
    if not keywords:
        LOGGER.info("keyword_search_completed", extra={"query": query, "keywords": [], "match_count": 0})
        return []

    results: list[dict[str, Any]] = []

    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        chunk_text = str(chunk.get("text", ""))
        if not chunk_id or not chunk_text.strip():
            continue

        score, matched_terms = _score_chunk(chunk_text, keywords)
        if score <= 0:
            continue

        results.append(
            {
                "chunk_id": chunk_id,
                "score": score,
                "matched_terms": matched_terms,
                "snippet": _build_snippet(chunk_text, matched_terms),
            }
        )

    ranked_results = sorted(results, key=lambda item: (-float(item["score"]), str(item["chunk_id"])))
    final_results = ranked_results[:top_k]

    LOGGER.info(
        "keyword_search_completed",
        extra={
            "query": query,
            "keywords": keywords,
            "match_count": len(final_results),
            "top_k": top_k,
        },
    )
    return final_results
