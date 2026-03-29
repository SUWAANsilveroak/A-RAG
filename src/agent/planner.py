"""Query planning node for strategy selection before tool execution."""

from __future__ import annotations

import logging
import re
from typing import Any


LOGGER = logging.getLogger(__name__)
_WHITESPACE_RE = re.compile(r"\s+")
_SYMBOL_ONLY_RE = re.compile(r"^[^\w]+$", flags=re.UNICODE)
_NON_WORD_PUNCT_RE = re.compile(r"[^\w\s-]", flags=re.UNICODE)
_CHUNK_LOOKUP_RE = re.compile(r"\bchunk[_\-\s]?\d+\b", flags=re.IGNORECASE)
_EXACT_QUOTE_RE = re.compile(r"[\"'`]")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
_ID_TOKEN_RE = re.compile(r"\b(?:[A-Za-z]+[-_]?\d+[A-Za-z\d_-]*|\d{3,})\b")
_BROAD_CONTEXT_TERMS = {
    "explain",
    "overview",
    "summarize",
    "meaning",
    "concept",
    "context",
    "understand",
    "how",
    "why",
    "what",
}
_CHUNK_LOOKUP_TERMS = {
    "chunk",
    "previous",
    "previously",
    "referenced",
    "above",
    "read",
}


def _normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace into single spaces."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _rewrite_query(query: str) -> str:
    """Rewrite query by removing unnecessary punctuation and normalizing spacing."""
    if not str(query).strip():
        return ""
    punctuation_cleaned = _NON_WORD_PUNCT_RE.sub(" ", str(query))
    return _normalize_whitespace(punctuation_cleaned)


def _is_chunk_lookup_query(original_query: str, rewritten_query: str) -> bool:
    """Detect whether query asks to read chunk or previously referenced context."""
    if _CHUNK_LOOKUP_RE.search(original_query):
        return True
    tokens = set(token.lower() for token in rewritten_query.split())
    return "chunk" in tokens and bool(tokens & _CHUNK_LOOKUP_TERMS)


def _has_exact_signal(original_query: str, rewritten_query: str) -> bool:
    """Detect exact-term style signals such as IDs, acronyms, or quoted text."""
    if _EXACT_QUOTE_RE.search(original_query):
        return True
    if _ID_TOKEN_RE.search(rewritten_query):
        return True
    if _ACRONYM_RE.search(original_query):
        return True
    return False


def _has_broad_context_signal(rewritten_query: str) -> bool:
    """Detect broad semantic intent from natural-language phrasing."""
    lowered_tokens = [token.lower() for token in rewritten_query.split()]
    if len(lowered_tokens) >= 7:
        return True
    if any(token in _BROAD_CONTEXT_TERMS for token in lowered_tokens):
        return True
    return False


def _classify_query(original_query: str, rewritten_query: str) -> tuple[str, str, str]:
    """Classify query type and return retrieval strategy + explanation."""
    if not rewritten_query:
        return "unknown", "fallback_search", "Query is empty after rewrite."

    if _SYMBOL_ONLY_RE.match(original_query.strip()):
        return "unknown", "fallback_search", "Query contains only symbols."

    if _is_chunk_lookup_query(original_query, rewritten_query):
        return "chunk_lookup", "chunk_read", "Query requests chunk lookup or prior context."

    has_exact = _has_exact_signal(original_query, rewritten_query)
    has_broad = _has_broad_context_signal(rewritten_query)

    if has_exact and has_broad:
        return "hybrid", "hybrid_search", "Query mixes exact identifiers with broad context."
    if has_exact:
        return "keyword", "keyword_search", "Query contains exact-term signals."
    if has_broad:
        return "semantic", "semantic_search", "Query is broad natural-language intent."

    if len(rewritten_query.split()) <= 2:
        return "keyword", "keyword_search", "Very short query favors precise lexical match."
    return "semantic", "semantic_search", "Defaulting to semantic retrieval for non-exact query."


def plan_query(query: str) -> dict[str, Any]:
    """Plan query type and retrieval strategy for downstream tool selection."""
    try:
        original_query = _normalize_whitespace(str(query))
        rewritten_query = _rewrite_query(original_query)
        query_type, retrieval_strategy, reason = _classify_query(original_query, rewritten_query)

        LOGGER.info(
            "query_planner_completed",
            extra={
                "original_query": original_query,
                "rewritten_query": rewritten_query,
                "query_type": query_type,
                "retrieval_strategy": retrieval_strategy,
            },
        )
        return {
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "query_type": query_type,
            "retrieval_strategy": retrieval_strategy,
            "reason": reason,
        }
    except Exception as error:
        LOGGER.exception("query_planner_failed", extra={"query": query, "error_message": str(error)})
        safe_query = _normalize_whitespace(str(query))
        return {
            "original_query": safe_query,
            "rewritten_query": _rewrite_query(safe_query),
            "query_type": "unknown",
            "retrieval_strategy": "fallback_search",
            "reason": f"Planner failed: {error}",
        }
