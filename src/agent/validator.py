"""Answer grounding validation utilities."""

from __future__ import annotations

import logging
import re
from typing import Any


LOGGER = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_-]*\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_STOPWORDS = {
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
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
    "explain",
    "describe",
    "tell",
    "show",
    "give",
    "provide",
    "what",
    "how",
    "why",
}
_OPPOSITE_PATTERNS = [
    ("yes", "no"),
    ("increase", "decrease"),
    ("increased", "decreased"),
    ("allowed", "not allowed"),
    ("allow", "not allow"),
    ("present", "absent"),
    ("before", "after"),
    ("active", "inactive"),
    ("enabled", "disabled"),
]


def _extract_context_chunk_ids(compressed_context: list[dict[str, Any]]) -> set[str]:
    """Extract valid unique chunk ids from compressed context."""
    context_chunk_ids: set[str] = set()
    for item in compressed_context:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        if chunk_id:
            context_chunk_ids.add(chunk_id)
    return context_chunk_ids


def _extract_supporting_chunks(final_output: dict[str, Any]) -> list[str]:
    """Extract ordered unique supporting chunk ids from final output."""
    raw_supporting_chunks = final_output.get("supporting_chunks", [])
    if not isinstance(raw_supporting_chunks, list):
        return []

    supporting_chunks: list[str] = []
    seen: set[str] = set()
    for chunk in raw_supporting_chunks:
        chunk_id = str(chunk).strip()
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        supporting_chunks.append(chunk_id)
    return supporting_chunks


def _build_notes(
    answer_present: bool,
    supporting_chunks: list[str],
    missing_chunks: list[str],
    grounded: bool,
) -> str:
    """Build human-readable notes describing grounding outcome."""
    if grounded:
        return "Answer is grounded in the provided supporting chunks."

    note_parts: list[str] = []
    if not answer_present:
        note_parts.append("Answer text is empty.")
    if not supporting_chunks:
        note_parts.append("No supporting chunks were provided.")
    if missing_chunks:
        note_parts.append(f"Missing supporting chunks: {', '.join(missing_chunks)}.")

    return " ".join(note_parts).strip() or "Grounding validation failed."


def validate_grounding(
    final_output: dict[str, Any],
    compressed_context: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate whether final answer is grounded in retrieved compressed context."""
    try:
        safe_output = final_output if isinstance(final_output, dict) else {}
        safe_context = compressed_context if isinstance(compressed_context, list) else []

        answer_text = str(safe_output.get("answer", "")).strip()
        answer_present = bool(answer_text)

        supporting_chunks = _extract_supporting_chunks(safe_output)
        context_chunk_ids = _extract_context_chunk_ids(safe_context)

        missing_chunks = [chunk_id for chunk_id in supporting_chunks if chunk_id not in context_chunk_ids]
        matched_count = len(supporting_chunks) - len(missing_chunks)
        coverage = (matched_count / len(supporting_chunks)) if supporting_chunks else 0.0

        grounded = answer_present and bool(supporting_chunks) and len(missing_chunks) == 0

        if not answer_present or not supporting_chunks:
            grounding_score = 0.0
        else:
            grounding_score = max(0.0, min(1.0, coverage))

        notes = _build_notes(answer_present, supporting_chunks, missing_chunks, grounded)

        LOGGER.info(
            "grounding_validation_completed",
            extra={
                "supporting_chunk_count": len(supporting_chunks),
                "missing_chunk_count": len(missing_chunks),
                "grounding_score": grounding_score,
                "grounded": grounded,
            },
        )
        return {
            "grounded": grounded,
            "grounding_score": grounding_score,
            "missing_chunks": missing_chunks,
            "notes": notes,
        }
    except Exception as error:
        LOGGER.exception(
            "grounding_validation_failed",
            extra={
                "final_output": final_output,
                "context_count": len(compressed_context) if isinstance(compressed_context, list) else 0,
            },
        )
        return {
            "grounded": False,
            "grounding_score": 0.0,
            "missing_chunks": [],
            "notes": f"Grounding validation error: {error}",
        }


def _normalize_for_conflict(text: str) -> str:
    """Normalize text for conflict heuristics."""
    return str(text).strip().lower()


def _keyword_set(text: str) -> set[str]:
    """Extract keyword set for concept overlap checks."""
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    return {token for token in tokens if token not in _STOPWORDS and len(token) > 2}


def _find_opposite_reason(text_a: str, text_b: str) -> str:
    """Return reason if opposite lexical patterns are detected."""
    for left, right in _OPPOSITE_PATTERNS:
        if (left in text_a and right in text_b) or (right in text_a and left in text_b):
            return f"Opposite statement detected: '{left}' vs '{right}'."
    return ""


def _find_numeric_conflict_reason(text_a: str, text_b: str) -> str:
    """Return reason for numeric mismatch on likely shared concept."""
    numbers_a = set(_NUMBER_RE.findall(text_a))
    numbers_b = set(_NUMBER_RE.findall(text_b))
    if not numbers_a or not numbers_b or numbers_a == numbers_b:
        return ""

    shared_keywords = _keyword_set(text_a) & _keyword_set(text_b)
    if len(shared_keywords) < 2:
        return ""
    return f"Numeric mismatch for overlapping concept keywords: {', '.join(sorted(shared_keywords)[:4])}."


def detect_conflicts(compressed_context: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect potential conflicts across compressed context chunk pairs."""
    try:
        safe_context = compressed_context if isinstance(compressed_context, list) else []
        valid_chunks: list[tuple[str, str]] = []
        for item in safe_context:
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("chunk_id", "")).strip()
            compressed_text = _normalize_for_conflict(str(item.get("compressed_text", "")))
            if not chunk_id or not compressed_text:
                continue
            valid_chunks.append((chunk_id, compressed_text))

        conflict_pairs: list[dict[str, str]] = []
        conflicting_chunks: list[str] = []
        seen_conflicting_chunk_ids: set[str] = set()

        for index_a in range(len(valid_chunks)):
            chunk_a, text_a = valid_chunks[index_a]
            for index_b in range(index_a + 1, len(valid_chunks)):
                chunk_b, text_b = valid_chunks[index_b]
                if chunk_a == chunk_b:
                    continue

                reason = _find_opposite_reason(text_a, text_b)
                if not reason:
                    reason = _find_numeric_conflict_reason(text_a, text_b)
                if not reason:
                    continue

                conflict_pairs.append(
                    {
                        "chunk_a": chunk_a,
                        "chunk_b": chunk_b,
                        "reason": reason,
                    }
                )
                if chunk_a not in seen_conflicting_chunk_ids:
                    seen_conflicting_chunk_ids.add(chunk_a)
                    conflicting_chunks.append(chunk_a)
                if chunk_b not in seen_conflicting_chunk_ids:
                    seen_conflicting_chunk_ids.add(chunk_b)
                    conflicting_chunks.append(chunk_b)

        has_conflict = len(conflict_pairs) > 0
        notes = (
            f"Detected {len(conflict_pairs)} conflict pair(s)."
            if has_conflict
            else "No conflicts detected in compressed context."
        )

        LOGGER.info(
            "conflict_detection_completed",
            extra={
                "chunk_count": len(valid_chunks),
                "conflict_pair_count": len(conflict_pairs),
                "has_conflict": has_conflict,
            },
        )
        return {
            "has_conflict": has_conflict,
            "conflicting_chunks": conflicting_chunks,
            "conflict_pairs": conflict_pairs,
            "notes": notes,
        }
    except Exception as error:
        LOGGER.exception(
            "conflict_detection_failed",
            extra={"context_count": len(compressed_context) if isinstance(compressed_context, list) else 0},
        )
        return {
            "has_conflict": False,
            "conflicting_chunks": [],
            "conflict_pairs": [],
            "notes": f"Conflict detection error: {error}",
        }


def _extract_major_terms(text: str) -> list[str]:
    """Extract normalized major terms from text by removing stop words."""
    normalized_text = _normalize_for_conflict(text)
    tokens = [token.lower() for token in _TOKEN_RE.findall(normalized_text)]
    major_terms: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in _STOPWORDS:
            continue
        if len(token) <= 2:
            continue
        if token in seen:
            continue
        seen.add(token)
        major_terms.append(token)
    return major_terms


def _context_text_blob(compressed_context: list[dict[str, Any]]) -> str:
    """Combine compressed context text into one normalized blob."""
    parts: list[str] = []
    for item in compressed_context:
        if not isinstance(item, dict):
            continue
        text = _normalize_for_conflict(str(item.get("compressed_text", "")))
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def check_completeness(
    query: str,
    final_output: dict[str, Any],
    compressed_context: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check whether answer sufficiently covers major query concepts."""
    try:
        normalized_query = _normalize_for_conflict(query)
        safe_output = final_output if isinstance(final_output, dict) else {}
        safe_context = compressed_context if isinstance(compressed_context, list) else []
        answer_text = _normalize_for_conflict(str(safe_output.get("answer", "")))

        query_terms = _extract_major_terms(normalized_query)
        answer_terms = set(_extract_major_terms(answer_text))
        context_blob = _context_text_blob(safe_context)

        if not normalized_query:
            LOGGER.info(
                "completeness_check_completed",
                extra={
                    "query_terms": [],
                    "covered_terms": [],
                    "missing_terms": [],
                    "completeness_score": 0.0,
                },
            )
            return {
                "is_complete": False,
                "completeness_score": 0.0,
                "missing_topics": [],
                "notes": "Query is empty; completeness cannot be determined.",
            }

        if not answer_text:
            LOGGER.info(
                "completeness_check_completed",
                extra={
                    "query_terms": query_terms,
                    "covered_terms": [],
                    "missing_terms": query_terms,
                    "completeness_score": 0.0,
                },
            )
            return {
                "is_complete": False,
                "completeness_score": 0.0,
                "missing_topics": query_terms,
                "notes": "Answer is empty; major query topics are not covered.",
            }

        if not query_terms:
            # Query may contain mostly stop words or symbols. Treat answer presence as partial completeness.
            LOGGER.info(
                "completeness_check_completed",
                extra={
                    "query_terms": [],
                    "covered_terms": [],
                    "missing_terms": [],
                    "completeness_score": 0.5,
                },
            )
            return {
                "is_complete": False,
                "completeness_score": 0.5,
                "missing_topics": [],
                "notes": "No major query terms found after normalization.",
            }

        covered_terms = [term for term in query_terms if term in answer_terms]
        missing_terms = [term for term in query_terms if term not in answer_terms]

        # If missing terms are at least present in retrieved context, note partial coverage.
        context_only_terms = [term for term in missing_terms if term in context_blob]

        completeness_score = len(covered_terms) / len(query_terms)
        completeness_score = max(0.0, min(1.0, float(completeness_score)))
        is_complete = len(missing_terms) == 0

        if is_complete:
            notes = "Answer covers all major query topics."
        elif context_only_terms:
            notes = (
                "Answer is partial; some missing topics exist in retrieved context: "
                + ", ".join(context_only_terms)
                + "."
            )
        else:
            notes = "Answer is partial; missing topics are not covered in the answer."

        LOGGER.info(
            "completeness_check_completed",
            extra={
                "query_terms": query_terms,
                "covered_terms": covered_terms,
                "missing_terms": missing_terms,
                "completeness_score": completeness_score,
            },
        )
        return {
            "is_complete": is_complete,
            "completeness_score": completeness_score,
            "missing_topics": missing_terms,
            "notes": notes,
        }
    except Exception as error:
        LOGGER.exception(
            "completeness_check_failed",
            extra={
                "query": query,
                "context_count": len(compressed_context) if isinstance(compressed_context, list) else 0,
            },
        )
        return {
            "is_complete": False,
            "completeness_score": 0.0,
            "missing_topics": [],
            "notes": f"Completeness check error: {error}",
        }


def decide_retry_action(
    grounding_result: dict[str, Any],
    conflict_result: dict[str, Any],
    completeness_result: dict[str, Any],
    max_retries: int,
    current_retry_count: int,
) -> dict[str, Any]:
    """Decide whether another retrieval retry is needed and which strategy to use."""
    try:
        safe_grounding = grounding_result if isinstance(grounding_result, dict) else {}
        safe_conflict = conflict_result if isinstance(conflict_result, dict) else {}
        safe_completeness = completeness_result if isinstance(completeness_result, dict) else {}

        normalized_max_retries = max(int(max_retries), 0)
        normalized_retry_count = max(int(current_retry_count), 0)
        remaining_retries = max(normalized_max_retries - normalized_retry_count, 0)

        grounding_failed = not bool(safe_grounding.get("grounded", False))
        conflict_detected = bool(safe_conflict.get("has_conflict", False))
        incomplete_answer = not bool(safe_completeness.get("is_complete", False))

        failure_reasons: list[str] = []
        if grounding_failed:
            failure_reasons.append("grounding_failed")
        if conflict_detected:
            failure_reasons.append("conflict_detected")
        if incomplete_answer:
            failure_reasons.append("incomplete_answer")

        if not failure_reasons:
            LOGGER.info(
                "retry_decision_completed",
                extra={
                    "should_retry": False,
                    "retry_reason": "validation_passed",
                    "remaining_retries": remaining_retries,
                    "recommended_strategy": "no_retry",
                },
            )
            return {
                "should_retry": False,
                "retry_reason": "validation_passed",
                "recommended_strategy": "no_retry",
                "remaining_retries": remaining_retries,
            }

        if normalized_retry_count >= normalized_max_retries:
            LOGGER.info(
                "retry_decision_completed",
                extra={
                    "should_retry": False,
                    "retry_reason": "retry_limit_reached",
                    "remaining_retries": remaining_retries,
                    "recommended_strategy": "no_retry",
                },
            )
            return {
                "should_retry": False,
                "retry_reason": "retry_limit_reached",
                "recommended_strategy": "no_retry",
                "remaining_retries": remaining_retries,
            }

        if len(failure_reasons) >= 2:
            recommended_strategy = "retry_with_hybrid_search"
        elif grounding_failed:
            recommended_strategy = "retry_with_chunk_read"
        elif conflict_detected:
            recommended_strategy = "retry_with_hybrid_search"
        else:
            recommended_strategy = "retry_with_semantic_search"

        retry_reason = ", ".join(failure_reasons)
        LOGGER.info(
            "retry_decision_completed",
            extra={
                "should_retry": True,
                "retry_reason": retry_reason,
                "remaining_retries": remaining_retries,
                "recommended_strategy": recommended_strategy,
            },
        )
        return {
            "should_retry": True,
            "retry_reason": retry_reason,
            "recommended_strategy": recommended_strategy,
            "remaining_retries": remaining_retries,
        }
    except Exception as error:
        LOGGER.exception(
            "retry_decision_failed",
            extra={
                "max_retries": max_retries,
                "current_retry_count": current_retry_count,
            },
        )
        return {
            "should_retry": False,
            "retry_reason": f"retry_decision_error: {error}",
            "recommended_strategy": "no_retry",
            "remaining_retries": 0,
        }
