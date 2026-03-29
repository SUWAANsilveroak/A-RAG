"""Evaluation metric calculators for retrieval, answer quality, and system efficiency."""

from __future__ import annotations

import logging
from typing import Any


LOGGER = logging.getLogger(__name__)


def _extract_score(result: dict[str, Any]) -> float:
    """Extract best-available relevance score from a retrieval result."""
    for key in ("combined_score", "score", "semantic_score", "keyword_score", "relevance_score"):
        value = result.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def calculate_retrieval_metrics(
    retrieval_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate retrieval quality metrics from result list."""
    try:
        safe_results = retrieval_results if isinstance(retrieval_results, list) else []
        valid_results = [result for result in safe_results if isinstance(result, dict)]
        result_count = len(valid_results)

        scores = [_extract_score(result) for result in valid_results]
        average_score = (sum(scores) / result_count) if result_count > 0 else 0.0
        top_score = max(scores) if scores else 0.0

        chunk_ids = [str(result.get("chunk_id", "")).strip() for result in valid_results if str(result.get("chunk_id", "")).strip()]
        duplicate_chunk_count = len(chunk_ids) - len(set(chunk_ids))

        LOGGER.info(
            "retrieval_metrics_calculated",
            extra={
                "result_count": result_count,
                "average_score": average_score,
                "top_score": top_score,
                "duplicate_chunk_count": duplicate_chunk_count,
            },
        )
        return {
            "result_count": result_count,
            "average_score": float(average_score),
            "top_score": float(top_score),
            "duplicate_chunk_count": int(max(duplicate_chunk_count, 0)),
        }
    except Exception as error:
        LOGGER.exception("retrieval_metrics_failed", extra={"error_message": str(error)})
        return {
            "result_count": 0,
            "average_score": 0.0,
            "top_score": 0.0,
            "duplicate_chunk_count": 0,
        }


def calculate_answer_metrics(
    final_output: dict[str, Any],
    grounding_result: dict[str, Any],
    completeness_result: dict[str, Any],
) -> dict[str, Any]:
    """Calculate answer-quality metrics from output and validation artifacts."""
    try:
        safe_output = final_output if isinstance(final_output, dict) else {}
        safe_grounding = grounding_result if isinstance(grounding_result, dict) else {}
        safe_completeness = completeness_result if isinstance(completeness_result, dict) else {}

        answer_text = str(safe_output.get("answer", "")).strip()
        answer_length = len(answer_text)

        grounding_score_raw = safe_grounding.get("grounding_score", 0.0)
        grounding_score = float(grounding_score_raw) if isinstance(grounding_score_raw, (int, float)) else 0.0

        completeness_score_raw = safe_completeness.get("completeness_score", 0.0)
        completeness_score = float(completeness_score_raw) if isinstance(completeness_score_raw, (int, float)) else 0.0

        confidence_raw = safe_output.get("confidence", 0.0)
        confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else 0.0
        confidence = max(0.0, min(confidence, 1.0))

        LOGGER.info(
            "answer_metrics_calculated",
            extra={
                "grounding_score": grounding_score,
                "completeness_score": completeness_score,
                "confidence": confidence,
                "answer_length": answer_length,
            },
        )
        return {
            "grounding_score": max(0.0, min(grounding_score, 1.0)),
            "completeness_score": max(0.0, min(completeness_score, 1.0)),
            "confidence": confidence,
            "answer_length": answer_length,
        }
    except Exception as error:
        LOGGER.exception("answer_metrics_failed", extra={"error_message": str(error)})
        return {
            "grounding_score": 0.0,
            "completeness_score": 0.0,
            "confidence": 0.0,
            "answer_length": 0,
        }


def calculate_system_metrics(
    token_usage: dict[str, Any],
    latency_seconds: float,
    retry_count: int,
) -> dict[str, Any]:
    """Calculate efficiency metrics from token usage, latency, and retry count."""
    try:
        safe_usage = token_usage if isinstance(token_usage, dict) else {}

        prompt_tokens_raw = safe_usage.get("prompt_tokens", 0)
        completion_tokens_raw = safe_usage.get("completion_tokens", 0)
        total_tokens_raw = safe_usage.get("total_tokens", 0)

        prompt_tokens = int(prompt_tokens_raw) if isinstance(prompt_tokens_raw, (int, float)) else 0
        completion_tokens = int(completion_tokens_raw) if isinstance(completion_tokens_raw, (int, float)) else 0
        total_tokens = int(total_tokens_raw) if isinstance(total_tokens_raw, (int, float)) else 0

        normalized_latency = float(latency_seconds) if isinstance(latency_seconds, (int, float)) else 0.0
        if normalized_latency < 0:
            LOGGER.warning("system_metrics_negative_latency", extra={"latency_seconds": normalized_latency})
            normalized_latency = 0.0

        normalized_retry_count = int(retry_count) if isinstance(retry_count, (int, float)) else 0
        if normalized_retry_count < 0:
            LOGGER.warning("system_metrics_negative_retry_count", extra={"retry_count": normalized_retry_count})
            normalized_retry_count = 0

        LOGGER.info(
            "system_metrics_calculated",
            extra={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_seconds": normalized_latency,
                "retry_count": normalized_retry_count,
            },
        )
        return {
            "prompt_tokens": max(prompt_tokens, 0),
            "completion_tokens": max(completion_tokens, 0),
            "total_tokens": max(total_tokens, 0),
            "latency_seconds": normalized_latency,
            "retry_count": normalized_retry_count,
        }
    except Exception as error:
        LOGGER.exception("system_metrics_failed", extra={"error_message": str(error)})
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
            "retry_count": 0,
        }
