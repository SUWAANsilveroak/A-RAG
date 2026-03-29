"""Debug utilities for concise and safe pipeline flow inspection."""

from __future__ import annotations

import logging
from typing import Any


LOGGER = logging.getLogger(__name__)
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "password",
    "secret",
    "token",
}
_PROMPT_PREVIEW_MAX = 500
_ANSWER_PREVIEW_MAX = 300
_TRUNCATED_SUFFIX = "...[truncated]"


def _sanitize_value(value: Any) -> Any:
    """Recursively sanitize values and redact likely sensitive fields."""
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if any(token in key_str.lower() for token in _SENSITIVE_KEYS):
                sanitized[key_str] = "[REDACTED]"
            else:
                sanitized[key_str] = _sanitize_value(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_sanitize_value(item) for item in value)
    return value


def _truncate_text(text: str, max_chars: int, field_name: str) -> str:
    """Truncate long text safely while logging truncation activity."""
    normalized_text = str(text or "").strip()
    if len(normalized_text) <= max_chars:
        return normalized_text

    LOGGER.info(
        "debug_text_truncated",
        extra={"field_name": field_name, "original_length": len(normalized_text), "max_chars": max_chars},
    )
    if max_chars <= len(_TRUNCATED_SUFFIX):
        return _TRUNCATED_SUFFIX[:max_chars]
    return normalized_text[: max_chars - len(_TRUNCATED_SUFFIX)] + _TRUNCATED_SUFFIX


def debug_retrieval_flow(
    planner_output: dict[str, Any],
    tool_selection_output: dict[str, Any],
    retrieval_output: dict[str, Any],
    evaluation_output: dict[str, Any],
) -> dict[str, Any]:
    """Return concise debug snapshot for planner -> selection -> retrieval -> evaluation."""
    try:
        safe_planner = _sanitize_value(planner_output if isinstance(planner_output, dict) else {})
        safe_tool_selection = _sanitize_value(tool_selection_output if isinstance(tool_selection_output, dict) else {})
        safe_retrieval = retrieval_output if isinstance(retrieval_output, dict) else {}
        safe_evaluation = _sanitize_value(evaluation_output if isinstance(evaluation_output, dict) else {})

        if not safe_retrieval:
            LOGGER.warning("debug_retrieval_flow_missing_retrieval_output")

        raw_result_count = safe_retrieval.get("result_count", 0)
        result_count = int(raw_result_count) if isinstance(raw_result_count, (int, float)) else 0
        retrieval_output_summary = {
            "tool_name": str(safe_retrieval.get("tool_name", "")).strip(),
            "status": str(safe_retrieval.get("status", "")).strip(),
            "result_count": max(result_count, 0),
            "execution_summary": str(safe_retrieval.get("execution_summary", "")).strip(),
        }

        debug_payload = {
            "planner_output": safe_planner,
            "tool_selection_output": safe_tool_selection,
            "retrieval_output_summary": _sanitize_value(retrieval_output_summary),
            "evaluation_output": safe_evaluation,
        }
        LOGGER.info(
            "debug_retrieval_flow_generated",
            extra={"result_count": retrieval_output_summary["result_count"]},
        )
        return debug_payload
    except Exception as error:
        LOGGER.exception("debug_retrieval_flow_failed", extra={"error_message": str(error)})
        return {
            "planner_output": {},
            "tool_selection_output": {},
            "retrieval_output_summary": {},
            "evaluation_output": {},
        }


def debug_validation_flow(
    grounding_result: dict[str, Any],
    conflict_result: dict[str, Any],
    completeness_result: dict[str, Any],
    retry_result: dict[str, Any],
) -> dict[str, Any]:
    """Return concise debug snapshot for validation and retry decision artifacts."""
    try:
        debug_payload = {
            "grounding_result": _sanitize_value(grounding_result if isinstance(grounding_result, dict) else {}),
            "conflict_result": _sanitize_value(conflict_result if isinstance(conflict_result, dict) else {}),
            "completeness_result": _sanitize_value(
                completeness_result if isinstance(completeness_result, dict) else {}
            ),
            "retry_result": _sanitize_value(retry_result if isinstance(retry_result, dict) else {}),
        }

        missing_count = sum(1 for value in debug_payload.values() if not value)
        if missing_count > 0:
            LOGGER.warning("debug_validation_flow_missing_fields", extra={"missing_sections": missing_count})

        LOGGER.info("debug_validation_flow_generated")
        return debug_payload
    except Exception as error:
        LOGGER.exception("debug_validation_flow_failed", extra={"error_message": str(error)})
        return {
            "grounding_result": {},
            "conflict_result": {},
            "completeness_result": {},
            "retry_result": {},
        }


def debug_answer_flow(
    prompt: str,
    answer_response: dict[str, Any],
    final_output: dict[str, Any],
) -> dict[str, Any]:
    """Return concise debug snapshot for prompt, raw answer response, and final formatted output."""
    try:
        safe_answer_response = answer_response if isinstance(answer_response, dict) else {}
        safe_final_output = _sanitize_value(final_output if isinstance(final_output, dict) else {})

        prompt_preview = _truncate_text(prompt, _PROMPT_PREVIEW_MAX, "prompt_preview")
        answer_text = str(safe_answer_response.get("answer", "")).strip()
        answer_preview = _truncate_text(answer_text, _ANSWER_PREVIEW_MAX, "answer_preview")

        if not answer_text:
            LOGGER.warning("debug_answer_flow_empty_answer")
        if not str(prompt or "").strip():
            LOGGER.warning("debug_answer_flow_empty_prompt")

        answer_summary = {
            "status": str(safe_answer_response.get("status", "")).strip(),
            "provider": str(safe_answer_response.get("provider", "")).strip(),
            "model_name": str(safe_answer_response.get("model_name", "")).strip(),
            "answer_preview": answer_preview,
            "answer_length": len(answer_text),
            "token_usage": _sanitize_value(
                safe_answer_response.get("token_usage", {})
                if isinstance(safe_answer_response.get("token_usage", {}), dict)
                else {}
            ),
            "latency_seconds": (
                float(safe_answer_response.get("latency_seconds", 0.0))
                if isinstance(safe_answer_response.get("latency_seconds", 0.0), (int, float))
                else 0.0
            ),
        }

        LOGGER.info(
            "debug_answer_flow_generated",
            extra={"prompt_preview_length": len(prompt_preview), "answer_preview_length": len(answer_preview)},
        )
        return {
            "prompt_preview": prompt_preview,
            "answer_summary": _sanitize_value(answer_summary),
            "final_output": safe_final_output,
        }
    except Exception as error:
        LOGGER.exception("debug_answer_flow_failed", extra={"error_message": str(error)})
        return {
            "prompt_preview": "",
            "answer_summary": {},
            "final_output": {},
        }

