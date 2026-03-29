"""Agent loop utilities for tool selection and execution preparation."""

from __future__ import annotations

import logging
from typing import Any

from src.tools.reader import run_chunk_read_tool
from src.tools.search_tools import (
    run_hybrid_search_tool,
    run_keyword_search_tool,
    run_semantic_search_tool,
)


LOGGER = logging.getLogger(__name__)
_DEFAULT_TOP_K = 5
_SUPPORTED_STRATEGY_TO_TOOL = {
    "keyword_search": "keyword_search",
    "semantic_search": "semantic_search",
    "hybrid_search": "hybrid_search",
    "chunk_read": "chunk_read",
    "fallback_search": "hybrid_search",
}
_EXECUTABLE_TOOLS = {
    "keyword_search",
    "semantic_search",
    "hybrid_search",
    "chunk_read",
    "fallback_search",
}


def _extract_query(planner_output: dict[str, Any]) -> str:
    """Extract best-available query text from planner output."""
    rewritten_query = str(planner_output.get("rewritten_query", "")).strip()
    if rewritten_query:
        return rewritten_query
    return str(planner_output.get("original_query", "")).strip()


def _resolve_tool_name(retrieval_strategy: str) -> tuple[str, bool]:
    """Resolve tool name from retrieval strategy and indicate fallback usage."""
    normalized_strategy = retrieval_strategy.strip()
    if normalized_strategy in _SUPPORTED_STRATEGY_TO_TOOL:
        return _SUPPORTED_STRATEGY_TO_TOOL[normalized_strategy], normalized_strategy == "fallback_search"

    if not normalized_strategy:
        LOGGER.warning("tool_selection_missing_strategy")
    else:
        LOGGER.warning(
            "tool_selection_unknown_strategy",
            extra={"retrieval_strategy": normalized_strategy},
        )
    return "hybrid_search", True


def _is_execution_ready(tool_name: str, tool_input: dict[str, Any]) -> bool:
    """Return whether the selected tool call has minimum required input fields."""
    if not tool_name.strip():
        return False
    query = str(tool_input.get("query", "")).strip()
    if not query:
        return False
    return "top_k" in tool_input


def select_tool(planner_output: dict[str, Any]) -> dict[str, Any]:
    """Select and prepare the tool call from planner output strategy."""
    try:
        safe_planner_output = planner_output if isinstance(planner_output, dict) else {}
        retrieval_strategy = str(safe_planner_output.get("retrieval_strategy", "")).strip()
        tool_name, fallback_used = _resolve_tool_name(retrieval_strategy)

        query = _extract_query(safe_planner_output)
        tool_input: dict[str, Any] = {
            "query": query,
            "top_k": int(safe_planner_output.get("top_k", _DEFAULT_TOP_K)),
        }

        if tool_name == "chunk_read":
            chunk_ids = safe_planner_output.get("chunk_ids")
            if isinstance(chunk_ids, list):
                tool_input["chunk_ids"] = [str(chunk_id).strip() for chunk_id in chunk_ids if str(chunk_id).strip()]

        execution_ready = _is_execution_ready(tool_name, tool_input)

        planner_reason = str(safe_planner_output.get("reason", "")).strip()
        reason_parts = []
        if planner_reason:
            reason_parts.append(planner_reason)
        if fallback_used:
            reason_parts.append("Fallback strategy applied.")
        if not execution_ready:
            reason_parts.append("Missing required tool input.")
        final_reason = " ".join(reason_parts).strip() or "Tool selected."

        LOGGER.info(
            "tool_selection_completed",
            extra={
                "tool_name": tool_name,
                "tool_input": tool_input,
                "execution_ready": execution_ready,
                "fallback_used": fallback_used,
            },
        )
        return {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "execution_ready": execution_ready,
            "reason": final_reason,
        }
    except Exception as error:
        LOGGER.exception(
            "tool_selection_failed",
            extra={"planner_output": planner_output, "error_message": str(error)},
        )
        return {
            "tool_name": "",
            "tool_input": {"query": "", "top_k": _DEFAULT_TOP_K},
            "execution_ready": False,
            "reason": f"Tool selection failed: {error}",
        }


def _build_execution_summary(tool_name: str, result_count: int) -> str:
    """Build a human-readable execution summary from tool result count."""
    if result_count <= 0:
        return f"{tool_name} returned no results"
    if tool_name == "chunk_read":
        return f"{tool_name} returned {result_count} chunks"
    return f"{tool_name} returned {result_count} results"


def _safe_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract results list from payload safely."""
    raw_results = payload.get("results", [])
    if isinstance(raw_results, list):
        return raw_results
    return []


def execute_retrieval(
    tool_selection_output: dict[str, Any],
    resources: dict[str, Any],
) -> dict[str, Any]:
    """Execute the selected retrieval tool with standardized input/output contract."""
    try:
        safe_selection = tool_selection_output if isinstance(tool_selection_output, dict) else {}
        tool_name = str(safe_selection.get("tool_name", "")).strip()
        tool_input = safe_selection.get("tool_input", {})
        execution_ready = bool(safe_selection.get("execution_ready", False))

        if not execution_ready:
            LOGGER.warning(
                "retrieval_execution_not_ready",
                extra={"tool_name": tool_name, "tool_input": tool_input},
            )
            return {
                "tool_name": tool_name,
                "status": "error",
                "results": [],
                "result_count": 0,
                "execution_summary": "Retrieval execution not ready",
            }

        if tool_name not in _EXECUTABLE_TOOLS:
            LOGGER.warning("retrieval_execution_invalid_tool", extra={"tool_name": tool_name})
            return {
                "tool_name": tool_name,
                "status": "error",
                "results": [],
                "result_count": 0,
                "execution_summary": f"Invalid tool: {tool_name}",
            }

        if not isinstance(tool_input, dict):
            tool_input = {}

        query = str(tool_input.get("query", "")).strip()
        top_k = int(tool_input.get("top_k", _DEFAULT_TOP_K))
        chunks = resources.get("chunks", [])
        model = resources.get("model")
        faiss_index = resources.get("faiss_index")
        metadata = resources.get("metadata", [])
        read_chunk_ids = resources.get("read_chunk_ids")

        LOGGER.info(
            "retrieval_execution_started",
            extra={
                "tool_name": tool_name,
                "tool_input": tool_input,
            },
        )

        tool_payload: dict[str, Any]
        if tool_name == "keyword_search":
            tool_payload = run_keyword_search_tool(
                query=query,
                chunks=chunks,
                top_k=top_k,
            )
        elif tool_name == "semantic_search":
            tool_payload = run_semantic_search_tool(
                query=query,
                model=model,
                faiss_index=faiss_index,
                metadata=metadata,
                top_k=top_k,
            )
        elif tool_name in {"hybrid_search", "fallback_search"}:
            tool_payload = run_hybrid_search_tool(
                query=query,
                chunks=chunks,
                model=model,
                faiss_index=faiss_index,
                metadata=metadata,
                top_k=top_k,
            )
        else:
            chunk_ids_raw = tool_input.get("chunk_ids", [])
            chunk_ids = (
                [str(chunk_id).strip() for chunk_id in chunk_ids_raw if str(chunk_id).strip()]
                if isinstance(chunk_ids_raw, list)
                else []
            )
            tool_payload = run_chunk_read_tool(
                chunk_ids=chunk_ids,
                chunks=chunks,
                read_chunk_ids=read_chunk_ids if isinstance(read_chunk_ids, set) else None,
            )

        status = str(tool_payload.get("status", "error"))
        results = _safe_results(tool_payload)
        result_count = len(results)
        execution_summary = _build_execution_summary(tool_name, result_count)

        LOGGER.info(
            "retrieval_execution_completed",
            extra={
                "tool_name": tool_name,
                "status": status,
                "result_count": result_count,
            },
        )
        return {
            "tool_name": str(tool_payload.get("tool_name", tool_name)),
            "status": status,
            "results": results,
            "result_count": result_count,
            "execution_summary": execution_summary,
        }
    except Exception as error:
        LOGGER.exception(
            "retrieval_execution_failed",
            extra={"tool_selection_output": tool_selection_output, "error_message": str(error)},
        )
        failed_tool_name = (
            str(tool_selection_output.get("tool_name", "")).strip()
            if isinstance(tool_selection_output, dict)
            else ""
        )
        return {
            "tool_name": failed_tool_name,
            "status": "error",
            "results": [],
            "result_count": 0,
            "execution_summary": f"Retrieval execution failed: {error}",
        }


def _extract_result_score(result: dict[str, Any]) -> float | None:
    """Extract the best available numeric score from a retrieval result item."""
    for score_key in ("combined_score", "score", "semantic_score", "keyword_score", "relevance_score"):
        score_value = result.get(score_key)
        if isinstance(score_value, (int, float)):
            return float(score_value)
    return None


def _retry_action_for_tool(tool_name: str) -> str:
    """Resolve retry action based on originating retrieval tool."""
    if tool_name == "keyword_search":
        return "retry_with_semantic_search"
    if tool_name == "semantic_search":
        return "retry_with_hybrid_search"
    if tool_name == "hybrid_search":
        return "retry_with_chunk_read"
    if tool_name == "fallback_search":
        return "retry_with_hybrid_search"
    if tool_name == "chunk_read":
        return "stop_no_results"
    return "retry_with_hybrid_search"


def evaluate_retrieval(
    retrieval_output: dict[str, Any],
    min_result_count: int = 1,
    min_score_threshold: float = 0.3,
) -> dict[str, Any]:
    """Evaluate retrieval sufficiency and recommend whether to retry retrieval."""
    try:
        safe_output = retrieval_output if isinstance(retrieval_output, dict) else {}
        tool_name = str(safe_output.get("tool_name", "")).strip()
        status = str(safe_output.get("status", "")).strip().lower()
        raw_results = safe_output.get("results", [])
        results = raw_results if isinstance(raw_results, list) else []
        result_count = int(safe_output.get("result_count", len(results)))
        if result_count != len(results):
            result_count = len(results)

        if status != "success":
            LOGGER.warning(
                "retrieval_evaluation_error_status",
                extra={"tool_name": tool_name, "status": status},
            )
            return {
                "sufficient_context": False,
                "needs_retry": True,
                "retry_reason": "Retrieval returned error status.",
                "recommended_next_action": "stop_no_results",
            }

        if result_count < int(min_result_count):
            next_action = _retry_action_for_tool(tool_name)
            LOGGER.info(
                "retrieval_evaluation_insufficient_result_count",
                extra={
                    "tool_name": tool_name,
                    "result_count": result_count,
                    "min_result_count": min_result_count,
                    "recommended_next_action": next_action,
                },
            )
            return {
                "sufficient_context": False,
                "needs_retry": True,
                "retry_reason": f"Result count {result_count} below minimum {min_result_count}.",
                "recommended_next_action": next_action,
            }

        if tool_name == "chunk_read":
            LOGGER.info(
                "retrieval_evaluation_chunk_read_sufficient",
                extra={"tool_name": tool_name, "result_count": result_count},
            )
            return {
                "sufficient_context": True,
                "needs_retry": False,
                "retry_reason": "",
                "recommended_next_action": "proceed_to_answer",
            }

        extracted_scores = [
            score
            for score in (_extract_result_score(result) for result in results if isinstance(result, dict))
            if score is not None
        ]
        score_quality_known = len(extracted_scores) > 0
        has_strong_score = any(score >= float(min_score_threshold) for score in extracted_scores)

        if (not score_quality_known) or (not has_strong_score):
            next_action = _retry_action_for_tool(tool_name)
            retry_reason = (
                "Missing score fields in retrieval results."
                if not score_quality_known
                else f"All scores below threshold {min_score_threshold}."
            )
            LOGGER.info(
                "retrieval_evaluation_retry_recommended",
                extra={
                    "tool_name": tool_name,
                    "result_count": result_count,
                    "score_quality_known": score_quality_known,
                    "scores": extracted_scores,
                    "min_score_threshold": min_score_threshold,
                    "recommended_next_action": next_action,
                },
            )
            return {
                "sufficient_context": False,
                "needs_retry": True,
                "retry_reason": retry_reason,
                "recommended_next_action": next_action,
            }

        LOGGER.info(
            "retrieval_evaluation_sufficient_context",
            extra={
                "tool_name": tool_name,
                "result_count": result_count,
                "scores": extracted_scores,
            },
        )
        return {
            "sufficient_context": True,
            "needs_retry": False,
            "retry_reason": "",
            "recommended_next_action": "proceed_to_answer",
        }
    except Exception as error:
        LOGGER.exception(
            "retrieval_evaluation_failed",
            extra={"retrieval_output": retrieval_output, "error_message": str(error)},
        )
        return {
            "sufficient_context": False,
            "needs_retry": True,
            "retry_reason": f"Evaluation failed: {error}",
            "recommended_next_action": "stop_no_results",
        }


def _normalize_recommended_action(evaluation_output: dict[str, Any]) -> str:
    """Return a safe recommended action string from evaluation output."""
    recommended_action = str(evaluation_output.get("recommended_next_action", "")).strip()
    if not recommended_action:
        return "stop_no_results"
    return recommended_action


def should_continue_loop(
    current_step: int,
    max_steps: int,
    evaluation_output: dict[str, Any],
) -> dict[str, Any]:
    """Decide whether retrieval loop should continue and what action should follow."""
    try:
        safe_evaluation_output = evaluation_output if isinstance(evaluation_output, dict) else {}
        normalized_current_step = int(current_step)
        normalized_max_steps = int(max_steps)

        if normalized_current_step < 0 or normalized_max_steps < 0:
            LOGGER.warning(
                "loop_control_invalid_step_count",
                extra={"current_step": current_step, "max_steps": max_steps},
            )
            return {
                "continue_loop": False,
                "stop_reason": "invalid_step_count",
                "next_action": "finalize_answer",
                "remaining_steps": 0,
            }

        remaining_steps = max(normalized_max_steps - normalized_current_step, 0)
        recommended_action = _normalize_recommended_action(safe_evaluation_output)
        sufficient_context = bool(safe_evaluation_output.get("sufficient_context", False))
        needs_retry = bool(safe_evaluation_output.get("needs_retry", False))

        continue_loop = False
        stop_reason = "no_retry_needed"
        next_action = "finalize_answer"

        if normalized_current_step >= normalized_max_steps:
            stop_reason = "max_steps_reached"
        elif recommended_action == "stop_no_results":
            stop_reason = "no_results"
        elif sufficient_context:
            stop_reason = "sufficient_context_found"
        elif needs_retry and normalized_current_step < normalized_max_steps:
            continue_loop = True
            stop_reason = ""
            next_action = recommended_action

        LOGGER.info(
            "loop_control_decision",
            extra={
                "current_step": normalized_current_step,
                "max_steps": normalized_max_steps,
                "remaining_steps": remaining_steps,
                "continue_loop": continue_loop,
                "stop_reason": stop_reason,
                "next_action": next_action,
            },
        )
        return {
            "continue_loop": continue_loop,
            "stop_reason": stop_reason,
            "next_action": next_action,
            "remaining_steps": remaining_steps,
        }
    except Exception as error:
        LOGGER.exception(
            "loop_control_failed",
            extra={
                "current_step": current_step,
                "max_steps": max_steps,
                "evaluation_output": evaluation_output,
                "error_message": str(error),
            },
        )
        return {
            "continue_loop": False,
            "stop_reason": f"loop_control_error: {error}",
            "next_action": "finalize_answer",
            "remaining_steps": 0,
        }
