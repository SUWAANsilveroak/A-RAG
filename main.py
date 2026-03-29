"""Application entry point and pipeline orchestration for A-RAG."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Final

from src.agent.answer_generator import format_final_output, generate_answer
from src.agent.loop import evaluate_retrieval, execute_retrieval, select_tool, should_continue_loop
from src.agent.planner import plan_query
from src.agent.validator import (
    check_completeness,
    decide_retry_action,
    detect_conflicts,
    validate_grounding,
)
from src.evaluation.metrics import (
    calculate_answer_metrics,
    calculate_retrieval_metrics,
    calculate_system_metrics,
)
from src.prompts import build_answer_prompt
from src.utils.compression import compress_snippets, select_top_snippets
from src.utils.logger import PipelineLogger


PROJECT_NAME: Final[str] = "A-RAG"
LOGGER = logging.getLogger(__name__)
_REQUIRED_RESOURCE_KEYS = {
    "chunks",
    "model",
    "faiss_index",
    "metadata",
    "read_chunk_ids",
    "model_name",
    "provider",
}
_RETRY_STRATEGY_TO_RETRIEVAL = {
    "retry_with_semantic_search": "semantic_search",
    "retry_with_hybrid_search": "hybrid_search",
    "retry_with_chunk_read": "chunk_read",
}


def configure_logging() -> None:
    """Configure a simple JSON-style logger for local development."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def bootstrap_status() -> dict[str, str]:
    """Return a structured snapshot of the bootstrap state."""
    return {
        "project": PROJECT_NAME,
        "status": "initialized",
        "root": str(Path(__file__).resolve().parent),
    }


def _normalize_retrieval_results_for_snippets(
    retrieval_output: dict[str, Any],
) -> list[dict[str, Any]]:
    """Normalize retrieval output results to snippet-selection input format."""
    safe_output = retrieval_output if isinstance(retrieval_output, dict) else {}
    raw_results = safe_output.get("results", [])
    if not isinstance(raw_results, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        if not chunk_id:
            continue

        snippet = str(item.get("snippet", item.get("text", ""))).strip()
        if not snippet:
            continue

        score = 0.0
        for key in ("combined_score", "score", "semantic_score", "keyword_score", "relevance_score"):
            value = item.get(key)
            if isinstance(value, (int, float)):
                score = float(value)
                break

        normalized.append(
            {
                "chunk_id": chunk_id,
                "score": score,
                "snippet": snippet,
            }
        )
    return normalized


def _validate_pipeline_input(query: str, resources: dict[str, Any]) -> str:
    """Return validation error message for pipeline input, empty string if valid."""
    normalized_query = str(query).strip()
    if not normalized_query:
        return "Query must not be empty."

    if not isinstance(resources, dict):
        return "Resources must be a dictionary."

    missing_keys = sorted(key for key in _REQUIRED_RESOURCE_KEYS if key not in resources)
    if missing_keys:
        return f"Missing required resources: {', '.join(missing_keys)}"

    return ""


def _build_chunk_read_ids(retrieval_output: dict[str, Any], limit: int = 5) -> list[str]:
    """Extract chunk ids from retrieval output for chunk-read retry strategy."""
    safe_output = retrieval_output if isinstance(retrieval_output, dict) else {}
    raw_results = safe_output.get("results", [])
    if not isinstance(raw_results, list):
        return []

    chunk_ids: list[str] = []
    seen: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunk_ids.append(chunk_id)
        if len(chunk_ids) >= max(limit, 1):
            break
    return chunk_ids


def _apply_retry_strategy_override(
    planner_output: dict[str, Any],
    strategy_override: str | None,
    previous_retrieval_output: dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply retry strategy override by updating planner output strategy/chunk ids."""
    safe_planner = dict(planner_output if isinstance(planner_output, dict) else {})
    if not strategy_override:
        return safe_planner

    retrieval_strategy = _RETRY_STRATEGY_TO_RETRIEVAL.get(strategy_override.strip(), "")
    if not retrieval_strategy:
        return safe_planner

    safe_planner["retrieval_strategy"] = retrieval_strategy
    safe_planner["reason"] = (
        f"{str(safe_planner.get('reason', '')).strip()} Retry override applied: {strategy_override}."
    ).strip()

    if retrieval_strategy == "chunk_read":
        safe_planner["chunk_ids"] = _build_chunk_read_ids(previous_retrieval_output or {})
    return safe_planner


def _run_single_attempt(
    query: str,
    resources: dict[str, Any],
    max_steps: int,
    max_retries: int,
    current_retry_count: int,
    pipeline_logger: PipelineLogger,
    attempt_number: int,
    strategy_override: str | None = None,
    previous_retrieval_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one pipeline attempt and return all intermediate artifacts."""
    safe_resources = resources if isinstance(resources, dict) else {}
    planner_output = plan_query(query)
    planner_output = _apply_retry_strategy_override(planner_output, strategy_override, previous_retrieval_output)
    tool_selection_output = select_tool(planner_output)

    retrieval_output = execute_retrieval(tool_selection_output, safe_resources)
    pipeline_logger.log_tool_execution(
        tool_name=str(retrieval_output.get("tool_name", "")),
        tool_input=tool_selection_output.get("tool_input", {}),
        result_count=int(retrieval_output.get("result_count", 0))
        if isinstance(retrieval_output.get("result_count", 0), (int, float))
        else 0,
        execution_summary=str(retrieval_output.get("execution_summary", "")),
    )

    evaluation_output = evaluate_retrieval(retrieval_output)
    _ = should_continue_loop(current_step=attempt_number, max_steps=max_steps, evaluation_output=evaluation_output)

    snippet_candidates = _normalize_retrieval_results_for_snippets(retrieval_output)
    selection_top_k = tool_selection_output.get("tool_input", {}).get("top_k", 5)
    top_k = int(selection_top_k) if isinstance(selection_top_k, (int, float)) else 5
    selected_snippets = select_top_snippets(snippet_candidates, top_k=max(top_k, 1))
    compressed_context = compress_snippets(selected_snippets)

    answer_prompt = build_answer_prompt(query, compressed_context)
    answer_response = generate_answer(
        prompt=answer_prompt,
        model_name=str(safe_resources.get("model_name", "")).strip(),
        provider=str(safe_resources.get("provider", "ollama")).strip() or "ollama",
    )
    final_output = format_final_output(answer_response, compressed_context)

    grounding_result = validate_grounding(final_output, compressed_context)
    conflict_result = detect_conflicts(compressed_context)
    completeness_result = check_completeness(query, final_output, compressed_context)
    validation_results = {
        "grounding_result": grounding_result,
        "conflict_result": conflict_result,
        "completeness_result": completeness_result,
    }
    pipeline_logger.log_validation(
        grounding_result=grounding_result,
        conflict_result=conflict_result,
        completeness_result=completeness_result,
    )

    retry_result = decide_retry_action(
        grounding_result=grounding_result,
        conflict_result=conflict_result,
        completeness_result=completeness_result,
        max_retries=max_retries,
        current_retry_count=current_retry_count,
    )

    retrieval_metrics = calculate_retrieval_metrics(
        retrieval_output.get("results", []) if isinstance(retrieval_output.get("results", []), list) else []
    )
    answer_metrics = calculate_answer_metrics(final_output, grounding_result, completeness_result)
    system_metrics = calculate_system_metrics(
        token_usage=final_output.get("token_usage", {})
        if isinstance(final_output.get("token_usage", {}), dict)
        else {},
        latency_seconds=float(final_output.get("latency_seconds", 0.0))
        if isinstance(final_output.get("latency_seconds", 0.0), (int, float))
        else 0.0,
        retry_count=current_retry_count,
    )
    metrics = {
        "retrieval_metrics": retrieval_metrics,
        "answer_metrics": answer_metrics,
        "system_metrics": system_metrics,
    }
    pipeline_logger.log_metrics(
        retrieval_metrics=retrieval_metrics,
        answer_metrics=answer_metrics,
        system_metrics=system_metrics,
    )

    return {
        "attempt_number": attempt_number,
        "planner_output": planner_output,
        "tool_selection_output": tool_selection_output,
        "retrieval_output": retrieval_output,
        "evaluation_output": evaluation_output,
        "compressed_context": compressed_context,
        "answer_response": answer_response,
        "final_output": final_output,
        "validation_results": validation_results,
        "retry_result": retry_result,
        "metrics": metrics,
    }


def run_a_rag_pipeline(
    query: str,
    resources: dict[str, Any],
    max_steps: int = 5,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Run single-pass A-RAG pipeline orchestration using existing project components."""
    pipeline_logger = PipelineLogger()
    normalized_query = str(query).strip()

    try:
        LOGGER.info(
            "a_rag_pipeline_started",
            extra={"query": normalized_query, "max_steps": max_steps, "max_retries": max_retries},
        )
        validation_error = _validate_pipeline_input(normalized_query, resources)
        if validation_error:
            pipeline_logger.log_error("run_a_rag_pipeline", validation_error)
            return {
                "status": "error",
                "query": normalized_query,
                "error_message": validation_error,
            }

        safe_resources = resources if isinstance(resources, dict) else {}
        pipeline_logger.log_query(normalized_query)

        attempt = _run_single_attempt(
            query=normalized_query,
            resources=safe_resources,
            max_steps=max_steps,
            max_retries=max_retries,
            current_retry_count=0,
            pipeline_logger=pipeline_logger,
            attempt_number=1,
        )

        planner_output = attempt["planner_output"]
        tool_selection_output = attempt["tool_selection_output"]
        retrieval_output = attempt["retrieval_output"]
        evaluation_output = attempt["evaluation_output"]

        loop_decision = should_continue_loop(current_step=1, max_steps=max_steps, evaluation_output=evaluation_output)
        if bool(loop_decision.get("continue_loop", False)):
            LOGGER.info(
                "a_rag_pipeline_retry_recommended",
                extra={"next_action": loop_decision.get("next_action", ""), "remaining_steps": loop_decision.get("remaining_steps", 0)},
            )

        output = {
            "query": normalized_query,
            "planner_output": planner_output,
            "tool_selection_output": tool_selection_output,
            "retrieval_output": retrieval_output,
            "evaluation_output": evaluation_output,
            "compressed_context": attempt["compressed_context"],
            "answer_response": attempt["answer_response"],
            "final_output": attempt["final_output"],
            "validation_results": attempt["validation_results"],
            "retry_result": attempt["retry_result"],
            "metrics": attempt["metrics"],
        }
        LOGGER.info(
            "a_rag_pipeline_completed",
            extra={
                "query": normalized_query,
                "final_status": output["final_output"].get("status", "unknown"),
                "should_retry": output["retry_result"].get("should_retry", False),
            },
        )
        return output
    except Exception as error:
        error_message = str(error)
        pipeline_logger.log_error("run_a_rag_pipeline", error_message)
        LOGGER.error(
            "a_rag_pipeline_failed",
            extra={"query": normalized_query, "error_message": error_message},
        )
        return {
            "status": "error",
            "query": normalized_query,
            "error_message": error_message,
        }


def run_retry_pipeline(
    query: str,
    resources: dict[str, Any],
    max_steps: int = 5,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Run retry-enabled A-RAG pipeline and preserve per-attempt artifacts."""
    pipeline_logger = PipelineLogger()
    normalized_query = str(query).strip()
    try:
        LOGGER.info(
            "a_rag_retry_pipeline_started",
            extra={"query": normalized_query, "max_steps": max_steps, "max_retries": max_retries},
        )
        validation_error = _validate_pipeline_input(normalized_query, resources)
        if validation_error:
            pipeline_logger.log_error("run_retry_pipeline", validation_error)
            return {"status": "error", "query": normalized_query, "error_message": validation_error}

        safe_resources = resources if isinstance(resources, dict) else {}
        pipeline_logger.log_query(normalized_query)

        attempts: list[dict[str, Any]] = []
        retry_history: list[dict[str, Any]] = []
        retry_count = 0
        strategy_override: str | None = None
        previous_retrieval_output: dict[str, Any] | None = None
        previous_override: str | None = None

        max_attempts = max(int(max_steps), 1)
        normalized_max_retries = max(int(max_retries), 0)

        for attempt_number in range(1, max_attempts + 1):
            attempt = _run_single_attempt(
                query=normalized_query,
                resources=safe_resources,
                max_steps=max_steps,
                max_retries=normalized_max_retries,
                current_retry_count=retry_count,
                pipeline_logger=pipeline_logger,
                attempt_number=attempt_number,
                strategy_override=strategy_override,
                previous_retrieval_output=previous_retrieval_output,
            )

            attempts.append(
                {
                    "attempt_number": attempt["attempt_number"],
                    "planner_output": attempt["planner_output"],
                    "tool_selection_output": attempt["tool_selection_output"],
                    "retrieval_output": attempt["retrieval_output"],
                    "evaluation_output": attempt["evaluation_output"],
                    "final_output": attempt["final_output"],
                    "validation_results": attempt["validation_results"],
                    "retry_result": attempt["retry_result"],
                }
            )

            retry_result = attempt["retry_result"]
            should_retry = bool(retry_result.get("should_retry", False))
            retry_reason = str(retry_result.get("retry_reason", "")).strip()
            recommended_strategy = str(retry_result.get("recommended_strategy", "")).strip()

            retry_history.append(
                {
                    "attempt_number": attempt_number,
                    "should_retry": should_retry,
                    "retry_reason": retry_reason,
                    "recommended_strategy": recommended_strategy,
                }
            )

            LOGGER.info(
                "a_rag_retry_attempt_completed",
                extra={
                    "attempt_number": attempt_number,
                    "should_retry": should_retry,
                    "retry_reason": retry_reason,
                    "recommended_strategy": recommended_strategy,
                },
            )

            if not should_retry:
                break
            if retry_count >= normalized_max_retries:
                LOGGER.info("a_rag_retry_limit_reached", extra={"retry_count": retry_count})
                break
            if recommended_strategy not in _RETRY_STRATEGY_TO_RETRIEVAL:
                LOGGER.info(
                    "a_rag_retry_no_useful_strategy",
                    extra={"recommended_strategy": recommended_strategy},
                )
                break
            if previous_override and recommended_strategy == previous_override:
                LOGGER.info(
                    "a_rag_retry_repeated_strategy_stop",
                    extra={"recommended_strategy": recommended_strategy},
                )
                break

            retry_count += 1
            previous_override = recommended_strategy
            strategy_override = recommended_strategy
            previous_retrieval_output = attempt["retrieval_output"]
            LOGGER.info(
                "a_rag_retry_started",
                extra={
                    "retry_count": retry_count,
                    "retry_reason": retry_reason,
                    "strategy_override": strategy_override,
                },
            )

        final_attempt = attempts[-1] if attempts else {}
        final_status = (
            str(final_attempt.get("final_output", {}).get("status", "error"))
            if isinstance(final_attempt, dict)
            else "error"
        )
        LOGGER.info(
            "a_rag_retry_pipeline_completed",
            extra={"query": normalized_query, "retry_count": retry_count, "final_status": final_status},
        )
        return {
            "query": normalized_query,
            "attempts": attempts,
            "final_attempt": final_attempt,
            "retry_history": retry_history,
            "retry_count": retry_count,
            "final_status": final_status,
        }
    except Exception as error:
        error_message = str(error)
        pipeline_logger.log_error("run_retry_pipeline", error_message)
        LOGGER.error(
            "a_rag_retry_pipeline_failed",
            extra={"query": normalized_query, "error_message": error_message},
        )
        return {"status": "error", "query": normalized_query, "error_message": error_message}


def main() -> None:
    """Log the current bootstrap status."""
    configure_logging()
    logging.info(json.dumps(bootstrap_status()))


if __name__ == "__main__":
    main()
