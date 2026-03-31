"""LLM answer generation via LiteLLM with provider-agnostic routing."""

from __future__ import annotations

import logging
import time
from typing import Any


LOGGER = logging.getLogger(__name__)
_SUPPORTED_PROVIDERS = {"ollama", "groq", "openrouter"}


def _call_litellm_completion(**kwargs: Any) -> Any:
    """Call LiteLLM completion lazily to keep import failures isolated."""
    try:
        from litellm import completion  # type: ignore
    except Exception as error:
        raise RuntimeError("LiteLLM is unavailable in the current environment.") from error
    return completion(**kwargs)


def _build_litellm_model_name(provider: str, model_name: str) -> str:
    """Build provider-qualified model name expected by LiteLLM."""
    normalized_provider = provider.strip().lower()
    normalized_model_name = model_name.strip()
    if "/" in normalized_model_name:
        return normalized_model_name
    return f"{normalized_provider}/{normalized_model_name}"


def _extract_token_usage(response: Any) -> dict[str, int]:
    """Extract token usage from LiteLLM response into a stable dictionary."""
    usage = None
    if isinstance(response, dict):
        usage = response.get("usage")
    else:
        usage = getattr(response, "usage", None)

    def _usage_value(key: str) -> int:
        if usage is None:
            return 0
        if isinstance(usage, dict):
            value = usage.get(key, 0)
        else:
            value = getattr(usage, key, 0)
        return int(value) if isinstance(value, (int, float)) else 0

    return {
        "prompt_tokens": _usage_value("prompt_tokens"),
        "completion_tokens": _usage_value("completion_tokens"),
        "total_tokens": _usage_value("total_tokens"),
    }


def _extract_answer_text(response: Any) -> str:
    """Extract generated answer text from first response choice."""
    choices = []
    if isinstance(response, dict):
        choices = response.get("choices", []) or []
    else:
        choices = getattr(response, "choices", []) or []

    if not choices:
        return ""

    first_choice = choices[0]
    message = None
    if isinstance(first_choice, dict):
        message = first_choice.get("message")
    else:
        message = getattr(first_choice, "message", None)

    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "") if message is not None else ""

    return str(content).strip()


def generate_answer(
    prompt: str,
    model_name: str,
    provider: str = "ollama",
    temperature: float = 0.1,
    max_tokens: int = 500,
) -> dict[str, Any]:
    """Generate an answer from prompt using configured provider through LiteLLM."""
    normalized_provider = str(provider).strip().lower()
    normalized_model_name = str(model_name).strip()
    normalized_prompt = str(prompt).strip()

    if not normalized_prompt:
        error_message = "Prompt must not be empty."
        LOGGER.warning(
            "answer_generation_invalid_prompt",
            extra={"provider": normalized_provider, "model_name": normalized_model_name},
        )
        return {
            "status": "error",
            "provider": normalized_provider,
            "model_name": normalized_model_name,
            "answer": "",
            "error_message": error_message,
        }

    if normalized_provider not in _SUPPORTED_PROVIDERS:
        error_message = f"Unsupported provider: {normalized_provider}"
        LOGGER.warning(
            "answer_generation_invalid_provider",
            extra={"provider": normalized_provider, "model_name": normalized_model_name},
        )
        return {
            "status": "error",
            "provider": normalized_provider,
            "model_name": normalized_model_name,
            "answer": "",
            "error_message": error_message,
        }

    if not normalized_model_name:
        error_message = "Model name must not be empty."
        LOGGER.warning("answer_generation_invalid_model_name", extra={"provider": normalized_provider})
        return {
            "status": "error",
            "provider": normalized_provider,
            "model_name": normalized_model_name,
            "answer": "",
            "error_message": error_message,
        }

    if max_tokens <= 0:
        error_message = "max_tokens must be greater than zero."
        LOGGER.warning(
            "answer_generation_invalid_max_tokens",
            extra={"provider": normalized_provider, "model_name": normalized_model_name, "max_tokens": max_tokens},
        )
        return {
            "status": "error",
            "provider": normalized_provider,
            "model_name": normalized_model_name,
            "answer": "",
            "error_message": error_message,
        }

    litellm_model_name = _build_litellm_model_name(normalized_provider, normalized_model_name)
    start = time.perf_counter()

    try:
        response = _call_litellm_completion(
            model=litellm_model_name,
            messages=[{"role": "user", "content": normalized_prompt}],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
    except Exception as error:
        LOGGER.error(
            "answer_generation_failed",
            extra={
                "provider": normalized_provider,
                "model_name": normalized_model_name,
                "error_message": str(error),
            },
        )
        return {
            "status": "error",
            "provider": normalized_provider,
            "model_name": normalized_model_name,
            "answer": "",
            "error_message": str(error),
        }

    latency_seconds = float(time.perf_counter() - start)
    answer_text = _extract_answer_text(response)
    if not answer_text:
        error_message = "Model returned an empty answer."
        LOGGER.warning(
            "answer_generation_empty_response",
            extra={
                "provider": normalized_provider,
                "model_name": normalized_model_name,
            },
        )
        return {
            "status": "error",
            "provider": normalized_provider,
            "model_name": normalized_model_name,
            "answer": "",
            "error_message": error_message,
        }

    token_usage = _extract_token_usage(response)
    LOGGER.info(
        "answer_generation_completed",
        extra={
            "provider": normalized_provider,
            "model_name": normalized_model_name,
            "latency_seconds": latency_seconds,
            "token_usage": token_usage,
        },
    )
    return {
        "status": "success",
        "provider": normalized_provider,
        "model_name": normalized_model_name,
        "answer": answer_text,
        "token_usage": token_usage,
        "latency_seconds": latency_seconds,
    }


def _extract_supporting_chunks(compressed_context: list[dict[str, Any]]) -> list[str]:
    """Extract ordered unique chunk ids from compressed context."""
    supporting_chunks: list[str] = []
    seen_chunk_ids: set[str] = set()

    for item in compressed_context:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        if not chunk_id or chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        supporting_chunks.append(chunk_id)

    return supporting_chunks


def _calculate_confidence(status: str, answer: str, supporting_chunks: list[str]) -> float:
    """Calculate confidence score in range [0.0, 1.0] from output signals."""
    normalized_status = str(status).strip().lower()
    has_answer = bool(str(answer).strip())
    has_supporting_chunks = len(supporting_chunks) > 0

    if normalized_status != "success":
        return 0.0
    if has_answer and has_supporting_chunks:
        return 0.9
    if has_answer and not has_supporting_chunks:
        return 0.5
    if (not has_answer) and has_supporting_chunks:
        return 0.3
    return 0.1


def format_final_output(
    answer_response: dict[str, Any],
    compressed_context: list[dict[str, Any]],
) -> dict[str, Any]:
    """Format standardized final answer output with supporting chunks and confidence."""
    try:
        safe_answer_response = answer_response if isinstance(answer_response, dict) else {}
        safe_context = compressed_context if isinstance(compressed_context, list) else []

        status = str(safe_answer_response.get("status", "error")).strip() or "error"
        answer = str(safe_answer_response.get("answer", "")).strip()
        provider = str(safe_answer_response.get("provider", "")).strip()
        model_name = str(safe_answer_response.get("model_name", "")).strip()

        raw_token_usage = safe_answer_response.get("token_usage", {})
        token_usage = raw_token_usage if isinstance(raw_token_usage, dict) else {}

        latency_value = safe_answer_response.get("latency_seconds", 0.0)
        latency_seconds = float(latency_value) if isinstance(latency_value, (int, float)) else 0.0

        supporting_chunks = _extract_supporting_chunks(safe_context)
        confidence = _calculate_confidence(status, answer, supporting_chunks)

        LOGGER.info(
            "answer_output_formatted",
            extra={
                "status": status,
                "answer_length": len(answer),
                "supporting_chunk_count": len(supporting_chunks),
                "confidence": confidence,
            },
        )
        return {
            "status": status,
            "answer": answer,
            "supporting_chunks": supporting_chunks,
            "confidence": confidence,
            "provider": provider,
            "model_name": model_name,
            "token_usage": token_usage,
            "latency_seconds": latency_seconds,
        }
    except Exception as error:
        LOGGER.exception(
            "answer_output_formatting_failed",
            extra={
                "answer_response": answer_response,
                "context_count": len(compressed_context) if isinstance(compressed_context, list) else 0,
            },
        )
        return {
            "status": "error",
            "answer": "",
            "supporting_chunks": [],
            "confidence": 0.0,
            "provider": "",
            "model_name": "",
            "token_usage": {},
            "latency_seconds": 0.0,
            "error_message": str(error),
        }


def build_fallback_answer(
    compressed_context: list[dict[str, Any]],
    error_message: str = "",
) -> dict[str, Any]:
    """Build a grounded extractive fallback answer when model generation is unavailable."""
    snippets: list[str] = []
    for item in compressed_context:
        if not isinstance(item, dict):
            continue
        compressed_text = str(item.get("compressed_text", "")).strip()
        if not compressed_text:
            continue
        snippets.append(compressed_text)
        if len(snippets) >= 2:
            break

    if snippets:
        answer_text = " ".join(snippets)
    else:
        answer_text = "Not found in retrieved context"

    if len(answer_text) > 500:
        answer_text = answer_text[:497].rstrip() + "..."

    return {
        "status": "success",
        "provider": "fallback",
        "model_name": "extractive-context",
        "answer": answer_text,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "latency_seconds": 0.0,
        "fallback_reason": str(error_message).strip(),
    }
