"""Prompt construction utilities for grounded answer generation."""

from __future__ import annotations

import logging
from typing import Any


LOGGER = logging.getLogger(__name__)


def _normalize_text(value: Any) -> str:
    """Normalize arbitrary input into a trimmed string."""
    return str(value).strip()


def _sorted_context_items(compressed_context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort context items by score descending with deterministic tie-breaking."""
    valid_items = [item for item in compressed_context if isinstance(item, dict)]
    return sorted(
        valid_items,
        key=lambda item: (
            -float(item.get("score", 0.0)) if isinstance(item.get("score"), (int, float)) else 0.0,
            _normalize_text(item.get("chunk_id", "")),
        ),
    )


def build_answer_prompt(
    query: str,
    compressed_context: list[dict[str, Any]],
) -> str:
    """Build the final grounded-answer prompt from query and compressed context."""
    try:
        normalized_query = _normalize_text(query)
        sorted_context = _sorted_context_items(compressed_context)

        context_lines: list[str] = []
        skipped_empty_context_entries = 0
        included_context_count = 0

        for item in sorted_context:
            compressed_text = _normalize_text(item.get("compressed_text", ""))
            if not compressed_text:
                skipped_empty_context_entries += 1
                continue

            chunk_id = _normalize_text(item.get("chunk_id", ""))
            if not chunk_id:
                chunk_id = "unknown_chunk"

            context_lines.append(f"[Chunk {chunk_id}]")
            context_lines.append(compressed_text)
            included_context_count += 1

        context_block = "\n".join(context_lines).strip()
        if not context_block:
            context_block = "No retrieved context available."

        prompt = (
            "SYSTEM:\n"
            "You are a grounded retrieval assistant. Answer using only the retrieved context.\n\n"
            "QUERY:\n"
            f"{normalized_query if normalized_query else '(empty query)'}\n\n"
            "CONTEXT:\n"
            f"{context_block}\n\n"
            "RULES:\n"
            "- Use only retrieved context.\n"
            "- Do not assume missing information.\n"
            '- If answer is not found, say exactly: "Not found in retrieved context".\n'
            "- Be concise.\n"
            "- Mention uncertainty if context is weak.\n\n"
            "ANSWER:\n"
        )

        LOGGER.info(
            "answer_prompt_built",
            extra={
                "context_item_count": included_context_count,
                "prompt_length": len(prompt),
                "skipped_empty_context_entries": skipped_empty_context_entries,
            },
        )
        return prompt
    except Exception:
        LOGGER.exception(
            "answer_prompt_build_failed",
            extra={
                "query": query,
                "context_count": len(compressed_context) if isinstance(compressed_context, list) else 0,
            },
        )
        return (
            "SYSTEM:\n"
            "You are a grounded retrieval assistant.\n\n"
            "QUERY:\n"
            "(failed to build query)\n\n"
            "CONTEXT:\n"
            "No retrieved context available.\n\n"
            "RULES:\n"
            '- If answer is not found, say exactly: "Not found in retrieved context".\n\n'
            "ANSWER:\n"
        )
