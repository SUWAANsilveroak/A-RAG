"""Tool wrapper for reading full chunk text by chunk id."""

from __future__ import annotations

import logging
from typing import Any


LOGGER = logging.getLogger(__name__)
_CHUNK_READ_TOOL_NAME = "chunk_read"


def _build_chunk_read_error_response(error_message: str) -> dict[str, Any]:
    """Return the standardized error payload for chunk read."""
    return {
        "tool_name": _CHUNK_READ_TOOL_NAME,
        "status": "error",
        "error_message": error_message,
        "results": [],
    }


def run_chunk_read_tool(
    chunk_ids: list[str],
    chunks: list[dict[str, Any]],
    read_chunk_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Return full chunk text for requested ids with read-status tracking."""
    LOGGER.info(
        "chunk_read_tool_started",
        extra={
            "tool_name": _CHUNK_READ_TOOL_NAME,
            "requested_chunk_ids": chunk_ids,
        },
    )

    try:
        chunk_map: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            if not chunk_id:
                continue
            chunk_map[chunk_id] = chunk

        tracked_read_ids = read_chunk_ids if read_chunk_ids is not None else set()

        results: list[dict[str, Any]] = []
        found_count = 0
        previously_read_count = 0
        missing_count = 0

        for requested_chunk_id in chunk_ids:
            normalized_chunk_id = str(requested_chunk_id).strip()
            if normalized_chunk_id not in chunk_map:
                missing_count += 1
                results.append(
                    {
                        "chunk_id": normalized_chunk_id,
                        "status": "not_found",
                        "text": "",
                        "position": -1,
                    }
                )
                continue

            chunk_record = chunk_map[normalized_chunk_id]
            chunk_text = str(chunk_record.get("text", ""))
            chunk_position = int(chunk_record.get("position", -1))

            if normalized_chunk_id in tracked_read_ids:
                status = "previously_read"
                previously_read_count += 1
            else:
                status = "new"
                tracked_read_ids.add(normalized_chunk_id)
                found_count += 1

            results.append(
                {
                    "chunk_id": normalized_chunk_id,
                    "status": status,
                    "text": chunk_text,
                    "position": chunk_position,
                }
            )

    except Exception as error:
        error_message = str(error)
        LOGGER.exception(
            "chunk_read_tool_failed",
            extra={
                "tool_name": _CHUNK_READ_TOOL_NAME,
                "requested_chunk_ids": chunk_ids,
                "error_message": error_message,
            },
        )
        return _build_chunk_read_error_response(error_message)

    LOGGER.info(
        "chunk_read_tool_completed",
        extra={
            "tool_name": _CHUNK_READ_TOOL_NAME,
            "requested_chunk_ids": chunk_ids,
            "found_chunks": found_count,
            "previously_read_chunks": previously_read_count,
            "missing_chunks": missing_count,
        },
    )
    return {
        "tool_name": _CHUNK_READ_TOOL_NAME,
        "status": "success",
        "requested_chunk_ids": [str(chunk_id) for chunk_id in chunk_ids],
        "results": results,
    }
