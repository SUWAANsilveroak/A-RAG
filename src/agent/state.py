"""Agent context tracking for chunk usage across loop steps."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any


LOGGER = logging.getLogger(__name__)


class ContextTracker:
    """Track chunk read usage to avoid redundant context retrieval."""

    def __init__(self) -> None:
        """Initialize empty tracker state."""
        self._read_chunk_ids: set[str] = set()
        self._access_log: dict[str, dict[str, Any]] = {}

    def mark_chunk_read(
        self,
        chunk_id: str,
        step: int,
        relevance_score: float = 0.0,
    ) -> None:
        """Record chunk access metadata and update read status."""
        normalized_chunk_id = str(chunk_id).strip()
        if not normalized_chunk_id:
            LOGGER.warning(
                "context_tracker_invalid_chunk_id",
                extra={"chunk_id": chunk_id, "step": step},
            )
            return

        is_repeat_read = normalized_chunk_id in self._read_chunk_ids
        self._read_chunk_ids.add(normalized_chunk_id)

        existing_record = self._access_log.get(normalized_chunk_id)
        if existing_record is None:
            self._access_log[normalized_chunk_id] = {
                "read_count": 1,
                "last_access_step": int(step),
                "relevance_score": float(relevance_score),
            }
            LOGGER.info(
                "context_tracker_new_chunk_read",
                extra={
                    "chunk_id": normalized_chunk_id,
                    "step": step,
                    "relevance_score": relevance_score,
                },
            )
            return

        existing_record["read_count"] = int(existing_record["read_count"]) + 1
        existing_record["last_access_step"] = int(step)
        if relevance_score != 0.0:
            existing_record["relevance_score"] = float(relevance_score)

        LOGGER.info(
            "context_tracker_repeated_chunk_read",
            extra={
                "chunk_id": normalized_chunk_id,
                "step": step,
                "read_count": existing_record["read_count"],
                "is_repeat_read": is_repeat_read,
            },
        )

    def has_been_read(self, chunk_id: str) -> bool:
        """Return whether the provided chunk id has already been read."""
        normalized_chunk_id = str(chunk_id).strip()
        if not normalized_chunk_id:
            return False
        return normalized_chunk_id in self._read_chunk_ids

    def get_read_chunks(self) -> set[str]:
        """Return a copy of all read chunk ids."""
        return set(self._read_chunk_ids)

    def get_chunk_access_info(self, chunk_id: str) -> dict[str, Any]:
        """Return access metadata for one chunk id, or an empty dict."""
        normalized_chunk_id = str(chunk_id).strip()
        if not normalized_chunk_id:
            return {}

        record = self._access_log.get(normalized_chunk_id)
        if record is None:
            return {}
        return deepcopy(record)

    def reset(self) -> None:
        """Clear all tracked chunk usage state."""
        self._read_chunk_ids.clear()
        self._access_log.clear()
        LOGGER.info("context_tracker_reset")
