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

    def get_access_log(self) -> dict[str, dict[str, Any]]:
        """Return a copy of the full chunk access log."""
        return deepcopy(self._access_log)

    def reset(self) -> None:
        """Clear all tracked chunk usage state."""
        self._read_chunk_ids.clear()
        self._access_log.clear()
        LOGGER.info("context_tracker_reset")


class AgentMemory:
    """Short-term execution memory for the agent loop."""

    def __init__(self) -> None:
        """Initialize an empty memory state."""
        self._current_query: str = ""
        self._retrieved_chunks: list[str] = []
        self._retrieved_chunk_ids: set[str] = set()
        self._reasoning_steps: list[str] = []
        self._tool_history: list[dict[str, Any]] = []

    def set_query(self, query: str) -> None:
        """Set the current user query, including empty-query updates."""
        normalized_query = str(query).strip()
        self._current_query = normalized_query

        if not normalized_query:
            LOGGER.warning("agent_memory_query_updated_empty")
            return

        LOGGER.info("agent_memory_query_updated", extra={"query": normalized_query})

    def get_query(self) -> str:
        """Return the current query."""
        return self._current_query

    def add_retrieved_chunk(self, chunk_id: str) -> None:
        """Add a retrieved chunk id while preserving order and avoiding duplicates."""
        normalized_chunk_id = str(chunk_id).strip()
        if not normalized_chunk_id:
            LOGGER.warning("agent_memory_invalid_chunk_id")
            return

        if normalized_chunk_id in self._retrieved_chunk_ids:
            LOGGER.info("agent_memory_chunk_duplicate_ignored", extra={"chunk_id": normalized_chunk_id})
            return

        self._retrieved_chunk_ids.add(normalized_chunk_id)
        self._retrieved_chunks.append(normalized_chunk_id)
        LOGGER.info("agent_memory_chunk_added", extra={"chunk_id": normalized_chunk_id})

    def get_retrieved_chunks(self) -> list[str]:
        """Return retrieved chunk ids in insertion order."""
        return list(self._retrieved_chunks)

    def add_reasoning_step(self, step: str) -> None:
        """Append one reasoning step while preserving sequence."""
        normalized_step = str(step).strip()
        if not normalized_step:
            LOGGER.warning("agent_memory_empty_reasoning_step")
            return

        self._reasoning_steps.append(normalized_step)
        LOGGER.info("agent_memory_reasoning_step_added", extra={"step": normalized_step})

    def get_reasoning_steps(self) -> list[str]:
        """Return reasoning steps in order."""
        return list(self._reasoning_steps)

    def add_tool_call(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        output_summary: str,
    ) -> None:
        """Append one tool-call history item with normalized fields."""
        normalized_tool_name = str(tool_name).strip()
        normalized_output_summary = str(output_summary).strip()

        if not normalized_tool_name:
            LOGGER.warning("agent_memory_empty_tool_name")
            return

        safe_input_data: dict[str, Any]
        if isinstance(input_data, dict):
            safe_input_data = deepcopy(input_data)
        else:
            safe_input_data = {"raw_input": str(input_data)}

        tool_record = {
            "tool_name": normalized_tool_name,
            "input_data": safe_input_data,
            "output_summary": normalized_output_summary,
        }
        self._tool_history.append(tool_record)
        LOGGER.info("agent_memory_tool_call_added", extra={"tool_name": normalized_tool_name})

    def get_tool_history(self) -> list[dict[str, Any]]:
        """Return tool-call history in order."""
        return deepcopy(self._tool_history)

    def reset(self) -> None:
        """Clear all short-term memory state."""
        self._current_query = ""
        self._retrieved_chunks.clear()
        self._retrieved_chunk_ids.clear()
        self._reasoning_steps.clear()
        self._tool_history.clear()
        LOGGER.info("agent_memory_reset")


class AgentStateManager:
    """Centralized state updater for context tracker and short-term memory."""

    def __init__(self) -> None:
        """Initialize managed tracker and memory components."""
        self._context_tracker = ContextTracker()
        self._agent_memory = AgentMemory()

    def update_after_chunk_read(
        self,
        chunk_id: str,
        step: int,
        relevance_score: float = 0.0,
    ) -> None:
        """Apply state updates after a chunk-read event."""
        try:
            self._context_tracker.mark_chunk_read(
                chunk_id=chunk_id,
                step=step,
                relevance_score=relevance_score,
            )
            self._agent_memory.add_retrieved_chunk(chunk_id)
            LOGGER.info(
                "agent_state_manager_chunk_read_updated",
                extra={"chunk_id": chunk_id, "step": step, "relevance_score": relevance_score},
            )
        except Exception:
            LOGGER.exception(
                "agent_state_manager_chunk_read_update_failed",
                extra={"chunk_id": chunk_id, "step": step},
            )
            raise

    def update_after_tool_call(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        output_summary: str,
    ) -> None:
        """Apply state updates after a tool call."""
        try:
            self._agent_memory.add_tool_call(
                tool_name=tool_name,
                input_data=input_data,
                output_summary=output_summary,
            )
            LOGGER.info(
                "agent_state_manager_tool_call_updated",
                extra={"tool_name": tool_name},
            )
        except Exception:
            LOGGER.exception(
                "agent_state_manager_tool_call_update_failed",
                extra={"tool_name": tool_name},
            )
            raise

    def update_after_reasoning_step(self, reasoning_step: str) -> None:
        """Apply state updates after a reasoning step."""
        try:
            self._agent_memory.add_reasoning_step(reasoning_step)
            LOGGER.info(
                "agent_state_manager_reasoning_updated",
                extra={"reasoning_step": reasoning_step},
            )
        except Exception:
            LOGGER.exception("agent_state_manager_reasoning_update_failed")
            raise

    def update_after_query(self, query: str) -> None:
        """Apply state updates after receiving a new query."""
        try:
            self._agent_memory.set_query(query)
            LOGGER.info(
                "agent_state_manager_query_updated",
                extra={"query": query},
            )
        except Exception:
            LOGGER.exception("agent_state_manager_query_update_failed")
            raise

    def get_full_state(self) -> dict[str, Any]:
        """Return the merged state from tracker and short-term memory."""
        try:
            return {
                "current_query": self._agent_memory.get_query(),
                "retrieved_chunks": self._agent_memory.get_retrieved_chunks(),
                "reasoning_steps": self._agent_memory.get_reasoning_steps(),
                "tool_history": self._agent_memory.get_tool_history(),
                "read_chunk_ids": sorted(self._context_tracker.get_read_chunks()),
                "access_log": self._context_tracker.get_access_log(),
            }
        except Exception:
            LOGGER.exception("agent_state_manager_get_full_state_failed")
            raise

    def reset_all(self) -> None:
        """Reset both tracker and memory state components."""
        try:
            self._context_tracker.reset()
            self._agent_memory.reset()
            LOGGER.info("agent_state_manager_reset_all")
        except Exception:
            LOGGER.exception("agent_state_manager_reset_all_failed")
            raise
