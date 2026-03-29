"""Structured pipeline logging utilities."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
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


class PipelineLogger:
    """Capture structured pipeline logs with optional JSON export."""

    def __init__(self) -> None:
        """Initialize empty in-memory log collections."""
        self.logs: dict[str, list[dict[str, Any]]] = {
            "queries": [],
            "tool_executions": [],
            "validations": [],
            "metrics": [],
            "errors": [],
        }
        self._dedupe_signatures: dict[str, set[str]] = {
            "queries": set(),
            "tool_executions": set(),
            "validations": set(),
            "metrics": set(),
            "errors": set(),
        }

    def log_query(self, query: str) -> None:
        """Log a query event with timestamp."""
        try:
            clean_query = str(query).strip()
            entry = {"query": clean_query}
            self._append_unique("queries", entry)
            LOGGER.info(
                "pipeline_log_query",
                extra={"query_length": len(clean_query), "stored": bool(clean_query)},
            )
        except Exception as error:
            LOGGER.exception("pipeline_log_query_failed", extra={"error_message": str(error)})

    def log_tool_execution(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        result_count: int,
        execution_summary: str,
    ) -> None:
        """Log a retrieval/tool execution event."""
        try:
            entry = {
                "tool_name": str(tool_name).strip(),
                "tool_input": self._sanitize_value(tool_input if isinstance(tool_input, dict) else {}),
                "result_count": max(int(result_count), 0) if isinstance(result_count, (int, float)) else 0,
                "execution_summary": str(execution_summary).strip(),
            }
            self._append_unique("tool_executions", entry)
            LOGGER.info(
                "pipeline_log_tool_execution",
                extra={
                    "tool_name": entry["tool_name"],
                    "result_count": entry["result_count"],
                },
            )
        except Exception as error:
            LOGGER.exception("pipeline_log_tool_execution_failed", extra={"error_message": str(error)})

    def log_validation(
        self,
        grounding_result: dict[str, Any],
        conflict_result: dict[str, Any],
        completeness_result: dict[str, Any],
    ) -> None:
        """Log validation outputs for grounding/conflict/completeness."""
        try:
            entry = {
                "grounding_result": self._sanitize_value(grounding_result if isinstance(grounding_result, dict) else {}),
                "conflict_result": self._sanitize_value(conflict_result if isinstance(conflict_result, dict) else {}),
                "completeness_result": self._sanitize_value(
                    completeness_result if isinstance(completeness_result, dict) else {}
                ),
            }
            self._append_unique("validations", entry)
            LOGGER.info("pipeline_log_validation")
        except Exception as error:
            LOGGER.exception("pipeline_log_validation_failed", extra={"error_message": str(error)})

    def log_metrics(
        self,
        retrieval_metrics: dict[str, Any],
        answer_metrics: dict[str, Any],
        system_metrics: dict[str, Any],
    ) -> None:
        """Log retrieval/answer/system metrics as a single event."""
        try:
            entry = {
                "retrieval_metrics": self._sanitize_value(
                    retrieval_metrics if isinstance(retrieval_metrics, dict) else {}
                ),
                "answer_metrics": self._sanitize_value(answer_metrics if isinstance(answer_metrics, dict) else {}),
                "system_metrics": self._sanitize_value(system_metrics if isinstance(system_metrics, dict) else {}),
            }
            self._append_unique("metrics", entry)
            LOGGER.info("pipeline_log_metrics")
        except Exception as error:
            LOGGER.exception("pipeline_log_metrics_failed", extra={"error_message": str(error)})

    def log_error(self, component: str, error_message: str) -> None:
        """Log an error event for a pipeline component."""
        try:
            clean_component = str(component).strip() or "unknown_component"
            clean_message = str(error_message).strip() or "unspecified_error"
            entry = {
                "component": clean_component,
                "error_message": clean_message,
            }
            self._append_unique("errors", entry)
            LOGGER.error(
                "pipeline_log_error",
                extra={"component": clean_component, "error_message": clean_message},
            )
        except Exception as error:
            LOGGER.exception("pipeline_log_error_failed", extra={"error_message": str(error)})

    def export_logs(self, output_path: str) -> None:
        """Export structured logs to a JSON file, creating parent directory as needed."""
        try:
            export_path = Path(str(output_path).strip())
            if not str(export_path):
                raise ValueError("output_path is required for export.")

            export_path.parent.mkdir(parents=True, exist_ok=True)
            with export_path.open("w", encoding="utf-8") as file_handle:
                json.dump(self.logs, file_handle, ensure_ascii=True, indent=2)

            LOGGER.info(
                "pipeline_log_exported",
                extra={"output_path": str(export_path), "entry_count": self._entry_count()},
            )
        except Exception as error:
            self.log_error("PipelineLogger.export_logs", str(error))
            LOGGER.exception("pipeline_log_export_failed", extra={"output_path": output_path})

    def reset_logs(self) -> None:
        """Clear all in-memory logs and dedupe signatures."""
        try:
            for key in self.logs:
                self.logs[key] = []
            for key in self._dedupe_signatures:
                self._dedupe_signatures[key].clear()
            LOGGER.info("pipeline_log_reset")
        except Exception as error:
            LOGGER.exception("pipeline_log_reset_failed", extra={"error_message": str(error)})

    def _append_unique(self, bucket: str, payload: dict[str, Any]) -> None:
        """Append payload with timestamp if not already present in the target bucket."""
        normalized_payload = self._sanitize_value(payload)
        signature = json.dumps(normalized_payload, sort_keys=True, ensure_ascii=True, default=str)

        if signature in self._dedupe_signatures[bucket]:
            LOGGER.debug("pipeline_log_duplicate_skipped", extra={"bucket": bucket})
            return

        self._dedupe_signatures[bucket].add(signature)
        entry = {
            "timestamp": self._timestamp(),
            **normalized_payload,
        }
        self.logs[bucket].append(entry)

    def _entry_count(self) -> int:
        """Return total number of log entries across all buckets."""
        return sum(len(entries) for entries in self.logs.values())

    def _timestamp(self) -> str:
        """Return UTC ISO-8601 timestamp for each log entry."""
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _sanitize_value(self, value: Any) -> Any:
        """Recursively sanitize values and redact likely sensitive fields."""
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for key, item in value.items():
                key_str = str(key)
                if any(token in key_str.lower() for token in _SENSITIVE_KEYS):
                    sanitized[key_str] = "[REDACTED]"
                else:
                    sanitized[key_str] = self._sanitize_value(item)
            return sanitized
        if isinstance(value, list):
            return [self._sanitize_value(item) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize_value(item) for item in value]
        if isinstance(value, set):
            return sorted(self._sanitize_value(item) for item in value)
        return value

