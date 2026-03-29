"""Unit tests for Task 10.2 structured pipeline logging."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.utils.logger import PipelineLogger


class PipelineLoggerTests(unittest.TestCase):
    """Validate in-memory logging, export, and reset behavior."""

    def setUp(self) -> None:
        """Create fresh logger and test output path."""
        self.logger = PipelineLogger()
        self.export_path = Path("tests/.tmp/logger/pipeline_logs.json")
        if self.export_path.exists():
            self.export_path.unlink()
        self.export_path.parent.mkdir(parents=True, exist_ok=True)

    def test_log_query(self) -> None:
        """Store query with timestamp and avoid duplicate query entries."""
        self.logger.log_query("Explain retrieval quality")
        self.logger.log_query("Explain retrieval quality")

        queries = self.logger.logs["queries"]
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0]["query"], "Explain retrieval quality")
        self.assertIn("timestamp", queries[0])

    def test_log_tool_execution(self) -> None:
        """Store tool execution metadata with sanitized tool input."""
        self.logger.log_tool_execution(
            tool_name="hybrid_search",
            tool_input={"query": "rag", "api_key": "super-secret"},
            result_count=3,
            execution_summary="hybrid_search returned 3 results",
        )

        executions = self.logger.logs["tool_executions"]
        self.assertEqual(len(executions), 1)
        entry = executions[0]
        self.assertEqual(entry["tool_name"], "hybrid_search")
        self.assertEqual(entry["result_count"], 3)
        self.assertEqual(entry["tool_input"]["api_key"], "[REDACTED]")
        self.assertIn("timestamp", entry)

    def test_log_validation(self) -> None:
        """Store grounding/conflict/completeness validation outputs."""
        self.logger.log_validation(
            grounding_result={"grounded": True, "grounding_score": 0.9},
            conflict_result={"has_conflict": False, "conflict_pairs": []},
            completeness_result={"is_complete": True, "completeness_score": 0.8},
        )

        validations = self.logger.logs["validations"]
        self.assertEqual(len(validations), 1)
        entry = validations[0]
        self.assertTrue(entry["grounding_result"]["grounded"])
        self.assertFalse(entry["conflict_result"]["has_conflict"])
        self.assertTrue(entry["completeness_result"]["is_complete"])

    def test_export_logs(self) -> None:
        """Export logs to JSON file and preserve structured buckets."""
        self.logger.log_query("What changed?")
        self.logger.log_error("retrieval_node", "timeout")

        self.logger.export_logs(str(self.export_path))

        self.assertTrue(self.export_path.exists())
        exported_data = json.loads(self.export_path.read_text(encoding="utf-8"))
        self.assertIn("queries", exported_data)
        self.assertIn("errors", exported_data)
        self.assertEqual(len(exported_data["queries"]), 1)
        self.assertEqual(len(exported_data["errors"]), 1)

    def test_reset_logs(self) -> None:
        """Clear all in-memory logs safely even after multiple event types."""
        self.logger.log_query("Q")
        self.logger.log_tool_execution("keyword_search", {"query": "q"}, 1, "ok")
        self.logger.log_metrics(
            retrieval_metrics={"result_count": 1},
            answer_metrics={"confidence": 0.5},
            system_metrics={"total_tokens": 10},
        )
        self.logger.reset_logs()

        self.assertEqual(self.logger.logs["queries"], [])
        self.assertEqual(self.logger.logs["tool_executions"], [])
        self.assertEqual(self.logger.logs["validations"], [])
        self.assertEqual(self.logger.logs["metrics"], [])
        self.assertEqual(self.logger.logs["errors"], [])


if __name__ == "__main__":
    unittest.main()

