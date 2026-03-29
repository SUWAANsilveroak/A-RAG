"""Unit tests for Task 12.2 CLI layer."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.cli import run_cli


class CliTests(unittest.TestCase):
    """Validate CLI commands and structured output behavior."""

    def _run_and_capture(self, argv: list[str]) -> tuple[dict, dict]:
        """Run CLI with argv and capture returned payload plus printed JSON."""
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            result = run_cli(argv)
        printed = buffer.getvalue().strip()
        printed_payload = json.loads(printed) if printed else {}
        return result, printed_payload

    def test_health_command(self) -> None:
        """Return health status for health command."""
        result, printed_payload = self._run_and_capture(["health"])
        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(printed_payload, {"status": "ok"})

    @patch("src.cli.run_a_rag_pipeline")
    def test_query_command(self, mock_run_pipeline: object) -> None:
        """Return success payload for query command."""
        mock_run_pipeline.return_value = {"query": "What is RAG?", "final_output": {"status": "success"}}
        result, printed_payload = self._run_and_capture(["query", "What is RAG?"])

        self.assertEqual(result["status"], "success")
        self.assertEqual(printed_payload["status"], "success")
        self.assertIn("result", result)

    @patch("src.cli.run_retry_pipeline")
    def test_retry_command(self, mock_run_retry_pipeline: object) -> None:
        """Return success payload for retry command."""
        mock_run_retry_pipeline.return_value = {"query": "Explain RAG", "retry_count": 1, "final_status": "success"}
        result, printed_payload = self._run_and_capture(["retry", "Explain RAG"])

        self.assertEqual(result["status"], "success")
        self.assertEqual(printed_payload["status"], "success")
        self.assertEqual(result["result"]["retry_count"], 1)

    def test_empty_query(self) -> None:
        """Return error payload for empty query command."""
        result, printed_payload = self._run_and_capture(["query", "   "])
        self.assertEqual(result["status"], "error")
        self.assertIn("Query must not be empty", result["error_message"])
        self.assertEqual(printed_payload["status"], "error")

    def test_unknown_command(self) -> None:
        """Return error payload for unknown command."""
        result, printed_payload = self._run_and_capture(["unknown"])
        self.assertEqual(result["status"], "error")
        self.assertIn("Unknown command", result["error_message"])
        self.assertEqual(printed_payload["status"], "error")


if __name__ == "__main__":
    unittest.main()

