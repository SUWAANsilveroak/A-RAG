"""Unit tests for Task 10.1 evaluation metrics."""

from __future__ import annotations

import unittest

from src.evaluation.metrics import (
    calculate_answer_metrics,
    calculate_retrieval_metrics,
    calculate_system_metrics,
)


class MetricsTests(unittest.TestCase):
    """Validate retrieval, answer, and system metric calculators."""

    def test_normal_retrieval_metrics(self) -> None:
        """Calculate retrieval metrics for a normal scored result set."""
        retrieval_results = [
            {"chunk_id": "chunk_1", "score": 0.8, "snippet": "A"},
            {"chunk_id": "chunk_2", "score": 0.6, "snippet": "B"},
            {"chunk_id": "chunk_1", "score": 0.7, "snippet": "C"},
        ]

        result = calculate_retrieval_metrics(retrieval_results)

        self.assertEqual(result["result_count"], 3)
        self.assertAlmostEqual(result["average_score"], (0.8 + 0.6 + 0.7) / 3)
        self.assertEqual(result["top_score"], 0.8)
        self.assertEqual(result["duplicate_chunk_count"], 1)

    def test_empty_retrieval_metrics(self) -> None:
        """Return safe zeroed retrieval metrics for empty input."""
        result = calculate_retrieval_metrics([])
        self.assertEqual(result["result_count"], 0)
        self.assertEqual(result["average_score"], 0.0)
        self.assertEqual(result["top_score"], 0.0)
        self.assertEqual(result["duplicate_chunk_count"], 0)

    def test_answer_metrics_with_missing_fields(self) -> None:
        """Handle missing confidence and malformed validation fields safely."""
        final_output = {"answer": "Short answer"}
        grounding_result = {"grounding_score": 0.7}
        completeness_result = {}

        result = calculate_answer_metrics(final_output, grounding_result, completeness_result)

        self.assertEqual(result["grounding_score"], 0.7)
        self.assertEqual(result["completeness_score"], 0.0)
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["answer_length"], len("Short answer"))

    def test_system_metrics_with_missing_token_usage(self) -> None:
        """Return safe defaults when token usage fields are missing."""
        result = calculate_system_metrics(token_usage={}, latency_seconds=1.25, retry_count=2)

        self.assertEqual(result["prompt_tokens"], 0)
        self.assertEqual(result["completion_tokens"], 0)
        self.assertEqual(result["total_tokens"], 0)
        self.assertEqual(result["latency_seconds"], 1.25)
        self.assertEqual(result["retry_count"], 2)

    def test_negative_latency_handling(self) -> None:
        """Clamp negative latency and retry count to safe values."""
        result = calculate_system_metrics(
            token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            latency_seconds=-4.2,
            retry_count=-1,
        )

        self.assertEqual(result["prompt_tokens"], 10)
        self.assertEqual(result["completion_tokens"], 5)
        self.assertEqual(result["total_tokens"], 15)
        self.assertEqual(result["latency_seconds"], 0.0)
        self.assertEqual(result["retry_count"], 0)


if __name__ == "__main__":
    unittest.main()
