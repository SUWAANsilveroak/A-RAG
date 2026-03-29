"""Unit tests for Task 6.4 evaluation node."""

from __future__ import annotations

import unittest

from src.agent.loop import evaluate_retrieval


class EvaluationNodeTests(unittest.TestCase):
    """Validate retrieval sufficiency scoring and retry recommendation logic."""

    def test_strong_retrieval_result(self) -> None:
        """Mark retrieval sufficient when at least one score is strong enough."""
        retrieval_output = {
            "tool_name": "semantic_search",
            "status": "success",
            "results": [{"chunk_id": "chunk_1", "score": 0.82}],
            "result_count": 1,
            "execution_summary": "semantic_search returned 1 results",
        }

        result = evaluate_retrieval(retrieval_output)

        self.assertTrue(result["sufficient_context"])
        self.assertFalse(result["needs_retry"])
        self.assertEqual(result["recommended_next_action"], "proceed_to_answer")

    def test_weak_retrieval_result(self) -> None:
        """Recommend retry when all available scores are below threshold."""
        retrieval_output = {
            "tool_name": "hybrid_search",
            "status": "success",
            "results": [
                {"chunk_id": "chunk_1", "combined_score": 0.1},
                {"chunk_id": "chunk_2", "combined_score": 0.2},
            ],
            "result_count": 2,
            "execution_summary": "hybrid_search returned 2 results",
        }

        result = evaluate_retrieval(retrieval_output, min_result_count=1, min_score_threshold=0.3)

        self.assertFalse(result["sufficient_context"])
        self.assertTrue(result["needs_retry"])
        self.assertEqual(result["recommended_next_action"], "retry_with_chunk_read")
        self.assertIn("below threshold", result["retry_reason"])

    def test_no_results(self) -> None:
        """Recommend semantic retry when keyword search returns no results."""
        retrieval_output = {
            "tool_name": "keyword_search",
            "status": "success",
            "results": [],
            "result_count": 0,
            "execution_summary": "keyword_search returned no results",
        }

        result = evaluate_retrieval(retrieval_output)

        self.assertFalse(result["sufficient_context"])
        self.assertTrue(result["needs_retry"])
        self.assertEqual(result["recommended_next_action"], "retry_with_semantic_search")

    def test_error_status(self) -> None:
        """Stop retry loop when retrieval output status is error."""
        retrieval_output = {
            "tool_name": "semantic_search",
            "status": "error",
            "results": [],
            "result_count": 0,
            "execution_summary": "semantic_search failed",
        }

        result = evaluate_retrieval(retrieval_output)

        self.assertFalse(result["sufficient_context"])
        self.assertTrue(result["needs_retry"])
        self.assertEqual(result["recommended_next_action"], "stop_no_results")
        self.assertIn("error", result["retry_reason"].lower())

    def test_missing_score_field(self) -> None:
        """Recommend retry when non-chunk-read results have no score fields."""
        retrieval_output = {
            "tool_name": "semantic_search",
            "status": "success",
            "results": [{"chunk_id": "chunk_1", "snippet": "no score present"}],
            "result_count": 1,
            "execution_summary": "semantic_search returned 1 results",
        }

        result = evaluate_retrieval(retrieval_output)

        self.assertFalse(result["sufficient_context"])
        self.assertTrue(result["needs_retry"])
        self.assertEqual(result["recommended_next_action"], "retry_with_hybrid_search")
        self.assertIn("Missing score fields", result["retry_reason"])


if __name__ == "__main__":
    unittest.main()
