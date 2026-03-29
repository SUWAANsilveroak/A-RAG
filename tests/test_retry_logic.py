"""Unit tests for Task 9.4 retry decision logic."""

from __future__ import annotations

import unittest

from src.agent.validator import decide_retry_action


class RetryLogicTests(unittest.TestCase):
    """Validate retry decision outputs from validation-layer signals."""

    def test_grounding_failure(self) -> None:
        """Recommend chunk read retry when grounding fails."""
        result = decide_retry_action(
            grounding_result={"grounded": False, "grounding_score": 0.0, "missing_chunks": ["chunk_1"], "notes": ""},
            conflict_result={"has_conflict": False, "conflicting_chunks": [], "conflict_pairs": [], "notes": ""},
            completeness_result={"is_complete": True, "completeness_score": 1.0, "missing_topics": [], "notes": ""},
            max_retries=3,
            current_retry_count=1,
        )

        self.assertTrue(result["should_retry"])
        self.assertEqual(result["recommended_strategy"], "retry_with_chunk_read")
        self.assertEqual(result["remaining_retries"], 2)
        self.assertIn("grounding_failed", result["retry_reason"])

    def test_conflict_detected(self) -> None:
        """Recommend hybrid retry when conflicts are detected."""
        result = decide_retry_action(
            grounding_result={"grounded": True, "grounding_score": 1.0, "missing_chunks": [], "notes": ""},
            conflict_result={"has_conflict": True, "conflicting_chunks": ["chunk_1", "chunk_2"], "conflict_pairs": [], "notes": ""},
            completeness_result={"is_complete": True, "completeness_score": 1.0, "missing_topics": [], "notes": ""},
            max_retries=3,
            current_retry_count=0,
        )

        self.assertTrue(result["should_retry"])
        self.assertEqual(result["recommended_strategy"], "retry_with_hybrid_search")
        self.assertIn("conflict_detected", result["retry_reason"])

    def test_incomplete_answer(self) -> None:
        """Recommend semantic retry when answer is incomplete."""
        result = decide_retry_action(
            grounding_result={"grounded": True, "grounding_score": 1.0, "missing_chunks": [], "notes": ""},
            conflict_result={"has_conflict": False, "conflicting_chunks": [], "conflict_pairs": [], "notes": ""},
            completeness_result={"is_complete": False, "completeness_score": 0.4, "missing_topics": ["latency"], "notes": ""},
            max_retries=2,
            current_retry_count=0,
        )

        self.assertTrue(result["should_retry"])
        self.assertEqual(result["recommended_strategy"], "retry_with_semantic_search")
        self.assertIn("incomplete_answer", result["retry_reason"])

    def test_retry_limit_reached(self) -> None:
        """Stop retries when retry limit has been reached."""
        result = decide_retry_action(
            grounding_result={"grounded": False, "grounding_score": 0.0, "missing_chunks": ["chunk_3"], "notes": ""},
            conflict_result={"has_conflict": False, "conflicting_chunks": [], "conflict_pairs": [], "notes": ""},
            completeness_result={"is_complete": False, "completeness_score": 0.0, "missing_topics": ["topic"], "notes": ""},
            max_retries=0,
            current_retry_count=0,
        )

        self.assertFalse(result["should_retry"])
        self.assertEqual(result["retry_reason"], "retry_limit_reached")
        self.assertEqual(result["remaining_retries"], 0)
        self.assertEqual(result["recommended_strategy"], "no_retry")

    def test_multiple_failures(self) -> None:
        """Prefer hybrid retry when multiple validation failures are present."""
        result = decide_retry_action(
            grounding_result={"grounded": False, "grounding_score": 0.2, "missing_chunks": ["chunk_1"], "notes": ""},
            conflict_result={"has_conflict": True, "conflicting_chunks": ["chunk_2"], "conflict_pairs": [], "notes": ""},
            completeness_result={"is_complete": False, "completeness_score": 0.3, "missing_topics": ["topic"], "notes": ""},
            max_retries=5,
            current_retry_count=2,
        )

        self.assertTrue(result["should_retry"])
        self.assertEqual(result["recommended_strategy"], "retry_with_hybrid_search")
        self.assertEqual(result["remaining_retries"], 3)
        self.assertIn("grounding_failed", result["retry_reason"])
        self.assertIn("conflict_detected", result["retry_reason"])
        self.assertIn("incomplete_answer", result["retry_reason"])


if __name__ == "__main__":
    unittest.main()
