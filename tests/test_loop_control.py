"""Unit tests for Task 6.5 loop control node."""

from __future__ import annotations

import unittest

from src.agent.loop import should_continue_loop


class LoopControlTests(unittest.TestCase):
    """Validate loop continuation and stop decisions."""

    def test_sufficient_context_found(self) -> None:
        """Stop loop when sufficient context is available."""
        evaluation_output = {
            "sufficient_context": True,
            "needs_retry": False,
            "retry_reason": "",
            "recommended_next_action": "proceed_to_answer",
        }

        result = should_continue_loop(current_step=1, max_steps=3, evaluation_output=evaluation_output)

        self.assertFalse(result["continue_loop"])
        self.assertEqual(result["stop_reason"], "sufficient_context_found")
        self.assertEqual(result["next_action"], "finalize_answer")
        self.assertEqual(result["remaining_steps"], 2)

    def test_retry_needed(self) -> None:
        """Continue loop when retry is needed and steps remain."""
        evaluation_output = {
            "sufficient_context": False,
            "needs_retry": True,
            "retry_reason": "weak scores",
            "recommended_next_action": "retry_with_hybrid_search",
        }

        result = should_continue_loop(current_step=1, max_steps=4, evaluation_output=evaluation_output)

        self.assertTrue(result["continue_loop"])
        self.assertEqual(result["stop_reason"], "")
        self.assertEqual(result["next_action"], "retry_with_hybrid_search")
        self.assertEqual(result["remaining_steps"], 3)

    def test_max_steps_reached(self) -> None:
        """Stop loop when max step budget has been exhausted."""
        evaluation_output = {
            "sufficient_context": False,
            "needs_retry": True,
            "retry_reason": "need more context",
            "recommended_next_action": "retry_with_semantic_search",
        }

        result = should_continue_loop(current_step=3, max_steps=3, evaluation_output=evaluation_output)

        self.assertFalse(result["continue_loop"])
        self.assertEqual(result["stop_reason"], "max_steps_reached")
        self.assertEqual(result["next_action"], "finalize_answer")
        self.assertEqual(result["remaining_steps"], 0)

    def test_no_results_stop(self) -> None:
        """Stop loop when evaluator recommends no-results termination."""
        evaluation_output = {
            "sufficient_context": False,
            "needs_retry": True,
            "retry_reason": "error status",
            "recommended_next_action": "stop_no_results",
        }

        result = should_continue_loop(current_step=0, max_steps=3, evaluation_output=evaluation_output)

        self.assertFalse(result["continue_loop"])
        self.assertEqual(result["stop_reason"], "no_results")
        self.assertEqual(result["next_action"], "finalize_answer")
        self.assertEqual(result["remaining_steps"], 3)

    def test_invalid_step_count(self) -> None:
        """Stop loop safely on invalid step inputs."""
        evaluation_output = {
            "sufficient_context": False,
            "needs_retry": True,
            "retry_reason": "invalid counters",
            "recommended_next_action": "retry_with_keyword_search",
        }

        result = should_continue_loop(current_step=-1, max_steps=3, evaluation_output=evaluation_output)

        self.assertFalse(result["continue_loop"])
        self.assertEqual(result["stop_reason"], "invalid_step_count")
        self.assertEqual(result["next_action"], "finalize_answer")
        self.assertEqual(result["remaining_steps"], 0)


if __name__ == "__main__":
    unittest.main()
