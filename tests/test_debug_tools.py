"""Unit tests for Task 10.3 debug utility functions."""

from __future__ import annotations

import unittest

from src.utils.debug import (
    debug_answer_flow,
    debug_retrieval_flow,
    debug_validation_flow,
)


class DebugToolsTests(unittest.TestCase):
    """Validate retrieval, validation, and answer debug snapshots."""

    def test_retrieval_flow_debug(self) -> None:
        """Return concise retrieval-flow debug snapshot with retrieval summary."""
        result = debug_retrieval_flow(
            planner_output={"query_type": "hybrid", "retrieval_strategy": "hybrid_search"},
            tool_selection_output={"tool_name": "hybrid_search", "execution_ready": True},
            retrieval_output={
                "tool_name": "hybrid_search",
                "status": "success",
                "results": [{"chunk_id": "chunk_1"}],
                "result_count": 1,
                "execution_summary": "hybrid_search returned 1 results",
            },
            evaluation_output={"sufficient_context": True, "needs_retry": False},
        )

        self.assertIn("planner_output", result)
        self.assertIn("tool_selection_output", result)
        self.assertIn("retrieval_output_summary", result)
        self.assertEqual(result["retrieval_output_summary"]["tool_name"], "hybrid_search")
        self.assertEqual(result["retrieval_output_summary"]["result_count"], 1)

    def test_validation_flow_debug(self) -> None:
        """Return validation debug snapshot with all validation sections."""
        result = debug_validation_flow(
            grounding_result={"grounded": True, "grounding_score": 0.9},
            conflict_result={"has_conflict": False, "conflict_pairs": []},
            completeness_result={"is_complete": True, "completeness_score": 0.8},
            retry_result={"should_retry": False, "remaining_retries": 2},
        )

        self.assertTrue(result["grounding_result"]["grounded"])
        self.assertFalse(result["conflict_result"]["has_conflict"])
        self.assertTrue(result["completeness_result"]["is_complete"])
        self.assertFalse(result["retry_result"]["should_retry"])

    def test_answer_flow_debug(self) -> None:
        """Return answer debug snapshot with prompt preview and answer summary."""
        result = debug_answer_flow(
            prompt="SYSTEM: grounded assistant\nQUERY: what is rag?",
            answer_response={
                "status": "success",
                "provider": "ollama",
                "model_name": "llama3.1",
                "answer": "RAG combines retrieval and generation.",
                "token_usage": {"total_tokens": 35},
                "latency_seconds": 0.52,
            },
            final_output={"status": "success", "confidence": 0.9},
        )

        self.assertIn("prompt_preview", result)
        self.assertIn("answer_summary", result)
        self.assertIn("final_output", result)
        self.assertEqual(result["answer_summary"]["status"], "success")
        self.assertGreater(result["answer_summary"]["answer_length"], 0)

    def test_large_prompt_truncation(self) -> None:
        """Truncate oversized prompt and answer previews to max lengths."""
        long_prompt = "P" * 800
        long_answer = "A" * 600
        result = debug_answer_flow(
            prompt=long_prompt,
            answer_response={"answer": long_answer, "status": "success"},
            final_output={},
        )

        self.assertLessEqual(len(result["prompt_preview"]), 500)
        self.assertLessEqual(len(result["answer_summary"]["answer_preview"]), 300)
        self.assertIn("[truncated]", result["prompt_preview"])
        self.assertIn("[truncated]", result["answer_summary"]["answer_preview"])

    def test_empty_values(self) -> None:
        """Handle empty inputs safely without crashing."""
        retrieval_debug = debug_retrieval_flow({}, {}, {}, {})
        validation_debug = debug_validation_flow({}, {}, {}, {})
        answer_debug = debug_answer_flow("", {}, {})

        self.assertEqual(retrieval_debug["retrieval_output_summary"]["result_count"], 0)
        self.assertEqual(validation_debug["grounding_result"], {})
        self.assertEqual(answer_debug["prompt_preview"], "")
        self.assertEqual(answer_debug["answer_summary"]["answer_preview"], "")


if __name__ == "__main__":
    unittest.main()

