"""Unit tests for Task 11.2 retry-enabled pipeline orchestration."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from main import run_retry_pipeline


class RetryPipelineTests(unittest.TestCase):
    """Validate retry loop behavior, strategy overrides, and safe stopping rules."""

    def setUp(self) -> None:
        """Prepare baseline resources used by retry pipeline tests."""
        self.resources = {
            "chunks": [{"chunk_id": "chunk_1", "text": "RAG improves grounding.", "position": 0}],
            "model": object(),
            "faiss_index": object(),
            "metadata": [{"chunk_id": "chunk_1", "text": "RAG improves grounding."}],
            "read_chunk_ids": set(),
            "model_name": "llama3.1",
            "provider": "ollama",
        }

    @patch("main.decide_retry_action")
    @patch("main.generate_answer")
    @patch("main.execute_retrieval")
    def test_no_retry_needed(
        self,
        mock_execute_retrieval: object,
        mock_generate_answer: object,
        mock_decide_retry_action: object,
    ) -> None:
        """Stop after first attempt when validation indicates no retry is needed."""
        mock_execute_retrieval.return_value = {
            "tool_name": "keyword_search",
            "status": "success",
            "results": [{"chunk_id": "chunk_1", "score": 0.8, "snippet": "RAG improves grounding."}],
            "result_count": 1,
            "execution_summary": "keyword_search returned 1 results",
        }
        mock_generate_answer.return_value = {
            "status": "success",
            "provider": "ollama",
            "model_name": "llama3.1",
            "answer": "RAG improves grounding with retrieved context.",
            "token_usage": {"total_tokens": 16},
            "latency_seconds": 0.2,
        }
        mock_decide_retry_action.return_value = {
            "should_retry": False,
            "retry_reason": "validation_passed",
            "recommended_strategy": "no_retry",
            "remaining_retries": 2,
        }

        output = run_retry_pipeline("ID-123", self.resources, max_steps=5, max_retries=2)
        self.assertEqual(len(output["attempts"]), 1)
        self.assertEqual(output["retry_count"], 0)
        self.assertEqual(output["final_status"], "success")

    @patch("main.decide_retry_action")
    @patch("main.generate_answer")
    @patch("main.execute_retrieval")
    def test_retry_due_to_incomplete_answer(
        self,
        mock_execute_retrieval: object,
        mock_generate_answer: object,
        mock_decide_retry_action: object,
    ) -> None:
        """Retry with semantic strategy and keep both attempts in history."""

        def retrieval_side_effect(selection: dict, _resources: dict) -> dict:
            tool_name = selection.get("tool_name", "")
            return {
                "tool_name": tool_name,
                "status": "success",
                "results": [{"chunk_id": "chunk_1", "score": 0.7, "snippet": f"{tool_name} snippet"}],
                "result_count": 1,
                "execution_summary": f"{tool_name} returned 1 results",
            }

        mock_execute_retrieval.side_effect = retrieval_side_effect
        mock_generate_answer.return_value = {
            "status": "success",
            "provider": "ollama",
            "model_name": "llama3.1",
            "answer": "Partial answer",
            "token_usage": {"total_tokens": 10},
            "latency_seconds": 0.1,
        }
        mock_decide_retry_action.side_effect = [
            {
                "should_retry": True,
                "retry_reason": "incomplete_answer",
                "recommended_strategy": "retry_with_semantic_search",
                "remaining_retries": 1,
            },
            {
                "should_retry": False,
                "retry_reason": "validation_passed",
                "recommended_strategy": "no_retry",
                "remaining_retries": 1,
            },
        ]

        output = run_retry_pipeline("ID-123", self.resources, max_steps=5, max_retries=2)
        self.assertEqual(len(output["attempts"]), 2)
        self.assertEqual(output["retry_count"], 1)
        self.assertEqual(output["attempts"][1]["tool_selection_output"]["tool_name"], "semantic_search")

    @patch("main.decide_retry_action")
    @patch("main.generate_answer")
    @patch("main.execute_retrieval")
    def test_retry_due_to_grounding_failure(
        self,
        mock_execute_retrieval: object,
        mock_generate_answer: object,
        mock_decide_retry_action: object,
    ) -> None:
        """Retry with chunk-read strategy when grounding failure is recommended."""

        def retrieval_side_effect(selection: dict, _resources: dict) -> dict:
            tool_name = selection.get("tool_name", "")
            return {
                "tool_name": tool_name,
                "status": "success",
                "results": [{"chunk_id": "chunk_1", "score": 0.6, "snippet": "chunk candidate", "text": "chunk text"}],
                "result_count": 1,
                "execution_summary": f"{tool_name} returned 1 results",
            }

        mock_execute_retrieval.side_effect = retrieval_side_effect
        mock_generate_answer.return_value = {
            "status": "success",
            "provider": "ollama",
            "model_name": "llama3.1",
            "answer": "Answer text",
            "token_usage": {"total_tokens": 10},
            "latency_seconds": 0.1,
        }
        mock_decide_retry_action.side_effect = [
            {
                "should_retry": True,
                "retry_reason": "grounding_failed",
                "recommended_strategy": "retry_with_chunk_read",
                "remaining_retries": 1,
            },
            {
                "should_retry": False,
                "retry_reason": "validation_passed",
                "recommended_strategy": "no_retry",
                "remaining_retries": 1,
            },
        ]

        output = run_retry_pipeline("Explain RAG", self.resources, max_steps=5, max_retries=2)
        self.assertEqual(len(output["attempts"]), 2)
        self.assertEqual(output["attempts"][1]["tool_selection_output"]["tool_name"], "chunk_read")

    @patch("main.decide_retry_action")
    @patch("main.generate_answer")
    @patch("main.execute_retrieval")
    def test_retry_limit_reached(
        self,
        mock_execute_retrieval: object,
        mock_generate_answer: object,
        mock_decide_retry_action: object,
    ) -> None:
        """Stop retry loop when max retry count is reached."""
        mock_execute_retrieval.return_value = {
            "tool_name": "keyword_search",
            "status": "success",
            "results": [{"chunk_id": "chunk_1", "score": 0.5, "snippet": "snippet"}],
            "result_count": 1,
            "execution_summary": "keyword_search returned 1 results",
        }
        mock_generate_answer.return_value = {
            "status": "success",
            "provider": "ollama",
            "model_name": "llama3.1",
            "answer": "Weak answer",
            "token_usage": {"total_tokens": 9},
            "latency_seconds": 0.1,
        }
        mock_decide_retry_action.return_value = {
            "should_retry": True,
            "retry_reason": "incomplete_answer",
            "recommended_strategy": "retry_with_semantic_search",
            "remaining_retries": 0,
        }

        output = run_retry_pipeline("Explain RAG", self.resources, max_steps=5, max_retries=1)
        self.assertEqual(len(output["attempts"]), 2)
        self.assertEqual(output["retry_count"], 1)

    @patch("main.decide_retry_action")
    @patch("main.generate_answer")
    @patch("main.execute_retrieval")
    def test_retrieval_failure_during_retry(
        self,
        mock_execute_retrieval: object,
        mock_generate_answer: object,
        mock_decide_retry_action: object,
    ) -> None:
        """Preserve retry attempt when retrieval fails on retry strategy."""

        def retrieval_side_effect(selection: dict, _resources: dict) -> dict:
            tool_name = selection.get("tool_name", "")
            if tool_name == "semantic_search":
                return {
                    "tool_name": tool_name,
                    "status": "error",
                    "results": [],
                    "result_count": 0,
                    "execution_summary": "semantic_search failed",
                }
            return {
                "tool_name": tool_name,
                "status": "success",
                "results": [{"chunk_id": "chunk_1", "score": 0.7, "snippet": "start"}],
                "result_count": 1,
                "execution_summary": f"{tool_name} returned 1 results",
            }

        mock_execute_retrieval.side_effect = retrieval_side_effect
        mock_generate_answer.return_value = {
            "status": "success",
            "provider": "ollama",
            "model_name": "llama3.1",
            "answer": "answer",
            "token_usage": {"total_tokens": 7},
            "latency_seconds": 0.1,
        }
        mock_decide_retry_action.side_effect = [
            {
                "should_retry": True,
                "retry_reason": "incomplete_answer",
                "recommended_strategy": "retry_with_semantic_search",
                "remaining_retries": 1,
            },
            {
                "should_retry": False,
                "retry_reason": "stop_after_error",
                "recommended_strategy": "no_retry",
                "remaining_retries": 1,
            },
        ]

        output = run_retry_pipeline("ID-123", self.resources, max_steps=5, max_retries=2)
        self.assertEqual(len(output["attempts"]), 2)
        self.assertEqual(output["final_attempt"]["retrieval_output"]["status"], "error")

    @patch("main.decide_retry_action")
    @patch("main.generate_answer")
    @patch("main.execute_retrieval")
    def test_model_failure_during_retry(
        self,
        mock_execute_retrieval: object,
        mock_generate_answer: object,
        mock_decide_retry_action: object,
    ) -> None:
        """Handle model failure on retry attempt and return safe final status."""
        mock_execute_retrieval.return_value = {
            "tool_name": "keyword_search",
            "status": "success",
            "results": [{"chunk_id": "chunk_1", "score": 0.6, "snippet": "snippet"}],
            "result_count": 1,
            "execution_summary": "keyword_search returned 1 results",
        }
        mock_generate_answer.side_effect = [
            {
                "status": "success",
                "provider": "ollama",
                "model_name": "llama3.1",
                "answer": "first answer",
                "token_usage": {"total_tokens": 9},
                "latency_seconds": 0.1,
            },
            {
                "status": "error",
                "provider": "ollama",
                "model_name": "llama3.1",
                "answer": "",
                "error_message": "Model unavailable",
            },
        ]
        mock_decide_retry_action.side_effect = [
            {
                "should_retry": True,
                "retry_reason": "incomplete_answer",
                "recommended_strategy": "retry_with_hybrid_search",
                "remaining_retries": 1,
            },
            {
                "should_retry": False,
                "retry_reason": "retry_limit_reached",
                "recommended_strategy": "no_retry",
                "remaining_retries": 1,
            },
        ]

        output = run_retry_pipeline("Explain RAG", self.resources, max_steps=5, max_retries=2)
        self.assertEqual(len(output["attempts"]), 2)
        self.assertEqual(output["final_status"], "error")


if __name__ == "__main__":
    unittest.main()

