"""Unit tests for Task 11.1 main A-RAG pipeline integration."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from main import run_a_rag_pipeline


class MainPipelineTests(unittest.TestCase):
    """Validate end-to-end pipeline orchestration and failure handling."""

    def setUp(self) -> None:
        """Prepare baseline resources for pipeline execution."""
        self.resources = {
            "chunks": [{"chunk_id": "chunk_1", "text": "RAG improves retrieval quality.", "position": 0}],
            "model": object(),
            "faiss_index": object(),
            "metadata": [{"chunk_id": "chunk_1", "text": "RAG improves retrieval quality."}],
            "read_chunk_ids": set(),
            "model_name": "llama3.1",
            "provider": "ollama",
        }

    @patch("main.generate_answer")
    @patch("main.execute_retrieval")
    def test_successful_pipeline_execution(self, mock_execute_retrieval: object, mock_generate_answer: object) -> None:
        """Run full pipeline successfully and return structured output keys."""
        mock_execute_retrieval.return_value = {
            "tool_name": "keyword_search",
            "status": "success",
            "results": [{"chunk_id": "chunk_1", "score": 0.9, "snippet": "RAG improves retrieval quality."}],
            "result_count": 1,
            "execution_summary": "keyword_search returned 1 results",
        }
        mock_generate_answer.return_value = {
            "status": "success",
            "provider": "ollama",
            "model_name": "llama3.1",
            "answer": "RAG improves answer grounding by using retrieved chunks.",
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
            "latency_seconds": 0.3,
        }

        output = run_a_rag_pipeline("What is RAG?", self.resources)

        self.assertEqual(output["query"], "What is RAG?")
        self.assertIn("planner_output", output)
        self.assertIn("tool_selection_output", output)
        self.assertIn("retrieval_output", output)
        self.assertIn("evaluation_output", output)
        self.assertIn("compressed_context", output)
        self.assertIn("answer_response", output)
        self.assertIn("final_output", output)
        self.assertIn("validation_results", output)
        self.assertIn("retry_result", output)
        self.assertIn("metrics", output)
        self.assertEqual(output["final_output"]["status"], "success")

    def test_empty_query(self) -> None:
        """Return standardized error payload for empty query input."""
        output = run_a_rag_pipeline("", self.resources)
        self.assertEqual(output["status"], "error")
        self.assertIn("Query must not be empty", output["error_message"])

    def test_missing_resources(self) -> None:
        """Return standardized error payload for missing required resources."""
        incomplete_resources = {"chunks": []}
        output = run_a_rag_pipeline("test query", incomplete_resources)
        self.assertEqual(output["status"], "error")
        self.assertIn("Missing required resources", output["error_message"])

    @patch("main.generate_answer")
    @patch("main.execute_retrieval")
    def test_retrieval_failure(self, mock_execute_retrieval: object, mock_generate_answer: object) -> None:
        """Continue safely when retrieval status is error and preserve retrieval output."""
        mock_execute_retrieval.return_value = {
            "tool_name": "keyword_search",
            "status": "error",
            "results": [],
            "result_count": 0,
            "execution_summary": "Retrieval execution failed",
        }
        mock_generate_answer.return_value = {
            "status": "success",
            "provider": "ollama",
            "model_name": "llama3.1",
            "answer": "Not found in retrieved context",
            "token_usage": {"prompt_tokens": 8, "completion_tokens": 6, "total_tokens": 14},
            "latency_seconds": 0.2,
        }

        output = run_a_rag_pipeline("What is RAG?", self.resources)

        self.assertEqual(output["retrieval_output"]["status"], "error")
        self.assertIn("evaluation_output", output)
        self.assertIn("final_output", output)

    @patch("main.generate_answer")
    @patch("main.execute_retrieval")
    def test_model_failure(self, mock_execute_retrieval: object, mock_generate_answer: object) -> None:
        """Fall back to grounded extractive output when model generation fails."""
        mock_execute_retrieval.return_value = {
            "tool_name": "keyword_search",
            "status": "success",
            "results": [{"chunk_id": "chunk_1", "score": 0.6, "snippet": "RAG context."}],
            "result_count": 1,
            "execution_summary": "keyword_search returned 1 results",
        }
        mock_generate_answer.return_value = {
            "status": "error",
            "provider": "ollama",
            "model_name": "llama3.1",
            "answer": "",
            "error_message": "Model unavailable",
        }

        output = run_a_rag_pipeline("Explain RAG", self.resources)

        self.assertEqual(output["answer_response"]["status"], "success")
        self.assertEqual(output["answer_response"]["provider"], "fallback")
        self.assertEqual(output["final_output"]["status"], "success")
        self.assertTrue(output["final_output"]["answer"])
        self.assertIn("validation_results", output)

    @patch("main.validate_grounding", side_effect=RuntimeError("Validation blew up"))
    @patch("main.generate_answer")
    @patch("main.execute_retrieval")
    def test_validation_failure(
        self,
        mock_execute_retrieval: object,
        mock_generate_answer: object,
        _mock_validate_grounding: object,
    ) -> None:
        """Return top-level error payload when validation stage raises unexpectedly."""
        mock_execute_retrieval.return_value = {
            "tool_name": "keyword_search",
            "status": "success",
            "results": [{"chunk_id": "chunk_1", "score": 0.8, "snippet": "RAG snippet."}],
            "result_count": 1,
            "execution_summary": "keyword_search returned 1 results",
        }
        mock_generate_answer.return_value = {
            "status": "success",
            "provider": "ollama",
            "model_name": "llama3.1",
            "answer": "RAG uses retrieval and generation.",
            "token_usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            "latency_seconds": 0.2,
        }

        output = run_a_rag_pipeline("Explain RAG", self.resources)

        self.assertEqual(output["status"], "error")
        self.assertIn("Validation blew up", output["error_message"])


if __name__ == "__main__":
    unittest.main()
