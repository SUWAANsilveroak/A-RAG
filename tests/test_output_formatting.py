"""Unit tests for Task 8.2 final output formatting."""

from __future__ import annotations

import unittest

from src.agent.answer_generator import format_final_output


class OutputFormattingTests(unittest.TestCase):
    """Validate standardized final output formatting and confidence behavior."""

    def test_successful_answer_response(self) -> None:
        """Format successful answer with supporting chunks and high confidence."""
        answer_response = {
            "status": "success",
            "provider": "ollama",
            "model_name": "llama3.1",
            "answer": "FAISS is a vector similarity library.",
            "token_usage": {"prompt_tokens": 22, "completion_tokens": 10, "total_tokens": 32},
            "latency_seconds": 0.5,
        }
        compressed_context = [
            {"chunk_id": "chunk_1", "score": 0.9, "compressed_text": "Context one"},
            {"chunk_id": "chunk_2", "score": 0.8, "compressed_text": "Context two"},
        ]

        result = format_final_output(answer_response, compressed_context)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], "FAISS is a vector similarity library.")
        self.assertEqual(result["supporting_chunks"], ["chunk_1", "chunk_2"])
        self.assertGreaterEqual(result["confidence"], 0.8)
        self.assertEqual(result["provider"], "ollama")
        self.assertEqual(result["model_name"], "llama3.1")

    def test_error_response(self) -> None:
        """Format error responses with low confidence."""
        answer_response = {
            "status": "error",
            "provider": "ollama",
            "model_name": "llama3.1",
            "answer": "",
            "error_message": "Model unavailable",
        }
        compressed_context = [{"chunk_id": "chunk_1", "score": 0.7, "compressed_text": "Context"}]

        result = format_final_output(answer_response, compressed_context)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["confidence"], 0.0)

    def test_empty_answer(self) -> None:
        """Handle empty answer text with reduced confidence."""
        answer_response = {
            "status": "success",
            "provider": "ollama",
            "model_name": "llama3.1",
            "answer": "   ",
            "token_usage": {"total_tokens": 12},
            "latency_seconds": 0.3,
        }
        compressed_context = [{"chunk_id": "chunk_1", "score": 0.9, "compressed_text": "Context"}]

        result = format_final_output(answer_response, compressed_context)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["supporting_chunks"], ["chunk_1"])
        self.assertLess(result["confidence"], 0.5)

    def test_missing_chunk_ids(self) -> None:
        """Skip missing chunk ids and preserve deduplicated order."""
        answer_response = {
            "status": "success",
            "provider": "ollama",
            "model_name": "llama3.1",
            "answer": "Answer exists",
            "token_usage": {},
            "latency_seconds": 0.1,
        }
        compressed_context = [
            {"score": 0.9, "compressed_text": "No chunk id"},
            {"chunk_id": "chunk_2", "score": 0.8, "compressed_text": "Context two"},
            {"chunk_id": "chunk_2", "score": 0.7, "compressed_text": "Duplicate chunk"},
        ]

        result = format_final_output(answer_response, compressed_context)

        self.assertEqual(result["supporting_chunks"], ["chunk_2"])

    def test_empty_context(self) -> None:
        """Handle empty context with valid output structure."""
        answer_response = {
            "status": "success",
            "provider": "groq",
            "model_name": "llama-3.1-8b",
            "answer": "Brief answer",
            "token_usage": {"total_tokens": 15},
            "latency_seconds": 0.2,
        }

        result = format_final_output(answer_response, [])

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["supporting_chunks"], [])
        self.assertIn("confidence", result)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
