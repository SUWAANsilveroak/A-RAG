"""Unit tests for Task 8.1 answer generation via LiteLLM."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.agent.answer_generator import generate_answer


class AnswerGeneratorTests(unittest.TestCase):
    """Validate provider routing, structured success, and error handling."""

    def test_valid_ollama_call(self) -> None:
        """Return structured success output for a valid Ollama completion call."""
        fake_response = {
            "choices": [{"message": {"content": "Grounded answer from context."}}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
        }

        with patch("src.agent.answer_generator._call_litellm_completion", return_value=fake_response):
            result = generate_answer(
                prompt="What is FAISS?",
                model_name="llama3.1",
                provider="ollama",
                temperature=0.1,
                max_tokens=500,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["provider"], "ollama")
        self.assertEqual(result["model_name"], "llama3.1")
        self.assertEqual(result["answer"], "Grounded answer from context.")
        self.assertEqual(result["token_usage"]["total_tokens"], 42)
        self.assertGreaterEqual(result["latency_seconds"], 0.0)

    def test_empty_prompt(self) -> None:
        """Return structured error when prompt is empty."""
        result = generate_answer(
            prompt="   ",
            model_name="llama3.1",
            provider="ollama",
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["answer"], "")
        self.assertIn("Prompt must not be empty", result["error_message"])

    def test_invalid_provider(self) -> None:
        """Return structured error for unsupported provider values."""
        result = generate_answer(
            prompt="Test prompt",
            model_name="llama3.1",
            provider="invalid_provider",
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["answer"], "")
        self.assertIn("Unsupported provider", result["error_message"])

    def test_model_unavailable(self) -> None:
        """Return structured error when provider/model call fails."""
        with patch("src.agent.answer_generator._call_litellm_completion", side_effect=RuntimeError("Model unavailable")):
            result = generate_answer(
                prompt="Explain retrieval",
                model_name="llama3.1",
                provider="ollama",
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["answer"], "")
        self.assertIn("Model unavailable", result["error_message"])

    def test_empty_response(self) -> None:
        """Return structured error when model response has no answer content."""
        empty_response = {
            "choices": [{"message": {"content": "   "}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
        }
        with patch("src.agent.answer_generator._call_litellm_completion", return_value=empty_response):
            result = generate_answer(
                prompt="What is retrieval?",
                model_name="llama3.1",
                provider="ollama",
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["answer"], "")
        self.assertIn("empty answer", result["error_message"].lower())


if __name__ == "__main__":
    unittest.main()
