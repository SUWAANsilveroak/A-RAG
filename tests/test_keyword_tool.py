"""Unit tests for the keyword search tool wrapper."""

from __future__ import annotations

import unittest

from src.tools.search_tools import run_keyword_search_tool


class KeywordSearchToolTests(unittest.TestCase):
    """Validate the standardized keyword tool contract."""

    def setUp(self) -> None:
        """Prepare chunk fixtures shared across tests."""
        self.chunks = [
            {
                "chunk_id": "chunk_0",
                "text": "FAISS stores dense vectors for semantic retrieval.",
                "position": 0,
            },
            {
                "chunk_id": "chunk_1",
                "text": "Keyword search helps exact acronym lookup and structured IDs.",
                "position": 1,
            },
        ]

    def test_valid_query_returns_success_structure(self) -> None:
        """Return standardized success output with preserved search results."""
        response = run_keyword_search_tool("FAISS", self.chunks, top_k=3)

        self.assertEqual(response["tool_name"], "keyword_search")
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["query"], "FAISS")
        self.assertEqual(response["top_k"], 3)
        self.assertEqual(len(response["results"]), 1)
        self.assertEqual(response["results"][0]["chunk_id"], "chunk_0")
        self.assertEqual(response["results"][0]["matched_terms"], ["faiss"])
        self.assertGreater(response["results"][0]["score"], 0.0)

    def test_empty_query_returns_success_with_no_results(self) -> None:
        """Treat an empty query as a safe no-result success response."""
        response = run_keyword_search_tool("", self.chunks, top_k=5)

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["results"], [])
        self.assertEqual(response["query"], "")
        self.assertEqual(response["top_k"], 5)

    def test_no_results_returns_empty_results(self) -> None:
        """Return a success response even when no chunks match."""
        response = run_keyword_search_tool("ollama", self.chunks, top_k=5)

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["results"], [])

    def test_invalid_chunk_input_returns_error_structure(self) -> None:
        """Return a standardized error payload when chunk input is invalid."""
        response = run_keyword_search_tool("FAISS", None)  # type: ignore[arg-type]

        self.assertEqual(response["tool_name"], "keyword_search")
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["results"], [])
        self.assertIn("error_message", response)
        self.assertTrue(response["error_message"])


if __name__ == "__main__":
    unittest.main()
