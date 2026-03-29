"""Unit tests for Task 7.1 snippet selection."""

from __future__ import annotations

import unittest

from src.utils.compression import select_top_snippets


class SnippetSelectionTests(unittest.TestCase):
    """Validate ranking, deduplication, and filtering for snippet selection."""

    def test_normal_ranked_results(self) -> None:
        """Select highest scored snippets in descending score order."""
        retrieval_results = [
            {"chunk_id": "chunk_1", "score": 0.2, "snippet": "Snippet 1"},
            {"chunk_id": "chunk_2", "score": 0.9, "snippet": "Snippet 2"},
            {"chunk_id": "chunk_3", "score": 0.6, "snippet": "Snippet 3"},
        ]

        selected = select_top_snippets(retrieval_results, top_k=2)

        self.assertEqual(len(selected), 2)
        self.assertEqual([item["chunk_id"] for item in selected], ["chunk_2", "chunk_3"])
        self.assertGreaterEqual(selected[0]["score"], selected[1]["score"])

    def test_duplicate_chunk_id(self) -> None:
        """Keep only highest scoring snippet for duplicate chunk ids."""
        retrieval_results = [
            {"chunk_id": "chunk_1", "score": 0.4, "snippet": "Lower snippet"},
            {"chunk_id": "chunk_1", "score": 0.8, "snippet": "Higher snippet"},
            {"chunk_id": "chunk_2", "score": 0.5, "snippet": "Second chunk"},
        ]

        selected = select_top_snippets(retrieval_results, top_k=5)

        self.assertEqual(len(selected), 2)
        chunk_1 = next(item for item in selected if item["chunk_id"] == "chunk_1")
        self.assertEqual(chunk_1["score"], 0.8)
        self.assertEqual(chunk_1["snippet"], "Higher snippet")

    def test_missing_snippet(self) -> None:
        """Remove results with missing or empty snippets."""
        retrieval_results = [
            {"chunk_id": "chunk_1", "score": 0.7, "snippet": ""},
            {"chunk_id": "chunk_2", "score": 0.6},
            {"chunk_id": "chunk_3", "score": 0.5, "snippet": "Valid snippet"},
        ]

        selected = select_top_snippets(retrieval_results, top_k=5)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["chunk_id"], "chunk_3")

    def test_empty_input(self) -> None:
        """Return empty selection for empty retrieval input."""
        selected = select_top_snippets([], top_k=5)
        self.assertEqual(selected, [])

    def test_top_k_larger_than_results(self) -> None:
        """Return all valid snippets when top_k exceeds result count."""
        retrieval_results = [
            {"chunk_id": "chunk_1", "score": 0.8, "snippet": "Snippet 1"},
            {"chunk_id": "chunk_2", "snippet": "Snippet without score"},
        ]

        selected = select_top_snippets(retrieval_results, top_k=10)

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["chunk_id"], "chunk_1")
        self.assertEqual(selected[1]["chunk_id"], "chunk_2")
        self.assertEqual(selected[1]["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
