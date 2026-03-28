"""Unit tests for Task 3.1 keyword search."""

from __future__ import annotations

import unittest

from src.retrieval.keyword import keyword_search


class KeywordSearchTests(unittest.TestCase):
    """Verify exact lexical chunk retrieval and ranking behavior."""

    def setUp(self) -> None:
        """Prepare reusable chunk fixtures."""
        self.chunks = [
            {
                "chunk_id": "chunk_0",
                "text": "FAISS stores dense vectors. The FAISS index supports exact vector lookup.",
                "position": 0,
            },
            {
                "chunk_id": "chunk_1",
                "text": "Keyword search is useful for exact acronym matching and structured IDs.",
                "position": 1,
            },
            {
                "chunk_id": "chunk_2",
                "text": "Metadata keeps sentence to chunk mapping for retrieval and traceability.",
                "position": 2,
            },
        ]

    def test_exact_term_match(self) -> None:
        """Return the chunk that contains the exact keyword."""
        results = keyword_search("FAISS", self.chunks, top_k=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "chunk_0")
        self.assertEqual(results[0]["matched_terms"], ["faiss"])
        self.assertIn("FAISS", results[0]["snippet"])

    def test_multi_keyword_query(self) -> None:
        """Support multiple keywords and keep the matched terms visible."""
        results = keyword_search("metadata retrieval", self.chunks, top_k=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "chunk_2")
        self.assertEqual(results[0]["matched_terms"], ["metadata", "retrieval"])
        self.assertGreater(results[0]["score"], 0.0)

    def test_no_match(self) -> None:
        """Return an empty list when no chunks match."""
        self.assertEqual(keyword_search("ollama", self.chunks, top_k=5), [])

    def test_empty_query(self) -> None:
        """Return an empty list for an empty or stop-word-only query."""
        self.assertEqual(keyword_search("", self.chunks, top_k=5), [])
        self.assertEqual(keyword_search("the and of", self.chunks, top_k=5), [])

    def test_ranking_correctness(self) -> None:
        """Rank chunks by frequency times keyword length."""
        ranking_chunks = [
            {
                "chunk_id": "chunk_a",
                "text": "index index index storage",
                "position": 0,
            },
            {
                "chunk_id": "chunk_b",
                "text": "index storage storage",
                "position": 1,
            },
        ]

        results = keyword_search("index storage", ranking_chunks, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "chunk_a")
        self.assertGreater(results[0]["score"], results[1]["score"])


if __name__ == "__main__":
    unittest.main()
