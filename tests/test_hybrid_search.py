"""Unit tests for Task 3.3 hybrid search."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.retrieval.hybrid import hybrid_search


class HybridSearchTests(unittest.TestCase):
    """Verify merging and ranking of keyword and semantic retrieval results."""

    def setUp(self) -> None:
        """Prepare reusable retrieval fixtures."""
        self.chunks = [
            {"chunk_id": "chunk_0", "text": "FAISS index stores vectors.", "position": 0},
            {"chunk_id": "chunk_1", "text": "Dense vector storage supports semantic lookup.", "position": 1},
            {"chunk_id": "chunk_2", "text": "Structured ID matching helps keyword retrieval.", "position": 2},
        ]
        self.metadata = [
            {"sentence_id": "s0", "chunk_id": "chunk_0", "text": "FAISS index stores vectors.", "position": 0},
            {"sentence_id": "s1", "chunk_id": "chunk_1", "text": "Dense vector storage supports semantic lookup.", "position": 0},
            {"sentence_id": "s2", "chunk_id": "chunk_2", "text": "Structured ID matching helps keyword retrieval.", "position": 0},
        ]
        self.model = object()
        self.faiss_index = object()

    def test_query_with_exact_keyword_and_semantic_intent(self) -> None:
        """Merge both result types and prefer semantic snippet when present."""
        keyword_results = [
            {"chunk_id": "chunk_0", "score": 10.0, "matched_terms": ["faiss"], "snippet": "keyword snippet"},
        ]
        semantic_results = [
            {
                "chunk_id": "chunk_0",
                "score": 2.0,
                "matched_sentences": ["FAISS index stores vectors."],
                "snippet": "FAISS index stores vectors.",
            }
        ]

        with (
            patch("src.retrieval.hybrid.keyword_search", return_value=keyword_results),
            patch("src.retrieval.hybrid.semantic_search", return_value=semantic_results),
        ):
            results = hybrid_search("faiss meaning", self.chunks, self.model, self.faiss_index, self.metadata, top_k=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "chunk_0")
        self.assertEqual(results[0]["matched_terms"], ["faiss"])
        self.assertEqual(results[0]["matched_sentences"], ["FAISS index stores vectors."])
        self.assertEqual(results[0]["snippet"], "FAISS index stores vectors.")

    def test_query_with_only_semantic_meaning(self) -> None:
        """Keep chunks returned only by semantic search."""
        with (
            patch("src.retrieval.hybrid.keyword_search", return_value=[]),
            patch(
                "src.retrieval.hybrid.semantic_search",
                return_value=[
                    {
                        "chunk_id": "chunk_1",
                        "score": 3.0,
                        "matched_sentences": ["Dense vector storage supports semantic lookup."],
                        "snippet": "Dense vector storage supports semantic lookup.",
                    }
                ],
            ),
        ):
            results = hybrid_search("vector meaning", self.chunks, self.model, self.faiss_index, self.metadata, top_k=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "chunk_1")
        self.assertEqual(results[0]["keyword_score"], 0.0)
        self.assertGreater(results[0]["semantic_score"], 0.0)

    def test_query_with_only_keyword_match(self) -> None:
        """Keep chunks returned only by keyword search."""
        with (
            patch(
                "src.retrieval.hybrid.keyword_search",
                return_value=[
                    {
                        "chunk_id": "chunk_2",
                        "score": 8.0,
                        "matched_terms": ["id"],
                        "snippet": "Structured ID matching helps keyword retrieval.",
                    }
                ],
            ),
            patch("src.retrieval.hybrid.semantic_search", return_value=[]),
        ):
            results = hybrid_search("ID", self.chunks, self.model, self.faiss_index, self.metadata, top_k=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "chunk_2")
        self.assertGreater(results[0]["keyword_score"], 0.0)
        self.assertEqual(results[0]["semantic_score"], 0.0)

    def test_empty_query(self) -> None:
        """Return an empty list for an empty query."""
        self.assertEqual(hybrid_search("", self.chunks, self.model, self.faiss_index, self.metadata, top_k=3), [])

    def test_ranking_correctness(self) -> None:
        """Combine normalized scores consistently across both retrieval types."""
        keyword_results = [
            {"chunk_id": "chunk_0", "score": 8.0, "matched_terms": ["faiss"], "snippet": "chunk 0 keyword"},
            {"chunk_id": "chunk_2", "score": 4.0, "matched_terms": ["id"], "snippet": "chunk 2 keyword"},
        ]
        semantic_results = [
            {
                "chunk_id": "chunk_1",
                "score": 5.0,
                "matched_sentences": ["Dense vector storage supports semantic lookup."],
                "snippet": "chunk 1 semantic",
            },
            {
                "chunk_id": "chunk_0",
                "score": 4.0,
                "matched_sentences": ["FAISS index stores vectors."],
                "snippet": "chunk 0 semantic",
            },
        ]

        with (
            patch("src.retrieval.hybrid.keyword_search", return_value=keyword_results),
            patch("src.retrieval.hybrid.semantic_search", return_value=semantic_results),
        ):
            results = hybrid_search(
                "faiss vector",
                self.chunks,
                self.model,
                self.faiss_index,
                self.metadata,
                top_k=3,
                keyword_weight=0.4,
                semantic_weight=0.6,
            )

        self.assertEqual([result["chunk_id"] for result in results], ["chunk_0", "chunk_1", "chunk_2"])
        self.assertGreater(results[0]["combined_score"], results[1]["combined_score"])


if __name__ == "__main__":
    unittest.main()
