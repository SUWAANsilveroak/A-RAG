"""Unit tests for Task 3.4 BM25 setup."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.retrieval.bm25_search import bm25_search, build_bm25_index


class FakeBM25Okapi:
    """Minimal BM25 stub for deterministic lexical-search tests."""

    def __init__(self, corpus_tokens: list[list[str]]) -> None:
        self.corpus_tokens = corpus_tokens

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for tokens in self.corpus_tokens:
            token_score = 0.0
            for query_token in query_tokens:
                token_score += float(tokens.count(query_token) * len(query_token))
            scores.append(token_score)
        return scores


class BM25SearchTests(unittest.TestCase):
    """Verify BM25 corpus construction and ranking behavior."""

    def setUp(self) -> None:
        """Prepare reusable chunk fixtures."""
        self.chunks = [
            {
                "chunk_id": "chunk_0",
                "text": "FAISS index stores dense vectors. Vector lookup stays fast.",
                "position": 0,
            },
            {
                "chunk_id": "chunk_1",
                "text": "Metadata mapping preserves sentence and chunk traceability.",
                "position": 1,
            },
            {
                "chunk_id": "chunk_2",
                "text": "Keyword retrieval works well for exact acronym search.",
                "position": 2,
            },
        ]

    def test_exact_keyword_match(self) -> None:
        """Return the chunk containing the exact BM25 keyword hit."""
        with patch("src.retrieval.bm25_search._get_bm25_class", return_value=FakeBM25Okapi):
            index = build_bm25_index(self.chunks)
            results = bm25_search("FAISS", index, self.chunks, top_k=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "chunk_0")
        self.assertIn("FAISS", results[0]["snippet"])

    def test_multi_word_query(self) -> None:
        """Rank a chunk higher when it matches multiple query terms."""
        with patch("src.retrieval.bm25_search._get_bm25_class", return_value=FakeBM25Okapi):
            index = build_bm25_index(self.chunks)
            results = bm25_search("metadata mapping", index, self.chunks, top_k=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "chunk_1")
        self.assertGreater(results[0]["score"], 0.0)

    def test_no_match(self) -> None:
        """Return an empty list when no chunk gets a positive score."""
        with patch("src.retrieval.bm25_search._get_bm25_class", return_value=FakeBM25Okapi):
            index = build_bm25_index(self.chunks)
            self.assertEqual(bm25_search("ollama", index, self.chunks, top_k=5), [])

    def test_empty_query(self) -> None:
        """Return an empty list for an empty query."""
        with patch("src.retrieval.bm25_search._get_bm25_class", return_value=FakeBM25Okapi):
            index = build_bm25_index(self.chunks)
            self.assertEqual(bm25_search("", index, self.chunks, top_k=5), [])

    def test_ranking_consistency(self) -> None:
        """Rank more relevant chunks above weaker lexical matches."""
        ranking_chunks = [
            {"chunk_id": "chunk_a", "text": "index index storage", "position": 0},
            {"chunk_id": "chunk_b", "text": "index storage", "position": 1},
            {"chunk_id": "chunk_c", "text": "storage", "position": 2},
        ]

        with patch("src.retrieval.bm25_search._get_bm25_class", return_value=FakeBM25Okapi):
            index = build_bm25_index(ranking_chunks)
            results = bm25_search("index storage", index, ranking_chunks, top_k=3)

        self.assertEqual([result["chunk_id"] for result in results], ["chunk_a", "chunk_b", "chunk_c"])
        self.assertGreater(results[0]["score"], results[1]["score"])
        self.assertGreater(results[1]["score"], results[2]["score"])


if __name__ == "__main__":
    unittest.main()
