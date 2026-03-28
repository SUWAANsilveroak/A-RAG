"""Unit tests for Task 3.5 reranking."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src import retrieval
from src.retrieval import reranker


class FakeCrossEncoder:
    """Deterministic cross-encoder stub for reranking tests."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(list(pairs))
        return self.scores[: len(pairs)]


class RerankerTests(unittest.TestCase):
    """Verify reranking behavior and score combination."""

    def setUp(self) -> None:
        """Reset cached reranker models before each test."""
        reranker._RERANK_MODELS.clear()

    def test_strongly_relevant_snippet(self) -> None:
        """Keep a strong relevant snippet at the top after reranking."""
        retrieval_results = [
            {"chunk_id": "chunk_0", "snippet": "FAISS index stores dense vectors.", "combined_score": 0.8},
            {"chunk_id": "chunk_1", "snippet": "Project planning notes.", "combined_score": 0.7},
        ]
        fake_model = FakeCrossEncoder([0.9, 0.1])

        with patch("src.retrieval.reranker._load_reranker_model", return_value=fake_model):
            results = reranker.rerank_results("faiss vectors", retrieval_results, top_k=2)

        self.assertEqual(results[0]["chunk_id"], "chunk_0")
        self.assertGreater(results[0]["final_score"], results[1]["final_score"])

    def test_weak_snippet(self) -> None:
        """Allow reranking to demote weakly relevant content."""
        retrieval_results = [
            {"chunk_id": "chunk_0", "snippet": "Weak lexical overlap only.", "combined_score": 0.9},
            {"chunk_id": "chunk_1", "snippet": "Dense vectors improve semantic retrieval.", "combined_score": 0.6},
        ]
        fake_model = FakeCrossEncoder([0.1, 0.8])

        with patch("src.retrieval.reranker._load_reranker_model", return_value=fake_model):
            results = reranker.rerank_results("semantic vectors", retrieval_results, top_k=2)

        self.assertEqual(results[0]["chunk_id"], "chunk_1")

    def test_empty_result_list(self) -> None:
        """Return an empty list for empty rerank input."""
        self.assertEqual(reranker.rerank_results("query", [], top_k=5), [])

    def test_duplicate_chunk_id(self) -> None:
        """Keep duplicate chunk IDs as separate candidates rather than dropping them."""
        retrieval_results = [
            {"chunk_id": "chunk_0", "snippet": "First snippet.", "combined_score": 0.5},
            {"chunk_id": "chunk_0", "snippet": "Second snippet.", "combined_score": 0.4},
        ]
        fake_model = FakeCrossEncoder([0.8, 0.2])

        with patch("src.retrieval.reranker._load_reranker_model", return_value=fake_model):
            results = reranker.rerank_results("snippet", retrieval_results, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual([result["chunk_id"] for result in results], ["chunk_0", "chunk_0"])

    def test_ranking_improvement_validation(self) -> None:
        """Promote the more relevant candidate even when retrieval score starts lower."""
        retrieval_results = [
            {"chunk_id": "chunk_a", "snippet": "General notes.", "combined_score": 0.95},
            {"chunk_id": "chunk_b", "snippet": "FAISS improves vector retrieval precision.", "combined_score": 0.55},
        ]
        fake_model = FakeCrossEncoder([0.05, 0.95])

        with patch("src.retrieval.reranker._load_reranker_model", return_value=fake_model):
            results = reranker.rerank_results("FAISS retrieval", retrieval_results, top_k=2)

        self.assertEqual(results[0]["chunk_id"], "chunk_b")
        self.assertTrue(all(isinstance(result["final_score"], float) for result in results))


if __name__ == "__main__":
    unittest.main()
