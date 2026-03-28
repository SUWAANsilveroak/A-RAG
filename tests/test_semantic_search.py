"""Unit tests for Task 3.2 semantic search."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.retrieval.semantic import semantic_search


class FakeMatrix:
    """Minimal matrix wrapper compatible with the semantic module."""

    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = [[float(value) for value in row] for row in rows]
        self.ndim = 2

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, index: int) -> list[float]:
        return self.rows[index]


class FakeNumpyModule:
    """Minimal NumPy stub used to prepare FAISS query vectors."""

    float32 = "float32"

    def asarray(self, values: list[list[float]], dtype: str) -> FakeMatrix:
        if dtype != self.float32:
            raise ValueError("Unexpected dtype passed to fake NumPy module.")
        return FakeMatrix(values)


class FakeSemanticModel:
    """Deterministic embedding stub for semantic-search tests."""

    def __init__(self) -> None:
        self.embedding_map = {
            "faiss index": [1.0, 0.0],
            "vector database": [1.0, 0.0],
            "storage mapping": [0.0, 1.0],
            "unknown topic": [0.0, 0.0],
        }

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        return [self.embedding_map.get(text.lower(), [0.0, 0.0]) for text in texts]


class FakeFaissIndex:
    """Simple FAISS-like search stub using dot-product similarity."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.ntotal = len(vectors)

    def search(self, query_matrix: FakeMatrix, top_k: int) -> tuple[list[list[float]], list[list[int]]]:
        query_vector = query_matrix[0]
        ranked = []

        for index, vector in enumerate(self.vectors):
            score = sum(query_component * vector_component for query_component, vector_component in zip(query_vector, vector, strict=True))
            ranked.append((float(score), index))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        top_hits = ranked[:top_k]
        return [[score for score, _ in top_hits]], [[index for _, index in top_hits]]


class SemanticSearchTests(unittest.TestCase):
    """Verify semantic retrieval and chunk-level aggregation."""

    def setUp(self) -> None:
        """Prepare shared fake model, index, and metadata."""
        self.fake_numpy = FakeNumpyModule()
        self.model = FakeSemanticModel()
        self.metadata = [
            {"sentence_id": "s0", "chunk_id": "chunk_0", "text": "FAISS index stores vectors.", "position": 0},
            {"sentence_id": "s1", "chunk_id": "chunk_0", "text": "Dense vectors support semantic lookup.", "position": 1},
            {"sentence_id": "s2", "chunk_id": "chunk_1", "text": "Metadata preserves sentence mapping.", "position": 0},
            {"sentence_id": "s3", "chunk_id": "chunk_2", "text": "Keyword search handles exact acronyms.", "position": 0},
        ]
        self.faiss_index = FakeFaissIndex(
            vectors=[
                [1.0, 0.0],
                [0.8, 0.0],
                [0.0, 1.0],
                [0.2, 0.1],
            ]
        )

    def test_exact_semantic_match(self) -> None:
        """Return the strongest chunk for a semantically aligned query."""
        with patch("src.retrieval.semantic._get_numpy_module", return_value=self.fake_numpy):
            results = semantic_search("faiss index", self.model, self.faiss_index, self.metadata, top_k=2)

        self.assertEqual(results[0]["chunk_id"], "chunk_0")
        self.assertIn("FAISS index stores vectors.", results[0]["matched_sentences"])
        self.assertEqual(results[0]["snippet"], "FAISS index stores vectors.")

    def test_synonym_query(self) -> None:
        """Support meaning-based matches beyond exact sentence text."""
        with patch("src.retrieval.semantic._get_numpy_module", return_value=self.fake_numpy):
            results = semantic_search("vector database", self.model, self.faiss_index, self.metadata, top_k=2)

        self.assertEqual(results[0]["chunk_id"], "chunk_0")
        self.assertGreater(results[0]["score"], 0.0)

    def test_no_results(self) -> None:
        """Return a score-sorted list even when similarity is effectively zero."""
        zero_index = FakeFaissIndex(vectors=[[0.0, 0.0], [0.0, 0.0]])
        zero_metadata = [self.metadata[0], self.metadata[2]]

        with patch("src.retrieval.semantic._get_numpy_module", return_value=self.fake_numpy):
            results = semantic_search("unknown topic", self.model, zero_index, zero_metadata, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["score"], 0.0)

    def test_empty_query(self) -> None:
        """Return an empty list for an empty query."""
        with patch("src.retrieval.semantic._get_numpy_module", return_value=self.fake_numpy):
            self.assertEqual(semantic_search("", self.model, self.faiss_index, self.metadata, top_k=2), [])

    def test_ranking_consistency(self) -> None:
        """Aggregate multiple sentence hits into a stable chunk ranking."""
        with patch("src.retrieval.semantic._get_numpy_module", return_value=self.fake_numpy):
            results = semantic_search("storage mapping", self.model, self.faiss_index, self.metadata, top_k=3)

        self.assertEqual(results[0]["chunk_id"], "chunk_1")
        self.assertEqual(results[0]["snippet"], "Metadata preserves sentence mapping.")
        self.assertTrue(all(result["score"] >= 0.0 for result in results))


if __name__ == "__main__":
    unittest.main()
