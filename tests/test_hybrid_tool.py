"""Unit tests for the hybrid search tool wrapper."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.tools.search_tools import run_hybrid_search_tool


class FakeMatrix:
    """Minimal matrix wrapper compatible with the semantic module."""

    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = [[float(value) for value in row] for row in rows]
        self.ndim = 2

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
    """Deterministic embedding stub for hybrid tool tests."""

    def __init__(self) -> None:
        self.embedding_map = {
            "faiss index": [1.0, 0.0],
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
            score = sum(
                query_component * vector_component
                for query_component, vector_component in zip(query_vector, vector, strict=True)
            )
            ranked.append((float(score), index))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        top_hits = ranked[:top_k]
        return [[score for score, _ in top_hits]], [[index for _, index in top_hits]]


class BrokenFaissIndex:
    """FAISS-like stub that forces the tool into its error path."""

    ntotal = 1


class HybridSearchToolTests(unittest.TestCase):
    """Validate the standardized hybrid tool contract."""

    def setUp(self) -> None:
        """Prepare shared fixtures for all hybrid tool tests."""
        self.fake_numpy = FakeNumpyModule()
        self.model = FakeSemanticModel()
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
        self.metadata = [
            {
                "sentence_id": "s0",
                "chunk_id": "chunk_0",
                "text": "FAISS stores dense vectors for semantic retrieval.",
                "position": 0,
            },
            {
                "sentence_id": "s1",
                "chunk_id": "chunk_1",
                "text": "Keyword search helps exact acronym lookup and structured IDs.",
                "position": 0,
            },
        ]
        self.faiss_index = FakeFaissIndex(vectors=[[1.0, 0.0], [0.0, 1.0]])

    def test_valid_hybrid_query_returns_success_structure(self) -> None:
        """Return standardized success output with preserved hybrid results."""
        with patch("src.retrieval.semantic._get_numpy_module", return_value=self.fake_numpy):
            response = run_hybrid_search_tool(
                "faiss index",
                self.chunks,
                self.model,
                self.faiss_index,
                self.metadata,
                top_k=2,
            )

        self.assertEqual(response["tool_name"], "hybrid_search")
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["query"], "faiss index")
        self.assertEqual(response["top_k"], 2)
        self.assertGreaterEqual(len(response["results"]), 1)
        first_result = response["results"][0]
        self.assertEqual(first_result["chunk_id"], "chunk_0")
        self.assertIn("keyword_score", first_result)
        self.assertIn("semantic_score", first_result)
        self.assertIn("combined_score", first_result)
        self.assertIn("matched_terms", first_result)
        self.assertIn("matched_sentences", first_result)
        self.assertIn("snippet", first_result)

    def test_empty_query_returns_success_with_no_results(self) -> None:
        """Treat an empty query as a safe no-result success response."""
        with patch("src.retrieval.semantic._get_numpy_module", return_value=self.fake_numpy):
            response = run_hybrid_search_tool(
                "",
                self.chunks,
                self.model,
                self.faiss_index,
                self.metadata,
                top_k=5,
            )

        self.assertEqual(response["tool_name"], "hybrid_search")
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["query"], "")
        self.assertEqual(response["top_k"], 5)
        self.assertEqual(response["results"], [])

    def test_no_results_returns_success_with_empty_results(self) -> None:
        """Return success with no results when neither lexical nor semantic hits exist."""
        empty_faiss_index = FakeFaissIndex(vectors=[])

        with patch("src.retrieval.semantic._get_numpy_module", return_value=self.fake_numpy):
            response = run_hybrid_search_tool(
                "unknown topic",
                self.chunks,
                self.model,
                empty_faiss_index,
                self.metadata,
                top_k=5,
            )

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["results"], [])

    def test_invalid_faiss_input_returns_error_structure(self) -> None:
        """Return a standardized error payload when the FAISS object is invalid."""
        with patch("src.retrieval.semantic._get_numpy_module", return_value=self.fake_numpy):
            response = run_hybrid_search_tool(
                "faiss index",
                self.chunks,
                self.model,
                BrokenFaissIndex(),
                self.metadata,
                top_k=2,
            )

        self.assertEqual(response["tool_name"], "hybrid_search")
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["results"], [])
        self.assertIn("error_message", response)
        self.assertTrue(response["error_message"])

    def test_missing_metadata_returns_keyword_only_success(self) -> None:
        """Allow missing metadata and preserve keyword-only results safely."""
        with patch("src.retrieval.semantic._get_numpy_module", return_value=self.fake_numpy):
            response = run_hybrid_search_tool(
                "faiss",
                self.chunks,
                self.model,
                self.faiss_index,
                [],
                top_k=2,
            )

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["tool_name"], "hybrid_search")
        self.assertGreaterEqual(len(response["results"]), 1)
        self.assertEqual(response["results"][0]["chunk_id"], "chunk_0")
        self.assertGreater(response["results"][0]["keyword_score"], 0.0)
        self.assertEqual(response["results"][0]["semantic_score"], 0.0)

    def test_missing_chunks_returns_error_structure(self) -> None:
        """Return a standardized error payload when chunks input is missing."""
        with patch("src.retrieval.semantic._get_numpy_module", return_value=self.fake_numpy):
            response = run_hybrid_search_tool(
                "faiss index",
                None,  # type: ignore[arg-type]
                self.model,
                self.faiss_index,
                self.metadata,
                top_k=2,
            )

        self.assertEqual(response["tool_name"], "hybrid_search")
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["results"], [])
        self.assertIn("error_message", response)
        self.assertTrue(response["error_message"])


if __name__ == "__main__":
    unittest.main()
