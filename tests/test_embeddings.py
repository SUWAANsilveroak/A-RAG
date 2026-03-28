"""Unit tests for Task 2.4 embedding generation."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src import indexer


class FakeEmbeddingModel:
    """Simple embedding model stub for deterministic tests."""

    def __init__(self, outputs: list[list[float]] | None = None) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, object]] = []

    def encode(
        self,
        texts: list[str],
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        self.calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "convert_to_numpy": convert_to_numpy,
                "show_progress_bar": show_progress_bar,
            }
        )
        if self.outputs is not None:
            return self.outputs[: len(texts)]

        return [[float(index + 1), float(index + 2), float(index + 3)] for index, _ in enumerate(texts)]


class EmbeddingGenerationTests(unittest.TestCase):
    """Verify sentence embedding generation behavior."""

    def setUp(self) -> None:
        """Reset cached models before each test."""
        indexer._EMBEDDING_MODELS.clear()

    def test_normal_sentence_list(self) -> None:
        """Embed valid sentence records and preserve metadata."""
        sentences = [
            {"sentence_id": "s1", "chunk_id": "c1", "text": "Alpha sentence.", "position": 0},
            {"sentence_id": "s2", "chunk_id": "c1", "text": "Beta sentence.", "position": 1},
        ]
        fake_model = FakeEmbeddingModel(outputs=[[0.1, 0.2], [0.3, 0.4]])

        with patch("src.indexer._load_embedding_model", return_value=fake_model):
            embedded = indexer.generate_sentence_embeddings(sentences)

        self.assertEqual(len(embedded), 2)
        self.assertEqual(embedded[0]["sentence_id"], "s1")
        self.assertEqual(embedded[1]["chunk_id"], "c1")
        self.assertEqual(embedded[0]["position"], 0)
        self.assertEqual(embedded[0]["embedding"], [0.1, 0.2])
        self.assertEqual(fake_model.calls[0]["texts"], ["Alpha sentence.", "Beta sentence."])
        self.assertTrue(fake_model.calls[0]["normalize_embeddings"])

    def test_empty_sentence_list(self) -> None:
        """Return empty output for empty input."""
        self.assertEqual(indexer.generate_sentence_embeddings([]), [])

    def test_sentence_with_empty_text_is_skipped(self) -> None:
        """Skip empty text and missing identifiers safely."""
        sentences = [
            {"sentence_id": "s1", "chunk_id": "c1", "text": "  ", "position": 0},
            {"sentence_id": "", "chunk_id": "c1", "text": "Missing id.", "position": 1},
            {"sentence_id": "s3", "chunk_id": "c2", "text": "Valid text.", "position": 2},
        ]
        fake_model = FakeEmbeddingModel(outputs=[[0.5, 0.6, 0.7]])

        with patch("src.indexer._load_embedding_model", return_value=fake_model):
            embedded = indexer.generate_sentence_embeddings(sentences)

        self.assertEqual(len(embedded), 1)
        self.assertEqual(embedded[0]["sentence_id"], "s3")
        self.assertEqual(embedded[0]["embedding"], [0.5, 0.6, 0.7])
        self.assertEqual(fake_model.calls[0]["texts"], ["Valid text."])

    def test_embedding_dimension_consistency(self) -> None:
        """All returned embeddings must share the same dimension."""
        sentences = [
            {"sentence_id": "s1", "chunk_id": "c1", "text": "One.", "position": 0},
            {"sentence_id": "s2", "chunk_id": "c1", "text": "Two.", "position": 1},
        ]
        fake_model = FakeEmbeddingModel(outputs=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        with patch("src.indexer._load_embedding_model", return_value=fake_model):
            embedded = indexer.generate_sentence_embeddings(sentences)

        dimensions = {len(record["embedding"]) for record in embedded}
        self.assertEqual(dimensions, {3})
        self.assertEqual(len({record["sentence_id"] for record in embedded}), 2)


if __name__ == "__main__":
    unittest.main()
