"""Unit tests for Task 2.5 index storage."""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from src.indexer import load_faiss_index, load_metadata, save_faiss_index, save_metadata


class FakeMatrix:
    """Minimal matrix wrapper that mimics the NumPy pieces used by the indexer."""

    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = [[float(value) for value in row] for row in rows]
        self.ndim = 2
        width = len(self._rows[0]) if self._rows else 0
        self.shape = (len(self._rows), width)

    def tolist(self) -> list[list[float]]:
        return [list(row) for row in self._rows]


class FakeNumpyModule:
    """Small subset of NumPy used for float32 matrix preparation."""

    float32 = "float32"

    def asarray(self, values: list[list[float]], dtype: str) -> FakeMatrix:
        if dtype != self.float32:
            raise ValueError("Unexpected dtype passed to fake NumPy module.")
        return FakeMatrix(values)


class FakeFaissIndex:
    """Minimal in-memory FAISS-like index for testing save/load flows."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.ntotal = 0
        self.vectors: list[list[float]] = []

    def add(self, matrix: FakeMatrix) -> None:
        rows = matrix.tolist()
        self.vectors.extend(rows)
        self.ntotal = len(self.vectors)


class FakeFaissModule:
    """FAISS stub that serializes indexes as JSON for tests."""

    def __init__(self) -> None:
        self.saved_paths: list[str] = []

    def IndexFlatIP(self, dimension: int) -> FakeFaissIndex:
        return FakeFaissIndex(dimension)

    def write_index(self, index: FakeFaissIndex, path: str) -> None:
        payload = {
            "dimension": index.dimension,
            "ntotal": index.ntotal,
            "vectors": index.vectors,
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")
        self.saved_paths.append(path)

    def read_index(self, path: str) -> FakeFaissIndex:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index = FakeFaissIndex(payload["dimension"])
        index.vectors = payload["vectors"]
        index.ntotal = payload["ntotal"]
        return index


class IndexStorageTests(unittest.TestCase):
    """Verify FAISS and metadata storage behavior."""

    def setUp(self) -> None:
        """Create writable test output paths."""
        self.output_dir = Path("tests/.tmp/test_index_storage")
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.output_dir / "faiss.index"
        self.metadata_path = self.output_dir / "metadata.json"
        self.fake_faiss = FakeFaissModule()
        self.fake_numpy = FakeNumpyModule()
        self.sentence_records = [
            {
                "sentence_id": "chunk_0_sentence_0",
                "chunk_id": "chunk_0",
                "text": "Alpha starts here.",
                "position": 0,
                "embedding": [0.1, 0.2, 0.3],
            },
            {
                "sentence_id": "chunk_0_sentence_1",
                "chunk_id": "chunk_0",
                "text": "Beta continues there.",
                "position": 1,
                "embedding": [0.4, 0.5, 0.6],
            },
        ]

    def tearDown(self) -> None:
        """Remove test output after each test."""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    def test_save_faiss_index_successfully(self) -> None:
        """Persist embedding vectors to a FAISS-compatible index file."""
        embeddings = [record["embedding"] for record in self.sentence_records]

        with (
            patch("src.indexer._get_faiss_module", return_value=self.fake_faiss),
            patch("src.indexer._get_numpy_module", return_value=self.fake_numpy),
        ):
            save_faiss_index(embeddings, str(self.index_path))

        self.assertTrue(self.index_path.exists())
        stored = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["dimension"], 3)
        self.assertEqual(stored["ntotal"], 2)

    def test_save_metadata_successfully(self) -> None:
        """Persist metadata and vector-position mapping as JSON."""
        save_metadata(self.sentence_records, str(self.metadata_path))

        self.assertTrue(self.metadata_path.exists())
        id_mapping_path = self.output_dir / "id_mapping.json"
        self.assertTrue(id_mapping_path.exists())

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        id_mapping = json.loads(id_mapping_path.read_text(encoding="utf-8"))
        self.assertEqual(len(metadata), 2)
        self.assertEqual(id_mapping["0"]["sentence_id"], "chunk_0_sentence_0")
        self.assertEqual(id_mapping["1"]["chunk_id"], "chunk_0")
        self.assertNotIn("embedding", metadata[0])

    def test_reload_index_and_metadata(self) -> None:
        """Load saved index and metadata without data loss."""
        embeddings = [record["embedding"] for record in self.sentence_records]

        with (
            patch("src.indexer._get_faiss_module", return_value=self.fake_faiss),
            patch("src.indexer._get_numpy_module", return_value=self.fake_numpy),
        ):
            save_faiss_index(embeddings, str(self.index_path))
            reloaded_index = load_faiss_index(str(self.index_path))

        save_metadata(self.sentence_records, str(self.metadata_path))
        reloaded_metadata = load_metadata(str(self.metadata_path))

        self.assertEqual(reloaded_index.ntotal, 2)
        self.assertEqual(reloaded_index.dimension, 3)
        self.assertEqual(len(reloaded_metadata["metadata"]), 2)
        self.assertEqual(reloaded_metadata["metadata"][0]["sentence_id"], "chunk_0_sentence_0")

    def test_validate_mapping_consistency(self) -> None:
        """Ensure vector positions map back to the correct sentence metadata."""
        save_metadata(self.sentence_records, str(self.metadata_path))

        payload = load_metadata(str(self.metadata_path))
        metadata = payload["metadata"]
        id_mapping = payload["id_mapping"]

        for position, record in enumerate(metadata):
            mapping_record = id_mapping[str(position)]
            self.assertEqual(mapping_record["sentence_id"], record["sentence_id"])
            self.assertEqual(mapping_record["chunk_id"], record["chunk_id"])

    def test_empty_input_handling(self) -> None:
        """Reject empty embeddings and allow empty metadata saves."""
        with (
            patch("src.indexer._get_faiss_module", return_value=self.fake_faiss),
            patch("src.indexer._get_numpy_module", return_value=self.fake_numpy),
        ):
            with self.assertRaises(ValueError):
                save_faiss_index([], str(self.index_path))

        save_metadata([], str(self.metadata_path))
        payload = load_metadata(str(self.metadata_path))
        self.assertEqual(payload["metadata"], [])
        self.assertEqual(payload["id_mapping"], {})


if __name__ == "__main__":
    unittest.main()
