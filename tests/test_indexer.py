"""Tests for the Phase 2.1 document loader."""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from src.indexer import build_chunks, clean_text, load_documents, load_or_build_knowledge_base


class DocumentLoaderTests(unittest.TestCase):
    """Verify text normalization and document loading behavior."""

    def setUp(self) -> None:
        """Create a writable workspace-local directory for test fixtures."""
        self.raw_dir = Path("tests/.tmp/test_indexer")
        if self.raw_dir.exists():
            shutil.rmtree(self.raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        """Remove workspace-local test fixtures after each test."""
        if self.raw_dir.exists():
            shutil.rmtree(self.raw_dir)

    def test_clean_text_removes_noise(self) -> None:
        """Collapse repeated whitespace and trim noisy lines."""
        raw_text = " Hello\tworld \r\n\r\n\r\nSecond   line\x00 "
        self.assertEqual(clean_text(raw_text), "Hello world\n\nSecond line")

    def test_load_documents_returns_structured_text_documents(self) -> None:
        """Load valid text files from the raw directory."""
        (self.raw_dir / "sample.txt").write_text(" Alpha\tbeta \n\nGamma ", encoding="utf-8")

        documents = load_documents(self.raw_dir)
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].title, "sample")
        self.assertEqual(documents[0].text, "Alpha beta\n\nGamma")
        self.assertTrue(documents[0].doc_id)
        self.assertTrue(documents[0].source.endswith("sample.txt"))

    def test_load_documents_skips_empty_and_invalid_files(self) -> None:
        """Ignore empty and unsupported files without crashing."""
        (self.raw_dir / "empty.txt").write_text(" \n\t ", encoding="utf-8")
        (self.raw_dir / "notes.md").write_text("# unsupported", encoding="utf-8")

        documents = load_documents(self.raw_dir)
        self.assertEqual(documents, [])

    def test_load_documents_handles_missing_directory(self) -> None:
        """Return an empty list for a missing raw-data directory."""
        documents = load_documents(Path("missing-directory-for-tests"))
        self.assertEqual(documents, [])

    def test_build_chunks_scopes_chunk_ids_per_document(self) -> None:
        """Create globally unique chunk ids when multiple documents are indexed."""
        (self.raw_dir / "alpha.txt").write_text("Alpha sentence one. Alpha sentence two.", encoding="utf-8")
        (self.raw_dir / "beta.txt").write_text("Beta sentence one. Beta sentence two.", encoding="utf-8")

        documents = load_documents(self.raw_dir)
        chunks = build_chunks(documents, max_tokens=4)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), len(chunks))
        self.assertTrue(all(chunk["doc_id"] for chunk in chunks))
        self.assertTrue(all(chunk["source"].endswith(".txt") for chunk in chunks))

    @patch("src.indexer.build_knowledge_base")
    @patch("src.indexer.load_knowledge_base")
    def test_load_or_build_prefers_existing_index(
        self,
        mock_load_knowledge_base: object,
        mock_build_knowledge_base: object,
    ) -> None:
        """Load persisted resources when knowledge-base artifacts already exist."""
        expected_resources = {
            "chunks": [],
            "model": object(),
            "faiss_index": object(),
            "metadata": [],
            "read_chunk_ids": set(),
            "model_name": "llama3.1",
            "provider": "ollama",
        }
        mock_load_knowledge_base.return_value = expected_resources

        resources = load_or_build_knowledge_base()

        self.assertIs(resources, expected_resources)
        mock_build_knowledge_base.assert_not_called()

    @patch("src.indexer.build_knowledge_base")
    @patch("src.indexer.load_knowledge_base", side_effect=FileNotFoundError("missing KB"))
    def test_load_or_build_rebuilds_when_index_missing(
        self,
        _mock_load_knowledge_base: object,
        mock_build_knowledge_base: object,
    ) -> None:
        """Build resources from raw documents when persisted artifacts are absent."""
        expected_resources = {
            "chunks": [],
            "model": object(),
            "faiss_index": object(),
            "metadata": [],
            "read_chunk_ids": set(),
            "model_name": "llama3.1",
            "provider": "ollama",
        }
        mock_build_knowledge_base.return_value = expected_resources

        resources = load_or_build_knowledge_base()

        self.assertIs(resources, expected_resources)
        mock_build_knowledge_base.assert_called_once()


if __name__ == "__main__":
    unittest.main()
