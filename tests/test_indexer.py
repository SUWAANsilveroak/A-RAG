"""Tests for the Phase 2.1 document loader."""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from src.indexer import clean_text, load_documents


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


if __name__ == "__main__":
    unittest.main()
