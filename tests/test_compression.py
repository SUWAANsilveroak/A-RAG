"""Unit tests for Task 7.2 rule-based snippet compression."""

from __future__ import annotations

import unittest

from src.utils.compression import compress_snippets


class CompressionTests(unittest.TestCase):
    """Validate snippet compression behavior and output structure."""

    def test_normal_snippet_compression(self) -> None:
        """Keep first N sentences and preserve score ordering."""
        snippets = [
            {
                "chunk_id": "chunk_1",
                "score": 0.8,
                "snippet": "Sentence one. Sentence two. Sentence three.",
            },
            {
                "chunk_id": "chunk_2",
                "score": 0.6,
                "snippet": "Only one sentence here.",
            },
        ]

        compressed = compress_snippets(snippets, max_sentences_per_snippet=2, max_total_characters=3000)

        self.assertEqual(len(compressed), 2)
        self.assertEqual(compressed[0]["chunk_id"], "chunk_1")
        self.assertEqual(compressed[0]["compressed_text"], "Sentence one. Sentence two.")
        self.assertEqual(compressed[1]["compressed_text"], "Only one sentence here.")

    def test_long_snippet_truncation(self) -> None:
        """Truncate snippet text when remaining character budget is small."""
        long_text = "A" * 500
        snippets = [
            {"chunk_id": "chunk_1", "score": 0.9, "snippet": long_text},
        ]

        compressed = compress_snippets(snippets, max_sentences_per_snippet=2, max_total_characters=120)

        self.assertEqual(len(compressed), 1)
        self.assertLessEqual(len(compressed[0]["compressed_text"]), 120)
        self.assertTrue(compressed[0]["compressed_text"])

    def test_empty_snippet(self) -> None:
        """Skip snippets with missing or empty text."""
        snippets = [
            {"chunk_id": "chunk_1", "score": 0.8, "snippet": ""},
            {"chunk_id": "chunk_2", "score": 0.7},
            {"chunk_id": "chunk_3", "score": 0.6, "snippet": "Valid sentence."},
        ]

        compressed = compress_snippets(snippets, max_sentences_per_snippet=2, max_total_characters=3000)

        self.assertEqual(len(compressed), 1)
        self.assertEqual(compressed[0]["chunk_id"], "chunk_3")
        self.assertEqual(compressed[0]["compressed_text"], "Valid sentence.")

    def test_duplicate_sentences(self) -> None:
        """Remove duplicate sentences inside one snippet."""
        snippets = [
            {
                "chunk_id": "chunk_1",
                "score": 0.9,
                "snippet": "Repeat sentence. Repeat sentence. Unique sentence.",
            }
        ]

        compressed = compress_snippets(snippets, max_sentences_per_snippet=3, max_total_characters=3000)

        self.assertEqual(len(compressed), 1)
        self.assertEqual(compressed[0]["compressed_text"], "Repeat sentence. Unique sentence.")

    def test_character_limit_enforcement(self) -> None:
        """Enforce global max_total_characters across multiple snippets."""
        snippets = [
            {"chunk_id": "chunk_1", "score": 0.9, "snippet": "First sentence is long enough to consume budget quickly."},
            {"chunk_id": "chunk_2", "score": 0.8, "snippet": "Second sentence should be partially or fully skipped due to limit."},
        ]

        compressed = compress_snippets(snippets, max_sentences_per_snippet=2, max_total_characters=70)
        total_characters = sum(len(item["compressed_text"]) for item in compressed)

        self.assertLessEqual(total_characters, 70)
        self.assertGreaterEqual(len(compressed), 1)
        self.assertTrue(all(item["compressed_text"] for item in compressed))


if __name__ == "__main__":
    unittest.main()
