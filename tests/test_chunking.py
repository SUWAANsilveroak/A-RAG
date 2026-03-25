"""Unit tests for Task 2.2 chunking."""

from __future__ import annotations

import unittest

from src.utils.chunking import create_chunks


class ChunkingTests(unittest.TestCase):
    """Verify sentence-preserving chunking behavior."""

    def test_normal_text_multiple_chunks(self) -> None:
        text = (
            "Alpha beta gamma. "
            "Delta epsilon zeta. "
            "Eta theta iota. "
            "Kappa lambda mu. "
            "Nu xi omicron."
        )

        # Each sentence has 3 words; using max_tokens=8 should pack 2 sentences/chunk.
        chunks = create_chunks(text, max_tokens=8)
        self.assertEqual(len(chunks), 3)

        expected_sentences = [
            "Alpha beta gamma.",
            "Delta epsilon zeta.",
            "Eta theta iota.",
            "Kappa lambda mu.",
            "Nu xi omicron.",
        ]
        expected_cleaned = " ".join(expected_sentences)
        reconstructed = " ".join(chunk["text"] for chunk in chunks)
        self.assertEqual(reconstructed, expected_cleaned)

        # Ensure sentence order and structure are preserved.
        for idx, chunk in enumerate(chunks):
            self.assertEqual(chunk["position"], idx)
            self.assertIsInstance(chunk["chunk_id"], str)
            self.assertTrue(chunk["text"])

        # No sentence should be split across chunks.
        for sentence in expected_sentences:
            containing = [c for c in chunks if sentence in c["text"]]
            self.assertEqual(len(containing), 1)

    def test_short_text_single_chunk(self) -> None:
        text = "Alpha beta gamma."
        chunks = create_chunks(text, max_tokens=1000)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["position"], 0)
        self.assertEqual(chunks[0]["text"], "Alpha beta gamma.")

    def test_empty_text_returns_empty_list(self) -> None:
        self.assertEqual(create_chunks("", max_tokens=1000), [])
        self.assertEqual(create_chunks("   \n\t  ", max_tokens=1000), [])

    def test_long_sentence_overflow_handled(self) -> None:
        # One sentence that is too large to fit max_tokens; it should still appear whole in a single chunk.
        sentence = " ".join(["word"] * 30).strip() + "."
        chunks = create_chunks(sentence, max_tokens=5)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["position"], 0)
        self.assertEqual(chunks[0]["text"], sentence)
        self.assertEqual(chunks[0]["chunk_id"], "chunk_0")

    def test_whitespace_normalization_preserves_text(self) -> None:
        # Extra whitespace should be normalized, but sentence boundaries must remain intact.
        text = "Alpha\tbeta \n\nGamma."
        chunks = create_chunks(text, max_tokens=1000)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["position"], 0)
        # Tabs/newlines are collapsed into single spaces; final sentence kept intact.
        self.assertEqual(chunks[0]["text"], "Alpha beta Gamma.")

    def test_question_mark_sentence_split(self) -> None:
        # Ensure regex fallback recognizes '?' as a sentence boundary.
        text = "Do you agree? Yes I do."
        chunks = create_chunks(text, max_tokens=12)

        self.assertGreaterEqual(len(chunks), 1)
        reconstructed = " ".join(chunk["text"] for chunk in chunks)
        self.assertEqual(reconstructed, "Do you agree? Yes I do.")

        # Both sentences must exist fully in order without splitting.
        self.assertEqual(sum(1 for c in chunks if "Do you agree?" in c["text"]), 1)
        self.assertEqual(sum(1 for c in chunks if "Yes I do." in c["text"]), 1)

    def test_no_punctuation_single_sentence(self) -> None:
        # Without punctuation, sentence segmentation may return the whole text as one sentence.
        text = "word word word word"
        chunks = create_chunks(text, max_tokens=2)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["position"], 0)
        self.assertEqual(chunks[0]["text"], "word word word word")


if __name__ == "__main__":
    unittest.main()

