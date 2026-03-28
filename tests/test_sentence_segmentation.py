"""Unit tests for Task 2.3 sentence segmentation."""

from __future__ import annotations

import unittest

from src.indexer import segment_sentences


class SentenceSegmentationTests(unittest.TestCase):
    """Verify sentence splitting preserves order, mapping, and text coverage."""

    def test_normal_chunk_multiple_sentences(self) -> None:
        """Split a standard chunk into ordered sentence records."""
        chunks = [
            {
                "chunk_id": "chunk_0",
                "text": "Alpha starts here. Beta continues there. Gamma ends it.",
                "position": 0,
            }
        ]

        sentences = segment_sentences(chunks)

        self.assertEqual(len(sentences), 3)
        self.assertEqual(
            [sentence["text"] for sentence in sentences],
            ["Alpha starts here.", "Beta continues there.", "Gamma ends it."],
        )
        self.assertTrue(all(sentence["chunk_id"] == "chunk_0" for sentence in sentences))
        self.assertEqual([sentence["position"] for sentence in sentences], [0, 1, 2])
        self.assertEqual(len({sentence["sentence_id"] for sentence in sentences}), 3)

    def test_single_sentence_chunk(self) -> None:
        """Keep a short chunk as one sentence."""
        chunks = [{"chunk_id": "chunk_4", "text": "Only one sentence here.", "position": 4}]

        sentences = segment_sentences(chunks)

        self.assertEqual(
            sentences,
            [
                {
                    "sentence_id": "chunk_4_sentence_0",
                    "chunk_id": "chunk_4",
                    "text": "Only one sentence here.",
                    "position": 0,
                }
            ],
        )

    def test_empty_chunk_is_skipped(self) -> None:
        """Skip empty or whitespace-only chunks."""
        chunks = [
            {"chunk_id": "chunk_empty", "text": "   \n\t  ", "position": 0},
            {"chunk_id": "chunk_ok", "text": "Still valid.", "position": 1},
        ]

        sentences = segment_sentences(chunks)

        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0]["chunk_id"], "chunk_ok")
        self.assertEqual(sentences[0]["text"], "Still valid.")

    def test_long_paragraph_preserves_order_and_text_coverage(self) -> None:
        """Segment a longer paragraph without losing chunk text meaning."""
        chunk_text = (
            "A-RAG improves retrieval quality by preserving traceability. "
            "Sentence segmentation supports semantic search and aggregation. "
            "Each sentence must still map back to the original chunk. "
            "This keeps auditability intact even for long paragraphs."
        )
        chunks = [{"chunk_id": "chunk_long", "text": chunk_text, "position": 0}]

        sentences = segment_sentences(chunks)

        self.assertEqual(len(sentences), 4)
        self.assertEqual(" ".join(sentence["text"] for sentence in sentences), chunk_text)
        self.assertEqual([sentence["position"] for sentence in sentences], list(range(4)))
        self.assertEqual(len({sentence["sentence_id"] for sentence in sentences}), len(sentences))
        self.assertTrue(all(sentence["chunk_id"] == "chunk_long" for sentence in sentences))


if __name__ == "__main__":
    unittest.main()
