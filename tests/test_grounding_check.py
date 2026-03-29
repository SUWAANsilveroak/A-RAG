"""Unit tests for Task 9.1 grounding validation."""

from __future__ import annotations

import unittest

from src.agent.validator import validate_grounding


class GroundingCheckTests(unittest.TestCase):
    """Validate answer grounding checks against compressed context."""

    def test_fully_grounded_answer(self) -> None:
        """Return grounded=true when all supporting chunks exist and answer is present."""
        final_output = {
            "status": "success",
            "answer": "FAISS is a vector index.",
            "supporting_chunks": ["chunk_1", "chunk_2"],
            "confidence": 0.9,
        }
        compressed_context = [
            {"chunk_id": "chunk_1", "score": 0.9, "compressed_text": "FAISS context 1"},
            {"chunk_id": "chunk_2", "score": 0.8, "compressed_text": "FAISS context 2"},
        ]

        result = validate_grounding(final_output, compressed_context)

        self.assertTrue(result["grounded"])
        self.assertEqual(result["missing_chunks"], [])
        self.assertEqual(result["grounding_score"], 1.0)
        self.assertIn("grounded", result["notes"].lower())

    def test_missing_supporting_chunks(self) -> None:
        """Detect missing supporting chunk references correctly."""
        final_output = {
            "status": "success",
            "answer": "Answer with missing support.",
            "supporting_chunks": ["chunk_1", "chunk_missing"],
            "confidence": 0.7,
        }
        compressed_context = [
            {"chunk_id": "chunk_1", "score": 0.9, "compressed_text": "Available context"},
        ]

        result = validate_grounding(final_output, compressed_context)

        self.assertFalse(result["grounded"])
        self.assertEqual(result["missing_chunks"], ["chunk_missing"])
        self.assertGreater(result["grounding_score"], 0.0)
        self.assertLess(result["grounding_score"], 1.0)

    def test_empty_answer(self) -> None:
        """Mark grounding false when answer text is empty."""
        final_output = {
            "status": "success",
            "answer": "   ",
            "supporting_chunks": ["chunk_1"],
            "confidence": 0.4,
        }
        compressed_context = [
            {"chunk_id": "chunk_1", "score": 0.8, "compressed_text": "Context"},
        ]

        result = validate_grounding(final_output, compressed_context)

        self.assertFalse(result["grounded"])
        self.assertEqual(result["grounding_score"], 0.0)
        self.assertIn("empty", result["notes"].lower())

    def test_empty_context(self) -> None:
        """Mark grounding false and report all supporting chunks as missing."""
        final_output = {
            "status": "success",
            "answer": "Has answer but no context.",
            "supporting_chunks": ["chunk_1", "chunk_2"],
            "confidence": 0.6,
        }

        result = validate_grounding(final_output, [])

        self.assertFalse(result["grounded"])
        self.assertEqual(result["missing_chunks"], ["chunk_1", "chunk_2"])
        self.assertEqual(result["grounding_score"], 0.0)

    def test_duplicate_chunk_ids(self) -> None:
        """Handle duplicate context chunk ids safely via set-based coverage."""
        final_output = {
            "status": "success",
            "answer": "Grounded answer",
            "supporting_chunks": ["chunk_1"],
            "confidence": 0.8,
        }
        compressed_context = [
            {"chunk_id": "chunk_1", "score": 0.9, "compressed_text": "Context A"},
            {"chunk_id": "chunk_1", "score": 0.7, "compressed_text": "Context B"},
        ]

        result = validate_grounding(final_output, compressed_context)

        self.assertTrue(result["grounded"])
        self.assertEqual(result["missing_chunks"], [])
        self.assertEqual(result["grounding_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
