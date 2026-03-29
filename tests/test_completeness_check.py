"""Unit tests for Task 9.3 completeness validation."""

from __future__ import annotations

import unittest

from src.agent.validator import check_completeness


class CompletenessCheckTests(unittest.TestCase):
    """Validate major-topic answer coverage checks."""

    def test_fully_covered_query(self) -> None:
        """Mark answer complete when all major query terms are covered."""
        query = "Explain faiss vector retrieval"
        final_output = {
            "status": "success",
            "answer": "FAISS supports vector retrieval for efficient similarity search.",
            "supporting_chunks": ["chunk_1"],
            "confidence": 0.9,
        }
        compressed_context = [
            {"chunk_id": "chunk_1", "score": 0.9, "compressed_text": "FAISS handles vector retrieval."},
        ]

        result = check_completeness(query, final_output, compressed_context)

        self.assertTrue(result["is_complete"])
        self.assertEqual(result["completeness_score"], 1.0)
        self.assertEqual(result["missing_topics"], [])

    def test_partial_answer(self) -> None:
        """Mark answer partial when some major query terms are missing."""
        query = "Explain faiss vector retrieval latency"
        final_output = {
            "status": "success",
            "answer": "FAISS supports vector retrieval.",
            "supporting_chunks": ["chunk_1"],
            "confidence": 0.7,
        }
        compressed_context = [
            {"chunk_id": "chunk_1", "score": 0.9, "compressed_text": "Latency details are available in docs."},
        ]

        result = check_completeness(query, final_output, compressed_context)

        self.assertFalse(result["is_complete"])
        self.assertGreater(result["completeness_score"], 0.0)
        self.assertLess(result["completeness_score"], 1.0)
        self.assertIn("latency", result["missing_topics"])

    def test_empty_answer(self) -> None:
        """Return incomplete when answer text is empty."""
        query = "What is semantic retrieval"
        final_output = {
            "status": "success",
            "answer": "   ",
            "supporting_chunks": ["chunk_1"],
            "confidence": 0.3,
        }
        compressed_context = [
            {"chunk_id": "chunk_1", "score": 0.8, "compressed_text": "Semantic retrieval compares meaning."},
        ]

        result = check_completeness(query, final_output, compressed_context)

        self.assertFalse(result["is_complete"])
        self.assertEqual(result["completeness_score"], 0.0)
        self.assertTrue(result["missing_topics"])

    def test_empty_query(self) -> None:
        """Return incomplete with explanatory note when query is empty."""
        final_output = {
            "status": "success",
            "answer": "Some answer",
            "supporting_chunks": ["chunk_1"],
            "confidence": 0.5,
        }
        compressed_context = []

        result = check_completeness("   ", final_output, compressed_context)

        self.assertFalse(result["is_complete"])
        self.assertEqual(result["completeness_score"], 0.0)
        self.assertIn("empty", result["notes"].lower())

    def test_missing_topic_detection(self) -> None:
        """Detect multiple missing topics accurately."""
        query = "Explain faiss indexing and metadata mapping"
        final_output = {
            "status": "success",
            "answer": "FAISS indexing is supported.",
            "supporting_chunks": ["chunk_1"],
            "confidence": 0.6,
        }
        compressed_context = [
            {"chunk_id": "chunk_1", "score": 0.8, "compressed_text": "Metadata mapping is mentioned in notes."},
        ]

        result = check_completeness(query, final_output, compressed_context)

        self.assertFalse(result["is_complete"])
        self.assertIn("metadata", result["missing_topics"])
        self.assertIn("mapping", result["missing_topics"])


if __name__ == "__main__":
    unittest.main()
