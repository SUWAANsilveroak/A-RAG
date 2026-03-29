"""Unit tests for Task 9.2 conflict detection."""

from __future__ import annotations

import unittest

from src.agent.validator import detect_conflicts


class ConflictDetectionTests(unittest.TestCase):
    """Validate heuristic conflict detection across compressed chunks."""

    def test_contradictory_text(self) -> None:
        """Detect opposite statements such as yes/no across chunks."""
        compressed_context = [
            {"chunk_id": "chunk_1", "score": 0.9, "compressed_text": "Feature is enabled. The answer is yes."},
            {"chunk_id": "chunk_2", "score": 0.8, "compressed_text": "Feature is disabled. The answer is no."},
        ]

        result = detect_conflicts(compressed_context)

        self.assertTrue(result["has_conflict"])
        self.assertGreaterEqual(len(result["conflict_pairs"]), 1)
        self.assertIn("chunk_1", result["conflicting_chunks"])
        self.assertIn("chunk_2", result["conflicting_chunks"])

    def test_numeric_mismatch(self) -> None:
        """Detect numeric mismatch for same concept keywords."""
        compressed_context = [
            {"chunk_id": "chunk_1", "score": 0.9, "compressed_text": "Latency for model api is 120 ms in production."},
            {"chunk_id": "chunk_2", "score": 0.8, "compressed_text": "Latency for model api is 340 ms in production."},
        ]

        result = detect_conflicts(compressed_context)

        self.assertTrue(result["has_conflict"])
        self.assertTrue(any("Numeric mismatch" in pair["reason"] for pair in result["conflict_pairs"]))

    def test_no_conflict(self) -> None:
        """Return no conflict when snippets are consistent."""
        compressed_context = [
            {"chunk_id": "chunk_1", "score": 0.8, "compressed_text": "Service is active and processing requests."},
            {"chunk_id": "chunk_2", "score": 0.7, "compressed_text": "System remains stable during operation."},
        ]

        result = detect_conflicts(compressed_context)

        self.assertFalse(result["has_conflict"])
        self.assertEqual(result["conflicting_chunks"], [])
        self.assertEqual(result["conflict_pairs"], [])

    def test_empty_context(self) -> None:
        """Handle empty context safely without errors."""
        result = detect_conflicts([])
        self.assertFalse(result["has_conflict"])
        self.assertEqual(result["conflict_pairs"], [])

    def test_single_chunk(self) -> None:
        """Handle single chunk input with no pairwise conflict checks."""
        compressed_context = [
            {"chunk_id": "chunk_1", "score": 0.9, "compressed_text": "Only one chunk available."},
        ]
        result = detect_conflicts(compressed_context)

        self.assertFalse(result["has_conflict"])
        self.assertEqual(result["conflict_pairs"], [])


if __name__ == "__main__":
    unittest.main()
