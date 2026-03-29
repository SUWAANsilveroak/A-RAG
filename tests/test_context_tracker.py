"""Unit tests for the ContextTracker state component."""

from __future__ import annotations

import unittest

from src.agent.state import ContextTracker


class ContextTrackerTests(unittest.TestCase):
    """Validate chunk usage tracking, counters, and reset behavior."""

    def setUp(self) -> None:
        """Create a fresh tracker for each test."""
        self.tracker = ContextTracker()

    def test_mark_chunk_read(self) -> None:
        """Track first-time chunk reads and store access metadata."""
        self.tracker.mark_chunk_read("chunk_1", step=1, relevance_score=0.75)

        self.assertTrue(self.tracker.has_been_read("chunk_1"))
        self.assertEqual(self.tracker.get_read_chunks(), {"chunk_1"})
        self.assertEqual(
            self.tracker.get_chunk_access_info("chunk_1"),
            {
                "read_count": 1,
                "last_access_step": 1,
                "relevance_score": 0.75,
            },
        )

    def test_read_same_chunk_twice(self) -> None:
        """Increment read count and update last step on duplicate reads."""
        self.tracker.mark_chunk_read("chunk_1", step=1, relevance_score=0.5)
        self.tracker.mark_chunk_read("chunk_1", step=2)

        access_info = self.tracker.get_chunk_access_info("chunk_1")
        self.assertEqual(access_info["read_count"], 2)
        self.assertEqual(access_info["last_access_step"], 2)
        self.assertEqual(access_info["relevance_score"], 0.5)

        self.tracker.mark_chunk_read("chunk_1", step=3, relevance_score=0.9)
        access_info = self.tracker.get_chunk_access_info("chunk_1")
        self.assertEqual(access_info["read_count"], 3)
        self.assertEqual(access_info["last_access_step"], 3)
        self.assertEqual(access_info["relevance_score"], 0.9)

    def test_has_been_read(self) -> None:
        """Return correct boolean values for read and unread chunk ids."""
        self.assertFalse(self.tracker.has_been_read("chunk_1"))
        self.tracker.mark_chunk_read("chunk_1", step=1)
        self.assertTrue(self.tracker.has_been_read("chunk_1"))
        self.assertFalse(self.tracker.has_been_read("chunk_2"))

    def test_reset_state(self) -> None:
        """Clear read chunk ids and access log with reset."""
        self.tracker.mark_chunk_read("chunk_1", step=1, relevance_score=0.7)
        self.tracker.mark_chunk_read("chunk_2", step=2, relevance_score=0.4)

        self.tracker.reset()

        self.assertEqual(self.tracker.get_read_chunks(), set())
        self.assertEqual(self.tracker.get_chunk_access_info("chunk_1"), {})
        self.assertEqual(self.tracker.get_chunk_access_info("chunk_2"), {})
        self.assertFalse(self.tracker.has_been_read("chunk_1"))

    def test_missing_chunk_id_handling(self) -> None:
        """Ignore empty or missing chunk ids safely."""
        self.tracker.mark_chunk_read("", step=1, relevance_score=0.2)
        self.tracker.mark_chunk_read("   ", step=2, relevance_score=0.4)

        self.assertEqual(self.tracker.get_read_chunks(), set())
        self.assertEqual(self.tracker.get_chunk_access_info(""), {})
        self.assertFalse(self.tracker.has_been_read(""))


if __name__ == "__main__":
    unittest.main()
