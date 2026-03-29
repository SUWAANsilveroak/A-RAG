"""Unit tests for the chunk read tool wrapper."""

from __future__ import annotations

import unittest

from src.tools.reader import run_chunk_read_tool


class ChunkReadToolTests(unittest.TestCase):
    """Validate full chunk reads, status labeling, and output structure."""

    def setUp(self) -> None:
        """Prepare reusable chunk fixtures."""
        self.chunks = [
            {
                "chunk_id": "chunk_1",
                "text": "Full context for chunk one.",
                "position": 0,
            },
            {
                "chunk_id": "chunk_2",
                "text": "Full context for chunk two.",
                "position": 1,
            },
        ]

    def test_valid_chunk_id_returns_full_text(self) -> None:
        """Return full chunk text with status 'new' for a valid id."""
        response = run_chunk_read_tool(chunk_ids=["chunk_1"], chunks=self.chunks)

        self.assertEqual(response["tool_name"], "chunk_read")
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["requested_chunk_ids"], ["chunk_1"])
        self.assertEqual(len(response["results"]), 1)
        self.assertEqual(response["results"][0]["chunk_id"], "chunk_1")
        self.assertEqual(response["results"][0]["status"], "new")
        self.assertEqual(response["results"][0]["text"], "Full context for chunk one.")
        self.assertEqual(response["results"][0]["position"], 0)

    def test_multiple_chunk_ids_preserve_request_order(self) -> None:
        """Preserve request order while returning multiple chunk reads."""
        response = run_chunk_read_tool(chunk_ids=["chunk_2", "chunk_1"], chunks=self.chunks)

        self.assertEqual(response["status"], "success")
        self.assertEqual([item["chunk_id"] for item in response["results"]], ["chunk_2", "chunk_1"])
        self.assertEqual([item["status"] for item in response["results"]], ["new", "new"])

    def test_previously_read_chunk_is_detected(self) -> None:
        """Mark chunk as previously_read when its id already exists in read set."""
        read_chunk_ids = {"chunk_1"}

        response = run_chunk_read_tool(
            chunk_ids=["chunk_1", "chunk_2"],
            chunks=self.chunks,
            read_chunk_ids=read_chunk_ids,
        )

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["results"][0]["chunk_id"], "chunk_1")
        self.assertEqual(response["results"][0]["status"], "previously_read")
        self.assertEqual(response["results"][1]["chunk_id"], "chunk_2")
        self.assertEqual(response["results"][1]["status"], "new")
        self.assertIn("chunk_2", read_chunk_ids)

    def test_missing_chunk_id_returns_not_found(self) -> None:
        """Return not_found status with safe empty text payload for missing id."""
        response = run_chunk_read_tool(chunk_ids=["chunk_404"], chunks=self.chunks)

        self.assertEqual(response["status"], "success")
        self.assertEqual(len(response["results"]), 1)
        self.assertEqual(response["results"][0]["chunk_id"], "chunk_404")
        self.assertEqual(response["results"][0]["status"], "not_found")
        self.assertEqual(response["results"][0]["text"], "")
        self.assertEqual(response["results"][0]["position"], -1)

    def test_empty_input_returns_success_with_empty_results(self) -> None:
        """Return a valid empty success structure when no chunk ids are requested."""
        response = run_chunk_read_tool(chunk_ids=[], chunks=self.chunks)

        self.assertEqual(response["tool_name"], "chunk_read")
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["requested_chunk_ids"], [])
        self.assertEqual(response["results"], [])


if __name__ == "__main__":
    unittest.main()
