"""Unit tests for Task 7.3 prompt builder."""

from __future__ import annotations

import unittest

from src.prompts import build_answer_prompt


class PromptBuilderTests(unittest.TestCase):
    """Validate grounded prompt structure and context insertion behavior."""

    def test_normal_query_with_context(self) -> None:
        """Build prompt with required sections and context chunks."""
        compressed_context = [
            {"chunk_id": "chunk_2", "score": 0.6, "compressed_text": "Second context."},
            {"chunk_id": "chunk_1", "score": 0.9, "compressed_text": "Top context."},
        ]

        prompt = build_answer_prompt("What is FAISS?", compressed_context)

        self.assertIn("SYSTEM:", prompt)
        self.assertIn("QUERY:", prompt)
        self.assertIn("CONTEXT:", prompt)
        self.assertIn("RULES:", prompt)
        self.assertIn("ANSWER:", prompt)
        self.assertIn("[Chunk chunk_1]", prompt)
        self.assertIn("Top context.", prompt)
        self.assertLess(prompt.find("[Chunk chunk_1]"), prompt.find("[Chunk chunk_2]"))

    def test_empty_context(self) -> None:
        """Handle empty context list with safe fallback context block."""
        prompt = build_answer_prompt("Any data?", [])
        self.assertIn("No retrieved context available.", prompt)
        self.assertIn("Any data?", prompt)

    def test_empty_query(self) -> None:
        """Handle empty query input without breaking prompt structure."""
        compressed_context = [
            {"chunk_id": "chunk_1", "score": 0.8, "compressed_text": "Context exists."},
        ]
        prompt = build_answer_prompt("   ", compressed_context)
        self.assertIn("(empty query)", prompt)
        self.assertIn("Context exists.", prompt)

    def test_missing_chunk_id(self) -> None:
        """Fallback to unknown_chunk label when chunk_id is missing."""
        compressed_context = [
            {"score": 0.7, "compressed_text": "Context with missing chunk id."},
        ]
        prompt = build_answer_prompt("Test query", compressed_context)
        self.assertIn("[Chunk unknown_chunk]", prompt)
        self.assertIn("Context with missing chunk id.", prompt)

    def test_large_context_list(self) -> None:
        """Include multiple context items and skip empty compressed text entries."""
        compressed_context = [
            {"chunk_id": "chunk_1", "score": 0.95, "compressed_text": "A"},
            {"chunk_id": "chunk_2", "score": 0.85, "compressed_text": "B"},
            {"chunk_id": "chunk_3", "score": 0.75, "compressed_text": ""},
            {"chunk_id": "chunk_4", "score": 0.65, "compressed_text": "D"},
        ]
        prompt = build_answer_prompt("Large context query", compressed_context)

        self.assertIn("[Chunk chunk_1]", prompt)
        self.assertIn("[Chunk chunk_2]", prompt)
        self.assertIn("[Chunk chunk_4]", prompt)
        self.assertNotIn("[Chunk chunk_3]", prompt)


if __name__ == "__main__":
    unittest.main()
