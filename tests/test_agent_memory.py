"""Unit tests for AgentMemory short-term state."""

from __future__ import annotations

import unittest

from src.agent.state import AgentMemory


class AgentMemoryTests(unittest.TestCase):
    """Validate query, chunk, reasoning, tool history, and reset behavior."""

    def setUp(self) -> None:
        """Create a fresh memory instance per test."""
        self.memory = AgentMemory()

    def test_set_query(self) -> None:
        """Set and update the current query, including empty query."""
        self.memory.set_query("What is RAG?")
        self.assertEqual(self.memory.get_query(), "What is RAG?")

        self.memory.set_query("   ")
        self.assertEqual(self.memory.get_query(), "")

    def test_add_retrieved_chunks(self) -> None:
        """Store retrieved chunks in insertion order."""
        self.memory.add_retrieved_chunk("chunk_1")
        self.memory.add_retrieved_chunk("chunk_2")

        self.assertEqual(self.memory.get_retrieved_chunks(), ["chunk_1", "chunk_2"])

    def test_prevent_duplicate_chunk_ids(self) -> None:
        """Ignore duplicate chunk ids while preserving order."""
        self.memory.add_retrieved_chunk("chunk_1")
        self.memory.add_retrieved_chunk("chunk_1")
        self.memory.add_retrieved_chunk("chunk_2")

        self.assertEqual(self.memory.get_retrieved_chunks(), ["chunk_1", "chunk_2"])

    def test_add_reasoning_step(self) -> None:
        """Append reasoning steps in sequence and ignore empty steps."""
        self.memory.add_reasoning_step("Use hybrid search first.")
        self.memory.add_reasoning_step("Read top chunk for context.")
        self.memory.add_reasoning_step("   ")

        self.assertEqual(
            self.memory.get_reasoning_steps(),
            ["Use hybrid search first.", "Read top chunk for context."],
        )

    def test_add_tool_call(self) -> None:
        """Store structured tool history entries and ignore empty tool names."""
        self.memory.add_tool_call(
            tool_name="semantic_search",
            input_data={"query": "RAG", "top_k": 5},
            output_summary="Returned 3 chunks.",
        )
        self.memory.add_tool_call(
            tool_name="",
            input_data={"query": "RAG"},
            output_summary="Should be ignored.",
        )

        history = self.memory.get_tool_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["tool_name"], "semantic_search")
        self.assertEqual(history[0]["input_data"], {"query": "RAG", "top_k": 5})
        self.assertEqual(history[0]["output_summary"], "Returned 3 chunks.")

    def test_reset_memory(self) -> None:
        """Clear all memory fields on reset."""
        self.memory.set_query("How does retrieval work?")
        self.memory.add_retrieved_chunk("chunk_1")
        self.memory.add_reasoning_step("Run keyword search.")
        self.memory.add_tool_call(
            tool_name="keyword_search",
            input_data={"query": "retrieval"},
            output_summary="Returned 2 chunks.",
        )

        self.memory.reset()

        self.assertEqual(self.memory.get_query(), "")
        self.assertEqual(self.memory.get_retrieved_chunks(), [])
        self.assertEqual(self.memory.get_reasoning_steps(), [])
        self.assertEqual(self.memory.get_tool_history(), [])


if __name__ == "__main__":
    unittest.main()
