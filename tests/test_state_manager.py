"""Unit tests for AgentStateManager centralized state updates."""

from __future__ import annotations

import unittest

from src.agent.state import AgentStateManager


class AgentStateManagerTests(unittest.TestCase):
    """Validate query, chunk, reasoning, tool, and reset update flows."""

    def setUp(self) -> None:
        """Create a fresh state manager for each test."""
        self.manager = AgentStateManager()

    def test_update_query(self) -> None:
        """Update current query through centralized state logic."""
        self.manager.update_after_query("What is hybrid retrieval?")
        full_state = self.manager.get_full_state()

        self.assertEqual(full_state["current_query"], "What is hybrid retrieval?")

    def test_update_chunk_read(self) -> None:
        """Update both tracker and memory after chunk read events."""
        self.manager.update_after_chunk_read("chunk_1", step=1, relevance_score=0.6)
        self.manager.update_after_chunk_read("chunk_1", step=2, relevance_score=0.9)

        full_state = self.manager.get_full_state()
        self.assertEqual(full_state["retrieved_chunks"], ["chunk_1"])
        self.assertEqual(full_state["read_chunk_ids"], ["chunk_1"])
        self.assertEqual(full_state["access_log"]["chunk_1"]["read_count"], 2)
        self.assertEqual(full_state["access_log"]["chunk_1"]["last_access_step"], 2)
        self.assertEqual(full_state["access_log"]["chunk_1"]["relevance_score"], 0.9)

    def test_update_reasoning_step(self) -> None:
        """Append reasoning steps through centralized state logic."""
        self.manager.update_after_reasoning_step("Run semantic search.")
        self.manager.update_after_reasoning_step("Read top chunk.")
        self.manager.update_after_reasoning_step("   ")

        full_state = self.manager.get_full_state()
        self.assertEqual(
            full_state["reasoning_steps"],
            ["Run semantic search.", "Read top chunk."],
        )

    def test_update_tool_call(self) -> None:
        """Append structured tool history records through manager updates."""
        self.manager.update_after_tool_call(
            tool_name="semantic_search",
            input_data={"query": "RAG", "top_k": 5},
            output_summary="Returned 3 chunks.",
        )
        self.manager.update_after_tool_call(
            tool_name="",
            input_data={"query": "RAG"},
            output_summary="Ignored empty tool name.",
        )

        full_state = self.manager.get_full_state()
        self.assertEqual(len(full_state["tool_history"]), 1)
        self.assertEqual(full_state["tool_history"][0]["tool_name"], "semantic_search")
        self.assertEqual(full_state["tool_history"][0]["input_data"], {"query": "RAG", "top_k": 5})
        self.assertEqual(full_state["tool_history"][0]["output_summary"], "Returned 3 chunks.")

    def test_full_state_retrieval(self) -> None:
        """Return merged state with required top-level keys."""
        self.manager.update_after_query("Explain FAISS usage")
        self.manager.update_after_chunk_read("chunk_2", step=3, relevance_score=0.8)
        self.manager.update_after_reasoning_step("Use retrieved context.")
        self.manager.update_after_tool_call(
            tool_name="chunk_read",
            input_data={"chunk_ids": ["chunk_2"]},
            output_summary="Read 1 chunk.",
        )

        full_state = self.manager.get_full_state()
        self.assertEqual(
            sorted(full_state.keys()),
            sorted(
                [
                    "current_query",
                    "retrieved_chunks",
                    "reasoning_steps",
                    "tool_history",
                    "read_chunk_ids",
                    "access_log",
                ]
            ),
        )
        self.assertEqual(full_state["current_query"], "Explain FAISS usage")
        self.assertEqual(full_state["retrieved_chunks"], ["chunk_2"])
        self.assertEqual(full_state["read_chunk_ids"], ["chunk_2"])
        self.assertEqual(full_state["access_log"]["chunk_2"]["read_count"], 1)

    def test_reset_all(self) -> None:
        """Clear both memory and tracker state via centralized reset."""
        self.manager.update_after_query("What is BM25?")
        self.manager.update_after_chunk_read("chunk_3", step=1, relevance_score=0.5)
        self.manager.update_after_reasoning_step("Try keyword search first.")
        self.manager.update_after_tool_call(
            tool_name="keyword_search",
            input_data={"query": "BM25"},
            output_summary="Returned 2 chunks.",
        )

        self.manager.reset_all()
        full_state = self.manager.get_full_state()

        self.assertEqual(full_state["current_query"], "")
        self.assertEqual(full_state["retrieved_chunks"], [])
        self.assertEqual(full_state["reasoning_steps"], [])
        self.assertEqual(full_state["tool_history"], [])
        self.assertEqual(full_state["read_chunk_ids"], [])
        self.assertEqual(full_state["access_log"], {})


if __name__ == "__main__":
    unittest.main()
