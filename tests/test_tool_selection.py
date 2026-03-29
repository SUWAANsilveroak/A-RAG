"""Unit tests for Task 6.2 tool selection node."""

from __future__ import annotations

import unittest

from src.agent.loop import select_tool


class ToolSelectionTests(unittest.TestCase):
    """Validate planner-to-tool mapping and execution readiness behavior."""

    def test_keyword_search_planner_output(self) -> None:
        """Map keyword strategy to keyword tool with execution-ready input."""
        planner_output = {
            "original_query": "Find BM25 mentions",
            "rewritten_query": "bm25 mentions",
            "query_type": "keyword",
            "retrieval_strategy": "keyword_search",
            "reason": "Exact term lookup required.",
        }

        result = select_tool(planner_output)

        self.assertEqual(result["tool_name"], "keyword_search")
        self.assertEqual(result["tool_input"]["query"], "bm25 mentions")
        self.assertEqual(result["tool_input"]["top_k"], 5)
        self.assertTrue(result["execution_ready"])

    def test_semantic_search_planner_output(self) -> None:
        """Map semantic strategy to semantic tool with default top_k."""
        planner_output = {
            "original_query": "Explain vector similarity",
            "rewritten_query": "vector similarity explanation",
            "query_type": "semantic",
            "retrieval_strategy": "semantic_search",
            "reason": "Meaning-based retrieval needed.",
        }

        result = select_tool(planner_output)

        self.assertEqual(result["tool_name"], "semantic_search")
        self.assertEqual(result["tool_input"]["query"], "vector similarity explanation")
        self.assertEqual(result["tool_input"]["top_k"], 5)
        self.assertTrue(result["execution_ready"])

    def test_hybrid_search_planner_output(self) -> None:
        """Map hybrid strategy to hybrid tool with execution-ready payload."""
        planner_output = {
            "original_query": "Find exact FAISS docs with semantic context",
            "rewritten_query": "faiss docs semantic context",
            "query_type": "hybrid",
            "retrieval_strategy": "hybrid_search",
            "reason": "Need lexical and semantic signals.",
        }

        result = select_tool(planner_output)

        self.assertEqual(result["tool_name"], "hybrid_search")
        self.assertEqual(result["tool_input"]["query"], "faiss docs semantic context")
        self.assertEqual(result["tool_input"]["top_k"], 5)
        self.assertTrue(result["execution_ready"])

    def test_chunk_read_planner_output(self) -> None:
        """Map chunk_read strategy and include chunk_ids in tool input."""
        planner_output = {
            "original_query": "Read selected chunks",
            "rewritten_query": "read selected chunks",
            "query_type": "read",
            "retrieval_strategy": "chunk_read",
            "reason": "Need full chunk text.",
            "chunk_ids": ["chunk_1", "chunk_2", "   "],
        }

        result = select_tool(planner_output)

        self.assertEqual(result["tool_name"], "chunk_read")
        self.assertEqual(result["tool_input"]["query"], "read selected chunks")
        self.assertEqual(result["tool_input"]["top_k"], 5)
        self.assertEqual(result["tool_input"]["chunk_ids"], ["chunk_1", "chunk_2"])
        self.assertTrue(result["execution_ready"])

    def test_unknown_strategy_uses_fallback(self) -> None:
        """Fallback safely to hybrid_search for unknown strategy values."""
        planner_output = {
            "original_query": "Some unclear request",
            "rewritten_query": "unclear request",
            "query_type": "unknown",
            "retrieval_strategy": "something_else",
            "reason": "Unknown strategy from planner.",
        }

        result = select_tool(planner_output)

        self.assertEqual(result["tool_name"], "hybrid_search")
        self.assertEqual(result["tool_input"]["query"], "unclear request")
        self.assertEqual(result["tool_input"]["top_k"], 5)
        self.assertTrue(result["execution_ready"])
        self.assertIn("Fallback strategy applied.", result["reason"])


if __name__ == "__main__":
    unittest.main()
