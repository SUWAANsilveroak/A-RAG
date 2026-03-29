"""Unit tests for Task 6.3 retrieval execution node."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.agent.loop import execute_retrieval


class RetrievalNodeTests(unittest.TestCase):
    """Validate retrieval node tool dispatch and standardized output handling."""

    def setUp(self) -> None:
        """Prepare baseline resources and tool selection payload."""
        self.resources = {
            "chunks": [{"chunk_id": "chunk_1", "text": "Chunk one text", "position": 0}],
            "model": object(),
            "faiss_index": object(),
            "metadata": [{"sentence_id": "s1", "chunk_id": "chunk_1", "text": "Chunk one text", "position": 0}],
            "read_chunk_ids": set(),
        }

    def test_execute_keyword_tool(self) -> None:
        """Execute keyword wrapper and preserve standardized result output."""
        selection = {
            "tool_name": "keyword_search",
            "tool_input": {"query": "faiss", "top_k": 5},
            "execution_ready": True,
            "reason": "keyword route",
        }
        expected_payload = {
            "tool_name": "keyword_search",
            "status": "success",
            "results": [{"chunk_id": "chunk_1"}],
        }

        with patch("src.agent.loop.run_keyword_search_tool", return_value=expected_payload):
            output = execute_retrieval(selection, self.resources)

        self.assertEqual(output["tool_name"], "keyword_search")
        self.assertEqual(output["status"], "success")
        self.assertEqual(output["result_count"], 1)
        self.assertEqual(output["execution_summary"], "keyword_search returned 1 results")

    def test_execute_semantic_tool(self) -> None:
        """Execute semantic wrapper and produce semantic summary text."""
        selection = {
            "tool_name": "semantic_search",
            "tool_input": {"query": "semantic meaning", "top_k": 5},
            "execution_ready": True,
            "reason": "semantic route",
        }
        expected_payload = {
            "tool_name": "semantic_search",
            "status": "success",
            "results": [{"chunk_id": "chunk_1"}, {"chunk_id": "chunk_2"}],
        }

        with patch("src.agent.loop.run_semantic_search_tool", return_value=expected_payload):
            output = execute_retrieval(selection, self.resources)

        self.assertEqual(output["tool_name"], "semantic_search")
        self.assertEqual(output["status"], "success")
        self.assertEqual(output["result_count"], 2)
        self.assertEqual(output["execution_summary"], "semantic_search returned 2 results")

    def test_execute_hybrid_tool(self) -> None:
        """Execute hybrid wrapper and preserve empty-result summaries."""
        selection = {
            "tool_name": "hybrid_search",
            "tool_input": {"query": "mixed intent", "top_k": 5},
            "execution_ready": True,
            "reason": "hybrid route",
        }
        expected_payload = {
            "tool_name": "hybrid_search",
            "status": "success",
            "results": [],
        }

        with patch("src.agent.loop.run_hybrid_search_tool", return_value=expected_payload):
            output = execute_retrieval(selection, self.resources)

        self.assertEqual(output["tool_name"], "hybrid_search")
        self.assertEqual(output["status"], "success")
        self.assertEqual(output["result_count"], 0)
        self.assertEqual(output["execution_summary"], "hybrid_search returned no results")

    def test_execute_chunk_read_tool(self) -> None:
        """Execute chunk-read wrapper and emit chunk-specific summary wording."""
        selection = {
            "tool_name": "chunk_read",
            "tool_input": {"query": "read chunk", "top_k": 5, "chunk_ids": ["chunk_1", "chunk_2"]},
            "execution_ready": True,
            "reason": "chunk read route",
        }
        expected_payload = {
            "tool_name": "chunk_read",
            "status": "success",
            "results": [
                {"chunk_id": "chunk_1", "status": "new", "text": "Chunk one text", "position": 0},
                {"chunk_id": "chunk_2", "status": "not_found", "text": "", "position": -1},
            ],
        }

        with patch("src.agent.loop.run_chunk_read_tool", return_value=expected_payload):
            output = execute_retrieval(selection, self.resources)

        self.assertEqual(output["tool_name"], "chunk_read")
        self.assertEqual(output["status"], "success")
        self.assertEqual(output["result_count"], 2)
        self.assertEqual(output["execution_summary"], "chunk_read returned 2 chunks")

    def test_invalid_tool_name(self) -> None:
        """Return standardized error payload for unsupported tool names."""
        selection = {
            "tool_name": "nonexistent_tool",
            "tool_input": {"query": "test", "top_k": 5},
            "execution_ready": True,
            "reason": "invalid route",
        }

        output = execute_retrieval(selection, self.resources)

        self.assertEqual(output["tool_name"], "nonexistent_tool")
        self.assertEqual(output["status"], "error")
        self.assertEqual(output["results"], [])
        self.assertEqual(output["result_count"], 0)
        self.assertIn("Invalid tool", output["execution_summary"])

    def test_execution_ready_false(self) -> None:
        """Return standardized error payload when selection is not execution-ready."""
        selection = {
            "tool_name": "keyword_search",
            "tool_input": {"query": "", "top_k": 5},
            "execution_ready": False,
            "reason": "missing query",
        }

        output = execute_retrieval(selection, self.resources)

        self.assertEqual(output["tool_name"], "keyword_search")
        self.assertEqual(output["status"], "error")
        self.assertEqual(output["results"], [])
        self.assertEqual(output["result_count"], 0)
        self.assertEqual(output["execution_summary"], "Retrieval execution not ready")


if __name__ == "__main__":
    unittest.main()
