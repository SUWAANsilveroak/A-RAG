"""Unit tests for Task 6.1 query planner node."""

from __future__ import annotations

import unittest

from src.agent.planner import plan_query


class QueryPlannerTests(unittest.TestCase):
    """Validate query classification, strategy mapping, and rewrite behavior."""

    def test_exact_keyword_query(self) -> None:
        """Prefer keyword search for exact identifiers/acronyms."""
        result = plan_query("Find record ID ABC123 in FAISS.")

        self.assertEqual(result["query_type"], "keyword")
        self.assertEqual(result["retrieval_strategy"], "keyword_search")
        self.assertEqual(result["original_query"], "Find record ID ABC123 in FAISS.")
        self.assertEqual(result["rewritten_query"], "Find record ID ABC123 in FAISS")

    def test_natural_language_query(self) -> None:
        """Prefer semantic search for broad natural-language requests."""
        result = plan_query("How does semantic retrieval work across this document collection?")

        self.assertEqual(result["query_type"], "semantic")
        self.assertEqual(result["retrieval_strategy"], "semantic_search")
        self.assertIn("semantic", result["retrieval_strategy"])
        self.assertTrue(result["rewritten_query"])

    def test_mixed_query(self) -> None:
        """Prefer hybrid search for mixed exact + broad context queries."""
        result = plan_query("Explain FAISS behavior for ID ABC123 with broader context.")

        self.assertEqual(result["query_type"], "hybrid")
        self.assertEqual(result["retrieval_strategy"], "hybrid_search")

    def test_chunk_lookup_query(self) -> None:
        """Prefer chunk_read for explicit chunk lookup requests."""
        result = plan_query("Read chunk_12 from previous results.")

        self.assertEqual(result["query_type"], "chunk_lookup")
        self.assertEqual(result["retrieval_strategy"], "chunk_read")
        self.assertIn("chunk_12", result["rewritten_query"])

    def test_empty_query(self) -> None:
        """Fallback safely for empty queries."""
        result = plan_query("   ")

        self.assertEqual(result["query_type"], "unknown")
        self.assertEqual(result["retrieval_strategy"], "fallback_search")
        self.assertEqual(result["rewritten_query"], "")


if __name__ == "__main__":
    unittest.main()
