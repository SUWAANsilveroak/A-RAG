"""Unit tests for Task 12.1 FastAPI layer."""

from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
    from src.api.app import app, set_pipeline_resources
    _FASTAPI_AVAILABLE = True
except ModuleNotFoundError:
    _FASTAPI_AVAILABLE = False


@unittest.skipUnless(_FASTAPI_AVAILABLE, "fastapi is not installed in the environment")
class ApiTests(unittest.TestCase):
    """Validate health, query, retry, and error handling endpoints."""

    def setUp(self) -> None:
        """Prepare API client and baseline pipeline resources."""
        set_pipeline_resources(
            {
                "chunks": [],
                "model": object(),
                "faiss_index": object(),
                "metadata": [],
                "read_chunk_ids": set(),
                "model_name": "llama3.1",
                "provider": "ollama",
            }
        )
        self.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        """Return API health status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    @patch("src.api.app.run_a_rag_pipeline")
    def test_successful_query_endpoint(self, mock_run_pipeline: object) -> None:
        """Return successful structured result for /query."""
        mock_run_pipeline.return_value = {
            "query": "What is RAG?",
            "final_output": {"status": "success"},
        }

        response = self.client.post("/query", json={"query": "What is RAG?"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "success")
        self.assertIn("result", payload)
        self.assertEqual(payload["result"]["query"], "What is RAG?")

    def test_empty_query(self) -> None:
        """Return error for empty query payload."""
        response = self.client.post("/query", json={"query": "   "})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "error")
        self.assertIn("Query must not be empty", payload["error_message"])

    @patch("src.api.app.run_retry_pipeline")
    def test_retry_endpoint(self, mock_run_retry_pipeline: object) -> None:
        """Return successful structured result for /query/retry."""
        mock_run_retry_pipeline.return_value = {
            "query": "Explain RAG",
            "retry_count": 1,
            "final_status": "success",
        }

        response = self.client.post("/query/retry", json={"query": "Explain RAG"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["result"]["retry_count"], 1)

    @patch("src.api.app.run_a_rag_pipeline")
    def test_pipeline_failure(self, mock_run_pipeline: object) -> None:
        """Return error response when pipeline reports failure."""
        mock_run_pipeline.return_value = {
            "status": "error",
            "query": "Bad query",
            "error_message": "Pipeline exploded",
        }

        response = self.client.post("/query", json={"query": "Bad query"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "error")
        self.assertIn("Pipeline exploded", payload["error_message"])


if __name__ == "__main__":
    unittest.main()
