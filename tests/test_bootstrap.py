"""Bootstrap smoke tests for Phase 1.1."""

from __future__ import annotations

import unittest

from main import bootstrap_status


class BootstrapTests(unittest.TestCase):
    """Verify the project bootstrap entry point is wired correctly."""

    def test_bootstrap_status(self) -> None:
        """Return a structured initialization payload."""
        status = bootstrap_status()
        self.assertEqual(status["project"], "A-RAG")
        self.assertEqual(status["status"], "initialized")
        self.assertTrue(status["root"])


if __name__ == "__main__":
    unittest.main()
