"""Application entry point for the A-RAG bootstrap scaffold."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final


PROJECT_NAME: Final[str] = "A-RAG"


def configure_logging() -> None:
    """Configure a simple JSON-style logger for local development."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def bootstrap_status() -> dict[str, str]:
    """Return a structured snapshot of the bootstrap state."""
    return {
        "project": PROJECT_NAME,
        "status": "initialized",
        "root": str(Path(__file__).resolve().parent),
    }


def main() -> None:
    """Log the current bootstrap status."""
    configure_logging()
    logging.info(json.dumps(bootstrap_status()))


if __name__ == "__main__":
    main()
