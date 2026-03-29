"""Command-line interface for A-RAG pipeline execution."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import run_a_rag_pipeline, run_retry_pipeline  # noqa: E402


LOGGER = logging.getLogger(__name__)


class _CliArgumentParser(argparse.ArgumentParser):
    """ArgumentParser variant that raises a ValueError instead of exiting."""

    def error(self, message: str) -> None:
        """Raise parsing errors for structured CLI error handling."""
        raise ValueError(message)


def _build_default_resources() -> dict[str, Any]:
    """Build safe default resource structure for local CLI execution."""
    return {
        "chunks": [],
        "model": None,
        "faiss_index": None,
        "metadata": [],
        "read_chunk_ids": set(),
        "model_name": "llama3.1",
        "provider": "ollama",
    }


def _build_parser() -> _CliArgumentParser:
    """Build CLI parser with health/query/retry commands."""
    parser = _CliArgumentParser(prog="a-rag-cli", description="A-RAG command-line interface")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("health", help="Run health check")

    query_parser = subparsers.add_parser("query", help="Run single-pass pipeline")
    query_parser.add_argument("query", nargs="?", default="", help="Question text to query")

    retry_parser = subparsers.add_parser("retry", help="Run retry-enabled pipeline")
    retry_parser.add_argument("query", nargs="?", default="", help="Question text to query with retries")

    return parser


def _error_payload(error_message: str) -> dict[str, Any]:
    """Build standardized CLI error payload."""
    return {
        "status": "error",
        "error_message": str(error_message).strip() or "Unknown CLI error.",
    }


def run_cli(argv: list[str] | None = None) -> dict[str, Any]:
    """Execute CLI command and return structured JSON-serializable payload."""
    parser = _build_parser()
    start_time = time.perf_counter()

    try:
        args = parser.parse_args(argv if argv is not None else sys.argv[1:])
        command = str(getattr(args, "command", "")).strip()
        LOGGER.info("cli_command_started", extra={"command": command})

        if command == "health":
            result = {"status": "ok"}
            LOGGER.info("cli_command_completed", extra={"command": command, "status": "ok"})
            print(json.dumps(result, ensure_ascii=True))
            return result

        if command in {"query", "retry"}:
            query = str(getattr(args, "query", "")).strip()
            if not query:
                payload = _error_payload("Query must not be empty.")
                LOGGER.warning("cli_empty_query", extra={"command": command})
                print(json.dumps(payload, ensure_ascii=True))
                return payload

            resources = _build_default_resources()
            if command == "query":
                pipeline_result = run_a_rag_pipeline(query=query, resources=resources)
            else:
                pipeline_result = run_retry_pipeline(query=query, resources=resources)

            if isinstance(pipeline_result, dict) and str(pipeline_result.get("status", "")).lower() == "error":
                payload = _error_payload(str(pipeline_result.get("error_message", "Pipeline failed.")))
                LOGGER.error(
                    "cli_command_failed",
                    extra={"command": command, "query": query, "error_message": payload["error_message"]},
                )
                print(json.dumps(payload, ensure_ascii=True))
                return payload

            payload = {"status": "success", "result": pipeline_result}
            LOGGER.info("cli_command_completed", extra={"command": command, "query": query, "status": "success"})
            print(json.dumps(payload, ensure_ascii=True))
            return payload

        payload = _error_payload("Unknown command. Use: health, query, or retry.")
        LOGGER.error("cli_unknown_command", extra={"command": command})
        print(json.dumps(payload, ensure_ascii=True))
        return payload
    except ValueError as error:
        message = str(error)
        if "invalid choice" in message:
            message = "Unknown command. Use: health, query, or retry."
        payload = _error_payload(message)
        LOGGER.error("cli_parse_error", extra={"error_message": message})
        print(json.dumps(payload, ensure_ascii=True))
        return payload
    except Exception as error:
        payload = _error_payload(str(error))
        LOGGER.error("cli_unhandled_error", extra={"error_message": str(error)})
        print(json.dumps(payload, ensure_ascii=True))
        return payload
    finally:
        latency = time.perf_counter() - start_time
        LOGGER.info("cli_command_latency", extra={"seconds": latency})


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_cli()


if __name__ == "__main__":
    main()

