"""Runtime validation script for the Phase 2.1 document loader."""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indexer import load_documents


RAW_DATA_DIR = Path("data/raw")


def _parse_log_lines(log_text: str) -> list[dict[str, Any]]:
    """Convert JSON log lines into structured records."""
    records: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append({"event": "unparsed_log", "raw": line})
    return records


def _collect_report() -> dict[str, Any]:
    """Run the loader and collect validation metrics."""
    logger = logging.getLogger("src.indexer")
    previous_level = logger.level
    previous_propagate = logger.propagate
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)

    try:
        files_detected = sorted(path.name for path in RAW_DATA_DIR.iterdir() if path.is_file())
        documents = load_documents(RAW_DATA_DIR)
    except Exception as error:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate
        return {
            "crashed": True,
            "error_messages": [str(error)],
            "files_detected": [],
            "loaded_files": [],
            "skipped_files": [],
            "loaded_documents": 0,
            "total_files_detected": 0,
            "logging_present": False,
        }

    logger.removeHandler(handler)
    logger.setLevel(previous_level)
    logger.propagate = previous_propagate

    log_records = _parse_log_lines(stream.getvalue())
    loaded_files = [Path(document.source).name for document in documents]

    skipped_paths = [
        record.get("path", "")
        for record in log_records
        if record.get("event") in {"document_skipped_empty", "document_skipped_unsupported"}
    ]
    skipped_files = sorted({Path(path).name for path in skipped_paths if path})

    error_messages = [
        record.get("error", "")
        for record in log_records
        if record.get("event") == "document_load_failed"
    ]

    valid_sample = next((document for document in documents if document.title == "sample"), None)
    valid_text_clean = valid_sample is not None and "multiple lines" in valid_sample.text

    empty_handled = "empty.txt" in skipped_files and "empty.txt" not in loaded_files
    unsupported_handled = "test.xyz" in skipped_files and "test.xyz" not in loaded_files
    logging_present = len(log_records) > 0
    crashed = False

    passed = all(
        [
            not crashed,
            valid_sample is not None,
            valid_text_clean,
            empty_handled,
            unsupported_handled,
            logging_present,
            not error_messages,
        ]
    )

    return {
        "crashed": crashed,
        "total_files_detected": len(files_detected),
        "loaded_documents": len(documents),
        "skipped_files_count": len(skipped_files),
        "errors_count": len(error_messages),
        "files_detected": files_detected,
        "loaded_files": loaded_files,
        "skipped_files": skipped_files,
        "error_messages": error_messages,
        "logging_present": logging_present,
        "valid_text_clean": valid_text_clean,
        "final_status": "PASS" if passed else "FAIL",
    }


def main() -> None:
    """Execute runtime validation and print a structured report."""
    report = _collect_report()

    print("VALIDATION REPORT")
    print()
    print(f"Total files detected: {report['total_files_detected']}")
    print(f"Valid documents loaded: {report['loaded_documents']}")
    print(f"Skipped files: {report.get('skipped_files_count', 0)}")
    print(f"Errors: {report.get('errors_count', 0)}")
    print()
    print("DETAILS:")
    print(f"- Loaded files: {report['loaded_files']}")
    print(f"- Skipped files: {report['skipped_files']}")
    print(f"- Error messages: {report['error_messages']}")
    print(f"- Logging present: {report['logging_present']}")
    print(f"- Clean text verified: {report.get('valid_text_clean', False)}")
    print()
    print("FINAL STATUS:")
    print(report["final_status"])


if __name__ == "__main__":
    main()
