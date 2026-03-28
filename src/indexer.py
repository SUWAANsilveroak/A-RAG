"""Document loading utilities for the indexing pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
SUPPORTED_TEXT_SUFFIXES = {".txt"}
SUPPORTED_PDF_SUFFIXES = {".pdf"}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SPACY_SENTENCIZER = None
_SPACY_AVAILABLE: bool | None = None


@dataclass(slots=True)
class DocumentRecord:
    """Structured document object returned by the loader."""

    doc_id: str
    title: str
    source: str
    created_at: str
    text: str


def _log_event(event: str, **fields: Any) -> None:
    """Emit structured logs for loader operations."""
    payload = {"event": event, **fields}
    LOGGER.info(json.dumps(payload, default=str))


def clean_text(text: str) -> str:
    """Normalize whitespace and remove common formatting noise."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    normalized = normalized.replace("\t", " ")
    normalized = re.sub(r"[ \f\v]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    cleaned = "\n".join(line.strip() for line in normalized.splitlines())
    return cleaned.strip()


def _build_doc_id(file_path: Path) -> str:
    """Create a deterministic document identifier from the file path."""
    digest = hashlib.sha256(str(file_path.resolve()).encode("utf-8")).hexdigest()
    return digest[:16]


def _build_document_record(file_path: Path, text: str) -> DocumentRecord:
    """Convert a file and cleaned text into a structured document object."""
    timestamp = datetime.fromtimestamp(file_path.stat().st_mtime, tz=UTC).isoformat()
    return DocumentRecord(
        doc_id=_build_doc_id(file_path),
        title=file_path.stem,
        source=str(file_path.resolve()),
        created_at=timestamp,
        text=text,
    )


def _read_text_file(file_path: Path) -> str:
    """Read a UTF-8 text file and tolerate decode issues safely."""
    return file_path.read_text(encoding="utf-8", errors="replace")


def _read_pdf_file(file_path: Path) -> str:
    """Read a PDF file if an optional PDF parser is available."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as error:
        raise RuntimeError("PDF support requires the optional 'pypdf' package.") from error

    reader = PdfReader(str(file_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _read_file(file_path: Path) -> str:
    """Dispatch file reading based on suffix."""
    suffix = file_path.suffix.lower()
    if suffix in SUPPORTED_TEXT_SUFFIXES:
        return _read_text_file(file_path)
    if suffix in SUPPORTED_PDF_SUFFIXES:
        return _read_pdf_file(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")


def load_documents(raw_data_dir: str | Path = Path("data/raw")) -> list[DocumentRecord]:
    """Load and normalize supported documents from the raw-data directory."""
    directory = Path(raw_data_dir)
    documents: list[DocumentRecord] = []

    if not directory.exists():
        _log_event("document_directory_missing", path=str(directory))
        return documents

    if not directory.is_dir():
        _log_event("document_directory_invalid", path=str(directory))
        return documents

    candidates = sorted(path for path in directory.iterdir() if path.is_file())
    _log_event("document_scan_started", path=str(directory.resolve()), file_count=len(candidates))

    for file_path in candidates:
        try:
            raw_text = _read_file(file_path)
            cleaned_text = clean_text(raw_text)
            if not cleaned_text:
                _log_event("document_skipped_empty", path=str(file_path.resolve()))
                continue

            document = _build_document_record(file_path, cleaned_text)
            documents.append(document)
            _log_event(
                "document_loaded",
                doc_id=document.doc_id,
                path=document.source,
                character_count=len(document.text),
            )
        except ValueError:
            _log_event("document_skipped_unsupported", path=str(file_path.resolve()))
        except Exception as error:
            _log_event(
                "document_load_failed",
                path=str(file_path.resolve()),
                error=str(error),
            )

    _log_event("document_scan_completed", loaded_count=len(documents))
    return documents


def _segment_text_into_sentences(text: str) -> list[str]:
    """Split text into ordered sentences using spaCy when available."""
    global _SPACY_SENTENCIZER
    global _SPACY_AVAILABLE

    stripped_text = text.strip()
    if not stripped_text:
        return []

    if _SPACY_AVAILABLE is False:
        parts = _SENTENCE_SPLIT_RE.split(stripped_text)
        return [part.strip() for part in parts if part and part.strip()]

    try:
        import spacy  # type: ignore

        if _SPACY_SENTENCIZER is None:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            _SPACY_SENTENCIZER = nlp

        _SPACY_AVAILABLE = True
        doc = _SPACY_SENTENCIZER(stripped_text)
        sentences = [sentence.text.strip() for sentence in doc.sents if sentence.text and sentence.text.strip()]
        return sentences
    except Exception:
        _SPACY_AVAILABLE = False
        parts = _SENTENCE_SPLIT_RE.split(stripped_text)
        return [part.strip() for part in parts if part and part.strip()]


def _build_sentence_id(chunk_id: str, position: int) -> str:
    """Create a deterministic sentence identifier within a chunk."""
    return f"{chunk_id}_sentence_{position}"


def segment_sentences(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split chunk text into ordered sentence records with strict chunk mapping."""
    sentence_records: list[dict[str, Any]] = []
    processed_chunks = 0

    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        chunk_text = str(chunk.get("text", "")).strip()
        if not chunk_text:
            continue

        processed_chunks += 1
        sentences = _segment_text_into_sentences(chunk_text)
        if not sentences:
            sentences = [chunk_text]

        for position, sentence_text in enumerate(sentences):
            sentence_records.append(
                {
                    "sentence_id": _build_sentence_id(chunk_id, position),
                    "chunk_id": chunk_id,
                    "text": sentence_text,
                    "position": position,
                }
            )

    average_sentences = (len(sentence_records) / processed_chunks) if processed_chunks else 0.0
    _log_event(
        "sentence_segmentation_completed",
        total_chunks_processed=processed_chunks,
        total_sentences_generated=len(sentence_records),
        avg_sentences_per_chunk=round(average_sentences, 2),
    )
    return sentence_records


def serialize_documents(documents: list[DocumentRecord]) -> list[dict[str, Any]]:
    """Convert document records into dictionaries for debugging or tests."""
    return [asdict(document) for document in documents]
