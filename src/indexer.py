"""Document loading utilities for the indexing pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Sequence
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
_EMBEDDING_MODELS: dict[str, Any] = {}
_FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_ID_MAPPING_FILENAME = "id_mapping.json"


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


def _load_embedding_model(model_name: str) -> Any:
    """Load and cache a sentence-transformer model by name."""
    cached_model = _EMBEDDING_MODELS.get(model_name)
    if cached_model is not None:
        return cached_model

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "Embedding generation requires the optional 'sentence-transformers' package."
        ) from error

    try:
        model = SentenceTransformer(model_name)
    except Exception as primary_error:
        if model_name == _FALLBACK_EMBEDDING_MODEL:
            raise RuntimeError(f"Failed to load embedding model: {model_name}") from primary_error

        LOGGER.warning(
            "embedding_model_load_failed",
            extra={"requested_model": model_name, "fallback_model": _FALLBACK_EMBEDDING_MODEL},
        )
        try:
            model = SentenceTransformer(_FALLBACK_EMBEDDING_MODEL)
            model_name = _FALLBACK_EMBEDDING_MODEL
        except Exception as fallback_error:
            raise RuntimeError(
                f"Failed to load embedding models: {model_name}, {_FALLBACK_EMBEDDING_MODEL}"
            ) from fallback_error

    _EMBEDDING_MODELS[model_name] = model
    return model


def _batched(items: Sequence[dict[str, Any]], batch_size: int) -> list[Sequence[dict[str, Any]]]:
    """Split sentence records into fixed-size batches."""
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def generate_sentence_embeddings(
    sentences: list[dict[str, Any]],
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> list[dict[str, Any]]:
    """Generate normalized embeddings for valid sentence records."""
    if not sentences:
        _log_event(
            "sentence_embedding_completed",
            total_sentences_embedded=0,
            batch_count=0,
            embedding_dimension=0,
            skipped_sentences=0,
            model_name=model_name,
        )
        return []

    valid_sentences: list[dict[str, Any]] = []
    skipped_sentences = 0

    for sentence in sentences:
        sentence_id = str(sentence.get("sentence_id", "")).strip()
        sentence_text = str(sentence.get("text", "")).strip()
        if not sentence_id or not sentence_text:
            skipped_sentences += 1
            continue
        valid_sentences.append(sentence)

    if not valid_sentences:
        _log_event(
            "sentence_embedding_completed",
            total_sentences_embedded=0,
            batch_count=0,
            embedding_dimension=0,
            skipped_sentences=skipped_sentences,
            model_name=model_name,
        )
        return []

    model = _load_embedding_model(model_name)
    batch_size = min(32, len(valid_sentences))
    batches = _batched(valid_sentences, batch_size)
    embedded_sentences: list[dict[str, Any]] = []
    embedding_dimension = 0

    for batch in batches:
        batch_texts = [str(sentence["text"]).strip() for sentence in batch]
        raw_embeddings = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            normalize_embeddings=True,
            convert_to_numpy=False,
            show_progress_bar=False,
        )

        for sentence, embedding in zip(batch, raw_embeddings, strict=True):
            vector = [float(value) for value in embedding]
            if embedding_dimension == 0:
                embedding_dimension = len(vector)
            elif len(vector) != embedding_dimension:
                raise ValueError("Inconsistent embedding dimensions detected across sentences.")

            embedded_sentences.append(
                {
                    "sentence_id": str(sentence["sentence_id"]),
                    "chunk_id": str(sentence["chunk_id"]),
                    "text": str(sentence["text"]),
                    "position": int(sentence["position"]),
                    "embedding": vector,
                }
            )

    _log_event(
        "sentence_embedding_completed",
        total_sentences_embedded=len(embedded_sentences),
        batch_count=len(batches),
        embedding_dimension=embedding_dimension,
        skipped_sentences=skipped_sentences,
        model_name=model_name,
    )
    return embedded_sentences


def _get_faiss_module() -> Any:
    """Import FAISS lazily so the module stays usable without retrieval dependencies."""
    try:
        import faiss  # type: ignore
    except ImportError as error:
        raise RuntimeError("Index storage requires the optional 'faiss-cpu' package.") from error
    return faiss


def _get_numpy_module() -> Any:
    """Import NumPy lazily for float32 embedding storage."""
    try:
        import numpy as np  # type: ignore
    except ImportError as error:
        raise RuntimeError("Index storage requires the optional 'numpy' package.") from error
    return np


def _ensure_parent_directory(path: str | Path) -> Path:
    """Create the parent directory for an output path if needed."""
    target_path = Path(path)
    if not target_path.parent.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
    return target_path


def _prepare_embedding_matrix(embeddings: list[list[float]]) -> Any:
    """Validate embeddings and convert them into a float32 matrix."""
    if not embeddings:
        raise ValueError("Embeddings list cannot be empty.")

    dimension = len(embeddings[0])
    if dimension == 0:
        raise ValueError("Embedding vectors must have a positive dimension.")

    for embedding in embeddings:
        if len(embedding) != dimension:
            raise ValueError("All embeddings must have the same dimension.")

    np = _get_numpy_module()
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Embeddings must be a 2D list of float values.")
    return matrix


def save_faiss_index(embeddings: list[list[float]], index_path: str) -> None:
    """Build and save a FAISS index from sentence embeddings."""
    matrix = _prepare_embedding_matrix(embeddings)
    faiss = _get_faiss_module()
    output_path = _ensure_parent_directory(index_path)

    index = faiss.IndexFlatIP(int(matrix.shape[1]))
    index.add(matrix)
    faiss.write_index(index, str(output_path))

    _log_event(
        "faiss_index_saved",
        total_embeddings_stored=int(matrix.shape[0]),
        embedding_dimension=int(matrix.shape[1]),
        index_size=int(index.ntotal),
        index_path=str(output_path.resolve()),
    )


def _build_metadata_record(sentence_record: dict[str, Any]) -> dict[str, Any]:
    """Strip embedding payloads and keep only storage metadata fields."""
    sentence_id = str(sentence_record.get("sentence_id", "")).strip()
    chunk_id = str(sentence_record.get("chunk_id", "")).strip()
    text = str(sentence_record.get("text", "")).strip()

    if not sentence_id:
        raise ValueError("Sentence metadata is missing 'sentence_id'.")
    if not chunk_id:
        raise ValueError("Sentence metadata is missing 'chunk_id'.")
    if not text:
        raise ValueError("Sentence metadata is missing 'text'.")

    return {
        "sentence_id": sentence_id,
        "chunk_id": chunk_id,
        "text": text,
        "position": int(sentence_record["position"]),
    }


def _build_id_mapping(metadata_records: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Create vector-position mapping for FAISS result reconstruction."""
    seen_sentence_ids: set[str] = set()
    id_mapping: dict[str, dict[str, str]] = {}

    for position, record in enumerate(metadata_records):
        sentence_id = str(record["sentence_id"])
        if sentence_id in seen_sentence_ids:
            raise ValueError(f"Duplicate sentence_id detected: {sentence_id}")
        seen_sentence_ids.add(sentence_id)
        id_mapping[str(position)] = {
            "sentence_id": sentence_id,
            "chunk_id": str(record["chunk_id"]),
        }

    return id_mapping


def save_metadata(sentence_records: list[dict[str, Any]], metadata_path: str) -> None:
    """Save sentence metadata and vector-position mapping as JSON."""
    output_path = _ensure_parent_directory(metadata_path)
    metadata_records = [_build_metadata_record(record) for record in sentence_records]
    id_mapping = _build_id_mapping(metadata_records)
    id_mapping_path = output_path.with_name(_DEFAULT_ID_MAPPING_FILENAME)

    with output_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata_records, metadata_file, ensure_ascii=False, indent=2)

    with id_mapping_path.open("w", encoding="utf-8") as mapping_file:
        json.dump(id_mapping, mapping_file, ensure_ascii=False, indent=2)

    _log_event(
        "metadata_saved",
        metadata_count=len(metadata_records),
        id_mapping_count=len(id_mapping),
        metadata_path=str(output_path.resolve()),
        id_mapping_path=str(id_mapping_path.resolve()),
    )


def load_faiss_index(index_path: str) -> Any:
    """Load a previously saved FAISS index from disk."""
    faiss = _get_faiss_module()
    resolved_path = Path(index_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {resolved_path}")
    return faiss.read_index(str(resolved_path))


def load_metadata(metadata_path: str) -> dict[str, Any]:
    """Load metadata and ID mapping used to reconstruct FAISS results."""
    resolved_metadata_path = Path(metadata_path)
    if not resolved_metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {resolved_metadata_path}")

    id_mapping_path = resolved_metadata_path.with_name(_DEFAULT_ID_MAPPING_FILENAME)
    if not id_mapping_path.exists():
        raise FileNotFoundError(f"ID mapping file not found: {id_mapping_path}")

    with resolved_metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata_records = json.load(metadata_file)

    with id_mapping_path.open("r", encoding="utf-8") as mapping_file:
        id_mapping = json.load(mapping_file)

    if len(metadata_records) != len(id_mapping):
        raise ValueError("Metadata count does not match ID mapping count.")

    for position, record in enumerate(metadata_records):
        mapping_record = id_mapping.get(str(position))
        if mapping_record is None:
            raise ValueError(f"Missing ID mapping for vector position {position}.")
        if mapping_record["sentence_id"] != record["sentence_id"]:
            raise ValueError(f"Sentence ID mismatch at vector position {position}.")
        if mapping_record["chunk_id"] != record["chunk_id"]:
            raise ValueError(f"Chunk ID mismatch at vector position {position}.")

    return {
        "metadata": metadata_records,
        "id_mapping": id_mapping,
    }


def serialize_documents(documents: list[DocumentRecord]) -> list[dict[str, Any]]:
    """Convert document records into dictionaries for debugging or tests."""
    return [asdict(document) for document in documents]
