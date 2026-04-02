"""Chunking utilities for A-RAG.

Task 2.2: Split text into ~`max_tokens` chunks while preserving sentence boundaries.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Dict, List

LOGGER = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def _clean_text_for_chunking(text: str) -> str:
    """Normalize whitespace before sentence segmentation.

    Note: chunking preserves semantic order, but this function also normalizes
    whitespace to keep output deterministic and testable.
    """

    # Collapse all whitespace runs into single spaces.
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned


def _estimate_tokens(text: str) -> int:
    """Approximate token count using the rule: 1 token ~= 0.75 words.

    Returns an integer approximation intended for chunk-size control.
    """

    # "Word count" fallback: count word-ish tokens.
    word_count = len(_WORD_RE.findall(text))
    # tokens ~= words / 0.75, round up to avoid exceeding max_tokens.
    return int(math.ceil(word_count / 0.75)) if word_count else 0


_NLTK_READY = False


def _segment_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK."""
    global _NLTK_READY
    
    if not _NLTK_READY:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        _NLTK_READY = True

    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


def create_chunks(text: str, max_tokens: int = 1000) -> List[Dict]:
    """Create sentence-preserving chunks.

    Output format (strict):
    [
      {
        "chunk_id": str,
        "text": str,
        "position": int
      },
      ...
    ]
    """

    if text is None:
        return []

    cleaned = _clean_text_for_chunking(text)
    if not cleaned:
        return []

    sentences = _segment_sentences(cleaned)

    chunks: List[Dict] = []
    chunk_index = 0

    current_sentences: List[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal chunk_index, current_sentences, current_tokens
        if not current_sentences:
            return

        chunk_text = " ".join(current_sentences).strip()
        chunk_payload = {
            "chunk_id": f"chunk_{chunk_index}",
            "text": chunk_text,
            "position": chunk_index,
        }
        chunks.append(chunk_payload)

        chunk_index += 1
        current_sentences = []
        current_tokens = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_tokens = _estimate_tokens(sentence)

        if current_sentences:
            # If we can still fit this whole sentence, append it.
            if current_tokens + sentence_tokens <= max_tokens:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
            else:
                # Otherwise, flush current chunk and start a new one.
                flush()
                current_sentences.append(sentence)
                current_tokens = sentence_tokens
        else:
            # Start first chunk (or start after a flush).
            current_sentences.append(sentence)
            current_tokens = sentence_tokens

        # Edge case: very long sentence overflow is allowed,
        # but it must not be split across chunks.
        if sentence_tokens > max_tokens:
            flush()

    flush()

    # Logging: number of chunks, avg chunk size, max chunk size.
    if chunks:
        sizes = [_estimate_tokens(chunk["text"]) for chunk in chunks if chunk.get("text")]
        num_chunks = len(chunks)
        avg_size = sum(sizes) / len(sizes) if sizes else 0.0
        max_size = max(sizes) if sizes else 0
        LOGGER.info(
            "chunking_created",
            extra={"num_chunks": num_chunks, "avg_chunk_tokens": avg_size, "max_chunk_tokens": max_size},
        )

    return chunks

