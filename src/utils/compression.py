"""Compression utilities for selecting high-value retrieval snippets."""

from __future__ import annotations

import logging
import re
from typing import Any


LOGGER = logging.getLogger(__name__)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")


def select_top_snippets(
    retrieval_results: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Select top snippets by score while removing empty snippets and chunk duplicates."""
    try:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        original_count = len(retrieval_results)
        if original_count == 0:
            LOGGER.info(
                "snippet_selection_completed",
                extra={
                    "original_result_count": 0,
                    "filtered_result_count": 0,
                    "duplicate_removals": 0,
                    "selected_chunk_ids": [],
                },
            )
            return []

        best_by_chunk: dict[str, dict[str, Any]] = {}
        duplicate_removals = 0
        empty_snippet_removals = 0

        for result in retrieval_results:
            if not isinstance(result, dict):
                continue

            chunk_id = str(result.get("chunk_id", "")).strip()
            snippet = str(result.get("snippet", "")).strip()
            if not chunk_id or not snippet:
                if chunk_id or ("snippet" in result):
                    empty_snippet_removals += 1
                continue

            score_value = result.get("score", 0.0)
            score = float(score_value) if isinstance(score_value, (int, float)) else 0.0

            normalized_result = {
                "chunk_id": chunk_id,
                "score": score,
                "snippet": snippet,
            }

            existing = best_by_chunk.get(chunk_id)
            if existing is None:
                best_by_chunk[chunk_id] = normalized_result
                continue

            duplicate_removals += 1
            if score > float(existing["score"]):
                best_by_chunk[chunk_id] = normalized_result

        ranked = sorted(
            best_by_chunk.values(),
            key=lambda item: (-float(item["score"]), str(item["chunk_id"])),
        )
        selected = ranked[:top_k]

        LOGGER.info(
            "snippet_selection_completed",
            extra={
                "original_result_count": original_count,
                "filtered_result_count": len(selected),
                "duplicate_removals": duplicate_removals,
                "empty_snippet_removals": empty_snippet_removals,
                "selected_chunk_ids": [str(item["chunk_id"]) for item in selected],
            },
        )
        return selected
    except Exception:
        LOGGER.exception(
            "snippet_selection_failed",
            extra={"top_k": top_k, "input_result_count": len(retrieval_results)},
        )
        return []


def _normalize_text(text: str) -> str:
    """Normalize whitespace in free text."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like segments while preserving order."""
    if not text.strip():
        return []
    raw_parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [_normalize_text(part) for part in raw_parts if _normalize_text(part)]


def _select_relevant_sentences(text: str, max_sentences: int) -> str:
    """Keep first N unique non-empty sentences from snippet text."""
    sentences = _split_sentences(text)
    if not sentences:
        return ""

    seen_sentences: set[str] = set()
    selected_sentences: list[str] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if lowered in seen_sentences:
            continue
        seen_sentences.add(lowered)
        selected_sentences.append(sentence)
        if len(selected_sentences) >= max_sentences:
            break

    return " ".join(selected_sentences).strip()


def compress_snippets(
    snippets: list[dict[str, Any]],
    max_sentences_per_snippet: int = 2,
    max_total_characters: int = 3000,
) -> list[dict[str, Any]]:
    """Compress snippets by sentence selection and global character budget."""
    try:
        if max_sentences_per_snippet <= 0:
            raise ValueError("max_sentences_per_snippet must be greater than zero.")
        if max_total_characters <= 0:
            raise ValueError("max_total_characters must be greater than zero.")

        if not snippets:
            LOGGER.info(
                "snippet_compression_completed",
                extra={
                    "original_character_count": 0,
                    "compressed_character_count": 0,
                    "skipped_snippets": 0,
                    "truncated_snippets": 0,
                },
            )
            return []

        ranked_snippets = sorted(
            [snippet for snippet in snippets if isinstance(snippet, dict)],
            key=lambda item: (-float(item.get("score", 0.0)) if isinstance(item.get("score"), (int, float)) else 0.0, str(item.get("chunk_id", ""))),
        )

        original_character_count = sum(len(str(snippet.get("snippet", ""))) for snippet in ranked_snippets)
        compressed_results: list[dict[str, Any]] = []
        remaining_characters = int(max_total_characters)
        skipped_snippets = 0
        truncated_snippets = 0

        for snippet_record in ranked_snippets:
            if remaining_characters <= 0:
                break

            chunk_id = str(snippet_record.get("chunk_id", "")).strip()
            if not chunk_id:
                skipped_snippets += 1
                continue

            score_value = snippet_record.get("score", 0.0)
            score = float(score_value) if isinstance(score_value, (int, float)) else 0.0
            raw_snippet_text = str(snippet_record.get("snippet", ""))
            compressed_text = _select_relevant_sentences(raw_snippet_text, max_sentences_per_snippet)

            if not compressed_text:
                skipped_snippets += 1
                continue

            if len(compressed_text) > remaining_characters:
                truncated_text = _normalize_text(compressed_text[:remaining_characters])
                if not truncated_text:
                    skipped_snippets += 1
                    break
                compressed_text = truncated_text
                truncated_snippets += 1

            compressed_results.append(
                {
                    "chunk_id": chunk_id,
                    "score": score,
                    "compressed_text": compressed_text,
                }
            )
            remaining_characters -= len(compressed_text)

            if remaining_characters <= 0:
                break

        compressed_character_count = sum(len(item["compressed_text"]) for item in compressed_results)
        LOGGER.info(
            "snippet_compression_completed",
            extra={
                "original_character_count": original_character_count,
                "compressed_character_count": compressed_character_count,
                "skipped_snippets": skipped_snippets,
                "truncated_snippets": truncated_snippets,
            },
        )
        return compressed_results
    except Exception:
        LOGGER.exception(
            "snippet_compression_failed",
            extra={
                "input_snippet_count": len(snippets),
                "max_sentences_per_snippet": max_sentences_per_snippet,
                "max_total_characters": max_total_characters,
            },
        )
        return []
