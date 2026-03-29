"""Tool wrappers for retrieval search functions."""

from __future__ import annotations

import logging
from typing import Any

from src.retrieval.keyword import keyword_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.semantic import semantic_search


LOGGER = logging.getLogger(__name__)
_KEYWORD_TOOL_NAME = "keyword_search"
_SEMANTIC_TOOL_NAME = "semantic_search"
_HYBRID_TOOL_NAME = "hybrid_search"


def _build_keyword_success_response(
    query: str,
    top_k: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the standardized success payload for keyword search."""
    return {
        "tool_name": _KEYWORD_TOOL_NAME,
        "status": "success",
        "query": query,
        "top_k": top_k,
        "results": results,
    }


def _build_keyword_error_response(error_message: str) -> dict[str, Any]:
    """Return the standardized error payload for keyword search."""
    return {
        "tool_name": _KEYWORD_TOOL_NAME,
        "status": "error",
        "error_message": error_message,
        "results": [],
    }


def run_keyword_search_tool(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Execute keyword search and wrap the result in the tool response format."""
    LOGGER.info(
        "keyword_search_tool_started",
        extra={
            "tool_name": _KEYWORD_TOOL_NAME,
            "query": query,
            "top_k": top_k,
        },
    )

    try:
        results = keyword_search(query=query, chunks=chunks, top_k=top_k)
    except Exception as error:
        error_message = str(error)
        LOGGER.exception(
            "keyword_search_tool_failed",
            extra={
                "tool_name": _KEYWORD_TOOL_NAME,
                "query": query,
                "top_k": top_k,
                "error_message": error_message,
            },
        )
        return _build_keyword_error_response(error_message)

    LOGGER.info(
        "keyword_search_tool_completed",
        extra={
            "tool_name": _KEYWORD_TOOL_NAME,
            "query": query,
            "top_k": top_k,
            "result_count": len(results),
        },
    )
    return _build_keyword_success_response(query=query, top_k=top_k, results=results)


def _build_semantic_success_response(
    query: str,
    top_k: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the standardized success payload for semantic search."""
    return {
        "tool_name": _SEMANTIC_TOOL_NAME,
        "status": "success",
        "query": query,
        "top_k": top_k,
        "results": results,
    }


def _build_semantic_error_response(error_message: str) -> dict[str, Any]:
    """Return the standardized error payload for semantic search."""
    return {
        "tool_name": _SEMANTIC_TOOL_NAME,
        "status": "error",
        "error_message": error_message,
        "results": [],
    }


def run_semantic_search_tool(
    query: str,
    model: Any,
    faiss_index: Any,
    metadata: list[dict[str, Any]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Execute semantic search and wrap the result in the tool response format."""
    LOGGER.info(
        "semantic_search_tool_started",
        extra={
            "tool_name": _SEMANTIC_TOOL_NAME,
            "query": query,
            "top_k": top_k,
        },
    )

    try:
        results = semantic_search(
            query=query,
            model=model,
            faiss_index=faiss_index,
            metadata=metadata,
            top_k=top_k,
        )
    except Exception as error:
        error_message = str(error)
        LOGGER.exception(
            "semantic_search_tool_failed",
            extra={
                "tool_name": _SEMANTIC_TOOL_NAME,
                "query": query,
                "top_k": top_k,
                "error_message": error_message,
            },
        )
        return _build_semantic_error_response(error_message)

    LOGGER.info(
        "semantic_search_tool_completed",
        extra={
            "tool_name": _SEMANTIC_TOOL_NAME,
            "query": query,
            "top_k": top_k,
            "result_count": len(results),
        },
    )
    return _build_semantic_success_response(query=query, top_k=top_k, results=results)


def _build_hybrid_success_response(
    query: str,
    top_k: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the standardized success payload for hybrid search."""
    return {
        "tool_name": _HYBRID_TOOL_NAME,
        "status": "success",
        "query": query,
        "top_k": top_k,
        "results": results,
    }


def _build_hybrid_error_response(error_message: str) -> dict[str, Any]:
    """Return the standardized error payload for hybrid search."""
    return {
        "tool_name": _HYBRID_TOOL_NAME,
        "status": "error",
        "error_message": error_message,
        "results": [],
    }


def run_hybrid_search_tool(
    query: str,
    chunks: list[dict[str, Any]],
    model: Any,
    faiss_index: Any,
    metadata: list[dict[str, Any]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Execute hybrid search and wrap the result in the tool response format."""
    LOGGER.info(
        "hybrid_search_tool_started",
        extra={
            "tool_name": _HYBRID_TOOL_NAME,
            "query": query,
            "top_k": top_k,
        },
    )

    try:
        results = hybrid_search(
            query=query,
            chunks=chunks,
            model=model,
            faiss_index=faiss_index,
            metadata=metadata,
            top_k=top_k,
        )
    except Exception as error:
        error_message = str(error)
        LOGGER.exception(
            "hybrid_search_tool_failed",
            extra={
                "tool_name": _HYBRID_TOOL_NAME,
                "query": query,
                "top_k": top_k,
                "error_message": error_message,
            },
        )
        return _build_hybrid_error_response(error_message)

    LOGGER.info(
        "hybrid_search_tool_completed",
        extra={
            "tool_name": _HYBRID_TOOL_NAME,
            "query": query,
            "top_k": top_k,
            "result_count": len(results),
        },
    )
    return _build_hybrid_success_response(query=query, top_k=top_k, results=results)
