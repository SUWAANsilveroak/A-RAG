"""FastAPI application exposing A-RAG pipeline endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from main import run_a_rag_pipeline, run_retry_pipeline


LOGGER = logging.getLogger(__name__)
app = FastAPI(title="A-RAG API", version="1.0.0")
_REQUIRED_RESOURCE_KEYS = {
    "chunks",
    "model",
    "faiss_index",
    "metadata",
    "read_chunk_ids",
    "model_name",
    "provider",
}


class QueryRequest(BaseModel):
    """Request model for pipeline query endpoints."""

    query: str


def _build_default_resources() -> dict[str, Any]:
    """Build safe default API resources with required key structure."""
    return {
        "chunks": [],
        "model": None,
        "faiss_index": None,
        "metadata": [],
        "read_chunk_ids": set(),
        "model_name": "llama3.1",
        "provider": "ollama",
    }


# Initialize immediately so API works even if startup hooks are bypassed in test clients/tools.
app.state.resources = _build_default_resources()


def _has_required_resources(resources: dict[str, Any]) -> bool:
    """Return whether resources include all required top-level keys."""
    if not isinstance(resources, dict):
        return False
    return all(key in resources for key in _REQUIRED_RESOURCE_KEYS)


@app.on_event("startup")
def initialize_default_resources() -> None:
    """Initialize default resource structure at API startup."""
    if not _has_required_resources(getattr(app.state, "resources", {})):
        app.state.resources = _build_default_resources()
        LOGGER.info("api_resources_initialized_default")


def set_pipeline_resources(resources: dict[str, Any]) -> None:
    """Set pipeline resources used by API endpoints."""
    app.state.resources = resources if isinstance(resources, dict) else {}


def _get_pipeline_resources() -> dict[str, Any]:
    """Get pipeline resources from app state with safe fallback."""
    resources = getattr(app.state, "resources", None)
    if isinstance(resources, dict):
        return resources
    return {}


def _resource_error_response() -> JSONResponse:
    """Return standardized 500 response for missing pipeline resources."""
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_message": "Pipeline resources are not initialized.",
        },
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    LOGGER.info("api_health_called")
    return {"status": "ok"}


@app.post("/query")
def run_query(request: QueryRequest) -> dict[str, Any]:
    """Run single-pass A-RAG pipeline."""
    query = str(request.query).strip()
    LOGGER.info("api_query_called", extra={"query": query})

    if not query:
        LOGGER.warning("api_query_empty_query")
        return {"status": "error", "error_message": "Query must not be empty."}

    resources = _get_pipeline_resources()
    if not _has_required_resources(resources):
        LOGGER.error("api_query_resources_uninitialized")
        return _resource_error_response()

    try:
        result = run_a_rag_pipeline(query=query, resources=resources)
        if isinstance(result, dict) and str(result.get("status", "")).lower() == "error":
            LOGGER.error("api_query_failed", extra={"query": query, "error_message": result.get("error_message", "")})
            return {"status": "error", "error_message": str(result.get("error_message", "Pipeline failed."))}

        LOGGER.info("api_query_succeeded", extra={"query": query})
        return {"status": "success", "result": result}
    except Exception as error:
        LOGGER.error("api_query_exception", extra={"query": query, "error_message": str(error)})
        return {"status": "error", "error_message": str(error)}


@app.post("/query/retry")
def run_query_retry(request: QueryRequest) -> dict[str, Any]:
    """Run retry-enabled A-RAG pipeline."""
    query = str(request.query).strip()
    LOGGER.info("api_retry_query_called", extra={"query": query})

    if not query:
        LOGGER.warning("api_retry_query_empty_query")
        return {"status": "error", "error_message": "Query must not be empty."}

    resources = _get_pipeline_resources()
    if not _has_required_resources(resources):
        LOGGER.error("api_retry_query_resources_uninitialized")
        return _resource_error_response()

    try:
        result = run_retry_pipeline(query=query, resources=resources)
        if isinstance(result, dict) and str(result.get("status", "")).lower() == "error":
            LOGGER.error(
                "api_retry_query_failed",
                extra={"query": query, "error_message": result.get("error_message", "")},
            )
            return {"status": "error", "error_message": str(result.get("error_message", "Retry pipeline failed."))}

        LOGGER.info("api_retry_query_succeeded", extra={"query": query})
        return {"status": "success", "result": result}
    except Exception as error:
        LOGGER.error("api_retry_query_exception", extra={"query": query, "error_message": str(error)})
        return {"status": "error", "error_message": str(error)}
