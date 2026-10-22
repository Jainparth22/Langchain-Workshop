"""
app/logging_/tracker.py
------------------------
Structured request logging and optional LangSmith observability integration.

Design Decision:
  WHY structured logging (JSON)?
  - Plain text logs are unqueryable at scale. JSON logs are trivially indexed
    by ELK Stack, Datadog, Google Cloud Logging, etc.
  - Each request gets a correlation_id for end-to-end trace reconstruction.

  WHY optional LangSmith?
  - LangSmith traces every LLM call with token counts, latency, and prompt/response.
  - Invaluable for debugging hallucinations and optimising prompts.
  - Optional (LANGCHAIN_TRACING_V2=true) to avoid mandatory external dependency.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

from app.core.config import get_settings

logger = logging.getLogger("finsight.tracker")


def setup_logging() -> None:
    """Configure application-wide structured JSON logging."""
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure root logger with JSON-style format
    logging.basicConfig(
        level=log_level,
        format='{"time": "%(asctime)s", "level": "%(levelname)s", '
               '"logger": "%(name)s", "message": "%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    # Enable LangSmith tracing if configured
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        logger.info("LangSmith tracing enabled for project: %s", settings.langchain_project)

    logger.info(
        "FinSight AI logging initialised (level=%s, langsmith=%s)",
        settings.log_level,
        settings.langchain_tracing_v2,
    )


class RequestTracker:
    """Tracks per-request metadata for observability and audit trails."""

    def __init__(self, session_id: str, query: str):
        self.correlation_id = str(uuid.uuid4())[:8]
        self.session_id = session_id
        self.query = query
        self.start_time = time.monotonic()
        self._metadata: Dict[str, Any] = {}

    def record(self, **kwargs) -> None:
        """Record arbitrary key-value metadata."""
        self._metadata.update(kwargs)

    def finish(self, final_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalise and log the request summary.

        Returns the complete request metadata dictionary.
        """
        elapsed_ms = round((time.monotonic() - self.start_time) * 1000, 1)
        metadata = final_state.get("metadata", {})

        summary = {
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "query_preview": self.query[:80],
            "total_latency_ms": elapsed_ms,
            "query_tier": metadata.get("query_tier"),
            "docs_retrieved": metadata.get("docs_retrieved", 0),
            "critic_verdict": metadata.get("critic_verdict"),
            "faithfulness_score": metadata.get("faithfulness_score"),
            "output_safe": final_state.get("output_safe"),
            "revision_count": final_state.get("revision_count", 0),
            "provider": metadata.get("provider", "unknown"),
            "latency_breakdown_ms": {
                "router": metadata.get("router_latency_ms"),
                "researcher": metadata.get("researcher_latency_ms"),
                "responder": metadata.get("responder_latency_ms"),
                "critic": metadata.get("critic_latency_ms"),
            },
        }

        logger.info("REQUEST_COMPLETE %s", json.dumps(summary))
        return summary
