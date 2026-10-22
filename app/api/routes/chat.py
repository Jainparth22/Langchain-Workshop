"""
app/api/routes/chat.py
-----------------------
POST /chat — Main multi-agent financial Q&A endpoint.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.graph import get_graph
from app.core.llm_factory import get_active_provider
from app.logging_.tracker import RequestTracker
from app.memory.session import SessionManager

router = APIRouter()
logger = logging.getLogger(__name__)

_session_manager = SessionManager()


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The financial question to ask FinSight AI",
        examples=["What was Apple's total revenue in Q3 2023?"],
    )
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session ID for conversation continuity. "
                    "Omit to start a new session.",
    )


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    query_tier: int
    critic_verdict: str
    faithfulness_score: float | None
    docs_retrieved: int
    revision_count: int
    sources: list[dict]
    total_latency_ms: float
    llm_provider: str
    output_safe: bool
    correlation_id: str


@router.post(
    "",
    response_model=ChatResponse,
    summary="Ask a financial question",
    description=(
        "Runs the full FinSight multi-agent pipeline: "
        "InputGuard → Router → Researcher (hybrid retrieval) → "
        "Responder → Critic → OutputGuard. "
        "Returns a grounded, source-cited financial answer."
    ),
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Execute the multi-agent RAG pipeline for a financial query."""
    tracker = RequestTracker(request.session_id, request.query)

    # Load conversation history
    history = _session_manager.get_history(request.session_id)

    initial_state = {
        "query": request.query,
        "session_id": request.session_id,
        "conversation_history": history,
        "revision_count": 0,
        "metadata": {},
    }

    try:
        graph = get_graph()
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        logger.error("Graph execution failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution error: {str(exc)}",
        )

    # Persist turn to conversation history
    answer = final_state.get("final_response", "No answer generated.")
    metadata = final_state.get("metadata", {})

    _session_manager.save_turn(
        session_id=request.session_id,
        query=request.query,
        response=answer,
        metadata=metadata,
    )

    summary = tracker.finish(final_state)

    return ChatResponse(
        answer=answer,
        session_id=request.session_id,
        query_tier=final_state.get("query_tier", 0),
        critic_verdict=final_state.get("critic_verdict", "UNKNOWN"),
        faithfulness_score=metadata.get("faithfulness_score"),
        docs_retrieved=metadata.get("docs_retrieved", 0),
        revision_count=final_state.get("revision_count", 0),
        sources=metadata.get("sources", []),
        total_latency_ms=summary["total_latency_ms"],
        llm_provider=get_active_provider(),
        output_safe=final_state.get("output_safe", False),
        correlation_id=tracker.correlation_id,
    )
