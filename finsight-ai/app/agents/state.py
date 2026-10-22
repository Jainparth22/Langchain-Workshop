"""
app/agents/state.py
--------------------
Shared state schema for the LangGraph multi-agent workflow.

Design Decision:
  WHY TypedDict?
  - LangGraph requires TypedDict for state definition (not dataclass/Pydantic).
  - TypedDict is a lightweight type-hint-only construct with zero runtime overhead.
  - All agent nodes read/write from this single shared state, ensuring type safety.

  FIELD LIFECYCLE:
  - query, session_id: Set at entry, never mutated.
  - retrieved_docs: Set by Researcher, read by Responder and Critic.
  - researcher_extracts: Researcher's structured analysis of retrieved docs.
  - draft_response: Set by Responder, evaluated by Critic.
  - critic_verdict, critic_feedback: Set by Critic, drives conditional edge.
  - final_response: Set by OutputGuard after safety validation.
  - revision_count: Incremented by Researcher on each revision loop.
  - metadata: Accumulated audit trail (latency, sources, model used).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Shared state dictionary passed between all LangGraph agent nodes."""

    # ── Immutable input fields ────────────────────────────────────────────
    query: str                          # Original user question
    session_id: str                     # Session ID for conversation history
    conversation_history: List[Dict]    # Prior turns in this session

    # ── Query routing ─────────────────────────────────────────────────────
    query_tier: int                     # 1=simple, 2=complex, 3=analytical
    query_routing_reason: str           # Routing decision explanation

    # ── Guardrails ────────────────────────────────────────────────────────
    input_safe: bool                    # Result of InputGuard check
    input_guard_reason: Optional[str]   # Reason if flagged

    # ── Researcher output ─────────────────────────────────────────────────
    retrieved_docs: List[Document]       # Chunks retrieved from vector store
    researcher_extracts: Dict[str, Any]  # Structured extraction from Researcher

    # ── Responder output ──────────────────────────────────────────────────
    draft_response: str                  # LLM-generated answer

    # ── Critic output ─────────────────────────────────────────────────────
    critic_verdict: str                  # "PASS" | "NEEDS_REVISION" | "FAIL"
    critic_feedback: Dict[str, Any]      # Full evaluation JSON
    revision_count: int                  # Number of revision loops completed

    # ── Final output ──────────────────────────────────────────────────────
    final_response: str                  # Post-output-guard validated response
    output_safe: bool                    # Output guard result

    # ── Observability ─────────────────────────────────────────────────────
    metadata: Dict[str, Any]            # Latency, sources, provider, scores
