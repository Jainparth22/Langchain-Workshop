"""
app/agents/graph.py
--------------------
LangGraph StateGraph — the heart of FinSight AI's multi-agent orchestration.

Design Decision:
  WHY LangGraph over vanilla LangChain chains?
  - LangChain chains are linear: A → B → C. No branching, no loops, no state.
  - LangGraph adds:
      * Stateful execution: shared typed State dict flows through all nodes.
      * Conditional edges: route based on runtime outputs (e.g., Critic verdict).
      * Cycles: Critic can send control back to Researcher for re-research.
      * Interruption: can pause for human approval in high-stakes queries.
      * Built-in checkpointing: resume interrupted workflows.

  STATE DESIGN:
  - Typed TypedDict ensures all agents read/write consistent fields.
  - Immutable source fields (query, session_id) prevent accidental mutation.
  - Mutable result fields are appended/overwritten by each agent.

  GRAPH TOPOLOGY:
  
  ┌─────────┐    ┌────────────┐    ┌────────────┐
  │  INPUT  │───▶│   ROUTER   │───▶│ RESEARCHER │
  │  GUARD  │    │            │    │            │
  └─────────┘    └────────────┘    └─────┬──────┘
                                         │
                                    ┌────▼──────┐
                                    │ RESPONDER │
                                    └─────┬─────┘
                                          │
                                    ┌─────▼─────┐
                    ┌──────────────▶│  CRITIC   │
                    │               └─────┬─────┘
                    │                     │
                    │          PASS / FAIL / NEEDS_REVISION
                    │                     │
                    │  NEEDS_REVISION ◀───┤
                    │                     │
                    │               ┌─────▼──────┐
                    └───────────────│   OUTPUT   │
                    (max 2 loops)   │   GUARD    │
                                    └─────┬──────┘
                                          │
                                    Final Response
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

from app.agents.researcher import researcher_node
from app.agents.responder import responder_node
from app.agents.critic import critic_node
from app.agents.router import router_node
from app.guardrails.input_guard import input_guard_node
from app.guardrails.output_guard import output_guard_node
from app.agents.state import AgentState

logger = logging.getLogger(__name__)

MAX_REVISION_LOOPS = 2  # Prevent infinite cycles


def _should_revise(state: AgentState) -> str:
    """
    Conditional edge: decide next node based on Critic verdict.

    Returns:
        "revise"    → back to researcher (re-research with critic instructions)
        "finish"    → proceed to output guard
    """
    verdict = state.get("critic_verdict", "PASS")
    revision_count = state.get("revision_count", 0)

    if verdict == "NEEDS_REVISION" and revision_count < MAX_REVISION_LOOPS:
        logger.info(
            "Critic verdict: NEEDS_REVISION (loop %d/%d). Sending back to researcher.",
            revision_count + 1, MAX_REVISION_LOOPS,
        )
        return "revise"

    if verdict == "FAIL":
        logger.warning(
            "Critic verdict: FAIL. Returning safe fallback response."
        )
        return "finish"

    return "finish"


def build_graph() -> Any:
    """
    Compile and return the FinSight AI LangGraph StateGraph.

    Returns:
        A compiled LangGraph Runnable (supports .invoke and .astream).
    """
    workflow = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────
    workflow.add_node("input_guard", input_guard_node)
    workflow.add_node("router", router_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("responder", responder_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("output_guard", output_guard_node)

    # ── Define edges (execution flow) ─────────────────────────────────────
    workflow.set_entry_point("input_guard")
    workflow.add_edge("input_guard", "router")
    workflow.add_edge("router", "researcher")
    workflow.add_edge("researcher", "responder")
    workflow.add_edge("responder", "critic")

    # Conditional edge from critic: revise loop or finish
    workflow.add_conditional_edges(
        "critic",
        _should_revise,
        {
            "revise": "researcher",   # Loop back for re-research
            "finish": "output_guard",  # Proceed to output validation
        },
    )

    workflow.add_edge("output_guard", END)

    # Compile with checkpointing for session continuity
    graph = workflow.compile()
    logger.info("LangGraph StateGraph compiled successfully.")
    return graph


# Singleton compiled graph (instantiate once per process)
_graph = None


def get_graph() -> Any:
    """Return the singleton compiled graph."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
