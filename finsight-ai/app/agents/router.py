"""
app/agents/router.py
---------------------
Query routing node — classifies query complexity before retrieval.

Design Decision:
  WHY route before retrieval?
  - Simple queries (Q1: single-fact) need fewer chunks (k=3), faster response.
  - Complex queries (Q3: analytical) need more chunks (k=8+), deeper reasoning.
  - Routing allows resource-proportional allocation: don't burn GPU/API tokens
    on a simple "what is Apple's revenue?" query as if it were an M&A analysis.

  FALLBACK:
  - If LLM routing call fails, defaults to Tier 2 (middle-ground safe default).
"""

from __future__ import annotations

import json
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import AgentState
from app.core.llm_factory import get_llm
from app.prompts.personas import ROUTER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def router_node(state: AgentState) -> AgentState:
    """Classify query complexity and update routing metadata in state."""
    query = state["query"]
    t0 = time.monotonic()

    try:
        llm = get_llm()
        response = llm.invoke(
            [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=f"Classify this query:\n\n{query}"),
            ]
        )
        routing = json.loads(response.content.strip())
        tier = routing.get("tier", 2)
        reason = routing.get("reasoning", "")
    except Exception as exc:
        logger.warning("Router classification failed (%s). Defaulting to Tier 2.", exc)
        tier = 2
        reason = "Fallback: router unavailable"

    elapsed = (time.monotonic() - t0) * 1000
    logger.info("Query routed to Tier %d in %.1f ms: %s", tier, elapsed, reason)

    return {
        **state,
        "query_tier": tier,
        "query_routing_reason": reason,
        "metadata": {
            **state.get("metadata", {}),
            "router_latency_ms": round(elapsed, 1),
            "query_tier": tier,
        },
    }
