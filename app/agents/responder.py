"""
app/agents/responder.py
------------------------
Responder agent — synthesises retrieved information into the final answer.
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import AgentState
from app.core.llm_factory import get_llm
from app.memory.session import SessionManager
from app.prompts.personas import RESPONDER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_session_manager = SessionManager()


def responder_node(state: AgentState) -> AgentState:
    """
    Synthesise a grounded financial answer from researcher extracts.

    Incorporates:
    - Researcher's structured extracts (key entities, relevant quotes)
    - Retrieved document chunks with source attribution
    - Conversation history for contextual continuity
    - Critic revision instructions (when in a revision loop)
    """
    query = state["query"]
    session_id = state.get("session_id", "default")
    extracts = state.get("researcher_extracts", {})
    retrieved_docs = state.get("retrieved_docs", [])
    critic_feedback = state.get("critic_feedback", {})
    revision_count = state.get("revision_count", 0)

    t0 = time.monotonic()

    # ── Build context from retrieved docs ───────────────────────────────
    context_parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        context_parts.append(
            f"[Document {i} | Source: {source} | Page: {page}]\n{doc.page_content}"
        )
    context = "\n\n".join(context_parts) if context_parts else "No context available."

    # ── Build researcher insights section ───────────────────────────────
    key_entities = extracts.get("key_entities", [])
    relevant_quotes = extracts.get("relevant_extracts", [])
    coverage = extracts.get("coverage_assessment", "unknown")
    missing = extracts.get("missing_aspects", [])

    researcher_summary = (
        f"Coverage Assessment: {coverage}\n"
        f"Key Entities Identified: {', '.join(key_entities) if key_entities else 'None'}\n"
        f"Missing Aspects: {'; '.join(missing) if missing else 'None'}"
    )

    # ── Add critic revision instructions if looping ──────────────────────
    revision_section = ""
    if revision_count > 0 and critic_feedback:
        instructions = critic_feedback.get("revision_instructions", "")
        revision_section = (
            f"\n\nPREVIOUS ANSWER WAS FLAGGED. Revision #{revision_count}.\n"
            f"Critic instructions: {instructions}\n"
            f"Please address these gaps in your revised answer."
        )

    # ── Load conversation history ────────────────────────────────────────
    history = _session_manager.get_history(session_id)
    history_text = ""
    if history:
        history_text = "\n\nCONVERSATION HISTORY (most recent):\n"
        for turn in history[-3:]:  # Last 3 turns only (context window efficiency)
            history_text += f"User: {turn['query']}\nAssistant: {turn['response'][:200]}...\n\n"

    # ── Compose final prompt ─────────────────────────────────────────────
    user_prompt = f"""DOCUMENT CONTEXT:
{context}

RESEARCHER ANALYSIS:
{researcher_summary}
{history_text}{revision_section}

USER QUESTION: {query}

Provide a comprehensive, accurately sourced answer."""

    # ── Invoke LLM ──────────────────────────────────────────────────────
    try:
        llm = get_llm()
        response = llm.invoke(
            [
                SystemMessage(content=RESPONDER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        draft = response.content
    except Exception as exc:
        logger.error("Responder LLM call failed: %s", exc)
        draft = (
            "I encountered an error generating a response. "
            "Please try again or contact support."
        )

    elapsed = (time.monotonic() - t0) * 1000
    logger.info("Responder generated answer in %.1f ms (revision=%d)", elapsed, revision_count)

    return {
        **state,
        "draft_response": draft,
        "metadata": {
            **state.get("metadata", {}),
            "responder_latency_ms": round(elapsed, 1),
            "revision_count": revision_count,
        },
    }
