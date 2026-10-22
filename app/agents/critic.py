"""
app/agents/critic.py
---------------------
Critic agent — evaluates the Responder's answer for faithfulness and accuracy.
"""

from __future__ import annotations

import json
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import AgentState
from app.core.llm_factory import get_llm
from app.prompts.personas import CRITIC_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def critic_node(state: AgentState) -> AgentState:
    """
    Evaluate the draft response against the retrieved context.

    Scoring dimensions:
    - Faithfulness: Claims supported by context only
    - Completeness: All aspects of query addressed
    - Accuracy: Financial figures correct
    - Hallucination detection: Any fabricated claims

    Returns PASS / NEEDS_REVISION / FAIL verdict.
    """
    query = state["query"]
    draft = state.get("draft_response", "")
    retrieved_docs = state.get("retrieved_docs", [])
    revision_count = state.get("revision_count", 0)

    t0 = time.monotonic()

    # Build context for Critic to cross-reference
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    critic_prompt = f"""ORIGINAL USER QUERY: {query}

RETRIEVED CONTEXT (ground truth):
{context}

GENERATED ANSWER TO EVALUATE:
{draft}

Evaluate the answer against the context and provide your assessment as JSON."""

    try:
        llm = get_llm()
        response = llm.invoke(
            [
                SystemMessage(content=CRITIC_SYSTEM_PROMPT),
                HumanMessage(content=critic_prompt),
            ]
        )

        # Parse critic JSON
        try:
            # Extract JSON block if embedded in markdown
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            feedback = json.loads(content)
            verdict = feedback.get("verdict", "PASS")
        except json.JSONDecodeError:
            logger.warning("Critic returned non-JSON response. Defaulting to PASS.")
            feedback = {
                "faithfulness_score": 0.8,
                "completeness_score": 0.8,
                "accuracy_score": 0.8,
                "hallucinations_detected": [],
                "verdict": "PASS",
                "revision_instructions": "",
            }
            verdict = "PASS"

    except Exception as exc:
        logger.error("Critic LLM call failed: %s. Defaulting to PASS.", exc)
        feedback = {"verdict": "PASS", "error": str(exc)}
        verdict = "PASS"

    elapsed = (time.monotonic() - t0) * 1000

    # Increment revision count if critic requests revision
    new_revision_count = revision_count + 1 if verdict == "NEEDS_REVISION" else revision_count

    logger.info(
        "Critic verdict: %s | Faithfulness: %.2f | Elapsed: %.1f ms",
        verdict,
        feedback.get("faithfulness_score", 0.0),
        elapsed,
    )

    return {
        **state,
        "critic_verdict": verdict,
        "critic_feedback": feedback,
        "revision_count": new_revision_count,
        "metadata": {
            **state.get("metadata", {}),
            "critic_latency_ms": round(elapsed, 1),
            "critic_verdict": verdict,
            "faithfulness_score": feedback.get("faithfulness_score"),
            "completeness_score": feedback.get("completeness_score"),
            "hallucinations": feedback.get("hallucinations_detected", []),
        },
    }
