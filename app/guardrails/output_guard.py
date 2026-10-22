"""
app/guardrails/output_guard.py
-------------------------------
Output safety guardrail — validates the Responder's answer before delivery.

CHECKS PERFORMED:
1. Faithfulness threshold: if Critic score < threshold, return safe fallback.
2. Investment advice disclaimer enforcement.
3. Hallucination flag injection if Critic detected fabrications.
4. Empty/error response detection.
"""

from __future__ import annotations

import logging

from app.agents.state import AgentState
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_INVESTMENT_DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **Disclaimer:** This information is sourced from indexed financial documents "
    "and is provided for informational purposes only. It does not constitute "
    "investment advice. Please consult a qualified financial advisor before "
    "making any investment decisions."
)

_SAFE_FALLBACK = (
    "I was unable to generate a reliable answer for this query based on the "
    "available documents. The information may not be present in the indexed "
    "knowledge base, or the generated answer did not meet quality thresholds. "
    "Please try rephrasing your question or upload additional relevant documents."
)


def output_guard_node(state: AgentState) -> AgentState:
    """
    Post-process and validate the draft response.

    Decisions:
    - FAIL verdict → replace with safe fallback.
    - Low faithfulness score → add uncertainty warning.
    - Hallucinations detected → prepend warning to response.
    - All responses → append investment disclaimer.
    """
    settings = get_settings()
    draft = state.get("draft_response", "")
    verdict = state.get("critic_verdict", "PASS")
    critic_feedback = state.get("critic_feedback", {})

    # Critic FAIL → safe fallback
    if verdict == "FAIL":
        logger.warning("Output guard: FAIL verdict. Returning safe fallback.")
        return {
            **state,
            "final_response": _SAFE_FALLBACK + _INVESTMENT_DISCLAIMER,
            "output_safe": False,
        }

    # Empty response check
    if not draft or len(draft.strip()) < 10:
        logger.warning("Output guard: Empty or minimal response detected.")
        return {
            **state,
            "final_response": _SAFE_FALLBACK + _INVESTMENT_DISCLAIMER,
            "output_safe": False,
        }

    final = draft

    # Low faithfulness warning
    faithfulness = critic_feedback.get("faithfulness_score", 1.0)
    if faithfulness is not None and faithfulness < settings.hallucination_threshold:
        logger.warning(
            "Low faithfulness score (%.2f < %.2f). Adding uncertainty warning.",
            faithfulness, settings.hallucination_threshold,
        )
        final = (
            "⚠️ **Note:** The confidence in this answer's accuracy is below the "
            f"threshold (score: {faithfulness:.2f}). Please verify against source documents.\n\n"
            + final
        )

    # Hallucination flag injection
    hallucinations = critic_feedback.get("hallucinations_detected", [])
    if hallucinations:
        hal_list = "\n".join(f"  - {h}" for h in hallucinations)
        final = (
            f"⚠️ **Potential inaccuracies detected:**\n{hal_list}\n\n"
            + final
        )
        logger.warning(
            "Output guard: %d hallucinations flagged by Critic.", len(hallucinations)
        )

    # Always append investment disclaimer
    final += _INVESTMENT_DISCLAIMER

    return {
        **state,
        "final_response": final,
        "output_safe": True,
    }
