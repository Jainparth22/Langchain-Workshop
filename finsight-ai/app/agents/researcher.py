"""
app/agents/researcher.py
-------------------------
Researcher agent node — retrieves and structures relevant context.
"""

from __future__ import annotations

import json
import logging
import time
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import AgentState
from app.core.llm_factory import get_llm
from app.prompts.personas import RESEARCHER_SYSTEM_PROMPT
from app.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

# Module-level retriever (shared across calls)
_retriever = HybridRetriever()

# Tier → top_k mapping: complex queries need more context
_TIER_TO_K = {1: 3, 2: 5, 3: 8}


def researcher_node(state: AgentState) -> AgentState:
    """
    Retrieve relevant document chunks and extract structured information.

    On revision loops, the researcher receives critic feedback and adjusts
    its extraction to address identified gaps.
    """
    query = state["query"]
    tier = state.get("query_tier", 2)
    revision_count = state.get("revision_count", 0)
    critic_feedback = state.get("critic_feedback", {})

    t0 = time.monotonic()

    # Augment query with critic's revision instructions if looping
    augmented_query = query
    if revision_count > 0 and critic_feedback:
        revision_hints = critic_feedback.get("revision_instructions", "")
        augmented_query = (
            f"Original query: {query}\n\n"
            f"Previous answer was insufficient. Focus on: {revision_hints}"
        )
        logger.info("Researcher revision %d: augmented query with critic hints.", revision_count)

    # Retrieve docs — tier-aware k
    top_k = _TIER_TO_K.get(tier, 5)
    retrieved_docs = _retriever.retrieve(augmented_query, top_n=top_k)

    if not retrieved_docs:
        logger.warning("No documents retrieved for query: '%s'", query[:60])
        return {
            **state,
            "retrieved_docs": [],
            "researcher_extracts": {
                "relevant_extracts": [],
                "key_entities": [],
                "coverage_assessment": "insufficient",
                "missing_aspects": ["No relevant documents found in the knowledge base."],
            },
            "revision_count": revision_count,
        }

    # Build context string
    context = "\n\n---\n\n".join(
        [
            f"[Source: {doc.metadata.get('source', 'Unknown')}, "
            f"Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
            for doc in retrieved_docs
        ]
    )

    # LLM-based structured extraction
    try:
        llm = get_llm()
        extraction_prompt = (
            f"User Query: {query}\n\n"
            f"Document Context:\n{context}\n\n"
            "Extract and structure all relevant information from the context."
        )
        response = llm.invoke(
            [
                SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
                HumanMessage(content=extraction_prompt),
            ]
        )
        # Parse JSON output from Researcher
        try:
            extracts = json.loads(response.content.strip())
        except json.JSONDecodeError:
            # If LLM didn't return valid JSON, wrap raw text
            extracts = {
                "relevant_extracts": [response.content],
                "key_entities": [],
                "coverage_assessment": "partial",
                "missing_aspects": [],
            }
    except Exception as exc:
        logger.error("Researcher LLM call failed: %s", exc)
        extracts = {
            "relevant_extracts": [doc.page_content for doc in retrieved_docs],
            "key_entities": [],
            "coverage_assessment": "partial",
            "missing_aspects": ["LLM extraction unavailable; raw chunks provided."],
        }

    elapsed = (time.monotonic() - t0) * 1000
    logger.info(
        "Researcher: %d docs retrieved, coverage=%s, %.1f ms",
        len(retrieved_docs),
        extracts.get("coverage_assessment"),
        elapsed,
    )

    # Build source citation list
    sources = [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "rrf_score": doc.metadata.get("rrf_score", 0.0),
            "reranker_score": doc.metadata.get("reranker_score", 0.0),
            "snippet": doc.page_content[:150],
        }
        for doc in retrieved_docs
    ]

    return {
        **state,
        "retrieved_docs": retrieved_docs,
        "researcher_extracts": extracts,
        "revision_count": revision_count,
        "metadata": {
            **state.get("metadata", {}),
            "researcher_latency_ms": round(elapsed, 1),
            "docs_retrieved": len(retrieved_docs),
            "coverage": extracts.get("coverage_assessment"),
            "sources": sources,
        },
    }
