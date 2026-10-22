"""
app/core/llm_factory.py
-----------------------
Multi-provider LLM factory with automatic waterfall fallback.

Design Decision:
  WHY multi-provider?
  - Production systems cannot rely on a single vendor's uptime.
  - Groq is the primary provider (fastest inference, ~300 tok/s on Llama/Qwen).
  - OpenAI GPT-4o-mini is the secondary fallback (widest capability).
  - Local Ollama is the tertiary fallback (fully offline / air-gapped deployments).

  WHY waterfall over load-balancing?
  - Load balancing distributes cost; waterfall minimises cost (cheapest first).
  - In a financial context, deterministic routing aids audit trails.

  WHY temperature=0?
  - Financial figures must be exact. Temperature > 0 introduces stochastic
    variation that can silently alter numbers (e.g., "$4.2B" → "$4.7B").
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from langchain_core.language_models import BaseChatModel

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _build_groq() -> Optional[BaseChatModel]:
    """Build Groq Chat LLM. Returns None if key absent or import fails."""
    try:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            temperature=settings.groq_temperature,
            max_tokens=settings.groq_max_tokens,
            groq_api_key=settings.groq_api_key,
        )
    except Exception as exc:
        logger.warning("Groq provider unavailable: %s", exc)
        return None


def _build_openai() -> Optional[BaseChatModel]:
    """Build OpenAI Chat LLM. Returns None if key absent or import fails."""
    if not settings.openai_api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            api_key=settings.openai_api_key,
        )
    except Exception as exc:
        logger.warning("OpenAI provider unavailable: %s", exc)
        return None


def _build_ollama() -> Optional[BaseChatModel]:
    """Build local Ollama LLM. Returns None if service unreachable."""
    try:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
    except Exception as exc:
        logger.warning("Ollama provider unavailable: %s", exc)
        return None


_PROVIDER_BUILDERS = {
    "groq": _build_groq,
    "openai": _build_openai,
    "ollama": _build_ollama,
}

_cached_llm: Optional[BaseChatModel] = None
_active_provider: str = "none"


def get_llm(force_provider: Optional[str] = None) -> BaseChatModel:
    """
    Return a ready-to-use chat LLM following the configured provider order.

    Args:
        force_provider: Override the waterfall and use a specific provider.
                        Useful for testing provider-specific behaviour.

    Returns:
        An instantiated BaseChatModel.

    Raises:
        RuntimeError: If no provider is available.
    """
    global _cached_llm, _active_provider

    # Return cached instance unless a specific provider is forced
    if _cached_llm and not force_provider:
        return _cached_llm

    order = [force_provider] if force_provider else settings.llm_provider_order

    for provider in order:
        builder = _PROVIDER_BUILDERS.get(provider)
        if builder is None:
            logger.warning("Unknown provider '%s', skipping.", provider)
            continue

        logger.info("Attempting LLM provider: %s", provider)
        start = time.monotonic()
        llm = builder()
        if llm:
            elapsed = (time.monotonic() - start) * 1000
            logger.info("Provider '%s' initialised in %.1f ms", provider, elapsed)
            _cached_llm = llm
            _active_provider = provider
            return llm

    raise RuntimeError(
        "No LLM provider is available. "
        "Please set GROQ_API_KEY, OPENAI_API_KEY, or start Ollama locally."
    )


def get_active_provider() -> str:
    """Return the name of the currently active provider."""
    return _active_provider
