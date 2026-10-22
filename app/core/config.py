"""
app/core/config.py
------------------
Centralised configuration management using Pydantic-Settings.

Design Decision:
  - Pydantic Settings reads from environment variables (12-factor app pattern).
  - Single source of truth for all tuneable parameters.
  - Validators enforce sensible bounds so misconfigured deployments fail fast
    at startup rather than silently misbehaving at runtime.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "FinSight AI"
    app_version: str = "1.0.0"
    app_description: str = (
        "Multi-Agent Financial Intelligence Platform powered by RAG + LangGraph"
    )
    debug: bool = False
    log_level: str = "INFO"

    # ── API Security ──────────────────────────────────────────────────────────
    api_keys: List[str] = Field(
        default=["dev-key-change-in-production"],
        description="Comma-separated list of valid API keys (or JSON list in env)",
    )
    rate_limit_requests_per_minute: int = 60

    # ── LLM Providers ────────────────────────────────────────────────────────
    # Primary
    groq_api_key: str = Field(default="", description="Groq Cloud API key")
    groq_model: str = "qwen/qwen3-32b"
    groq_temperature: float = 0.0
    groq_max_tokens: int = 2048

    # Secondary fallback
    openai_api_key: str = Field(default="", description="OpenAI API key (fallback)")
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.0

    # Tertiary (local Ollama — completely offline)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # Provider selection order
    llm_provider_order: List[str] = ["groq", "openai", "ollama"]

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"  # "cuda" if GPU available

    # ── Vector Store ─────────────────────────────────────────────────────────
    faiss_index_path: str = "data/faiss_index"
    faiss_index_name: str = "finsight"

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size_narrative: int = 512      # For text-heavy sections (MD&A, Risk)
    chunk_size_tabular: int = 256        # For tables and structured data
    chunk_overlap: int = 64
    chunking_strategy: Literal["recursive", "semantic", "sentence"] = "recursive"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retriever_top_k: int = 6            # Retrieve more, rerank to top-3
    reranker_top_n: int = 3             # Final chunks passed to LLM
    hybrid_alpha: float = 0.6          # Weight: 0=pure BM25, 1=pure dense
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Memory / Sessions ────────────────────────────────────────────────────
    sqlite_db_path: str = "data/finsight.db"
    session_max_history: int = 20       # Max turns kept in context

    # ── Guardrails ───────────────────────────────────────────────────────────
    enable_input_guard: bool = True
    enable_output_guard: bool = True
    hallucination_threshold: float = 0.3   # Flag if faithfulness < threshold

    # ── Evaluation (RAGAS) ───────────────────────────────────────────────────
    ragas_sample_size: int = 10

    # ── LangSmith (optional observability) ───────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "finsight-ai"

    @field_validator("groq_api_key")
    @classmethod
    def groq_key_required(cls, v: str) -> str:
        if not v:
            raise ValueError(
                "GROQ_API_KEY is required. "
                "Get one free at https://console.groq.com/"
            )
        return v

    @field_validator("hybrid_alpha")
    @classmethod
    def alpha_bounds(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("hybrid_alpha must be between 0.0 and 1.0")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance (instantiated once per process)."""
    return Settings()
