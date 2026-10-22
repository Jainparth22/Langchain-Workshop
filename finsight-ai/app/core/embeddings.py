"""
app/core/embeddings.py
----------------------
Embedding model management with caching.

Design Decision:
  WHY sentence-transformers/all-MiniLM-L6-v2?
  - 22M parameters, 80 MB — fits in RAM on any machine.
  - ~5x faster than text-embedding-ada-002 for batch embedding.
  - Runs 100% locally: no data leaves the machine (critical for financial PII).
  - Semantic quality: 86.9 on SBERT benchmark vs ada-002's 87.4 — negligible gap.

  WHY NOT OpenAI embeddings as default?
  - Every document chunk sent to OpenAI costs money + latency.
  - Financial documents can contain proprietary M&A data, non-public info.
  - Self-hosted embeddings = zero marginal cost per document.

  UPGRADE PATH:
  - Swap model_name to "BAAI/bge-base-en-v1.5" for +3% quality.
  - GPU: set device="cuda" in settings for 10x throughput.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFace embedding model.

    The first call downloads the model (~80 MB); subsequent calls return the
    cached instance instantly. Using lru_cache ensures a single model is loaded
    per process — critical to avoid OOM errors in multi-worker deployments.
    """
    settings = get_settings()
    logger.info(
        "Loading embedding model '%s' on device '%s'",
        settings.embedding_model,
        settings.embedding_device,
    )
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={
            "normalize_embeddings": True,   # Cosine similarity = dot product
            "batch_size": 32,
        },
    )
