"""
app/retrieval/vectorstore.py
-----------------------------
FAISS vector store manager with persistence.

Design Decision:
  WHY FAISS over Pinecone/Chroma/Weaviate?
  - FAISS (Facebook AI Similarity Search) is the industry-standard in-process
    library for dense vector search.
  - Zero external service dependency → works fully offline.
  - For corpus sizes under ~1M chunks, FAISS.IndexFlatIP (exact inner product)
    achieves sub-10ms latency — faster than any managed service with network RTT.

  UPGRADE PATH for scale:
  - 1M–100M chunks: Swap to FAISS.IndexIVFFlat (ANN with recall trade-off).
  - 100M+ chunks: Migrate to Pinecone/Weaviate with the same retriever interface
    (LangChain's VectorStore abstraction makes this a 3-line change).

  WHY persist to disk?
  - Re-embedding all documents on every server restart would take 10+ minutes
    for a real corpus. Disk persistence makes restarts instantaneous.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.embeddings import get_embeddings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages a persistent FAISS vector store."""

    def __init__(self):
        self.settings = get_settings()
        self._store: Optional[FAISS] = None
        self._index_path = Path(self.settings.faiss_index_path)
        self._index_name = self.settings.faiss_index_name
        self._embeddings = get_embeddings()

        # Auto-load if persisted index exists
        if self._index_exists():
            self._load()

    # ── Public API ──────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to the store, merging with existing index."""
        if not documents:
            logger.warning("add_documents called with empty list.")
            return

        if self._store is None:
            logger.info("Creating new FAISS index with %d chunks", len(documents))
            self._store = FAISS.from_documents(documents, self._embeddings)
        else:
            logger.info(
                "Merging %d new chunks into existing FAISS index", len(documents)
            )
            new_store = FAISS.from_documents(documents, self._embeddings)
            self._store.merge_from(new_store)

        self._persist()
        logger.info(
            "FAISS index updated. Total vectors: %d",
            self._store.index.ntotal,
        )

    def get_retriever(self, k: Optional[int] = None):
        """Return a LangChain retriever with optional k override."""
        self._require_store()
        top_k = k or self.settings.retriever_top_k
        return self._store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

    def similarity_search_with_score(
        self, query: str, k: Optional[int] = None
    ) -> List[tuple[Document, float]]:
        """Return (document, similarity_score) pairs for hybrid search."""
        self._require_store()
        top_k = k or self.settings.retriever_top_k
        return self._store.similarity_search_with_score(query, k=top_k)

    @property
    def total_vectors(self) -> int:
        if self._store is None:
            return 0
        return self._store.index.ntotal

    # ── Private Helpers ─────────────────────────────────────────────────────

    def _require_store(self) -> None:
        if self._store is None:
            raise RuntimeError(
                "Vector store is empty. Ingest documents first via POST /ingest."
            )

    def _index_exists(self) -> bool:
        index_file = self._index_path / f"{self._index_name}.faiss"
        return index_file.exists()

    def _persist(self) -> None:
        self._index_path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(
            str(self._index_path), index_name=self._index_name
        )
        logger.debug("FAISS index persisted to %s", self._index_path)

    def _load(self) -> None:
        logger.info("Loading existing FAISS index from %s", self._index_path)
        self._store = FAISS.load_local(
            str(self._index_path),
            self._embeddings,
            index_name=self._index_name,
            allow_dangerous_deserialization=True,
        )
        logger.info(
            "FAISS index loaded. Vectors: %d", self._store.index.ntotal
        )
