"""
app/retrieval/bm25_store.py
---------------------------
BM25 keyword index for the hybrid search system.

Design Decision:
  WHY BM25 alongside dense vectors?
  - BM25 (Best Match 25) is a probabilistic keyword ranking function.
  - Dense vectors excel at semantic similarity but struggle with:
      * Exact financial figures: "Q3 EPS of $1.47" → needs exact term match.
      * Acronyms: "EBITDA", "COGS", "ROE" — unknown to embedding space.
      * Rare named entities: company names, ticker symbols, executive names.
  - BM25 catches these lexically precise queries that dense search misses.

  WHY rank_bm25 library over Elastic/OpenSearch?
  - rank_bm25 is pure Python, zero infrastructure dependency.
  - For corpus < 100K chunks, Python BM25 runs in <50ms — acceptable latency.
  - For production scale, drop-in replacement: Elasticsearch BM25 via 
    langchain_elasticsearch.ElasticsearchStore.

  Persistence: BM25 index is serialised to disk with pickle for fast restarts.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_BM25_PICKLE_PATH = Path("data/bm25_index.pkl")


class BM25Store:
    """BM25 keyword retrieval index backed by rank_bm25."""

    def __init__(self):
        self._documents: List[Document] = []
        self._bm25 = None  # rank_bm25.BM25Okapi instance

        if _BM25_PICKLE_PATH.exists():
            self._load()

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents and rebuild the BM25 index."""
        self._documents.extend(documents)
        self._rebuild()
        self._persist()
        logger.info("BM25 index updated. Total documents: %d", len(self._documents))

    def search(self, query: str, k: int = 6) -> List[Document]:
        """
        Return the top-k most relevant documents for a keyword query.

        Args:
            query: The user query string.
            k: Number of results to return.

        Returns:
            Ranked list of Documents.
        """
        if self._bm25 is None or not self._documents:
            logger.warning("BM25 index is empty. Returning empty results.")
            return []

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.error("rank_bm25 not installed. Run: pip install rank-bm25")
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = [self._documents[i] for i in top_indices if scores[i] > 0]

        logger.debug("BM25 search '%s' → %d results", query[:50], len(results))
        return results

    @property
    def total_documents(self) -> int:
        return len(self._documents)

    # ── Private ─────────────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        """Rebuild the BM25 index from all stored documents."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.error("rank_bm25 not installed. BM25 search disabled.")
            return

        tokenized_corpus = [
            doc.page_content.lower().split() for doc in self._documents
        ]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def _persist(self) -> None:
        _BM25_PICKLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_BM25_PICKLE_PATH, "wb") as f:
            pickle.dump(
                {"documents": self._documents, "bm25": self._bm25}, f
            )
        logger.debug("BM25 index persisted to %s", _BM25_PICKLE_PATH)

    def _load(self) -> None:
        logger.info("Loading BM25 index from %s", _BM25_PICKLE_PATH)
        with open(_BM25_PICKLE_PATH, "rb") as f:
            data = pickle.load(f)
        self._documents = data["documents"]
        self._bm25 = data["bm25"]
        logger.info("BM25 index loaded. Documents: %d", len(self._documents))
