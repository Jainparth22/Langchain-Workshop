"""
app/ingestion/pipeline.py
-------------------------
End-to-end document ingestion pipeline.

Orchestrates: Load → Chunk → Embed → Index (FAISS + BM25)

This is the single entry point for adding new knowledge to FinSight AI.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app.core.config import get_settings
from app.ingestion.chunker import chunk_documents
from app.ingestion.loaders import load_directory, load_document
from app.retrieval.bm25_store import BM25Store
from app.retrieval.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Full document ingestion pipeline.

    Usage:
        pipeline = IngestionPipeline()
        stats = pipeline.ingest_file("data/apple_10k.pdf")
        # stats = {"chunks": 142, "elapsed_s": 8.3, "strategy": "recursive"}
    """

    def __init__(self):
        self.settings = get_settings()
        self.vector_store = VectorStoreManager()
        self.bm25_store = BM25Store()

    def ingest_file(self, file_path: str) -> dict:
        """
        Ingest a single file into both FAISS and BM25 indexes.

        Args:
            file_path: Path to the document.

        Returns:
            Statistics dict with chunk count, elapsed time, and strategy used.
        """
        start = time.monotonic()
        logger.info("Ingesting file: %s", file_path)

        # Step 1: Load
        documents = load_document(file_path)
        logger.info("Loaded %d raw segment(s)", len(documents))

        # Step 2: Chunk
        chunks = chunk_documents(documents)
        logger.info("Produced %d chunks", len(chunks))

        # Step 3: Index into FAISS (dense vectors)
        self.vector_store.add_documents(chunks)

        # Step 4: Index into BM25 (keyword index)
        self.bm25_store.add_documents(chunks)

        elapsed = time.monotonic() - start
        stats = {
            "file": Path(file_path).name,
            "raw_segments": len(documents),
            "chunks": len(chunks),
            "elapsed_s": round(elapsed, 2),
            "strategy": self.settings.chunking_strategy,
        }
        logger.info("Ingestion complete: %s", stats)
        return stats

    def ingest_directory(self, dir_path: str) -> dict:
        """
        Ingest all supported files from a directory.

        Args:
            dir_path: Path to the directory.

        Returns:
            Aggregated statistics.
        """
        start = time.monotonic()
        logger.info("Ingesting directory: %s", dir_path)

        documents = load_directory(dir_path)
        chunks = chunk_documents(documents)

        self.vector_store.add_documents(chunks)
        self.bm25_store.add_documents(chunks)

        elapsed = time.monotonic() - start
        stats = {
            "directory": dir_path,
            "raw_segments": len(documents),
            "chunks": len(chunks),
            "elapsed_s": round(elapsed, 2),
            "strategy": self.settings.chunking_strategy,
        }
        logger.info("Directory ingestion complete: %s", stats)
        return stats

    def ingest_bytes(self, file_bytes: bytes, filename: str) -> dict:
        """
        Ingest a file uploaded as raw bytes (for FastAPI file upload endpoint).

        Args:
            file_bytes: Raw file content.
            filename: Original filename (used to detect type).

        Returns:
            Statistics dict.
        """
        import tempfile

        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, mode="wb"
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            return self.ingest_file(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
