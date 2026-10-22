"""
app/ingestion/loaders.py
------------------------
Multi-format document loaders with unified output schema.

Design Decision:
  WHY support PDF, DOCX, CSV, TXT, MD?
  - Financial professionals work across all these formats:
      PDF  → Annual reports, SEC filings, analyst research
      DOCX → Internal memos, board presentations
      CSV  → Financial tables, income statements
      TXT/MD → Policy docs, changelogs, internal wikis
  - A production RAG must ingest *any* document type without manual preprocessing.

  WHY add metadata at load time?
  - Source attribution is mandatory in financial Q&A (audit, compliance).
  - Metadata (filename, page_number, document_type) flows through the pipeline
    and surfaces in the final answer as citations.

  FALLBACK: If PyMuPDF (fitz) is unavailable, falls back to PyPDF2.
"""

from __future__ import annotations

import logging
import mimetypes
import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Supported MIME type → loader strategy mapping
_EXTENSION_MAP = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".csv": "csv",
    ".txt": "text",
    ".md": "text",
    ".markdown": "text",
}


def _load_pdf(file_path: str) -> List[Document]:
    """Load PDF with page-level metadata."""
    try:
        # Prefer PyMuPDF for better table/layout preservation
        from langchain_community.document_loaders import PyMuPDFLoader

        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        logger.debug("Loaded %d PDF pages via PyMuPDF from %s", len(docs), file_path)
        return docs
    except ImportError:
        logger.warning("PyMuPDF not installed, falling back to PyPDFLoader")
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(file_path)
        return loader.load()


def _load_docx(file_path: str) -> List[Document]:
    """Load Word document with paragraph-level chunking hint."""
    from langchain_community.document_loaders import Docx2txtLoader

    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    logger.debug("Loaded DOCX: %s (%d docs)", file_path, len(docs))
    return docs


def _load_csv(file_path: str) -> List[Document]:
    """
    Load CSV as row-level documents.

    Design Decision:
      Each row becomes an independent document so that the retriever can
      surface specific financial rows (e.g., "Q3 revenue for Apple").
      Row → Document avoids the 'needle in a haystack' problem of large CSVs.
    """
    from langchain_community.document_loaders.csv_loader import CSVLoader

    loader = CSVLoader(
        file_path=file_path,
        csv_args={"delimiter": ","},
        encoding="utf-8",
    )
    docs = loader.load()
    logger.debug("Loaded CSV: %s (%d row-documents)", file_path, len(docs))
    return docs


def _load_text(file_path: str) -> List[Document]:
    """Load plain text / Markdown with UTF-8 encoding."""
    from langchain_community.document_loaders import TextLoader

    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    logger.debug("Loaded TEXT: %s", file_path)
    return docs


_LOADER_MAP = {
    "pdf": _load_pdf,
    "docx": _load_docx,
    "csv": _load_csv,
    "text": _load_text,
}


def load_document(file_path: str) -> List[Document]:
    """
    Load any supported document type and enrich with source metadata.

    Args:
        file_path: Absolute or relative path to the document.

    Returns:
        List of LangChain Document objects with enriched metadata.

    Raises:
        ValueError: If the file type is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    extension = path.suffix.lower()
    strategy = _EXTENSION_MAP.get(extension)
    if not strategy:
        raise ValueError(
            f"Unsupported file type '{extension}'. "
            f"Supported: {list(_EXTENSION_MAP.keys())}"
        )

    loader_fn = _LOADER_MAP[strategy]
    docs = loader_fn(str(path))

    # ── Enrich metadata for every chunk ──────────────────────────────────────
    file_size_kb = round(path.stat().st_size / 1024, 1)
    for i, doc in enumerate(docs):
        doc.metadata.update(
            {
                "source": path.name,
                "file_path": str(path.resolve()),
                "file_type": strategy,
                "file_size_kb": file_size_kb,
                "document_index": i,
            }
        )

    logger.info(
        "Loaded '%s' (%s, %.1f KB) → %d document segment(s)",
        path.name, strategy, file_size_kb, len(docs),
    )
    return docs


def load_directory(dir_path: str, recursive: bool = True) -> List[Document]:
    """
    Load all supported documents from a directory.

    Args:
        dir_path: Path to the directory.
        recursive: If True, walks subdirectories.

    Returns:
        Combined list of Documents from all files.
    """
    root = Path(dir_path)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    all_docs: List[Document] = []
    pattern = "**/*" if recursive else "*"

    for file_path in root.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in _EXTENSION_MAP:
            try:
                docs = load_document(str(file_path))
                all_docs.extend(docs)
            except Exception as exc:
                logger.error("Failed to load '%s': %s", file_path.name, exc)

    logger.info(
        "Directory load complete: %d total document segments from '%s'",
        len(all_docs), dir_path,
    )
    return all_docs
