"""
tests/test_ingestion.py
-----------------------
Unit tests for document loading and chunking pipeline.
"""

import os
import tempfile

import pytest

from app.ingestion.loaders import load_document
from app.ingestion.chunker import chunk_documents, _detect_content_type


class TestDocumentLoaders:
    def test_load_text_file(self):
        """Test basic text file loading."""
        content = "TechNova Solutions Annual Report 2024.\n\nRevenue: $4.2B"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name

        try:
            docs = load_document(tmp_path)
            assert len(docs) >= 1
            assert "TechNova" in docs[0].page_content
            assert docs[0].metadata["file_type"] == "text"
            assert docs[0].metadata["source"] == os.path.basename(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_load_markdown_file(self):
        """Test Markdown file loading."""
        content = "# FinSight AI\n\n## Q3 Results\n\nEBITDA: $1.2B (+12% YoY)"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name

        try:
            docs = load_document(tmp_path)
            assert len(docs) >= 1
        finally:
            os.unlink(tmp_path)

    def test_unsupported_format_raises(self):
        """Test that unsupported file formats raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            tmp_path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                load_document(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_missing_file_raises(self):
        """Test that missing files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_document("/nonexistent/path/report.pdf")

    def test_metadata_enrichment(self):
        """Test that metadata is correctly attached to loaded documents."""
        content = "Financial report content."
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name

        try:
            docs = load_document(tmp_path)
            meta = docs[0].metadata
            assert "source" in meta
            assert "file_path" in meta
            assert "file_size_kb" in meta
            assert "file_type" in meta
        finally:
            os.unlink(tmp_path)


class TestChunker:
    def _make_doc(self, content: str):
        from langchain_core.documents import Document
        return Document(page_content=content, metadata={"source": "test.txt"})

    def test_recursive_chunking_produces_chunks(self):
        """Test that chunking splits long documents."""
        long_text = "The company reported strong earnings. " * 100
        doc = self._make_doc(long_text)
        chunks = chunk_documents([doc], strategy="recursive")
        assert len(chunks) > 1

    def test_chunk_metadata_added(self):
        """Test that chunk index metadata is attached."""
        text = "Annual revenue grew by 15% year-over-year. " * 30
        doc = self._make_doc(text)
        chunks = chunk_documents([doc], strategy="recursive")
        assert "chunk_index" in chunks[0].metadata
        assert "chunking_strategy" in chunks[0].metadata
        assert "content_type" in chunks[0].metadata

    def test_detect_tabular_content(self):
        """Test that tabular content is correctly detected."""
        tabular = "Balance Sheet Q3 2023: Cash $1.2B, Revenue $4.5B, EBITDA $0.9B"
        narrative = "The company has shown consistent growth in international markets."
        assert _detect_content_type(tabular) == "tabular"
        assert _detect_content_type(narrative) == "narrative"

    def test_empty_documents(self):
        """Test chunking with empty document list."""
        chunks = chunk_documents([])
        assert chunks == []
