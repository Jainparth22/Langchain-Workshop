"""
app/api/routes/ingest.py
-------------------------
Document ingestion endpoints.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.ingestion.pipeline import IngestionPipeline

router = APIRouter()
logger = logging.getLogger(__name__)

_pipeline = IngestionPipeline()


class IngestResponse(BaseModel):
    status: str
    file: str
    raw_segments: int
    chunks: int
    elapsed_s: float
    strategy: str


class DirectoryIngestRequest(BaseModel):
    directory_path: str


@router.post(
    "/file",
    response_model=IngestResponse,
    summary="Upload and index a document",
    description=(
        "Upload a PDF, DOCX, TXT, MD, or CSV file. "
        "The system will load, chunk, embed, and index it into both "
        "the FAISS vector store and the BM25 keyword index."
    ),
)
async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
    """Ingest an uploaded document into the RAG knowledge base."""
    logger.info("Ingest request: %s (%s)", file.filename, file.content_type)

    try:
        contents = await file.read()
        stats = _pipeline.ingest_bytes(contents, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Ingestion failed for '%s': %s", file.filename, exc)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(exc)}")

    return IngestResponse(status="success", **stats)


@router.post(
    "/directory",
    response_model=IngestResponse,
    summary="Index all documents in a server-side directory",
)
async def ingest_directory(request: DirectoryIngestRequest) -> IngestResponse:
    """Index all supported documents from a directory on the server."""
    try:
        stats = _pipeline.ingest_directory(request.directory_path)
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return IngestResponse(status="success", **stats)
