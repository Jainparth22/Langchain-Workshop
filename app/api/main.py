"""
app/api/main.py
----------------
FastAPI application entrypoint for FinSight AI.

Endpoints:
  POST /chat                   — Multi-agent financial Q&A
  POST /ingest/file            — Upload and index a document
  POST /ingest/directory       — Index all documents in a directory
  POST /evaluate               — Run RAGAS quality evaluation
  GET  /sessions/{session_id}  — Get conversation history
  DELETE /sessions/{session_id}— Delete a session
  GET  /health                 — Service health check
  GET  /docs                   — Swagger UI (auto-generated)
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.middleware.auth import APIKeyMiddleware
from app.api.middleware.rate_limit import RateLimitMiddleware
from app.api.routes import chat, evaluate, health, ingest, sessions
from app.core.config import get_settings
from app.core.exceptions import FinSightBaseException, GuardrailViolation
from app.logging_.tracker import setup_logging

# Initialise structured logging before anything else
setup_logging()
logger = logging.getLogger(__name__)

settings = get_settings()

# ── FastAPI App ────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Chat", "description": "Multi-agent financial Q&A pipeline"},
        {"name": "Ingest", "description": "Document ingestion and indexing"},
        {"name": "Evaluate", "description": "RAGAS quality evaluation"},
        {"name": "Sessions", "description": "Conversation history management"},
        {"name": "Health", "description": "Service health and metrics"},
    ],
)

# ── Middleware ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(RateLimitMiddleware)

# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(ingest.router, prefix="/ingest", tags=["Ingest"])
app.include_router(evaluate.router, prefix="/evaluate", tags=["Evaluate"])
app.include_router(sessions.router, prefix="/sessions", tags=["Sessions"])
app.include_router(health.router, tags=["Health"])


# ── Global Exception Handlers ─────────────────────────────────────────────
@app.exception_handler(GuardrailViolation)
async def guardrail_handler(request, exc: GuardrailViolation):
    return JSONResponse(
        status_code=422,
        content={
            "error": "guardrail_violation",
            "guard_type": exc.guard_type,
            "message": exc.message,
        },
    )


@app.exception_handler(FinSightBaseException)
async def finsight_exception_handler(request, exc: FinSightBaseException):
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "message": exc.message},
    )


@app.on_event("startup")
async def startup():
    logger.info(
        "FinSight AI v%s starting up | debug=%s",
        settings.app_version,
        settings.debug,
    )


@app.on_event("shutdown")
async def shutdown():
    logger.info("FinSight AI shutting down.")
