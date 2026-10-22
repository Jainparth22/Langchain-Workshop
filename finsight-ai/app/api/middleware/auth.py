"""
app/api/middleware/auth.py
---------------------------
API key authentication middleware.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings

_PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate X-API-Key header on all non-public endpoints."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key", "")
        settings = get_settings()

        if api_key not in settings.api_keys:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "message": "Valid X-API-Key header required.",
                },
            )

        return await call_next(request)
