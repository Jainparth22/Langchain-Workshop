"""
app/api/routes/health.py
-------------------------
Health check and system metrics endpoints.
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.llm_factory import get_active_provider

router = APIRouter()

_START_TIME = datetime.now(timezone.utc)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    uptime_seconds: float
    active_llm_provider: str
    python_version: str
    platform: str
    timestamp: str


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["Health"],
)
async def health() -> HealthResponse:
    """Returns service health status and runtime metadata."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    uptime = (now - _START_TIME).total_seconds()

    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=settings.app_version,
        uptime_seconds=round(uptime, 1),
        active_llm_provider=get_active_provider() or "none",
        python_version=sys.version.split()[0],
        platform=platform.system(),
        timestamp=now.isoformat(),
    )
