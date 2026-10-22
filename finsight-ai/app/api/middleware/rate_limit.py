"""
app/api/middleware/rate_limit.py
---------------------------------
Simple in-memory rate limiter (requests per minute per API key).

Production Upgrade Path:
  Replace with Redis-backed rate limiting (slowapi + redis) for distributed deployments.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding window rate limiter keyed by API key."""

    def __init__(self, app):
        super().__init__(app)
        self._windows: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        limit = settings.rate_limit_requests_per_minute
        window = 60  # seconds

        api_key = request.headers.get("X-API-Key", request.client.host)
        now = time.monotonic()
        q = self._windows[api_key]

        # Remove timestamps older than the window
        while q and now - q[0] > window:
            q.popleft()

        if len(q) >= limit:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit: {limit} requests/minute. Retry after {window}s.",
                },
                headers={"Retry-After": str(window)},
            )

        q.append(now)
        return await call_next(request)
