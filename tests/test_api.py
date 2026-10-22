"""
tests/test_api.py
-----------------
Integration tests for FastAPI endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)

HEADERS = {"X-API-Key": "dev-key-change-in-production"}


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data

    def test_health_no_auth_required(self):
        # Health endpoint must be public (no API key required)
        response = client.get("/health")
        assert response.status_code == 200


class TestAuthMiddleware:
    def test_chat_without_api_key_returns_401(self):
        response = client.post("/chat", json={"query": "test query"})
        assert response.status_code == 401

    def test_chat_with_invalid_api_key_returns_401(self):
        response = client.post(
            "/chat",
            json={"query": "test query"},
            headers={"X-API-Key": "invalid-key"},
        )
        assert response.status_code == 401


class TestInputValidation:
    def test_chat_query_too_short(self):
        response = client.post(
            "/chat",
            json={"query": "Hi"},
            headers=HEADERS,
        )
        assert response.status_code in (400, 422)

    def test_chat_missing_query_field(self):
        response = client.post(
            "/chat",
            json={},
            headers=HEADERS,
        )
        assert response.status_code == 422


class TestSessionEndpoints:
    def test_create_new_session(self):
        response = client.post("/sessions/new", headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data

    def test_list_sessions(self):
        response = client.get("/sessions", headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data

    def test_get_nonexistent_session_returns_404(self):
        response = client.get(
            "/sessions/nonexistent-session-id/history",
            headers=HEADERS,
        )
        assert response.status_code == 404
