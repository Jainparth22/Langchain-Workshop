"""
app/api/routes/sessions.py
---------------------------
Conversation session management endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.memory.session import SessionManager

router = APIRouter()
_session_manager = SessionManager()


class SessionSummary(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    message_count: int


@router.get(
    "/{session_id}/history",
    summary="Get conversation history for a session",
)
async def get_history(session_id: str, limit: int = 20) -> dict:
    """Return conversation turns for a session."""
    history = _session_manager.get_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or empty.")
    return {"session_id": session_id, "turns": history, "count": len(history)}


@router.delete(
    "/{session_id}",
    summary="Delete a session and its history",
)
async def delete_session(session_id: str) -> dict:
    """Permanently delete a session and all its conversation history."""
    deleted = _session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"status": "deleted", "session_id": session_id}


@router.get(
    "",
    summary="List all sessions",
)
async def list_sessions() -> dict:
    """Return a list of all active sessions with summary metadata."""
    sessions = _session_manager.list_sessions()
    return {"sessions": sessions, "count": len(sessions)}


@router.post(
    "/new",
    summary="Create a new session",
)
async def new_session() -> dict:
    """Generate a new session ID."""
    session_id = _session_manager.new_session()
    return {"session_id": session_id, "status": "created"}
