"""
app/memory/session.py
----------------------
Session manager — thin interface over ConversationStore.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from app.memory.store import ConversationStore

_store = ConversationStore()


class SessionManager:
    """High-level session API used by agent nodes."""

    def new_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        _store.create_session(session_id)
        return session_id

    def get_history(self, session_id: str) -> List[Dict]:
        """Return conversation history for a session."""
        return _store.get_history(session_id)

    def save_turn(
        self,
        session_id: str,
        query: str,
        response: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Persist a user-assistant turn."""
        _store.append_turn(session_id, query, response, metadata)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        return _store.delete_session(session_id)

    def list_sessions(self) -> List[Dict]:
        """Return all session summaries."""
        return _store.list_sessions()
