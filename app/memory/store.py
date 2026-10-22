"""
app/memory/store.py
--------------------
SQLite-backed conversation persistence.

Design Decision:
  WHY SQLite over Redis?
  - SQLite: zero infrastructure, perfect for single-node dev/demo deployments.
  - Redis would be the production choice for multi-node / stateless deployments.
  - Upgrade path: swap SQLite with Redis via a 10-line change in session.py
    (langchain-redis provides a drop-in replacement for conversation history).

  SCHEMA:
  - sessions: session_id, created_at, user_id
  - messages: id, session_id, role, content, timestamp, metadata_json
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class ConversationStore:
    """Manages conversation history in a SQLite database."""

    def __init__(self):
        settings = get_settings()
        self.db_path = Path(settings.sqlite_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id  TEXT PRIMARY KEY,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    message_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    timestamp   TEXT NOT NULL,
                    metadata    TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, timestamp);
            """)
        logger.debug("SQLite conversation store initialised at %s", self.db_path)

    def create_session(self, session_id: str) -> None:
        """Create a new session or no-op if it already exists."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at)
                VALUES (?, ?, ?)
                """,
                (session_id, now, now),
            )

    def append_turn(
        self,
        session_id: str,
        query: str,
        response: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Append a user-assistant turn to the session history."""
        self.create_session(session_id)
        now = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(metadata or {})

        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO messages (session_id, role, content, timestamp, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, "user", query, now, "{}"),
            )
            conn.execute(
                """INSERT INTO messages (session_id, role, content, timestamp, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, "assistant", response, now, metadata_json),
            )
            conn.execute(
                """UPDATE sessions
                   SET updated_at = ?, message_count = message_count + 2
                   WHERE session_id = ?""",
                (now, session_id),
            )

    def get_history(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Return the conversation history for a session.

        Args:
            session_id: The session identifier.
            limit: Maximum number of message pairs to return.

        Returns:
            List of {"role": str, "content": str, "timestamp": str} dicts.
        """
        settings = get_settings()
        max_rows = (limit or settings.session_max_history) * 2  # pairs → rows

        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT role, content, timestamp FROM messages
                   WHERE session_id = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (session_id, max_rows),
            ).fetchall()

        # Return in chronological order
        return [dict(row) for row in reversed(rows)]

    def delete_session(self, session_id: str) -> bool:
        """Delete all messages and session record. Returns True if existed."""
        with self._get_conn() as conn:
            result = conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
        return result.rowcount > 0

    def list_sessions(self) -> List[Dict]:
        """Return summary of all sessions."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT session_id, created_at, updated_at, message_count
                   FROM sessions ORDER BY updated_at DESC"""
            ).fetchall()
        return [dict(row) for row in rows]
