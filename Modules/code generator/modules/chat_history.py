"""
Chat History Module
Persists multi-turn conversation history per session (keyed by request_id / session_id).
Each session stores the sequence of user prompts and assistant responses,
which are injected into the LLM context on subsequent turns.

Storage: SQLite (same DB as memory module, separate table).
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH  = Path(__file__).parent.parent / "data" / "memory.db"
MAX_TURNS_IN_CONTEXT = 10  # How many past turns to inject into the LLM


class ChatHistory:
    def __init__(self):
        self.conn: Optional[sqlite3.Connection] = None

    async def connect(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self._init_schema()
        logger.info("ChatHistory: connected.")

    async def disconnect(self):
        if self.conn:
            self.conn.close()

    # ─── Schema ───────────────────────────────────────────────────────────────

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                role        TEXT    NOT NULL CHECK(role IN ('user', 'assistant')),
                content     TEXT    NOT NULL,
                intent      TEXT,
                language    TEXT,
                metadata    TEXT,
                created     TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_session ON chat_sessions(session_id, id);
        """)
        self.conn.commit()

    # ─── Public Interface ─────────────────────────────────────────────────────

    async def get_history(self, session_id: str) -> list[dict]:
        """Returns the last MAX_TURNS_IN_CONTEXT turns for a session."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_history, session_id)

    async def append_user(
        self,
        session_id: str,
        prompt: str,
        intent: str,
        language: str,
    ):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._insert, session_id, "user", prompt, intent, language, {}
        )

    async def append_assistant(
        self,
        session_id: str,
        code: str,
        metadata: dict,
    ):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._insert, session_id, "assistant", code, None, None, metadata
        )

    async def list_sessions(self) -> list[dict]:
        """Returns all sessions with their last message timestamp and turn count."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_sessions)

    async def get_session(self, session_id: str) -> list[dict]:
        """Returns full history for a session (all turns)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_full_session, session_id)

    async def delete_session(self, session_id: str):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._delete_session, session_id)

    # ─── Sync Implementations ─────────────────────────────────────────────────

    def _insert(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: Optional[str],
        language: Optional[str],
        metadata: dict,
    ):
        self.conn.execute(
            """INSERT INTO chat_sessions
               (session_id, role, content, intent, language, metadata, created)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id, role, content, intent, language,
                json.dumps(metadata), datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()

    def _fetch_history(self, session_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT role, content, intent, language, metadata, created
               FROM chat_sessions
               WHERE session_id = ?
               ORDER BY id DESC
               LIMIT ?""",
            (session_id, MAX_TURNS_IN_CONTEXT * 2),  # *2 because each turn = user + assistant
        ).fetchall()

        # Reverse to chronological order
        rows = list(reversed(rows))

        return [
            {
                "role":     r[0],
                "content":  r[1],
                "intent":   r[2],
                "language": r[3],
                "metadata": json.loads(r[4]) if r[4] else {},
                "created":  r[5],
            }
            for r in rows
        ]

    def _fetch_full_session(self, session_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT role, content, intent, language, metadata, created
               FROM chat_sessions
               WHERE session_id = ?
               ORDER BY id ASC""",
            (session_id,),
        ).fetchall()
        return [
            {
                "role":     r[0],
                "content":  r[1],
                "intent":   r[2],
                "language": r[3],
                "metadata": json.loads(r[4]) if r[4] else {},
                "created":  r[5],
            }
            for r in rows
        ]

    def _fetch_sessions(self) -> list[dict]:
        rows = self.conn.execute(
            """SELECT session_id, COUNT(*) as turns,
                      MIN(created) as started, MAX(created) as last_active
               FROM chat_sessions
               GROUP BY session_id
               ORDER BY last_active DESC"""
        ).fetchall()
        return [
            {
                "session_id":   r[0],
                "turns":        r[1],
                "started":      r[2],
                "last_active":  r[3],
            }
            for r in rows
        ]

    def _delete_session(self, session_id: str):
        self.conn.execute(
            "DELETE FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        )
        self.conn.commit()