# Alpha AI (AVA) — SQLite do session store
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "db" / "ava_sessions.db"


def default_db_path() -> Path:
    """Caminho do banco de sessões.

    Separado do banco de memória de propósito: migrações de sessão não podem
    arriscar o grafo/LanceDB. ``AVA_SESSION_DB_PATH`` permite apontar para
    outro volume em produção.
    """
    raw = os.getenv("AVA_SESSION_DB_PATH", "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_DB_PATH


def canonical_json(value: Any) -> str:
    """JSON canônico para hash/armazenamento determinístico."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class SessionDatabase:
    """Conexão SQLite única, serializada por um RLock.

    O volume de escrita é baixo (uma mensagem por step/tool), então um lock
    global é suficiente e evita surpresas de concorrência entre threads do
    uvicorn. Um lock por sessão pode ser adicionado se o volume crescer.
    """

    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path).expanduser() if path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; transações explícitas abaixo
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._lock = threading.RLock()
        self._migrate()

    # ── schema ────────────────────────────────────────────────────────────
    def _migrate(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS session (
                    id            TEXT PRIMARY KEY,
                    parent_id     TEXT,
                    project_id    TEXT,
                    person_id     INTEGER,
                    agent         TEXT,
                    model         TEXT,
                    title         TEXT,
                    time_created  REAL NOT NULL,
                    time_updated  REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_message (
                    id            TEXT PRIMARY KEY,
                    session_id    TEXT NOT NULL,
                    seq           INTEGER NOT NULL,
                    type          TEXT NOT NULL,
                    data          TEXT NOT NULL,
                    time_created  REAL NOT NULL,
                    UNIQUE(session_id, seq)
                );
                CREATE INDEX IF NOT EXISTS idx_msg_session_seq
                    ON session_message(session_id, seq);
                CREATE INDEX IF NOT EXISTS idx_msg_session_type
                    ON session_message(session_id, type);

                CREATE TABLE IF NOT EXISTS session_part (
                    message_id    TEXT NOT NULL,
                    ordinal       INTEGER NOT NULL,
                    type          TEXT NOT NULL,
                    data          TEXT NOT NULL,
                    PRIMARY KEY(message_id, ordinal)
                );

                CREATE TABLE IF NOT EXISTS session_blob (
                    hash          TEXT PRIMARY KEY,
                    data          BLOB NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_usage (
                    session_id    TEXT NOT NULL,
                    message_id    TEXT,
                    input_tokens  INTEGER,
                    output_tokens INTEGER,
                    cache_read    INTEGER,
                    cache_write   INTEGER,
                    cost          REAL,
                    time_created  REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_usage_session
                    ON session_usage(session_id, time_created);

                CREATE TABLE IF NOT EXISTS instruction_state (
                    session_id     TEXT PRIMARY KEY,
                    baseline_text  TEXT NOT NULL,
                    values_json    TEXT NOT NULL,
                    deltas_json    TEXT NOT NULL,
                    time_updated   REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS instruction_blob (
                    hash          TEXT PRIMARY KEY,
                    value         TEXT NOT NULL
                );
                """
            )

    # ── transações ────────────────────────────────────────────────────────
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    # ── consultas ─────────────────────────────────────────────────────────
    def execute(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def fetchone(self, sql: str, params: tuple | list = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
