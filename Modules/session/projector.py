# Alpha AI (AVA) — Projeção de eventos em linhas duráveis
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import sqlite3
from typing import Any, Optional

from .db import SessionDatabase, canonical_json
from .events import Event
from .ids import new_message_id, now


def normalize_message(message: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Normaliza uma mensagem OpenAI-like para (type, data).

    O store guarda o formato que o modelo vê, para que a leitura seja uma
    tradução direta. Campos desconhecidos são preservados em ``data``.
    """
    data = dict(message)
    message_id = data.pop("id", None)
    explicit_type = data.pop("type", None)
    role = str(data.get("role") or explicit_type or "system")
    data["role"] = role
    if "content" not in data:
        data["content"] = ""
    if data["content"] is None:
        data["content"] = ""
    if message_id is not None:
        data["_message_id"] = message_id
    return str(explicit_type or role), data


class Projector:
    """Aplica eventos no SQLite.

    A escrita de mensagens é serializada por sessão pelo lock do banco. O
    projector é deliberadamente simples: a lógica de negócio (compactação,
    instruções) vive nos módulos que publicam os eventos.
    """

    def __init__(self, db: SessionDatabase):
        self.db = db

    # ── API principal ─────────────────────────────────────────────────────
    def apply(self, event: Event) -> None:
        if event.type == "session.created":
            self._create_session(event.session_id, event.data, event.time)
            return
        if event.type == "session.message.appended":
            self.append_message(event.session_id, event.data.get("message") or {})
            return
        if event.type == "session.usage.recorded":
            self._record_usage(event.session_id, event.data.get("usage") or {}, event.time)
            return
        # Eventos de step/tool/compaction/instruções não viram mensagem por si
        # só; apenas mantêm o timestamp da sessão vivo.
        self._touch(event.session_id, event.time)

    # ── mensagens ─────────────────────────────────────────────────────────
    def append_message(
        self,
        session_id: str,
        message: dict[str, Any],
        *,
        conn: Optional[sqlite3.Connection] = None,
    ) -> str:
        if conn is not None:
            return self._insert_message(conn, session_id, message)
        with self.db.transaction() as tx:
            return self._insert_message(tx, session_id, message)

    def _insert_message(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        message: dict[str, Any],
    ) -> str:
        message_type, data = normalize_message(message)
        message_id = str(data.pop("_message_id", "") or new_message_id())
        existing = conn.execute(
            "SELECT id FROM session_message WHERE id = ?",
            (message_id,),
        ).fetchone()
        if existing is not None:
            return message_id

        row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM session_message WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        seq = int(row["next_seq"]) if row is not None else 1
        created = now()
        conn.execute(
            "INSERT INTO session_message (id, session_id, seq, type, data, time_created) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (message_id, session_id, seq, message_type, canonical_json(data), created),
        )
        conn.execute(
            "UPDATE session SET time_updated = ? WHERE id = ?",
            (created, session_id),
        )
        return message_id

    # ── sessão ────────────────────────────────────────────────────────────
    def _create_session(self, session_id: str, data: dict[str, Any], created: float) -> None:
        with self.db.transaction() as tx:
            tx.execute(
                "INSERT INTO session "
                "(id, parent_id, project_id, person_id, agent, model, title, time_created, time_updated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "  parent_id = COALESCE(excluded.parent_id, session.parent_id), "
                "  project_id = COALESCE(excluded.project_id, session.project_id), "
                "  person_id = COALESCE(excluded.person_id, session.person_id), "
                "  agent = COALESCE(excluded.agent, session.agent), "
                "  model = COALESCE(excluded.model, session.model), "
                "  title = COALESCE(excluded.title, session.title), "
                "  time_updated = excluded.time_updated",
                (
                    session_id,
                    data.get("parent_id"),
                    data.get("project_id"),
                    data.get("person_id"),
                    data.get("agent"),
                    data.get("model"),
                    data.get("title"),
                    created,
                    created,
                ),
            )

    def _touch(self, session_id: str, at: float) -> None:
        with self.db.transaction() as tx:
            tx.execute(
                "UPDATE session SET time_updated = ? WHERE id = ?",
                (at, session_id),
            )

    # ── usage ─────────────────────────────────────────────────────────────
    def _record_usage(self, session_id: str, usage: dict[str, Any], at: float) -> None:
        with self.db.transaction() as tx:
            tx.execute(
                "INSERT INTO session_usage "
                "(session_id, message_id, input_tokens, output_tokens, cache_read, cache_write, cost, time_created) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    usage.get("message_id"),
                    usage.get("input_tokens"),
                    usage.get("output_tokens"),
                    usage.get("cache_read"),
                    usage.get("cache_write"),
                    usage.get("cost"),
                    at,
                ),
            )
