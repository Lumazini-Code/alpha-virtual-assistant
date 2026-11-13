# Alpha AI (AVA) — Store de sessões e mensagens
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .db import SessionDatabase
from .events import session_created
from .ids import new_session_id, now
from .projector import Projector


class SessionStore:
    """Fonte durável de sessões, mensagens e usage.

    O store é a base do novo histórico e é usado em dual-write: o
    orchestrator continua lendo o ST antigo, mas grava aqui quando
    ``AVA_SESSION_STORE`` está ligado (padrão: ligado).
    """

    def __init__(self, path: Optional[str | Path] = None, db: Optional[SessionDatabase] = None):
        self.db = db or SessionDatabase(path)
        self.projector = Projector(self.db)

    # ── sessão ────────────────────────────────────────────────────────────
    def create(
        self,
        *,
        session_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        project_id: Optional[str] = None,
        person_id: Optional[int] = None,
        agent: Optional[str] = None,
        model: Optional[str] = None,
        title: Optional[str] = None,
    ) -> str:
        sid = session_id or new_session_id()
        self.projector.apply(
            session_created(
                sid,
                parent_id=parent_id,
                project_id=project_id,
                person_id=person_id,
                agent=agent,
                model=model,
                title=title,
            )
        )
        return sid

    def get(self, session_id: str) -> Optional[dict[str, Any]]:
        row = self.db.fetchone("SELECT * FROM session WHERE id = ?", (session_id,))
        return dict(row) if row is not None else None

    def list_sessions(
        self,
        limit: int = 50,
        *,
        root_only: bool = False,
        project: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Sessões mais recentes primeiro.

        `root_only` esconde sessões filhas (subagentes) — é o que a lista de
        chats do WebUI mostra. `project` filtra pelo escopo de memória.
        """
        sql = "SELECT * FROM session WHERE 1=1"
        params: list[Any] = []
        if root_only:
            sql += " AND parent_id IS NULL"
        if project:
            sql += " AND project_id = ?"
            params.append(project)
        sql += " ORDER BY time_updated DESC LIMIT ?"
        params.append(limit)
        rows = self.db.fetchall(sql, tuple(params))
        return [dict(row) for row in rows]

    # ── mensagens ─────────────────────────────────────────────────────────
    def append_message(self, session_id: str, message: dict[str, Any]) -> str:
        if not session_id:
            session_id = self.create()
        elif self.get(session_id) is None:
            self.create(session_id=session_id)
        return self.projector.append_message(session_id, message)

    def append_turn(
        self,
        session_id: str,
        user: str | list[dict[str, Any]],
        assistant: str | list[dict[str, Any]],
        *,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        tool_results: Optional[list[dict[str, Any]]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Grava um turno completo em uma única transação.

        ``tool_calls``/``tool_results`` são opcionais e servem para o caso de
        um turno que terminou com tool loop. O loop normal grava cada mensagem
        durante a execução; este método é o fallback atômico.
        """
        if not session_id:
            session_id = self.create()
        elif self.get(session_id) is None:
            self.create(session_id=session_id)
        with self.db.transaction() as tx:
            self.projector.append_message(
                session_id,
                {"role": "user", "content": user, "metadata": metadata or {}},
                conn=tx,
            )
            assistant_message: dict[str, Any] = {"role": "assistant", "content": assistant}
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            if metadata:
                assistant_message["metadata"] = metadata
            self.projector.append_message(session_id, assistant_message, conn=tx)
            for result in tool_results or []:
                self.projector.append_message(session_id, result, conn=tx)

    def last_message(self, session_id: str) -> Optional[dict[str, Any]]:
        row = self.db.fetchone(
            "SELECT * FROM session_message WHERE session_id = ? ORDER BY seq DESC LIMIT 1",
            (session_id,),
        )
        return self._row_to_message(row) if row is not None else None

    def load_messages(self, session_id: str, *, boundary: str = "latest") -> list[dict[str, Any]]:
        """Mensagens na ordem do modelo, a partir da última compactação.

        A compactação é reexpandida como ``[resumo, ...cauda mantida, ...]``:
        o ``metadata.keepID`` aponta a primeira mensagem que continua verbatim.
        ``boundary="local"`` (reservado) ignora compactações nativas.
        """
        return [entry["message"] for entry in self.entries(session_id, boundary=boundary)]

    def entries(self, session_id: str, *, boundary: str = "latest") -> list[dict[str, Any]]:
        """Mensagens com id/seq, já reexpandidas a partir da compactação.

        Formato de cada item: ``{"id", "seq", "type", "data", "message"}``.
        """
        compaction = self._latest_compaction_row(session_id)
        if compaction is None:
            rows = self.db.fetchall(
                "SELECT * FROM session_message WHERE session_id = ? ORDER BY seq ASC",
                (session_id,),
            )
            return [self._entry(row) for row in rows]

        compaction_entry = self._entry(compaction)
        data = json.loads(compaction["data"])
        keep_id = (data.get("metadata") or {}).get("keepID")
        kept: list[dict[str, Any]] = []
        if keep_id:
            keep_row = self.db.fetchone(
                "SELECT seq FROM session_message WHERE session_id = ? AND id = ?",
                (session_id, keep_id),
            )
            keep_seq = int(keep_row["seq"]) if keep_row is not None else int(compaction["seq"])
            kept_rows = self.db.fetchall(
                "SELECT * FROM session_message "
                "WHERE session_id = ? AND seq >= ? AND seq < ? AND type != 'compaction' "
                "ORDER BY seq ASC",
                (session_id, keep_seq, compaction["seq"]),
            )
            kept = [self._entry(row) for row in kept_rows]
        after_rows = self.db.fetchall(
            "SELECT * FROM session_message WHERE session_id = ? AND seq > ? ORDER BY seq ASC",
            (session_id, compaction["seq"]),
        )
        return [compaction_entry, *kept, *[self._entry(row) for row in after_rows]]

    def append_compaction(
        self,
        session_id: str,
        *,
        text: str,
        keep_id: Optional[str] = None,
        reason: str = "auto",
        model: Optional[str] = None,
        usage: Optional[dict[str, Any]] = None,
        status: str = "completed",
    ) -> str:
        message: dict[str, Any] = {
            "type": "compaction",
            "status": status,
            "reason": reason,
            "text": text,
            "metadata": {"keepID": keep_id} if keep_id else {},
        }
        if model:
            message["model"] = model
        if usage:
            message["usage"] = usage
        return self.append_message(session_id, message)

    def latest_compaction(self, session_id: str) -> Optional[dict[str, Any]]:
        row = self._latest_compaction_row(session_id)
        if row is None:
            return None
        return {
            "id": row["id"],
            "seq": row["seq"],
            "message": self._row_to_message(row),
            "data": json.loads(row["data"]),
        }

    def _latest_compaction_row(self, session_id: str) -> Any:
        return self.db.fetchone(
            "SELECT * FROM session_message "
            "WHERE session_id = ? AND type = 'compaction' "
            "AND json_extract(data, '$.status') = 'completed' "
            "ORDER BY seq DESC LIMIT 1",
            (session_id,),
        )

    def _entry(self, row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "seq": row["seq"],
            "type": row["type"],
            "data": json.loads(row["data"]),
            "message": self._row_to_message(row),
        }

    def _row_to_message(self, row: Any) -> dict[str, Any]:
        data = json.loads(row["data"])
        message_type = row["type"]
        if message_type == "compaction":
            return {
                "role": "system",
                "content": data.get("text", ""),
                "type": "compaction",
                "metadata": data.get("metadata") or {},
            }
        if message_type == "synthetic":
            return {
                "role": data.get("role") or "system",
                "content": data.get("text") or data.get("content") or "",
                "type": "synthetic",
                "metadata": data.get("metadata") or {},
            }
        return data

    # ── usage ─────────────────────────────────────────────────────────────
    def record_usage(self, session_id: str, usage: dict[str, Any]) -> None:
        from .events import usage_recorded

        self.projector.apply(usage_recorded(session_id, usage))

    def usage(self, session_id: str) -> dict[str, Any]:
        row = self.db.fetchone(
            "SELECT "
            "  COALESCE(SUM(input_tokens), 0) AS input_tokens, "
            "  COALESCE(SUM(output_tokens), 0) AS output_tokens, "
            "  COALESCE(SUM(cache_read), 0) AS cache_read, "
            "  COALESCE(SUM(cache_write), 0) AS cache_write, "
            "  COALESCE(SUM(cost), 0) AS cost "
            "FROM session_usage WHERE session_id = ?",
            (session_id,),
        )
        return dict(row) if row is not None else {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read": 0,
            "cache_write": 0,
            "cost": 0,
        }

    def usage_total(self) -> dict[str, Any]:
        row = self.db.fetchone(
            "SELECT "
            "  COALESCE(SUM(input_tokens), 0) AS input_tokens, "
            "  COALESCE(SUM(output_tokens), 0) AS output_tokens, "
            "  COALESCE(SUM(cache_read), 0) AS cache_read, "
            "  COALESCE(SUM(cache_write), 0) AS cache_write, "
            "  COALESCE(SUM(cost), 0) AS cost "
            "FROM session_usage",
        )
        return dict(row) if row is not None else {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read": 0,
            "cache_write": 0,
            "cost": 0,
        }

    # ── manutenção ────────────────────────────────────────────────────────
    def clear(self, session_id: str) -> None:
        with self.db.transaction() as tx:
            tx.execute("DELETE FROM session_message WHERE session_id = ?", (session_id,))
            tx.execute("DELETE FROM session_usage WHERE session_id = ?", (session_id,))
            tx.execute("DELETE FROM instruction_state WHERE session_id = ?", (session_id,))
            tx.execute("UPDATE session SET time_updated = ? WHERE id = ?", (now(), session_id))

    def has_source(self, session_id: str, source_id: int) -> bool:
        """Idempotência da migração do ST (``metadata.st_group_id``)."""
        row = self.db.fetchone(
            "SELECT 1 FROM session_message "
            "WHERE session_id = ? AND json_extract(data, '$.metadata.st_group_id') = ? LIMIT 1",
            (session_id, source_id),
        )
        return row is not None

    def close(self) -> None:
        self.db.close()
