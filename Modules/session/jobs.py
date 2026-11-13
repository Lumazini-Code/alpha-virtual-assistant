# Alpha AI (AVA) — Jobs em background
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

from .store import SessionStore


def _now() -> float:
    return time.time()


def new_job_id() -> str:
    return f"job_{uuid.uuid4().hex}"


class JobStore:
    """Tabela `session_job`: subagentes e outros trabalhos em background."""

    def __init__(self, store: SessionStore):
        self.store = store
        self.db = store.db

    def create(
        self,
        *,
        parent_id: str,
        child_id: Optional[str],
        kind: str,
        description: Optional[str] = None,
    ) -> str:
        job_id = new_job_id()
        now = _now()
        with self.db.transaction() as tx:
            tx.execute(
                "INSERT INTO session_job "
                "(id, parent_id, child_id, kind, status, description, time_created, time_updated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (job_id, parent_id, child_id, kind, "running", description, now, now),
            )
        return job_id

    def update(self, job_id: str, status: str, error: Optional[str] = None) -> None:
        with self.db.transaction() as tx:
            tx.execute(
                "UPDATE session_job SET status = ?, error = ?, time_updated = ? WHERE id = ?",
                (status, error, _now(), job_id),
            )

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        row = self.db.fetchone("SELECT * FROM session_job WHERE id = ?", (job_id,))
        return dict(row) if row is not None else None

    def list(
        self,
        *,
        parent_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if parent_id is not None:
            clauses.append("parent_id = ?")
            params.append(parent_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.fetchall(
            f"SELECT * FROM session_job {where} ORDER BY time_created DESC LIMIT ?",
            (*params, limit),
        )
        return [dict(row) for row in rows]

    def active_count(self, parent_id: str) -> int:
        row = self.db.fetchone(
            "SELECT COUNT(*) AS count FROM session_job WHERE parent_id = ? AND status = 'running'",
            (parent_id,),
        )
        return int(row["count"]) if row is not None else 0

    def recover_orphans(self) -> int:
        """Processos que morreram deixam jobs `running`; marca como interrupted."""
        with self.db.transaction() as tx:
            cursor = tx.execute(
                "UPDATE session_job SET status = 'interrupted', error = ?, time_updated = ? "
                "WHERE status = 'running'",
                ("process restarted before the job finished", _now()),
            )
            return cursor.rowcount or 0

    def as_json(self, job: dict[str, Any]) -> str:
        return json.dumps(job, ensure_ascii=False, default=str)
