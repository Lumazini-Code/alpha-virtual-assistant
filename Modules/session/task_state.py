# Alpha AI (AVA) — Estado explícito da tarefa atual
# Copyright (c) 2026 Felipe Lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

"""
task_state.py — Objeto explícito de "tarefa atual" por sessão.

O task state é estado de trabalho, não memória durável: fica no SessionStore
e é injetado no system prompt enquanto existir. O extrator o atualiza após
cada turno; um fallback heurístico mantém o objeto vivo se o extrator falhar.
"""

import json
import time
import uuid
from typing import Any, Optional

from .store import SessionStore

TASK_STATE_MAX_CHARS = 1200


def _now() -> float:
    return time.time()


def get_task_state(store: SessionStore, session_id: str) -> Optional[dict[str, Any]]:
    if not store or not session_id:
        return None
    row = store.db.fetchone(
        "SELECT data FROM session_task_state WHERE session_id = ?", (session_id,)
    )
    if row is None:
        return None
    try:
        data = json.loads(row["data"] or "{}")
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def set_task_state(store: SessionStore, session_id: str,
                   data: dict[str, Any]) -> dict[str, Any]:
    if not store or not session_id:
        return {}
    now = _now()
    previous = get_task_state(store, session_id) or {}
    task_id = str(data.get("task_id") or previous.get("task_id") or f"tsk_{uuid.uuid4().hex[:12]}")
    created = float(previous.get("created_at") or now)
    payload = {
        "task_id": task_id,
        "session_id": session_id,
        "goal": str(data.get("goal") or previous.get("goal") or "").strip(),
        "status": str(data.get("status") or previous.get("status") or "active").strip(),
        "decisions": _clean_list(data.get("decisions") if "decisions" in data else previous.get("decisions")),
        "open_questions": _clean_list(
            data.get("open_questions") if "open_questions" in data else previous.get("open_questions")
        ),
        "next_steps": _clean_list(
            data.get("next_steps") if "next_steps" in data else previous.get("next_steps")
        ),
        "constraints": _clean_list(
            data.get("constraints") if "constraints" in data else previous.get("constraints")
        ),
        "created_at": created,
        "updated_at": now,
    }
    store.db.execute(
        """
        INSERT INTO session_task_state (session_id, task_id, data, time_created, time_updated)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            task_id=excluded.task_id,
            data=excluded.data,
            time_updated=excluded.time_updated
        """,
        (session_id, task_id, json.dumps(payload, ensure_ascii=False), created, now),
    )
    return payload


def _clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out[:12]


def merge_task_state(store: SessionStore, session_id: str,
                     update: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Aplica a atualização do extrator preservando o que ele não repetiu."""
    if not isinstance(update, dict):
        return get_task_state(store, session_id)
    previous = get_task_state(store, session_id) or {}
    merged = {
        "goal": update.get("goal") or previous.get("goal") or "",
        "status": update.get("status") or previous.get("status") or "active",
        "decisions": _clean_list(update.get("decisions")) or _clean_list(previous.get("decisions")),
        "open_questions": _clean_list(update.get("open_questions")) or _clean_list(previous.get("open_questions")),
        "next_steps": _clean_list(update.get("next_steps")) or _clean_list(previous.get("next_steps")),
        "constraints": _clean_list(update.get("constraints")) or _clean_list(previous.get("constraints")),
    }
    return set_task_state(store, session_id, merged)


def format_task_state(data: Optional[dict[str, Any]]) -> str:
    if not isinstance(data, dict):
        return ""
    goal = str(data.get("goal") or "").strip()
    if not goal:
        return ""
    status = str(data.get("status") or "active").strip()
    lines = [f"Goal: {goal}", f"Status: {status}"]
    for label, key in (
        ("Decisions", "decisions"),
        ("Open questions", "open_questions"),
        ("Next steps", "next_steps"),
        ("Constraints", "constraints"),
    ):
        values = _clean_list(data.get(key))
        if values:
            lines.append(f"{label}:")
            lines.extend(f"- {value}" for value in values[:6])
    text = "\n".join(lines)
    if len(text) > TASK_STATE_MAX_CHARS:
        text = text[: TASK_STATE_MAX_CHARS - 1].rstrip() + "…"
    return text
