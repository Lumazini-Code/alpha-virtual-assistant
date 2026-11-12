# Alpha AI (AVA) — Eventos duráveis da sessão
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ids import now


@dataclass(frozen=True)
class Event:
    """Um fato novo e imutável sobre uma sessão.

    O projector traduz o evento para linhas duráveis. Nem todo evento vira
    mensagem: ``step.*`` e ``tool.*`` podem apenas atualizar metadados/usage.
    """

    type: str
    session_id: str
    data: dict[str, Any] = field(default_factory=dict)
    time: float = field(default_factory=now)


# ── fábricas (mantêm os nomes de evento em um só lugar) ───────────────────
def session_created(session_id: str, **data: Any) -> Event:
    return Event("session.created", session_id, data)


def message_appended(session_id: str, message: dict[str, Any]) -> Event:
    return Event("session.message.appended", session_id, {"message": message})


def usage_recorded(session_id: str, usage: dict[str, Any]) -> Event:
    return Event("session.usage.recorded", session_id, {"usage": usage})


def step_started(session_id: str, **data: Any) -> Event:
    return Event("session.step.started", session_id, data)


def step_ended(session_id: str, **data: Any) -> Event:
    return Event("session.step.ended", session_id, data)


def tool_started(session_id: str, **data: Any) -> Event:
    return Event("session.tool.started", session_id, data)


def tool_ended(session_id: str, **data: Any) -> Event:
    return Event("session.tool.ended", session_id, data)


def compaction_started(session_id: str, **data: Any) -> Event:
    return Event("session.compaction.started", session_id, data)


def compaction_completed(session_id: str, **data: Any) -> Event:
    return Event("session.compaction.completed", session_id, data)


def compaction_failed(session_id: str, **data: Any) -> Event:
    return Event("session.compaction.failed", session_id, data)


def instructions_updated(session_id: str, **data: Any) -> Event:
    return Event("session.instructions.updated", session_id, data)


def execution_started(session_id: str, **data: Any) -> Event:
    return Event("session.execution.started", session_id, data)


def execution_succeeded(session_id: str, **data: Any) -> Event:
    return Event("session.execution.succeeded", session_id, data)


def execution_failed(session_id: str, **data: Any) -> Event:
    return Event("session.execution.failed", session_id, data)


def execution_interrupted(session_id: str, **data: Any) -> Event:
    return Event("session.execution.interrupted", session_id, data)
