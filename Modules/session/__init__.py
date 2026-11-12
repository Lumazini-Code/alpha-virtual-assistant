# Alpha AI (AVA) — Session store (fundação durável)
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Session store durável do AVA.

Este pacote implementa um log de eventos e um store de sessões/mensagens
inspirado no OpenCode. Ele ainda não substitui o ST memory como fonte de
leitura; o orchestrator grava nos dois (dual-write) enquanto a migração é
validada.
"""

from .db import SessionDatabase, default_db_path
from .events import Event
from .history import History, SessionHistory
from .ids import new_message_id, new_part_id, new_session_id, now
from .projector import Projector
from .store import SessionStore

__all__ = [
    "Event",
    "History",
    "Projector",
    "SessionDatabase",
    "SessionHistory",
    "SessionStore",
    "default_db_path",
    "new_message_id",
    "new_part_id",
    "new_session_id",
    "now",
]
