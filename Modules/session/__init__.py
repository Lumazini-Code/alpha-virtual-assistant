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
from .compaction import CompactionManager, Settings as CompactionSettings, estimate_tokens
from .events import Event
from .history import History, SessionHistory, message_text, sanitize_for_model
from .ids import new_message_id, new_part_id, new_session_id, now
from .projector import Projector
from .store import SessionStore

__all__ = [
    "CompactionManager",
    "CompactionSettings",
    "Event",
    "History",
    "Projector",
    "SessionDatabase",
    "SessionHistory",
    "SessionStore",
    "default_db_path",
    "estimate_tokens",
    "message_text",
    "new_message_id",
    "new_part_id",
    "new_session_id",
    "now",
    "sanitize_for_model",
]
