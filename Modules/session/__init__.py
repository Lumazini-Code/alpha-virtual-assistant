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
from .runner import (
    CONTINUE_AFTER_OUTPUT_LIMIT,
    MAX_OUTPUT_CONTINUATIONS,
    Outcome,
    OutcomeKind,
    OutputLimitTracker,
    completed,
    finish_reason,
    interrupted,
    is_output_limit,
    output_limit,
    retry,
)
from .store import SessionStore

__all__ = [
    "CONTINUE_AFTER_OUTPUT_LIMIT",
    "MAX_OUTPUT_CONTINUATIONS",
    "CompactionManager",
    "CompactionSettings",
    "Event",
    "History",
    "Outcome",
    "OutcomeKind",
    "OutputLimitTracker",
    "Projector",
    "SessionDatabase",
    "SessionHistory",
    "SessionStore",
    "completed",
    "default_db_path",
    "estimate_tokens",
    "finish_reason",
    "interrupted",
    "is_output_limit",
    "message_text",
    "new_message_id",
    "new_part_id",
    "new_session_id",
    "now",
    "output_limit",
    "retry",
    "sanitize_for_model",
]
