# Alpha AI (AVA) — Leitura de histórico para o runner
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .store import SessionStore


@dataclass
class History:
    """O que o runner precisa para montar um request.

    ``initial`` será preenchido pelo sistema de instruções versionadas. Nesta
    fase fica vazio e o orchestrator continua usando o system prompt atual.
    """

    initial: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)


class SessionHistory:
    def __init__(self, store: SessionStore):
        self.store = store

    def load(self, session_id: str, *, boundary: str = "latest") -> History:
        return History(initial=[], messages=self.store.load_messages(session_id, boundary=boundary))
