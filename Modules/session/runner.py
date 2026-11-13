# Alpha AI (AVA) — Política do loop agentico por steps
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

MAX_OUTPUT_CONTINUATIONS = 2
CONTINUE_AFTER_OUTPUT_LIMIT = (
    "Your last response hit the output token limit. Do not apologize, recap, or "
    "repeat yourself. Break the remaining work into smaller pieces and continue."
)


class OutcomeKind(str, Enum):
    COMPLETED = "completed"
    OUTPUT_LIMIT = "output_limit"
    RETRY = "retry"
    COMPACTED = "compacted"
    INTERRUPTED = "interrupted"


@dataclass
class Outcome:
    kind: OutcomeKind
    needs_continuation: bool = False
    error: Optional[str] = None
    fresh_assistant: bool = False


def finish_reason(message: dict[str, Any]) -> Optional[str]:
    value = message.get("_finish_reason") or message.get("finish_reason")
    return str(value) if value is not None else None


def is_output_limit(message: dict[str, Any]) -> bool:
    return finish_reason(message) == "length"


def is_stream_interrupted(message: dict[str, Any]) -> bool:
    return bool(message.get("_interrupted"))


class OutputLimitTracker:
    """Até duas continuações por sequência de respostas truncadas."""

    def __init__(self, limit: int = MAX_OUTPUT_CONTINUATIONS):
        self.limit = limit
        self.count = 0

    def record(self, message: dict[str, Any]) -> bool:
        """Retorna True se ainda pode continuar; False se esgotou."""
        if not is_output_limit(message):
            self.count = 0
            return True
        self.count += 1
        return self.count <= self.limit

    @property
    def exhausted(self) -> bool:
        return self.count > self.limit

    def reset(self) -> None:
        self.count = 0


def completed(needs_continuation: bool) -> Outcome:
    return Outcome(OutcomeKind.COMPLETED, needs_continuation=needs_continuation)


def output_limit(needs_continuation: bool) -> Outcome:
    return Outcome(OutcomeKind.OUTPUT_LIMIT, needs_continuation=needs_continuation)


def retry(error: str, fresh_assistant: bool = False) -> Outcome:
    return Outcome(OutcomeKind.RETRY, error=error, fresh_assistant=fresh_assistant)


def compacted() -> Outcome:
    return Outcome(OutcomeKind.COMPACTED)


def interrupted(error: Optional[str] = None) -> Outcome:
    return Outcome(OutcomeKind.INTERRUPTED, error=error)
