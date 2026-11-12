# Alpha AI (AVA) — Compactação de contexto
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from .history import message_text
from .store import SessionStore

SUMMARY_SYSTEM_PROMPT = """You compact a conversation so another assistant can continue the work without losing context.
Return only a Markdown summary with these sections, in this order:

## Goal
## Work State
## Decisions
## Files and Artifacts
## Next Steps
## Open Questions

Rules:
- Preserve exact file paths, symbols, commands, error strings, URLs, identifiers, and numbers.
- Carry forward only user requests that remain unanswered or need action. Do not repeat resolved ones.
- Preserve consequential workflow state (uncommitted/committed/pushed/reviewed/merged).
- When information is uncertain, say so briefly instead of inventing.
- Do not mention the compaction process or that context was compacted.
- Do not include a preamble or commentary outside the sections.
"""


def estimate_tokens(text: str) -> int:
    """Estimativa conservadora: ~4 caracteres por token."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _entry_text(entry: dict[str, Any]) -> str:
    message = entry.get("message") or {}
    text = message_text(message.get("content"))
    calls = message.get("tool_calls")
    if calls:
        text += "\n" + json.dumps(calls, ensure_ascii=False, default=str)
    return text


@dataclass
class Settings:
    auto: bool = True
    context_tokens: int = 128_000
    buffer: float = 0.15
    keep_tokens: int = 15_000
    input_max_chars: int = 120_000
    message_max_chars: int = 4_000

    @classmethod
    def from_env(cls) -> "Settings":
        def _int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except ValueError:
                return default

        def _float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except ValueError:
                return default

        return cls(
            auto=os.getenv("AVA_COMPACTION", "1").strip().lower() in ("1", "true", "yes", "on"),
            context_tokens=_int("AVA_COMPACTION_CONTEXT_TOKENS", 128_000),
            buffer=max(0.0, min(0.9, _float("AVA_COMPACTION_BUFFER", 0.15))),
            keep_tokens=max(1_000, _int("AVA_COMPACTION_KEEP_TOKENS", 15_000)),
            input_max_chars=max(10_000, _int("AVA_COMPACTION_INPUT_MAX_CHARS", 120_000)),
            message_max_chars=max(500, _int("AVA_COMPACTION_MESSAGE_MAX_CHARS", 4_000)),
        )


class CompactionManager:
    """Decide quando compactar, resume e grava o checkpoint no store."""

    def __init__(self, store: SessionStore, settings: Optional[Settings] = None):
        self.store = store
        self.settings = settings or Settings.from_env()

    @property
    def threshold_tokens(self) -> int:
        return int(self.settings.context_tokens * (1.0 - self.settings.buffer))

    def should_compact(self, total_tokens: int) -> bool:
        return self.settings.auto and total_tokens >= self.threshold_tokens

    def estimate_entries(self, entries: list[dict[str, Any]]) -> int:
        return sum(estimate_tokens(_entry_text(entry)) for entry in entries)

    def keep_index(self, entries: list[dict[str, Any]]) -> int:
        """Índice da primeira mensagem que fica verbatim.

        A cauda começa na mensagem mais antiga que ainda cabe no orçamento de
        ``keep_tokens`` e é ajustada para começar em um turno do usuário.
        """
        if not entries:
            return 0
        budget = self.settings.keep_tokens
        used = 0
        index = len(entries)
        for position in range(len(entries) - 1, -1, -1):
            used += estimate_tokens(_entry_text(entries[position]))
            index = position
            if used >= budget and (len(entries) - position) >= 2:
                break
        while index < len(entries) and entries[index].get("message", {}).get("role") != "user":
            index += 1
        return index

    async def compact(
        self,
        session_id: str,
        entries: list[dict[str, Any]],
        summarize: Callable[[str], Awaitable[str]],
        *,
        reason: str = "auto",
        model: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        if not entries:
            return None
        keep_index = self.keep_index(entries)
        if keep_index <= 0:
            return None

        transcript = self._render(entries[:keep_index])
        if not transcript.strip():
            return None

        summary = await summarize(
            f"{SUMMARY_SYSTEM_PROMPT}\n\n## Conversation transcript\n\n{transcript}"
        )
        summary = (summary or "").strip()
        if not summary:
            return None

        keep_id = entries[keep_index]["id"]
        self.store.append_compaction(
            session_id,
            text=summary,
            keep_id=keep_id,
            reason=reason,
            model=model,
        )
        return {
            "text": summary,
            "keepID": keep_id,
            "summarized": keep_index,
            "reason": reason,
        }

    def _render(self, entries: list[dict[str, Any]]) -> str:
        """Transcreve as mensagens mais recentes que cabem no orçamento.

        A compactação precisa cobrir o passado, mas um transcript gigante
        estouraria o próprio request de resumo. Mantém as mensagens mais
        recentes e sinaliza o corte no início.
        """
        budget = self.settings.input_max_chars
        blocks: list[str] = []
        omitted = False
        for entry in reversed(entries):
            message = entry.get("message") or {}
            role = str(message.get("role") or entry.get("type") or "message").upper()
            text = _entry_text(entry)
            if len(text) > self.settings.message_max_chars:
                text = text[: self.settings.message_max_chars] + "\n[…truncated…]"
            block = f"{role}: {text}\n\n"
            if len(block) > budget:
                omitted = True
                break
            blocks.append(block)
            budget -= len(block)
        body = "".join(reversed(blocks))
        return ("[earlier messages omitted]\n\n" if omitted else "") + body
