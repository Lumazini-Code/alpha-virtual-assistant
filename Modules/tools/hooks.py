# Alpha AI (AVA) — Hooks de execução de tools
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from .registry import ToolContext, ToolResult

BeforeHook = Callable[[str, dict[str, Any], "ToolContext"], Awaitable[dict[str, Any]]]
AfterHook = Callable[[str, "ToolResult", "ToolContext"], Awaitable["ToolResult"]]


@dataclass
class ToolHooks:
    """Hooks before/after, na ordem de registro.

    ``before`` pode reescrever os argumentos; ``after`` pode reescrever o
    resultado. Hooks, como no OpenCode, não podem transformar um erro em
    sucesso silenciosamente — quem decide o status é a execução.
    """

    before: list[BeforeHook] = field(default_factory=list)
    after: list[AfterHook] = field(default_factory=list)

    async def run_before(self, name: str, args: dict[str, Any], context: "ToolContext") -> dict[str, Any]:
        current = args
        for hook in self.before:
            current = await hook(name, current, context)
        return current

    async def run_after(self, name: str, result: "ToolResult", context: "ToolContext") -> "ToolResult":
        current = result
        for hook in self.after:
            current = await hook(name, current, context)
        return current
