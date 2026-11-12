# Alpha AI (AVA) — Política de permissões de tools
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass


def _split(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return ()
    return tuple(
        item.strip()
        for item in raw.replace(",", " ").split()
        if item.strip()
    )


@dataclass(frozen=True)
class ToolPolicy:
    """Regras allow/deny por nome de tool (aceita glob)."""

    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "ToolPolicy":
        return cls(allow=_split("AVA_TOOL_ALLOW"), deny=_split("AVA_TOOL_DENY"))

    def permits(self, name: str) -> bool:
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in self.deny):
            return False
        if self.allow and not any(fnmatch.fnmatchcase(name, pattern) for pattern in self.allow):
            return False
        return True

    def reason(self, name: str) -> str:
        return f"tool '{name}' is disabled by the current tool policy"
