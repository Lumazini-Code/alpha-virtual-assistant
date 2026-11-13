# Alpha AI (AVA) — Política de skills
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
    return tuple(item for item in raw.replace(",", " ").split() if item)


@dataclass(frozen=True)
class SkillPolicy:
    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "SkillPolicy":
        return cls(allow=_split("AVA_SKILL_ALLOW"), deny=_split("AVA_SKILL_DENY"))

    def permits(self, skill_id: str) -> bool:
        if any(fnmatch.fnmatchcase(skill_id, pattern) for pattern in self.deny):
            return False
        if self.allow and not any(fnmatch.fnmatchcase(skill_id, pattern) for pattern in self.allow):
            return False
        return True
