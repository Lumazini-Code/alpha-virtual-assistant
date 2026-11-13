# Alpha AI (AVA) — Operações da tool `skill`
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from typing import Optional

from .discovery import Skill
from .loader import load_skill, load_skill_file, parse_skill_uri
from .permissions import SkillPolicy


def resolve_skill(skills: list[Skill], skill_id: str) -> Optional[Skill]:
    parsed = parse_skill_uri(skill_id)
    target = parsed[0] if parsed else skill_id
    return next((skill for skill in skills if skill.id == target), None)


def load_request(
    skills: list[Skill],
    skill_id: str,
    file: Optional[str] = None,
    *,
    policy: Optional[SkillPolicy] = None,
) -> str:
    policy = policy or SkillPolicy.from_env()
    parsed = parse_skill_uri(skill_id)
    if parsed is not None:
        skill_id, uri_path = parsed
        file = file or uri_path
    skill = resolve_skill(skills, skill_id)
    if skill is None:
        available = ", ".join(skill.id for skill in skills) or "(none installed)"
        return f"ERROR: skill '{skill_id}' is not installed. Available skills: {available}."
    if not policy.permits(skill.id):
        return f"ERROR: skill '{skill.id}' is disabled by the current policy."
    if file:
        try:
            return load_skill_file(skill, file)
        except Exception as error:
            return f"ERROR: could not read '{file}' from skill '{skill.id}': {error}"
    return load_skill(skill)
