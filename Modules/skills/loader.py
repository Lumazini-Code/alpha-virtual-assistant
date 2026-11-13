# Alpha AI (AVA) — Carregamento de skills e arquivos
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..agent_skills import read_skill_body, resolve_skill_file
from .discovery import Skill

SKILL_BODY_MAX_CHARS = 64_000
SKILL_FILE_MAX_CHARS = 32_000


def skill_uri(skill_id: str, path: Optional[str] = None) -> str:
    return f"skill://{skill_id}/{path}" if path else f"skill://{skill_id}"


def parse_skill_uri(value: str) -> Optional[tuple[str, Optional[str]]]:
    if not isinstance(value, str) or not value.startswith("skill://"):
        return None
    rest = value[len("skill://"):]
    if not rest:
        return None
    if "/" in rest:
        skill_id, path = rest.split("/", 1)
        return (skill_id, path or None) if skill_id else None
    return rest, None


def load_skill(skill: Skill, max_chars: int = SKILL_BODY_MAX_CHARS) -> str:
    body = read_skill_body(skill.path, max_chars)
    files = skill.files[:10]
    return "\n".join([
        f'<skill_content name="{skill.name}">',
        f"# Skill: {skill.name}",
        "",
        body,
        "",
        f"Base directory for this skill: {skill_uri(skill.id)}",
        "Relative paths in this skill (e.g., scripts/, reference/) are relative to this base directory.",
        "Use the skill_file tool (or skill with file=...) to read a listed file.",
        "",
        "<skill_files>",
        *[f"  <file>{skill_uri(skill.id, file)}</file>" for file in files],
        "</skill_files>",
        "</skill_content>",
    ])


def load_skill_file(skill: Skill, path: str, max_chars: int = SKILL_FILE_MAX_CHARS) -> str:
    return resolve_skill_file(skill.directory, path, max_chars)


def load_uri(skills: list[Skill], uri: str) -> Optional[str]:
    parsed = parse_skill_uri(uri)
    if parsed is None:
        return None
    skill_id, path = parsed
    skill = next((item for item in skills if item.id == skill_id), None)
    if skill is None:
        return None
    return load_skill_file(skill, path) if path else load_skill(skill)
