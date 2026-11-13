# Alpha AI (AVA) — Descoberta de skills
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ..agent_skills import parse_frontmatter, scan_agent_skills
from .permissions import SkillPolicy

SKILL_MD = "SKILL.md"
MAX_FILES = 10


@dataclass
class Skill:
    id: str
    name: str
    description: str
    directory: Path
    path: Path
    autoinvoke: bool = True
    version: Optional[str] = None
    files: list[str] = field(default_factory=list)


def _default_dirs() -> list[str]:
    base = Path(__file__).resolve().parents[2]
    dirs = [
        str(base / ".agents" / "skills"),
        str(base / ".agents" / "skills-created"),
    ]
    try:
        dirs.append(str(Path.home() / ".agents" / "skills"))
    except Exception:
        pass
    return dirs


def _dirs(dirs: Optional[Iterable[str]]) -> list[str]:
    if dirs is not None:
        return [str(item) for item in dirs]
    raw = os.getenv("AVA_SKILLS_DIRS", "").strip()
    if raw:
        return [item.strip() for item in raw.replace(",", os.pathsep).split(os.pathsep) if item.strip()]
    return _default_dirs()


def _list_files(directory: Path) -> list[str]:
    try:
        files = [
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file() and path.name != SKILL_MD
        ]
    except OSError:
        return []
    return sorted(files)[:MAX_FILES]


def _parse_meta(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    meta, _ = parse_frontmatter(text)
    return meta


def discover(dirs: Optional[Iterable[str]] = None) -> list[Skill]:
    found, _ = scan_agent_skills(_dirs(dirs))
    skills: list[Skill] = []
    for agent_skill in found:
        meta = _parse_meta(agent_skill.path)
        auto = str(meta.get("autoinvoke", "true")).strip().lower() not in (
            "false", "0", "no", "off",
        )
        skills.append(
            Skill(
                id=agent_skill.name,
                name=str(meta.get("name") or agent_skill.name),
                description=agent_skill.description,
                directory=agent_skill.dir,
                path=agent_skill.path,
                autoinvoke=auto,
                version=str(meta.get("version") or "") or None,
                files=_list_files(agent_skill.dir),
            )
        )
    return sorted(skills, key=lambda skill: skill.id)


def load_skills(
    dirs: Optional[Iterable[str]] = None,
    policy: Optional[SkillPolicy] = None,
) -> list[Skill]:
    policy = policy or SkillPolicy.from_env()
    return [skill for skill in discover(dirs) if policy.permits(skill.id)]
