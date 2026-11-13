# Alpha AI (AVA) — Catálogo de skills como instrução
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import html
from typing import Any

from ..instructions.source import REMOVED, Source
from .discovery import Skill


def entries(skills: list[Skill]) -> list[dict[str, str]]:
    return [
        {"id": skill.id, "name": skill.name, "description": skill.description}
        for skill in skills
        if skill.description and skill.autoinvoke
    ]


def _render_items(items: list[dict[str, str]]) -> list[str]:
    lines: list[str] = []
    for item in items:
        lines.extend([
            "  <skill>",
            f"    <id>{html.escape(item['id'])}</id>",
            f"    <name>{html.escape(item['name'])}</name>",
            f"    <description>{html.escape(item['description'])}</description>",
            "  </skill>",
        ])
    return lines


def render(skills: list[Skill] | list[dict[str, str]]) -> str:
    items = entries(skills) if skills and isinstance(skills[0], Skill) else list(skills)  # type: ignore[arg-type]
    if not items:
        return (
            "Skills provide specialized instructions and workflows for specific tasks.\n"
            "Use the skill tool to load a skill when a task matches its description.\n"
            "No skills are currently available."
        )
    return "\n".join([
        "Skills provide specialized instructions and workflows for specific tasks.",
        "Use the skill tool to load a skill when a task matches its description.",
        "The user may also invoke a skill directly; a skill already present as <skill_content>",
        "does not need to be loaded again.",
        "<available_skills>",
        *_render_items(items),
        "</available_skills>",
    ])


def _decode(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    return [
        {
            "id": str(item["id"]),
            "name": str(item.get("name") or item["id"]),
            "description": str(item.get("description") or ""),
        }
        for item in value
        if isinstance(item, dict) and "id" in item
    ]


def _delta(previous: list[dict[str, str]], current: list[dict[str, str]]) -> str:
    before = {item["id"]: item for item in previous}
    after = {item["id"]: item for item in current}
    added = [after[key] for key in after if key not in before]
    removed = [before[key] for key in before if key not in after]
    changed = [
        after[key]
        for key in after
        if key in before and before[key] != after[key]
    ]
    parts: list[str] = []
    if added:
        parts.extend(["New skills are available in addition to those previously listed:", *_render_items(added)])
    if changed:
        parts.extend(["The following skills have changed. Each entry replaces the previous one:", *_render_items(changed)])
    if removed:
        parts.append(
            "The following skill IDs are no longer available and must not be used: "
            + ", ".join(item["id"] for item in removed)
            + "."
        )
    if not parts:
        return render(current)
    return "\n".join(["The available skills have changed. This list supersedes the previous one.", *parts])


def make_source(skills: list[Skill]) -> Source:
    current = entries(skills)
    return Source(
        key="core/skills",
        read=lambda: current if current else REMOVED,
        initial=lambda value: render(_decode(value)),
        changed=lambda previous, current_value: _delta(_decode(previous), _decode(current_value)),
        removed=lambda _previous: "Skill guidance is no longer available. Do not use any previously listed skill.",
    )
