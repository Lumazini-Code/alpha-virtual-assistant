# Alpha AI (AVA) — Skills (catálogo + carregamento sob demanda)
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from .catalog import make_source, render, entries
from .discovery import Skill, discover, load_skills
from .loader import load_skill, load_skill_file, load_uri, parse_skill_uri, skill_uri
from .permissions import SkillPolicy
from .tool import load_request, resolve_skill

__all__ = [
    "Skill",
    "SkillPolicy",
    "discover",
    "entries",
    "load_request",
    "load_skill",
    "load_skill_file",
    "load_skills",
    "load_uri",
    "make_source",
    "parse_skill_uri",
    "render",
    "resolve_skill",
    "skill_uri",
]
