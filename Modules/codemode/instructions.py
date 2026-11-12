# Alpha AI (AVA) — Instruções do Code Mode
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from .catalog import Catalog


def render(catalog: Catalog, max_entries: int = 80) -> str:
    return catalog.render(max_entries=max_entries)
