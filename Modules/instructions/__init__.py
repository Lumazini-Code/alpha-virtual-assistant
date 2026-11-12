# Alpha AI (AVA) — Instruções versionadas (AGENTS.md)
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from .discovery import InstructionFile, discover_files, discover_near, load_sources, make_source
from .source import REMOVED, UNAVAILABLE, Source
from .state import InstructionState, Prepared

__all__ = [
    "InstructionFile",
    "InstructionState",
    "Prepared",
    "REMOVED",
    "Source",
    "UNAVAILABLE",
    "discover_files",
    "discover_near",
    "load_sources",
    "make_source",
]
