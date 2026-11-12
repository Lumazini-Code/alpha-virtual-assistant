# Alpha AI (AVA) — IDs e tempo do session store
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import time
import uuid


def now() -> float:
    """Epoch em segundos (REAL), como o resto do projeto."""
    return time.time()


def _new(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def new_session_id() -> str:
    return _new("ses")


def new_message_id() -> str:
    return _new("msg")


def new_part_id() -> str:
    return _new("part")
