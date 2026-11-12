# Alpha AI (AVA) — Fontes de instrução versionadas
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Optional


class _Sentinel:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name


UNAVAILABLE = _Sentinel("unavailable")
REMOVED = _Sentinel("removed")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Source:
    """Uma fonte composável de instrução.

    ``read`` devolve o valor atual, ``UNAVAILABLE`` (falha temporária; mantém o
    valor anterior) ou ``REMOVED`` (a fonte existe, mas o valor sumiu).
    """

    key: str
    read: Callable[[], Any]
    initial: Callable[[Any], str]
    changed: Callable[[Any, Any], Optional[str]]
    removed: Callable[[Any], Optional[str]]


@dataclass
class Observation:
    key: str
    value: Any


def read_sources(sources: list[Source]) -> list[Observation]:
    observed: list[Observation] = []
    for source in sources:
        try:
            value = source.read()
        except Exception:
            value = UNAVAILABLE
        observed.append(Observation(source.key, value))
    return observed


def render_initial(sources: list[Source], values: dict[str, Any]) -> str:
    parts: list[str] = []
    by_key = {source.key: source for source in sources}
    for key, value in values.items():
        source = by_key.get(key)
        if source is None or value is REMOVED or value is UNAVAILABLE:
            continue
        text = source.initial(value)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def render_delta(
    sources: list[Source],
    previous: dict[str, Any],
    current: dict[str, Any],
) -> str:
    parts: list[str] = []
    by_key = {source.key: source for source in sources}
    for key, value in current.items():
        source = by_key.get(key)
        if source is None:
            continue
        if value is REMOVED:
            old = previous.get(key)
            text = source.removed(old) if old is not None else None
        elif value is UNAVAILABLE:
            text = None
        elif key not in previous or previous[key] is UNAVAILABLE:
            text = source.initial(value)
        elif previous[key] == value:
            text = None
        else:
            text = source.changed(previous[key], value)
        if text:
            parts.append(text)
    for key, value in previous.items():
        if key in current or value is REMOVED or value is UNAVAILABLE:
            continue
        source = by_key.get(key)
        if source is None:
            continue
        text = source.removed(value)
        if text:
            parts.append(text)
    return "\n\n".join(parts)
