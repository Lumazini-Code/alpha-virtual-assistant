# Alpha AI (AVA) — Métricas operacionais
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import re
import threading
from typing import Any


def _name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", value)


class Metrics:
    """Contadores/gauges em memória, resetados no restart."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {
            "turns_total": 0,
            "steps_total": 0,
            "turns_degraded": 0,
            "tool_calls_total": 0,
            "tool_failures_total": 0,
            "degraded": {
                "skill_selector_fallback": 0,
                "mcp_manager_offline": 0,
                "mcp_server_load_failures": 0,
                "agent_skills_unavailable": 0,
            },
            "tool_failures_by_category": {},
            "unknown_tool_calls": 0,
            "invalid_json_calls": 0,
            "compactions_total": 0,
            "compaction_failures_total": 0,
            "subagent_jobs_total": 0,
            "background_jobs_total": 0,
            "queue_drops": 0,
            "memory_save_failures": 0,
            "memory_extraction_failures": 0,
            "force_finish_calls": 0,
        }

    def inc(self, name: str, key: str | None = None, n: int = 1) -> None:
        try:
            with self._lock:
                if key is None:
                    self._data[name] = int(self._data.get(name, 0)) + n
                    return
                bucket = self._data.get(name)
                if not isinstance(bucket, dict):
                    bucket = self._data[name] = {}
                bucket[key] = int(bucket.get(key, 0)) + n
        except Exception:
            pass

    def set(self, name: str, value: float, key: str | None = None) -> None:
        try:
            with self._lock:
                if key is None:
                    self._data[name] = value
                    return
                bucket = self._data.get(name)
                if not isinstance(bucket, dict):
                    bucket = self._data[name] = {}
                bucket[key] = value
        except Exception:
            pass

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                key: dict(value) if isinstance(value, dict) else value
                for key, value in self._data.items()
            }

    def render_prometheus(self, prefix: str = "ava_") -> str:
        lines: list[str] = []
        with self._lock:
            for key, value in self._data.items():
                metric = prefix + _name(key)
                if isinstance(value, dict):
                    for label, count in value.items():
                        lines.append(
                            f'{metric}{{kind="{_label(label)}"}} {count}'
                        )
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    lines.append(f"{metric} {value}")
        return "\n".join(lines) + "\n"


def _label(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


METRICS = Metrics()


def inc(name: str, key: str | None = None, n: int = 1) -> None:
    METRICS.inc(name, key, n)


def snapshot() -> dict[str, Any]:
    return METRICS.snapshot()


def render_prometheus(prefix: str = "ava_") -> str:
    return METRICS.render_prometheus(prefix)
