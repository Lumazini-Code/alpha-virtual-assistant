# Alpha AI (AVA) — Baseline e deltas de instruções por sessão
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from ..session.store import SessionStore
from .source import (
    REMOVED,
    UNAVAILABLE,
    Observation,
    Source,
    hash_value,
    read_sources,
    render_delta,
    render_initial,
)


@dataclass
class Prepared:
    """Texto para o system prompt: baseline estável + deltas acumulados."""

    baseline: str
    deltas: list[str]

    @property
    def text(self) -> str:
        parts = [self.baseline, *self.deltas]
        return "\n\n".join(part for part in parts if part.strip())


class InstructionState:
    """Guarda o baseline e os deltas de instruções de cada sessão.

    O baseline é a primeira versão vista pela sessão e nunca é reescrito. Cada
    mudança vira um delta textual anexado ao fim do bloco de instruções, o que
    preserva o prefixo cacheado do prompt.
    """

    def __init__(self, store: SessionStore):
        self.store = store
        self.db = store.db

    def prepare(self, session_id: str, sources: list[Source]) -> Prepared:
        if not session_id:
            return Prepared("", [])
        observed = read_sources(sources)
        row = self.db.fetchone(
            "SELECT * FROM instruction_state WHERE session_id = ?",
            (session_id,),
        )
        if row is None:
            return self._initialize(session_id, sources, observed)
        return self._update(session_id, sources, observed, row)

    def _initialize(
        self,
        session_id: str,
        sources: list[Source],
        observed: list[Observation],
    ) -> Prepared:
        values: dict[str, Any] = {}
        hashes: dict[str, str] = {}
        with self.db.transaction() as tx:
            for observation in observed:
                if observation.value is UNAVAILABLE or observation.value is REMOVED:
                    continue
                digest = hash_value(observation.value)
                values[observation.key] = observation.value
                hashes[observation.key] = digest
                self._put_blob(tx, digest, observation.value)
            baseline = render_initial(sources, values)
            tx.execute(
                "INSERT INTO instruction_state "
                "(session_id, baseline_text, values_json, deltas_json, time_updated) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, baseline, json.dumps(hashes), json.dumps([]), _now()),
            )
        return Prepared(baseline, [])

    def _update(
        self,
        session_id: str,
        sources: list[Source],
        observed: list[Observation],
        row: Any,
    ) -> Prepared:
        previous_hashes = json.loads(row["values_json"] or "{}")
        previous = self._load_values(previous_hashes)
        current_hashes: dict[str, str] = {}
        current: dict[str, Any] = {}
        new_blobs: dict[str, Any] = {}

        for observation in observed:
            key = observation.key
            if observation.value is UNAVAILABLE:
                digest = previous_hashes.get(key)
                if digest:
                    current_hashes[key] = digest
                    current[key] = previous.get(key, UNAVAILABLE)
                continue
            if observation.value is REMOVED:
                current[key] = REMOVED
                continue
            digest = hash_value(observation.value)
            current_hashes[key] = digest
            current[key] = observation.value
            new_blobs[digest] = observation.value

        delta = render_delta(sources, previous, current)
        deltas = json.loads(row["deltas_json"] or "[]")
        if not isinstance(deltas, list):
            deltas = []
        if delta:
            deltas.append(delta)

        with self.db.transaction() as tx:
            for digest, value in new_blobs.items():
                self._put_blob(tx, digest, value)
            tx.execute(
                "UPDATE instruction_state SET values_json = ?, deltas_json = ?, time_updated = ? "
                "WHERE session_id = ?",
                (json.dumps(current_hashes), json.dumps(deltas), _now(), session_id),
            )
        return Prepared(str(row["baseline_text"] or ""), deltas)

    def _load_values(self, hashes: dict[str, str]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for key, digest in hashes.items():
            row = self.db.fetchone(
                "SELECT value FROM instruction_blob WHERE hash = ?",
                (digest,),
            )
            if row is not None:
                values[key] = json.loads(row["value"])
        return values

    def _put_blob(self, tx: Any, digest: str, value: Any) -> None:
        tx.execute(
            "INSERT INTO instruction_blob (hash, value) VALUES (?, ?) ON CONFLICT(hash) DO NOTHING",
            (digest, json.dumps(value, ensure_ascii=False, sort_keys=True)),
        )

    def source_paths(self, session_id: str, key: str = "core/instructions") -> set[str]:
        """Caminhos de AGENTS.md já presentes no baseline da sessão."""
        row = self.db.fetchone(
            "SELECT values_json FROM instruction_state WHERE session_id = ?",
            (session_id,),
        )
        if row is None:
            return set()
        hashes = json.loads(row["values_json"] or "{}")
        digest = hashes.get(key)
        if not digest:
            return set()
        blob = self.db.fetchone(
            "SELECT value FROM instruction_blob WHERE hash = ?",
            (digest,),
        )
        if blob is None:
            return set()
        try:
            value = json.loads(blob["value"])
        except (TypeError, ValueError):
            return set()
        if not isinstance(value, list):
            return set()
        return {
            str(item["path"])
            for item in value
            if isinstance(item, dict) and "path" in item
        }

    # ── AGENTS.md aninhados descobertos por leitura ───────────────────────
    def append_synthetic(self, session_id: str, path: str, content: str) -> bool:
        """Injeta instruções de um AGENTS.md próximo, uma vez por caminho."""
        if not session_id or not path or not content.strip():
            return False
        if self._has_synthetic(session_id, path):
            return False
        text = f"Instructions from: {path}\n{content}"
        self.store.append_message(
            session_id,
            {
                "type": "synthetic",
                "role": "system",
                "content": text,
                "metadata": {"instruction": {"paths": [path]}},
            },
        )
        return True

    def synthetic_text(self, session_id: str) -> str:
        rows = self.db.fetchall(
            "SELECT data FROM session_message WHERE session_id = ? AND type = 'synthetic' ORDER BY seq ASC",
            (session_id,),
        )
        parts: list[str] = []
        for row in rows:
            try:
                data = json.loads(row["data"])
            except (TypeError, ValueError):
                continue
            metadata = data.get("metadata") or {}
            if not isinstance(metadata, dict) or "instruction" not in metadata:
                continue
            content = data.get("content") or data.get("text") or ""
            if content:
                parts.append(str(content))
        return "\n\n".join(parts)

    def _has_synthetic(self, session_id: str, path: str) -> bool:
        rows = self.db.fetchall(
            "SELECT data FROM session_message WHERE session_id = ? AND type = 'synthetic'",
            (session_id,),
        )
        for row in rows:
            try:
                data = json.loads(row["data"])
            except (TypeError, ValueError):
                continue
            paths = (
                data.get("metadata", {}).get("instruction", {}).get("paths", [])
                if isinstance(data.get("metadata"), dict)
                else []
            )
            if path in paths:
                return True
        return False

    def reset(self, session_id: str) -> None:
        with self.db.transaction() as tx:
            tx.execute("DELETE FROM instruction_state WHERE session_id = ?", (session_id,))


def _now() -> float:
    import time

    return time.time()
