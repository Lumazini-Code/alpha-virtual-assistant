#!/usr/bin/env python3
# Alpha AI (AVA) — Migração do ST antigo para o session store
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Migra ``turn_groups`` (ShortTermDB) para ``session_message``.

Uso:
    python scripts/migrate_st_to_sessions.py --dry-run
    python scripts/migrate_st_to_sessions.py
    python scripts/migrate_st_to_sessions.py --st-db /caminho/ava_short_term.db

O script é idempotente: cada grupo migrado carrega
``metadata.st_group_id`` e é ignorado em execuções seguintes.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Modules.session import SessionStore  # noqa: E402

DEFAULT_ST_DB = ROOT / "Modules" / "memory" / "db" / "ava_short_term.db"


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _turns(raw: str) -> list[dict]:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []


def migrate(st_db: Path, sessions_db: Path | None, *, dry_run: bool = False) -> dict:
    if not st_db.exists():
        raise SystemExit(f"ST DB não encontrado: {st_db}")

    conn = _connect(st_db)
    rows = conn.execute(
        "SELECT id, session_id, person_id, turns_json, created_at "
        "FROM turn_groups ORDER BY created_at ASC, id ASC"
    ).fetchall()
    conn.close()

    store = None if dry_run else SessionStore(sessions_db)
    stats = {"groups": 0, "sessions": 0, "messages": 0, "skipped": 0}
    seen_sessions: set[str] = set()

    for row in rows:
        stats["groups"] += 1
        sid = (row["session_id"] or "default").strip() or "default"
        group_id = int(row["id"])
        if store is not None and store.has_source(sid, group_id):
            stats["skipped"] += 1
            continue

        turns = _turns(row["turns_json"])
        if not turns:
            stats["skipped"] += 1
            continue

        if store is not None:
            if sid not in seen_sessions:
                store.create(
                    session_id=sid,
                    person_id=row["person_id"],
                    title=_title(turns),
                )
                seen_sessions.add(sid)
            for turn in turns:
                role = str(turn.get("role") or "").strip()
                content = turn.get("content")
                if role not in ("user", "assistant") or not content:
                    continue
                store.append_message(
                    sid,
                    {
                        "role": role,
                        "content": content,
                        "metadata": {
                            "migrated_from_st": True,
                            "st_group_id": group_id,
                            "st_created_at": row["created_at"],
                        },
                    },
                )
                stats["messages"] += 1
        else:
            stats["messages"] += sum(
                1
                for turn in turns
                if turn.get("role") in ("user", "assistant") and turn.get("content")
            )

    if store is not None:
        stats["sessions"] = len(seen_sessions)
        store.close()
    else:
        stats["sessions"] = len({(row["session_id"] or "default") for row in rows})

    return stats


def _title(turns: list[dict]) -> str | None:
    for turn in turns:
        if turn.get("role") == "user" and turn.get("content"):
            text = str(turn["content"]).strip().replace("\n", " ")
            return text[:80] or None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--st-db", type=Path, default=DEFAULT_ST_DB)
    parser.add_argument("--sessions-db", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stats = migrate(args.st_db, args.sessions_db, dry_run=args.dry_run)
    mode = "DRY-RUN" if args.dry_run else "MIGRADO"
    print(
        f"[{mode}] grupos={stats['groups']} sessões={stats['sessions']} "
        f"mensagens={stats['messages']} ignorados={stats['skipped']}"
    )


if __name__ == "__main__":
    main()
