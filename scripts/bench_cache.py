#!/usr/bin/env python3
# Alpha AI (AVA) — benchmark de uso/cache a partir do session store
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Relatório de tokens/custo/cache por sessão.

Uso:
    python scripts/bench_cache.py
    python scripts/bench_cache.py --db /caminho/ava_sessions.db
    python scripts/bench_cache.py --session ses_...
    python scripts/bench_cache.py --json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "Modules" / "session" / "db" / "ava_sessions.db"


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def report(db: Path, session_id: str | None = None) -> dict:
    conn = _connect(db)
    where = "WHERE session_id = ?" if session_id else ""
    params = (session_id,) if session_id else ()
    rows = conn.execute(
        f"SELECT "
        f"  session_id, "
        f"  COALESCE(SUM(input_tokens), 0) AS input_tokens, "
        f"  COALESCE(SUM(output_tokens), 0) AS output_tokens, "
        f"  COALESCE(SUM(cache_read), 0) AS cache_read, "
        f"  COALESCE(SUM(cache_write), 0) AS cache_write, "
        f"  COALESCE(SUM(cost), 0) AS cost "
        f"FROM session_usage {where} GROUP BY session_id ORDER BY input_tokens DESC",
        params,
    ).fetchall()
    conn.close()
    sessions = []
    total = {"input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_write": 0, "cost": 0.0}
    for row in rows:
        item = dict(row)
        prompt = int(item.get("input_tokens") or 0)
        cached = int(item.get("cache_read") or 0)
        item["cache_read_ratio"] = (cached / prompt) if prompt else 0.0
        sessions.append(item)
        for key in total:
            total[key] += float(item.get(key) or 0)
    prompt = int(total["input_tokens"] or 0)
    total["cache_read_ratio"] = (float(total["cache_read"]) / prompt) if prompt else 0.0
    return {"sessions": sessions, "total": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--session", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.db.exists():
        raise SystemExit(f"DB não encontrado: {args.db}")
    data = report(args.db, args.session)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        return
    for item in data["sessions"]:
        print(
            f"{item['session_id']}: input={item['input_tokens']} output={item['output_tokens']} "
            f"cache_read={item['cache_read']} ratio={item['cache_read_ratio']:.2%} "
            f"cost=${item['cost']:.4f}"
        )
    total = data["total"]
    print(
        f"TOTAL: input={int(total['input_tokens'])} output={int(total['output_tokens'])} "
        f"cache_read={int(total['cache_read'])} ratio={total['cache_read_ratio']:.2%} "
        f"cost=${total['cost']:.4f}"
    )


if __name__ == "__main__":
    main()
