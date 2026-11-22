# Alpha AI (AVA) — Episódios da memória de longo prazo
# Copyright (C) 2026  Felipe Lumazini
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
from __future__ import annotations

"""
memory_episodes.py — Memória episódica: um resumo por turno.

O ledger (`SessionStore`) guarda o texto bruto de tudo. Um episódio é um
átomo narrativo derivado: resumo, intenção, resultado, ferramentas úteis,
entidades, decisões e pendências. Ele é gravado SEMPRE (em ambos os modos de
contexto) e recuperado por recall no modo conversacional.

Episódios não substituem fatos nem o ledger: são ponteiros com contexto.
"""

import asyncio
import json
import time
from typing import Optional

from memory_config import (
    EPISODES_ENABLED,
    EPISODE_MAX_CHARS,
    EPISODE_MIN_SCORE,
    EPISODE_TOP_K,
    log,
)
from memory_models import (
    EpisodeResponse,
    EpisodeSearchRequest,
    EpisodeSearchResponse,
    EpisodeWriteRequest,
    EpisodeWriteResponse,
)
from memory_state import state


def _db_for(scope: Optional[str] = None):
    name = (scope or "").strip()
    if name:
        sc = state.scopes.get(name)
        if sc is not None:
            return sc.lt_db
    return state.scope.lt_db


def _index_for(scope: Optional[str] = None):
    name = (scope or "").strip()
    if name:
        sc = state.scopes.get(name)
        if sc is not None:
            return sc.ep_index
    return state.scope.ep_index


def _parse_list(raw) -> list:
    if isinstance(raw, list):
        return raw
    try:
        value = json.loads(raw or "[]")
        return value if isinstance(value, list) else []
    except Exception:
        return []


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _row_to_episode(row, scope: Optional[str] = None,
                    score: Optional[float] = None) -> EpisodeResponse:
    return EpisodeResponse(
        episode_id=int(row["id"]),
        session_id=row["session_id"] or "",
        turn_id=row["turn_id"] or "",
        summary=row["summary"] or "",
        intent=row["intent"] or "",
        outcome=row["outcome"] or "",
        tools=[str(x) for x in _parse_list(row["tools"])],
        tool_results=[x for x in _parse_list(row["tool_results"]) if isinstance(x, dict)],
        entities=[str(x) for x in _parse_list(row["entities"])],
        decisions=[str(x) for x in _parse_list(row["decisions"])],
        open_questions=[str(x) for x in _parse_list(row["open_questions"])],
        message_ids=[str(x) for x in _parse_list(row["message_ids"])],
        importance=float(row["importance"] or 0.5),
        created_at=row["created_at"],
        last_accessed=row["last_accessed"],
        access_count=int(row["access_count"] or 0),
        parent_episode_id=row["parent_episode_id"],
        score=score,
        scope=scope,
    )


def _episode_text(req: EpisodeWriteRequest) -> str:
    parts = [req.summary, req.intent, req.outcome]
    if req.entities:
        parts.append("Entities: " + ", ".join(req.entities))
    if req.decisions:
        parts.append("Decisions: " + "; ".join(req.decisions))
    if req.open_questions:
        parts.append("Open questions: " + "; ".join(req.open_questions))
    return "\n".join(p for p in parts if p).strip()


async def _embed_episode(episode_id: int, text: str, scope: Optional[str] = None) -> None:
    """Indexa o episódio. Nunca levanta: sem vetor ele continua no SQLite."""
    try:
        emb = await state.embed_engine.embed_passage_one(text)
        idx = _index_for(scope)
        if idx is None:
            return

        def _upsert():
            idx.remove_ids({int(episode_id)})
            idx.add(emb, int(episode_id))

        await asyncio.get_running_loop().run_in_executor(None, _upsert)
    except Exception as e:
        log.debug(f"Episódio #{episode_id}: embedding ignorado ({type(e).__name__}: {e})")


async def episode_write(req: EpisodeWriteRequest) -> EpisodeWriteResponse:
    if not EPISODES_ENABLED:
        return EpisodeWriteResponse(stored=False, reason="episodes_disabled")
    if not (req.session_id or "").strip() or not (req.turn_id or "").strip():
        return EpisodeWriteResponse(stored=False, reason="missing_session_or_turn")
    summary = _truncate(req.summary, EPISODE_MAX_CHARS)
    if not summary:
        return EpisodeWriteResponse(stored=False, reason="empty_summary")
    db = _db_for(req.scope)
    payload = req.model_dump()
    payload["summary"] = summary
    episode_id = db.episode_insert(payload)
    if not episode_id:
        return EpisodeWriteResponse(stored=False, reason="insert_failed")
    await _embed_episode(episode_id, _episode_text(req), scope=req.scope)
    return EpisodeWriteResponse(stored=True, episode_id=episode_id)


async def episode_search(req: EpisodeSearchRequest) -> EpisodeSearchResponse:
    if not EPISODES_ENABLED or not (req.query or "").strip():
        return EpisodeSearchResponse(episodes=[])
    db = _db_for(req.scope)
    top_k = max(1, min(int(req.top_k or EPISODE_TOP_K), 20))
    hits: list[tuple[int, float]] = []
    try:
        emb = await state.embed_engine.embed_query_one(req.query)
        idx = _index_for(req.scope)
        if idx is not None:
            hits = idx.search(emb, top_k)
    except Exception as e:
        log.debug(f"Busca de episódios: embedding falhou ({type(e).__name__}: {e})")

    episodes: list[EpisodeResponse] = []
    seen: set[int] = set()
    for episode_id, score in hits:
        if score < EPISODE_MIN_SCORE:
            continue
        row = db.episode_get(int(episode_id))
        if row is None:
            continue
        seen.add(int(episode_id))
        episodes.append(_row_to_episode(row, scope=req.scope, score=round(float(score), 4)))

    # Continuidade: episódios recentes da mesma sessão entram mesmo sem match
    # semântico forte — o fio da conversa importa.
    if req.session_id:
        for row in db.episode_recent(req.session_id, limit=3):
            eid = int(row["id"])
            if eid in seen:
                continue
            seen.add(eid)
            episodes.append(_row_to_episode(row, scope=req.scope, score=0.0))

    episodes.sort(key=lambda e: (e.score or 0.0, e.created_at or 0.0), reverse=True)
    return EpisodeSearchResponse(episodes=episodes[:top_k])


def episode_list(session_id: Optional[str] = None, scope: Optional[str] = None,
                 limit: int = 50) -> list[EpisodeResponse]:
    db = _db_for(scope)
    rows = db.episode_list(session_id=session_id, limit=max(1, min(limit, 200)))
    return [_row_to_episode(row, scope=scope) for row in rows]


def episode_recent(session_id: str, scope: Optional[str] = None,
                   limit: int = 6) -> list[EpisodeResponse]:
    db = _db_for(scope)
    rows = db.episode_recent(session_id, limit=max(1, min(limit, 20)))
    return [_row_to_episode(row, scope=scope) for row in rows]


def episode_mark_accessed(episode_ids: list[int], scope: Optional[str] = None) -> None:
    db = _db_for(scope)
    for eid in episode_ids:
        db.episode_update_access(int(eid))


def format_episode_line(ep: EpisodeResponse, now: Optional[float] = None) -> str:
    """Linha compacta para o bloco de recall."""
    now = now or time.time()
    age = max(0.0, now - float(ep.created_at or now))
    if age < 3600:
        age_s = f"{int(age // 60)}m"
    elif age < 86400:
        age_s = f"{int(age // 3600)}h"
    else:
        age_s = f"{int(age // 86400)}d"
    line = f"- [{age_s} ago] {ep.summary}"
    if ep.outcome:
        line += f" → {ep.outcome}"
    if ep.open_questions:
        line += " (open: " + "; ".join(ep.open_questions[:2]) + ")"
    return line
