# Alpha AI (AVA) — Blocos de memória de longo prazo
# Copyright (C) 2026  Felipe lumazini
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
from __future__ import annotations

"""
memory_blocks.py — Blocos de memória: consolidação, CRUD e leitura.

Um bloco é um PONTEIRO para um conjunto de memórias atômicas. Ele ocupa um
slot no recall (gist + cues) e desdobra os átomos sob demanda. O bloco nunca
substitui o texto dos átomos: `gist` é uma etiqueta de ativação e `guide` é
orientação advisory (mini-AGENTS.md do bloco).

A consolidação roda em background: candidatos saem do grafo hebbiano
(co-ativação e precedência temporal), passam por um gate e viram um bloco
via LLM (ou heurística, quando o LLM não está disponível). É idempotente por
`summary_hash` (hash ordenado dos membros).
"""

import hashlib
import json
import os
import re
import time
import asyncio
from typing import Optional

import httpx

from memory_config import (
    BLOCK_CONSOLIDATE_MAX_CANDIDATES,
    BLOCK_CONSOLIDATE_INTERVAL_S,
    BLOCK_EPISODE_GAP_S,
    BLOCK_EXPAND_MAX_CHARS,
    BLOCK_GUIDE_MAX_BULLETS,
    BLOCK_GUIDE_MAX_CHARS,
    BLOCK_MIN_AVG_WEIGHT,
    BLOCK_MIN_DENSITY,
    BLOCK_MIN_EVOCATIONS,
    BLOCK_MIN_MEMBERS,
    BLOCK_PARENT_OVERLAP,
    BLOCK_PARENT_MIN_CHILDREN,
    BLOCK_PRUNE_THRESHOLD,
    BLOCK_TOP_K,
    DECAY_HALF_LIFE_DAYS,
    EDGE_CO_ACTIVATION,
    EDGE_CONTAINS,
    EDGE_PART_OF,
    EDGE_TEMPORAL,
    MEMORY_BLOCKS_ENABLED,
    block_node_id,
    log,
)
from memory_models import (
    BlockConsolidateRequest,
    BlockConsolidateResponse,
    BlockGuideResponse,
    BlockListResponse,
    BlockMember,
    BlockProposal,
    BlockWriteRequest,
    BlockWriteResponse,
    MemoryBlock,
)
from memory_state import state

_BLOCK_LLM_URL = os.getenv("AVA_LLM_URL", "http://127.0.0.1:4003").rstrip("/")
_BLOCK_LLM_TIMEOUT = float(os.getenv("AVA_BLOCK_LLM_TIMEOUT", "45"))

# Contadores do processo de memória (expostos em /status via memory_status).
_STATS = {
    "consolidations_total": 0,
    "consolidation_failures_total": 0,
    "prunes_total": 0,
    "guide_regenerations_total": 0,
    "guide_stale_total": 0,
    "guide_manual_total": 0,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash_members(ids: list[int]) -> str:
    ordered = ",".join(str(int(i)) for i in sorted(set(ids)))
    return hashlib.sha256(ordered.encode()).hexdigest()


def _block_text(gist: str, cues: list[str]) -> str:
    return (gist or "").strip() + ("\n" + "\n".join(cues) if cues else "")


def _guide_hash(guide: str, summary_hash: str) -> str:
    return hashlib.sha256((guide + "\n" + summary_hash).encode()).hexdigest()


def _parse_cues(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(c) for c in raw if str(c).strip()]
    try:
        data = json.loads(raw or "[]")
        return [str(c) for c in data if str(c).strip()] if isinstance(data, list) else []
    except Exception:
        return []


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _members_with_text(db, block_id: int) -> list[BlockMember]:
    rows = db.block_members(block_id)
    out: list[BlockMember] = []
    for r in rows:
        mem = db.get_by_id(r["memory_id"])
        out.append(BlockMember(
            memory_id=r["memory_id"],
            role=r["role"] or "core",
            weight=float(r["weight"] or 1.0),
            text=(mem["text"] if mem else ""),
        ))
    return out


def _row_to_model(row, db, scope: Optional[str] = None,
                  include_members: bool = True) -> MemoryBlock:
    members = _members_with_text(db, row["id"]) if include_members else []
    return MemoryBlock(
        block_id=row["id"],
        name=row["name"],
        kind=row["kind"],
        gist=row["gist"],
        cues=_parse_cues(row["cues"]),
        guide=row["guide"] or "",
        guide_version=int(row["guide_version"] or 1),
        guide_source=row["guide_source"] or "consolidation",
        parent_block_id=row["parent_block_id"],
        level=int(row["level"] or 1),
        confidence=float(row["confidence"] or 1.0),
        member_count=int(row["member_count"] or 0),
        members=members,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_accessed=row["last_accessed"],
        access_count=int(row["access_count"] or 0),
        scope=scope,
    )


async def _embed_block(block_id: int, gist: str, cues: list[str],
                       scope: Optional[str] = None) -> None:
    """Adiciona/atualiza o vetor do bloco. Nunca levanta: sem embedding o
    bloco continua recuperável por nome e pelas arestas do grafo."""
    try:
        emb = await state.embed_engine.embed_passage_one(_block_text(gist, cues))
        sc = state.scopes.get((scope or "").strip()) if scope else None
        idx = sc.blk_index if sc is not None else _blk_index()
        if idx is not None:
            def _upsert():
                idx.remove_ids({int(block_id)})
                idx.add(emb, block_id)
            await asyncio.get_running_loop().run_in_executor(None, _upsert)
    except Exception as e:
        log.debug(f"Bloco #{block_id}: embedding ignorado ({type(e).__name__}: {e})")


def _blk_index():
    """Índice de blocos do escopo ativo (fallback: None)."""
    try:
        return state.scope.blk_index
    except Exception:
        return None


def _db_for(scope: Optional[str] = None):
    """MemoryDB do escopo pedido; sem escopo, o ativo. Leitura não troca o
    escopo global — apenas resolve o DB já aberto no cache."""
    name = (scope or "").strip()
    if name:
        sc = state.scopes.get(name)
        if sc is not None:
            return sc.lt_db
    return state.scope.lt_db


# ── CRUD / escrita ────────────────────────────────────────────────────────────

async def block_write(req: BlockWriteRequest) -> BlockWriteResponse:
    """Cria ou atualiza um bloco. Idempotente por summary_hash e por
    sobreposição de membros (Jaccard >= 0.8)."""
    if not MEMORY_BLOCKS_ENABLED:
        return BlockWriteResponse(stored=False, reason="blocks_disabled")
    db = _db_for(req.scope)
    ids = [m.memory_id for m in req.members]
    if not ids:
        return BlockWriteResponse(stored=False, reason="no_members")
    summary_hash = _hash_members(ids)
    existing = db.block_get_by_hash(summary_hash)
    if existing is None:
        # Sobreposição alta: atualiza o bloco existente em vez de criar irmão.
        for row in db.block_list(limit=200):
            old = set(db.block_member_ids(row["id"]))
            if old and len(old & set(ids)) / len(old | set(ids)) >= 0.8:
                existing = row
                break
    now = time.time()
    if existing is not None:
        block_id = existing["id"]
        db.block_update(
            block_id, name=req.name, kind=req.kind, gist=req.gist,
            cues=json.dumps(req.cues, ensure_ascii=False), summary_hash=summary_hash,
            parent_block_id=req.parent_block_id, level=req.level,
            confidence=req.confidence, source=req.source,
        )
        _remove_missing_members(db, block_id, ids)
        for m in req.members:
            db.block_add_member(block_id, m.memory_id, m.role, m.weight)
        _sync_edges(db, block_id, ids)
        guide = _sanitize_guide(req.guide)
        if guide and existing["guide_source"] != "manual":
            gh = _guide_hash(guide, summary_hash)
            if existing["guide_hash"] != gh or existing["guide_source"] == "stale":
                if existing["guide_source"] == "stale":
                    _STATS["guide_stale_total"] += 1
                db.block_set_guide(block_id, guide, "extractor", gh)
                _STATS["guide_regenerations_total"] += 1
        await _embed_block(block_id, req.gist, req.cues, req.scope)
        return BlockWriteResponse(stored=True, reason="updated", block_id=block_id, created=False)

    guide = _sanitize_guide(req.guide)
    block_id = db.block_insert(
        name=req.name, kind=req.kind, gist=req.gist, cues=req.cues,
        summary_hash=summary_hash, parent_block_id=req.parent_block_id,
        level=req.level, confidence=req.confidence, source=req.source,
        guide=guide, guide_source="extractor",
        guide_hash=_guide_hash(guide, summary_hash) if guide else "",
    )
    for m in req.members:
        db.block_add_member(block_id, m.memory_id, m.role, m.weight)
    _sync_edges(db, block_id, ids)
    await _embed_block(block_id, req.gist, req.cues, req.scope)
    return BlockWriteResponse(stored=True, reason="created", block_id=block_id, created=True)


def _remove_missing_members(db, block_id: int, keep_ids: list[int]) -> None:
    """Remove membros que saíram do conjunto do bloco (sem tocar nos átomos)."""
    keep = {int(i) for i in keep_ids}
    for mid in db.block_member_ids(block_id):
        if int(mid) not in keep:
            db.block_remove_member(block_id, int(mid))


def _sync_edges(db, block_id: int, member_ids: list[int]) -> None:
    """Mantém contains/part_of coerentes com a tabela de membros."""
    node = block_node_id(block_id)
    try:
        current = set(db.block_member_ids(block_id))
        for mid in current:
            db.upsert_edge(node, int(mid), 1.0, time.time(), EDGE_CONTAINS)
            db.upsert_edge(int(mid), node, 1.0, time.time(), EDGE_PART_OF)
        # Remove arestas de membros que saíram do bloco.
        for row in db._conn.execute(
            "SELECT memory_id_a, memory_id_b FROM memory_edges "
            "WHERE edge_type IN (?, ?) AND (memory_id_a = ? OR memory_id_b = ?)",
            (EDGE_CONTAINS, EDGE_PART_OF, node, node),
        ).fetchall():
            other = row["memory_id_b"] if row["memory_id_a"] == node else row["memory_id_a"]
            if int(other) not in current:
                db._conn.execute(
                    "DELETE FROM memory_edges WHERE edge_type IN (?, ?) AND "
                    "((memory_id_a = ? AND memory_id_b = ?) OR (memory_id_a = ? AND memory_id_b = ?))",
                    (EDGE_CONTAINS, EDGE_PART_OF, node, other, other, node),
                )
    except Exception as e:
        log.debug(f"Bloco #{block_id}: falha ao sincronizar arestas ({e})")


async def block_list(kind: Optional[str] = None, scope: Optional[str] = None) -> BlockListResponse:
    if not MEMORY_BLOCKS_ENABLED:
        return BlockListResponse(blocks=[], total=0)
    db = _db_for(scope)
    rows = db.block_list(kind=kind)
    blocks = [_row_to_model(r, db, scope=scope, include_members=False) for r in rows]
    return BlockListResponse(blocks=blocks, total=len(blocks))


async def block_get(block_id: int, scope: Optional[str] = None) -> Optional[MemoryBlock]:
    if not MEMORY_BLOCKS_ENABLED:
        return None
    db = _db_for(scope)
    row = db.block_get(block_id)
    if row is None:
        return None
    db.block_update_access(block_id)
    return _row_to_model(row, db, scope=scope)


async def block_set_guide(block_id: int, guide: str, source: str = "manual",
                          scope: Optional[str] = None) -> BlockGuideResponse:
    if not MEMORY_BLOCKS_ENABLED:
        return BlockGuideResponse(stored=False, reason="blocks_disabled")
    db = _db_for(scope)
    row = db.block_get(block_id)
    if row is None:
        return BlockGuideResponse(stored=False, reason="not_found")
    guide = _truncate(guide, BLOCK_GUIDE_MAX_CHARS)
    gh = _guide_hash(guide, row["summary_hash"])
    db.block_set_guide(block_id, guide, source, gh)
    if source == "manual":
        _STATS["guide_manual_total"] += 1
    return BlockGuideResponse(stored=True, reason="updated", block=_row_to_model(db.block_get(block_id), db, scope))


async def block_delete(block_id: int, scope: Optional[str] = None) -> bool:
    if not MEMORY_BLOCKS_ENABLED:
        return False
    db = _db_for(scope)
    if db.block_get(block_id) is None:
        return False
    db.block_delete(block_id)
    try:
        sc = state.scopes.get((scope or "").strip()) if scope else None
        idx = sc.blk_index if sc is not None else _blk_index()
        if idx is not None:
            idx.remove_ids([int(block_id)])
    except Exception as e:
        log.debug(f"Bloco #{block_id}: remoção do índice ignorada ({type(e).__name__}: {e})")
    return True


# ── Candidatos e consolidação ─────────────────────────────────────────────────

def _find_candidates(db, min_members: int, min_avg_weight: float,
                     min_density: float, min_evocations: int) -> list[dict]:
    """Candidatos saem do grafo existente: co-ativação densa e cadeias
    temporais. Nenhuma chamada de LLM aqui."""
    edges = db.get_edges_above_weight(0.0)
    adj: dict[int, dict[int, float]] = {}
    temporal: dict[int, dict[int, float]] = {}
    for e in edges:
        a, b = int(e["memory_id_a"]), int(e["memory_id_b"])
        if a < 0 or b < 0:
            continue
        et = e["edge_type"]
        w = float(e["weight"])
        if et == EDGE_CO_ACTIVATION:
            adj.setdefault(a, {})[b] = max(adj.setdefault(a, {}).get(b, 0.0), w)
            adj.setdefault(b, {})[a] = max(adj.setdefault(b, {}).get(a, 0.0), w)
        elif et == EDGE_TEMPORAL:
            temporal.setdefault(a, {})[b] = max(temporal.setdefault(a, {}).get(b, 0.0), w)
            temporal.setdefault(b, {})[a] = max(temporal.setdefault(b, {}).get(a, 0.0), w)

    candidates: list[dict] = []
    seen: set[frozenset] = set()

    def consider(cluster: set[int], kind: str) -> None:
        if len(cluster) < min_members:
            return
        key = frozenset(cluster)
        if key in seen:
            return
        seen.add(key)
        # Densidade: arestas observadas / possíveis dentro do cluster.
        observed = 0
        total_w = 0.0
        for a in cluster:
            for b, w in adj.get(a, {}).items():
                if b in cluster and a < b:
                    observed += 1
                    total_w += w
        possible = len(cluster) * (len(cluster) - 1) / 2
        density = observed / possible if possible else 0.0
        avg_w = total_w / observed if observed else 0.0
        evocations = 0
        for a in cluster:
            row = db.get_by_id(a)
            if row is not None:
                evocations += int(row["access_count"] or 0)
        if density >= min_density and avg_w >= min_avg_weight and evocations >= min_evocations:
            candidates.append({
                "members": sorted(cluster),
                "kind": kind,
                "density": round(density, 3),
                "avg_weight": round(avg_w, 3),
                "evocations": evocations,
            })

    # 1) Vizinhança imediata de cada nó (cluster = nó + vizinhos).
    for node, nbrs in adj.items():
        consider({node, *nbrs.keys()}, "topic")

    # 2) Componentes conexos de co-ativação (assunto mais amplo).
    visited: set[int] = set()
    for node in adj:
        if node in visited:
            continue
        comp: set[int] = set()
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp.add(cur)
            stack.extend(n for n in adj.get(cur, {}) if n not in comp)
        visited |= comp
        consider(comp, "topic")

    # 3) Cadeias temporais (episódio). Densidade de co-ativação não se
    # aplica aqui; o gate usa o peso médio das arestas temporais.
    visited_t: set[int] = set()
    for node in temporal:
        if node in visited_t:
            continue
        comp: set[int] = set()
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp.add(cur)
            stack.extend(n for n in temporal.get(cur, {}) if n not in comp)
        visited_t |= comp
        if len(comp) < min_members:
            continue
        observed = 0
        total_w = 0.0
        for a in comp:
            for b, w in temporal.get(a, {}).items():
                if b in comp and a < b:
                    observed += 1
                    total_w += w
        avg_w = total_w / observed if observed else 0.0
        evocations = sum(
            int((db.get_by_id(a) or {"access_count": 0})["access_count"] or 0)
            for a in comp
        )
        if avg_w >= min_avg_weight and evocations >= min_evocations:
            key = frozenset(comp)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "members": sorted(comp),
                "kind": "episode",
                "density": 0.0,
                "avg_weight": round(avg_w, 3),
                "evocations": evocations,
            })

    # 4) Remove clusters que já são subconjuntos de outro candidato maior.
    candidates.sort(key=lambda c: len(c["members"]), reverse=True)
    kept: list[dict] = []
    for c in candidates:
        s = set(c["members"])
        if any(s <= set(k["members"]) for k in kept):
            continue
        kept.append(c)
    return kept


def _heuristic_plan(candidate: dict, db) -> dict:
    """Fallback sem LLM: gist com os primeiros membros e nome pelo assunto
    mais frequente. Nunca falha."""
    ids = candidate["members"]
    texts = []
    for mid in ids:
        row = db.get_by_id(mid)
        if row:
            texts.append(row["text"])
    gist = _truncate("; ".join(texts[:4]), 400)
    name = _truncate((texts[0] if texts else "Memórias relacionadas"), 60)
    return {
        "name": name,
        "kind": candidate.get("kind", "topic"),
        "gist": gist or "Conjunto de memórias relacionadas.",
        "cues": [t[:120] for t in texts[:3]],
        "guide": "",
        "members": [{"id": mid, "role": "core", "weight": 1.0} for mid in ids],
    }


async def _describe_block_via_llm(candidate: dict, db) -> Optional[dict]:
    """1 chamada de LLM por candidato. Devolve None quando o LLM não está
    disponível ou a resposta não é um JSON válido."""
    ids = candidate["members"]
    facts = []
    for mid in ids:
        row = db.get_by_id(mid)
        if row:
            facts.append(f"[M{mid}] {row['text']}")
    if not facts:
        return None
    _allowed_kinds = {"topic", "pattern", "behavior", "person", "project", "episode"}
    prompt = (
        "You are the memory consolidation module. Group the atomic memories below "
        "into ONE block. A block is a pointer/etag to a set of facts: it must NOT "
        "rewrite or replace them. Return ONLY valid JSON.\n\n"
        "Rules:\n"
        "- name: short, specific (max 60 chars).\n"
        "- kind: topic|pattern|behavior|person|project|episode.\n"
        "- gist: 1-2 sentences describing what this block IS (max 400 chars).\n"
        "- cues: up to 6 short trigger phrases that should retrieve this block.\n"
        "- guide: optional advisory guidance (max 1200 chars, up to 6 bullets) on how to "
        "use these facts. It is advisory, never a replacement for the facts.\n"
        "- members: every id in the input, each with role core|example|exception|history.\n"
        "- If the facts do NOT form a coherent concept, return {\"skip\": true}.\n\n"
        "Input facts:\n" + "\n".join(facts) +
        "\n\nJSON schema:\n"
        '{"name":"...","kind":"topic","gist":"...","cues":["..."],'
        '"guide":"...","members":[{"id":1,"role":"core","weight":1.0}]}'
    )
    headers = {"Content-Type": "application/json"}
    try:
        from Modules.internal_auth import internal_token  # type: ignore
        tok = internal_token()
        if tok:
            headers["Authorization"] = f"Bearer {tok}"
    except Exception:
        pass
    try:
        async with httpx.AsyncClient(timeout=_BLOCK_LLM_TIMEOUT) as client:
            r = await client.post(
                f"{_BLOCK_LLM_URL}/chat/tools",
                headers=headers,
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 900,
                },
            )
        if r.status_code != 200:
            return None
        data = r.json()
        content = ((data.get("message") or {}).get("content") or "").strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.S).strip()
        plan = json.loads(content)
        if not isinstance(plan, dict) or plan.get("skip"):
            return None
        valid = {int(m["id"]) for m in plan.get("members", []) if str(m.get("id", "")).isdigit()}
        if not valid or not valid <= set(ids):
            return None
        plan["members"] = [m for m in plan.get("members", []) if int(m.get("id", -1)) in set(ids)]
        plan["name"] = _truncate(str(plan.get("name") or ""), 60)
        plan["gist"] = _truncate(str(plan.get("gist") or ""), 400)
        plan["cues"] = [str(c) for c in (plan.get("cues") or [])][:6]
        plan["guide"] = _truncate(str(plan.get("guide") or ""), BLOCK_GUIDE_MAX_CHARS)
        kind = str(plan.get("kind") or "topic")
        plan["kind"] = kind if kind in _allowed_kinds else "topic"
        return plan
    except Exception as e:
        log.debug(f"Consolidação: LLM indisponível ({type(e).__name__}: {e})")
        return None


def _sanitize_guide(guide: str) -> str:
    """Remove bullets excessivos e conteúdo perigoso; o guia é advisory."""
    if not guide:
        return ""
    lines = [ln.strip() for ln in guide.splitlines() if ln.strip()]
    bullets = [ln for ln in lines if ln.startswith(("-", "*", "•"))]
    if len(bullets) > BLOCK_GUIDE_MAX_BULLETS:
        keep = set(id(b) for b in bullets[:BLOCK_GUIDE_MAX_BULLETS])
        lines = [ln for ln in lines if not ln.startswith(("-", "*", "•")) or id(ln) in keep]
    text = "\n".join(lines)
    text = re.sub(r"(?i)(ignore|disregard)\s+(all\s+)?(previous|above)\s+instructions", "", text)
    # Nunca deixar o guia carregar segredos/credenciais (é advisory, mas
    # entra no prompt do modelo e pode vazar em logs).
    text = re.sub(r"(?i)\b(sk-[a-z0-9]{12,}|api[_-]?key\s*[:=]\s*\S+|"
                  r"password\s*[:=]\s*\S+|token\s*[:=]\s*\S+)\b", "[redacted]", text)
    return _truncate(text, BLOCK_GUIDE_MAX_CHARS)


async def _apply_plan(db, plan: dict, candidate: dict, source: str,
                      scope: Optional[str] = None) -> tuple[bool, bool]:
    """Grava/atualiza o bloco. Retorna (stored, created)."""
    ids = [int(m["id"]) for m in plan.get("members", []) if str(m.get("id", "")).isdigit()]
    if not ids:
        return False, False
    summary_hash = _hash_members(ids)
    existing = db.block_get_by_hash(summary_hash)
    if existing is None:
        for row in db.block_list(limit=200):
            old = set(db.block_member_ids(row["id"]))
            if old and len(old & set(ids)) / len(old | set(ids)) >= 0.8:
                existing = row
                break
    cues = [str(c) for c in (plan.get("cues") or [])]
    guide = _sanitize_guide(str(plan.get("guide") or ""))
    if existing is not None:
        block_id = existing["id"]
        db.block_update(
            block_id, name=_truncate(plan.get("name") or existing["name"], 60),
            kind=plan.get("kind") or existing["kind"],
            gist=_truncate(plan.get("gist") or existing["gist"], 400),
            cues=json.dumps(cues, ensure_ascii=False), summary_hash=summary_hash,
            confidence=float(candidate.get("avg_weight", 0.5)),
        )
        _remove_missing_members(db, block_id, ids)
        for m in plan.get("members", []):
            db.block_add_member(block_id, int(m["id"]), m.get("role", "core"), float(m.get("weight", 1.0)))
        _sync_edges(db, block_id, ids)
        if guide and existing["guide_source"] != "manual":
            gh = _guide_hash(guide, summary_hash)
            if existing["guide_hash"] != gh or existing["guide_source"] == "stale":
                if existing["guide_source"] == "stale":
                    _STATS["guide_stale_total"] += 1
                db.block_set_guide(block_id, guide, source, gh)
                _STATS["guide_regenerations_total"] += 1
        elif existing["summary_hash"] != summary_hash and existing["guide_source"] != "manual":
            # Membros mudaram sem guia novo: o guia antigo pode contradizer
            # os átomos. Marca stale para regeneração no próximo ciclo.
            db.block_update(block_id, guide_source="stale")
            _STATS["guide_stale_total"] += 1
        await _embed_block(block_id, plan.get("gist") or "", cues, scope)
        return True, False

    block_id = db.block_insert(
        name=_truncate(plan.get("name") or "Bloco", 60),
        kind=plan.get("kind") or "topic",
        gist=_truncate(plan.get("gist") or "", 400),
        cues=cues, summary_hash=summary_hash,
        confidence=float(candidate.get("avg_weight", 0.5)), source=source,
    )
    for m in plan.get("members", []):
        db.block_add_member(block_id, int(m["id"]), m.get("role", "core"), float(m.get("weight", 1.0)))
    _sync_edges(db, block_id, ids)
    if guide:
        db.block_set_guide(block_id, guide, source, _guide_hash(guide, summary_hash))
    await _embed_block(block_id, plan.get("gist") or "", cues, scope)
    return True, True


async def consolidate_blocks(req: Optional[BlockConsolidateRequest] = None) -> BlockConsolidateResponse:
    """Consolida candidatos do escopo ativo em blocos. Idempotente."""
    req = req or BlockConsolidateRequest()
    if not MEMORY_BLOCKS_ENABLED:
        return BlockConsolidateResponse(candidates=0, created=0, updated=0)
    db = _db_for(req.scope)
    min_members = req.min_members or BLOCK_MIN_MEMBERS
    min_avg = req.min_avg_weight if req.min_avg_weight is not None else BLOCK_MIN_AVG_WEIGHT
    candidates = _find_candidates(db, min_members, min_avg, BLOCK_MIN_DENSITY, BLOCK_MIN_EVOCATIONS)
    candidates = candidates[: (req.max_candidates or BLOCK_CONSOLIDATE_MAX_CANDIDATES)]
    created = updated = skipped = 0
    blocks: list[MemoryBlock] = []
    for cand in candidates:
        plan = None if req.dry_run else await _describe_block_via_llm(cand, db)
        if plan is None:
            plan = _heuristic_plan(cand, db)
        if req.dry_run:
            skipped += 1
            continue
        stored, was_created = await _apply_plan(db, plan, cand, "consolidation", req.scope)
        if not stored:
            skipped += 1
            continue
        if was_created:
            created += 1
        else:
            updated += 1
    _STATS["consolidations_total"] += 1
    if not req.dry_run:
        _STATS["prunes_total"] += _prune_blocks(db)
        await _refresh_parent_blocks(db, req.scope)
        for row in db.block_list(limit=20):
            blocks.append(_row_to_model(row, db, include_members=False))
    return BlockConsolidateResponse(
        candidates=len(candidates), created=created, updated=updated,
        skipped=skipped, dry_run=req.dry_run, blocks=blocks,
    )


async def _refresh_parent_blocks(db, scope: Optional[str] = None) -> int:
    """Agrupa blocos irmãos com membros sobrepostos sob um bloco-pai.
    Heurística: mesmo kind, >= BLOCK_PARENT_MIN_CHILDREN filhos e Jaccard
    médio >= BLOCK_PARENT_OVERLAP. O pai não copia membros: aponta para os
    filhos via parent_block_id (hierarquia letras → palavras → frases)."""
    rows = [r for r in db.block_list(limit=500) if r["parent_block_id"] is None]
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r["kind"] or "topic", []).append(r)
    created = 0
    for kind, kids in groups.items():
        if len(kids) < BLOCK_PARENT_MIN_CHILDREN:
            continue
        sets = [set(db.block_member_ids(k["id"])) for k in kids]
        # Agrupa por sobreposição: union-find simples.
        parent = list(range(len(kids)))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri
        for i in range(len(kids)):
            for j in range(i + 1, len(kids)):
                if not sets[i] or not sets[j]:
                    continue
                jac = len(sets[i] & sets[j]) / len(sets[i] | sets[j])
                if jac >= BLOCK_PARENT_OVERLAP:
                    union(i, j)
        comps: dict[int, list[int]] = {}
        for i in range(len(kids)):
            comps.setdefault(find(i), []).append(i)
        for comp in comps.values():
            if len(comp) < BLOCK_PARENT_MIN_CHILDREN:
                continue
            child_rows = [kids[i] for i in comp]
            # Já existe pai? (filhos com parent_block_id preenchido)
            if any(c["parent_block_id"] for c in child_rows):
                continue
            name = _truncate(f"{kind}: " + " / ".join(c["name"] for c in child_rows[:3]), 60)
            gist = _truncate("; ".join(c["gist"] for c in child_rows[:3]), 400)
            pid = db.block_insert(
                name=name, kind=kind, gist=gist, cues=[],
                # IDs negativos: o hash do pai nunca colide com um bloco de
                # átomos cujos memory_ids coincidam com os ids dos filhos.
                summary_hash=_hash_members([-int(c["id"]) for c in child_rows]),
                level=max(int(c["level"] or 1) for c in child_rows) + 1,
                confidence=min(float(c["confidence"] or 1.0) for c in child_rows),
                source="hierarchy",
            )
            for c in child_rows:
                db.block_update(c["id"], parent_block_id=pid)
                # Estrutura pai↔filho: o pai aponta para os filhos e o PPR
                # desce a hierarquia sem copiar membros.
                db.upsert_edge(block_node_id(pid), block_node_id(c["id"]),
                               1.0, time.time(), EDGE_CONTAINS)
                db.upsert_edge(block_node_id(c["id"]), block_node_id(pid),
                               1.0, time.time(), EDGE_PART_OF)
            db.block_update(
                pid,
                member_count=sum(int(c["member_count"] or 0) for c in child_rows),
            )
            await _embed_block(pid, gist, [], scope)
            created += 1
    return created


def _prune_blocks(db) -> int:
    """Remove blocos sem uso e blocos com menos de 2 membros. Não apaga átomos.

    A confiança decai com o tempo (meia-vida de DECAY_HALF_LIFE_DAYS) e o
    bloco é podado abaixo de BLOCK_PRUNE_THRESHOLD. Blocos-pai (com filhos)
    são preservados: a hierarquia é estrutural, não um item de recall."""
    removed = 0
    now = time.time()
    for row in db.block_list(limit=500):
        last = float(row["last_accessed"] or row["updated_at"] or now)
        age_days = max(0.0, (now - last) / 86400.0)
        decayed = float(row["confidence"] or 1.0) * (0.5 ** (age_days / max(1.0, DECAY_HALF_LIFE_DAYS)))
        stale = age_days > 180
        small = int(row["member_count"] or 0) < 2
        has_children = db._conn.execute(
            "SELECT COUNT(*) FROM memory_blocks WHERE parent_block_id = ?", (row["id"],)
        ).fetchone()[0]
        if has_children:
            continue
        if decayed < BLOCK_PRUNE_THRESHOLD or stale or small:
            db.block_delete(row["id"])
            try:
                idx = _blk_index()
                if idx is not None:
                    idx.remove_ids([int(row["id"])])
            except Exception:
                pass
            removed += 1
        elif abs(decayed - float(row["confidence"] or 1.0)) > 1e-6:
            db.block_update(row["id"], confidence=round(decayed, 4))
    return removed


def _dissolve_small_blocks(db) -> int:
    return _prune_blocks(db)


# ── Integração com o extrator / escrita de átomos ─────────────────────────────

async def attach_memory_to_block(memory_id: int, block_id: int,
                                 role: str = "core", weight: float = 1.0,
                                 scope: Optional[str] = None) -> bool:
    """Associa um átomo recém-gravado a um bloco existente."""
    db = _db_for(scope)
    row = db.block_get(block_id)
    if row is None:
        return False
    db.block_add_member(block_id, memory_id, role, weight)
    _sync_edges(db, block_id, [memory_id])
    return True


async def propose_block_from_memories(memory_ids: list[int], name: str,
                                      gist: str = "", kind: str = "topic",
                                      guide: str = "",
                                      scope: Optional[str] = None) -> Optional[int]:
    """Cria um bloco provisório a partir de ids já gravados (usado pelo
    extrator quando ele propõe um bloco novo)."""
    if not memory_ids:
        return None
    req = BlockWriteRequest(
        name=name or "Bloco", kind=kind, gist=gist or name,
        members=[BlockMember(memory_id=i) for i in memory_ids],
        confidence=0.5, source="extractor", scope=scope,
    )
    res = await block_write(req)
    if res.stored and guide:
        await block_set_guide(res.block_id, guide, "extractor", scope=scope)
    return res.block_id if res.stored else None


__all__ = [
    "block_write", "block_list", "block_get", "block_set_guide", "block_delete",
    "consolidate_blocks", "attach_memory_to_block", "propose_block_from_memories",
    "BLOCK_CONSOLIDATE_INTERVAL_S",
]
