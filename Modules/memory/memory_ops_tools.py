from __future__ import annotations

"""
memory_ops_tools.py — Handlers do RAG de tool selection.

/tools/register (e batch com prune), /select (kNN com max-pooling por tool,
expansão por PPR no grafo de co-uso, tools core sempre presentes),
/record_usage (arestas co-uso/sequência + queries aprendidas), /list, /delete.

Lock _tl_write_lock vive aqui — mesmo padrão dos demais módulos de handlers.
"""

import asyncio
import hashlib
import time

from memory_config import (
    EDGE_CO_ACTIVATION,
    EDGE_LEARNING_RATE,
    EDGE_TEMPORAL,
    TOOLS_FALLBACK_TOP_K,
    TOOLS_LEARN_DEDUP_SCORE,
    TOOLS_LOW_CONFIDENCE_SCORE,
    TOOLS_MAX_EXAMPLES,
    TOOLS_MAX_LEARNED_EXAMPLES,
    TOOLS_MAX_RELATED,
    TOOLS_USAGE_MAX_SEQUENCE,
    log,
)
from memory_db import ToolsDB
from memory_models import (
    MemoryToolError,
    ToolMatch,
    ToolRegisterBatchRequest,
    ToolRegisterBatchResponse,
    ToolRegisterRequest,
    ToolRegisterResponse,
    ToolSelectRequest,
    ToolSelectResponse,
    ToolUsageRequest,
)
from memory_search import _ppr_spread_on
from memory_state import state

# Lock de escrita das tools (mesmo racional do comentário em
# memory_ops_memory.py — um asyncio.Lock por fluxo de escrita).
_tl_write_lock = asyncio.Lock()


# ── NEW: RAG de tools — seleção dinâmica das tools expostas ao LLM ────────────
#
# Fluxo esperado no orquestrador:
#   1. no startup: tools_register_batch(todas as tools, prune_missing=True) —
#      idempotente (tool inalterada não é re-embedada);
#   2. a cada turno, ANTES de montar o campo `tools` da chamada ao LLM:
#      tools_select(query) → só as tools devolvidas vão no payload;
#   3. depois do turno: tools_record_usage(tools realmente chamadas, em ordem,
#      + a query) → alimenta o grafo de co-uso/sequência e os exemplos aprendidos.
#
# Seleção = kNN em vetores (doc + exemplos + queries aprendidas) com max-pooling
# por tool, threshold permissivo, expansão por PPR no grafo de co-uso, e tools
# "core" sempre presentes. Confiança baixa no top-1 → rede mais larga.

def _tool_doc_text(name: str, description: str) -> str:
    return f"{name.replace('_', ' ')}: {description.strip()}"


async def _delete_tool_row(row: sqlite3.Row) -> None:
    loop = asyncio.get_event_loop()
    vids = state.tl_db.get_vector_ids(row["id"])
    if vids:
        await loop.run_in_executor(None, state.tl_index.remove_ids, set(vids))
    await loop.run_in_executor(None, state.tl_db.delete_tool, row["id"])


async def _register_tool_locked(req: ToolRegisterRequest) -> ToolRegisterResponse:
    name        = req.name.strip()
    description = req.description.strip()
    if not name or not description:
        raise MemoryToolError("tool precisa de name e description não vazios")
    examples: list[str] = []
    for ex in req.examples:
        ex = ex.strip()
        if ex and ex not in examples:
            examples.append(ex)
    examples = examples[:TOOLS_MAX_EXAMPLES]

    doc_text = _tool_doc_text(name, description)
    doc_hash = hashlib.sha256("\n".join([doc_text, *examples]).encode()).hexdigest()
    loop = asyncio.get_event_loop()

    existing = state.tl_db.get_by_name(name)
    if existing is not None and existing["doc_hash"] == doc_hash:
        if bool(existing["is_core"]) != req.core:
            state.tl_db.set_core(existing["id"], req.core)
        return ToolRegisterResponse(
            name=name, tool_id=existing["id"], status="unchanged",
            vectors=len(state.tl_db.get_vector_ids(existing["id"])),
        )

    texts = [doc_text, *examples]
    embeddings = await state.embed_engine.embed_passages(texts)   # antes de mutar qualquer coisa

    if existing is not None:
        tool_id = existing["id"]
        # troca doc + exemplos; queries "learned" (uso real) são preservadas
        old_ids = state.tl_db.get_vector_ids(tool_id, kinds=("doc", "example"))
        if old_ids:
            await loop.run_in_executor(None, state.tl_index.remove_ids, set(old_ids))
            state.tl_db.delete_vectors(old_ids)
        state.tl_db.update_tool(tool_id, description, req.core, doc_hash)
        status = "updated"
    else:
        tool_id = state.tl_db.insert_tool(name, description, req.core, doc_hash)
        status = "created"

    vids = [
        state.tl_db.insert_vector(tool_id, "doc" if i == 0 else "example", t)
        for i, t in enumerate(texts)
    ]
    await loop.run_in_executor(None, state.tl_index.add_batch, embeddings, vids)
    log.info(f"Tools: '{name}' {status} ({len(vids)} vetores, core={req.core})")
    return ToolRegisterResponse(name=name, tool_id=tool_id, status=status, vectors=len(vids))


async def tools_register(req: ToolRegisterRequest):
    async with _tl_write_lock:
        return await _register_tool_locked(req)


async def tools_register_batch(req: ToolRegisterBatchRequest):
    results: list[ToolRegisterResponse] = []
    pruned: list[str] = []
    async with _tl_write_lock:
        for item in req.tools:
            results.append(await _register_tool_locked(item))
        if req.prune_missing:
            keep = {ToolsDB._normalize_key(t.name) for t in req.tools}
            for row in state.tl_db.list_tools():
                if row["name_key"] not in keep:
                    await _delete_tool_row(row)
                    pruned.append(row["name"])
    if pruned:
        log.info(f"Tools: {len(pruned)} tools removidas do índice (prune_missing): {pruned}")
    return ToolRegisterBatchResponse(results=results, pruned=pruned)


async def tools_select(req: ToolSelectRequest):
    query = req.query.strip()
    if not query:
        raise MemoryToolError("query vazia")
    loop = asyncio.get_event_loop()

    query_emb = await state.embed_engine.embed_query_one(query)
    # sobre-amostra: vários vetores (doc + exemplos) apontam pra mesma tool
    hits = await loop.run_in_executor(
        None, state.tl_index.search, query_emb, max(req.top_k * 6, 30)
    )
    vec_to_tool = state.tl_db.get_tool_ids_by_vectors([v for v, _ in hits])
    best: dict[int, float] = {}
    for vid, score in hits:
        tid = vec_to_tool.get(vid)
        if tid is not None and score > best.get(tid, -1.0):
            best[tid] = score
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)

    top1 = ranked[0][1] if ranked else 0.0
    low_confidence = top1 < TOOLS_LOW_CONFIDENCE_SCORE
    if low_confidence:
        # confiança baixa: recall > precisão — rede mais larga, sem threshold
        primary = ranked[:TOOLS_FALLBACK_TOP_K]
    else:
        primary = [(t, s) for t, s in ranked if s >= req.min_score][:req.top_k]

    # tools alcançadas SÓ pelo grafo de co-uso/sequência (listar → ler → editar)
    related: list[tuple[int, float]] = []
    if req.include_related and primary:
        ctx = await _ppr_spread_on(state.tl_db, primary, loop)
        taken = {t for t, _ in primary}
        related = [(t, s) for t, s in ctx["related_raw"] if t not in taken][:TOOLS_MAX_RELATED]

    core_rows = state.tl_db.list_core()
    core_ids = {r["id"] for r in core_rows}

    rows = {r["id"]: r for r in state.tl_db.get_by_ids(
        [t for t, _ in primary] + [t for t, _ in related] + list(core_ids)
    )}
    out: list[ToolMatch] = []
    seen: set[int] = set()

    def push(tid: int, score: float, match_type: str):
        row = rows.get(tid)
        if row is None or tid in seen:
            return
        seen.add(tid)
        out.append(ToolMatch(
            name=row["name"], description=row["description"],
            score=round(score, 4), match_type=match_type,
        ))

    for tid, s in primary:
        push(tid, s, "primary")
    for tid, s in related:
        push(tid, s, "related")
    for tid in core_ids:
        push(tid, best.get(tid, 0.0), "core")

    log.info(
        f"tools_select query='{query[:50]}' top1={top1:.3f} low_conf={low_confidence} "
        f"primary={len(primary)} related={len(related)} core={len(core_ids)} -> {len(out)} tools"
    )
    return ToolSelectResponse(tools=out, top1_score=round(top1, 4), low_confidence=low_confidence)


def _record_tool_edges_sync(tool_ids: list[int]) -> None:
    now = time.time()
    for i in range(len(tool_ids)):
        for j in range(i + 1, len(tool_ids)):
            state.tl_db.upsert_edge(tool_ids[i], tool_ids[j], EDGE_LEARNING_RATE, now, EDGE_CO_ACTIVATION)
    for prev, nxt in zip(tool_ids, tool_ids[1:]):
        state.tl_db.upsert_edge(prev, nxt, EDGE_LEARNING_RATE, now, EDGE_TEMPORAL)


async def tools_record_usage(req: ToolUsageRequest):
    """Registra o uso real: (1) arestas de co-uso e de sequência no grafo de
    tools, (2) a query vira exemplo aprendido das tools usadas."""
    loop = asyncio.get_event_loop()
    ordered: list[int] = []
    for name in req.tools_used[:TOOLS_USAGE_MAX_SEQUENCE]:
        row = state.tl_db.get_by_name(name)
        if row is not None and (not ordered or ordered[-1] != row["id"]):
            ordered.append(row["id"])
    if not ordered:
        return {"recorded": False, "reason": "no_known_tools", "tools": 0, "learned": 0}

    await loop.run_in_executor(None, state.tl_db.mark_used, ordered)
    if len(ordered) > 1:
        await loop.run_in_executor(None, _record_tool_edges_sync, ordered)

    learned = 0
    query = req.query.strip()
    if query:
        async with _tl_write_lock:
            # `query` aqui vira um EXEMPLO ARMAZENADO no índice de tools
            # (insert_vector abaixo + tl_index.add), pra ser encontrado
            # depois por outras queries em tools_select — por isso é
            # embedado como passage, não como query, apesar do nome da
            # variável.
            emb = await state.embed_engine.embed_passage_one(query)
            for tid in dict.fromkeys(ordered):
                if state.tl_db.count_learned(tid) >= TOOLS_MAX_LEARNED_EXAMPLES:
                    continue
                existing_vids = state.tl_db.get_vector_ids(tid)
                near = await loop.run_in_executor(
                    None, state.tl_index.search_subset, emb, existing_vids, 1
                )
                if near and near[0][1] >= TOOLS_LEARN_DEDUP_SCORE:
                    continue
                vid = state.tl_db.insert_vector(tid, "learned", query)
                await loop.run_in_executor(None, state.tl_index.add, emb, vid)
                learned += 1
    return {"recorded": True, "tools": len(ordered), "learned": learned}


async def tools_list():
    rows = state.tl_db.list_tools()
    return {
        "total": len(rows),
        "tools": [
            {
                "name": r["name"], "description": r["description"],
                "core": bool(r["is_core"]), "vectors": r["vectors_count"],
                "use_count": r["use_count"], "last_used": r["last_used"],
            }
            for r in rows
        ],
    }


async def tools_delete(name: str):
    row = state.tl_db.get_by_name(name)
    if row is None:
        raise MemoryToolError(f"Tool '{name}' não encontrada")
    async with _tl_write_lock:
        await _delete_tool_row(row)
    return {"deleted": True, "name": row["name"]}


__all__ = [
    '_delete_tool_row',
    '_record_tool_edges_sync',
    '_register_tool_locked',
    '_tl_write_lock',
    '_tool_doc_text',
    'tools_delete',
    'tools_list',
    'tools_record_usage',
    'tools_register',
    'tools_register_batch',
    'tools_select',
]
