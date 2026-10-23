from __future__ import annotations

"""
memory_ops_plans.py — Handlers dos planos de execução híbridos.

/plans/write (criar), /plans/read, /plans/list, /plans/update (versiona),
/plans/delete. Storage em PlansDB (SQLite puro — sem vetores, planos são
buscados por nome/id, não por similaridade).

Lock _plans_write_lock vive aqui — mesmo padrão dos demais módulos de handlers.
"""

import asyncio
import json
import time

from memory_config import (
    EDGE_CO_ACTIVATION,
    EDGE_LEARNING_RATE,
    EDGE_TEMPORAL,
    log,
)
from memory_models import (
    MemoryToolError,
    PlanCreateRequest,
    PlanCreateResponse,
    PlanDeleteRequest,
    PlanDeleteResponse,
    PlanEntry,
    PlanListResponse,
    PlanReadRequest,
    PlanReadResponse,
    PlanStep,
    PlanUpdateRequest,
    PlanUpdateResponse,
)
from memory_state import state

_plans_write_lock = asyncio.Lock()


def _row_to_entry(row) -> PlanEntry:
    return PlanEntry(
        plan_id=row["id"],
        name=row["name"],
        description=row["description"],
        version=row["version"],
        steps=[PlanStep(**s) for s in json.loads(row["steps_json"])],
        corrections=json.loads(row["corrections_json"]),
        # ── NEW (people cards / tools ↔ planos): vínculos persistidos.
        person_id=row["person_id"],
        tool_ids=json.loads(row["tools_json"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _resolve_tool_ids(tool_names: list[str]) -> list[int]:
    """NEW (tools ↔ planos): resolve nomes de tools (tool_name dos steps,
    distintos e em ordem de aparição) para ids no ToolsDB. Nome desconhecido
    → warning e ignora (não bloqueia o plano); tool sem cadastro não tem nó
    no grafo de tools."""
    resolved: list[int] = []
    for name in tool_names:
        tool = state.tl_db.get_by_name(name)
        if tool is None:
            log.warning(f"Plans: tool '{name}' não cadastrada — vínculo ignorado")
            continue
        if tool["id"] not in resolved:
            resolved.append(tool["id"])
    return resolved


def _link_plan_tools(tool_ids: list[int]) -> None:
    """NEW (tools ↔ planos): grava as arestas de co-uso no grafo de tools.
    temporal_precedence (dirigida: a tool do passo anterior veio antes) entre
    tools consecutivas + co_activation (simétrica) entre TODAS as pairs —
    mesmo padrão de _record_tool_edges_sync em memory_ops_tools.py."""
    now = time.time()
    for i in range(len(tool_ids)):
        for j in range(i + 1, len(tool_ids)):
            state.tl_db.upsert_edge(
                tool_ids[i], tool_ids[j], EDGE_LEARNING_RATE, now, EDGE_CO_ACTIVATION
            )
    for prev, nxt in zip(tool_ids, tool_ids[1:]):
        state.tl_db.upsert_edge(
            prev, nxt, EDGE_LEARNING_RATE, now, EDGE_TEMPORAL
        )
    if tool_ids:
        log.info(f"Plans: arestas de co-uso gravadas para tools {tool_ids}")


def _resolve_row(plan_id: int | None, name: str | None):
    if plan_id is not None:
        row = state.plans_db.get_by_id(plan_id)
        if row is None:
            raise MemoryToolError(f"Plano id={plan_id} não encontrado")
        return row
    if name:
        row = state.plans_db.get_by_name(name)
        if row is None:
            raise MemoryToolError(f"Plano '{name}' não encontrado")
        return row
    raise MemoryToolError("Informe plan_id ou name para identificar o plano")


async def plan_create(req: PlanCreateRequest) -> PlanCreateResponse:
    async with _plans_write_lock:
        existing = state.plans_db.get_by_name(req.name)
        if existing is not None:
            raise MemoryToolError(
                f"Plano '{req.name}' já existe (id={existing['id']}) — use update"
            )
        # ── NEW (people cards): valida o card antes de gravar — inválido →
        # warning e grava sem vínculo.
        person_id = req.person_id
        if person_id is not None and state.pd_db.get_person_by_id(person_id) is None:
            log.warning(f"people: person_id #{person_id} não existe — plano gravado sem vínculo")
            person_id = None
        # ── NEW (tools ↔ planos): resolve tool_name dos steps → tool_ids e
        # persiste em tools_json (fonte única do conjunto de tools do plano).
        tool_ids = _resolve_tool_ids([s.tool_name for s in req.steps if s.tool_name])
        steps_json = json.dumps([s.model_dump() for s in req.steps], ensure_ascii=False)
        plan_id = state.plans_db.insert_plan(
            req.name, req.description, steps_json,
            person_id=person_id, tools_json=json.dumps(tool_ids),
        )
    if tool_ids:
        _link_plan_tools(tool_ids)
    log.info(f"Plans: '{req.name}' criado (id={plan_id}, {len(req.steps)} passos, tools={tool_ids})")
    return PlanCreateResponse(plan_id=plan_id, version=1, created=True)


async def plan_read(req: PlanReadRequest) -> PlanReadResponse:
    row = _resolve_row(req.plan_id, req.name)
    return PlanReadResponse(plan=_row_to_entry(row))


async def plan_list() -> PlanListResponse:
    rows = state.plans_db.list_plans()
    # list_plans não traz steps/corrections (leve); completa com get_by_id
    entries = []
    for row in rows:
        full = state.plans_db.get_by_id(row["id"])
        if full is not None:
            entries.append(_row_to_entry(full))
    return PlanListResponse(total=len(entries), plans=entries)


async def plan_update(req: PlanUpdateRequest) -> PlanUpdateResponse:
    async with _plans_write_lock:
        row = _resolve_row(req.plan_id, req.name)
        plan_id = row["id"]
        # renomeio: checa duplicidade ANTES de gravar (name_key é UNIQUE);
        # renomear para o mesmo nome (key) é no-op e não falha.
        new_name = req.new_name.strip() if req.new_name else None
        if new_name:
            existing = state.plans_db.get_by_name(new_name)
            if existing is not None and existing["id"] != plan_id:
                raise MemoryToolError(
                    f"Plano '{new_name}' já existe (id={existing['id']}) — nome duplicado"
                )
        steps_json = (
            json.dumps([s.model_dump() for s in req.steps], ensure_ascii=False)
            if req.steps is not None else None
        )
        corrections_json = None
        if req.corrections is not None:
            current = json.loads(row["corrections_json"])
            current.update(req.corrections)
            corrections_json = json.dumps(current, ensure_ascii=False)
        # ── NEW (people cards): novo vínculo com card de pessoa. None = mantém
        # o atual (semântica do update_plan); id inexistente → warning e mantém.
        person_id = req.person_id
        if person_id is not None and state.pd_db.get_person_by_id(person_id) is None:
            log.warning(f"people: person_id #{person_id} não existe — vínculo do plano mantido")
            person_id = None
        # ── NEW (tools ↔ planos): steps mudaram → re-resolve tools_json e
        # regrava as arestas de co-uso com o conjunto novo.
        tools_json = None
        if req.steps is not None:
            tool_ids = _resolve_tool_ids([s.tool_name for s in req.steps if s.tool_name])
            tools_json = json.dumps(tool_ids)
        version = state.plans_db.update_plan(
            plan_id, req.description, steps_json, corrections_json, name=new_name,
            person_id=person_id, tools_json=tools_json,
        )
    if req.steps is not None and tool_ids:
        _link_plan_tools(tool_ids)
    if version == 0:
        raise MemoryToolError(f"Plano id={plan_id} não encontrado")
    log.info(f"Plans: id={plan_id} atualizado → versão {version}"
             + (f" (renomeado para {new_name!r})" if new_name else ""))
    return PlanUpdateResponse(plan_id=plan_id, version=version, updated=True)


async def plan_delete(req: PlanDeleteRequest) -> PlanDeleteResponse:
    async with _plans_write_lock:
        row = _resolve_row(req.plan_id, req.name)
        state.plans_db.delete_plan(row["id"])
    log.info(f"Plans: '{row['name']}' (id={row['id']}) apagado")
    return PlanDeleteResponse(deleted=True, plan_id=row["id"])


__all__ = [
    '_link_plan_tools',
    '_plans_write_lock',
    '_resolve_tool_ids',
    '_row_to_entry',
    '_resolve_row',
    'plan_create',
    'plan_delete',
    'plan_list',
    'plan_read',
    'plan_update',
]
