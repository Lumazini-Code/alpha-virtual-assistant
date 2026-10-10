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

from memory_config import log
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
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


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
        steps_json = json.dumps([s.model_dump() for s in req.steps], ensure_ascii=False)
        plan_id = state.plans_db.insert_plan(req.name, req.description, steps_json)
    log.info(f"Plans: '{req.name}' criado (id={plan_id}, {len(req.steps)} passos)")
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
        steps_json = (
            json.dumps([s.model_dump() for s in req.steps], ensure_ascii=False)
            if req.steps is not None else None
        )
        corrections_json = None
        if req.corrections is not None:
            current = json.loads(row["corrections_json"])
            current.update(req.corrections)
            corrections_json = json.dumps(current, ensure_ascii=False)
        version = state.plans_db.update_plan(
            plan_id, req.description, steps_json, corrections_json
        )
    if version == 0:
        raise MemoryToolError(f"Plano id={plan_id} não encontrado")
    log.info(f"Plans: id={plan_id} atualizado → versão {version}")
    return PlanUpdateResponse(plan_id=plan_id, version=version, updated=True)


async def plan_delete(req: PlanDeleteRequest) -> PlanDeleteResponse:
    async with _plans_write_lock:
        row = _resolve_row(req.plan_id, req.name)
        state.plans_db.delete_plan(row["id"])
    log.info(f"Plans: '{row['name']}' (id={row['id']}) apagado")
    return PlanDeleteResponse(deleted=True, plan_id=row["id"])


__all__ = [
    '_plans_write_lock',
    '_row_to_entry',
    '_resolve_row',
    'plan_create',
    'plan_delete',
    'plan_list',
    'plan_read',
    'plan_update',
]
