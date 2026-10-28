# Alpha AI (AVA) — Handlers dos planos de execução híbridos
# Copyright (C) 2026  Felipe lumazini
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

"""
memory_ops_plans.py — Handlers dos planos de execução híbridos.

/plans/write (criar), /plans/read, /plans/list, /plans/update (versiona),
/plans/delete. Storage em PlansDB (SQLite puro — sem vetores, planos são
buscados por nome/id, não por similaridade).

NEW (projetos): leitura mesclada — /plans/read/list resolvem ids compostas
(offset +10^12) para o escopo geral e fundem os planos do escopo ativo com
os do geral (ids do geral saem compostas + tag `scope`). Criação acontece
sempre no escopo ativo.

Lock _plans_write_lock vive aqui — mesmo padrão dos demais módulos de handlers.
"""

import asyncio
import json
import time

from memory_config import (
    EDGE_CO_ACTIVATION,
    EDGE_LEARNING_RATE,
    EDGE_TEMPORAL,
    GENERAL_SCOPE,
    encode_ref,
    is_general_ref,
    log,
    ref_real_id,
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


def _row_to_entry(row, encode: bool = False, scope: str | None = None) -> PlanEntry:
    return PlanEntry(
        plan_id=encode_ref(scope, row["id"]) if encode else row["id"],
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
        # ── NEW (projetos): escopo dono (só na leitura mesclada) ──
        scope=scope,
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
    """Resolve (plans_db, row) do plano — offset-aware (projetos).

    Id composta (offset +10^12) → escopo geral. Id crua → escopo ativo
    primeiro; não achou → sonda o geral (update/delete de plano criado em
    outro modo permanecem no dono). Por nome → prefere o escopo ativo, cai
    no geral. Erro (MemoryToolError) se nada for encontrado."""
    gen = state.scopes.get(GENERAL_SCOPE)
    if plan_id is not None:
        if is_general_ref(plan_id):
            real = ref_real_id(plan_id)
            if gen is None:
                raise MemoryToolError(f"Plano id={plan_id} não encontrado")
            row = gen.plans_db.get_by_id(real)
            if row is None:
                raise MemoryToolError(f"Plano id={plan_id} não encontrado")
            return gen.plans_db, row
        row = state.plans_db.get_by_id(plan_id)
        if row is not None:
            return state.plans_db, row
        if gen is not None:
            row = gen.plans_db.get_by_id(plan_id)
            if row is not None:
                return gen.plans_db, row
        raise MemoryToolError(f"Plano id={plan_id} não encontrado")
    if name:
        row = state.plans_db.get_by_name(name)
        if row is not None:
            return state.plans_db, row
        if gen is not None:
            row = gen.plans_db.get_by_name(name)
            if row is not None:
                return gen.plans_db, row
        raise MemoryToolError(f"Plano '{name}' não encontrado")
    raise MemoryToolError("Informe plan_id ou name para identificar o plano")


def _plan_out_id(db, real_id: int) -> int:
    """Id de RESPOSTA: crua quando o plano é do escopo ativo; composta
    (offset) quando é do geral com projeto ativo (leitura mesclada)."""
    if db is not state.plans_db and state.active_scope != GENERAL_SCOPE:
        return encode_ref(GENERAL_SCOPE, real_id)
    return real_id


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
    plans_db, row = _resolve_row(req.plan_id, req.name)
    merged = state.active_scope != GENERAL_SCOPE
    scope_tag = GENERAL_SCOPE if (merged and plans_db is not state.plans_db) else None
    encode = scope_tag is not None
    return PlanReadResponse(
        plan=_row_to_entry(row, encode=encode, scope=scope_tag)
    )


async def plan_list() -> PlanListResponse:
    """Leitura MESCLADA (projetos): planos do escopo ativo (ids cruas) + os
    do geral (ids compostas + scope="general"). Modo geral → comportamento
    idêntico ao de antes."""
    merged = state.active_scope != GENERAL_SCOPE
    entries = []
    # list_plans não traz steps/corrections (leve); completa com get_by_id
    for row in state.plans_db.list_plans():
        full = state.plans_db.get_by_id(row["id"])
        if full is not None:
            entries.append(_row_to_entry(full))
    if merged:
        gen = state.scopes.get(GENERAL_SCOPE)
        if gen is not None:
            for row in gen.plans_db.list_plans():
                full = gen.plans_db.get_by_id(row["id"])
                if full is not None:
                    entries.append(_row_to_entry(full, encode=True, scope=GENERAL_SCOPE))
    return PlanListResponse(total=len(entries), plans=entries)


async def plan_update(req: PlanUpdateRequest) -> PlanUpdateResponse:
    async with _plans_write_lock:
        plans_db, row = _resolve_row(req.plan_id, req.name)
        plan_id = row["id"]
        # renomeio: checa duplicidade ANTES de gravar (name_key é UNIQUE);
        # renomear para o mesmo nome (key) é no-op e não falha. A checagem é
        # no DB DONO do plano (o update não troca o plano de escopo).
        new_name = req.new_name.strip() if req.new_name else None
        if new_name:
            existing = plans_db.get_by_name(new_name)
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
        version = plans_db.update_plan(
            plan_id, req.description, steps_json, corrections_json, name=new_name,
            person_id=person_id, tools_json=tools_json,
        )
    if req.steps is not None and tool_ids:
        _link_plan_tools(tool_ids)
    if version == 0:
        raise MemoryToolError(f"Plano id={plan_id} não encontrado")
    log.info(f"Plans: id={plan_id} atualizado → versão {version}"
             + (f" (renomeado para {new_name!r})" if new_name else ""))
    return PlanUpdateResponse(plan_id=_plan_out_id(plans_db, plan_id), version=version, updated=True)


async def plan_delete(req: PlanDeleteRequest) -> PlanDeleteResponse:
    async with _plans_write_lock:
        plans_db, row = _resolve_row(req.plan_id, req.name)
        plan_id = row["id"]
        plans_db.delete_plan(plan_id)
    log.info(f"Plans: '{row['name']}' (id={plan_id}) apagado")
    return PlanDeleteResponse(deleted=True, plan_id=_plan_out_id(plans_db, plan_id))


__all__ = [
    '_link_plan_tools',
    '_plans_write_lock',
    '_plan_out_id',
    '_resolve_tool_ids',
    '_row_to_entry',
    '_resolve_row',
    'plan_create',
    'plan_delete',
    'plan_list',
    'plan_read',
    'plan_update',
]
