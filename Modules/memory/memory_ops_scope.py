# Alpha AI (AVA) — Handlers de escopo (modo geral × projetos)
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
memory_ops_scope.py — Handlers de escopo (modo geral × projetos).

GET /project/current (estado do escopo ativo), POST /project/switch (troca —
abre o escopo se ainda não estiver em cache) e DELETE /project/{slug} (fecha
e APAGA o diretório db/<slug>/ inteiro). A troca em si vive em
memory_state.switch_scope/delete_scope — aqui é só validação + modelos.

Consumidores: orchestrator (re-assert a cada turno — a memory é reiniciada
pelo start.sh e o escopo ativo precisa ser restaurado) e TUI (/projects).
"""

from memory_config import (
    GENERAL_SCOPE,
    is_valid_scope,
    log,
    resolve_scope_paths,
)
from memory_models import (
    MemoryToolError,
    ScopeDeleteResponse,
    ScopeInfoResponse,
    ScopeSwitchRequest,
)
from memory_state import state, switch_scope, delete_scope


async def project_current() -> ScopeInfoResponse:
    """Estado do escopo ativo (GET /project/current)."""
    sc = state.scope
    return ScopeInfoResponse(
        scope=state.active_scope,
        db_dir=sc.paths.db_dir if sc is not None else resolve_scope_paths(state.active_scope).db_dir,
    )


async def project_switch(req: ScopeSwitchRequest) -> ScopeInfoResponse:
    """Troca o escopo ativo (POST /project/switch). "general" = modo geral;
    slug = abre (se preciso) e ativa o conjunto de artefatos do projeto."""
    scope = (req.scope or "").strip()
    if not scope:
        raise MemoryToolError("scope vazio")
    if scope != GENERAL_SCOPE and not is_valid_scope(scope):
        raise MemoryToolError(
            f"scope inválido: {scope!r} — use 'general' ou slug [a-z0-9][a-z0-9_-]{{0,63}}"
        )
    previous = state.active_scope
    result = await switch_scope(scope)
    sc = state.scope
    return ScopeInfoResponse(
        scope=scope,
        db_dir=sc.paths.db_dir if sc is not None else resolve_scope_paths(scope).db_dir,
        previous=result.get("previous", previous),
        opened=result.get("opened"),
    )


async def project_delete(slug: str) -> ScopeDeleteResponse:
    """Remove um projeto (DELETE /project/{slug}): fecha o escopo (se aberto)
    e apaga o diretório db/<slug>/ INTEIRO — memórias LT, grafo Hebbiano,
    planos, arquivos indexados, LanceDB e log de ativação. "general" e o
    escopo ativo são recusados (desative antes)."""
    scope = (slug or "").strip()
    if not scope:
        raise MemoryToolError("slug vazio")
    if scope == GENERAL_SCOPE:
        raise MemoryToolError("O escopo 'general' não pode ser removido")
    if not is_valid_scope(scope):
        raise MemoryToolError(
            f"slug inválido: {scope!r} — use [a-z0-9][a-z0-9_-]{{0,63}}"
        )
    await delete_scope(scope)
    log.info(f"Projeto '{scope}' removido")
    return ScopeDeleteResponse(scope=scope, deleted=True)


__all__ = [
    'project_current',
    'project_delete',
    'project_switch',
]
