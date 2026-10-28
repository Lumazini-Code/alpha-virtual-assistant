# Alpha AI (AVA) — Tool Tips: dicas de execução gravadas PELO MODELO
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
memory_ops_tips.py — Tool Tips: dicas de execução gravadas PELO MODELO.

Diferença crucial pro antigo sistema de snippets (REMOVIDO do mcp_manager):
o snippet era descrição CURADA por humano que substituía a description real;
a dica é NOTA DE USO REAL do próprio modelo ("falha com payload > 5MB",
"chame X antes de Y"), ADITIVA à description — a fonte da verdade da tool
continua sendo o MCP. Gravação via tool_tip_add / tool_tip_remove (alpha-mcp);
leitura em lote pelo orquestrador SÓ das tools carregadas na rodada (o bloco
"Tool tips" vai no tail dinâmico do system prompt — prefixo cacheável intacto).

Cap firme (TOOLS_TIP_MAX_PER_TOOL): excedente REJEITA com orientação — o
modelo cura as próprias dicas; nada é apagado por trás dele. Dedup SEMÂNTICO
por embedding (mesmo motor do ToolsDB, e5 passage-side): limiar ALTO por
desenho (TOOLS_TIP_DEDUP_EMBED) — funde só reescrita quase literal (o bug
real: paráfrase morfológica); paráfrase distante vira duplicata VISÍVEL que
o modelo cura (cap-rejeição), pois fundir dicas distintas silenciosamente é
destrutivo (o e5-small não separa as famílias: "query vs payload" dá 0.975).

Escrita serializa na MESMA _tl_write_lock do registro de tools (uma dica não
pode ser gravada no meio de um re-register/prune da tool alvo).
"""

import numpy as np

from memory_config import (
    TOOLS_TIP_DEDUP_EMBED,
    TOOLS_TIP_MAX_CHARS,
    TOOLS_TIP_MAX_PER_TOOL,
    log,
)
from memory_models import (
    MemoryToolError,
    ToolTipAddRequest,
    ToolTipAddResponse,
    ToolTipsListResponse,
    ToolTipRemoveRequest,
)
from memory_ops_tools import _tl_write_lock
from memory_state import state


def _tip_text_norm(text: str) -> str:
    """Normalização p/ dedup exato (fallback se o embed engine falhar)."""
    return " ".join(text.strip().lower().split())


async def _tip_find_dup(name_key: str, tip: str) -> int | None:
    """Dica ativa da tool que já diz a mesma coisa (a nova deve ATUALIZAR em
    vez de gastar slot). Exato primeiro; se o embed engine estiver de pé,
    dedup semântico por cosine (vetores normalizados → dot). Falha de embed →
    só o exato conta (o cap continua protegendo contra duplicatas)."""
    rows = state.tl_db.tips_active_for_tools([name_key])
    if not rows:
        return None
    exact = _tip_text_norm(tip)
    for r in rows:
        if _tip_text_norm(r["text"]) == exact:
            return r["tip_id"]
    try:
        embs = await state.embed_engine.embed_passages(
            [tip] + [r["text"] for r in rows]
        )
        sims = embs[1:] @ embs[0]
        best = int(np.argmax(sims))
        if float(sims[best]) >= TOOLS_TIP_DEDUP_EMBED:
            return rows[best]["tip_id"]
    except Exception as e:
        log.warning(f"ToolTips: dedup semântico indisponível ({type(e).__name__}: {e}) — só exato")
    return None


async def tool_tip_add(req: ToolTipAddRequest) -> ToolTipAddResponse:
    name = req.tool.strip()
    tip = req.tip.strip()
    if not name:
        raise MemoryToolError("tool vazia — passe o nome completo ('servidor__tool')")
    if not tip:
        raise MemoryToolError("tip vazia")
    if len(tip) > TOOLS_TIP_MAX_CHARS:
        raise MemoryToolError(
            f"tip com {len(tip)} chars — limite {TOOLS_TIP_MAX_CHARS}. "
            "Reescreva em 1 frase imperativa e geral (não o caso específico desta conversa)."
        )

    async with _tl_write_lock:
        row = state.tl_db.get_by_name(name)
        if row is None:
            raise MemoryToolError(
                f"Tool '{name}' não existe no ToolsDB — só é possível dicas de "
                "tools MCP registradas/sincronizadas (rode o sync do servidor primeiro)."
            )

        # Dedup SEMÂNTICO (ver _tip_find_dup) — duplicata vira UPDATE (o modelo
        # refinou a própria dica; não deve gastar um slot novo).
        similar_id = await _tip_find_dup(row["name_key"], tip)

        count = state.tl_db.tip_active_count(row["id"])
        if similar_id is not None:
            state.tl_db.tip_update_text(similar_id, tip)
            status = "updated"
        else:
            # Cap ANTES do insert: excedente rejeita com orientação explícita —
            # o modelo decide o que remover (nada é evictado por trás dele).
            if count >= TOOLS_TIP_MAX_PER_TOOL:
                raise MemoryToolError(
                    f"Tool '{name}' já tem {count} dicas (limite {TOOLS_TIP_MAX_PER_TOOL}). "
                    "Remova a mais fraca/desatualizada com tool_tip_remove antes de gravar outra."
                )
            similar_id = state.tl_db.tip_insert(row["id"], tip)
            status = "created"
            count += 1

    log.info(f"ToolTips: '{name}' dica #{similar_id} {status} ({count} ativas)")
    return ToolTipAddResponse(tool=name, tip_id=int(similar_id), status=status, tips_count=count)


async def tool_tips_list(names: str) -> ToolTipsListResponse:
    """Batch p/ o orquestrador: `names` = graph_names separados por vírgula.
    Devolve SÓ tools que têm dica ativa (o bloco do prompt não renderiza vazio)."""
    requested = [n.strip() for n in names.split(",") if n.strip()]
    if not requested:
        return ToolTipsListResponse(tips={})
    name_keys = [state.tl_db._normalize_key(n) for n in requested]
    rows = state.tl_db.tips_active_for_tools(name_keys)
    grouped: dict[str, list] = {}
    for r in rows:
        grouped.setdefault(r["name"], []).append({
            "id": r["tip_id"], "text": r["text"], "created_at": r["created_at"],
        })
    return ToolTipsListResponse(tips=grouped)


async def tool_tip_remove(req: ToolTipRemoveRequest) -> dict:
    name = req.tool.strip()
    if not name:
        raise MemoryToolError("tool vazia — passe o nome completo ('servidor__tool')")
    row = state.tl_db.get_by_name(name)
    if row is None:
        raise MemoryToolError(f"Tool '{name}' não existe no ToolsDB")
    tip = state.tl_db.tip_get(req.tip_id)
    if tip is None or tip["tool_id"] != row["id"] or not tip["active"]:
        raise MemoryToolError(
            f"Dica #{req.tip_id} não encontrada em '{name}' — confira o id no bloco "
            "'Tool tips' do system prompt (ids mudam se a dica foi regravada)."
        )
    async with _tl_write_lock:
        state.tl_db.tip_deactivate(req.tip_id)
    log.info(f"ToolTips: '{name}' dica #{req.tip_id} retratada")
    return {"removed": True, "tool": name, "tip_id": req.tip_id}


__all__ = [
    'tool_tip_add',
    'tool_tip_remove',
    'tool_tips_list',
]
