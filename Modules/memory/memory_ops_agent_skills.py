from __future__ import annotations

"""
memory_ops_agent_skills.py — Handlers do RAG de Agent Skills (SKILL.md).

/agent-skills/select — kNN (mesmo índice vetorial do ToolsDB) + PPR no grafo
de co-uso, FILTRADO pelo namespace AGENT_SKILL_PREFIX. As skills compartilham
a tabela `tools` com as MCP tools (cada uma é uma linha "agentskill__<name>"
registrada pelo orchestrator via /tools/register — idempotente por doc_hash,
com self-heal e exemplos aprendidos de graça), mas:

  - o seletor de SERVIDORES MCP (/tools/select) EXCLUI o prefixo — skills não
    competem por slots de servidores (ver memory_ops_tools.py);
  - este endpoint FILTRA pelo prefixo — só skills voltam;
  - o uso real é registrado pelo MESMO /tools/record-usage (nomes com
    prefixo na lista) → arestas de co-uso skill↔tool + query vira exemplo
    aprendido pros dois mundos.

Sem cota por servidor aqui: cada skill é seu próprio grupo (o peso do
"primary" é o cap de SKILLS_TOP_K no system prompt, não quota de schema).
"""

import asyncio

from memory_config import (
    AGENT_SKILL_PREFIX,
    SKILLS_FALLBACK_TOP_K,
    TOOLS_LOW_CONFIDENCE_SCORE,
    TOOLS_MAX_RELATED,
    log,
)
from memory_models import (
    AgentSkillMatch,
    AgentSkillSelectRequest,
    AgentSkillSelectResponse,
    MemoryToolError,
)
from memory_search import _ppr_spread_on
from memory_state import state


async def agent_skills_select(req: AgentSkillSelectRequest):
    """Seleção semântica de Agent Skills para o pedido: kNN nas descrições
    (mesmo índice LanceDB do ToolsDB, max-pooling por skill) + expansão por
    PPR no grafo de co-uso (skills↔tools↔skills) + skills "core" (always-on)
    sempre presentes. Devolve `name` SEM o prefixo — é o argumento da tool
    `use_skill` do orchestrator."""
    query = req.query.strip()
    if not query:
        raise MemoryToolError("query vazia")
    loop = asyncio.get_event_loop()

    query_emb = await state.embed_engine.embed_query_one(query)
    # sobre-amostra: doc + exemplos + queries aprendidas apontam p/ a mesma skill
    hits = await loop.run_in_executor(
        None, state.tl_index.search, query_emb, max(req.top_k * 6, 30)
    )
    vec_to_skill = state.tl_db.get_tool_ids_by_vectors([v for v, _ in hits])
    best: dict[int, float] = {}
    for vid, score in hits:
        tid = vec_to_skill.get(vid)
        if tid is not None and score > best.get(tid, -1.0):
            best[tid] = score
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)

    top1 = ranked[0][1] if ranked else 0.0
    low_confidence = top1 < TOOLS_LOW_CONFIDENCE_SCORE
    if low_confidence:
        # confiança baixa: recall > precisão — rede mais larga, sem threshold
        primary = [(t, s) for t, s in ranked][:SKILLS_FALLBACK_TOP_K]
    else:
        primary = [(t, s) for t, s in ranked if s >= req.min_score][:req.top_k]

    # skills alcançadas SÓ pelo grafo (co-uso com MCP tools que o kNN não pega)
    related_raw: list[tuple[int, float]] = []
    if req.include_related and primary:
        ctx = await _ppr_spread_on(state.tl_db, primary, loop)
        taken = {t for t, _ in primary}
        related_raw = [(t, s) for t, s in ctx["related_raw"] if t not in taken][:TOOLS_MAX_RELATED]

    # skills "core" (always-on — o orchestrator registra com core=True)
    core_ids = {
        r["id"] for r in state.tl_db.list_core() if r["name"].startswith(AGENT_SKILL_PREFIX)
    }

    rows = {r["id"]: r for r in state.tl_db.get_by_ids(
        [t for t, _ in ranked] + [t for t, _ in related_raw] + list(core_ids)
    )}
    related = [(t, s) for t, s in related_raw
               if t in rows and rows[t]["name"].startswith(AGENT_SKILL_PREFIX)]

    out: list[AgentSkillMatch] = []
    seen: set[int] = set()

    def push(tid: int, score: float, match_type: str):
        row = rows.get(tid)
        if row is None or tid in seen or not row["name"].startswith(AGENT_SKILL_PREFIX):
            return
        seen.add(tid)
        out.append(AgentSkillMatch(
            name=row["name"][len(AGENT_SKILL_PREFIX):],
            description=row["description"],
            score=round(score, 4), match_type=match_type,
        ))

    for tid, s in primary:
        push(tid, s, "primary")
    for tid, s in related:
        push(tid, s, "related")
    for tid in core_ids:
        push(tid, best.get(tid, 0.0), "core")

    log.info(
        f"agent_skills_select query='{query[:50]}' top1={top1:.3f} low_conf={low_confidence} "
        f"primary={len(primary)} related={len(related)} core={len(core_ids)} -> {len(out)} skills"
    )
    return AgentSkillSelectResponse(skills=out, top1_score=round(top1, 4), low_confidence=low_confidence)


__all__ = ['agent_skills_select']
