# Alpha AI (AVA) — Handlers de leitura/escrita de memória (MCP tools)
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
memory_ops_memory.py — Handlers de leitura/escrita de memória (MCP tools).

/write, /write_batch, /write_st, /read, /read_st, /session (clear) e o
pipeline de gravação de longo prazo (_store_long_term_text: dedup exata +
semântica, confirmação por reranker cross-encoder, action=update, arestas
tipadas via link_to/link_type).

Locks de escrita de LT e ST vivem aqui (os de dicionários/tools ficam nos
seus respectivos módulos de handlers).
"""

import asyncio
import sqlite3
import time
from typing import Literal, Optional

from memory_config import (
    AUTO_LINK_ENABLED,
    AUTO_LINK_INITIAL_WEIGHT,
    AUTO_LINK_MAX_EDGES,
    AUTO_LINK_MIN_COSINE,
    AUTO_LINK_MIN_CROSS,
    AUTO_LINK_TOP_K,
    DEDUP_THRESHOLD,
    EDGE_CO_ACTIVATION,
    EDGE_LEARNING_RATE,
    EDGE_TEMPORAL,
    GENERAL_SCOPE,
    IF_MIN_SCORE_READ,
    READ_MIN_SCORE_STRICT,
    READ_RERANK_COSINE_WEIGHT,
    READ_RERANK_CROSS_WEIGHT,
    READ_RERANK_ENABLED,
    READ_RERANK_TOP_N,
    READ_TOP_K_FINAL,
    READ_TOP_K_FINAL_MAX,
    READ_TOTAL_MAX_CHARS,
    READ_TOTAL_MAX_CHARS_MAX,
    RERANK_DUPLICATE_SCORE,
    RERANK_UPDATE_SCORE,
    ST_DEDUP_WINDOW_S,
    ST_READ_DEFAULT_PAIRS,
    UPDATE_SIM_THRESHOLD,
    VS_MIN_SCORE,
    encode_ref,
    is_general_ref,
    log,
    ref_real_id,
)
from memory_models import (
    MemoryEntry,
    MemoryToolError,
    ReadRequest,
    ReadResponse,
    ReadSTRequest,
    ReadSTResponse,
    Turn,
    WriteBatchRequest,
    WriteBatchResponse,
    WriteRequest,
    WriteResponse,
    WriteSTRequest,
    WriteSTResponse,
)
from memory_search import (
    _apply_token_budget,
    _build_context_block,
    _build_if_entries,
    _build_lt_entries,
    _build_st_entries,
    _build_vs_entries,
    _classify_query,
    _fuse_max,
    _hebbian_link_task,
    _lateral_inhibition_filter,
    _ppr_reinforce_task,
    _ppr_spread_on,
    _search_lt_dual_on,
    _search_lt_expanded_on,
    _search_lt_plain,
    _search_lt_segmented_on,
    _search_st_dual,
    _search_st_expanded,
    _search_st_plain,
    _search_st_segmented,
    _split_query_segments,
    lt_views,
)
from memory_state import state

# ── POST /write ────────────────────────────────────────────────────────────────

# ── Locks de escrita (ITEM 5 da revisão) ───────────────────────────────────────
# Cada fluxo de escrita "check existe? → embed → dedup semântico → insert" tem
# um `await` (a chamada de embedding) no meio da checagem. Sem serializar essa
# sequência, duas escritas quase simultâneas do mesmo conteúdo podem passar as
# duas pela checagem de duplicata antes de qualquer uma delas inserir —
# resultando em duplicata (exact ou semântica) ou, no caso de `text_hash`/
# `person_key`/`concept_key` serem UNIQUE, uma exceção não tratada no insert.
# Um `asyncio.Lock` por fluxo elimina essa corrida sem serializar o processo
# inteiro (cada tipo de memória tem o seu).
_lt_write_lock = asyncio.Lock()
_st_write_lock = asyncio.Lock()


async def _auto_link_task(new_id: int, embedding, text: str,
                          lt_db=None, lt_index=None) -> None:
    """Auto-link na ESCRITA (grafo Hebbiano) — liga a memória recém-gravada
    aos seus vizinhos semânticos já existentes, sem esperar uma leitura
    futura que co-ative os dois.

    NEW (projetos): `lt_db`/`lt_index` = grafo/índice DO ESCOPO da memória
    nova (None = escopo ativo). Arestas nunca cruzam escopos — o auto-link
    só considera vizinhos do MESMO escopo.

    Pipeline (fire-and-forget — falhas nunca derrubam a escrita):
      1. top-K vizinhos no FAISS (o embedding da nova memória já estava em
         mãos — zero custo de embed extra); a própria memória, que já está
         no índice, é excluída. O piso de cosine é só PRÉ-FILTRO de custo:
         o bi-encoder (E5-small) é anisotrópico — não-relacionados saem com
         cosine ~0.78 (medido) — então QUEM DECIDE é o cross-encoder;
      2. gate OBRIGATÓRIO por cross-encoder (score_many, 1 round-trip):
         ~0.000 p/ não-relacionados (medido em produção); se o reranker
         estiver indisponível, o link NÃO acontece (conservador);
      3. upsert_edge(co_activation, initial_weight=AUTO_LINK_INITIAL_WEIGHT)
         — cria ou reforça; decay natural (90 dias) poda o que não for usado.

    INSTRUMENTAÇÃO (solução 3 da calibração): sempre loga UMA linha INFO por
    escrita com candidatos + scores top — o gate que filtra tudo deixou de
    ser silencioso, e recalibrar thresholds vira tarefa orientada a dados.
    """
    try:
        loop = asyncio.get_event_loop()
        if lt_db is None:
            lt_db = state.lt_db
        if lt_index is None:
            lt_index = state.lt_index

        # +1 no top_k: a própria memória (recém-adicionada ao índice) pode
        # voltar como hit — filtrada abaixo.
        hits = await loop.run_in_executor(
            None, lt_index.search, embedding, AUTO_LINK_TOP_K + 1
        )
        candidates = [
            (mid, sim) for mid, sim in hits
            if mid != new_id and sim >= AUTO_LINK_MIN_COSINE
        ]
        if not candidates:
            # Nada acima do pré-filtro de cosine — com o piso em 0.80 e ruído
            # medido em 0.71-0.86, isso só acontece no grafo vazio/primeira
            # memória de um tema totalmente novo. DEBUG p/ não poluir.
            log.debug(f"[AUTO-LINK] LT #{new_id}: nenhum candidato acima do "
                      f"pré-filtro de cosine {AUTO_LINK_MIN_COSINE:.2f}")
            return

        rows = await loop.run_in_executor(
            None, lt_db.get_by_ids, [mid for mid, _ in candidates]
        )
        by_id = {r["id"]: r for r in rows}
        cand_ids = [mid for mid, _ in candidates if mid in by_id]
        if not cand_ids:
            return

        # Gate por cross-encoder em lote (1 round-trip). Obrigatório: sem
        # score, sem aresta — evita grafo poluído por falsos positivos do
        # bi-encoder.
        try:
            cross_scores = await state.rerank_engine.score_many(
                text, [by_id[mid]["text"] for mid in cand_ids]
            )
        except Exception as e:
            log.warning(
                f"[AUTO-LINK] reranker indisponível — auto-link pulado p/ LT "
                f"#{new_id} ({len(cand_ids)} candidato(s)): {type(e).__name__}: {e}"
            )
            return

        linked, linked_ids = 0, []
        now = time.time()
        top_cos = max(sim for _, sim in candidates)
        top_cross = max(cross_scores) if cross_scores else 0.0
        for mid, cross in zip(cand_ids, cross_scores):
            if cross < AUTO_LINK_MIN_CROSS:
                continue
            await loop.run_in_executor(
                None, lt_db.upsert_edge,
                new_id, mid, EDGE_LEARNING_RATE, now,
                EDGE_CO_ACTIVATION, AUTO_LINK_INITIAL_WEIGHT,
            )
            linked += 1
            linked_ids.append(mid)
            if linked >= AUTO_LINK_MAX_EDGES:
                break

        # UMA linha INFO sempre que houver candidatos — ligou ou não, o
        # "porquê" fica no log (o gate cross que filtra tudo deixou de ser
        # invisível; recalibração fica orientada a dados).
        log.info(
            f"[AUTO-LINK] LT #{new_id}: {len(cand_ids)} candidato(s), "
            f"top cos={top_cos:.3f}/cross={top_cross:.3f} "
            f"(gate cross>={AUTO_LINK_MIN_CROSS:.2f}) → {linked} aresta(s) "
            f"{linked_ids if linked_ids else '(gate filtrou tudo)'}"
        )
    except Exception as e:
        log.warning(
            f"[AUTO-LINK] falhou (não-bloqueante) p/ LT #{new_id}: "
            f"{type(e).__name__}: {e}"
        )





async def _store_long_term_text(
    text: str,
    source: str,
    confidence: float,
    forgettable: bool = True,
    # ── NEW (solução 3) ──
    ttl_days: Optional[float] = None,
    # ── NEW (solução 2) ──
    action: Literal["create", "update"] = "create",
    memory_id: Optional[int] = None,
    # ── NEW (arestas tipadas): True → o chamador já declarou (link_to +
    # link_type) que este texto é um fato DISTINTO relacionado a outro; a faixa
    # "possible_update" não bloqueia o write.
    allow_near: bool = False,
    # ── NEW (projetos): True = caller INTERNO (dicionários/cards) que trabalha
    # no espaço de ids CRUAS — a resposta devolve o id real mesmo quando o alvo
    # está no escopo geral (o id é gravado de volta no card/conceito e depois
    # sondado cru). False (default) = caller externo (/write) recebe id de API
    # (composta com offset quando o alvo é do geral em leitura mesclada).
    internal: bool = False,
) -> tuple[bool, str, Optional[int], Optional[dict]]:
    """
    Lógica compartilhada de gravação em memória de longo prazo — usada tanto
    pelo endpoint /write (e /write_batch) quanto pelo /visual-dict/write
    (para persistir a descrição textual de um conceito visual como memória
    normal).
    Retorna (stored, reason, memory_id, candidate) — `candidate` só vem
    preenchido (dict com id/text/score) quando `reason` começa com
    "possible_update:".

    NEW (projetos): a CRIAÇÃO acontece sempre no ESCOPO ATIVO. O UPDATE
    resolve o dono do alvo: id composta (offset) → escopo geral; id crua
    inexistente no ativo → sonda o geral (nós de identidade de cards criados
    no modo geral permanecem estáveis no grafo original — ver
    _card_identity_memory). Fora isso, escrita só no ativo.
    """
    text = text.strip()
    if len(text) < 10:
        return False, "too_short", None, None

    loop = asyncio.get_event_loop()

    # ── NEW (solução 2): fluxo de atualização/correção ─────────────────
    # Em vez de inserir um fato novo que conflita com um já existente, o
    # texto substitui o conteúdo de uma memória já gravada.
    if action == "update":
        async with _lt_write_lock:
            target_id = memory_id
            # Escopo dono do alvo (default: escopo ativo)
            target_db, target_index = state.lt_db, state.lt_index
            if target_id is not None and is_general_ref(target_id):
                # Id COMPOSTA (leitura mesclada) → alvo no escopo geral.
                gen = state.scopes.get(GENERAL_SCOPE)
                if gen is None:
                    return False, "update_target_not_found", None, None
                target_id = ref_real_id(target_id)
                target_db, target_index = gen.lt_db, gen.lt_index
            elif target_id is not None:
                # Id CRUA: se não existe no escopo ativo, sonda o geral — o
                # update de um nó existente mantém o nó no dono original
                # (arestas/permanência no grafo certo), sem criar cópia no
                # escopo ativo.
                row = await loop.run_in_executor(None, state.lt_db.get_by_id, target_id)
                if row is None:
                    gen = state.scopes.get(GENERAL_SCOPE)
                    grow = await loop.run_in_executor(
                        None, gen.lt_db.get_by_id, target_id
                    ) if gen is not None else None
                    if grow is not None:
                        target_db, target_index = gen.lt_db, gen.lt_index
            if target_id is None:
                # Sem memory_id explícito: acha a memória de LT mais
                # parecida semanticamente e usa ela como alvo, desde que
                # a similaridade seja alta o bastante para termos certeza
                # de que é "a mesma coisa, dita de outro jeito" — não
                # queremos "atualizar" um fato não relacionado por engano.
                #
                # O cosine do bi-encoder aqui é só RECALL (candidato a
                # checar); quem confirma é o reranker cross-encoder, que
                # não sofre do mesmo problema de anisotropia (ver
                # RERANK_UPDATE_SCORE / comentário perto de UPDATE_SIM_THRESHOLD).
                #
                # Busca SÓ no escopo ativo (escrita não vira update no geral
                # por semelhança — isso seria criação disfarçada fora do
                # escopo ativo).
                embedding = await state.embed_engine.embed_passage_one(text)
                hits = state.lt_index.search(embedding, top_k=1)
                if hits and hits[0][1] >= UPDATE_SIM_THRESHOLD:
                    candidate_id, candidate_sim = hits[0]
                    candidate_row = state.lt_db.get_by_id(candidate_id)
                    if candidate_row is not None:
                        cross_score = await state.rerank_engine.score_one(text, candidate_row["text"])
                        if cross_score >= RERANK_UPDATE_SCORE:
                            target_id = candidate_id
                        else:
                            log.info(
                                f"update sem memory_id: bi-encoder achou candidata #{candidate_id} "
                                f"(cosine={candidate_sim:.3f}) mas reranker discordou "
                                f"(cross_score={cross_score:.3f} < {RERANK_UPDATE_SCORE}) — ignorando"
                            )

            if target_id is None:
                return False, "update_target_not_found", None, None

            row = target_db.get_by_id(target_id)
            if row is None:
                return False, "update_target_not_found", None, None

            ok = target_db.update(
                target_id, text,
                source=source, confidence=confidence,
                forgettable=forgettable,
                ttl_days=ttl_days, ttl_days_set=True,
            )
            if not ok:
                return False, "update_conflict", None, None

            # O vetor antigo aponta pro texto anterior — remove e adiciona
            # de novo com o mesmo id, pra busca semântica continuar
            # refletindo o texto atual em vez do corrigido.
            new_embedding = await state.embed_engine.embed_passage_one(text)
            await loop.run_in_executor(None, target_index.remove_ids, {target_id})
            await loop.run_in_executor(None, target_index.add, new_embedding, target_id)

            # Auto-link: o texto atualizado deve ligar aos seus vizinhos
            # semânticos ATUAIS (as arestas antigas permanecem; esta task só
            # acrescenta o que o novo conteúdo trouxe de novo) — no escopo
            # dono do alvo.
            if AUTO_LINK_ENABLED:
                asyncio.ensure_future(
                    _auto_link_task(target_id, new_embedding, text,
                                    lt_db=target_db, lt_index=target_index)
                )

        # Id de RESPOSTA: cru p/ callers internos (cards/conceitos guardam id
        # crua) ou quando o alvo é do próprio escopo ativo; composta quando o
        # alvo é do geral e o caller é externo com projeto ativo.
        out_id = target_id
        if not internal and target_db is not state.lt_db:
            out_id = encode_ref(GENERAL_SCOPE, target_id)
        owner = "ativo" if target_db is state.lt_db else GENERAL_SCOPE
        log.info(f"LT #{target_id} atualizada [{owner}]: {text[:60]}")
        return True, "updated", out_id, None


    # ── Fluxo normal de criação ─────────────────────────────────────────
    async with _lt_write_lock:
        if state.lt_db.exists_exact(text):
            return False, "duplicate_exact", None, None

        embedding = await state.embed_engine.embed_passage_one(text)

        hits = state.lt_index.search(embedding, top_k=1)
        max_sim, nearest_id = (hits[0][1], hits[0][0]) if hits else (0.0, None)

        # ── Bi-encoder = recall (achar candidato), reranker = decisão ────────
        # Só vale a pena checar o cross-encoder se existe um candidato
        # plausível pelo cosine; abaixo de UPDATE_SIM_THRESHOLD nem chega
        # perto do piso de ruído do modelo, então é claramente um fato novo
        # e pulamos a chamada extra de rerank (mais rápido).
        if not allow_near and max_sim >= UPDATE_SIM_THRESHOLD and nearest_id is not None:
            candidate_row = state.lt_db.get_by_id(nearest_id)
            cross_score = None
            if candidate_row is not None:
                cross_score = await state.rerank_engine.score_one(text, candidate_row["text"])

            if cross_score is not None and cross_score >= RERANK_DUPLICATE_SCORE:
                return False, f"duplicate_semantic:{max_sim:.3f}:rerank={cross_score:.3f}", None, None

            if cross_score is not None and cross_score >= RERANK_UPDATE_SCORE:
                candidate = {
                    "id":    candidate_row["id"],
                    "text":  candidate_row["text"],
                    "score": round(max_sim, 4),
                    "rerank_score": round(cross_score, 4),
                }
                return False, f"possible_update:{max_sim:.3f}:rerank={cross_score:.3f}", None, candidate

            # cross_score baixo (ou candidate_row sumiu) → bi-encoder deu
            # falso positivo (anisotropia/domínio compartilhado); segue como
            # fato novo mesmo com cosine alto.
            if cross_score is not None:
                log.info(
                    f"LT: candidato #{nearest_id} descartado pelo reranker "
                    f"(cosine={max_sim:.3f} mas cross_score={cross_score:.3f} < {RERANK_UPDATE_SCORE}) "
                    f"— gravando como fato novo: {text[:60]}"
                )

        try:
            memory_id = state.lt_db.insert(text, source, confidence, forgettable, ttl_days)
        except sqlite3.IntegrityError:
            # Rede de segurança: mesmo com o lock, cobre o caso de outro
            # processo/writer ter inserido o mesmo texto entre o check e o
            # insert (ex.: dois workers do MCP).
            log.warning(f"LT: corrida de duplicata detectada no insert — '{text[:60]}'")
            return False, "duplicate_exact", None, None

        # add() faz uma cópia/realocação em C++ — pequena, mas offload pro
        # executor mantém o event loop livre mesmo sob concorrência alta.
        await loop.run_in_executor(None, state.lt_index.add, embedding, memory_id)

        # ── Auto-link na escrita ────────────────────────────────────────────
        # Liga a memória nova aos vizinhos semânticos já existentes (gate por
        # cross-encoder, fire-and-forget) — a memória não fica mais isolada
        # no grafo até uma co-ativação futura. Ver _auto_link_task.
        if AUTO_LINK_ENABLED:
            asyncio.ensure_future(_auto_link_task(memory_id, embedding, text))

    log.info(f"LT #{memory_id} gravada: {text[:60]}")
    return True, "ok", memory_id, None


async def _link_new_memory(new_id: int, target_id: int, link_type: str) -> None:
    """Cria a aresta tipada entre uma memória recém-gravada e `target_id`.
    Direção: updates = nova → antiga; temporal_precedence = `target_id` (veio
    antes) → nova; contradicts/co_activation são simétricas.

    NEW (projetos): arestas nunca cruzam escopos — `target_id` composto
    (memória do geral referenciada num /write feito com projeto ativo) é
    recusado com warning (a memória nova está no escopo ativo)."""
    loop = asyncio.get_event_loop()
    try:
        if is_general_ref(target_id):
            log.warning(
                f"link_to #{target_id} é do escopo GERAL (id composta) — arestas "
                f"nunca cruzam escopos; ligação ignorada"
            )
            return
        if await loop.run_in_executor(None, state.lt_db.get_by_id, target_id) is None:
            log.warning(f"link_to #{target_id} não existe — aresta {link_type} não criada")
            return
        src, dst = (target_id, new_id) if link_type == EDGE_TEMPORAL else (new_id, target_id)
        await loop.run_in_executor(
            None, state.lt_db.upsert_edge, src, dst, EDGE_LEARNING_RATE, time.time(), link_type,
        )
    except Exception as e:
        log.error(f"Grafo: falha ao criar aresta {link_type} #{new_id}→#{target_id}: {e}")


# ── NEW (people cards): vínculo memória LT ↔ card de pessoa ────────────────────

async def _link_memory_to_person(memory_id: int, person_id: int) -> None:
    """Cria a aresta co_activation entre a memória recém-gravada e o nó de
    identidade do card de pessoa no grafo Hebbiano — o PPR espalha ativação
    entre a pessoa e as memórias sobre ela. Se o card ainda não tem memória
    de identidade (memory_id vazio), cria uma (`"Nome: descrição"`,
    source="people_card") e grava o id no card. person_id inválido → só
    warning, nunca bloqueia a gravação da memória.

    NEW (projetos): o nó de identidade pode morar no escopo ativo OU no geral
    (o card foi criado em outro modo) — sonda os dois ANTES de criar a aresta:
    dono = ativo → aresta no grafo ativo; dono = geral com projeto ativo →
    recusada (arestas nunca cruzam escopos); nó não existe em nenhum →
    warning, sem aresta (não recria o nó — preserva o ponteiro do card)."""
    loop = asyncio.get_event_loop()
    try:
        card = await loop.run_in_executor(None, state.pd_db.get_person_by_id, person_id)
        if card is None:
            log.warning(f"people: person_id #{person_id} não existe — memória gravada sem vínculo")
            return
        identity_id = card["memory_id"]
        if identity_id is None:
            stored, reason, identity_id, _ = await _store_long_term_text(
                f"{card['person_name']}: {card['description'] or 'sem descrição'}",
                "people_card", 1.0, False,
                internal=True,
            )
            if not stored or identity_id is None:
                log.warning(
                    f"people: falha ao criar memória de identidade do card "
                    f"#{person_id} ({reason}) — memória gravada sem vínculo"
                )
                return
            await loop.run_in_executor(
                None, state.pd_db.set_memory_id, person_id, identity_id,
            )
            edge_db = state.lt_db   # nó criado agora, no escopo ativo
        else:
            # Sonda o dono do nó de identidade: escopo ativo primeiro, geral
            # depois (ids de cards são cruas — vêm do banco de pessoas).
            if await loop.run_in_executor(None, state.lt_db.get_by_id, identity_id) is not None:
                edge_db = state.lt_db
            else:
                gen = state.scopes.get(GENERAL_SCOPE)
                if gen is not None and await loop.run_in_executor(
                    None, gen.lt_db.get_by_id, identity_id
                ) is not None:
                    if state.active_scope == GENERAL_SCOPE:
                        edge_db = gen.lt_db
                    else:
                        log.warning(
                            f"people: nó de identidade #{identity_id} do card #{person_id} está "
                            f"no escopo GERAL — aresta não criada (arestas nunca cruzam escopos)"
                        )
                        return
                else:
                    log.warning(
                        f"people: nó de identidade #{identity_id} do card #{person_id} não "
                        f"encontrado em nenhum escopo — memória gravada sem vínculo"
                    )
                    return
        await loop.run_in_executor(
            None, edge_db.upsert_edge, memory_id, identity_id,
            EDGE_LEARNING_RATE, time.time(), EDGE_CO_ACTIVATION,
        )
        log.info(f"people: memória #{memory_id} ↔ card #{person_id} (co_activation)")
    except Exception as e:
        log.error(f"people: falha ao vincular memória #{memory_id} à pessoa #{person_id}: {e}")


async def _process_write_request(req: WriteRequest) -> WriteResponse:
    """Ponto único usado tanto por /write quanto por /write_batch (solução 1),
    pra garantir que os dois caminhos tenham exatamente a mesma lógica de
    dedup/update/ttl."""
    if (req.link_to is None) != (req.link_type is None):
        log.warning("write: link_to e link_type devem vir juntos — ligação ignorada")
    linking = req.link_to is not None and req.link_type is not None
    stored, reason, memory_id, candidate = await _store_long_term_text(
        req.text, req.source, req.confidence, req.forgettable,
        ttl_days=req.ttl_days, action=req.action, memory_id=req.memory_id,
        allow_near=linking,
    )
    if linking and stored and reason == "ok" and memory_id is not None:
        await _link_new_memory(memory_id, req.link_to, req.link_type)
    # ── NEW (people cards): vincula a memória ao nó de identidade do card
    # (aresta co_activation). Depois da ligação explícita — person_id inválido
    # → warning e grava sem vínculo, sem bloquear.
    if req.person_id is not None and stored and reason == "ok" and memory_id is not None:
        await _link_memory_to_person(memory_id, req.person_id)
    resp = WriteResponse(stored=stored, reason=reason, memory_id=memory_id)
    if candidate is not None:
        resp.candidate_id    = candidate["id"]
        resp.candidate_text  = candidate["text"]
        resp.candidate_score = candidate["score"]
        resp.candidate_rerank_score = candidate.get("rerank_score")
    return resp


async def write_memory(req: WriteRequest):
    return await _process_write_request(req)


# ── POST /write_batch ──────────────────────────────────────────────────────
# NEW (solução 1): grava vários WriteRequest numa chamada só, em vez de N
# chamadas separadas ao /write. Cada item é processado com a mesma lógica de
# _process_write_request (dedup/update/ttl inclusos) — um item que falha
# (ex.: duplicata) não impede os demais de serem processados.

async def write_memory_batch(req: WriteBatchRequest):
    results: list[WriteResponse] = []
    loop = asyncio.get_event_loop()
    prev_id: Optional[int] = None
    for item in req.items:
        resp = await _process_write_request(item)
        results.append(resp)
        # ── NEW: batch sequencial → aresta temporal_precedence (anterior → seguinte)
        # NEW (projetos): aresta só quando AMBAS as ids são do MESMO escopo —
        # ids compostas (offset) indicam memória do geral (update em modo
        # projeto); arestas nunca cruzam escopos.
        if req.sequential and resp.stored and resp.memory_id is not None:
            if (
                prev_id is not None and prev_id != resp.memory_id
                and not is_general_ref(prev_id) and not is_general_ref(resp.memory_id)
            ):
                await loop.run_in_executor(
                    None, state.lt_db.upsert_edge, prev_id, resp.memory_id,
                    EDGE_LEARNING_RATE, time.time(), EDGE_TEMPORAL,
                )
            prev_id = resp.memory_id
    stored_count = sum(1 for r in results if r.stored)
    log.info(f"write_batch: {stored_count}/{len(results)} memórias gravadas")
    return WriteBatchResponse(
        results=results, stored_count=stored_count, total=len(results),
    )


# ── POST /write_st ─────────────────────────────────────────────────────────────

async def write_short_term(req: WriteSTRequest):
    if not req.turns:
        return WriteSTResponse(stored=False, reason="no_turns")

    all_text = " ".join(t.content.strip() for t in req.turns)
    if len(all_text) < 10:
        return WriteSTResponse(stored=False, reason="too_short")

    embed_text = "\n".join(f"{t.role}: {t.content}" for t in req.turns)
    loop = asyncio.get_event_loop()

    # ── NEW (people cards): valida o card antes de gravar — person_id inválido
    # → warning e grava sem vínculo (FK sem checagem destruiria a leitura por
    # pessoa com ids órfãos).
    person_id = req.person_id
    if person_id is not None:
        card = await loop.run_in_executor(None, state.pd_db.get_person_by_id, person_id)
        if card is None:
            log.warning(f"people: person_id #{person_id} não existe — ST gravado sem vínculo")
            person_id = None

    async with _st_write_lock:
        embedding = await state.embed_engine.embed_passage_one(embed_text)

        # ── SESSÕES REMOVIDAS: dedup com JANELA TEMPORAL. O st_index é global,
        # então sim ≥ threshold contra um grupo ANTIGO de outra sessão não pode
        # recusar o turno novo (destruiria o conteúdo mais recente do histórico
        # — bug 27/10). Só recusa quando o duplicado é RECENTE (corrida com o
        # resumo episódico da extração / retry do _save_turn / reenvio imediato).
        hits = await loop.run_in_executor(None, state.st_index.search, embedding, 5)
        dups = [(gid, sim) for gid, sim in hits if sim >= DEDUP_THRESHOLD]
        if dups:
            created = {
                r["id"]: r["created_at"] for r in state.st_db.get_by_ids([gid for gid, _ in dups])
            }
            now_ts = time.time()
            recent = [(gid, sim) for gid, sim in dups
                      if now_ts - created.get(gid, 0.0) < ST_DEDUP_WINDOW_S]
            if recent:
                gid, sim = max(recent, key=lambda t: t[1])
                log.info(
                    f"/write_st dedup: grupo #{gid} recente ({now_ts - created[gid]:.0f}s, "
                    f"< {ST_DEDUP_WINDOW_S:.0f}s) com sim={sim:.3f} — recusado"
                )
                return WriteSTResponse(stored=False, reason=f"duplicate_semantic:{sim:.3f}")
            log.info(
                f"/write_st dedup: sim={dups[0][1]:.3f} apenas contra grupo(s) antigo(s) "
                f"(fora da janela de {ST_DEDUP_WINDOW_S:.0f}s) — turno novo gravado"
            )

        group_id = state.st_db.insert(req.session_id, req.turns, embed_text, person_id=person_id)
        await loop.run_in_executor(None, state.st_index.add, embedding, group_id)

    log.info(
        f"ST #{group_id} gravado — session={req.session_id or '-'} turnos={len(req.turns)} "
        f"person_id={person_id}: {embed_text[:80]}"
    )
    return WriteSTResponse(stored=True, reason="ok", turn_ids=[group_id])


# ── POST /read_st ────────────────────────────────────────────────────────────
# Leitura crua do short-term: as N duplas pergunta-resposta mais recentes do
# ST GLOBAL (sem filtro de sessão — sessões removidas; session_id no payload
# é aceito por compatibilidade e ignorado), sem embeddings/scoring — pensado
# para o LLM montar o histórico de conversa como contexto (mais leve/rápido
# que /read).

async def read_short_term(req: ReadSTRequest):
    # ── SESSÕES REMOVIDAS: a leitura é GLOBAL — o histórico tem que sobreviver
    # à troca de sessão/restart (bug 26/10: cada sessão nova nascia amnésica).
    # `person_id` (people cards) continua filtrável — o histórico devolvido
    # pode se restringir a conversas com aquela pessoa.
    n_pairs = req.n_pairs if req.n_pairs > 0 else ST_READ_DEFAULT_PAIRS
    loop = asyncio.get_event_loop()
    raw_turns, groups_fetched = await loop.run_in_executor(
        None, state.st_db.get_recent_turn_groups, n_pairs, req.person_id
    )
    turns = [Turn(**t) for t in raw_turns]

    log.info(f"/read_st global n_pairs={n_pairs} grupos={groups_fetched} turnos={len(turns)} "
             f"session_pedida={req.session_id or '-'}")
    return ReadSTResponse(session_id=req.session_id, turns=turns, pairs_returned=groups_fetched)


# ── POST /read ─────────────────────────────────────────────────────────────────

async def read_memory(req: ReadRequest):
    """Leitura mesclada — pipeline completo do /read.

    NEW (projetos): `lt_views()` devolve o escopo ATIVO primeiro e, com um
    projeto ativo, o GERAL ao final (leitura mesclada). Cada estágio roda
    POR VIEW com os MESMOS embeddings (calculados uma vez — sem custo extra
    de ONNX): busca LT estratégica, PPR, VS, IF. ST é global (busca única).
    Entradas do escopo geral saem com id composta (offset +10^12) e tag
    `scope` — escrita/update resolve de volta (ref_real_id). Com escopo
    geral ativo há uma view só e o comportamento é idêntico ao de antes.
    """
    query = req.query.strip()
    if not query:
        raise MemoryToolError("query vazia")

    loop = asyncio.get_event_loop()
    effective_strategy = "none"

    views = lt_views()
    merged = len(views) > 1
    # Escopo ativo é sempre views[0]; state.scopes dá acesso a vs/if_index/if_db.
    scope_states = [state.scopes[v.name] for v in views]

    query_emb = await state.embed_engine.embed_query_one(query)

    # ── VS + IF: busca POR ESCOPO em background (cada escopo tem os seus) ──
    vs_futures: dict[str, object] = {}
    if_futures: dict[str, object] = {}
    for v, sc in zip(views, scope_states):
        if sc.vs is not None and sc.vs.total > 0:
            vs_min_score = min(req.min_score, VS_MIN_SCORE) if req.min_score < VS_MIN_SCORE else VS_MIN_SCORE
            vs_futures[v.name] = asyncio.ensure_future(
                loop.run_in_executor(None, sc.vs.search, query_emb, req.top_k, vs_min_score)
            )
        if sc.if_index.total > 0:
            if_futures[v.name] = asyncio.ensure_future(
                loop.run_in_executor(None, sc.if_index.search, query_emb, req.top_k * 2)
            )

    # Hits LT por view — ids CRUAS do escopo (PPR/vetor usam ids cruas; a
    # codificação de resposta acontece só no _build_* com encode/scope).
    lt_raw_by_view: dict[str, list[tuple[int, float]]] = {v.name: [] for v in views}
    st_raw: list[tuple[int, float]] = []

    if req.strategy != "none":
        # SESSÕES REMOVIDAS: o contexto de expansão são os turnos recentes
        # GLOBAIS do ST (session_id deixou de filtrar).
        context_block = await loop.run_in_executor(None, _build_context_block)

        if context_block:
            if req.strategy == "auto":
                effective_strategy = _classify_query(query)
            elif req.strategy in ("expanded", "dual"):
                effective_strategy = req.strategy
            else:
                log.warning(f"Estratégia desconhecida '{req.strategy}', usando 'auto'")
                effective_strategy = _classify_query(query)

            # ── Embeddings da estratégia: UMA vez (ST global + LT por view) ──
            if effective_strategy == "expanded":
                expanded = f"{context_block}\n\nquery atual: {query}" if context_block else query
                strat_emb = await state.embed_engine.embed_query_one(expanded)
                lt_coros = [_search_lt_expanded_on(v.index, strat_emb, req.top_k) for v in views]
                st_task = _search_st_expanded(strat_emb, req.top_k)
            else:
                emb_query, emb_ctx = await state.embed_engine.embed_query_two(query, context_block)
                lt_coros = [_search_lt_dual_on(v.index, emb_query, emb_ctx, req.top_k) for v in views]
                st_task = _search_st_dual(emb_query, emb_ctx, req.top_k)

            # gather devolve uma lista FLAT ([lt_view0, …, lt_viewN, st]) —
            # fatia por posição (views primeiro, ST por último).
            gathered = await asyncio.gather(*lt_coros, st_task)
            lt_results, st_raw = gathered[:len(views)], gathered[len(views)]
            for v, lt_res in zip(views, lt_results):
                lt_raw_by_view[v.name] = list(lt_res)

            log.info(
                f"/read session={req.session_id} strategy={effective_strategy} "
                f"ctx_chars={len(context_block)} query='{query[:60]}'"
            )
        else:
            effective_strategy = "none"

    if effective_strategy == "none":
        lt_coros = [_search_lt_plain(v.index, query_emb, req.top_k) for v in views]
        # gather devolve lista FLAT (views primeiro, ST por último) — ver acima.
        gathered = await asyncio.gather(*lt_coros, _search_st_plain(query_emb, req.top_k))
        lt_results, st_raw = gathered[:len(views)], gathered[len(views)]
        for v, lt_res in zip(views, lt_results):
            lt_raw_by_view[v.name] = list(lt_res)

    # ── NEW: query composta → busca adicional por segmento, sem LLM ───────────
    # Se a pergunta tem vários sub-pedidos numa frase só (ex.: "quero fazer X
    # usando Y para conseguir Z"), a busca acima (query inteira) tende a
    # trazer só o que é mais "central" na frase. Aqui cada sub-tópico é
    # embedado/buscado à parte e o resultado é fundido por max-score — sem
    # afogar um sub-tópico no outro nem gerar custo de LLM. Os embeddings dos
    # segmentos são calculados UMA vez e buscados em cada view.
    query_segments = _split_query_segments(query)
    if len(query_segments) > 1:
        seg_emb = await state.embed_engine.embed_queries(query_segments)  # shape (N, dim)
        lt_seg_coros = [_search_lt_segmented_on(v.index, seg_emb, req.top_k) for v in views]
        # gather devolve lista FLAT (views primeiro, ST por último).
        gathered = await asyncio.gather(*lt_seg_coros, _search_st_segmented(seg_emb, req.top_k))
        lt_segs, st_seg = gathered[:len(views)], gathered[len(views)]
        for v, lt_seg in zip(views, lt_segs):
            lt_raw_by_view[v.name] = _fuse_max(lt_raw_by_view[v.name], lt_seg)
        st_raw = _fuse_max(st_raw, st_seg)
        log.info(f"/read query segmentada em {len(query_segments)} partes: {query_segments}")

    # ── NEW (people cards): leitura centrada em uma pessoa. O nó de identidade
    # do card (memory_id) é injetado como semente extra no PPR — memórias
    # relacionadas à pessoa vêm como "related" mesmo sem match semântico.
    # Card inexistente ou sem nó de identidade → só warning, leitura segue
    # normal (sem semente). NEW (projetos): a semente pode morar no escopo
    # ativo OU no geral (o card pode ter sido criado em outro modo) — sonda
    # as views em ordem (ativo primeiro).
    if req.person_id is not None:
        card = await loop.run_in_executor(None, state.pd_db.get_person_by_id, req.person_id)
        if card is None:
            log.warning(f"people: person_id #{req.person_id} não existe — /read sem semente de pessoa")
        elif card["memory_id"] is None:
            log.warning(f"people: card #{req.person_id} sem memória de identidade — /read sem semente")
        else:
            for v in views:
                row = await loop.run_in_executor(None, v.db.get_by_id, card["memory_id"])
                if row is not None:
                    lt_raw_by_view[v.name].append((card["memory_id"], 1.0))
                    log.info(
                        f"people: semente de identidade #{card['memory_id']} (card #{req.person_id}, "
                        f"escopo {v.name}) injetada no PPR"
                    )
                    break
            else:
                log.warning(
                    f"people: memória de identidade #{card['memory_id']} do card #{req.person_id} "
                    f"não encontrada em nenhum escopo — /read sem semente"
                )

    # ── NEW (Hebbian graph / Step 6): PPR POR ESCOPO — cada view propaga no
    # SEU grafo (sementes com ids cruas do escopo) e guarda o ctx próprio.
    # ctx enriquece o log de ativação (query/strategy/session) consumido pela
    # visualização em tempo real (graph_activation.py), que segue o escopo
    # ativo via /status.
    read_ctx = {
        "query": query,
        "strategy": effective_strategy,
        "session_id": req.session_id,
    }
    view_names = [v.name for v in views]
    ctx_list = await asyncio.gather(*(
        _ppr_spread_on(v.db, lt_raw_by_view[v.name], loop, ctx=read_ctx)
        for v in views
    ))
    per_view_ctx: dict[str, dict] = dict(zip(view_names, ctx_list))
    ppr_ms = sum(c["elapsed_ms"] for c in per_view_ctx.values())

    # ── Otimização de tokens ───────────────────────────────────────────────────
    # 1. Threshold mais seletivo para /read (combinação de 4 fontes gera ruído)
    #    Usa max(req.min_score, READ_MIN_SCORE_STRICT) — sempre >= 0.85
    # 2. Corrige bug do IF_MIN_SCORE: usar max (mais seletivo) não min
    #    Antes: min(req.min_score, IF_MIN_SCORE) → retornava chunks com score 0.75
    #    Agora: max(req.min_score, IF_MIN_SCORE_READ) → >= 0.82
    strict_min_score = max(req.min_score, READ_MIN_SCORE_STRICT)
    if_strict_min_score = max(strict_min_score, IF_MIN_SCORE_READ)

    # ── Build POR ESCOPO: ids compostas + tag `scope` só na leitura mesclada ──
    # ST é global — entra uma vez (i == 0, escopo ativo).
    results: list[MemoryEntry] = []
    for i, v in enumerate(views):
        vctx = per_view_ctx[v.name]
        scope_tag = v.name if merged else None     # None = compatibilidade (modo geral)
        related_raw = vctx["related_raw"]
        related_ids = {mid for mid, _ in related_raw}
        results += _build_lt_entries(
            lt_raw_by_view[v.name], strict_min_score, loop,
            lt_db=v.db, encode=v.encode, scope=scope_tag,
        )
        # "related" tem threshold próprio: score PPR é massa de probabilidade
        # (<= PPR_SPREAD_WEIGHT), jamais passaria no strict_min_score (0.85).
        # O piso real de elegibilidade é PPR_MIN_ACTIVATION_REINFORCE, já
        # aplicado dentro de _ppr_spread no próprio rank.
        results += _build_lt_entries(
            related_raw, 0.0, loop, related_ids=related_ids,
            lt_db=v.db, encode=v.encode, scope=scope_tag,
        )
        if i == 0:
            results += _build_st_entries(st_raw, strict_min_score, loop)
        vs_results = []
        vs_future = vs_futures.get(v.name)
        if vs_future is not None:
            try:
                vs_results = await vs_future
            except Exception as e:
                log.error(f"VectorStore search failed [{v.name}]: {e}")
                vs_results = []
        results += _build_vs_entries(vs_results, strict_min_score, encode=v.encode, scope=scope_tag)
        if_raw = []
        if_future = if_futures.get(v.name)
        if if_future is not None:
            try:
                if_raw = await if_future
            except Exception as e:
                log.error(f"Indexed files search failed [{v.name}]: {e}")
                if_raw = []
        results += _build_if_entries(
            if_raw, if_strict_min_score, loop,
            if_db=scope_states[i].if_db, encode=v.encode, scope=scope_tag,
        )  # NEW
    results.sort(key=lambda r: r.score * r.confidence, reverse=True)

    # ── NEW: rerank dos candidatos primary com o cross-encoder ───────────────
    # O ranking até aqui vem do bi-encoder (cosine) — rápido, mas anisotrópico.
    # O cross-encoder (ms-marco, já carregado) compara query×texto diretamente
    # e decide MUITO melhor relevância; com <= READ_RERANK_TOP_N candidatos, é
    # 1 round-trip /v1/score. Entradas "related" (PPR — score de massa de
    # probabilidade, nunca passariam no strict_min_score) ficam de fora: sua
    # ordenação é responsabilidade do MMR/inibição lateral abaixo, e misturar
    # escalas (cosine ~[0,1] vs PPR <= PPR_SPREAD_WEIGHT) quebraria o contrato
    # primary < related usado por _lateral_inhibition_filter e pelo corte.
    if READ_RERANK_ENABLED and results:
        primary = [r for r in results if r.match_type != "related"]
        related = [r for r in results if r.match_type == "related"]
        candidates = primary[:READ_RERANK_TOP_N]
        if len(candidates) >= 2:
            try:
                cross_scores = await state.rerank_engine.score_many(
                    query, [r.text or "" for r in candidates]
                )
                for entry, cross in zip(candidates, cross_scores):
                    entry.score = round(
                        READ_RERANK_CROSS_WEIGHT * cross
                        + READ_RERANK_COSINE_WEIGHT * entry.score, 4
                    )
                primary = candidates + primary[READ_RERANK_TOP_N:]
                primary.sort(key=lambda r: r.score * r.confidence, reverse=True)
                results = primary + related
                log.debug(
                    f"/read rerank: {len(candidates)} candidatos reordenados "
                    f"pelo cross-encoder (related preservados: {len(related)})"
                )
            except Exception as e:
                # Rerank é melhoria de precisão, não requisito — qualquer falha
                # (onnx serving fora, timeout) mantém a ordem original.
                log.warning(f"/read rerank falhou (seguindo com ranking cosine): {e}")

    # ── NEW (Step 7): inibição lateral (MMR) SÓ no subconjunto "related",
    # antes do orçamento de tokens — primary não é penalizado. Por escopo. ──
    results = await loop.run_in_executor(None, _lateral_inhibition_filter, results)

    # ── Orçamento global de tokens ─────────────────────────────────────────────
    # Limita o total de caracteres retornados, cortando entradas de menor score.
    # Sempre retorna pelo menos 1 entrada se existir.
    # NEW: overrides por request (ReadRequest.top_k_final / total_max_chars),
    # clampeados aos tetos globais — callers de recall de chat podem pedir um
    # contexto mais rico sem mexer no default conservador.
    if req.top_k and req.top_k > 0:
        base_top_k = min(req.top_k, READ_TOP_K_FINAL)
    else:
        base_top_k = READ_TOP_K_FINAL
    if req.top_k_final is not None and req.top_k_final > 0:
        final_top_k = min(req.top_k_final, READ_TOP_K_FINAL_MAX)
    else:
        final_top_k = base_top_k
    if req.total_max_chars is not None and req.total_max_chars > 0:
        total_budget = min(req.total_max_chars, READ_TOTAL_MAX_CHARS_MAX)
    else:
        total_budget = READ_TOTAL_MAX_CHARS
    results = _apply_token_budget(results, total_budget, final_top_k)

    # Log de diagnóstico (nível INFO para acompanhar redução no pipeline)
    total_chars = sum(len(r.text or "") for r in results)
    log.info(
        f"/read query='{query[:50]}' strategy={effective_strategy} "
        f"results={len(results)} total_chars={total_chars} "
        f"budget={total_budget} strict_score={strict_min_score:.2f} "
        f"ppr_ms={ppr_ms:.1f} views={view_names}"
    )

    response = ReadResponse(results=results, query=query, strategy=effective_strategy)

    # ── NEW (Step 4, background): criação de arestas hebbianas entre as
    # memórias LT FINAIS com score >= EDGE_MIN_SCORE_TO_LINK. Fire-and-forget
    # — o caller já recebeu a resposta; falhas só logam. A tarefa agrupa por
    # escopo (entry.scope) — arestas nunca cruzam escopos. ──
    asyncio.ensure_future(_hebbian_link_task(results))
    # ── NEW (Step 8, background): reforço das arestas EXISTENTES dos subgrafos
    # PPR pela regra de co-ativação (Opção B). Também fire-and-forget. Na
    # leitura mesclada o ctx vem por escopo ({"scopes": {nome: ctx}}); no modo
    # geral, o formato plano de antes. ──
    if merged:
        asyncio.ensure_future(_ppr_reinforce_task({"scopes": per_view_ctx}))
    else:
        asyncio.ensure_future(_ppr_reinforce_task(per_view_ctx[views[0].name]))

    return response

# ── DELETE /session/{session_id} ───────────────────────────────────────────────

async def clear_session(session_id: str):
    ids_to_remove = set(state.st_db.get_ids_by_session(session_id))

    if not ids_to_remove:
        return {"cleared": 0, "session_id": session_id}

    state.st_db.delete_by_session(session_id)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, state.st_index.remove_ids, ids_to_remove)
    log.info(f"Sessão {session_id}: {len(ids_to_remove)} grupos removidos")
    return {"cleared": len(ids_to_remove), "session_id": session_id}


__all__ = [
    '_auto_link_task',
    '_link_new_memory',
    '_lt_write_lock',
    '_process_write_request',
    '_st_write_lock',
    '_store_long_term_text',
    'clear_session',
    'read_memory',
    'read_short_term',
    'write_memory',
    'write_memory_batch',
    'write_short_term',
]
