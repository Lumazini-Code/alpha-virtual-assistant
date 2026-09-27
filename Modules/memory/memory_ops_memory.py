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
    AUTO_LINK_BACKFILL_BATCH,
    AUTO_LINK_BACKFILL_MAX_NODES,
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
    ST_READ_DEFAULT_PAIRS,
    UPDATE_SIM_THRESHOLD,
    VS_MIN_SCORE,
    log,
)
from memory_models import (
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
    _ppr_spread,
    _search_dual,
    _search_expanded,
    _search_segmented,
    _split_query_segments,
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


async def _auto_link_task(new_id: int, embedding, text: str) -> None:
    """Auto-link na ESCRITA (grafo Hebbiano) — liga a memória recém-gravada
    aos seus vizinhos semânticos já existentes, sem esperar uma leitura
    futura que co-ative os dois.

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

        # +1 no top_k: a própria memória (recém-adicionada ao índice) pode
        # voltar como hit — filtrada abaixo.
        hits = await loop.run_in_executor(
            None, state.lt_index.search, embedding, AUTO_LINK_TOP_K + 1
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
            None, state.lt_db.get_by_ids, [mid for mid, _ in candidates]
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
                None, state.lt_db.upsert_edge,
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


async def _auto_link_backfill() -> dict:
    """Consolidação retroativa do grafo — o "sono" do memory.

    Roda no startup (agendado por memory_state.startup, com delay p/ o ONNX
    subir) e liga memórias que JÁ estavam gravadas antes do auto-link existir
    ou sob thresholds antigos — exatamente o caso do grafo atual: só cadeias
    temporais, zero ligações semânticas transversais.

    Pipeline:
      1. carrega todos os (id, text) da LT (teto AUTO_LINK_BACKFILL_MAX_NODES);
      2. embeda TUDO em lote (chunks de AUTO_LINK_BACKFILL_BATCH — 1 chamada
         por chunk, não N);
      3. para cada nó: top-K vizinhos por cosine (matmul local — sem buscas
         no índice), pré-filtro AUTO_LINK_MIN_COSINE;
      4. gate por cross-encoder (1 score_many por nó com candidatos);
      5. upsert_edge — IDEMPOTENTE: arestas já existentes são apenas
         reforçadas (crescimento saturante até 1.0), por isso pode rodar a
         cada boot sem inflar o grafo.

    Retorna estatísticas p/ o log de boot."""
    import numpy as np

    loop = asyncio.get_event_loop()
    rows = await loop.run_in_executor(
        None,
        lambda: state.lt_db._conn.execute(
            "SELECT id, text FROM memories ORDER BY id LIMIT ?",
            (AUTO_LINK_BACKFILL_MAX_NODES,),
        ).fetchall(),
    )
    nodes = [(r["id"], r["text"]) for r in rows]
    if len(nodes) < 2:
        log.info(f"[AUTO-LINK-BACKFILL] grafo pequeno demais p/ consolidar "
                 f"({len(nodes)} nó(s)) — pulando")
        return {"nodes": len(nodes), "created": 0, "reinforced": 0, "skipped_gate": 0}

    t0 = time.perf_counter()

    # ── 1+2: embed em lote (chunks) ─────────────────────────────────────────
    texts = [t for _, t in nodes]
    vecs = []
    for i in range(0, len(texts), AUTO_LINK_BACKFILL_BATCH):
        vecs.append(await state.embed_engine.embed_passages(
            texts[i:i + AUTO_LINK_BACKFILL_BATCH]
        ))
    X = np.concatenate(vecs, axis=0).astype(np.float32)
    # E5 já sai normalizado — renormaliza defensivamente (custo trivial).
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X /= norms

    # ── 3: pré-filtro por cosine (por nó, matmul local) ─────────────────────
    per_node: dict[int, list[tuple[int, float]]] = {}
    for i in range(len(nodes)):
        sims = X @ X[i]
        sims[i] = -1.0  # exclui a si mesmo
        idx = np.argsort(sims)[::-1][:AUTO_LINK_TOP_K]
        cand = [(int(j), float(sims[j])) for j in idx if sims[j] >= AUTO_LINK_MIN_COSINE]
        if cand:
            per_node[i] = cand

    if not per_node:
        log.info("[AUTO-LINK-BACKFILL] nenhum par acima do pré-filtro de "
                 f"cosine ({AUTO_LINK_MIN_COSINE:.2f}) — nada a consolidar")
        return {"nodes": len(nodes), "created": 0, "reinforced": 0, "skipped_gate": 0}

    # Arestas co_activation já existentes (p/ estatística criada vs reforçada).
    existing: set[tuple[int, int]] = set()
    for row in await loop.run_in_executor(
        None,
        lambda: state.lt_db._conn.execute(
            "SELECT memory_id_a, memory_id_b FROM memory_edges WHERE edge_type = ?",
            (EDGE_CO_ACTIVATION,),
        ).fetchall(),
    ):
        existing.add((row["memory_id_a"], row["memory_id_b"]))

    # ── 4+5: gate cross-encoder + upsert ────────────────────────────────────
    created = reinforced = skipped_gate = 0
    now = time.time()
    for i, cand in per_node.items():
        nid, ntext = nodes[i]
        cand_ids = [j for j, _ in cand]
        try:
            cross_scores = await state.rerank_engine.score_many(
                ntext, [texts[j] for j in cand_ids]
            )
        except Exception as e:
            log.warning(f"[AUTO-LINK-BACKFILL] rerank falhou p/ #{nid} "
                        f"(pulando nó): {type(e).__name__}: {e}")
            continue

        linked = 0
        for j, cross in zip(cand_ids, cross_scores):
            if cross < AUTO_LINK_MIN_CROSS:
                continue
            other = nodes[j][0]
            a, b = min(nid, other), max(nid, other)  # co_activation é simétrica
            await loop.run_in_executor(
                None, state.lt_db.upsert_edge,
                nid, other, EDGE_LEARNING_RATE, now,
                EDGE_CO_ACTIVATION, AUTO_LINK_INITIAL_WEIGHT,
            )
            if (a, b) in existing:
                reinforced += 1
            else:
                created += 1
                existing.add((a, b))
            linked += 1
            if linked >= AUTO_LINK_MAX_EDGES:
                break
        if not linked:
            skipped_gate += 1

    elapsed = time.perf_counter() - t0
    log.info(
        f"[AUTO-LINK-BACKFILL] sono concluído: {len(nodes)} memórias, "
        f"{len(per_node)} com candidatos, {created} aresta(s) nova(s), "
        f"{reinforced} reforçada(s), {skipped_gate} nó(s) sem nada acima do "
        f"gate ({AUTO_LINK_MIN_CROSS:.2f}) — {elapsed:.1f}s"
    )
    return {
        "nodes": len(nodes), "created": created, "reinforced": reinforced,
        "skipped_gate": skipped_gate, "elapsed_s": round(elapsed, 1),
    }


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
) -> tuple[bool, str, Optional[int], Optional[dict]]:
    """
    Lógica compartilhada de gravação em memória de longo prazo — usada tanto
    pelo endpoint /write (e /write_batch) quanto pelo /visual-dict/write
    (para persistir a descrição textual de um conceito visual como memória
    normal).
    Retorna (stored, reason, memory_id, candidate) — `candidate` só vem
    preenchido (dict com id/text/score) quando `reason` começa com
    "possible_update:".
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

            row = state.lt_db.get_by_id(target_id)
            if row is None:
                return False, "update_target_not_found", None, None

            ok = state.lt_db.update(
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
            await loop.run_in_executor(None, state.lt_index.remove_ids, {target_id})
            await loop.run_in_executor(None, state.lt_index.add, new_embedding, target_id)

            # Auto-link: o texto atualizado deve ligar aos seus vizinhos
            # semânticos ATUAIS (as arestas antigas permanecem; esta task só
            # acrescenta o que o novo conteúdo trouxer de novo).
            if AUTO_LINK_ENABLED:
                asyncio.ensure_future(_auto_link_task(target_id, new_embedding, text))

        log.info(f"LT #{target_id} atualizada: {text[:60]}")
        return True, "updated", target_id, None

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
    antes) → nova; contradicts/co_activation são simétricas."""
    loop = asyncio.get_event_loop()
    try:
        if await loop.run_in_executor(None, state.lt_db.get_by_id, target_id) is None:
            log.warning(f"link_to #{target_id} não existe — aresta {link_type} não criada")
            return
        src, dst = (target_id, new_id) if link_type == EDGE_TEMPORAL else (new_id, target_id)
        await loop.run_in_executor(
            None, state.lt_db.upsert_edge, src, dst, EDGE_LEARNING_RATE, time.time(), link_type,
        )
    except Exception as e:
        log.error(f"Grafo: falha ao criar aresta {link_type} #{new_id}→#{target_id}: {e}")


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
        if req.sequential and resp.stored and resp.memory_id is not None:
            if prev_id is not None and prev_id != resp.memory_id:
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

    async with _st_write_lock:
        embedding = await state.embed_engine.embed_passage_one(embed_text)

        max_sim = state.st_index.search_similar(embedding)
        if max_sim >= DEDUP_THRESHOLD:
            return WriteSTResponse(stored=False, reason=f"duplicate_semantic:{max_sim:.3f}")

        group_id = state.st_db.insert(req.session_id, req.turns, embed_text)
        await loop.run_in_executor(None, state.st_index.add, embedding, group_id)

    log.info(f"ST #{group_id} gravado — session={req.session_id} turnos={len(req.turns)}: {embed_text[:80]}")
    return WriteSTResponse(stored=True, reason="ok", turn_ids=[group_id])


# ── POST /read_st ────────────────────────────────────────────────────────────
# Leitura crua do short-term: só as N duplas pergunta-resposta mais recentes
# de uma sessão, sem embeddings/scoring — pensado para o LLM montar o
# histórico de conversa como contexto (mais leve/rápido que /read).

async def read_short_term(req: ReadSTRequest):
    session_id = req.session_id.strip()
    if not session_id:
        raise MemoryToolError("session_id vazio")

    n_pairs = req.n_pairs if req.n_pairs > 0 else ST_READ_DEFAULT_PAIRS
    loop = asyncio.get_event_loop()
    raw_turns, groups_fetched = await loop.run_in_executor(
        None, state.st_db.get_recent_turn_groups, session_id, n_pairs
    )
    turns = [Turn(**t) for t in raw_turns]

    log.info(f"/read_st session={session_id} n_pairs={n_pairs} grupos={groups_fetched} turnos={len(turns)}")
    return ReadSTResponse(session_id=session_id, turns=turns, pairs_returned=groups_fetched)


# ── POST /read ─────────────────────────────────────────────────────────────────

async def read_memory(req: ReadRequest):
    query = req.query.strip()
    if not query:
        raise MemoryToolError("query vazia")

    loop = asyncio.get_event_loop()
    effective_strategy = "none"

    query_emb = await state.embed_engine.embed_query_one(query)

    # Start VS search in background
    vs_future = None
    if state.vs is not None and state.vs.total > 0:
        vs_min_score = min(req.min_score, VS_MIN_SCORE) if req.min_score < VS_MIN_SCORE else VS_MIN_SCORE
        vs_future = asyncio.ensure_future(
            loop.run_in_executor(
                None,
                state.vs.search,
                query_emb,
                req.top_k,
                vs_min_score,
            )
        )

    # ── NEW: Start indexed files search in background ──
    if_future = None
    if state.if_index.total > 0:
        if_future = asyncio.ensure_future(
            loop.run_in_executor(None, state.if_index.search, query_emb, req.top_k * 2)
        )

    if req.session_id and req.strategy != "none":
        context_block = await loop.run_in_executor(None, _build_context_block, req.session_id)

        if context_block:
            if req.strategy == "auto":
                effective_strategy = _classify_query(query)
            elif req.strategy in ("expanded", "dual"):
                effective_strategy = req.strategy
            else:
                log.warning(f"Estratégia desconhecida '{req.strategy}', usando 'auto'")
                effective_strategy = _classify_query(query)

            if effective_strategy == "expanded":
                lt_raw, st_raw = await _search_expanded(query, context_block, req.top_k)
            else:
                lt_raw, st_raw = await _search_dual(query, context_block, req.top_k)

            log.info(
                f"/read session={req.session_id} strategy={effective_strategy} "
                f"ctx_chars={len(context_block)} query='{query[:60]}'"
            )
        else:
            effective_strategy = "none"
            lt_raw, st_raw = await asyncio.gather(
                loop.run_in_executor(None, state.lt_index.search, query_emb, req.top_k * 2),
                loop.run_in_executor(None, state.st_index.search, query_emb, req.top_k * 2),
            )
    else:
        lt_raw, st_raw = await asyncio.gather(
            loop.run_in_executor(None, state.lt_index.search, query_emb, req.top_k * 2),
            loop.run_in_executor(None, state.st_index.search, query_emb, req.top_k * 2),
        )

    # ── NEW: query composta → busca adicional por segmento, sem LLM ───────────
    # Se a pergunta tem vários sub-pedidos numa frase só (ex.: "quero fazer X
    # usando Y para conseguir Z"), a busca acima (query inteira) tende a
    # trazer só o que é mais "central" na frase. Aqui cada sub-tópico é
    # embedado/buscado à parte e o resultado é fundido por max-score — sem
    # afogar um sub-tópico no outro nem gerar custo de LLM.
    query_segments = _split_query_segments(query)
    if len(query_segments) > 1:
        lt_seg, st_seg = await _search_segmented(query_segments, req.top_k)
        lt_raw = _fuse_max(lt_raw, lt_seg)
        st_raw = _fuse_max(st_raw, st_seg)
        log.info(f"/read query segmentada em {len(query_segments)} partes: {query_segments}")

    # ── NEW (Hebbian graph / Step 6): PPR no grafo, com sementes em lt_raw ──
    # lt_raw já está finalizado aqui (todas as estratégias convergem nele) e
    # a conversão pra MemoryEntry ainda não aconteceu — posição exata do Step 6.
    # ctx enriquece o log de ativação (query/strategy/session) consumido pela
    # visualização em tempo real (graph_activation.py).
    ppr_ctx = await _ppr_spread(lt_raw, loop, ctx={
        "query": query,
        "strategy": effective_strategy,
        "session_id": req.session_id,
    })
    related_raw: list[tuple[int, float]] = ppr_ctx["related_raw"]
    related_ids = {mid for mid, _ in related_raw}

    # Await VS results
    vs_results = []
    if vs_future is not None:
        try:
            vs_results = await vs_future
        except Exception as e:
            log.error(f"VectorStore search failed: {e}")
            vs_results = []

    # ── NEW: Await indexed files results ──
    if_raw = []
    if if_future is not None:
        try:
            if_raw = await if_future
        except Exception as e:
            log.error(f"Indexed files search failed: {e}")
            if_raw = []

    # ── Otimização de tokens ───────────────────────────────────────────────────
    # 1. Threshold mais seletivo para /read (combinação de 4 fontes gera ruído)
    #    Usa max(req.min_score, READ_MIN_SCORE_STRICT) — sempre >= 0.85
    # 2. Corrige bug do IF_MIN_SCORE: usar max (mais seletivo) não min
    #    Antes: min(req.min_score, IF_MIN_SCORE) → retornava chunks com score 0.75
    #    Agora: max(req.min_score, IF_MIN_SCORE_READ) → >= 0.82
    strict_min_score = max(req.min_score, READ_MIN_SCORE_STRICT)
    if_strict_min_score = max(strict_min_score, IF_MIN_SCORE_READ)

    results: list[MemoryEntry] = (
        _build_lt_entries(lt_raw, strict_min_score, loop) +
        # "related" tem threshold próprio: score PPR é massa de probabilidade
        # (<= PPR_SPREAD_WEIGHT), jamais passaria no strict_min_score (0.85).
        # O piso real de elegibilidade é PPR_MIN_ACTIVATION_REINFORCE, já
        # aplicado dentro de _ppr_spread no próprio rank.
        _build_lt_entries(related_raw, 0.0, loop, related_ids=related_ids) +
        _build_st_entries(st_raw, strict_min_score, loop) +
        _build_vs_entries(vs_results, strict_min_score) +
        _build_if_entries(if_raw, if_strict_min_score, loop)  # NEW
    )
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
    # antes do orçamento de tokens — primary não é penalizado. ──
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
        f"ppr_ms={ppr_ctx['elapsed_ms']:.1f} related={len(related_raw)}"
    )

    response = ReadResponse(results=results, query=query, strategy=effective_strategy)

    # ── NEW (Step 4, background): criação de arestas hebbianas entre as
    # memórias LT FINAIS com score >= EDGE_MIN_SCORE_TO_LINK. Fire-and-forget
    # — o caller já recebeu a resposta; falhas só logam. ──
    asyncio.ensure_future(_hebbian_link_task(results))
    # ── NEW (Step 8, background): reforço das arestas EXISTENTES do subgrafo
    # PPR pela regra de co-ativação (Opção B). Também fire-and-forget. ──
    asyncio.ensure_future(_ppr_reinforce_task(ppr_ctx))

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
    '_auto_link_backfill',
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
