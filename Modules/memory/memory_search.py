from __future__ import annotations

"""
memory_search.py — Pipeline de busca, ranqueamento e construção de resultados.

Entendimento de query (classificação, segmentação heurística), estratégias
de busca (expanded/dual/segmented), fusão de scores, propagação PPR no grafo
Hebbiano (_ppr_spread*), inibição lateral (MMR), tasks hebbianas em
background e os builders _build_*_entries (LT/ST/VS/IF → MemoryEntry).
"""

import asyncio
import json
import re
import sqlite3
import time
from typing import Optional

import numpy as np

from memory_config import (
    CONTEXT_MAX_CHARS,
    CONTEXT_TURNS_FETCH,
    DICT_GRAPH_MAX_RELATED,
    DUAL_CONTEXT_WEIGHT,
    EDGE_LEARNING_RATE,
    EDGE_MIN_SCORE_TO_LINK,
    EDGE_MMR_FACTORS,
    HEBBIAN_LINK_MAX_IDS,
    MMR_COSINE_FLOOR,
    MMR_LAMBDA,
    PPR_DAMPING,
    PPR_EXPAND_MAX_NODES,
    PPR_EXPAND_MIN_PATH_WEIGHT,
    PPR_MAX_ITER,
    PPR_CONVERGENCE_EPS,
    PPR_MIN_ACTIVATION_REINFORCE,
    PPR_MIN_NEIGHBORS_PER_HOP,
    PPR_SPREAD_WEIGHT,
    QUERY_AMBIGUOUS_RATIO,
    QUERY_SHORT_WORDS,
    READ_IF_MAX_CHARS,
    READ_LT_MAX_CHARS,
    READ_ST_MAX_CHARS,
    READ_VS_MAX_CHARS,
    SEGMENT_MAX_COUNT,
    SEGMENT_MIN_WORDS,
    SEGMENT_TRIGGER_WORDS,
    _STOP_WORDS,
    log,
)
from memory_graph import (
    _build_adjacency,
    _edge_key,
    _edge_traversal_weights,
    _expand_graph_sync,
    _log_activation_sync,
    personalized_pagerank,
)
from memory_models import MemoryEntry, RelatedMemory, Turn
from memory_state import state

# ── Helpers de busca contextual ────────────────────────────────────────────────

def _truncate_text(text: str, max_chars: int) -> str:
    """
    Trunca texto preservando o início e quebrando em fronteira de palavra.
    Adiciona '...' quando truncado. Usado para reduzir tokens no /read.
    """
    if not text or len(text) <= max_chars:
        return text
    # Reserva 3 chars para '...'
    cut = text[:max(max_chars - 3, 1)]
    # Tenta cortar em fronteira de palavra para não cortar no meio de token
    last_space = cut.rfind(' ')
    if last_space > max_chars // 2:
        cut = cut[:last_space]
    return cut.rstrip() + "..."


def _apply_token_budget(
    entries: list[MemoryEntry],
    total_max_chars: int,
    top_k: int,
) -> list[MemoryEntry]:
    """
    Aplica orçamento global de caracteres no resultado final do /read.
    Corta entradas de menor score primeiro quando o total excede o teto.
    """
    if not entries:
        return entries
    # Primeiro garante o top_k
    selected = entries[:top_k]
    # Depois corta de trás pra frente (menor score) enquanto exceder o teto
    while len(selected) > 1:
        total = sum(len(e.text or "") for e in selected)
        if total <= total_max_chars:
            break
        selected.pop()  # remove o último (menor score, já ordenado)
    return selected


# ── NEW: Segmentação heurística de query composta (sem LLM) ───────────────────

# Fronteira antes de gerúndios ("usando", "utilizando", "aplicando", "seguindo"
# ...) — em português, costumam introduzir um novo sub-meio/estratégia dentro
# da mesma frase.
_GERUND_BOUNDARY_RE = re.compile(r'(?<=\s)(?=\w+(?:ando|endo|indo)\b)', re.IGNORECASE)

# Fronteira antes de "para <verbo no infinitivo>" — costuma introduzir um
# novo sub-objetivo/resultado dentro da mesma frase.
_PURPOSE_BOUNDARY_RE = re.compile(r'(?<=\s)(?=para\s+\w+(?:ar|er|ir)\b)', re.IGNORECASE)


def _split_query_segments(query: str) -> list[str]:
    """
    Decompõe a query em sub-tópicos SEM LLM, para busca por segmento.

    Regra: (1) separa por pontuação forte → sentenças; (2) dentro de
    sentenças longas, separa por vírgula e por fronteiras heurísticas
    (gerúndio, "para <verbo>") que tendem a introduzir um novo sub-pedido;
    (3) funde fragmentos curtos demais no vizinho; (4) deduplica e limita
    a quantidade de segmentos.

    Cada segmento resultante é embedado e buscado separadamente — isso é o
    que evita a "diluição de atenção semântica" de embedar a pergunta
    inteira de uma vez.
    """
    query = query.strip()
    if not query:
        return []

    sentences = [s.strip() for s in re.split(r'[.!?;]+', query) if s.strip()]

    raw_segments: list[str] = []
    for sent in sentences:
        if len(sent.split()) <= SEGMENT_TRIGGER_WORDS:
            raw_segments.append(sent)
            continue

        for part in re.split(r'\s*,\s*', sent):
            for sub in _GERUND_BOUNDARY_RE.split(part):
                raw_segments.extend(
                    p.strip() for p in _PURPOSE_BOUNDARY_RE.split(sub) if p.strip()
                )

    # funde fragmentos curtos demais no segmento anterior
    merged: list[str] = []
    for seg in raw_segments:
        if merged and len(seg.split()) < SEGMENT_MIN_WORDS:
            merged[-1] = f"{merged[-1]} {seg}"
        else:
            merged.append(seg)
    if len(merged) >= 2 and len(merged[0].split()) < SEGMENT_MIN_WORDS:
        merged[1] = f"{merged[0]} {merged[1]}"
        merged.pop(0)

    # dedup preservando ordem
    seen: set[str] = set()
    dedup: list[str] = []
    for seg in merged:
        key = seg.lower()
        if key not in seen:
            seen.add(key)
            dedup.append(seg)

    # teto de segmentos — mantém os mais "carregados" semanticamente
    if len(dedup) > SEGMENT_MAX_COUNT:
        dedup = sorted(dedup, key=lambda s: len(s.split()), reverse=True)[:SEGMENT_MAX_COUNT]

    return dedup


def _fuse_max(
    base: list[tuple[int, float]],
    extra: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Une dois conjuntos (id, score) mantendo o maior score por id. Usado
    para combinar a busca da query inteira com a busca por segmento, sem
    que um sub-tópico "afogue" o score de outro (diferente de uma média)."""
    scores: dict[int, float] = dict(base)
    for mid, score in extra:
        if score > scores.get(mid, -1.0):
            scores[mid] = score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


async def _search_segmented(
    segments: list[str],
    top_k: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """
    Embeda TODOS os segmentos em uma única chamada em lote à API de
    embeddings, busca todos de uma vez no FAISS (batch nativo) e funde por
    max-score. Determinístico, sem LLM — apenas cosine similarity aplicado
    a cada pedaço da pergunta em vez da pergunta inteira.
    """
    embeddings = await state.embed_engine.embed_queries(segments)  # shape (N, dim)
    loop = asyncio.get_event_loop()
    lt_batches, st_batches = await asyncio.gather(
        loop.run_in_executor(None, state.lt_index.search_batch, embeddings, top_k),
        loop.run_in_executor(None, state.st_index.search_batch, embeddings, top_k),
    )
    lt_fused: list[tuple[int, float]] = []
    st_fused: list[tuple[int, float]] = []
    for seg_results in lt_batches:
        lt_fused = _fuse_max(lt_fused, seg_results)
    for seg_results in st_batches:
        st_fused = _fuse_max(st_fused, seg_results)
    return lt_fused, st_fused


def _classify_query(query: str) -> str:
    tokens = re.findall(r"\w+", query.lower())
    if not tokens:
        return "expanded"
    n_words      = len(tokens)
    n_stopwords  = sum(1 for t in tokens if t in _STOP_WORDS)
    stop_ratio   = n_stopwords / n_words
    is_short     = n_words <= QUERY_SHORT_WORDS
    is_ambiguous = stop_ratio >= QUERY_AMBIGUOUS_RATIO
    strategy = "dual" if (is_short or is_ambiguous) else "expanded"
    log.debug(f"Query classify: words={n_words} stop_ratio={stop_ratio:.2f} → {strategy}")
    return strategy


def _build_context_block(session_id: str) -> str:
    lines = state.st_db.get_recent_turns_text(session_id, CONTEXT_TURNS_FETCH)
    if not lines:
        return ""
    block = "\n".join(lines)
    if len(block) > CONTEXT_MAX_CHARS:
        block = block[-CONTEXT_MAX_CHARS:]
        newline_pos = block.find("\n")
        if newline_pos != -1:
            block = block[newline_pos + 1:]
    return block


def _fuse_scores(
    q_results: list[tuple[int, float]],
    c_results: list[tuple[int, float]],
    context_weight: float,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for mid, score in q_results:
        scores[mid] = (1.0 - context_weight) * score
    for mid, score in c_results:
        scores[mid] = scores.get(mid, 0.0) + context_weight * score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


async def _search_expanded(
    query: str,
    context_block: str,
    top_k: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    expanded = f"{context_block}\n\nquery atual: {query}" if context_block else query
    emb = await state.embed_engine.embed_query_one(expanded)
    loop = asyncio.get_event_loop()
    lt_raw, st_raw = await asyncio.gather(
        loop.run_in_executor(None, state.lt_index.search, emb, top_k * 2),
        loop.run_in_executor(None, state.st_index.search, emb, top_k * 2),
    )
    return lt_raw, st_raw


async def _search_dual(
    query: str,
    context_block: str,
    top_k: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    emb_query, emb_ctx = await state.embed_engine.embed_query_two(query, context_block)
    loop = asyncio.get_event_loop()
    lt_q, st_q, lt_c, st_c = await asyncio.gather(
        loop.run_in_executor(None, state.lt_index.search, emb_query, top_k * 2),
        loop.run_in_executor(None, state.st_index.search, emb_query, top_k * 2),
        loop.run_in_executor(None, state.lt_index.search, emb_ctx,   top_k * 2),
        loop.run_in_executor(None, state.st_index.search, emb_ctx,   top_k * 2),
    )
    lt_fused = _fuse_scores(lt_q, lt_c, DUAL_CONTEXT_WEIGHT)
    st_fused = _fuse_scores(st_q, st_c, DUAL_CONTEXT_WEIGHT)
    return lt_fused, st_fused


async def _ppr_spread_on(
    db,
    seeds: list[tuple[int, float]],
    loop: asyncio.AbstractEventLoop,
    ctx: Optional[dict] = None,
) -> dict:
    """Propagação PPR num grafo de arestas tipadas (`db` = MemoryDB ou ToolsDB),
    com sementes em `seeds` = [(id, score)]. Retorna:
      related_raw  — [(id, score)] dos ids alcançados SÓ pelo grafo
      ppr_rank     — {id: rank} completo (sementes + vizinhos), p/ Step 8
      adjacency    — {id: {vizinho: weight}} do subgrafo local, p/ Step 8
      edge_rows    — linhas brutas de aresta do subgrafo
      elapsed_ms   — tempo da propagação (profiling)

    `ctx` (opcional) é repassado ao log de ativação (query/strategy/session
    da leitura que semeou — enriquece a visualização em tempo real).
    """
    result: dict = {
        "related_raw": [], "ppr_rank": {}, "adjacency": {}, "edge_rows": [],
        "elapsed_ms": 0.0,
    }
    scores: dict[int, float] = {}
    for mid, score in seeds:
        scores[mid] = max(score, scores.get(mid, 0.0))
    seed_ids = list(scores)
    if not seed_ids:
        return result
    t0 = time.perf_counter()
    edges, discovered = await loop.run_in_executor(
        None, _expand_graph_sync, db, seed_ids,
        PPR_EXPAND_MAX_NODES, PPR_EXPAND_MIN_PATH_WEIGHT, PPR_MIN_NEIGHBORS_PER_HOP,
    )
    if not edges:
        return result
    adjacency = _build_adjacency(edges, discovered)
    if not adjacency:
        return result
    # Vetor de personalização: scores clipados (>= 0) e normalizados.
    clipped = {mid: max(score, 0.0) for mid, score in scores.items()}
    total = sum(clipped.values())
    if total <= 0:
        return result
    personalization = {mid: s / total for mid, s in clipped.items()}
    rank = await loop.run_in_executor(
        None, personalized_pagerank,
        adjacency, personalization, PPR_DAMPING, PPR_MAX_ITER, PPR_CONVERGENCE_EPS,
    )
    # ── NEW: dispara log de ativação p/ visualização em tempo real, sem bloquear ──
    asyncio.ensure_future(
        loop.run_in_executor(
            None, _log_activation_sync, seed_ids, rank,
            (time.perf_counter() - t0) * 1000.0, ctx,
        )
    )
    seed_set = set(seed_ids)
    related_raw = [
        (mid, PPR_SPREAD_WEIGHT * r)
        for mid, r in rank.items()
        if mid not in seed_set and r >= PPR_MIN_ACTIVATION_REINFORCE
    ]
    related_raw.sort(key=lambda x: x[1], reverse=True)
    result.update(
        related_raw=related_raw, ppr_rank=rank,
        adjacency=adjacency, edge_rows=edges,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )
    return result


async def _ppr_spread(
    lt_raw: list[tuple[int, float]],
    loop: asyncio.AbstractEventLoop,
    ctx: Optional[dict] = None,
) -> dict:
    """Step 6 — propagação PPR no grafo Hebbiano de memórias LT, com sementes
    nos hits LT. Roda DEPOIS que `lt_raw` está finalizado e ANTES de converter
    em MemoryEntry. `ctx` (opcional) enriquece o log de ativação."""
    return await _ppr_spread_on(state.lt_db, lt_raw, loop, ctx)


async def _dict_graph_related(
    seeds: list[tuple[Optional[int], float]],
    loop: asyncio.AbstractEventLoop,
) -> list[RelatedMemory]:
    """Busca por grafo a partir de hits de dicionário (objeto/rosto/voz): usa o
    `memory_id` de cada candidato como semente (score = similaridade do hit)
    e devolve as memórias LT alcançadas SÓ pelas arestas. Só leitura — não
    cria nem reforça arestas (leituras de dicionário são frequentes/por-frame
    e não devem fortalecer o grafo sozinhas). Candidatos sem memory_id não
    têm nó no grafo e são ignorados."""
    real = [(mid, s) for mid, s in seeds if mid is not None]
    if not real:
        return []
    ctx = await _ppr_spread(real, loop)
    related_raw = ctx["related_raw"][:DICT_GRAPH_MAX_RELATED]
    if not related_raw:
        return []
    rows = await loop.run_in_executor(
        None, state.lt_db.get_by_ids, [mid for mid, _ in related_raw]
    )
    by_id = {r["id"]: r for r in rows}
    return [
        RelatedMemory(
            memory_id=mid,
            text=_truncate_text(by_id[mid]["text"], READ_LT_MAX_CHARS),
            score=round(score, 4),
        )
        for mid, score in related_raw if mid in by_id
    ]


def _lateral_inhibition_filter(results: list[MemoryEntry]) -> list[MemoryEntry]:
    """Step 7 — MMR (inibição lateral) restrito às entradas "related".

    "Similaridade" entre dois candidatos:
      * se existe aresta entre eles → peso da aresta × EDGE_MMR_FACTORS[tipo]
        (o maior entre as arestas do par). Suprimir hub genérico é redundância
        GRÁFICA; contradicts/updates contam como redundância máxima (mesmo
        assunto, versões conflitantes); temporal_precedence conta pouco;
      * se NÃO existe aresta → cosine direta entre os embeddings (antes: 0.0,
        "totalmente diverso" mesmo p/ candidatos quase idênticos que nunca
        co-ativaram), descontada por MMR_COSINE_FLOOR.
    Entradas "primary" passam intactas."""
    primary = [r for r in results if r.match_type != "related"]
    related = [r for r in results if r.match_type == "related"]
    if len(related) <= 1:
        return results
    ids = [r.id for r in related]

    edge_sim: dict[tuple[int, int], float] = {}
    for row in state.lt_db.get_edges_for_ids(ids):
        sim = row["weight"] * EDGE_MMR_FACTORS.get(row["edge_type"], 1.0)
        key = _edge_key(row["memory_id_a"], row["memory_id_b"])
        edge_sim[key] = max(edge_sim.get(key, 0.0), sim)

    try:
        vecs = state.lt_index.get_vectors(ids)
    except Exception as e:
        log.warning(f"MMR: não foi possível buscar vetores p/ o fallback de cosine: {e}")
        vecs = {}
    unit: dict[int, np.ndarray] = {}
    for mid, v in vecs.items():
        n = float(np.linalg.norm(v))
        if n > 0:
            unit[mid] = v / n

    pair_cache: dict[tuple[int, int], float] = {}

    def pair_sim(i: int, j: int) -> float:
        key = _edge_key(i, j)
        cached = pair_cache.get(key)
        if cached is not None:
            return cached
        sim = edge_sim.get(key)
        if sim is None:                      # sem aresta → cosine direta
            vi, vj = unit.get(i), unit.get(j)
            if vi is None or vj is None:
                sim = 0.0
            else:
                cos = float(np.dot(vi, vj))
                sim = max(0.0, (cos - MMR_COSINE_FLOOR) / (1.0 - MMR_COSINE_FLOOR))
        pair_cache[key] = sim
        return sim

    selected: list[MemoryEntry] = []
    pool = list(related)
    while pool:
        def mmr_score(c: MemoryEntry) -> float:
            relevance = c.score * c.confidence
            if not selected:
                return relevance
            max_sim = max(pair_sim(c.id, s.id) for s in selected)
            return MMR_LAMBDA * relevance - (1.0 - MMR_LAMBDA) * max_sim

        best = max(pool, key=mmr_score)
        selected.append(best)
        pool.remove(best)

    # ATENÇÃO: NÃO reordenar por score aqui. O laço acima seleciona TODOS os
    # "related" (até esvaziar o pool), então um sort por relevância depois
    # descartava exatamente a ordem que o MMR acabou de calcular — o filtro
    # virava no-op e o corte do orçamento de tokens (entries[:top_k]) pegava os
    # mais relevantes, não os mais diversos. Os "related" sempre têm score PPR
    # (<= PPR_SPREAD_WEIGHT) abaixo dos "primary" (>= READ_MIN_SCORE_STRICT), então
    # primary (já ordenado por score) + related (ordem MMR) preserva a ordem
    # global e deixa o MMR decidir QUEM sobrevive ao corte.
    return primary + selected


def _upsert_edges_sync(pairs: list[tuple[int, int]], now: float) -> int:
    """Helper síncrono p/ executor: cria/reforça arestas em lote (Step 4)."""
    for id_a, id_b in pairs:
        state.lt_db.upsert_edge(id_a, id_b, EDGE_LEARNING_RATE, now)
    return len(pairs)


async def _hebbian_link_task(results: list[MemoryEntry]) -> None:
    """Step 4 (background) — regra hebbiana NO CONJUNTO FINAL do /read: todo
    par de memórias LT com score >= EDGE_MIN_SCORE_TO_LINK co-ativou, então
    cria/reforça a aresta. Fire-and-forget: loga falhas, nunca levanta —
    o cliente já recebeu a resposta antes disso começar."""
    try:
        scored = [
            (r.id, r.score)
            for r in results
            if r.memory_type == "long_term" and r.score >= EDGE_MIN_SCORE_TO_LINK
        ]
        if len(scored) < 2:
            return
        now = time.time()
        if len(scored) > HEBBIAN_LINK_MAX_IDS:
            # Safeguard: leitura multi-tópico não pode virar O(n²) de arestas
            # — liga cada id apenas ao de maior score e avisa.
            scored.sort(key=lambda x: x[1], reverse=True)
            hub_id = scored[0][0]
            pairs = [(hub_id, other) for other, _ in scored[1:]]
            log.warning(
                f"Grafo Hebbiano: {len(scored)} ids acima de {EDGE_MIN_SCORE_TO_LINK} "
                f"no /read — ligando cada um apenas ao top-score (anti-O(n²))"
            )
        else:
            ids = [mid for mid, _ in scored]
            pairs = [
                (ids[i], ids[j])
                for i in range(len(ids))
                for j in range(i + 1, len(ids))
            ]
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _upsert_edges_sync, pairs, now)
    except Exception as e:
        log.error(f"Grafo Hebbiano: falha ao criar arestas pós-/read: {e}")


async def _ppr_reinforce_task(ppr_ctx: dict) -> None:
    """Step 8 (background, Opção B) — reforço das arestas EXISTENTES do
    subgrafo PPR pela regra literal de co-ativação:
        reinforcement = rank_i * rank_j * w_ij
        w_new = w_old + lr * reinforcement * (1 - w_old)
    NÃO cria aresta nova (isso é exclusivamente do Step 4). Elegibilidade:
    rank_i e rank_j ambos > PPR_MIN_ACTIVATION_REINFORCE. Mais barato que
    rastrear caminhos exatos de propagação — e mais hebbiano de qualquer
    forma (co-ativação, não causalidade de caminho)."""
    try:
        rank = ppr_ctx.get("ppr_rank") or {}
        adjacency = ppr_ctx.get("adjacency") or {}
        if not rank or not adjacency:
            return
        updates: list[tuple[int, int, float]] = []
        seen: set[tuple[int, int]] = set()
        for a, neighbors in adjacency.items():
            rank_a = rank.get(a, 0.0)
            if rank_a <= PPR_MIN_ACTIVATION_REINFORCE:
                continue
            for b in neighbors:
                rank_b = rank.get(b, 0.0)
                if rank_b <= PPR_MIN_ACTIVATION_REINFORCE:
                    continue
                key = _edge_key(a, b)
                if key in seen:
                    continue  # adjacência é simétrica — não reforça 2x a mesma aresta
                seen.add(key)
                updates.append((key[0], key[1], rank_a * rank_b))
        if not updates:
            return
        now = time.time()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, state.lt_db.reinforce_edges_batch,
            updates, EDGE_LEARNING_RATE, PPR_MIN_ACTIVATION_REINFORCE, now,
        )
    except Exception as e:
        log.error(f"Grafo Hebbiano: falha no reforço por ativação PPR: {e}")


def _build_lt_entries(
    lt_raw: list[tuple[int, float]],
    min_score: float,
    loop: asyncio.AbstractEventLoop,
    max_chars: int = READ_LT_MAX_CHARS,
    related_ids: Optional[set[int]] = None,
) -> list[MemoryEntry]:
    ids_filtered = [mid for mid, score in lt_raw if score >= min_score]
    score_map    = {mid: score for mid, score in lt_raw}
    if not ids_filtered:
        return []
    related_ids = related_ids or set()
    entries = []
    for row in state.lt_db.get_by_ids(ids_filtered):
        entries.append(MemoryEntry(
            id           = row["id"],
            text         = _truncate_text(row["text"], max_chars),
            score        = round(score_map[row["id"]], 4),
            confidence   = round(row["confidence"], 4),
            created_at   = row["created_at"],
            access_count = row["access_count"],
            memory_type  = "long_term",
            source       = row["source"],
            forgettable  = bool(row["forgettable"]),
            ttl_days     = row["ttl_days"],
            # ── NEW (Hebbian graph): id alcançado só via PPR vira "related" ──
            match_type   = "related" if row["id"] in related_ids else "primary",
        ))
        loop.run_in_executor(None, state.lt_db.update_access, row["id"])
    return entries


def _build_st_entries(
    st_raw: list[tuple[int, float]],
    min_score: float,
    loop: asyncio.AbstractEventLoop,
    max_chars: int = READ_ST_MAX_CHARS,
) -> list[MemoryEntry]:
    ids_filtered = [mid for mid, score in st_raw if score >= min_score]
    score_map    = {mid: score for mid, score in st_raw}
    if not ids_filtered:
        return []
    entries = []
    for row in state.st_db.get_by_ids(ids_filtered):
        score      = score_map[row["id"]]
        turns_data = json.loads(row["turns_json"])
        turns      = [Turn(**t) for t in turns_data]
        # Representação compacta: prioriza último user turn, depois assistant
        # Antes: " | ".join(f"[role] content[:120]" for t in turns) — acumulava
        # Agora: apenas primeiro e último turn, cada um com 80 chars no máx
        if len(turns) <= 2:
            parts = [f"[{t.role}] {t.content[:100]}" for t in turns]
        else:
            first = turns[0]
            last  = turns[-1]
            parts = [
                f"[{first.role}] {first.content[:80]}",
                f"...(+{len(turns)-2} turns)...",
                f"[{last.role}] {last.content[:100]}",
            ]
        text_repr = " | ".join(parts)
        entries.append(MemoryEntry(
            id           = row["id"],
            text         = _truncate_text(text_repr, max_chars),
            score        = round(score, 4),
            confidence   = 1.0,
            created_at   = row["created_at"],
            access_count = row["access_count"],
            memory_type  = "short_term",
            session_id   = row["session_id"],
        ))
        loop.run_in_executor(None, state.st_db.update_access, row["id"])
    return entries


def _build_vs_entries(
    vs_results: list,
    min_score: float,
    max_chars: int = READ_VS_MAX_CHARS,
) -> list[MemoryEntry]:
    if not vs_results:
        return []
    entries = []
    for ventry, score in vs_results:
        if score < min_score:
            continue
        entries.append(MemoryEntry(
            id           = ventry.id,
            text         = _truncate_text(ventry.text, max_chars),
            score        = round(score, 4),
            confidence   = 1.0,
            created_at   = 0.0,
            access_count = 0,
            memory_type  = "knowledge",
            source       = ventry.source,
        ))
    return entries


# ── NEW: Build indexed file entries from FAISS search results ─────────────────

def _build_if_entries(
    if_raw: list[tuple[int, float]],
    min_score: float,
    loop: asyncio.AbstractEventLoop,
    return_full_content: bool = False,
    max_chars: int = READ_IF_MAX_CHARS,
) -> list[MemoryEntry]:
    """
    Converte resultados de busca FAISS de chunks em MemoryEntry.

    Se return_full_content=True, text contém o conteúdo completo do arquivo.
    Se False, text contém apenas o chunk que deu match (mais conciso para /read).
    Deduplica por file_id — mantém apenas o melhor score por arquivo.
    """
    if not if_raw:
        return []

    # Filtra por score mínimo
    filtered = [(cid, score) for cid, score in if_raw if score >= min_score]
    if not filtered:
        return []

    # Busca chunk records
    chunk_ids = [cid for cid, _ in filtered]
    chunk_rows = state.if_db.get_chunks_by_ids(chunk_ids)
    chunk_map = {row["id"]: row for row in chunk_rows}

    # Agrupa por file_id — mantém melhor score por arquivo
    file_best: dict[int, tuple[float, sqlite3.Row]] = {}
    for cid, score in filtered:
        chunk_row = chunk_map.get(cid)
        if chunk_row is None:
            continue
        fid = chunk_row["file_id"]
        if fid not in file_best or score > file_best[fid][0]:
            file_best[fid] = (score, chunk_row)

    if not file_best:
        return []

    # Busca file records
    file_ids = list(file_best.keys())
    file_rows = state.if_db._conn.execute(
        f"SELECT * FROM indexed_files WHERE id IN ({','.join('?' * len(file_ids))})",
        file_ids,
    ).fetchall()
    file_map = {row["id"]: row for row in file_rows}

    entries = []
    for fid, (score, chunk_row) in file_best.items():
        file_row = file_map.get(fid)
        if file_row is None:
            continue

        # No modo /read (return_full_content=False), sempre trunca o chunk
        # No modo full content (indexed-file/read), não trunca — preserva o original
        if return_full_content:
            text = file_row["content"]
        else:
            text = _truncate_text(chunk_row["chunk_text"], max_chars)

        entries.append(MemoryEntry(
            id           = fid,
            text         = text,
            score        = round(score, 4),
            confidence   = round(file_row["confidence"], 4),
            created_at   = file_row["created_at"],
            access_count = file_row["access_count"],
            memory_type  = "indexed_file",
            source       = file_row["source"],
            file_path    = file_row["file_path"],
            file_name    = file_row["file_name"],
            extension    = file_row["extension"],
            content_hash = file_row["content_hash"],
            file_hash    = file_row["file_hash"],
        ))
        loop.run_in_executor(None, state.if_db.update_access, fid)

    return entries


__all__ = [
    '_GERUND_BOUNDARY_RE',
    '_PURPOSE_BOUNDARY_RE',
    '_apply_token_budget',
    '_build_context_block',
    '_build_if_entries',
    '_build_lt_entries',
    '_build_st_entries',
    '_build_vs_entries',
    '_classify_query',
    '_dict_graph_related',
    '_fuse_max',
    '_fuse_scores',
    '_hebbian_link_task',
    '_lateral_inhibition_filter',
    '_ppr_reinforce_task',
    '_ppr_spread',
    '_ppr_spread_on',
    '_search_dual',
    '_search_expanded',
    '_search_segmented',
    '_split_query_segments',
    '_truncate_text',
    '_upsert_edges_sync',
]
