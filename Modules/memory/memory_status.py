from __future__ import annotations

"""
memory_status.py — Endpoint de status/diagnóstico.

Agrega contagens e parâmetros de todos os stores num único dict síncrono
(roda via run_in_executor para não bloquear o event loop).
"""

import asyncio

from memory_config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONTEXT_MAX_CHARS,
    CONTEXT_TURNS_FETCH,
    DECAY_HALF_LIFE_DAYS,
    DEDUP_THRESHOLD,
    DUAL_CONTEXT_WEIGHT,
    EDGE_DECAY_HALF_LIFE_DAYS,
    EDGE_LEARNING_RATE,
    EDGE_MIN_SCORE_TO_LINK,
    EDGE_PRUNE_THRESHOLD,
    EDGE_TYPES,
    FD_AMBIGUOUS_MARGIN,
    FD_EMBED_DIM,
    FD_MIN_SCORE,
    FD_TOP_K,
    IF_MAX_CHUNKS,
    IF_MAX_CONTENT_SIZE,
    IF_MIN_SCORE,
    IF_MIN_SCORE_READ,
    LANCE_DIR,
    LTP_ACCESS_BOOST,
    MMR_COSINE_FLOOR,
    MMR_LAMBDA,
    ONNX_SERVING_URL,
    PPR_DAMPING,
    PPR_EXPAND_MAX_NODES,
    PPR_EXPAND_MIN_PATH_WEIGHT,
    PPR_SPREAD_WEIGHT,
    QUERY_AMBIGUOUS_RATIO,
    QUERY_SHORT_WORDS,
    READ_IF_MAX_CHARS,
    READ_LT_MAX_CHARS,
    READ_MIN_SCORE_STRICT,
    READ_ST_MAX_CHARS,
    READ_TOP_K_FINAL,
    READ_TOTAL_MAX_CHARS,
    READ_VS_MAX_CHARS,
    RERANK_DUPLICATE_SCORE,
    RERANK_UPDATE_SCORE,
    ST_CLEANUP_INTERVAL_S,
    ST_TTL_HOURS,
    TOOLS_FALLBACK_TOP_K,
    TOOLS_LOW_CONFIDENCE_SCORE,
    TOOLS_MIN_SCORE,
    TOOLS_TOP_K,
    UPDATE_SIM_THRESHOLD,
    VD_AMBIGUOUS_MARGIN,
    VD_EMBED_DIM,
    VD_MIN_SCORE,
    VD_TOP_K,
    VO_AMBIGUOUS_MARGIN,
    VO_EMBED_DIM,
    VO_MIN_SCORE,
    VO_TOP_K,
    VS_MIN_SCORE,
)

from memory_state import state

# ── GET /status ────────────────────────────────────────────────────────────────

def _gather_status_sync() -> dict:
    """Coleta bloqueante (várias queries SQLite + state.vs.status()).
    Extraída para função síncrona própria para poder rodar via
    run_in_executor (ver status() abaixo) — igual ao padrão já usado no
    resto do arquivo para .search()/.apply_decay()/etc. Sem isso, essas
    ~8 queries síncronas rodavam direto no corpo da coroutine e, sob
    contenção com uma escrita concorrente seguranco o lock/transação
    SQLite (embedding batch, decay job, cleanup de short-term — todas via
    run_in_executor em outra thread), travavam o event loop inteiro
    enquanto esperavam — inclusive impedindo o servidor de responder ao
    handshake MCP de outros clientes (ver orchestrator._get_mcp_session)."""
    resp = {
        "long_term": {
            "memories_total":       state.lt_db.count(),
            "index_vectors":        state.lt_index.total,
            "decay_half_life_days": DECAY_HALF_LIFE_DAYS,
            "dedup_threshold":      DEDUP_THRESHOLD,
            "update_sim_threshold": UPDATE_SIM_THRESHOLD,
            "rerank_duplicate_score": RERANK_DUPLICATE_SCORE,
            "rerank_update_score":    RERANK_UPDATE_SCORE,
        },
        # ── NEW: grafo Hebbiano de co-ativação ──
        "hebbian_graph": {
            "edges_total":              state.lt_db.count_edges(),
            "edge_decay_half_life_d":   EDGE_DECAY_HALF_LIFE_DAYS,
            "edge_prune_threshold":     EDGE_PRUNE_THRESHOLD,
            "edge_learning_rate":       EDGE_LEARNING_RATE,
            "edge_min_score_to_link":   EDGE_MIN_SCORE_TO_LINK,
            "ltp_access_boost":         LTP_ACCESS_BOOST,
            "ppr_damping":              PPR_DAMPING,
            "ppr_spread_weight":        PPR_SPREAD_WEIGHT,
            "mmr_lambda":               MMR_LAMBDA,
            "ppr_expand_max_nodes":     PPR_EXPAND_MAX_NODES,
            "ppr_expand_min_path_w":    PPR_EXPAND_MIN_PATH_WEIGHT,
            "mmr_cosine_floor":         MMR_COSINE_FLOOR,
            "edge_types":               list(EDGE_TYPES),
        },
        # ── NEW: LanceDB ──
        "vector_store": {
            "backend": "lancedb",
            "path":    LANCE_DIR,
        },
        # ── NEW: RAG de tools ──
        "tools": {
            "tools_total":        state.tl_db.count(),
            "vectors_total":      state.tl_index.total,
            "edges_total":        state.tl_db.count_edges(),
            "top_k":              TOOLS_TOP_K,
            "min_score":          TOOLS_MIN_SCORE,
            "low_confidence":     TOOLS_LOW_CONFIDENCE_SCORE,
            "fallback_top_k":     TOOLS_FALLBACK_TOP_K,
        },
        "short_term": {
            "turn_groups_total":  state.st_db.count(),
            "index_vectors":      state.st_index.total,
            "ttl_hours":          ST_TTL_HOURS,
            "cleanup_interval_s": ST_CLEANUP_INTERVAL_S,
        },
        "contextual_search": {
            "query_short_words":     QUERY_SHORT_WORDS,
            "query_ambiguous_ratio": QUERY_AMBIGUOUS_RATIO,
            "context_max_chars":     CONTEXT_MAX_CHARS,
            "context_turns_fetch":   CONTEXT_TURNS_FETCH,
            "dual_context_weight":   DUAL_CONTEXT_WEIGHT,
        },
        # ── Otimização de tokens (anti 413 Payload Too Large) ──
        "token_optimization": {
            "read_top_k_final":       READ_TOP_K_FINAL,
            "read_total_max_chars":   READ_TOTAL_MAX_CHARS,
            "read_lt_max_chars":      READ_LT_MAX_CHARS,
            "read_st_max_chars":      READ_ST_MAX_CHARS,
            "read_vs_max_chars":      READ_VS_MAX_CHARS,
            "read_if_max_chars":      READ_IF_MAX_CHARS,
            "read_min_score_strict":  READ_MIN_SCORE_STRICT,
            "if_min_score_read":      IF_MIN_SCORE_READ,
        },
        "onnx_serving": {
            "url": ONNX_SERVING_URL,
            "mode": "remote_api",
        },
        # ── NEW: Indexed files status ──
        "indexed_files": {
            "files_total":    state.if_db.count_files(),
            "chunks_total":   state.if_db.get_total_chunks(),
            "index_vectors":  state.if_index.total,
            "min_score":      IF_MIN_SCORE,
            "chunk_size":     CHUNK_SIZE,
            "chunk_overlap":  CHUNK_OVERLAP,
            "max_content":    IF_MAX_CONTENT_SIZE,
            "max_chunks":     IF_MAX_CHUNKS,
        },
        # ── NEW: Visual dictionary status ──
        "visual_dict": {
            "concepts_total":  state.vd_db.count(),
            "embeddings_total": state.vd_index.total,
            "embed_dim":       VD_EMBED_DIM,
            "min_score":       VD_MIN_SCORE,
            "top_k":           VD_TOP_K,
            "ambiguous_margin": VD_AMBIGUOUS_MARGIN,
        },
        # ── NEW: Face dictionary status ──
        "face_dict": {
            "people_total":     state.fd_db.count(),
            "embeddings_total": state.fd_index.total,
            "embed_dim":        FD_EMBED_DIM,
            "min_score":        FD_MIN_SCORE,
            "top_k":            FD_TOP_K,
            "ambiguous_margin": FD_AMBIGUOUS_MARGIN,
        },
        # ── NEW: Voice dictionary status ──
        "voice_dict": {
            "people_total":     state.vo_db.count(),
            "embeddings_total": state.vo_index.total,
            "embed_dim":        VO_EMBED_DIM,
            "min_score":        VO_MIN_SCORE,
            "top_k":            VO_TOP_K,
            "ambiguous_margin": VO_AMBIGUOUS_MARGIN,
        },
    }
    if state.vs is not None:
        vs_status = state.vs.status()
        resp["knowledge"] = {
            "available":        True,
            "chunks_in_db":     vs_status["chunks_in_db"],
            "vectors_in_index": vs_status["vectors_in_index"],
            "embed_dim":        vs_status["embed_dim"],
            "dedup_threshold":  vs_status["dedup_threshold"],
            "vs_min_score":     VS_MIN_SCORE,
        }
    else:
        resp["knowledge"] = {
            "available":        False,
            "chunks_in_db":     0,
            "vectors_in_index": 0,
        }
    return resp


async def status():
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _gather_status_sync)


__all__ = [
    '_gather_status_sync',
    'status',
]
