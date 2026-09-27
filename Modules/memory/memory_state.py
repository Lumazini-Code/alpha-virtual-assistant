from __future__ import annotations

"""
memory_state.py — Estado global do processo + ciclo de vida.

AppState (dataclass com todos os DBs/índices/engines), singleton `state`,
os três jobs em background (decay, cleanup de curto prazo, flush/compactação
LanceDB), startup()/shutdown() e a integração opcional com o VectorStore
(modules.vector_store — busca de conhecimento KG-RAG).
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from memory_config import (
    AUTO_LINK_BACKFILL,
    AUTO_LINK_BACKFILL_DELAY_S,
    AUTO_LINK_ENABLED,
    DB_PATH,
    DECAY_HALF_LIFE_DAYS,
    DECAY_JOB_INTERVAL_S,
    EDGE_DECAY_HALF_LIFE_DAYS,
    EDGE_PRUNE_THRESHOLD,
    EMBED_DIM,
    FAISS_ID_MAP_PATH,
    FAISS_INDEX_PATH,
    FD_DB_PATH,
    FD_EMBED_DIM,
    FD_FAISS_ID_MAP_PATH,
    FD_FAISS_INDEX_PATH,
    IF_DB_PATH,
    IF_FAISS_ID_MAP_PATH,
    IF_FAISS_INDEX_PATH,
    INDEX_FLUSH_INTERVAL_S,
    LANCE_DIR,
    MT_FACE_DICT,
    MT_INDEXED_FILE,
    MT_LONG_TERM,
    MT_SHORT_TERM,
    MT_TOOL,
    MT_VOICE_DICT,
    MT_VISUAL_DICT,
    ONNX_SERVING_URL,
    ST_CLEANUP_INTERVAL_S,
    ST_DB_PATH,
    ST_FAISS_ID_MAP_PATH,
    ST_FAISS_INDEX_PATH,
    ST_TTL_HOURS,
    TOOLS_DB_PATH,
    VD_DB_PATH,
    VD_EMBED_DIM,
    VD_FAISS_ID_MAP_PATH,
    VD_FAISS_INDEX_PATH,
    VO_DB_PATH,
    VO_EMBED_DIM,
    VO_FAISS_ID_MAP_PATH,
    VO_FAISS_INDEX_PATH,
    VS_DB_PATH,
    VS_FAISS_ID_MAP_PATH,
    VS_FAISS_INDEX_PATH,
    log,
)
from memory_db import (
    FaceDictDB,
    IndexedFilesDB,
    MemoryDB,
    ShortTermDB,
    ToolsDB,
    VisualDictDB,
    VoiceDictDB,
)
from memory_engines import EmbeddingEngine, RerankEngine
from memory_vector import LanceStore, MemoryIndex

# ── MODIFIED: Import VectorStore from vector_store (graceful fallback) ────────
try:
    from modules.vector_store import VectorStore, VectorEntry as VSVectorEntry
    _VS_AVAILABLE = True
except ImportError as _imp_err:
    _VS_AVAILABLE = False
    VectorStore = None          # type: ignore[assignment,misc]
    VSVectorEntry = None        # type: ignore[assignment,misc]

if not _VS_AVAILABLE:
    log.warning("vector_store module not available — knowledge search disabled")
else:
    log.info("vector_store module found — knowledge search enabled")


# ── Estado global ──────────────────────────────────────────────────────────────

@dataclass
class AppState:
    embed_engine: EmbeddingEngine = field(default=None)
    rerank_engine: RerankEngine   = field(default=None)
    lance:        LanceStore      = field(default=None)   # NEW: LanceDB (todos os índices vetoriais)
    lt_db:        MemoryDB        = field(default=None)
    lt_index:     MemoryIndex     = field(default=None)
    st_db:        ShortTermDB     = field(default=None)
    st_index:     MemoryIndex     = field(default=None)
    vs:           Optional[VectorStore] = field(default=None)
    # ── NEW: Indexed files ──
    if_db:        IndexedFilesDB  = field(default=None)
    if_index:     MemoryIndex     = field(default=None)
    # ── NEW: Dicionário Visual ──
    vd_db:        VisualDictDB    = field(default=None)
    vd_index:     MemoryIndex     = field(default=None)
    # ── NEW: Dicionário de Rostos ──
    fd_db:        FaceDictDB      = field(default=None)
    fd_index:     MemoryIndex     = field(default=None)
    # ── NEW: Dicionário de Voz ──
    vo_db:        VoiceDictDB     = field(default=None)
    vo_index:     MemoryIndex     = field(default=None)
    # ── NEW: RAG de tools ──
    tl_db:        ToolsDB         = field(default=None)
    tl_index:     MemoryIndex     = field(default=None)
    decay_task:   asyncio.Task    = field(default=None)
    cleanup_task: asyncio.Task    = field(default=None)
    flush_task:   asyncio.Task    = field(default=None)
    backfill_task: asyncio.Task   = field(default=None)

state = AppState()


def _all_indices() -> list["MemoryIndex"]:
    """Todos os índices vetoriais (LanceDB) do processo — usado pelo job de
    manutenção periódica e pelo shutdown para compactar tudo de uma vez."""
    return [
        idx for idx in (
            state.lt_index, state.st_index,
            state.if_index, state.vd_index, state.fd_index, state.vo_index,
            state.tl_index,
        )
        if idx is not None
    ]


# ── Jobs em background ─────────────────────────────────────────────────────────

async def decay_job():
    while True:
        await asyncio.sleep(DECAY_JOB_INTERVAL_S)
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, state.lt_db.apply_decay, DECAY_HALF_LIFE_DAYS)
            # ── NEW (Hebbian graph): decay + poda das arestas no MESMO job —
            # sem criar segundo loop/task (reusa o decay_task existente).
            pruned = await loop.run_in_executor(
                None, state.lt_db.decay_and_prune_edges,
                EDGE_DECAY_HALF_LIFE_DAYS, EDGE_PRUNE_THRESHOLD,
            )
            if pruned:
                log.info(
                    f"Grafo Hebbiano: {pruned} arestas podadas "
                    f"(threshold={EDGE_PRUNE_THRESHOLD})"
                )
        except Exception as e:
            log.error(f"Erro no decay job: {e}")


async def index_flush_job():
    """Compacta os índices LanceDB que receberam escritas desde o último ciclo
    (cada add() já é durável; isto só junta os fragmentos pequenos). Substitui o antigo `_save()` a cada escrita (ver ITEM 1 da
    revisão) — o custo de I/O passa a ser amortizado em vez de pago a cada
    add()/remove_ids()."""
    while True:
        await asyncio.sleep(INDEX_FLUSH_INTERVAL_S)
        loop = asyncio.get_event_loop()
        for index in _all_indices():
            try:
                await loop.run_in_executor(None, index.flush)
            except Exception as e:
                log.error(f"Erro ao compactar índice LanceDB: {e}")


async def st_cleanup_job():
    while True:
        await asyncio.sleep(ST_CLEANUP_INTERVAL_S)
        try:
            cutoff = time.time() - ST_TTL_HOURS * 3600
            rows = state.st_db._conn.execute(
                "SELECT id FROM turn_groups WHERE last_accessed < ?", (cutoff,)
            ).fetchall()
            expired_ids = {row["id"] for row in rows}
            if expired_ids:
                state.st_db.expire_old(ST_TTL_HOURS)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, state.st_index.remove_ids, expired_ids)
        except Exception as e:
            log.error(f"Erro no cleanup de curto prazo: {e}")


# ── Lifespan ───────────────────────────────────────────────────────────────────

def _resolve_vs_paths() -> tuple[str, str, str]:
    try:
        from config import FAISS_INDEX_PATH as _cfg_idx, FAISS_META_PATH as _cfg_meta
        idx_path    = str(_cfg_idx)
        meta_str    = str(_cfg_meta)
        db_path     = meta_str.replace(".json", ".db") if meta_str.endswith(".json") else meta_str + ".db"
        id_map_path = idx_path.replace(".index", "_id_map.npy")
        log.info(f"VS paths from config: index={idx_path} db={db_path}")
        return idx_path, db_path, id_map_path
    except ImportError:
        log.info(f"VS paths from defaults: index={VS_FAISS_INDEX_PATH} db={VS_DB_PATH}")
        return VS_FAISS_INDEX_PATH, VS_DB_PATH, VS_FAISS_ID_MAP_PATH


async def startup():
    """Inicializa o estado global da memória (DBs, índices FAISS, engine de
    embeddings, jobs em background). Chamado pelo MCP server no startup do
    processo — substitui o antigo `lifespan` do FastAPI."""
    log.info("Iniciando AVA Memory (MCP)...")

    try:
        from onnx_client import check_health
        health = await check_health(ONNX_SERVING_URL)
        log.info(f"ONNX Serving API healthy: {health}")
    except Exception as e:
        log.warning(f"ONNX Serving API not reachable at {ONNX_SERVING_URL}: {e}")
        log.warning("Memory API will start but embedding calls will fail until ONNX serving is available.")

    state.embed_engine = EmbeddingEngine(base_url=ONNX_SERVING_URL)
    state.rerank_engine = RerankEngine(base_url=ONNX_SERVING_URL)

    # ── NEW: LanceDB — um diretório só; FAISS_*_PATH viram fontes de migração ──
    state.lance = LanceStore(LANCE_DIR)

    state.lt_db    = MemoryDB(DB_PATH)
    state.lt_index = MemoryIndex(state.lance, MT_LONG_TERM, EMBED_DIM,
                                 FAISS_INDEX_PATH, FAISS_ID_MAP_PATH)

    state.st_db    = ShortTermDB(ST_DB_PATH)
    state.st_index = MemoryIndex(state.lance, MT_SHORT_TERM, EMBED_DIM,
                                 ST_FAISS_INDEX_PATH, ST_FAISS_ID_MAP_PATH)

    # ── NEW: Indexed files ──
    state.if_db    = IndexedFilesDB(IF_DB_PATH)
    state.if_index = MemoryIndex(state.lance, MT_INDEXED_FILE, EMBED_DIM,
                                 IF_FAISS_INDEX_PATH, IF_FAISS_ID_MAP_PATH)

    # ── NEW: Dicionário Visual ──
    state.vd_db    = VisualDictDB(VD_DB_PATH)
    state.vd_index = MemoryIndex(state.lance, MT_VISUAL_DICT, VD_EMBED_DIM,
                                 VD_FAISS_INDEX_PATH, VD_FAISS_ID_MAP_PATH)

    # ── NEW: Dicionário de Rostos ──
    state.fd_db    = FaceDictDB(FD_DB_PATH)
    state.fd_index = MemoryIndex(state.lance, MT_FACE_DICT, FD_EMBED_DIM,
                                 FD_FAISS_INDEX_PATH, FD_FAISS_ID_MAP_PATH)

    # ── NEW: Dicionário de Voz ──
    state.vo_db    = VoiceDictDB(VO_DB_PATH)
    state.vo_index = MemoryIndex(state.lance, MT_VOICE_DICT, VO_EMBED_DIM,
                                 VO_FAISS_INDEX_PATH, VO_FAISS_ID_MAP_PATH)

    # ── NEW: RAG de tools (índice novo — sem legado FAISS) ──
    state.tl_db    = ToolsDB(TOOLS_DB_PATH)
    state.tl_index = MemoryIndex(state.lance, MT_TOOL, EMBED_DIM)

    if _VS_AVAILABLE:
        try:
            vs_idx, vs_db, vs_idmap = _resolve_vs_paths()
            state.vs = VectorStore(
                index_path  = vs_idx,
                db_path     = vs_db,
                id_map_path = vs_idmap,
                embed_dim   = EMBED_DIM,
            )
            log.info(f"VectorStore integrado — {state.vs.total} chunks de conhecimento")
        except Exception as e:
            log.error(f"VectorStore initialization failed: {e} — knowledge search disabled")
            state.vs = None
    else:
        state.vs = None
        log.warning("VectorStore not available — knowledge search disabled")

    state.decay_task   = asyncio.create_task(decay_job())
    state.cleanup_task = asyncio.create_task(st_cleanup_job())
    state.flush_task   = asyncio.create_task(index_flush_job())

    # ── Auto-link backfill ("sono" do memory) ────────────────────────────────
    # Consolida retroativamente o grafo a cada boot: liga memórias gravadas
    # antes do auto-link existir / sob thresholds antigos. Fire-and-forget com
    # delay (ONNX serving ainda aquecendo). Idempotente — arestas existentes
    # só são reforçadas. Import lazy: memory_ops_memory importa este módulo
    # (state), então o import direto no topo criaria ciclo.
    if AUTO_LINK_ENABLED and AUTO_LINK_BACKFILL:
        async def _autolink_sleep():
            await asyncio.sleep(AUTO_LINK_BACKFILL_DELAY_S)
            try:
                import memory_ops_memory  # noqa: PLC0415 — ciclo evitado de propósito
                await memory_ops_memory._auto_link_backfill()
            except Exception as e:
                log.warning(f"[AUTO-LINK-BACKFILL] falhou (não-bloqueante): "
                            f"{type(e).__name__}: {e}")
        state.backfill_task = asyncio.ensure_future(_autolink_sleep())

    log.info(
        f"Pronto — {state.lt_db.count()} memórias LT | "
        f"{state.st_db.count()} grupos ST | "
        f"{state.vs.total if state.vs else 0} chunks de conhecimento | "
        f"{state.if_db.count_files()} arquivos indexados ({state.if_db.get_total_chunks()} chunks) | "
        f"{state.vd_db.count()} conceitos visuais ({state.vd_index.total} embeddings) | "
        f"{state.fd_db.count()} pessoas cadastradas ({state.fd_index.total} embeddings de rosto) | "
        f"{state.vo_db.count()} pessoas com voz cadastrada ({state.vo_index.total} embeddings de voz) | "
        f"{state.tl_db.count()} tools indexadas ({state.tl_index.total} vetores)"
    )


async def shutdown():
    """Encerra o estado global da memória de forma limpa. Chamado pelo MCP
    server no shutdown do processo — substitui a parte pós-`yield` do antigo
    `lifespan` do FastAPI."""
    await state.embed_engine.client.close()

    if state.decay_task is not None:
        state.decay_task.cancel()
    if state.cleanup_task is not None:
        state.cleanup_task.cancel()
    if state.flush_task is not None:
        state.flush_task.cancel()
    if state.backfill_task is not None and not state.backfill_task.done():
        state.backfill_task.cancel()

    # Flush final e síncrono de todos os índices — garante que nada gravado
    # desde o último ciclo do index_flush_job seja perdido ao encerrar.
    for index in _all_indices():
        try:
            index.flush()
        except Exception as e:
            log.error(f"Erro ao compactar índice LanceDB no shutdown: {e}")

    log.info("AVA Memory (MCP) encerrada")


__all__ = [
    'AppState',
    '_all_indices',
    '_resolve_vs_paths',
    'decay_job',
    'index_flush_job',
    'shutdown',
    'st_cleanup_job',
    'startup',
    'state',
]
