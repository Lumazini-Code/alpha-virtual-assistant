# Alpha AI (AVA) — Estado global do processo + ciclo de vida
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
memory_state.py — Estado global do processo + ciclo de vida.

AppState (dataclass com todos os DBs/índices/engines), singleton `state`,
os três jobs em background (decay, cleanup de curto prazo, flush/compactação
LanceDB), startup()/shutdown() e a integração opcional com o VectorStore
(modules.vector_store — busca de conhecimento KG-RAG).
"""

import asyncio
import asyncio
import shutil
import time
from dataclasses import dataclass, field
from typing import Optional

from memory_config import (
    AUTO_LINK_ENABLED,
    DB_PATH,
    DECAY_HALF_LIFE_DAYS,
    DECAY_JOB_INTERVAL_S,
    EDGE_DECAY_HALF_LIFE_DAYS,
    EDGE_PRUNE_THRESHOLD,
    EMBED_DIM,
    FAISS_ID_MAP_PATH,
    FAISS_INDEX_PATH,
    FD_EMBED_DIM,
    FD_FAISS_ID_MAP_PATH,
    FD_FAISS_INDEX_PATH,
    GENERAL_SCOPE,
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
    PD_DB_PATH,
    PLANS_DB_PATH,
    ScopePaths,
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
    VO_EMBED_DIM,
    VO_FAISS_ID_MAP_PATH,
    VO_FAISS_INDEX_PATH,
    VS_DB_PATH,
    VS_FAISS_ID_MAP_PATH,
    VS_FAISS_INDEX_PATH,
    VS_MIN_SCORE,
    is_valid_scope,
    log,
    resolve_scope_paths,
    scope_db_dir,
)
from memory_db import (
    IndexedFilesDB,
    MemoryDB,
    PeopleDB,
    PlansDB,
    ShortTermDB,
    ToolsDB,
    VisualDictDB,
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
class ScopeState:
    """Artefatos ESCOPADOS de um escopo (modo geral ou projeto): LT + grafo
    Hebbiano, arquivos indexados, planos, LanceDB próprio, KG-RAG e log de
    ativação — todos sob db/<slug>/ (o geral usa db/ raiz, sem migração).
    ST, dicionários, cards de pessoa e RAG de tools são GLOBAIS (AppState)."""
    name:      str                 # "general" ou slug do projeto
    paths:     ScopePaths          = field(default=None)
    lance:     LanceStore          = field(default=None)   # store DO escopo (o geral compartilha state.lance)
    lt_db:     MemoryDB            = field(default=None)
    lt_index:  MemoryIndex         = field(default=None)
    if_db:     IndexedFilesDB      = field(default=None)
    if_index:  MemoryIndex         = field(default=None)
    plans_db:  PlansDB             = field(default=None)
    vs:        Optional[VectorStore] = field(default=None)

    def close(self):
        """Flush dos índices + fechamento das conexões SQLite. Chamado ao
        remover um escopo (delete_scope) — silencioso por design."""
        for idx in (self.lt_index, self.if_index):
            if idx is not None:
                try:
                    idx.flush()
                except Exception as e:
                    log.error(f"[{self.name}] falha ao compactar índice no close: {e}")
        for db in (self.lt_db, self.if_db, self.plans_db):
            if db is not None:
                try:
                    db.close()
                except Exception as e:
                    log.error(f"[{self.name}] falha ao fechar DB no close: {e}")


@dataclass
class AppState:
    embed_engine: EmbeddingEngine = field(default=None)
    rerank_engine: RerankEngine   = field(default=None)
    # ── GLOBAIS (compartilhados entre o geral e todos os projetos) ──────────
    # `lance` é o store GLOBAL (db/ava_lance) — usado por ST/dicionários/
    # pessoas/tools. O escopo geral o COMPARTILHA (mesmo diretório); projetos
    # têm store próprio (db/<slug>/ava_lance).
    lance:        LanceStore      = field(default=None)
    st_db:        ShortTermDB     = field(default=None)
    st_index:     MemoryIndex     = field(default=None)
    # ── NEW: Dicionário Visual ──
    vd_db:        VisualDictDB    = field(default=None)
    vd_index:     MemoryIndex     = field(default=None)
    # ── Unificação: Dicionário de Pessoas (Voice + Face) ──
    pd_db:        PeopleDB        = field(default=None)
    fd_index:     MemoryIndex     = field(default=None)
    vo_index:     MemoryIndex     = field(default=None)
    # ── NEW: RAG de tools ──
    tl_db:        ToolsDB         = field(default=None)
    tl_index:     MemoryIndex     = field(default=None)
    decay_task:   asyncio.Task    = field(default=None)
    cleanup_task: asyncio.Task    = field(default=None)
    flush_task:   asyncio.Task    = field(default=None)
    # Auto-heal do seed de identidade (pode falhar no boot se o ONNX serving
    # subir depois da memory — re-tenta em background até o vínculo nascer).
    identity_seed_task: asyncio.Task = field(default=None)
    # ── ESCOPOS (projetos × geral) ──────────────────────────────────────────
    # `scopes` mantém TODO escopo já aberto em cache (switch = troca de
    # ponteiro, O(1) — sem reabrir DBs). "general" nasce aberto no startup.
    scopes:       dict            = field(default_factory=dict)
    active_scope: str             = GENERAL_SCOPE
    scope_lock:   asyncio.Lock    = field(default_factory=asyncio.Lock)

    # ── Compatibilidade: os handlers (memory_ops_*) acessam state.lt_db /
    # state.lt_index / state.if_db / ... — viram propriedades do ESCOPO ATIVO.
    # Nenhum ponto de call-site precisou mudar para a separação por projeto.
    @property
    def scope(self) -> ScopeState:
        s = self.scopes.get(self.active_scope)
        if s is None:                      # boot safety — geral sempre aberto
            s = self.scopes.get(GENERAL_SCOPE)
        return s

    @property
    def lt_db(self) -> MemoryDB:
        return self.scope.lt_db

    @property
    def lt_index(self) -> MemoryIndex:
        return self.scope.lt_index

    @property
    def if_db(self) -> IndexedFilesDB:
        return self.scope.if_db

    @property
    def if_index(self) -> MemoryIndex:
        return self.scope.if_index

    @property
    def plans_db(self) -> PlansDB:
        return self.scope.plans_db

    @property
    def vs(self) -> Optional[VectorStore]:
        return self.scope.vs

    # Índices globais que usam o store global (st/vd/fd/vo/tl) — sem propriedade
    # porque os handlers os acessam por state.st_index etc. direto.

state = AppState()


# ── Escopos — abertura/troca/remoção ───────────────────────────────────────────

def _open_scope(name: str, shared_lance: Optional[LanceStore] = None) -> ScopeState:
    """Abre os artefatos de conteúdo de um escopo (bloqueante — chame via
    run_in_executor). `shared_lance`: quando informado (escopo geral), o
    escopo COMPARTILHA o store global em vez de abrir uma 2ª conexão no
    mesmo diretório — preserva o comportamento de processo único de antes."""
    paths = resolve_scope_paths(name)
    sc = ScopeState(name=name, paths=paths)
    sc.lance    = shared_lance if shared_lance is not None else LanceStore(paths.LANCE_DIR)
    sc.lt_db    = MemoryDB(paths.LT_DB_PATH)
    sc.lt_index = MemoryIndex(sc.lance, MT_LONG_TERM, EMBED_DIM,
                              paths.LT_FAISS_INDEX_PATH, paths.LT_FAISS_ID_MAP_PATH)
    # Log de ativação do grafo por escopo (memory_graph lê via getattr).
    sc.lt_db._activation_log_path = paths.ACTIVATION_LOG_PATH
    sc.if_db    = IndexedFilesDB(paths.IF_DB_PATH)
    sc.if_index = MemoryIndex(sc.lance, MT_INDEXED_FILE, EMBED_DIM,
                              paths.IF_FAISS_INDEX_PATH, paths.IF_FAISS_ID_MAP_PATH)
    sc.plans_db = PlansDB(paths.PLANS_DB_PATH)
    if _VS_AVAILABLE:
        try:
            sc.vs = VectorStore(
                index_path  = paths.VS_FAISS_INDEX_PATH,
                db_path     = paths.VS_DB_PATH,
                id_map_path = paths.VS_FAISS_ID_MAP_PATH,
                embed_dim   = EMBED_DIM,
            )
        except Exception as e:
            log.error(f"[{name}] VectorStore initialization failed: {e} — knowledge search disabled")
            sc.vs = None
    log.info(
        f"Escopo '{name}' aberto — {sc.lt_db.count()} memórias LT | "
        f"{sc.lt_db.count_edges()} arestas | {sc.if_db.count_files()} arquivos "
        f"indexados | {sc.plans_db.count()} planos ({paths.db_dir})"
    )
    return sc


async def switch_scope(name: str) -> dict:
    """Ativa um escopo (projeto ou geral). Escopo já em cache → troca de
    ponteiro O(1); novo → abre sob o lock (a troca nunca afeta um handler em
    voo mais do que a própria atribuição do ponteiro — handlers capturam os
    objetos via state.* na hora do uso)."""
    if not is_valid_scope(name):
        raise RuntimeError(f"Escopo inválido: {name!r}")
    async with state.scope_lock:
        sc = state.scopes.get(name)
        opened = sc is None
        if sc is None:
            # Escopo geral compartilha o store global (mesmo diretório).
            shared = state.lance if name == GENERAL_SCOPE else None
            sc = await asyncio.get_event_loop().run_in_executor(
                None, _open_scope, name, shared
            )
            state.scopes[name] = sc
        previous = state.active_scope
        state.active_scope = name
    log.info(f"Escopo ativo: '{previous}' → '{name}'" + (" (aberto agora)" if opened else " (cache)"))
    return {"scope": name, "opened": opened, "previous": previous}


async def delete_scope(name: str) -> None:
    """Fecha (se aberto) e APAGA o diretório do escopo — memórias LT, grafo,
    planos, arquivos indexados, LanceDB e log, tudo. Recusa o escopo geral e
    o escopo ativo (o chamador deve desativar antes)."""
    if name == GENERAL_SCOPE:
        raise RuntimeError("O escopo 'general' não pode ser removido")
    if not is_valid_scope(name):
        raise RuntimeError(f"Escopo inválido: {name!r}")
    async with state.scope_lock:
        if state.active_scope == name:
            raise RuntimeError(f"Escopo '{name}' está ATIVO — desative antes de remover")
        sc = state.scopes.pop(name, None)
        if sc is not None:
            await asyncio.get_event_loop().run_in_executor(None, sc.close)
        d = scope_db_dir(name)
        # Guarda de segurança: só remove diretórios DENTRO de db/ (o scope_db_dir
        # já valida o slug; a checagem explícita é defesa em profundidade).
        if str(d) != str(scope_db_dir(GENERAL_SCOPE)) and d.is_dir():
            await asyncio.get_event_loop().run_in_executor(None, shutil.rmtree, d)
    log.info(f"Escopo '{name}' removido ({d})")


def _all_indices() -> list["MemoryIndex"]:
    """Todos os índices vetoriais (LanceDB) do processo — globais + os de
    TODOS os escopos abertos. Usado pelo job de manutenção periódica e pelo
    shutdown para compactar tudo de uma vez."""
    idxs = [
        idx for idx in (
            state.st_index, state.vd_index, state.fd_index, state.vo_index,
            state.tl_index,
        )
        if idx is not None
    ]
    for sc in state.scopes.values():
        idxs.extend(idx for idx in (sc.lt_index, sc.if_index) if idx is not None)
    return idxs


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

async def startup():
    """Inicializa o estado global da memória (DBs, índices, engines, jobs em
    background). Artefatos GLOBAIS (ST, dicionários, pessoas, tools) + o
    escopo 'general' (LT+grafo, arquivos indexados, planos) — projetos são
    abertos sob demanda via switch_scope (o orchestrator re-afirma o escopo
    ativo no boot e a cada turno)."""
    log.info("Iniciando AVA Memory...")

    try:
        from onnx_client import check_health
        health = await check_health(ONNX_SERVING_URL)
        log.info(f"ONNX Serving API healthy: {health}")
    except Exception as e:
        log.warning(f"ONNX Serving API not reachable at {ONNX_SERVING_URL}: {e}")
        log.warning("Memory API will start but embedding calls will fail until ONNX serving is available.")

    state.embed_engine = EmbeddingEngine(base_url=ONNX_SERVING_URL)
    state.rerank_engine = RerankEngine(base_url=ONNX_SERVING_URL)

    # ── GLOBAIS: um LanceDB só (db/ava_lance) p/ ST/dicionários/pessoas/tools;
    # FAISS_*_PATH viram fontes de migração ──────────────────────────────────
    state.lance = LanceStore(LANCE_DIR)

    state.st_db    = ShortTermDB(ST_DB_PATH)
    state.st_index = MemoryIndex(state.lance, MT_SHORT_TERM, EMBED_DIM,
                                 ST_FAISS_INDEX_PATH, ST_FAISS_ID_MAP_PATH)

    # ── NEW: Dicionário Visual ──
    state.vd_db    = VisualDictDB(VD_DB_PATH)
    state.vd_index = MemoryIndex(state.lance, MT_VISUAL_DICT, VD_EMBED_DIM,
                                 VD_FAISS_INDEX_PATH, VD_FAISS_ID_MAP_PATH)

    # ── Unificação: Card de Pessoa (PeopleDB) — Voice + Face ──
    # Um único banco de pessoas: os metadados (nome, descrição, gostos, etc.)
    # moram na tabela `people`, enquanto os embeddings de voz e rosto ficam
    # em tabelas separadas (dimensionalidades diferentes), ambos apontando
    # pro mesmo person_id. Os índices vetoriais (fd_index/vo_index) continuam
    # separados pelas mesmas razões.
    state.pd_db    = PeopleDB(PD_DB_PATH)
    state.fd_index = MemoryIndex(state.lance, MT_FACE_DICT, FD_EMBED_DIM,
                                 FD_FAISS_INDEX_PATH, FD_FAISS_ID_MAP_PATH)
    state.vo_index = MemoryIndex(state.lance, MT_VOICE_DICT, VO_EMBED_DIM,
                                 VO_FAISS_INDEX_PATH, VO_FAISS_ID_MAP_PATH)

    # ── NEW: RAG de tools (índice novo — sem legado FAISS) ──
    state.tl_db    = ToolsDB(TOOLS_DB_PATH)
    state.tl_index = MemoryIndex(state.lance, MT_TOOL, EMBED_DIM)

    # ── ESCOPO GERAL (db/ raiz — paths idênticos aos de sempre, zero migração;
    # compartilha o store Lance global que fica no mesmo diretório) ──
    state.scopes[GENERAL_SCOPE] = await asyncio.get_event_loop().run_in_executor(
        None, _open_scope, GENERAL_SCOPE, state.lance,
    )
    state.active_scope = GENERAL_SCOPE

    # ── NEW: seed do card de identidade da Alpha (PeopleDB `identity`) ──
    # Só popula se a tabela estiver vazia (ou existir sem vínculo de memória).
    # Import local de propósito: memory_ops_dicts importa memory_state — um
    # import de topo aqui criaria ciclo. Falha (ex.: ONNX ainda fora do ar)
    # vira warning — o boot da memory nunca quebra por causa disso.
    try:
        from memory_ops_dicts import identity_seed
        await identity_seed()
    except Exception as e:
        log.warning(f"Identity: seed do card de identidade pulado: {e}")
    # Auto-heal em BACKGROUND: no boot a memory API pode subir ANTES do ONNX
    # serving (embeddings) — o seed direto falha com ConnectError e, sem este
    # retry, o card (e o nó "a1" no graph activation) só nasceria no próximo
    # restart. A task retorna de imediato se o seed deu certo.
    state.identity_seed_task = asyncio.create_task(_identity_seed_retry())

    state.decay_task   = asyncio.create_task(decay_job())
    state.cleanup_task = asyncio.create_task(st_cleanup_job())
    state.flush_task   = asyncio.create_task(index_flush_job())

    log.info(
        f"Pronto — {state.lt_db.count()} memórias LT | "
        f"{state.st_db.count()} grupos ST | "
        f"{state.vs.total if state.vs else 0} chunks de conhecimento | "
        f"{state.if_db.count_files()} arquivos indexados ({state.if_db.get_total_chunks()} chunks) | "
        f"{state.vd_db.count()} conceitos visuais ({state.vd_index.total} embeddings) | "
        f"{state.pd_db.count()} pessoas cadastradas ({state.fd_index.total} emb. de rosto, {state.vo_index.total} emb. de voz) | "
        f"{state.tl_db.count()} tools indexadas ({state.tl_index.total} vetores) | "
        f"{state.plans_db.count()} planos"
    )


async def _identity_seed_retry(max_attempts: int = 40, delay: float = 15.0) -> None:
    """Auto-heal do seed de identidade em background.

    No boot a memory API pode subir ANTES do ONNX serving (embeddings) — o
    seed direto falha com ConnectError e, sem este retry, o card de
    identidade (e o nó "a1" no graph activation) só nasceria no próximo
    restart. Idempotente: pronto quando o card existe COM memory_id; se o
    seed direto do boot deu certo, a 1ª checagem já encerra a task."""
    from memory_ops_dicts import identity_seed   # import local (ciclo)

    def _ok() -> bool:
        try:
            row = state.pd_db.get_identity()
            return row is not None and row["memory_id"] is not None
        except Exception:
            return False

    if _ok():
        return
    for attempt in range(1, max_attempts + 1):
        try:
            await identity_seed()
        except Exception as e:
            log.info(
                f"Identity: retry {attempt}/{max_attempts} do seed — "
                f"{type(e).__name__}: {e}"
            )
        if _ok():
            log.info(f"Identity: seed em background concluído (tentativa {attempt})")
            return
        await asyncio.sleep(delay)
    log.warning(
        f"Identity: retry do seed esgotado ({max_attempts} tentativas de "
        f"{delay}s) — o card só ganha vínculo de memória quando o ONNX "
        f"serving responder (ou no próximo boot)."
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
    if state.identity_seed_task is not None:
        state.identity_seed_task.cancel()

    # Flush final e síncrono de todos os índices (globais + todos os escopos
    # abertos) — garante que nada gravado desde o último ciclo do
    # index_flush_job seja perdido ao encerrar.
    for index in _all_indices():
        try:
            index.flush()
        except Exception as e:
            log.error(f"Erro ao compactar índice LanceDB no shutdown: {e}")

    log.info("AVA Memory encerrada")


__all__ = [
    'AppState',
    'ScopeState',
    '_all_indices',
    '_open_scope',
    'decay_job',
    'delete_scope',
    'index_flush_job',
    'shutdown',
    'st_cleanup_job',
    'startup',
    'state',
    'switch_scope',
]
