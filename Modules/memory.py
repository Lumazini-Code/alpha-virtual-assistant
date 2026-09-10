from __future__ import annotations

import re
import time
import json
import sqlite3
import hashlib
import asyncio
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np
import faiss
from pydantic import BaseModel

# ── MODIFIED: Import from onnx_client instead of local ONNX ───────────────────
from onnx_client import EmbeddingClient, DEFAULT_ONNX_BASE_URL

# ── MODIFIED: Import VectorStore from vector_store (graceful fallback) ────────
try:
    from modules.vector_store import VectorStore, VectorEntry as VSVectorEntry
    _VS_AVAILABLE = True
except ImportError as _imp_err:
    _VS_AVAILABLE = False
    VectorStore = None          # type: ignore[assignment,misc]
    VSVectorEntry = None        # type: ignore[assignment,misc]

# ── Configuração ───────────────────────────────────────────────────────────────

ONNX_SERVING_URL = DEFAULT_ONNX_BASE_URL

# Longo prazo
DB_PATH           = "./memory/ava_memory.db"
FAISS_INDEX_PATH  = "./memory/ava_memory.index"
FAISS_ID_MAP_PATH = "./memory/ava_id_map.npy"

# Curto prazo
ST_DB_PATH           = "./memory/ava_short_term.db"
ST_FAISS_INDEX_PATH  = "./memory/ava_short_term.index"
ST_FAISS_ID_MAP_PATH = "./memory/ava_short_term_id_map.npy"

# Knowledge (Vector Store / KG-RAG)
VS_DB_PATH           = "./memory/ava_kg_chunks.db"
VS_FAISS_INDEX_PATH  = "./memory/ava_kg_vectors.index"
VS_FAISS_ID_MAP_PATH = "./memory/ava_kg_vectors_id_map.npy"
VS_MIN_SCORE         = 0.70

# ── NEW: Indexed Files (local-scraping) ────────────────────────────────────────
IF_DB_PATH           = "./memory/ava_indexed_files.db"
IF_FAISS_INDEX_PATH  = "./memory/ava_indexed_files.index"
IF_FAISS_ID_MAP_PATH = "./memory/ava_indexed_files_id_map.npy"
IF_MIN_SCORE         = 0.75
IF_MAX_CONTENT_SIZE  = 500_000    # 500 KB — mesmo limite do local-scraping
IF_MAX_CHUNKS        = 2000       # máx chunks por arquivo
CHUNK_SIZE           = 500        # chars por chunk
CHUNK_OVERLAP        = 100        # chars de sobreposição entre chunks
IF_EMBED_BATCH_SIZE  = 64         # chunks por batch de embedding

# ── NEW: Dicionário Visual (módulo vision.py — "Tradução de Objetos") ─────────
# Armazena, por conceito visual, N embeddings de exemplo (DINOv3) + a
# descrição textual ("significado") do objeto. A descrição também é gravada
# na memória de longo prazo (mesma tabela `memories`), então ela é
# recuperável pela leitura normal (/read) igual a qualquer outra memória.
VD_DB_PATH           = "./memory/ava_visual_dict.db"
VD_FAISS_INDEX_PATH  = "./memory/ava_visual_dict.index"
VD_FAISS_ID_MAP_PATH = "./memory/ava_visual_dict_id_map.npy"
VD_EMBED_DIM         = 384     # DINOv3 ViT-S/16 (ajustar se trocar de encoder)
VD_MIN_SCORE         = 0.55    # similaridade mínima p/ considerar candidato válido
VD_TOP_K             = 5
VD_AMBIGUOUS_MARGIN  = 0.05    # se score#1 - score#2 < margem → resposta ambígua

# ── NEW: Dicionário de Rostos (módulo vision.py — reconhecimento facial) ──────
# Mesmo padrão do dicionário visual acima, só que pra pessoas: por pessoa,
# guarda N embeddings de exemplo (EdgeFace) — permite reconhecer o mesmo
# rosto em ângulos/luz diferentes. Fica em banco/índice separados porque a
# dimensão do embedding e os thresholds de similaridade de rosto costumam
# ser bem diferentes dos de objeto genérico (DINOv3).
FD_DB_PATH           = "./memory/ava_face_dict.db"
FD_FAISS_INDEX_PATH  = "./memory/ava_face_dict.index"
FD_FAISS_ID_MAP_PATH = "./memory/ava_face_dict_id_map.npy"
FD_EMBED_DIM         = 512     # EdgeFace — ajustar se a saída do seu .onnx for outra dim
FD_MIN_SCORE         = 0.42    # cosine similarity — modelos estilo ArcFace/EdgeFace costumam
                                # precisar de threshold mais baixo que embeddings genéricos
                                # (DINOv3 usa 0.55); CALIBRE em cima dos seus próprios exemplos
                                # antes de confiar nisso em produção
FD_TOP_K             = 3
FD_AMBIGUOUS_MARGIN  = 0.05

EMBED_DIM            = 384
READ_MIN_SCORE       = 0.83
DEDUP_THRESHOLD      = 0.92
TOP_K_READ           = 5
DECAY_HALF_LIFE_DAYS = 90
DECAY_JOB_INTERVAL_S = 3600

# ── NEW (solução 2): faixa de similaridade "provável correção" ────────────────
# Entre UPDATE_SIM_THRESHOLD e DEDUP_THRESHOLD, um texto novo não é nem uma
# duplicata clara (>= DEDUP_THRESHOLD, rejeitada) nem algo totalmente
# diferente (< UPDATE_SIM_THRESHOLD, vira memória nova). Nessa faixa, o
# write é recusado com reason="possible_update:<score>" e a memória mais
# próxima (candidate_id/candidate_text/candidate_score) volta na resposta
# — cabe a quem chamou (o extrator LLM) decidir se reenvia o /write com
# action="update" e memory_id=candidate_id, ou se era mesmo um fato novo.
UPDATE_SIM_THRESHOLD = 0.75

ST_TTL_HOURS          = 24.0
ST_CLEANUP_INTERVAL_S = 1800

# ── Otimização de tokens (reduzir payload enviado ao LLM) ─────────────────────
# Evita erros 413 Payload Too Large no Groq quando o /read é chamado múltiplas
# vezes pelo pipeline CoT → LLM (5 passos × N entradas × ~1KB cada).
#
# Estratégia em 3 camadas:
#   1. Thresholds mais seletivos para o /read (combinação de 4 fontes)
#   2. Teto de caracteres por entrada individual (trunca preservando palavra)
#   3. Teto global de caracteres no resultado final (corta entradas de menor score)
READ_TOP_K_FINAL        = 3        # era TOP_K_READ (5) — menos entradas no /read
READ_TOTAL_MAX_CHARS    = 2400     # teto global de chars retornados pelo /read
READ_LT_MAX_CHARS       = 600      # teto por entrada de memória de longo prazo
READ_ST_MAX_CHARS       = 350      # teto menor para curto prazo (texto denso)
READ_VS_MAX_CHARS       = 500      # teto para chunks de conhecimento (KG-RAG)
READ_IF_MAX_CHARS       = 500      # teto para chunks de arquivos indexados
READ_MIN_SCORE_STRICT   = 0.85     # mais seletivo que READ_MIN_SCORE (0.83)
IF_MIN_SCORE_READ       = 0.82     # threshold p/ /read (era min(0.83,0.75)=0.75)
                                   # corrige bug que retornava chunks demais de IF

# ── Parâmetros de busca contextual ─────────────────────────────────────────────
QUERY_SHORT_WORDS     = 6
QUERY_AMBIGUOUS_RATIO = 0.55
CONTEXT_MAX_CHARS     = 1200
CONTEXT_TURNS_FETCH   = 6
DUAL_CONTEXT_WEIGHT   = 0.35

# ── NEW: Decomposição de query composta (busca por segmento, sem LLM) ─────────
# Quando a query tem vários "sub-pedidos" numa frase só (ex.: "quero fazer X
# usando Y para conseguir Z"), embedar a frase inteira dilui a atenção
# semântica e nenhum sub-tópico fica bem representado no vetor final. Em vez
# de pedir a um LLM para gerar sub-queries, a query é quebrada por regras
# heurísticas (pontuação, vírgula, gerúndio, "para <verbo>") e cada pedaço é
# embedado/buscado separadamente — puro cosine similarity, sem custo de LLM.
SEGMENT_TRIGGER_WORDS = 12   # frases com até isso não valem a pena segmentar
SEGMENT_MIN_WORDS     = 3    # fragmento menor que isso é remendado no vizinho
SEGMENT_MAX_COUNT     = 4    # teto de segmentos por query (custo de embed/batch)

# ── /read_st — leitura crua do short-term (sem busca semântica) ───────────────
# Usada pelo LLM para montar o histórico recente da conversa como contexto,
# sem passar pelo pipeline de embeddings/scoring do /read.
ST_READ_DEFAULT_PAIRS = 5

_STOP_WORDS: frozenset[str] = frozenset({
    "o","a","os","as","um","uma","uns","umas","de","do","da","dos","das",
    "em","no","na","nos","nas","por","para","com","que","se","é","são",
    "foi","isso","esse","essa","eu","tu","ele","ela","nós","vocês","eles",
    "elas","me","te","nos","como","mas","mais","já","não","sim","tem",
    "ter","ser","fazer","ir","vou","vai","aqui","lá","também","então",
    "quando","onde","porque","qual","quais",
    "the","a","an","is","are","was","were","be","been","being","have",
    "has","had","do","does","did","will","would","could","should","may",
    "might","shall","can","to","of","in","on","at","by","for","with",
    "about","this","that","it","he","she","we","they","i","you","and",
    "or","but","so","if","my","your","his","her","our","their","what",
    "how","when","where","why","which","who","then","there","here","also",
})

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [Memory] %(message)s")
log = logging.getLogger("ava.memory")

if not _VS_AVAILABLE:
    log.warning("vector_store module not available — knowledge search disabled")
else:
    log.info("vector_store module found — knowledge search enabled")


# ── Modelos de request/response — memória ─────────────────────────────────────

class Turn(BaseModel):
    role:    Literal["user", "assistant"]
    content: str

class WriteRequest(BaseModel):
    text:        str
    source:      str   = "chat"
    confidence:  float = 1.0
    # "esquecível?" — quando False, a memória fica isenta do decay por
    # inatividade (apply_decay a ignora), mas ainda pode ser removida por
    # deleção explícita. Default True preserva o comportamento anterior.
    forgettable: bool  = True
    # ── NEW (solução 3): meia-vida específica desta memória, em dias.
    # Quando None, apply_decay cai para o default global
    # (DECAY_HALF_LIFE_DAYS). Ex.: ttl_days=7 para "viagem essa semana",
    # deixado em branco para memórias sem prazo específico.
    ttl_days:    Optional[float] = None
    # ── NEW (solução 2): "criar" (default) ou "atualizar/corrigir" uma
    # memória existente. Em action="update":
    #   - se memory_id vier preenchido, atualiza diretamente essa memória;
    #   - se memory_id vier vazio, procura a memória de longo prazo mais
    #     semelhante (score >= UPDATE_SIM_THRESHOLD) e atualiza ela; se
    #     nenhuma candidata suficientemente parecida existir, o write falha
    #     com reason="update_target_not_found" em vez de criar uma memória
    #     nova (evita que uma correção vire um fato solto).
    action:      Literal["create", "update"] = "create"
    memory_id:   Optional[int] = None

class WriteBatchRequest(BaseModel):
    # ── NEW (solução 1): grava várias memórias em uma única chamada,
    # evitando N round-trips HTTP/MCP quando o extrator LLM devolve um
    # array de fatos para uma mesma dupla pergunta-resposta.
    items: list[WriteRequest]

class WriteSTRequest(BaseModel):
    session_id: str
    turns:      list[Turn]

class ReadRequest(BaseModel):
    query:      str
    top_k:      int   = TOP_K_READ
    min_score:  float = READ_MIN_SCORE
    session_id: Optional[str] = None
    strategy:   str = "auto"

class WriteResponse(BaseModel):
    stored:    bool
    reason:    str
    memory_id: Optional[int] = None
    # ── NEW (solução 2): só populados quando reason começa com
    # "possible_update:" — a memória de longo prazo mais parecida
    # encontrada na faixa [UPDATE_SIM_THRESHOLD, DEDUP_THRESHOLD), para o
    # chamador decidir se reenvia como action="update".
    candidate_id:    Optional[int]   = None
    candidate_text:  Optional[str]   = None
    candidate_score: Optional[float] = None

class WriteBatchResponse(BaseModel):
    # ── NEW (solução 1): um WriteResponse por item de entrada, na mesma
    # ordem de WriteBatchRequest.items.
    results: list[WriteResponse]
    stored_count: int
    total: int

class WriteSTResponse(BaseModel):
    stored:   bool
    reason:   str
    turn_ids: list[int] = []

class ReadSTRequest(BaseModel):
    session_id: str
    n_pairs:    int = ST_READ_DEFAULT_PAIRS

class ReadSTResponse(BaseModel):
    session_id:     str
    turns:          list[Turn]
    pairs_returned: int

class MemoryEntry(BaseModel):
    id:           int
    text:         str
    score:        float
    confidence:   float
    created_at:   float
    access_count: int
    memory_type:  Literal["long_term", "short_term", "knowledge", "indexed_file"]
    session_id:   Optional[str]        = None
    turns:        Optional[list[Turn]] = None
    source:       Optional[str]        = None
    # ── NEW: esquecível — só populado para memory_type == "long_term" ──
    forgettable:  Optional[bool]       = None
    # ── NEW (solução 3): meia-vida por memória, em dias — só populado
    # para memory_type == "long_term" quando definida (senão usa o
    # default global DECAY_HALF_LIFE_DAYS).
    ttl_days:     Optional[float]      = None
    # ── NEW: Indexed file metadata ──
    file_path:    Optional[str]        = None
    file_name:    Optional[str]        = None
    extension:    Optional[str]        = None
    content_hash: Optional[str]        = None
    file_hash:    Optional[str]        = None

class ReadResponse(BaseModel):
    results:  list[MemoryEntry]
    query:    str
    strategy: str


# ── NEW: Modelos de request/response — arquivos indexados ─────────────────────

class IndexedFileWriteRequest(BaseModel):
    """Store a complete indexed file with full content, hash, and auto-chunking."""
    file_path:    str
    file_name:    str
    extension:    str   = ""
    content:      str
    file_hash:    str   = ""     # SHA-256 do arquivo original no disco
    size:         int   = 0
    modified:     str   = ""     # data de modificação ISO
    source:       str   = "local_scraping"
    confidence:   float = 1.0
    force_reindex: bool = False

class IndexedFileWriteResponse(BaseModel):
    stored:         bool
    reason:         str
    file_id:        Optional[int] = None
    chunks_created: int  = 0
    was_reindexed:  bool = False
    hash_match:     bool = True

class IndexedFileReadRequest(BaseModel):
    # ── NEW: leitura exata por caminho absoluto ──
    # Quando `file_path` é enviado, a busca semântica (via `query`) é
    # ignorada — o lookup é feito diretamente por igualdade de caminho
    # (mesma chave usada em `/indexed-file/write`), garantindo que dois
    # arquivos com o mesmo nome em pastas diferentes nunca se confundam.
    # `query` continua obrigatório apenas quando `file_path` não é enviado.
    file_path: Optional[str] = None
    query:     Optional[str] = None
    top_k:     int   = 5
    min_score: float = IF_MIN_SCORE
    # ── NEW: quando False, as entradas retornadas vêm com `content=""` —
    # útil no modo "file_path + query" para arquivos grandes, onde o
    # chamador só quer as chunks e não o payload inteiro do arquivo.
    # Não afeta o modo "file_path" sozinho (arquivo inteiro), que sempre
    # devolve `content` preenchido — é o próprio propósito desse modo.
    include_full_content: bool = True

class IndexedFileEntry(BaseModel):
    file_id:      int
    file_path:    str
    file_name:    str
    extension:    str
    content:      str                   # conteúdo completo do arquivo
    file_hash:    str
    content_hash: str
    size:         int
    modified:     str
    score:        float
    confidence:   float
    created_at:   float
    access_count: int
    source:       str = "local_scraping"
    chunk_text:   Optional[str] = None  # chunk específico que deu match
    # ── NEW: posição da chunk dentro do arquivo — ajuda o chamador a se
    # orientar sem precisar do arquivo inteiro (útil pra arquivos grandes) ──
    chunk_index:  Optional[int] = None
    char_start:   Optional[int] = None
    char_end:     Optional[int] = None
    chunk_id: Optional[int] = None 
    # ── NEW: "exact_path" (lookup direto por file_path) ou "semantic" (busca vetorial) ──
    match_type:   str = "semantic"

class IndexedFileReadResponse(BaseModel):
    results:   list[IndexedFileEntry]
    query:     Optional[str] = None
    file_path: Optional[str] = None

class IndexedFileCheckResponse(BaseModel):
    indexed:          bool
    hash_match:       Optional[bool]   = None
    file_id:          Optional[int]    = None
    stored_file_hash: Optional[str]    = None
    stored_content_hash: Optional[str] = None
    stored_modified:  Optional[str]    = None
    chunks_count:     Optional[int]    = None


# ── NEW: Modelos de request/response — Dicionário Visual (vision.py) ──────────
#
# O módulo vision.py (pipeline de "Tradução de Objetos") faz a segmentação,
# extração de crops e geração de embeddings (DINOv3) de cada objeto detectado
# numa imagem. Ele NÃO guarda nenhum estado — apenas manda o embedding (+ meta)
# pra cá via HTTP, e este arquivo é o único responsável por persistir o
# "dicionário" (embeddings no FAISS + texto/significado no SQLite).
#
# A descrição textual de cada conceito também é replicada na tabela
# `memories` (longo prazo), então ela aparece normalmente em qualquer
# chamada de /read — não é preciso um endpoint de leitura separado pra isso.

class VisualDictWriteRequest(BaseModel):
    concept_name:   str                 # nome curto do objeto/conceito, ex.: "caneca azul"
    description:    str                 # "significado" textual — o que é, contexto, uso etc.
    embedding:      list[float]         # embedding do crop (DINOv3), normalizado ou não
    source:         str   = "vision_pipeline"
    confidence:     float = 1.0
    # quando True (padrão), a descrição também é gravada como memória de
    # longo prazo normal (pesquisável via /read). Só é feito na primeira vez
    # que o conceito é criado — exemplos adicionais do mesmo conceito não
    # duplicam a entrada de texto, só adicionam mais um vetor de embedding.
    link_to_memory: bool  = True

class VisualDictWriteResponse(BaseModel):
    stored:       bool
    reason:       str
    concept_id:   Optional[int] = None
    embedding_id: Optional[int] = None
    memory_id:    Optional[int] = None   # id em `memories`, se link_to_memory=True
    new_concept:  bool = False           # True se um conceito novo foi criado agora

class VisualDictReadRequest(BaseModel):
    embedding: list[float]        # embedding do crop consultado (DINOv3)
    top_k:     int   = VD_TOP_K
    min_score: float = VD_MIN_SCORE

class VisualDictCandidate(BaseModel):
    concept_id:   int
    concept_name: str
    description:  str
    score:        float
    confidence:   float
    access_count: int
    memory_id:    Optional[int] = None

class VisualDictReadResponse(BaseModel):
    results:   list[VisualDictCandidate]
    ambiguous: bool     # True → nenhum candidato confiável / candidatos muito próximos;
                         # quem chama (vision.py / orquestrador) deve perguntar ao usuário

class VisualDictEntry(BaseModel):
    concept_id:    int
    concept_name:  str
    description:   str
    source:        str
    confidence:    float
    memory_id:     Optional[int] = None
    examples_count: int
    created_at:    float
    access_count:  int


# ── NEW: Modelos de request/response — dicionário de rostos ───────────────────

class FaceDictWriteRequest(BaseModel):
    person_name: str                 # nome da pessoa, ex.: "eu" / "Fulano"
    embedding:   list[float]         # embedding do rosto alinhado (EdgeFace)
    description: str   = ""          # quem é essa pessoa (relação, contexto etc.)
    source:      str   = "vision_pipeline"
    confidence:  float = 1.0

class FaceDictWriteResponse(BaseModel):
    stored:       bool
    reason:       str
    person_id:    Optional[int] = None
    embedding_id: Optional[int] = None
    new_person:   bool = False        # True se a pessoa foi criada agora

class FaceDictReadRequest(BaseModel):
    embedding: list[float]        # embedding do rosto consultado (EdgeFace)
    top_k:     int   = FD_TOP_K
    min_score: float = FD_MIN_SCORE

class FaceCandidate(BaseModel):
    person_id:    int
    person_name:  str
    description:  str
    score:        float
    confidence:   float
    access_count: int

class FaceDictReadResponse(BaseModel):
    results:   list[FaceCandidate]
    ambiguous: bool     # True → nenhum candidato confiável / candidatos muito próximos

class FaceDictEntry(BaseModel):
    person_id:      int
    person_name:    str
    description:    str
    source:         str
    confidence:     float
    examples_count: int
    created_at:     float
    access_count:   int


# ── MODIFIED: EmbeddingEngine now delegates to EmbeddingClient ────────────────

class EmbeddingEngine:
    """Motor de embeddings via ONNX Serving API."""

    def __init__(self, base_url: str = ONNX_SERVING_URL):
        self._client = EmbeddingClient(base_url=base_url)
        log.info(f"EmbeddingEngine carregado — via API: {base_url}")

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        return await self._client.embed(texts)

    async def embed_one(self, text: str) -> np.ndarray:
        result = await self._client.embed([text])
        return result[0]

    async def embed_batch_two(self, text_a: str, text_b: str) -> tuple[np.ndarray, np.ndarray]:
        results = await self._client.embed([text_a, text_b])
        return results[0], results[1]

    @property
    def client(self) -> EmbeddingClient:
        return self._client


# ── Banco de dados de longo prazo ──────────────────────────────────────────────

class MemoryDB:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # ITEM 4 da revisão: esta conexão é usada tanto pelo event loop quanto
        # por threads do executor (endpoints e jobs em background). O módulo
        # sqlite3 não serializa automaticamente chamadas concorrentes vindas
        # de threads diferentes sobre a MESMA conexão — este lock garante que
        # cada statement de escrita rode de forma atômica em relação aos
        # outros.
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                text          TEXT    NOT NULL,
                text_hash     TEXT    NOT NULL UNIQUE,
                source        TEXT    NOT NULL DEFAULT 'chat',
                confidence    REAL    NOT NULL DEFAULT 1.0,
                created_at    REAL    NOT NULL,
                last_accessed REAL    NOT NULL,
                access_count  INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_confidence    ON memories(confidence);
            CREATE INDEX IF NOT EXISTS idx_last_accessed ON memories(last_accessed);
        """)
        # Migração leve: bancos criados antes do campo "esquecível?" existir
        # não têm a coluna — adiciona com default 1 (esquecível, comportamento
        # anterior) sem quebrar instalações já em uso.
        cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(memories)")}
        if "forgettable" not in cols:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN forgettable INTEGER NOT NULL DEFAULT 1"
            )
        # ── NEW (solução 3): mesma migração leve de sempre — bancos criados
        # antes de ttl_days existir não têm a coluna; adiciona como NULL
        # (== "usar o default global DECAY_HALF_LIFE_DAYS"), sem quebrar
        # instalações já em uso.
        if "ttl_days" not in cols:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN ttl_days REAL DEFAULT NULL"
            )

    def insert(
        self,
        text: str,
        source: str,
        confidence: float,
        forgettable: bool = True,
        ttl_days: Optional[float] = None,
    ) -> int:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO memories (text, text_hash, source, confidence, forgettable, "
                "ttl_days, created_at, last_accessed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (text, text_hash, source, confidence, int(forgettable), ttl_days, now, now),
            )
            return cur.lastrowid

    def exists_exact(self, text: str) -> bool:
        h = hashlib.sha256(text.encode()).hexdigest()
        return self._conn.execute(
            "SELECT 1 FROM memories WHERE text_hash = ?", (h,)
        ).fetchone() is not None

    def get_by_id(self, memory_id: int) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()

    def get_by_ids(self, ids: list[int]) -> list[sqlite3.Row]:
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        return self._conn.execute(
            f"SELECT * FROM memories WHERE id IN ({ph})", ids
        ).fetchall()

    def update_access(self, memory_id: int):
        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                    (time.time(), memory_id),
                )
        except sqlite3.OperationalError:
            pass

    def update(
        self,
        memory_id: int,
        text: str,
        source: Optional[str]      = None,
        confidence: Optional[float] = None,
        forgettable: Optional[bool] = None,
        ttl_days: Optional[float]   = None,
        ttl_days_set: bool          = False,
    ) -> bool:
        """
        Atualiza o texto (e opcionalmente source/confidence/forgettable/
        ttl_days) de uma memória já existente, em vez de inserir uma nova
        — usado pelo fluxo action="update" (solução 2 da revisão), quando
        o extrator identifica que um texto novo é uma correção de um fato
        já gravado, e não um fato adicional.

        `text_hash` é recalculado a partir do novo texto para que a
        dedup exata (`exists_exact`) continue funcionando corretamente
        depois da correção. `created_at` não é alterado — só o conteúdo,
        a confiança e o "relógio" de acesso.

        `ttl_days_set` distingue "não mexer no ttl_days atual" (default)
        de "setar ttl_days para None explicitamente" (chamador passou
        ttl_days=None de propósito) — sem isso não daria pra diferenciar
        os dois casos só olhando `ttl_days is None`.
        """
        row = self.get_by_id(memory_id)
        if row is None:
            return False

        new_text        = text.strip()
        new_text_hash   = hashlib.sha256(new_text.encode()).hexdigest()
        new_source      = source if source is not None else row["source"]
        new_confidence  = confidence if confidence is not None else row["confidence"]
        new_forgettable = int(forgettable) if forgettable is not None else row["forgettable"]
        new_ttl_days    = ttl_days if ttl_days_set else row["ttl_days"]
        now = time.time()

        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE memories SET text = ?, text_hash = ?, source = ?, "
                    "confidence = ?, forgettable = ?, ttl_days = ?, last_accessed = ? "
                    "WHERE id = ?",
                    (
                        new_text, new_text_hash, new_source, new_confidence,
                        new_forgettable, new_ttl_days, now, memory_id,
                    ),
                )
            except sqlite3.IntegrityError:
                # O texto novo já é idêntico (mesmo text_hash) a OUTRA
                # memória existente — não faz sentido duplicar o hash.
                return False
        return True

    def apply_decay(self, half_life_days: float):
        now  = time.time()
        with self._lock:
            # forgettable = 0 → memória marcada como "não esquecível":
            # fica de fora do decay por completo, mesmo que fique muito
            # tempo sem ser acessada.
            rows = self._conn.execute(
                "SELECT id, confidence, last_accessed, ttl_days FROM memories "
                "WHERE confidence > 0.01 AND forgettable = 1"
            ).fetchall()
            updates = []
            for row in rows:
                # ── NEW (solução 3): ttl_days por linha tem prioridade
                # sobre o half_life_days global quando presente (> 0);
                # cai para o default global quando ausente/NULL.
                row_ttl = row["ttl_days"]
                effective_half_life = row_ttl if (row_ttl is not None and row_ttl > 0) else half_life_days
                days_idle    = (now - row["last_accessed"]) / 86400.0
                decay_factor = 0.5 ** (days_idle / effective_half_life)
                updates.append((row["confidence"] * decay_factor, row["id"]))
            if updates:
                self._conn.executemany("UPDATE memories SET confidence = ? WHERE id = ?", updates)
                self._conn.execute("DELETE FROM memories WHERE confidence < 0.01")
                log.info(f"Decay aplicado em {len(updates)} memórias de longo prazo")

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]


# ── Banco de dados de curto prazo ──────────────────────────────────────────────

class ShortTermDB:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()  # ITEM 4 — ver comentário em MemoryDB
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS turn_groups (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT    NOT NULL,
                turns_json    TEXT    NOT NULL,
                embed_text    TEXT    NOT NULL,
                created_at    REAL    NOT NULL,
                last_accessed REAL    NOT NULL,
                access_count  INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_st_session  ON turn_groups(session_id);
            CREATE INDEX IF NOT EXISTS idx_st_accessed ON turn_groups(last_accessed);
        """)

    def insert(self, session_id: str, turns: list[Turn], embed_text: str) -> int:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO turn_groups (session_id, turns_json, embed_text, created_at, last_accessed) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, json.dumps([t.model_dump() for t in turns]), embed_text, now, now),
            )
            return cur.lastrowid

    def get_by_ids(self, ids: list[int]) -> list[sqlite3.Row]:
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        return self._conn.execute(
            f"SELECT * FROM turn_groups WHERE id IN ({ph})", ids
        ).fetchall()

    def get_recent_turns_text(self, session_id: str, n_groups: int) -> list[str]:
        rows = self._conn.execute(
            "SELECT turns_json FROM turn_groups "
            "WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, n_groups),
        ).fetchall()
        lines: list[str] = []
        for row in reversed(rows):
            for t in json.loads(row["turns_json"]):
                lines.append(f"{t['role']}: {t['content']}")
        return lines

    def get_recent_turn_groups(self, session_id: str, n_groups: int) -> tuple[list[dict], int]:
        """
        Leitura crua (sem busca semântica) das N duplas pergunta-resposta mais
        recentes de uma sessão, em ordem cronológica (mais antiga primeiro).
        Usada pelo /read_st — contexto de conversa direto, sem embeddings.
        """
        rows = self._conn.execute(
            "SELECT turns_json FROM turn_groups "
            "WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, n_groups),
        ).fetchall()
        turns: list[dict] = []
        for row in reversed(rows):
            turns.extend(json.loads(row["turns_json"]))
        return turns, len(rows)

    def update_access(self, group_id: int):
        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE turn_groups SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                    (time.time(), group_id),
                )
        except sqlite3.OperationalError:
            pass

    def expire_old(self, ttl_hours: float) -> int:
        cutoff = time.time() - ttl_hours * 3600
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM turn_groups WHERE last_accessed < ?", (cutoff,)
            )
            removed = cur.rowcount
        if removed:
            log.info(f"Curto prazo: {removed} grupos expirados (TTL={ttl_hours}h)")
        return removed

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM turn_groups").fetchone()[0]

    def get_ids_by_session(self, session_id: str) -> list[int]:
        rows = self._conn.execute(
            "SELECT id FROM turn_groups WHERE session_id = ?", (session_id,)
        ).fetchall()
        return [row["id"] for row in rows]

    def delete_by_session(self, session_id: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM turn_groups WHERE session_id = ?", (session_id,)
            )
            return cur.rowcount


# ── NEW: Banco de dados de arquivos indexados ──────────────────────────────────

class IndexedFilesDB:
    """
    Armazena o conteúdo COMPLETO dos arquivos indexados pelo local-scraping,
    com hash para detectar mudanças e chunks para busca semântica via FAISS.
    """

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()  # ITEM 4 — ver comentário em MemoryDB
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS indexed_files (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path     TEXT    NOT NULL UNIQUE,
                file_name     TEXT    NOT NULL,
                extension     TEXT    NOT NULL DEFAULT '',
                content       TEXT    NOT NULL,
                content_hash  TEXT    NOT NULL,
                file_hash     TEXT    NOT NULL DEFAULT '',
                size          INTEGER NOT NULL DEFAULT 0,
                modified      TEXT    NOT NULL DEFAULT '',
                source        TEXT    NOT NULL DEFAULT 'local_scraping',
                confidence    REAL    NOT NULL DEFAULT 1.0,
                created_at    REAL    NOT NULL,
                last_accessed REAL    NOT NULL,
                access_count  INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_if_path      ON indexed_files(file_path);
            CREATE INDEX IF NOT EXISTS idx_if_file_hash ON indexed_files(file_hash);

            CREATE TABLE IF NOT EXISTS indexed_file_chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id     INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text  TEXT    NOT NULL,
                char_start  INTEGER NOT NULL DEFAULT 0,
                char_end    INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (file_id) REFERENCES indexed_files(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_ifc_file_id ON indexed_file_chunks(file_id);
        """)

    # ── File operations ──

    def insert_file(
        self,
        file_path:   str,
        file_name:   str,
        extension:   str,
        content:     str,
        content_hash: str,
        file_hash:   str,
        size:        int,
        modified:    str,
        source:      str,
        confidence:  float,
    ) -> int:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO indexed_files "
                "(file_path, file_name, extension, content, content_hash, file_hash, "
                "size, modified, source, confidence, created_at, last_accessed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (file_path, file_name, extension, content, content_hash, file_hash,
                 size, modified, source, confidence, now, now),
            )
            return cur.lastrowid

    def update_file(
        self,
        file_id:     int,
        content:     str,
        content_hash: str,
        file_hash:   str,
        size:        int,
        modified:    str,
    ):
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE indexed_files SET content=?, content_hash=?, file_hash=?, "
                "size=?, modified=?, last_accessed=? WHERE id=?",
                (content, content_hash, file_hash, size, modified, now, file_id),
            )

    def get_by_path(self, file_path: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM indexed_files WHERE file_path = ?", (file_path,)
        ).fetchone()

    def get_by_id(self, file_id: int) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM indexed_files WHERE id = ?", (file_id,)
        ).fetchone()

    def update_access(self, file_id: int):
        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE indexed_files SET access_count = access_count + 1, "
                    "last_accessed = ? WHERE id = ?",
                    (time.time(), file_id),
                )
        except sqlite3.OperationalError:
            pass

    def delete_file(self, file_id: int) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM indexed_files WHERE id = ?", (file_id,))
            return cur.rowcount

    def count_files(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()[0]

    def list_files(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT id, file_path, file_name, extension, content_hash, file_hash, "
            "size, modified, created_at, access_count FROM indexed_files "
            "ORDER BY last_accessed DESC"
        ).fetchall()

    # ── Chunk operations ──

    def insert_chunks(self, file_id: int, chunks: list[dict]):
        """Insert multiple chunks for a file. chunks = [{text, index, char_start, char_end}]"""
        rows = [
            (file_id, c["index"], c["text"], c["char_start"], c["char_end"])
            for c in chunks
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT INTO indexed_file_chunks (file_id, chunk_index, chunk_text, char_start, char_end) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def get_chunks_by_file(self, file_id: int) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM indexed_file_chunks WHERE file_id = ? ORDER BY chunk_index",
            (file_id,),
        ).fetchall()

    def get_chunk_ids_by_file(self, file_id: int) -> list[int]:
        rows = self._conn.execute(
            "SELECT id FROM indexed_file_chunks WHERE file_id = ?", (file_id,)
        ).fetchall()
        return [row["id"] for row in rows]

    def count_chunks(self, file_id: Optional[int] = None) -> int:
        if file_id:
            return self._conn.execute(
                "SELECT COUNT(*) FROM indexed_file_chunks WHERE file_id = ?", (file_id,)
            ).fetchone()[0]
        return self._conn.execute(
            "SELECT COUNT(*) FROM indexed_file_chunks"
        ).fetchone()[0]

    def delete_chunks_by_file(self, file_id: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM indexed_file_chunks WHERE file_id = ?", (file_id,)
            )
            return cur.rowcount

    def get_chunks_by_ids(self, chunk_ids: list[int]) -> list[sqlite3.Row]:
        if not chunk_ids:
            return []
        ph = ",".join("?" * len(chunk_ids))
        return self._conn.execute(
            f"SELECT * FROM indexed_file_chunks WHERE id IN ({ph})", chunk_ids
        ).fetchall()

    def get_total_chunks(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM indexed_file_chunks").fetchone()[0]


# ── NEW: Banco de dados do Dicionário Visual ───────────────────────────────────

class VisualDictDB:
    """
    Persiste os "conceitos visuais" do módulo vision.py: um conceito é um
    objeto/tipo de objeto (ex.: "caneca azul", "controle remoto da TV"),
    com uma descrição textual e N embeddings de exemplo (um por crop
    registrado — permite reconhecer o mesmo conceito em ângulos/condições
    de luz diferentes).

    Os embeddings em si moram no FAISS (`vd_index`); aqui só ficam o texto
    e o mapeamento embedding_id → concept_id.
    """

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()  # ITEM 4 — ver comentário em MemoryDB
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS visual_concepts (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_name  TEXT    NOT NULL,
                concept_key   TEXT    NOT NULL UNIQUE,   -- nome normalizado (lower/strip)
                description   TEXT    NOT NULL,
                source        TEXT    NOT NULL DEFAULT 'vision_pipeline',
                confidence    REAL    NOT NULL DEFAULT 1.0,
                memory_id     INTEGER,                    -- FK lógica p/ memories.id
                created_at    REAL    NOT NULL,
                last_accessed REAL    NOT NULL,
                access_count  INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_vc_key ON visual_concepts(concept_key);

            CREATE TABLE IF NOT EXISTS visual_concept_embeddings (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_id INTEGER NOT NULL,
                created_at REAL    NOT NULL,
                FOREIGN KEY (concept_id) REFERENCES visual_concepts(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_vce_concept ON visual_concept_embeddings(concept_id);
        """)

    @staticmethod
    def _normalize_key(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip().lower())

    # ── Concept operations ──

    def insert_concept(
        self, concept_name: str, description: str, source: str,
        confidence: float, memory_id: Optional[int],
    ) -> int:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO visual_concepts "
                "(concept_name, concept_key, description, source, confidence, memory_id, "
                "created_at, last_accessed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (concept_name, self._normalize_key(concept_name), description, source,
                 confidence, memory_id, now, now),
            )
            return cur.lastrowid

    def get_by_name(self, concept_name: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM visual_concepts WHERE concept_key = ?",
            (self._normalize_key(concept_name),),
        ).fetchone()

    def get_concept_by_id(self, concept_id: int) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM visual_concepts WHERE id = ?", (concept_id,)
        ).fetchone()

    def update_access(self, concept_id: int):
        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE visual_concepts SET access_count = access_count + 1, "
                    "last_accessed = ? WHERE id = ?",
                    (time.time(), concept_id),
                )
        except sqlite3.OperationalError:
            pass

    def delete_concept(self, concept_id: int) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM visual_concepts WHERE id = ?", (concept_id,))
            return cur.rowcount

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM visual_concepts").fetchone()[0]

    def list_concepts(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT c.*, "
            "(SELECT COUNT(*) FROM visual_concept_embeddings e WHERE e.concept_id = c.id) "
            "AS examples_count "
            "FROM visual_concepts c ORDER BY c.last_accessed DESC"
        ).fetchall()

    # ── Embedding-row operations (mapeamento embedding_id → concept_id) ──

    def insert_embedding(self, concept_id: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO visual_concept_embeddings (concept_id, created_at) VALUES (?, ?)",
                (concept_id, time.time()),
            )
            return cur.lastrowid

    def get_concept_id_by_embedding(self, embedding_id: int) -> Optional[int]:
        row = self._conn.execute(
            "SELECT concept_id FROM visual_concept_embeddings WHERE id = ?",
            (embedding_id,),
        ).fetchone()
        return row["concept_id"] if row else None

    def get_embedding_ids_by_concept(self, concept_id: int) -> list[int]:
        rows = self._conn.execute(
            "SELECT id FROM visual_concept_embeddings WHERE concept_id = ?", (concept_id,)
        ).fetchall()
        return [r["id"] for r in rows]

    def count_examples(self, concept_id: int) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM visual_concept_embeddings WHERE concept_id = ?",
            (concept_id,),
        ).fetchone()[0]


# ── NEW: Dicionário de Rostos ──────────────────────────────────────────────────

class FaceDictDB:
    """
    Persiste as pessoas cadastradas pro reconhecimento facial: uma pessoa
    tem N embeddings de exemplo (um por rosto registrado — permite
    reconhecer o mesmo rosto em ângulos/luz diferentes).

    Os embeddings em si moram no FAISS (`fd_index`); aqui só ficam o nome
    e o mapeamento embedding_id → person_id. Estrutura idêntica à
    VisualDictDB, só que sem `description`/`memory_id` — reconhecimento
    facial não precisa de "significado" textual gravado na memória geral.
    """

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()  # ITEM 4 — ver comentário em MemoryDB
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS face_people (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name   TEXT    NOT NULL,
                person_key    TEXT    NOT NULL UNIQUE,   -- nome normalizado (lower/strip)
                description   TEXT    NOT NULL DEFAULT '',  -- quem é a pessoa (relação/contexto)
                source        TEXT    NOT NULL DEFAULT 'vision_pipeline',
                confidence    REAL    NOT NULL DEFAULT 1.0,
                created_at    REAL    NOT NULL,
                last_accessed REAL    NOT NULL,
                access_count  INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_fp_key ON face_people(person_key);

            CREATE TABLE IF NOT EXISTS face_embeddings (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id  INTEGER NOT NULL,
                created_at REAL    NOT NULL,
                FOREIGN KEY (person_id) REFERENCES face_people(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_fe_person ON face_embeddings(person_id);
        """)
        # Migração leve: bancos criados antes deste campo existir não têm a
        # coluna `description` — adiciona se estiver faltando, sem quebrar
        # instalações já em uso.
        cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(face_people)")}
        if "description" not in cols:
            self._conn.execute("ALTER TABLE face_people ADD COLUMN description TEXT NOT NULL DEFAULT ''")

    @staticmethod
    def _normalize_key(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip().lower())

    # ── Person operations ──

    def insert_person(self, person_name: str, description: str, source: str, confidence: float) -> int:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO face_people "
                "(person_name, person_key, description, source, confidence, created_at, last_accessed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (person_name, self._normalize_key(person_name), description, source, confidence, now, now),
            )
            return cur.lastrowid

    def get_by_name(self, person_name: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM face_people WHERE person_key = ?",
            (self._normalize_key(person_name),),
        ).fetchone()

    def get_person_by_id(self, person_id: int) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM face_people WHERE id = ?", (person_id,)
        ).fetchone()

    def update_description(self, person_id: int, description: str):
        """Atualiza/edita a descrição de uma pessoa já cadastrada. Chamado
        quando `person_name` já existe e um `description` não-vazio é
        enviado num novo /face-dict/write (ex.: cadastrando mais um
        exemplo de rosto e aproveitando pra corrigir/completar o texto)."""
        with self._lock:
            self._conn.execute(
                "UPDATE face_people SET description = ? WHERE id = ?",
                (description, person_id),
            )

    def update_access(self, person_id: int):
        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE face_people SET access_count = access_count + 1, "
                    "last_accessed = ? WHERE id = ?",
                    (time.time(), person_id),
                )
        except sqlite3.OperationalError:
            pass

    def delete_person(self, person_id: int) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM face_people WHERE id = ?", (person_id,))
            return cur.rowcount

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM face_people").fetchone()[0]

    def list_people(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT p.*, "
            "(SELECT COUNT(*) FROM face_embeddings e WHERE e.person_id = p.id) "
            "AS examples_count "
            "FROM face_people p ORDER BY p.last_accessed DESC"
        ).fetchall()

    # ── Embedding-row operations (mapeamento embedding_id → person_id) ──

    def insert_embedding(self, person_id: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO face_embeddings (person_id, created_at) VALUES (?, ?)",
                (person_id, time.time()),
            )
            return cur.lastrowid

    def get_person_id_by_embedding(self, embedding_id: int) -> Optional[int]:
        row = self._conn.execute(
            "SELECT person_id FROM face_embeddings WHERE id = ?",
            (embedding_id,),
        ).fetchone()
        return row["person_id"] if row else None

    def get_embedding_ids_by_person(self, person_id: int) -> list[int]:
        rows = self._conn.execute(
            "SELECT id FROM face_embeddings WHERE person_id = ?", (person_id,)
        ).fetchall()
        return [r["id"] for r in rows]

    def count_examples(self, person_id: int) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM face_embeddings WHERE person_id = ?",
            (person_id,),
        ).fetchone()[0]


# ── NEW: Chunking de texto ─────────────────────────────────────────────────────

def _chunk_text(
    text:         str,
    chunk_size:   int = CHUNK_SIZE,
    overlap:      int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Divide texto em chunks sobrepostos com metadados de posição.

    Estratégia de split (prioridade):
      1. Quebra de parágrafo (\\n\\n)
      2. Quebra de linha (\\n)
      3. Fim de sentença (. ! ?)
      4. Espaço (limite de palavra)
      5. Corte duro no chunk_size

    Retorna lista de dicts:
      {"text": str, "index": int, "char_start": int, "char_end": int}
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Tenta encontrar um ponto de quebra natural na segunda metade do chunk
            search_start = start + chunk_size // 2

            # 1. Parágrafo duplo
            split_pos = text.rfind('\n\n', search_start, end)
            if split_pos != -1:
                end = split_pos + 2  # inclui o \n\n
            else:
                # 2. Linha simples
                split_pos = text.rfind('\n', search_start, end)
                if split_pos != -1:
                    end = split_pos + 1
                else:
                    # 3. Fim de sentença
                    best_sep = -1
                    for sep in ('. ', '.\n', '! ', '? ', '。', '！', '？', '; ', ';\n'):
                        pos = text.rfind(sep, search_start, end)
                        if pos != -1 and pos > best_sep:
                            best_sep = pos + len(sep)
                    if best_sep != -1:
                        end = best_sep
                    else:
                        # 4. Espaço
                        split_pos = text.rfind(' ', search_start, end)
                        if split_pos != -1:
                            end = split_pos + 1
                        # 5. Corte duro — end já está setado

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text":       chunk_text,
                "index":      chunk_index,
                "char_start": start,
                "char_end":   end,
            })
            chunk_index += 1

        # Avança com sobreposição
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # garante progresso

        # Limite de chunks
        if chunk_index >= IF_MAX_CHUNKS:
            log.warning(f"Chunking atingiu limite de {IF_MAX_CHUNKS} — truncando")
            break

    return chunks


# ── Índice FAISS genérico ──────────────────────────────────────────────────────

# Intervalo do job de persistência periódica dos índices (ver index_flush_job).
# Trocamos "salvar a cada escrita" por "marcar sujo + descarregar em lote" —
# evita reescrever o índice inteiro em disco a cada add() (ver ITEM 1 da
# revisão).
INDEX_FLUSH_INTERVAL_S = 60


class MemoryIndex:
    """
    FAISS IndexIDMap2(IndexFlatIP) — inner product em vetores L2-normalizados
    = cosine similarity. Usar IndexIDMap2 (em vez de IndexFlatIP + lista
    paralela `_id_map` mantida à mão em Python) resolve dois problemas do
    design anterior:

      1. `remove_ids` deixa de precisar reconstruir o índice inteiro
         (reconstruct_n + rebuild em Python) — quem faz a remoção agora é o
         próprio FAISS, em C++, via IDSelectorBatch.
      2. O mapeamento id→vetor é interno ao índice e persiste junto no
         arquivo .index — elimina a possibilidade de `_id_map` e o índice
         ficarem fora de sincronia (ex.: se o processo morrer entre os dois
         `np.save`/`write_index`).

    Todas as mutações (add/add_batch/remove_ids/reset) e leituras (search/
    search_subset) são protegidas por um lock — os métodos podem ser chamados
    tanto do event loop quanto de threads do executor (ver ITEM 3/4 da
    revisão: FAISS não garante thread-safety para add/search concorrentes).
    """

    def __init__(
        self,
        index_path: str,
        id_map_path: str,
        persist: bool = True,
        embed_dim: int = EMBED_DIM,
    ):
        self._index_path  = index_path
        self._id_map_path = id_map_path  # mantido só para migrar índices antigos
        self._persist      = persist
        self._embed_dim    = embed_dim
        self._lock         = threading.Lock()
        self._dirty        = False
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        if persist and Path(index_path).exists():
            loaded = faiss.read_index(index_path)
            if isinstance(loaded, faiss.IndexIDMap2):
                self._index = loaded
                log.info(f"Índice FAISS carregado [{index_path}] — {self._index.ntotal} vetores")
            else:
                # Migração de índice no formato antigo (IndexFlatIP + .npy
                # externo) — lê os ids do id_map legado, se existir, e
                # reconstrói como IndexIDMap2.
                self._index = self._migrate_legacy_index(loaded, id_map_path)
        else:
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(embed_dim))
            log.info(f"Novo índice FAISS criado [{index_path}] (dim={embed_dim})")

    def _migrate_legacy_index(self, flat_index, id_map_path: str) -> "faiss.IndexIDMap2":
        n = flat_index.ntotal
        ids: list[int] = []
        if Path(id_map_path).exists():
            ids = list(np.load(id_map_path).tolist())
        if len(ids) != n:
            log.warning(
                f"Migração [{self._index_path}]: id_map legado tem {len(ids)} "
                f"entradas mas o índice tem {n} vetores — usando posição como id"
            )
            ids = list(range(n))
        new_index = faiss.IndexIDMap2(faiss.IndexFlatIP(self._embed_dim))
        if n > 0:
            vecs = flat_index.reconstruct_n(0, n)
            new_index.add_with_ids(vecs, np.array(ids, dtype=np.int64))
        log.info(f"Índice migrado para IndexIDMap2 [{self._index_path}] — {n} vetores")
        self._dirty = True
        return new_index

    def add(self, embedding: np.ndarray, record_id: int):
        with self._lock:
            self._index.add_with_ids(
                embedding.reshape(1, -1).astype(np.float32),
                np.array([record_id], dtype=np.int64),
            )
            self._dirty = True

    def add_batch(self, embeddings: np.ndarray, record_ids: list[int]):
        """Adiciona múltiplos vetores de uma vez — mais eficiente que add() individual."""
        if embeddings.shape[0] != len(record_ids):
            raise ValueError(
                f"embeddings ({embeddings.shape[0]}) e record_ids ({len(record_ids)}) "
                f"devem ter o mesmo tamanho"
            )
        if embeddings.shape[0] == 0:
            return
        with self._lock:
            self._index.add_with_ids(
                embeddings.astype(np.float32),
                np.array(record_ids, dtype=np.int64),
            )
            self._dirty = True

    def remove_ids(self, record_ids: set[int]):
        if not record_ids:
            return
        with self._lock:
            if self._index.ntotal == 0:
                return
            # Construtor "cru" (n + ponteiro) em vez do atalho
            # `IDSelectorBatch(array)` — compatível com um leque maior de
            # versões do faiss-cpu/faiss-gpu.
            ids_arr  = np.ascontiguousarray(list(record_ids), dtype=np.int64)
            selector = faiss.IDSelectorBatch(ids_arr.size, faiss.swig_ptr(ids_arr))
            n_removed = self._index.remove_ids(selector)
            self._dirty = True
        log.info(f"FAISS: {n_removed} vetores removidos [{self._index_path}]")

    def reset(self):
        with self._lock:
            self._index.reset()
            self._dirty = True

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        with self._lock:
            ntotal = self._index.ntotal
            if ntotal == 0:
                return []
            k = min(top_k, ntotal)
            scores, indices = self._index.search(
                query_embedding.reshape(1, -1).astype(np.float32), k
            )
        # Com IndexIDMap2 os `indices` retornados já SÃO os record_id — não
        # há mais tradução via `_id_map`. -1 indica slot vazio (padding).
        return [
            (int(idx), float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

    # ── NEW: busca em lote — N queries de uma vez em UMA chamada ao FAISS ──
    def search_batch(
        self, query_embeddings: np.ndarray, top_k: int
    ) -> list[list[tuple[int, float]]]:
        """
        Mesma lógica de search(), mas para várias queries simultâneas
        (ex.: os N segmentos de uma pergunta composta). O FAISS já faz
        busca em lote nativamente — uma única chamada, sem loop em Python —
        então isso é praticamente tão rápido quanto uma busca única.
        """
        n = query_embeddings.shape[0] if query_embeddings.ndim == 2 else 0
        with self._lock:
            ntotal = self._index.ntotal
            if ntotal == 0 or n == 0:
                return [[] for _ in range(n)]
            k = min(top_k, ntotal)
            scores, indices = self._index.search(
                query_embeddings.astype(np.float32), k
            )
        return [
            [
                (int(idx), float(score))
                for score, idx in zip(row_scores, row_indices)
                if idx != -1
            ]
            for row_scores, row_indices in zip(scores, indices)
        ]

    def search_similar(self, embedding: np.ndarray) -> float:
        results = self.search(embedding, top_k=1)
        return results[0][1] if results else 0.0

    def search_subset(
        self, query_embedding: np.ndarray, record_ids: list[int], top_k: int
    ) -> list[tuple[int, float]]:
        """Ranqueia `record_ids` por similaridade com `query_embedding`, SEM
        comparar com o restante do índice. Usado para restringir a busca às
        chunks de um único arquivo (ex.: leitura de um arquivo específico
        com top_k de chunks) em vez do índice inteiro.
        """
        if not record_ids:
            return []
        q = query_embedding.reshape(-1).astype(np.float32)
        scored: list[tuple[int, float]] = []
        with self._lock:
            if self._index.ntotal == 0:
                return []
            for rid in record_ids:
                try:
                    vec = self._index.reconstruct(int(rid))
                except RuntimeError:
                    continue  # id não presente no índice
                scored.append((rid, float(np.dot(vec, q))))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def flush(self):
        """Persiste o índice em disco SE houver mudanças pendentes desde o
        último flush. Chamado periodicamente pelo `index_flush_job` e uma
        última vez, de forma síncrona, no `shutdown()` — em vez de escrever o
        índice inteiro a cada add()/remove_ids() (ver ITEM 1 da revisão)."""
        with self._lock:
            if not self._persist or not self._dirty:
                return
            faiss.write_index(self._index, self._index_path)
            self._dirty = False

    @property
    def total(self) -> int:
        with self._lock:
            return self._index.ntotal


# ── Estado global ──────────────────────────────────────────────────────────────

@dataclass
class AppState:
    embed_engine: EmbeddingEngine = field(default=None)
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
    decay_task:   asyncio.Task    = field(default=None)
    cleanup_task: asyncio.Task    = field(default=None)
    flush_task:   asyncio.Task    = field(default=None)

state = AppState()


def _all_indices() -> list["MemoryIndex"]:
    """Todos os índices FAISS do processo — usado pelo job de flush periódico
    e pelo shutdown para persistir tudo de uma vez."""
    return [
        idx for idx in (
            state.lt_index, state.st_index,
            state.if_index, state.vd_index, state.fd_index,
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
        except Exception as e:
            log.error(f"Erro no decay job: {e}")


async def index_flush_job():
    """Persiste em disco os índices FAISS marcados como 'sujos' desde o
    último ciclo. Substitui o antigo `_save()` a cada escrita (ver ITEM 1 da
    revisão) — o custo de I/O passa a ser amortizado em vez de pago a cada
    add()/remove_ids()."""
    while True:
        await asyncio.sleep(INDEX_FLUSH_INTERVAL_S)
        loop = asyncio.get_event_loop()
        for index in _all_indices():
            try:
                await loop.run_in_executor(None, index.flush)
            except Exception as e:
                log.error(f"Erro ao persistir índice FAISS: {e}")


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

    state.lt_db    = MemoryDB(DB_PATH)
    state.lt_index = MemoryIndex(FAISS_INDEX_PATH, FAISS_ID_MAP_PATH)

    state.st_db    = ShortTermDB(ST_DB_PATH)
    state.st_index = MemoryIndex(ST_FAISS_INDEX_PATH, ST_FAISS_ID_MAP_PATH)

    # ── NEW: Indexed files ──
    state.if_db    = IndexedFilesDB(IF_DB_PATH)
    state.if_index = MemoryIndex(IF_FAISS_INDEX_PATH, IF_FAISS_ID_MAP_PATH)

    # ── NEW: Dicionário Visual ──
    state.vd_db    = VisualDictDB(VD_DB_PATH)
    state.vd_index = MemoryIndex(VD_FAISS_INDEX_PATH, VD_FAISS_ID_MAP_PATH, embed_dim=VD_EMBED_DIM)

    # ── NEW: Dicionário de Rostos ──
    state.fd_db    = FaceDictDB(FD_DB_PATH)
    state.fd_index = MemoryIndex(FD_FAISS_INDEX_PATH, FD_FAISS_ID_MAP_PATH, embed_dim=FD_EMBED_DIM)

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

    log.info(
        f"Pronto — {state.lt_db.count()} memórias LT | "
        f"{state.st_db.count()} grupos ST | "
        f"{state.vs.total if state.vs else 0} chunks de conhecimento | "
        f"{state.if_db.count_files()} arquivos indexados ({state.if_db.get_total_chunks()} chunks) | "
        f"{state.vd_db.count()} conceitos visuais ({state.vd_index.total} embeddings) | "
        f"{state.fd_db.count()} pessoas cadastradas ({state.fd_index.total} embeddings de rosto)"
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

    # Flush final e síncrono de todos os índices — garante que nada gravado
    # desde o último ciclo do index_flush_job seja perdido ao encerrar.
    for index in _all_indices():
        try:
            index.flush()
        except Exception as e:
            log.error(f"Erro ao persistir índice FAISS no shutdown: {e}")

    log.info("AVA Memory (MCP) encerrada")


# ── Erros ──────────────────────────────────────────────────────────────────────

class MemoryToolError(Exception):
    """Erro de validação/negócio de uma tool de memória — equivalente ao
    antigo HTTPException do FastAPI, mas agnóstico de transporte (MCP não
    tem código de status HTTP). A mensagem vira o texto de erro da tool."""
    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


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
    embeddings = await state.embed_engine.embed(segments)  # shape (N, dim)
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
    emb = await state.embed_engine.embed_one(expanded)
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
    emb_query, emb_ctx = await state.embed_engine.embed_batch_two(query, context_block)
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


def _build_lt_entries(
    lt_raw: list[tuple[int, float]],
    min_score: float,
    loop: asyncio.AbstractEventLoop,
    max_chars: int = READ_LT_MAX_CHARS,
) -> list[MemoryEntry]:
    ids_filtered = [mid for mid, score in lt_raw if score >= min_score]
    score_map    = {mid: score for mid, score in lt_raw}
    if not ids_filtered:
        return []
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
_vd_write_lock = asyncio.Lock()
_fd_write_lock = asyncio.Lock()


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
                embedding = await state.embed_engine.embed_one(text)
                hits = state.lt_index.search(embedding, top_k=1)
                if hits and hits[0][1] >= UPDATE_SIM_THRESHOLD:
                    target_id = hits[0][0]

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
            new_embedding = await state.embed_engine.embed_one(text)
            await loop.run_in_executor(None, state.lt_index.remove_ids, {target_id})
            await loop.run_in_executor(None, state.lt_index.add, new_embedding, target_id)

        log.info(f"LT #{target_id} atualizada: {text[:60]}")
        return True, "updated", target_id, None

    # ── Fluxo normal de criação ─────────────────────────────────────────
    async with _lt_write_lock:
        if state.lt_db.exists_exact(text):
            return False, "duplicate_exact", None, None

        embedding = await state.embed_engine.embed_one(text)

        hits = state.lt_index.search(embedding, top_k=1)
        max_sim, nearest_id = (hits[0][1], hits[0][0]) if hits else (0.0, None)

        if max_sim >= DEDUP_THRESHOLD:
            return False, f"duplicate_semantic:{max_sim:.3f}", None, None

        # ── NEW (solução 2): faixa "provável correção" — nem duplicata
        # clara nem claramente um fato novo. Não decide sozinho: devolve
        # a candidata pro chamador decidir (reenviar com action="update").
        if max_sim >= UPDATE_SIM_THRESHOLD and nearest_id is not None:
            candidate_row = state.lt_db.get_by_id(nearest_id)
            candidate = None
            if candidate_row is not None:
                candidate = {
                    "id":    candidate_row["id"],
                    "text":  candidate_row["text"],
                    "score": round(max_sim, 4),
                }
            return False, f"possible_update:{max_sim:.3f}", None, candidate

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

    log.info(f"LT #{memory_id} gravada: {text[:60]}")
    return True, "ok", memory_id, None


async def _process_write_request(req: WriteRequest) -> WriteResponse:
    """Ponto único usado tanto por /write quanto por /write_batch (solução 1),
    pra garantir que os dois caminhos tenham exatamente a mesma lógica de
    dedup/update/ttl."""
    stored, reason, memory_id, candidate = await _store_long_term_text(
        req.text, req.source, req.confidence, req.forgettable,
        ttl_days=req.ttl_days, action=req.action, memory_id=req.memory_id,
    )
    resp = WriteResponse(stored=stored, reason=reason, memory_id=memory_id)
    if candidate is not None:
        resp.candidate_id    = candidate["id"]
        resp.candidate_text  = candidate["text"]
        resp.candidate_score = candidate["score"]
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
    for item in req.items:
        results.append(await _process_write_request(item))
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
        embedding = await state.embed_engine.embed_one(embed_text)

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

    query_emb = await state.embed_engine.embed_one(query)

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
        _build_st_entries(st_raw, strict_min_score, loop) +
        _build_vs_entries(vs_results, strict_min_score) +
        _build_if_entries(if_raw, if_strict_min_score, loop)  # NEW
    )
    results.sort(key=lambda r: r.score * r.confidence, reverse=True)

    # ── Orçamento global de tokens ─────────────────────────────────────────────
    # Limita o total de caracteres retornados, cortando entradas de menor score.
    # Sempre retorna pelo menos 1 entrada se existir.
    final_top_k = min(req.top_k, READ_TOP_K_FINAL) if req.top_k > 0 else READ_TOP_K_FINAL
    results = _apply_token_budget(results, READ_TOTAL_MAX_CHARS, final_top_k)

    # Log de diagnóstico (nível INFO para acompanhar redução no pipeline)
    total_chars = sum(len(r.text or "") for r in results)
    log.info(
        f"/read query='{query[:50]}' strategy={effective_strategy} "
        f"results={len(results)} total_chars={total_chars} "
        f"budget={READ_TOTAL_MAX_CHARS} strict_score={strict_min_score:.2f}"
    )

    return ReadResponse(results=results, query=query, strategy=effective_strategy)


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


# ══════════════════════════════════════════════════════════════════════════════
# NEW: Arquivos Indexados — /indexed-file/*
# ══════════════════════════════════════════════════════════════════════════════

async def indexed_file_write(req: IndexedFileWriteRequest):
    """
    Armazena o conteúdo COMPLETO de um arquivo indexado.

    Fluxo:
      1. Verifica se o arquivo já está indexado (por file_path)
      2. Se existe e file_hash é igual → sem reindexação necessária
      3. Se existe e file_hash difere → remove chunks antigos, reindexa
      4. Se não existe → insere e indexa

    O conteúdo é dividido em chunks, cada chunk é embedado e
    adicionado ao FAISS para busca semântica. O hash do arquivo
    é armazenado para detectar mudanças futuras.
    """
    content = req.content
    if not content or not content.strip():
        return IndexedFileWriteResponse(stored=False, reason="content_empty")

    if len(content) > IF_MAX_CONTENT_SIZE:
        return IndexedFileWriteResponse(
            stored=False,
            reason=f"content_too_large:{len(content)}>{IF_MAX_CONTENT_SIZE}",
        )

    content_hash = hashlib.sha256(content.encode()).hexdigest()
    loop = asyncio.get_event_loop()

    # ── Check existing ──
    existing = state.if_db.get_by_path(req.file_path)

    if existing and not req.force_reindex:
        if existing["file_hash"] == req.file_hash and existing["content_hash"] == content_hash:
            # Arquivo inalterado — nada a fazer
            log.info(f"Indexed file unchanged: {req.file_path} (hash_match=True)")
            return IndexedFileWriteResponse(
                stored=False,
                reason="unchanged",
                file_id=existing["id"],
                chunks_created=0,
                was_reindexed=False,
                hash_match=True,
            )

    # ── Remove old chunks if re-indexing ──
    if existing:
        old_chunk_ids = state.if_db.get_chunk_ids_by_file(existing["id"])
        if old_chunk_ids:
            await loop.run_in_executor(None, state.if_index.remove_ids, set(old_chunk_ids))
            state.if_db.delete_chunks_by_file(existing["id"])
            log.info(f"Removed {len(old_chunk_ids)} old chunks for: {req.file_path}")

    # ── Chunk the content ──
    chunks = _chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        return IndexedFileWriteResponse(stored=False, reason="chunking_failed")

    # ── Embed chunks in batches ──
    chunk_texts = [c["text"] for c in chunks]
    all_embeddings = []

    for batch_start in range(0, len(chunk_texts), IF_EMBED_BATCH_SIZE):
        batch = chunk_texts[batch_start:batch_start + IF_EMBED_BATCH_SIZE]
        try:
            batch_embs = await state.embed_engine.embed(batch)
            all_embeddings.append(batch_embs)
        except Exception as e:
            log.error(f"Embedding batch failed for {req.file_path}: {e}")
            return IndexedFileWriteResponse(
                stored=False,
                reason=f"embedding_failed:{e}",
            )

    embeddings = np.vstack(all_embeddings) if all_embeddings else np.empty((0, EMBED_DIM), dtype=np.float32)

    # ── Insert or update file record ──
    was_reindexed = existing is not None
    if existing:
        file_id = existing["id"]
        state.if_db.update_file(
            file_id=file_id,
            content=content,
            content_hash=content_hash,
            file_hash=req.file_hash,
            size=req.size or len(content),
            modified=req.modified,
        )
    else:
        file_id = state.if_db.insert_file(
            file_path=req.file_path,
            file_name=req.file_name,
            extension=req.extension,
            content=content,
            content_hash=content_hash,
            file_hash=req.file_hash,
            size=req.size or len(content),
            modified=req.modified,
            source=req.source,
            confidence=req.confidence,
        )

    # ── Insert chunks into DB ──
    state.if_db.insert_chunks(file_id, chunks)

    # ── Get chunk IDs (just inserted) ──
    chunk_rows = state.if_db.get_chunks_by_file(file_id)
    chunk_ids = [row["id"] for row in chunk_rows]

    if len(chunk_ids) != len(chunks):
        log.warning(
            f"Chunk ID mismatch: expected {len(chunks)}, got {len(chunk_ids)} "
            f"for {req.file_path}"
        )

    # ── Add embeddings to FAISS in batch ──
    if len(chunk_ids) == embeddings.shape[0]:
        await loop.run_in_executor(
            None,
            state.if_index.add_batch,
            embeddings,
            chunk_ids,
        )
    else:
        # Fallback: add one by one if sizes don't match
        log.warning("Chunk/embedding size mismatch — adding individually")
        for emb, cid in zip(embeddings, chunk_ids):
            await loop.run_in_executor(None, state.if_index.add, emb, cid)

    hash_match = (
        existing is not None
        and existing["file_hash"] == req.file_hash
        and existing["content_hash"] == content_hash
    )

    log.info(
        f"Indexed file: {req.file_path} → file_id={file_id} "
        f"chunks={len(chunks)} reindexed={was_reindexed} "
        f"hash_match={hash_match} content_size={len(content)}"
    )

    return IndexedFileWriteResponse(
        stored=True,
        reason="ok" if not was_reindexed else "reindexed",
        file_id=file_id,
        chunks_created=len(chunks),
        was_reindexed=was_reindexed,
        hash_match=hash_match,
    )


async def indexed_file_read(req: IndexedFileReadRequest):
    """
    Lê arquivos indexados. Três modos, mutuamente exclusivos:

      1. `file_path` sozinho → lookup EXATO por caminho absoluto (mesma
         chave usada em `/indexed-file/write`). Não passa pelo FAISS —
         é uma busca direta no SQLite por igualdade de `file_path`, o que
         garante que dois arquivos com o mesmo nome em pastas diferentes
         nunca sejam confundidos. Retorna o arquivo INTEIRO (1 resultado).

      2. `file_path` + `query` → mesmo lookup exato por caminho, mas em vez
         de devolver o arquivo inteiro de uma vez, ranqueia as chunks DESSE
         MESMO ARQUIVO (e só dele — não compara com outros arquivos do
         índice) contra `query` e devolve até `top_k` chunks mais
         relevantes, cada uma com seu `chunk_text` e `score`.

      3. Nem um nem outro, só `query` → busca semântica original entre
         TODOS os arquivos indexados, retornando até `top_k` arquivos
         cujos chunks tiveram similaridade acima do threshold.
    """
    file_path = (req.file_path or "").strip()
    query     = (req.query or "").strip()

    # ── Modo 1/2: caminho absoluto informado ──
    if file_path:
        row = state.if_db.get_by_path(file_path)
        if row is None:
            return IndexedFileReadResponse(results=[], file_path=file_path, query=query or None)

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, state.if_db.update_access, row["id"])

        # ── Modo 2: sem query → devolve o arquivo inteiro (comportamento original) ──
        if not query:
            entry = IndexedFileEntry(
                file_id      = row["id"],
                file_path    = row["file_path"],
                file_name    = row["file_name"],
                extension    = row["extension"],
                content      = row["content"],           # conteúdo COMPLETO
                file_hash    = row["file_hash"],
                content_hash = row["content_hash"],
                size         = row["size"],
                modified     = row["modified"],
                score        = 1.0,                       # lookup exato — sem score de similaridade
                confidence   = round(row["confidence"], 4),
                created_at   = row["created_at"],
                access_count = row["access_count"],
                source       = row["source"],
                chunk_text   = None,
                match_type   = "exact_path",
            )
            return IndexedFileReadResponse(results=[entry], file_path=file_path)

        # ── Modo 2b: com query → top_k chunks ranqueadas, restritas a este arquivo ──
        chunk_rows = state.if_db.get_chunks_by_file(row["id"])
        chunk_ids  = [c["id"] for c in chunk_rows]
        if not chunk_ids:
            return IndexedFileReadResponse(results=[], file_path=file_path, query=query)

        query_emb = await state.embed_engine.embed_one(query)
        ranked = await loop.run_in_executor(
            None, state.if_index.search_subset, query_emb, chunk_ids, req.top_k
        )
        ranked = [(cid, score) for cid, score in ranked if score >= req.min_score]
        if not ranked:
            return IndexedFileReadResponse(results=[], file_path=file_path, query=query)

        chunk_by_id = {c["id"]: c for c in chunk_rows}
        full_content = row["content"] if req.include_full_content else ""
        results = []
        for cid, score in ranked:
            chunk_row = chunk_by_id.get(cid)
            if chunk_row is None:
                continue
            results.append(IndexedFileEntry(
                file_id      = row["id"],
                file_path    = row["file_path"],
                file_name    = row["file_name"],
                extension    = row["extension"],
                content      = full_content,              # arquivo completo — vazio se include_full_content=False
                file_hash    = row["file_hash"],
                content_hash = row["content_hash"],
                size         = row["size"],
                modified     = row["modified"],
                score        = round(score, 4),
                confidence   = round(row["confidence"], 4),
                created_at   = row["created_at"],
                access_count = row["access_count"],
                source       = row["source"],
                chunk_text   = chunk_row["chunk_text"],   # a chunk específica ranqueada
                chunk_index  = chunk_row["chunk_index"],
                char_start   = chunk_row["char_start"],
                char_end     = chunk_row["char_end"],
                chunk_id = chunk_row["id"],
                match_type   = "exact_path_chunks",
            ))
        return IndexedFileReadResponse(results=results, file_path=file_path, query=query)

    # ── Modo 3: busca semântica entre todos os arquivos ──
    if not query:
        raise MemoryToolError("informe 'file_path' (leitura exata) ou 'query' (busca semântica)")

    if state.if_index.total == 0:
        return IndexedFileReadResponse(results=[], query=query)

    loop = asyncio.get_event_loop()
    query_emb = await state.embed_engine.embed_one(query)

    if_raw = await loop.run_in_executor(
        None, state.if_index.search, query_emb, req.top_k * 2
    )

    # Filtra por score
    filtered = [(cid, score) for cid, score in if_raw if score >= req.min_score]
    if not filtered:
        return IndexedFileReadResponse(results=[], query=query)

    # Busca chunk records
    chunk_ids = [cid for cid, _ in filtered]
    chunk_rows = state.if_db.get_chunks_by_ids(chunk_ids)
    chunk_map = {row["id"]: row for row in chunk_rows}

    # Agrupa por file_id — melhor score por arquivo
    file_best: dict[int, tuple[float, sqlite3.Row]] = {}
    for cid, score in filtered:
        chunk_row = chunk_map.get(cid)
        if chunk_row is None:
            continue
        fid = chunk_row["file_id"]
        if fid not in file_best or score > file_best[fid][0]:
            file_best[fid] = (score, chunk_row)

    if not file_best:
        return IndexedFileReadResponse(results=[], query=query)

    # Busca file records
    file_ids = list(file_best.keys())
    file_rows = state.if_db._conn.execute(
        f"SELECT * FROM indexed_files WHERE id IN ({','.join('?' * len(file_ids))})",
        file_ids,
    ).fetchall()
    file_map = {row["id"]: row for row in file_rows}

    results = []
    for fid, (score, chunk_row) in sorted(file_best.items(), key=lambda x: x[1][0], reverse=True):
        file_row = file_map.get(fid)
        if file_row is None:
            continue

        results.append(IndexedFileEntry(
            file_id      = fid,
            file_path    = file_row["file_path"],
            file_name    = file_row["file_name"],
            extension    = file_row["extension"],
            content      = file_row["content"],          # conteúdo COMPLETO
            file_hash    = file_row["file_hash"],
            content_hash = file_row["content_hash"],
            size         = file_row["size"],
            modified     = file_row["modified"],
            score        = round(score, 4),
            confidence   = round(file_row["confidence"], 4),
            created_at   = file_row["created_at"],
            access_count = file_row["access_count"],
            source       = file_row["source"],
            chunk_text   = chunk_row["chunk_text"],       # chunk que deu match
            chunk_index  = chunk_row["chunk_index"],
            char_start   = chunk_row["char_start"],
            char_end     = chunk_row["char_end"],
        ))
        loop.run_in_executor(None, state.if_db.update_access, fid)

    return IndexedFileReadResponse(results=results[:req.top_k], query=query)


async def indexed_file_check(file_path: str):
    """
    Verifica se um arquivo está indexado e se o hash bate.

    Usado pelo local-scraping para decidir se precisa reindexar:
      - indexed=False  → arquivo nunca foi indexado
      - hash_match=True → já indexado e inalterado
      - hash_match=False → indexado mas arquivo mudou → reindexar
    """
    if not file_path:
        raise MemoryToolError("file_path vazio")

    row = state.if_db.get_by_path(file_path)
    if row is None:
        return IndexedFileCheckResponse(indexed=False)

    chunks_count = state.if_db.count_chunks(row["id"])

    return IndexedFileCheckResponse(
        indexed            = True,
        file_id            = row["id"],
        stored_file_hash   = row["file_hash"],
        stored_content_hash = row["content_hash"],
        stored_modified    = row["modified"],
        chunks_count       = chunks_count,
        hash_match         = None,  # caller compara com o hash atual
    )


async def indexed_file_get(file_id: int):
    """
    Retorna o conteúdo completo de um arquivo indexado pelo seu ID.
    """
    row = state.if_db.get_by_id(file_id)
    if row is None:
        raise MemoryToolError(f"Arquivo indexado #{file_id} não encontrado")

    chunks = state.if_db.get_chunks_by_file(file_id)

    return {
        "file_id":      row["id"],
        "file_path":    row["file_path"],
        "file_name":    row["file_name"],
        "extension":    row["extension"],
        "content":      row["content"],
        "content_hash": row["content_hash"],
        "file_hash":    row["file_hash"],
        "size":         row["size"],
        "modified":     row["modified"],
        "source":       row["source"],
        "confidence":   row["confidence"],
        "created_at":   row["created_at"],
        "access_count": row["access_count"],
        "chunks_count": len(chunks),
    }



async def get_chunk_by_id(chunk_id: int):
    row = state.if_db._conn.execute(
        "SELECT c.*, f.file_path, f.file_hash, f.content "
        "FROM indexed_file_chunks c "
        "JOIN indexed_files f ON c.file_id = f.id "
        "WHERE c.id = ?", (chunk_id,)
    ).fetchone()
    if row is None:
        raise MemoryToolError(f"Chunk #{chunk_id} não encontrado")
    return {
        "chunk_id": row["id"],
        "file_path": row["file_path"],
        "file_hash": row["file_hash"],
        "char_start": row["char_start"],
        "char_end": row["char_end"],
        "chunk_text": row["chunk_text"],
        "file_content": row["content"],  # conteúdo completo para validação
    }


async def indexed_file_delete(file_id: int):
    """
    Remove um arquivo indexado e todos os seus chunks (DB + FAISS).
    """
    row = state.if_db.get_by_id(file_id)
    if row is None:
        raise MemoryToolError(f"Arquivo indexado #{file_id} não encontrado")

    # Remove FAISS vectors first
    chunk_ids = state.if_db.get_chunk_ids_by_file(file_id)
    if chunk_ids:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, state.if_index.remove_ids, set(chunk_ids))

    # Delete from DB (cascades to chunks)
    deleted = state.if_db.delete_file(file_id)

    log.info(f"Indexed file deleted: #{file_id} ({len(chunk_ids)} chunks removed)")
    return {"deleted": deleted, "file_id": file_id, "chunks_removed": len(chunk_ids)}


async def indexed_file_delete_by_path(file_path: str):
    """
    Remove um arquivo indexado pelo caminho.
    """
    if not file_path:
        raise MemoryToolError("file_path vazio")

    row = state.if_db.get_by_path(file_path)
    if row is None:
        return {"deleted": 0, "file_path": file_path, "message": "not indexed"}

    return await indexed_file_delete(row["id"])


async def indexed_file_list():
    """
    Lista todos os arquivos indexados com metadados.
    """
    rows = state.if_db.list_files()
    files = []
    for row in rows:
        files.append({
            "file_id":      row["id"],
            "file_path":    row["file_path"],
            "file_name":    row["file_name"],
            "extension":    row["extension"],
            "content_hash": row["content_hash"],
            "file_hash":    row["file_hash"],
            "size":         row["size"],
            "modified":     row["modified"],
            "created_at":   row["created_at"],
            "access_count": row["access_count"],
        })
    return {"total": len(files), "files": files}


# ── NEW: Dicionário Visual — endpoints usados pelo vision.py ──────────────────
#
# vision.py NÃO persiste nada localmente: ele roda o pipeline (depth →
# clustering → segmentação → embeddings) e manda cada embedding de crop pra
# cá. Este arquivo decide se é um conceito novo ou um exemplo a mais de um
# conceito já existente, e é quem guarda tudo (FAISS + SQLite + memória de
# longo prazo).

async def visual_dict_write(req: VisualDictWriteRequest):
    concept_name = req.concept_name.strip()
    description  = req.description.strip()
    if not concept_name:
        return VisualDictWriteResponse(stored=False, reason="empty_concept_name")
    if len(req.embedding) != VD_EMBED_DIM:
        raise MemoryToolError(f"embedding deve ter dimensão {VD_EMBED_DIM}, recebido {len(req.embedding)}")

    vec = np.asarray(req.embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm

    loop = asyncio.get_event_loop()

    async with _vd_write_lock:
        existing = state.vd_db.get_by_name(concept_name)
        new_concept = existing is None

        if existing is not None:
            concept_id = existing["id"]
            memory_id  = existing["memory_id"]
        else:
            memory_id = None
            if req.link_to_memory and description:
                _, _, memory_id, _ = await _store_long_term_text(
                    f"{concept_name}: {description}", "visual_dict", req.confidence,
                )
            try:
                concept_id = state.vd_db.insert_concept(
                    concept_name, description, req.source, req.confidence, memory_id,
                )
            except sqlite3.IntegrityError:
                # Corrida: outro write criou o mesmo concept_key nesse meio-tempo.
                existing = state.vd_db.get_by_name(concept_name)
                if existing is None:
                    raise
                concept_id, memory_id, new_concept = existing["id"], existing["memory_id"], False
            else:
                log.info(f"Visual-dict: novo conceito #{concept_id} criado — '{concept_name}'")

        embedding_id = state.vd_db.insert_embedding(concept_id)
        await loop.run_in_executor(None, state.vd_index.add, vec, embedding_id)

    log.info(
        f"Visual-dict: embedding #{embedding_id} gravado p/ conceito #{concept_id} "
        f"({'novo' if new_concept else 'exemplo adicional'})"
    )

    return VisualDictWriteResponse(
        stored=True, reason="ok", concept_id=concept_id,
        embedding_id=embedding_id, memory_id=memory_id, new_concept=new_concept,
    )


async def visual_dict_read(req: VisualDictReadRequest):
    if len(req.embedding) != VD_EMBED_DIM:
        raise MemoryToolError(f"embedding deve ter dimensão {VD_EMBED_DIM}, recebido {len(req.embedding)}")

    vec = np.asarray(req.embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm

    # sobre-amostra o kNN porque vários embeddings podem apontar pro mesmo
    # conceito (múltiplos exemplos) — precisamos deduplicar por concept_id
    hits = state.vd_index.search(vec, max(req.top_k * 4, 20))

    best_score_by_concept: dict[int, float] = {}
    for embedding_id, score in hits:
        concept_id = state.vd_db.get_concept_id_by_embedding(embedding_id)
        if concept_id is None:
            continue
        if concept_id not in best_score_by_concept or score > best_score_by_concept[concept_id]:
            best_score_by_concept[concept_id] = score

    ranked = sorted(best_score_by_concept.items(), key=lambda kv: kv[1], reverse=True)[:req.top_k]

    results: list[VisualDictCandidate] = []
    for concept_id, score in ranked:
        if score < req.min_score:
            continue
        row = state.vd_db.get_concept_by_id(concept_id)
        if row is None:
            continue
        state.vd_db.update_access(concept_id)
        results.append(VisualDictCandidate(
            concept_id=row["id"],
            concept_name=row["concept_name"],
            description=row["description"],
            score=score,
            confidence=row["confidence"],
            access_count=row["access_count"] + 1,
            memory_id=row["memory_id"],
        ))

    # ambíguo quando: nenhum resultado confiável, OU os dois melhores
    # candidatos estão muito próximos (o objeto pode ser qualquer um dos dois)
    ambiguous = (
        len(results) == 0
        or (len(results) > 1 and (results[0].score - results[1].score) < VD_AMBIGUOUS_MARGIN)
    )

    return VisualDictReadResponse(results=results, ambiguous=ambiguous)


async def visual_dict_list():
    rows = state.vd_db.list_concepts()
    concepts = [
        VisualDictEntry(
            concept_id=row["id"],
            concept_name=row["concept_name"],
            description=row["description"],
            source=row["source"],
            confidence=row["confidence"],
            memory_id=row["memory_id"],
            examples_count=row["examples_count"],
            created_at=row["created_at"],
            access_count=row["access_count"],
        ).model_dump()
        for row in rows
    ]
    return {"total": len(concepts), "concepts": concepts}


async def visual_dict_get(concept_id: int):
    row = state.vd_db.get_concept_by_id(concept_id)
    if row is None:
        raise MemoryToolError(f"Conceito visual #{concept_id} não encontrado")
    return VisualDictEntry(
        concept_id=row["id"],
        concept_name=row["concept_name"],
        description=row["description"],
        source=row["source"],
        confidence=row["confidence"],
        memory_id=row["memory_id"],
        examples_count=state.vd_db.count_examples(concept_id),
        created_at=row["created_at"],
        access_count=row["access_count"],
    )


async def visual_dict_delete(concept_id: int):
    """Remove um conceito visual e todos os seus embeddings (DB + FAISS)."""
    row = state.vd_db.get_concept_by_id(concept_id)
    if row is None:
        raise MemoryToolError(f"Conceito visual #{concept_id} não encontrado")

    embedding_ids = state.vd_db.get_embedding_ids_by_concept(concept_id)
    if embedding_ids:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, state.vd_index.remove_ids, set(embedding_ids))

    deleted = state.vd_db.delete_concept(concept_id)
    log.info(f"Visual-dict: conceito #{concept_id} removido ({len(embedding_ids)} embeddings)")
    return {"deleted": deleted, "concept_id": concept_id, "embeddings_removed": len(embedding_ids)}


# ── NEW: Dicionário de Rostos — endpoints usados pelo vision.py ───────────────
#
# Mesma lógica do dicionário visual acima, adaptada pra reconhecimento
# facial: vision.py detecta+alinha o rosto, extrai o embedding (EdgeFace) e
# manda pra cá. Aqui decide-se se é uma pessoa nova ou mais um exemplo de
# uma pessoa já cadastrada.

async def face_dict_write(req: FaceDictWriteRequest):
    person_name = req.person_name.strip()
    if not person_name:
        return FaceDictWriteResponse(stored=False, reason="empty_person_name")

    if len(req.embedding) != FD_EMBED_DIM:
        raise MemoryToolError(f"embedding deve ter dimensão {FD_EMBED_DIM}, recebido {len(req.embedding)}")

    vec = np.asarray(req.embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm

    description = req.description.strip()
    loop = asyncio.get_event_loop()

    async with _fd_write_lock:
        existing = state.fd_db.get_by_name(person_name)
        new_person = existing is None

        if existing is not None:
            person_id = existing["id"]
            # se veio uma descrição não-vazia num cadastro de exemplo adicional,
            # atualiza/completa a descrição já salva (permite corrigir depois)
            if description:
                state.fd_db.update_description(person_id, description)
        else:
            try:
                person_id = state.fd_db.insert_person(
                    person_name=person_name, description=description,
                    source=req.source, confidence=req.confidence,
                )
            except sqlite3.IntegrityError:
                # Corrida: outro write criou a mesma person_key nesse meio-tempo.
                existing = state.fd_db.get_by_name(person_name)
                if existing is None:
                    raise
                person_id, new_person = existing["id"], False

        embedding_id = state.fd_db.insert_embedding(person_id)
        await loop.run_in_executor(None, state.fd_index.add, vec, embedding_id)

    log.info(
        f"Face-dict: {'nova pessoa' if new_person else 'novo exemplo'} "
        f"'{person_name}' (person_id={person_id}, embedding_id={embedding_id})"
    )

    return FaceDictWriteResponse(
        stored=True, reason="ok", person_id=person_id,
        embedding_id=embedding_id, new_person=new_person,
    )


async def face_dict_read(req: FaceDictReadRequest):
    if len(req.embedding) != FD_EMBED_DIM:
        raise MemoryToolError(f"embedding deve ter dimensão {FD_EMBED_DIM}, recebido {len(req.embedding)}")

    vec = np.asarray(req.embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm

    # sobre-amostra o kNN porque vários embeddings podem apontar pra mesma
    # pessoa (múltiplos exemplos) — precisamos deduplicar por person_id
    hits = state.fd_index.search(vec, max(req.top_k * 4, 20))

    best_score_by_person: dict[int, float] = {}
    for embedding_id, score in hits:
        person_id = state.fd_db.get_person_id_by_embedding(embedding_id)
        if person_id is None:
            continue
        if person_id not in best_score_by_person or score > best_score_by_person[person_id]:
            best_score_by_person[person_id] = score

    ranked = sorted(best_score_by_person.items(), key=lambda kv: kv[1], reverse=True)[:req.top_k]

    results: list[FaceCandidate] = []
    for person_id, score in ranked:
        if score < req.min_score:
            continue
        row = state.fd_db.get_person_by_id(person_id)
        if row is None:
            continue
        state.fd_db.update_access(person_id)
        results.append(FaceCandidate(
            person_id=row["id"],
            person_name=row["person_name"],
            description=row["description"],
            score=score,
            confidence=row["confidence"],
            access_count=row["access_count"] + 1,
        ))

    # ambíguo quando: ninguém bateu com confiança suficiente, OU os dois
    # melhores candidatos estão muito próximos (pode ser qualquer um dos dois)
    ambiguous = (
        len(results) == 0
        or (len(results) > 1 and (results[0].score - results[1].score) < FD_AMBIGUOUS_MARGIN)
    )

    return FaceDictReadResponse(results=results, ambiguous=ambiguous)


async def face_dict_list():
    rows = state.fd_db.list_people()
    people = [
        FaceDictEntry(
            person_id=row["id"],
            person_name=row["person_name"],
            description=row["description"],
            source=row["source"],
            confidence=row["confidence"],
            examples_count=row["examples_count"],
            created_at=row["created_at"],
            access_count=row["access_count"],
        ).model_dump()
        for row in rows
    ]
    return {"total": len(people), "people": people}


async def face_dict_get(person_id: int):
    row = state.fd_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")
    return FaceDictEntry(
        person_id=row["id"],
        person_name=row["person_name"],
        description=row["description"],
        source=row["source"],
        confidence=row["confidence"],
        examples_count=state.fd_db.count_examples(person_id),
        created_at=row["created_at"],
        access_count=row["access_count"],
    )


class FaceDictUpdateRequest(BaseModel):
    description: str

async def face_dict_update(person_id: int, req: FaceDictUpdateRequest):
    """Edita só a descrição de uma pessoa já cadastrada, sem precisar
    mandar um novo embedding junto."""
    row = state.fd_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")
    state.fd_db.update_description(person_id, req.description.strip())
    row = state.fd_db.get_person_by_id(person_id)
    return FaceDictEntry(
        person_id=row["id"],
        person_name=row["person_name"],
        description=row["description"],
        source=row["source"],
        confidence=row["confidence"],
        examples_count=state.fd_db.count_examples(person_id),
        created_at=row["created_at"],
        access_count=row["access_count"],
    )


async def face_dict_delete(person_id: int):
    """Remove uma pessoa e todos os seus embeddings de rosto (DB + FAISS)."""
    row = state.fd_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")

    embedding_ids = state.fd_db.get_embedding_ids_by_person(person_id)
    if embedding_ids:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, state.fd_index.remove_ids, set(embedding_ids))

    deleted = state.fd_db.delete_person(person_id)
    log.info(f"Face-dict: pessoa #{person_id} removida ({len(embedding_ids)} embeddings)")
    return {"deleted": deleted, "person_id": person_id, "embeddings_removed": len(embedding_ids)}


# ── GET /status ────────────────────────────────────────────────────────────────

async def status():
    resp = {
        "long_term": {
            "memories_total":       state.lt_db.count(),
            "index_vectors":        state.lt_index.total,
            "decay_half_life_days": DECAY_HALF_LIFE_DAYS,
            "dedup_threshold":      DEDUP_THRESHOLD,
            "update_sim_threshold": UPDATE_SIM_THRESHOLD,
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