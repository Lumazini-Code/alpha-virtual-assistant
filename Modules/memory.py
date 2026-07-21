from __future__ import annotations

import re
import time
import json
import sqlite3
import hashlib
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
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

# Cache de planos (CoT)
PC_DB_PATH           = "./memory/ava_plan_cache.db"
PC_FAISS_INDEX_PATH  = "./memory/ava_plan_cache.index"
PC_FAISS_ID_MAP_PATH = "./memory/ava_plan_cache_id_map.npy"

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

EMBED_DIM            = 384
READ_MIN_SCORE       = 0.83
DEDUP_THRESHOLD      = 0.92
TOP_K_READ           = 5
DECAY_HALF_LIFE_DAYS = 90
DECAY_JOB_INTERVAL_S = 3600

ST_TTL_HOURS          = 24.0
ST_CLEANUP_INTERVAL_S = 1800

PC_HIT_THRESHOLD = 0.92

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
    text:       str
    source:     str   = "chat"
    confidence: float = 1.0

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

class WriteSTResponse(BaseModel):
    stored:   bool
    reason:   str
    turn_ids: list[int] = []

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


# ── Modelos de request/response — cache de planos ─────────────────────────────

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
    query:     str
    top_k:     int   = 5
    min_score: float = IF_MIN_SCORE

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

class IndexedFileReadResponse(BaseModel):
    results: list[IndexedFileEntry]
    query:   str

class IndexedFileCheckResponse(BaseModel):
    indexed:          bool
    hash_match:       Optional[bool]   = None
    file_id:          Optional[int]    = None
    stored_file_hash: Optional[str]    = None
    stored_content_hash: Optional[str] = None
    stored_modified:  Optional[str]    = None
    chunks_count:     Optional[int]    = None


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

    def insert(self, text: str, source: str, confidence: float) -> int:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO memories (text, text_hash, source, confidence, created_at, last_accessed) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (text, text_hash, source, confidence, now, now),
        )
        return cur.lastrowid

    def exists_exact(self, text: str) -> bool:
        h = hashlib.sha256(text.encode()).hexdigest()
        return self._conn.execute(
            "SELECT 1 FROM memories WHERE text_hash = ?", (h,)
        ).fetchone() is not None

    def get_by_ids(self, ids: list[int]) -> list[sqlite3.Row]:
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        return self._conn.execute(
            f"SELECT * FROM memories WHERE id IN ({ph})", ids
        ).fetchall()

    def update_access(self, memory_id: int):
        try:
            self._conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (time.time(), memory_id),
            )
        except sqlite3.OperationalError:
            pass

    def apply_decay(self, half_life_days: float):
        now  = time.time()
        rows = self._conn.execute(
            "SELECT id, confidence, last_accessed FROM memories WHERE confidence > 0.01"
        ).fetchall()
        updates = []
        for row in rows:
            days_idle    = (now - row["last_accessed"]) / 86400.0
            decay_factor = 0.5 ** (days_idle / half_life_days)
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

    def update_access(self, group_id: int):
        try:
            self._conn.execute(
                "UPDATE turn_groups SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (time.time(), group_id),
            )
        except sqlite3.OperationalError:
            pass

    def expire_old(self, ttl_hours: float) -> int:
        cutoff = time.time() - ttl_hours * 3600
        cur = self._conn.execute(
            "DELETE FROM turn_groups WHERE last_accessed < ?", (cutoff,)
        )
        removed = cur.rowcount
        if removed:
            log.info(f"Curto prazo: {removed} grupos expirados (TTL={ttl_hours}h)")
        return removed

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM turn_groups").fetchone()[0]


# ── Banco de dados do cache de planos ─────────────────────────────────────────

class PlanCacheDB:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS plan_cache (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT    NOT NULL,
                plan_json  TEXT    NOT NULL,
                hit_count  INTEGER NOT NULL DEFAULT 0,
                created_at REAL    NOT NULL,
                last_hit   REAL    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pc_last_hit ON plan_cache(last_hit);
        """)

    def insert(self, query_text: str, plan: dict) -> int:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO plan_cache (query_text, plan_json, created_at, last_hit) "
            "VALUES (?, ?, ?, ?)",
            (query_text, json.dumps(plan), now, now),
        )
        return cur.lastrowid

    def get_by_id(self, cache_id: int) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM plan_cache WHERE id = ?", (cache_id,)
        ).fetchone()

    def update_hit(self, cache_id: int):
        try:
            self._conn.execute(
                "UPDATE plan_cache SET hit_count = hit_count + 1, last_hit = ? WHERE id = ?",
                (time.time(), cache_id),
            )
        except sqlite3.OperationalError:
            pass

    def delete_all(self) -> int:
        cur = self._conn.execute("DELETE FROM plan_cache")
        return cur.rowcount

    def delete_by_id(self, cache_id: int) -> int:
        cur = self._conn.execute("DELETE FROM plan_cache WHERE id = ?", (cache_id,))
        return cur.rowcount

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM plan_cache").fetchone()[0]


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
            self._conn.execute(
                "UPDATE indexed_files SET access_count = access_count + 1, "
                "last_accessed = ? WHERE id = ?",
                (time.time(), file_id),
            )
        except sqlite3.OperationalError:
            pass

    def delete_file(self, file_id: int) -> int:
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

class MemoryIndex:
    """FAISS IndexFlatIP — inner product em vetores L2-normalizados = cosine similarity."""

    def __init__(self, index_path: str, id_map_path: str, persist: bool = True):
        self._index_path  = index_path
        self._id_map_path = id_map_path
        self._persist     = persist
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        if persist and Path(index_path).exists() and Path(id_map_path).exists():
            self._index  = faiss.read_index(index_path)
            self._id_map = list(np.load(id_map_path).tolist())
            log.info(f"Índice FAISS carregado [{index_path}] — {self._index.ntotal} vetores")
        else:
            self._index  = faiss.IndexFlatIP(EMBED_DIM)
            self._id_map = []
            log.info(f"Novo índice FAISS criado [{index_path}]")

    def add(self, embedding: np.ndarray, record_id: int):
        self._index.add(embedding.reshape(1, -1))
        self._id_map.append(record_id)
        if self._persist:
            self._save()

    def add_batch(self, embeddings: np.ndarray, record_ids: list[int]):
        """Adiciona múltiplos vetores de uma vez — mais eficiente que add() individual."""
        if embeddings.shape[0] != len(record_ids):
            raise ValueError(
                f"embeddings ({embeddings.shape[0]}) e record_ids ({len(record_ids)}) "
                f"devem ter o mesmo tamanho"
            )
        if embeddings.shape[0] == 0:
            return
        self._index.add(embeddings)
        self._id_map.extend(record_ids)
        if self._persist:
            self._save()

    def remove_ids(self, record_ids: set[int]):
        if not record_ids or self._index.ntotal == 0:
            return
        all_vectors = self._index.reconstruct_n(0, self._index.ntotal)
        new_vecs, new_map = [], []
        for vec, rid in zip(all_vectors, self._id_map):
            if rid not in record_ids:
                new_vecs.append(vec)
                new_map.append(rid)
        self._index = faiss.IndexFlatIP(EMBED_DIM)
        if new_vecs:
            self._index.add(np.array(new_vecs, dtype=np.float32))
        self._id_map = new_map
        if self._persist:
            self._save()
        log.info(f"FAISS: {len(record_ids)} vetores removidos [{self._index_path}]")

    def reset(self):
        self._index  = faiss.IndexFlatIP(EMBED_DIM)
        self._id_map = []
        if self._persist:
            self._save()

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self._index.ntotal == 0:
            return []
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_embedding.reshape(1, -1), k)
        return [
            (self._id_map[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if 0 <= idx < len(self._id_map)
        ]

    def search_similar(self, embedding: np.ndarray) -> float:
        results = self.search(embedding, top_k=1)
        return results[0][1] if results else 0.0

    def _save(self):
        faiss.write_index(self._index, self._index_path)
        np.save(self._id_map_path, np.array(self._id_map, dtype=np.int64))

    @property
    def total(self) -> int:
        return self._index.ntotal


# ── Estado global ──────────────────────────────────────────────────────────────

@dataclass
class AppState:
    embed_engine: EmbeddingEngine = field(default=None)
    lt_db:        MemoryDB        = field(default=None)
    lt_index:     MemoryIndex     = field(default=None)
    st_db:        ShortTermDB     = field(default=None)
    st_index:     MemoryIndex     = field(default=None)
    pc_db:        PlanCacheDB     = field(default=None)
    pc_index:     MemoryIndex     = field(default=None)
    vs:           Optional[VectorStore] = field(default=None)
    # ── NEW: Indexed files ──
    if_db:        IndexedFilesDB  = field(default=None)
    if_index:     MemoryIndex     = field(default=None)
    decay_task:   asyncio.Task    = field(default=None)
    cleanup_task: asyncio.Task    = field(default=None)

state = AppState()


# ── Jobs em background ─────────────────────────────────────────────────────────

async def decay_job():
    while True:
        await asyncio.sleep(DECAY_JOB_INTERVAL_S)
        try:
            state.lt_db.apply_decay(DECAY_HALF_LIFE_DAYS)
        except Exception as e:
            log.error(f"Erro no decay job: {e}")


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Iniciando AVA Memory API...")

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

    state.pc_db    = PlanCacheDB(PC_DB_PATH)
    state.pc_index = MemoryIndex(PC_FAISS_INDEX_PATH, PC_FAISS_ID_MAP_PATH)

    # ── NEW: Indexed files ──
    state.if_db    = IndexedFilesDB(IF_DB_PATH)
    state.if_index = MemoryIndex(IF_FAISS_INDEX_PATH, IF_FAISS_ID_MAP_PATH)

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

    log.info(
        f"Pronto — {state.lt_db.count()} memórias LT | "
        f"{state.st_db.count()} grupos ST | "
        f"{state.pc_db.count()} planos em cache | "
        f"{state.vs.total if state.vs else 0} chunks de conhecimento | "
        f"{state.if_db.count_files()} arquivos indexados ({state.if_db.get_total_chunks()} chunks)"
    )
    yield

    await state.embed_engine.client.close()

    state.decay_task.cancel()
    state.cleanup_task.cancel()
    log.info("AVA Memory API encerrada")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="AVA Memory API", lifespan=lifespan)


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

@app.post("/write", response_model=WriteResponse)
async def write_memory(req: WriteRequest):
    text = req.text.strip()
    if len(text) < 10:
        return WriteResponse(stored=False, reason="too_short")
    if state.lt_db.exists_exact(text):
        return WriteResponse(stored=False, reason="duplicate_exact")

    embedding = await state.embed_engine.embed_one(text)

    max_sim = state.lt_index.search_similar(embedding)
    if max_sim >= DEDUP_THRESHOLD:
        return WriteResponse(stored=False, reason=f"duplicate_semantic:{max_sim:.3f}")

    memory_id = state.lt_db.insert(text, req.source, req.confidence)
    state.lt_index.add(embedding, memory_id)
    log.info(f"LT #{memory_id} gravada: {text[:60]}")
    return WriteResponse(stored=True, reason="ok", memory_id=memory_id)


# ── POST /write_st ─────────────────────────────────────────────────────────────

@app.post("/write_st", response_model=WriteSTResponse)
async def write_short_term(req: WriteSTRequest):
    if not req.turns:
        return WriteSTResponse(stored=False, reason="no_turns")

    all_text = " ".join(t.content.strip() for t in req.turns)
    if len(all_text) < 10:
        return WriteSTResponse(stored=False, reason="too_short")

    embed_text = "\n".join(f"{t.role}: {t.content}" for t in req.turns)
    embedding = await state.embed_engine.embed_one(embed_text)

    max_sim = state.st_index.search_similar(embedding)
    if max_sim >= DEDUP_THRESHOLD:
        return WriteSTResponse(stored=False, reason=f"duplicate_semantic:{max_sim:.3f}")

    group_id = state.st_db.insert(req.session_id, req.turns, embed_text)
    state.st_index.add(embedding, group_id)
    log.info(f"ST #{group_id} gravado — session={req.session_id} turnos={len(req.turns)}: {embed_text[:80]}")
    return WriteSTResponse(stored=True, reason="ok", turn_ids=[group_id])


# ── POST /read ─────────────────────────────────────────────────────────────────

@app.post("/read", response_model=ReadResponse)
async def read_memory(req: ReadRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query vazia")

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

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    rows = state.st_db._conn.execute(
        "SELECT id FROM turn_groups WHERE session_id = ?", (session_id,)
    ).fetchall()
    ids_to_remove = {row["id"] for row in rows}

    if not ids_to_remove:
        return {"cleared": 0, "session_id": session_id}

    state.st_db._conn.execute("DELETE FROM turn_groups WHERE session_id = ?", (session_id,))
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, state.st_index.remove_ids, ids_to_remove)
    log.info(f"Sessão {session_id}: {len(ids_to_remove)} grupos removidos")
    return {"cleared": len(ids_to_remove), "session_id": session_id}


# ══════════════════════════════════════════════════════════════════════════════
# NEW: Arquivos Indexados — /indexed-file/*
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/indexed-file/write", response_model=IndexedFileWriteResponse)
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
        for i, (emb, cid) in enumerate(zip(embeddings, chunk_ids)):
            state.if_index.add(emb, cid)

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


@app.post("/indexed-file/read", response_model=IndexedFileReadResponse)
async def indexed_file_read(req: IndexedFileReadRequest):
    """
    Busca semântica nos arquivos indexados.

    Retorna o conteúdo COMPLETO de cada arquivo que teve um chunk
    com similaridade acima do threshold, junto com o chunk que
    deu match para contexto.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query vazia")

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
        ))
        loop.run_in_executor(None, state.if_db.update_access, fid)

    return IndexedFileReadResponse(results=results[:req.top_k], query=query)


@app.get("/indexed-file/check", response_model=IndexedFileCheckResponse)
async def indexed_file_check(file_path: str):
    """
    Verifica se um arquivo está indexado e se o hash bate.

    Usado pelo local-scraping para decidir se precisa reindexar:
      - indexed=False  → arquivo nunca foi indexado
      - hash_match=True → já indexado e inalterado
      - hash_match=False → indexado mas arquivo mudou → reindexar
    """
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path vazio")

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


@app.get("/indexed-file/{file_id}")
async def indexed_file_get(file_id: int):
    """
    Retorna o conteúdo completo de um arquivo indexado pelo seu ID.
    """
    row = state.if_db.get_by_id(file_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Arquivo indexado #{file_id} não encontrado")

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


@app.delete("/indexed-file/{file_id}")
async def indexed_file_delete(file_id: int):
    """
    Remove um arquivo indexado e todos os seus chunks (DB + FAISS).
    """
    row = state.if_db.get_by_id(file_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Arquivo indexado #{file_id} não encontrado")

    # Remove FAISS vectors first
    chunk_ids = state.if_db.get_chunk_ids_by_file(file_id)
    if chunk_ids:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, state.if_index.remove_ids, set(chunk_ids))

    # Delete from DB (cascades to chunks)
    deleted = state.if_db.delete_file(file_id)

    log.info(f"Indexed file deleted: #{file_id} ({len(chunk_ids)} chunks removed)")
    return {"deleted": deleted, "file_id": file_id, "chunks_removed": len(chunk_ids)}


@app.delete("/indexed-file/path")
async def indexed_file_delete_by_path(file_path: str):
    """
    Remove um arquivo indexado pelo caminho.
    """
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path vazio")

    row = state.if_db.get_by_path(file_path)
    if row is None:
        return {"deleted": 0, "file_path": file_path, "message": "not indexed"}

    return await indexed_file_delete(row["id"])


@app.get("/indexed-file")
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


# ── GET /status ────────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    resp = {
        "long_term": {
            "memories_total":       state.lt_db.count(),
            "index_vectors":        state.lt_index.total,
            "decay_half_life_days": DECAY_HALF_LIFE_DAYS,
            "dedup_threshold":      DEDUP_THRESHOLD,
        },
        "short_term": {
            "turn_groups_total":  state.st_db.count(),
            "index_vectors":      state.st_index.total,
            "ttl_hours":          ST_TTL_HOURS,
            "cleanup_interval_s": ST_CLEANUP_INTERVAL_S,
        },
        "plan_cache": {
            "entries_total":  state.pc_db.count(),
            "index_vectors":  state.pc_index.total,
            "hit_threshold":  PC_HIT_THRESHOLD,
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


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("memory:app", host="0.0.0.0", port=3001, log_level="info")