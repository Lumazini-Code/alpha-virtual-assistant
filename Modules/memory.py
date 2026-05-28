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
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── MODIFIED: Import VectorStore from vector_store (graceful fallback) ────────
try:
    from modules.vector_store import VectorStore, VectorEntry as VSVectorEntry
    _VS_AVAILABLE = True
except ImportError as _imp_err:
    _VS_AVAILABLE = False
    VectorStore = None          # type: ignore[assignment,misc]
    VSVectorEntry = None        # type: ignore[assignment,misc]
    # Will log after logging is configured below

# ── Configuração ───────────────────────────────────────────────────────────────

EMBED_MODEL_PATH  = "./Models/multilingual-e5-small/multilingual-e5-small.onnx"
TOKENIZER_PATH    = "./Models/multilingual-e5-small/tokenizer.json"

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

# ── NEW: Knowledge (Vector Store / KG-RAG) paths ─────────────────────────────
# These must match the paths used by vector_store.py / config.
# If config is available, the actual paths are resolved at init time.
VS_DB_PATH           = "./memory/ava_kg_chunks.db"
VS_FAISS_INDEX_PATH  = "./memory/ava_kg_vectors.index"
VS_FAISS_ID_MAP_PATH = "./memory/ava_kg_vectors_id_map.npy"
VS_MIN_SCORE         = 0.70   # Lower floor for knowledge retrieval

EMBED_DIM            = 384
READ_MIN_SCORE       = 0.83
DEDUP_THRESHOLD      = 0.92
TOP_K_READ           = 5
DECAY_HALF_LIFE_DAYS = 90
DECAY_JOB_INTERVAL_S = 3600

ST_TTL_HOURS          = 24.0
ST_CLEANUP_INTERVAL_S = 1800

# Cache de planos: threshold de similaridade para considerar hit
PC_HIT_THRESHOLD = 0.92

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ava.memory")

# ── NEW: Log VectorStore availability ─────────────────────────────────────────
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
    strategy:   str = "auto"   # "auto" | "expanded" | "dual" | "none"

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
    # ── MODIFIED: Added "knowledge" to memory_type literal ────────────────────
    memory_type:  Literal["long_term", "short_term", "knowledge"]
    session_id:   Optional[str]        = None
    turns:        Optional[list[Turn]] = None
    # ── NEW: source field — populated for long_term and knowledge entries ─────
    source:       Optional[str]        = None

class ReadResponse(BaseModel):
    results:  list[MemoryEntry]
    query:    str
    strategy: str


# ── Modelos de request/response — cache de planos ─────────────────────────────

class PlanCacheGetRequest(BaseModel):
    query:     str
    threshold: float = PC_HIT_THRESHOLD

class PlanCachePutRequest(BaseModel):
    query: str
    plan:  dict   # PlanResponse serializado pelo CoT — sem schema fixo aqui

class PlanCacheGetResponse(BaseModel):
    hit:        bool
    plan:       Optional[dict] = None
    score:      Optional[float] = None
    cache_id:   Optional[int]  = None
    hit_count:  Optional[int]  = None

class PlanCacheDeleteResponse(BaseModel):
    deleted: int


# ── Tokenizer ──────────────────────────────────────────────────────────────────

class FastTokenizer:
    def __init__(self, path: str):
        try:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(path)
            self._tok.enable_padding(length=128)
            self._tok.enable_truncation(max_length=128)
            self._backend = "tokenizers"
        except ImportError:
            self._backend = "simple"
            log.warning("tokenizers não encontrado — usando tokenizer simples.")

    def encode_batch(self, texts: list[str]) -> dict[str, np.ndarray]:
        if self._backend == "tokenizers":
            encoded = self._tok.encode_batch(texts)
            return {
                "input_ids":      np.array([e.ids            for e in encoded], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
                "token_type_ids": np.zeros((len(texts), 128), dtype=np.int64),
            }
        max_len = 128
        ids, masks = [], []
        for t in texts:
            tokens = t.lower().split()[:max_len - 2]
            pad    = max_len - len(tokens) - 2
            ids.append([101] + [hash(w) % 30000 + 100 for w in tokens] + [102] + [0] * pad)
            masks.append([1] * (len(tokens) + 2) + [0] * pad)
        return {
            "input_ids":      np.array(ids,   dtype=np.int64),
            "attention_mask": np.array(masks, dtype=np.int64),
            "token_type_ids": np.zeros((len(texts), max_len), dtype=np.int64),
        }


# ── Engine de embeddings ───────────────────────────────────────────────────────

class EmbeddingEngine:
    def __init__(self, model_path: str, tokenizer_path: str):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads     = 4
        opts.inter_op_num_threads     = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session   = ort.InferenceSession(model_path, opts, providers=providers)
        self._tokenizer = FastTokenizer(tokenizer_path)
        ep = self._session.get_providers()[0]
        log.info(f"EmbeddingEngine carregado — provider: {ep}")

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        inputs = self._tokenizer.encode_batch(texts)
        output = self._session.run(None, {
            "input_ids":      inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs["token_type_ids"],
        })
        token_embeddings = output[0]
        mask    = inputs["attention_mask"][:, :, np.newaxis]
        summed  = np.sum(token_embeddings * mask, axis=1)
        counts  = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = (summed / counts).astype(np.float32)
        norms   = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, a_min=1e-9, a_max=None)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def embed_batch_two(self, text_a: str, text_b: str) -> tuple[np.ndarray, np.ndarray]:
        """Dois textos numa única chamada ONNX — evita overhead duplo."""
        results = self.embed([text_a, text_b])
        return results[0], results[1]


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
    """
    Armazena pares (query_text → plan_json) para o CoT generator.

    Diferenças intencionais em relação à MemoryDB:
    - Sem decay de confiança: um plano válido não fica "menos válido" com o tempo.
    - Sem deduplicação na gravação: o CoT controla quando cachear.
    - hit_count rastreado para análise de utilização.
    - Invalidação apenas por DELETE explícito (módulos mudaram, etc.).
    """

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
        """Zera o índice completamente — usado para invalidar o cache de planos."""
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
    # Memória conversacional
    lt_db:        MemoryDB        = field(default=None)
    lt_index:     MemoryIndex     = field(default=None)
    st_db:        ShortTermDB     = field(default=None)
    st_index:     MemoryIndex     = field(default=None)
    # Cache de planos CoT
    pc_db:        PlanCacheDB     = field(default=None)
    pc_index:     MemoryIndex     = field(default=None)
    # ── NEW: VectorStore (knowledge / KG-RAG) ─────────────────────────────────
    vs:           Optional[VectorStore] = field(default=None)
    # Tasks
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
    """
    Resolve VectorStore paths — tries config first, falls back to local constants.
    Ensures memory.py reads the same data written by vector_store.py.
    """
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
    for path in (EMBED_MODEL_PATH, TOKENIZER_PATH):
        if not Path(path).exists():
            raise RuntimeError(f"Arquivo não encontrado: {path}")

    state.embed_engine = EmbeddingEngine(EMBED_MODEL_PATH, TOKENIZER_PATH)

    state.lt_db    = MemoryDB(DB_PATH)
    state.lt_index = MemoryIndex(FAISS_INDEX_PATH, FAISS_ID_MAP_PATH)

    state.st_db    = ShortTermDB(ST_DB_PATH)
    state.st_index = MemoryIndex(ST_FAISS_INDEX_PATH, ST_FAISS_ID_MAP_PATH)

    state.pc_db    = PlanCacheDB(PC_DB_PATH)
    state.pc_index = MemoryIndex(PC_FAISS_INDEX_PATH, PC_FAISS_ID_MAP_PATH)

    # ── NEW: Initialize VectorStore for knowledge retrieval ───────────────────
    if _VS_AVAILABLE:
        try:
            vs_idx, vs_db, vs_idmap = _resolve_vs_paths()
            state.vs = VectorStore(
                index_path  = vs_idx,
                db_path     = vs_db,
                id_map_path = vs_idmap,
                embed_dim   = EMBED_DIM,
            )
            log.info(
                f"VectorStore integrado — {state.vs.total} chunks de conhecimento"
            )
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
        f"{state.vs.total if state.vs else 0} chunks de conhecimento"
    )
    yield

    state.decay_task.cancel()
    state.cleanup_task.cancel()
    log.info("AVA Memory API encerrada")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="AVA Memory API", lifespan=lifespan)


# ── Helpers de busca contextual ────────────────────────────────────────────────

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
    loop = asyncio.get_event_loop()
    emb  = await loop.run_in_executor(None, state.embed_engine.embed_one, expanded)
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
    loop = asyncio.get_event_loop()
    emb_query, emb_ctx = await loop.run_in_executor(
        None, state.embed_engine.embed_batch_two, query, context_block
    )
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
) -> list[MemoryEntry]:
    ids_filtered = [mid for mid, score in lt_raw if score >= min_score]
    score_map    = {mid: score for mid, score in lt_raw}
    if not ids_filtered:
        return []
    entries = []
    for row in state.lt_db.get_by_ids(ids_filtered):
        entries.append(MemoryEntry(
            id           = row["id"],
            text         = row["text"],
            score        = round(score_map[row["id"]], 4),
            confidence   = round(row["confidence"], 4),
            created_at   = row["created_at"],
            access_count = row["access_count"],
            memory_type  = "long_term",
            source       = row["source"],            # ── MODIFIED: expose source
        ))
        loop.run_in_executor(None, state.lt_db.update_access, row["id"])
    return entries


def _build_st_entries(
    st_raw: list[tuple[int, float]],
    min_score: float,
    loop: asyncio.AbstractEventLoop,
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
        text_repr  = " | ".join(f"[{t.role}] {t.content[:120]}" for t in turns)
        entries.append(MemoryEntry(
            id           = row["id"],
            text         = text_repr,
            score        = round(score, 4),
            confidence   = 1.0,
            created_at   = row["created_at"],
            access_count = row["access_count"],
            memory_type  = "short_term",
            session_id   = row["session_id"],
        ))
        loop.run_in_executor(None, state.st_db.update_access, row["id"])
    return entries


# ── NEW: Build knowledge entries from VectorStore results ─────────────────────

def _build_vs_entries(
    vs_results: list,
    min_score: float,
) -> list[MemoryEntry]:
    """
    Converte resultados do VectorStore em MemoryEntry com memory_type="knowledge".

    VectorStore.search() retorna List[Tuple[VectorEntry, float]].
    A filtragem por min_score já é feita dentro de VectorStore.search(),
    mas aplicamos novamente como salvaguarda.
    """
    if not vs_results:
        return []
    entries = []
    for ventry, score in vs_results:
        if score < min_score:
            continue
        entries.append(MemoryEntry(
            id           = ventry.id,
            text         = ventry.text,
            score        = round(score, 4),
            confidence   = 1.0,              # Knowledge entries have full confidence
            created_at   = 0.0,              # Not tracked by VectorStore
            access_count = 0,                # Not tracked by VectorStore
            memory_type  = "knowledge",
            source       = ventry.source,     # Document source path/URL
        ))
    return entries


# ── POST /write ────────────────────────────────────────────────────────────────

@app.post("/write", response_model=WriteResponse)
async def write_memory(req: WriteRequest):
    text = req.text.strip()
    if len(text) < 10:
        return WriteResponse(stored=False, reason="too_short")
    if state.lt_db.exists_exact(text):
        return WriteResponse(stored=False, reason="duplicate_exact")

    loop      = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(None, state.embed_engine.embed_one, text)

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
    loop       = asyncio.get_event_loop()
    embedding  = await loop.run_in_executor(None, state.embed_engine.embed_one, embed_text)

    max_sim = state.st_index.search_similar(embedding)
    if max_sim >= DEDUP_THRESHOLD:
        return WriteSTResponse(stored=False, reason=f"duplicate_semantic:{max_sim:.3f}")

    group_id = state.st_db.insert(req.session_id, req.turns, embed_text)
    state.st_index.add(embedding, group_id)
    log.info(f"ST #{group_id} gravado — session={req.session_id} turnos={len(req.turns)}: {embed_text[:80]}")
    return WriteSTResponse(stored=True, reason="ok", turn_ids=[group_id])


# ── POST /read ─────────────────────────────────────────────────────────────────
# ── POST /read ─────────────────────────────────────────────────────────────────

@app.post("/read", response_model=ReadResponse)
async def read_memory(req: ReadRequest):
    """
    Busca nas memórias de longo prazo, curto prazo E na base de conhecimento
    (VectorStore / KG-RAG) com suporte a busca contextual.

    session_id ausente  → busca simples.
    session_id presente → busca contextual com estratégia automática ou forçada:
      "auto"     → classifica por comprimento/stop-words → expanded ou dual
      "expanded" → um embed (contexto + query)
      "dual"     → dois embeds separados com fusão ponderada de scores
      "none"     → ignora session_id, busca simples

    A busca na base de conhecimento (VectorStore) é sempre feita com o embedding
    da query original — o contexto conversacional não é aplicado ao VS, pois o
    conhecimento não é conversacional.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query vazia")

    loop = asyncio.get_event_loop()
    effective_strategy = "none"

    # ── Always compute base query embedding (used for VS + simple search) ─────
    query_emb = await loop.run_in_executor(None, state.embed_engine.embed_one, query)

    # ── Start VS search in background (parallel with LT/ST search) ────────────
    # FIX: use ensure_future() — run_in_executor returns a Future, not a coroutine.
    #      asyncio.create_task() only accepts coroutines.
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

    # ── Search LT and ST memories (existing logic) ────────────────────────────
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

    # ── Collect VS results ────────────────────────────────────────────────────
    vs_results = []
    if vs_future is not None:
        try:
            vs_results = await vs_future
        except Exception as e:
            log.error(f"VectorStore search failed: {e}")
            vs_results = []

    # ── Build all entry lists ─────────────────────────────────────────────────
    results: list[MemoryEntry] = (
        _build_lt_entries(lt_raw, req.min_score, loop) +
        _build_st_entries(st_raw, req.min_score, loop) +
        _build_vs_entries(vs_results, req.min_score)
    )
    results.sort(key=lambda r: r.score * r.confidence, reverse=True)
    return ReadResponse(results=results[:req.top_k], query=query, strategy=effective_strategy)

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
# Cache de planos CoT — /cache/*
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/cache/get", response_model=PlanCacheGetResponse)
async def cache_get(req: PlanCacheGetRequest):
    """
    Busca um plano cacheado semanticamente similar à query.

    Fluxo no CoT:
      1. Chama /cache/get antes de inferir.
      2. Se hit=True, usa o plano diretamente (latência ~5ms vs ~800ms).
      3. Se hit=False, infere e chama /cache/put com o plano gerado.

    O threshold padrão é PC_HIT_THRESHOLD (0.92). O CoT pode passar um valor
    mais conservador (ex: 0.95) para domínios onde erros de plano são custosos.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query vazia")

    loop      = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(None, state.embed_engine.embed_one, query)

    results = await loop.run_in_executor(None, state.pc_index.search, embedding, 1)

    if not results:
        return PlanCacheGetResponse(hit=False)

    cache_id, score = results[0]
    if score < req.threshold:
        log.debug(f"/cache/get miss — score={score:.3f} < threshold={req.threshold}")
        return PlanCacheGetResponse(hit=False)

    row = state.pc_db.get_by_id(cache_id)
    if row is None:
        # Índice e DB dessincronizados — silenciosamente retorna miss
        log.warning(f"/cache/get: id={cache_id} no índice mas ausente no DB")
        return PlanCacheGetResponse(hit=False)

    # Atualiza hit_count em background sem bloquear a resposta
    loop.run_in_executor(None, state.pc_db.update_hit, cache_id)

    log.info(f"/cache/get HIT — id={cache_id} score={score:.3f} query='{query[:60]}'")
    return PlanCacheGetResponse(
        hit       = True,
        plan      = json.loads(row["plan_json"]),
        score     = round(score, 4),
        cache_id  = cache_id,
        hit_count = row["hit_count"] + 1,
    )


@app.post("/cache/put")
async def cache_put(req: PlanCachePutRequest):
    """
    Grava um novo par (query → plano) no cache.

    Não faz deduplicação interna — o CoT já verificou com /cache/get antes
    de inferir, então chegou aqui porque não havia hit. Gravar duplicatas não
    é catastrófico (aumenta o índice), mas o CoT deve evitar chamadas
    desnecessárias a este endpoint.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query vazia")
    if not req.plan:
        raise HTTPException(status_code=400, detail="plano vazio")

    loop      = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(None, state.embed_engine.embed_one, query)

    cache_id = state.pc_db.insert(query, req.plan)
    state.pc_index.add(embedding, cache_id)

    log.info(f"/cache/put #{cache_id} — query='{query[:60]}'")
    return {"stored": True, "cache_id": cache_id}


@app.delete("/cache", response_model=PlanCacheDeleteResponse)
async def cache_clear_all():
    """
    Invalida todo o cache de planos.

    Usar quando:
    - Módulos do AVA foram adicionados/removidos (planos existentes referenciam
      executores que podem não existir mais).
    - System prompt do CoT foi alterado significativamente.
    - Detecção de planos incorretos em produção.
    """
    deleted = state.pc_db.delete_all()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, state.pc_index.reset)
    log.info(f"/cache DELETE ALL — {deleted} entradas removidas")
    return PlanCacheDeleteResponse(deleted=deleted)


@app.delete("/cache/{cache_id}", response_model=PlanCacheDeleteResponse)
async def cache_delete_one(cache_id: int):
    """Remove uma entrada específica do cache pelo ID retornado em /cache/get."""
    deleted = state.pc_db.delete_by_id(cache_id)
    if deleted:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, state.pc_index.remove_ids, {cache_id})
    return PlanCacheDeleteResponse(deleted=deleted)


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
    }
    # ── NEW: Include VectorStore / knowledge status ───────────────────────────
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