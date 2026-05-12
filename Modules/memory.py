from __future__ import annotations

import os
import time
import json
import sqlite3
import hashlib
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import faiss
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Configuração ───────────────────────────────────────────────────────────────

EMBED_MODEL_PATH  = "./Models/multilingual-e5-small/multilingual-e5-small.onnx"       # paraphrase-multilingual-MiniLM
TOKENIZER_PATH    = "./Models/multilingual-e5-small/tokenizer.json"        # tokenizer do modelo de embedding
DB_PATH           = "./memory/ava_memory.db"
FAISS_INDEX_PATH  = "./memory/ava_memory.index"
FAISS_ID_MAP_PATH = "./memory/ava_id_map.npy"

EMBED_DIM              = 384      # dimensão do MiniLM-L12
READ_MIN_SCORE  = 0.83   # abaixo disso é ruído
DEDUP_THRESHOLD = 0.92   # cosine mínimo pra retornar na leitura
TOP_K_READ             = 5       # máximo de memórias retornadas por busca
DECAY_HALF_LIFE_DAYS   = 90      # memória perde 50% de confiança em 90 dias sem acesso
DECAY_JOB_INTERVAL_S   = 3600    # roda decay a cada 1 hora

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ava.memory")


# ── Modelos de request/response ────────────────────────────────────────────────

class WriteRequest(BaseModel):
    text: str                         # texto a memorizar (já filtrado pelo zero-shot)
    source: str = "chat"              # origem: chat | user_explicit | system
    confidence: float = 1.0          # confiança inicial (0.0–1.0)

class ReadRequest(BaseModel):
    query: str                        # pergunta ou keywords extraídas
    top_k: int = TOP_K_READ
    min_score: float = READ_MIN_SCORE

class WriteResponse(BaseModel):
    stored: bool
    reason: str                       # "ok" | "duplicate" | "too_short"
    memory_id: Optional[int] = None

class MemoryEntry(BaseModel):
    id: int
    text: str
    score: float                      # similaridade com a query
    confidence: float                 # confiança atual (decaída)
    created_at: float
    access_count: int

class ReadResponse(BaseModel):
    results: list[MemoryEntry]
    query: str


# ── Tokenizer minimalista (sem transformers) ────────────────────────────────────

class FastTokenizer:
    """
    Tokenizer BPE leve carregado do tokenizer.json do HuggingFace.
    Evita dependência do transformers inteiro só pra tokenizar.
    Alternativa: usar tokenizers (Rust) que é muito mais leve.
    """

    def __init__(self, path: str):
        try:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(path)
            self._tok.enable_padding(length=128)
            self._tok.enable_truncation(max_length=128)
            self._backend = "tokenizers"
        except ImportError:
            # fallback: wordpiece simples (qualidade menor)
            self._backend = "simple"
            log.warning("tokenizers não encontrado — usando tokenizer simples. "
                        "pip install tokenizers para melhor qualidade.")

    def encode_batch(self, texts: list[str]) -> dict[str, np.ndarray]:
        if self._backend == "tokenizers":
            encoded = self._tok.encode_batch(texts)
            return {
                "input_ids":      np.array([e.ids          for e in encoded], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
                "token_type_ids": np.zeros((len(texts), 128), dtype=np.int64),
            }
        # fallback — apenas split por espaço, sem BPE
        max_len = 128
        ids, masks = [], []
        for t in texts:
            tokens = t.lower().split()[:max_len - 2]
            pad = max_len - len(tokens) - 2
            ids.append([101] + [hash(w) % 30000 + 100 for w in tokens] + [102] + [0] * pad)
            masks.append([1] * (len(tokens) + 2) + [0] * pad)
        return {
            "input_ids":      np.array(ids,   dtype=np.int64),
            "attention_mask": np.array(masks, dtype=np.int64),
            "token_type_ids": np.zeros((len(texts), max_len), dtype=np.int64),
        }


# ── Engine de embeddings ONNX ──────────────────────────────────────────────────

class EmbeddingEngine:
    def __init__(self, model_path: str, tokenizer_path: str):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Tenta CUDA, cai pro CPU se não disponível
        providers = ["CUDAExecutionProvider", "AzureExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(model_path, opts, providers=providers)
        self._tokenizer = FastTokenizer(tokenizer_path)

        ep = self._session.get_providers()[0]
        log.info(f"EmbeddingEngine carregado — provider: {ep}")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Retorna embeddings L2-normalizados shape (N, EMBED_DIM)."""
        if not texts:
            return np.empty((0, EMBED_DIM), dtype=np.float32)

        inputs = self._tokenizer.encode_batch(texts)

        # Inferência ONNX
        output = self._session.run(None, {
            "input_ids":      inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs["token_type_ids"],
        })

        # Mean pooling sobre os tokens
        token_embeddings = output[0]                           # (N, seq, dim)
        mask = inputs["attention_mask"][:, :, np.newaxis]     # (N, seq, 1)
        summed = np.sum(token_embeddings * mask, axis=1)       # (N, dim)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = (summed / counts).astype(np.float32)     # (N, dim)

        # L2 normalização — necessária para cosine via inner product no FAISS
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)
        return embeddings

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# ── Banco de dados SQLite ──────────────────────────────────────────────────────

class MemoryDB:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            path,
            check_same_thread=False,
            isolation_level=None   # ← autocommit, sem transações implícitas
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                text         TEXT    NOT NULL,
                text_hash    TEXT    NOT NULL UNIQUE,   -- evita duplicata exata
                source       TEXT    NOT NULL DEFAULT 'chat',
                confidence   REAL    NOT NULL DEFAULT 1.0,
                created_at   REAL    NOT NULL,
                last_accessed REAL   NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_confidence ON memories(confidence);
            CREATE INDEX IF NOT EXISTS idx_last_accessed ON memories(last_accessed);
        """)
        self._conn.commit()

    def insert(self, text: str, source: str, confidence: float) -> int:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO memories (text, text_hash, source, confidence, created_at, last_accessed) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (text, text_hash, source, confidence, now, now)
        )
        self._conn.commit()
        return cur.lastrowid

    def exists_exact(self, text: str) -> bool:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        row = self._conn.execute(
            "SELECT 1 FROM memories WHERE text_hash = ?", (text_hash,)
        ).fetchone()
        return row is not None

    def get_by_ids(self, ids: list[int]) -> list[sqlite3.Row]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        return self._conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", ids
        ).fetchall()

    def update_access(self, memory_id: int):
        try:
            self._conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (time.time(), memory_id)
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            self._conn.rollback()

    def apply_decay(self, half_life_days: float):
        """
        Decai confidence de memórias não acessadas recentemente.
        Fórmula: confidence *= 0.5 ^ (dias_sem_acesso / half_life)
        """
        now = time.time()
        rows = self._conn.execute(
            "SELECT id, confidence, last_accessed FROM memories WHERE confidence > 0.01"
        ).fetchall()

        updates = []
        for row in rows:
            days_idle = (now - row["last_accessed"]) / 86400.0
            decay_factor = 0.5 ** (days_idle / half_life_days)
            new_conf = row["confidence"] * decay_factor
            updates.append((new_conf, row["id"]))

        if updates:
            self._conn.executemany(
                "UPDATE memories SET confidence = ? WHERE id = ?", updates
            )
            # Remove memórias com confiança muito baixa (lixo)
            self._conn.execute("DELETE FROM memories WHERE confidence < 0.01")
            self._conn.commit()
            log.info(f"Decay aplicado em {len(updates)} memórias")

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]


# ── Índice FAISS ───────────────────────────────────────────────────────────────

class MemoryIndex:
    """
    FAISS IndexFlatIP (inner product) para embeddings L2-normalizados.
    IP em vetores normalizados = cosine similarity.
    id_map mapeia posição FAISS → id SQLite.
    """

    def __init__(self, index_path: str, id_map_path: str):
        self._index_path  = index_path
        self._id_map_path = id_map_path
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        if Path(index_path).exists() and Path(id_map_path).exists():
            self._index  = faiss.read_index(index_path)
            self._id_map = list(np.load(id_map_path).tolist())
            log.info(f"Índice FAISS carregado — {self._index.ntotal} vetores")
        else:
            self._index  = faiss.IndexFlatIP(EMBED_DIM)
            self._id_map = []
            log.info("Novo índice FAISS criado")

    def add(self, embedding: np.ndarray, memory_id: int):
        self._index.add(embedding.reshape(1, -1))
        self._id_map.append(memory_id)
        self._save()

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Retorna lista de (memory_id, cosine_score)."""
        if self._index.ntotal == 0:
            return []
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_embedding.reshape(1, -1), k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._id_map):
                results.append((self._id_map[idx], float(score)))
        return results

    def search_similar(self, embedding: np.ndarray, threshold: float) -> float:
        """Retorna o maior score de similaridade encontrado (para deduplicação)."""
        if self._index.ntotal == 0:
            return 0.0
        results = self.search(embedding, top_k=1)
        return results[0][1] if results else 0.0

    def _save(self):
        faiss.write_index(self._index, self._index_path)
        np.save(self._id_map_path, np.array(self._id_map, dtype=np.int64))

    @property
    def total(self) -> int:
        return self._index.ntotal


# ── Estado global da aplicação ─────────────────────────────────────────────────

@dataclass
class AppState:
    embed_engine: EmbeddingEngine = field(default=None)
    db:           MemoryDB        = field(default=None)
    index:        MemoryIndex     = field(default=None)
    decay_task:   asyncio.Task    = field(default=None)


state = AppState()


# ── Job de decay em background ─────────────────────────────────────────────────

async def decay_job():
    """Roda periodicamente para decair confiança de memórias antigas."""
    while True:
        await asyncio.sleep(DECAY_JOB_INTERVAL_S)
        try:
            state.db.apply_decay(DECAY_HALF_LIFE_DAYS)
        except Exception as e:
            log.error(f"Erro no decay job: {e}")


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Iniciando AVA Memory API...")

    # Verifica modelos
    if not Path(EMBED_MODEL_PATH).exists():
        raise RuntimeError(f"Modelo de embedding não encontrado: {EMBED_MODEL_PATH}")
    if not Path(TOKENIZER_PATH).exists():
        raise RuntimeError(f"Tokenizer não encontrado: {TOKENIZER_PATH}")

    state.embed_engine = EmbeddingEngine(EMBED_MODEL_PATH, TOKENIZER_PATH)
    state.db           = MemoryDB(DB_PATH)
    state.index        = MemoryIndex(FAISS_INDEX_PATH, FAISS_ID_MAP_PATH)
    state.decay_task   = asyncio.create_task(decay_job())

    log.info(f"Pronto — {state.db.count()} memórias indexadas")
    yield

    # Shutdown
    state.decay_task.cancel()
    log.info("AVA Memory API encerrada")


# ── App FastAPI ────────────────────────────────────────────────────────────────

app = FastAPI(title="AVA Memory API", lifespan=lifespan)


# ── POST /write ────────────────────────────────────────────────────────────────

@app.post("/write", response_model=WriteResponse)
async def write_memory(req: WriteRequest):
    text = req.text.strip()

    # Rejeita textos muito curtos
    if len(text) < 10:
        return WriteResponse(stored=False, reason="too_short")

    # Deduplicação exata (hash — O(1))
    if state.db.exists_exact(text):
        return WriteResponse(stored=False, reason="duplicate_exact")

    # Deduplicação semântica (embedding + FAISS cosine)
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(
        None, state.embed_engine.embed_one, text
    )

    max_similarity = state.index.search_similar(embedding, DEDUP_THRESHOLD)
    if max_similarity >= DEDUP_THRESHOLD:
        log.debug(f"Duplicata semântica descartada (score={max_similarity:.3f}): {text[:60]}")
        return WriteResponse(stored=False, reason=f"duplicate_semantic:{max_similarity:.3f}")

    # Grava no SQLite
    memory_id = state.db.insert(text, req.source, req.confidence)

    # Indexa no FAISS
    state.index.add(embedding, memory_id)

    log.info(f"Memória #{memory_id} gravada: {text[:60]}...")
    return WriteResponse(stored=True, reason="ok", memory_id=memory_id)


# ── POST /read ─────────────────────────────────────────────────────────────────

@app.post("/read", response_model=ReadResponse)
async def read_memory(req: ReadRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query vazia")

    if state.index.total == 0:
        return ReadResponse(results=[], query=query)

    # Embedding da query
    loop = asyncio.get_event_loop()
    query_embedding = await loop.run_in_executor(
        None, state.embed_engine.embed_one, query
    )

    # Busca FAISS
    raw_results = state.index.search(query_embedding, req.top_k * 2)  # busca mais, filtra depois

    # Filtra por score mínimo e confiança
    filtered_ids   = [mid for mid, score in raw_results if score >= req.min_score]
    score_map      = {mid: score for mid, score in raw_results}

    if not filtered_ids:
        return ReadResponse(results=[], query=query)

    # Busca metadados no SQLite
    rows = state.db.get_by_ids(filtered_ids)

    # Monta resposta e atualiza access_count de forma assíncrona (fire-and-forget)
    results = []
    for row in rows:
        score = score_map.get(row["id"], 0.0)
        results.append(MemoryEntry(
            id           = row["id"],
            text         = row["text"],
            score        = round(score, 4),
            confidence   = round(row["confidence"], 4),
            created_at   = row["created_at"],
            access_count = row["access_count"],
        ))
        # Atualiza acesso em background — não bloqueia a resposta
        asyncio.get_event_loop().run_in_executor(
            None, state.db.update_access, row["id"]
        )

    # Ordena por score × confidence (memórias confiáveis e relevantes primeiro)
    results.sort(key=lambda r: r.score * r.confidence, reverse=True)
    results = results[:req.top_k]

    return ReadResponse(results=results, query=query)


# ── GET /status ────────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    return {
        "memories_total": state.db.count(),
        "index_vectors":  state.index.total,
        "db_path":        DB_PATH,
        "dedup_threshold": DEDUP_THRESHOLD,
        "decay_half_life_days": DECAY_HALF_LIFE_DAYS,
    }


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("memory:app", host="0.0.0.0", port=3001, log_level="info")