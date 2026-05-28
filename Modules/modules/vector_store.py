"""
AVA KG-RAG — Vector Store (FAISS + SQLite)
Padrão AVA: SQLite WAL + IndexFlatIP + id_map .npy — idêntico ao memory.py.

Mudanças em relação à versão anterior (FAISS + JSON):
- Metadados migrados de JSON para SQLite (WAL, synchronous=NORMAL).
- id_map agora é um arquivo .npy com IDs de linhas do DB (igual ao MemoryIndex do memory.py).
- Deduplicação dupla: SHA-256 exato no DB + threshold semântico no FAISS.
- KGVectorDB espelha a estrutura de MemoryDB / ShortTermDB / PlanCacheDB.
- MemoryIndex reutilizado sem modificação — só os paths mudam.
"""

import faiss
import numpy as np
import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from config import FAISS_INDEX_PATH, FAISS_META_PATH, RETRIEVAL

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────

# FAISS_META_PATH vira o caminho da DB SQLite (.db).
# Para o id_map .npy, derivamos o path trocando a extensão.
_DB_PATH      = str(FAISS_META_PATH).replace(".json", ".db") if str(FAISS_META_PATH).endswith(".json") else str(FAISS_META_PATH) + ".db"
_ID_MAP_PATH  = str(FAISS_INDEX_PATH).replace(".index", "_id_map.npy")
_EMBED_DIM    = RETRIEVAL.embed_dim

# Threshold semântico para deduplicação (igual ao DEDUP_THRESHOLD do memory.py)
DEDUP_THRESHOLD: float = 0.92


# ── Estrutura de entrada ──────────────────────────────────────────────────────

@dataclass
class VectorEntry:
    id:          int           # PK do SQLite (autoincrement)
    chunk_id:    str           # SHA-256 do texto (dedup exato)
    text:        str
    source:      str           # Caminho do arquivo ou URL de origem
    chunk_index: int           # Posição no documento original
    node_ids:    List[str]     # IDs de nós do KG relacionados ao chunk


# ── Banco de dados SQLite ─────────────────────────────────────────────────────

class KGVectorDB:
    """
    Persistência de metadados dos chunks do KG-RAG.

    Espelha o padrão de MemoryDB/ShortTermDB/PlanCacheDB do memory.py:
    - sqlite3 com WAL e synchronous=NORMAL.
    - Deduplicação por SHA-256 (coluna UNIQUE).
    - Método get_by_ids() para leitura em lote após busca FAISS.
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
            CREATE TABLE IF NOT EXISTS kg_chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id    TEXT    NOT NULL UNIQUE,
                text        TEXT    NOT NULL,
                source      TEXT    NOT NULL DEFAULT '',
                chunk_index INTEGER NOT NULL DEFAULT 0,
                node_ids    TEXT    NOT NULL DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_chunk_id ON kg_chunks(chunk_id);
            CREATE INDEX IF NOT EXISTS idx_source   ON kg_chunks(source);
        """)

    # ── Escrita ──────────────────────────────────────────────────────────────

    def insert(
        self,
        chunk_id:    str,
        text:        str,
        source:      str,
        chunk_index: int,
        node_ids:    List[str],
    ) -> int:
        """
        Insere um chunk e retorna o id autoincrement.
        Lança sqlite3.IntegrityError se chunk_id já existir
        (trate externamente ou use exists_exact() antes).
        """
        import json
        cur = self._conn.execute(
            "INSERT INTO kg_chunks (chunk_id, text, source, chunk_index, node_ids) "
            "VALUES (?, ?, ?, ?, ?)",
            (chunk_id, text, source, chunk_index, json.dumps(node_ids)),
        )
        return cur.lastrowid

    # ── Leitura ──────────────────────────────────────────────────────────────

    def exists_exact(self, chunk_id: str) -> bool:
        """Deduplicação exata por SHA-256 — idêntico a MemoryDB.exists_exact()."""
        return self._conn.execute(
            "SELECT 1 FROM kg_chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone() is not None

    def get_by_ids(self, ids: List[int]) -> List[sqlite3.Row]:
        """Leitura em lote por PKs — idêntico a MemoryDB.get_by_ids()."""
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        return self._conn.execute(
            f"SELECT * FROM kg_chunks WHERE id IN ({ph})", ids
        ).fetchall()

    def get_by_node_ids(self, node_ids: List[str]) -> List["VectorEntry"]:
        """
        Recupera chunks que referenciam determinados nós do KG.
        Necessita varredura linear — aceitável para KGs de tamanho moderado.
        """
        import json
        target = set(node_ids)
        rows = self._conn.execute("SELECT * FROM kg_chunks").fetchall()
        results = []
        for row in rows:
            stored = set(json.loads(row["node_ids"]))
            if target & stored:
                results.append(_row_to_entry(row))
        return results

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM kg_chunks").fetchone()[0]


# ── Índice FAISS (padrão MemoryIndex do memory.py) ────────────────────────────

class KGMemoryIndex:
    """
    FAISS IndexFlatIP com id_map persistido em .npy.
    Estrutura idêntica ao MemoryIndex do memory.py — apenas o nome muda
    para evitar conflito de importação caso ambos os módulos coexistam.

    IndexFlatIP + vetores L2-normalizados = similaridade cosseno.
    """

    def __init__(self, index_path: str, id_map_path: str):
        self._index_path  = index_path
        self._id_map_path = id_map_path
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        if Path(index_path).exists() and Path(id_map_path).exists():
            self._index  = faiss.read_index(index_path)
            self._id_map = list(np.load(id_map_path).tolist())
            logger.info(
                "KGMemoryIndex carregado [%s] — %d vetores",
                index_path, self._index.ntotal,
            )
        else:
            self._index  = faiss.IndexFlatIP(_EMBED_DIM)
            self._id_map = []
            logger.info("Novo KGMemoryIndex criado [%s]", index_path)

    def add(self, embedding: np.ndarray, record_id: int):
        self._index.add(embedding.reshape(1, -1))
        self._id_map.append(record_id)
        self._save()

    def add_batch(self, embeddings: np.ndarray, record_ids: List[int]):
        """Inserção em lote — evita N saves individuais."""
        self._index.add(embeddings.astype(np.float32))
        self._id_map.extend(record_ids)
        self._save()

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
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
        """Retorna o score máximo do vizinho mais próximo — usado para dedup semântico."""
        results = self.search(embedding, top_k=1)
        return results[0][1] if results else 0.0

    def remove_ids(self, record_ids: set):
        """Remove vetores por record_id — reconstrói o índice sem os IDs removidos."""
        if not record_ids or self._index.ntotal == 0:
            return
        all_vectors = self._index.reconstruct_n(0, self._index.ntotal)
        new_vecs, new_map = [], []
        for vec, rid in zip(all_vectors, self._id_map):
            if rid not in record_ids:
                new_vecs.append(vec)
                new_map.append(rid)
        self._index = faiss.IndexFlatIP(_EMBED_DIM)
        if new_vecs:
            self._index.add(np.array(new_vecs, dtype=np.float32))
        self._id_map = new_map
        self._save()
        logger.info("KGMemoryIndex: %d vetores removidos", len(record_ids))

    def reset(self):
        self._index  = faiss.IndexFlatIP(_EMBED_DIM)
        self._id_map = []
        self._save()

    def _save(self):
        faiss.write_index(self._index, self._index_path)
        np.save(self._id_map_path, np.array(self._id_map, dtype=np.int64))

    @property
    def total(self) -> int:
        return self._index.ntotal


# ── Helper interno ────────────────────────────────────────────────────────────

def _row_to_entry(row: sqlite3.Row) -> VectorEntry:
    import json
    return VectorEntry(
        id          = row["id"],
        chunk_id    = row["chunk_id"],
        text        = row["text"],
        source      = row["source"],
        chunk_index = row["chunk_index"],
        node_ids    = json.loads(row["node_ids"]),
    )


# ── VectorStore público ───────────────────────────────────────────────────────

class VectorStore:
    """
    Armazenamento vetorial do KG-RAG alinhado ao padrão do memory.py.

    Backends:
    - KGVectorDB  → SQLite WAL  (metadados dos chunks)
    - KGMemoryIndex → FAISS IndexFlatIP + id_map.npy  (busca semântica)

    Deduplicação dupla (igual ao memory.py):
    1. SHA-256 exato no DB  → rejeita textos idênticos byte a byte.
    2. Similaridade cosseno no FAISS  → rejeita textos quase-duplicados
       (threshold = DEDUP_THRESHOLD, default 0.92).
    """

    def __init__(
        self,
        index_path:  str = str(FAISS_INDEX_PATH),
        db_path:     str = _DB_PATH,
        id_map_path: str = _ID_MAP_PATH,
        embed_dim:   int = None,
    ):
        global _EMBED_DIM
        if embed_dim:
            _EMBED_DIM = embed_dim

        self._db    = KGVectorDB(db_path)
        self._index = KGMemoryIndex(index_path, id_map_path)

        logger.info(
            "VectorStore iniciado — %d chunks no DB | %d vetores no índice (dim=%d)",
            self._db.count(), self._index.total, _EMBED_DIM,
        )

    # ── Escrita ───────────────────────────────────────────────────────────────

    def add(
        self,
        texts:      List[str],
        embeddings: np.ndarray,
        source:     str,
        node_ids:   Optional[List[List[str]]] = None,
    ) -> int:
        """
        Adiciona chunks ao índice com deduplicação dupla.

        Parâmetros
        ----------
        texts      : lista de textos dos chunks.
        embeddings : array (N, dim) L2-normalizado — saída do EmbeddingEngine.
        source     : caminho/URL de origem dos chunks.
        node_ids   : lista de listas de IDs de nós do KG por chunk (opcional).

        Retorna
        -------
        Número de novos vetores efetivamente inseridos.
        """
        if node_ids is None:
            node_ids = [[] for _ in texts]

        inserted       = 0
        new_vectors:   List[np.ndarray] = []
        new_record_ids: List[int]        = []

        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            chunk_id = hashlib.sha256(text.encode()).hexdigest()

            # 1) Dedup exato — SHA-256 no DB
            if self._db.exists_exact(chunk_id):
                logger.debug("Dedup exato ignorado: chunk_id=%s", chunk_id[:12])
                continue

            # 2) Dedup semântico — similaridade cosseno no FAISS
            max_sim = self._index.search_similar(emb)
            if max_sim >= DEDUP_THRESHOLD:
                logger.debug("Dedup semântico ignorado: sim=%.3f chunk=%.40s", max_sim, text)
                continue

            nids      = node_ids[i] if i < len(node_ids) else []
            record_id = self._db.insert(chunk_id, text, source, i, nids)

            new_vectors.append(emb)
            new_record_ids.append(record_id)
            inserted += 1

        if new_vectors:
            batch = np.vstack(new_vectors).astype(np.float32)
            self._index.add_batch(batch, new_record_ids)

        logger.debug(
            "VectorStore.add: +%d inseridos / %d total (source=%s)",
            inserted, self._index.total, source,
        )
        return inserted

    # ── Leitura ───────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k:           int = None,
        min_score:       float = 0.0,
    ) -> List[Tuple[VectorEntry, float]]:
        """
        Busca os top_k chunks mais similares.

        Retorna lista de (VectorEntry, score) ordenada por relevância descendente.
        min_score filtra resultados abaixo do threshold (opcional).
        """
        top_k = top_k or RETRIEVAL.top_k_dense

        raw = self._index.search(query_embedding.reshape(1, -1)[0], top_k)
        if not raw:
            return []

        ids_filtered = [rid for rid, score in raw if score >= min_score]
        score_map    = {rid: score for rid, score in raw}

        rows    = self._db.get_by_ids(ids_filtered)
        results = [
            (_row_to_entry(row), round(score_map[row["id"]], 4))
            for row in rows
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_by_node_ids(self, node_ids: List[str]) -> List[VectorEntry]:
        """Recupera entries que referenciam determinados nós do KG."""
        return self._db.get_by_node_ids(node_ids)

    def remove_by_source(self, source: str) -> int:
        """
        Remove todos os chunks de uma fonte específica.
        Útil para re-indexar um documento sem duplicar entradas.
        """
        rows = self._db._conn.execute(
            "SELECT id FROM kg_chunks WHERE source = ?", (source,)
        ).fetchall()
        ids_to_remove = {row["id"] for row in rows}

        if not ids_to_remove:
            return 0

        self._db._conn.execute("DELETE FROM kg_chunks WHERE source = ?", (source,))
        self._index.remove_ids(ids_to_remove)
        logger.info("VectorStore: %d chunks removidos (source=%s)", len(ids_to_remove), source)
        return len(ids_to_remove)

    # ── Status ────────────────────────────────────────────────────────────────

    @property
    def total(self) -> int:
        return self._index.total

    def status(self) -> dict:
        return {
            "chunks_in_db":     self._db.count(),
            "vectors_in_index": self._index.total,
            "embed_dim":        _EMBED_DIM,
            "dedup_threshold":  DEDUP_THRESHOLD,
        }