from __future__ import annotations

"""
memory_vector.py — Índice vetorial LanceDB + chunking de texto.

LanceStore (conexão única, uma tabela por dimensão), MemoryIndex (visão de
um store com a interface do antigo índice FAISS + migração one-shot) e
_chunk_text (chunking sobreposto com metadados de posição).
"""

import threading
from pathlib import Path
from typing import Optional

import lancedb
import numpy as np
import pyarrow as pa

from memory_config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_DIM,
    IF_MAX_CHUNKS,
    LANCE_TABLE_PREFIX,
    log,
)

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


class LanceStore:
    """Conexão única com o LanceDB + uma tabela por dimensão de embedding.

    Schema de cada tabela `vectors_<dim>`:
        memory_type : string          — qual store (long_term, short_term, tool…)
        id          : int64           — id do registro DENTRO do store
        vector      : fixed_size_list<float32>[dim]
    A chave lógica é (memory_type, id): os ids vêm de sequências SQLite
    independentes por store, então repetem entre stores."""

    def __init__(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self._path   = path
        self._db     = lancedb.connect(path)
        self.lock    = threading.RLock()
        self._tables: dict[int, object] = {}
        self._writes: dict[int, int]    = {}

    def schema(self, dim: int) -> "pa.Schema":
        return pa.schema([
            pa.field("memory_type", pa.string()),
            pa.field("id", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ])

    def table(self, dim: int):
        with self.lock:
            tbl = self._tables.get(dim)
            if tbl is None:
                tbl = self._db.create_table(
                    f"{LANCE_TABLE_PREFIX}_{dim}", schema=self.schema(dim), exist_ok=True
                )
                self._tables[dim] = tbl
            return tbl

    def note_write(self, dim: int, n: int = 1):
        with self.lock:
            self._writes[dim] = self._writes.get(dim, 0) + n

    def compact(self, dim: int):
        with self.lock:
            if not self._writes.get(dim):
                return
            try:
                self.table(dim).optimize()
            except Exception as e:
                log.error(f"LanceDB: falha ao compactar vectors_{dim}: {e}")
                return
            self._writes[dim] = 0


class MemoryIndex:
    """
    Visão de UM store (`memory_type`) sobre a tabela LanceDB da sua dimensão.
    Mantém a mesma interface do antigo índice FAISS (add, add_batch,
    remove_ids, reset, search, search_batch, search_similar, search_subset,
    flush, total), então o resto do arquivo não muda — e ganha:

      * sem .npy de id-map e sem IndexIDMap2: o id vive na própria linha;
      * remove_ids = DELETE com filtro (sem reconstruir nada);
      * search_subset com filtro nativo `id IN (...)` (antes: reconstruct em
        loop Python);
      * get_vectors(ids), usado pelo fallback de cosine do MMR.

    Métrica: cosine (score = 1 - distância cosine), equivalente ao produto
    interno do FAISS para vetores L2-normalizados — e correta se algum não for.

    Migração: se `legacy_index_path` existir e o store ainda estiver vazio no
    LanceDB, o índice FAISS antigo (IndexIDMap2, ou IndexFlatIP + .npy) é
    importado uma vez e renomeado para `<path>.migrated` (backup).
    """

    def __init__(
        self,
        store: LanceStore,
        memory_type: str,
        embed_dim: int = EMBED_DIM,
        legacy_index_path: Optional[str] = None,
        legacy_id_map_path: Optional[str] = None,
    ):
        self._store       = store
        self._memory_type = memory_type
        self._embed_dim   = embed_dim
        self._table       = store.table(embed_dim)
        self._filter      = f"memory_type = '{memory_type}'"
        if legacy_index_path:
            self._migrate_from_faiss(legacy_index_path, legacy_id_map_path)
        log.info(f"Índice LanceDB [{memory_type}] pronto — {self.total} vetores (dim={embed_dim})")

    # ── migração one-shot FAISS → LanceDB ──
    def _migrate_from_faiss(self, index_path: str, id_map_path: Optional[str]):
        p = Path(index_path)
        if not p.exists():
            return
        with self._store.lock:
            if self._table.count_rows(self._filter) > 0:
                log.warning(
                    f"[{self._memory_type}] {index_path} existe mas o LanceDB já tem dados "
                    f"desse store — migração ignorada (apague/renomeie o arquivo legado)"
                )
                return
        try:
            import faiss  # só necessário para migrar índices antigos
        except ImportError:
            log.warning(f"[{self._memory_type}] faiss não instalado — não foi possível migrar {index_path}")
            return
        loaded = faiss.read_index(str(p))
        n = loaded.ntotal
        if n > 0:
            if isinstance(loaded, faiss.IndexIDMap2) or hasattr(loaded, "id_map"):
                ids  = faiss.vector_to_array(loaded.id_map).astype(np.int64)
                vecs = faiss.downcast_index(loaded.index).reconstruct_n(0, n)
            else:
                # formato antigo: IndexFlatIP + .npy externo com os ids
                ids = np.load(id_map_path).astype(np.int64) if id_map_path and Path(id_map_path).exists() \
                      else np.arange(n, dtype=np.int64)
                if len(ids) != n:
                    log.warning(f"[{self._memory_type}] id_map legado com {len(ids)} ids p/ {n} vetores — usando posição")
                    ids = np.arange(n, dtype=np.int64)
                vecs = loaded.reconstruct_n(0, n)
            self.add_batch(np.asarray(vecs, dtype=np.float32), [int(i) for i in ids])
            if self.total != n:
                raise RuntimeError(
                    f"[{self._memory_type}] migração FAISS→LanceDB inconsistente "
                    f"({self.total} != {n}) — arquivo legado mantido"
                )
        p.rename(str(p) + ".migrated")
        log.info(f"[{self._memory_type}] {n} vetores migrados de {index_path} (backup: .migrated)")

    def _arrow(self, ids: list[int], embeddings: np.ndarray):
        emb = np.ascontiguousarray(embeddings, dtype=np.float32).reshape(len(ids), self._embed_dim)
        return pa.table(
            {
                "memory_type": pa.array([self._memory_type] * len(ids), pa.string()),
                "id":          pa.array(ids, pa.int64()),
                "vector":      pa.FixedSizeListArray.from_arrays(
                                   pa.array(emb.ravel(), pa.float32()), self._embed_dim),
            },
            schema=self._store.schema(self._embed_dim),
        )

    def add(self, embedding: np.ndarray, record_id: int):
        self.add_batch(np.asarray(embedding, dtype=np.float32).reshape(1, -1), [record_id])

    def add_batch(self, embeddings: np.ndarray, record_ids: list[int]):
        """Adiciona múltiplos vetores de uma vez — bem mais barato que add()
        individual (cada chamada gera um fragmento novo no LanceDB)."""
        if embeddings.shape[0] != len(record_ids):
            raise ValueError(
                f"embeddings ({embeddings.shape[0]}) e record_ids ({len(record_ids)}) "
                f"devem ter o mesmo tamanho"
            )
        if embeddings.shape[0] == 0:
            return
        data = self._arrow([int(i) for i in record_ids], embeddings)
        with self._store.lock:
            self._table.add(data)
            self._store.note_write(self._embed_dim, len(record_ids))

    def remove_ids(self, record_ids: set[int]):
        if not record_ids:
            return
        id_list = ",".join(str(int(i)) for i in record_ids)
        with self._store.lock:
            self._table.delete(f"{self._filter} AND id IN ({id_list})")
            self._store.note_write(self._embed_dim)
        log.info(f"LanceDB: remoção de até {len(record_ids)} vetores [{self._memory_type}]")

    def reset(self):
        with self._store.lock:
            self._table.delete(self._filter)
            self._store.note_write(self._embed_dim)

    def _query(self, q: np.ndarray, where: str, k: int) -> list[tuple[int, float]]:
        with self._store.lock:
            rows = (
                self._table.search(q)
                .distance_type("cosine")
                .where(where, prefilter=True)
                .select(["id", "_distance"])
                .limit(k)
                .to_list()
            )
        return [(int(r["id"]), 1.0 - float(r["_distance"])) for r in rows]

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        return self._query(q, self._filter, top_k)

    def search_batch(
        self, query_embeddings: np.ndarray, top_k: int
    ) -> list[list[tuple[int, float]]]:
        """N queries (ex.: os segmentos de uma pergunta composta). N é pequeno
        (SEGMENT_MAX_COUNT), então é um loop simples sobre search()."""
        n = query_embeddings.shape[0] if query_embeddings.ndim == 2 else 0
        return [self.search(query_embeddings[i], top_k) for i in range(n)]

    def search_similar(self, embedding: np.ndarray) -> float:
        results = self.search(embedding, top_k=1)
        return results[0][1] if results else 0.0

    def search_subset(
        self, query_embedding: np.ndarray, record_ids: list[int], top_k: int
    ) -> list[tuple[int, float]]:
        """Ranqueia `record_ids` por similaridade com `query_embedding`, SEM
        comparar com o restante do store (filtro nativo `id IN (...)`)."""
        if not record_ids or top_k <= 0:
            return []
        id_list = ",".join(str(int(i)) for i in record_ids)
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        return self._query(q, f"{self._filter} AND id IN ({id_list})", min(top_k, len(record_ids)))

    def get_vectors(self, record_ids: list[int]) -> dict[int, np.ndarray]:
        """Vetores armazenados dos ids pedidos (ids ausentes ficam de fora)."""
        if not record_ids:
            return {}
        id_list = ",".join(str(int(i)) for i in record_ids)
        with self._store.lock:
            rows = (
                self._table.search()
                .where(f"{self._filter} AND id IN ({id_list})")
                .select(["id", "vector"])
                .limit(len(record_ids) * 2)
                .to_list()
            )
        out: dict[int, np.ndarray] = {}
        for r in rows:
            out.setdefault(int(r["id"]), np.asarray(r["vector"], dtype=np.float32))
        return out

    def flush(self):
        """Compacta os fragmentos gerados por adds/deletes desde a última
        compactação (no-op se não houve escrita). Chamado pelo
        `index_flush_job` e no `shutdown()`. A durabilidade em si é imediata."""
        self._store.compact(self._embed_dim)

    @property
    def total(self) -> int:
        with self._store.lock:
            return int(self._table.count_rows(self._filter))


__all__ = [
    'LanceStore',
    'MemoryIndex',
    '_chunk_text',
]
