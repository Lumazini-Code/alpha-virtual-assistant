from __future__ import annotations

"""
memory_db.py — Camada de persistência SQLite (sem lógica de rede/vetor).

MemoryDB (longo prazo), ShortTermDB (turnos), IndexedFilesDB (arquivos +
chunks), VisualDictDB / FaceDictDB / VoiceDictDB (dicionários multimodais)
e ToolsDB (RAG de tool selection). MemoryDB e ToolsDB herdam EdgeGraphMixin.
"""

import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from memory_config import LTP_ACCESS_BOOST, log
from memory_graph import EdgeGraphMixin
from memory_models import Turn

# ── Banco de dados de longo prazo ──────────────────────────────────────────────

class MemoryDB(EdgeGraphMixin):
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
        # ── NEW: grafo Hebbiano com arestas tipadas (cria/migra memory_edges)
        self._create_edge_table()
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
                # ── NEW (LTP): reforço saturante de confiança a cada
                # acesso — "use it and improve it". Compõe com o decay:
                # o decay multiplica o que quer que a confiança seja; o
                # reforço puxa pra cima no acesso, o decay puxa pra baixo
                # com o tempo. Caminho único — nenhum outro lugar incrementa
                # access_count de LT.
                self._conn.execute(
                    "UPDATE memories SET access_count = access_count + 1, last_accessed = ?, "
                    "confidence = confidence + ? * (1 - confidence) WHERE id = ?",
                    (time.time(), LTP_ACCESS_BOOST, memory_id),
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
                # ── NEW (Hebbian graph): antes de apagar as memórias
                # expiradas, apaga as arestas que apontam pra elas — o
                # grafo nunca mantém aresta pra linha morta.
                expired = self._conn.execute(
                    "SELECT id FROM memories WHERE confidence < 0.01"
                ).fetchall()
                for expired_row in expired:
                    self._conn.execute(
                        "DELETE FROM memory_edges WHERE memory_id_a = ? OR memory_id_b = ?",
                        (expired_row["id"], expired_row["id"]),
                    )
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

    Os embeddings em si moram no LanceDB (`fd_index`); aqui só ficam o nome
    e o mapeamento embedding_id → person_id. `memory_id` é OPCIONAL (default
    NULL): só é preenchido quando o cadastro pede link_to_memory — reconhecimento
    facial não precisa, por padrão, de descrição gravada na memória geral, mas o
    link dá ao rosto um nó no grafo Hebbiano.
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
        # NEW: link opcional com a memória de longo prazo (nó no grafo Hebbiano)
        if "memory_id" not in cols:
            self._conn.execute("ALTER TABLE face_people ADD COLUMN memory_id INTEGER DEFAULT NULL")

    @staticmethod
    def _normalize_key(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip().lower())

    # ── Person operations ──

    def insert_person(
        self, person_name: str, description: str, source: str, confidence: float,
        memory_id: Optional[int] = None,
    ) -> int:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO face_people "
                "(person_name, person_key, description, source, confidence, memory_id, "
                "created_at, last_accessed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (person_name, self._normalize_key(person_name), description, source,
                 confidence, memory_id, now, now),
            )
            return cur.lastrowid

    def set_memory_id(self, person_id: int, memory_id: Optional[int]):
        with self._lock:
            self._conn.execute(
                "UPDATE face_people SET memory_id = ? WHERE id = ?", (memory_id, person_id)
            )

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


# ── NEW: Dicionário de Voz ─────────────────────────────────────────────────────

class VoiceDictDB:
    """
    Persiste as pessoas cadastradas pela identificação de locutor: uma
    pessoa tem N embeddings de exemplo (um por frase falada registrada —
    permite reconhecer a mesma voz em condições de captação diferentes).

    Os embeddings em si moram no FAISS (`vo_index`); aqui ficam o nome, a
    descrição, o `memory_id` (link para a memória de longo prazo onde a
    descrição também foi gravada) e o mapeamento embedding_id → person_id.
    Estrutura idêntica à FaceDictDB, com a coluna extra `memory_id`.
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
            CREATE TABLE IF NOT EXISTS voice_people (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name   TEXT    NOT NULL,
                person_key    TEXT    NOT NULL UNIQUE,   -- nome normalizado (lower/strip)
                description   TEXT    NOT NULL DEFAULT '',  -- quem é a pessoa (relação/contexto)
                source        TEXT    NOT NULL DEFAULT 'voice_pipeline',
                confidence    REAL    NOT NULL DEFAULT 1.0,
                memory_id     INTEGER,                    -- FK lógica p/ memories.id
                created_at    REAL    NOT NULL,
                last_accessed REAL    NOT NULL,
                access_count  INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_vp_key ON voice_people(person_key);

            CREATE TABLE IF NOT EXISTS voice_embeddings (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id  INTEGER NOT NULL,
                created_at REAL    NOT NULL,
                FOREIGN KEY (person_id) REFERENCES voice_people(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_ve_person ON voice_embeddings(person_id);
        """)

    @staticmethod
    def _normalize_key(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip().lower())

    # ── Person operations ──

    def insert_person(self, person_name: str, description: str, source: str,
                      confidence: float, memory_id: Optional[int]) -> int:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO voice_people "
                "(person_name, person_key, description, source, confidence, memory_id, "
                "created_at, last_accessed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (person_name, self._normalize_key(person_name), description, source,
                 confidence, memory_id, now, now),
            )
            return cur.lastrowid

    def get_by_name(self, person_name: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM voice_people WHERE person_key = ?",
            (self._normalize_key(person_name),),
        ).fetchone()

    def get_person_by_id(self, person_id: int) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM voice_people WHERE id = ?", (person_id,)
        ).fetchone()

    def update_description(self, person_id: int, description: str):
        """Atualiza/edita a descrição de uma pessoa já cadastrada."""
        with self._lock:
            self._conn.execute(
                "UPDATE voice_people SET description = ? WHERE id = ?",
                (description, person_id),
            )

    def update_access(self, person_id: int):
        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE voice_people SET access_count = access_count + 1, "
                    "last_accessed = ? WHERE id = ?",
                    (time.time(), person_id),
                )
        except sqlite3.OperationalError:
            pass

    def delete_person(self, person_id: int) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM voice_people WHERE id = ?", (person_id,))
            return cur.rowcount

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM voice_people").fetchone()[0]

    def list_people(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT p.*, "
            "(SELECT COUNT(*) FROM voice_embeddings e WHERE e.person_id = p.id) "
            "AS examples_count "
            "FROM voice_people p ORDER BY p.last_accessed DESC"
        ).fetchall()

    # ── Embedding-row operations (mapeamento embedding_id → person_id) ──

    def insert_embedding(self, person_id: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO voice_embeddings (person_id, created_at) VALUES (?, ?)",
                (person_id, time.time()),
            )
            return cur.lastrowid

    def get_person_id_by_embedding(self, embedding_id: int) -> Optional[int]:
        row = self._conn.execute(
            "SELECT person_id FROM voice_embeddings WHERE id = ?",
            (embedding_id,),
        ).fetchone()
        return row["person_id"] if row else None

    def get_embedding_ids_by_person(self, person_id: int) -> list[int]:
        rows = self._conn.execute(
            "SELECT id FROM voice_embeddings WHERE person_id = ?", (person_id,)
        ).fetchall()
        return [r["id"] for r in rows]

    def count_examples(self, person_id: int) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM voice_embeddings WHERE person_id = ?",
            (person_id,),
        ).fetchone()[0]


# ── NEW: Banco de tools (RAG de tool selection) ────────────────────────────────

class ToolsDB(EdgeGraphMixin):
    """
    Registro das tools expostas ao LLM. Cada tool tem N vetores no LanceDB
    (`tl_index`, memory_type="tool"): 1 do "documento" (nome + descrição), 1 por
    exemplo de query fornecido no cadastro (kind="example") e 1 por query real
    aprendida com o uso (kind="learned"). A busca faz max-pooling por tool.

    O grafo de co-uso vive aqui mesmo (tabela `tool_edges`, mesma lógica de
    arestas tipadas do grafo Hebbiano de memórias): temporal_precedence captura
    sequências (listar → ler → editar) e co_activation, tools usadas juntas.
    """
    EDGE_TABLE = "tool_edges"

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS tools (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                name_key    TEXT    NOT NULL UNIQUE,
                description TEXT    NOT NULL,
                is_core     INTEGER NOT NULL DEFAULT 0,
                doc_hash    TEXT    NOT NULL,
                created_at  REAL    NOT NULL,
                last_used   REAL,
                use_count   INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS tool_vectors (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_id    INTEGER NOT NULL,
                kind       TEXT    NOT NULL,      -- doc | example | learned
                text       TEXT    NOT NULL,
                created_at REAL    NOT NULL,
                FOREIGN KEY (tool_id) REFERENCES tools(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_tv_tool ON tool_vectors(tool_id);
        """)
        self._create_edge_table()

    @staticmethod
    def _normalize_key(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip().lower())

    # ── Tools ──
    def insert_tool(self, name: str, description: str, is_core: bool, doc_hash: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO tools (name, name_key, description, is_core, doc_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (name, self._normalize_key(name), description, int(is_core), doc_hash, time.time()),
            )
            return cur.lastrowid

    def update_tool(self, tool_id: int, description: str, is_core: bool, doc_hash: str):
        with self._lock:
            self._conn.execute(
                "UPDATE tools SET description = ?, is_core = ?, doc_hash = ? WHERE id = ?",
                (description, int(is_core), doc_hash, tool_id),
            )

    def set_core(self, tool_id: int, is_core: bool):
        with self._lock:
            self._conn.execute("UPDATE tools SET is_core = ? WHERE id = ?", (int(is_core), tool_id))

    def get_by_name(self, name: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM tools WHERE name_key = ?", (self._normalize_key(name),)
        ).fetchone()

    def get_by_ids(self, ids: list[int]) -> list[sqlite3.Row]:
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        return self._conn.execute(f"SELECT * FROM tools WHERE id IN ({ph})", list(ids)).fetchall()

    def list_tools(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT t.*, (SELECT COUNT(*) FROM tool_vectors v WHERE v.tool_id = t.id) "
            "AS vectors_count FROM tools t ORDER BY t.name"
        ).fetchall()

    def list_core(self) -> list[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM tools WHERE is_core = 1").fetchall()

    def mark_used(self, tool_ids: list[int]):
        if not tool_ids:
            return
        now = time.time()
        with self._lock:
            self._conn.executemany(
                "UPDATE tools SET use_count = use_count + 1, last_used = ? WHERE id = ?",
                [(now, tid) for tid in tool_ids],
            )

    def delete_tool(self, tool_id: int) -> int:
        """Apaga a tool (vetores em cascata) e todas as arestas dela."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM tools WHERE id = ?", (tool_id,))
            self._conn.execute(
                "DELETE FROM tool_edges WHERE memory_id_a = ? OR memory_id_b = ?",
                (tool_id, tool_id),
            )
            return cur.rowcount

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM tools").fetchone()[0]

    # ── Vetores (mapeamento vector_id → tool_id) ──
    def insert_vector(self, tool_id: int, kind: str, text: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO tool_vectors (tool_id, kind, text, created_at) VALUES (?, ?, ?, ?)",
                (tool_id, kind, text, time.time()),
            )
            return cur.lastrowid

    def get_vector_ids(self, tool_id: int, kinds: Optional[tuple[str, ...]] = None) -> list[int]:
        if kinds:
            ph = ",".join("?" * len(kinds))
            rows = self._conn.execute(
                f"SELECT id FROM tool_vectors WHERE tool_id = ? AND kind IN ({ph})",
                (tool_id, *kinds),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id FROM tool_vectors WHERE tool_id = ?", (tool_id,)
            ).fetchall()
        return [r["id"] for r in rows]

    def get_tool_ids_by_vectors(self, vector_ids: list[int]) -> dict[int, int]:
        if not vector_ids:
            return {}
        ph = ",".join("?" * len(vector_ids))
        rows = self._conn.execute(
            f"SELECT id, tool_id FROM tool_vectors WHERE id IN ({ph})", list(vector_ids)
        ).fetchall()
        return {r["id"]: r["tool_id"] for r in rows}

    def delete_vectors(self, vector_ids: list[int]):
        if not vector_ids:
            return
        ph = ",".join("?" * len(vector_ids))
        with self._lock:
            self._conn.execute(f"DELETE FROM tool_vectors WHERE id IN ({ph})", list(vector_ids))

    def count_learned(self, tool_id: int) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM tool_vectors WHERE tool_id = ? AND kind = 'learned'", (tool_id,)
        ).fetchone()[0]


__all__ = [
    'FaceDictDB',
    'IndexedFilesDB',
    'MemoryDB',
    'ShortTermDB',
    'ToolsDB',
    'VisualDictDB',
    'VoiceDictDB',
]
