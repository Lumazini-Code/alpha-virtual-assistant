"""
Memory Module
Persists and retrieves:
  - User preferences (coding style, frameworks, conventions)
  - Past bug fixes (error → solution pairs)
  - Generation feedback (accepted / rejected / modified)

Uses SQLite for local persistence with semantic similarity via ONNX embeddings.
"""

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

DB_PATH     = Path(__file__).parent.parent / "data" / "memory.db"
MODEL_PATH  = Path(__file__).parent.parent / "models" / "embedder" / "model.onnx"
TOKENIZER_PATH = Path(__file__).parent.parent / "models" / "embedder"
TOP_K = 5


class MemoryModule:
    def __init__(self):
        self.conn: Optional[sqlite3.Connection] = None
        self.session: Optional[ort.InferenceSession] = None
        self.tokenizer = None

    async def connect(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self._init_schema()
        await self._load_embedder()
        logger.info("MemoryModule: connected.")

    async def disconnect(self):
        if self.conn:
            self.conn.close()

    # ─── Schema ───────────────────────────────────────────────────────────────

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS preferences (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                scope     TEXT    NOT NULL DEFAULT 'global',
                key       TEXT    NOT NULL,
                value     TEXT    NOT NULL,
                updated   TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS bug_fixes (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                language   TEXT,
                error_hash TEXT    NOT NULL,
                error_ctx  TEXT,
                solution   TEXT    NOT NULL,
                embedding  BLOB,
                created    TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id   TEXT    NOT NULL,
                accepted     INTEGER NOT NULL,
                original     TEXT,
                modified     TEXT,
                notes        TEXT,
                embedding    BLOB,
                created      TEXT    NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_pref_scope_key ON preferences(scope, key);
        """)
        self.conn.commit()

    # ─── Public Interface ─────────────────────────────────────────────────────

    async def retrieve(self, prompt: str, language: str, intent: str) -> dict:
        """Returns relevant preferences and past bug/feedback entries for a prompt."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._retrieve_sync, prompt, language, intent)

    async def store_preference(self, key: str, value: str, scope: str = "global"):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._upsert_preference, key, value, scope)

    async def store_feedback(self, payload: dict):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._insert_feedback, payload)

    # ─── Sync Implementations ─────────────────────────────────────────────────

    def _retrieve_sync(self, prompt: str, language: str, intent: str) -> dict:
        entries = []

        # Always load all preferences
        rows = self.conn.execute(
            "SELECT key, value, scope FROM preferences WHERE scope='global' OR scope=?",
            (language,),
        ).fetchall()
        for key, value, scope in rows:
            entries.append({
                "type": "preference",
                "content": f"{key}: {value}",
                "relevance": 1.0,
                "scope": scope,
            })

        # Semantic search over bug fixes when in fix/refactor mode
        if intent in ("fix", "refactor"):
            bug_entries = self._semantic_search_bugs(prompt, language)
            entries.extend(bug_entries)

        # Semantic search over feedback
        feedback_entries = self._semantic_search_feedback(prompt)
        entries.extend(feedback_entries)

        return {"entries": entries, "preferences": {k: v for k, v, _ in rows}}

    def _upsert_preference(self, key: str, value: str, scope: str):
        self.conn.execute(
            """INSERT INTO preferences (scope, key, value, updated)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(scope, key) DO UPDATE SET value=excluded.value, updated=excluded.updated""",
            (scope, key, value, datetime.utcnow().isoformat()),
        )
        self.conn.commit()

    def _insert_feedback(self, payload: dict):
        embedding = self._embed(payload.get("notes") or payload.get("modified") or "")
        self.conn.execute(
            """INSERT INTO feedback (request_id, accepted, original, modified, notes, embedding, created)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                payload["request_id"],
                int(payload["accepted"]),
                payload.get("original"),
                payload.get("modifications"),
                payload.get("notes"),
                embedding.tobytes() if embedding is not None else None,
                datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()

    # ─── Semantic Search ──────────────────────────────────────────────────────

    def _semantic_search_bugs(self, prompt: str, language: str) -> list[dict]:
        query_emb = self._embed(prompt)
        if query_emb is None:
            return []

        rows = self.conn.execute(
            "SELECT error_ctx, solution, embedding FROM bug_fixes WHERE language=? OR language IS NULL",
            (language,),
        ).fetchall()

        return self._rank_by_similarity(query_emb, rows, content_key="solution", context_key="error_ctx", entry_type="bug_fix")

    def _semantic_search_feedback(self, prompt: str) -> list[dict]:
        query_emb = self._embed(prompt)
        if query_emb is None:
            return []

        rows = self.conn.execute(
            "SELECT notes, modified, embedding FROM feedback WHERE accepted=1 AND modified IS NOT NULL"
        ).fetchall()

        return self._rank_by_similarity(query_emb, rows, content_key="modified", context_key="notes", entry_type="feedback")

    def _rank_by_similarity(self, query_emb, rows, content_key, context_key, entry_type) -> list[dict]:
        scored = []
        for row in rows:
            content, context, emb_bytes = row
            if emb_bytes is None or content is None:
                continue
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            sim = float(self._cosine(query_emb, emb))
            if sim > 0.5:
                scored.append({"type": entry_type, "content": content, "relevance": sim})

        scored.sort(key=lambda x: x["relevance"], reverse=True)
        return scored[:TOP_K]

    # ─── Embedder ─────────────────────────────────────────────────────────────

    async def _load_embedder(self):
        if MODEL_PATH.exists():
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            self.session = ort.InferenceSession(
                str(MODEL_PATH),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            try:
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH / "tokenizer.json"))
                logger.info("MemoryModule: embedder loaded.")
            except Exception as e:
                logger.warning(f"MemoryModule: tokenizer unavailable — semantic search disabled. {e}")
        else:
            logger.warning("MemoryModule: embedder model not found — semantic search disabled.")

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if self.session is None or self.tokenizer is None or not text:
            return None
        enc  = self.tokenizer.encode(text[:512])
        ids  = np.array([enc.ids],            dtype=np.int64)
        mask = np.array([enc.attention_mask],  dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        expected = {i.name for i in self.session.get_inputs()}
        if "token_type_ids" in expected:
            feed["token_type_ids"] = np.zeros_like(ids, dtype=np.int64)
        out  = self.session.run(None, feed)
        # Mean-pool token embeddings
        emb = out[0][0].mean(axis=0).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            return 0.0
        return float(np.dot(a, b))