"""
Research Module
Retrieves relevant documentation, patterns, and references for code generation.
Sources: local vector index (ONNX), and optionally a web search fallback.
"""

import asyncio
import json
import logging
import httpx
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.45
MAX_TIMEOUT_SECONDS = 8
INDEX_PATH = Path(__file__).parent.parent / "data" / "research_index.json"
MODEL_PATH = Path(__file__).parent.parent / "models" / "embedder" / "model.onnx"
TOKENIZER_PATH = Path(__file__).parent.parent / "models" / "embedder"


class ResearchModule:
    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.tokenizer = None
        self.index: list[dict] = []  # [{"content": str, "url": str, "embedding": np.ndarray}]

    async def connect(self):
        await self._load_embedder()
        self._load_index()
        logger.info("ResearchModule: ready.")

    # ─── Public Interface ─────────────────────────────────────────────────────

    async def query(
        self,
        topic: str,
        language: Optional[str] = None,
        max_results: int = 5,
    ) -> list[dict]:
        """
        Returns relevant docs from the local index.
        Falls back to a web-search stub if local results are insufficient.
        """
        loop = asyncio.get_event_loop()

        # Local semantic search
        local_results = await loop.run_in_executor(
            None, self._local_search, topic, language, max_results
        )

        if len(local_results) >= max_results:
            return local_results[:max_results]

        # Fallback — web search stub (replace with real API key integration)
        remaining = max_results - len(local_results)
        web_results = await self._web_search_fallback(topic, language, remaining)

        combined = local_results + web_results
        combined.sort(key=lambda r: r.get("relevance", 0), reverse=True)
        return combined[:max_results]

    # ─── Local Index ──────────────────────────────────────────────────────────

    def _local_search(self, topic: str, language: Optional[str], max_results: int) -> list[dict]:
        if not self.index or self.session is None:
            return []

        query_emb = self._embed(topic)
        if query_emb is None:
            return []

        scored = []
        for entry in self.index:
            if language and entry.get("language") and entry["language"] != language:
                continue
            emb = np.array(entry["embedding"], dtype=np.float32)
            sim = float(np.dot(query_emb, emb))
            if sim >= RELEVANCE_THRESHOLD:
                scored.append({
                    "content": entry["content"],
                    "url": entry.get("url", ""),
                    "relevance": sim,
                    "source": "local_index",
                })

        scored.sort(key=lambda x: x["relevance"], reverse=True)
        return scored[:max_results]

    def _load_index(self):
        if INDEX_PATH.exists():
            with open(INDEX_PATH) as f:
                self.index = json.load(f)
            logger.info(f"ResearchModule: loaded {len(self.index)} index entries.")
        else:
            logger.warning("ResearchModule: no local index found. Starting with empty index.")

    # ─── Web Search Fallback ──────────────────────────────────────────────────

    async def _web_search_fallback(
        self, topic: str, language: Optional[str], max_results: int
    ) -> list[dict]:
        """
        Stub for web search. Replace with a real provider
        (SerpAPI, Brave Search, DuckDuckGo API, etc.)
        by adding the API key to config and implementing the HTTP call.
        """
        query = f"{language} {topic}" if language else topic
        logger.info(f"ResearchModule: web search fallback for '{query}'")

        # Placeholder — returns empty until a real provider is wired up
        # Example integration point:
        # async with httpx.AsyncClient(timeout=MAX_TIMEOUT_SECONDS) as client:
        #     resp = await client.get(
        #         "https://api.search-provider.io/search",
        #         params={"q": query, "num": max_results},
        #         headers={"Authorization": f"Bearer {API_KEY}"},
        #     )
        #     results = resp.json()["results"]
        #     return [{"content": r["snippet"], "url": r["url"], "relevance": 0.5} for r in results]

        return []

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
                logger.info("ResearchModule: embedder loaded.")
            except Exception as e:
                logger.warning(f"ResearchModule: tokenizer unavailable. {e}")
        else:
            logger.warning("ResearchModule: embedder not found — local search disabled.")

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if self.session is None or self.tokenizer is None:
            return None
        enc  = self.tokenizer.encode(text[:512])
        ids  = np.array([enc.ids],            dtype=np.int64)
        mask = np.array([enc.attention_mask],  dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        expected = {i.name for i in self.session.get_inputs()}
        if "token_type_ids" in expected:
            feed["token_type_ids"] = np.zeros_like(ids, dtype=np.int64)
        out  = self.session.run(None, feed)
        emb  = out[0][0].mean(axis=0).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)