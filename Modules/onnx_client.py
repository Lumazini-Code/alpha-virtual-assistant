"""
AVA ONNX Client — Lightweight async HTTP client for the unified ONNX serving API.

Provides two client classes:
  - EmbeddingClient  → calls /v1/embed
  - RerankerClient   → calls /v1/rerank and /v1/score

Uses httpx.AsyncClient with connection pooling for maximum throughput.
Thread-safe: a single client instance can be shared across the application.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [Memory] %(message)s")
log = logging.getLogger("ava.memory")

# Default ONNX serving endpoint
DEFAULT_ONNX_BASE_URL = "http://localhost:2002"


# ── Embedding Client ──────────────────────────────────────────────────────────

class EmbeddingClient:
    """
    Async client for the /v1/embed endpoint.

    Usage:
        client = EmbeddingClient()
        vecs = await client.embed(["hello world", "another text"])
        vec  = await client.embed_one("single text")
        qvec = await client.embed_query("search query")
        pvec = await client.embed_passages(["doc1 text", "doc2 text"])
    """

    def __init__(
        self,
        base_url: str = DEFAULT_ONNX_BASE_URL,
        timeout: float = 30.0,
        max_connections: int = 20,
    ):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections // 2,
            ),
            headers={"Content-Type": "application/json"},
        )

    # ── Core embed ───────────────────────────────────────────────────────────

    async def embed(
        self,
        texts: list[str],
        prefix: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate L2-normalized embeddings for a batch of texts.

        Args:
            texts: List of input strings.
            prefix: Optional e5 prefix ('query' or 'passage').

        Returns:
            np.ndarray of shape (len(texts), 384), float32.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        payload = {"texts": texts}
        if prefix:
            payload["prefix"] = prefix

        resp = await self._client.post("/v1/embed", json=payload)
        resp.raise_for_status()
        data = resp.json()

        return np.array(data["embeddings"], dtype=np.float32)

    # ── Convenience methods ──────────────────────────────────────────────────

    async def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text — returns 1D array of shape (384,)."""
        result = await self.embed([text])
        return result[0]

    async def embed_query(self, text: str) -> np.ndarray:
        """Embed a query with the 'query' prefix (e5 convention)."""
        result = await self.embed([text], prefix="query")
        return result[0]

    async def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed passages with the 'passage' prefix (e5 convention)."""
        return await self.embed(texts, prefix="passage")

    async def embed_batch_two(self, text_a: str, text_b: str) -> tuple[np.ndarray, np.ndarray]:
        """Embed two texts in a single API call — avoids double overhead."""
        result = await self.embed([text_a, text_b])
        return result[0], result[1]

    # ── Similarity utilities ─────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors (assumes L2-normalized → dot product)."""
        return float(np.dot(a, b))

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()


# ── Reranker Client ────────────────────────────────────────────────────────────

class RerankerClient:
    """
    Async client for the /v1/rerank and /v1/score endpoints.

    Usage:
        client = RerankerClient()
        scored = await client.rerank("query text", ["passage 1", "passage 2"])
        scores = await client.score("query text", ["passage 1", "passage 2"])
    """

    def __init__(
        self,
        base_url: str = DEFAULT_ONNX_BASE_URL,
        timeout: float = 30.0,
        max_connections: int = 20,
    ):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections // 2,
            ),
            headers={"Content-Type": "application/json"},
        )

    # ── Core methods ─────────────────────────────────────────────────────────

    async def score(self, query: str, passages: list[str]) -> np.ndarray:
        """
        Raw cross-encoder scores for (query, passage) pairs.

        Returns:
            np.ndarray of shape (len(passages),), float32, sigmoid-normalized [0, 1].
        """
        if not passages:
            return np.array([], dtype=np.float32)

        resp = await self._client.post("/v1/score", json={
            "query": query,
            "passages": passages,
        })
        resp.raise_for_status()
        data = resp.json()

        return np.array(data["scores"], dtype=np.float32)

    async def rerank(
        self,
        query: str,
        candidates: list[str],
        top_k: Optional[int] = None,
    ) -> list[tuple[int, float, str]]:
        """
        Rerank candidates by relevance to query.

        Returns:
            List of (original_index, score, text), sorted descending by score.
            Truncated to top_k if specified.
        """
        if not candidates:
            return []

        resp = await self._client.post("/v1/rerank", json={
            "query": query,
            "passages": candidates,
        })
        resp.raise_for_status()
        data = resp.json()

        results = [
            (r["index"], r["score"], r["text"])
            for r in data["results"]
        ]

        if top_k is not None:
            results = results[:top_k]

        return results

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()


# ── Health Check ───────────────────────────────────────────────────────────────

async def check_health(base_url: str = DEFAULT_ONNX_BASE_URL) -> dict:
    """Quick health check of the ONNX serving API."""
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=5.0) as client:
        resp = await client.get("/health")
        resp.raise_for_status()
        return resp.json()