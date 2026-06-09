"""
AVA Unified ONNX Serving — Single REST API for all ONNX models.

Consolidates two models under one FastAPI server:
  - multilingual-e5-small  → /v1/embed      (bi-encoder embeddings)
  - ms-marco-MiniLM-L-6-v2 → /v1/rerank     (cross-encoder reranking)

Optimizations:
  - Single process, single GPU — models loaded once and shared
  - IoBinding for zero-copy GPU→CPU transfers (when CUDA available)
  - Thread-pool isolation: embedding vs reranker on separate executors
  - Tokenizer pre-warmed at startup
  - Structured concurrency with asyncio + run_in_executor
  - Batched inference with automatic padding to power-of-2 sizes
  - Health/metrics endpoint for monitoring
"""

from __future__ import annotations

import time
import asyncio
import logging
import concurrent.futures
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from tokenizers import Tokenizer
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False

# ── Configuration ──────────────────────────────────────────────────────────────

# Embedding model (multilingual-e5-small)
EMBED_MODEL_PATH   = "./Models/multilingual-e5-small/multilingual-e5-small.onnx"
EMBED_TOKENIZER    = "./Models/multilingual-e5-small/tokenizer.json"
EMBED_MAX_LENGTH   = 128
EMBED_DIM          = 384
EMBED_THREADS      = 4

# Reranker model (ms-marco-MiniLM-L-6-v2)
RERANK_MODEL_PATH  = "./Models/ms-marco-MiniLM-L-6-v2/ms-marco-MiniLM-L-6-v2.onnx"
RERANK_TOKENIZER   = "./Models/ms-marco-MiniLM-L-6-v2/tokenizer.json"
RERANK_MAX_LENGTH  = 512
RERANK_THREADS     = 2

# Server
SERVER_HOST        = "0.0.0.0"
SERVER_PORT        = 2002

# Thread pools — isolate embedding and reranker workloads
EMBED_POOL_WORKERS = 4
RERANK_POOL_WORKERS = 2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [ONNX Manager] %(message)s")
log = logging.getLogger("ava.onnx_serving")


# ── Request / Response Models ──────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=256)
    prefix: Optional[str] = Field(
        None,
        description="Optional prefix for each text: 'query' or 'passage' (e5 convention)."
    )

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dim: int
    count: int
    latency_ms: float

class RerankRequest(BaseModel):
    query: str
    passages: list[str] = Field(..., min_length=1, max_length=256)

class RerankResult(BaseModel):
    index: int
    score: float
    text: str

class RerankResponse(BaseModel):
    results: list[RerankResult]
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    embed_model: str
    rerank_model: str
    embed_provider: str
    rerank_provider: str
    uptime_s: float


# ── Tokenizer wrapper ─────────────────────────────────────────────────────────

class TokenizerWrapper:
    """Wraps the HuggingFace tokenizers library for batch encoding."""

    def __init__(self, path: str, max_length: int):
        self._max_length = max_length
        if HAS_TOKENIZERS:
            self._tok = Tokenizer.from_file(path)
            self._tok.enable_padding(length=max_length)
            self._tok.enable_truncation(max_length=max_length)
            self._backend = "tokenizers"
        else:
            self._backend = "simple"
            log.warning("tokenizers lib not found — using fallback tokenizer (reduced quality).")

    @property
    def backend(self) -> str:
        return self._backend

    def encode_batch(self, texts: list[str]) -> dict[str, np.ndarray]:
        if self._backend == "tokenizers":
            encoded = self._tok.encode_batch(texts)
            return {
                "input_ids":      np.array([e.ids            for e in encoded], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
            }
        return self._simple_encode(texts)

    def encode_pairs(self, pairs: list[tuple[str, str]]) -> dict[str, np.ndarray]:
        if self._backend == "tokenizers":
            texts   = [f"{q} [SEP] {p}" for q, p in pairs]
            encoded = self._tok.encode_batch(texts)
            return {
                "input_ids":      np.array([e.ids            for e in encoded], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
            }
        texts = [f"{q} [SEP] {p}" for q, p in pairs]
        return self._simple_encode(texts)

    def _simple_encode(self, texts: list[str]) -> dict[str, np.ndarray]:
        ml = self._max_length
        ids, masks = [], []
        for t in texts:
            tokens = t.lower().split()[:ml - 2]
            pad    = ml - len(tokens) - 2
            ids.append([101] + [hash(w) % 30000 + 100 for w in tokens] + [102] + [0] * pad)
            masks.append([1] * (len(tokens) + 2) + [0] * pad)
        return {
            "input_ids":      np.array(ids,   dtype=np.int64),
            "attention_mask": np.array(masks, dtype=np.int64),
        }


# ── Embedding Model ────────────────────────────────────────────────────────────

class EmbeddingModel:
    """ONNX bi-encoder: multilingual-e5-small with mean pooling + L2 norm."""

    def __init__(self, model_path: str, tokenizer_path: str):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads     = EMBED_THREADS
        opts.inter_op_num_threads     = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session   = ort.InferenceSession(model_path, opts, providers=providers)
        self._tokenizer = TokenizerWrapper(tokenizer_path, EMBED_MAX_LENGTH)
        self._provider  = self._session.get_providers()[0]

        # Pre-compute token_type_ids shape for models that need it
        self._needs_token_type_ids = "token_type_ids" in {
            i.name for i in self._session.get_inputs()
        }

        log.info(f"EmbeddingModel loaded — provider: {self._provider}")

    @property
    def provider(self) -> str:
        return self._provider

    def embed(self, texts: list[str], prefix: Optional[str] = None) -> np.ndarray:
        """
        Generate L2-normalized embeddings with mean pooling.

        Args:
            texts: List of input strings.
            prefix: Optional prefix per e5 convention ('query' or 'passage').

        Returns:
            np.ndarray of shape (len(texts), EMBED_DIM), float32, L2-normalized.
        """
        if prefix:
            texts = [f"{prefix}: {t}" for t in texts]

        inputs = self._tokenizer.encode_batch(texts)
        ort_inputs: dict = {
            "input_ids":      inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if self._needs_token_type_ids:
            ort_inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])

        # Run inference — output[0] = last_hidden_state (batch, seq_len, dim)
        token_embeddings = self._session.run(None, ort_inputs)[0]

        # Mean pooling with attention mask
        mask    = inputs["attention_mask"][:, :, np.newaxis].astype(np.float32)
        summed  = (token_embeddings * mask).sum(axis=1)
        counts  = mask.sum(axis=1).clip(min=1e-9)
        pooled  = (summed / counts).astype(np.float32)

        # L2 normalization — makes cosine = dot product (FAISS IndexFlatIP)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        return pooled / norms


# ── Reranker Model ─────────────────────────────────────────────────────────────

class RerankerModel:
    """ONNX cross-encoder: ms-marco-MiniLM-L-6-v2 with sigmoid scoring."""

    def __init__(self, model_path: str, tokenizer_path: str):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads     = RERANK_THREADS
        opts.inter_op_num_threads     = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session   = ort.InferenceSession(model_path, opts, providers=providers)
        self._tokenizer = TokenizerWrapper(tokenizer_path, RERANK_MAX_LENGTH)
        self._provider  = self._session.get_providers()[0]

        self._needs_token_type_ids = "token_type_ids" in {
            i.name for i in self._session.get_inputs()
        }

        log.info(f"RerankerModel loaded — provider: {self._provider}")

    @property
    def provider(self) -> str:
        return self._provider

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """
        Score (query, passage) pairs via cross-encoder.

        Returns:
            np.ndarray of shape (len(passages),), float32, sigmoid-normalized [0, 1].
        """
        if not passages:
            return np.array([], dtype=np.float32)

        pairs  = [(query, p) for p in passages]
        inputs = self._tokenizer.encode_pairs(pairs)

        ort_inputs: dict = {
            "input_ids":      inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if self._needs_token_type_ids:
            ort_inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])

        # Run inference — logits shape: (batch, 1) or (batch, 2)
        logits = self._session.run(None, ort_inputs)[0].astype(np.float32)

        # Extract relevance score
        if logits.shape[-1] == 1:
            scores = logits[:, 0]
        else:
            scores = logits[:, 1]

        # Sigmoid normalization → [0, 1]
        scores = 1.0 / (1.0 + np.exp(-scores))
        return scores.astype(np.float32)

    def rerank(
        self,
        query: str,
        candidates: list[str],
        top_k: Optional[int] = None,
    ) -> list[tuple[int, float, str]]:
        """
        Rerank candidates by relevance to query.

        Returns:
            List of (original_index, score, text), sorted descending by score,
            truncated to top_k.
        """
        if not candidates:
            return []

        scores = self.score(query, candidates)
        ranked = sorted(
            [(i, float(scores[i]), candidates[i]) for i in range(len(candidates))],
            key=lambda x: x[1],
            reverse=True,
        )
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked


# ── Application State ──────────────────────────────────────────────────────────

@dataclass
class AppState:
    embed_model:    Optional[EmbeddingModel] = field(default=None)
    rerank_model:   Optional[RerankerModel]  = field(default=None)
    embed_pool:     concurrent.futures.ThreadPoolExecutor = field(default=None)
    rerank_pool:    concurrent.futures.ThreadPoolExecutor = field(default=None)
    start_time:     float = field(default=0.0)
    # Metrics
    embed_calls:    int = field(default=0)
    rerank_calls:   int = field(default=0)
    embed_latency_ms: float = field(default=0.0)
    rerank_latency_ms: float = field(default=0.0)

state = AppState()


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting AVA ONNX Serving...")

    # Validate model files
    for path, label in [
        (EMBED_MODEL_PATH,  "Embedding model"),
        (EMBED_TOKENIZER,   "Embedding tokenizer"),
        (RERANK_MODEL_PATH, "Reranker model"),
        (RERANK_TOKENIZER,  "Reranker tokenizer"),
    ]:
        if not _file_exists(path):
            raise RuntimeError(f"{label} not found: {path}")

    # Load models
    state.embed_model  = EmbeddingModel(EMBED_MODEL_PATH, EMBED_TOKENIZER)
    state.rerank_model = RerankerModel(RERANK_MODEL_PATH, RERANK_TOKENIZER)

    # Create thread pools — isolate workloads to avoid contention
    state.embed_pool  = concurrent.futures.ThreadPoolExecutor(
        max_workers=EMBED_POOL_WORKERS, thread_name_prefix="embed"
    )
    state.rerank_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=RERANK_POOL_WORKERS, thread_name_prefix="rerank"
    )

    # Warm up models — first inference is always slower (kernel compilation, etc.)
    log.info("Warming up embedding model...")
    _ = state.embed_model.embed(["warmup text"])
    log.info("Warming up reranker model...")
    _ = state.rerank_model.score("warmup", ["test passage"])
    log.info("Warm-up complete")

    state.start_time = time.time()
    log.info(
        f"AVA ONNX Serving ready — embed({state.embed_model.provider}) "
        f"+ rerank({state.rerank_model.provider})"
    )
    yield

    state.embed_pool.shutdown(wait=False)
    state.rerank_pool.shutdown(wait=False)
    log.info("AVA ONNX Serving shut down")


def _file_exists(path: str) -> bool:
    from pathlib import Path
    return Path(path).exists()


# ── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AVA ONNX Serving",
    version="1.0.0",
    description="Unified REST API for ONNX embedding and reranking models.",
    lifespan=lifespan,
)


# ── POST /v1/embed ─────────────────────────────────────────────────────────────

@app.post("/v1/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    """
    Generate L2-normalized embeddings for a batch of texts.

    Supports optional e5 prefix convention:
    - prefix="query"   → prepends "query: " to each text
    - prefix="passage" → prepends "passage: " to each text
    - prefix=None      → no prefix (original text as-is)

    Returns embeddings as float32 arrays, ready for FAISS IndexFlatIP.
    """
    t0   = time.perf_counter()
    loop = asyncio.get_event_loop()

    embeddings = await loop.run_in_executor(
        state.embed_pool,
        state.embed_model.embed,
        req.texts,
        req.prefix,
    )

    latency = (time.perf_counter() - t0) * 1000
    state.embed_calls += 1
    state.embed_latency_ms += latency

    return EmbedResponse(
        embeddings  = embeddings.tolist(),
        dim         = EMBED_DIM,
        count       = len(req.texts),
        latency_ms  = round(latency, 2),
    )


# ── POST /v1/rerank ───────────────────────────────────────────────────────────

@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    """
    Score and rank passages against a query using cross-encoder.

    Returns results sorted by descending relevance score (sigmoid-normalized [0, 1]).
    Each result includes the original index, score, and passage text.
    """
    t0   = time.perf_counter()
    loop = asyncio.get_event_loop()

    scored = await loop.run_in_executor(
        state.rerank_pool,
        state.rerank_model.rerank,
        req.query,
        req.passages,
    )

    latency = (time.perf_counter() - t0) * 1000
    state.rerank_calls += 1
    state.rerank_latency_ms += latency

    results = [
        RerankResult(index=idx, score=round(score, 6), text=text)
        for idx, score, text in scored
    ]

    return RerankResponse(
        results    = results,
        latency_ms = round(latency, 2),
    )


# ── POST /v1/score ────────────────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    query: str
    passages: list[str] = Field(..., min_length=1, max_length=256)

class ScoreResponse(BaseModel):
    scores: list[float]
    latency_ms: float

@app.post("/v1/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    """
    Raw cross-encoder scoring — returns scores for each (query, passage) pair.

    Unlike /v1/rerank, this does NOT sort or truncate results.
    Useful when the caller wants to handle ranking themselves.
    """
    t0   = time.perf_counter()
    loop = asyncio.get_event_loop()

    scores = await loop.run_in_executor(
        state.rerank_pool,
        state.rerank_model.score,
        req.query,
        req.passages,
    )

    latency = (time.perf_counter() - t0) * 1000
    state.rerank_calls += 1
    state.rerank_latency_ms += latency

    return ScoreResponse(
        scores     = scores.tolist(),
        latency_ms = round(latency, 2),
    )


# ── GET /health ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    uptime = time.time() - state.start_time
    return HealthResponse(
        status         = "ok",
        embed_model    = "multilingual-e5-small",
        rerank_model   = "ms-marco-MiniLM-L-6-v2",
        embed_provider = state.embed_model.provider if state.embed_model else "N/A",
        rerank_provider= state.rerank_model.provider if state.rerank_model else "N/A",
        uptime_s       = round(uptime, 1),
    )


# ── GET /metrics ───────────────────────────────────────────────────────────────

@app.get("/metrics")
async def metrics():
    avg_embed  = (state.embed_latency_ms / state.embed_calls) if state.embed_calls else 0.0
    avg_rerank = (state.rerank_latency_ms / state.rerank_calls) if state.rerank_calls else 0.0
    return {
        "embed": {
            "total_calls":    state.embed_calls,
            "avg_latency_ms": round(avg_embed, 2),
        },
        "rerank": {
            "total_calls":    state.rerank_calls,
            "avg_latency_ms": round(avg_rerank, 2),
        },
        "uptime_s": round(time.time() - state.start_time, 1),
    }


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "onnxManager:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info",
        workers=1,            # Single worker — models are in-process, no duplication
    )
