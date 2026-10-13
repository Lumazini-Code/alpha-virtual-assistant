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

import ctypes
import os
import time
import asyncio
import logging
import concurrent.futures
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from tokenizers import Tokenizer
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False

# ── Configuration ──────────────────────────────────────────────────────────────
# Tudo por env para trocar modelo/afinar consumo SEM rebuild: basta reiniciar
# o container (Modules/ é montado por volume).

# Embedding model — troca de modelo não requer mudança de código: aponte
# ONNX_EMBED_MODEL_PATH/ONNX_EMBED_TOKENIZER/ONNX_EMBED_DIM pro novo modelo.
# Candidatas com recall melhor que o e5-small (testar com rebuild_indices.py):
#   - intfloat/multilingual-e5-base   (278M, dim 768, int8 ≈ 280MB)
#   - Alibaba-NLP/gte-multilingual-base (305M, dim 768)
#   - BAAI/bge-m3 (568M, dim 1024) — só em máquina boa
EMBED_MODEL_PATH   = os.getenv("ONNX_EMBED_MODEL_PATH",  "./Models/multilingual-e5-small/multilingual-e5-small.onnx")
EMBED_TOKENIZER    = os.getenv("ONNX_EMBED_TOKENIZER", "./Models/multilingual-e5-small/tokenizer.json")
EMBED_MAX_LENGTH   = max(32, int(os.getenv("ONNX_EMBED_MAX_LENGTH", "128")))
EMBED_DIM          = int(os.getenv("ONNX_EMBED_DIM", "384"))
EMBED_THREADS      = max(1, int(os.getenv("ONNX_EMBED_THREADS", "2")))
# Teto de textos por chamada de inferência. A arena do ORT cresce com o batch
# e com workers concorrentes (medido: 3 batches de 128 → +850MB que não voltam).
# Chunkar aqui bounda o pico sem mudar a API.
EMBED_MAX_BATCH    = max(8, int(os.getenv("ONNX_EMBED_MAX_BATCH", "64")))
# Quantização int8 automática (fp32 → ~25% do tamanho). Mesmo padrão do
# vision.py (_ensure_quantized): quantiza na 1ª execução e cacheia o arquivo.
ONNX_EMBED_AUTO_QUANTIZE = os.getenv("ONNX_EMBED_AUTO_QUANTIZE", "1").strip().lower() in ("1", "true", "yes", "on")

# Reranker model (ms-marco-MiniLM-L-6-v2)
RERANK_MODEL_PATH  = "./Models/ms-marco-MiniLM-L-6-v2/ms-marco-MiniLM-L-6-v2.onnx"
RERANK_TOKENIZER   = "./Models/ms-marco-MiniLM-L-6-v2/tokenizer.json"
RERANK_MAX_LENGTH  = 512
RERANK_THREADS     = max(1, int(os.getenv("ONNX_RERANK_THREADS", "2")))
RERANK_MAX_BATCH   = max(8, int(os.getenv("ONNX_RERANK_MAX_BATCH", "32")))

# Server
SERVER_HOST        = "0.0.0.0"
SERVER_PORT        = 2002

# Thread pools — isolate embedding and reranker workloads.
# 4 workers × sessão intra-op concorrente era o que explodia a arena + arenas
# glibc; 2 já dão paralelismo suficiente para query + indexação.
EMBED_POOL_WORKERS  = max(1, int(os.getenv("ONNX_EMBED_POOL_WORKERS", "2")))
RERANK_POOL_WORKERS = max(1, int(os.getenv("ONNX_RERANK_POOL_WORKERS", "1")))

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


# ── Session options (memória) ─────────────────────────────────────────────────

def _memory_opts(intra: int, inter: int) -> ort.SessionOptions:
    """SessionOptions com tuning de memória.

    O default do ORT (arena CPU ativada + extend power-of-two) faz o RSS
    crescer até o PICO histórico e nunca voltar (medido: +850MB após 3
    batches de 128 textos). Aqui:
      - arena desativada → cada run aloca/desaloca direto (heap glibc, que
        o malloc_trim watchdog devolve ao OS);
      - mem_pattern off → não cacheia plano por shape (batches variam);
      - intra/inter-op baixos → menos buffers intermediários e menos
        threads concorrentes na mesma sessão.
    Custo: ~5-10% mais lento por run; ganho: RSS estável sob uso prolongado.
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads     = intra
    opts.inter_op_num_threads     = inter
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_cpu_mem_arena     = False
    opts.enable_mem_pattern       = False
    return opts


def _malloc_trim_watchdog(interval_s: float = 60.0) -> None:
    """Thread daemon: devolve ao OS o heap glibc liberado pelo free().

    glibc mantém memória free()da no heap do processo (brk/mmap trimming só
    acontece nas pontas). malloc_trim(0) força a devolução — essencial aqui
    porque os runs do ORT + numpy geram picos repetidos de alocação."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        return

    def _loop():
        while True:
            time.sleep(interval_s)
            try:
                libc.malloc_trim(0)
            except Exception:
                return

    threading.Thread(target=_loop, daemon=True, name="malloc-trim").start()


def _ensure_quantized(fp32_path: str) -> tuple[str, bool]:
    """Retorna (caminho_a_usar, usou_int8). Quantiza int8 na 1ª execução e
    cacheia ao lado do fp32 (`<nome>_int8.onnx`). Mesmo padrão do vision.py."""
    if not ONNX_EMBED_AUTO_QUANTIZE:
        return fp32_path, False
    p = Path(fp32_path)
    if p.suffix != ".onnx" or "int8" in p.stem.lower():
        return fp32_path, False
    int8_path = p.with_name(f"{p.stem}_int8.onnx")
    if int8_path.exists():
        return str(int8_path), True
    try:
        log.info(f"Quantizando embedding model int8 (uma única vez): {p.name} → {int8_path.name}")
        quantize_dynamic(fp32_path, str(int8_path), weight_type=QuantType.QInt8)
        log.info(f"Modelo int8 salvo: {int8_path} ({int8_path.stat().st_size // (1024*1024)} MB)")
        return str(int8_path), True
    except Exception as e:
        log.warning(f"Falha ao quantizar ({e}) — usando fp32")
        return fp32_path, False


# ── Embedding Model ────────────────────────────────────────────────────────────

class EmbeddingModel:
    """ONNX bi-encoder: multilingual-e5-small with mean pooling + L2 norm."""

    def __init__(self, model_path: str, tokenizer_path: str):
        opts = _memory_opts(EMBED_THREADS, 1)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        model_path, used_int8 = _ensure_quantized(model_path)
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

        # Chunking: bounda o tamanho de cada run do ORT (arena/alocador não
        # explode com requests grandes) preservando a ordem dos resultados.
        chunks = [texts[i:i + EMBED_MAX_BATCH] for i in range(0, len(texts), EMBED_MAX_BATCH)]
        out: list[np.ndarray] = []
        for chunk in chunks:
            out.append(self._embed_batch(chunk))
        return np.concatenate(out, axis=0)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
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
        opts = _memory_opts(RERANK_THREADS, 1)

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

        # Chunking igual ao embed: bounda cada run do ORT (passages de 256 ×
        # 512 tokens alocavam centenas de MB por request).
        chunks = [passages[i:i + RERANK_MAX_BATCH] for i in range(0, len(passages), RERANK_MAX_BATCH)]
        return np.concatenate([self._score_batch(query, c) for c in chunks]).astype(np.float32)

    def _score_batch(self, query: str, passages: list[str]) -> np.ndarray:
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

# Exit code EX_CONFIG (sysexits): config/dados irrecuperáveis. O supervisor
# restart_on_fail do start.sh NÃO reinicia serviços que saíram com 78 —
# sem isso um modelo corrompido gerava crash-loop eterno (restart a cada 5s).
EXIT_CONFIG = 78

# Piso de tamanho de um .onnx válido — e5-small/MiniLM têm dezenas de MB.
# Arquivo truncado (download/zip interrompido) fica bem abaixo disso.
MIN_MODEL_BYTES = 10 * 1024 * 1024

_FATAL_FIX_HINT = (
    "Correção: rode `python3 scripts/verify_models.py` (revalida/rebaixa os "
    "modelos) e reinicie o container. O serviço fica fora até o modelo "
    "ser reparado — reiniciar em loop não resolve."
)


def _fatal_exit(msg: str) -> None:
    """Erro de startup irrecuperável: loga acionável e sai com EX_CONFIG.
    os._exit garante o código no container (o uvicorn captura exceções de
    lifespan e sairia com o código DELE, mascarando o 78)."""
    for h in log.handlers:
        try:
            h.flush()
        except Exception:
            pass
    log.error(msg)
    os._exit(EXIT_CONFIG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting AVA ONNX Serving...")

    # ── Validação de integridade dos modelos ──
    # ONNX truncado/corrompido (download ou extração interrompida — caso
    # real em instalação Windows) estoura InvalidProtobuf em
    # ort.InferenceSession. Valida existência + tamanho mínimo ANTES e
    # captura falha de carga com mensagem acionável — em vez de crash
    # opaque + loop de restart.
    for path, label in [
        (EMBED_MODEL_PATH,  "Embedding model"),
        (EMBED_TOKENIZER,   "Embedding tokenizer"),
        (RERANK_MODEL_PATH, "Reranker model"),
        (RERANK_TOKENIZER,  "Reranker tokenizer"),
    ]:
        p = Path(path)
        if not p.exists():
            _fatal_exit(f"[FATAL] {label} não encontrado: {path} — {_FATAL_FIX_HINT}")
        if p.suffix == ".onnx" and p.stat().st_size < MIN_MODEL_BYTES:
            _fatal_exit(
                f"[FATAL] {label} TRUNCADO/corrompido: {path} tem "
                f"{p.stat().st_size} bytes (< {MIN_MODEL_BYTES} esperados) — {_FATAL_FIX_HINT}"
            )

    try:
        state.embed_model  = EmbeddingModel(EMBED_MODEL_PATH, EMBED_TOKENIZER)
        state.rerank_model = RerankerModel(RERANK_MODEL_PATH, RERANK_TOKENIZER)
    except Exception as e:
        name = type(e).__name__
        hint = ""
        if "INVALID_PROTOBUF" in str(e) or "Protobuf" in name:
            hint = f" — arquivo do modelo CORROMPIDO. {_FATAL_FIX_HINT}"
        _fatal_exit(f"[FATAL] falha ao carregar modelos ONNX: {name}: {e}{hint}")

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

    # Watchdog de memória: devolve heap glibc ao OS a cada minuto.
    _malloc_trim_watchdog()

    state.start_time = time.time()
    log.info(
        f"AVA ONNX Serving ready — embed({state.embed_model.provider}, "
        f"dim={EMBED_DIM}, max_batch={EMBED_MAX_BATCH}) "
        f"+ rerank({state.rerank_model.provider})"
    )
    yield

    state.embed_pool.shutdown(wait=False)
    state.rerank_pool.shutdown(wait=False)
    log.info("AVA ONNX Serving shut down")


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
        embed_model    = Path(EMBED_MODEL_PATH).stem,
        rerank_model   = Path(RERANK_MODEL_PATH).stem,
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