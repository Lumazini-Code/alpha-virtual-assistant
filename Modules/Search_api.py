from __future__ import annotations

import time
import hashlib
import asyncio
import logging
import concurrent.futures
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional
from collections import OrderedDict
from pathlib import Path

import numpy as np
import yake
import onnxruntime as ort
from ddgs import DDGS
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from tokenizers import Tokenizer
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False

# ── Configuração ───────────────────────────────────────────────────────────────

CROSS_ENCODER_MODEL_PATH = "Models/ms-marco-MiniLM-L-6-v2/ms-marco-MiniLM-L-6-v2.onnx"
CROSS_ENCODER_TOKENIZER  = "Models/ms-marco-MiniLM-L-6-v2/tokenizer.json"

MAX_DDG_RESULTS       = 8     # snippets buscados — mais que o retornado para o ranqueador ter mais opções
TOP_RESULTS_FINAL     = 5     # snippets retornados na resposta
CROSS_ENCODER_THREADS = 4
DDG_TIMEOUT_S         = 12.0

CACHE_TTL_SECONDS     = 3600
CACHE_MAX_SIZE        = 128

_IO_POOL  = concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="io")
_CPU_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ava.search")

# ── Modelos de request/response ────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query:       str
    max_results: int  = TOP_RESULTS_FINAL
    use_cache:   bool = True

class SearchResult(BaseModel):
    text:   str
    score:  float
    source: str
    title:  str

class SearchResponse(BaseModel):
    results:    list[SearchResult]
    query:      str
    from_cache: bool
    latency_ms: float

# ── Cache TTL ──────────────────────────────────────────────────────────────────

class TTLCache:
    def __init__(self, max_size: int = CACHE_MAX_SIZE, ttl: float = CACHE_TTL_SECONDS):
        self._cache:    OrderedDict[str, dict] = OrderedDict()
        self._max_size = max_size
        self._ttl      = ttl

    def _key(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def get(self, query: str) -> Optional[list]:
        key   = self._key(query)
        entry = self._cache.get(key)
        if not entry:
            return None
        if time.time() - entry["ts"] > self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return entry["data"]

    def put(self, query: str, data: list):
        key = self._key(query)
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = {"data": data, "ts": time.time()}

    def clear(self):
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)

# ── Tokenizer ──────────────────────────────────────────────────────────────────

class FastTokenizer:
    def __init__(self, path: str, max_length: int = 512):
        self._max_length = max_length
        if HAS_TOKENIZERS:
            self._tok = Tokenizer.from_file(path)
            self._tok.enable_padding(length=max_length)
            self._tok.enable_truncation(max_length=max_length)
            self._backend = "tokenizers"
        else:
            self._backend = "simple"
            log.warning("tokenizers não encontrado — qualidade reduzida")

    def encode_pairs(self, pairs: list[tuple[str, str]]) -> dict[str, np.ndarray]:
        if self._backend == "tokenizers":
            texts   = [f"{q} [SEP] {p}" for q, p in pairs]
            encoded = self._tok.encode_batch(texts)
            return {
                "input_ids":      np.array([e.ids            for e in encoded], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
                "token_type_ids": np.zeros((len(pairs), self._max_length), dtype=np.int64),
            }
        ml = self._max_length
        ids, masks = [], []
        for q, p in pairs:
            tokens = (q + " " + p).lower().split()[:ml - 2]
            pad    = ml - len(tokens) - 2
            ids.append([101] + [hash(w) % 30000 + 100 for w in tokens] + [102] + [0] * pad)
            masks.append([1] * (len(tokens) + 2) + [0] * pad)
        return {
            "input_ids":      np.array(ids,   dtype=np.int64),
            "attention_mask": np.array(masks, dtype=np.int64),
            "token_type_ids": np.zeros((len(pairs), ml), dtype=np.int64),
        }

# ── Cross-encoder ONNX ─────────────────────────────────────────────────────────

class CrossEncoder:
    def __init__(self, model_path: str, tokenizer_path: str):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads    = CROSS_ENCODER_THREADS
        opts.inter_op_num_threads    = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode          = ort.ExecutionMode.ORT_SEQUENTIAL

        self._session   = ort.InferenceSession(
            model_path, opts, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = FastTokenizer(tokenizer_path)
        log.info("CrossEncoder carregado")

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.array([], dtype=np.float32)

        pairs  = [(query, p) for p in passages]
        inputs = self._tokenizer.encode_pairs(pairs)

        output = self._session.run(None, {
            "input_ids":      inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs["token_type_ids"],
        })
        # logits shape (N, 1) — squeeze para (N,)
        logits = output[0].squeeze(-1).astype(np.float32)

        # Normaliza por min-max para range 0–1 preservando ordem relativa
        # Sigmoid em logits não calibrados do ms-marco produz scores ~0.0001
        if len(logits) == 1:
            return np.array([1.0], dtype=np.float32)
        lo, hi = logits.min(), logits.max()
        if hi - lo < 1e-6:
            return np.ones(len(logits), dtype=np.float32)
        return ((logits - lo) / (hi - lo)).astype(np.float32)

# ── Keyword extractor ──────────────────────────────────────────────────────────

class KeywordExtractor:
    def __init__(self):
        self._extractor = yake.KeywordExtractor(
            lan="pt", n=2, dedupLim=0.9, top=6, features=None
        )

    def extract_query(self, text: str) -> str:
        try:
            keywords = self._extractor.extract_keywords(text)
            terms    = [kw for kw, _ in sorted(keywords, key=lambda x: x[1])[:5]]
            return " ".join(terms).strip() or text
        except Exception:
            return text

# ── DuckDuckGo ─────────────────────────────────────────────────────────────────

def _ddg_search_sync(query: str, max_results: int) -> list[dict]:
    try:
        with DDGS(timeout=10) as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        log.warning(f"DuckDuckGo falhou: {type(e).__name__}: {e}")
        return []

# ── Estado global ──────────────────────────────────────────────────────────────

@dataclass
class AppState:
    cross_encoder:     CrossEncoder     = field(default=None)
    keyword_extractor: KeywordExtractor = field(default=None)
    cache:             TTLCache         = field(default=None)

state = AppState()

# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Iniciando AVA Search API...")

    if not Path(CROSS_ENCODER_MODEL_PATH).exists():
        raise RuntimeError(f"Modelo não encontrado: {CROSS_ENCODER_MODEL_PATH}")
    if not Path(CROSS_ENCODER_TOKENIZER).exists():
        raise RuntimeError(f"Tokenizer não encontrado: {CROSS_ENCODER_TOKENIZER}")

    state.cross_encoder     = CrossEncoder(CROSS_ENCODER_MODEL_PATH, CROSS_ENCODER_TOKENIZER)
    state.keyword_extractor = KeywordExtractor()
    state.cache             = TTLCache()

    log.info("Search API pronta")
    yield

    _IO_POOL.shutdown(wait=False)
    _CPU_POOL.shutdown(wait=False)
    log.info("AVA Search API encerrada")

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="AVA Search API", lifespan=lifespan)

# ── POST /search ───────────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query vazia")

    t0   = time.perf_counter()
    loop = asyncio.get_event_loop()

    # Cache
    if req.use_cache:
        cached = state.cache.get(query)
        if cached:
            return SearchResponse(
                results    = [SearchResult(**r) for r in cached],
                query      = query,
                from_cache = True,
                latency_ms = round((time.perf_counter() - t0) * 1000, 2),
            )

    # 1. Keywords
    search_query = await loop.run_in_executor(
        _IO_POOL, state.keyword_extractor.extract_query, query
    )
    log.info(f"Query → '{search_query}'")

    # 2. DuckDuckGo com timeout de segurança
    try:
        ddg_results = await asyncio.wait_for(
            loop.run_in_executor(_IO_POOL, _ddg_search_sync, search_query, MAX_DDG_RESULTS),
            timeout=DDG_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        log.warning("DuckDuckGo timeout")
        return SearchResponse(results=[], query=query, from_cache=False,
                              latency_ms=round((time.perf_counter() - t0) * 1000, 2))

    if not ddg_results:
        return SearchResponse(results=[], query=query, from_cache=False,
                              latency_ms=round((time.perf_counter() - t0) * 1000, 2))

    # 3. Extrai snippets válidos preservando referência ao resultado original
    indexed = [(i, r) for i, r in enumerate(ddg_results) if r.get("body", "").strip()]
    if not indexed:
        return SearchResponse(results=[], query=query, from_cache=False,
                              latency_ms=round((time.perf_counter() - t0) * 1000, 2))

    snippets = [r["body"] for _, r in indexed]

    # 4. Cross-encoder ranqueia no _CPU_POOL
    scores = await loop.run_in_executor(
        _CPU_POOL, state.cross_encoder.score, query, snippets
    )

    # 5. Monta resultados ranqueados
    ranked = sorted(zip(scores, indexed), key=lambda x: x[0], reverse=True)
    results = []
    for score, (_, r) in ranked[:req.max_results]:
        results.append(SearchResult(
            text   = r["body"],
            score  = round(float(score), 4),
            source = r.get("href", ""),
            title  = r.get("title", ""),
        ))

    latency = round((time.perf_counter() - t0) * 1000, 2)
    log.info(f"Busca em {latency}ms — {len(results)} resultados: {query[:50]}")

    if req.use_cache:
        state.cache.put(query, [r.model_dump() for r in results])

    return SearchResponse(results=results, query=query, from_cache=False,
                          latency_ms=latency)

# ── GET /status ────────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    return {
        "cache_size":        state.cache.size,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "max_ddg_results":   MAX_DDG_RESULTS,
        "top_results":       TOP_RESULTS_FINAL,
    }

# ── DELETE /cache ──────────────────────────────────────────────────────────────

@app.delete("/cache")
async def clear_cache():
    state.cache.clear()
    return {"cleared": True}

# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3002, log_level="info")