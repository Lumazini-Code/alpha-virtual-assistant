from __future__ import annotations

import io
import re
import time
import hashlib
import asyncio
import logging
import tempfile
import concurrent.futures
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional
from collections import OrderedDict
from pathlib import Path

import numpy as np
import httpx
import yake
import pdfplumber
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

MAX_DDG_RESULTS       = 8     # snippets buscados pelo DDG
MAX_PDF_RESULTS       = 5     # PDFs buscados para download
TOP_RESULTS_FINAL     = 5     # snippets retornados na resposta final
CROSS_ENCODER_THREADS = 4
DDG_TIMEOUT_S         = 12.0

PDF_DOWNLOAD_TIMEOUT_S = 20.0  # timeout por PDF download
PDF_MAX_BYTES          = 20 * 1024 * 1024  # 20 MB — ignora PDFs maiores
PDF_CHUNK_SIZE         = 400   # ~palavras por chunk
PDF_CHUNK_OVERLAP      = 80    # sobreposição entre chunks para não perder contexto

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
    search_pdfs: bool = False  # ativa busca em PDFs encontrados na internet

class SearchResult(BaseModel):
    text:      str
    score:     float
    source:    str
    title:     str
    from_pdf:  bool  = False   # True quando o trecho veio de um PDF
    pdf_chunk: Optional[int] = None  # índice do chunk dentro do PDF

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

    def _key(self, query: str, search_pdfs: bool) -> str:
        raw = f"{query.lower().strip()}|pdf={search_pdfs}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, query: str, search_pdfs: bool) -> Optional[list]:
        key   = self._key(query, search_pdfs)
        entry = self._cache.get(key)
        if not entry:
            return None
        if time.time() - entry["ts"] > self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return entry["data"]

    def put(self, query: str, search_pdfs: bool, data: list):
        key = self._key(query, search_pdfs)
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
        opts.intra_op_num_threads     = CROSS_ENCODER_THREADS
        opts.inter_op_num_threads     = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL

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
        logits = output[0].squeeze(-1).astype(np.float32)

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

# ── PDF utils ──────────────────────────────────────────────────────────────────

def _extract_pdf_chunks(pdf_bytes: bytes, chunk_size: int = PDF_CHUNK_SIZE,
                         overlap: int = PDF_CHUNK_OVERLAP) -> list[tuple[int, str]]:
    """
    Extrai texto de um PDF em memória e divide em chunks sobrepostos.
    Retorna lista de (chunk_index, texto).
    """
    chunks: list[tuple[int, str]] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            full_text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
    except Exception as e:
        log.warning(f"pdfplumber falhou: {e}")
        return chunks

    # Normaliza espaços e quebras excessivas
    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()
    if not full_text:
        return chunks

    words = full_text.split()
    step  = max(1, chunk_size - overlap)

    for idx, start in enumerate(range(0, len(words), step)):
        chunk = " ".join(words[start : start + chunk_size])
        if len(chunk.strip()) > 60:          # descarta chunks trivialmente curtos
            chunks.append((idx, chunk))

    return chunks


def _ddg_search_sync(query: str, max_results: int) -> list[dict]:
    try:
        with DDGS(timeout=10) as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        log.warning(f"DuckDuckGo falhou: {type(e).__name__}: {e}")
        return []


def _ddg_pdf_search_sync(query: str, max_results: int) -> list[dict]:
    """Busca no DDG restringindo a arquivos PDF (filetype:pdf)."""
    pdf_query = f"{query} filetype:pdf"
    try:
        with DDGS(timeout=10) as ddgs:
            results = list(ddgs.text(pdf_query, max_results=max_results * 2))
        # Filtra apenas URLs que terminam em .pdf ou passam por redirecionadores comuns
        pdf_results = [
            r for r in results
            if r.get("href", "").lower().endswith(".pdf")
            or "pdf" in r.get("href", "").lower()
        ]
        return pdf_results[:max_results]
    except Exception as e:
        log.warning(f"DuckDuckGo PDF falhou: {type(e).__name__}: {e}")
        return []


async def _download_pdf(url: str, client: httpx.AsyncClient) -> Optional[bytes]:
    """Baixa um PDF de forma assíncrona respeitando limite de tamanho."""
    try:
        async with client.stream("GET", url, timeout=PDF_DOWNLOAD_TIMEOUT_S) as resp:
            if resp.status_code != 200:
                return None
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                return None

            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes(chunk_size=65536):
                total += len(chunk)
                if total > PDF_MAX_BYTES:
                    log.warning(f"PDF muito grande (>{PDF_MAX_BYTES // 1024 // 1024}MB): {url}")
                    return None
                chunks.append(chunk)
            return b"".join(chunks)
    except Exception as e:
        log.warning(f"Download falhou [{url}]: {type(e).__name__}: {e}")
        return None


async def fetch_pdf_chunks(
    pdf_refs: list[dict],
    query: str,
    cross_encoder: CrossEncoder,
    loop: asyncio.AbstractEventLoop,
    top_k: int = TOP_RESULTS_FINAL,
) -> list[SearchResult]:
    """
    Baixa PDFs em paralelo, extrai chunks, reranqueia com o cross-encoder
    e retorna os top_k melhores trechos.
    """
    if not pdf_refs:
        return []

    # Download paralelo com um único cliente httpx
    async with httpx.AsyncClient(
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; AVASearch/1.0)"},
    ) as client:
        download_tasks = [_download_pdf(r["href"], client) for r in pdf_refs]
        pdf_bytes_list = await asyncio.gather(*download_tasks)

    # Extrai chunks de cada PDF baixado com sucesso
    all_chunks: list[tuple[str, str, int]] = []  # (chunk_text, source_url, chunk_idx)
    for ref, pdf_bytes in zip(pdf_refs, pdf_bytes_list):
        if pdf_bytes is None:
            continue
        chunks = await loop.run_in_executor(
            _CPU_POOL, _extract_pdf_chunks, pdf_bytes
        )
        url = ref.get("href", "")
        for chunk_idx, chunk_text in chunks:
            all_chunks.append((chunk_text, url, chunk_idx))

    if not all_chunks:
        log.info("Nenhum chunk extraído dos PDFs")
        return []

    log.info(f"{len(all_chunks)} chunks extraídos de {len(pdf_refs)} PDFs")

    # Reranqueia todos os chunks com o cross-encoder
    passages = [c[0] for c in all_chunks]
    scores   = await loop.run_in_executor(
        _CPU_POOL, cross_encoder.score, query, passages
    )

    # Seleciona top_k chunks por score
    ranked = sorted(
        zip(scores, all_chunks),
        key=lambda x: x[0],
        reverse=True,
    )

    results: list[SearchResult] = []
    for score, (chunk_text, source_url, chunk_idx) in ranked[:top_k]:
        # Encontra o título do PDF na lista original
        title = next(
            (r.get("title", source_url) for r in pdf_refs if r.get("href") == source_url),
            source_url,
        )
        results.append(SearchResult(
            text      = chunk_text,
            score     = round(float(score), 4),
            source    = source_url,
            title     = title,
            from_pdf  = True,
            pdf_chunk = chunk_idx,
        ))

    return results

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

    # Cache — chave diferente se search_pdfs=True para não misturar resultados
    if req.use_cache:
        cached = state.cache.get(query, req.search_pdfs)
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

    # 2. DDG texto + DDG PDF em paralelo (se search_pdfs=True)
    ddg_text_task = asyncio.wait_for(
        loop.run_in_executor(_IO_POOL, _ddg_search_sync, search_query, MAX_DDG_RESULTS),
        timeout=DDG_TIMEOUT_S,
    )

    if req.search_pdfs:
        ddg_pdf_task = asyncio.wait_for(
            loop.run_in_executor(_IO_POOL, _ddg_pdf_search_sync, search_query, MAX_PDF_RESULTS),
            timeout=DDG_TIMEOUT_S,
        )
        gathered = await asyncio.gather(ddg_text_task, ddg_pdf_task, return_exceptions=True)
        ddg_results  = gathered[0] if not isinstance(gathered[0], Exception) else []
        pdf_refs     = gathered[1] if not isinstance(gathered[1], Exception) else []
    else:
        try:
            ddg_results = await ddg_text_task
        except asyncio.TimeoutError:
            log.warning("DuckDuckGo timeout")
            ddg_results = []
        pdf_refs = []

    # 3. Pipeline de texto (igual ao original)
    text_results: list[SearchResult] = []
    indexed = [(i, r) for i, r in enumerate(ddg_results) if r.get("body", "").strip()]
    if indexed:
        snippets = [r["body"] for _, r in indexed]
        scores   = await loop.run_in_executor(
            _CPU_POOL, state.cross_encoder.score, query, snippets
        )
        ranked = sorted(zip(scores, indexed), key=lambda x: x[0], reverse=True)
        for score, (_, r) in ranked[:req.max_results]:
            text_results.append(SearchResult(
                text     = r["body"],
                score    = round(float(score), 4),
                source   = r.get("href", ""),
                title    = r.get("title", ""),
                from_pdf = False,
            ))

    # 4. Pipeline de PDF (se ativado)
    pdf_results: list[SearchResult] = []
    if req.search_pdfs and pdf_refs:
        log.info(f"{len(pdf_refs)} PDFs encontrados — iniciando download e extração")
        pdf_results = await fetch_pdf_chunks(
            pdf_refs      = pdf_refs,
            query         = query,
            cross_encoder = state.cross_encoder,
            loop          = loop,
            top_k         = req.max_results,
        )

    # 5. Merge e reranqueamento final entre texto e PDF
    all_results = text_results + pdf_results
    if req.search_pdfs and all_results:
        # Re-normaliza scores entre as duas fontes para comparação justa
        all_scores = np.array([r.score for r in all_results], dtype=np.float32)
        lo, hi = all_scores.min(), all_scores.max()
        if hi - lo > 1e-6:
            all_scores = (all_scores - lo) / (hi - lo)
        for r, s in zip(all_results, all_scores):
            r.score = round(float(s), 4)
        all_results.sort(key=lambda r: r.score, reverse=True)

    results = all_results[:req.max_results]

    latency = round((time.perf_counter() - t0) * 1000, 2)
    log.info(
        f"Busca em {latency}ms — {len(text_results)} texto + {len(pdf_results)} PDF → "
        f"{len(results)} retornados: {query[:50]}"
    )

    if req.use_cache:
        state.cache.put(query, req.search_pdfs, [r.model_dump() for r in results])

    return SearchResponse(
        results    = results,
        query      = query,
        from_cache = False,
        latency_ms = latency,
    )

# ── GET /status ────────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    return {
        "cache_size":           state.cache.size,
        "cache_ttl_seconds":    CACHE_TTL_SECONDS,
        "max_ddg_results":      MAX_DDG_RESULTS,
        "max_pdf_results":      MAX_PDF_RESULTS,
        "top_results":          TOP_RESULTS_FINAL,
        "pdf_chunk_size_words": PDF_CHUNK_SIZE,
        "pdf_chunk_overlap":    PDF_CHUNK_OVERLAP,
        "pdf_max_bytes":        PDF_MAX_BYTES,
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