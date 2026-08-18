from __future__ import annotations
import os
import re
import time
import hashlib
import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Literal, Optional
from collections import OrderedDict
from urllib.parse import urlparse
from dotenv import load_dotenv
load_dotenv()  # carrega variáveis de ambiente do .env local (não version
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Configuração ─────────────────────────────────────────────────────────────
# Toda a infraestrutura de busca (antes: DDG + fetch próprio via httpx/
# trafilatura + PDF via pdfplumber) foi substituída pela API da Tavily
# (https://tavily.com), que é feita pra consumo por agentes/LLMs: já devolve
# resultados ranqueados por relevância + conteúdo real da página (não só um
# snippet) numa única chamada. Isso elimina o passo separado de "buscar
# depois enriquecer" que a versão DDG precisava, e reduz o número de
# requisições — o que importa porque o free tier da Tavily é limitado em
# CRÉDITOS por mês, não em requisições soltas.

TAVILY_API_KEY  = os.environ.get("TAVILY_API_KEY", "").strip()
TAVILY_BASE_URL = "https://api.tavily.com"
TAVILY_TIMEOUT_S = 20.0

# Créditos da Tavily (free tier = 1000/mês, ver docs.tavily.com):
#   /search com search_depth="basic"    -> 1 crédito
#   /search com search_depth="advanced" -> 2 créditos
#   /extract                            -> cobrado por URL processada com
#                                          sucesso (lotes), não é 1:1 por
#                                          chamada — o contador abaixo é uma
#                                          APROXIMAÇÃO pra monitoramento, não
#                                          o número exato de créditos. Pro
#                                          número real, consulte o dashboard
#                                          da Tavily.
#
# Estratégia pra esticar o free tier ao máximo:
#   - search_depth fica em "basic" por padrão (1 crédito) SEMPRE combinado
#     com include_raw_content=True — isso já devolve o conteúdo real da
#     página no mesmo crédito de uma busca simples, então não pagamos o
#     dobro (search_depth="advanced") só pra ter conteúdo completo.
#   - Cache TTL agressivo (ver CACHE_TTL_SECONDS) evita queimar crédito de
#     novo em queries repetidas dentro da janela de cache.
#   - PDFs: primeiro tentamos usar o raw_content que a própria busca já
#     devolve pro PDF: só chamamos /extract (custo adicional) pros PDFs
#     onde a busca não trouxe conteúdo suficiente.
SEARCH_DEPTH_DEFAULT = "basic"
TOPIC_DEFAULT: Literal["general", "news"] = "general"

MAX_RESULTS_HARD_CAP = 20   # limite aceito pela API da Tavily por chamada
TOP_RESULTS_FINAL    = 5
PDF_MAX_RESULTS       = 5

PDF_CHUNK_SIZE        = 400
PDF_CHUNK_OVERLAP     = 80
PAGE_TEXT_MAX_CHARS   = 3000

CACHE_TTL_SECONDS = int(os.environ.get("SEARCH_CACHE_TTL_SECONDS", 3600))
CACHE_MAX_SIZE    = 128

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [Search API] %(message)s")
log = logging.getLogger("ava.search")

if not TAVILY_API_KEY:
    log.warning(
        "TAVILY_API_KEY não definida no ambiente — configure antes de subir o "
        "serviço; toda chamada a /search ou /extract vai falhar com 502 até lá."
    )

# ── Modelos de request/response ────────────────────────────────────────────

class SearchRequest(BaseModel):
    query:       str
    max_results: int  = TOP_RESULTS_FINAL
    use_cache:   bool = True
    search_pdfs: bool = False
    # Campos novos, todos opcionais e retrocompatíveis: quem já chamava o
    # serviço sem eles continua funcionando exatamente igual (só que com
    # resultados melhores, já que a Tavily rankeia por relevância).
    topic:           Literal["general", "news"] = TOPIC_DEFAULT
    days:            Optional[int]       = None   # só com topic="news": limita aos últimos N dias
    include_domains: Optional[list[str]] = None
    exclude_domains: Optional[list[str]] = None

class SearchResult(BaseModel):
    text:      str
    source:    str
    title:     str
    from_pdf:  bool  = False
    pdf_chunk: Optional[int]   = None
    score:     Optional[float] = None   # score de relevância devolvido pela Tavily

class SearchResponse(BaseModel):
    results:    list[SearchResult]
    query:      str
    from_cache: bool
    latency_ms: float
    answer:     Optional[str] = None   # resposta curta gerada pela Tavily (include_answer)

class ExtractRequest(BaseModel):
    urls:          list[str]
    extract_depth: Literal["basic", "advanced"] = "basic"

class ExtractedPage(BaseModel):
    url:     str
    title:   str = ""
    text:    str
    success: bool = True

class ExtractResponse(BaseModel):
    results:    list[ExtractedPage]
    failed:     list[str]
    latency_ms: float

# ── Cache TTL ────────────────────────────────────────────────────────────────

class TTLCache:
    def __init__(self, max_size: int = CACHE_MAX_SIZE, ttl: float = CACHE_TTL_SECONDS):
        self._cache:    OrderedDict[str, dict] = OrderedDict()
        self._max_size = max_size
        self._ttl      = ttl

    def _key(self, req: SearchRequest) -> str:
        raw = (
            f"{req.query.lower().strip()}|pdf={req.search_pdfs}|topic={req.topic}|"
            f"days={req.days}|inc={sorted(req.include_domains or [])}|"
            f"exc={sorted(req.exclude_domains or [])}|max={req.max_results}"
        )
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, req: SearchRequest) -> Optional[list]:
        key   = self._key(req)
        entry = self._cache.get(key)
        if not entry:
            return None
        if time.time() - entry["ts"] > self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return entry["data"]

    def put(self, req: SearchRequest, data: list, answer: Optional[str]):
        key = self._key(req)
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = {"data": data, "answer": answer, "ts": time.time()}

    def clear(self):
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


# ── Filtro de páginas-índice (homepages) ────────────────────────────────────
# A Tavily rankeia bem a homepage de portais grandes (NYT, Guardian, etc.)
# pra queries amplas/genéricas ("world news today", "notícias hoje"). Uma
# homepage não é um artigo: é navegação + categorias + trechos soltos de
# várias matérias diferentes, e o LLM não tem como distinguir isso de fato
# noticiado — por isso essas URLs são descartadas antes de entrar no
# resultado final, em vez de devolvidas como se fossem uma matéria.
_INDEX_PATH_RE = re.compile(r"^/(index\.\w+|home\.\w+|news/?|noticias/?)?$", re.IGNORECASE)

def _is_index_page(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    path = parsed.path or "/"
    return not parsed.query and bool(_INDEX_PATH_RE.match(path))


def _clean_text(text: str, max_chars: int = PAGE_TEXT_MAX_CHARS) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:max_chars]


def _chunk_text(text: str, chunk_size: int = PDF_CHUNK_SIZE,
                 overlap: int = PDF_CHUNK_OVERLAP) -> list[tuple[int, str]]:
    """Quebra um texto longo (ex: PDF já extraído pela Tavily) em chunks
    sobrepostos, na mesma lógica usada antes pra texto extraído via
    pdfplumber — só que agora opera direto sobre texto, sem precisar baixar
    e parsear o PDF localmente."""
    text = re.sub(r"\n{3,}", "\n\n", text or "").strip()
    if not text:
        return []
    words = text.split()
    step  = max(1, chunk_size - overlap)
    chunks: list[tuple[int, str]] = []
    for idx, start in enumerate(range(0, len(words), step)):
        chunk = " ".join(words[start: start + chunk_size])
        if len(chunk.strip()) > 60:
            chunks.append((idx, chunk))
    return chunks


# ── Contador aproximado de créditos ─────────────────────────────────────────
# Só pra monitoramento (exposto em /status) — não é a fonte de verdade de
# billing da Tavily, que reseta o mês e cobra em lotes pro /extract.

@dataclass
class _CreditTracker:
    month:           str = field(default_factory=lambda: time.strftime("%Y-%m"))
    search_basic:    int = 0
    search_advanced: int = 0
    extract_calls:   int = 0

    def _roll_if_new_month(self):
        cur = time.strftime("%Y-%m")
        if cur != self.month:
            log.info(f"Novo mês ({cur}) — zerando contador aproximado de créditos Tavily")
            self.month = cur
            self.search_basic = self.search_advanced = self.extract_calls = 0

    def record_search(self, depth: str):
        self._roll_if_new_month()
        if depth == "advanced":
            self.search_advanced += 1
        else:
            self.search_basic += 1

    def record_extract(self):
        self._roll_if_new_month()
        self.extract_calls += 1

    @property
    def estimated_credits(self) -> int:
        return self.search_basic * 1 + self.search_advanced * 2 + self.extract_calls


# ── Cliente Tavily ───────────────────────────────────────────────────────────

async def _tavily_post(client: httpx.AsyncClient, path: str, payload: dict) -> dict:
    if not TAVILY_API_KEY:
        raise HTTPException(status_code=502, detail="TAVILY_API_KEY não configurada no servidor")
    try:
        resp = await client.post(path, json=payload, timeout=TAVILY_TIMEOUT_S)
    except httpx.TimeoutException:
        log.warning(f"Tavily timeout em {path}")
        raise HTTPException(status_code=504, detail="Tavily timeout")
    except httpx.HTTPError as e:
        log.warning(f"Tavily erro de rede em {path}: {type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail=f"Tavily erro de rede: {e}")

    if resp.status_code == 401:
        raise HTTPException(status_code=502, detail="Tavily rejeitou a API key (401) — confira TAVILY_API_KEY")
    if resp.status_code == 429:
        log.warning("Tavily: rate limit ou créditos do mês esgotados (429)")
        raise HTTPException(status_code=502, detail="Tavily: rate limit ou créditos esgotados (429)")
    if resp.status_code >= 400:
        log.warning(f"Tavily {resp.status_code} em {path}: {resp.text[:300]}")
        raise HTTPException(status_code=502, detail=f"Tavily retornou {resp.status_code}")

    return resp.json()


async def _tavily_search(
    client: httpx.AsyncClient,
    query: str,
    max_results: int,
    topic: str,
    days: Optional[int],
    include_domains: Optional[list[str]],
    exclude_domains: Optional[list[str]],
    search_depth: str = SEARCH_DEPTH_DEFAULT,
) -> dict:
    payload: dict = {
        "query":               query,
        "search_depth":        search_depth,
        "topic":                topic,
        "max_results":         max(1, min(max_results, MAX_RESULTS_HARD_CAP)),
        "include_answer":      "basic",
        "include_raw_content": "markdown",
        "include_images":      False,
    }
    if topic == "news" and days:
        payload["days"] = days
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains

    state.credits.record_search(search_depth)
    return await _tavily_post(client, "/search", payload)


async def _tavily_extract(client: httpx.AsyncClient, urls: list[str],
                           extract_depth: str = "basic") -> dict:
    if not urls:
        return {"results": [], "failed_results": []}
    payload = {
        "urls":          urls[:20],   # limite por chamada da API
        "extract_depth": extract_depth,
        "format":        "markdown",
    }
    state.credits.record_extract()
    return await _tavily_post(client, "/extract", payload)


# ── PDFs ─────────────────────────────────────────────────────────────────────
# A Tavily não tem um operador `filetype:pdf` documentado/garantido — então
# a estratégia é: (1) pedir a busca normal com "filetype:pdf" anexado à
# query como dica pro ranking, (2) filtrar client-side só URLs que terminam
# em .pdf, (3) usar o raw_content que a própria busca já trouxe quando for
# suficiente, e só chamar /extract (custo extra) pros PDFs em que a busca
# não trouxe conteúdo — bem mais barato que baixar+parsear cada PDF
# localmente como a versão anterior fazia com pdfplumber.

async def _search_pdfs(client: httpx.AsyncClient, query: str, top_k: int) -> list[SearchResult]:
    data = await _tavily_search(
        client, query=f"{query} filetype:pdf", max_results=PDF_MAX_RESULTS * 2,
        topic="general", days=None, include_domains=None, exclude_domains=None,
    )
    raw_results = [
        r for r in data.get("results", [])
        if r.get("url", "").lower().split("?")[0].endswith(".pdf")
    ][:PDF_MAX_RESULTS]

    if not raw_results:
        return []

    # Separa os que já vieram com conteúdo útil dos que precisam de /extract
    needs_extract = [r["url"] for r in raw_results if len(r.get("raw_content") or "") < 200]
    extracted_by_url: dict[str, str] = {}
    if needs_extract:
        try:
            ex = await _tavily_extract(client, needs_extract, extract_depth="advanced")
            for item in ex.get("results", []):
                extracted_by_url[item.get("url", "")] = item.get("raw_content", "") or ""
        except HTTPException as e:
            log.warning(f"Extract de PDF falhou, seguindo só com o que a busca trouxe: {e.detail}")

    pdf_results: list[SearchResult] = []
    for r in raw_results:
        url  = r.get("url", "")
        text = r.get("raw_content") or extracted_by_url.get(url, "") or r.get("content", "")
        chunks = _chunk_text(text)
        if not chunks:
            continue
        title = r.get("title", url)
        for chunk_idx, chunk_text in chunks[: max(1, top_k)]:
            pdf_results.append(SearchResult(
                text      = chunk_text,
                source    = url,
                title     = title,
                from_pdf  = True,
                pdf_chunk = chunk_idx,
            ))

    return pdf_results[:top_k]


# ── Estado global ────────────────────────────────────────────────────────────

@dataclass
class AppState:
    cache:         TTLCache               = field(default=None)
    credits:       _CreditTracker         = field(default=None)
    tavily_client: httpx.AsyncClient      = field(default=None)

state = AppState()


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Iniciando AVA Search API (Tavily)...")

    state.cache   = TTLCache()
    state.credits = _CreditTracker()
    state.tavily_client = httpx.AsyncClient(
        base_url = TAVILY_BASE_URL,
        headers  = {
            "Authorization": f"Bearer {TAVILY_API_KEY}",
            "Content-Type":  "application/json",
        },
    )

    log.info("Search API pronta" + ("" if TAVILY_API_KEY else " (SEM TAVILY_API_KEY — chamadas vão falhar)"))
    yield

    await state.tavily_client.aclose()
    log.info("AVA Search API encerrada")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="AVA Search API", lifespan=lifespan)


# ── POST /search ─────────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query vazia")

    t0 = time.perf_counter()

    if req.use_cache:
        cached = state.cache.get(req)
        if cached is not None:
            entry_data = cached
            return SearchResponse(
                results    = [SearchResult(**r) for r in entry_data],
                query      = query,
                from_cache = True,
                latency_ms = round((time.perf_counter() - t0) * 1000, 2),
            )

    client = state.tavily_client

    # 1. Busca de texto + busca de PDF (se pedida), em paralelo.
    #    A query do caller (LLM via tool-calling) é usada direto — sem
    #    passar por extração de keywords, que só degradava queries já
    #    curtas/objetivas. A Tavily já rankeia por relevância, então (ao
    #    contrário da versão DDG) a ordem dos resultados já é significativa.
    search_task = _tavily_search(
        client, query=query, max_results=max(req.max_results * 2, 10),
        topic=req.topic, days=req.days,
        include_domains=req.include_domains, exclude_domains=req.exclude_domains,
    )

    if req.search_pdfs:
        gathered = await asyncio.gather(
            search_task, _search_pdfs(client, query, req.max_results),
            return_exceptions=True,
        )
        search_data = gathered[0] if not isinstance(gathered[0], Exception) else {}
        pdf_results = gathered[1] if not isinstance(gathered[1], Exception) else []
        if isinstance(gathered[0], Exception):
            log.warning(f"Busca de texto (Tavily) falhou: {gathered[0]}")
        if isinstance(gathered[1], Exception):
            log.warning(f"Busca de PDF (Tavily) falhou: {gathered[1]}")
    else:
        try:
            search_data = await search_task
        except Exception as e:
            log.warning(f"Busca de texto (Tavily) falhou: {e}")
            search_data = {}
        pdf_results = []

    answer = search_data.get("answer") if isinstance(search_data, dict) else None

    # 2. Pipeline de texto — mantém a ordem de relevância devolvida pela
    #    Tavily; só descarta homepages/páginas-índice (ver _is_index_page).
    raw_hits = search_data.get("results", []) if isinstance(search_data, dict) else []
    indexed  = [r for r in raw_hits if not _is_index_page(r.get("url", ""))]
    skipped  = len(raw_hits) - len(indexed)
    if skipped:
        log.info(f"{skipped} resultado(s) descartado(s) por parecerem homepage/página-índice")

    text_results: list[SearchResult] = []
    for r in indexed[: req.max_results]:
        content = r.get("raw_content") or r.get("content", "")
        if not content:
            continue
        text_results.append(SearchResult(
            text     = _clean_text(content),
            source   = r.get("url", ""),
            title    = r.get("title", ""),
            from_pdf = False,
            score    = r.get("score"),
        ))

    # 3. Merge — texto primeiro (já vem ranqueado pela Tavily), PDF como complemento
    all_results = text_results + pdf_results
    results = all_results[: req.max_results]

    latency = round((time.perf_counter() - t0) * 1000, 2)
    log.info(
        f"Busca em {latency}ms — {len(text_results)} texto + {len(pdf_results)} PDF → "
        f"{len(results)} retornados: {query[:50]}"
    )

    if req.use_cache:
        state.cache.put(req, [r.model_dump() for r in results], answer)

    return SearchResponse(
        results    = results,
        query      = query,
        from_cache = False,
        latency_ms = latency,
        answer     = answer,
    )


# ── POST /extract ────────────────────────────────────────────────────────────
# Extração direta de conteúdo pra URLs específicas (ex: o usuário colou um
# link e quer que o agente leia aquela página), sem passar pelo /search.
# Também usa a API grátis da Tavily — dá pro orchestrator expor isso como
# uma tool própria (ver tool "read_url").

@app.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest):
    urls = [u.strip() for u in req.urls if u.strip()]
    if not urls:
        raise HTTPException(status_code=400, detail="lista de urls vazia")

    t0 = time.perf_counter()
    data = await _tavily_extract(state.tavily_client, urls, req.extract_depth)

    results = [
        ExtractedPage(
            url     = item.get("url", ""),
            title   = item.get("title") or "",
            text    = _clean_text(item.get("raw_content", "")),
            success = True,
        )
        for item in data.get("results", [])
    ]
    failed = [f.get("url", "") for f in data.get("failed_results", [])]

    return ExtractResponse(
        results    = results,
        failed     = failed,
        latency_ms = round((time.perf_counter() - t0) * 1000, 2),
    )


# ── GET /status ──────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    return {
        "provider":              "tavily",
        "api_key_configured":    bool(TAVILY_API_KEY),
        "cache_size":            state.cache.size,
        "cache_ttl_seconds":     CACHE_TTL_SECONDS,
        "search_depth_default":  SEARCH_DEPTH_DEFAULT,
        "topic_default":         TOPIC_DEFAULT,
        "max_pdf_results":       PDF_MAX_RESULTS,
        "top_results":           TOP_RESULTS_FINAL,
        "pdf_chunk_size_words":  PDF_CHUNK_SIZE,
        "pdf_chunk_overlap":     PDF_CHUNK_OVERLAP,
        "page_text_max_chars":   PAGE_TEXT_MAX_CHARS,
        "estimated_credits_used_this_month": state.credits.estimated_credits,
        "estimated_credits_breakdown": {
            "search_basic":    state.credits.search_basic,
            "search_advanced": state.credits.search_advanced,
            "extract_calls":   state.credits.extract_calls,
        },
        "note": (
            "estimated_credits_* é aproximado (extract é cobrado em lotes pela "
            "Tavily) — confira o dashboard da Tavily pro número exato de créditos."
        ),
    }


# ── DELETE /cache ────────────────────────────────────────────────────────────

@app.delete("/cache")
async def clear_cache():
    state.cache.clear()
    return {"cleared": True}


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3002, log_level="info")