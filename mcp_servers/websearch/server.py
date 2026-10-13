"""websearch-mcp — pipeline de busca web do AVA (SearXNG → filtro/BM25 → crawl → chunks).

Expõe UMA única tool MCP: `web_search(query, max_results)` → trechos relevantes
com fontes. O LLM não vê as etapas internas — recebe só o resultado final.

Fluxo:
  1. Cache de busca no Valkey (hash da query, TTL de horas) → hit retorna direto
  2. SearXNG `GET /search?format=json` (~15 resultados, timeout curto)
  3. Filtro: dedup por domínio, blocklist de SEO/spam/paywall
  4. BM25 sobre os snippets → top 3-5 URLs (crawlear pouco e bem)
  5. Fetch em camadas por URL: HTTP simples + trafilatura (estático) e, se
     falhar/vazio, Crawl4AI (navegador reutilizado, text_mode) — concorrência
     limitada, timeout por página, falha descartada sem travar o lote
  6. Markdown da página cacheado no Valkey (TTL de ~1 dia)
  7. Chunking por headers markdown → BM25 dos chunks contra a query → envia
     só os melhores trechos até o orçamento de tokens, ≥1 chunk por página
  8. Retorno: texto plano com [n] título / url / trecho (fontes pra citação)

Transporte: streamable-http em /mcp (o mcp_manager do AVA registra via
`POST /servers {"transport":"http","url":...}`).

Config: resource/websearch/config.json — caminho via env WEBSEARCH_CONFIG;
no container default é /app/resource/websearch/config.json, em dev cai no
relativo ../../resource/websearch/config.json.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

import httpx
import trafilatura
from rank_bm25 import BM25Plus
import redis.asyncio as aioredis

from mcp.server.mcpserver import MCPServer

logging.basicConfig(
    level=logging.DEBUG if os.environ.get("AVA_DEBUG") else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ava.websearch")

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULTS = {
    "searxng": {"url": "http://127.0.0.1:8081", "results": 15, "timeout_s": 3.0, "language": "pt-BR"},
    "crawl4ai": {"url": "http://127.0.0.1:11235", "concurrency": 5, "timeout_s": 15.0},
    "static_fetch": {"timeout_s": 8.0, "min_chars": 400},
    "valkey": {"url": "redis://127.0.0.1:6379/0", "search_ttl_s": 21600, "page_ttl_s": 86400},
    "filter": {"top_urls": 5, "max_per_domain": 1, "block_domains": []},
    "chunks": {"max_chunk_chars": 1200, "min_chunk_chars": 80, "max_per_page": 3, "max_tokens_total": 4000},
    "fetch": {"min_pages": 3},
}


def _load_config() -> dict:
    import copy
    cfg = copy.deepcopy(_DEFAULTS)
    path = os.getenv("WEBSEARCH_CONFIG")
    if not path:
        candidates = [
            Path("/app/resource/websearch/config.json"),
            Path(__file__).resolve().parent.parent.parent / "resource" / "websearch" / "config.json",
        ]
        path = next((str(p) for p in candidates if p.is_file()), None)
    if path and Path(path).is_file():
        try:
            user = json.loads(Path(path).read_text(encoding="utf-8"))
            for section, values in user.items():
                if isinstance(values, dict) and section in cfg:
                    cfg[section] = {**cfg[section], **values}
                else:
                    cfg[section] = values
            log.info(f"config carregada de {path}")
        except Exception as e:
            log.warning(f"config inválida em {path} ({e}) — usando defaults")
    else:
        log.warning("config.json não encontrado — usando defaults")

    # Env sobrepõe config (docker-compose aponta pros serviços vizinhos)
    if v := os.getenv("SEARXNG_URL"):
        cfg["searxng"]["url"] = v
    if v := os.getenv("CRAWL4AI_URL"):
        cfg["crawl4ai"]["url"] = v
    if v := os.getenv("VALKEY_URL"):
        cfg["valkey"]["url"] = v
    return cfg


CFG = _load_config()
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# ══════════════════════════════════════════════════════════════════════════════
# Crawl4AI on-demand — o container (~490MB: Playwright + Chromium) só roda
# quando o fetch estático não resolve. Sobe/para via socket do Docker montado
# no compose. CRAWL4AI_ON_DEMAND=0 volta ao comportamento "sempre ligado".
# ══════════════════════════════════════════════════════════════════════════════

C4_CONTAINER    = os.getenv("CRAWL4AI_CONTAINER", "ava-crawl4ai")
C4_ON_DEMAND    = os.getenv("CRAWL4AI_ON_DEMAND", "1").strip().lower() in ("1", "true", "yes", "on")
C4_IDLE_STOP_S  = max(60, int(os.getenv("CRAWL4AI_IDLE_STOP_S", "600")))
C4_BOOT_TIMEOUT = max(30, int(os.getenv("CRAWL4AI_BOOT_TIMEOUT_S", "120")))

_docker_client: object | None | bool = None   # None=uninit, False=indisponível
_started_by_us  = False
_last_crawl_use = 0.0
_crawl_watchdog: asyncio.Task | None = None


def _docker():
    global _docker_client
    if _docker_client is False:
        return None
    if _docker_client is None:
        try:
            import docker
            _docker_client = docker.from_env()
        except Exception as e:
            log.warning(f"Docker SDK indisponível ({e}) — crawl4ai permanece gerenciado manualmente")
            _docker_client = False
    return _docker_client or None


async def _crawl4ai_healthy() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{CFG['crawl4ai']['url']}/health")
            return r.status_code == 200
    except Exception:
        return False


async def _ensure_crawl4ai() -> bool:
    """Garante o crawl4ai no ar (subindo on-demand se necessário).
    Retorna True se saudável; False = pipeline segue só com fetch estático."""
    global _started_by_us, _last_crawl_use
    _last_crawl_use = time.monotonic()
    if await _crawl4ai_healthy():
        return True
    if not C4_ON_DEMAND:
        return False
    d = _docker()
    if d is None:
        return False
    try:
        c = d.containers.get(C4_CONTAINER)
        if c.status != "running":
            log.info(f"[crawl4ai] indisponível — subindo {C4_CONTAINER} on-demand "
                     f"(1ª vez pode levar até {C4_BOOT_TIMEOUT}s)...")
            c.start()
            _started_by_us = True
        deadline = time.monotonic() + C4_BOOT_TIMEOUT
        while time.monotonic() < deadline:
            if await _crawl4ai_healthy():
                log.info("[crawl4ai] pronto")
                _schedule_crawl_stop()
                return True
            await asyncio.sleep(2)
        log.warning("[crawl4ai] não ficou saudável dentro do timeout — seguindo sem ele")
    except Exception as e:
        log.warning(f"[crawl4ai] falha ao subir container ({e}) — seguindo sem ele")
    return False


async def _stop_crawl4ai() -> None:
    global _started_by_us
    try:
        d = _docker()
        if d is None:
            return
        c = d.containers.get(C4_CONTAINER)
        if c.status == "running":
            c.stop(timeout=30)
            log.info(f"[crawl4ai] parado após {C4_IDLE_STOP_S}s sem uso (economia de RAM)")
    except Exception as e:
        log.debug(f"[crawl4ai] stop falhou ({e})")
    finally:
        _started_by_us = False


def _schedule_crawl_stop() -> None:
    """Watchdog idempotente: para o container após TTL sem uso — só se NÓS
    o subimos (não interfere em quem subiu manualmente)."""
    global _crawl_watchdog
    if _crawl_watchdog and not _crawl_watchdog.done():
        return

    async def _watch():
        while True:
            await asyncio.sleep(C4_IDLE_STOP_S)
            if time.monotonic() - _last_crawl_use < C4_IDLE_STOP_S:
                continue  # houve uso recente — segue esperando
            if _started_by_us:
                await _stop_crawl4ai()
            return

    _crawl_watchdog = asyncio.create_task(_watch())


# ══════════════════════════════════════════════════════════════════════════════
# Valkey (cache) — degrada com graça se indisponível
# ══════════════════════════════════════════════════════════════════════════════

_redis: aioredis.Redis | None = None


async def _cache() -> aioredis.Redis | None:
    global _redis
    if _redis is None:
        try:
            r = aioredis.from_url(CFG["valkey"]["url"], decode_responses=True,
                                  socket_timeout=2.0, socket_connect_timeout=2.0)
            await r.ping()
            _redis = r
            log.info(f"Valkey conectado em {CFG['valkey']['url']}")
        except Exception as e:
            log.warning(f"Valkey indisponível ({e}) — pipeline segue sem cache")
            _redis = False  # type: ignore[assignment]
    return _redis or None


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:24]


async def cache_get(key: str) -> str | None:
    r = await _cache()
    if not r:
        return None
    try:
        return await r.get(key)
    except Exception as e:
        log.warning(f"cache_get falhou ({e})"); return None


async def cache_set(key: str, value: str, ttl: int) -> None:
    r = await _cache()
    if not r:
        return
    try:
        await r.set(key, value, ex=ttl)
    except Exception as e:
        log.warning(f"cache_set falhou ({e})")


# ══════════════════════════════════════════════════════════════════════════════
# BM25 helpers
# ══════════════════════════════════════════════════════════════════════════════

_STOPWORDS = set("""
a o os as de da do das dos e é em no na nos nas um uma uns umas por para com sem
que se ao aos à às pelo pela como mais menos muito pouco sobre entre até desde
the of and to in on for with is are was were be been it this that as at by from
""".split())


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w{2,}", text.lower(), re.UNICODE)
            if t not in _STOPWORDS]


def _bm25_rank(query: str, docs: list[str]) -> list[float]:
    """Scores BM25 dos docs contra a query; lista vazia/1 doc → scores triviais.
    BM25Plus (não Okapi) porque em corpus pequenos o idf do Okapi vira 0
    pra termo presente em metade dos docs — e é exatamente o nosso caso."""
    if not docs:
        return []
    if len(docs) == 1:
        return [1.0]
    corpus = [_tokenize(d) for d in docs]
    corpus = [d if d else ["vazio"] for d in corpus]
    try:
        bm25 = BM25Plus(corpus)
        return list(bm25.get_scores(_tokenize(query)))
    except Exception as e:
        log.warning(f"BM25 falhou ({e}) — ordem original mantida")
        return [1.0 / (i + 1) for i in range(len(docs))]


# ══════════════════════════════════════════════════════════════════════════════
# Etapa 1-2: SearXNG (descoberta)
# ══════════════════════════════════════════════════════════════════════════════


async def searxng_search(query: str) -> list[dict]:
    sx = CFG["searxng"]
    async with httpx.AsyncClient(timeout=sx["timeout_s"]) as client:
        r = await client.get(f"{sx['url']}/search", params={
            "q": query, "format": "json", "safesearch": 0, "language": sx["language"],
        })
        r.raise_for_status()
        return r.json().get("results", [])


def filter_results(results: list[dict], max_urls: int) -> list[dict]:
    """Dedup por domínio + blocklist + ordenação por score do SearXNG."""
    blocked = tuple(CFG["filter"]["block_domains"])
    max_per_domain = CFG["filter"]["max_per_domain"]
    seen_domain: dict[str, int] = {}
    seen_url: set[str] = set()
    kept = []
    for item in sorted(results, key=lambda x: x.get("score", 0), reverse=True):
        url = item.get("url") or ""
        if not url or url in seen_url or not url.startswith("http"):
            continue
        domain = re.sub(r"^www\.", "", url.split("//", 1)[-1].split("/", 1)[0].lower())
        if domain.endswith(blocked):
            continue
        if seen_domain.get(domain, 0) >= max_per_domain:
            continue
        seen_url.add(url)
        seen_domain[domain] = seen_domain.get(domain, 0) + 1
        kept.append(item)
        if len(kept) >= max_urls * 2:  # folga: crawl pode descartar falhas
            break
    return kept


# ══════════════════════════════════════════════════════════════════════════════
# Etapa 3-4: Fetch em camadas (HTTP estático → Crawl4AI)
# ══════════════════════════════════════════════════════════════════════════════

_static_client: httpx.AsyncClient | None = None


async def _static_fetch(url: str) -> str | None:
    """Página estática? HTTP puro + trafilatura resolve em milissegundos."""
    sf = CFG["static_fetch"]
    try:
        r = await _static_client.get(url, headers={"User-Agent": _UA}, follow_redirects=True)
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
            return None
        md = trafilatura.extract(r.text, output_format="markdown",
                                 include_comments=False, include_links=False,
                                 favor_recall=True)
        if md and len(md) >= sf["min_chars"]:
            return md
    except Exception:
        pass
    return None


async def _crawl4ai_batch(urls: list[str]) -> dict[str, str]:
    """Lote pro Crawl4AI (navegador reutilizado). Falha individual vira
    ausência no dict retornado — nunca trava o resto do lote."""
    if not await _ensure_crawl4ai():
        log.info("[crawl4ai] ausente — batch ignorado (só fetch estático nesta busca)")
        return {}
    c4 = CFG["crawl4ai"]
    payload = {
        "urls": urls,
        "browser_config": {"type": "BrowserConfig", "params": {
            "headless": True, "text_mode": True,
        }},
        "crawler_config": {"type": "CrawlerRunConfig", "params": {
            "stream": False, "cache_mode": "ENABLED",
            "page_timeout": int(c4["timeout_s"] * 1000),
            "exclude_external_images": True, "exclude_social_media_links": True,
        }},
    }
    timeout = c4["timeout_s"] * (len(urls) // c4["concurrency"] + 1) + 10.0
    headers = {}
    if token := os.getenv("CRAWL4AI_API_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    out: dict[str, str] = {}
    try:
        r = await _static_client.post(f"{c4['url']}/crawl", json=payload,
                                      timeout=timeout, headers=headers)
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or ([data["result"]] if data.get("result") else [])
        for res in results:
            if not res.get("success"):
                continue
            md = res.get("markdown")
            if isinstance(md, dict):
                md = md.get("fit_markdown") or md.get("raw_markdown") or ""
            text = (md or "").strip()
            if text and res.get("url"):
                out[res["url"]] = text
    except Exception as e:
        log.warning(f"crawl4ai falhou ({e})")
    return out


async def fetch_pages(pages: list[dict]) -> list[dict]:
    """Cache → HTTP estático → Crawl4AI. Preenche `markdown` em cada page."""
    global _static_client
    sem = asyncio.Semaphore(CFG["crawl4ai"]["concurrency"])

    async def one(page: dict) -> None:
        key = f"ws:page:{_hash(page['url'])}"
        cached = await cache_get(key)
        if cached:
            page["markdown"] = cached
            return
        async with sem:
            page["markdown"] = await _static_fetch(page["url"]) or ""
        if page["markdown"]:
            await cache_set(key, page["markdown"], CFG["valkey"]["page_ttl_s"])

    async with httpx.AsyncClient(timeout=CFG["static_fetch"]["timeout_s"]) as client:
        _static_client = client
        await asyncio.gather(*(one(p) for p in pages))

        # Estático resolveu pouco? Manda o lote inteiro de uma vez pro
        # Crawl4AI (mais eficiente que N chamadas unitárias de browser).
        fetched = [p for p in pages if p.get("markdown")]
        missing = [p for p in pages if not p.get("markdown")]
        if len(fetched) < CFG["fetch"]["min_pages"] and missing:
            got = await _crawl4ai_batch([p["url"] for p in missing])
            for p in missing:
                if p["url"] in got:
                    p["markdown"] = got[p["url"]]
                    await cache_set(f"ws:page:{_hash(p['url'])}", p["markdown"],
                                    CFG["valkey"]["page_ttl_s"])
            fetched = [p for p in pages if p.get("markdown")]
        return fetched


# ══════════════════════════════════════════════════════════════════════════════
# Etapa 5: Chunking + seleção de trechos
# ══════════════════════════════════════════════════════════════════════════════

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def chunk_markdown(md: str, title: str, url: str) -> list[dict]:
    """Quebra o markdown por seção (headers); seções gigantes são fatiadas
    por parágrafos. Cada chunk carrega url+title pra citação."""
    cc = CFG["chunks"]
    chunks: list[dict] = []
    section, buf = "intro", []

    def flush():
        text = "\n\n".join(buf).strip()
        buf.clear()
        if len(text) < cc["min_chunk_chars"]:
            return
        while len(text) > cc["max_chunk_chars"]:
            cut = text.rfind("\n\n", 0, cc["max_chunk_chars"])
            cut = cut if cut > cc["max_chunk_chars"] // 2 else cc["max_chunk_chars"]
            chunks.append({"url": url, "title": title, "section": section,
                           "text": text[:cut].strip()})
            text = text[cut:].strip()
        if len(text) >= cc["min_chunk_chars"]:
            chunks.append({"url": url, "title": title, "section": section, "text": text})

    for line in md.splitlines():
        m = _HEADER_RE.match(line)
        if m:
            flush()
            section = m.group(2).strip()[:80] or section
        else:
            buf.append(line)
    flush()
    return chunks


def select_chunks(chunks: list[dict], query: str) -> list[dict]:
    """BM25 dos chunks contra a query. Garante ≥1 chunk por página (o melhor
    de cada), respeita max_per_page e para no orçamento de tokens."""
    cc = CFG["chunks"]
    if not chunks:
        return []
    scores = _bm25_rank(query, [c["text"] for c in chunks])
    for c, s in zip(chunks, scores):
        c["score"] = s

    best_per_page: dict[str, dict] = {}
    for c in sorted(chunks, key=lambda x: x["score"], reverse=True):
        best_per_page.setdefault(c["url"], c)
    picked: list[dict] = []
    budget = cc["max_tokens_total"]
    per_page_count: dict[str, int] = {}  # inclui o passo 1 — max_per_page é TOTAL por página

    def take(c: dict) -> bool:
        nonlocal budget
        if any(c is p for p in picked) or budget <= 0:
            return False
        est = len(c["text"]) // 4 + 1
        if est > budget and picked:
            return False
        picked.append(c)
        budget -= est
        per_page_count[c["url"]] = per_page_count.get(c["url"], 0) + 1
        return True

    for c in best_per_page.values():  # 1) melhor chunk de cada página
        take(c)
    for c in sorted(chunks, key=lambda x: x["score"], reverse=True):  # 2) preenche por score
        if per_page_count.get(c["url"], 0) >= cc["max_per_page"]:
            continue
        take(c)
    return picked


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline completa
# ══════════════════════════════════════════════════════════════════════════════


def _format(chunks: list[dict], query: str) -> str:
    parts = [f'Resultados da busca por "{query}":', ""]
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] {c['title']}")
        parts.append(c["url"])
        parts.append(c["text"])
        parts.append("")
    return "\n".join(parts).strip()


async def run_pipeline(query: str, max_results: int) -> str:
    # 0) cache da busca inteira
    ck = f"ws:search:{_hash(f'{query}::{max_results}')}"
    cached = await cache_get(ck)
    if cached:
        log.info(f"cache hit: {query!r}")
        return cached

    # 1) descoberta
    results = await searxng_search(query)
    if not results:
        return "Nenhum resultado encontrado."
    candidates = filter_results(results, max(CFG["filter"]["top_urls"], max_results))

    # 2) rerank pelos snippets → top N URLs
    scores = _bm25_rank(query, [(c.get("title") or "") + " " + (c.get("content") or "")
                                for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top = min(max_results, CFG["filter"]["top_urls"])
    pages = [{"url": c["url"], "title": c.get("title") or c["url"],
              "snippet": c.get("content") or ""} for c, _ in ranked[:top]]

    # 3) conteúdo
    fetched = await fetch_pages(pages)
    if not fetched:
        # sem crawl: devolve só os snippets (melhor que nada)
        fallback = "\n\n".join(f"[{i}] {p['title']}\n{p['url']}\n{p['snippet']}"
                               for i, p in enumerate(pages, 1))
        return f"Não foi possível acessar o conteúdo das páginas; snippets:\n\n{fallback}"

    # 4) chunks + seleção
    chunks = [c for p in fetched for c in chunk_markdown(p["markdown"], p["title"], p["url"])]
    picked = select_chunks(chunks, query)
    if not picked:
        return "As páginas foram acessadas mas nenhum trecho relevante foi extraído."

    out = _format(picked, query)
    await cache_set(ck, out, CFG["valkey"]["search_ttl_s"])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Servidor MCP
# ══════════════════════════════════════════════════════════════════════════════

mcp = MCPServer("websearch")


@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> str:
    """Busca na web e retorna os trechos mais relevantes das melhores páginas,
    cada um com título e URL de origem para citação. Use para fatos atuais,
    verificação de informações ou qualquer coisa fora do seu conhecimento.
    Para pesquisar a fundo, chame várias vezes refinando a query."""
    if not query.strip():
        return "Erro: query vazia."
    max_results = max(1, min(max_results, 10))
    try:
        return await run_pipeline(query, max_results)
    except Exception as e:
        log.exception("pipeline falhou")
        return f"Erro na busca web: {e}"


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host=os.getenv("WEBSEARCH_HOST", "0.0.0.0"),
        port=int(os.getenv("WEBSEARCH_PORT", "4010")),
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
    )
