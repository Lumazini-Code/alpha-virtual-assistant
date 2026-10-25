"""alpha-mcp — tools self-hosted do AVA (busca web + memória).

Tools MCP expostas:
  - `web_search(query, max_results)` → trechos relevantes com fontes.
    O LLM não vê as etapas internas — recebe só o resultado final.
  - `memory_write_batch(facts)` → grava múltiplos fatos PERMANENTES na
    memória de longo prazo (LT) via a memory_api — o análogo da extração
    automática de memórias, mas disparado pelo próprio modelo (ex.: extrair
    fatos de um documento ou fonte que o usuário pediu para memorizar).
    Mesmo contrato de gravação do extrator de LLM.py (_MEMORY_EXTRACTION_SCHEMA):
    fatos semânticos, forgettable=False, source=tool:memory_write_batch.
  - `memory_people_card_create(person_name, ...)` → cria um card de pessoa
    (people card) na memória — nome, descrição, gostos, religião e
    informações adicionais. Card textual puro sobre a mesma tabela unificada
    `people` dos dicionários de rosto/voz; recusa duplicado por nome.
  - `memory_people_card_edit(person_id, ...)` → edita campos de um card
    existente (só os campos enviados; None mantém o valor atual).
  - `memory_people_card_list()` → lista os cards para o modelo descobrir
    person_ids e evitar duplicados antes de criar.
  - `memory_identity_get()` → lê o card de IDENTIDADE da própria Alpha
    (quem É a assistente — nome, descrição, papéis/valores, infos extras).
    Card singleton semeado no boot da memory a partir da persona.
  - `memory_identity_edit(description?, roles?, ...)` → edita campos do
    card de identidade (só os campos enviados; None mantém o atual).
    Mesma autenticação do write LT: client persistente com Bearer token
    interno (Modules.internal_auth / AVA_INTERNAL_TOKEN).

Fluxo da busca web:
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

Config: resource/alpha-mcp/config.json — caminho via env ALPHA_MCP_CONFIG;
no container default é /app/resource/alpha-mcp/config.json, em dev cai no
relativo ../../resource/alpha-mcp/config.json. A memory_api é apontada via
env MEMORY_API_URL (default http://127.9.12.22:3000; no compose, http://ava:3000).
"""

import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import trafilatura
from rank_bm25 import BM25Plus
import redis.asyncio as aioredis

from mcp.server.mcpserver import MCPServer

logging.basicConfig(
    level=logging.DEBUG if os.environ.get("AVA_DEBUG") else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ava.alpha_mcp")

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULTS = {
    "searxng": {"url": "http://127.9.12.22:8081", "results": 15, "timeout_s": 3.0, "language": "pt-BR"},
    "crawl4ai": {"url": "http://127.9.12.22:11235", "concurrency": 5, "timeout_s": 15.0},
    "static_fetch": {"timeout_s": 8.0, "min_chars": 400},
    "valkey": {"url": "redis://127.9.12.22:6379/0", "search_ttl_s": 21600, "page_ttl_s": 86400},
    "filter": {"top_urls": 5, "max_per_domain": 1, "block_domains": []},
    "chunks": {"max_chunk_chars": 1200, "min_chunk_chars": 80, "max_per_page": 3, "max_tokens_total": 4000},
    "fetch": {"min_pages": 3},
}


def _load_config() -> dict:
    import copy
    cfg = copy.deepcopy(_DEFAULTS)
    path = os.getenv("ALPHA_MCP_CONFIG")
    if not path:
        candidates = [
            Path("/app/resource/alpha-mcp/config.json"),
            Path(__file__).resolve().parent.parent.parent / "resource" / "alpha-mcp" / "config.json",
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

def _is_private_ip(ip_str: str) -> bool:
    """Verifica se um IP é privado/loopback/link-local/reserved (T6 — SSRF)."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    # IPv4-mapped IPv6 (ex: ::ffff:127.9.12.22)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return (ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast or ip.is_unspecified)


def _assert_public_url(url: str) -> None:
    """
    T6 — SSRF guard.
    Resolve o hostname da URL e bloqueia IPs privados, loopback, link-local,
    metadata (ex: 169.254.169.254), etc.
    Raises ValueError se a URL for considerada insegura.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("URL deve usar http ou https")
        if not parsed.hostname:
            raise ValueError("URL sem hostname")
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            infos = socket.getaddrinfo(parsed.hostname, port)
        except socket.gaierror:
            raise ValueError(f"Não foi possível resolver hostname: {parsed.hostname}")
        for info in infos:
            if _is_private_ip(info[4][0]):
                raise ValueError(f"URL resolve para IP bloqueado ({info[4][0]}) — SSRF suspeito")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Validação de URL falhou: {e}")


async def _ssrf_guard_hook(request: httpx.Request) -> None:
    """httpx event hook: valida a URL de CADA request (inclui redirects)."""
    _assert_public_url(str(request.url))


_static_client: httpx.AsyncClient | None = None


def _get_static_client() -> httpx.AsyncClient:
    """Client HTTP persistente do fetch estático (connection pooling, mesmo
    padrão do _memory_client). Criado sob demanda e reutilizado por TODAS as
    buscas — nunca fechado no meio de uma busca em curso (o padrão antigo,
    `async with` em fetch_pages + global, fechava o client enquanto outra
    busca concorrente ainda o usava → RuntimeError). TLS reutilizado:
    ~50-150ms economizados por página."""
    global _static_client
    if _static_client is None or _static_client.is_closed:
        _static_client = httpx.AsyncClient(
            timeout=CFG["static_fetch"]["timeout_s"],
            event_hooks={"request": [_ssrf_guard_hook]},
        )
    return _static_client



async def _static_fetch(url: str) -> str | None:
    """Página estática? HTTP puro + trafilatura resolve em milissegundos."""
    sf = CFG["static_fetch"]
    try:
        r = await _get_static_client().get(url, headers={"User-Agent": _UA}, follow_redirects=True)
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
            return None
        # T6: trafilatura é CPU-bound puro — rodando fora do event loop para
        # não bloquear concorrência (o semáforo do caller segue limitando).
        md = await asyncio.to_thread(
            trafilatura.extract, r.text, output_format="markdown",
            include_comments=False, include_links=False,
            favor_recall=True,
        )
        if md and len(md) >= sf["min_chars"]:
            return md
    except Exception:
        pass
    return None


_crawl4ai_client: httpx.AsyncClient | None = None


def _get_crawl4ai_client() -> httpx.AsyncClient:
    """Client persistente do lote crawl4ai (connection pooling). SEM o hook
    SSRF de propósito: o serviço roda na rede docker interna e o hook do
    fetch estático bloquearia o IP dele. Timeout é passado por-request —
    varia com o tamanho do lote."""
    global _crawl4ai_client
    if _crawl4ai_client is None or _crawl4ai_client.is_closed:
        _crawl4ai_client = httpx.AsyncClient()
    return _crawl4ai_client


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
        # Client persistente próprio (_get_crawl4ai_client) — o _static_client
        # tem hook SSRF que bloquearia o IP interno do crawl4ai (rede docker
        # privada). Timeout por-request: varia com o tamanho do lote.
        r = await _get_crawl4ai_client().post(
            f"{c4['url']}/crawl", json=payload, headers=headers, timeout=timeout,
        )
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

    # T7: client persistente do módulo (_get_static_client) — sem `async with`,
    # o client nunca é fechado sob buscas concorrentes.
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
    top = min(max_results, CFG["filter"]["top_urls"])
    if len(candidates) <= top:
        # T9: candidatos já cabem no top — rerank por snippets é redundante.
        ranked = [(c, 0.0) for c in candidates]
    else:
        scores = _bm25_rank(query, [(c.get("title") or "") + " " + (c.get("content") or "")
                                    for c in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
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
# Memory API — gravação de fatos em memória de longo prazo (LT)
# ══════════════════════════════════════════════════════════════════════════════

MEMORY_API_URL = os.getenv("MEMORY_API_URL", "http://127.9.12.22:3000")

_memory_client: httpx.AsyncClient | None = None


def _get_memory_client() -> httpx.AsyncClient:
    """Client persistente para a memory_api (connection pooling, mesmo padrão
    do LLM.py). Recria de forma segura se a anterior foi fechada."""
    global _memory_client
    if _memory_client is None or _memory_client.is_closed:
        try:
            from Modules.internal_auth import internal_token as _tok_fn  # type: ignore
        except Exception:
            _tok_fn = lambda: os.getenv("AVA_INTERNAL_TOKEN", "").strip() or None  # type: ignore
        _tok = _tok_fn()
        _h: dict[str, str] = {"Authorization": f"Bearer {_tok}"} if _tok else {}
        _memory_client = httpx.AsyncClient(
            base_url=MEMORY_API_URL,
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=2, max_connections=4),
            headers=_h,
        )
    return _memory_client


async def _resolve_possible_update(item: dict, res: dict) -> dict:
    """Espelha `_resolve_possible_update` de Modules/LLM.py: quando o write é
    recusado com reason="possible_update:<score>" e o servidor devolve
    candidate_id, reenvia a escrita com action="update" + memory_id para
    efetivar a correção na memória candidata em vez de descartar o fato.
    Sem candidate_id, o fato NÃO vira registro solto (por design, uma correção
    não deve criar fato desancorado) — mas o descarte é logado em WARNING."""
    reason = str(res.get("reason") or "")
    if not reason.startswith("possible_update:"):
        return res
    candidate_id = res.get("candidate_id")
    if candidate_id is None:
        log.warning(
            f"[memory] possible_update sem candidate_id — fato descartado "
            f"(reason={reason!r}, text={str(item.get('text'))[:80]!r})"
        )
        return res
    log.info(
        f"[memory] possible_update — candidate_id={candidate_id}, "
        f"candidate_score={res.get('candidate_score')}, "
        f"candidate_text={str(res.get('candidate_text'))[:80]!r}"
    )
    try:
        r = await _get_memory_client().post("/write", json={
            **item, "action": "update", "memory_id": candidate_id,
        })
        r.raise_for_status()
        data = r.json()
        data["resolved"] = "update" if data.get("stored") else "not_applied"
        if not data.get("stored"):
            log.info(
                f"[memory] update LT não efetivado (candidate_id={candidate_id}): "
                f"{data.get('reason')}"
            )
        return data
    except Exception as e:
        log.warning(
            f"[memory] possible_update resolve falhou (candidate_id={candidate_id}): {e}"
        )
        return res


async def _memory_write_facts(texts: list[str]) -> list[dict]:
    """Grava os fatos em LT via POST /write/batch (1 round-trip) e resolve
    possible_update por item. Mesmo contrato do extrator de LLM.py para
    memórias semânticas não-esquecíveis: forgettable=False (isenta de decay),
    confidence=0.9 (prioridade na busca) e sequential=True (o servidor cria
    cadeia de arestas temporal_precedence entre fatos consecutivos — o PPR
    propaga contexto entre memórias do mesmo episódio)."""
    items = [
        {
            "text":        t,
            "source":      "tool:memory_write_batch",
            "confidence":  0.9,
            "forgettable": False,
            "ttl_days":    None,
        }
        for t in texts
    ]
    r = await _get_memory_client().post(
        "/write/batch", json={"items": items, "sequential": True},
    )
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])
    # T8: resoluções de possible_update em paralelo (gather).
    return await asyncio.gather(*(
        _resolve_possible_update(item, res)
        for item, res in zip(items, results)
    ))


# ══════════════════════════════════════════════════════════════════════════════
# Servidor MCP
# ══════════════════════════════════════════════════════════════════════════════

mcp = MCPServer("alpha-mcp")


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


@mcp.tool()
async def memory_write_batch(facts: list[str]) -> str:
    """Grava múltiplos fatos na memória de LONGO prazo (permanente) do usuário.
    Cada item de `facts` deve ser um fato AUTOCONTIDO em terceira pessoa
    (ex.: "O usuário chama-se João", "O usuário é alérgico a amendoim",
    "O autor do livro X é Y") — sem referências a "esta conversa" nem
    pronomes soltos. Use ao extrair informações úteis de documentos, páginas
    da web ou qualquer fonte que o usuário pedir para memorizar; não use
    para eventos transitórios da conversa em andamento."""
    if not isinstance(facts, list):
        return "Erro: `facts` deve ser uma lista de strings."
    cleaned = [f.strip() for f in facts if isinstance(f, str) and f.strip()]
    if not cleaned:
        return "Erro: nenhum fato válido fornecido."
    try:
        results = await _memory_write_facts(cleaned)
    except Exception as e:
        log.exception("memory_write_batch falhou")
        return f"Erro ao gravar memórias: {e}"
    stored = sum(1 for r in results if r.get("stored"))
    lines = []
    for i, (f, r) in enumerate(zip(cleaned, results), 1):
        if r.get("stored") and r.get("resolved") == "update":
            lines.append(f"[{i}] Atualizada memória candidata #{r.get('memory_id')}: {f[:100]}")
        elif r.get("stored"):
            lines.append(f"[{i}] Gravada (memory_id={r.get('memory_id')}): {f[:100]}")
        else:
            lines.append(f"[{i}] Recusada ({r.get('reason')}): {f[:100]}")
    log.info(f"[memory] memory_write_batch: {stored}/{len(cleaned)} fato(s) gravado(s)")
    return (
        f"{stored}/{len(cleaned)} fato(s) gravados na memória de longo prazo (permanente):\n"
        + "\n".join(lines)
    )


# ══════════════════════════════════════════════════════════════════════════════
# People Cards — criar/editar cards de pessoa na memória (mesma auth do write)
# ══════════════════════════════════════════════════════════════════════════════

class MemoryToolError(Exception):
    """Erro de negócio devolvido pela memory_api (HTTP 400, JSON {"error": ...}).
    O `detail` vira o texto de erro da tool — mesmo espírito do
    MemoryToolError da camada de memória, mas definido aqui para o servidor
    MCP não depender do pacote Modules."""  # noqa: matching-memory-models-name
    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


async def _memory_api_request(method: str, path: str, payload: dict | None = None) -> dict:
    """Request autenticada à memory_api (mesmo client/Bearer do write LT).
    Levanta MemoryToolError com o `detail` do servidor em erro 400."""
    r = await _get_memory_client().request(method, path, json=payload)
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError:
        detail = None
        try:
            detail = r.json().get("error") or r.json().get("detail")
        except Exception:
            pass
        raise MemoryToolError(detail or f"memory_api respondeu HTTP {r.status_code}")
    return r.json()


def _format_card(c: dict) -> str:
    """Card de pessoa em linha compacta: '#id Nome — descrição (gostos/religião/info)'."""
    extra = " / ".join(
        f"{label}: {c[label]}" for label in ("likes", "religion", "additional_info")
        if c.get(label)
    )
    desc = c.get("description") or "sem descrição"
    return f"#{c['person_id']} {c['person_name']} — {desc}" + (f" ({extra})" if extra else "")


@mcp.tool()
async def memory_people_card_create(
    person_name: str,
    description: str = "",
    likes: str = "",
    religion: str = "",
    additional_info: str = "",
) -> str:
    """Cria um card de pessoa na memória do usuário (Pessoas). Use ao aprender
    sobre alguém novo e estável na vida do usuário — quem é a pessoa
    (`description`), gostos/preferências (`likes`), crenças (`religion`) e
    outros fatos persistentes (`additional_info`). NÃO crie cards para fatos
    soltos sobre pessoas sem relevância duradoura — para isso use
    memory_write_batch. Se a pessoa já existir, o servidor recusa a criação:
    use memory_people_card_list para descobrir o person_id e
    memory_people_card_edit para atualizar o card dela."""
    if not person_name.strip():
        return "Erro: person_name vazio."
    try:
        data = await _memory_api_request("POST", "/people-card/write", {
            "person_name":     person_name.strip(),
            "description":     description.strip(),
            "likes":           likes.strip(),
            "religion":        religion.strip(),
            "additional_info": additional_info.strip(),
            "link_to_memory":  True,
        })
    except MemoryToolError as e:
        return f"Erro ao criar card de pessoa: {e.detail}"
    except Exception as e:
        log.exception("memory_people_card_create falhou")
        return f"Erro ao criar card de pessoa: {e}"
    if not data.get("stored"):
        if data.get("reason") == "duplicate":
            pid = data.get("person_id")
            return (
                f"Card duplicado: '{person_name.strip()}' já existe "
                f"(person_id={pid}). Use memory_people_card_edit para atualizar "
                f"o card dela."
            )
        return f"Card não criado ({data.get('reason')})."
    log.info(f"[people] card criado: #{data.get('person_id')} '{person_name.strip()}'")
    return (
        f"Card de pessoa criado (person_id={data.get('person_id')}, "
        f"identity_memory_id={data.get('memory_id')}): "
        f"{_format_card({'person_id': data.get('person_id'), 'person_name': person_name.strip(), 'description': description, 'likes': likes, 'religion': religion, 'additional_info': additional_info})}"
    )


@mcp.tool()
async def memory_people_card_edit(
    person_id: int,
    person_name: str | None = None,
    description: str | None = None,
    likes: str | None = None,
    religion: str | None = None,
    additional_info: str | None = None,
) -> str:
    """Edita um card de pessoa existente por person_id (descubra com
    memory_people_card_list). Campos enviados SOBRESCREVEM o valor atual
    (string vazia limpa o campo); campos deixados em None NÃO são alterados.
    Mudar description ou person_name também atualiza a memória de identidade
    da pessoa na memória de longo prazo. Use ao corrigir/complementar
    informações de alguém já cadastrado — para fatos soltos prefira
    memory_write_batch."""
    payload = {
        k: v for k, v in {
            "person_name":     person_name,
            "description":     description,
            "likes":           likes,
            "religion":        religion,
            "additional_info": additional_info,
        }.items() if v is not None
    }
    if not payload:
        return "Erro: nenhum campo para editar (todos em None)."
    try:
        c = await _memory_api_request("PATCH", f"/people-card/{person_id}", payload)
    except MemoryToolError as e:
        return f"Erro ao editar card de pessoa: {e.detail}"
    except Exception as e:
        log.exception("memory_people_card_edit falhou")
        return f"Erro ao editar card de pessoa: {e}"
    log.info(f"[people] card editado: #{person_id}")
    return f"Card atualizado: {_format_card(c)}"


@mcp.tool()
async def memory_people_card_list() -> str:
    """Lista todas as pessoas cadastradas na memória do usuário (people cards).
    Use antes de memory_people_card_create para descobrir se a pessoa já
    existe (evitar duplicado) e obter o person_id para edição. Retorna uma
    linha por pessoa: '#person_id Nome — descrição (extras)'."""
    try:
        data = await _memory_api_request("GET", "/people-card/list")
    except MemoryToolError as e:
        return f"Erro ao listar cards de pessoa: {e.detail}"
    except Exception as e:
        log.exception("memory_people_card_list falhou")
        return f"Erro ao listar cards de pessoa: {e}"
    people = data.get("people", [])
    if not people:
        return "Nenhuma pessoa cadastrada."
    lines = [_format_card(c) for c in people]
    return f"{len(people)} pessoa(s) cadastrada(s):\n" + "\n".join(lines)


def _format_identity(c: dict) -> str:
    """Card de identidade em linha compacta: 'Nome — descrição (papéis/info)'."""
    extra = " / ".join(
        f"{label}: {c[label]}" for label in ("roles", "additional_info")
        if c.get(label)
    )
    desc = c.get("description") or "sem descrição"
    return f"{c.get('name', '?')} — {desc}" + (f" ({extra})" if extra else "")


@mcp.tool()
async def memory_identity_get() -> str:
    """Lê o card de IDENTIDADE da Alpha — quem VOCÊ é (nome, descrição,
    papéis/valores e informações extras persistentes). Use para responder
    perguntas sobre si mesma e antes de memory_identity_edit. O card é
    único (singleton) e nasce no boot da memória a partir da persona."""
    try:
        c = await _memory_api_request("GET", "/identity")
    except MemoryToolError as e:
        return f"Erro ao ler card de identidade: {e.detail}"
    except Exception as e:
        log.exception("memory_identity_get falhou")
        return f"Erro ao ler card de identidade: {e}"
    return (
        f"memory_id={c.get('memory_id')} — {_format_identity(c)}"
    )


@mcp.tool()
async def memory_identity_edit(
    name: str | None = None,
    description: str | None = None,
    roles: str | None = None,
    additional_info: str | None = None,
) -> str:
    """Edita o card de identidade da Alpha (card único/singleton — quem a
    própria assistente É). Campos enviados SOBRESCREVEM o valor atual
    (string vazia limpa o campo); campos deixados em None NÃO são alterados.
    Mudar name ou description também atualiza a memória de identidade LT
    dela na memória de longo prazo (o nó no grafo permanece estável). Use ao
    aprender algo duradouro sobre si mesma (papéis, valores, relações,
    preferências próprias) — para fatos soltos use memory_write_batch."""
    payload = {
        k: v for k, v in {
            "name":            name,
            "description":     description,
            "roles":           roles,
            "additional_info": additional_info,
        }.items() if v is not None
    }
    if not payload:
        return "Erro: nenhum campo para editar (todos em None)."
    try:
        c = await _memory_api_request("PATCH", "/identity", payload)
    except MemoryToolError as e:
        return f"Erro ao editar card de identidade: {e.detail}"
    except Exception as e:
        log.exception("memory_identity_edit falhou")
        return f"Erro ao editar card de identidade: {e}"
    log.info(f"[identity] card editado: campos={sorted(payload)}")
    return f"Card de identidade atualizado: {_format_identity(c)}"


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host=os.getenv("ALPHA_MCP_HOST", "127.9.12.22"),
        port=int(os.getenv("ALPHA_MCP_PORT", "4010")),
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
    )
