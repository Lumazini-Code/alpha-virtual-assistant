"""
AVA KG-RAG — Web Researcher (Etapa 2)

Estratégia adaptada para ambientes com IP bloqueado por Cloudflare:
  1. DDGS().text() → retorna URLs + snippets (título + resumo de cada resultado)
  2. Tenta fetch direto da página (funciona quando o site não bloqueia)
  3. Fallback para o snippet da busca (sempre disponível, qualidade razoável)
  4. Agrega múltiplos snippets do mesmo sub-tópico quando fetch falha

Dependência: pip install duckduckgo-search
"""

import re
import logging
import asyncio
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS

from config import WEB

logger = logging.getLogger(__name__)

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
]

_BASE_HEADERS = {
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

_JUNK_TAGS = {
    "script", "style", "noscript", "nav", "footer", "header",
    "aside", "form", "button", "iframe", "svg", "img", "figure",
}

_JUNK_CLASS_ID = re.compile(
    r"^(nav|menu|sidebar|cookie|banner|popup|advertisement|"
    r"breadcrumb|pagination|social\-share|share\-bar|ads?)$",
    re.I,
)

_STOP_WORDS = {
    "de", "do", "da", "dos", "das", "e", "em", "para", "com", "por",
    "um", "uma", "os", "as", "na", "no", "nos", "nas", "ao", "aos",
    "análise", "revisão", "detalhada", "detalhado", "utilizados",
    "etapas", "tipos", "sobre", "entre", "que", "se", "ou",
}

# Sites que sistematicamente bloqueiam scrapers
_BLOCKED_DOMAINS = {
    "youtube.com", "youtu.be", "twitter.com", "x.com",
    "facebook.com", "instagram.com", "tiktok.com", "reddit.com",
    "scribd.com", "studocu.com", "researchgate.net",
    "scielo.br", "pmc.ncbi.nlm.nih.gov", "academia.edu",
    "1library.org", "livrozilla.com",
}

# Mínimo de chars para considerar texto válido
_MIN_TEXT = 120


@dataclass
class WebPage:
    url:   str
    title: str
    text:  str
    query: str


class WebResearcher:

    def __init__(self):
        self._timeout     = httpx.Timeout(WEB.timeout_seconds)
        self._max_results = WEB.max_results_per_query
        self._max_chars   = WEB.max_chars_per_page
        self._delay       = WEB.delay_between_requests

    # ─── API pública ─────────────────────────────────────────────────────────

    async def research_async(
        self,
        sub_topics: List[str],
        verbose: bool = True,
    ) -> List[Tuple[str, dict]]:
        """Versão async direta — use dentro de FastAPI."""
        return await self._research_core(sub_topics, verbose)

    def research(
        self,
        sub_topics: List[str],
        verbose: bool = True,
    ) -> List[Tuple[str, dict]]:
        """Wrapper síncrono para scripts/testes fora de contexto async."""
        try:
            asyncio.get_running_loop()
            import threading
            result_container = []
            exception_container = []

            def run():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result_container.append(
                        loop.run_until_complete(self._research_core(sub_topics, verbose))
                    )
                except Exception as e:
                    exception_container.append(e)
                    logger.error("Erro na thread de pesquisa: %s", e, exc_info=True)
                finally:
                    loop.close()

            t = threading.Thread(target=run, daemon=True)
            t.start()
            t.join()
            if exception_container:
                raise exception_container[0]
            return result_container[0] if result_container else []
        except RuntimeError:
            return asyncio.run(self._research_core(sub_topics, verbose))

    # ─── Core ────────────────────────────────────────────────────────────────

    async def _research_core(
        self,
        sub_topics: List[str],
        verbose: bool,
    ) -> List[Tuple[str, dict]]:
        results: List[Tuple[str, dict]] = []

        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            verify=False,
        ) as client:
            for topic in sub_topics:
                if verbose:
                    print(f"  🌐 Pesquisando: {topic!r}")

                search_results = await self._ddg_search_full(topic, verbose)
                if not search_results:
                    logger.warning("Nenhum resultado de busca para: %r", topic)
                    continue

                if verbose:
                    print(f"       {len(search_results)} resultados encontrados")

                pages = await self._process_results(client, search_results, topic, verbose)

                for page in pages:
                    results.append((
                        page.text,
                        {"source": page.url, "title": page.title, "query": page.query},
                    ))

                if verbose:
                    print(f"       {len(pages)} páginas com conteúdo")

                await asyncio.sleep(self._delay + random.uniform(0.5, 1.5))

        logger.info("WebResearcher coletou %d páginas no total.", len(results))
        return results

    # ─── Busca ───────────────────────────────────────────────────────────────

    async def _ddg_search_full(self, query: str, verbose: bool) -> List[dict]:
        """Retorna lista de {href, title, body} onde body é o snippet."""
        short_query = self._shorten_query(query)
        if verbose:
            print(f"       Query: {short_query!r}")
        try:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None,
                lambda: list(DDGS().text(short_query, max_results=self._max_results))
            )
            return [r for r in raw if self._is_valid_url(r.get("href", ""))]
        except Exception as e:
            logger.warning("DDGS falhou para %r: %s", short_query, e)
            return []

    def _shorten_query(self, topic: str) -> str:
        topic = re.sub(r"\(.*?\)", "", topic)
        topic = re.sub(r"[^\w\s\-]", " ", topic)
        words = [w for w in topic.split() if w.lower() not in _STOP_WORDS and len(w) > 2]
        return " ".join(words[:6])

    # ─── Processamento de resultados ─────────────────────────────────────────

    async def _process_results(
        self,
        client: httpx.AsyncClient,
        search_results: List[dict],
        query: str,
        verbose: bool,
    ) -> List[WebPage]:
        """
        Para cada resultado de busca:
        1. Tenta fetch direto → extrai texto completo
        2. Se falhar → usa o snippet da busca

        Ao final, se TODOS os fetchs falharam, agrega todos os snippets
        em um único WebPage consolidado (garante que sempre há conteúdo).
        """
        sem = asyncio.Semaphore(3)
        pages: List[WebPage] = []
        snippet_fallbacks: List[dict] = []

        async def fetch_one(result: dict) -> Optional[WebPage]:
            async with sem:
                await asyncio.sleep(random.uniform(0.4, 1.0))
                url     = result.get("href", "")
                title   = result.get("title", url)
                snippet = result.get("body", "")

                # Tenta fetch direto
                text = await self._fetch_url(client, url)

                if text:
                    return WebPage(url=url, title=title, text=text[:self._max_chars], query=query)

                # Registra snippet para fallback agregado
                if snippet and len(snippet) >= 80:
                    snippet_fallbacks.append(result)
                return None

        tasks = [fetch_one(r) for r in search_results]
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        pages = [p for p in raw if isinstance(p, WebPage)]

        # Se nenhum fetch funcionou mas há snippets → agrega tudo num único WebPage
        if not pages and snippet_fallbacks:
            aggregated = self._aggregate_snippets(snippet_fallbacks, query)
            if aggregated:
                pages.append(aggregated)
                if verbose:
                    print(f"       ⚠️  Fetch bloqueado — usando {len(snippet_fallbacks)} snippets agregados")

        return pages

    def _aggregate_snippets(self, results: List[dict], query: str) -> Optional[WebPage]:
        """
        Combina snippets de múltiplos resultados em um único texto coerente.
        Usado quando todos os fetchs são bloqueados (ex: IP em datacenter).
        """
        parts = []
        for r in results:
            title   = r.get("title", "")
            snippet = r.get("body", "").strip()
            url     = r.get("href", "")
            if snippet:
                parts.append(f"{title}. {snippet}")

        text = " ".join(parts)
        if len(text) < _MIN_TEXT:
            return None

        # URL do resultado com snippet mais longo como referência
        best = max(results, key=lambda r: len(r.get("body", "")))
        return WebPage(
            url=best.get("href", "aggregated"),
            title=f"Resultados agregados: {query[:50]}",
            text=text[:self._max_chars],
            query=query,
        )

    # ─── Fetch ───────────────────────────────────────────────────────────────

    async def _fetch_url(self, client: httpx.AsyncClient, url: str) -> Optional[str]:
        """Faz fetch de uma URL com UA aleatório. Retorna texto limpo ou None."""
        headers = {**_BASE_HEADERS, "User-Agent": random.choice(_USER_AGENTS)}
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code >= 400:
                return None
            if "text/html" not in resp.headers.get("content-type", ""):
                return None
            text, _ = self._extract_text(resp.text, url)
            return text if len(text) >= _MIN_TEXT else None
        except Exception as e:
            logger.debug("Fetch falhou %s: %s", url, e)
            return None

    # ─── Extração de texto ───────────────────────────────────────────────────

    def _extract_text(self, html: str, url: str) -> Tuple[str, str]:
        soup = BeautifulSoup(html, "lxml")

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else urlparse(url).netloc

        for tag in soup.find_all(_JUNK_TAGS):
            tag.decompose()

        for tag in soup.find_all(True):
            cls_list = tag.get("class", [])
            iid = tag.get("id", "")
            if any(_JUNK_CLASS_ID.match(c) for c in cls_list) or _JUNK_CLASS_ID.match(iid):
                tag.decompose()

        main = (
            soup.find("article") or
            soup.find("main") or
            soup.find(attrs={"role": "main"}) or
            soup.find("div", id=re.compile(r"^(content|main|article|post|entry)$", re.I)) or
            soup.find("div", class_=re.compile(r"^(content|main|article|post|entry|body)$", re.I)) or
            soup.body
        )

        if main is None:
            return "", title

        parts = []
        for el in main.find_all(["p", "h1", "h2", "h3", "h4", "li", "td", "th", "blockquote"]):
            t = el.get_text(separator=" ", strip=True)
            if t and len(t) > 30:
                parts.append(t)

        text = re.sub(r"\s{2,}", " ", " ".join(parts)).strip()
        return text, title

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _is_valid_url(self, url: str) -> bool:
        if not url.startswith(("http://", "https://")):
            return False
        domain = urlparse(url).netloc.lower().replace("www.", "")
        if any(s in domain for s in _BLOCKED_DOMAINS):
            return False
        if url.lower().endswith(".pdf"):
            return False
        return True