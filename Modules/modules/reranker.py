"""
AVA KG-RAG — Cross-Encoder Reranker (via ONNX Serving API)
Substitui o carregamento local do ONNX por chamadas HTTP ao servidor unificado.
Mantém a mesma interface pública — drop-in replacement.
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

from onnx_client import RerankerClient, DEFAULT_ONNX_BASE_URL

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranker cross-encoder via API REST (onnx_serving).
    Interface idêntica à versão ONNX local — substituição transparente.
    """

    def __init__(
        self,
        onnx_path: str = None,         # Ignorado — mantido para compatibilidade
        tokenizer_path: str = None,    # Ignorado — mantido para compatibilidade
        base_url: str = DEFAULT_ONNX_BASE_URL,
    ):
        self._client = RerankerClient(base_url=base_url)
        self._base_url = base_url
        logger.info("CrossEncoderReranker iniciado — via API: %s", base_url)

    # ─── API pública ─────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: int = None,
    ) -> List[Tuple[int, float, str]]:
        """
        Reordena candidatos por relevância à query.

        Retorna lista de (índice_original, score, texto) ordenada do mais
        ao menos relevante, truncada em top_k.

        ATENÇÃO: Método síncrono por compatibilidade. Prefira rerank_async()
        quando em contexto assíncrono.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        asyncio.run,
                        self._client.rerank(query, candidates, top_k)
                    )
                    return future.result()
        except RuntimeError:
            pass
        return asyncio.run(self._client.rerank(query, candidates, top_k))

    def score(self, query: str, passages: List[str]) -> np.ndarray:
        """
        Raw cross-encoder scores — retorna array de scores sigmoid-normalizados.

        ATENÇÃO: Método síncrono por compatibilidade. Prefira score_async()
        quando em contexto assíncrono.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        asyncio.run,
                        self._client.score(query, passages)
                    )
                    return future.result()
        except RuntimeError:
            pass
        return asyncio.run(self._client.score(query, passages))

    async def rerank_async(
        self,
        query: str,
        candidates: List[str],
        top_k: int = None,
    ) -> List[Tuple[int, float, str]]:
        """Versão assíncrona nativa — preferida em contexto async."""
        return await self._client.rerank(query, candidates, top_k)

    async def score_async(self, query: str, passages: List[str]) -> np.ndarray:
        """Versão assíncrona nativa — preferida em contexto async."""
        return await self._client.score(query, passages)

    @property
    def client(self) -> RerankerClient:
        """Acesso direto ao cliente HTTP para chamadas assíncronas avançadas."""
        return self._client
