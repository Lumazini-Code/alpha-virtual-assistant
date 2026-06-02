"""
AVA KG-RAG — Motor de Embeddings (via ONNX Serving API)
Substitui o carregamento local do ONNX por chamadas HTTP ao servidor unificado.
Mantém a mesma interface pública — drop-in replacement.
"""

import numpy as np
from typing import List, Optional
import logging

from onnx_client import EmbeddingClient, DEFAULT_ONNX_BASE_URL

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Motor de embeddings via API REST (onnx_serving).
    Compatível com multilingual-e5-small (prefixos query:/passage:).
    Interface idêntica à versão ONNX local — substituição transparente.
    """

    def __init__(
        self,
        onnx_path: str = None,         # Ignorado — mantido para compatibilidade
        tokenizer_path: str = None,    # Ignorado — mantido para compatibilidade
        base_url: str = DEFAULT_ONNX_BASE_URL,
    ):
        self._client = EmbeddingClient(base_url=base_url)
        self._base_url = base_url
        logger.info("EmbeddingEngine iniciado — via API: %s", base_url)

    # ─── API pública ─────────────────────────────────────────────────────────

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed batch de textos (sem prefixo).
        ATENÇÃO: Este método é síncrono por compatibilidade, mas faz chamada HTTP assíncrona
        internamente via asyncio.run(). Prefira embed_async() quando possível.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Estamos dentro de um event loop — usar nest_asyncio ou thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, self._client.embed(texts))
                    return future.result()
        except RuntimeError:
            pass
        return asyncio.run(self._client.embed(texts))

    def embed_one(self, text: str) -> np.ndarray:
        """Embed de texto único — retorna vetor 1D."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, self._client.embed_one(text))
                    return future.result()
        except RuntimeError:
            pass
        return asyncio.run(self._client.embed_one(text))

    def embed_batch_two(self, text_a: str, text_b: str) -> tuple:
        """Dois textos numa única chamada — evita overhead duplo."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, self._client.embed_batch_two(text_a, text_b))
                    return future.result()
        except RuntimeError:
            pass
        return asyncio.run(self._client.embed_batch_two(text_a, text_b))

    async def embed_async(self, texts: List[str]) -> np.ndarray:
        """Embed batch — versão assíncrona nativa (preferida em contexto async)."""
        return await self._client.embed(texts)

    async def embed_query(self, text: str) -> np.ndarray:
        """Embed de query — usa prefixo 'query:' (padrão e5)."""
        return await self._client.embed_query(text)

    async def embed_passages(self, texts: List[str]) -> np.ndarray:
        """Embed de passagens/chunks — usa prefixo 'passage:'. """
        return await self._client.embed_passages(texts)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Similaridade cosseno entre dois vetores normalizados."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def cosine_matrix(self, query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Similaridade cosseno entre query (1D) e matriz de embeddings (N×D)."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
        normalized = matrix / norms
        return normalized @ query / (np.linalg.norm(query) + 1e-9)

    @property
    def client(self) -> EmbeddingClient:
        """Acesso direto ao cliente HTTP para chamadas assíncronas avançadas."""
        return self._client
