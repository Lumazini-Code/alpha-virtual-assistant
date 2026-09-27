from __future__ import annotations

"""
memory_engines.py — Motores de embedding e rerank (ONNX Serving API).

EmbeddingEngine (E5: query/passage prefixados) e RerankEngine
(cross-encoder p/ confirmação de duplicata/update).
"""

import numpy as np

# memory_config primeiro: ele faz o bootstrap que coloca a pasta-mãe (Modules/)
# no sys.path, tornando o import de onnx_client abaixo possível quando este
# pacote roda a partir do próprio diretório (Modules/memory/).
from memory_config import (
    EMBED_DIM,
    EMBED_PASSAGE_PREFIX,
    EMBED_QUERY_PREFIX,
    ONNX_SERVING_URL,
    log,
)

from onnx_client import EmbeddingClient, RerankerClient

# ── MODIFIED: EmbeddingEngine now delegates to EmbeddingClient ────────────────

class EmbeddingEngine:
    """
    Motor de embeddings via ONNX Serving API.

    IMPORTANTE (E5): toda chamada precisa dizer se o texto é uma QUERY (algo
    que vai ser usado para *buscar*) ou uma PASSAGE (algo que vai ser
    *armazenado* e depois encontrado por uma query). Os métodos genéricos
    antigos (embed/embed_one/embed_batch_two) foram removidos de propósito —
    cada call site abaixo precisa escolher explicitamente `_query`/`_passage`,
    pra não reintroduzir por engano uma chamada sem prefixo.
    """

    def __init__(self, base_url: str = ONNX_SERVING_URL):
        self._client = EmbeddingClient(base_url=base_url)
        log.info(f"EmbeddingEngine carregado — via API: {base_url}")

    @staticmethod
    def _prefixed(texts: list[str], prefix: str) -> list[str]:
        return [f"{prefix}{t}" for t in texts]

    # ── lado "documento" (o que fica gravado/indexado) ──────────────────────

    async def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        return await self._client.embed(self._prefixed(texts, EMBED_PASSAGE_PREFIX))

    async def embed_passage_one(self, text: str) -> np.ndarray:
        result = await self.embed_passages([text])
        return result[0]

    # ── lado "busca" (o que vai comparar contra o índice) ───────────────────

    async def embed_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        return await self._client.embed(self._prefixed(texts, EMBED_QUERY_PREFIX))

    async def embed_query_one(self, text: str) -> np.ndarray:
        result = await self.embed_queries([text])
        return result[0]

    async def embed_query_two(self, text_a: str, text_b: str) -> tuple[np.ndarray, np.ndarray]:
        results = await self.embed_queries([text_a, text_b])
        return results[0], results[1]

    @property
    def client(self) -> EmbeddingClient:
        return self._client


class RerankEngine:
    """
    Motor de rerank via ONNX Serving API (/v1/score) — usado para CONFIRMAR
    duplicata/atualização quando o bi-encoder (cosine) encontra um candidato
    na faixa "parecido", em vez de decidir só pelo cosine bruto (que sofre de
    anisotropia em modelos multilíngues pequenos como o multilingual-e5-small
    — ver comentário perto de DEDUP_THRESHOLD/UPDATE_SIM_THRESHOLD).
    """

    def __init__(self, base_url: str = ONNX_SERVING_URL):
        self._client = RerankerClient(base_url=base_url)

    async def score_one(self, text_a: str, text_b: str) -> float:
        """Score cross-encoder sigmoid [0,1] de quão relacionados text_a e
        text_b são — não confundir com a cosine do bi-encoder."""
        scores = await self._client.score(text_a, [text_b])
        return float(scores[0]) if len(scores) else 0.0

    async def score_many(self, query: str, passages: list[str]) -> list[float]:
        """Scores cross-encoder em lote (1 round-trip /v1/score) — usado pelo
        rerank do /read: reordena os candidatos primary contra a query real.
        Retorna lista alinhada com `passages` (0.0 para furos vazios)."""
        if not passages:
            return []
        scores = await self._client.score(query, passages)
        out = [float(s) for s in scores]
        if len(out) < len(passages):
            out.extend([0.0] * (len(passages) - len(out)))
        return out[:len(passages)]

    @property
    def client(self) -> RerankerClient:
        return self._client


__all__ = [
    'EmbeddingEngine',
    'RerankEngine',
]
