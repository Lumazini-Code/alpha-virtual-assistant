"""
AVA KG-RAG — Semantic Chunker
Quebra texto em chunks com base em distância cosseno entre embeddings de sentenças.
Muito superior ao chunking por tokens fixos — respeita fronteiras semânticas reais.
"""

import re
import numpy as np
import logging
from typing import List
from dataclasses import dataclass

from config import RETRIEVAL

logger = logging.getLogger(__name__)


@dataclass
class SemanticChunk:
    text: str
    sentences: List[str]
    start_sentence: int   # Índice da primeira sentença no documento original
    embedding: np.ndarray = None  # Preenchido após embed


# Abreviações que NÃO devem quebrar sentença — compiladas uma vez
_ABBREV_PATTERN = re.compile(
    r"\b(Dr|Sr|Sra|Prof|Fig|Ref|vs|etc|p\.ex|e\.g|i\.e)\.\s+(?=[a-záéíóúàâêôãõç])",
    re.IGNORECASE,
)

# Quebra de sentença: ponto/!/? seguido de espaço + maiúscula
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ\"\(])")


class SemanticChunker:
    """
    Chunker semântico baseado em distância cosseno entre embeddings consecutivos.

    Algoritmo:
    1. Tokeniza texto em sentenças
    2. Gera embedding para cada sentença
    3. Calcula distância cosseno entre sentença[i] e sentença[i+1]
    4. Quebra onde a distância excede o percentil configurado (padrão: 95°)
    5. Retorna chunks com texto completo e metadados
    """

    def __init__(self, embedding_engine=None, percentile: float = None):
        self._embed         = embedding_engine
        self._pct           = percentile or RETRIEVAL.sem_chunk_pct
        self._min_sentences = 2
        self._max_sentences = 20

    def chunk(self, text: str) -> List[SemanticChunk]:
        """
        Divide texto em chunks semanticamente coerentes.
        Requer embedding_engine configurado.
        """
        if self._embed is None:
            raise RuntimeError("SemanticChunker precisa de um EmbeddingEngine para chunking semântico.")

        sentences = self._split_sentences(text)
        if len(sentences) <= self._min_sentences:
            return [SemanticChunk(text=text, sentences=sentences, start_sentence=0)]

        embeddings = self._embed.embed_passages(sentences)  # (N, D)
        distances  = self._consecutive_distances(embeddings)
        threshold  = float(np.percentile(distances, self._pct))
        breakpoints = [i + 1 for i, d in enumerate(distances) if d > threshold]

        chunks = self._build_chunks(sentences, embeddings, breakpoints)
        logger.debug("SemanticChunker: %d sentenças → %d chunks", len(sentences), len(chunks))
        return chunks

    def chunk_simple(self, text: str, max_chars: int = 1000) -> List[SemanticChunk]:
        """
        Fallback sem embedding engine: chunking por parágrafos/caracteres.
        """
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: List[SemanticChunk] = []
        current_sents: List[str] = []
        current_chars = 0
        sentence_idx  = 0

        for para in paragraphs:
            sents = self._split_sentences(para)
            for sent in sents:
                if current_chars + len(sent) > max_chars and current_sents:
                    chunks.append(SemanticChunk(
                        text=" ".join(current_sents),
                        sentences=current_sents[:],
                        start_sentence=sentence_idx - len(current_sents),
                    ))
                    current_sents = []
                    current_chars = 0
                current_sents.append(sent)
                current_chars += len(sent)
                sentence_idx  += 1

        if current_sents:
            chunks.append(SemanticChunk(
                text=" ".join(current_sents),
                sentences=current_sents,
                start_sentence=sentence_idx - len(current_sents),
            ))
        return chunks

    # ─── Internals ───────────────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> List[str]:
        """
        Tokenização de sentenças robusta para PT-BR e EN.

        Estratégia:
        1. Substitui temporariamente pontos de abreviações conhecidas por placeholder
        2. Divide no padrão ponto/!/? + espaço + maiúscula
        3. Restaura os placeholders
        """
        text = re.sub(r"\s+", " ", text.strip())

        # Substitui "Dr. " / "Sr. " etc. por placeholder para não quebrar ali
        placeholder = "\x00"
        protected = _ABBREV_PATTERN.sub(
            lambda m: m.group(0).replace(". ", placeholder),
            text,
        )

        parts = _SENTENCE_SPLIT.split(protected)

        # Restaura placeholders e limpa
        sentences = [s.replace(placeholder, ". ").strip() for s in parts if s.strip()]
        return sentences

    def _consecutive_distances(self, embeddings: np.ndarray) -> List[float]:
        """Distância cosseno entre embeddings consecutivos (1 - similaridade)."""
        distances = []
        for i in range(len(embeddings) - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            distances.append(1.0 - sim)
        return distances

    def _build_chunks(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        breakpoints: List[int],
    ) -> List[SemanticChunk]:
        """Constrói chunks a partir dos pontos de quebra."""
        chunks: List[SemanticChunk] = []
        starts = [0] + breakpoints
        ends   = breakpoints + [len(sentences)]

        for start, end in zip(starts, ends):
            for sub_start in range(start, end, self._max_sentences):
                sub_end     = min(sub_start + self._max_sentences, end)
                chunk_sents = sentences[sub_start:sub_end]
                if not chunk_sents:
                    continue

                chunk_emb = embeddings[sub_start:sub_end].mean(axis=0)
                norm      = np.linalg.norm(chunk_emb) + 1e-9
                chunk_emb = (chunk_emb / norm).astype(np.float32)

                chunks.append(SemanticChunk(
                    text=" ".join(chunk_sents),
                    sentences=chunk_sents,
                    start_sentence=sub_start,
                    embedding=chunk_emb,
                ))
        return chunks