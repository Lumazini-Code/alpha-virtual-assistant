"""
AVA KG-RAG — Motor de Embeddings (ONNX Runtime)
Usa tokenizers Rust (sem transformers) — padrão do AVA.
Implementa mean pooling correto com attention mask.
"""

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from pathlib import Path
from typing import List, Union
import logging

from config import EMBEDDING_ONNX, EMBEDDING_TOKENIZER, RETRIEVAL

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Motor de embeddings sobre ONNX Runtime.
    Compatível com multilingual-e5-small (prefixos query:/passage:).
    Usa tokenizers Rust — sem overhead do transformers.
    """

    def __init__(
        self,
        onnx_path: str = EMBEDDING_ONNX,
        tokenizer_path: str = EMBEDDING_TOKENIZER,
    ):
        if not Path(onnx_path).exists():
            raise FileNotFoundError(
                f"Modelo ONNX não encontrado: {onnx_path}\n"
                "Baixe o multilingual-e5-small convertido para ONNX e coloque em models/."
            )

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4

        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._tokenizer.enable_truncation(max_length=512)

        logger.info("EmbeddingEngine iniciado — %s", Path(onnx_path).name)

    # ─── API pública ─────────────────────────────────────────────────────────

    def embed_query(self, text: str) -> np.ndarray:
        """Embed de query — usa prefixo 'query:' (padrão e5)."""
        return self._embed_batch([f"query: {text}"])[0]

    def embed_passages(self, texts: List[str]) -> np.ndarray:
        """Embed de passagens/chunks — usa prefixo 'passage:'."""
        prefixed = [f"passage: {t}" for t in texts]
        return self._embed_batch(prefixed)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Similaridade cosseno entre dois vetores normalizados."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def cosine_matrix(self, query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Similaridade cosseno entre query (1D) e matriz de embeddings (N×D)."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
        normalized = matrix / norms
        return normalized @ query / (np.linalg.norm(query) + 1e-9)

    # ─── Internals ───────────────────────────────────────────────────────────

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Executa ONNX inference e aplica mean pooling com attention mask.
        Retorna embeddings L2-normalizados — prontos para cosine similarity.
        """
        encoded = self._tokenizer.encode_batch(texts)

        input_ids      = np.array([e.ids          for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        # Alguns modelos e5 também usam token_type_ids
        input_names = {i.name for i in self._session.get_inputs()}
        ort_inputs: dict = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in input_names:
            ort_inputs["token_type_ids"] = np.zeros_like(input_ids)

        # last_hidden_state shape: (batch, seq_len, hidden_dim)
        last_hidden: np.ndarray = self._session.run(None, ort_inputs)[0]

        # Mean pooling — média apenas sobre tokens não-padding
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_hidden    = (last_hidden * mask_expanded).sum(axis=1)
        count         = mask_expanded.sum(axis=1).clip(min=1e-9)
        pooled        = sum_hidden / count                              # (batch, hidden_dim)

        # L2 normalização — compatível com FAISS cosine (IndexFlatIP após normalize)
        norms     = np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-9
        return (pooled / norms).astype(np.float32)
