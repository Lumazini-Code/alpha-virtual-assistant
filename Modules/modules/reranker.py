"""
AVA KG-RAG — Cross-Encoder Reranker (ONNX Runtime)
Modelo: ms-marco-MiniLM-L-6-v2
Entrada: (query, passagem) → score de relevância
"""

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from typing import List, Tuple
from pathlib import Path
import logging

from config import RERANKER_ONNX, RERANKER_TOKENIZER, RETRIEVAL

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranker cross-encoder via ONNX.
    Recebe candidatos (texto) + query e retorna os top_k mais relevantes.
    """

    def __init__(
        self,
        onnx_path: str  = RERANKER_ONNX,
        tokenizer_path: str = RERANKER_TOKENIZER,
    ):
        if not Path(onnx_path).exists():
            raise FileNotFoundError(
                f"Reranker ONNX não encontrado: {onnx_path}\n"
                "Baixe ms-marco-MiniLM-L-6-v2 em ONNX e coloque em models/."
            )

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 2

        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._tokenizer.enable_truncation(max_length=512)

        logger.info("CrossEncoderReranker iniciado — %s", Path(onnx_path).name)

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
        """
        if not candidates:
            return []

        top_k = top_k or RETRIEVAL.top_k_final

        # Tokeniza pares (query, passagem) — cross-encoder espera sequência A+B
        pairs = [f"{query} [SEP] {c}" for c in candidates]
        encoded = self._tokenizer.encode_batch(pairs)

        input_ids      = np.array([e.ids           for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        input_names = {i.name for i in self._session.get_inputs()}
        ort_inputs: dict = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in input_names:
            ort_inputs["token_type_ids"] = np.zeros_like(input_ids)

        # logits shape: (batch, 1) ou (batch, 2)
        logits: np.ndarray = self._session.run(None, ort_inputs)[0]

        # Extrai score de relevância
        if logits.shape[-1] == 1:
            scores = logits[:, 0]
        else:
            # Modelos de classificação binária — usa logit da classe positiva
            scores = logits[:, 1]

        # Sigmoid para converter em probabilidade 0-1
        scores = 1.0 / (1.0 + np.exp(-scores))

        ranked = sorted(
            [(i, float(scores[i]), candidates[i]) for i in range(len(candidates))],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]
