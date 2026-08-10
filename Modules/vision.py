"""
AVA Vision API — Pipeline de "Tradução de Objetos"
=====================================================

Implementa o pipeline descrito no projeto:

    imagem → [1] depth estimation (Depth Anything V2)
           → clusterização por camadas de profundidade (k-means / watershed)
           → pontos amostrados por camada (viram prompts)
           → [3] segmentação class-agnostic (EdgeSAM)
           → [4] pós-processamento / verificação de consistência
           → crops finais (um por objeto)
           → [5] embeddings visuais (DINOv3, CPU-otimizado)
           → [6]/[7] consulta ao "dicionário" (kNN) — delegada ao memory.py
           → [8] NÃO chama nenhum LLM aqui: apenas retorna as imagens (crops)
                 + os candidatos de significado recuperados. Quem decide o
                 que fazer com isso (perguntar ao usuário em caso de
                 ambiguidade, mandar pro LLM final etc.) é o orquestrador
                 externo, não este módulo.

Este arquivo NÃO persiste nenhum estado de dicionário. Todo o "dicionário
visual" (embeddings de exemplo + significado textual de cada conceito) é
lido e escrito no memory.py, via HTTP (endpoints /visual-dict/write e
/visual-dict/read) — exatamente como pedido: os dicionários vivem no
memory.py, e a parte textual deles é automaticamente pesquisável pela
leitura normal de memória (/read), junto com as memórias de curto e longo
prazo.

Segue o mesmo estilo arquitetural do memory.py: FastAPI + Pydantic + httpx,
API REST simples, sem estado externo além dos modelos ONNX carregados em
memória.
"""

from __future__ import annotations

import os
import io
import base64
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

import cv2
import httpx
import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.cluster import KMeans
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Carrega variáveis de um arquivo .env na pasta atual, se existir — assim as
# configurações (caminhos dos modelos etc.) não precisam ser reexportadas a
# cada sessão de terminal. Opcional: se python-dotenv não estiver instalado,
# só usa as variáveis de ambiente já exportadas no shell.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Configuração ────────────────────────────────────────────────────────────
# Tudo ajustável via variável de ambiente, sem precisar tocar no código —
# principalmente os caminhos dos modelos ONNX, que cada um exporta/baixa
# separadamente (ver README_vision.md).

def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default

def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# memory.py — onde o dicionário visual (e a memória normal) é lido/escrito
MEMORY_API_URL = _env_str("VISION_MEMORY_API_URL", "http://localhost:3001")

# Modelos — caminhos default assumem uma pasta ./models ao lado deste arquivo.
# Todos são opcionais: se o arquivo não existir, o componente correspondente
# fica "indisponível" e o /vision/process retorna 503 (sem derrubar a API).
DEPTH_MODEL_PATH = _env_str("VISION_DEPTH_MODEL_PATH", "./Models/depth_anything_v2_small_FP32.onnx")
DEPTH_INPUT_SIZE = _env_int("VISION_DEPTH_INPUT_SIZE", 518)  # padrão da exportação oficial DA-V2

EDGESAM_ENCODER_PATH = _env_str("VISION_EDGESAM_ENCODER_PATH", "./Models/EdgeSAM/edge_sam_3x_encoder.onnx")
EDGESAM_DECODER_PATH = _env_str("VISION_EDGESAM_DECODER_PATH", "./Models/EdgeSAM/edge_sam_3x_decoder.onnx")
EDGESAM_INPUT_SIZE   = _env_int("VISION_EDGESAM_INPUT_SIZE", 1024)  # padrão estilo-SAM
SAM_SCORE_THRESH     = _env_float("VISION_SAM_SCORE_THRESH", 0.75)
# Override manual — só necessário se a auto-detecção (por nome/formato) não
# acertar os outputs do seu export específico do decoder. Descubra os nomes
# reais rodando o servidor e olhando o log "EdgeSAM decoder outputs: ..."
# na primeira chamada, ou inspecionando o .onnx com o Netron.
EDGESAM_MASKS_OUTPUT_NAME  = _env_str("VISION_EDGESAM_MASKS_OUTPUT_NAME", "")
EDGESAM_SCORES_OUTPUT_NAME = _env_str("VISION_EDGESAM_SCORES_OUTPUT_NAME", "")

# DINOv3 — o embedder visual. Suporta um modelo fp32 (qualidade máxima) e,
# opcionalmente, um int8 já quantizado (mais rápido em CPU). Se só o fp32
# existir e AUTO_QUANTIZE estiver ligado, quantiza uma vez no startup e
# reaproveita o arquivo int8 nas próximas execuções.
DINOV3_FP32_MODEL_PATH = _env_str("VISION_DINOV3_FP32_MODEL_PATH", "./Models/dinov3/dinov3_vits_FP32.onnx")
DINOV3_INT8_MODEL_PATH = _env_str("VISION_DINOV3_INT8_MODEL_PATH", "./Models/dinov3/dinov3_vits_FP32.onnx")
DINOV3_INPUT_SIZE      = _env_int("VISION_DINOV3_INPUT_SIZE", 224)
USE_INT8_EMBEDDER      = _env_bool("VISION_USE_INT8_EMBEDDER", True)
AUTO_QUANTIZE          = _env_bool("VISION_AUTO_QUANTIZE", True)

# CPU tuning — onnxruntime
CPU_THREADS = _env_int("VISION_CPU_THREADS", max(1, os.cpu_count() or 4))

# Clusterização por profundidade
N_DEPTH_LAYERS   = _env_int("VISION_N_DEPTH_LAYERS", 4)
POINTS_PER_LAYER = _env_int("VISION_POINTS_PER_LAYER", 3)
CLUSTER_METHOD   = _env_str("VISION_CLUSTER_METHOD", "kmeans")  # "kmeans" | "watershed"

# Filtragem / dedup de máscaras
MIN_MASK_AREA_RATIO = _env_float("VISION_MIN_MASK_AREA_RATIO", 0.002)
MAX_MASK_AREA_RATIO = _env_float("VISION_MAX_MASK_AREA_RATIO", 0.90)
MASK_IOU_NMS_THRESH  = _env_float("VISION_MASK_IOU_NMS_THRESH", 0.85)
# Supressão por contenção: quando uma máscara grande engloba quase inteira
# uma máscara menor (ex.: "objeto + metade do fundo" contendo "só o
# objeto"), descarta a maior e fica só a menor/mais específica.
CONTAINMENT_SUPPRESS_THRESH = _env_float("VISION_CONTAINMENT_SUPPRESS_THRESH", 0.90)
CONTAINMENT_SIZE_RATIO      = _env_float("VISION_CONTAINMENT_SIZE_RATIO", 1.30)

# Pós-processamento / verificação de consistência
DEPTH_STD_RESPLIT_THRESH  = _env_float("VISION_DEPTH_STD_RESPLIT_THRESH", 0.15)
DEPTH_MERGE_EPS           = _env_float("VISION_DEPTH_MERGE_EPS", 0.03)
COLOR_EDGE_MERGE_THRESH   = _env_float("VISION_COLOR_EDGE_MERGE_THRESH", 12.0)

# Crops finais
CROP_PADDING_RATIO = _env_float("VISION_CROP_PADDING_RATIO", 0.05)
CROP_MODE          = _env_str("VISION_CROP_MODE", "masked")  # "bbox" | "masked" (RGBA c/ fundo transparente)

# Dicionário visual — defaults espelham os do memory.py (VD_*)
DICT_TOP_K     = _env_int("VISION_DICT_TOP_K", 5)
DICT_MIN_SCORE = _env_float("VISION_DICT_MIN_SCORE", 0.55)
MAX_OBJECTS_PER_IMAGE = _env_int("VISION_MAX_OBJECTS", 20)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [Vision] %(message)s")
log = logging.getLogger("ava.vision")


# ── Sessões ONNX Runtime — otimizadas para CPU ─────────────────────────────

def _make_ort_session(model_path: str, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads   = threads
    so.inter_op_num_threads   = 1
    so.execution_mode         = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern    = True
    so.enable_cpu_mem_arena  = True
    return ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])


def _ensure_quantized(fp32_path: str, int8_path: str) -> tuple[str, bool]:
    """
    Retorna (caminho_a_usar, usou_int8). Prioriza int8 se já existir; se só
    o fp32 existir e AUTO_QUANTIZE=True, gera a versão int8 uma única vez
    (dynamic quantization — não precisa de dataset de calibração) e passa a
    usá-la. Se a quantização falhar por qualquer motivo, cai de volta pro
    fp32 sem derrubar a API.

    Quantização dinâmica reduz o modelo ~4x e costuma manter a maior parte
    da acurácia de retrieval em embedders tipo DINO — mas SEMPRE vale medir
    a perda em cima do seu próprio dataset antes de assumir em produção
    (ver nota do projeto sobre isso).

    Nota: se VISION_DINOV3_FP32_MODEL_PATH e VISION_DINOV3_INT8_MODEL_PATH
    apontarem pro MESMO arquivo, essa função "acha" que já existe um int8
    pronto (o caminho existe) mesmo sendo o fp32 — configure os dois
    caminhos como arquivos DIFERENTES pra esse mecanismo funcionar direito.
    """
    if not USE_INT8_EMBEDDER:
        return fp32_path, False
    if int8_path == fp32_path:
        log.warning(
            "VISION_DINOV3_FP32_MODEL_PATH e VISION_DINOV3_INT8_MODEL_PATH apontam pro mesmo "
            "arquivo — usando-o como fp32 (não há quantização real acontecendo)."
        )
        return fp32_path, False
    if Path(int8_path).exists():
        return int8_path, True
    if not AUTO_QUANTIZE or not Path(fp32_path).exists():
        return fp32_path, False
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        Path(int8_path).parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Quantizando {fp32_path} → int8 (uma única vez)...")
        quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
        log.info(f"Modelo int8 salvo em {int8_path}")
        return int8_path, True
    except Exception as e:
        log.warning(f"Falha ao quantizar {fp32_path} ({e}) — usando fp32")
        return fp32_path, False


# ── [1] Depth Estimation (Depth Anything V2) ───────────────────────────────

class DepthEstimator:
    def __init__(self, model_path: str, input_size: int = DEPTH_INPUT_SIZE, threads: int = CPU_THREADS):
        self.input_size = input_size
        self._session: Optional[ort.InferenceSession] = None
        self._available = False
        if Path(model_path).exists():
            try:
                self._session = _make_ort_session(model_path, threads)
                self._input_name  = self._session.get_inputs()[0].name
                self._output_name = self._session.get_outputs()[0].name
                self._available = True
                log.info(f"DepthEstimator carregado: {model_path} (threads={threads})")
            except Exception as e:
                log.error(f"Falha ao carregar DepthEstimator ({model_path}): {e}")
        else:
            log.warning(
                f"Modelo de depth não encontrado em '{model_path}' — "
                f"/vision/process ficará indisponível até configurar VISION_DEPTH_MODEL_PATH"
            )

    @property
    def available(self) -> bool:
        return self._available

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        x = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        return x.transpose(2, 0, 1)[None, ...].astype(np.float32)

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Retorna mapa de profundidade normalizado em [0, 1], no tamanho original."""
        if not self._available:
            raise RuntimeError("Depth model não carregado")
        h, w = image.shape[:2]
        x = self._preprocess(image)
        out = self._session.run([self._output_name], {self._input_name: x})[0]
        depth = np.squeeze(out).astype(np.float32)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)
        return depth


# ── Clusterização por camadas de profundidade + amostragem de pontos ──────

class DepthClusterer:
    def __init__(
        self,
        n_layers: int = N_DEPTH_LAYERS,
        points_per_layer: int = POINTS_PER_LAYER,
        method: str = CLUSTER_METHOD,
    ):
        self.n_layers = n_layers
        self.points_per_layer = points_per_layer
        self.method = method

    def cluster(self, depth_map: np.ndarray) -> list[np.ndarray]:
        """Retorna uma lista de máscaras booleanas (uma por camada), ordenadas
        da mais próxima (menor profundidade normalizada) pra mais distante."""
        if self.method == "watershed":
            return self._cluster_watershed(depth_map)
        return self._cluster_kmeans(depth_map)

    def _cluster_kmeans(self, depth_map: np.ndarray) -> list[np.ndarray]:
        h, w = depth_map.shape
        flat = depth_map.reshape(-1, 1)
        n_unique = len(np.unique(np.round(flat, 3)))
        k = max(1, min(self.n_layers, n_unique))
        km = KMeans(n_clusters=k, n_init=4, random_state=0)
        labels = km.fit_predict(flat).reshape(h, w)
        order = np.argsort([
            depth_map[labels == i].mean() if np.any(labels == i) else 1e9
            for i in range(k)
        ])
        return [labels == i for i in order]

    def _cluster_watershed(self, depth_map: np.ndarray) -> list[np.ndarray]:
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
        from scipy import ndimage as ndi

        d = (depth_map * 255).astype(np.uint8)
        distance = ndi.distance_transform_edt(255 - d)
        coords = peak_local_max(distance, num_peaks=self.n_layers)
        mask = np.zeros(distance.shape, dtype=bool)
        if coords.size:
            mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers)
        uniq = [u for u in np.unique(labels) if u != 0]
        return [labels == u for u in uniq]

    def sample_points(self, layer_mask: np.ndarray) -> list[tuple[int, int]]:
        """Amostra pontos "centrais" da camada (máximos locais da distance
        transform) — bons prompts pra segmentação, porque ficam longe da
        borda do cluster e tendem a cair dentro de um único objeto."""
        if not np.any(layer_mask):
            return []
        dist = cv2.distanceTransform(layer_mask.astype(np.uint8), cv2.DIST_L2, 5)
        pts: list[tuple[int, int]] = []
        d = dist.copy()
        for _ in range(self.points_per_layer):
            idx = np.unravel_index(np.argmax(d), d.shape)
            if d[idx] <= 0:
                break
            y, x = int(idx[0]), int(idx[1])
            pts.append((x, y))
            rr = max(5, int(d[idx] * 0.6))
            y0, y1 = max(0, y - rr), min(d.shape[0], y + rr)
            x0, x1 = max(0, x - rr), min(d.shape[1], x + rr)
            d[y0:y1, x0:x1] = 0
        return pts


# ── [3] Proposer / Segmentação class-agnostic (EdgeSAM) ────────────────────

class EdgeSAMSegmenter:
    """
    Wrapper para um EdgeSAM exportado em duas partes (padrão estilo-SAM):
    um encoder de imagem (roda 1x por imagem) e um decoder de prompts (roda
    1x por ponto). Os nomes exatos de input/output variam conforme a
    ferramenta de exportação usada — o decoder tenta detectar quais inputs
    o grafo espera e só preenche esses.
    """

    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        input_size: int = EDGESAM_INPUT_SIZE,
        threads: int = CPU_THREADS,
    ):
        self.input_size = input_size
        self._enc: Optional[ort.InferenceSession] = None
        self._dec: Optional[ort.InferenceSession] = None
        self._available = False
        self._outputs_logged = False
        if Path(encoder_path).exists() and Path(decoder_path).exists():
            try:
                self._enc = _make_ort_session(encoder_path, threads)
                self._dec = _make_ort_session(decoder_path, threads)
                self._available = True
                log.info(f"EdgeSAM carregado (encoder={encoder_path}, decoder={decoder_path})")
            except Exception as e:
                log.error(f"Falha ao carregar EdgeSAM: {e}")
        else:
            log.warning(
                "Modelos EdgeSAM não encontrados — configure "
                "VISION_EDGESAM_ENCODER_PATH / VISION_EDGESAM_DECODER_PATH"
            )

    @property
    def available(self) -> bool:
        return self._available

    def encode_image(self, image: np.ndarray) -> dict:
        h, w = image.shape[:2]
        resized = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        x = (x - mean) / std
        x = x.transpose(2, 0, 1)[None, ...].astype(np.float32)

        in_name  = self._enc.get_inputs()[0].name
        out_name = self._enc.get_outputs()[0].name
        emb = self._enc.run([out_name], {in_name: x})[0]
        return {"embedding": emb, "orig_size": (h, w)}

    def _pick_mask_and_score_outputs(
        self, out_names: list[str], outputs: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Identifica, entre os outputs do decoder, qual é o tensor de máscaras
        e qual é o de scores (iou_predictions) — os nomes/posições variam
        conforme a ferramenta usada pra exportar o EdgeSAM.
        """
        # 1) override explícito via env var, se configurado
        if EDGESAM_MASKS_OUTPUT_NAME and EDGESAM_SCORES_OUTPUT_NAME:
            if EDGESAM_MASKS_OUTPUT_NAME in out_names and EDGESAM_SCORES_OUTPUT_NAME in out_names:
                return (
                    outputs[out_names.index(EDGESAM_MASKS_OUTPUT_NAME)],
                    outputs[out_names.index(EDGESAM_SCORES_OUTPUT_NAME)],
                )
            log.warning(
                "VISION_EDGESAM_MASKS_OUTPUT_NAME/SCORES_OUTPUT_NAME configurados mas não "
                f"encontrados nos outputs do decoder ({out_names}) — caindo pra auto-detecção"
            )

        # 2) por convenção de nome (padrão SAM/EdgeSAM: "masks" + "iou_predictions")
        mask_by_name  = [o for n, o in zip(out_names, outputs) if "mask" in n.lower() and "low_res" not in n.lower()]
        score_by_name = [o for n, o in zip(out_names, outputs) if "iou" in n.lower() or "score" in n.lower()]
        if len(mask_by_name) == 1 and len(score_by_name) == 1:
            return mask_by_name[0], score_by_name[0]

        # 3) fallback por formato: máscara = tensor com >=3 dims; score = tensor
        # pequeno (poucos elementos) que não seja a própria máscara
        mask_like  = [o for o in outputs if o.ndim >= 3]
        score_like = [o for o in outputs if o.ndim <= 2 and o.size <= 16]
        if len(mask_like) == 1 and len(score_like) == 1:
            return mask_like[0], score_like[0]

        raise RuntimeError(
            "Não consegui identificar automaticamente os outputs 'masks' e 'scores' "
            "(iou_predictions) do decoder EdgeSAM. Outputs disponíveis: "
            + ", ".join(f"{n} shape={list(o.shape)}" for n, o in zip(out_names, outputs))
            + ". Configure VISION_EDGESAM_MASKS_OUTPUT_NAME e VISION_EDGESAM_SCORES_OUTPUT_NAME "
              "com os nomes corretos (inspecione o .onnx com o Netron, por exemplo)."
        )

    def predict_mask(self, image_embedding: dict, point_xy: tuple[int, int]) -> tuple[np.ndarray, float]:
        h, w = image_embedding["orig_size"]
        scale = self.input_size / max(h, w)
        px, py = point_xy[0] * scale, point_xy[1] * scale

        feed_candidates = {
            "image_embeddings": image_embedding["embedding"],
            "point_coords": np.array([[[px, py]]], dtype=np.float32),
            "point_labels": np.array([[1]], dtype=np.float32),
            "orig_im_size": np.array([h, w], dtype=np.float32),
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.array([0], dtype=np.float32),
        }
        expected = {i.name for i in self._dec.get_inputs()}
        feed = {k: v for k, v in feed_candidates.items() if k in expected}

        out_names = [o.name for o in self._dec.get_outputs()]
        outputs = self._dec.run(out_names, feed)

        if not self._outputs_logged:
            shapes = ", ".join(f"{n}={list(o.shape)}" for n, o in zip(out_names, outputs))
            log.info(f"EdgeSAM decoder outputs (1a chamada): {shapes}")
            self._outputs_logged = True

        masks, scores = self._pick_mask_and_score_outputs(out_names, outputs)

        scores_flat = scores.reshape(-1)
        masks_stack = masks.reshape((-1,) + masks.shape[-2:])
        best = int(np.argmax(scores_flat))
        if best >= masks_stack.shape[0]:
            # nº de scores não bate com o nº de máscaras (auto-detecção pegou o
            # tensor errado, ou o modelo tem uma convenção diferente) — não
            # estoura, cai pra máscara 0 e loga pra investigar
            log.warning(
                f"scores({scores_flat.shape}, argmax={best}) e masks({masks_stack.shape}) com "
                f"contagens incompatíveis — usando a máscara 0. Configure "
                f"VISION_EDGESAM_MASKS_OUTPUT_NAME/VISION_EDGESAM_SCORES_OUTPUT_NAME se persistir."
            )
            best = 0
        mask = masks_stack[best]
        mask_resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
        score = float(scores_flat[best]) if best < scores_flat.shape[0] else 1.0
        return mask_resized, score


# ── [5] Encoder de embeddings visuais (DINOv3) ─────────────────────────────

def _flatten_for_embedding(pil_image: Image.Image) -> Image.Image:
    """
    Prepara uma imagem (RGB ou RGBA) pra virar embedding.

    Se vier de um crop no modo "masked" (RGBA, fundo com alpha=0),
    `Image.convert("RGB")` sozinho NÃO remove o fundo — ele só descarta o
    canal alpha, e os pixels do fundo (que continuam lá, só "marcados" como
    transparentes) voltam a aparecer inteiros. Aqui a gente compõe de
    verdade sobre um fundo neutro antes de descartar o alpha, pra o
    embedder realmente "ver" só o objeto.
    """
    if pil_image.mode == "RGBA":
        bg = Image.new("RGB", pil_image.size, (128, 128, 128))
        bg.paste(pil_image, mask=pil_image.split()[3])
        return bg
    return pil_image.convert("RGB")


class DinoV3Embedder:
    def __init__(
        self, model_path: str, input_size: int = DINOV3_INPUT_SIZE,
        threads: int = CPU_THREADS, is_int8: bool = False,
    ):
        self.input_size = input_size
        self._session: Optional[ort.InferenceSession] = None
        self._available = False
        self._model_path_used = model_path
        self._is_int8 = is_int8
        if Path(model_path).exists():
            try:
                self._session = _make_ort_session(model_path, threads)
                self._in_name  = self._session.get_inputs()[0].name
                self._out_name = self._session.get_outputs()[0].name
                self._available = True
                log.info(f"DINOv3 embedder carregado: {model_path} (threads={threads})")
            except Exception as e:
                log.error(f"Falha ao carregar DINOv3 ({model_path}): {e}")
        else:
            log.warning(
                f"Modelo DINOv3 não encontrado em '{model_path}' — configure "
                f"VISION_DINOV3_FP32_MODEL_PATH / VISION_DINOV3_INT8_MODEL_PATH"
            )

    @property
    def available(self) -> bool:
        return self._available

    @property
    def model_path_used(self) -> str:
        return self._model_path_used

    @property
    def is_int8(self) -> bool:
        return self._is_int8

    def embed(self, pil_image: Image.Image) -> np.ndarray:
        if not self._available:
            raise RuntimeError("DINOv3 não carregado")
        img = _flatten_for_embedding(pil_image).resize((self.input_size, self.input_size), Image.BICUBIC)
        x = np.asarray(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        x = x.transpose(2, 0, 1)[None, ...].astype(np.float32)
        out = self._session.run([self._out_name], {self._in_name: x})[0]
        vec = out.reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec


# ── [4] Pós-processamento / verificação de consistência ────────────────────

def _mask_depth_stats(mask: np.ndarray, depth_map: np.ndarray) -> tuple[float, float, float]:
    vals = depth_map[mask]
    if vals.size == 0:
        return 0.0, 0.0, 0.0
    return float(vals.mean()), float(vals.std()), float(vals.max() - vals.min())


def _should_resplit(mask: np.ndarray, depth_map: np.ndarray) -> bool:
    """Máscara com profundidade muito heterogênea provavelmente engloba mais
    de um objeto (ex.: ponto caiu numa "sombra" entre dois objetos próximos)."""
    _, std, rng = _mask_depth_stats(mask, depth_map)
    return std > DEPTH_STD_RESPLIT_THRESH and rng > DEPTH_STD_RESPLIT_THRESH * 2


def _resplit_mask(
    mask: np.ndarray, depth_map: np.ndarray, sam: EdgeSAMSegmenter, image_embedding: dict,
) -> list[np.ndarray]:
    """Re-segmenta uma máscara heterogênea: agrupa os pixels da máscara em 2
    sub-populações de profundidade, pega o centróide de cada uma como novo
    prompt, e pede pro SAM uma máscara nova pra cada — restrita à máscara
    original, pra não vazar pra fora do que já era considerado 1 região."""
    ys, xs = np.where(mask)
    vals = depth_map[mask]
    if len(vals) < 20:
        return [mask]
    km = KMeans(n_clusters=2, n_init=4, random_state=0)
    sub_labels = km.fit_predict(vals.reshape(-1, 1))

    sub_masks = []
    for lbl in (0, 1):
        sel = sub_labels == lbl
        if sel.sum() < 10:
            continue
        cy, cx = int(ys[sel].mean()), int(xs[sel].mean())
        new_mask, score = sam.predict_mask(image_embedding, (cx, cy))
        new_mask = new_mask & mask
        if score >= SAM_SCORE_THRESH and new_mask.sum() > 0:
            sub_masks.append(new_mask)

    return sub_masks if len(sub_masks) >= 2 else [mask]


def _masks_adjacent(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    dil_a = cv2.dilate(mask_a.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return bool(np.any(dil_a & mask_b))


def _border_edge_strength(mask_a: np.ndarray, mask_b: np.ndarray, gray_image: np.ndarray) -> float:
    """Intensidade média do gradiente de cor (Sobel) na fronteira entre as
    duas máscaras — usado pra decidir se a borda é "real" (objetos
    diferentes) ou só um artefato de segmentação (mesmo objeto)."""
    dil_a = cv2.dilate(mask_a.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    border = dil_a & mask_b
    if not np.any(border):
        dil_b = cv2.dilate(mask_b.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        border = dil_b & mask_a
    if not np.any(border):
        return 0.0
    gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(mag[border].mean())


def postprocess_masks(
    masks: list[np.ndarray],
    depth_map: np.ndarray,
    gray_image: np.ndarray,
    sam: EdgeSAMSegmenter,
    image_embedding: dict,
) -> list[np.ndarray]:
    """
    Regras de consistência do pipeline:
      1. máscara com profundidade muito heterogênea → re-split
      2. duas máscaras adjacentes, profundidade parecida (diff < eps) e
         borda de cor FRACA → provavelmente é o mesmo objeto → funde
      3. borda de cor FORTE + profundidade igual → mantém separadas
         (comportamento default — não faz nada além de não fundir)
    """
    refined: list[np.ndarray] = []
    for m in masks:
        if _should_resplit(m, depth_map):
            refined.extend(_resplit_mask(m, depth_map, sam, image_embedding))
        else:
            refined.append(m)

    merged: list[np.ndarray] = []
    used = [False] * len(refined)
    for i in range(len(refined)):
        if used[i]:
            continue
        current = refined[i]
        used[i] = True
        for j in range(i + 1, len(refined)):
            if used[j] or not _masks_adjacent(current, refined[j]):
                continue
            mean_a, _, _ = _mask_depth_stats(current, depth_map)
            mean_b, _, _ = _mask_depth_stats(refined[j], depth_map)
            if abs(mean_a - mean_b) > DEPTH_MERGE_EPS:
                continue  # profundidades diferentes → não funde
            edge = _border_edge_strength(current, refined[j], gray_image)
            if edge < COLOR_EDGE_MERGE_THRESH:
                current = current | refined[j]
                used[j] = True
            # else: borda forte + profundidade igual → mantém separadas
        merged.append(current)

    return merged


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def dedup_masks(masks: list[np.ndarray], iou_thresh: float = MASK_IOU_NMS_THRESH) -> list[np.ndarray]:
    """NMS simples por IoU — pontos de camadas diferentes às vezes geram a
    mesma máscara (objeto grande cobrindo mais de uma "faixa" de profundidade)."""
    kept: list[np.ndarray] = []
    for m in sorted(masks, key=lambda x: int(x.sum()), reverse=True):
        if all(_mask_iou(m, k) < iou_thresh for k in kept):
            kept.append(m)
    return kept


def suppress_contained(
    masks: list[np.ndarray],
    containment_thresh: float = CONTAINMENT_SUPPRESS_THRESH,
    size_ratio_thresh: float = CONTAINMENT_SIZE_RATIO,
) -> list[np.ndarray]:
    """
    Remove máscaras "infladas" que engolem quase inteira uma máscara menor
    já detectada — sinal típico de um prompt que caiu numa região ampla
    (fundo, mesa, parte do cenário) em vez de num objeto específico. IoU
    sozinho não pega esse caso: uma máscara pequena totalmente dentro de
    uma bem maior pode ter IoU baixo mesmo sendo, na prática, redundante
    (o "objeto real" + um monte de fundo).

    Mantém a menor/mais específica quando a maior a contém quase inteira
    e é significativamente maior.
    """
    masks_sorted = sorted(masks, key=lambda m: int(m.sum()))  # menor primeiro
    dropped = [False] * len(masks_sorted)
    for i, small in enumerate(masks_sorted):
        if dropped[i]:
            continue
        area_small = int(small.sum())
        for j in range(i + 1, len(masks_sorted)):
            if dropped[j]:
                continue
            big = masks_sorted[j]
            area_big = int(big.sum())
            if area_big < area_small * size_ratio_thresh:
                continue
            inter = int(np.logical_and(small, big).sum())
            containment = inter / area_small if area_small > 0 else 0.0
            if containment >= containment_thresh:
                dropped[j] = True
    return [m for m, d in zip(masks_sorted, dropped) if not d]


def filter_by_area(
    masks: list[np.ndarray],
    image_area: int,
    min_ratio: float = MIN_MASK_AREA_RATIO,
    max_ratio: float = MAX_MASK_AREA_RATIO,
) -> list[np.ndarray]:
    out = []
    for m in masks:
        ratio = float(m.sum()) / float(image_area)
        if min_ratio <= ratio <= max_ratio:
            out.append(m)
    return out


def crop_object(
    image: np.ndarray, mask: np.ndarray, padding_ratio: float = CROP_PADDING_RATIO, mode: str = CROP_MODE,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    h, w = image.shape[:2]
    pad_y = int((y1 - y0) * padding_ratio)
    pad_x = int((x1 - x0) * padding_ratio)
    y0, y1 = max(0, y0 - pad_y), min(h, y1 + pad_y + 1)
    x0, x1 = max(0, x0 - pad_x), min(w, x1 + pad_x + 1)

    crop = image[y0:y1, x0:x1].copy()
    if mode == "masked":
        local_mask = mask[y0:y1, x0:x1]
        rgba = np.dstack([crop, (local_mask * 255).astype(np.uint8)])
        return Image.fromarray(rgba, mode="RGBA"), (x0, y0, x1, y1)
    return Image.fromarray(crop, mode="RGB"), (x0, y0, x1, y1)


# ── Cliente HTTP do Dicionário Visual (memory.py) ──────────────────────────

class MemoryDictClient:
    """
    Fala com os endpoints /visual-dict/* do memory.py. Este módulo é
    stateless em relação ao dicionário — toda leitura e escrita passa por
    aqui, igual ao onnx_client.py faz para os embeddings de texto do
    memory.py.
    """

    def __init__(self, base_url: str = MEMORY_API_URL, timeout: float = 20.0):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def read(self, embedding: np.ndarray, top_k: int = DICT_TOP_K, min_score: float = DICT_MIN_SCORE) -> dict:
        resp = await self._client.post(
            "/visual-dict/read",
            json={"embedding": embedding.tolist(), "top_k": top_k, "min_score": min_score},
        )
        resp.raise_for_status()
        return resp.json()

    async def write(
        self,
        concept_name: str,
        description: str,
        embedding: np.ndarray,
        source: str = "vision_pipeline",
        confidence: float = 1.0,
        link_to_memory: bool = True,
    ) -> dict:
        resp = await self._client.post(
            "/visual-dict/write",
            json={
                "concept_name": concept_name,
                "description": description,
                "embedding": embedding.tolist(),
                "source": source,
                "confidence": confidence,
                "link_to_memory": link_to_memory,
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def status(self) -> dict:
        resp = await self._client.get("/status")
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        await self._client.aclose()


# ── Modelos de request/response — API REST ─────────────────────────────────

class VisionProcessRequest(BaseModel):
    image_base64: str
    top_k:       int   = DICT_TOP_K
    min_score:   float = DICT_MIN_SCORE
    max_objects: int   = MAX_OBJECTS_PER_IMAGE

class ObjectCandidate(BaseModel):
    concept_id:   int
    concept_name: str
    description:  str
    score:        float

class DetectedObject(BaseModel):
    object_index: int
    bbox:         tuple[int, int, int, int]   # (x0, y0, x1, y1) na imagem original
    depth_mean:   float
    # Só retornamos a imagem do crop — nenhuma chamada a LLM acontece aqui.
    crop_base64:  str
    candidates:   list[ObjectCandidate]
    ambiguous:    bool
    # Preenchido só quando a consulta ao dicionário (memory.py) falha — não
    # confundir com "ambiguous=True" (que é uma resposta válida sem match).
    # Deixa o erro real visível no JSON em vez de só no log do servidor.
    dict_error:   Optional[str] = None

class VisionProcessResponse(BaseModel):
    objects:        list[DetectedObject]
    total_detected: int
    image_width:    int
    image_height:   int

class VisionRegisterRequest(BaseModel):
    image_base64: str
    concept_name: str
    description:  str
    source:       str   = "manual"
    confidence:   float = 1.0

class VisionRegisterResponse(BaseModel):
    stored:       bool
    reason:       str
    concept_id:   Optional[int] = None
    embedding_id: Optional[int] = None
    memory_id:    Optional[int] = None
    new_concept:  bool = False


# ── Estado global ────────────────────────────────────────────────────────────

@dataclass
class VisionAppState:
    depth:       DepthEstimator     = field(default=None)
    clusterer:   DepthClusterer     = field(default=None)
    sam:         EdgeSAMSegmenter   = field(default=None)
    embedder:    DinoV3Embedder     = field(default=None)
    dict_client: MemoryDictClient   = field(default=None)

state = VisionAppState()


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Iniciando AVA Vision API...")

    state.depth     = DepthEstimator(DEPTH_MODEL_PATH, DEPTH_INPUT_SIZE, CPU_THREADS)
    state.clusterer = DepthClusterer()
    state.sam       = EdgeSAMSegmenter(EDGESAM_ENCODER_PATH, EDGESAM_DECODER_PATH, EDGESAM_INPUT_SIZE, CPU_THREADS)

    dinov3_path, dinov3_is_int8 = _ensure_quantized(DINOV3_FP32_MODEL_PATH, DINOV3_INT8_MODEL_PATH)
    state.embedder = DinoV3Embedder(dinov3_path, DINOV3_INPUT_SIZE, CPU_THREADS, is_int8=dinov3_is_int8)

    state.dict_client = MemoryDictClient(MEMORY_API_URL)
    try:
        status = await state.dict_client.status()
        log.info(f"memory.py acessível — visual_dict={status.get('visual_dict')}")
    except Exception as e:
        log.warning(
            f"memory.py não acessível em {MEMORY_API_URL} ({e}) — "
            f"/vision/process vai falhar ao consultar o dicionário até ele subir"
        )

    log.info(
        f"Pronto — depth={state.depth.available} sam={state.sam.available} "
        f"embedder={state.embedder.available} (modelo={state.embedder.model_path_used}) "
        f"threads_cpu={CPU_THREADS}"
    )
    yield

    await state.dict_client.close()
    log.info("AVA Vision API encerrada")


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(title="AVA Vision API", lifespan=lifespan)


def _decode_image(image_base64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(image_base64)
        return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Não foi possível decodificar a imagem: {e}")


@app.post("/vision/process", response_model=VisionProcessResponse)
async def vision_process(req: VisionProcessRequest):
    """
    Roda o pipeline inteiro numa imagem e retorna, para cada objeto
    detectado: o crop (base64) e os candidatos de significado recuperados
    do dicionário visual (memory.py). Não chama nenhum LLM — a decisão do
    que fazer com objetos ambíguos (`ambiguous=True`, sem candidato
    confiável) é do orquestrador externo.
    """
    missing = [
        name for name, comp in (
            ("depth", state.depth), ("edgesam", state.sam), ("dinov3", state.embedder),
        ) if comp is None or not comp.available
    ]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Modelos não carregados: {', '.join(missing)}. Configure os caminhos via variáveis de ambiente.",
        )

    image = _decode_image(req.image_base64)
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # [1] depth estimation
    depth_map = state.depth.estimate(image)

    # clusterização por camadas de profundidade + amostragem de pontos
    layers = state.clusterer.cluster(depth_map)
    prompt_points: list[tuple[int, int]] = []
    for layer in layers:
        prompt_points.extend(state.clusterer.sample_points(layer))

    if not prompt_points:
        return VisionProcessResponse(objects=[], total_detected=0, image_width=w, image_height=h)

    # [3] proposer / segmentação class-agnostic
    image_embedding = state.sam.encode_image(image)
    candidate_masks: list[np.ndarray] = []
    for pt in prompt_points:
        mask, score = state.sam.predict_mask(image_embedding, pt)
        if score >= SAM_SCORE_THRESH and mask.sum() > 0:
            candidate_masks.append(mask)

    candidate_masks = dedup_masks(filter_by_area(candidate_masks, h * w))

    # [4] pós-processamento / verificação de consistência
    final_masks = postprocess_masks(candidate_masks, depth_map, gray, state.sam, image_embedding)
    final_masks = dedup_masks(filter_by_area(final_masks, h * w))
    # remove máscaras grandes que só engolem uma menor já detectada
    # (objeto + fundo, em vez do objeto em si)
    final_masks = suppress_contained(final_masks)
    final_masks = final_masks[: min(req.max_objects, MAX_OBJECTS_PER_IMAGE)]

    objects: list[DetectedObject] = []
    for i, mask in enumerate(final_masks):
        crop_img, bbox = crop_object(image, mask)
        buf = io.BytesIO()
        crop_img.save(buf, format="PNG")
        crop_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        # [5] embedding visual do crop
        crop_embedding = state.embedder.embed(crop_img)

        # [6]/[7] short-list via kNN — delegado ao memory.py
        candidates: list[ObjectCandidate] = []
        ambiguous = True
        dict_error: Optional[str] = None
        try:
            result = await state.dict_client.read(crop_embedding, top_k=req.top_k, min_score=req.min_score)
            candidates = [ObjectCandidate(**c) for c in result.get("results", [])]
            ambiguous = bool(result.get("ambiguous", True))
        except httpx.HTTPStatusError as e:
            dict_error = f"{e.response.status_code}: {e.response.text}"
            log.exception(f"Falha ao consultar o dicionário visual (memory.py) — objeto {i}")
        except Exception as e:
            dict_error = str(e)
            log.exception(f"Falha ao consultar o dicionário visual (memory.py) — objeto {i}")

        depth_mean, _, _ = _mask_depth_stats(mask, depth_map)

        objects.append(DetectedObject(
            object_index=i,
            bbox=bbox,
            depth_mean=depth_mean,
            crop_base64=crop_b64,
            candidates=candidates,
            ambiguous=ambiguous,
            dict_error=dict_error,
        ))

    log.info(f"Processada imagem {w}x{h} — {len(objects)} objeto(s) detectado(s)")
    return VisionProcessResponse(objects=objects, total_detected=len(objects), image_width=w, image_height=h)


@app.post("/vision/register", response_model=VisionRegisterResponse)
async def vision_register(req: VisionRegisterRequest):
    """
    Registra um novo exemplo no dicionário visual: embeda a imagem enviada
    (idealmente já um crop de um objeto) e grava no memory.py — cria um
    conceito novo se `concept_name` ainda não existir, ou só adiciona mais
    um embedding de exemplo se já existir (reconhecimento mais robusto a
    ângulo/luz diferentes).
    """
    if state.embedder is None or not state.embedder.available:
        raise HTTPException(status_code=503, detail="Modelo DINOv3 não carregado")

    image = _decode_image(req.image_base64)
    vec = state.embedder.embed(Image.fromarray(image))

    try:
        result = await state.dict_client.write(
            req.concept_name, req.description, vec,
            source=req.source, confidence=req.confidence,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Falha ao gravar no dicionário visual (memory.py): {e}")

    return VisionRegisterResponse(**result)


@app.get("/vision/status")
async def vision_status():
    memory_reachable = True
    memory_status: dict = {}
    try:
        memory_status = await state.dict_client.status()
    except Exception:
        memory_reachable = False

    return {
        "models": {
            "depth_estimation": {
                "available": state.depth.available if state.depth else False,
                "model_path": DEPTH_MODEL_PATH,
                "input_size": DEPTH_INPUT_SIZE,
            },
            "segmentation_edgesam": {
                "available": state.sam.available if state.sam else False,
                "encoder_path": EDGESAM_ENCODER_PATH,
                "decoder_path": EDGESAM_DECODER_PATH,
                "input_size": EDGESAM_INPUT_SIZE,
                "score_threshold": SAM_SCORE_THRESH,
            },
            "embeddings_dinov3": {
                "available": state.embedder.available if state.embedder else False,
                "model_path_used": state.embedder.model_path_used if state.embedder else None,
                "input_size": DINOV3_INPUT_SIZE,
                "int8_quantized": bool(state.embedder and state.embedder.is_int8),
            },
        },
        "memory_api": {
            "url": MEMORY_API_URL,
            "reachable": memory_reachable,
            "visual_dict": memory_status.get("visual_dict"),
        },
        "pipeline_config": {
            "cpu_threads":       CPU_THREADS,
            "depth_layers":      N_DEPTH_LAYERS,
            "points_per_layer":  POINTS_PER_LAYER,
            "cluster_method":    CLUSTER_METHOD,
            "sam_score_thresh":  SAM_SCORE_THRESH,
            "min_mask_area_ratio": MIN_MASK_AREA_RATIO,
            "max_mask_area_ratio": MAX_MASK_AREA_RATIO,
            "mask_iou_nms_thresh": MASK_IOU_NMS_THRESH,
            "depth_std_resplit_thresh": DEPTH_STD_RESPLIT_THRESH,
            "depth_merge_eps":   DEPTH_MERGE_EPS,
            "color_edge_merge_thresh": COLOR_EDGE_MERGE_THRESH,
            "crop_mode":         CROP_MODE,
            "crop_padding_ratio": CROP_PADDING_RATIO,
            "dict_top_k":        DICT_TOP_K,
            "dict_min_score":    DICT_MIN_SCORE,
        },
    }


# ── Entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("vision:app", host="0.0.0.0", port=4002, log_level="info")