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
# Margem de score: só descarta a máscara MAIOR por contenção se o score dela
# for pior que o da menor por mais do que essa margem. Evita descartar um
# objeto "limpo" (ex.: o livro inteiro) só porque ele contém sub-partes
# (ícones, olhos, rodas...) que também viraram máscaras candidatas.
CONTAINMENT_SCORE_MARGIN    = _env_float("VISION_CONTAINMENT_SCORE_MARGIN", 0.05)

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

# ── NEW: Reconhecimento facial (detecção + EdgeFace) ───────────────────────
# Detector: YuNet, já embutido no OpenCV (cv2.FaceDetectorYN) — não precisa
# de pós-processamento manual de anchors como SCRFD. Baixe o .onnx oficial
# (~230KB) em: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
FACE_DETECTOR_MODEL_PATH = _env_str("VISION_FACE_DETECTOR_MODEL_PATH", "./Models/face_detection_yunet_2023mar.onnx")
FACE_DET_SCORE_THRESH    = _env_float("VISION_FACE_DET_SCORE_THRESH", 0.80)
FACE_DET_NMS_THRESH      = _env_float("VISION_FACE_DET_NMS_THRESH", 0.30)
FACE_DET_TOP_K           = _env_int("VISION_FACE_DET_TOP_K", 10)

# EdgeFace — embedder de rosto. Ajuste input_size/dim conforme seu export.
EDGEFACE_MODEL_PATH = _env_str("VISION_EDGEFACE_MODEL_PATH", "./Models/edgeface_xs_gamma_06.onnx")
EDGEFACE_INPUT_SIZE = _env_int("VISION_EDGEFACE_INPUT_SIZE", 112)   # padrão ArcFace/EdgeFace
EDGEFACE_EMBED_DIM  = _env_int("VISION_EDGEFACE_EMBED_DIM", 512)    # deve bater com FD_EMBED_DIM no memory.py

# Dicionário de rostos — defaults espelham os do memory.py (FD_*)
FACE_TOP_K     = _env_int("VISION_FACE_TOP_K", 3)
FACE_MIN_SCORE = _env_float("VISION_FACE_MIN_SCORE", 0.42)
MAX_FACE_REGISTER_IMAGES = _env_int("VISION_MAX_FACE_REGISTER_IMAGES", 10)

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


# ── NEW: Reconhecimento facial — detecção (YuNet) + alinhamento + EdgeFace ─

class FaceDetector:
    """
    Wrapper fino sobre cv2.FaceDetectorYN (YuNet) — vem embutido no OpenCV,
    então não precisamos decodificar anchors/strides manualmente como seria
    necessário com SCRFD "cru". Retorna bbox + 5 landmarks (olhos, nariz,
    cantos da boca) por rosto — os landmarks são o que permite alinhar o
    rosto antes de mandar pro EdgeFace (alinhamento melhora bastante a
    acurácia de embedders estilo ArcFace/EdgeFace).
    """

    def __init__(
        self,
        model_path: str,
        score_thresh: float = FACE_DET_SCORE_THRESH,
        nms_thresh: float = FACE_DET_NMS_THRESH,
        top_k: int = FACE_DET_TOP_K,
    ):
        self._detector = None
        self._available = False
        if Path(model_path).exists():
            try:
                # input_size inicial é um placeholder — setInputSize() é
                # chamado com o tamanho real da imagem a cada detect()
                self._detector = cv2.FaceDetectorYN.create(
                    model_path, "", (320, 320), score_thresh, nms_thresh, top_k,
                )
                self._available = True
                log.info(f"FaceDetector (YuNet) carregado: {model_path}")
            except Exception as e:
                log.error(f"Falha ao carregar FaceDetector ({model_path}): {e}")
        else:
            log.warning(
                f"Modelo de detecção facial não encontrado em '{model_path}' — "
                f"configure VISION_FACE_DETECTOR_MODEL_PATH (baixe o YuNet.onnx "
                f"do opencv_zoo)"
            )

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, image: np.ndarray) -> list[dict]:
        """image: RGB numpy array. Retorna lista de rostos ordenados por
        score desc, cada um com bbox (x, y, w, h), landmarks (5, 2) e score."""
        if not self._available:
            raise RuntimeError("FaceDetector não carregado")
        h, w = image.shape[:2]
        self._detector.setInputSize((w, h))
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, faces = self._detector.detect(bgr)
        results: list[dict] = []
        if faces is not None:
            for f in faces:
                bbox = tuple(f[0:4].astype(int).tolist())
                landmarks = f[4:14].reshape(5, 2).astype(np.float32)
                score = float(f[14])
                results.append({"bbox": bbox, "landmarks": landmarks, "score": score})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results


# Template padrão de 5 pontos (ArcFace/insightface) para alinhamento em
# imagem de saída 112x112 — olho esq, olho dir, nariz, canto boca esq, canto
# boca dir. Mesma convenção usada pela maioria dos embedders estilo EdgeFace.
_FACE_ALIGN_TEMPLATE_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(image: np.ndarray, landmarks: np.ndarray, output_size: int = EDGEFACE_INPUT_SIZE) -> np.ndarray:
    """Alinha o rosto via transformação de similaridade estimada entre os 5
    landmarks detectados e o template padrão — corrige rotação/escala antes
    do embedding, o que reduz bastante a variação entre poses do mesmo
    rosto (fundamental pra threshold de similaridade ficar estável)."""
    template = _FACE_ALIGN_TEMPLATE_112 * (output_size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.LMEDS)
    if M is None:
        # fallback: crop simples pela bbox dos landmarks, sem correção de pose
        x0, y0 = landmarks.min(axis=0).astype(int)
        x1, y1 = landmarks.max(axis=0).astype(int)
        crop = image[max(0, y0):y1, max(0, x0):x1]
        return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    return cv2.warpAffine(image, M, (output_size, output_size), borderValue=(0, 0, 0))


class EdgeFaceEmbedder:
    """
    Embedder de rosto (EdgeFace, ONNX). Mesmo padrão de código do
    DinoV3Embedder — espera um rosto já alinhado (112x112, RGB) como
    entrada. Normalização segue o padrão comum a modelos estilo
    ArcFace/insightface: (px/255 - 0.5) / 0.5, ou seja, valores em [-1, 1].

    IMPORTANTE: confira a normalização exata usada no export do SEU .onnx
    (varia entre repositórios de EdgeFace) — se o embedding não bater com o
    esperado, esse é o primeiro lugar a revisar.
    """

    def __init__(self, model_path: str, input_size: int = EDGEFACE_INPUT_SIZE, threads: int = CPU_THREADS):
        self.input_size = input_size
        self._session: Optional[ort.InferenceSession] = None
        self._available = False
        if Path(model_path).exists():
            try:
                self._session = _make_ort_session(model_path, threads)
                self._in_name  = self._session.get_inputs()[0].name
                self._out_name = self._session.get_outputs()[0].name
                self._available = True
                log.info(f"EdgeFace embedder carregado: {model_path} (threads={threads})")
            except Exception as e:
                log.error(f"Falha ao carregar EdgeFace ({model_path}): {e}")
        else:
            log.warning(
                f"Modelo EdgeFace não encontrado em '{model_path}' — configure "
                f"VISION_EDGEFACE_MODEL_PATH"
            )

    @property
    def available(self) -> bool:
        return self._available

    def embed(self, aligned_face_rgb: np.ndarray) -> np.ndarray:
        if not self._available:
            raise RuntimeError("EdgeFace não carregado")
        img = cv2.resize(aligned_face_rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        x = img.astype(np.float32)
        x = (x / 255.0 - 0.5) / 0.5
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
    mask: np.ndarray, score: float, depth_map: np.ndarray, sam: EdgeSAMSegmenter, image_embedding: dict,
) -> list[tuple[np.ndarray, float]]:
    """Re-segmenta uma máscara heterogênea: agrupa os pixels da máscara em 2
    sub-populações de profundidade, pega o centróide de cada uma como novo
    prompt, e pede pro SAM uma máscara nova pra cada — restrita à máscara
    original, pra não vazar pra fora do que já era considerado 1 região."""
    ys, xs = np.where(mask)
    vals = depth_map[mask]
    if len(vals) < 20:
        return [(mask, score)]
    km = KMeans(n_clusters=2, n_init=4, random_state=0)
    sub_labels = km.fit_predict(vals.reshape(-1, 1))

    sub_masks: list[tuple[np.ndarray, float]] = []
    for lbl in (0, 1):
        sel = sub_labels == lbl
        if sel.sum() < 10:
            continue
        cy, cx = int(ys[sel].mean()), int(xs[sel].mean())
        new_mask, new_score = sam.predict_mask(image_embedding, (cx, cy))
        new_mask = new_mask & mask
        if new_score >= SAM_SCORE_THRESH and new_mask.sum() > 0:
            sub_masks.append((new_mask, new_score))

    return sub_masks if len(sub_masks) >= 2 else [(mask, score)]


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
    masks_scores: list[tuple[np.ndarray, float]],
    depth_map: np.ndarray,
    gray_image: np.ndarray,
    sam: EdgeSAMSegmenter,
    image_embedding: dict,
) -> list[tuple[np.ndarray, float]]:
    """
    Regras de consistência do pipeline:
      1. máscara com profundidade muito heterogênea → re-split
      2. duas máscaras adjacentes, profundidade parecida (diff < eps) e
         borda de cor FRACA → provavelmente é o mesmo objeto → funde
      3. borda de cor FORTE + profundidade igual → mantém separadas
         (comportamento default — não faz nada além de não fundir)

    Recebe e retorna pares (mask, score) — o score do EdgeSAM acompanha a
    máscara em todas as etapas pra poder ser usado depois em
    suppress_contained (ver CONTAINMENT_SCORE_MARGIN).
    """
    refined: list[tuple[np.ndarray, float]] = []
    for m, s in masks_scores:
        if _should_resplit(m, depth_map):
            refined.extend(_resplit_mask(m, s, depth_map, sam, image_embedding))
        else:
            refined.append((m, s))

    merged: list[tuple[np.ndarray, float]] = []
    used = [False] * len(refined)
    for i in range(len(refined)):
        if used[i]:
            continue
        current, current_score = refined[i]
        used[i] = True
        for j in range(i + 1, len(refined)):
            if used[j]:
                continue
            other, other_score = refined[j]
            if not _masks_adjacent(current, other):
                continue
            mean_a, _, _ = _mask_depth_stats(current, depth_map)
            mean_b, _, _ = _mask_depth_stats(other, depth_map)
            if abs(mean_a - mean_b) > DEPTH_MERGE_EPS:
                continue  # profundidades diferentes → não funde
            edge = _border_edge_strength(current, other, gray_image)
            if edge < COLOR_EDGE_MERGE_THRESH:
                current = current | other
                # máscara fundida herda o melhor score das duas partes
                current_score = max(current_score, other_score)
                used[j] = True
            # else: borda forte + profundidade igual → mantém separadas
        merged.append((current, current_score))

    return merged


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def dedup_masks(
    masks_scores: list[tuple[np.ndarray, float]], iou_thresh: float = MASK_IOU_NMS_THRESH
) -> list[tuple[np.ndarray, float]]:
    """NMS simples por IoU — pontos de camadas diferentes às vezes geram a
    mesma máscara (objeto grande cobrindo mais de uma "faixa" de profundidade)."""
    kept: list[tuple[np.ndarray, float]] = []
    for m, s in sorted(masks_scores, key=lambda x: int(x[0].sum()), reverse=True):
        if all(_mask_iou(m, k) < iou_thresh for k, _ in kept):
            kept.append((m, s))
    return kept


def suppress_contained(
    masks_scores: list[tuple[np.ndarray, float]],
    containment_thresh: float = CONTAINMENT_SUPPRESS_THRESH,
    size_ratio_thresh: float = CONTAINMENT_SIZE_RATIO,
    score_margin: float = CONTAINMENT_SCORE_MARGIN,
) -> list[tuple[np.ndarray, float]]:
    """
    Resolve máscaras redundantes por contenção — mas usando o score do
    EdgeSAM pra decidir QUAL das duas (a maior ou a menor) é a "espúria",
    em vez de sempre assumir que é a maior.

    Duas situações distintas geram o mesmo padrão geométrico (máscara
    pequena quase inteira dentro de uma bem maior):

      1. "objeto + fundo": o prompt caiu numa região ampla (mesa, parede,
         parte do cenário) e a máscara grande é, na prática, ruído — a
         máscara pequena é o objeto real. Aqui a máscara GRANDE costuma ter
         score pior que a pequena.
      2. "objeto + sub-partes": a máscara grande é um objeto legítimo (ex.:
         o livro inteiro) e a pequena é uma sub-região dele (ex.: um ícone
         na capa) que também virou candidata. Aqui a máscara grande tem
         score igual ou melhor que a pequena — descartá-la (como o código
         antigo sempre fazia) jogava fora o objeto certo e deixava só o
         fragmento.

    Regra: só descarta a máscara MAIOR se o score dela for pior que o da
    menor por mais que `score_margin`. Caso contrário, quem é descartada é
    a máscara MENOR (sub-parte redundante), e a maior (objeto real) fica.
    """
    # menor primeiro, pra comparar cada pequena contra as maiores que a contêm
    ordered = sorted(masks_scores, key=lambda ms: int(ms[0].sum()))
    dropped = [False] * len(ordered)
    for i, (small, small_score) in enumerate(ordered):
        if dropped[i]:
            continue
        area_small = int(small.sum())
        for j in range(i + 1, len(ordered)):
            if dropped[j]:
                continue
            big, big_score = ordered[j]
            area_big = int(big.sum())
            if area_big < area_small * size_ratio_thresh:
                continue
            inter = int(np.logical_and(small, big).sum())
            containment = inter / area_small if area_small > 0 else 0.0
            if containment < containment_thresh:
                continue

            if big_score < small_score - score_margin:
                # máscara grande é a "ruim" (objeto + fundo) → descarta a grande
                dropped[j] = True
            else:
                # grande é o objeto real, pequena é sub-parte → descarta a pequena
                dropped[i] = True
                break  # "small" já foi descartada, não faz sentido comparar mais

    return [ms for ms, d in zip(ordered, dropped) if not d]


def filter_by_area(
    masks_scores: list[tuple[np.ndarray, float]],
    image_area: int,
    min_ratio: float = MIN_MASK_AREA_RATIO,
    max_ratio: float = MAX_MASK_AREA_RATIO,
) -> list[tuple[np.ndarray, float]]:
    out = []
    for m, s in masks_scores:
        ratio = float(m.sum()) / float(image_area)
        if min_ratio <= ratio <= max_ratio:
            out.append((m, s))
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


# ── NEW: Cliente HTTP do Dicionário de Rostos (memory.py) ──────────────────

class FaceDictClient:
    """Mesmo papel do MemoryDictClient, mas para /face-dict/* — reconhecimento
    facial mora num dicionário separado no memory.py (banco/índice próprios,
    dimensão de embedding diferente)."""

    def __init__(self, base_url: str = MEMORY_API_URL, timeout: float = 20.0):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def read(self, embedding: np.ndarray, top_k: int = FACE_TOP_K, min_score: float = FACE_MIN_SCORE) -> dict:
        resp = await self._client.post(
            "/face-dict/read",
            json={"embedding": embedding.tolist(), "top_k": top_k, "min_score": min_score},
        )
        resp.raise_for_status()
        return resp.json()

    async def write(
        self,
        person_name: str,
        embedding: np.ndarray,
        description: str = "",
        source: str = "vision_pipeline",
        confidence: float = 1.0,
    ) -> dict:
        resp = await self._client.post(
            "/face-dict/write",
            json={
                "person_name": person_name,
                "embedding": embedding.tolist(),
                "description": description,
                "source": source,
                "confidence": confidence,
            },
        )
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


# ── NEW: Modelos de request/response — reconhecimento facial ──────────────

class FaceRegisterRequest(BaseModel):
    images_base64: list[str]         # uma ou mais fotos da mesma pessoa, ângulos/luz diferentes
    person_name:   str                # ex.: "eu" — ou seu nome, se preferir
    description:   str   = ""         # quem é essa pessoa (relação, contexto etc.)
    source:        str   = "manual"
    confidence:    float = 1.0

class FaceRegisterImageResult(BaseModel):
    index:          int              # posição da imagem em images_base64
    stored:         bool
    reason:         str
    embedding_id:   Optional[int] = None
    faces_detected: int = 0

class FaceRegisterResponse(BaseModel):
    stored:          bool             # True se PELO MENOS uma imagem foi gravada com sucesso
    person_id:       Optional[int] = None
    new_person:      bool = False
    images_received: int
    images_stored:   int
    results:         list[FaceRegisterImageResult]

class FaceIdentifyRequest(BaseModel):
    image_base64: str
    top_k:        int   = FACE_TOP_K
    min_score:    float = FACE_MIN_SCORE

class FaceMatch(BaseModel):
    person_id:   int
    person_name: str
    description: str
    score:       float

class FaceIdentifyResponse(BaseModel):
    faces_detected: int
    is_known:       bool           # True → o rosto principal do frame bateu com alguém cadastrado
    matches:        list[FaceMatch]
    ambiguous:      bool


# ── Estado global ────────────────────────────────────────────────────────────

@dataclass
class VisionAppState:
    depth:            DepthEstimator     = field(default=None)
    clusterer:        DepthClusterer     = field(default=None)
    sam:              EdgeSAMSegmenter   = field(default=None)
    embedder:         DinoV3Embedder     = field(default=None)
    dict_client:      MemoryDictClient   = field(default=None)
    # ── NEW: Reconhecimento facial ──
    face_detector:    FaceDetector       = field(default=None)
    face_embedder:    EdgeFaceEmbedder   = field(default=None)
    face_dict_client: FaceDictClient     = field(default=None)

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

    # ── NEW: Reconhecimento facial ──
    state.face_detector    = FaceDetector(FACE_DETECTOR_MODEL_PATH, FACE_DET_SCORE_THRESH, FACE_DET_NMS_THRESH, FACE_DET_TOP_K)
    state.face_embedder    = EdgeFaceEmbedder(EDGEFACE_MODEL_PATH, EDGEFACE_INPUT_SIZE, CPU_THREADS)
    state.face_dict_client = FaceDictClient(MEMORY_API_URL)

    log.info(
        f"Pronto — depth={state.depth.available} sam={state.sam.available} "
        f"embedder={state.embedder.available} (modelo={state.embedder.model_path_used}) "
        f"face_detector={state.face_detector.available} face_embedder={state.face_embedder.available} "
        f"threads_cpu={CPU_THREADS}"
    )
    yield

    await state.dict_client.close()
    await state.face_dict_client.close()
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
    candidate_masks: list[tuple[np.ndarray, float]] = []
    for pt in prompt_points:
        mask, score = state.sam.predict_mask(image_embedding, pt)
        if score >= SAM_SCORE_THRESH and mask.sum() > 0:
            candidate_masks.append((mask, score))

    candidate_masks = dedup_masks(filter_by_area(candidate_masks, h * w))

    # [4] pós-processamento / verificação de consistência
    final_masks_scores = postprocess_masks(candidate_masks, depth_map, gray, state.sam, image_embedding)
    final_masks_scores = dedup_masks(filter_by_area(final_masks_scores, h * w))
    # resolve máscaras redundantes por contenção, usando o score do EdgeSAM
    # pra decidir se a "espúria" é a máscara grande (objeto + fundo) ou a
    # pequena (sub-parte de um objeto legítimo) — ver suppress_contained()
    final_masks_scores = suppress_contained(final_masks_scores)
    final_masks_scores = final_masks_scores[: min(req.max_objects, MAX_OBJECTS_PER_IMAGE)]
    final_masks = [m for m, _ in final_masks_scores]

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


# ── NEW: Reconhecimento facial ──────────────────────────────────────────────

@app.post("/vision/register-face", response_model=FaceRegisterResponse)
async def vision_register_face(req: FaceRegisterRequest):
    """
    Cadastra um rosto a partir de uma ou mais fotos da mesma pessoa (ângulos
    e luz diferentes ajudam bastante). Pra cada imagem: detecta o rosto de
    maior confiança, alinha, extrai o embedding (EdgeFace) e grava mais um
    exemplo no dicionário de rostos (memory.py) — todas viram exemplos do
    mesmo `person_name`. Uma foto sem rosto detectado não derruba as
    demais: o resultado por imagem vem em `results`.
    """
    missing = [
        name for name, comp in (
            ("face_detector", state.face_detector), ("edgeface", state.face_embedder),
        ) if comp is None or not comp.available
    ]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Modelos não carregados: {', '.join(missing)}. Configure os caminhos via variáveis de ambiente.",
        )

    person_name = req.person_name.strip()
    if not person_name:
        return FaceRegisterResponse(
            stored=False, images_received=len(req.images_base64), images_stored=0, results=[],
        )

    if not req.images_base64:
        return FaceRegisterResponse(stored=False, images_received=0, images_stored=0, results=[])

    if len(req.images_base64) > MAX_FACE_REGISTER_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Máximo de {MAX_FACE_REGISTER_IMAGES} imagens por chamada (recebido {len(req.images_base64)})",
        )

    description = req.description.strip()
    person_id: Optional[int] = None
    new_person = False
    results: list[FaceRegisterImageResult] = []

    for i, image_b64 in enumerate(req.images_base64):
        try:
            image = _decode_image(image_b64)
        except HTTPException as e:
            results.append(FaceRegisterImageResult(index=i, stored=False, reason=f"decode_error: {e.detail}"))
            continue

        faces = state.face_detector.detect(image)
        if not faces:
            results.append(FaceRegisterImageResult(index=i, stored=False, reason="no_face_detected", faces_detected=0))
            continue

        # usa o rosto de maior score — assume-se uma pessoa por foto de cadastro
        best = faces[0]
        aligned = align_face(image, best["landmarks"])
        vec = state.face_embedder.embed(aligned)

        try:
            # só manda description nas escritas ainda sem person_id resolvido
            # (evita reenviar update de descrição redundante N vezes seguidas
            # — embora seja inofensivo, é desnecessário)
            write_result = await state.face_dict_client.write(
                person_name, vec, description=(description if person_id is None else ""),
                source=req.source, confidence=req.confidence,
            )
        except Exception as e:
            results.append(FaceRegisterImageResult(index=i, stored=False, reason=f"memory_error: {e}", faces_detected=len(faces)))
            continue

        person_id = write_result.get("person_id", person_id)
        new_person = new_person or write_result.get("new_person", False)
        results.append(FaceRegisterImageResult(
            index=i, stored=write_result.get("stored", False), reason=write_result.get("reason", "ok"),
            embedding_id=write_result.get("embedding_id"), faces_detected=len(faces),
        ))

    images_stored = sum(1 for r in results if r.stored)
    log.info(
        f"Face-register '{person_name}': {images_stored}/{len(req.images_base64)} "
        f"imagem(ns) gravada(s) (person_id={person_id})"
    )

    return FaceRegisterResponse(
        stored=images_stored > 0, person_id=person_id, new_person=new_person,
        images_received=len(req.images_base64), images_stored=images_stored, results=results,
    )


@app.post("/vision/identify-face", response_model=FaceIdentifyResponse)
async def vision_identify_face(req: FaceIdentifyRequest):
    """
    Identifica o rosto principal (maior score de detecção) de uma imagem
    contra o dicionário de rostos cadastrado. `is_known=True` só quando há
    um candidato com score >= min_score e sem ambiguidade — pra um caso de
    "só me identificar" (1 pessoa cadastrada), na prática isso já funciona
    como um booleano "é você ou não é".
    """
    missing = [
        name for name, comp in (
            ("face_detector", state.face_detector), ("edgeface", state.face_embedder),
        ) if comp is None or not comp.available
    ]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Modelos não carregados: {', '.join(missing)}. Configure os caminhos via variáveis de ambiente.",
        )

    image = _decode_image(req.image_base64)
    faces = state.face_detector.detect(image)
    if not faces:
        return FaceIdentifyResponse(faces_detected=0, is_known=False, matches=[], ambiguous=True)

    best = faces[0]
    aligned = align_face(image, best["landmarks"])
    vec = state.face_embedder.embed(aligned)

    try:
        result = await state.face_dict_client.read(vec, top_k=req.top_k, min_score=req.min_score)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Falha ao consultar o dicionário de rostos (memory.py): {e}")

    matches = [FaceMatch(**m) for m in result.get("results", [])]
    ambiguous = bool(result.get("ambiguous", True))
    is_known = len(matches) > 0 and not ambiguous

    return FaceIdentifyResponse(
        faces_detected=len(faces), is_known=is_known, matches=matches, ambiguous=ambiguous,
    )


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
            # ── NEW: Reconhecimento facial ──
            "face_detector_yunet": {
                "available": state.face_detector.available if state.face_detector else False,
                "model_path": FACE_DETECTOR_MODEL_PATH,
                "score_threshold": FACE_DET_SCORE_THRESH,
            },
            "embeddings_edgeface": {
                "available": state.face_embedder.available if state.face_embedder else False,
                "model_path": EDGEFACE_MODEL_PATH,
                "input_size": EDGEFACE_INPUT_SIZE,
                "embed_dim": EDGEFACE_EMBED_DIM,
            },
        },
        "memory_api": {
            "url": MEMORY_API_URL,
            "reachable": memory_reachable,
            "visual_dict": memory_status.get("visual_dict"),
            "face_dict": memory_status.get("face_dict"),
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
            "containment_suppress_thresh": CONTAINMENT_SUPPRESS_THRESH,
            "containment_size_ratio": CONTAINMENT_SIZE_RATIO,
            "containment_score_margin": CONTAINMENT_SCORE_MARGIN,
            "depth_std_resplit_thresh": DEPTH_STD_RESPLIT_THRESH,
            "depth_merge_eps":   DEPTH_MERGE_EPS,
            "color_edge_merge_thresh": COLOR_EDGE_MERGE_THRESH,
            "crop_mode":         CROP_MODE,
            "crop_padding_ratio": CROP_PADDING_RATIO,
            "dict_top_k":        DICT_TOP_K,
            "dict_min_score":    DICT_MIN_SCORE,
            "face_top_k":        FACE_TOP_K,
            "face_min_score":    FACE_MIN_SCORE,
        },
    }


# ── Entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("vision:app", host="0.0.0.0", port=4002, log_level="info")