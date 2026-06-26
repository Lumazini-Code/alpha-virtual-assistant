"""
AVA Orchestrator — Unified Execution & Routing Engine
======================================================
Receives user requests, classifies intent via internal ONNX/Heuristic Router,
and either routes directly to a single module or generates a multi-step 
execution plan via the CoT module, executing it respecting dependency DAGs.

Integrates ALL AVA microservices:
  - Router          (INTERNAL)    — ONNX intent classification + heuristic fallback
  - CoT Generator   (port 3000)  — plan generation + semantic cache
  - Memory          (port 3001)  — long-term, short-term, knowledge
  - Search          (port 3002)  — web search + cross-encoder rerank
  - Local Scraping  (port 3003)  — local file search, read & indexing
  - TTS             (port 3004)  — text-to-speech (Supertonic)
  - LLM Chat        (port 4003)  — conversational inference (llama-server)
  - Vision / VQA    (port 4002)  — image understanding (Qwen3VL)
  - Deep Search     (port 4005)  — KG-RAG with automatic web research
"""

from __future__ import annotations

import json 
import asyncio
import logging
import os
import re
import shlex
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tokenizers import Tokenizer
from fastapi.responses import StreamingResponse

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

MODELS_DIR =  "./Modules/Models"
DEBERTA_DIR  = os.path.join(MODELS_DIR, "DebertaV2ForSequenceClassification")
MINILM_DIR   = os.path.join(MODELS_DIR, "ms-marco-MiniLM-L-6-v2")

COT_URL              = "http://localhost:3000"
MEMORY_URL           = "http://localhost:3001"
SEARCH_URL           = "http://localhost:3002"
LOCAL_SCRAPING_URL   = "http://localhost:3003"   # ← NEW: local file scraping
TTS_URL              = "http://localhost:3004"
LLM_URL              = "http://localhost:4003"
VISION_URL           = "http://localhost:4002"
DEEP_SEARCH_URL      = "http://localhost:4005"

HEALTH_PATHS: dict[str, str] = {
    "cot": "/status", "memory": "/status", "search": "/status",
    "local_scraping": "/status", "tts": "/status",          # ← NEW
    "llm": "/health", "vision": "/health", "deep_search": "/health",
}

EXECUTOR_TIMEOUTS: dict[str, float] = {
    "llm": 9999999.0, "memory": 9999999.0, "search": 9999999.0, "deep_search": 9999999.0,
    "vision": 9999999.0, "tts": 9999999.0, "local_scraping": 9999999.0,  # ← NEW
}

EXECUTOR_MAX_RETRIES: dict[str, int] = {
    "llm": 2, "memory": 3, "search": 2, "deep_search": 1, "vision": 1,
    "tts": 2,"local_scraping": 2,  # ← NEW
}

RETRY_BACKOFF_S   = 0.15
COT_TIMEOUT_S     = 9999999.0
MAX_CONTEXT_CHARS = 1500
DEFAULT_TOP_K     = 5
DEFAULT_MIN_SCORE = 0.30

# Router Configuration
CONFIDENCE_MIN = 0.45        # Se acima disso e não ambíguo, aceita
AMBIGUITY_THRESHOLD = 0.08   # Se top2 rotas diferem por menos, considerar ambíguo
COT_THRESHOLD = 0.30  
ONNX_PROVIDERS = os.environ.get("ONNX_PROVIDERS", "AzureExecutionProvider, CPUExecutionProvider").split(",")

# Direct Route Mapping
# Direct Route Mapping (ATUALIZADO)
ROUTE_TO_EXECUTOR: dict[str, str] = {
    "llm": "llm", "search": "search", "memory_read": "memory", "memory_write": "memory",
    "vision": "vision", "deep_search": "deep_search", "tts": "tts",
    "local_scraping": "local_scraping",  
}


THINK_DEPTH_INSTRUCTIONS: dict[int, str] = {
    0: (
        "This is a trivial interaction — a greeting, acknowledgment, or simple social exchange. "
        "Respond naturally and briefly. No reasoning needed."
    ),
    1: (
        "This is a simple factual question with a direct answer. "
        "Retrieve the fact and respond concisely. No chain of thought needed."
    ),
    2: (
        "This requires minimal reasoning — a basic comparison, definition, or short explanation. "
        "Answer directly and clearly in a few sentences."
    ),
    3: (
        "This requires light reasoning. Think step by step briefly before answering, "
        "but keep your response focused and avoid unnecessary elaboration."
    ),
    4: (
        "This requires moderate reasoning. Break the problem into clear parts, "
        "think through each one, then synthesize a coherent answer."
    ),
    5: (
        "This requires balanced analytical thinking. Identify the key variables, "
        "consider different angles, weigh trade-offs, and build your answer progressively. "
        "Show your reasoning where helpful."
    ),
    6: (
        "This is a complex question. Think carefully before answering: "
        "identify assumptions, explore multiple perspectives, anticipate edge cases, "
        "and structure your response logically."
    ),
    7: (
        "This requires deep reasoning. Use a thorough chain of thought: "
        "decompose the problem, reason through each component independently, "
        "identify dependencies between parts, and synthesize a well-argued response."
    ),
    8: (
        "This is a highly complex task. Think extensively before responding. "
        "Map out the full problem space, consider competing hypotheses, "
        "validate intermediate conclusions, and build your final answer step by step. "
        "Precision and completeness matter here."
    ),
    9: (
        "This requires expert-level reasoning. Engage in rigorous multi-step thinking: "
        "define the problem formally, reason from first principles, explore edge cases, "
        "challenge your own intermediate conclusions, and produce a thorough, well-structured response. "
        "Do not skip reasoning steps."
    ),
    10: (
        "This is a maximally complex task requiring deep, exhaustive reasoning. "
        "Think as carefully and thoroughly as possible before responding. "
        "Decompose every sub-problem, reason from first principles at each step, "
        "validate every intermediate conclusion, consider all relevant edge cases and counter-arguments, "
        "and synthesize a complete, precise, and well-justified response. "
        "Take as much reasoning space as needed — correctness and depth are the priority."
    ),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ava.orchestrator")


# ══════════════════════════════════════════════════════════════════════════════
# Data Models
# ══════════════════════════════════════════════════════════════════════════════

class ExecuteRequest(BaseModel):
    input:       str
    session_id:  Optional[str]  = None
    voice:       str            = "M1"
    lang:        str            = "pt"
    tts:         bool           = True
    use_cache:   bool           = True
    image_path:  Optional[str]  = None
    search_pdfs: bool           = False
    strategy:    Literal["parallel", "sequential", "fail_fast"] = "parallel"
    stream:      bool           = True   # ← NOVO: ativa resposta em streaming

class StepResult(BaseModel):
    step: int; executor: str; action: str; success: bool
    result: Optional[Any] = None; error: Optional[str] = None
    retries: int = 0; latency_ms: float = 0.0

class ExecuteResponse(BaseModel):
    execution_id: str; input: str; session_id: str; final_response: str
    steps: list[StepResult]; plan_from_cache: bool; total_latency_ms: float
    errors: list[str]
    route: str = "cot"; route_confidence: float = 0.0
    route_method: str = "cot"; routed_directly: bool = False

class ClassifyRequest(BaseModel):
    text: str = Field(..., description="User query to classify")
    image_path: Optional[str] = None; lang: str = "pt"

class ClassifyResponse(BaseModel):
    route: str; confidence: float; method: str
    all_scores: dict[str, float]; needs_cot: bool; latency_ms: float = 0.0

class DeepSearchRequest(BaseModel):
    text: str = Field(..., description="Question or research objective")

class VisionRequest(BaseModel):
    img_path: str = Field(..., description="Local path to the image file")
    prompt: str = Field(default="Descreva a imagem detalhadamente.")

class MemoryReadRequest(BaseModel):
    query: str; top_k: int = DEFAULT_TOP_K; min_score: float = DEFAULT_MIN_SCORE; session_id: Optional[str] = None

class MemoryWriteRequest(BaseModel):
    text: str; source: str = "chat"; confidence: float = 1.0

# ── NEW: Local Scraping Models ──────────────────────────────────────────────

class LocalScrapingRequest(BaseModel):
    """Request to search and read a local file on the user's machine."""
    query: str = Field(..., description="Nome ou descrição do arquivo a buscar")
    search_path: Optional[str] = Field(None, description="Caminho base para a busca (padrão: diretório da Alpha)")
    force_reindex: bool = Field(False, description="Forçar reindexação mesmo se o arquivo não mudou")
    session_id: Optional[str] = None

class LocalScrapingChooseRequest(BaseModel):
    """Choose a specific file when multiple matches are found."""
    query: str = Field(..., description="Query original da busca")
    file_path: str = Field(..., description="Caminho completo do arquivo escolhido")
    force_reindex: bool = Field(False, description="Forçar reindexação")
    session_id: Optional[str] = None

# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL ROUTER — ONNX + Heuristics
# ══════════════════════════════════════════════════════════════════════════════

class RouteLabel(IntEnum):
    COT = 0; LLM = 1; MEMORY_READ = 3; MEMORY_WRITE = 4
    VISION = 5; DEEP_SEARCH = 6;  # ← NEW

LABEL_NAMES: list[str] = [rl.name.lower() for rl in RouteLabel]
NUM_LABELS = len(RouteLabel)   # agora 12

def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def _find_onnx_file(directory: str, preferred_name: str = "model.onnx") -> Optional[str]:
    d = Path(directory)
    if not d.exists(): return None
    for name in [preferred_name, "model_quantized.onnx"]:
        c = d / name
        if c.is_file(): return str(c)
    for f in sorted(d.glob("*.onnx")):
        if f.is_file(): return str(f)
    return None

def _load_tokenizer(model_dir: str, max_length: int = 128) -> Optional[Tokenizer]:
    d = Path(model_dir)
    if not d.exists(): return None
    tokenizer_path = None
    if (d / "tokenizer.json").is_file(): tokenizer_path = str(d / "tokenizer.json")
    else:
        for f in sorted(d.glob("*.json")):
            if "tokenizer" in f.name.lower(): tokenizer_path = str(f); break
    if not tokenizer_path: return None
    try:
        tok = Tokenizer.from_file(tokenizer_path)
        tok.enable_truncation(max_length=max_length)
        tok.enable_padding(length=max_length)
        return tok
    except Exception: return None

def _encode_for_onnx(tokenizer: Tokenizer, text: str) -> dict[str, np.ndarray]:
    encoding = tokenizer.encode(text)
    result = {"input_ids": np.array([encoding.ids], dtype=np.int64), "attention_mask": np.array([encoding.attention_mask], dtype=np.int64)}
    if encoding.type_ids: result["token_type_ids"] = np.array([encoding.type_ids], dtype=np.int64)
    return result

def _sanitize_step_value(val):
    """Extrai o valor primitivo caso venha como dict do CoT."""
    if isinstance(val, dict):
        # Tenta pegar chaves comuns como id, name, step, etc.
        return val.get("id") or val.get("name") or val.get("step") or val.get("executor") or str(val)
    return val

def _sanitize_steps(raw_steps: list) -> list:
    """Garante que os steps do CoT tenham tipos primitivos (int, str, list)."""
    sanitized = []
    for s in raw_steps:
        if not isinstance(s, dict):
            continue
        
        step_num = _sanitize_step_value(s.get("step"))
        try: step_num = int(step_num)
        except: step_num = 0
            
        executor = _sanitize_step_value(s.get("executor"))
        if not isinstance(executor, str):
            executor = str(executor).lower()
            
        action = s.get("action") or s.get("task") or ""
        
        deps = s.get("depends_on") or []
        clean_deps = []
        for d in deps:
            d_val = _sanitize_step_value(d)
            try: clean_deps.append(int(d_val))
            except: pass
            
        sanitized.append({
            "step": step_num,
            "executor": executor,
            "action": str(action),
            "depends_on": clean_deps
        })
    return sanitized

class OnnxMiniLMRouter:
    ROUTE_PROTOTYPES: dict[str, list[str]] = {
        "cot": ["search and then translate", "research and summarize", "complex multi step task"],
        "llm": ["hello how are you", "explain what is ai", "tell me a joke", "write a poem"],
        "memory_read": ["what do you remember", "recall my conversation", "what do you know about me"],
        "memory_write": ["remember my favorite color", "save this info", "grave na memória"],
        "vision": ["describe this image", "analyze photo", "what does picture show"],
        "deep_search": ["deep research about", "learn about", "pesquisa profunda"],
    }
    def __init__(self): self.ready = False; self.tokenizer = None; self.session = None; self.input_names = []; self.output_names = []; self.provider = ""; self._prototype_embeddings = None
    def load(self, model_dir: str) -> bool:
        try: import onnxruntime as ort
        except ImportError: return False
        self.tokenizer = _load_tokenizer(model_dir)
        if not self.tokenizer: return False
        onnx_path = _find_onnx_file(model_dir)
        if not onnx_path: return False
        try:
            opts = ort.SessionOptions(); opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL; opts.intra_op_num_threads = 4
            self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=[p for p in ONNX_PROVIDERS if p in ort.get_available_providers()] or ["CPUExecutionProvider"])
            self.input_names = [i.name for i in self.session.get_inputs()]; self.output_names = [o.name for o in self.session.get_outputs()]
            self.provider = self.session.get_providers()[0]; self.ready = True; self._precompute_prototypes(); return True
        except Exception: return False
    def _encode(self, text: str) -> np.ndarray:
        enc = _encode_for_onnx(self.tokenizer, text); feed = {k: v for k, v in enc.items() if k in self.input_names}
        out = self.session.run(self.output_names, feed)[0]
        emb = out.squeeze() if len(out.shape) != 3 else (out * enc["attention_mask"][..., np.newaxis].astype(np.float32)).sum(axis=1).squeeze() / np.clip(enc["attention_mask"].sum(axis=1), 1e-9, None).squeeze()[..., np.newaxis]
        norm = np.linalg.norm(emb); return (emb / norm).astype(np.float32) if norm > 1e-9 else emb.astype(np.float32)
    def _precompute_prototypes(self):
        self._prototype_embeddings = {}
        for route, descs in self.ROUTE_PROTOTYPES.items():
            embs = [self._encode(d) for d in descs]; avg = np.mean(embs, axis=0); norm = np.linalg.norm(avg)
            self._prototype_embeddings[route] = (avg / norm if norm > 1e-9 else avg).astype(np.float32)
    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        if not self.ready or not self._prototype_embeddings: raise RuntimeError("Not ready")
        q_emb = self._encode(text); scores = {r: round(float(np.dot(q_emb, p)), 4) for r, p in self._prototype_embeddings.items()}
        sims = np.array([scores.get(n, 0.0) for n in LABEL_NAMES]); probs = _softmax(sims * 5.0)
        scores = {LABEL_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_LABELS)}
        idx = int(probs.argmax()); return LABEL_NAMES[idx], round(float(probs[idx]), 4), scores


# Global Router State
minilm_router  = OnnxMiniLMRouter()
def _internal_classify(text: str, image_path: Optional[str] = None) -> dict:
    """
    Classifica texto utilizando APENAS o modelo ONNX (MiniLM).
    CoT só é usado quando há real ambiguidade ou confiança muito baixa.
    """
    all_scores = {n: 0.0 for n in LABEL_NAMES}
    method = "default"
    route = "llm"  # Fallback seguro
    confidence = 0.0

    # ─────────────────────────────────────────
    # PASSO 1: Tentar MiniLM (ONNX)
    # ─────────────────────────────────────────
    if minilm_router.ready:
        try:
            # predict retorna (route, confidence, all_scores)
            route, confidence, all_scores = minilm_router.predict(text)
            method = "onnx_minilm"
        except Exception as e:
            log.warning(f"Erro ao classificar com ONNX: {e}")
            method = "onnx_error"
            # Em caso de erro no ONNX, joga pra CoT ou LLM com baixa confiança
            route = "llm"
            confidence = 0.0
    else:
        method = "router_offline"
        route = "llm"
        confidence = 0.0

    # ─────────────────────────────────────────
    # PASSO 2: Análise de confiança
    # ─────────────────────────────────────────
    # Calcula a segunda maior confiança para detectar ambiguidade
    sorted_scores = sorted(all_scores.values(), reverse=True)
    second_best = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    confidence_gap = confidence - second_best

    # ─────────────────────────────────────────
    # PASSO 3: Decisão sobre CoT
    # ─────────────────────────────────────────
    needs_cot = False
    reason_cot = None

    # Caso 1: Ambíguo — 2+ rotas com scores próximos e sem confiança alta
    if confidence_gap < AMBIGUITY_THRESHOLD and confidence < 0.70:
        needs_cot = True
        reason_cot = "ambiguous_top2"

    # Caso 2: Confiança MUITO baixa (modelo muito incerto)
    elif confidence < COT_THRESHOLD:
        needs_cot = True
        reason_cot = "low_confidence"

    # ─────────────────────────────────────────
    # PASSO 4: Ajuste Final de Segurança
    # ─────────────────────────────────────────
    if needs_cot:
        # Se decidiu por CoT, força a rota
        route = "cot"
        confidence = 0.0
        method = f"{method}→cot"
    else:
        # 🛡️ SEGURANÇA CRÍTICA: Se NÃO precisa de CoT, mas a rota calculada 
        # foi "cot" (porque o modelo deu um leve pico de ruído nela), 
        # impedimos que caia no pipeline CoT forçando o fallback para a 
        # melhor rota DIRETA disponível (geralmente "llm").
        if route not in ROUTE_TO_EXECUTOR:
            valid_routes = {k: v for k, v in all_scores.items() if k in ROUTE_TO_EXECUTOR}
            if valid_routes:
                route = max(valid_routes, key=valid_routes.get)
                confidence = valid_routes[route]
            else:
                route = "llm"
                confidence = 0.5

    return {
        "route": route,
        "confidence": confidence,
        "method": method,
        "all_scores": all_scores,
        "needs_cot": needs_cot,
        "reason_cot": reason_cot,
        "confidence_gap": confidence_gap,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Global State & Lifespan
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AppState:
    cot_client: httpx.AsyncClient = field(default=None); memory_client: httpx.AsyncClient = field(default=None)
    search_client: httpx.AsyncClient = field(default=None); tts_client: httpx.AsyncClient = field(default=None)
    llm_client: httpx.AsyncClient = field(default=None); vision_client: httpx.AsyncClient = field(default=None)
    deep_search_client: httpx.AsyncClient = field(default=None)
    local_scraping_client: httpx.AsyncClient = field(default=None)  # ← NEW

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Iniciando AVA Unified Orchestrator & Router...")
    
    # Load internal ONNX Routers
    m_loaded = minilm_router.load(MINILM_DIR)
    log.info(f"Router Init →  MiniLM: {'✓' if m_loaded else '✗'} | Heuristic: ✓")

    # Probe external services
    service_urls = {
        "cot": COT_URL, "memory": MEMORY_URL, "search": SEARCH_URL,
        "local_scraping": LOCAL_SCRAPING_URL, "tts": TTS_URL,          # ← NEW
        "llm": LLM_URL, "vision": VISION_URL, "deep_search": DEEP_SEARCH_URL,
    }
    async with httpx.AsyncClient(timeout=5.0) as probe:
        for name, url in service_urls.items():
            try:
                r = await probe.get(f"{url}{HEALTH_PATHS.get(name, '/status')}")
                log.info(f"  ✓ {name:16s} OK" if r.status_code == 200 else f"  ⚠ {name:16s} {r.status_code}")
            except httpx.ConnectError: log.warning(f"  ✗ {name:16s} OFFLINE")

    def _make_client(base_url: str, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout), limits=httpx.Limits(max_keepalive_connections=4, max_connections=8))

    state.cot_client = _make_client(COT_URL, COT_TIMEOUT_S); state.memory_client = _make_client(MEMORY_URL, EXECUTOR_TIMEOUTS["memory"])
    state.search_client = _make_client(SEARCH_URL, EXECUTOR_TIMEOUTS["search"]); state.tts_client = _make_client(TTS_URL, EXECUTOR_TIMEOUTS["tts"])
    state.llm_client = _make_client(LLM_URL, EXECUTOR_TIMEOUTS["llm"]); state.vision_client = _make_client(VISION_URL, EXECUTOR_TIMEOUTS["vision"])
    state.deep_search_client = _make_client(DEEP_SEARCH_URL, EXECUTOR_TIMEOUTS["deep_search"])
    state.local_scraping_client = _make_client(LOCAL_SCRAPING_URL, EXECUTOR_TIMEOUTS["local_scraping"])  # ← NEW
    
    log.info("Orchestrator pronto — todos os clientes HTTP inicializados")
    yield
    for c in (state.cot_client, state.memory_client, state.search_client, state.tts_client,
              state.llm_client, state.vision_client, state.deep_search_client,
              state.local_scraping_client):                             # ← NEW
        await c.aclose()
    log.info("AVA Orchestrator encerrado")


# ══════════════════════════════════════════════════════════════════════════════
# Executor Adapters & Logic
# ══════════════════════════════════════════════════════════════════════════════

_WRITE_VERBS = frozenset({"gravar", "salvar", "registrar", "armazenar", "lembrar", "memorizar", "store", "save", "write", "record"})
_OPEN_PREFIXES = ("open ", "abrir ", "launch ", "start ", "iniciar ", "executar ")


async def _stream_llm(
    message: str,
    voice: str = "M1",
    lang: str = "pt",
    tts: bool = False,
    session_id: str = "default",
    max_turns: int = 10,
) -> str:
    """
    Call LLM /chat/stream and accumulate the full response.
    When tts=True, the LLM server fires TTS chunks in real-time,
    so the user starts hearing audio in seconds instead of waiting
    for the full generation to complete.
    """
    full_response = ""
    try:
        async with state.llm_client.stream(
            "POST",
            "/chat/stream",
            json={
                "message":    message,
                "voice":      voice,
                "lang":       lang,
                "tts":        tts,
                "session_id": session_id,
                "max_turns":  max_turns,
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                try:
                    chunk = json.loads(payload)
                    if "delta" in chunk:
                        full_response += chunk["delta"]
                    elif "done" in chunk:
                        break
                    elif "error" in chunk:
                        log.warning(f"LLM stream error: {chunk['error']}")
                        break
                except json.JSONDecodeError:
                    continue
    except httpx.StreamError as e:
        log.warning(f"LLM stream interrupted after {len(full_response)} chars: {e}")
    except Exception as e:
        if not full_response:
            raise
        log.warning(f"LLM stream failed after {len(full_response)} chars: {e}")

    return full_response


async def _adapt_llm(action, context, req, stream_tts=False, step_queue=None, step_num=None) -> str:
    ctx = _format_context_for_llm(action, context)
    msg = f"{action}\n\n{ctx}" if ctx else action
    
    # Se houver fila (CoT pipeline), faz streaming via fila em tempo real
    if step_queue and step_num is not None:
        full_resp = ""
        try:
            async for event_type, data in _stream_llm_chunks(
                msg,
                voice=req.voice,
                lang=req.lang,
                tts=stream_tts,
                session_id=req.session_id or "default",
            ):
                if event_type == "reasoning":
                    await step_queue.put({
                        "type": "step_reasoning", 
                        "step": step_num, 
                        "data": data
                    })
                elif event_type == "delta":
                    full_resp += data
                    await step_queue.put({
                        "type": "step_delta", 
                        "step": step_num, 
                        "data": data
                    })
            return full_resp
        except Exception as e:
            await step_queue.put({
                "type": "step_error", 
                "step": step_num, 
                "data": str(e)
            })
            raise

    # Caminho antigo (sem fila, caso seja chamado fora do pipeline)
    return await _stream_llm(
        msg,
        voice=req.voice,
        lang=req.lang,
        tts=stream_tts,
        session_id=req.session_id or "default",
        max_turns=10,
    )

async def _adapt_memory(action, context, req) -> list[dict]:
    fw = action.lower().split()[0] if action.split() else ""
    if fw in _WRITE_VERBS:
        txt = _resolve_action_text(action, context) or action; r = await state.memory_client.post("/write", json={"text": txt, "source": "orchestrator", "confidence": 1.0}); r.raise_for_status(); return [r.json()]
    else:
        r = await state.memory_client.post("/read", json={"query": action, "top_k": DEFAULT_TOP_K, "min_score": DEFAULT_MIN_SCORE, "session_id": req.session_id, "strategy": "auto"}); r.raise_for_status(); return r.json().get("results", [])

async def _adapt_search(action, context, req) -> list[dict]:
    r = await state.search_client.post("/search", json={"query": action, "max_results": DEFAULT_TOP_K, "use_cache": True, "search_pdfs": req.search_pdfs}); r.raise_for_status(); return r.json().get("results", [])

async def _adapt_deep_search(action, context, req) -> str:
    res = _resolve_action_text(action, context) or action; r = await state.deep_search_client.post("/query", json={"text": res}); r.raise_for_status(); d = r.json(); return d.get("answer") or str(d)

async def _adapt_vision(action, context, req) -> str:
    if not req.image_path: return f"[vision: nenhuma imagem fornecida — ação: {action}]"
    pr = _resolve_action_text(action, context) or action; r = await state.vision_client.post("/describe", json={"img_path": req.image_path, "prompt": pr}); r.raise_for_status(); d = r.json(); return d.get("result") or str(d)

async def _adapt_tts(action, context, req) -> str:
    txt = _resolve_action_text(action, context) or action; r = await state.tts_client.post("/speak", json={"text": txt[:2000], "voice": req.voice, "lang": req.lang}); r.raise_for_status(); return "tts_ok"

async def _adapt_stt(action, context, req) -> str: return action


async def _adapt_local_scraping(action, context, req) -> dict:
    """
    Adapter for the local-scraping microservice.
    
    Flow:
      1. Orchestrator forwards the user query to local-scraping /scrape
      2. local-scraping uses a model to generate a file search command
      3. A REST API client on the user's machine executes the command
         in the Alpha execution directory
      4. Returns file content + path
      5. If multiple files with the same name are found, returns a
         list for the user to choose
      6. If the file was already indexed, compares hashes:
         - Same hash  → use cached index
         - Different  → reindex
      7. Indexed info is saved to the memory folder
    """
    query = _resolve_action_text(action, context) or action
    
    r = await state.local_scraping_client.post(
        "/scrape",
        json={
            "query":         query,
            "search_path":   None,   # let the service use the Alpha path
            "force_reindex": False,
            "session_id":    req.session_id,
        },
    )
    r.raise_for_status()
    data = r.json()
    
    # ── If multiple files matched, return the choice list to the caller ──
    # The caller (orchestrator /execute or direct endpoint) will present
    # the choices and the user picks via /local-scraping/choose.
    if data.get("multiple_matches"):
        return {
            "status":           "multiple_matches",
            "matches":          data["matches"],
            "message":          data.get("message", "Múltiplos arquivos encontrados. Escolha qual deseja ler."),
            "requires_choice":  True,
        }
    
    # ── Single file result ──
    # If the file was just indexed or reindexed, also persist to memory
    file_content = data.get("content", "")
    file_path    = data.get("file_path", "")
    was_reindexed = data.get("was_reindexed", False)
    hash_match    = data.get("hash_match", True)
    
    # Persist indexed file info to AVA memory (long-term)
    if file_content and file_path:
        try:
            summary = file_content[:500] if len(file_content) > 500 else file_content
            await state.memory_client.post("/indexed-file/write", json={
                "file_path":   file_path,
                "file_name":   Path(file_path).name,
                "extension":   Path(file_path).suffix.lower(),
                "content":     file_content,           # conteúdo COMPLETO
                "file_hash":   data.get("file_hash"),  # hash do arquivo no disco
                "size":        len(file_content),
                "modified":    data.get("modified", ""),
                "source":      "local_scraping",
                "confidence":  1.0 if hash_match else 0.9,
                "force_reindex": False,
            })
            log.info(f"Local scraping: arquivo '{file_path}' salvo na memória (reindexed={was_reindexed}, hash_match={hash_match})")
        except Exception as e:
            log.warning(f"Falha ao salvar arquivo indexado na memória: {e}")
    
    return {
        "status":        "success",
        "content":       file_content,
        "file_path":     file_path,
        "was_reindexed": was_reindexed,
        "hash_match":    hash_match,
        "requires_choice": False,
    }


EXECUTOR_ADAPTERS = {
    "llm": _adapt_llm, "memory": _adapt_memory, "search": _adapt_search,
    "deep_search": _adapt_deep_search, "vision": _adapt_vision, "tts": _adapt_tts,
    "stt": _adapt_stt, "local_scraping": _adapt_local_scraping,  # ← NEW
}

def _resolve_action_text(action, context):
    return re.sub(r"result of step (\d+)", lambda m: _result_to_text(context.get(f"step_{m.group(1)}")) if context.get(f"step_{m.group(1)}") else m.group(0), action, flags=re.IGNORECASE)

def _result_to_text(res):
    if not res: return ""
    if isinstance(res, str): return res[:MAX_CONTEXT_CHARS]
    # ── Handle local_scraping result dicts ──
    if isinstance(res, dict):
        if res.get("status") == "multiple_matches":
            matches = res.get("matches", [])
            lines = [f"  [{i+1}] {m.get('file_path','?')} ({m.get('size','?')})" for i, m in enumerate(matches[:10])]
            return f"Múltiplos arquivos encontrados:\n" + "\n".join(lines)
        if res.get("status") == "success" and res.get("content"):
            return res["content"][:MAX_CONTEXT_CHARS]
        return str(res.get("text") or res.get("content") or res.get("response") or res)[:MAX_CONTEXT_CHARS]
    if isinstance(res, list): return "\n".join([f"- {(i.get('text') or i.get('content') or str(i))[:300]}" for i in res[:6]])[:MAX_CONTEXT_CHARS]
    return str(res)[:MAX_CONTEXT_CHARS]

def _format_context_for_llm(action, context):
    refs = set(int(m) for m in re.findall(r"result of step (\d+)", action, re.IGNORECASE))
    keys = {f"step_{n}" for n in refs} & set(context.keys()) if refs else set(context.keys())
    if not keys: return ""
    return "\n\n".join([f"[Step {k.split('_')[1]} result]\n{_result_to_text(context[k])}" for k in sorted(keys)])[:MAX_CONTEXT_CHARS]

async def _run_step_with_retry(step_num, action, executor, context, req, stream_tts=False, step_queue=None) -> StepResult:
    # Sanitiza executor caso venha como dict
    if isinstance(step_num, dict):
        step_num = step_num.get("id") or step_num.get("step") or step_num.get("num") or 0
    try:
        step_num = int(step_num)
    except (TypeError, ValueError):
        step_num = 0

    # ✅ Sanitizar executor (já existe)
    if isinstance(executor, dict):
        executor = executor.get("name") or executor.get("executor") or "llm"
    executor = str(executor).lower()

    adapter = EXECUTOR_ADAPTERS.get(executor)
    max_retries = EXECUTOR_MAX_RETRIES.get(executor, 1)
    
    if not adapter:
        return StepResult(step=step_num, executor=executor, action=action, success=False, error=f"Unknown executor: {executor}")
        
    t0 = time.perf_counter()
    retries = 0
    last_err = ""
    res_act = _resolve_action_text(action, context)
    
    while retries <= max_retries:
        try:
            if executor == "llm":
                res = await asyncio.wait_for(
                    adapter(res_act, context, req, stream_tts=(stream_tts or req.tts), step_queue=step_queue, step_num=step_num),
                    timeout=EXECUTOR_TIMEOUTS.get(executor, 30.0),
                )
            else:
                res = await asyncio.wait_for(
                    adapter(res_act, context, req),
                    timeout=EXECUTOR_TIMEOUTS.get(executor, 30.0),
                )
                
            lat = round((time.perf_counter() - t0) * 1000, 2)
            return StepResult(step=step_num, executor=executor, action=action, success=True,
                              result=res, retries=retries, latency_ms=lat)
                              
        except (httpx.ConnectError, httpx.TimeoutException, asyncio.TimeoutError) as e:
            last_err = f"{type(e).__name__}: {e}"
            retries += 1
            await asyncio.sleep(RETRY_BACKOFF_S * retries)
            continue
        except httpx.HTTPStatusError as e:
            last_err = f"HTTP {e.response.status_code}"
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            break
            
    lat = round((time.perf_counter() - t0) * 1000, 2)
    return StepResult(step=step_num, executor=executor, action=action, success=False,
                      error=last_err, retries=retries, latency_ms=lat)
    
    

async def _execute_plan(steps, req, strategy, step_queue: Optional[asyncio.Queue] = None) -> tuple[list[StepResult], dict[str, Any], bool]:
    step_map = {s["step"]: s for s in steps}
    pending = set(step_map.keys())
    
    dependents = set()
    for s in steps:
        for d in (s.get("depends_on") or []):
            dependents.add(d)
            
    output_steps = {s["step"] for s in steps if s["step"] not in dependents}
    # Faz um dump completo da lista de steps para vermos o que o CoT enviou
    log.info(f"ESTRUTURA DOS STEPS RECEBIDOS: {json.dumps(steps, indent=2, default=str, ensure_ascii=False)}")

    results = {}
    context = {}
    tts_fired = False

    while pending:
        ready = [
            sn for sn in sorted(pending)
            if all(d in results and results[d].success for d in (step_map[sn].get("depends_on") or []))
            and not any(d in results and not results[d].success for d in (step_map[sn].get("depends_on") or []))
        ]
        if not ready:
            for sn in list(pending):
                results[sn] = StepResult(step=sn, executor=step_map[sn]["executor"],
                                         action=step_map[sn]["action"], success=False,
                                         error="Deadlock/Deps failed")
                pending.discard(sn)
            break
        if strategy == "fail_fast" and any(not r.success for r in results.values()):
            for sn in list(pending):
                results[sn] = StepResult(step=sn, executor=step_map[sn]["executor"],
                                         action=step_map[sn]["action"], success=False,
                                         error="Abortado (fail_fast)")
                pending.discard(sn)
            break
        if strategy == "sequential":
            ready = [ready[0]]

        # ── NOTIFICAÇÃO DE INÍCIO (Tempo Real) ──
        if step_queue:
            for sn in ready:
                await step_queue.put({
                    "type": "step_start",
                    "step": sn,
                    "executor": step_map[sn]["executor"],
                    "action": step_map[sn]["action"][:50]
                })

        batch = await asyncio.gather(*[
            _run_step_with_retry(
                sn, step_map[sn]["action"], step_map[sn]["executor"], context, req,
                stream_tts=(sn in output_steps and step_map[sn]["executor"] == "llm" and req.tts),
                step_queue=step_queue,
            )
            for sn in ready
        ])
        
        for sr in batch:
            results[sr.step] = sr
            pending.discard(sr.step)
            context[f"step_{sr.step}"] = sr.result if sr.success else None
            if sr.success and sr.step in output_steps and sr.executor == "llm" and req.tts:
                tts_fired = True
            
            # ── NOTIFICAÇÃO DE CONCLUSÃO (Tempo Real) ──
            if step_queue:
                await step_queue.put({"type": "step_done", "data": sr})

    if step_queue is not None:
        await step_queue.put(None)  # sentinela
    return [results[sn] for sn in sorted(results.keys())], context, tts_fired
    

async def _fire_tts(text, req):
    try: await state.tts_client.post("/speak", json={"text": text[:2000], "voice": req.voice, "lang": req.lang})
    except: pass

async def _save_turn(u, a, sid):
    try: await state.memory_client.post("/write_st", json={"session_id": sid, "turns": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]})
    except: pass

async def _save_lt(text, src="chat"):
    try: await state.memory_client.post("/write", json={"text": text[:500], "source": src, "confidence": 0.8})
    except: pass
    
async def _stream_llm_chunks(
    message: str,
    voice: str = "M1",
    lang: str = "pt",
    tts: bool = False,
    session_id: str = "default",
    max_turns: int = 10,
    stream_reasoning: bool = True,  # ← NOVO PARÂMETRO
):
    """
    Gerador que yielda tuplas (event_type, data):
      - ("reasoning", texto) → thinking do modelo (se stream_reasoning=True)
      - ("delta", texto)     → resposta final
    """
    try:
        async with state.llm_client.stream(
            "POST",
            "/chat/stream",
            json={
                "message": message,
                "voice": voice,
                "lang": lang,
                "tts": tts,
                "session_id": session_id,
                "max_turns": max_turns,
                "stream_reasoning": stream_reasoning,  # ← ENVIAR PARA O LLM
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                try:
                    chunk = json.loads(payload)
                    
                    # Só processa reasoning se a flag estiver True E o LLM enviou
                    if "reasoning" in chunk and stream_reasoning:
                        yield ("reasoning", chunk["reasoning"])
                    elif "delta" in chunk:
                        yield ("delta", chunk["delta"])
                    elif "done" in chunk:
                        break
                    elif "error" in chunk:
                        log.warning(f"LLM stream error: {chunk['error']}")
                        break
                except json.JSONDecodeError:
                    continue
    except httpx.StreamError as e:
        log.warning(f"LLM stream interrupted: {e}")
        raise
    except Exception as e:
        log.error(f"LLM stream failed: {e}")
        raise
    
    
async def _execute_stream_generator(req: ExecuteRequest):
    """
    Gerador assíncrono que emite Server-Sent Events para o /execute.

    Tipos de evento:
      meta   – metadados de roteamento (route, confidence, execution_id)
      delta  – fragmento de token LLM (streaming em tempo real, TEXTO PURO)
      step   – notificação de conclusão de step do plano CoT
      result – resultado completo (busca, memória, local_scraping, etc., TEXTO PURO)
      error  – mensagem de erro
      done   – evento final com resposta completa + estatísticas
    """
    t0 = time.perf_counter()
    eid = str(uuid.uuid4())
    sid = req.session_id or str(uuid.uuid4())
    log.info(f"[{eid[:8]}] Executando (stream): '{req.input[:80]}'")
    try:
        req.input = await _add_think_instruction(req)
        log.info(f"[{eid[:8]}] Think depth aplicado")
    except Exception as e:
        log.warning(f"[{eid[:8]}] _verify_think falhou, usando input original: {e}")

    route_info   = _internal_classify(req.input, req.image_path)
    route_name   = route_info["route"]
    route_conf   = route_info["confidence"]
    route_method = route_info["method"]
    needs_cot    = route_info["needs_cot"]
    routed_directly = not needs_cot and route_name in ROUTE_TO_EXECUTOR

    log.info(
        f"[{eid[:8]}] Router → {route_name} ({route_conf:.0%} via {route_method})"
        f" {'→ CoT' if needs_cot else '→ Direct'}"
    )

    # ── 1. Metadados iniciais ───────────────────────────────────────────
    yield _sse("meta", {
        "execution_id": eid,
        "session_id": sid,
        "route": route_name,
        "route_confidence": route_conf,
        "route_method": route_method,
        "routed_directly": routed_directly,
    })

    step_results: list[StepResult] = []
    context: dict[str, Any] = {}
    plan_cache = False
    tts_already_fired = False
    final_response = ""

    try:
        # ────────────────────────────────────────────────────────────────
        #  ROTA DIRETA
        # ────────────────────────────────────────────────────────────────
        if not needs_cot and route_name in ROUTE_TO_EXECUTOR:
            executor = ROUTE_TO_EXECUTOR.get(route_name, "llm")
            action = f"gravar {req.input}" if route_name == "memory_write" else req.input

            if executor == "llm":
                t0_step = time.perf_counter()
                try:
                    async for event_type, data in _stream_llm_chunks(
                        action,
                        voice=req.voice,
                        lang=req.lang,
                        tts=req.tts,
                        session_id=sid,
                    ):
                        if event_type == "reasoning":
                            yield _sse("reasoning", data)  # ← NOVO
                        else:
                            final_response += data
                            yield _sse("delta", data)

                    lat = round((time.perf_counter() - t0_step) * 1000, 2)
                    step_results = [StepResult(
                        step=1, executor="llm", action=action,
                        success=True, result=final_response,
                        latency_ms=lat,
                    )]
                    tts_already_fired = req.tts
                except Exception as e:
                    lat = round((time.perf_counter() - t0_step) * 1000, 2)
                    step_results = [StepResult(
                        step=1, executor="llm", action=action,
                        success=False, error=str(e), latency_ms=lat,
                    )]
                    final_response = f"Erro: {e}"
                    yield _sse("error", {"error": str(e)})
            else:
                # ── Rota direta não-LLM ──
                sr = await _run_step_with_retry(1, action, executor, {}, req)
                step_results = [sr]
                context = {"step_1": sr.result} if sr.success else {}
                final_response = (
                    _result_to_text(sr.result) if sr.success
                    else f"Erro: {sr.error}"
                )
                yield _sse("result", final_response)  # TEXTO PURO

        # ────────────────────────────────────────────────────────────────
        #  PIPELINE CoT
        # ────────────────────────────────────────────────────────────────
        else:
            log.info(f"[{eid[:8]}] 🧠 CoT pipeline")
            raw = []
            try:
                cr = await state.cot_client.post(
                    "/plan", json={"input": req.input, "use_cache": req.use_cache}
                )
                cr.raise_for_status()
                pd = cr.json()
                raw = pd.get("steps", [])
                plan_cache = pd.get("from_cache", False)
                
                
            except Exception as e:
                yield _sse("error", {"error": f"CoT falhou: {e}"})
                final_response = f"CoT falhou: {e}"

            if raw:
                # Log de antes e depois da sanitização
                raw = _sanitize_steps(raw)
                
                # ... (resto do código continua)
                # ── Aplica saneamento para evitar erro de unhashable type ──
                raw = _sanitize_steps(raw)

                # ── Emite o plano completo antes de executar ──────────────────
                yield _sse("plan", {
                    "from_cache": plan_cache,
                    "steps": raw,  # Agora é seguro usar diretamente
                })

                # ── Executa com notificações em tempo real via Queue ──────────
                step_queue: asyncio.Queue = asyncio.Queue()
                plan_task = asyncio.create_task(
                    _execute_plan(raw, req, req.strategy, step_queue)
                )

                while True:
                    item = await step_queue.get()
                    if item is None:
                        break
                    
                    if item["type"] == "step_start":
                        yield _sse("step_start", {
                            "step":     item["step"],
                            "executor": item["executor"],
                            "action":   item.get("action", ""),
                        })
                    elif item["type"] == "step_delta":
                        # Manda o token do LLM direto para a UI em tempo real!
                        yield _sse("delta", item["data"])
                    elif item["type"] == "step_reasoning":
                        # Manda o raciocínio do LLM direto para a UI em tempo real!
                        yield _sse("reasoning", item["data"])
                    elif item["type"] == "step_done":
                        sr = item["data"]
                        yield _sse("step_done", {
                            "step":       sr.step,
                            "executor":   sr.executor,
                            "success":    sr.success,
                            "latency_ms": sr.latency_ms,
                            "error":      sr.error,
                        })

                step_results, context, tts_already_fired = await plan_task

                # ── Monta resposta final ──
                succ = [s for s in step_results if s.success]
                if not succ:
                    final_response = (
                        f"Erros: {'; '.join(s.error for s in step_results if s.error)}"
                    )
                    yield _sse("error", {"error": final_response})
                else:
                    # Caso especial: local_scraping com múltiplos matches
                    local_scraping_handled = False
                    for s in succ:
                        if s.executor == "local_scraping" and isinstance(s.result, dict):
                            if s.result.get("status") == "multiple_matches":
                                matches = s.result.get("matches", [])
                                msg = s.result.get(
                                    "message", "Múltiplos arquivos encontrados."
                                )
                                lines = [
                                    f"  [{i+1}] {m.get('file_path','?')}  "
                                    f"({m.get('size','?')})"
                                    for i, m in enumerate(matches[:10])
                                ]
                                final_response = f"{msg}\n\n" + "\n".join(lines)
                                yield _sse("result", final_response)  # TEXTO PURO
                                local_scraping_handled = True
                                break

                            if (
                                s.result.get("status") == "success"
                                and s.result.get("content")
                            ):
                                content   = s.result["content"]
                                file_path = s.result.get("file_path", "")
                                reindex_info = ""
                                if s.result.get("was_reindexed"):
                                    reindex_info = " (reindexado)"
                                elif not s.result.get("hash_match", True):
                                    reindex_info = " (hash diferente — reindexado)"
                                final_response = (
                                    f"📄 Arquivo: {file_path}{reindex_info}"
                                    f"\n\n{content}"
                                )
                                yield _sse("result", final_response)  # TEXTO PURO
                                local_scraping_handled = True
                                break

                    if not local_scraping_handled:
                        last = succ[-1]
                        # Último step é LLM/deep_search com texto útil → enviar direto
                        if (
                            last.executor in ("llm", "deep_search")
                            and isinstance(last.result, str)
                            and len(last.result) > 20
                        ):
                            final_response = last.result
                            yield _sse("result", final_response)  # TEXTO PURO
                        else:
                            # ── Síntese via LLM em streaming ──
                            parts = [
                                f"[{s.executor.upper()} — step {s.step}]\n"
                                f"{_result_to_text(s.result)}"
                                for s in succ
                                if _result_to_text(s.result)
                            ]
                            if parts:
                                ctx_text = "\n\n".join(parts)[:MAX_CONTEXT_CHARS]
                                synthesis_prompt = (
                                    f"Original: {req.input}\nInfo:\n{ctx_text}\n"
                                    f"Direct response in same language:"
                                )
                                try:
                                    async for event_type, data in _stream_llm_chunks(
                                        synthesis_prompt,
                                        voice=req.voice,
                                        lang=req.lang,
                                        tts=req.tts,
                                        session_id=sid,
                                        max_turns=1,
                                    ):
                                        if event_type == "delta":
                                            final_response += data
                                            yield _sse("delta", data)
                                    tts_already_fired = req.tts
                                except Exception:
                                    final_response = ctx_text
                                    yield _sse("result", final_response)  # TEXTO PURO
                            else:
                                final_response = (
                                    "Tarefa concluída sem resultado textual."
                                )
                                yield _sse("result", final_response)  # TEXTO PURO

    except Exception as e:
        final_response = f"Erro interno: {e}"
        yield _sse("error", {"error": str(e)})

    # ── TTS se não foi disparado durante o streaming ──
    if req.tts and final_response and not tts_already_fired:
        asyncio.create_task(_fire_tts(final_response, req))

    # ── Persistir turno ──
    if final_response:
        asyncio.create_task(_save_turn(req.input, final_response, sid))
        asyncio.create_task(_save_lt(f"Usuário disse: {req.input[:200]}"))

    # ── Evento final ──
    lat = round((time.perf_counter() - t0) * 1000, 2)
    errors = [
        f"Step {s.step} [{s.executor}]: {s.error}"
        for s in step_results
        if not s.success
    ]

    yield _sse("done", {
        "execution_id": eid,
        "final_response": final_response,
        "steps": [s.model_dump() for s in step_results],
        "plan_from_cache": plan_cache,
        "total_latency_ms": lat,
        "errors": errors,
        "route": route_name,
        "route_confidence": route_conf,
        "route_method": route_method,
        "routed_directly": routed_directly,
    })



def _sse(event: str, data: Any) -> str:
    """
    Formata um frame Server-Sent Events robusto.
    - Usa o campo 'event' do SSE para tipificar a mensagem.
    - Se `data` for dict, serializa como JSON (single-line).
    - Se `data` for str, envia como TEXTO PURO, sem serialização JSON,
      preservando Markdown e LaTeX perfeitamente (sem duplo escape).
    - Quebras de linha internas viram múltiplas linhas 'data:' conforme a spec SSE.
    """
    if isinstance(data, dict):
        payload = json.dumps(data, ensure_ascii=False)
    else:
        payload = str(data)
    
    # A spec do SSE exige que quebras de linha no dado virem múltiplas linhas "data:"
    lines = payload.split('\n')
    frame = f"event: {event}\n"
    for line in lines:
        frame += f"data: {line}\n"
    frame += "\n"  # Fim do evento (linha em branco)
    return frame


async def _verify_think(req: ExecuteRequest) -> int:
    grammar = r"""
root   ::= single-digit | double-digit
double-digit ::= "10"
single-digit ::= [0-9]
"""

    payload = {
        "model": "local",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a thinking-depth classifier. "
                    "Given a user input, respond with a single integer from 0 to 10 "
                    "representing how much deep reasoning or complex thinking is required. "
                    "0 = trivial (greetings, simple facts). "
                    "10 = very complex (multi-step reasoning, math proofs, deep research). "
                    "Respond with the number only. No explanation, no punctuation."
                ),
            },
            # Few-shot examples so the model calibrates correctly
            {"role": "user",      "content": "Oi, tudo bem?"},
            {"role": "assistant", "content": "0"},
            {"role": "user",      "content": "Qual a capital da França?"},
            {"role": "assistant", "content": "1"},
            {"role": "user",      "content": "Explique o que é machine learning."},
            {"role": "assistant", "content": "4"},
            {"role": "user",      "content": "Pesquise sobre os impactos econômicos da IA e resuma em tópicos."},
            {"role": "assistant", "content": "7"},
            {"role": "user",      "content": "Prove que existem infinitos números primos e explique cada passo."},
            {"role": "assistant", "content": "10"},
            # Actual user input
            {"role": "user",      "content": req.input},
        ],
        "grammar":     grammar,
        "temperature": 0.0,
        "max_tokens":  4,
        "stream":      False,
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(9999999.0)) as client:
        response = await client.post(
            "http://localhost:2001/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"].strip()
    return int(content)

async def _add_think_instruction(req: ExecuteRequest) -> str:
    depth = await _verify_think(req)
    think_instruction = THINK_DEPTH_INSTRUCTIONS[depth]

    # Injeta como system message adicional ou prefixo do contexto
    return f"[Reasoning depth: {depth}/10]\n{think_instruction}\n\n{req.input}"
# ══════════════════════════════════════════════════════════════════════════════
# FastAPI Application Endpoints
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="AVA Unified Orchestrator", version="3.1.0", description="Central Execution + Internal ONNX Routing Engine + Local Scraping", lifespan=lifespan)

@app.post("/execute")
    
async def execute(req: ExecuteRequest):
    async def stream_with_flush():
        async for chunk in _execute_stream_generator(req):
            yield chunk
            # Força o uvicorn a não bufferizar
    
    return StreamingResponse(
        stream_with_flush(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",  # <- adicione este
        },
    )


@app.post("/classify", response_model=ClassifyResponse)
async def classify_endpoint(req: ClassifyRequest):
    t0 = time.perf_counter(); ri = _internal_classify(req.text, req.image_path)
    return ClassifyResponse(route=ri["route"], confidence=ri["confidence"], method=ri["method"], all_scores=ri["all_scores"], needs_cot=ri["needs_cot"], latency_ms=round((time.perf_counter()-t0)*1000, 2))

@app.post("/deep-search")
async def deep_search(req: DeepSearchRequest):
    if not req.text.strip(): raise HTTPException(400, "text vazio")
    try: r = await state.deep_search_client.post("/query", json={"text": req.text}); r.raise_for_status(); return r.json()
    except Exception as e: raise HTTPException(502, f"Deep Search falhou: {e}")

@app.post("/vision")
async def vision(req: VisionRequest):
    if not req.img_path.strip(): raise HTTPException(400, "img_path vazio")
    try: r = await state.vision_client.post("/describe", json={"img_path": req.img_path, "prompt": req.prompt}); r.raise_for_status(); return r.json()
    except Exception as e: raise HTTPException(502, f"Vision falhou: {e}")

@app.post("/memory/read")
async def memory_read(req: MemoryReadRequest):
    try: r = await state.memory_client.post("/read", json={"query": req.query, "top_k": req.top_k, "min_score": req.min_score, "session_id": req.session_id, "strategy": "auto"}); r.raise_for_status(); return r.json()
    except Exception as e: raise HTTPException(502, f"Memory falhou: {e}")

@app.post("/memory/write")
async def memory_write(req: MemoryWriteRequest):
    try: r = await state.memory_client.post("/write", json={"text": req.text, "source": req.source, "confidence": req.confidence}); r.raise_for_status(); return r.json()
    except Exception as e: raise HTTPException(502, f"Memory falhou: {e}")

@app.post("/search")
async def search(query: str, max_results: int = 5, search_pdfs: bool = False):
    try: r = await state.search_client.post("/search", json={"query": query, "max_results": max_results, "use_cache": True, "search_pdfs": search_pdfs}); r.raise_for_status(); return r.json()
    except Exception as e: raise HTTPException(502, f"Search falhou: {e}")

@app.post("/chat")
async def chat(message: str, voice: str = "M1", lang: str = "pt", tts: bool = True):
    try: r = await state.llm_client.post("/chat", json={"message": message, "voice": voice, "lang": lang, "max_turns": 10, "tts": tts}); r.raise_for_status(); return r.json()
    except Exception as e: raise HTTPException(502, f"Chat falhou: {e}")

# ── NEW: Local Scraping Endpoints ───────────────────────────────────────────

@app.post("/local-scraping")
async def local_scraping(req: LocalScrapingRequest):
    """
    Search and read a local file on the user's machine.
    
    Flow:
      1. Orchestrator forwards query to local-scraping service
      2. Service generates a search command via model
      3. REST API client on user's machine executes the command
      4. Returns file content + path
      5. If multiple files match → returns list for user choice
      6. If file was indexed → hash comparison → reindex if changed
      7. Indexed info saved to memory
    """
    if not req.query.strip():
        raise HTTPException(400, "query vazio")
    try:
        r = await state.local_scraping_client.post(
            "/scrape",
            json={
                "query":         req.query,
                "search_path":   req.search_path,
                "force_reindex": req.force_reindex,
                "session_id":    req.session_id,
            },
        )
        r.raise_for_status()
        data = r.json()
        
        # If multiple matches, return the choice list
        if data.get("multiple_matches"):
            return data
        
        # Single file — persist indexed info to memory
        file_content = data.get("content", "")
        file_path    = data.get("file_path", "")
        was_reindexed = data.get("was_reindexed", False)
        hash_match    = data.get("hash_match", True)
        
        if file_content and file_path:
            try:
                summary = file_content[:500] if len(file_content) > 500 else file_content
                await state.memory_client.post(
                    "/write",
                    json={
                        "text":       f"[ARQUIVO INDEXADO] {file_path}: {summary}",
                        "source":     "local_scraping",
                        "confidence": 1.0 if hash_match else 0.9,
                    },
                )
            except Exception as e:
                log.warning(f"Falha ao salvar arquivo indexado na memória: {e}")
        
        return data
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"Local Scraping falhou: HTTP {e.response.status_code}")
    except httpx.ConnectError:
        raise HTTPException(502, "Local Scraping serviço offline")
    except Exception as e:
        raise HTTPException(502, f"Local Scraping falhou: {e}")


@app.post("/local-scraping/choose")
async def local_scraping_choose(req: LocalScrapingChooseRequest):
    """
    Choose a specific file when multiple matches were found.
    
    After a /local-scraping call returns `multiple_matches: true`,
    the user selects one file_path from the list and calls this
    endpoint to actually read and index the chosen file.
    """
    if not req.file_path.strip():
        raise HTTPException(400, "file_path vazio")
    try:
        r = await state.local_scraping_client.post(
            "/choose",
            json={
                "query":         req.query,
                "file_path":     req.file_path,
                "force_reindex": req.force_reindex,
                "session_id":    req.session_id,
            },
        )
        r.raise_for_status()
        data = r.json()
        
        # Persist chosen file to memory
        file_content = data.get("content", "")
        file_path    = data.get("file_path", req.file_path)
        hash_match   = data.get("hash_match", True)
        
        if file_content and file_path:
            try:
                summary = file_content[:500] if len(file_content) > 500 else file_content
                await state.memory_client.post(
                    "/write",
                    json={
                        "text":       f"[ARQUIVO INDEXADO] {file_path}: {summary}",
                        "source":     "local_scraping",
                        "confidence": 1.0 if hash_match else 0.9,
                    },
                )
            except Exception as e:
                log.warning(f"Falha ao salvar arquivo escolhido na memória: {e}")
        
        return data
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"Local Scraping choose falhou: HTTP {e.response.status_code}")
    except httpx.ConnectError:
        raise HTTPException(502, "Local Scraping serviço offline")
    except Exception as e:
        raise HTTPException(502, f"Local Scraping choose falhou: {e}")


@app.get("/local-scraping/indexed")
async def local_scraping_indexed(file_path: Optional[str] = None):
    """
    List indexed files, or check if a specific file is indexed.
    Optionally filter by file_path to check indexing status and hash.
    """
    try:
        params = {}
        if file_path:
            params["file_path"] = file_path
        r = await state.local_scraping_client.get("/indexed", params=params)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        raise HTTPException(502, "Local Scraping serviço offline")
    except Exception as e:
        raise HTTPException(502, f"Local Scraping indexed falhou: {e}")


@app.delete("/local-scraping/index/{file_id}")
async def local_scraping_delete_index(file_id: str):
    """Remove a specific file from the local-scraping index."""
    try:
        r = await state.local_scraping_client.delete(f"/index/{file_id}")
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        raise HTTPException(502, "Local Scraping serviço offline")
    except Exception as e:
        raise HTTPException(502, f"Local Scraping delete falhou: {e}")


# ── Existing Endpoints (unchanged) ──────────────────────────────────────────

@app.get("/status")
async def status():
    checks = {}
    cfg = {
        "cot": (state.cot_client, HEALTH_PATHS["cot"]),
        "memory": (state.memory_client, HEALTH_PATHS["memory"]),
        "search": (state.search_client, HEALTH_PATHS["search"]),
        "local_scraping": (state.local_scraping_client, HEALTH_PATHS["local_scraping"]),  # ← NEW
        "tts": (state.tts_client, HEALTH_PATHS["tts"]),
        "llm": (state.llm_client, HEALTH_PATHS["llm"]),
        "vision": (state.vision_client, HEALTH_PATHS["vision"]),
        "deep_search": (state.deep_search_client, HEALTH_PATHS["deep_search"]),
    }
    for n, (c, p) in cfg.items():
        try: r = await c.get(p, timeout=2.0); checks[n] = {"healthy": r.status_code == 200, "status_code": r.status_code}
        except: checks[n] = {"healthy": False, "status_code": None}
    return {"orchestrator": "ok", "internal_router": {"minilm_loaded": minilm_router.ready}, "services": checks, "executors": list(EXECUTOR_ADAPTERS.keys())}
    
@app.delete("/cache")
async def invalidate_cot_cache():
    try: r = await state.cot_client.delete("/cache", timeout=10.0); r.raise_for_status(); return r.json()
    except Exception as e: raise HTTPException(502, f"Cache invalidation falhou: {e}")

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    try: r = await state.memory_client.delete(f"/session/{session_id}", timeout=5.0); r.raise_for_status(); return r.json()
    except Exception as e: raise HTTPException(502, f"Session clear falhou: {e}")

@app.post("/tts/cancel")
async def cancel_tts():
    try: r = await state.tts_client.post("/cancel", timeout=5.0); r.raise_for_status(); return r.json()
    except Exception as e: raise HTTPException(502, f"TTS cancel falhou: {e}")

@app.get("/tts/voices")
async def list_voices():
    try: r = await state.tts_client.get("/voices", timeout=5.0); r.raise_for_status(); return r.json()
    except Exception as e: raise HTTPException(502, f"TTS voices falhou: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("orchestrator:app", host="0.0.0.0", port=9000, log_level="info")