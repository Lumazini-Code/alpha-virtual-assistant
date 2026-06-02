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

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

MODELS_DIR =  "./Modules/Models"
DEBERTA_DIR  = os.path.join(MODELS_DIR, "DebertaV2ForSequenceClassification")
MINILM_DIR   = os.path.join(MODELS_DIR, "ms-marco-MiniLM-L-6-v2")

COT_URL         = "http://localhost:3000"
MEMORY_URL      = "http://localhost:3001"
SEARCH_URL      = "http://localhost:3002"
TTS_URL         = "http://localhost:3004"
LLM_URL         = "http://localhost:4003"
VISION_URL      = "http://localhost:4002"
DEEP_SEARCH_URL = "http://localhost:4005"

HEALTH_PATHS: dict[str, str] = {
    "cot": "/status", "memory": "/status", "search": "/status",
    "tts": "/status", "llm": "/health", "vision": "/health", "deep_search": "/health",
}

EXECUTOR_TIMEOUTS: dict[str, float] = {
    "llm": 120.0, "memory": 5.0, "search": 25.0, "deep_search": 90.0,
    "vision": 60.0, "tts": 15.0, "stt": 30.0, "commander": 10.0,
    "translator": 30.0, "calculator": 5.0,
}

EXECUTOR_MAX_RETRIES: dict[str, int] = {
    "llm": 2, "memory": 3, "search": 2, "deep_search": 1, "vision": 1,
    "tts": 2, "stt": 0, "commander": 0, "translator": 2, "calculator": 3,
}

RETRY_BACKOFF_S   = 0.15
COT_TIMEOUT_S     = 45.0
MAX_CONTEXT_CHARS = 1500
DEFAULT_TOP_K     = 5
DEFAULT_MIN_SCORE = 0.30

# Router Configuration
CONFIDENCE_THRESHOLD = float(os.environ.get("ROUTER_CONFIDENCE", "0.65"))
HEURISTIC_MIN_SCORE  = 0.35
ONNX_PROVIDERS = os.environ.get("ONNX_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider").split(",")

# Direct Route Mapping
ROUTE_TO_EXECUTOR: dict[str, str] = {
    "llm": "llm", "search": "search", "memory_read": "memory", "memory_write": "memory",
    "vision": "vision", "deep_search": "deep_search", "calculator": "calculator",
    "commander": "commander", "translator": "translator", "tts": "tts",
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


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL ROUTER — ONNX + Heuristics
# ══════════════════════════════════════════════════════════════════════════════

class RouteLabel(IntEnum):
    COT = 0; LLM = 1; SEARCH = 2; MEMORY_READ = 3; MEMORY_WRITE = 4
    VISION = 5; DEEP_SEARCH = 6; CALCULATOR = 7; COMMANDER = 8; TRANSLATOR = 9; TTS = 10

LABEL_NAMES: list[str] = [rl.name.lower() for rl in RouteLabel]
NUM_LABELS = len(RouteLabel)

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

class OnnxDebertaRouter:
    def __init__(self): self.ready = False; self.fine_tuned = False; self.tokenizer = None; self.session = None; self.input_names = []; self.output_names = []; self.num_labels = 0; self.provider = ""; self.model_path = ""
    def load(self, model_dir: str) -> bool:
        try:
            import onnxruntime as ort
        except ImportError: return False
        self.tokenizer = _load_tokenizer(model_dir)
        if not self.tokenizer: return False
        onnx_path = _find_onnx_file(model_dir)
        if not onnx_path: return False
        try:
            opts = ort.SessionOptions(); opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL; opts.intra_op_num_threads = 4
            self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=[p for p in ONNX_PROVIDERS if p in ort.get_available_providers()] or ["CPUExecutionProvider"])
            self.input_names = [i.name for i in self.session.get_inputs()]; self.output_names = [o.name for o in self.session.get_outputs()]
            self.provider = self.session.get_providers()[0]; self.model_path = onnx_path
            shape = self.session.get_outputs()[0].shape
            self.num_labels = shape[-1] if len(shape) >= 2 and isinstance(shape[-1], int) else 0
            self.fine_tuned = (self.num_labels == NUM_LABELS); self.ready = True; return True
        except Exception: return False
    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        if not self.ready or not self.fine_tuned: raise RuntimeError("Not ready")
        feed = {k: v for k, v in _encode_for_onnx(self.tokenizer, text).items() if k in self.input_names}
        logits = self.session.run(self.output_names, feed)[0].squeeze().astype(np.float64)
        probs = _softmax(logits)
        scores = {LABEL_NAMES[i]: round(float(probs[i]), 4) for i in range(min(len(probs), NUM_LABELS)) if i < len(LABEL_NAMES)}
        idx = int(probs[:NUM_LABELS].argmax()); return LABEL_NAMES[idx], round(float(probs[idx]), 4), scores

class OnnxMiniLMRouter:
    ROUTE_PROTOTYPES: dict[str, list[str]] = {
        "cot": ["search and then translate", "research and summarize", "complex multi step task"],
        "llm": ["hello how are you", "explain what is ai", "tell me a joke", "write a poem"],
        "search": ["search the web for", "find information about", "look up price", "procure na internet"],
        "memory_read": ["what do you remember", "recall my conversation", "what do you know about me"],
        "memory_write": ["remember my favorite color", "save this info", "grave na memória"],
        "vision": ["describe this image", "analyze photo", "what does picture show"],
        "deep_search": ["deep research about", "investigate causes of", "pesquisa profunda"],
        "calculator": ["calculate 15 times 37", "square root of 144", "quanto é 100"],
        "commander": ["open firefox", "launch vscode", "abrir spotify"],
        "translator": ["translate to english", "traduzir para espanhol"],
        "tts": ["speak text aloud", "fale em voz alta"],
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

# Heuristic Rules
_HEURISTIC_RULES = [
    ("memory_write", ["lembrar","memorizar","gravar","salvar","remember","save"], re.compile(r"(lembrar|memorizar|gravar|salvar|remember|save|store|memo)\b", re.I)),
    ("memory_read", ["o que você sabe","recall"], re.compile(r"(o que (você|voce) (sabe|lembra)|recall|what do you (know|remember))", re.I)),
    ("translator", ["traduz","translate"], re.compile(r"(traduz|tradu[çc][aã]|translate|em (ingl[eê]s|espanhol)|to (english|spanish))", re.I)),
    ("calculator", ["quanto é","calcule"], re.compile(r"(quanto [eé]|calcule?|calculate|\d+\s*[\+\-\*\/\^x×÷]\s*\d+)", re.I)),
    ("commander", ["abrir","open"], re.compile(r"^(abrir|open|launch|iniciar|executar|start|run)\b", re.I)),
    ("vision", ["imagem","foto","image"], re.compile(r"(imagem|foto|picture|image|screenshot|descreva a|what is in the)", re.I)),
    ("deep_search", ["pesquisa profunda","research"], re.compile(r"(pesquisa profunda|investigue|deep.?search|research|estude sobre)", re.I)),
    ("search", ["procure","busque","search"], re.compile(r"(procure|busque|pesquis|search|look up|find|not[ií]cia|news)", re.I)),
    ("tts", ["fale","speak"], re.compile(r"^(fale|diga em voz alta|speak|read aloud)\b", re.I)),
]

def _heuristic_classify(text: str, image_path: Optional[str] = None) -> tuple[str, float, dict[str, float]]:
    scores: dict[str, float] = {name: 0.0 for name in LABEL_NAMES}
    if image_path: scores["vision"] += 0.5
    tl = text.lower().strip()
    for rn, kws, pat in _HEURISTIC_RULES:
        if pat.search(tl): scores[rn] += 0.4
        if any(kw in tl for kw in kws): scores[rn] += 0.15
    if sum(1 for s in scores.values() if s > 0.2) >= 2: scores["cot"] += 0.5
    if re.search(r"\b(e\s+(depois|então)|and\s+then|compare|resuma\s+e|primeiro|step by step)\b", tl): scores["cot"] += 0.3
    wc = len(text.split())
    if wc > 25: scores["cot"] += 0.15
    if tl.endswith("?") and max(scores.values()) < 0.15: scores["llm"] += 0.3
    if re.search(r"^(oi|olá|hello|hi|bom dia)\b", tl): scores["llm"] += 0.5
    total = sum(scores.values())
    if total > 0: scores = {k: v/total for k, v in scores.items()}
    best = max(scores, key=scores.get); bscore = scores[best]
    if bscore < HEURISTIC_MIN_SCORE: best = "cot" if wc > 15 else "llm"; bscore = 0.35
    return best, round(bscore, 4), scores

# Global Router State
deberta_router = OnnxDebertaRouter()
minilm_router  = OnnxMiniLMRouter()

def _internal_classify(text: str, image_path: Optional[str] = None) -> dict:
    route, confidence, method, all_scores = "cot", 0.0, "default", {n: 0.0 for n in LABEL_NAMES}
    d_scores, m_scores = {}, {}
    if deberta_router.ready and deberta_router.fine_tuned:
        try: route, confidence, d_scores = deberta_router.predict(text); method = "onnx_deberta"
        except Exception: pass
    if minilm_router.ready:
        try: _, _, m_scores = minilm_router.predict(text); method = "onnx_minilm" if method == "default" else method + "+minilm"
        except Exception: pass
    h_route, h_conf, h_scores = _heuristic_classify(text, image_path)
    if method == "default": method = "heuristic"
    else: method += "+heuristic"
    
    if d_scores and m_scores:
        for n in LABEL_NAMES: all_scores[n] = round(d_scores.get(n,0)*0.5 + m_scores.get(n,0)*0.2 + h_scores.get(n,0)*0.3, 4)
    elif d_scores:
        for n in LABEL_NAMES: all_scores[n] = round(d_scores.get(n,0)*0.6 + h_scores.get(n,0)*0.4, 4)
    elif m_scores:
        for n in LABEL_NAMES: all_scores[n] = round(m_scores.get(n,0)*0.4 + h_scores.get(n,0)*0.6, 4)
    else: all_scores = h_scores
        
    route = max(all_scores, key=all_scores.get); confidence = all_scores[route]
    if confidence < HEURISTIC_MIN_SCORE: route, confidence, method = "cot", 0.3, "default_safe"
    return {"route": route, "confidence": confidence, "method": method, "all_scores": all_scores, "needs_cot": route == "cot"}


# ══════════════════════════════════════════════════════════════════════════════
# Global State & Lifespan
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AppState:
    cot_client: httpx.AsyncClient = field(default=None); memory_client: httpx.AsyncClient = field(default=None)
    search_client: httpx.AsyncClient = field(default=None); tts_client: httpx.AsyncClient = field(default=None)
    llm_client: httpx.AsyncClient = field(default=None); vision_client: httpx.AsyncClient = field(default=None)
    deep_search_client: httpx.AsyncClient = field(default=None)

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Iniciando AVA Unified Orchestrator & Router...")
    
    # Load internal ONNX Routers
    d_loaded = deberta_router.load(DEBERTA_DIR)
    m_loaded = minilm_router.load(MINILM_DIR)
    log.info(f"Router Init → Deberta: {'✓' if d_loaded else '✗'} | MiniLM: {'✓' if m_loaded else '✗'} | Heuristic: ✓")

    # Probe external services
    service_urls = {"cot": COT_URL, "memory": MEMORY_URL, "search": SEARCH_URL, "tts": TTS_URL, "llm": LLM_URL, "vision": VISION_URL, "deep_search": DEEP_SEARCH_URL}
    async with httpx.AsyncClient(timeout=5.0) as probe:
        for name, url in service_urls.items():
            try:
                r = await probe.get(f"{url}{HEALTH_PATHS.get(name, '/status')}")
                log.info(f"  ✓ {name:12s} OK" if r.status_code == 200 else f"  ⚠ {name:12s} {r.status_code}")
            except httpx.ConnectError: log.warning(f"  ✗ {name:12s} OFFLINE")

    def _make_client(base_url: str, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout), limits=httpx.Limits(max_keepalive_connections=4, max_connections=8))

    state.cot_client = _make_client(COT_URL, COT_TIMEOUT_S); state.memory_client = _make_client(MEMORY_URL, EXECUTOR_TIMEOUTS["memory"])
    state.search_client = _make_client(SEARCH_URL, EXECUTOR_TIMEOUTS["search"]); state.tts_client = _make_client(TTS_URL, EXECUTOR_TIMEOUTS["tts"])
    state.llm_client = _make_client(LLM_URL, EXECUTOR_TIMEOUTS["llm"]); state.vision_client = _make_client(VISION_URL, EXECUTOR_TIMEOUTS["vision"])
    state.deep_search_client = _make_client(DEEP_SEARCH_URL, EXECUTOR_TIMEOUTS["deep_search"])
    
    log.info("Orchestrator pronto — todos os clientes HTTP inicializados")
    yield
    for c in (state.cot_client, state.memory_client, state.search_client, state.tts_client, state.llm_client, state.vision_client, state.deep_search_client): await c.aclose()
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


async def _adapt_llm(action, context, req, stream_tts=False) -> str:
    ctx = _format_context_for_llm(action, context)
    msg = f"{action}\n\n{ctx}" if ctx else action
    return await _stream_llm(
        msg,
        voice=req.voice,
        lang=req.lang,
        tts=stream_tts,          # ← True = user hears audio in real-time
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

async def _adapt_commander(action, context, req) -> str:
    res = _resolve_action_text(action, context) or action; cl = res.lower()
    if any(cl.startswith(p) for p in _OPEN_PREFIXES): cmd = ["xdg-open", res.split(None, 1)[1] if len(res.split()) > 1 else res]
    else:
        try: cmd = shlex.split(res)
        except ValueError as e: raise RuntimeError(f"Comando inválido: {e}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8.0); out = proc.stdout.strip() or proc.stderr.strip() or "ok"
        if proc.returncode != 0: raise RuntimeError(f"Exit {proc.returncode}: {out}"); return out
    except subprocess.TimeoutExpired: raise RuntimeError("Timeout 8s")
    except FileNotFoundError: raise RuntimeError(f"Cmd não encontrado: {cmd[0]}")

async def _adapt_translator(action, context, req) -> str:
    txt = _resolve_action_text(action, context) or action
    return await _stream_llm(
        f"Translate accurately. Return ONLY translation:\n\n{txt}",
        voice=req.voice,
        lang=req.lang,
        tts=False,
        max_turns=1,
    )

async def _adapt_calculator(action, context, req) -> str:
    res = _resolve_action_text(action, context) or action
    m = re.search(r"[\d\s\+\-\*\/\(\)\.\^%]+", res)
    if m:
        expr = m.group().strip().replace("^", "**")
        try:
            return str(eval(expr, {"__builtins__": {}}, {}))
        except Exception:
            pass
    return await _stream_llm(
        f"Solve calculation, return ONLY numeric result:\n{res}",
        voice=req.voice,
        lang=req.lang,
        tts=False,
        max_turns=1,
    )

EXECUTOR_ADAPTERS = {"llm": _adapt_llm, "memory": _adapt_memory, "search": _adapt_search, "deep_search": _adapt_deep_search, "vision": _adapt_vision, "tts": _adapt_tts, "stt": _adapt_stt, "commander": _adapt_commander, "translator": _adapt_translator, "calculator": _adapt_calculator}

def _resolve_action_text(action, context):
    return re.sub(r"result of step (\d+)", lambda m: _result_to_text(context.get(f"step_{m.group(1)}")) if context.get(f"step_{m.group(1)}") else m.group(0), action, flags=re.IGNORECASE)

def _result_to_text(res):
    if not res: return ""
    if isinstance(res, str): return res[:MAX_CONTEXT_CHARS]
    if isinstance(res, list): return "\n".join([f"- {(i.get('text') or i.get('content') or str(i))[:300]}" for i in res[:6]])[:MAX_CONTEXT_CHARS]
    if isinstance(res, dict): return str(res.get("text") or res.get("content") or res.get("response") or res)[:MAX_CONTEXT_CHARS]
    return str(res)[:MAX_CONTEXT_CHARS]

def _format_context_for_llm(action, context):
    refs = set(int(m) for m in re.findall(r"result of step (\d+)", action, re.IGNORECASE))
    keys = {f"step_{n}" for n in refs} & set(context.keys()) if refs else set(context.keys())
    if not keys: return ""
    return "\n\n".join([f"[Step {k.split('_')[1]} result]\n{_result_to_text(context[k])}" for k in sorted(keys)])[:MAX_CONTEXT_CHARS]

async def _run_step_with_retry(step_num, action, executor, context, req, stream_tts=False) -> StepResult:
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
            # Pass stream_tts only to the LLM adapter
            if executor == "llm" and stream_tts:
                res = await asyncio.wait_for(
                    adapter(res_act, context, req, stream_tts=True),
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
    
    
async def _execute_plan(steps, req, strategy) -> tuple[list[StepResult], dict[str, Any], bool]:
    step_map = {s["step"]: s for s in steps}
    results = {}
    context = {}
    pending = set(step_map.keys())
    tts_fired = False

    # ── Identify "output steps" — steps no other step depends on ──
    # These are the final/terminal steps whose output reaches the user.
    # For LLM output steps, we stream with TTS so audio plays in real-time.
    dependents = set()
    for s in steps:
        for d in (s.get("depends_on") or []):
            dependents.add(d)
    output_steps = {s["step"] for s in steps if s["step"] not in dependents}

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

        # ── For terminal LLM steps, enable streaming TTS ──
        batch = await asyncio.gather(*[
            _run_step_with_retry(
                sn, step_map[sn]["action"], step_map[sn]["executor"], context, req,
                stream_tts=(sn in output_steps and step_map[sn]["executor"] == "llm" and req.tts),
            )
            for sn in ready
        ])
        for sr in batch:
            results[sr.step] = sr
            pending.discard(sr.step)
            context[f"step_{sr.step}"] = sr.result if sr.success else None
            # Track whether TTS was fired during execution
            if sr.success and sr.step in output_steps and sr.executor == "llm" and req.tts:
                tts_fired = True

    return [results[sn] for sn in sorted(results.keys())], context, tts_fired

async def _build_final_response(orig, steps, context, req) -> tuple[str, bool]:
    """
    Returns (final_text, tts_already_fired).
    When synthesis calls LLM, streams with tts=True so user hears audio immediately.
    """
    succ = [s for s in steps if s.success]
    if not succ:
        return f"Erros: {'; '.join(s.error for s in steps if s.error)}", False
    last = succ[-1]
    # If the last step was LLM/deep_search, return its result directly (no re-synthesis)
    if last.executor in ("llm", "deep_search") and isinstance(last.result, str) and len(last.result) > 20:
        return last.result, False  # TTS may have been fired during step execution
    parts = [f"[{s.executor.upper()} — step {s.step}]\n{_result_to_text(s.result)}"
             for s in succ if _result_to_text(s.result)]
    if not parts:
        return "Tarefa concluída sem resultado textual.", False
    ctx = "\n\n".join(parts)[:MAX_CONTEXT_CHARS]
    try:
        # Stream synthesis with TTS → user hears audio in real-time
        text = await _stream_llm(
            f"Original: {orig}\nInfo:\n{ctx}\nDirect response in same language:",
            voice=req.voice,
            lang=req.lang,
            tts=True,           # ← KEY CHANGE: TTS plays during generation
            max_turns=1,
        )
        return text or ctx, True  # tts_already_fired = True
    except Exception:
        return ctx, False

async def _execute_direct(route: str, req: ExecuteRequest) -> tuple[list[StepResult], dict[str, Any], bool]:
    """
    Returns (step_results, context, tts_already_fired).
    For direct LLM routes, streams with TTS for immediate audio.
    """
    executor = ROUTE_TO_EXECUTOR.get(route, "llm")
    action = f"gravar {req.input}" if route == "memory_write" else req.input
    tts_fired = False

    if executor == "llm":
        # ── Stream with TTS so user hears audio in real-time ──
        t0 = time.perf_counter()
        try:
            text = await _stream_llm(
                action,
                voice=req.voice,
                lang=req.lang,
                tts=req.tts,      # ← TTS fires during generation
                session_id=req.session_id or "default",
            )
            lat = round((time.perf_counter() - t0) * 1000, 2)
            sr = StepResult(step=1, executor="llm", action=action, success=True,
                            result=text, latency_ms=lat)
            tts_fired = req.tts
        except Exception as e:
            lat = round((time.perf_counter() - t0) * 1000, 2)
            sr = StepResult(step=1, executor="llm", action=action, success=False,
                            error=str(e), latency_ms=lat)
    else:
        sr = await _run_step_with_retry(1, action, executor, {}, req)

    ctx = {"step_1": sr.result} if sr.success else {}
    return [sr], ctx, tts_fired

async def _fire_tts(text, req):
    try: await state.tts_client.post("/speak", json={"text": text[:2000], "voice": req.voice, "lang": req.lang})
    except: pass

async def _save_turn(u, a, sid):
    try: await state.memory_client.post("/write_st", json={"session_id": sid, "turns": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]})
    except: pass

async def _save_lt(text, src="chat"):
    try: await state.memory_client.post("/write", json={"text": text[:500], "source": src, "confidence": 0.8})
    except: pass


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI Application Endpoints
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="AVA Unified Orchestrator", version="3.0.0", description="Central Execution + Internal ONNX Routing Engine", lifespan=lifespan)

@app.post("/execute", response_model=ExecuteResponse)
async def execute(req: ExecuteRequest):
    t0 = time.perf_counter()
    eid = str(uuid.uuid4())
    sid = req.session_id or str(uuid.uuid4())
    log.info(f"[{eid[:8]}] Executando: '{req.input[:80]}'")

    route_info = _internal_classify(req.input, req.image_path)
    route_name  = route_info["route"]
    route_conf  = route_info["confidence"]
    route_method = route_info["method"]
    needs_cot   = route_info["needs_cot"]
    log.info(f"[{eid[:8]}] Router → {route_name} ({route_conf:.0%} via {route_method}) {'→ CoT' if needs_cot else '→ Direct'}")

    step_results = []
    context = {}
    plan_cache = False
    tts_already_fired = False              # ← NEW

    if not needs_cot and route_name in ROUTE_TO_EXECUTOR:
        log.info(f"[{eid[:8]}] ⚡ Direct route: {route_name}")
        step_results, context, tts_already_fired = await _execute_direct(route_name, req)
    else:
        log.info(f"[{eid[:8]}] 🧠 CoT pipeline")
        try:
            cr = await state.cot_client.post("/plan", json={"input": req.input, "use_cache": req.use_cache})
            cr.raise_for_status()
            pd = cr.json()
            raw = pd.get("steps", [])
            plan_cache = pd.get("from_cache", False)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"CoT falhou: {e}")
        if not raw:
            raise HTTPException(status_code=500, detail="CoT plano vazio")
        step_results, context, tts_already_fired = await _execute_plan(raw, req, req.strategy)

    errors = [f"Step {s.step} [{s.executor}]: {s.error}" for s in step_results if not s.success]
    final, synthesis_tts_fired = await _build_final_response(req.input, step_results, context, req)

    # ── Only fire TTS if it wasn't already streamed during LLM generation ──
    any_tts_fired = tts_already_fired or synthesis_tts_fired
    if req.tts and final and not any_tts_fired:
        asyncio.create_task(_fire_tts(final, req))

    asyncio.create_task(_save_turn(req.input, final, sid))
    asyncio.create_task(_save_lt(f"Usuário disse: {req.input[:200]}"))

    lat = round((time.perf_counter() - t0) * 1000, 2)
    routed_directly = not needs_cot and route_name in ROUTE_TO_EXECUTOR
    return ExecuteResponse(
        execution_id=eid, input=req.input, session_id=sid, final_response=final,
        steps=step_results, plan_from_cache=plan_cache, total_latency_ms=lat,
        errors=errors, route=route_name, route_confidence=route_conf,
        route_method=route_method, routed_directly=routed_directly,
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

@app.get("/status")
async def status():
    checks = {}
    cfg = {"cot": (state.cot_client, HEALTH_PATHS["cot"]), "memory": (state.memory_client, HEALTH_PATHS["memory"]), "search": (state.search_client, HEALTH_PATHS["search"]), "tts": (state.tts_client, HEALTH_PATHS["tts"]), "llm": (state.llm_client, HEALTH_PATHS["llm"]), "vision": (state.vision_client, HEALTH_PATHS["vision"]), "deep_search": (state.deep_search_client, HEALTH_PATHS["deep_search"])}
    for n, (c, p) in cfg.items():
        try: r = await c.get(p, timeout=2.0); checks[n] = {"healthy": r.status_code == 200, "status_code": r.status_code}
        except: checks[n] = {"healthy": False, "status_code": None}
    return {"orchestrator": "ok", "internal_router": {"deberta_loaded": deberta_router.ready, "minilm_loaded": minilm_router.ready}, "services": checks, "executors": list(EXECUTOR_ADAPTERS.keys())}

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