"""
AVA Orchestrator — Tool-Calling Execution Engine
==================================================
Arquitetura (tool-calling nativo via Jinja chat template):

  1. O LLM recebe o histórico completo + a lista de tools no formato OpenAI
     (function-calling). A própria chat template do modelo (renderizada via
     Jinja pelo llama-server com `--jinja`) embute as definições das tools no
     prompt, e o parser nativo do servidor devolve `message.tool_calls` já
     estruturado (nome + argumentos JSON), sem precisarmos de uma segunda
     chamada com grammar manual.
  2. Se a resposta NÃO tiver tool_calls, o texto é apenas adicionado ao
     histórico como mensagem do assistant e o loop continua normalmente —
     sem lembrete, sem penalidade.
  3. Se houver tool_calls, cada uma é executada; o resultado volta ao
     histórico como mensagem role="tool" (atrelada ao tool_call_id) e o
     ciclo reinicia.
  4. O loop SÓ termina quando o modelo chama a tool "finish" (cujos
     argumentos contêm a resposta final). Não há limite de rodadas.

Resumo do fluxo:
  pergunta → LLM (+ tools via Jinja) → [sem tool_calls: continua o loop]
                                     → [com tool_calls X: executa X
                                        → resultado (role=tool) → repete]
                                     → [com tool_call finish: FIM]

Módulos expostos como tools: memory (read/write), search, deep_search, vision,
local_scraping, alpha_code.
TTS NÃO é uma tool do modelo — o sistema dispara TTS automaticamente sobre a
resposta final (igual antes), o modelo nunca decide chamar TTS.

Integra os microserviços AVA:
  - Memory          (port 3001)  — long-term, short-term, knowledge
  - Search          (port 3002)  — web search + cross-encoder rerank
  - Local Scraping  (port 3003)  — local file search, read & indexing
  - TTS             (port 3004)  — text-to-speech (Supertonic) — SISTEMA, não tool
  - LLM Chat        (port 4003)  — conversational inference (llama-server)
  - Vision / VQA    (port 4002)  — image understanding
  - Deep Search     (port 4005)  — KG-RAG com pesquisa web automática
  - Alpha Code      (port 4006)  — agente de geração/edição de código
"""
from __future__ import annotations

import json
import asyncio
import base64
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

MEMORY_URL           = "http://localhost:3001"
SEARCH_URL           = "http://localhost:3002"
LOCAL_SCRAPING_URL   = "http://localhost:3003"
TTS_URL              = "http://localhost:3004"
LLM_URL              = "http://localhost:4003"
VISION_URL           = "http://localhost:4002"
DEEP_SEARCH_URL      = "http://localhost:4005"
ALPHA_CODE_URL       = "http://localhost:4006"
# Gerenciador de processos (Rust/axum) que sobe/derruba o llama-server e
# permite trocar entre modo texto e multimodal via /llama/switch_mode.
PROCESS_MANAGER_URL  = "http://localhost:9001"

HEALTH_PATHS: dict[str, str] = {
    "memory": "/status", "search": "/status", "local_scraping": "/status", "tts": "/status",
    "llm": "/health", "vision": "/vision/status", "deep_search": "/health", "alpha_code": "/health",
    "process_manager": "/status",
}

# Timeout por tool (usado tanto para chamadas de tool quanto pro loop de seleção)
EXECUTOR_TIMEOUTS: dict[str, float] = {
    "llm": 9999999.0, "memory_read": 60.0, "memory_write": 30.0, "search": 60.0,
    "deep_search": 9999999.0, "vision_objects": 300.0, "tts": 60.0,
    "local_scraping": 9999999.0, "alpha_code": 9999999.0,
}

MAX_CONTEXT_CHARS = 1500
DEFAULT_TOP_K     = 5
DEFAULT_MIN_SCORE = 0.30
# Sem limite de rodadas: o loop só termina quando a tool "finish" é chamada.

THINK_DEPTH_INSTRUCTIONS: dict[int, str] = {
    0: "This is a trivial interaction — a greeting, acknowledgment, or simple social exchange. Respond naturally and briefly. No reasoning needed.",
    1: "This is a simple factual question with a direct answer. Retrieve the fact and respond concisely. No chain of thought needed.",
    2: "This requires minimal reasoning — a basic comparison, definition, or short explanation. Answer directly and clearly in a few sentences.",
    3: "This requires light reasoning. Think step by step briefly before answering, but keep your response focused and avoid unnecessary elaboration.",
    4: "This requires moderate reasoning. Break the problem into clear parts, think through each one, then synthesize a coherent answer.",
    5: "This requires balanced analytical thinking. Identify the key variables, consider different angles, weigh trade-offs, and build your answer progressively. Show your reasoning where helpful.",
    6: "This is a complex question. Think carefully before answering: identify assumptions, explore multiple perspectives, anticipate edge cases, and structure your response logically.",
    7: "This requires deep reasoning. Use a thorough chain of thought: decompose the problem, reason through each component independently, identify dependencies between parts, and synthesize a well-argued response.",
    8: "This is a highly complex task. Think extensively before responding. Map out the full problem space, consider competing hypotheses, validate intermediate conclusions, and build your final answer step by step. Precision and completeness matter here.",
    9: "This requires expert-level reasoning. Engage in rigorous multi-step thinking: define the problem formally, reason from first principles, explore edge cases, challenge your own intermediate conclusions, and produce a thorough, well-structured response. Do not skip reasoning steps.",
    10: "This is a maximally complex task requiring deep, exhaustive reasoning. Think as carefully and thoroughly as possible before responding. Decompose every sub-problem, reason from first principles at each step, validate every intermediate conclusion, consider all relevant edge cases and counter-arguments, and synthesize a complete, precise, and well-justified response. Take as much reasoning space as needed — correctness and depth are the priority.",
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
    image_path:  Optional[str]  = None
    search_pdfs: bool           = False
    stream:      bool           = True

class StepResult(BaseModel):
    step: int; executor: str; action: str; success: bool
    result: Optional[Any] = None; error: Optional[str] = None
    retries: int = 0; latency_ms: float = 0.0

class ExecuteResponse(BaseModel):
    execution_id: str; input: str; session_id: str; final_response: str
    steps: list[StepResult]; total_latency_ms: float; errors: list[str]

class DeepSearchRequest(BaseModel):
    text: str = Field(..., description="Question or research objective")

class MemoryReadRequest(BaseModel):
    query: str; top_k: int = DEFAULT_TOP_K; min_score: float = DEFAULT_MIN_SCORE; session_id: Optional[str] = None

class MemoryWriteRequest(BaseModel):
    text: str; source: str = "chat"; confidence: float = 1.0

class LocalScrapingRequest(BaseModel):
    query: str = Field(..., description="Nome ou descrição do arquivo a buscar")
    search_path: Optional[str] = Field(None, description="Caminho base para a busca (padrão: diretório da Alpha)")
    force_reindex: bool = Field(False, description="Forçar reindexação mesmo se o arquivo não mudou")
    session_id: Optional[str] = None

class LocalScrapingChooseRequest(BaseModel):
    query: str = Field(..., description="Query original da busca")
    file_path: str = Field(..., description="Caminho completo do arquivo escolhido")
    force_reindex: bool = Field(False, description="Forçar reindexação")
    session_id: Optional[str] = None

class AlphaCodeRequest(BaseModel):
    task: str = Field(..., description="Descrição da tarefa em linguagem natural")
    session_id: Optional[str] = Field(None, description="ID de sessão. None = gera novo.")
    project_dir: Optional[str] = Field(None, description="Diretório alvo. None = BASE_DIR do scraping_client.")
    max_steps: int = Field(25, ge=1, le=100, description="Limite de iterações ReAct")
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    model_override: Optional[str] = Field(None, description="Força modelo para TODOS os steps. None = roteamento adaptativo.")
    stream: bool = Field(True, description="True = SSE streaming, False = síncrono")


# ══════════════════════════════════════════════════════════════════════════════
# Global State & Lifespan
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AppState:
    memory_client: httpx.AsyncClient = field(default=None)
    search_client: httpx.AsyncClient = field(default=None)
    tts_client: httpx.AsyncClient = field(default=None)
    llm_client: httpx.AsyncClient = field(default=None)
    vision_client: httpx.AsyncClient = field(default=None)
    deep_search_client: httpx.AsyncClient = field(default=None)
    local_scraping_client: httpx.AsyncClient = field(default=None)
    alpha_code_client: httpx.AsyncClient = field(default=None)
    process_manager_client: httpx.AsyncClient = field(default=None)

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Iniciando AVA Orchestrator (tool-calling engine, sem router/CoT)...")

    service_urls = {
        "memory": MEMORY_URL, "search": SEARCH_URL, "local_scraping": LOCAL_SCRAPING_URL,
        "tts": TTS_URL, "llm": LLM_URL, "vision": VISION_URL,
        "deep_search": DEEP_SEARCH_URL, "alpha_code": ALPHA_CODE_URL,
        "process_manager": PROCESS_MANAGER_URL,
    }
    async with httpx.AsyncClient(timeout=5.0) as probe:
        for name, url in service_urls.items():
            try:
                r = await probe.get(f"{url}{HEALTH_PATHS.get(name, '/status')}")
                log.info(f"  ✓ {name:16s} OK" if r.status_code == 200 else f"  ⚠ {name:16s} {r.status_code}")
            except httpx.ConnectError:
                log.warning(f"  ✗ {name:16s} OFFLINE")

    def _make_client(base_url: str, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout), limits=httpx.Limits(max_keepalive_connections=4, max_connections=8))

    state.memory_client = _make_client(MEMORY_URL, 9999999.0)
    state.search_client = _make_client(SEARCH_URL, 9999999.0)
    state.tts_client = _make_client(TTS_URL, 9999999.0)
    state.llm_client = _make_client(LLM_URL, 9999999.0)
    state.vision_client = _make_client(VISION_URL, 9999999.0)
    state.deep_search_client = _make_client(DEEP_SEARCH_URL, 9999999.0)
    state.local_scraping_client = _make_client(LOCAL_SCRAPING_URL, 9999999.0)
    state.alpha_code_client = _make_client(ALPHA_CODE_URL, 9999999.0)
    # Timeout curto e finito (não 9999999.0): chamadas ao gerenciador de
    # processos (status / switch_mode) devem falhar rápido se ele estiver
    # offline, em vez de travar o loop de tools esperando para sempre.
    state.process_manager_client = _make_client(PROCESS_MANAGER_URL, 15.0)

    log.info("Orchestrator pronto — todos os clientes HTTP inicializados")
    yield
    for c in (state.memory_client, state.search_client, state.tts_client, state.llm_client,
              state.vision_client, state.deep_search_client, state.local_scraping_client,
              state.alpha_code_client, state.process_manager_client):
        await c.aclose()
    log.info("AVA Orchestrator encerrado")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers de resultado / contexto
# ══════════════════════════════════════════════════════════════════════════════

def _result_to_text(res) -> str:
    if not res:
        return "(no result)"
    if isinstance(res, str):
        return res[:MAX_CONTEXT_CHARS]
    if isinstance(res, dict):
        if res.get("status") == "multiple_matches":
            matches = res.get("matches", [])
            lines = [f"  [{i+1}] {m.get('file_path','?')} ({m.get('size','?')})" for i, m in enumerate(matches[:10])]
            return "Multiple files found:\n" + "\n".join(lines)
        if res.get("status") == "success" and res.get("content"):
            return res["content"][:MAX_CONTEXT_CHARS]
        return str(res.get("text") or res.get("content") or res.get("answer") or res.get("response") or res)[:MAX_CONTEXT_CHARS]
    if isinstance(res, list):
        return "\n".join([f"- {(i.get('text') or i.get('content') or str(i))[:300]}" for i in res[:6]])[:MAX_CONTEXT_CHARS]
    return str(res)[:MAX_CONTEXT_CHARS]


async def _fire_tts(text, req):
    """TTS é disparado pelo SISTEMA sobre a resposta final — o modelo nunca chama TTS."""
    try:
        await state.tts_client.post("/speak", json={"text": text[:2000], "voice": req.voice, "lang": req.lang})
    except Exception:
        pass

async def _save_turn(u, a, sid):
    try:
        await state.memory_client.post("/write_st", json={"session_id": sid, "turns": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]})
    except Exception:
        pass

async def _save_lt(text, src="chat"):
    try:
        await state.memory_client.post("/write", json={"text": text[:500], "source": src, "confidence": 0.8})
    except Exception:
        pass

def _sse(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else str(data)
    lines = payload.split("\n")
    frame = f"event: {event}\n"
    for line in lines:
        frame += f"data: {line}\n"
    frame += "\n"
    return frame


# ══════════════════════════════════════════════════════════════════════════════
# Thinking-depth calibration (independente do tool loop — não é "roteamento
# de módulo", apenas calibra o quanto o LLM deve raciocinar antes de responder)
# ══════════════════════════════════════════════════════════════════════════════

_THINK_GRAMMAR = r"""
root   ::= single-digit | double-digit
double-digit ::= "10"
single-digit ::= [0-9]
"""

async def _verify_think(text: str) -> int:
    payload = {
        "model": "local",
        "messages": [
            {"role": "system", "content": (
                "You are a thinking-depth classifier. Given a user input, respond with a single "
                "integer from 0 to 10 representing how much deep reasoning or complex thinking is "
                "required. 0 = trivial (greetings, simple facts). 10 = very complex (multi-step "
                "reasoning, math proofs, deep research, complex coding). Respond with the number only."
            )},
            {"role": "user", "content": "Hi, how are you?"}, {"role": "assistant", "content": "0"},
            {"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "1"},
            {"role": "user", "content": "Explain what machine learning is."}, {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Research the economic impacts of AI and summarize them in bullet points."}, {"role": "assistant", "content": "7"},
            {"role": "user", "content": "Prove that there are infinitely many prime numbers and explain each step."}, {"role": "assistant", "content": "10"},
            {"role": "user", "content": text},
        ],
        "grammar": _THINK_GRAMMAR, "temperature": 0.0, "max_tokens": 4, "stream": False,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(9999999.0)) as client:
        response = await client.post("http://localhost:2001/v1/chat/completions", json=payload)
        response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"].strip()
    return int(content)

async def _think_instruction(text: str) -> tuple[int, str]:
    try:
        depth = await _verify_think(text)
    except Exception as e:
        log.warning(f"_verify_think falhou, usando depth padrão (5): {e}")
        depth = 5
    return depth, THINK_DEPTH_INSTRUCTIONS.get(depth, THINK_DEPTH_INSTRUCTIONS[5])


# ══════════════════════════════════════════════════════════════════════════════
# TOOL REGISTRY — cada tool tem: descrição, schema de argumentos (→ JSON
# Schema convertido para o formato OpenAI tools/function-calling) e um
# executor async(args, req) -> Any
# ══════════════════════════════════════════════════════════════════════════════

_WRITE_HINT = "Use this when the user asks you to remember/record/save something."

async def _tool_memory_read(args: dict, req: ExecuteRequest):
    r = await state.memory_client.post("/read", json={
        "query": args.get("query", ""), "top_k": DEFAULT_TOP_K,
        "min_score": DEFAULT_MIN_SCORE, "session_id": req.session_id, "strategy": "auto",
    })
    r.raise_for_status()
    return r.json().get("results", [])

async def _tool_memory_write(args: dict, req: ExecuteRequest):
    r = await state.memory_client.post("/write", json={
        "text": args.get("text", ""), "source": "orchestrator", "confidence": 1.0,
    })
    r.raise_for_status()
    return r.json()

async def _tool_search(args: dict, req: ExecuteRequest):
    r = await state.search_client.post("/search", json={
        "query": args.get("query", ""), "max_results": DEFAULT_TOP_K,
        "use_cache": True, "search_pdfs": req.search_pdfs,
    })
    r.raise_for_status()
    return r.json().get("results", [])

async def _tool_deep_search(args: dict, req: ExecuteRequest):
    r = await state.deep_search_client.post("/query", json={"text": args.get("query", "")})
    r.raise_for_status()
    d = r.json()
    return d.get("answer") or str(d)


# ── Tradução de Objetos: vision.py (crops) -> garante modo multimodal
#    no gerenciador de processos (porta 9001) -> LLM.py (multi-imagem) ──

async def _get_process_manager_status() -> dict:
    r = await state.process_manager_client.get("/status")
    r.raise_for_status()
    return r.json()


async def _ensure_llama_mode(mode: str, timeout: float = 60.0, poll_interval: float = 1.0) -> None:
    """
    Garante que o llama-server gerenciado pelo tray (porta 9001) esteja
    carregado no modo `mode` ("text" ou "multimodal").

    Consulta GET /status; se `llama.mode` já for o modo pedido e o processo
    estiver "Ativo", não faz nada (evita reiniciar o servidor à toa a cada
    chamada). Caso contrário, dispara POST /llama/switch_mode e espera
    (polling) o processo voltar a ficar Ativo no novo modo antes de
    retornar — a troca de modo reinicia o llama-server, então leva alguns
    segundos até o modelo terminar de carregar.
    """
    try:
        status = await _get_process_manager_status()
    except Exception as e:
        raise RuntimeError(
            f"Não foi possível consultar o gerenciador de processos ({PROCESS_MANAGER_URL}/status): {e}"
        )

    llama_status = status.get("llama", {})
    if llama_status.get("mode") == mode and llama_status.get("status") == "Ativo":
        log.info(f"llama-server já está em modo {mode}, nenhuma troca necessária.")
        return

    log.info(
        f"llama-server em modo '{llama_status.get('mode')}' "
        f"(status: {llama_status.get('status')}) — trocando para {mode}..."
    )
    r = await state.process_manager_client.post("/llama/switch_mode", json={"mode": mode})
    r.raise_for_status()
    body = r.json()
    if not body.get("ok", False):
        raise RuntimeError(f"Falha ao trocar o llama-server para modo {mode}: {body.get('message')}")

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(poll_interval)
        try:
            status = await _get_process_manager_status()
        except Exception:
            continue
        llama_status = status.get("llama", {})
        if llama_status.get("mode") == mode and llama_status.get("status") == "Ativo":
            log.info(f"llama-server em modo {mode} e pronto.")
            return

    raise RuntimeError(
        f"Timeout ({timeout}s) esperando o llama-server ficar pronto em modo {mode}."
    )


async def _ensure_multimodal_mode(timeout: float = 60.0, poll_interval: float = 1.0) -> None:
    """Garante o llama-server em modo multimodal (usado antes de enviar crops/imagens)."""
    await _ensure_llama_mode("multimodal", timeout=timeout, poll_interval=poll_interval)


async def _ensure_text_mode(timeout: float = 60.0, poll_interval: float = 1.0) -> None:
    """Garante o llama-server de volta em modo texto (usado ao final de uma requisição de visão)."""
    await _ensure_llama_mode("text", timeout=timeout, poll_interval=poll_interval)


async def _describe_crops_with_llm(objects: list[dict], user_prompt: str) -> str:
    """
    Manda todos os crops detectados numa única chamada multi-imagem para
    o LLM multimodal (LLM.py, via llama-server já trocado para o modelo
    multimodal por _ensure_multimodal_mode). Cada crop vem acompanhado do
    seu índice e, se houver, do melhor candidato já recuperado do
    dicionário visual (memory.py) — o LLM usa isso como contexto e pode
    confirmar ou corrigir o candidato.
    """
    content: list[dict] = [{
        "type": "text",
        "text": (
            f"{user_prompt}\n\n"
            f"Você recebeu {len(objects)} recorte(s) (crops) de objetos detectados numa "
            "cena, na mesma ordem em que aparecem abaixo (a primeira imagem é o objeto "
            "de índice 0, a segunda o de índice 1, e assim por diante). Para cada um, "
            "diga o que é o objeto. Alguns já têm um candidato de significado vindo de "
            "um dicionário visual — use isso como contexto, mas corrija se a imagem "
            "mostrar algo diferente do candidato."
        ),
    }]
    for obj in objects:
        candidates = obj.get("candidates") or []
        if candidates:
            top = candidates[0]
            cand_txt = f" Candidato do dicionário: {top['concept_name']} (score {top['score']:.2f})."
        else:
            cand_txt = " Sem candidato no dicionário (objeto novo ou ambíguo)."
        content.append({"type": "text", "text": f"Objeto {obj['object_index']}:{cand_txt}"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{obj['crop_base64']}"},
        })

    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
        "stream": False,
    }
    r = await state.llm_client.post("/v1/chat/completions", json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"].get("content", "")


async def _tool_vision_objects(args: dict, req: ExecuteRequest):
    """
    Roda o pipeline de "Tradução de Objetos" (vision.py /vision/process) na
    imagem anexada ao request: depth estimation + segmentação + dicionário
    visual, retornando um crop por objeto detectado. Em seguida garante
    que o llama-server esteja em modo multimodal (trocando via o
    gerenciador de processos, porta 9001, se necessário) e manda todos os
    crops numa única chamada multi-imagem para o LLM final descrever/
    identificar cada objeto.
    """
    if not req.image_path:
        return "[vision_objects: no image was provided with this request]"

    try:
        image_bytes = Path(req.image_path).read_bytes()
    except Exception as e:
        return f"[vision_objects: falha ao ler a imagem em '{req.image_path}': {e}]"
    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    r = await state.vision_client.post("/vision/process", json={
        "image_base64": image_b64, "top_k": DEFAULT_TOP_K, "min_score": DEFAULT_MIN_SCORE,
    })
    r.raise_for_status()
    data = r.json()

    objects = data.get("objects", [])
    if not objects:
        return {
            "total_detected": 0,
            "image_width": data.get("image_width"), "image_height": data.get("image_height"),
            "description": "Nenhum objeto foi detectado na imagem.",
        }

    await _ensure_multimodal_mode()

    prompt = args.get("prompt", req.input)
    description = await _describe_crops_with_llm(objects, prompt)

    # Ao final da requisição de visão, volta o llama-server para o modo
    # texto comum — o restante do loop de tool-calling (e o resto da
    # conversa) usa o modelo de texto, então não faz sentido deixá-lo
    # carregado em multimodal. Uma falha aqui não deve derrubar o
    # resultado já obtido da vision, então só logamos o erro.
    try:
        await _ensure_text_mode()
    except Exception as e:
        log.warning(f"Falha ao voltar o llama-server para modo texto após vision_objects: {e}")

    # Remove os crops (base64 pesado) do retorno que volta pro histórico do
    # LLM de texto — já foram consumidos pela chamada multimodal acima;
    # o LLM de texto só precisa dos metadados + da descrição já gerada.
    lean_objects = [
        {
            "object_index": o["object_index"], "bbox": o["bbox"],
            "candidates": o.get("candidates", []), "ambiguous": o.get("ambiguous", True),
        }
        for o in objects
    ]

    return {
        "total_detected": data.get("total_detected", len(objects)),
        "image_width": data.get("image_width"), "image_height": data.get("image_height"),
        "objects": lean_objects, "description": description,
    }

async def _tool_local_scraping(args: dict, req: ExecuteRequest):
    query = args.get("query", "")
    r = await state.local_scraping_client.post("/scrape", json={
        "query": query, "search_path": None, "force_reindex": False, "session_id": req.session_id,
    })
    r.raise_for_status()
    data = r.json()

    if data.get("multiple_matches"):
        return {
            "status": "multiple_matches", "matches": data["matches"],
            "message": data.get("message", "Múltiplos arquivos encontrados. Escolha qual deseja ler."),
            "requires_choice": True,
        }

    file_content = data.get("content", "")
    file_path = data.get("file_path", "")
    was_reindexed = data.get("was_reindexed", False)
    hash_match = data.get("hash_match", True)

    if file_content and file_path:
        try:
            await state.memory_client.post("/indexed-file/write", json={
                "file_path": file_path, "file_name": Path(file_path).name,
                "extension": Path(file_path).suffix.lower(), "content": file_content,
                "file_hash": data.get("file_hash"), "size": len(file_content),
                "modified": data.get("modified", ""), "source": "local_scraping",
                "confidence": 1.0 if hash_match else 0.9, "force_reindex": False,
            })
        except Exception as e:
            log.warning(f"Falha ao salvar arquivo indexado na memória: {e}")

    return {
        "status": "success", "content": file_content, "file_path": file_path,
        "was_reindexed": was_reindexed, "hash_match": hash_match, "requires_choice": False,
    }

async def _tool_alpha_code(args: dict, req: ExecuteRequest):
    payload = {
        "task": args.get("task", ""), "session_id": req.session_id, "project_dir": None,
        "max_steps": 25, "temperature": 0.3, "streaming": False,
    }
    r = await state.alpha_code_client.post("/run", json=payload)
    r.raise_for_status()
    data = r.json()
    return {
        "status": "success" if data.get("success") else "failed",
        "answer": data.get("answer", ""), "files_changed": data.get("files_changed", []),
        "steps_executed": data.get("steps_executed", 0), "tools_called": data.get("tools_called", 0),
        "session_id": data.get("session_id", ""),
    }


# name -> (description, [(field, type, required)], async executor)
TOOLS: dict[str, dict[str, Any]] = {
    "memory_read": {
        "description": f"Retrieves relevant information saved in long/short-term memory.",
        "fields": [("query", "string", True)],
        "executor": _tool_memory_read,
    },
    "memory_write": {
        "description": f"Saves a piece of information to long-term memory. {_WRITE_HINT}",
        "fields": [("text", "string", True)],
        "executor": _tool_memory_write,
    },
    "search": {
        "description": "Searches the web for current/general information.",
        "fields": [("query", "string", True)],
        "executor": _tool_search,
    },
    "deep_search": {
        "description": "Deep research (knowledge-RAG) with automatic web research for complex questions.",
        "fields": [("query", "string", True)],
        "executor": _tool_deep_search,
    },
    "vision_objects": {
        "description": (
            "Detects and identifies individual objects in the image attached to the current "
            "request, using depth-based segmentation and a visual dictionary, then asks the "
            "multimodal model to respond based on each detected object. Only works if an image was sent."
        ),
        "fields": [("prompt", "string", False)],
        "executor": _tool_vision_objects,
    },
    "local_scraping": {
        "description": "Searches for and reads a local file on the user's machine.",
        "fields": [("query", "string", True)],
        "executor": _tool_local_scraping,
    },
    "alpha_code": {
        "description": "Autonomous code generation/editing agent for a project.",
        "fields": [("task", "string", True)],
        "executor": _tool_alpha_code,
    },
    "finish": {
        "description": "Ends the cycle and delivers the final response to the user. Call this tool once you already have everything you need (or if no other tool is required).",
        "fields": [("response", "string", True)],
        "executor": None,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# Tools schema (formato OpenAI function-calling) — usado pelo llama-server
# em modo `--jinja`, que renderiza as definições na chat template do próprio
# modelo e aplica a grammar de tool-calling automaticamente no lado servidor.
# ══════════════════════════════════════════════════════════════════════════════

_JSON_SCHEMA_TYPES = {"string": "string", "number": "number"}

def _build_tools_schema() -> list[dict]:
    """Converte o TOOLS registry para a lista `tools` no formato OpenAI."""
    schema = []
    for name, spec in TOOLS.items():
        properties = {}
        required = []
        for field_name, field_type, field_required in spec["fields"]:
            properties[field_name] = {"type": _JSON_SCHEMA_TYPES.get(field_type, "string")}
            if field_required:
                required.append(field_name)
        schema.append({
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": {
                    "type": "object", "properties": properties, "required": required,
                },
            },
        })
    return schema


# ══════════════════════════════════════════════════════════════════════════════
# LLM call helper (tool-calling nativo)
# ══════════════════════════════════════════════════════════════════════════════

async def _llm_chat(messages: list[dict], tools: Optional[list[dict]] = None,
                     temperature: float = 0.0, max_tokens: Optional[int] = None) -> dict:
    """Chama o endpoint de chat e retorna a mensagem completa (content + tool_calls)."""
    payload = {
        "model": "local", "messages": messages, "temperature": temperature, "stream": False,
    }
    if tools:
        payload["tools"] = tools
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    r = await state.llm_client.post("/v1/chat/completions", json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def _tools_system_prompt(think_instruction: Optional[str] = None) -> str:
    lines = [
        "You are AVA, a helpful assistant. You have access to a set of tools/functions.",
        "",
        "Call a tool whenever you need information or need to take an action. You may also "
        "respond with plain text — reasoning, a plan, a clarifying remark — without calling "
        "a tool; in that case you will simply be prompted again on the next turn, so use it "
        "to think out loud if needed.",
        "",
        "To deliver your final answer to the user, call the `finish` tool with the complete "
        "response text as its `response` argument. The task is not done until you call `finish`.",
        "",
        "Available tools:",
    ]
    for name, spec in TOOLS.items():
        lines.append(f"- {name}: {spec['description']}")
    if think_instruction:
        lines.append("")
        lines.append(think_instruction)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Tool-calling loop (substitui router MiniLM + CoT/DAG)
# ══════════════════════════════════════════════════════════════════════════════

async def _run_tool_loop(req: ExecuteRequest, eid: str, sid: str,
                          on_event=None) -> tuple[list[StepResult], str]:
    """
    Fluxo (tool-calling nativo via Jinja):
      1. LLM recebe o histórico + a lista `tools` (formato OpenAI). O servidor
         renderiza isso na chat template do modelo e devolve `message` com
         `content` (texto livre, opcional) e/ou `tool_calls` (já estruturado).
      2. Se NÃO houver tool_calls: o texto vira uma mensagem assistant no
         histórico e o loop simplesmente continua — sem lembrete.
      3. Se houver tool_calls: cada uma é executada e o resultado volta ao
         histórico como mensagem role="tool" (tool_call_id correspondente).
      4. O loop SÓ termina quando a tool "finish" é chamada — não há limite
         de rodadas.
    """
    depth, think_instruction = await _think_instruction(req.input)
    system_prompt = _tools_system_prompt(think_instruction)
    tools_schema = _build_tools_schema()

    history: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": req.input},
    ]

    step_results: list[StepResult] = []
    final_response = ""
    turn = 0

    while True:
        # ── Passo 1: LLM responde, podendo incluir tool_calls estruturadas ──
        message = await _llm_chat(history, tools=tools_schema, max_tokens=2048, temperature=0.3)
        tool_calls = message.get("tool_calls") or []

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        history.append(assistant_msg)

        if not tool_calls:
            # Nenhuma tool chamada — apenas segue o loop, sem lembrete
            log.debug(f"[{eid[:8]}] Turn {turn}: resposta sem tool_calls, continuando o loop")
            turn += 1
            continue

        finished = False
        for call in tool_calls:
            fn = call.get("function", {}) or {}
            tool_name = fn.get("name", "")
            call_id = call.get("id", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}

            if tool_name not in TOOLS:
                log.warning(f"[{eid[:8]}] Tool desconhecida '{tool_name}'")
                history.append({"role": "tool", "tool_call_id": call_id,
                                 "content": f"ERROR: tool '{tool_name}' não existe"})
                continue

            if on_event:
                await on_event("tool_call", {"step": turn, "tool": tool_name})

            # ── Se a tool for "finish", extrai a resposta final dos argumentos ──
            if tool_name == "finish":
                final_response = args.get("response", "") or "(sem resposta)"
                step_results.append(StepResult(
                    step=turn, executor="finish",
                    action=json.dumps(args, ensure_ascii=False),
                    success=True, result=final_response,
                ))
                finished = True
                break

            # ── Executa a tool ──
            spec = TOOLS[tool_name]
            t0 = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    spec["executor"](args, req),
                    timeout=EXECUTOR_TIMEOUTS.get(tool_name, 60.0),
                )
                success, err = True, None
            except Exception as e:
                result, success, err = None, False, f"{type(e).__name__}: {e}"
            lat = round((time.perf_counter() - t0) * 1000, 2)

            sr = StepResult(
                step=turn, executor=tool_name,
                action=json.dumps(args, ensure_ascii=False),
                success=success, result=result, error=err, latency_ms=lat,
            )
            step_results.append(sr)
            if on_event:
                await on_event("tool_result", sr.model_dump())

            # ── Resultado no histórico como mensagem role="tool" ──
            history.append({
                "role": "tool", "tool_call_id": call_id,
                "content": _result_to_text(result) if success else f"ERROR: {err}",
            })

        if finished:
            break
        turn += 1

    return step_results, final_response


# ══════════════════════════════════════════════════════════════════════════════
# SSE generator para /execute
# ══════════════════════════════════════════════════════════════════════════════

async def _execute_stream_generator(req: ExecuteRequest):
    t0 = time.perf_counter()
    eid = str(uuid.uuid4())
    sid = req.session_id or str(uuid.uuid4())
    log.info(f"[{eid[:8]}] Executando (tool loop): '{req.input[:80]}'")

    yield _sse("meta", {"execution_id": eid, "session_id": sid})

    events_queue: asyncio.Queue = asyncio.Queue()

    async def on_event(name, data):
        await events_queue.put((name, data))

    async def run():
        try:
            sr, fr = await _run_tool_loop(req, eid, sid, on_event=on_event)
            await events_queue.put(("__done__", (sr, fr)))
        except Exception as e:
            log.exception(f"[{eid[:8]}] erro no tool loop")
            await events_queue.put(("__error__", str(e)))

    task = asyncio.create_task(run())

    step_results: list[StepResult] = []
    final_response = ""
    error_msg: Optional[str] = None

    while True:
        name, data = await events_queue.get()
        if name == "__done__":
            step_results, final_response = data
            break
        if name == "__error__":
            error_msg = data
            yield _sse("error", {"error": error_msg})
            break
        if name == "tool_call":
            yield _sse("tool_call", data)
        elif name == "tool_result":
            yield _sse("step_done", data)
            res = data.get("result")
            if data.get("success") and res:
                yield _sse("result", _result_to_text(res))

    await task

    if final_response:
        yield _sse("delta", final_response)

    if req.tts and final_response:
        asyncio.create_task(_fire_tts(final_response, req))

    if final_response:
        asyncio.create_task(_save_turn(req.input, final_response, sid))
        asyncio.create_task(_save_lt(f"Usuário disse: {req.input[:200]}"))

    lat = round((time.perf_counter() - t0) * 1000, 2)
    errors = [f"Step {s.step} [{s.executor}]: {s.error}" for s in step_results if not s.success]
    if error_msg:
        errors.append(error_msg)

    yield _sse("done", {
        "execution_id": eid, "final_response": final_response,
        "steps": [s.model_dump() for s in step_results],
        "total_latency_ms": lat, "errors": errors,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Alpha-code direct streaming (endpoint separado — não passa pelo tool loop)
# ══════════════════════════════════════════════════════════════════════════════

async def _alpha_code_stream_generator(req: AlphaCodeRequest):
    t0 = time.perf_counter()
    eid = str(uuid.uuid4())
    sid = req.session_id or str(uuid.uuid4())

    log.info(f"[alpha_code:{eid[:8]}] Iniciando tarefa: '{req.task[:80]}'")

    yield _sse("meta", {"execution_id": eid, "session_id": sid, "route": "alpha_code", "routed_directly": True})

    payload = {
        "task": req.task, "session_id": sid, "project_dir": req.project_dir,
        "max_steps": req.max_steps, "temperature": req.temperature,
        "model_override": req.model_override, "streaming": True,
    }

    final_answer = ""
    files_changed: list[str] = []
    tools_called = 0
    steps_executed = 0
    tokens_used = 0
    error_msg: Optional[str] = None

    try:
        async with state.alpha_code_client.stream("POST", "/run/stream", json=payload,
                                                    timeout=EXECUTOR_TIMEOUTS["alpha_code"]) as resp:
            if resp.status_code != 200:
                body = ""
                async for chunk in resp.aiter_text():
                    body += chunk
                error_msg = f"alpha_code HTTP {resp.status_code}: {body[:300]}"
                yield _sse("error", {"error": error_msg})
            else:
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    try:
                        evt = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue

                    ev_type = evt.get("event", "")
                    ev_data = evt.get("data", {}) or {}
                    ev_step = evt.get("step")

                    if ev_type == "plan":
                        yield _sse("plan", ev_data)
                    elif ev_type == "thinking":
                        yield _sse("reasoning", ev_data.get("text", ""))
                    elif ev_type == "tool_call":
                        tools_called += 1
                        yield _sse("tool_call", {"step": ev_step, "tool": ev_data.get("name", ""), "arguments": ev_data.get("arguments", {})})
                        yield _sse("step_start", {"step": ev_step or tools_called, "executor": "alpha_code",
                                                    "action": f"{ev_data.get('name', '')}({json.dumps(ev_data.get('arguments', {}), ensure_ascii=False)[:120]})"})
                    elif ev_type == "tool_result":
                        success = ev_data.get("success", False)
                        out = ev_data.get("output", "")
                        err = ev_data.get("error", "")
                        yield _sse("step_done", {"step": ev_step or tools_called, "executor": "alpha_code",
                                                   "success": success, "latency_ms": ev_data.get("elapsed_ms", 0), "error": err})
                        yield _sse("result", f"❌ {err[:500]}" if (not success and err) else (out[:2000] if out else "(no output)"))
                    elif ev_type == "model_choice":
                        yield _sse("model_choice", {"step": ev_step, "model": ev_data.get("model", ""),
                                                      "reasoning_effort": ev_data.get("reasoning_effort"),
                                                      "temperature": ev_data.get("temperature", 0.3),
                                                      "step_kind": ev_data.get("step_kind", "")})
                    elif ev_type == "context_budget":
                        tokens_used = ev_data.get("tokens_used", tokens_used)
                        yield _sse("context_budget", ev_data)
                    elif ev_type == "error":
                        fatal = ev_data.get("fatal", False)
                        error_msg = ev_data.get("error", "unknown error")
                        yield _sse("error", {"error": error_msg, "fatal": fatal})
                        if fatal:
                            break
                    elif ev_type == "final":
                        final_answer = ev_data.get("answer", "")
                        files_changed = ev_data.get("files_changed", []) or []
                        tools_called = ev_data.get("tools_called", tools_called)
                        steps_executed = ev_data.get("steps_executed", steps_executed)
                        tokens_used = ev_data.get("tokens_used", tokens_used)
                        if final_answer:
                            yield _sse("delta", final_answer)
                        break

    except httpx.ConnectError as e:
        error_msg = f"alpha_code offline: {e}"
        yield _sse("error", {"error": error_msg, "fatal": True})
    except httpx.ReadTimeout as e:
        error_msg = f"alpha_code timeout: {e}"
        yield _sse("error", {"error": error_msg, "fatal": True})
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        log.exception("[alpha_code] erro inesperado")
        yield _sse("error", {"error": error_msg, "fatal": True})

    lat = round((time.perf_counter() - t0) * 1000, 2)
    yield _sse("done", {
        "execution_id": eid, "session_id": sid,
        "final_response": final_answer or (f"Erro: {error_msg}" if error_msg else "(no answer)"),
        "steps_executed": steps_executed, "tools_called": tools_called, "tokens_used": tokens_used,
        "files_changed": files_changed, "total_latency_ms": lat,
        "errors": [error_msg] if error_msg else [], "route": "alpha_code", "routed_directly": True,
    })


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI Application Endpoints
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="AVA Orchestrator", version="4.0.0",
              description="Tool-Calling Execution Engine (sem router MiniLM, sem CoT/DAG)",
              lifespan=lifespan)

@app.post("/execute")
async def execute(req: ExecuteRequest):
    async def stream_with_flush():
        async for chunk in _execute_stream_generator(req):
            yield chunk

    return StreamingResponse(
        stream_with_flush(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no", "Transfer-Encoding": "chunked"},
    )


@app.post("/code")
async def code_endpoint(req: AlphaCodeRequest):
    """Chama o agente alpha_code diretamente via SSE — não passa pelo tool loop do /execute."""
    if not req.task.strip():
        raise HTTPException(400, "task vazio")

    async def stream_alpha_code():
        async for chunk in _alpha_code_stream_generator(req):
            yield chunk

    return StreamingResponse(
        stream_alpha_code(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no", "Transfer-Encoding": "chunked"},
    )


@app.post("/code/sync")
async def code_sync_endpoint(req: AlphaCodeRequest):
    if not req.task.strip():
        raise HTTPException(400, "task vazio")
    payload = {
        "task": req.task, "session_id": req.session_id, "project_dir": req.project_dir,
        "max_steps": req.max_steps, "temperature": req.temperature,
        "model_override": req.model_override, "streaming": False,
    }
    try:
        r = await state.alpha_code_client.post("/run", json=payload, timeout=EXECUTOR_TIMEOUTS["alpha_code"])
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError as e:
        raise HTTPException(502, f"alpha_code offline: {e}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, f"alpha_code: {e.response.text[:500]}")
    except Exception as e:
        raise HTTPException(500, f"alpha_code falhou: {e}")


@app.post("/deep-search")
async def deep_search(req: DeepSearchRequest):
    if not req.text.strip():
        raise HTTPException(400, "text vazio")
    try:
        r = await state.deep_search_client.post("/query", json={"text": req.text})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(502, f"Deep Search falhou: {e}")

@app.post("/memory/read")
async def memory_read(req: MemoryReadRequest):
    try:
        r = await state.memory_client.post("/read", json={"query": req.query, "top_k": req.top_k, "min_score": req.min_score, "session_id": req.session_id, "strategy": "auto"})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(502, f"Memory falhou: {e}")

@app.post("/memory/write")
async def memory_write(req: MemoryWriteRequest):
    try:
        r = await state.memory_client.post("/write", json={"text": req.text, "source": req.source, "confidence": req.confidence})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(502, f"Memory falhou: {e}")

@app.post("/search")
async def search(query: str, max_results: int = 5, search_pdfs: bool = False):
    try:
        r = await state.search_client.post("/search", json={"query": query, "max_results": max_results, "use_cache": True, "search_pdfs": search_pdfs})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(502, f"Search falhou: {e}")

@app.post("/chat")
async def chat(message: str, voice: str = "M1", lang: str = "pt", tts: bool = True):
    try:
        r = await state.llm_client.post("/chat", json={"message": message, "voice": voice, "lang": lang, "max_turns": 10, "tts": tts})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(502, f"Chat falhou: {e}")


# ── Local Scraping Endpoints ────────────────────────────────────────────────

@app.post("/local-scraping")
async def local_scraping(req: LocalScrapingRequest):
    if not req.query.strip():
        raise HTTPException(400, "query vazio")
    try:
        r = await state.local_scraping_client.post("/scrape", json={
            "query": req.query, "search_path": req.search_path,
            "force_reindex": req.force_reindex, "session_id": req.session_id,
        })
        r.raise_for_status()
        data = r.json()

        if data.get("multiple_matches"):
            return data

        file_content = data.get("content", "")
        file_path = data.get("file_path", "")
        hash_match = data.get("hash_match", True)

        if file_content and file_path:
            try:
                summary = file_content[:500] if len(file_content) > 500 else file_content
                await state.memory_client.post("/write", json={
                    "text": f"[ARQUIVO INDEXADO] {file_path}: {summary}",
                    "source": "local_scraping", "confidence": 1.0 if hash_match else 0.9,
                })
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
    if not req.file_path.strip():
        raise HTTPException(400, "file_path vazio")
    try:
        r = await state.local_scraping_client.post("/choose", json={
            "query": req.query, "file_path": req.file_path,
            "force_reindex": req.force_reindex, "session_id": req.session_id,
        })
        r.raise_for_status()
        data = r.json()

        file_content = data.get("content", "")
        file_path = data.get("file_path", req.file_path)
        hash_match = data.get("hash_match", True)

        if file_content and file_path:
            try:
                summary = file_content[:500] if len(file_content) > 500 else file_content
                await state.memory_client.post("/write", json={
                    "text": f"[ARQUIVO INDEXADO] {file_path}: {summary}",
                    "source": "local_scraping", "confidence": 1.0 if hash_match else 0.9,
                })
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
    try:
        params = {"file_path": file_path} if file_path else {}
        r = await state.local_scraping_client.get("/indexed", params=params)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        raise HTTPException(502, "Local Scraping serviço offline")
    except Exception as e:
        raise HTTPException(502, f"Local Scraping indexed falhou: {e}")


@app.delete("/local-scraping/index/{file_id}")
async def local_scraping_delete_index(file_id: str):
    try:
        r = await state.local_scraping_client.delete(f"/index/{file_id}")
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        raise HTTPException(502, "Local Scraping serviço offline")
    except Exception as e:
        raise HTTPException(502, f"Local Scraping delete falhou: {e}")


# ── Status / utilitários ─────────────────────────────────────────────────────

@app.get("/status")
async def status():
    checks = {}
    cfg = {
        "memory": (state.memory_client, HEALTH_PATHS["memory"]),
        "search": (state.search_client, HEALTH_PATHS["search"]),
        "local_scraping": (state.local_scraping_client, HEALTH_PATHS["local_scraping"]),
        "tts": (state.tts_client, HEALTH_PATHS["tts"]),
        "llm": (state.llm_client, HEALTH_PATHS["llm"]),
        "vision": (state.vision_client, HEALTH_PATHS["vision"]),
        "deep_search": (state.deep_search_client, HEALTH_PATHS["deep_search"]),
        "alpha_code": (state.alpha_code_client, HEALTH_PATHS["alpha_code"]),
        "process_manager": (state.process_manager_client, HEALTH_PATHS["process_manager"]),
    }
    for n, (c, p) in cfg.items():
        try:
            r = await c.get(p, timeout=2.0)
            checks[n] = {"healthy": r.status_code == 200, "status_code": r.status_code}
        except Exception:
            checks[n] = {"healthy": False, "status_code": None}
    return {"orchestrator": "ok", "architecture": "tool-calling", "services": checks, "tools": list(TOOLS.keys())}

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    try:
        r = await state.memory_client.delete(f"/session/{session_id}", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(502, f"Session clear falhou: {e}")

@app.post("/tts/cancel")
async def cancel_tts():
    try:
        r = await state.tts_client.post("/cancel", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(502, f"TTS cancel falhou: {e}")

@app.get("/tts/voices")
async def list_voices():
    try:
        r = await state.tts_client.get("/voices", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(502, f"TTS voices falhou: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("orchestrator:app", host="0.0.0.0", port=9000, log_level="info")