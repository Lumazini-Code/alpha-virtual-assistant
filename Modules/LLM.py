"""
AVA — LLM Inference API
========================
REST API para inferência conversacional com:
  - Gerenciamento do llama-server (subida/desligamento)
  - Integração correta com API de Memória (LT + ST via session_id)
  - Integração com API de TTS (localhost:3004)
  - Streaming de texto + disparo paralelo de áudio
  - Histórico de chat persistido via módulo de memória externo
  - Detecção de idioma para resposta automática

TTFT OPTIMIZATIONS (v2.1):
  1. Persistent httpx clients — no TCP handshake per request (saves ~50-150ms)
  2. Stable system prompt prefix — enables llama-server prompt caching (saves ~200-500ms)
  3. Memory read parallel with prompt construction (saves ~100-300ms)
  4. Real-length warmup — KV cache pre-allocated for actual context sizes
  5. Connection pooling — keep-alive to llama-server and memory API

Porta: localhost:4003
"""
import sys
import json
import datetime
import asyncio
import time
from pathlib import Path
from typing import Optional
import re
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langdetect import detect
import logging
from dotenv import load_dotenv
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [LLM] %(message)s")
log = logging.getLogger("ava.llm")




load_dotenv()  # Carrega variáveis de ambiente do arquivo .env

# ─────────────────────────────────────────────────────────────
#                          CONFIG
# ─────────────────────────────────────────────────────────────

BASEFOLDER = Path(__file__).parent.parent

# URLs das APIs satélite
MEMORY_URL = "http://localhost:3001"
TTS_URL    = "http://localhost:3004"

# Contexto de curto prazo: quantas duplas pergunta-resposta (turn groups)
# são lidas cruas do /read_st a cada turno, em paralelo com o /read semântico.
ST_CONTEXT_PAIRS = 5

# llama-server
LLAMA_SERVER_PATH = r".\llama-cpp\llama-server"
LLAMA_HOST        = "localhost"
LLAMA_PORT        = 2001
LLAMA_URL         = f"http://{LLAMA_HOST}:{LLAMA_PORT}"

# ─────────────────────────────────────────────────────────────
#          OPTIMIZATION 1: PERSISTENT HTTP CLIENTS
# ─────────────────────────────────────────────────────────────
# Instead of creating a new httpx.AsyncClient per request (which
# costs ~50-150ms for TCP handshake + HTTP/1.1 upgrade), we
# create them once at startup and reuse across all requests.
# This is the SINGLE BIGGEST latency win for TTFT.

_llama_client: httpx.AsyncClient | None = None
_memory_client: httpx.AsyncClient | None = None
_tts_http: httpx.AsyncClient | None = None


async def _get_llama_client() -> httpx.AsyncClient:
    """Persistent client to llama-server — connection pooling + keep-alive."""
    global _llama_client
    if _llama_client is None or _llama_client.is_closed:
        _llama_client = httpx.AsyncClient(
            base_url=LLAMA_URL,
            timeout=httpx.Timeout(9999999.0, connect=5.0),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=6,
                keepalive_expiry=60.0,       # Keep connections warm for 60s
            ),
            headers={"Content-Type": "application/json"},
        )
    return _llama_client




async def _get_memory_client() -> httpx.AsyncClient:
    """Persistent client to memory API — connection pooling + keep-alive."""
    global _memory_client
    if _memory_client is None or _memory_client.is_closed:
        _memory_client = httpx.AsyncClient(
            base_url=MEMORY_URL,
            timeout=httpx.Timeout(9999999.0, connect=3.0),
            limits=httpx.Limits(
                max_connections=6,
                max_keepalive_connections=4,
                keepalive_expiry=60.0,
            ),
        )
    return _memory_client


def _get_tts_client() -> httpx.AsyncClient:
    global _tts_http
    if _tts_http is None or _tts_http.is_closed:
        _tts_http = httpx.AsyncClient(
            base_url=TTS_URL,
            timeout=httpx.Timeout(9999999.0),
            limits=httpx.Limits(max_keepalive_connections=2, max_connections=4),
        )
    return _tts_http


# ─────────────────────────────────────────────────────────────
#                       LOGGING DUPLO
# ─────────────────────────────────────────────────────────────

def _setup_logging():
    log_dir = BASEFOLDER / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"LLM_api_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

    class LogDuplicado:
        def __init__(self, terminal, path):
            self.terminal = terminal
            self.log = open(path, "w", encoding="utf-8")

        def write(self, msg):
            try: self.terminal.write(msg)
            except Exception: pass
            self.log.write(msg)

        def flush(self):
            try: self.terminal.flush()
            except Exception: pass
            self.log.flush()

        def isatty(self): return False

    sys.stdout = LogDuplicado(sys.__stdout__, log_path)
    sys.stderr = LogDuplicado(sys.__stderr__, log_path)

_setup_logging()

# ─────────────────────────────────────────────────────────────
#                     LEITURA DE CONFIGS
# ─────────────────────────────────────────────────────────────

def _read(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

username   = _read(BASEFOLDER / r"resource/username.dll")
voiceModel = _read(BASEFOLDER / r"resource/VoiceModel.dll") or "F1"
ctxUsed    = _read(BASEFOLDER / r"resource/ctxConfig.dll")
context    = _read(BASEFOLDER / f"ctxBin/{ctxUsed}.bin")
searchCfg  = _read(BASEFOLDER / r"resource/SearchCfg.dll")

model_raw = _read(BASEFOLDER / r"resource/Aiconfig.dll")
model_path = re.split(r"[\\/]", model_raw)
MODEL_PATH = model_path[-1]
MODEL_NAME = model_raw

try:
    with open(BASEFOLDER / f"CfgModels/{model_raw}.json", "r", encoding="utf-8") as f:
        MODELCFG = json.load(f)
except FileNotFoundError:
    MODELCFG = {}

import re
import time
import asyncio
import logging
import httpx

log = logging.getLogger("ava.llm")

# ════════════════════════════════════════════════════════════════════════════
# Inferência 100% local via llama-server — um único modelo, sem rate-limit
# tracking por modelo e sem estado de "esgotado".
# ════════════════════════════════════════════════════════════════════════════

def _calc_max_tokens(messages: list, requested: int = 4096) -> int:
    """
    Calcula max_tokens para a resposta. O llama-server local não impõe
    limite prático de tokens por minuto, então isso só evita pedir um
    max_tokens absurdamente alto por engano.
    """
    return min(requested, 8000)


def _reasoning_payload_extra(thinking_depth: int) -> dict:
    """
    Traduz `thinking_depth` (0-10) em parâmetros de reasoning para o
    llama-server. LFM2.5-8B-A1B é um reasoning model — a decisão do time
    (2026-08-05) é priorizar QUALIDADE por padrão, então depth=0 não manda
    nenhum override e deixa o comportamento nativo do template agir
    (raciocínio sem teto).

    Para depth > 0, tentamos limitar via `reasoning_budget_tokens` — mas o
    nome desse campo mudou entre versões recentes do llama.cpp (havia
    `think_budget` via CLI, `thinking_budget_tokens` numa proposta anterior,
    e `reasoning_budget_tokens` é o confirmado como funcional per-request na
    build mais recente checada). Se o seu build não reconhecer o campo, ele
    é ignorado silenciosamente pelo llama-server — sem quebrar o request,
    só sem efeito. Vale conferir contra `--help`/`/props` do seu binário.
    """
    if thinking_depth <= 0:
        return {}
    return {"reasoning_budget_tokens": thinking_depth * 300}


async def _execute_inference(json_payload: dict, stream: bool = False):
    """
    Executa a inferência contra o llama-server local (único backend).
    """
    client = await _get_llama_client()
    messages = json_payload.get("messages", [])
    max_tok = _calc_max_tokens(messages, requested=4096)
    payload = {**json_payload, "model": MODEL_NAME, "stream": stream, "max_tokens": max_tok}

    if stream:
        ctx = client.stream("POST", "/v1/chat/completions", json=payload)
        return client, MODEL_NAME, ctx
    else:
        r = await client.post("/v1/chat/completions", json=payload)
        return client, MODEL_NAME, r


# ─────────────────────────────────────────────────────────────
#          INFERÊNCIA SÍNCRONA COM RETRY (falhas transitórias)
# ─────────────────────────────────────────────────────────────
# Único backend: llama-server local. Apenas um pequeno retry para
# 5xx/erro de rede transitório.

LLAMA_MAX_RETRIES = 2
LLAMA_RETRY_BACKOFF_S = 1.0


async def _llama_post_with_retry(
    json_payload: dict,
    stream: bool = False,
):
    """
    Executa POST no /v1/chat/completions do llama-server local, com um
    pequeno retry em caso de 5xx ou erro de rede transitório.

    Retorna tupla: (client, model_name, response_obj).
    """
    client = await _get_llama_client()
    messages = json_payload.get("messages", [])
    max_tok = _calc_max_tokens(messages, requested=4096)
    payload = {**json_payload, "model": MODEL_NAME, "stream": stream, "max_tokens": max_tok}

    if stream:
        # No streaming, devolvemos o context manager — 5xx/erro de rede
        # dentro do stream é tratado por quem consome (ver /chat/stream).
        ctx = client.stream("POST", "/v1/chat/completions", json=payload)
        return client, MODEL_NAME, ctx

    last_error: Optional[Exception] = None
    for attempt in range(LLAMA_MAX_RETRIES + 1):
        try:
            r = await client.post("/v1/chat/completions", json=payload)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            last_error = e
            log.warning(f"llama-server: erro de rede ({type(e).__name__}), tentativa {attempt + 1}/{LLAMA_MAX_RETRIES + 1}.")
            if attempt < LLAMA_MAX_RETRIES:
                await asyncio.sleep(LLAMA_RETRY_BACKOFF_S)
                continue
            raise RuntimeError(f"llama-server inacessível em {LLAMA_URL}: {e}") from e

        if r.status_code >= 500 and attempt < LLAMA_MAX_RETRIES:
            log.warning(f"llama-server: {r.status_code} (tentativa {attempt + 1}/{LLAMA_MAX_RETRIES + 1}).")
            await asyncio.sleep(LLAMA_RETRY_BACKOFF_S)
            continue

        return client, MODEL_NAME, r

    raise RuntimeError(f"llama-server: falhou após {LLAMA_MAX_RETRIES + 1} tentativa(s): {last_error}")


# ─────────────────────────────────────────────────────────────
#          OPTIMIZATION 2: STABLE SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
# The original code rebuilds the system prompt EVERY request with:
#   - Full context string
#   - Memory block (changes every request)
#   - Current timestamp (changes every second)
#   - Username + language instruction
#
# This DESTROYS prompt caching because llama-server's --cache-prompt
# works by matching the prompt PREFIX. If the prefix changes even one
# character, the entire KV cache is invalidated.
#
# FIX: Split into:
#   1. STABLE system prompt (context + username + instructions)
#      → This part is cached by llama-server after the first request
#   2. DYNAMIC memory block expressed in the ASSISTANT voice
#      → Model "recalls" its own knowledge instead of receiving external data
#   3. Timestamp moved to a user message (not in the system prompt)
#
# MEMORY SEMANTICS FIX (v2.2):
# Injecting memories as role:user causes the model to treat them as
# external data provided by the user, not as its own knowledge.
# The fix is two-pronged:
#   a) The system prompt explicitly explains the memory injection mechanism
#      so the model understands the semantic contract.
#   b) The memories are expressed in the ASSISTANT voice (role:assistant),
#      as a first-person recall, so the model "owns" that knowledge.
#
# CACHE IMPACT: The stable prefix [system: _SYSTEM_PROMPT_BASE] is always
# the same → KV cache HIT. The assistant memory recall turn varies but is
# small and comes AFTER the cached prefix.

_SYSTEM_PROMPT_BASE = (
    f"{context}\n\n"
    f"O nome do usuário é {username}. "
    f"Responda sempre no idioma em que o usuário escrever.\n\n"
    # ── MEMORY SEMANTICS INSTRUCTION ─────────────────────────────────────────
    # This tells the model HOW to interpret the injected memory block.
    # Without this, the model may treat the assistant recall turn as a
    # previous response rather than as retrieved self-knowledge.
    "Você possui um sistema de memória persistente. Antes de cada resposta, "
    "fragmentos relevantes da sua memória de longo prazo e do histórico recente "
    "são recuperados e apresentados em uma mensagem sua anterior nesta conversa. "
    "Trate essas informações como conhecimento próprio e utilize-as naturalmente, "
    "sem mencionar explicitamente que são 'memórias recuperadas' ou citar o "
    "mecanismo de memória ao usuário."
)


async def _warmup():
    """
    Warmup with a REPRESENTATIVE prompt — not just "ok".
    This pre-allocates the KV cache for the actual context sizes we use,
    so the first real request doesn't pay the allocation cost.
    """
    log.info("[LLM] Warmup do modelo (representative prompt)...")
    try:
        client = await _get_llama_client()

        # Send a warmup request that's similar in structure to real requests
        # This allocates KV cache for the system prompt + a user message
        warmup_messages = [
            {"role": "system", "content": _SYSTEM_PROMPT_BASE},
            {"role": "user", "content": "ok"},
        ]

        r = await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": warmup_messages,
                "max_tokens": 1,
                "temperature": 0.1,
            },
        )

        if r.status_code == 200:
            # Check if prompt was cached
            usage = r.json().get("usage", {})
            cached_tokens = usage.get("prompt_tokens_cached", 0)
            log.info(f"[LLM] Warmup concluído. Cached tokens: {cached_tokens}")
        else:
            log.info(f"[LLM] Warmup response: {r.status_code}")

    except Exception as e:
        log.info(f"[LLM] Warmup falhou (não crítico): {e}")


# ─────────────────────────────────────────────────────────────
#                     INTEGRAÇÃO: MEMÓRIA
# ─────────────────────────────────────────────────────────────

async def memory_read(query: str, session_id: Optional[str] = None, top_k: int = 10) -> list[dict]:
    """
    Busca memórias relevantes (Long-Term e Short-Term) para o contexto da conversa.
    OPTIMIZED: Uses persistent HTTP client — no TCP handshake per call.
    """
    try:
        client = await _get_memory_client()
        r = await client.post(
            "/read",
            json={
                "query": query,
                "top_k": top_k,
                "min_score": 0.3,
                "session_id": session_id,
                "strategy": "auto",
            },
        )
        return r.json().get("results", [])
    except Exception as e:
        log.info(f"[MEMORY] Falha na leitura: {e}")
        return []


async def memory_read_st(session_id: Optional[str], n_pairs: int = ST_CONTEXT_PAIRS) -> list[dict]:
    """
    Lê as N duplas pergunta-resposta mais recentes do short-term (histórico
    cru da conversa, sem busca semântica) via /read_st. Roda em paralelo com
    o /read semântico — juntos formam o contexto completo enviado ao modelo.
    """
    if not session_id:
        return []
    try:
        client = await _get_memory_client()
        r = await client.post(
            "/read_st",
            json={"session_id": session_id, "n_pairs": n_pairs},
        )
        return r.json().get("turns", [])
    except Exception as e:
        log.info(f"[MEMORY] Falha na leitura ST (read_st): {e}")
        return []


async def memory_save_turn(session_id: str, user_input: str, assistant_response: str):
    """Grava o par de turnos na memória de curto prazo — fire-and-forget."""
    if not session_id:
        return
    try:
        client = await _get_memory_client()
        await client.post(
            "/write_st",
            json={
                "session_id": session_id,
                "turns": [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": assistant_response}
                ]
            },
        )
    except Exception as e:
        log.info(f"[MEMORY] Falha ao salvar turno ST: {e}")


async def memory_write_fact(text: str, source: str = "chat", confidence: float = 0.7):
    """Grava informações na memória de longo prazo — fire-and-forget."""
    try:
        client = await _get_memory_client()
        await client.post(
            "/write",
            json={"text": text, "source": source, "confidence": confidence},
        )
    except Exception as e:
        log.info(f"[MEMORY] Falha na escrita LT: {e}")


# ─────────────────────────────────────────────────────────────
#                      INTEGRAÇÃO: TTS
# ─────────────────────────────────────────────────────────────

_tts_queue: Optional[asyncio.Queue] = None

async def _tts_sender_worker():
    """
    Worker em background que envia sentenças para o TTS SEQUENCIALMENTE.
    OPTIMIZED: Uses persistent HTTP client.
    """
    tts_client = _get_tts_client()
    while True:
        text, voice, lang = await _tts_queue.get()
        try:
            r = await tts_client.post(
                "/stream",
                json={"text": text, "voice": voice, "lang": lang},
            )
            if r.status_code >= 400:
                log.info(f"[TTS] Chunk rejeitado ({r.status_code}): '{text[:60]}'")
        except Exception as e:
            log.info(f"[TTS] Falha no chunk: {type(e).__name__}: {e}")
        finally:
            _tts_queue.task_done()


def _ensure_tts_queue():
    global _tts_queue
    if _tts_queue is None:
        _tts_queue = asyncio.Queue()
        asyncio.create_task(_tts_sender_worker())


async def tts_speak(text: str, voice: str, lang: str):
    _ensure_tts_queue()
    await _tts_queue.put((text[:2000], voice, lang))


# ─────────────────────────────────────────────────────────────
#          OPTIMIZATION 2 (cont): PROMPT CONSTRUCTION
# ─────────────────────────────────────────────────────────────

def _build_memory_recall(memories: list[dict]) -> str | None:
    """
    Formata o bloco de memórias como texto de recall em primeira pessoa.

    Returns None se não houver memórias válidas.

    SEMANTIC RATIONALE:
    The memory block is expressed as an ASSISTANT turn (first-person recall)
    instead of a USER turn (external data injection). This ensures the model
    treats the information as self-knowledge it's retrieving, not as
    instructions or data provided by the user.

    The phrasing uses verbs of recall ("Lembro que", "Sei que") to reinforce
    the epistemic ownership. The closing line signals readiness, anchoring
    the model's stance before the actual user message arrives.
    """
    if not memories:
        return None

    lines = [
        f"- {m.get('text', m.get('content', ''))}"
        for m in memories
        if m.get("text") or m.get("content")
    ]
    if not lines:
        return None

    mem_block = "\n".join(lines)

    return (
        "Resgatando contexto relevante da minha memória antes de responder:\n\n"
        f"{mem_block}\n\n"
        "Tenho isso em mente e vou usar esse conhecimento de forma natural na conversa."
    )


def _build_messages(
    user_input: str,
    lang: str,
    memories: list[dict],
    recent_turns: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Monta a lista de mensagens para o llama-server.

    CRITICAL for prompt caching:
    ─────────────────────────────────
    The system prompt is STABLE (never changes at runtime).
    llama-server's --cache-prompt works by matching the PREFIX of
    the message list. If the system prompt is always the same, it
    gets cached after the first request, and subsequent requests
    only need to prefill the NEW tokens.

    Structure (with memories + recent turns):
      [0] system:      STABLE prompt — context + username + memory semantics instruction
      [1] assistant:   first-person memory recall (dynamic, AFTER cached prefix) — from /read
      [2..N] user/assistant: last N pairs of real conversation history — from /read_st
      [N+1] user:       "[Data: DD/MM/YYYY]\n{user_input}"

    Structure (without memories/turns):
      [0] system:    STABLE prompt
      [1] user:      "[Data: DD/MM/YYYY]\n{user_input}"

    MEMORY SEMANTICS:
    Memories from /read are expressed in the ASSISTANT voice (role:assistant)
    as a first-person recall — semantic knowledge the model is "remembering",
    not external data from the user. `recent_turns` (from /read_st) are the
    opposite: they're the actual conversation as it happened, so they're
    inserted as real user/assistant turns instead of being folded into the
    recall block — that keeps the model's literal short-term context (what
    was actually said) separate from associative long-term recall (relevant
    facts a semantic search surfaced).

    KV CACHE IMPACT:
    - Message [0] (system) is always identical → KV cache HIT
    - Message [1] (assistant recall) varies by query → small prefill cost
    - Recent turns + user input vary every turn → prefill cost, but small
      relative to the alternative of re-embedding the whole history via /read
    """
    messages = [
        # STABLE: This is the cached portion — never changes at runtime
        {"role": "system", "content": _SYSTEM_PROMPT_BASE},
    ]

    # DYNAMIC: Memory recall expressed in the assistant's own voice.
    # Role is "assistant" so the model treats this as self-knowledge,
    # not as external input from the user.
    recall_text = _build_memory_recall(memories)
    if recall_text:
        messages.append({
            "role": "assistant",
            "content": recall_text,
        })

    # DYNAMIC: Last N pairs of real conversation turns (from /read_st),
    # inserted as-is — this is literal history, not associative recall.
    for turn in (recent_turns or []):
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    # DYNAMIC: Current date + user input
    today = datetime.datetime.now().strftime("%d/%m/%Y")
    messages.append({
        "role": "user",
        "content": f"[Data de hoje: {today}]\n{user_input}",
    })

    return messages



# ─────────────────────────────────────────────────────────────
#                           FASTAPI
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="AVA — LLM API", version="2.2.0")

# ── Schemas ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = Field(default="default", description="ID da sessão para memória de curto prazo")
    voice:   Optional[str] = Field(default=None, description="Voz TTS. None = usa padrão.")
    lang:    Optional[str] = Field(default=None, description="Idioma forçado. None = detectado.")
    max_turns: int = Field(default=10,  ge=1, le=40, description="Limite de contexto recuperado")
    tts: bool = Field(default=True, description="Dispara TTS após gerar resposta.")
    thinking_depth: int = Field(
        default=0, ge=0, le=10,
        description=(
            "Orçamento de raciocínio (best-effort). LFM2.5-8B-A1B é um reasoning "
            "model — por padrão (0) usamos o comportamento nativo do template "
            "(raciocina livremente, sem teto), priorizando qualidade sobre "
            "latência. Valores 1-10 tentam limitar o raciocínio via "
            "`reasoning_budget_tokens` do llama-server (depth * 300 tokens) "
            "para reduzir latência quando isso importar mais — mas esse campo "
            "do servidor mudou de nome recentemente entre builds do llama.cpp, "
            "então trate como melhor-esforço, não garantia."
        ),
    )
    stream_reasoning: bool = Field(default=True, description="Se True e o modelo emitir reasoning_content, envia para a UI. Se False, ignora completamente.")
    model: Optional[str] = Field(default=None, description="Ignorado — mantido por compatibilidade. Sempre usa o único modelo servido pelo llama-server local.")

class ClearRequest(BaseModel):
    confirm: bool = False
    session_id: Optional[str] = "default"


# ── Schemas para tool use nativo (endpoint /chat/tools) ───────────────────────
# Usado pelo módulo alpha_code para ReAct loop. Não compartilha memória/TTS
# do /chat padrão — mensagens e tools são controlados pelo caller.

class ToolCallMessage(BaseModel):
    role: str = Field(..., description="system | user | assistant | tool")
    content: Optional[str | list[dict]] = Field(
        default=None,
        description=(
            "Texto simples (str) na maioria dos casos. Também aceita o formato "
            "multi-parte do OpenAI vision (lista de blocos {type: text|image_url, ...}) "
            "para requests multimodais — repassado como está para o llama-server, "
            "que só interpreta corretamente quando o modelo carregado está em "
            "modo multimodal."
        ),
    )
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class ToolUseRequest(BaseModel):
    messages: list[ToolCallMessage]
    tools: list[dict] = Field(default_factory=list, description="Lista de tool schemas no formato OpenAI function-calling. IGNORADO quando `grammar` está presente — nesse caso o llama-server usa a grammar para restringir o output e o caller deve parsear `message.content` em vez de `message.tool_calls`.")
    tool_choice: Optional[str] = Field(default="auto", description="auto | required | none | {type:function,function:{name:...}}. IGNORADO quando `grammar` está presente.")
    grammar: Optional[str] = Field(default=None, description="Gramática GBNF (llama.cpp) para restringir o output do modelo. Quando presente, o payload enviado ao llama-server NÃO inclui `tools`/`tool_choice` (desabilita function-calling nativo) e a resposta deve ser lida de `message.content`. O caller é responsável por parsear o content segundo a grammar — garantidamente válido pelo servidor.")
    model: Optional[str] = Field(default=None, description="Ignorado — mantido por compatibilidade. Sempre usa o único modelo servido pelo llama-server local.")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=32000)
    reasoning_effort: Optional[str] = Field(
        default=None,
        description=(
            "Passado direto para o llama-server. Padrão (None) = comportamento "
            "nativo do template (LFM2.5-8B-A1B raciocina antes de responder — "
            "prioriza qualidade). Envie \"none\" para desligar reasoning nesta "
            "chamada específica (único valor com efeito documentado no "
            "tools/server/README.md do llama.cpp; use quando latência por "
            "passo importar mais que qualidade — ex.: loops rápidos do "
            "alpha_code)."
        ),
    )
    allow_llama_fallback: bool = Field(
        default=True,
        description="Ignorado — mantido por compatibilidade. O llama-server local já é o único backend."
    )
    max_retries: int = Field(default=3, ge=0, le=10, description="Máximo de retries em falha transitória (5xx/rede) do llama-server local antes de desistir.")

class ToolUseResponse(BaseModel):
    message: dict
    model: str
    usage: dict
    elapsed_ms: float
    fallback_used: bool = False
    too_large: bool = Field(default=False, description="True se request excedeu limite TPM do modelo (caller deve reduzir contexto ou trocar modelo)")


# ── Lifecycle ────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """
    OPTIMIZED: Async warmup that uses the persistent client
    and sends a representative-length prompt.
    """
    await _warmup()


@app.on_event("shutdown")
async def shutdown():
    """Close persistent HTTP clients gracefully."""
    global _llama_client, _memory_client, _tts_http
    if _llama_client and not _llama_client.is_closed:
        await _llama_client.aclose()
    if _memory_client and not _memory_client.is_closed:
        await _memory_client.aclose()
    if _tts_http and not _tts_http.is_closed:
        await _tts_http.aclose()


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Verifica se a API e o llama-server estão no ar."""
    try:
        client = await _get_llama_client()
        r = await client.get("/health")
        llama_ok = r.status_code == 200
    except Exception:
        llama_ok = False
    return {"api": "ok", "llama_server": "ok" if llama_ok else "down"}


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Inferência síncrona — retorna a resposta completa em JSON.
    OPTIMIZED:
      - Persistent llama + memory clients (no TCP handshake)
      - Stable system prompt (prompt cache hits after 1st request)
      - Parallel memory read + language detection
    """
    user_input = req.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Mensagem vazia.")

    # ── OPTIMIZATION 3: PARALLEL prep ────────────────────────────────────────
    # Run language detection, memory read (semantic) and memory read_st
    # (raw recent history) IN PARALLEL instead of sequentially.
    # This saves ~100-300ms when memory API is slow.
    lang_task = asyncio.ensure_future(
        asyncio.get_event_loop().run_in_executor(None, _safe_detect, user_input)
    )
    memory_task = asyncio.ensure_future(
        memory_read(user_input, session_id=req.session_id, top_k=req.max_turns)
    )
    memory_st_task = asyncio.ensure_future(
        memory_read_st(req.session_id)
    )

    # Wait for all three to complete
    lang, memories, recent_turns = await asyncio.gather(lang_task, memory_task, memory_st_task)
    if not lang:
        lang = "pt"

    # 3. Montar prompt (with stable system prompt for caching)
    messages = _build_messages(user_input, lang, memories, recent_turns)
    log.info(f"tamanho do contexto do assistente: {str(messages).count(chr(0))} caracteres.")
    # 4. Inferência (llama-server local, com retry para falhas transitórias)
    t0 = time.perf_counter()
    log.info(messages)
    try:
        client_used, model_used, r = await _llama_post_with_retry(
            json_payload={
                "messages":    messages,
                "temperature": 0.7,
                **_reasoning_payload_extra(req.thinking_depth),
            },
            stream=False,
        )
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"inference error: {e}")

    resp_data = r.json()
    response_text = resp_data["choices"][0]["message"]["content"]
    reasoning_text = resp_data["choices"][0]["message"].get("reasoning_content", "") or \
                      resp_data["choices"][0]["message"].get("reasoning", "")
    elapsed = time.perf_counter() - t0

    # Log cache stats
    usage = resp_data.get("usage", {})
    cached = usage.get("prompt_tokens_cached", 0)
    total_prompt = usage.get("prompt_tokens", 0)
    log.info(
        f"[CHAT] {elapsed:.2f}s | {len(response_text)} chars | "
        f"reasoning: {len(reasoning_text)} chars | "
        f"prompt: {total_prompt} tokens (cached: {cached})"
    )

    # 5. Gravar turno na memória (fire-and-forget)
    asyncio.create_task(
        memory_save_turn(req.session_id, user_input, response_text)
    )
    asyncio.create_task(
        memory_write_fact(f"Usuário disse: {user_input[:300]}", "chat", 0.7)
    )

    # 6. Disparar TTS em background
    voice = req.voice or voiceModel
    if req.tts and voice:
        asyncio.create_task(tts_speak(response_text, voice, lang))

    return {
        "response": response_text,
        "session_id": req.session_id,
        "lang":     lang,
        "elapsed":  round(elapsed, 3),
        "prompt_cached_tokens": cached,
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Inferência com streaming — retorna Server-Sent Events (SSE).
    OPTIMIZED:
      - Persistent llama + memory clients (no TCP handshake)
      - Stable system prompt (prompt cache hits after 1st request)
      - Parallel memory read + language detection
      - Connection reuse for streaming
    """
    user_input = req.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Mensagem vazia.")

    # ── OPTIMIZATION 3: PARALLEL prep ────────────────────────────────────────
    lang_task = asyncio.ensure_future(
        asyncio.get_event_loop().run_in_executor(None, _safe_detect, user_input)
    )
    memory_task = asyncio.ensure_future(
        memory_read(user_input, session_id=req.session_id, top_k=req.max_turns)
    )
    memory_st_task = asyncio.ensure_future(
        memory_read_st(req.session_id)
    )
    lang, memories, recent_turns = await asyncio.gather(lang_task, memory_task, memory_st_task)
    if not lang:
        lang = "pt"

    messages = _build_messages(user_input, lang, memories, recent_turns)
    log.info(f"tamanho do contexto do assistente: {str(messages).count(chr(0))} caracteres.")
    voice = req.voice or voiceModel

    _ensure_tts_queue()
    _tts_buf = ""

    def _clean_for_tts(text: str) -> str:
        """Remove formatação Markdown que o TTS não consegue ler."""
        t = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)
        t = re.sub(r'`{1,3}[^`]*`{1,3}', '', t)
        t = re.sub(r'#{1,6}\s+', '', t)
        t = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', t)
        t = re.sub(r'^\s*[-*]\s+', '', t, flags=re.M)
        t = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', t)
        return t.strip()

    def _flush_tts_buf():
        nonlocal _tts_buf
        cleaned = _clean_for_tts(_tts_buf)
        if len(cleaned) >= 3:
            _tts_queue.put_nowait((cleaned, voice, lang))
        _tts_buf = ""

    async def generator():
        nonlocal _tts_buf
        full_response = ""
        full_reasoning = ""  
        t0 = time.perf_counter()
        cached_tokens = 0

        # Retry simples para falhas transitórias (5xx/erro de rede) do llama-server local
        for attempt in range(LLAMA_MAX_RETRIES + 1):
            client_used, model_used, r_ctx = await _execute_inference(
                json_payload={
                    "messages":    messages,
                    "temperature": 0.7,
                    **_reasoning_payload_extra(req.thinking_depth),
                },
                stream=True,
            )

            try:
                async with r_ctx as r:
                    if r.status_code >= 500 and attempt < LLAMA_MAX_RETRIES:
                        log.warning(f"llama-server: {r.status_code} no stream (tentativa {attempt + 1}/{LLAMA_MAX_RETRIES + 1}).")
                        await asyncio.sleep(LLAMA_RETRY_BACKOFF_S)
                        continue

                    r.raise_for_status()

                    # Se chegou aqui, a conexão foi aceita e não há erros. Processa as linhas:
                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta_obj = chunk["choices"][0].get("delta", {})

                            if req.stream_reasoning:
                                reasoning = delta_obj.get("reasoning_content", "") or delta_obj.get("reasoning", "")
                                if reasoning:
                                    full_reasoning += reasoning
                                    yield f"data: {json.dumps({'reasoning': reasoning})}\n\n"

                            content = delta_obj.get("content", "")
                            if not content:
                                timings = chunk.get("timings", {})
                                if timings and "cache_n" in timings:
                                    cached_tokens = timings.get("cache_n", 0)
                                continue

                            full_response += content
                            yield f"data: {json.dumps({'delta': content})}\n\n"

                            if req.tts and voice:
                                _tts_buf += content
                                buf_rstrip = _tts_buf.rstrip()
                                if (buf_rstrip and buf_rstrip[-1] in '.!?\n。') \
                                   or len(_tts_buf) > 150:
                                    _flush_tts_buf()
                        except (json.JSONDecodeError, KeyError):
                            continue
                    
                    # Se o stream terminou com sucesso, quebra o loop de retry
                    break 

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                return

        # Rotina de finalização (TTS e memória)
        if _tts_buf.strip():
            _flush_tts_buf()

        elapsed = time.perf_counter() - t0
        log.info(
            f"[STREAM] {elapsed:.2f}s | {len(full_response)} chars | "
            f"reasoning: {len(full_reasoning)} chars | "
            f"cached: {cached_tokens} tokens"
        )

        yield f"data: {json.dumps({'done': True, 'elapsed': round(elapsed, 3), 'prompt_cached_tokens': cached_tokens})}\n\n"

        if full_response:
            asyncio.create_task(
                memory_save_turn(req.session_id, user_input, full_response)
            )
            asyncio.create_task(
                memory_write_fact(f"Usuário disse: {user_input[:300]}", "chat", 0.7)
            )

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        }
    )



@app.delete("/history")
async def clear_history(req: ClearRequest):
    """Limpa o histórico de chat da sessão no servidor de memória."""
    if not req.confirm:
        raise HTTPException(status_code=400, detail="Envie confirm=true para confirmar.")

    try:
        client = await _get_memory_client()
        await client.delete(f"/session/{req.session_id}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Falha ao limpar sessão: {e}")

    return {"cleared": True, "session_id": req.session_id}


@app.get("/history")
async def get_history(session_id: str = "default", last_n: int = 20):
    """Retorna as últimas N mensagens do histórico via módulo de memória."""
    try:
        client = await _get_memory_client()
        r = await client.post(
            "/read",
            json={"query": "histórico recente", "session_id": session_id, "top_k": last_n}
        )
        results = r.json().get("results", [])
        return {"history": results, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Falha ao buscar histórico: {e}")


# ── Cache stats endpoint ─────────────────────────────────────

@app.get("/cache_stats")
async def cache_stats():
    """
    Check how well the prompt cache is working.
    If prompt_cached_tokens is always 0, the system prompt is changing
    between requests and caching is not effective.
    """
    try:
        client = await _get_llama_client()
        # Send a minimal request with the stable system prompt
        r = await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT_BASE},
                    {"role": "user", "content": "test"},
                ],
                "max_tokens": 1,
            },
        )
        if r.status_code == 200:
            usage = r.json().get("usage", {})
            return {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "prompt_tokens_cached": usage.get("prompt_tokens_cached", 0),
                "cache_hit_rate": (
                    round(usage.get("prompt_tokens_cached", 0) / max(usage.get("prompt_tokens", 1), 1) * 100, 1)
                ),
                "system_prompt_length": len(_SYSTEM_PROMPT_BASE),
            }
        return {"error": f"llama-server returned {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────
#                         UTILITÁRIOS
# ─────────────────────────────────────────────────────────────

def _safe_detect(text: str) -> str:
    """
    Detecta o idioma do texto para escolher a pronúncia do TTS.

    Estava hardcoded para sempre retornar "pt" (langdetect importado mas
    nunca usado) — provavelmente um atalho de debug que ficou. Isso não
    afeta a RESPOSTA do modelo (o system prompt já instrui a responder no
    idioma do usuário, e o LFM2.5 segue isso bem por conta própria), mas
    afeta o TTS: `lang` é passado para o serviço de voz, então uma
    conversa em qualquer idioma diferente de português seria falada com
    pronúncia errada. Restaurado com fallback seguro para "pt" se a
    detecção falhar (texto curto demais, erro do langdetect, etc.).
    """
    try:
        return detect(text) if len(text.strip()) >= 3 else "pt"
    except Exception:
        return "pt"


# ─────────────────────────────────────────────────────────────
#                   TOOL USE NATIVO (llama-server local)
# ─────────────────────────────────────────────────────────────
# Endpoint para o módulo alpha_code (agente ReAct). Não compartilha
# memória/TTS do /chat — mensagens e tools são controlados pelo caller.

class RequestTooLargeError(RuntimeError):
    """Sinaliza que o request excedeu o contexto máximo do llama-server local."""
    pass


def _is_request_too_large_error(msg: str) -> bool:
    """Detecta mensagens de erro indicando que o prompt excede o contexto do modelo."""
    if not msg:
        return False
    msg_l = msg.lower()
    return any(s in msg_l for s in ("context", "too large", "exceeds", "n_ctx", "exceed"))


async def _llama_tool_call(
    messages: list[dict],
    tools: list[dict],
    tool_choice,
    temperature: float,
    max_retries: int = 3,
    grammar: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> tuple[dict, str, bool]:
    """
    Executa tool call no llama-server local, com retry em falhas transitórias.

    Estratégia:
      - Em erro de rede (ConnectError/ReadTimeout/RemoteProtocolError): retry
        com backoff, até esgotar max_retries.
      - Em 5xx: retry com backoff, até esgotar max_retries.
      - Em erro indicando contexto grande demais: NÃO retenta — levanta
        RequestTooLargeError (esperar não adianta, o tamanho não muda).
      - Outros erros (4xx): não retry.

    MODO GRAMMAR (otimização extrema):
      Quando `grammar` (string GBNF) é fornecida, o payload NÃO inclui
      `tools`/`tool_choice` — isso desabilita o function-calling nativo do
      OpenAI, que consome ~500-1000 tokens de schemas por request e é a fonte
      #1 de JSON malformado em modelos locais. Em vez disso, a grammar força
      o output a seguir um formato JSON compacto (definido pelo caller), e a
      resposta vem em `message.content` — não em `message.tool_calls`.
      O caller é responsável por parsear o content. Como a grammar garante
      validade estrutural, o parse só falha se o caller cometer erro na
      definição da grammar — nunca por output malformado do modelo.

    Retorna (message_dict, model_used, fallback_used=False sempre — não há
    mais fallback entre provedores, só o llama-server local).
    """
    client = await _get_llama_client()
    payload: dict = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }
    # `reasoning_effort` é passado direto — o único valor com efeito
    # documentado no llama-server é "none" (desliga reasoning nesta
    # chamada). Padrão (None) = não manda o campo, deixa o comportamento
    # nativo do template agir (prioriza qualidade). Ver ToolUseRequest.
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    # ── Modo grammar vs. modo tool-calling nativo ─────────────────────────
    # São mutuamente exclusivos no llama-server: se `grammar` está presente,
    # não enviamos `tools` — o output vem em `content` e a grammar garante o
    # formato. Isso economiza tokens de prompt (schemas) e elimina retries
    # por JSON malformado.
    if grammar:
        payload["grammar"] = grammar
    elif tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            r = await client.post("/v1/chat/completions", json=payload)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            last_error = e
            log.warning(
                f"llama-server /chat/tools: erro de rede ({type(e).__name__}), "
                f"tentativa {attempt + 1}/{max_retries + 1}."
            )
            if attempt < max_retries:
                await asyncio.sleep(LLAMA_RETRY_BACKOFF_S)
                continue
            raise RuntimeError(f"llama-server inacessível em {LLAMA_URL}: {e}") from e

        if r.status_code == 200:
            data = r.json()
            return data["choices"][0]["message"], MODEL_NAME, False

        # Erro de contexto grande demais: não faz sentido retentar
        try:
            body = r.json()
            err_msg = body.get("error", {}).get("message", "") or r.text
        except Exception:
            err_msg = r.text

        if _is_request_too_large_error(err_msg):
            log.warning(f"llama-server /chat/tools: contexto excede o limite. Erro: {err_msg[:200]}")
            raise RequestTooLargeError(f"Contexto excede o limite do llama-server: {err_msg[:300]}")

        # 5xx: retry com backoff
        if r.status_code >= 500:
            log.warning(f"llama-server /chat/tools: {r.status_code} (tentativa {attempt + 1}/{max_retries + 1}).")
            if attempt < max_retries:
                await asyncio.sleep(LLAMA_RETRY_BACKOFF_S)
                continue
            raise RuntimeError(f"llama-server /chat/tools: {r.status_code} persistente. Erro: {err_msg[:300]}")

        # Outros erros (4xx): não retry
        raise RuntimeError(f"llama-server /chat/tools: {r.status_code} - {err_msg[:300]}")

    raise RuntimeError(f"llama-server /chat/tools: falhou após {max_retries + 1} tentativa(s): {last_error}")


@app.post("/chat/tools", response_model=ToolUseResponse)
async def chat_tools(req: ToolUseRequest):
    """
    Tool use nativo (llama-server local) — para o módulo alpha_code.

    Recebe messages + tools (formato OpenAI function-calling) e retorna
    a mensagem do assistant (pode conter tool_calls ou content).

    MODO GRAMMAR (otimizado):
      Se `req.grammar` estiver presente, o payload enviado ao llama-server
      NÃO inclui `tools`/`tool_choice`. A grammar GBNF restringe o output
      do modelo a um formato JSON compacto definido pelo caller, e a
      resposta é devolvida em `message.content` (não em `message.tool_calls`).
      O caller faz o parse do content — garantidamente válido pelo servidor.
      Economiza ~500-1000 tokens de prompt por request (schemas) e elimina
      retries por JSON malformado.

    Diferenças vs /chat:
      - Sem memória persistida (caller gerencia)
      - Sem TTS, sem detecção de idioma
      - Sem streaming (síncrono — alpha_code faz seu próprio streaming de steps)
      - too_large=true quando o contexto excede o limite do llama-server
        local (caller deve reduzir o contexto)
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages vazio.")

    messages = [m.model_dump(exclude_none=True) for m in req.messages]

    t0 = time.perf_counter()
    try:
        message, model_used, fallback = await _llama_tool_call(
            messages=messages,
            tools=req.tools,
            tool_choice=req.tool_choice,
            temperature=req.temperature,
            max_retries=req.max_retries,
            grammar=req.grammar,
            reasoning_effort=req.reasoning_effort,
        )
    except RequestTooLargeError as e:
        # Contexto excede o limite do llama-server local — caller precisa agir
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return ToolUseResponse(
            message={"role": "assistant", "content": "", "tool_calls": None},
            model=MODEL_NAME,
            usage={"error": "request_too_large", "detail": str(e)[:500]},
            elapsed_ms=round(elapsed_ms, 1),
            fallback_used=False,
            too_large=True,
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {e.response.text[:500]}")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"inference error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference error: {e}")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    approx_prompt_tokens = sum(len(str(m)) for m in messages) // 4
    approx_completion_tokens = len(str(message)) // 4

    return ToolUseResponse(
        message=message,
        model=model_used,
        usage={
            "prompt_tokens_approx": approx_prompt_tokens,
            "completion_tokens_approx": approx_completion_tokens,
            "total_approx": approx_prompt_tokens + approx_completion_tokens,
        },
        elapsed_ms=round(elapsed_ms, 1),
        fallback_used=fallback,
        too_large=False,
    )


# ─────────────────────────────────────────────────────────────
#                         ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4003, log_level="info")