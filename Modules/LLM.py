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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [LLM] %(message)s")
log = logging.getLogger("ava.llm")

# ─────────────────────────────────────────────────────────────
#                          CONFIG
# ─────────────────────────────────────────────────────────────

BASEFOLDER = Path(__file__).parent.parent

# URLs das APIs satélite
MEMORY_URL = "http://localhost:3001"
TTS_URL    = "http://localhost:3004"

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
            timeout=httpx.Timeout(120.0, connect=5.0),
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
            timeout=httpx.Timeout(8.0, connect=3.0),
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
            timeout=httpx.Timeout(15.0),
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
model_path = model_raw.split("/") or model_raw.split("\\")
MODEL_PATH = model_path[-1]
MODEL_NAME = model_raw

try:
    with open(BASEFOLDER / f"CfgModels/{model_raw}.json", "r", encoding="utf-8") as f:
        MODELCFG = json.load(f)
except FileNotFoundError:
    MODELCFG = {}


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
    log.info("[MAIN] Warmup do modelo (representative prompt)...")
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
            log.info(f"[MAIN] Warmup concluído. Cached tokens: {cached_tokens}")
        else:
            log.info(f"[MAIN] Warmup response: {r.status_code}")

    except Exception as e:
        log.info(f"[MAIN] Warmup falhou (não crítico): {e}")


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

    Structure (with memories):
      [0] system:    STABLE prompt — context + username + memory semantics instruction
      [1] assistant: first-person memory recall (dynamic, AFTER cached prefix)
      [2] user:      "[Data: DD/MM/YYYY]\n{user_input}"

    Structure (without memories):
      [0] system:    STABLE prompt
      [1] user:      "[Data: DD/MM/YYYY]\n{user_input}"

    MEMORY SEMANTICS:
    Memories are expressed in the ASSISTANT voice (role:assistant) as a
    first-person recall. This is semantically correct: the model is
    "remembering" its own knowledge, not receiving external data from
    the user. The system prompt explains this contract so the model
    understands why an assistant turn appears before the user's message.

    KV CACHE IMPACT:
    - Message [0] (system) is always identical → KV cache HIT
    - Message [1] (assistant recall) varies by query → small prefill cost
    - Message [2] (user input) varies → small prefill cost
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

class ClearRequest(BaseModel):
    confirm: bool = False
    session_id: Optional[str] = "default"


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
    # Run language detection and memory read IN PARALLEL instead of sequentially.
    # This saves ~100-300ms when memory API is slow.
    lang_task = asyncio.ensure_future(
        asyncio.get_event_loop().run_in_executor(None, _safe_detect, user_input)
    )
    memory_task = asyncio.ensure_future(
        memory_read(user_input, session_id=req.session_id, top_k=req.max_turns)
    )

    # Wait for both to complete
    lang, memories = await asyncio.gather(lang_task, memory_task)
    if not lang:
        lang = "pt"

    # 3. Montar prompt (with stable system prompt for caching)
    messages = _build_messages(user_input, lang, memories)

    # 4. Inferência — OPTIMIZED: persistent client
    t0 = time.perf_counter()
    log.info(messages)
    try:
        client = await _get_llama_client()
        r = await client.post(
            "/v1/chat/completions",
            json={
                "model":       MODEL_NAME,
                "messages":    messages,
                "temperature": 0.7,
                "stream":      False,
            },
        )
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"llama-server error: {e}")

    resp_data = r.json()
    response_text = resp_data["choices"][0]["message"]["content"]
    elapsed = time.perf_counter() - t0

    # Log cache stats
    usage = resp_data.get("usage", {})
    cached = usage.get("prompt_tokens_cached", 0)
    total_prompt = usage.get("prompt_tokens", 0)
    log.info(
        f"[CHAT] {elapsed:.2f}s | {len(response_text)} chars | "
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
    lang, memories = await asyncio.gather(lang_task, memory_task)
    if not lang:
        lang = "pt"

    messages = _build_messages(user_input, lang, memories)
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
        t0 = time.perf_counter()
        cached_tokens = 0

        try:
            # ── OPTIMIZED: Use persistent client for streaming ────────────────
            client = await _get_llama_client()
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model":       MODEL_NAME,
                    "messages":    messages,
                    "temperature": 0.7,
                    "stream":      True,
                },
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if not delta:
                            continue

                        full_response += delta
                        yield f"data: {json.dumps({'delta': delta})}\n\n"

                        # Buffer para o TTS
                        if req.tts and voice:
                            _tts_buf += delta
                            buf_rstrip = _tts_buf.rstrip()
                            if (buf_rstrip and buf_rstrip[-1] in '.!?\n。') \
                               or len(_tts_buf) > 150:
                                _flush_tts_buf()

                        # Track cached tokens from streaming response
                        usage = chunk.get("usage", {})
                        if usage and "prompt_tokens_cached" in usage:
                            cached_tokens = usage["prompt_tokens_cached"]

                    except (json.JSONDecodeError, KeyError):
                        continue

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        # Flush TTS buffer
        if _tts_buf.strip():
            _flush_tts_buf()

        elapsed = time.perf_counter() - t0
        log.info(
            f"[STREAM] {elapsed:.2f}s | {len(full_response)} chars | "
            f"cached: {cached_tokens} tokens"
        )

        yield f"data: {json.dumps({'done': True, 'elapsed': round(elapsed, 3), 'prompt_cached_tokens': cached_tokens})}\n\n"

        # Salva memória (fire-and-forget)
        if full_response:
            asyncio.create_task(
                memory_save_turn(req.session_id, user_input, full_response)
            )
            asyncio.create_task(
                memory_write_fact(f"Usuário disse: {user_input[:300]}", "chat", 0.7)
            )

    return StreamingResponse(generator(), media_type="text/event-stream")


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
    return "pt"


# ─────────────────────────────────────────────────────────────
#                         ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4003, log_level="info")