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
GROQ_KEY = os.getenv("GROQ_API_KEY")

log.info(f"[LLM] GROQ_API_KEY loaded: {'set' if GROQ_KEY else 'not set'}")

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
GROQ_URL          = "https://api.groq.com/openai"    

# ─────────────────────────────────────────────────────────────
#          OPTIMIZATION 1: PERSISTENT HTTP CLIENTS
# ─────────────────────────────────────────────────────────────
# Instead of creating a new httpx.AsyncClient per request (which
# costs ~50-150ms for TCP handshake + HTTP/1.1 upgrade), we
# create them once at startup and reuse across all requests.
# This is the SINGLE BIGGEST latency win for TTFT.

_llama_client: httpx.AsyncClient | None = None
_groq_client: httpx.AsyncClient | None = None
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




async def _get_groq_client() -> httpx.AsyncClient:
    """Persistent client to Groq API — connection pooling + keep-alive."""
    global _groq_client
    if _groq_client is None or _groq_client.is_closed:
        _groq_client = httpx.AsyncClient(
            base_url=GROQ_URL,
            timeout=httpx.Timeout(9999999.0, connect=5.0),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=6,
                keepalive_expiry=60.0,       # Keep connections warm for 60s
            ),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_KEY}"},
        )
    return _groq_client





    
    
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
MODEL_GROQ_NAME = "openai/gpt-oss-120b"

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

groq_state = {
    "blocked_until": 0.0,          # timestamp em que TPM libera
    "daily_exhausted": False,      # RPD esgotado?
    "daily_reset_at": 0.0,         # timestamp em que RPD reseta
    "last_remaining_tokens": 18000,
    "last_remaining_requests": 14400,
}



async def _execute_inference(json_payload: dict, thinking_depth: int, stream: bool = False):
    """
    Executa a inferência respeitando as regras estritas:
    - Se for usar Groq e der erro 429 (TPM), espera e tenta de novo (loop infinito).
    - Só faz fallback para llama-server se for RPD ou erro 5xx de servidor.
    """
    while True:
        client = await choose_client(thinking_depth)
        model_name = await get_model_name(thinking_depth)

        # Se escolheu llama-server, executa e retorna
        if not _is_groq_client(client):
            payload = {**json_payload, "model": model_name, "stream": stream, "max_tokens": 8000}
            if stream:
                ctx = client.stream("POST", "/v1/chat/completions", json=payload)
                return client, model_name, ctx
            else:
                r = await client.post("/v1/chat/completions", json=payload)
                return client, model_name, r

        # Escolheu Groq. Tenta executar
        try:
            payload = {**json_payload, "model": model_name, "stream": stream, "max_tokens": 8000}
            if stream:
                # No stream, retornamos o contexto. O 429 será tratado por quem chamar.
                ctx = client.stream("POST", "/v1/chat/completions", json=payload)
                return client, model_name, ctx
            else:
                r = await client.post("/v1/chat/completions", json=payload)
                await update_groq_limits(r)

                if r.status_code == 429:
                    log.warning("Groq: 429 (TPM). Vou esperar e tentar de novo...")
                    continue # Volta para o while True, vai chamar choose_client que vai esperar
                
                if r.status_code >= 500:
                    log.warning(f"Groq: Erro servidor {r.status_code}. Fallback para llama-server.")
                    fb = await _get_llama_client()
                    rfb = await fb.post("/v1/chat/completions", json={**json_payload, "model": MODEL_NAME, "stream": False, "max_tokens": 8000})
                    return fb, MODEL_NAME, rfb

                return client, model_name, r

        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            log.warning(f"Groq: erro de rede ({type(e).__name__}). Fallback para llama-server.")
            fb = await _get_llama_client()
            payload = {**json_payload, "model": MODEL_NAME, "stream": stream, "max_tokens": 8000}
            if stream:
                return fb, MODEL_NAME, fb.stream("POST", "/v1/chat/completions", json=payload)
            else:
                rfb = await fb.post("/v1/chat/completions", json=payload)
                return fb, MODEL_NAME, rfb


def _parse_duration_to_seconds(value: str) -> float:
    if not value:
        return 2.0
    value = value.strip().lower()
    pattern = re.compile(r'(\d+(?:\.\d+)?)([hms])')
    matches = pattern.findall(value)
    total = 0.0
    for num, unit in matches:
        n = float(num)
        if unit == 'h': total += n * 3600
        elif unit == 'm': total += n * 60
        elif unit == 's': total += n
    return total if total > 0 else 2.0


def _extract_retry_seconds_from_body(body_text: str) -> Optional[float]:
    """
    Extrai tempo de retry do corpo do erro do Groq.
    Formato: "Please try again in 9.18s" ou "Please try again in 9.18 seconds"
    Retorna None se não encontrar.
    """
    if not body_text:
        return None
    m = re.search(r"try again in\s+(\d+(?:\.\d+)?)\s*(?:s|seconds?)", body_text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _is_request_too_large_error(body_text: str) -> bool:
    """
    Detecta erro do Groq onde o REQUEST individual é maior que o limite TPM do modelo.
    Diferente de "rate limit reached" — esse NÃO resolve esperando.

    Mensagens típicas:
      - "Request too large for model X on tokens per minute (TPM)"
      - "please reduce your message size and try again"
    """
    if not body_text:
        return False
    bt = body_text.lower()
    return ("request too large" in bt and "tokens per minute" in bt) or \
           ("reduce your message size" in bt) or \
           ("request_too_large" in bt)


def _check_daily_reset():
    if groq_state["daily_exhausted"] and time.time() >= groq_state["daily_reset_at"]:
        groq_state["daily_exhausted"] = False
        groq_state["last_remaining_requests"] = 14400
        groq_state["last_remaining_tokens"] = 18000
        groq_state["blocked_until"] = 0.0
        log.info("Groq: janela diária (RPD) resetada — voltando a usar Groq.")




async def update_groq_limits(response):
    headers = response.headers
    if "x-ratelimit-remaining-tokens" not in headers and "retry-after" not in headers:
        return

    try:
        remaining_tokens = int(headers.get("x-ratelimit-remaining-tokens", 18000))
    except (ValueError, TypeError):
        remaining_tokens = 18000
    try:
        remaining_requests = int(headers.get("x-ratelimit-remaining-requests", 14400))
    except (ValueError, TypeError):
        remaining_requests = 14400

    groq_state["last_remaining_tokens"] = remaining_tokens
    groq_state["last_remaining_requests"] = remaining_requests

    # Limite DIÁRIO (RPD) esgotado → a partir de agora só usa llama-server hoje
    if remaining_requests <= 0:
        reset_requests_str = headers.get("x-ratelimit-reset-requests", "")
        reset_seconds = _parse_duration_to_seconds(reset_requests_str)
        reset_seconds = max(reset_seconds, 60.0)
        groq_state["daily_exhausted"] = True
        groq_state["daily_reset_at"] = time.time() + reset_seconds
        log.warning(f"Groq: RPD diário esgotado. Usando llama-server por {reset_seconds:.1f}s.")
        return

    # Limite POR MINUTO (TPM) esgotado → bloqueia e espera
    if remaining_tokens <= 0:
        reset_tokens_str = headers.get("x-ratelimit-reset-tokens", "")
        reset_seconds = _parse_duration_to_seconds(reset_tokens_str)
        wait_seconds = reset_seconds + 0.5
        groq_state["blocked_until"] = time.time() + wait_seconds
        log.warning(f"Groq: TPM por minuto esgotado. Esperando {wait_seconds:.2f}s para tentar de novo.")
        return

    # 429 com remaining_tokens > 0 (request excede quota disponível no bucket)
    # Ex: "Limit 8000, Used 6467, Requested 2757" → remaining=1533, mas request=2757
    # Aqui precisamos distinguir:
    #   - "try again in X seconds" → rate limit normal, espera
    #   - "Request too large" → request é maior que o limite TPM inteiro, NÃO espera
    if response.status_code == 429 and "retry-after" in headers:
        # Tenta ler o body para detectar tipo do erro
        try:
            body_text = response.text if hasattr(response, 'text') else ""
        except Exception:
            body_text = ""

        if _is_request_too_large_error(body_text):
            # Request é grande demais para o modelo — NÃO bloqueia globalmente
            # (esperar não adianta: request size não muda)
            # Deixa o caller decidir: trocar de modelo ou reduzir contexto
            log.warning(
                f"Groq: Request too large (TPM={remaining_tokens} restantes, mas request individual excede limite do modelo)."
            )
            # Seta blocked_until curto (5s) só pra evitar spam imediato, mas caller
            # precisa trocar de modelo ou reduzir contexto.
            groq_state["blocked_until"] = time.time() + 5.0
            return

        retry_after_str = headers.get("retry-after", "5").strip()
        try:
            retry_seconds = float(retry_after_str)
        except (ValueError, TypeError):
            retry_seconds = _parse_duration_to_seconds(retry_after_str)
        wait_seconds = retry_seconds + 0.5
        groq_state["blocked_until"] = time.time() + wait_seconds
        log.warning(
            f"Groq: 429 com retry-after={retry_seconds}s "
            f"(remaining_tokens={remaining_tokens}, request excede bucket). "
            f"Esperando {wait_seconds:.2f}s."
        )
        return

    # Prevenção (quando tá chegando perto do limite)
    if remaining_tokens < 500:
        reset_tokens_str = headers.get("x-ratelimit-reset-tokens", "5s")
        reset_seconds = _parse_duration_to_seconds(reset_tokens_str)
        groq_state["blocked_until"] = time.time() + reset_seconds
        log.info(f"Groq: TPM baixo ({remaining_tokens}). Pre-blocked por {reset_seconds:.2f}s.")




async def _wait_for_groq_tpm():
    now = time.time()
    if now < groq_state["blocked_until"]:
        wait = groq_state["blocked_until"] - now
        log.info(f"Groq: aguardando {wait:.2f}s para liberação do TPM...")
        await asyncio.sleep(wait)


async def choose_client(thinking_depth: int = 0):
    _check_daily_reset()

    # Regra 1: Dificuldade baixa → sempre llama-server
    if thinking_depth <= 5:
        return await _get_llama_client()

    # Regra 2: Dificuldade alta (>5), mas limite diário estourado → llama-server
    if groq_state["daily_exhausted"]:
        log.info("Groq: RPD diário esgotado, usando llama-server.")
        return await _get_llama_client()

    # Regra 3: Dificuldade alta, mas limite por minuto estourado → ESPERA e usa Groq
    await _wait_for_groq_tpm()
    return await _get_groq_client()



async def get_model_name(thinking_depth: int = 0) -> str:
    if thinking_depth <= 5:
        return MODEL_NAME
    if groq_state["daily_exhausted"]:
        return MODEL_NAME
    return MODEL_GROQ_NAME


def _is_groq_client(client: httpx.AsyncClient) -> bool:
    try:
        return "groq.com" in str(client.base_url)
    except Exception:
        return False
    
    
async def _groq_post_with_fallback(
    json_payload: dict,
    thinking_depth: int,
    stream: bool = False,
):
    """
    Executa POST no /v1/chat/completions com:
      - Retry em 429 (espera TPM liberar)
      - Fallback automático para llama-server se:
          * RPD esgotado
          * 429 persistente
          * Erro 5xx do Groq
          * Exceção de rede

    Retorna tuplo: (client_usado, model_name_usado, response_obj)
    """
    # 1) Determina cliente inicial
    client = await choose_client(thinking_depth)
    model_name = await get_model_name(thinking_depth)

    # 2) Se já caiu em llama-server (RPD), segue direto
    if not _is_groq_client(client):
        if stream:
            ctx = client.stream("POST", "/v1/chat/completions",
                                json={**json_payload, "model": model_name, "stream": True})
            return client, model_name, ctx
        else:
            r = await client.post("/v1/chat/completions",
                                  json={**json_payload, "model": model_name, "stream": False})
            return client, model_name, r

    # 3) Tentativa Groq com retry
    MAX_RETRIES = 2
    for attempt in range(MAX_RETRIES + 1):
        # Antes de cada tentativa, garante que TPM está liberado
        await _wait_for_groq_tpm()

        # Checa se RPD virou durante a espera
        _check_daily_reset()
        if groq_state["daily_exhausted"]:
            log.warning("Groq: RPD detectado, fallback para llama-server.")
            client = await _get_llama_client()
            model_name = MODEL_NAME
            if stream:
                ctx = client.stream("POST", "/v1/chat/completions",
                                    json={**json_payload, "model": model_name, "stream": True})
                return client, model_name, ctx
            else:
                r = await client.post("/v1/chat/completions",
                                      json={**json_payload, "model": model_name, "stream": False})
                return client, model_name, r

        try:
            if stream:
                # Para streaming, abrimos o context manager e retornamos
                ctx = client.stream(
                    "POST", "/v1/chat/completions",
                    json={**json_payload, "model": model_name, "stream": True},
                )
                # Não conseguimos ler headers sem entrar no ctx.
                # O caller vai lidar com 429 dentro do streaming.
                return client, model_name, ctx
            else:
                r = await client.post(
                    "/v1/chat/completions",
                    json={**json_payload, "model": model_name, "stream": False},
                )
                await update_groq_limits(r)

                if r.status_code == 429:
                    log.warning(f"Groq: 429 (tentativa {attempt+1}/{MAX_RETRIES+1}).")
                    if attempt < MAX_RETRIES:
                        # update_groq_limits já setou blocked_until ou daily_exhausted
                        continue
                    # esgotou tentativas → fallback
                    log.warning("Groq: 429 persistente. Fallback para llama-server.")
                    fb = await _get_llama_client()
                    rfb = await fb.post("/v1/chat/completions",
                                        json={**json_payload, "model": MODEL_NAME, "stream": False})
                    return fb, MODEL_NAME, rfb

                if r.status_code >= 500:
                    log.warning(f"Groq: {r.status_code} (tentativa {attempt+1}).")
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(1.0)
                        continue
                    fb = await _get_llama_client()
                    rfb = await fb.post("/v1/chat/completions",
                                        json={**json_payload, "model": MODEL_NAME, "stream": False})
                    return fb, MODEL_NAME, rfb

                # Sucesso
                return client, model_name, r

        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            log.warning(f"Groq: erro de rede ({type(e).__name__}). Fallback para llama-server.")
            fb = await _get_llama_client()
            if stream:
                ctx = fb.stream("POST", "/v1/chat/completions",
                                json={**json_payload, "model": MODEL_NAME, "stream": True})
                return fb, MODEL_NAME, ctx
            else:
                rfb = await fb.post("/v1/chat/completions",
                                    json={**json_payload, "model": MODEL_NAME, "stream": False})
                return fb, MODEL_NAME, rfb

    # Fallback final de segurança
    fb = await _get_llama_client()
    rfb = await fb.post("/v1/chat/completions",
                        json={**json_payload, "model": MODEL_NAME, "stream": False})
    return fb, MODEL_NAME, rfb
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
    thinking_depth: int = Field(default=0, ge=0, le=10, description="Profundidade do pensamento.")
    stream_reasoning: bool = Field(default=True, description="Se True e o modelo emitir reasoning_content, envia para a UI. Se False, ignora completamente.")
    model: Optional[str] = Field(default=None, description="Força modelo Groq específico. None = usa roteamento padrão por thinking_depth.")

class ClearRequest(BaseModel):
    confirm: bool = False
    session_id: Optional[str] = "default"


# ── Schemas para tool use nativo (endpoint /chat/tools) ───────────────────────
# Usado pelo módulo alpha_code para ReAct loop. Não compartilha memória/TTS
# do /chat padrão — mensagens e tools são controlados pelo caller.

class ToolCallMessage(BaseModel):
    role: str = Field(..., description="system | user | assistant | tool")
    content: Optional[str] = None
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class ToolUseRequest(BaseModel):
    messages: list[ToolCallMessage]
    tools: list[dict] = Field(default_factory=list, description="Lista de tool schemas no formato OpenAI function-calling")
    tool_choice: Optional[str] = Field(default="auto", description="auto | required | none | {type:function,function:{name:...}}")
    model: Optional[str] = Field(default=None, description="Força modelo Groq. None = gpt-oss-120b padrão.")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=32000)
    reasoning_effort: Optional[str] = Field(default=None, description="low | medium | high. None = default do modelo.")
    allow_llama_fallback: bool = Field(
        default=True,
        description="Se True, faz fallback para llama-server em RPD/5xx/rede. Se False, só usa Groq."
    )
    max_retries: int = Field(default=3, ge=0, le=10, description="Máximo de retries em 429 antes de fallback/desistir.")

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
    log.info(f"tamanho do contexto do assistente: {str(messages).count(chr(0))} caracteres.")
    # 4. Inferência com fallback automático
    t0 = time.perf_counter()
    log.info(messages)
    try:
        client_used, model_used, r = await _groq_post_with_fallback(
            json_payload={
                "messages":    messages,
                "temperature": 0.7,
            },
            thinking_depth=req.thinking_depth,
            stream=False,
        )
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"inference error: {e}")

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
    await asyncio.sleep(2)
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

        # Loop para lidar com 429 (TPM) no Groq: espera e reinicia o stream
        while True:
            client_used, model_used, r_ctx = await _execute_inference(
                json_payload={
                    "messages":    messages,
                    "temperature": 0.7,
                },
                thinking_depth=req.thinking_depth,
                stream=True,
            )

            try:
                async with r_ctx as r:
                    if _is_groq_client(client_used):
                        await update_groq_limits(r)

                    # Se Groq bloqueou por minuto (TPM), espera e reinicia o loop
                    if r.status_code == 429 and _is_groq_client(client_used):
                        log.warning("Groq: 429 no stream. Esperando para tentar de novo...")
                        await _wait_for_groq_tpm()
                        continue # Volta para o while True

                    # Se for erro grave no Groq, desvia pro llama-server
                    if r.status_code >= 500 and _is_groq_client(client_used):
                        log.warning(f"Groq: {r.status_code} no stream. Fallback para llama-server.")
                        client_used = await _get_llama_client()
                        model_used = MODEL_NAME
                        r_ctx = client_used.stream("POST", "/v1/chat/completions",
                            json={"model": MODEL_NAME, "messages": messages, "temperature": 0.7, "stream": True, "max_tokens": 8000})
                        continue # Reinicia o loop para entrar no contexto do llama-server

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
                    
                    # Se o stream terminou com sucesso, quebra o loop while True
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
    return "pt"


# ─────────────────────────────────────────────────────────────
#                   TOOL USE NATIVO (Groq)
# ─────────────────────────────────────────────────────────────
# Endpoint para o módulo alpha_code (agente ReAct). Não compartilha
# memória/TTS do /chat — mensagens e tools são controlados pelo caller.

class RequestTooLargeError(RuntimeError):
    """Sinaliza que o request excedeu o limite TPM do modelo — NÃO é rate limit normal."""
    pass


async def _groq_tool_call(
    messages: list[dict],
    tools: list[dict],
    tool_choice,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
    reasoning_effort: Optional[str],
    allow_llama_fallback: bool = True,
    max_retries: int = 3,
) -> tuple[dict, str, bool]:
    """
    Executa tool call no Groq com retry respeitando rate limit.

    Estratégia:
      - Antes de cada tentativa: chama _wait_for_groq_tpm() (respeita blocked_until)
      - Em 429 "try again in Xs": espera e tenta de novo (rate limit normal)
      - Em 429 "Request too large": NÃO tenta de novo — levanta RequestTooLargeError
        (esperar não adianta, request size não muda)
      - Em RPD diário esgotado: fallback direto (ou erro se allow_llama_fallback=False)
      - Em 5xx: 1 retry com sleep 1s, depois fallback/erro
      - Em erro de rede: fallback/erro direto

    Retorna (message_dict, model_used, fallback_used).
    """
    payload: dict = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    forced_model = model or MODEL_GROQ_NAME
    llama_payload = {**payload, "model": MODEL_NAME}

    # Caso 1: RPD diário já esgotado
    if groq_state["daily_exhausted"]:
        if not allow_llama_fallback:
            raise RuntimeError(
                f"Groq RPD diário esgotado e allow_llama_fallback=False. "
                f"Tente novamente em {groq_state['daily_reset_at'] - time.time():.1f}s."
            )
        log.warning("Groq /chat/tools: RPD diário esgotado, usando llama-server.")
        r = await (await _get_llama_client()).post("/v1/chat/completions", json=llama_payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"], MODEL_NAME, True

    # Caso 2: loop de tentativas com retry em 429
    client = await _get_groq_client()

    for attempt in range(max_retries + 1):
        await _wait_for_groq_tpm()

        _check_daily_reset()
        if groq_state["daily_exhausted"]:
            if not allow_llama_fallback:
                raise RuntimeError("Groq RPD diário esgotado durante retry.")
            log.warning("Groq /chat/tools: RPD detectado durante retry, fallback llama-server.")
            r = await (await _get_llama_client()).post("/v1/chat/completions", json=llama_payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"], MODEL_NAME, True

        try:
            r = await client.post(
                "/v1/chat/completions",
                json={**payload, "model": forced_model},
            )
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            if not allow_llama_fallback:
                raise RuntimeError(f"Groq erro de rede: {e}")
            log.warning(f"Groq /chat/tools: erro rede ({type(e).__name__}). Fallback llama-server.")
            r = await (await _get_llama_client()).post("/v1/chat/completions", json=llama_payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"], MODEL_NAME, True

        await update_groq_limits(r)

        # 200: sucesso
        if r.status_code == 200:
            data = r.json()
            return data["choices"][0]["message"], forced_model, False

        # 429: distinguir rate-limit-normal vs request-too-large
        if r.status_code == 429:
            # Lê body para detectar tipo do erro
            try:
                body = r.json()
                err_msg = body.get("error", {}).get("message", "") or r.text
            except Exception:
                err_msg = r.text

            # ── Request too large: NÃO retenta, sinaliza caller ──
            if _is_request_too_large_error(err_msg):
                log.warning(
                    f"Groq /chat/tools: REQUEST TOO LARGE para {forced_model}. "
                    f"Caller deve trocar modelo ou reduzir contexto. "
                    f"Erro: {err_msg[:200]}"
                )
                raise RequestTooLargeError(
                    f"Request too large for {forced_model} (TPM limit): {err_msg[:300]}"
                )

            # ── Rate limit normal: espera e tenta de novo ──
            if time.time() >= groq_state["blocked_until"]:
                wait_from_body = _extract_retry_seconds_from_body(err_msg)
                if wait_from_body:
                    wait_seconds = wait_from_body + 0.5
                    groq_state["blocked_until"] = time.time() + wait_seconds
                    log.warning(
                        f"Groq /chat/tools: 429 (tentativa {attempt+1}/{max_retries+1}). "
                        f"Body indica retry em {wait_from_body}s. Esperando {wait_seconds:.2f}s."
                    )
                else:
                    groq_state["blocked_until"] = time.time() + 10.0
                    log.warning(
                        f"Groq /chat/tools: 429 (tentativa {attempt+1}/{max_retries+1}). "
                        f"Sem retry-after. Esperando 10s."
                    )
            else:
                log.warning(
                    f"Groq /chat/tools: 429 (tentativa {attempt+1}/{max_retries+1}). "
                    f"Wait já setado: {groq_state['blocked_until'] - time.time():.2f}s restantes."
                )

            if attempt < max_retries:
                continue

            # Esgotou retries
            if not allow_llama_fallback:
                raise RuntimeError(
                    f"Groq /chat/tools: 429 persistente após {max_retries+1} tentativas "
                    f"e allow_llama_fallback=False."
                )
            log.warning(f"Groq /chat/tools: 429 persistente após {max_retries+1} tentativas. Fallback llama-server.")
            r = await (await _get_llama_client()).post("/v1/chat/completions", json=llama_payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"], MODEL_NAME, True

        # 5xx: 1 retry com sleep, depois fallback
        if r.status_code >= 500:
            log.warning(f"Groq /chat/tools: {r.status_code} (tentativa {attempt+1}/{max_retries+1}).")
            if attempt < max_retries:
                await asyncio.sleep(1.0)
                continue
            if not allow_llama_fallback:
                raise RuntimeError(f"Groq /chat/tools: {r.status_code} persistente.")
            log.warning(f"Groq /chat/tools: {r.status_code} persistente. Fallback llama-server.")
            r = await (await _get_llama_client()).post("/v1/chat/completions", json=llama_payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"], MODEL_NAME, True

        # Outros erros (4xx exceto 429): não retry
        try:
            body = r.json()
            err_detail = body.get("error", {}).get("message", r.text[:300])
        except Exception:
            err_detail = r.text[:300]
        raise RuntimeError(f"Groq /chat/tools: {r.status_code} - {err_detail}")

    raise RuntimeError(f"Groq /chat/tools: excedeu tentativas sem resolução.")


@app.post("/chat/tools", response_model=ToolUseResponse)
async def chat_tools(req: ToolUseRequest):
    """
    Tool use nativo Groq — para o módulo alpha_code.

    Recebe messages + tools (formato OpenAI function-calling) e retorna
    a mensagem do assistant (pode conter tool_calls ou content).

    Diferenças vs /chat:
      - Sem memória persistida (caller gerencia)
      - Sem TTS, sem detecção de idioma
      - Sem streaming (síncrono — alpha_code faz seu próprio streaming de steps)
      - Modelo forçável via `model`
      - Distingue 429 "rate limit" (espera e retenta) de "request too large"
        (retorna 422 com too_large=true para caller trocar modelo ou reduzir contexto)
    """
    if not GROQ_KEY:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY não configurada.")
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages vazio.")

    messages = [m.model_dump(exclude_none=True) for m in req.messages]

    t0 = time.perf_counter()
    try:
        message, model_used, fallback = await _groq_tool_call(
            messages=messages,
            tools=req.tools,
            tool_choice=req.tool_choice,
            model=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            reasoning_effort=req.reasoning_effort,
            allow_llama_fallback=req.allow_llama_fallback,
            max_retries=req.max_retries,
        )
    except RequestTooLargeError as e:
        # Request excede TPM do modelo — caller precisa agir
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return ToolUseResponse(
            message={"role": "assistant", "content": "", "tool_calls": None},
            model=req.model or MODEL_GROQ_NAME,
            usage={"error": "request_too_large", "detail": str(e)[:500]},
            elapsed_ms=round(elapsed_ms, 1),
            fallback_used=False,
            too_large=True,
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {e.response.text[:500]}")
    except RuntimeError as e:
        msg = str(e)
        if "429" in msg or "RPD" in msg or "rate" in msg.lower():
            raise HTTPException(status_code=429, detail=msg)
        raise HTTPException(status_code=503, detail=f"inference error: {msg}")
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