"""
AVA — LLM Inference API (OpenRouter Edition)
=============================================
REST API para inferência conversacional com:
  - Backend 100% via OpenRouter (sem llama-server local)
  - Modelo principal: z-ai/glm-5.3-flash
  - Modelo extrator de memórias: env MEMORY_EXTRACTOR_MODEL (com fallback p/ MAIN_MODEL)
  - Integração com API de Memória via MCP (LT + ST via session_id)
  - Integração com API de TTS (localhost:3004)
  - Streaming de texto + disparo paralelo de áudio
  - Histórico de chat persistido via servidor MCP de memória externo
  - Detecção de idioma para resposta automática
  - EXTRAÇÃO AUTOMÁTICA DE MEMÓRIAS de longo prazo:
      Após cada dupla pergunta-resposta, um modelo extractor
      dedicado identifica informações úteis (esquecíveis ou não,
      episódicas ou semânticas) usando JSON schema estruturado.

OPTIMIZATIONS:
  1. Clients/sessões persistentes — sem handshake de transporte por
     request (httpx para OpenRouter/TTS, sessão MCP para a memória)
  2. Stable system prompt prefix — habilita prompt caching no provedor
  3. Memory read paralelo com construção de prompt (saves ~100-300ms)
  4. Connection pooling — keep-alive para OpenRouter e TTS; sessão MCP
     única e persistente para a Memória
  5. Extração de memórias fire-and-forget em background

Porta: localhost:4003
"""
import sys
import json
import datetime
import asyncio
import time
import math
import difflib
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

# Memória agora é acessada via API REST (memory_api.py — FastAPI sobre
# memory.py), não mais via iceoryx2/MCP. Client httpx persistente, criado
# lazy na primeira chamada e reutilizado pelo resto do processo.
MEMORY_API_URL = os.getenv("MEMORY_API_URL", "http://127.0.0.1:3000")
TTS_URL    = "http://localhost:3004"

# OpenRouter API
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")

# Modelos (fixos — não há mais servido local via llama-server)
MAIN_MODEL  = "z-ai/glm-5.3-flash"
# Extrator de memórias pós-turno. Env-configurável: modelos free/pequenos
# frequentemente NÃO suportam o `response_format: json_schema` do OpenRouter
# e retornam 200 com `content` vazio — a cadeia de tentativas do extrator
# (ver _extract_and_save_memories) cai para: (1) extrator sem response_format,
# (2) modelo de fallback (default = MAIN_MODEL).
MEMORY_EXTRACTOR_MODEL = os.getenv(
    "MEMORY_EXTRACTOR_MODEL", "meta-llama/llama-3.1-8b-instruct",
)
# "" (default) = usa MAIN_MODEL como último recurso da cadeia.
MEMORY_EXTRACTOR_FALLBACK_MODEL = os.getenv("MEMORY_EXTRACTOR_FALLBACK_MODEL", "")

# Contexto de curto prazo: quantas duplas pergunta-resposta (turn groups)
# são lidas cruas do /read_st a cada turno, em paralelo com o /read semântico.
ST_CONTEXT_PAIRS = 5

# Recall de chat: overrides de orçamento enviados ao /read (ReadRequest do
# memory_api). O default server-side é conservador (3 entradas / 2400 chars) —
# o recall de conversação pede um pouco mais de contexto, clampeado server-side.
CHAT_RECALL_TOP_K_FINAL     = int(os.getenv("CHAT_RECALL_TOP_K_FINAL", "6"))
CHAT_RECALL_TOTAL_MAX_CHARS = int(os.getenv("CHAT_RECALL_TOTAL_MAX_CHARS", "3200"))

# ─────────────────────────────────────────────────────────────
#   RECUPERAÇÃO CONTÍNUA DE MEMÓRIA (confidence-watch + prefill)
# ─────────────────────────────────────────────────────────────
# Ideia: durante o streaming, acompanhamos o logprob médio dos tokens dentro
# de uma "janela de frase" (do início da frase até o próximo `.`/`!`/`?`/`\n`).
# Se a confiança média da janela cair abaixo de um limiar, a geração é
# INTERROMPIDA (fechamos a conexão de streaming atual), o texto da janela +
# algumas linhas anteriores de contexto são enviados como query para
# `memory_read` (mesma API de memória usada no resto do arquivo). Se vier
# alguma evidência relevante o suficiente, um modelo pequeno classifica a
# relação evidência↔frase interrompida (confirmação/complementação/correção
# pequena/contradição forte/erro factual/mudança de raciocínio) e essa
# classificação escolhe o prefill de retomada. A retomada é uma NOVA
# requisição a /chat/completions cuja última mensagem é role="assistant"
# com o texto já gerado + o bloco <MEMORY_CORRECTION> (evidência embutida) +
# o prefill escolhido — o modelo "continua" a própria fala a partir dali
# (o bloco de correção nunca é emitido ao usuário via SSE; só o prefill e o
# texto gerado depois dele são).
#
# Ativado por request via `watch_confidence=true` no ChatRequest — é opt-in
# porque adiciona latência (round-trips extras de memory_read/classificação)
# e só funciona no modo CHAT (precisa de session_id e de uma query semântica
# com sentido; no modo TOOLS o caller já gerencia seu próprio histórico).

CONFIDENCE_THRESHOLD        = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
CONFIDENCE_MIN_TOKENS       = int(os.getenv("CONFIDENCE_MIN_TOKENS", "6"))
CONFIDENCE_MAX_CORRECTIONS  = int(os.getenv("CONFIDENCE_MAX_CORRECTIONS", "2"))
CONFIDENCE_CONTEXT_CHARS    = int(os.getenv("CONFIDENCE_CONTEXT_CHARS", "400"))
MEMORY_MATCH_MIN_SIMILARITY = float(os.getenv("MEMORY_MATCH_MIN_SIMILARITY", "0.28"))
# Probe do vigia: após N chunks de conteúdo sem NENHUM logprob, assume que o
# provedor não suporta logprobs e desativa/avisa (o vigia antes falhava em
# silêncio — window nunca acumulava e o memory_read de correção nunca rodava).
WATCH_LP_PROBE_CHUNKS       = int(os.getenv("WATCH_LP_PROBE_CHUNKS", "30"))

_SENTENCE_END_RE = re.compile(r'[.!?…]["\')\]]?\s*$|\n\s*$')

# Tipo de recuperação → prefill de retomada.
_CORRECTION_PREFILL: dict[str, str] = {
    "confirmacao":        "",                          # nada — apenas confirma, segue normalmente
    "complementacao":     "",                           # nada — soma informação, segue normalmente
    "correcao_pequena":   "Na verdade, ",
    "contradicao_forte":  "Espera, na verdade, ",
    "erro_factual":       "Corrigindo o que eu disse: ",
    "mudanca_raciocinio": "Pensando melhor, ",
}

# Injetado (só no payload enviado ao modelo, nunca no SSE) para dar ao
# modelo o mesmo protocolo de interpretação descrito no prompt original.
_MEMORY_CORRECTION_TEMPLATE = (
    "<MEMORY_CORRECTION>\n"
    "Sua geração foi interrompida porque surgiu evidência relevante.\n\n"
    "A evidência pode:\n"
    "1. confirmar sua hipótese;\n"
    "2. adicionar informação;\n"
    "3. contradizer sua hipótese.\n\n"
    "Evidência recuperada da memória:\n{evidencia}\n\n"
    "Classificação automática: {classificacao}.\n\n"
    "Revise o trecho que você estava produzindo.\n"
    "Se necessário, reconheça a correção naturalmente antes de continuar.\n"
    "</MEMORY_CORRECTION>"
)

_CORRECTION_CLASSIFICATION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "classification": {
            "type": "string",
            "enum": list(_CORRECTION_PREFILL.keys()),
            "description": (
                "confirmacao: a evidência só confirma o que já estava sendo dito. "
                "complementacao: adiciona informação nova, sem conflito. "
                "correcao_pequena: corrige um detalhe menor. "
                "contradicao_forte: contradiz diretamente a frase. "
                "erro_factual: a frase contém um erro factual importante. "
                "mudanca_raciocinio: exige mudar toda a linha de raciocínio."
            ),
        },
    },
    "required": ["classification"],
    "additionalProperties": False,
}

_MEMORY_CORRECTION_CLASSIFIER_SYSTEM = (
    "Você classifica a relação entre uma frase que um assistente estava "
    "gerando (foi interrompida no meio) e uma evidência recuperada da "
    "memória de longo prazo dele. Responda apenas com a classificação, "
    "usando EXATAMENTE uma destas categorias:\n"
    "- confirmacao: a evidência só confirma o que já estava sendo dito;\n"
    "- complementacao: a evidência adiciona informação nova, sem conflito;\n"
    "- correcao_pequena: a evidência corrige um detalhe menor (data, número, nome);\n"
    "- contradicao_forte: a evidência contradiz diretamente a frase;\n"
    "- erro_factual: a frase contém um erro factual importante segundo a evidência;\n"
    "- mudanca_raciocinio: a evidência exige mudar toda a linha de raciocínio, "
    "não só um detalhe."
)


class _ConfidenceWindow:
    """Acumula tokens + logprobs desde o início da frase atual até o
    próximo terminador de frase (`.`, `!`, `?`, `…`, quebra de linha).
    `avg_confidence()` converte cada logprob em probabilidade (exp(logprob))
    e tira a média — uma proxy simples de "quão confiante o modelo estava,
    em média, gerando esta frase"."""

    __slots__ = ("text", "logprobs")

    def __init__(self):
        self.text = ""
        self.logprobs: list[float] = []

    def add(self, token_text: str, logprob: Optional[float]) -> None:
        self.text += token_text or ""
        if logprob is not None:
            self.logprobs.append(logprob)

    def sentence_ready(self) -> bool:
        """True quando a janela já acumulou tokens suficientes E terminou
        numa fronteira de frase — só então vale a pena calcular a média
        (janelas muito curtas têm confiança ruidosa)."""
        return len(self.logprobs) >= CONFIDENCE_MIN_TOKENS and bool(_SENTENCE_END_RE.search(self.text))

    def avg_confidence(self) -> float:
        if not self.logprobs:
            return 1.0
        probs = [math.exp(lp) for lp in self.logprobs]
        return sum(probs) / len(probs)

    def reset(self) -> None:
        self.text = ""
        self.logprobs = []


def _best_matching_memory(window_text: str, results: list[dict]) -> Optional[dict]:
    """Entre os resultados de `memory_read` (já rankeados semanticamente
    por memory.py — embeddings + PPR/MMR), escolhe qual melhor corresponde
    ao trecho que estava sendo gerado. Usa o `score` semântico devolvido
    pela própria memória como sinal primário, e um SequenceMatcher
    (difflib, leve e sem dependência nova) como sinal secundário de
    sobreposição lexical — evita agir sobre matches semânticos fracos que
    não têm nenhuma relação textual perceptível com a frase interrompida.
    Retorna None se nada bater um limiar mínimo de confiança combinada."""
    if not results:
        return None
    best, best_val = None, -1.0
    for item in results:
        text = item.get("text") or item.get("content") or ""
        if not text:
            continue
        score = float(item.get("score") or item.get("relevance") or 0.0)
        overlap = difflib.SequenceMatcher(None, window_text.lower(), text.lower()).ratio()
        combined = 0.7 * score + 0.3 * overlap
        if combined > best_val:
            best_val, best = combined, {**item, "_match_score": combined}
    if best is None or best_val < MEMORY_MATCH_MIN_SIMILARITY:
        return None
    return best


async def _classify_memory_correction(previous_context: str, window_text: str, evidence_text: str) -> str:
    """Chama um modelo pequeno (mesmo MEMORY_EXTRACTOR_MODEL usado na
    extração pós-resposta) forçado por JSON schema a classificar a relação
    entre a frase interrompida e a evidência recuperada. Em qualquer falha
    (sem API key, timeout, resposta inválida), cai em "complementacao" —
    o fallback mais seguro (prefill vazio, geração só é enriquecida com o
    contexto extra, sem afirmar uma correção que não foi confirmada)."""
    if not OPENROUTER_API_KEY:
        return "complementacao"
    payload = {
        "model": MEMORY_EXTRACTOR_MODEL,
        "messages": [
            {"role": "system", "content": _MEMORY_CORRECTION_CLASSIFIER_SYSTEM},
            {"role": "user", "content": (
                f"Contexto anterior: {previous_context}\n\n"
                f"Frase sendo gerada (interrompida): {window_text}\n\n"
                f"Evidência recuperada: {evidence_text}"
            )},
        ],
        "temperature": 0.0,
        "max_tokens": 40,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name":   "memory_correction_classification",
                "strict": True,
                "schema": _CORRECTION_CLASSIFICATION_SCHEMA,
            },
        },
    }
    try:
        client = await _get_openrouter_client()
        r = await client.post("/chat/completions", json=payload, timeout=httpx.Timeout(8.0, connect=5.0))
        if r.status_code != 200:
            log.info(f"[MEMORY-WATCH] Classificador: {r.status_code} — {r.text[:200]}")
            return "complementacao"
        content = r.json()["choices"][0]["message"]["content"]
        cls = json.loads(content).get("classification", "complementacao")
        return cls if cls in _CORRECTION_PREFILL else "complementacao"
    except Exception as e:
        log.info(f"[MEMORY-WATCH] Falha na classificação da correção: {e}")
        return "complementacao"

# ─────────────────────────────────────────────────────────────
#          OPTIMIZATION 1: PERSISTENT HTTP CLIENTS
# ─────────────────────────────────────────────────────────────
# Em vez de criar um httpx.AsyncClient novo a cada request (custo de
# ~50-150ms de handshake TCP + HTTP upgrade), criamos uma vez na
# inicialização e reutilizamos em todas as requests. Essa é a maior
# otimização de latência para TTFT.

_openrouter_client: httpx.AsyncClient | None = None
_tts_http: httpx.AsyncClient | None = None
_memory_http: httpx.AsyncClient | None = None

# ── Memória via memory_api.py (REST) ──
# Client httpx persistente e lazy, com pooling/keep-alive, igual ao padrão
# usado para OpenRouter/TTS. Ver _get_memory_client().


async def _get_openrouter_client() -> httpx.AsyncClient:
    """Persistent client para OpenRouter — connection pooling + keep-alive."""
    global _openrouter_client
    if _openrouter_client is None or _openrouter_client.is_closed:
        if not OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY não configurada. Defina a variável de "
                "ambiente OPENROUTER_API_KEY (arquivo .env ou shell)."
            )
        _openrouter_client = httpx.AsyncClient(
            base_url=OPENROUTER_BASE_URL,
            timeout=httpx.Timeout(9999999.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=6,
                keepalive_expiry=60.0,       # Mantém conexões quentes por 60s
            ),
            headers={
                "Content-Type":   "application/json",
                "Authorization":   f"Bearer {OPENROUTER_API_KEY}",
                # OpenRouter usa esses cabeçalhos para ranking/atribuição
                "HTTP-Referer":    "http://localhost:4003",
                "X-Title":         "AVA-LLM",
            },
        )
    return _openrouter_client


def _get_memory_client() -> httpx.AsyncClient:
    """Persistent client para a API REST de memória (memory_api.py) —
    connection pooling + keep-alive, mesmo padrão do client OpenRouter/TTS."""
    global _memory_http
    if _memory_http is None or _memory_http.is_closed:
        _memory_http = httpx.AsyncClient(
            base_url=MEMORY_API_URL,
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )
    return _memory_http


# Mapeia o antigo nome de "tool" (era usado no protocolo MCP/iceoryx2) para
# (método HTTP, rota) das rotas REST expostas em memory_api.py. Rotas com
# parâmetro de path (ex.: /session/{session_id}) usam esse nome de argumento
# como placeholder — o valor é retirado de `arguments` e removido do body.
_MEMORY_TOOL_ROUTES: dict[str, tuple[str, str]] = {
    "memory_read":               ("POST",   "/read"),
    "memory_read_short_term":    ("POST",   "/read-short-term"),
    "memory_write_short_term":   ("POST",   "/write-short-term"),
    "memory_write":               ("POST",   "/write"),
    "memory_write_batch":        ("POST",   "/write/batch"),
    "memory_status":             ("GET",    "/status"),
    "memory_clear_session":      ("DELETE", "/session/{session_id}"),
}


async def _call_memory_tool(name: str, arguments: dict) -> dict:
    """Chama uma "tool" de memória via a API REST exposta por memory_api.py
    (que por sua vez delega 100% para memory.py). Mantém a MESMA assinatura
    (name, arguments) do antigo `_call_memory_tool` baseado em iceoryx2/MCP —
    nenhum call site precisou mudar. Erros de negócio (HTTP 400, mapeados de
    MemoryToolError pelo exception handler de memory_api.py) chegam como
    RuntimeError, igual antes."""
    if name not in _MEMORY_TOOL_ROUTES:
        raise RuntimeError(f"Tool de memória desconhecida: {name!r}")
    method, route = _MEMORY_TOOL_ROUTES[name]

    body = dict(arguments)
    if "{" in route:
        route = route.format(**{k: body.pop(k) for k in list(body) if f"{{{k}}}" in route})

    client = _get_memory_client()
    try:
        if method == "GET":
            r = await client.get(route, params=body or None)
        elif method == "DELETE":
            r = await client.delete(route, params=body or None)
        else:
            r = await client.post(route, json=body)
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
        raise RuntimeError(f"API de memória inacessível ({MEMORY_API_URL}): {e}") from e

    if r.status_code == 400:
        try:
            detail = r.json().get("error", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"memory_api: {detail}")
    r.raise_for_status()
    return r.json() if r.content else {}


async def _close_memory_session():
    """Encerra o client HTTP persistente da memória. Mantido com esse nome
    por compatibilidade com o startup/shutdown existentes."""
    global _memory_http
    if _memory_http and not _memory_http.is_closed:
        await _memory_http.aclose()


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


# ════════════════════════════════════════════════════════════════════════════
# Inferência 100% via OpenRouter — modelo principal fixo (z-ai/glm-5.3-flash).
# Para o extractor de memórias, modelo dedicado (env MEMORY_EXTRACTOR_MODEL).
#
# /chat/stream é o endpoint UNIFICADO: streaming (SSE) + tool calling nativo
# no mesmo request. Dois modos:
#   - modo CHAT (`message`): memória LT+ST, TTS, extração de memórias;
#   - modo TOOLS (`messages` + `tools`): loop ReAct gerenciado pelo caller —
#     as tool_calls são emitidas em evento único ao FINAL do stream.
# /chat/tools permanece como wrapper SÍNCRONO de compatibilidade.
# ════════════════════════════════════════════════════════════════════════════


def _reasoning_payload_extra(thinking_depth: int) -> dict:
    """
    Traduz `thinking_depth` (0-10) no formato `reasoning: {effort: ...}`
    aceito pela API OpenRouter para modelos que suportam reasoning.

    Para depth=0 (padrão), nenhum override é enviado — deixa o comportamento
    nativo do template agir.

    Mapeamento (aproximado, melhor-esforço):
      depth 0     → sem override
      depth 1-3   → reasoning.effort = "low"
      depth 4-7   → reasoning.effort = "medium"
      depth 8-10  → reasoning.effort = "high"

    Se o modelo não suportar reasoning, o provedor deve ignorar o campo.
    """
    if thinking_depth <= 0:
        return {}
    if thinking_depth <= 3:
        effort = "low"
    elif thinking_depth <= 7:
        effort = "medium"
    else:
        effort = "high"
    return {"reasoning": {"effort": effort}}


async def _execute_inference(json_payload: dict, stream: bool = False, model: str = MAIN_MODEL):
    """
    Executa a inferência contra a API OpenRouter.

    `stream=True` retorna um context manager pronto para iteração SSE.
    `stream=False` retorna a resposta síncrona completa.
    """
    client = await _get_openrouter_client()
    messages = json_payload.get("messages", [])
    payload = {**json_payload, "model": model, "stream": stream}

    if stream:
        ctx = client.stream("POST", "/chat/completions", json=payload)
        return client, model, ctx
    else:
        r = await client.post("/chat/completions", json=payload)
        return client, model, r


# ─────────────────────────────────────────────────────────────
#          INFERÊNCIA SÍNCRONA COM RETRY (falhas transitórias)
# ─────────────────────────────────────────────────────────────
# Único backend: OpenRouter. Pequeno retry para 5xx ou erro de rede
# transitório. Erros 429 (rate limit) respeitam Retry-After quando presente.

OPENROUTER_MAX_RETRIES   = 2
OPENROUTER_RETRY_BACKOFF = 1.0


async def _openrouter_post_with_retry(
    json_payload: dict,
    stream: bool = False,
    model: str = MAIN_MODEL,
):
    """
    Executa POST no /chat/completions do OpenRouter, com um pequeno retry
    em caso de 5xx / 429 / erro de rede transitório.

    Retorna tupla: (client, model_name, response_obj | stream_ctx).
    """
    client = await _get_openrouter_client()
    messages = json_payload.get("messages", [])
    payload = {**json_payload, "model": model, "stream": stream}

    if stream:
        # No streaming, devolvemos o context manager — 5xx/erro de rede
        # dentro do stream é tratado por quem consome (ver /chat/stream).
        ctx = client.stream("POST", "/chat/completions", json=payload)
        return client, model, ctx

    last_error: Optional[Exception] = None
    for attempt in range(OPENROUTER_MAX_RETRIES + 1):
        try:
            r = await client.post("/chat/completions", json=payload)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            last_error = e
            log.warning(
                f"OpenRouter: erro de rede ({type(e).__name__}), "
                f"tentativa {attempt + 1}/{OPENROUTER_MAX_RETRIES + 1}."
            )
            if attempt < OPENROUTER_MAX_RETRIES:
                await asyncio.sleep(OPENROUTER_RETRY_BACKOFF)
                continue
            raise RuntimeError(f"OpenRouter inacessível: {e}") from e

        # 429: rate limit — respeita Retry-After se presente
        if r.status_code == 429 and attempt < OPENROUTER_MAX_RETRIES:
            retry_after = r.headers.get("Retry-After")
            wait_s = float(retry_after) if retry_after else OPENROUTER_RETRY_BACKOFF
            log.warning(
                f"OpenRouter: 429 (rate limit). Aguardando {wait_s}s "
                f"antes de tentar {attempt + 2}/{OPENROUTER_MAX_RETRIES + 1}."
            )
            await asyncio.sleep(wait_s)
            continue

        if r.status_code >= 500 and attempt < OPENROUTER_MAX_RETRIES:
            log.warning(
                f"OpenRouter: {r.status_code} (tentativa {attempt + 1}/{OPENROUTER_MAX_RETRIES + 1})."
            )
            await asyncio.sleep(OPENROUTER_RETRY_BACKOFF)
            continue

        return client, model, r

    raise RuntimeError(
        f"OpenRouter: falhou após {OPENROUTER_MAX_RETRIES + 1} tentativa(s): {last_error}"
    )


# ─────────────────────────────────────────────────────────────
#          OPTIMIZATION 2: STABLE SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
# O prompt de sistema é mantido ESTÁVEL (não muda entre requests) para
# maximizar a taxa de prompt cache hits no provedor OpenRouter. O bloco
# dinâmico de memória vai DEPOIS do prefixo estável, em uma mensagem
# separada (role:assistant) — o custo de prefill é pequeno, mas o prefixo
# ainda aproveita o cache.

_SYSTEM_PROMPT_BASE = (
    f"{context}\n\n"
    f"O nome do usuário é {username}. "
    f"Responda sempre no idioma em que o usuário escrever.\n\n"
    # ── MEMORY SEMANTICS INSTRUCTION ─────────────────────────────────────────
    # Diz ao modelo COMO interpretar o bloco de memória injetado.
    # Sem isso, o modelo pode tratar o recall em voz de assistant como
    # uma resposta anterior em vez de conhecimento próprio recuperado.
    "Você possui um sistema de memória persistente. Antes de cada resposta, "
    "fragmentos relevantes da sua memória de longo prazo e do histórico recente "
    "são recuperados e apresentados em uma mensagem sua anterior nesta conversa. "
    "Trate essas informações como conhecimento próprio e utilize-as naturalmente, "
    "sem mencionar explicitamente que são 'memórias recuperadas' ou citar o "
    "mecanismo de memória ao usuário."
)


# ─────────────────────────────────────────────────────────────
#                     INTEGRAÇÃO: MEMÓRIA
# ─────────────────────────────────────────────────────────────

async def memory_read(
    query: str,
    session_id: Optional[str] = None,
    top_k: int = 10,
    top_k_final: Optional[int] = None,
    total_max_chars: Optional[int] = None,
) -> list[dict]:
    """
    Busca memórias relevantes (LT semântico + ST + knowledge + arquivos
    indexados) para o contexto da conversa, via a API REST de memória.

    `top_k_final`/`total_max_chars` são overrides de orçamento (server-side
    ReadRequest) — o default do /read é conservador (3 entradas / 2400 chars);
    o recall de chat pede um pouco mais. Server clampa aos tetos globais.
    """
    try:
        payload = {
            "query": query,
            "top_k": top_k,
            "min_score": 0.3,
            "session_id": session_id,
            "strategy": "auto",
        }
        if top_k_final:
            payload["top_k_final"] = top_k_final
        if total_max_chars:
            payload["total_max_chars"] = total_max_chars
        resp = await _call_memory_tool("memory_read", payload)
        return resp.get("results", [])
    except Exception as e:
        log.info(f"[MEMORY] Falha na leitura: {e}")
        return []


async def memory_read_st(session_id: Optional[str], n_pairs: int = ST_CONTEXT_PAIRS) -> list[dict]:
    """
    Lê as N duplas pergunta-resposta mais recentes do short-term (histórico
    cru da conversa, sem busca semântica) via a tool memory_read_short_term.
    Roda em paralelo com o memory_read semântico — juntos formam o contexto
    completo enviado ao modelo.
    """
    if not session_id:
        return []
    try:
        resp = await _call_memory_tool(
            "memory_read_short_term",
            {"session_id": session_id, "n_pairs": n_pairs},
        )
        return resp.get("turns", [])
    except Exception as e:
        log.info(f"[MEMORY] Falha na leitura ST (read_st): {e}")
        return []


async def memory_save_turn(session_id: str, user_input: str, assistant_response: str):
    """Grava o par de turnos na memória de curto prazo — fire-and-forget."""
    if not session_id:
        return
    try:
        await _call_memory_tool(
            "memory_write_short_term",
            {
                "session_id": session_id,
                "turns": [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": assistant_response},
                ],
            },
        )
    except Exception as e:
        log.info(f"[MEMORY] Falha ao salvar turno ST: {e}")


async def memory_write_st_episodic(session_id: str, text: str):
    """
    Grava uma memória EPISÓDICA na memória de curto prazo (ST) — fire-and-forget.

    Episódica = evento específico (algo que aconteceu em um momento). Como ST
    é organizada como turnos de conversa por session_id, gravamos o evento
    como um turno único role=assistant, sem um par user correspondente — isso
    evita poluir o histórico de chat com mensagens vazias do usuário.
    """
    if not session_id or not text:
        return
    try:
        await _call_memory_tool(
            "memory_write_short_term",
            {
                "session_id": session_id,
                "turns": [
                    {"role": "assistant", "content": text},
                ],
            },
        )
    except Exception as e:
        log.info(f"[MEMORY] Falha na escrita ST episódica: {e}")


async def _resolve_possible_update(payload: dict, data: dict) -> None:
    """Segue o protocolo de `reason="possible_update:<score>"` de um único
    resultado de `memory_write`/`memory_write_batch`: reenvia a escrita com
    action="update" e memory_id=candidate_id, efetivando a correção na
    memória candidata em vez de descartar o fato silenciosamente.

    Se possible_update vier sem candidate_id (resposta inconsistente do
    serviço de memória), o fato NÃO é criado como registro solto — por
    design, uma correção não deve virar um fato novo desancorado — mas o
    descarte é logado em WARNING para não passar despercebido. Compartilhado
    entre `memory_write_fact` (write único) e `memory_write_facts_batch`
    (write em lote), que têm exatamente a mesma lógica de retry."""
    reason = data.get("reason") or ""
    if not reason.startswith("possible_update:"):
        return

    candidate_id = data.get("candidate_id")
    text = payload.get("text", "")
    if candidate_id is None:
        log.warning(
            f"[MEMORY] possible_update sem candidate_id — fato descartado "
            f"sem reenvio (reason={reason!r}, text={text[:80]!r})"
        )
        return

    log.info(
        f"[MEMORY] possible_update — candidate_id={candidate_id}, "
        f"candidate_score={data.get('candidate_score')}, "
        f"candidate_text={data.get('candidate_text')!r}"
    )

    try:
        data2 = await _call_memory_tool(
            "memory_write",
            {**payload, "action": "update", "memory_id": candidate_id},
        )
    except Exception as e:
        log.info(f"[MEMORY] Falha no update LT (candidate_id={candidate_id}): {e}")
        return

    if not data2.get("stored"):
        log.info(
            f"[MEMORY] Update LT não efetivado (candidate_id={candidate_id}): "
            f"{data2.get('reason')}"
        )


async def memory_write_fact(
    text: str,
    source: str = "chat",
    confidence: float = 0.7,
    forgettable: bool = True,
    ttl_days: Optional[float] = None,
):
    """
    Grava UMA informação na memória de longo prazo (LT) — fire-and-forget.
    Para gravar várias memórias extraídas de uma mesma dupla
    pergunta-resposta, prefira `memory_write_facts_batch` (1 round-trip MCP
    em vez de N).

    A LT armazena apenas memórias SEMÂNTICAS (fatos/conhecimento geral sobre
    o usuário ou o mundo). Memórias EPISÓDICAS (eventos específicos da
    conversa) não entram aqui — elas vão para a ST via
    `memory_write_st_episodic`.

    Parâmetros:
      - forgettable: True se a memória pode decair com o tempo; False se é
        considerada permanente (ex.: nome do usuário, alergias).
      - ttl_days: meia-vida específica desta memória, em dias (None usa o
        default global do serviço de memória).
    """
    payload = {
        "text":         text,
        "source":       source,
        "confidence":   confidence,
        "forgettable":  forgettable,
        "ttl_days":     ttl_days,
    }
    try:
        data = await _call_memory_tool("memory_write", payload)
    except Exception as e:
        log.info(f"[MEMORY] Falha na escrita LT: {e}")
        return

    await _resolve_possible_update(payload, data)


async def memory_write_facts_batch(items: list[dict]) -> list[dict]:
    """
    Grava VÁRIAS memórias de longo prazo (LT) em uma única chamada MCP via
    `memory_write_batch` — usada pelo extrator (`_extract_and_save_memories`)
    para evitar N round-trips quando uma mesma dupla pergunta-resposta
    produz várias memórias semânticas.

    Cada item de `items` é um dict com as mesmas chaves de `memory_write_fact`
    (text, source, confidence, forgettable, ttl_days). Itens que caírem em
    "possible_update" recebem o mesmo reenvio com action="update" que
    `memory_write_fact` faz — só que aplicado individualmente APÓS o batch,
    já que o retry (achar/confirmar a memória candidata) é inerentemente
    por-item. Retorna a lista bruta de resultados (um dict por item, mesma
    ordem de `items`) para fins de logging/inspeção.
    """
    if not items:
        return []
    try:
        # sequential=True: fatos extraídos da MESMA dupla pergunta-resposta
        # chegam em ordem narrativa — o servidor cria a cadeia de arestas
        # temporal_precedence (anterior → seguinte) entre eles, então o
        # PPR consegue propagar contexto entre memórias do mesmo episódio
        # mesmo sem co-ativação futura.
        resp = await _call_memory_tool(
            "memory_write_batch", {"items": items, "sequential": True},
        )
    except Exception as e:
        log.info(f"[MEMORY] Falha na escrita LT em lote: {e}")
        return []

    results = resp.get("results", [])
    log.info(
        f"[MEMORY] write_batch: {resp.get('stored_count', 0)}/{resp.get('total', len(items))} "
        f"memórias gravadas em 1 round-trip"
    )
    for item, data in zip(items, results):
        await _resolve_possible_update(item, data)
    return results


# ─────────────────────────────────────────────────────────────
#              EXTRAÇÃO AUTOMÁTICA DE MEMÓRIAS (LT + ST)
# ─────────────────────────────────────────────────────────────
# Para cada dupla pergunta-resposta (considerando o assistant prefill e
# múltiplas requisições na mesma resposta — ou seja, usamos o user_input
# ORIGINAL e a resposta FINAL agregada, não importa quantas chamadas
# internas tenham ocorrido), um modelo extractor dedicado recebe o par
# e decide:
#   1. Quais informações (pergunta + resposta) valem gravar.
#   2. Se cada uma é "esquecível" ou não (aplica-se apenas às semânticas).
#   3. Se cada uma é "episódica" ou "semântica" — este campo define
#      o ROTEAMENTO da memória:
#        - semantic  → LT (memória de longo prazo, via /write)
#        - episodic  → ST (memória de curto prazo, via /write_st)
# A resposta é forçada via JSON schema estruturado para garantir parsing
# confiável.

_MEMORY_EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "description": (
                "Lista de memórias úteis extraídas da dupla pergunta-resposta. "
                "Pode ser vazia se nenhuma informação merecer ser gravada."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": (
                            "Texto autocontido da memória, redigido em terceira "
                            "pessoa. Ex.: 'O usuário chama-se João' ou "
                            "'Usuário preferiu Python para o novo projeto'."
                        ),
                    },
                    "forgettable": {
                        "type": "boolean",
                        "description": (
                            "true  → memória transitória, pode decair com o tempo "
                            "(ex.: projeto atual, tarefa em andamento). "
                            "false → memória permanente (ex.: nome, alergia, "
                            "preferência gastronômica estável). Aplicável apenas "
                            "a memórias semânticas (LT); para episódicas (ST) "
                            "ignore este campo, pois ST já é efêmera por natureza."
                        ),
                    },
                    "type": {
                        "type": "string",
                        "enum": ["episodic", "semantic"],
                        "description": (
                            "Define o ROTEAMENTO da memória: "
                            "semantic  → gravada na memória de longo prazo (LT) "
                            "como fato/conhecimento geral (ex.: 'usuário é "
                            "alérgico a amendoim'). "
                            "episodic  → gravada na memória de curto prazo (ST) "
                            "como evento específico do momento (ex.: 'em 2024 o "
                            "usuário viajou para Paris')."
                        ),
                    },
                },
                "required": ["content", "forgettable", "type"],
            },
        }
    },
    "required": ["memories"],
}


_MEMORY_EXTRACTOR_SYSTEM_PROMPT = (
    "Você é um extrator de memórias para um assistente conversacional chamado alfa. "
    "Com base na dupla PERGUNTA-RESPOSTA fornecida, identifique TODAS as informações "
    "presentes (tanto na pergunta do usuário quanto na resposta do assistente) que "
    "seriam úteis de serem gravadas sobre o usuário, o contexto da conversa, ou o "
    "mundo.\n\n"
    "Para cada memória identificada, classifique:\n"
    "  - type: 'semantic' se é um fato ou conhecimento geral (ex.: 'usuário sabe "
    "    programar em Python', 'usuário é alérgico a amendoim'). Essas memórias "
    "    vão para a memória de LONGO prazo (LT). 'episodic' se descreve um evento "
    "    específico (algo que aconteceu em um momento — ex.: 'em 2024 o usuário "
    "    viajou para Paris', 'hoje o usuário perguntou sobre X'). Essas memórias "
    "    vão para a memória de CURTO prazo (ST).\n"
    "  - forgettable: aplicável apenas a memórias semânticas (LT). true se é uma "
    "    informação transitória que pode perder relevância com o tempo (ex.: "
    "    tarefa atual, projeto do momento, humor atual); false se é permanente "
    "    (ex.: nome do usuário, alergias, profissão, preferências estáveis). Para "
    "    memórias episódicas (ST), preencha com true — ST é efêmera por natureza.\n\n"
    "Redija cada memória em terceira pessoa e de forma autocontida (sem referenciar "
    "esta instrução nem a conversa). Evite duplicar informações que já estão "
    "implícitas em outra memória da mesma resposta.\n\n"
    "Se nenhuma informação merecer ser gravada, retorne uma lista vazia."
)


async def _extract_and_save_memories(
    user_input: str,
    assistant_response: str,
    *,
    session_id: Optional[str] = None,
) -> list[dict]:
    """
    Roda o modelo extractor (MEMORY_EXTRACTOR_MODEL) sobre a dupla
    pergunta-resposta, força a saída em JSON schema estruturado, e roteia
    cada memória identificada:
      - semantic  → LT (memória de longo prazo, via /write)
      - episodic  → ST (memória de curto prazo, via /write_st)

    `user_input`        — texto ORIGINAL que o usuário enviou (não o prefill
                          nem um turno intermediário de tool use).
    `assistant_response`— texto FINAL agregado da resposta (mesmo que tenha
                          sido montado a partir de múltiplas requisições
                          internas: streaming, reasoning, múltiplos turnos
                          de tool use). Só chamamos o extractor UMA vez por
                          turno de conversa, sobre o produto final.

    Retorna a lista de memórias extraídas (para fins de logging/inspeção).
    """
    if not user_input or not assistant_response:
        return []
    if not OPENROUTER_API_KEY:
        log.info("[MEMORY-EXTRACT] OPENROUTER_API_KEY ausente — extração pulada.")
        return []

    messages_base = [
        {"role": "system", "content": _MEMORY_EXTRACTOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"=== PERGUNTA DO USUÁRIO ===\n{user_input}\n\n"
                f"=== RESPOSTA DO ASSISTENTE ===\n{assistant_response}\n\n"
                "=== TAREFA ===\n"
                "Com base nessa dupla pergunta-resposta, identifique se alguma "
                "informação presente (tanto na pergunta quanto na resposta) seria "
                "útil de ser gravada. Para cada uma, indique se é esquecível ou "
                "não, e se é episódica (→ ST) ou semântica (→ LT). "
                "Responda SOMENTE no formato JSON definido."
            ),
        },
    ]

    # ── Cadeia de tentativas ────────────────────────────────────────────────
    # BUG corrigido: o fluxo antigo só caía para o fallback quando o status
    # NÃO era 200. Modelos free/pequenos (liquid/lfm-2.5-2.6b:free etc.)
    # frequentemente retornam 200 com `message.content` VAZIO quando não
    # suportam `response_format: json_schema` — e o retorno silencioso
    # `return []` descartava o turno sem logar NADA (o bug visto em produção:
    # /memories/extract → 200, zero memórias, zero logs).
    # Cadeia atual: (1) extrator + json_schema; (2) extrator SEM response_format
    # (instrução JSON explícita no prompt); (3) modelo de fallback (default =
    # MAIN_MODEL) + json_schema. Cada passo loga; só os três vazios/falhos
    # desistem — e desistindo, desiste COM log.
    json_schema_fmt: dict = {
        "type": "json_schema",
        "json_schema": {
            "name":   "memory_extraction",
            "strict": True,
            "schema": _MEMORY_EXTRACTION_SCHEMA,
        },
    }
    _plain_json_instruction = (
        "\n\nFORMATO OBRIGATÓRIO — responda APENAS com um objeto JSON válido, "
        "sem texto antes ou depois e sem markdown:\n"
        '{"memories": [{"content": "<fato em 3ª pessoa>", '
        '"forgettable": true|false, "type": "semantic"|"episodic"}]}\n'
        "Se nada merecer ser gravado, responda exatamente: {\"memories\": []}"
    )
    fallback_model = MEMORY_EXTRACTOR_FALLBACK_MODEL or MAIN_MODEL
    attempts: list[tuple[str, Optional[dict], str]] = [
        (MEMORY_EXTRACTOR_MODEL, json_schema_fmt, ""),
        (MEMORY_EXTRACTOR_MODEL, None, _plain_json_instruction),
        (fallback_model, json_schema_fmt, ""),
    ]

    client = await _get_openrouter_client()
    content = ""
    last_err = ""
    for attempt_i, (model, fmt, suffix) in enumerate(attempts, start=1):
        fmt_label = fmt.get("type") if fmt else "sem response_format"
        msg_user = messages_base[-1]["content"] + suffix if suffix else messages_base[-1]["content"]
        payload: dict = {
            "model":       model,
            "messages":    messages_base[:-1] + [{"role": "user", "content": msg_user}],
            "temperature": 0.2,
            "max_tokens":  1024,
        }
        if fmt:
            payload["response_format"] = fmt
        try:
            r = await client.post("/chat/completions", json=payload)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            log.warning(
                f"[MEMORY-EXTRACT] tentativa {attempt_i}/{len(attempts)} "
                f"({model}, {fmt_label}) erro de rede: {last_err}"
            )
            continue

        if r.status_code != 200:
            last_err = f"HTTP {r.status_code}: {r.text[:300]}"
            log.warning(
                f"[MEMORY-EXTRACT] tentativa {attempt_i}/{len(attempts)} "
                f"({model}, {fmt_label}) não-200: {last_err}"
            )
            continue

        data = r.json()
        try:
            choice = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            last_err = f"resposta malformada: {str(data)[:300]}"
            log.warning(
                f"[MEMORY-EXTRACT] tentativa {attempt_i}/{len(attempts)} "
                f"({model}, {fmt_label}) {last_err}"
            )
            continue
        content = choice.get("content") or ""
        if content.strip():
            break  # tentativa bem-sucedida — content preenchido

        # 200 com content vazio — NÃO descartar em silêncio. Logar com
        # diagnóstico e tentar o próximo elo da cadeia.
        finish_reason = data["choices"][0].get("finish_reason")
        usage = json.dumps(data.get("usage", {}), ensure_ascii=False)[:200]
        last_err = "content vazio"
        log.warning(
            f"[MEMORY-EXTRACT] tentativa {attempt_i}/{len(attempts)} "
            f"({model}, {fmt_label}) 200 com content VAZIO "
            f"(finish_reason={finish_reason!r}, usage={usage}, "
            f"raw={r.text[:300]!r}) — tentando próximo fallback"
        )
        content = ""
    else:
        log.warning(
            f"[MEMORY-EXTRACT] todas as {len(attempts)} tentativas falharam "
            f"(último erro: {last_err or 'desconhecido'}) — memórias deste turno "
            "NÃO gravadas."
        )
        return []

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as je:
        # Tenta recuperar extraindo o primeiro bloco JSON da string.
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            log.info(f"[MEMORY-EXTRACT] JSON inválido: {je}. Raw: {content[:200]}")
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            log.info(f"[MEMORY-EXTRACT] Recovery falhou. Raw: {content[:200]}")
            return []

    memories_raw = parsed.get("memories", []) if isinstance(parsed, dict) else []

    # Roteia cada memória identificada:
    #   semantic  → LT (memória de longo prazo) — TODAS de uma vez via
    #               memory_write_batch (1 round-trip MCP em vez de N).
    #   episodic  → ST (memória de curto prazo, via /write_st) —
    #               continua individual (não existe write_st em lote).
    saved: list[dict] = []
    semantic_payloads: list[dict] = []
    for mem in memories_raw:
        if not isinstance(mem, dict):
            continue
        text = (mem.get("content") or "").strip()
        if not text:
            continue
        mem_type = mem.get("type") or "semantic"
        if mem_type not in ("episodic", "semantic"):
            mem_type = "semantic"
        forgettable = bool(mem.get("forgettable", True))

        if mem_type == "semantic":
            # LT: confidence mais alta para memórias não-esquecíveis —
            # elas tendem a ser fatos estáveis sobre o usuário e
            # merecem prioridade na busca.
            confidence = 0.9 if not forgettable else 0.6
            semantic_payloads.append({
                "text":        text,
                "source":      "chat:semantic",
                "confidence":  confidence,
                "forgettable": forgettable,
                "ttl_days":    None,
            })
        else:
            # episodic → ST (curto prazo). ST é efêmera por natureza,
            # então o forgetable do extractor é ignorado neste caso.
            asyncio.create_task(
                memory_write_st_episodic(session_id=session_id, text=text)
            )

        saved.append({
            "content":    text,
            "type":       mem_type,
            "target":     "LT" if mem_type == "semantic" else "ST",
            "forgettable": forgettable,
        })

    if semantic_payloads:
        # fire-and-forget: o /chat não espera a gravação terminar para
        # responder, mas as N memórias semânticas viram 1 chamada MCP.
        asyncio.create_task(memory_write_facts_batch(semantic_payloads))

    log.info(
        f"[MEMORY-EXTRACT] {len(saved)} memória(s) extraída(s) "
        f"(session={session_id}, modelo={model})."
    )
    return saved


# ─────────────────────────────────────────────────────────────
#                      INTEGRAÇÃO: TTS
# ─────────────────────────────────────────────────────────────

_tts_queue: Optional[asyncio.Queue] = None

async def _tts_sender_worker():
    """
    Worker em background que envia sentenças para o TTS SEQUENCIALMENTE.
    Otimizado: usa HTTP client persistente.
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

# Tamanho mínimo de fragmento de ST p/ o dedup contra o histórico cru —
# substrings curtos demais causariam falsos positivos (saudações genéricas).
_ST_DEDUP_MIN_CHARS = 12


def _norm_fragments(text: str) -> list[str]:
    """Extrai fragmentos normalizados (lowercase, sem os prefixos [role]) da
    representação compacta de um hit de ST ('[user] xxx | [assistant] yyy'),
    para o dedup contra os turnos recentes do /read_st."""
    parts = re.split(r"\s*\|\s*", text)
    frags = []
    for p in parts:
        p = re.sub(r"^\[(?:user|assistant)\]\s*", "", p.strip(), flags=re.I)
        p = p.lower().strip()
        if p:
            frags.append(p)
    return frags


def _build_memory_recall(
    memories: list[dict],
    recent_turns: Optional[list[dict]] = None,
) -> str | None:
    """
    Formata o bloco de memórias como texto de recall em primeira pessoa.
    Retorna None se não houver memórias válidas.

    Rationale semântico:
      O bloco de memória é expresso como um turno ASSISTANT (recall em
      primeira pessoa) em vez de USER (injeção de dados externos). Assim
      o modelo trata a informação como conhecimento próprio que está
      recuperando, e não como instrução ou dado fornecido pelo usuário.

    NEW (type-aware): as entradas do /read vêm de 4 fontes com naturezas
    epistêmicas DIFERENTES — tratá-las como um bolo só faz o modelo citar
    documentos como se fossem memória pessoal. O bloco separa:
      * long_term primary  → autoconhecimento ("Lembro que...", 1ª pessoa);
      * long_term related  → associação indireta do grafo Hebbiano (PPR) —
        evidência mais fraca, marcada como "pode ser relevante";
      * short_term         → episódios de conversa; os que sobrepõem os
        turnos recentes do /read_st são DESCARTADOS (dedup — já entram
        como histórico literal, repetir no recall é token jogado fora);
      * knowledge/indexed_file → material de referência (chunks de KG-RAG e
        arquivos indexados) — apresentado como acervo consultado, não como
        memória vivida, com a fonte quando disponível.
    """
    if not memories:
        return None

    # Blob dos conteúdos dos turnos recentes (lowercase) para o dedup de ST.
    recent_blob = ""
    if recent_turns:
        recent_blob = " ".join(
            (t.get("content") or "").lower() for t in recent_turns if t.get("content")
        )

    lt_primary: list[str] = []
    lt_related: list[str] = []
    references: list[str] = []

    for m in memories:
        text = (m.get("text") or m.get("content") or "").strip()
        if not text:
            continue
        mtype = m.get("memory_type") or "long_term"
        match = m.get("match_type") or "primary"

        if mtype == "short_term":
            # Dedup contra o histórico cru: se QUALQUER fragmento com peso
            # suficiente (>= _ST_DEDUP_MIN_CHARS) do hit já aparece nos turnos
            # recentes, é o mesmo episódio chegando duas vezes (recall
            # semântico + /read_st). Hits de sessões antigas não têm os
            # fragmentos no blob e passam normalmente.
            frags = _norm_fragments(text)
            if recent_blob and any(
                f in recent_blob for f in frags if len(f) >= _ST_DEDUP_MIN_CHARS
            ):
                continue
            lt_primary.append(text)
            continue

        if mtype in ("knowledge", "indexed_file"):
            src = m.get("file_name") or m.get("source")
            references.append(text + (f" (fonte: {src})" if src else ""))
            continue

        if match == "related":
            lt_related.append(text)
        else:
            lt_primary.append(text)

    sections: list[str] = []
    if lt_primary:
        sections.append(
            "Memória (fatos e episódios que conheço):\n"
            + "\n".join(f"- {t}" for t in lt_primary)
        )
    if lt_related:
        sections.append(
            "Associações do grafo de memória (contexto indireto, pode ser relevante):\n"
            + "\n".join(f"- {t}" for t in lt_related)
        )
    if references:
        sections.append(
            "Material de referência no meu acervo:\n"
            + "\n".join(f"- {t}" for t in references)
        )

    if not sections:
        return None

    return (
        "Resgatando contexto relevante da minha memória antes de responder:\n\n"
        + "\n\n".join(sections)
        + "\n\nTenho isso em mente e vou usar esse conhecimento de forma natural na conversa."
    )


def _build_messages(
    user_input: str,
    lang: str,
    memories: list[dict],
    recent_turns: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Monta a lista de mensagens para o OpenRouter.

    Crítico para prompt caching:
      O prompt de sistema é ESTÁVEL (não muda em runtime). O provedor
      (OpenRouter/modelo subjacente) tipicamente mantém cache do prefixo;
      manter o system prompt idêntico maximiza cache hits.

    Estrutura (com memórias + turnos recentes):
      [0] system:     prompt ESTÁVEL — contexto + nome + instrução de memória
      [1] assistant:  recall de memória em 1ª pessoa (DINÂMICO, depois do prefixo)
      [2..N] user/assistant: últimas N duplas da conversa real — de /read_st
      [N+1] user:     "[Data: DD/MM/YYYY]\n{user_input}"

    Estrutura (sem memórias/turnos):
      [0] system:     prompt ESTÁVEL
      [1] user:       "[Data: DD/MM/YYYY]\n{user_input}"
    """
    messages = [
        # ESTÁVEL: porção que aproveita cache no provedor
        {"role": "system", "content": _SYSTEM_PROMPT_BASE},
    ]

    # DINÂMICO: recall de memória em voz do próprio assistant.
    # role=assistant para o modelo tratar como autoconhecimento.
    # recent_turns entra para o dedup: hits de ST que sobrepõem o histórico
    # cru não são repetidos dentro do recall.
    recall_text = _build_memory_recall(memories, recent_turns)
    if recall_text:
        messages.append({
            "role": "assistant",
            "content": recall_text,
        })

    # DINÂMICO: últimas N duplas de turnos reais (de /read_st),
    # inseridas como estão — é histórico literal, não recall associativo.
    for turn in (recent_turns or []):
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    # DINÂMICO: data atual + input do usuário
    today = datetime.datetime.now().strftime("%d/%m/%Y")
    messages.append({
        "role": "user",
        "content": f"[Data de hoje: {today}]\n{user_input}",
    })

    return messages



# ─────────────────────────────────────────────────────────────
#                           FASTAPI
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="AVA — LLM API", version="3.0.0")

# ── Schemas ───────────────────────────────────────────────────

class ToolCallMessage(BaseModel):
    role: str = Field(..., description="system | user | assistant | tool")
    content: Optional[str | list[dict]] = Field(
        default=None,
        description=(
            "Texto simples (str) na maioria dos casos. Também aceita o formato "
            "multi-parte do OpenAI vision (lista de blocos {type: text|image_url, ...}) "
            "para requests multimodais — repassado como está para o OpenRouter."
        ),
    )
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatRequest(BaseModel):
    # ── Modo CHAT: mensagem única — o servidor monta o prompt (memória + data) ──
    message: Optional[str] = Field(default=None, description="Mensagem do usuário (modo chat). Obrigatório quando `messages` não for enviado.")
    session_id: Optional[str] = Field(default="default", description="ID da sessão para memória de curto prazo")
    voice:   Optional[str] = Field(default=None, description="Voz TTS. None = usa padrão.")
    lang:    Optional[str] = Field(default=None, description="Idioma forçado. None = detectado.")
    max_turns: int = Field(default=10,  ge=1, le=40, description="Limite de contexto recuperado")
    tts: bool = Field(default=True, description="Dispara TTS após gerar resposta.")
    thinking_depth: int = Field(
        default=0, ge=0, le=10,
        description=(
            "Orçamento de raciocínio (melhor-esforço). Por padrão (0) nenhum "
            "override é enviado — usa o comportamento nativo do modelo. "
            "Valores 1-10 são mapeados para reasoning.effort = low/medium/high "
            "no payload OpenRouter (modelos que não suportam ignorarão o campo)."
        ),
    )
    stream_reasoning: bool = Field(default=True, description="Se True e o modelo emitir reasoning_content, envia para a UI. Se False, ignora completamente.")
    extract_memories: bool = Field(
        default=True,
        description=(
            "Se True, após gerar a resposta final, dispara em background o "
            "modelo extractor de memórias (MEMORY_EXTRACTOR_MODEL) "
            "para identificar informações úteis a serem gravadas em LT, "
            "rotulando-as como esquecíveis/permanentes e episódicas/semânticas."
        ),
    )
    model: Optional[str] = Field(default=None, description="Ignorado — mantido por compatibilidade. Sempre usa o modelo principal fixo (z-ai/glm-5.3-flash).")

    # ── Tool calling nativo (unificação com o antigo /chat/tools) ──────────
    messages: Optional[list[ToolCallMessage]] = Field(
        default=None,
        description=(
            "Modo TOOLS: mensagens prontas no formato OpenAI (system/user/assistant/"
            "tool, com tool_call_id nas resultados). Quando presente, o endpoint NÃO "
            "lê/grava memória nem dispara TTS — o caller gerencia o estado do loop "
            "ReAct. Mutuamente exclusivo com `message`."
        ),
    )
    tools: list[dict] = Field(
        default_factory=list,
        description=(
            "Tool schemas no formato OpenAI function-calling. O texto é streamado "
            "normalmente (eventos delta); como a decisão de tool call só acontece "
            "no FINAL da geração, as tool_calls completas são emitidas em um único "
            "evento SSE {'tool_calls': [...]} ao final da resposta (antes do done)."
        ),
    )
    tool_choice: Optional[str] = Field(
        default="auto",
        description="auto | required | none | {type:function,function:{name:...}}.",
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=2.0,
        description="None = usa o padrão do modo (chat: 0.7, tools: 0.3).",
    )
    max_tokens: Optional[int] = Field(
        default=None, ge=1, le=32000,
        description="None = não envia o campo (usa o padrão do provedor).",
    )
    reasoning_effort: Optional[str] = Field(
        default=None,
        description=(
            "Override direto de reasoning.effort no OpenRouter (low|medium|high|"
            "none). Tem precedência sobre `thinking_depth` quando definido."
        ),
    )
    watch_confidence: bool = Field(
        default=False,
        description=(
            "Se True (só no modo CHAT), ativa logprobs no OpenRouter e monitora "
            "a confiança média dos tokens por janela de frase. Quando a "
            "confiança cai abaixo de CONFIDENCE_THRESHOLD, a geração é "
            "interrompida, o trecho + contexto recente é enviado a memory_read, "
            "e — havendo evidência relevante o suficiente — um modelo pequeno "
            "classifica a relação (confirmação/complementação/correção "
            "pequena/contradição forte/erro factual/mudança de raciocínio) e "
            "a resposta é retomada via prefill de assistant apropriado. Opt-in: "
            "adiciona latência (memory_read + classificação a cada interrupção)."
        ),
    )

class ClearRequest(BaseModel):
    confirm: bool = False
    session_id: Optional[str] = "default"


# ── Schemas para tool use nativo (endpoint /chat/tools — COMPATIBILIDADE) ─────
# O /chat/tools permanece como wrapper SÍNCRONO para o orquestrador/alpha_code.
# A unificação (tools + streaming no mesmo request) vive no /chat/stream — os
# schemas (ToolCallMessage) estão definidos junto ao ChatRequest.

class ToolUseRequest(BaseModel):
    messages: list[ToolCallMessage]
    tools: list[dict] = Field(default_factory=list, description="Lista de tool schemas no formato OpenAI function-calling. IGNORADO quando `grammar` está presente — nesse caso o OpenRouter usa response_format json_object para restringir o output.")
    tool_choice: Optional[str] = Field(default="auto", description="auto | required | none | {type:function,function:{name:...}}. IGNORADO quando `grammar` está presente.")
    grammar: Optional[str] = Field(default=None, description="Mantido por compatibilidade com chamadores antigos do llama-server. No OpenRouter, quando presente, ativamos response_format=json_object (equivalente aproximado de GBNF). A resposta vem em message.content e o caller é responsável por parsear.")
    model: Optional[str] = Field(default=None, description="Ignorado — mantido por compatibilidade. Sempre usa o modelo principal fixo (z-ai/glm-5.3-flash).")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=32000)
    reasoning_effort: Optional[str] = Field(
        default=None,
        description=(
            "Passado direto para o OpenRouter como reasoning.effort. Padrão "
            "(None) = não envia o campo. Envie \"low\"|\"medium\"|\"high\" "
            "para modelos que suportam, ou \"none\" para desligar reasoning "
            "nesta chamada específica (use quando latência por passo importar "
            "mais que qualidade — ex.: loops rápidos do alpha_code)."
        ),
    )
    max_retries: int = Field(default=3, ge=0, le=10, description="Máximo de retries em falha transitória (5xx/rede/429) do OpenRouter antes de desistir.")
class ToolUseResponse(BaseModel):
    message: dict
    model: str
    usage: dict
    elapsed_ms: float
    fallback_used: bool = False
    too_large: bool = Field(default=False, description="True se request excedeu limite de contexto do modelo (caller deve reduzir contexto ou trocar modelo)")


# ── Lifecycle ────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Verifica a OPENROUTER_API_KEY e faz uma chamada de teste (memory_status)
    pra API REST de memória (memory_api.py), só pra logar conectividade —
    o client em si é lazy (ver _get_memory_client) e reconecta sozinho a
    qualquer momento, então uma falha aqui não impede a API de subir."""
    if not OPENROUTER_API_KEY:
        log.warning(
            "[STARTUP] OPENROUTER_API_KEY não configurada — inferência vai falhar "
            "até a variável ser definida."
        )
    else:
        log.info(f"[STARTUP] OpenRouter configurado. Modelo principal: {MAIN_MODEL}")

    try:
        await _call_memory_tool("memory_status", {})
        log.info(f"[MEMORY] API REST de memória OK ({MEMORY_API_URL}).")
    except asyncio.CancelledError as e:
        log.warning(
            f"[STARTUP] Checagem da memory_api cancelada durante o "
            f"startup: {e}. A API sobe mesmo assim."
        )
    except Exception as e:
        log.warning(
            f"[STARTUP] Memória (memory_api em {MEMORY_API_URL}) não acessível: {e}. "
            "A API sobe mesmo assim — chamadas de memória vão falhar (e "
            "tentar reconectar sozinhas) até o servidor de memória subir."
        )


@app.on_event("shutdown")
async def shutdown():
    """Fecha HTTP clients (OpenRouter, TTS e memória) graciosamente."""
    global _openrouter_client, _tts_http
    if _openrouter_client and not _openrouter_client.is_closed:
        await _openrouter_client.aclose()
    if _tts_http and not _tts_http.is_closed:
        await _tts_http.aclose()
    await _close_memory_session()


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Verifica se a API está no ar e se a chave OpenRouter está configurada."""
    return {
        "api":               "ok",
        "openrouter_key":    "ok" if OPENROUTER_API_KEY else "missing",
        "main_model":        MAIN_MODEL,
        "memory_extractor":  MEMORY_EXTRACTOR_MODEL,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Inferência síncrona — retorna a resposta completa em JSON.
    Otimizações:
      - HTTP client persistente para OpenRouter (sem TCP handshake)
      - System prompt estável (maximiza cache hits no provedor)
      - Memória read paralelo com detecção de idioma
      - Extração de memórias LT em background (fire-and-forget)
    """
    user_input = (req.message or "").strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Mensagem vazia.")

    # ── PARALLEL prep: detecção de idioma + leitura LT + leitura ST ────────
    lang_task = asyncio.ensure_future(
        asyncio.get_event_loop().run_in_executor(None, _safe_detect, user_input)
    )
    memory_task = asyncio.ensure_future(
        memory_read(
            user_input,
            session_id=req.session_id,
            top_k=req.max_turns,
            top_k_final=CHAT_RECALL_TOP_K_FINAL,
            total_max_chars=CHAT_RECALL_TOTAL_MAX_CHARS,
        )
    )
    memory_st_task = asyncio.ensure_future(
        memory_read_st(req.session_id)
    )

    lang, memories, recent_turns = await asyncio.gather(lang_task, memory_task, memory_st_task)
    if not lang:
        lang = "pt"

    # Monta prompt
    messages = _build_messages(user_input, lang, memories, recent_turns)
    log.info(f"tamanho do contexto do assistente: {str(messages).count(chr(0))} caracteres.")

    # Inferência (OpenRouter, com retry para falhas transitórias)
    t0 = time.perf_counter()
    log.info(messages)
    try:
        client_used, model_used, r = await _openrouter_post_with_retry(
            json_payload={
                "messages":    messages,
                "temperature": 0.7,
                **_reasoning_payload_extra(req.thinking_depth),
            },
            stream=False,
            model=MAIN_MODEL,
        )
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"openrouter error {e.response.status_code}: {e.response.text[:300]}",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"inference error: {e}")

    resp_data = r.json()
    response_text = resp_data["choices"][0]["message"]["content"]
    reasoning_text = resp_data["choices"][0]["message"].get("reasoning_content", "") or \
                      resp_data["choices"][0]["message"].get("reasoning", "")
    elapsed = time.perf_counter() - t0

    # Log cache stats (quando o provedor reportar)
    usage = resp_data.get("usage", {})
    cached = usage.get("prompt_tokens_cached", 0) or usage.get("cached_tokens", 0)
    total_prompt = usage.get("prompt_tokens", 0)
    log.info(
        f"[CHAT] {elapsed:.2f}s | {len(response_text)} chars | "
        f"reasoning: {len(reasoning_text)} chars | "
        f"prompt: {total_prompt} tokens (cached: {cached})"
    )

    # Gravar turno na memória de curto prazo (fire-and-forget)
    asyncio.create_task(
        memory_save_turn(req.session_id, user_input, response_text)
    )

    # ── EXTRAÇÃO DE MEMÓRIAS LT (fire-and-forget) ─────────────────────────
    # Dispara o modelo extractor sobre a dupla (user_input, response_text)
    # — isto é a "resposta final agregada" do assistente. Não importa se
    # internamente o modelo fez reasoning, tool calls, ou múltiplas
    # requisições: aqui só interessa o produto final visto pelo usuário.
    if req.extract_memories:
        asyncio.create_task(
            _extract_and_save_memories(user_input, response_text, session_id=req.session_id)
        )

    # Disparar TTS em background
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
    Endpoint UNIFICADO de inferência com streaming (SSE) — absorve o antigo
    /chat/tools, permitindo tool calling nativo e streaming no MESMO request:

      - Modo CHAT (envie `message`): o servidor monta o prompt completo
        (memória LT + ST, data, idioma) e cuida do pipeline inteiro —
        TTS em streaming, persistência do turno em ST e extração de
        memórias LT em background ao final.

      - Modo TOOLS (envie `messages`): mensagens prontas no formato OpenAI
        (com role="tool" para resultados), sem memória/TTS — o caller
        (orquestrador/agente) gerencia o histórico, executa as tools e
        re-chama este endpoint com os resultados (loop ReAct).

    TOOLS (opcional nos dois modos, via `tools`): o texto é streamado
    normalmente (eventos `delta`). Como a decisão de tool call só ocorre no
    FINAL da geração, os fragmentos de tool_calls que chegam no stream são
    apenas ACUMULADOS e emitidos UMA vez, no evento `{"tool_calls": [...]}`,
    logo antes do `done`. Se houver tool_calls, TTS/persistência são pulados
    (o turno não terminou — o caller executa as tools e continua o loop).

    Eventos SSE:
      {"reasoning": "..."}   — raciocínio (se stream_reasoning)
      {"delta": "..."}       — texto do content
      {"memory_correction": {"classification": ..., "evidence": "..."}}
                              — emitido quando `watch_confidence=true` e a
                                confiança média de uma janela de frase caiu
                                abaixo do limiar, disparando releitura da
                                memória e retomada com prefill (ver
                                CONFIDENCE_* no topo do arquivo)
      {"tool_calls": [...]}  — tool_calls completas (ao final, se houver)
      {"done": true, "elapsed": ..., "prompt_cached_tokens": ..., "had_tool_calls": bool}
      {"error": "...", "too_large": bool}  — aborta o stream
    """
    if req.messages is not None and (req.message or "").strip():
        raise HTTPException(
            status_code=400,
            detail="Envie apenas um dos campos: `message` (modo chat) ou `messages` (modo tools).",
        )

    # ── Resolução de modo ──────────────────────────────────────────────────
    tool_mode = req.messages is not None
    if tool_mode:
        messages = [m.model_dump(exclude_none=True) for m in req.messages]
        if not messages:
            raise HTTPException(status_code=400, detail="messages vazio.")
        user_input = None
        lang = None
        temperature = req.temperature if req.temperature is not None else 0.3
    else:
        user_input = (req.message or "").strip()
        if not user_input:
            raise HTTPException(
                status_code=400,
                detail="Envie `message` (modo chat) ou `messages` (modo tools).",
            )

        # ── PARALLEL prep: detecção de idioma + leitura LT + leitura ST ────
        lang_task = asyncio.ensure_future(
            asyncio.get_event_loop().run_in_executor(None, _safe_detect, user_input)
        )
        memory_task = asyncio.ensure_future(
            memory_read(
                user_input,
                session_id=req.session_id,
                top_k=req.max_turns,
                top_k_final=CHAT_RECALL_TOP_K_FINAL,
                total_max_chars=CHAT_RECALL_TOTAL_MAX_CHARS,
            )
        )
        memory_st_task = asyncio.ensure_future(
            memory_read_st(req.session_id)
        )
        lang, memories, recent_turns = await asyncio.gather(lang_task, memory_task, memory_st_task)
        if not lang:
            lang = "pt"

        messages = _build_messages(user_input, lang, memories, recent_turns)
        temperature = req.temperature if req.temperature is not None else 0.7

    log.info(f"tamanho do contexto do assistente: {str(messages).count(chr(0))} caracteres.")

    # ── Payload OpenRouter (com tool calling nativo quando `tools` presente) ──
    reasoning_extra = _reasoning_payload_extra(req.thinking_depth)
    if req.reasoning_effort:
        reasoning_extra = {"reasoning": {"effort": req.reasoning_effort}}

    payload: dict = {
        "messages":    messages,
        "temperature": temperature,
        **reasoning_extra,
    }
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens
    if req.tools:
        payload["tools"] = req.tools
        payload["tool_choice"] = req.tool_choice or "auto"

    # ── Confidence-watch: pede logprobs por token ao OpenRouter ────────────
    # Só faz sentido no modo CHAT (precisa de session_id + query semântica
    # com sentido de conversa; no modo TOOLS o caller já gerencia o histórico).
    watch_confidence = req.watch_confidence and not tool_mode
    if watch_confidence:
        payload["logprobs"] = True

    # TTS só existe no modo chat (modo tools é stateless — caller cuida do áudio)
    voice = req.voice or voiceModel
    use_tts = (not tool_mode) and bool(req.tts and voice)
    if use_tts:
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
        full_response = ""       # texto exposto ao usuário (sem os blocos <MEMORY_CORRECTION>)
        full_reasoning = ""
        tool_calls_acc: dict[int, dict] = {}   # index -> tool_call em montagem
        t0 = time.perf_counter()
        cached_tokens = 0

        # ── Confidence-watch: estado entre passes ──────────────────────────
        window = _ConfidenceWindow() if watch_confidence else None
        watch_events_used = 0   # conta interrupções (com ou sem correção efetiva)
        # ── Diagnóstico do vigia: um vigia sem logprobs é silenciosamente
        # inútil. Contamos se o provedor de fato devolve logprobs e avisamos
        # (probe + resumo final) quando não devolve.
        watch_lp_seen = 0       # total de logprobs recebidos
        watch_chunks_no_lp = 0  # chunks de conteúdo observados sem nenhum logprob
        current_messages = list(messages)   # cresce a cada correção (novo request)

        while True:
            interrupted_for_correction = False

            # Retry simples para falhas transitórias (5xx/rede/429) do OpenRouter
            for attempt in range(OPENROUTER_MAX_RETRIES + 1):
                client_used, model_used, r_ctx = await _execute_inference(
                    json_payload={**payload, "messages": current_messages},
                    stream=True,
                    model=MAIN_MODEL,
                )

                try:
                    async with r_ctx as r:
                        if r.status_code == 429 and attempt < OPENROUTER_MAX_RETRIES:
                            retry_after = r.headers.get("Retry-After")
                            wait_s = float(retry_after) if retry_after else OPENROUTER_RETRY_BACKOFF
                            log.warning(
                                f"OpenRouter stream: 429. Aguardando {wait_s}s "
                                f"(tentativa {attempt + 2}/{OPENROUTER_MAX_RETRIES + 1})."
                            )
                            await asyncio.sleep(wait_s)
                            continue
                        if r.status_code >= 500 and attempt < OPENROUTER_MAX_RETRIES:
                            log.warning(
                                f"OpenRouter stream: {r.status_code} "
                                f"(tentativa {attempt + 1}/{OPENROUTER_MAX_RETRIES + 1})."
                            )
                            await asyncio.sleep(OPENROUTER_RETRY_BACKOFF)
                            continue

                        if r.status_code >= 400:
                            # Erro definitivo (retries esgotados ou 4xx): reporta no
                            # stream — com a marcação too_large (mesmo contrato do
                            # /chat/tools) para o caller reduzir o contexto.
                            try:
                                err_body = (await r.aread()).decode("utf-8", errors="replace")
                            except Exception:
                                err_body = ""
                            log.warning(
                                f"OpenRouter stream: {r.status_code} — {err_body[:200]}"
                            )
                            yield f"data: {json.dumps({'error': f'openrouter {r.status_code}: {err_body[:400]}', 'too_large': _is_request_too_large_error(err_body)})}\n\n"
                            return

                        # Conexão aceita. Processa as linhas SSE:
                        async for line in r.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            data = line[len("data:"):].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                choices = chunk.get("choices") or []
                                if not choices:
                                    # Chunk final só com usage (sem choices)
                                    usage = chunk.get("usage", {})
                                    if usage:
                                        cached_tokens = usage.get("prompt_tokens_cached", 0) or \
                                                         usage.get("cached_tokens", 0)
                                    continue
                                delta_obj = choices[0].get("delta", {}) or {}

                                if req.stream_reasoning:
                                    reasoning = delta_obj.get("reasoning_content", "") or delta_obj.get("reasoning", "")
                                    if reasoning:
                                        full_reasoning += reasoning
                                        yield f"data: {json.dumps({'reasoning': reasoning})}\n\n"

                                # ── Fragments de tool_calls: apenas ACUMULA aqui.
                                # A "ativação" das tools acontece ao FINAL da
                                # resposta, num único evento — os deltas de text
                                # continuam fluindo normalmente durante a geração.
                                for tc in (delta_obj.get("tool_calls") or []):
                                    idx = tc.get("index", 0)
                                    entry = tool_calls_acc.setdefault(idx, {
                                        "id": "", "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    })
                                    if tc.get("id"):
                                        entry["id"] = tc["id"]
                                    if tc.get("type"):
                                        entry["type"] = tc["type"]
                                    fn = tc.get("function") or {}
                                    if fn.get("name"):
                                        entry["function"]["name"] += fn["name"]
                                    if fn.get("arguments"):
                                        entry["function"]["arguments"] += fn["arguments"]

                                content = delta_obj.get("content", "") or ""
                                if not content:
                                    # Alguns provedores enviam usage na última chunk
                                    usage = chunk.get("usage", {})
                                    if usage:
                                        cached_tokens = usage.get("prompt_tokens_cached", 0) or \
                                                         usage.get("cached_tokens", 0)
                                    continue

                                full_response += content
                                yield f"data: {json.dumps({'delta': content})}\n\n"

                                # TTS incremental apenas quando não há tools na
                                # jogada — com tools, o content pode ser texto
                                # intermediário (a tool_call vem no fim), então
                                # bufferizamos e só falamos se o turno for final.
                                if use_tts:
                                    _tts_buf += content
                                    if not req.tools:
                                        buf_rstrip = _tts_buf.rstrip()
                                        if (buf_rstrip and buf_rstrip[-1] in '.!?\n。') \
                                           or len(_tts_buf) > 150:
                                            _flush_tts_buf()

                                # ── VIGIA DE CONFIANÇA (logprobs) ───────────
                                # OpenRouter/OpenAI: choices[0].logprobs.content
                                # é uma lista de {token, logprob, ...} — 1 item
                                # por token do delta (normalmente 1 por chunk).
                                # Alguns provedores aninham em delta.logprobs —
                                # checamos os dois. Se NENHUM logprob chegar
                                # (provedor sem suporte), o vigia é desativado
                                # com aviso explícito — sem isso, ele falhava
                                # silenciosamente para sempre (window nunca
                                # acumula, sentence_ready nunca dispara).
                                if watch_confidence and window is not None \
                                   and watch_events_used < CONFIDENCE_MAX_CORRECTIONS:
                                    lp_obj = choices[0].get("logprobs") \
                                             or delta_obj.get("logprobs") or {}
                                    lp_content = lp_obj.get("content") or []
                                    if lp_content:
                                        watch_lp_seen += len(lp_content)
                                        for lp_item in lp_content:
                                            window.add(lp_item.get("token", ""), lp_item.get("logprob"))
                                    else:
                                        watch_chunks_no_lp += 1
                                        if watch_chunks_no_lp == WATCH_LP_PROBE_CHUNKS and not watch_lp_seen:
                                            log.warning(
                                                "[MEMORY-WATCH] "
                                                f"{WATCH_LP_PROBE_CHUNKS} chunks de conteúdo e nenhum "
                                                "logprob recebido — o provedor/modelo não devolve "
                                                "logprobs. Vigia de confiança desativado nesta "
                                                "resposta (memory_read de correção não vai disparar)."
                                            )
                                    if window.sentence_ready():
                                        conf = window.avg_confidence()
                                        if conf < CONFIDENCE_THRESHOLD:
                                            log.info(
                                                f"[MEMORY-WATCH] Confiança baixa ({conf:.2f} < "
                                                f"{CONFIDENCE_THRESHOLD}) na janela: {window.text!r}"
                                            )
                                            interrupted_for_correction = True
                                            break
                                        window.reset()
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue

                            if interrupted_for_correction:
                                break

                        # Se saiu do `async for line` por causa da interrupção,
                        # fechar o `async with r_ctx` aqui já encerra a conexão
                        # de streaming atual (é o "interromper a geração").
                        break   # sai do loop de retry — sucesso (interrompido ou não)

                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    return

            if not interrupted_for_correction:
                break   # resposta terminou normalmente — sai do while de passes

            # ── CORREÇÃO EM ANDAMENTO ────────────────────────────────────────
            # A conexão de streaming atual já foi fechada (saímos do `async
            # with`). Busca evidência, classifica, e monta um novo request
            # cuja última mensagem é role="assistant" com o texto já gerado +
            # o bloco <MEMORY_CORRECTION> (não emitido ao usuário) + o
            # prefill escolhido — o modelo "continua" a partir dali.
            watch_events_used += 1
            window_text = window.text
            prev_context = full_response[:-len(window_text)] if window_text and full_response.endswith(window_text) else full_response
            prev_context = prev_context[-CONFIDENCE_CONTEXT_CHARS:]
            query = f"{prev_context} {window_text}".strip()

            try:
                results = await memory_read(query, session_id=req.session_id, top_k=5)
            except Exception as e:
                log.info(f"[MEMORY-WATCH] Falha no memory_read de correção: {e}")
                results = []

            match = _best_matching_memory(window_text, results)
            window.reset()

            if match is None:
                # Nenhuma evidência relevante o suficiente — falso alarme.
                # Não há correção real: apenas reabre o streaming de onde
                # parou (prefill puro, sem bloco de correção), consumindo
                # o orçamento de watch_events_used para não repetir sem fim
                # sobre a mesma frase difícil.
                current_messages = messages + [{"role": "assistant", "content": full_response}]
                continue

            evidence_text = match.get("text") or match.get("content") or ""
            classification = await _classify_memory_correction(prev_context, window_text, evidence_text)
            prefill = _CORRECTION_PREFILL.get(classification, "")

            log.info(
                f"[MEMORY-WATCH] Correção: classificação={classification!r} "
                f"evidência={evidence_text[:120]!r}"
            )
            yield f"data: {json.dumps({'memory_correction': {'classification': classification, 'evidence': evidence_text[:300]}})}\n\n"

            correction_block = _MEMORY_CORRECTION_TEMPLATE.format(
                evidencia=evidence_text, classificacao=classification,
            )
            # O bloco <MEMORY_CORRECTION> só entra no payload enviado ao
            # modelo (dá contexto/instrução pra retomada) — nunca é
            # streamado ao usuário. Só o prefill (se houver) e o que vier
            # depois dele são emitidos como deltas normais.
            assistant_with_correction = f"{full_response}\n\n{correction_block}\n{prefill}"
            current_messages = messages + [{"role": "assistant", "content": assistant_with_correction}]

            if prefill:
                full_response += prefill
                yield f"data: {json.dumps({'delta': prefill})}\n\n"
                if use_tts:
                    _tts_buf += prefill
            # "confirmacao"/"complementacao": prefill vazio — a geração só
            # segue, agora com a evidência já incorporada ao contexto do
            # próximo pass (sem alegar uma correção que não houve).

            # volta ao topo do while: nova requisição de streaming, continuando

        # ── FINALIZAÇÃO: monta as tool_calls acumuladas ──────────────────────
        tool_calls_final = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
        turno_final = not tool_calls_final

        # Rotina de finalização (TTS e memória) — só em turno FINAL
        if use_tts and turno_final and _tts_buf.strip():
            _flush_tts_buf()

        elapsed = time.perf_counter() - t0
        log.info(
            f"[STREAM] {elapsed:.2f}s | {len(full_response)} chars | "
            f"reasoning: {len(full_reasoning)} chars | "
            f"tool_calls: {[tc['function']['name'] for tc in tool_calls_final] or 'nenhuma'} | "
            f"cached: {cached_tokens} tokens"
            + (f" | memory_corrections: {watch_events_used}" if watch_confidence else "")
            + (f" | watch_logprobs: {watch_lp_seen}" if watch_confidence else "")
        )
        if watch_confidence and watch_lp_seen == 0:
            log.warning(
                "[MEMORY-WATCH] Nenhum logprob na resposta inteira — o vigia de "
                "confiança não pôde atuar (provedor provavelmente não suporta "
                f"logprobs para {MAIN_MODEL}). Considere desligar watch_confidence "
                "ou trocar de provedor/modelo."
            )

        if tool_calls_final:
            # ── ATIVAÇÃO DAS TOOLS: evento único ao final da resposta ──
            # Formato idêntico a message.tool_calls do OpenAI — o caller pode
            # anexar direto no histórico como mensagem role="assistant" e
            # responder com role="tool".
            yield f"data: {json.dumps({'tool_calls': tool_calls_final})}\n\n"

        yield f"data: {json.dumps({'done': True, 'elapsed': round(elapsed, 3), 'prompt_cached_tokens': cached_tokens, 'had_tool_calls': bool(tool_calls_final)})}\n\n"

        # Persistência pós-stream — apenas turno FINAL do modo chat
        if turno_final and (not tool_mode) and full_response:
            # Grava turno na memória ST
            asyncio.create_task(
                memory_save_turn(req.session_id, user_input, full_response)
            )
            # ── EXTRAÇÃO DE MEMÓRIAS LT ─────────────────────────────────────
            # Mesma lógica do /chat síncrono: usa o texto final agregado,
            # independente de quantas chunks SSE tenham vindo. Garante que
            # mesmo respostas longas com reasoning/tool calls intermediários
            # só disparem o extractor UMA vez, sobre o produto final.
            if req.extract_memories:
                asyncio.create_task(
                    _extract_and_save_memories(user_input, full_response, session_id=req.session_id)
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
        await _call_memory_tool("memory_clear_session", {"session_id": req.session_id})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Falha ao limpar sessão: {e}")

    return {"cleared": True, "session_id": req.session_id}


@app.get("/history")
async def get_history(session_id: str = "default", last_n: int = 20):
    """Retorna as últimas N mensagens do histórico via módulo de memória."""
    try:
        resp = await _call_memory_tool(
            "memory_read",
            {"query": "histórico recente", "session_id": session_id, "top_k": last_n},
        )
        results = resp.get("results", [])
        return {"history": results, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Falha ao buscar histórico: {e}")


# ── Endpoint manual para extração de memórias (debug / testes) ────────────────

class MemoryExtractionRequest(BaseModel):
    user_input: str = Field(..., description="Pergunta original do usuário.")
    assistant_response: str = Field(..., description="Resposta final do assistente.")
    session_id: Optional[str] = Field(default=None)


@app.post("/memories/extract")
async def memories_extract(req: MemoryExtractionRequest):
    """
    Endpoint manual para disparar o extractor de memórias sobre uma dupla
    pergunta-resposta arbitrária. Útil para depurar o modelo extractor ou
    para reprocessar turnos antigos em batch.

    Em condições normais, o /chat e /chat/stream já disparam o extractor
    automaticamente em background (quando extract_memories=true).
    """
    saved = await _extract_and_save_memories(
        req.user_input,
        req.assistant_response,
        session_id=req.session_id,
    )
    return {
        "extractor_model": MEMORY_EXTRACTOR_MODEL,
        "session_id":      req.session_id,
        "memories":        saved,
        "count":           len(saved),
    }


# ─────────────────────────────────────────────────────────────
#                         UTILITÁRIOS
# ─────────────────────────────────────────────────────────────

def _safe_detect(text: str) -> str:
    """
    Detecta o idioma do texto para escolher a pronúncia do TTS.
    Fallback seguro para "pt" se a detecção falhar.
    """
    try:
        return detect(text) if len(text.strip()) >= 3 else "pt"
    except Exception:
        return "pt"


# ─────────────────────────────────────────────────────────────
#                   TOOL USE NATIVO (OpenRouter)
# ─────────────────────────────────────────────────────────────
# Endpoint para o módulo alpha_code (agente ReAct). Não compartilha
# memória/TTS do /chat — mensagens e tools são controlados pelo caller.

class RequestTooLargeError(RuntimeError):
    """Sinaliza que o request excedeu o contexto máximo do modelo."""
    pass


def _is_request_too_large_error(msg: str) -> bool:
    """Detecta mensagens de erro indicando que o prompt excede o contexto."""
    if not msg:
        return False
    msg_l = msg.lower()
    return any(s in msg_l for s in ("context", "too large", "exceeds", "n_ctx", "exceed", "maximum context"))


async def _openrouter_tool_call(
    messages: list[dict],
    tools: list[dict],
    tool_choice,
    temperature: float,
    max_retries: int = 3,
    grammar: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> tuple[dict, str, bool]:
    """
    Executa tool call no OpenRouter, com retry em falhas transitórias.

    Estratégia:
      - Erro de rede (ConnectError/ReadTimeout/RemoteProtocolError): retry com backoff.
      - 429 (rate limit): retry respeitando Retry-After.
      - 5xx: retry com backoff.
      - Erro de contexto grande demais: NÃO retenta — levanta RequestTooLargeError.
      - Outros 4xx: não retry.

    MODO GRAMMAR (compatibilidade com chamadores antigos do llama-server):
      Quando `grammar` (string GBNF) é fornecida, o payload NÃO inclui
      `tools`/`tool_choice` e ativamos `response_format: {type: "json_object"}`
      no OpenRouter — isso força o output em JSON válido (sem validação de
      schema estrito). A resposta vem em `message.content` (não em
      `message.tool_calls`) e o caller é responsável por parsear.
      Nota: o OpenRouter não aceita GBNF nativamente; a gramática é
      interpretada como "forçar JSON".

    Retorna (message_dict, model_used, fallback_used=False sempre — não há
    mais fallback entre provedores, só OpenRouter).
    """
    client = await _get_openrouter_client()
    payload: dict = {
        "model":       MAIN_MODEL,
        "messages":    messages,
        "temperature": temperature,
    }
    if reasoning_effort:
        # reasoning.effort é o formato OpenRouter para controlar raciocínio
        payload["reasoning"] = {"effort": reasoning_effort}

    # ── Modo grammar vs. tool-calling nativo ─────────────────────────────
    # Mutuamente exclusivos: se grammar está presente, usamos response_format
    # para forçar JSON (equivalente aproximado do GBNF do llama-server).
    if grammar:
        payload["response_format"] = {"type": "json_object"}
    elif tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            r = await client.post("/chat/completions", json=payload)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            last_error = e
            log.warning(
                f"OpenRouter /chat/tools: erro de rede ({type(e).__name__}), "
                f"tentativa {attempt + 1}/{max_retries + 1}."
            )
            if attempt < max_retries:
                await asyncio.sleep(OPENROUTER_RETRY_BACKOFF)
                continue
            raise RuntimeError(f"OpenRouter inacessível: {e}") from e

        if r.status_code == 200:
            data = r.json()
            return data["choices"][0]["message"], MAIN_MODEL, False

        # Erro de contexto grande demais: não faz sentido retentar
        try:
            body = r.json()
            err_msg = body.get("error", {}).get("message", "") or r.text
        except Exception:
            err_msg = r.text

        if _is_request_too_large_error(err_msg):
            log.warning(f"OpenRouter /chat/tools: contexto excede o limite. Erro: {err_msg[:200]}")
            raise RequestTooLargeError(f"Contexto excede o limite do modelo: {err_msg[:300]}")

        # 429: rate limit — respeita Retry-After
        if r.status_code == 429 and attempt < max_retries:
            retry_after = r.headers.get("Retry-After")
            wait_s = float(retry_after) if retry_after else OPENROUTER_RETRY_BACKOFF
            log.warning(
                f"OpenRouter /chat/tools: 429. Aguardando {wait_s}s "
                f"(tentativa {attempt + 2}/{max_retries + 1})."
            )
            await asyncio.sleep(wait_s)
            continue

        # 5xx: retry com backoff
        if r.status_code >= 500:
            log.warning(f"OpenRouter /chat/tools: {r.status_code} (tentativa {attempt + 1}/{max_retries + 1}).")
            if attempt < max_retries:
                await asyncio.sleep(OPENROUTER_RETRY_BACKOFF)
                continue
            raise RuntimeError(f"OpenRouter /chat/tools: {r.status_code} persistente. Erro: {err_msg[:300]}")

        # Outros 4xx: não retry
        raise RuntimeError(f"OpenRouter /chat/tools: {r.status_code} - {err_msg[:300]}")

    raise RuntimeError(f"OpenRouter /chat/tools: falhou após {max_retries + 1} tentativa(s): {last_error}")


@app.post("/chat/tools", response_model=ToolUseResponse)
async def chat_tools(req: ToolUseRequest):
    """
    [COMPATIBILIDADE] Tool use nativo síncrono (OpenRouter) — mantido para o
    orquestrador (_llm_chat) e alpha_code, que esperam resposta JSON única.

    A versão UNIFICADA deste comportamento com streaming agora vive em
    /chat/stream: envie `messages` + `tools` no ChatRequest e as tool_calls
    chegam no evento SSE {"tool_calls": [...]} ao final da resposta.

    Recebe messages + tools (formato OpenAI function-calling) e retorna
    a mensagem do assistant (pode conter tool_calls ou content).

    Recebe messages + tools (formato OpenAI function-calling) e retorna
    a mensagem do assistant (pode conter tool_calls ou content).

    MODO GRAMMAR (compatibilidade):
      Se `req.grammar` estiver presente, o payload enviado ao OpenRouter
      NÃO inclui `tools`/`tool_choice`. Em vez disso, ativamos
      `response_format: {type: "json_object"}` (equivalente aproximado
      do GBNF do llama-server). A resposta vem em `message.content` e o
      caller faz o parse — garantidamente JSON válido pelo provedor.

    Diferenças vs /chat:
      - Sem memória persistida (caller gerencia)
      - Sem TTS, sem detecção de idioma
      - Sem streaming (síncrono — alpha_code faz seu próprio streaming de steps)
      - too_large=true quando o contexto excede o limite do modelo
        (caller deve reduzir o contexto)
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages vazio.")

    messages = [m.model_dump(exclude_none=True) for m in req.messages]

    t0 = time.perf_counter()
    try:
        message, model_used, fallback = await _openrouter_tool_call(
            messages=messages,
            tools=req.tools,
            tool_choice=req.tool_choice,
            temperature=req.temperature,
            max_retries=req.max_retries,
            grammar=req.grammar,
            reasoning_effort=req.reasoning_effort,
        )
    except RequestTooLargeError as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return ToolUseResponse(
            message={"role": "assistant", "content": "", "tool_calls": None},
            model=MAIN_MODEL,
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
            "prompt_tokens_approx":     approx_prompt_tokens,
            "completion_tokens_approx": approx_completion_tokens,
            "total_approx":             approx_prompt_tokens + approx_completion_tokens,
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