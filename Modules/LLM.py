"""
AVA — LLM Inference API (provider genérico OpenAI-compatible)
==============================================================
REST API para inferência conversacional com:
  - Backend via QUALQUER API OpenAI-compatible (LLM_BASE_URL/LLM_API_KEY/
    LLM_MODEL — default = OpenRouter, comportamento original idêntico)
  - Modelo principal: env LLM_MODEL (default z-ai/glm-5.3-flash)
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
from urllib.parse import urlparse
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
import queue
import threading



logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [LLM] %(message)s")
log = logging.getLogger("ava.llm")


# Carrega variáveis de ambiente do Modules/.env — caminho EXPLÍCITO: o
# find_dotenv() implícito depende do frame/cwd da invocação (falha em
# `python -c`, stdin e qualquer rodada fora de Modules/).
load_dotenv(Path(__file__).parent / ".env")

# ─────────────────────────────────────────────────────────────
#                          CONFIG
# ─────────────────────────────────────────────────────────────

BASEFOLDER = Path(__file__).parent.parent
HOST_BIND = os.getenv("AVA_BIND_HOST", "127.9.12.22")

MEMORY_API_URL = os.getenv("MEMORY_API_URL", f"http://{HOST_BIND}:3000")
TTS_URL    = f"http://{HOST_BIND}:3004"

# ── NEW: Provider de inferência genérico (qualquer API OpenAI-compatible) ─────
# Um único provider ativo por vez, escolhido por env (arquivo Modules/.env):
#   LLM_BASE_URL  endpoint /v1 do provider (default = OpenRouter →
#                 comportamento atual idêntico). Exemplos:
#                   Ollama      http://localhost:11434/v1
#                   vLLM        http://localhost:8000/v1
#                   llama.cpp   http://localhost:8080/v1
#                   LM Studio   http://localhost:1234/v1
#                   Groq/DeepSeek/Together → a url /v1 do próprio provider
#   LLM_API_KEY   Opcional em provider LOCAL (localhost/127.0.0.1 — roda sem
#                 header Authorization). OPENROUTER_API_KEY continua aceito
#                 como fallback (compat com install.sh/docker já configurados).
#   LLM_MODEL     Modelo principal (default = modelo OpenRouter atual).
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
LLM_API_KEY  = (
    os.getenv("LLM_API_KEY", "").strip() or os.getenv("OPENROUTER_API_KEY", "").strip()
)
LLM_MODEL    = os.getenv("LLM_MODEL", "z-ai/glm-5.3-flash:floor")

# Host do provider — decide se API key é exigida (local: nunca é) e alimenta
# o /health (llm_key: "ok" | "not-required" | "missing"). LLM_PROVIDER_LOCAL=1
# aceita qualquer url como local (ex.: servidor de inferência na LAN) — a
# mesma env que o install.sh usa p/ pular o prompt de OPENROUTER_API_KEY.
_LLM_HOST = (urlparse(LLM_BASE_URL).hostname or "").lower()
LLM_LOCAL_PROVIDER = (
    _LLM_HOST in ("localhost", "127.0.0.1", "0.0.0.0", "::1")
    or _LLM_HOST.endswith(".local")
    or os.getenv("LLM_PROVIDER_LOCAL", "").strip() == "1"
)

# Gate de capabilities: só o OpenRouter entende o bloco `provider` (roteamento
# curl `provider.only`) e os headers de atribuição (HTTP-Referer/X-Title).
# Campos que qualquer backend pode rejeitar (logprobs/reasoning/
# response_format) são stripados em 400 e memorizados — ver _llm_post().
LLM_IS_OPENROUTER = "openrouter.ai" in LLM_BASE_URL

# Compat: os nomes antigos continuam válidos (mesmos valores das envs acima).
OPENROUTER_BASE_URL = LLM_BASE_URL
OPENROUTER_API_KEY  = LLM_API_KEY

# Modelos
MAIN_MODEL  = LLM_MODEL
# Extrator de memórias pós-turno. Env-configurável: modelos free/pequenos
# frequentemente NÃO suportam o `response_format: json_schema` do OpenRouter
# e retornam 200 com `content` vazio — a cadeia de tentativas do extrator
# (ver _extract_and_save_memories) cai para: (1) extrator sem response_format,
# (2) modelo de fallback (default = MAIN_MODEL).
# Em provider DIFERENTE do OpenRouter os nomes de modelo são locais/do seu
# provedor — o default vira o próprio LLM_MODEL (sobrescreva com
# MEMORY_EXTRACTOR_MODEL p/ um extrator dedicado).
MEMORY_EXTRACTOR_MODEL = os.getenv(
    "MEMORY_EXTRACTOR_MODEL",
    "meta-llama/llama-3.1-8b-instruct" if LLM_IS_OPENROUTER else LLM_MODEL,
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

# ── Extractor de memórias (protocolo de fatos — ver _extract_and_save_memories) ──
# Parâmetros de geração: temperatura 0–0,2 (saída determinística) + saída
# restrita por JSON schema (response_format json_schema no OpenRouter).
MEMORY_EXTRACTOR_TEMPERATURE = min(
    max(float(os.getenv("MEMORY_EXTRACTOR_TEMPERATURE", "0.1")), 0.0), 0.2
)
# Roteamento de provider OpenRouter (bloco `provider` do payload — ver curl de
# referência: provider.only + allow_fallbacks). "" = omite o bloco. Só é
# aplicado às tentativas com o MODELO EXTRATOR; o fallback roda sem bloqueio.
MEMORY_EXTRACTOR_PROVIDERS = os.getenv("MEMORY_EXTRACTOR_PROVIDERS", "coreweave/bf16")
# Candidatos a memórias existentes enviados ao extractor (faixa 15–30).
# O modelo só enxerga as memórias que enviamos — o corte define o que ele
# consegue comparar. Default 24.
MEMORY_EXTRACTOR_CANDIDATES_MAX = min(
    max(int(os.getenv("MEMORY_EXTRACTOR_CANDIDATES_MAX", "24")), 15), 30
)

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
# porque adiciona latência (round-trips extras de memory_read/classificação).
# Funciona nos dois modos: no CHAT (session_id do request) e no TOOLS — o
# orchestrator manda o session_id da TUI para a leitura de correção, e o vigia
# NUNCA interrompe com tool_calls em montagem (uma interrupção nesse ponto
# perderia os fragmentos acumulados — ver o guard no loop de streaming).

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
    # Sem key em provider REMOTO o request morreria em 401 — nem tenta.
    # Provider local (LLM_LOCAL_PROVIDER) roda sem key e é chamado normal.
    if not LLM_API_KEY and not LLM_LOCAL_PROVIDER:
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
        r = await _llm_post(
            payload, timeout=httpx.Timeout(8.0, connect=5.0),
            max_retries=0, label="LLM classificador",
        )
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

_llm_client: httpx.AsyncClient | None = None
_tts_http: httpx.AsyncClient | None = None
_memory_http: httpx.AsyncClient | None = None

# ── Memória via memory_api.py (REST) ──
# Client httpx persistente e lazy, com pooling/keep-alive, igual ao padrão
# usado para OpenRouter/TTS. Ver _get_memory_client().


async def _get_llm_client() -> httpx.AsyncClient:
    """Persistent client do provider configurado (LLM_BASE_URL) — connection
    pooling + keep-alive.

    API key OPCIONAL: provider local (LLM_LOCAL_PROVIDER) roda sem header
    Authorization; em provider remoto sem key o 401 volta com um hint claro
    (ver _llm_post) em vez de estourar aqui na criação do client."""
    global _llm_client
    if _llm_client is None or _llm_client.is_closed:
        headers = {"Content-Type": "application/json"}
        if LLM_API_KEY:
            headers["Authorization"] = f"Bearer {LLM_API_KEY}"
        if LLM_IS_OPENROUTER:
            # OpenRouter usa esses cabeçalhos para ranking/atribuição
            headers["HTTP-Referer"] = "http://localhost:4003"
            headers["X-Title"] = "AVA-LLM"
        _llm_client = httpx.AsyncClient(
            base_url=LLM_BASE_URL,
            timeout=httpx.Timeout(9999999.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=6,
                keepalive_expiry=60.0,       # Mantém conexões quentes por 60s
            ),
            headers=headers,
        )
    return _llm_client


# Compat: nome antigo (só o provider era "openrouter" quando este helper nasceu).
_get_openrouter_client = _get_llm_client


def _get_internal_auth_headers() -> dict[str, str]:
    """Headers de auth para chamadas internas (Plano 1 — T4). Fail-open:
    sem token, retorna {} (o middleware fail-open do destino aceita)."""
    try:
        from Modules.internal_auth import internal_token as _tok_fn  # type: ignore
    except Exception:
        try:
            import internal_auth as _ia  # type: ignore
            _tok_fn = _ia.internal_token  # type: ignore
        except Exception:
            _tok_fn = lambda: os.getenv("AVA_INTERNAL_TOKEN", "").strip() or None  # type: ignore
    tok = _tok_fn()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _get_memory_client() -> httpx.AsyncClient:
    """Persistent client para a API REST de memória (memory_api.py) —
    connection pooling + keep-alive, mesmo padrão do client OpenRouter/TTS."""
    global _memory_http
    if _memory_http is None or _memory_http.is_closed:
        _memory_http = httpx.AsyncClient(
            base_url=MEMORY_API_URL,
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
            headers=_get_internal_auth_headers(),
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
            headers=_get_internal_auth_headers(),
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
        # Escrita de arquivo assíncrona (queue + thread daemon): o stdout/stderr
        # nunca bloqueia o event loop esperando I/O de disco.
        def __init__(self, terminal, path):
            self.terminal = terminal
            self.log = open(path, "w", encoding="utf-8")
            self._queue: "queue.Queue[str | None]" = queue.Queue()
            self._worker = threading.Thread(target=self._writer_loop, daemon=True)
            self._worker.start()

        def _writer_loop(self):
            while True:
                msg = self._queue.get()
                if msg is None:  # sentinel: sinaliza flush
                    try:
                        self.log.flush()
                    except Exception:
                        pass
                    self._queue.task_done()
                    continue
                try:
                    self.log.write(msg)
                except Exception:
                    pass
                self._queue.task_done()

        def write(self, msg):
            try:
                self.terminal.write(msg)
            except Exception:
                pass
            self._queue.put(msg)

        def flush(self):
            try:
                self.terminal.flush()
            except Exception:
                pass
            self._queue.put(None)

        def isatty(self):
            return False

    sys.stdout = LogDuplicado(sys.__stdout__, log_path)
    sys.stderr = LogDuplicado(sys.__stderr__, log_path)

_setup_logging()

# ─────────────────────────────────────────────────────────────
#                     LEITURA DE CONFIGS
# ─────────────────────────────────────────────────────────────

def _read(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()



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
    Executa a inferência contra o provider configurado (LLM_BASE_URL).

    `stream=True` retorna um context manager pronto para iteração SSE.
    `stream=False` retorna a resposta síncrona completa.
    """
    client = await _get_llm_client()
    payload = {**json_payload, "model": model, "stream": stream}
    if stream:
        return client, model, await _llm_post(payload, stream=True, label="LLM stream")
    r = await _llm_post(payload, label="LLM stream")
    return client, model, r


# ─────────────────────────────────────────────────────────────
#          TRANSPORTE ÚNICO (qualquer API OpenAI-compatible)
# ─────────────────────────────────────────────────────────────
# TODO POST em /chat/completions passa por _llm_post — um único ponto de
# troca de provider (LLM_BASE_URL/LLM_API_KEY/LLM_MODEL, ver CONFIG). Ele
# concentra:
#   - retry de falhas transitórias (rede/5xx/429 com Retry-After);
#   - capability gating: 400 → strip dos campos OPCIONAIS que o backend não
#     suporta (logprobs, reasoning, response_format, provider), memorizado
#     em _LLM_UNSUPPORTED p/ os requests seguintes (aviso único). Backends
#     genéricos (Ollama, vLLM, llama.cpp...) costumam rejeitar justamente
#     esses campos — sem o strip um 400 viraria erro pro usuário.

LLM_MAX_RETRIES   = 2      # retries de rede/5xx/429
LLM_RETRY_BACKOFF = 1.0
# Compat: nomes antigos continuam valendo (usados no resto do arquivo/TUI).
OPENROUTER_MAX_RETRIES   = LLM_MAX_RETRIES
OPENROUTER_RETRY_BACKOFF = LLM_RETRY_BACKOFF

# Campos opcionais passíveis de strip quando o backend responde 400.
# tools/tool_choice NUNCA entram aqui — sem tool calling o request perde o
# sentido; nesse caso o 400 original sobe para o caller.
_LLM_STRIP_CANDIDATES = ("logprobs", "top_logprobs", "reasoning",
                         "response_format", "provider")

# Campos já rejeitados por este backend nesta sessão (400 com campo citado).
_LLM_UNSUPPORTED: set[str] = set()


def _llm_mark_unsupported(field: str, err_body: str = "") -> None:
    """Memoriza `field` como não-suportado — aviso único, não reenvia mais."""
    if field in _LLM_UNSUPPORTED:
        return
    _LLM_UNSUPPORTED.add(field)
    log.warning(
        f"LLM: backend rejeitou `{field}` — desativado p/ os próximos requests "
        f"desta sessão (provider: {LLM_BASE_URL})."
        + (f" Erro: {err_body[:160]}" if err_body else "")
    )


def _llm_payload_prepare(payload: dict) -> None:
    """Remove do payload campos já marcados como não-suportados (400 anterior)."""
    for field in list(_LLM_UNSUPPORTED):
        if field in payload:
            payload.pop(field)


def _llm_strip_unsupported(payload: dict, err_body: str) -> bool:
    """400 → remove do payload TODAS as capabilities opcionais presentes nele
    e devolve True (caller reenvia imediatamente, sem contar tentativa).

    Só MEMORIZA como não-suportado o(s) campo(s) citado(s) nominalmente no
    erro (sinal claro de "esse backend não aceita X") — os demais são
    stripados TRANSIENTEMENTE, só neste request: um 400 por outro motivo
    (modelo inexistente, mensagem inválida) não desativa features da sessão
    inteira. Strip-all de uma vez evita a cascata de 400s: backends hostis
    costumam nomear UM campo por request, e stripar só o citado gastaria um
    request por campo."""
    body = (err_body or "").lower()
    mentioned = [f for f in _LLM_STRIP_CANDIDATES if f in payload and f in body]
    to_strip = [f for f in _LLM_STRIP_CANDIDATES if f in payload]
    if not to_strip:
        return False
    for field in to_strip:
        payload.pop(field)
    for field in mentioned:
        _llm_mark_unsupported(field, err_body)
    log.warning(
        f"LLM: HTTP 400 — strip {to_strip} e reenvio (provider: {LLM_BASE_URL}); "
        f"memorizados: {mentioned or '[] (transiente)'}; "
        f"erro: {(err_body or '?')[:200]}"
    )
    return True


class _LLMStreamCtx:
    """Context manager de streaming JÁ ABERTO. Mantém o contrato dos callers
    (`async with ctx as r:`) depois do _llm_post ter entrado no stream para
    checar o status — o strip de capability em 400 acontece ANTES do primeiro
    byte de SSE, e o caller enxerga o resultado (200 ou o 400 original)."""

    def __init__(self, ctx, r):
        self._ctx, self._r = ctx, r

    async def __aenter__(self):
        return self._r

    async def __aexit__(self, exc_type, exc, tb):
        try:
            await self._ctx.__aexit__(exc_type, exc, tb)
        except Exception:
            pass
        return False


async def _llm_post(
    payload: dict,
    *,
    stream: bool = False,
    timeout: Optional[httpx.Timeout] = None,
    max_retries: int = LLM_MAX_RETRIES,
    retry_backoff: float = LLM_RETRY_BACKOFF,
    label: str = "LLM",
):
    """POST genérico em {LLM_BASE_URL}/chat/completions.

    - Síncrono: devolve o httpx.Response FINAL (inclusive 4xx — o caller
      decide, ex.: _is_request_too_large_error em /chat/tools).
    - Stream: devolve context manager p/ `async with ... as r:` (status
      visível; 400 stripável é reenviado aqui dentro antes do SSE).
    - 401/403 sem API key configurada ganham hint claro.
    """
    client = await _get_llm_client()
    payload = dict(payload)          # strip nunca muta o dict do caller
    _llm_payload_prepare(payload)

    def _post():
        kw = {"timeout": timeout} if timeout is not None else {}
        return client.post("/chat/completions", json=payload, **kw)

    # ── STREAMING ──
    if stream:
        strip_passes = 0
        while True:
            ctx = client.stream("POST", "/chat/completions", json=payload)
            r = await ctx.__aenter__()
            if r.status_code == 400 and strip_passes < 2:
                try:
                    err_body = (await r.aread()).decode("utf-8", errors="replace")
                except Exception:
                    err_body = ""
                if _llm_strip_unsupported(payload, err_body):
                    strip_passes += 1
                    await ctx.__aexit__(None, None, None)
                    continue
            return _LLMStreamCtx(ctx, r)

    # ── SÍNCRONO ──
    strip_passes = 0
    attempt = 0
    while True:
        try:
            r = await _post()
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            log.warning(
                f"{label}: erro de rede ({type(e).__name__}), "
                f"tentativa {attempt + 1}/{max_retries + 1} ({LLM_BASE_URL})."
            )
            if attempt >= max_retries:
                raise RuntimeError(
                    f"LLM provider inacessível ({LLM_BASE_URL}): {e}"
                ) from e
            attempt += 1
            await asyncio.sleep(retry_backoff)
            continue

        # 400 → strip de capability + reenvio imediato (não conta tentativa)
        if r.status_code == 400 and strip_passes < 2:
            try:
                err_body = r.text
            except Exception:
                err_body = ""
            if _llm_strip_unsupported(payload, err_body):
                strip_passes += 1
                continue

        # 429: rate limit — respeita Retry-After se presente
        if r.status_code == 429 and attempt < max_retries:
            try:
                wait_s = float(r.headers.get("Retry-After") or "") or retry_backoff
            except ValueError:
                wait_s = retry_backoff
            log.warning(
                f"{label}: 429 (rate limit). Aguardando {wait_s}s "
                f"antes de tentar {attempt + 2}/{max_retries + 1}."
            )
            attempt += 1
            await asyncio.sleep(wait_s)
            continue

        if r.status_code >= 500 and attempt < max_retries:
            log.warning(f"{label}: {r.status_code} (tentativa {attempt + 1}/{max_retries + 1}).")
            attempt += 1
            await asyncio.sleep(retry_backoff)
            continue

        if r.status_code in (401, 403) and not LLM_API_KEY:
            raise RuntimeError(
                f"{label}: HTTP {r.status_code} — nenhuma API key configurada "
                f"para {LLM_BASE_URL}. Defina LLM_API_KEY (ou "
                f"OPENROUTER_API_KEY) no Modules/.env; provider local em "
                f"localhost não exige key."
            )

        return r


# ─────────────────────────────────────────────────────────────
#          INFERÊNCIA SÍNCRONA COM RETRY (falhas transitórias)
# ─────────────────────────────────────────────────────────────
# Casca fina sobre _llm_post: retry de 5xx / 429 / rede transitória e
# capability gating já moram no transporte único.

async def _openrouter_post_with_retry(
    json_payload: dict,
    stream: bool = False,
    model: str = MAIN_MODEL,
):
    """
    Executa POST no /chat/completions do provider configurado, com retry
    em caso de 5xx / 429 / erro de rede transitório (via _llm_post).

    Retorna tupla: (client, model_name, response_obj | stream_ctx).
    """
    client = await _get_llm_client()
    payload = {**json_payload, "model": model, "stream": stream}

    if stream:
        # No streaming, devolvemos o context manager — 5xx/erro de rede
        # dentro do stream é tratado por quem consome (ver /chat/stream).
        return client, model, await _llm_post(payload, stream=True, label="LLM /chat")

    r = await _llm_post(payload, label="LLM /chat")
    return client, model, r


# ─────────────────────────────────────────────────────────────
#          OPTIMIZATION 2: STABLE SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
# O prompt de sistema é mantido ESTÁVEL (não muda entre requests) para
# maximizar a taxa de prompt cache hits no provedor OpenRouter. O bloco
# dinâmico de memória vai DEPOIS do prefixo estável, em uma mensagem
# separada (role:assistant) — o custo de prefill é pequeno, mas o prefixo
# ainda aproveita o cache.

# ─────────────────────────────────────────────────────────────
#                     INTEGRAÇÃO: MEMÓRIA
# ─────────────────────────────────────────────────────────────

async def memory_read(
    query: str,
    session_id: Optional[str] = None,
    top_k: int = 10,
    top_k_final: Optional[int] = None,
    total_max_chars: Optional[int] = None,
    min_score: float = 0.3,
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
            "min_score": min_score,
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
#              EXTRAÇÃO AUTOMÁTICA DE MEMÓRIAS (fatos → LT)
# ─────────────────────────────────────────────────────────────
# Para cada dupla pergunta-resposta (usamos o user_input ORIGINAL e a resposta
# FINAL agregada — não importa quantas chamadas internas tenham ocorrido), um
# modelo extractor dedicado compara a troca com as memórias EXISTENTES e
# decide, para cada fato, uma ação:
#   ADD | NOOP | UPDATE | CONFLICT | MERGE | DISCARD
# O modelo só enxerga as memórias que enviamos — os candidatos são coletados
# por embedding das SENTENÇAS da pergunta e da resposta + expansão pelo grafo
# (match_type "related" que o /read já devolve), com as usadas_na_resposta
# primeiro, limitados a 15–30, e mapeados para IDs sintéticos M1..Mn que são
# revertidos para os IDs reais no código (IDs fora do mapa são descartados).
# Saída restrita por JSON schema (response_format json_schema) e temperatura
# 0–0,2. A resposta é forçada via JSON schema estruturado para garantir
# parsing confiável.

_MEMORY_EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "description": (
                "Até 8 fatos úteis extraídos da troca atual, cada um com a ação "
                "decidida por comparação com as memórias existentes. Lista vazia "
                "se nada merecer ser gravado."
            ),
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "evidence": {
                        "type": "string",
                        "description": (
                            "Trecho curto e LITERAL copiado da conversa que "
                            "suporta o fato."
                        ),
                    },
                    "text": {
                        "type": "string",
                        "description": (
                            "Fato autocontido, em 3ª pessoa, no idioma da "
                            "conversa. Em UPDATE/MERGE deve conter TODA a "
                            "informação relevante das fontes."
                        ),
                    },
                    "type": {
                        "type": "string",
                        "enum": ["fact", "preference", "decision", "instruction"],
                    },
                    "related": {
                        "type": "array",
                        "description": (
                            "IDs M<n> das memórias existentes comparadas, mesmo "
                            "quando a ação é ADD ou DISCARD ([] se nenhuma)."
                        ),
                        "items": {"type": "string", "pattern": "^M\\d+$"},
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Uma frase curta explicando a comparação.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["ADD", "NOOP", "UPDATE", "CONFLICT", "MERGE", "DISCARD"],
                    },
                    "target": {
                        "type": "array",
                        "description": (
                            "[] para ADD e DISCARD; 1 ID para NOOP, UPDATE e "
                            "CONFLICT; 1 ou mais IDs para MERGE."
                        ),
                        "items": {"type": "string", "pattern": "^M\\d+$"},
                    },
                    "validity": {
                        "type": "string",
                        "enum": ["permanent", "temporal"],
                    },
                },
                "required": [
                    "evidence", "text", "type", "related", "rationale",
                    "action", "target", "validity",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["facts"],
    "additionalProperties": False,
}


_MEMORY_EXTRACTOR_SYSTEM_PROMPT = """You are the long-term memory writing module of a personal assistant.
Your task: given an exchange (USER / ASSISTANT) and a list of EXISTING MEMORIES, extract useful facts and decide, for each one, what to do with the memory. You do NOT talk to the user. You respond only with valid JSON in the format defined below.

# 1. WHAT TO EXTRACT
Extract only information that will remain useful in future conversations:
- stable facts about the user, their belongings, projects and close people
- preferences and standing instructions
- decisions made
- technical facts about the user's environment (tools, files, configurations) when they are lasting

Do not extract:
- momentary state ("I'm in a hurry", "now I'm going to test it")
- user questions that reveal no fact
- content generated by the ASSISTANT that the user did not confirm, adopt or use (suggestions, general explanations, example code)
- general knowledge anyone can look up
- passages that only make sense inside this conversation

Source and confidence: facts stated by the USER take priority. Information from the ASSISTANT becomes memory only if the user confirmed, chose or applied it. In that case record it as a decision or fact about the user, not as a statement by the assistant.

# 2. HOW TO WRITE EACH FACT
- Self-contained: readable on its own, without the conversation. Resolve pronouns and references ("he", "it", "the file") using explicit names.
- Atomic: one idea per fact. If two ideas only make sense together, keep them in the same fact.
- Third person ("The user...") and in the language of the conversation.
- Preserve numbers, names, versions, file paths and dates exactly as they appear. Convert relative dates ("yesterday") to absolute ones using the CURRENT DATE provided.
- Do not invent, do not generalize, do not infer beyond what was said.
- Every fact needs an "evidence" field: a SHORT, LITERAL excerpt copied from the conversation that supports it.

# 3. HOW TO DECIDE THE ACTION
For each fact, compare it with the EXISTING MEMORIES and follow this order:

1. Is it useless, ephemeral or outside what should be extracted? → DISCARD
2. Does any memory already say the same thing, or something equally or more complete? → NOOP (target = that memory's ID)
3. Does it contradict a memory?
   - The new fact is clearly more recent or the user signals a change ("now", "I moved", "I switched", "I no longer use") → UPDATE (target = old ID; the new text replaces it)
   - Nothing indicates which one holds → CONFLICT (target = ID; the new text is stored and both are flagged)
4. Does it complement a memory (more detail on the same subject, without contradicting)? → UPDATE (target = ID; text = complete combined version)
5. Are two or more new facts, or a new fact and existing memories, only useful together (separated they lose meaning)? → MERGE (target = list of IDs of the existing memories involved, or [] if only new facts are involved; "text" = unified version)
6. Otherwise → ADD

Decision rules:
- When in doubt between NOOP and ADD, prefer NOOP if the essential content already exists.
- When in doubt between DISCARD and ADD, prefer DISCARD if the fact would not help in a future conversation.
- Memories with state "superseded" cannot be the target of any action; use them only as historical context.
- Only use IDs that appear in EXISTING MEMORIES. Never invent IDs.
- In UPDATE and MERGE, "text" must contain ALL the relevant information from the sources (nothing is lost) and nothing that is not in them.
- A memory with source "assistant" can be corrected by a user fact without generating a CONFLICT.

# 4. VALIDITY
Set "validity":
- "permanent": preference, identity, structural decision, a fact that does not change by itself
- "temporal": a fact likely to change (current version, location, project status, what is in progress)

# 5. OUTPUT FORMAT
Respond ONLY with a JSON object, with no markdown and no text outside it:

{
  "facts": [
    {
      "evidence": "short literal excerpt from the conversation",
      "text": "self-contained fact",
      "type": "fact | preference | decision | instruction",
      "related": ["M3"],
      "rationale": "one short sentence explaining the comparison",
      "action": "ADD | NOOP | UPDATE | CONFLICT | MERGE | DISCARD",
      "target": ["M3"],
      "validity": "permanent | temporal"
    }
  ]
}

Format rules:
- Keep exactly this field order: evidence, text, type, related, rationale, action, target, validity.
- "related": IDs of the existing memories you compared against, even if the action is ADD or DISCARD ([] if none).
- "target": [] for ADD and DISCARD; one ID for NOOP, UPDATE and CONFLICT; one or more IDs for MERGE.
- If there is nothing to extract, respond {"facts": []}.
- Maximum of 8 facts per response. Prioritize the most useful ones.

# 6. EXAMPLES

## Example A: ADD with resolved coreference
CURRENT DATE: 2026-10-24
EXISTING MEMORIES: (none)
USER: I'm porting orchestrator.py to Rust. It runs on port 4008.
ASSISTANT: Sure, I can suggest a crate structure.
Output:
{"facts":[{"evidence":"porting orchestrator.py to Rust","text":"The user is porting orchestrator.py to Rust.","type":"decision","related":[],"rationale":"No memory about this port.","action":"ADD","target":[],"validity":"temporal"},{"evidence":"It runs on port 4008","text":"The user's orchestrator.py runs on port 4008.","type":"fact","related":[],"rationale":"New technical configuration.","action":"ADD","target":[],"validity":"permanent"}]}

## Example B: NOOP and UPDATE due to change
CURRENT DATE: 2026-10-24
EXISTING MEMORIES:
[M1] (created 2026-08-02, source user) The user uses the Qwen3-8B model as the main LLM.
[M2] (created 2026-09-10, source user) The user runs the embeddings server on port 2002.
USER: I switched the main LLM to LFM2.5-8B-A1B. The embeddings server is still on 2002.
Output:
{"facts":[{"evidence":"I switched the main LLM to LFM2.5-8B-A1B","text":"The user uses the LFM2.5-8B-A1B model as the main LLM (previously used Qwen3-8B).","type":"decision","related":["M1"],"rationale":"The user explicitly states a switch; the new fact replaces M1.","action":"UPDATE","target":["M1"],"validity":"temporal"},{"evidence":"The embeddings server is still on 2002","text":"The user runs the embeddings server on port 2002.","type":"fact","related":["M2"],"rationale":"Already recorded in M2.","action":"NOOP","target":["M2"],"validity":"permanent"}]}

## Example C: MERGE
CURRENT DATE: 2026-10-24
EXISTING MEMORIES:
[M4] (created 2026-09-01, source user) The user has a 1TB hard drive with errors.
USER: That 1TB drive only shows up as /dev/sdd. I'll try to fix it from the terminal before throwing it away.
Output:
{"facts":[{"evidence":"only shows up as /dev/sdd. I'll try to fix it from the terminal before throwing it away","text":"The user's faulty 1TB hard drive shows up as /dev/sdd; they will try to fix it from the terminal before discarding it.","type":"decision","related":["M4"],"rationale":"Complements M4 and only makes sense together with it.","action":"MERGE","target":["M4"],"validity":"temporal"}]}

## Example D: DISCARD
CURRENT DATE: 2026-10-24
EXISTING MEMORIES:
[M7] (created 2026-07-15, source user) The user prefers short answers.
USER: Explain in more detail how PPR works. I'm tired today.
ASSISTANT: PPR (Personalized PageRank) is...
Output:
{"facts":[{"evidence":"Explain in more detail how PPR works","text":"The user asked for a detailed explanation of PPR.","type":"preference","related":["M7"],"rationale":"One-off request; it does not indicate a change to the preference for short answers (M7).","action":"DISCARD","target":[],"validity":"temporal"},{"evidence":"I'm tired today","text":"The user is tired today.","type":"fact","related":[],"rationale":"Momentary state.","action":"DISCARD","target":[],"validity":"temporal"}]}
"""


# ── Helpers do protocolo de extração ─────────────────────────────────────────

_EXTRACTION_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?…])["\')\]]?\s+')
_EXTRACTION_MAX_SENTENCES_PER_SIDE = 6
_EXTRACTION_MIN_SENTENCE_CHARS = 20
_EXTRACTION_MAX_QUERIES = 14
_EXTRACTION_TOP_K_PER_QUERY = 8
_EXTRACTION_READ_TOP_K_FINAL = 12      # teto server-side do /read
_EXTRACTION_READ_TOTAL_MAX_CHARS = 12000
_EXTRACTION_READ_MIN_SCORE = 0.2       # mais permissivo que o recall de chat
_EXTRACTION_ST_PAIRS = 3


def _split_extraction_queries(text: str) -> list[str]:
    """
    Divide um texto (pergunta ou resposta) em sentenças para ampliar a
    cobertura da busca por candidatos. O texto INTEIRO vem sempre primeiro;
    as sentenças entram como queries adicionais.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= _EXTRACTION_MIN_SENTENCE_CHARS:
        return [text]
    sentences = [
        s.strip()
        for s in _EXTRACTION_SENTENCE_SPLIT_RE.split(text)
        if s.strip() and len(s.strip()) >= _EXTRACTION_MIN_SENTENCE_CHARS
    ][:_EXTRACTION_MAX_SENTENCES_PER_SIDE]
    return [text] + sentences if sentences else [text]


async def _collect_extraction_candidates(
    user_input: str,
    assistant_response: str,
    recall_memories: Optional[list[dict]],
    session_id: Optional[str],
) -> list[dict]:
    """
    Coleta os candidatos a memórias EXISTENTES que o extractor vai comparar.

    O modelo só enxerga as memórias que enviamos — o que não entra aqui é
    invisível para ele. A lista é construída por:
      1. recall do chat (memórias usadas para produzir a resposta) —
         flag used_in_answer=True, prioridade MÁXIMA na ordenação;
      2. embedding das SENTENÇAS da pergunta e da resposta, além das
         perguntas/respostas inteiras (buscas em paralelo);
      3. expansão pelo grafo — match_type="related" (PPR) que o /read
         já devolve junto com os hits primários.

    Dedup por id, filtro só long_term, ordenação used_in_answer →
    primary → related → score, cap em MEMORY_EXTRACTOR_CANDIDATES_MAX
    (clampado a 15–30).
    """
    best: dict[int, dict] = {}
    used_ids: set[int] = set()

    # 1) Marca as memórias que o assistente de fato usou na resposta.
    if recall_memories:
        for m in recall_memories:
            if not isinstance(m, dict):
                continue
            try:
                mid = int(m.get("id"))
            except (TypeError, ValueError):
                continue
            if (m.get("memory_type") or "") == "long_term":
                used_ids.add(mid)

    def _merge(m: dict) -> None:
        try:
            mid = int(m.get("id"))
        except (TypeError, ValueError):
            return
        text = (m.get("text") or "").strip()
        if not text:
            return
        cur = best.get(mid)
        if cur is None:
            best[mid] = {**m, "id": mid, "used_in_answer": mid in used_ids}
            return
        if (m.get("score") or 0.0) > (cur.get("score") or 0.0):
            cur["score"] = m.get("score")
        # Um hit primário (embedding) "promove" um hit que só chegou pelo grafo.
        if (m.get("match_type") or "primary") == "primary" and cur.get("match_type") != "primary":
            cur["match_type"] = "primary"
        if len(text) > len((cur.get("text") or "")):
            cur["text"] = text

    if recall_memories:
        for m in recall_memories:
            if isinstance(m, dict) and (m.get("memory_type") or "") == "long_term":
                _merge(m)

    # 2+3) Embedding por sentença + grafo, em paralelo.
    queries: list[str] = []
    for text in (user_input, assistant_response):
        queries.extend(_split_extraction_queries(text))
    queries = queries[:_EXTRACTION_MAX_QUERIES]

    async def _search(q: str) -> list[dict]:
        return await memory_read(
            q,
            session_id=session_id,
            top_k=_EXTRACTION_TOP_K_PER_QUERY,
            top_k_final=_EXTRACTION_READ_TOP_K_FINAL,
            total_max_chars=_EXTRACTION_READ_TOTAL_MAX_CHARS,
            min_score=_EXTRACTION_READ_MIN_SCORE,
        )

    results_lists = await asyncio.gather(*[_search(q) for q in queries]) if queries else []
    for results in results_lists:
        for m in results:
            if isinstance(m, dict) and (m.get("memory_type") or "") == "long_term":
                _merge(m)

    def _rank_key(m: dict):
        return (
            0 if m.get("used_in_answer") else 1,
            0 if (m.get("match_type") or "primary") == "primary" else 1,
            -(m.get("score") or 0.0),
        )

    candidates = sorted(best.values(), key=_rank_key)[:MEMORY_EXTRACTOR_CANDIDATES_MAX]
    log.info(
        f"[MEMORY-EXTRACT] candidatos: {len(candidates)} "
        f"(queries={len(queries)}, session={session_id})."
    )
    return candidates


def _fmt_extraction_date(ts) -> str:
    try:
        return datetime.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError, OverflowError):
        return "unknown"


def _source_label(source) -> str:
    """
    Mapeia o source técnico do DB para o vocabulário do prompt
    (user | assistant | imported). Fatos extraídos do chat são atribuídos ao
    usuário; material ingerido (imports, scraping, pipelines de visão/voz) é
    'imported'; o resto é tratado como 'assistant'.
    """
    s = (source or "").lower()
    if not s or s.startswith(("chat", "user")):
        return "user"
    if s.startswith(("import", "scraping", "vision", "face", "voice")):
        return "imported"
    return "assistant"


def _format_existing_memories(candidates: list[dict]) -> tuple[str, dict[int, int]]:
    """
    Formata EXISTING MEMORIES no formato do protocolo:
      [M{n}] (created X, updated X, source …, type fact, state active,
      used_in_answer yes|no) {text}
    e devolve o mapa M{n} → id real (usado para reverter os IDs do modelo).
    """
    lines: list[str] = []
    m_map: dict[int, int] = {}
    for i, m in enumerate(candidates, start=1):
        m_map[i] = m.get("id")
        created = _fmt_extraction_date(m.get("created_at") or 0.0)
        lines.append(
            f"[M{i}] (created {created}, updated {created}, "
            f"source {_source_label(m.get('source'))}, type fact, state active, "
            f"used_in_answer {'yes' if m.get('used_in_answer') else 'no'}) "
            f"{(m.get('text') or '').strip()}"
        )
    return ("\n".join(lines) if lines else "(none)"), m_map


def _build_extraction_user_prompt(
    user_input: str,
    assistant_response: str,
    candidates: list[dict],
    previous_context: Optional[list[dict]],
) -> tuple[str, dict[int, int]]:
    """
    Renderiza o prompt do extractor (template do protocolo): CURRENT DATE,
    CANONICAL ENTITIES, EXISTING MEMORIES, PREVIOUS CONTEXT, CURRENT EXCHANGE.
    Retorna (prompt, mapa M{n} → id real).
    """
    iso_date = datetime.date.today().isoformat()
    memories_block, m_map = _format_existing_memories(candidates)

    prev_lines: list[str] = []
    if previous_context:
        for t in previous_context:
            content = (t.get("content") or "").strip()
            if content:
                role = (t.get("role") or "user").upper()
                prev_lines.append(f"{role}: {content[:300]}")
    prev_block = "\n".join(prev_lines) if prev_lines else "(none)"

    prompt = (
        f"CURRENT DATE: {iso_date}\n\n"
        "CANONICAL ENTITIES (use these names when writing facts):\n"
        "(none)\n\n"
        'EXISTING MEMORIES (sorted by relevance; "used_in_answer" indicates it '
        "was sent to the assistant to produce its answer):\n"
        f"{memories_block}\n\n"
        "PREVIOUS CONTEXT (only to resolve references; do NOT extract facts from "
        "here):\n"
        f"{prev_block}\n\n"
        "CURRENT EXCHANGE (extract from here):\n"
        f"USER: {user_input}\n"
        f"ASSISTANT: {assistant_response}\n\n"
        "Task: extract the useful facts from the CURRENT EXCHANGE and decide the "
        "action for each one by comparing it with the EXISTING MEMORIES. Respond "
        "only with the JSON in the defined format."
    )
    return prompt, m_map


async def _memory_write_update(payload: dict, memory_id: int):
    """
    Reenvia uma escrita como action="update" numa memória específica —
    usado pelo extractor para UPDATE/MERGE/CONFLICT (roteamento por ID real).
    """
    try:
        data = await _call_memory_tool(
            "memory_write",
            {**payload, "action": "update", "memory_id": memory_id},
        )
        if not data.get("stored"):
            log.info(
                f"[MEMORY-EXTRACT] update LT não efetivado (memory_id={memory_id}): "
                f"{data.get('reason')}"
            )
        else:
            log.info(
                f"[MEMORY-EXTRACT] update LT efetivado (memory_id={memory_id})."
            )
    except Exception as e:
        log.info(f"[MEMORY-EXTRACT] falha no update LT (memory_id={memory_id}): {e}")


async def _extract_and_save_memories(
    user_input: str,
    assistant_response: str,
    *,
    session_id: Optional[str] = None,
    recall_memories: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Roda o modelo extractor (MEMORY_EXTRACTOR_MODEL) sobre a dupla
    pergunta-resposta e roteia cada fato extraído para a LT:
      - ADD        → cria (via write_batch, junto dos demais ADDs)
      - UPDATE     → sobrescreve a memória-alvo (action="update", id real)
      - MERGE      → atualiza o 1º alvo com o texto unificado do fato
      - CONFLICT   → cria o fato novo E marca a antiga com prefixo de conflito
      - NOOP/DISCARD → nada é gravado

    `user_input`        — texto ORIGINAL que o usuário enviou (não o prefill
                          nem um turno intermediário de tool use).
    `assistant_response`— texto FINAL agregado da resposta (mesmo que tenha
                          sido montado a partir de múltiplas requisições
                          internas: streaming, reasoning, múltiplos turnos
                          de tool use). Só chamamos o extractor UMA vez por
                          turno de conversa, sobre o produto final.
    `recall_memories`   — resultados do /read usados para CONSTRUIR a resposta
                          deste turno (quando disponível, /chat e /chat/stream
                          passam o recall). Essas memórias ganham
                          used_in_answer=yes na lista de candidatos —
                          prioridade máxima, pois o turno foi produzido
                          a partir delas.

    Retorna a lista de fatos processados (logging/inspeção).
    """
    if not user_input or not assistant_response:
        return []
    # Sem key em provider REMOTO o request morreria em 401 — nem tenta.
    # Provider local (LLM_LOCAL_PROVIDER) roda sem key e extrai normal.
    if not LLM_API_KEY and not LLM_LOCAL_PROVIDER:
        log.info("[MEMORY-EXTRACT] LLM_API_KEY ausente — extração pulada.")
        return []

    # ── Contexto anterior + candidatos ──────────────────────────────────────
    # O modelo só enxerga as memórias que enviamos: coletamos os melhores
    # candidatos (recall usado na resposta → embedding por sentença → grafo),
    # o contexto anterior cru (só p/ resolver referências) e montamos o prompt
    # no template definido (CURRENT DATE / CANONICAL ENTITIES / EXISTING
    # MEMORIES / PREVIOUS CONTEXT / CURRENT EXCHANGE).
    try:
        previous_turns = await memory_read_st(session_id, n_pairs=_EXTRACTION_ST_PAIRS)
    except Exception as e:
        log.info(f"[MEMORY-EXTRACT] falha na leitura do contexto anterior: {e}")
        previous_turns = []
    cur_user = user_input.strip().lower()
    cur_assist = assistant_response.strip().lower()
    previous_context = [
        t for t in previous_turns
        if (t.get("content") or "").strip().lower() not in (cur_user, cur_assist)
    ]

    candidates = await _collect_extraction_candidates(
        user_input, assistant_response, recall_memories, session_id,
    )

    prompt_user, m_map = _build_extraction_user_prompt(
        user_input, assistant_response, candidates, previous_context,
    )

    messages_base = [
        {"role": "system", "content": _MEMORY_EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_user},
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
        '{"facts": [{"evidence": "<trecho literal curto>", '
        '"text": "<fato autocontido em 3ª pessoa>", '
        '"type": "fact"|"preference"|"decision"|"instruction", '
        '"related": [], "rationale": "<uma frase curta>", '
        '"action": "ADD"|"NOOP"|"UPDATE"|"CONFLICT"|"MERGE"|"DISCARD", '
        '"target": [], "validity": "permanent"|"temporal"}]}\n'
        "Regras: target=[] para ADD e DISCARD; 1 ID (M<n>) para NOOP, UPDATE e "
        "CONFLICT; 1 ou mais para MERGE. Máximo de 8 fatos. "
        "Se nada merecer ser gravado, responda exatamente: "
        '{"facts": []}'
    )
    fallback_model = MEMORY_EXTRACTOR_FALLBACK_MODEL or MAIN_MODEL
    # Bloco de roteamento de provider (curl `provider.only`) — só acompanha as
    # tentativas com o MODELO EXTRATOR: o fallback pode rodar em outro modelo
    # que não esteja no provider configurado. Só o OpenRouter entende este
    # campo — em qualquer outro backend ele viraria 400 (gate aqui + strip
    # defensivo em _llm_post).
    provider_block: Optional[dict] = None
    if LLM_IS_OPENROUTER and MEMORY_EXTRACTOR_PROVIDERS.strip():
        provider_block = {
            "only": [p.strip() for p in MEMORY_EXTRACTOR_PROVIDERS.split(",") if p.strip()],
            "allow_fallbacks": True,
        }
    attempts: list[tuple[str, Optional[dict], str, bool]] = [
        (MEMORY_EXTRACTOR_MODEL, json_schema_fmt, "", True),
        (MEMORY_EXTRACTOR_MODEL, None, _plain_json_instruction, True),
        (fallback_model, json_schema_fmt, "", False),
    ]

    content = ""
    last_err = ""
    for attempt_i, (model, fmt, suffix, use_provider) in enumerate(attempts, start=1):
        fmt_label = fmt.get("type") if fmt else "sem response_format"
        msg_user = messages_base[-1]["content"] + suffix if suffix else messages_base[-1]["content"]
        payload: dict = {
            "model":       model,
            "messages":    messages_base[:-1] + [{"role": "user", "content": msg_user}],
            "temperature": MEMORY_EXTRACTOR_TEMPERATURE,
            "max_tokens":  1500,
        }
        if fmt:
            payload["response_format"] = fmt
        if use_provider and provider_block:
            payload["provider"] = provider_block
        try:
            # max_retries=0: a cadeia de tentativas (json_schema → plain JSON →
            # modelo fallback) é o retry desta função — o _llm_post só faz o
            # strip de capability em 400 (ex.: backend sem json_schema).
            r = await _llm_post(payload, max_retries=0, label="[MEMORY-EXTRACT]")
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

    facts_raw = parsed.get("facts", []) if isinstance(parsed, dict) else []

    # ── Pós-validação + roteamento ──────────────────────────────────────────
    # Qualquer ID fora do padrão ^M\d+$ ou fora do mapa M→id real é
    # DESCARTADO — modelos pequenos alucinam IDs; um id que não corresponde a
    # nenhuma memória enviada não tem destino válido.
    today_iso = datetime.date.today().isoformat()

    def _resolve_ids(m_ids) -> list[int]:
        resolved: list[int] = []
        for mid in (m_ids or []):
            match = re.match(r"^M(\d+)$", str(mid).strip())
            if not match:
                log.warning(f"[MEMORY-EXTRACT] id inválido {mid!r} — descartado.")
                continue
            real_id = m_map.get(int(match.group(1)))
            if real_id is None:
                log.warning(
                    f"[MEMORY-EXTRACT] id {mid!r} fora do mapa (M1..M{len(m_map)}) "
                    "— descartado."
                )
                continue
            resolved.append(real_id)
        return resolved

    def _lt_payload(text: str, validity: str) -> dict:
        # permanent → fato estável (preferência, identidade, decisão estrutural):
        #             não decai, confidence alta.
        # temporal  → fato que pode mudar sozinho (versão, local, projeto): decai.
        return {
            "text":        text,
            "source":      "chat:extractor",
            "confidence":  0.9 if validity == "permanent" else 0.65,
            "forgettable": validity != "permanent",
            "ttl_days":    None,
        }

    saved: list[dict] = []
    add_payloads: list[dict] = []
    update_ops: list[tuple[dict, int]] = []

    for fact in facts_raw:
        if not isinstance(fact, dict):
            continue
        text = (fact.get("text") or "").strip()
        if not text:
            continue
        action = str(fact.get("action") or "").strip().upper()
        if action not in ("ADD", "NOOP", "UPDATE", "CONFLICT", "MERGE", "DISCARD"):
            log.warning(
                f"[MEMORY-EXTRACT] action desconhecida {action!r} — fato pulado: "
                f"{text[:80]!r}"
            )
            continue
        validity = fact.get("validity")
        if validity not in ("permanent", "temporal"):
            validity = "temporal"
        targets = _resolve_ids(fact.get("target"))
        related_ids = _resolve_ids(fact.get("related"))
        evidence = (fact.get("evidence") or "").strip()[:200]
        rationale = (fact.get("rationale") or "").strip()[:200]

        entry: dict = {
            "content":      text,
            "type":         fact.get("type") or "fact",
            "action":       action,
            "target_ids":   targets,
            "related_ids":  related_ids,
            "validity":     validity,
            "target":       "LT",
            "forgettable":  validity != "permanent",
        }
        if evidence:
            entry["evidence"] = evidence
        if rationale:
            entry["rationale"] = rationale

        if action == "ADD":
            add_payloads.append(_lt_payload(text, validity))
        elif action in ("UPDATE", "MERGE"):
            if not targets:
                # UPDATE/MERGE alegado mas alvo inválido → downgrade para ADD
                # para não perder o fato.
                log.warning(
                    f"[MEMORY-EXTRACT] {action} sem alvo válido — downgrade para ADD. "
                    f"rationale={rationale!r}"
                )
                add_payloads.append(_lt_payload(text, validity))
                entry["action"] = "ADD"
                entry["target_ids"] = []
            else:
                target_id = targets[0]
                if action == "MERGE" and len(targets) > 1:
                    # Sem estado "superseded" no DB: só o 1º alvo é atualizado
                    # com o texto unificado; os demais ficam intocados (ficam
                    # redundantes, mas nada é perdido).
                    log.info(
                        f"[MEMORY-EXTRACT] MERGE com {len(targets)} alvos — só o "
                        f"primeiro ({target_id}) é atualizado; demais ignorados: "
                        f"{targets[1:]}"
                    )
                # O "text" do fato já deve conter TODA a informação relevante
                # das fontes (regra do prompt) — o update sobrescreve com ele.
                update_ops.append((_lt_payload(text, validity), target_id))
                entry["target_ids"] = [target_id]
        elif action == "CONFLICT":
            # Cria o fato novo E marca a memória antiga preservando o texto
            # original com um prefixo de conflito (não há flag de conflito no
            # DB — assim nada é destruído).
            add_payloads.append(_lt_payload(text, validity))
            if targets:
                old = next((c for c in candidates if c.get("id") == targets[0]), None)
                old_text = (old.get("text") or "").strip() if old else ""
                if old_text:
                    update_ops.append((
                        {
                            "text":        f"[CONFLITO {today_iso}] {old_text}",
                            "source":      old.get("source") or "chat",
                            "confidence":  old.get("confidence") or 0.65,
                            "forgettable": bool(old.get("forgettable", True)),
                            "ttl_days":    None,
                        },
                        targets[0],
                    ))
        elif action == "NOOP" and not targets:
            # NOOP sem alvo: sem destino, o fato viraria órfão — trata como
            # DISCARD para o registro refletir o que de fato aconteceu.
            log.info("[MEMORY-EXTRACT] NOOP sem alvo — tratado como DISCARD.")
            entry["action"] = "DISCARD"
        # DISCARD → apenas registrado em `saved`; nada é gravado.

        saved.append(entry)

    if add_payloads:
        # fire-and-forget: o /chat não espera a gravação terminar para
        # responder; os N fatos ADD viram 1 chamada MCP de batch.
        asyncio.create_task(memory_write_facts_batch(add_payloads))
    for upd_payload, upd_target in update_ops:
        asyncio.create_task(_memory_write_update(upd_payload, upd_target))

    log.info(
        f"[MEMORY-EXTRACT] {len(saved)} fato(s) processado(s) "
        f"(adds={len(add_payloads)}, updates={len(update_ops)}, "
        f"discards={sum(1 for s in saved if s['action'] == 'DISCARD')}, "
        f"session={session_id}, modelo={model})."
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
    active_project: Optional[str] = None,
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

    NEW (projetos): `active_project` (slug ≠ None) injeta o hint de escopo
    como mensagem system — o /read mescla o escopo do projeto com o geral e
    fatos novos pertencem ao projeto. None = modo geral, sem system extra
    (prefixo de cache inalterado).
    """
    messages = []

    # NEW (projetos): hint de escopo — vira o [0] do array (system), mantendo
    # o recall de memória como assistant logo depois.
    if active_project and active_project != "general":
        messages.append({
            "role": "system",
            "content": (
                f"Active project: {active_project}. Your long-term memory currently "
                f"merges this project's dedicated memory scope with your general memory; "
                f"new facts you save (memory write) belong to this project's context."
            ),
        })

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

try:
    from Modules.internal_auth import install_internal_auth, internal_bind_host
except ImportError:
    import internal_auth as _ia
    install_internal_auth = _ia.install_internal_auth
    internal_bind_host = _ia.internal_bind_host

app = FastAPI(title="AVA — LLM API", version="3.0.0")
install_internal_auth(app, service_name="LLM")

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

    # ── NEW (projetos): escopo ativo de memória (slug ou None/"general").
    # Repassado pelo orchestrator (re-afirmado a cada turno — a memory pode
    # ter reiniciado). Só muda o prompt quando um projeto está ativo: injeta
    # o hint de escopo (o /read mescla projeto+geral; fatos novos pertencem
    # ao projeto). None/"general" = modo geral, prefixo inalterado (cache).
    active_project: Optional[str] = Field(
        default=None,
        description="Slug do projeto ativo (escopo de memória). None = modo geral.",
    )

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
    """Verifica a API key do provider e faz uma chamada de teste (memory_status)
    pra API REST de memória (memory_api.py), só pra logar conectividade —
    o client em si é lazy (ver _get_memory_client) e reconecta sozinho a
    qualquer momento, então uma falha aqui não impede a API de subir."""
    if not LLM_API_KEY and not LLM_LOCAL_PROVIDER:
        log.warning(
            f"[STARTUP] LLM_API_KEY não configurada para {LLM_BASE_URL} — "
            "inferência vai falhar até a variável ser definida (provider "
            "local não exige key; LLM_PROVIDER_LOCAL=1 marca url LAN como local)."
        )
    else:
        log.info(
            f"[STARTUP] Provider {LLM_BASE_URL} configurado. "
            f"Modelo principal: {MAIN_MODEL}"
        )

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
    """Verifica se a API está no ar e se o provider de inferência (LLM_*)
    está configurado. `llm_key`: "ok" (key presente), "not-required"
    (provider local — localhost/127.0.0.1 sem key é válido) ou "missing"
    (provider remoto sem key). `openrouter_key` é o alias de compat."""
    llm_key = "ok" if LLM_API_KEY else ("not-required" if LLM_LOCAL_PROVIDER else "missing")
    return {
        "api":               "ok",
        "base_url":          LLM_BASE_URL,
        "llm_key":           llm_key,
        "openrouter_key":    llm_key,   # compat: checks antigos do install.sh
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
    messages = _build_messages(user_input, lang, memories, recent_turns,
                               active_project=req.active_project)
    log.info(f"tamanho do contexto do assistente: {str(messages).count(chr(0))} caracteres.")

    # Inferência (OpenRouter, com retry para falhas transitórias)
    t0 = time.perf_counter()
    if os.environ.get("AVA_DEBUG") == "1":
        log.info(messages)  # dump completo apenas em debug
    else:
        log.info(
            "llm request: n_msgs=%d approx_chars=%d session=%s",
            len(messages),
            sum(len(str(m.get("content", ""))) for m in messages),
            req.session_id,
        )
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
            detail=f"provider error {e.response.status_code}: {e.response.text[:300]}",
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
            _extract_and_save_memories(
                user_input, response_text,
                session_id=req.session_id,
                recall_memories=memories,
            )
        )

    # Disparar TTS em background
    if req.tts and req.voice:
        asyncio.create_task(tts_speak(response_text, req.voice, lang))

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
      {"status": "preparing"}     — modo chat: emitido IMEDIATAMENTE (a conexão
                                    SSE abre sem esperar a prep — Fase 3 TTFB);
                                    a prep do prompt (memória LT/ST + idioma)
                                    roda dentro do generator
      {"status": "ready", "prep_ms": ...} — prep concluída; inferência vai começar
      {"reasoning": "..."}   — raciocínio (se stream_reasoning)
      {"delta": "..."}       — texto do content
      {"memory_correction": {"classification": ..., "evidence": "..."}}
                              — emitido quando `watch_confidence=true` e a
                                confiança média de uma janela de frase caiu
                                abaixo do limiar, disparando releitura da
                                memória e retomada com prefill (ver
                                CONFIDENCE_* no topo do arquivo)
      {"tool_calls": [...]}  — tool_calls completas (ao final, se houver)
      {"done": true, "elapsed": ..., "prep_ms": ..., "prompt_cached_tokens": ..., "had_tool_calls": bool}
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
        base_messages = [m.model_dump(exclude_none=True) for m in req.messages]
        if not base_messages:
            raise HTTPException(status_code=400, detail="messages vazio.")
        user_input = None
        temperature = req.temperature if req.temperature is not None else 0.3
    else:
        user_input = (req.message or "").strip()
        if not user_input:
            raise HTTPException(
                status_code=400,
                detail="Envie `message` (modo chat) ou `messages` (modo tools).",
            )

        temperature = req.temperature if req.temperature is not None else 0.7

    # ── Estado mutável compartilhado endpoint ↔ generator ──────────────────
    # Fase 3: a prep do prompt (memória LT/ST + idioma) roda DENTRO do
    # generator — o StreamingResponse é devolvido imediatamente ao caller.
    # `lang` só fica definitivo após a prep; _flush_tts_buf lê daqui por closure.
    state = {"lang": "pt"}

    # ── Payload extras (independem de messages — computados fora) ──────────
    reasoning_extra = _reasoning_payload_extra(req.thinking_depth)
    if req.reasoning_effort:
        reasoning_extra = {"reasoning": {"effort": req.reasoning_effort}}

    # ── Confidence-watch: pede logprobs por token ao OpenRouter ────────────
    # Vale nos dois modos (CHAT e TOOLS). No TOOLS o guard no loop de
    # streaming impede a interrupção com tool_calls em montagem; a leitura de
    # correção usa req.session_id (o orchestrator manda o sid do /execute).
    watch_confidence = req.watch_confidence
    # ── NEW: backend sem logprobs (400 memorizado em _LLM_UNSUPPORTED) →
    # vigia desligado já na hora, sem gastar um request p/ descobrir. O probe
    # do vigia continua p/ o caso do backend ACEITAR logprobs e só ignorá-los.
    if watch_confidence and "logprobs" in _LLM_UNSUPPORTED:
        log.warning("LLM: watch_confidence ignorado — o backend rejeitou logprobs nesta sessão.")
        watch_confidence = False

    # TTS só existe no modo chat (modo tools é stateless — caller cuida do áudio)
    
    use_tts = (not tool_mode) and bool(req.tts and req.voice)
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
            _tts_queue.put_nowait((cleaned, req.voice, state["lang"]))
        _tts_buf = ""

    async def generator():
        nonlocal _tts_buf

        # ── FASE 3: prep do prompt DENTRO do stream ─────────────────────────
        # O StreamingResponse já foi devolvido ao caller (headers SSE saem
        # imediatamente; o orquestrador ignora eventos "status" desconhecidos).
        # A leitura de memória LT/ST e a detecção de idioma — todas failure-safe
        # (retornam []/"pt" em falha) — rodam aqui dentro.
        t_prep_start = time.perf_counter()
        if not tool_mode:
            yield f"data: {json.dumps({'status': 'preparing'})}\n\n"
            try:
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
                lang, memories, recent_turns = await asyncio.gather(
                    lang_task, memory_task, memory_st_task
                )
                if not lang:
                    lang = "pt"
                state["lang"] = lang

                messages = _build_messages(user_input, lang, memories, recent_turns,
                               active_project=req.active_project)
            except Exception as e:
                yield f"data: {json.dumps({'error': f'prep do prompt falhou: {e}'})}\n\n"
                return
            # ── FASE 4: tempo de PREPARAÇÃO (idioma + memória) separado ────
            prep_elapsed = time.perf_counter() - t_prep_start
            yield f"data: {json.dumps({'status': 'ready', 'prep_ms': round(prep_elapsed * 1000)})}\n\n"
        else:
            messages = base_messages
            prep_elapsed = 0.0

        log.info(
            "tamanho do contexto do assistente: n_msgs=%d approx_chars=%d session=%s",
            len(messages),
            sum(len(str(m.get("content", ""))) for m in messages),
            req.session_id,
        )

        # ── Payload OpenRouter (com tool calling nativo quando `tools` presente) ──
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
        if watch_confidence:
            payload["logprobs"] = True

        # ── FASE 4: t0 marca o início da INFERÊNCIA — o `elapsed` do log
        # final e do evento done EXCLUI o tempo de prep (reportado à parte).
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

            # Retry simples p/ falhas transitórias (5xx/rede/429) do provider
            # (strip de capability em 400 acontece dentro de _llm_post, antes
            # do SSE — aqui só sobra 4xx definitivo p/ reportar).
            for attempt in range(OPENROUTER_MAX_RETRIES + 1):
                client_used, model_used, r_ctx = await _execute_inference(
                    json_payload={**payload, "messages": current_messages},
                    stream=True,
                    model=MAIN_MODEL,
                )

                try:
                    async with r_ctx as r:
                        if r.status_code == 429 and attempt < OPENROUTER_MAX_RETRIES:
                            try:
                                wait_s = float(r.headers.get("Retry-After") or "") or OPENROUTER_RETRY_BACKOFF
                            except ValueError:
                                wait_s = OPENROUTER_RETRY_BACKOFF
                            log.warning(
                                f"LLM stream: 429. Aguardando {wait_s}s "
                                f"(tentativa {attempt + 2}/{OPENROUTER_MAX_RETRIES + 1})."
                            )
                            await asyncio.sleep(wait_s)
                            continue
                        if r.status_code >= 500 and attempt < OPENROUTER_MAX_RETRIES:
                            log.warning(
                                f"LLM stream: {r.status_code} "
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
                                f"LLM stream: {r.status_code} — {err_body[:200]}"
                            )
                            yield f"data: {json.dumps({'error': f'provider {r.status_code}: {err_body[:400]}', 'too_large': _is_request_too_large_error(err_body)})}\n\n"
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
                                        # Flush imediato em pausa forte (.!?\n。).
                                        # Flush em pausa curta (,;:) só com um mínimo
                                        # de 40 chars acumulados — evita sínteses
                                        # fragmentadas palavra a palavra a cada vírgula.
                                        # Buffer cheio (>80 chars, antes 150) força o
                                        # flush — reduz a latência percebida da voz.
                                        if buf_rstrip and buf_rstrip[-1] in '.!?\n。':
                                            _flush_tts_buf()
                                        elif buf_rstrip and buf_rstrip[-1] in ',;:' \
                                             and len(_tts_buf) > 40:
                                            _flush_tts_buf()
                                        elif len(_tts_buf) > 80:
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
                                #
                                # Modo TOOLS: com tool_calls em montagem o vigia
                                # NÃO age — interromper aqui perderia os
                                # fragmentos já acumulados, e os logprobs desse
                                # trecho são de JSON de argumento, não de prosa.
                                # (a montagem das fragments acontece no bloco
                                # `for tc in delta_obj.get("tool_calls")` logo
                                # acima, então tool_calls_acc cobre o caso de
                                # chunk misto content+tool_calls também.)
                                if watch_confidence and window is not None:
                                    if tool_calls_acc or delta_obj.get("tool_calls"):
                                        # reseta a janela — a confiança daqui
                                        # por diante não é de prosa conversável
                                        if window.text:
                                            window.reset()
                                    elif watch_events_used < CONFIDENCE_MAX_CORRECTIONS:
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
            f"[STREAM] {elapsed:.2f}s (prep: {prep_elapsed:.2f}s) | {len(full_response)} chars | "
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

        yield f"data: {json.dumps({'done': True, 'elapsed': round(elapsed, 3), 'prep_ms': round(prep_elapsed * 1000), 'prompt_cached_tokens': cached_tokens, 'had_tool_calls': bool(tool_calls_final)})}\n\n"

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
                    _extract_and_save_memories(
                        user_input, full_response,
                        session_id=req.session_id,
                        recall_memories=memories,
                    )
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
    Executa tool call no provider configurado (LLM_BASE_URL).

    Estratégia (retry e capability gating moram no transporte único
    _llm_post — um único ponto de troca de provider):
      - Erro de rede / 429 (Retry-After) / 5xx: retry com backoff.
      - 400 com capability stripável (logprobs/reasoning/response_format):
        strip + reenvio, invisível pro caller.
      - Erro de contexto grande demais: NÃO retenta — RequestTooLargeError.
      - Outros 4xx: não retry.

    MODO GRAMMAR (compatibilidade com chamadores antigos do llama-server):
      Quando `grammar` (string GBNF) é fornecida, o payload NÃO inclui
      `tools`/`tool_choice` e ativamos `response_format: {type: "json_object"}`
      no provider — isso força o output em JSON válido (sem validação de
      schema estrito). A resposta vem em `message.content` (não em
      `message.tool_calls`) e o caller é responsável por parsear.
      Nota: o OpenRouter não aceita GBNF nativamente; a gramática é
      interpretada como "forçar JSON" (backends sem response_format caem
      no strip de 400 e seguem sem a força-de-JSON).

    Retorna (message_dict, model_used, fallback_used=False sempre — um
    único provider configurado, sem fallback entre provedores).
    """
    payload: dict = {
        "model":       MAIN_MODEL,
        "messages":    messages,
        "temperature": temperature,
    }
    if reasoning_effort:
        # reasoning.effort é o formato OpenAI/OpenRouter p/ controlar raciocínio
        payload["reasoning"] = {"effort": reasoning_effort}

    # ── Modo grammar vs. tool-calling nativo ─────────────────────────────
    # Mutuamente exclusivos: se grammar está presente, usamos response_format
    # para forçar JSON (equivalente aproximado do GBNF do llama-server).
    if grammar:
        payload["response_format"] = {"type": "json_object"}
    elif tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"

    r = await _llm_post(payload, max_retries=max_retries, label="LLM /chat/tools")

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
        log.warning(f"LLM /chat/tools: contexto excede o limite. Erro: {err_msg[:200]}")
        raise RequestTooLargeError(f"Contexto excede o limite do modelo: {err_msg[:300]}")

    # Rede/429/5xx já esgotaram o retry dentro de _llm_post — aqui só sobra
    # 4xx definitivo (ou 5xx persistente).
    raise RuntimeError(f"LLM /chat/tools: {r.status_code} - {err_msg[:300]}")


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
    uvicorn.run(app, host=internal_bind_host(), port=4003, log_level="info")