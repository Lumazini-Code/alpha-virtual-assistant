"""
AVA Orchestrator — Tool-Calling Execution Engine
=================================================
Arquitetura (tool-calling nativo via chat template):

  1. O LLM recebe o histórico completo + a lista de tools no formato OpenAI
     (function-calling). As definições das tools são embutidas no prompt e o
     backend devolve `message.tool_calls` já estruturado (nome + argumentos
     JSON), sem precisarmos de uma segunda chamada com grammar manual.
  2. Se a resposta NÃO tiver tool_calls, o texto é tratado como resposta
     final direta (equivalente a um "finish" implícito) e o loop termina —
     não há re-chamada com o histórico inalterado, pois isso violaria a
     alternância estrita user/assistant/tool exigida pelo chat template.
  3. Se houver tool_calls, cada uma é executada; o resultado volta ao
     histórico como mensagem role="tool" (atrelada ao tool_call_id) e o
     ciclo reinicia.
  4. O loop termina quando o modelo chama a tool "finish" (cujos
     argumentos contêm a resposta final) ou quando responde sem chamar
     nenhuma tool (resposta direta = finish implícito). Não há limite
     artificial de rodadas além disso.

Resumo do fluxo:
  pergunta → LLM (+ tools) → [sem tool_calls: continua o loop]
                            → [com tool_calls X: executa X
                               → resultado (role=tool) → repete]
                            → [com tool_call finish: FIM]

Tools NATIVAS (permanecem hardcoded — acopladas demais ao loop pra virar MCP):
  - vision_objects: reusa o histórico real da conversa. Auto-finish especial.
  - finish: primitiva de controle do próprio loop, não uma capacidade externa.
TTS NÃO é uma tool do modelo — o sistema dispara TTS automaticamente sobre a
resposta final (igual antes), o modelo nunca decide chamar TTS.

Tools via MCP (descoberta em duas etapas — ver MCP_SKILLS):
  memory (read/write), search, read_url, alpha_code,
  playwright (automação de navegador — microsoft/playwright-mcp, via npx,
  conectado ao Chrome REAL do usuário via CDP — ver _ensure_chrome_debug —
  em vez de um Chromium automatizado isolado, pra evitar detecção antibot).
  Em vez de um executor Python hardcoded por tool + conversão manual pro
  schema OpenAI, cada um desses agora é um servidor MCP separado. O loop
  primeiro escolhe a "skill" (servidor) relevante pra pergunta do usuário
  usando só os resumos em MCP_SKILLS (~1 linha cada, barato em tokens),
  e SÓ ENTÃO conecta e lista as tools reais daquele servidor.

  Memory deixou de falar REST e virou MCP (`mcp_servers/memory_server.py`,
  que importa a lógica de `Modules/memory.py` diretamente) — mas continua
  sendo um processo HTTP compartilhado (streamable-http, porta 3001,
  MEMORY_MCP_URL), porque LLM.py também precisa enxergar o MESMO estado
  (SQLite/FAISS). Diferente das demais skills MCP (playwright etc.), que
  este orchestrator spawna como subprocesso stdio sob demanda, "memory" é
  um processo externo que precisa estar de pé antes do primeiro uso (ver
  MCP_SKILLS["memory"], transport="http"). Tanto as leituras/escritas que o
  modelo decide chamar quanto o bookkeeping interno do orchestrator (turnos
  de curto prazo, limpeza de sessão) passam pela mesma sessão MCP — ver
  `_get_mcp_session("memory")`
  e o helper `_call_memory_tool(...)`.

Integra os microserviços AVA que continuam como acesso HTTP direto:
  - TTS             (port 3004)  — text-to-speech (Supertonic) — SISTEMA, não tool
  - LLM Chat        (port 4003)  — conversational inference (OpenRouter)
  - Vision / VQA    (port 4002)  — image understanding (tool nativa)
Process manager (port 9001) segue controlando o ambiente Docker.
"""
from __future__ import annotations

import json
import asyncio
import base64
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager, AsyncExitStack
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional
import datetime
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

# ── MCP (Model Context Protocol) — cliente usado pelas tools externas ──
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

# MEMORY_URL (REST, porta 3001) removido — a memória não é mais um serviço
# REST. Ela agora é o servidor MCP "memory" (ver MCP_SKILLS), rodando como
# processo HTTP COMPARTILHADO (mcp_servers/memory_server.py, streamable-http)
# — não é mais spawnado como subprocesso stdio por este orchestrator, porque
# LLM.py também precisa falar com o MESMO processo/estado (SQLite/FAISS).
# Acessado via _get_mcp_session("memory") / _call_memory_tool(...). Mesma env
# var (mesmo default) usada por LLM.py — os dois processos devem apontar
# para a mesma instância do memory_server.py.
MEMORY_MCP_URL       = os.getenv("MEMORY_MCP_URL", "http://localhost:3001/mcp")
TTS_URL              = "http://localhost:3004"
LLM_URL              = "http://localhost:4003"
VISION_URL           = "http://localhost:4002"
DEEP_SEARCH_URL      = "http://localhost:4005"
ALPHA_CODE_URL       = "http://localhost:4006"
# shell_command_api.py — abre/derruba o Chrome real do usuário (debugging
# remoto via CDP), usado para dar ao playwright-mcp um navegador com
# fingerprint/cookies/sessão genuínos em vez do Chromium automatizado que
# ele lançaria sozinho (ver MCP_SKILLS["playwright"] e _ensure_chrome_debug).
#
# ATENÇÃO — CONFLITO DE PORTA: o default de shell_command_api.py
# (AGENT_SHELL_API_PORT) também é 4005, igual ao DEEP_SEARCH_URL acima.
# Suba o shell_command_api.py com `AGENT_SHELL_API_PORT=4007` (env var) ou
# ajuste SHELL_COMMAND_URL/CHROME_DEBUG_PORT abaixo para combinar com a
# porta que você realmente usar.
SHELL_COMMAND_URL    = "http://localhost:4007"
CHROME_DEBUG_PORT    = 9222  # mesmo default de COMMAND_REGISTRY["launch_chrome_debug"]
# Gerenciador de processos (Rust/axum) que sobe/derruba o ambiente Docker.
PROCESS_MANAGER_URL  = "http://localhost:9001"

# ══════════════════════════════════════════════════════════════════════════════
# MCP Skills Registry — descoberta em duas etapas
# ══════════════════════════════════════════════════════════════════════════════
# Etapa 1: o modelo vê só `skill_summary` (barato, ~10-20 tokens cada) e
# escolhe quais servidores são relevantes pro pedido do usuário.
# Etapa 2: SÓ ENTÃO conectamos (stdio) e chamamos list_tools() nos
# servidores escolhidos, carregando o schema completo das tools reais.
#
# Substitui os antigos executores hardcoded (_tool_memory_read,
# _tool_memory_write, _tool_search, _tool_read_url,
# _tool_alpha_code) — cada um vira, em vez de uma função Python fixa, um
# servidor MCP que expõe sua própria lista de tools dinamicamente.
@dataclass
class MCPSkill:
    skill_summary: str                 # descrição curta, usada na etapa 1
    # ── stdio (default): subimos um subprocesso próprio por skill ──
    command: Optional[str] = None      # executável do servidor MCP (stdio)
    args: list[str] = field(default_factory=list)
    env: Optional[dict[str, str]] = None
    # Hook opcional, chamado (awaited) uma única vez, ANTES de subir o
    # processo stdio, na primeira conexão da skill. Devolve uma lista de
    # args extras a concatenar em `args` — usado pelo "playwright" pra
    # resolver dinamicamente o --cdp-endpoint do Chrome real (só se sabe a
    # porta depois de perguntar pro shell_command_api). None = nenhum extra.
    pre_connect: Optional[Callable[[], "asyncio.Future[list[str]]"]] = None
    # ── http: conecta a um processo MCP JÁ RODANDO (streamable-http) em vez
    # de spawnar um subprocesso — necessário para skills com estado que
    # precisa ser compartilhado com OUTROS processos além deste orchestrator
    # (ex.: "memory", também consumido por LLM.py). Quando transport="http",
    # `url` é obrigatório e `command`/`args`/`pre_connect` são ignorados.
    transport: Literal["stdio", "http"] = "stdio"
    url: Optional[str] = None

MCP_SKILLS: dict[str, MCPSkill] = {
    "memory": MCPSkill(
        skill_summary="Grava e busca fatos/contexto salvos na memória de longo/curto prazo.",
        # HTTP, não stdio: o memory_server.py roda como processo único e
        # compartilhado (ver seu docstring) — LLM.py conecta na MESMA
        # instância para que SQLite/FAISS fiquem consistentes entre os dois
        # consumidores. Suba-o separadamente (ex.: via start.sh) ANTES do
        # orchestrator ou do LLM.py tentarem usar a skill "memory".
        transport="http", url=MEMORY_MCP_URL,
    ),
    "playwright": MCPSkill(
        skill_summary=(
        "Controla um navegador de verdade (Chrome/Chromium) para acessar sites, clicar em "
        "botões/links, preencher formulários, rolar a página, tirar screenshot e ler o "
        "conteúdo real de uma página depois de carregada — inclusive sites que mudam "
        "dinamicamente (JavaScript, login, resultados de busca renderizados no navegador). "
        "Use sempre que o pedido envolver ABRIR, NAVEGAR, VISITAR ou INTERAGIR com um site "
        "específico (ex: 'abra o YouTube', 'entra no meu email', 'clica no primeiro "
        "resultado', 'tira um print da página', 'preenche esse formulário'). "
        "Diferente de busca simples: isso não é pra pesquisar um termo na web, é pra "
        "efetivamente controlar um navegador e ver/interagir com uma página real."
    ),  command="npx", args=["-y", "@playwright/mcp@latest"],
        pre_connect=lambda: _ensure_chrome_debug(),
    ),
    # Novos MCPs (ex.: sandbox, telegram, tts) entram aqui — só precisam de
    # um `skill_summary` e dos parâmetros de conexão, nada no loop principal
    # muda pra suportar um servidor a mais.
}

# "memory" removido daqui — não é mais um serviço HTTP com endpoint de
# health próprio; sua saúde é checada via MCP (ver /status).
HEALTH_PATHS: dict[str, str] = {
    "search": "/status", "tts": "/status",
    "llm": "/health", "vision": "/vision/status", "deep_search": "/health", "alpha_code": "/health",
    "process_manager": "/status",
}

# Timeout por tool (usado tanto para chamadas de tool quanto pro loop de seleção)
# memory_read/memory_write saíram daqui — agora são tools MCP, com timeout
# controlado pela própria sessão MCP, não por este dict de tools nativas.
EXECUTOR_TIMEOUTS: dict[str, float] = {
    "llm": 9999999.0, "search": 60.0,
    "read_url": 30.0, "deep_search": 9999999.0, "vision_objects": 300.0, "tts": 60.0,
    "alpha_code": 9999999.0,
}

MAX_CONTEXT_CHARS = 3000
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

logging.basicConfig(
    level=logging.DEBUG if os.environ.get("AVA_DEBUG") else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ava.orchestrator")


# ══════════════════════════════════════════════════════════════════════════════
# Data Models
# ══════════════════════════════════════════════════════════════════════════════

class ExecuteRequest(BaseModel):
    input:         str
    session_id:    Optional[str]  = None
    voice:         str            = "M1"
    lang:          str            = "pt"
    tts:           bool           = True
    # Base64 puro da imagem (sem prefixo "data:...;base64,"), já codificado
    # pelo cliente (TUI) — o orchestrator NUNCA lê arquivo de disco nem
    # decodifica/recodifica isso, só repassa para o vision service. Isso
    # elimina o problema de path resolution entre cliente e orchestrator
    # (ex.: orchestrator rodando em container sem o filesystem do host).
    image_base64:  Optional[str]  = None
    search_pdfs:   bool           = False
    stream:        bool           = True

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
    forgettable: bool = True
    ttl_days: Optional[float] = None
    action: Literal["create", "update"] = "create"
    memory_id: Optional[int] = None

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
    # memory_client (httpx REST) removido — memória agora é MCP, ver
    # mcp_sessions["memory"] / _call_memory_tool().
    search_client: httpx.AsyncClient = field(default=None)
    tts_client: httpx.AsyncClient = field(default=None)
    llm_client: httpx.AsyncClient = field(default=None)
    vision_client: httpx.AsyncClient = field(default=None)
    deep_search_client: httpx.AsyncClient = field(default=None)
    alpha_code_client: httpx.AsyncClient = field(default=None)
    process_manager_client: httpx.AsyncClient = field(default=None)
    # shell_command_api.py — abre/derruba o Chrome real (debug CDP), usado
    # pela skill "playwright" via _ensure_chrome_debug.
    shell_client: httpx.AsyncClient = field(default=None)
    # ── MCP: pilha de contexto que mantém os processos stdio vivos, e um
    # cache de sessões já conectadas (lazy — só conecta na primeira vez que
    # uma skill é escolhida, não em todas as MCP_SKILLS no boot) ──
    mcp_stack: AsyncExitStack = field(default=None)
    mcp_sessions: dict[str, ClientSession] = field(default_factory=dict)

state = AppState()


async def _ensure_chrome_debug() -> list[str]:
    """`pre_connect` da skill "playwright": garante que existe um Chrome de
    verdade (com o profile REAL do usuário — cookies, sessões logadas,
    extensões) escutando debugging remoto via CDP, pedindo isso ao
    shell_command_api.py (mesmo padrão de client HTTP do resto do
    orchestrator — ver alpha_code.py). Devolve os args extras
    (`--cdp-endpoint ...`) que fazem o playwright-mcp conectar NELE em vez
    de lançar seu próprio Chromium automatizado — é isso que evita o
    antibot: aos olhos do site, é o mesmo Chrome que o usuário já usa
    normalmente, não um navegador headless/isolado recém-criado.

    `launch_chrome_debug` é idempotente (o próprio shell_command_api
    devolve `already_running: True` se a porta já estiver de pé), então
    chamar isso toda vez que a skill "playwright" conecta é seguro — não
    abre um Chrome novo a cada conversa.
    """
    if state.shell_client is None:
        state.shell_client = httpx.AsyncClient(
            base_url=SHELL_COMMAND_URL, timeout=httpx.Timeout(30.0, connect=5.0),
        )
    try:
        r = await state.shell_client.post("/command", json={
            "command": "launch_chrome_debug",
            "params": {"port": CHROME_DEBUG_PORT},
        })
        r.raise_for_status()
        data = r.json()
        if not data.get("success"):
            raise RuntimeError(data.get("error") or "launch_chrome_debug falhou")
        port = (data.get("result") or {}).get("port", CHROME_DEBUG_PORT)
    except Exception as e:
        # Falha aqui não deveria travar o resto do orchestrator — só a
        # skill "playwright" fica indisponível até o shell_command_api
        # voltar. Propaga pra _get_mcp_session logar e devolver erro à
        # tool call, em vez de silenciosamente conectar num Chrome que não
        # existe.
        raise RuntimeError(
            f"Não consegui garantir o Chrome debug via shell_command_api "
            f"({SHELL_COMMAND_URL}): {e}"
        ) from e

    log.info(f"Chrome debug pronto na porta {port} (via shell_command_api) — playwright-mcp vai conectar via CDP")
    return ["--cdp-endpoint", f"http://localhost:{port}"]


async def _get_mcp_session(skill_name: str) -> ClientSession:
    """Retorna a ClientSession do servidor MCP daquela skill, conectando
    (e mantendo viva pro resto do processo, via state.mcp_stack) na
    primeira vez que a skill é usada. Duas topologias, conforme
    `skill.transport`:
      - "stdio" (default): spawna um subprocesso próprio para esta skill.
      - "http": conecta via streamable-http a um processo MCP que já está
        rodando de forma independente (ex.: "memory" — precisa ser o MESMO
        processo que LLM.py também usa, não uma cópia spawnada aqui)."""
    if skill_name in state.mcp_sessions:
        log.debug(f"[_get_mcp_session] '{skill_name}' já conectado, reusando sessão")
        return state.mcp_sessions[skill_name]
    skill = MCP_SKILLS[skill_name]
    t0 = time.perf_counter()

    if skill.transport == "http":
        if not skill.url:
            raise RuntimeError(f"MCP skill '{skill_name}': transport='http' sem `url` configurado")
        log.debug(f"[_get_mcp_session] '{skill_name}': conectando via streamable-http em {skill.url}")
        try:
            read, write, _ = await state.mcp_stack.enter_async_context(streamable_http_client(skill.url))
        except Exception as e:
            log.error(
                f"[_get_mcp_session] '{skill_name}': falha ao conectar em {skill.url} — "
                f"o processo do servidor MCP está rodando? ({type(e).__name__}: {e})"
            )
            raise
        session = await state.mcp_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        state.mcp_sessions[skill_name] = session
        log.info(f"MCP: conectado à skill '{skill_name}' (http, {skill.url}) em {(time.perf_counter()-t0)*1000:.0f}ms")
        return session

    # ── stdio: comportamento original (subprocesso próprio) ──
    args = list(skill.args)
    if skill.pre_connect is not None:
        log.debug(f"[_get_mcp_session] '{skill_name}': rodando pre_connect...")
        try:
            extra_args = await skill.pre_connect()
        except Exception as e:
            log.error(f"[_get_mcp_session] '{skill_name}': pre_connect FALHOU — skill "
                      f"ficará indisponível nesta rodada: {type(e).__name__}: {e}")
            raise
        log.debug(f"[_get_mcp_session] '{skill_name}': pre_connect ok, args extras: {extra_args}")
        args.extend(extra_args)
    log.debug(f"[_get_mcp_session] '{skill_name}': lançando `{skill.command} {' '.join(args)}`")
    server_params = StdioServerParameters(command=skill.command, args=args, env=skill.env)
    read, write = await state.mcp_stack.enter_async_context(stdio_client(server_params))
    session = await state.mcp_stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    state.mcp_sessions[skill_name] = session
    log.info(f"MCP: conectado à skill '{skill_name}' (stdio) em {(time.perf_counter()-t0)*1000:.0f}ms")
    return session


async def _call_memory_tool(tool_name: str, arguments: dict) -> Any:
    """Chama uma tool do servidor MCP "memory" (mcp_servers/memory_server.py,
    processo HTTP compartilhado — ver MCP_SKILLS["memory"]) e devolve o
    resultado já desserializado (dict/list/str). Substitui as antigas
    chamadas REST `state.memory_client.post(...)`.

    Prefere `structuredContent` (populado automaticamente pelo FastMCP/
    MCPServer para tools com retorno tipado) e cai para parsing do primeiro
    content-block como JSON quando ausente — mesma lógica de
    `_unwrap_tool_result` em LLM.py, para os dois consumidores tratarem a
    resposta do mesmo jeito.

    Levanta a exceção original se a tool reportar erro (isError=True) ou se
    a conexão MCP falhar — quem chama decide como tratar (fallback,
    HTTPException, log-and-ignore, etc.).
    """
    session = await _get_mcp_session("memory")
    result = await session.call_tool(tool_name, arguments)
    if getattr(result, "isError", False):
        detail = result.content[0].text if result.content else "erro desconhecido"
        raise RuntimeError(f"memory tool '{tool_name}' falhou: {detail}")
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    if not result.content:
        return None
    text = result.content[0].text
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Iniciando AVA Orchestrator (tool-calling engine, sem router/CoT)...")

    service_urls = {
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
    # "memory" não entra no probe HTTP simples acima (é MCP, não REST puro),
    # mas AGORA é um processo externo de longa duração que precisa estar de
    # pé ANTES do primeiro uso — diferente das demais skills MCP (stdio),
    # que este orchestrator spawna sozinho sob demanda. Sua conexão é lazy
    # (ver _get_mcp_session) e só é testada quando a skill "memory" é
    # escolhida pela primeira vez; se o processo não estiver rodando em
    # MEMORY_MCP_URL, essa primeira chamada falhará com erro de conexão.
    log.info(f"  ℹ memory           serviço MCP compartilhado esperado em {MEMORY_MCP_URL} "
             f"(suba mcp_servers/memory_server.py separadamente antes do primeiro uso)")

    def _make_client(base_url: str, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout), limits=httpx.Limits(max_keepalive_connections=4, max_connections=8))

    state.tts_client = _make_client(TTS_URL, 9999999.0)
    state.llm_client = _make_client(LLM_URL, 9999999.0)
    state.vision_client = _make_client(VISION_URL, 9999999.0)
    state.deep_search_client = _make_client(DEEP_SEARCH_URL, 9999999.0)
    state.alpha_code_client = _make_client(ALPHA_CODE_URL, 9999999.0)
    # Timeout curto e finito (não 9999999.0): chamadas ao gerenciador de
    # processos (status) devem falhar rápido se ele estiver
    # offline, em vez de travar o loop de tools esperando para sempre.
    state.process_manager_client = _make_client(PROCESS_MANAGER_URL, 15.0)

    # MCP: só prepara a pilha de contexto — conexão real é lazy (ver
    # _get_mcp_session), disparada quando o loop escolhe uma skill pela
    # primeira vez. Conectar tudo aqui no boot atrasaria a inicialização
    # à toa com servidores que talvez nunca sejam usados nesta sessão.
    state.mcp_stack = AsyncExitStack()
    log.info(f"MCP: {len(MCP_SKILLS)} skills registradas (conexão lazy): {list(MCP_SKILLS.keys())}")

    log.info("Orchestrator pronto — todos os clientes HTTP inicializados")
    yield
    for c in (state.tts_client, state.llm_client,
              state.vision_client, state.deep_search_client,
              state.alpha_code_client, state.process_manager_client):
        await c.aclose()
    if state.shell_client is not None:
        await state.shell_client.aclose()
    await state.mcp_stack.aclose()
    log.info("AVA Orchestrator encerrado")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers de resultado / contexto
# ══════════════════════════════════════════════════════════════════════════════

def _result_to_text(res) -> str:
    if not res:
        return ("(empty result — this tool returned nothing usable. Do NOT repeat the same "
                 "query worded differently; either try a materially different angle once, "
                 "or call `finish` now and be upfront that you couldn't find the information.)")
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
        if not res:
            return ("(empty result — this tool returned zero items. Do NOT repeat the same "
                     "query worded differently; either try a materially different angle once, "
                     "or call `finish` now and be upfront that you couldn't find the information.)")
        # Structured rendering (search-like items): keep title/url separate from
        # the snippet so the model can actually judge relevance/date/source
        # instead of an undifferentiated wall of text truncated mid-sentence.
        lines = []
        for idx, item in enumerate(res[:5]):
            if isinstance(item, dict):
                title = item.get("title") or item.get("name")
                url = item.get("url") or item.get("link") or item.get("source")
                snippet = (item.get("text") or item.get("content") or item.get("snippet") or "").strip()
                snippet = snippet[:600]
                header = f"[{idx + 1}] " + (title or "(untitled)")
                if url:
                    header += f" — {url}"
                lines.append(header if not snippet else f"{header}\n    {snippet}")
            else:
                lines.append(f"[{idx + 1}] {str(item)[:280]}")
        return ("\n".join(lines))[:MAX_CONTEXT_CHARS]
    return str(res)[:MAX_CONTEXT_CHARS]


async def _fire_tts(text, req):
    """TTS é disparado pelo SISTEMA sobre a resposta final — o modelo nunca chama TTS."""
    try:
        await state.tts_client.post("/speak", json={"text": text[:2000], "voice": req.voice, "lang": req.lang})
    except Exception:
        pass

async def _save_turn(u, a, sid):
    try:
        await _call_memory_tool("memory_write_short_term", {
            "session_id": sid,
            "turns": [{"role": "user", "content": u}, {"role": "assistant", "content": a}],
        })
    except Exception:
        pass

async def _save_lt(text, src="chat"):
    try:
        await _call_memory_tool("memory_write", {"text": text[:500], "source": src, "confidence": 0.8})
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

async def _verify_think(text: str) -> int:
    """
    Classifica a profundidade de raciocínio (0-10) necessária para o input.
    O llama-server local que fazia essa classificação foi removido — o
    backend de substituição será plugado aqui (a chamada sobe exceção até lá;
    _think_instruction trata a falha e usa o depth padrão).
    """
    raise NotImplementedError(
        "Classificador de thinking-depth sem backend (llama-server removido; "
        "aguardando substituição)."
    )

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

# _tool_memory_read, _tool_memory_write, _tool_search, _tool_read_url e
# _tool_deep_search foram REMOVIDOS — memory e search agora são servidores
# MCP (ver MCP_SKILLS + _get_mcp_session), descobertos e despachados
# dinamicamente pelo loop em vez de uma função Python fixa por tool.

# ── Tradução de Objetos: vision.py (crops) -> LLM.py (multi-imagem) ──

# Dá acesso, de dentro de um executor de tool, ao histórico REAL da
# conversa deste turno (system + user + tool_calls já resolvidas) sem
# precisar mudar a assinatura `executor(args, req)` de todas as tools —
# só quem realmente precisa de contexto conversacional (vision_objects)
# lê isso. Setado pelo loop principal logo antes de cada `executor(...)`.
_CURRENT_HISTORY: ContextVar[list[dict]] = ContextVar("_current_history", default=[])


def _sanitize_history_for_submodel(history: list[dict]) -> list[dict]:
    """Cópia do histórico pronta pra ser usada numa chamada paralela ao LLM
    (ex.: o passo multimodal dentro de vision_objects). Remove a última
    mensagem se for um assistant com tool_calls ainda pendentes (a tool
    que está rodando agora mesmo) — do contrário a alternância estrita
    user/assistant/tool exigida pelo chat template quebra, porque essa
    tool_call ainda não tem uma mensagem role="tool" respondendo ela."""
    if history and history[-1].get("role") == "assistant" and history[-1].get("tool_calls"):
        return list(history[:-1])
    return list(history)


def _vision_object_context_line(obj: dict) -> str:
    """Texto de contexto pra acompanhar um crop — já resolvido (nome de
    pessoa via face-dict, ou candidato do dicionário visual de objetos).
    Isso é só CONTEXTO pro modelo multimodal formar a resposta; ele não tem
    (nem precisa ter) acesso a nenhum mecanismo de verificação por trás
    disso — só lê o que já veio pronto do vision.py.

    Pra rosto reconhecido, o texto é deliberadamente afirmativo ("É ... .")
    em vez de sugestivo ("pode ser...") — LLMs multimodais tendem a se
    recusar a nomear pessoas em fotos por reflexo de segurança/privacidade,
    mesmo quando a identidade já veio resolvida de um banco de dados local
    que o próprio usuário cadastrou. Framing como fato de banco de dados
    (não como "reconhecimento facial ao vivo feito pelo modelo") reduz
    esse hedging."""
    idx = obj["object_index"]
    if obj.get("is_face"):
        if obj.get("person_name"):
            desc = f" Descrição cadastrada: {obj['person_description']}" if obj.get("person_description") else ""
            return (
                f"[nota interna sobre recorte {idx}, não mencione essa nota nem o número do "
                f"recorte na resposta] já consultamos o banco de dados local de rostos "
                f"cadastrados pelo próprio usuário — este rosto É {obj['person_name']} "
                f"(confirmado, não é uma sugestão).{desc} Refira-se a essa pessoa pelo nome, "
                "naturalmente, como se você mesmo tivesse reconhecido."
            )
        return (
            f"[nota interna sobre recorte {idx}, não mencione essa nota na resposta] rosto "
            "detectado, mas não encontrado no banco de rostos cadastrados — pessoa "
            "desconhecida/não cadastrada."
        )
    candidates = obj.get("candidates") or []
    if candidates:
        top = candidates[0]
        return (
            f"[nota interna sobre recorte {idx}, não mencione essa nota na resposta] "
            f"candidato do dicionário visual — {top['concept_name']} (score {top['score']:.2f}); "
            "confirme ou corrija olhando a imagem."
        )
    return (
        f"[nota interna sobre recorte {idx}, não mencione essa nota na resposta] "
        "sem candidato no dicionário visual — descreva pelo que você vê."
    )


async def _answer_vision_with_llm(objects: list[dict], user_prompt: str, history: list[dict]) -> str:
    """
    Pede pro LLM multimodal RESPONDER DE FATO a pergunta do usuário usando
    os crops como evidência visual — não apenas rotular/descrever cada
    objeto isoladamente pra depois um segundo LLM (de texto) reinterpretar
    isso do zero. O retorno desta função já é a resposta a ser entregue
    (o modelo de texto, no passo seguinte do loop, só a relaia/formata).

    Reaproveita o histórico real desta conversa (via _CURRENT_HISTORY,
    setado pelo loop principal) em vez de uma chamada isolada — assim o
    modelo multimodal responde com o mesmo contexto que o de texto teria,
    incluindo o que já foi dito antes nesse turno.
    """
    instruction = (
        "Você é a etapa de visão deste assistente. O usuário fez a pergunta abaixo em "
        "relação à imagem que ele anexou, e você recebeu recortes (crops) das regiões "
        "relevantes dessa imagem, numerados na mesma ordem em que aparecem a seguir, cada "
        "um com uma linha de contexto interna acima dele. "
        "RESPONDA DIRETAMENTE à pergunta do usuário, como se você estivesse vendo a cena "
        "— não liste os objetos genericamente nem descreva cada recorte isoladamente. "
        "Use as linhas de contexto (nomes de pessoas já reconhecidas, candidatos do "
        "dicionário) para embasar a resposta, mas corrija se a imagem mostrar algo "
        "diferente disso.\n\n"
        "IMPORTANTE sobre identificação de pessoas: quando uma linha de contexto disser "
        "que um rosto JÁ FOI CONFIRMADO como uma pessoa específica, isso não é você "
        "reconhecendo um estranho numa foto — é um sistema de biometria local, restrito "
        "aos contatos que o próprio usuário cadastrou previamente, já tendo feito essa "
        "verificação. Você deve declarar esse nome normalmente na resposta, sem hedging "
        "do tipo 'não é possível confirmar quem é' — essa confirmação já aconteceu antes "
        "de a imagem chegar até você. Só evite declarar identidade se a linha de contexto "
        "disser explicitamente que a pessoa não foi encontrada/reconhecida.\n\n"
        "IMPORTANTE sobre a forma da resposta: as linhas de contexto e os números de "
        "'recorte' são informação interna sua, não algo que o usuário sabe que existe. "
        "NUNCA mencione 'recorte', 'linha de contexto', 'crop', 'sistema de biometria', "
        "'dicionário visual' ou qualquer coisa do mecanismo por trás disso na resposta — "
        "fale como se você mesmo tivesse simplesmente reconhecido a pessoa/objeto ao "
        "olhar a imagem. Exemplo do que NÃO fazer: 'a linha de contexto confirma que é o "
        "Felipe'. Faça assim: 'Esse é o Felipe' (e siga respondendo à pergunta com essa "
        "informação incorporada naturalmente).\n\n"
        "Leve em conta também o restante da conversa até aqui para responder de forma "
        "coerente com o que já foi dito.\n\n"
        f"Pergunta do usuário: {user_prompt}"
    )
    content: list[dict] = [{"type": "text", "text": instruction}]
    context_lines: list[str] = []
    for obj in objects:
        line = _vision_object_context_line(obj)
        context_lines.append(line)
        content.append({"type": "text", "text": line})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{obj['crop_base64']}"},
        })
    log.info(f"vision_objects — linhas de contexto enviadas ao multimodal: {context_lines}")

    base_history = _sanitize_history_for_submodel(history)
    messages = base_history + [{"role": "user", "content": content}]

    payload = {"messages": messages, "tools": [], "temperature": 0.2}
    # Retry com backoff para 503 (falha transitória do backend de LLM)
    for attempt in range(ORCHESTRATOR_LLM_RETRIES + 1):
        r = await state.llm_client.post("/chat/tools", json=payload)
        if r.status_code == 200:
            data = r.json()
            if data.get("too_large"):
                raise RuntimeError(f"Contexto excede o limite do modelo: {data.get('usage')}")
            return data["message"].get("content", "")
        if r.status_code == 503 and attempt < ORCHESTRATOR_LLM_RETRIES:
            wait = ORCHESTRATOR_LLM_BACKOFF_S * (attempt + 1)
            log.warning(
                f"LLM.py retornou 503 em _answer_vision_with_llm, "
                f"tentativa {attempt + 1}/{ORCHESTRATOR_LLM_RETRIES + 1}, "
                f"aguardando {wait:.0f}s..."
            )
            await asyncio.sleep(wait)
            continue
        r.raise_for_status()
    raise RuntimeError(f"LLM.py /chat/tools falhou com 503 após {ORCHESTRATOR_LLM_RETRIES + 1} tentativas em _answer_vision_with_llm")


async def _tool_vision_objects(args: dict, req: ExecuteRequest):
    """
    Roda o pipeline de visão (vision.py /vision/process) na imagem anexada
    ao request: rostos primeiro (resolvidos direto contra o face-dict),
    depois objetos genéricos via depth estimation + segmentação +
    dicionário visual. Em seguida manda os crops, numa única chamada
    multi-imagem, para o LLM multimodal RESPONDER a pergunta do usuário
    diretamente — ver _answer_vision_with_llm.
    """
    if not req.image_base64:
        return "[vision_objects: no image was provided with this request]"

    # O base64 já chega pronto do cliente (TUI) — não há arquivo em disco
    # pra ler nem path pra resolver. Só repassamos direto pro vision
    # service. Nenhuma validação de conteúdo aqui: se o base64 estiver
    # corrompido/inválido, quem detecta isso é o próprio vision service
    # (ele decodifica antes de processar).
    r = await state.vision_client.post("/vision/process", json={
        "image_base64": req.image_base64, "top_k": DEFAULT_TOP_K, "min_score": DEFAULT_MIN_SCORE,
    })
    r.raise_for_status()
    data = r.json()

    objects = data.get("objects", [])

    # Log da resposta CRUA do vision.py (antes do corte pra lean_objects
    # mais abaixo, que remove is_face/person_name/face_score/etc pra não
    # inchar o histórico do LLM de texto). Sem isso, o docker.log só mostra
    # a versão enxuta e não dá pra saber se o vision.py de fato resolveu
    # (ou não) a identidade de um rosto antes da chamada ao multimodal.
    face_summary = [
        {
            "object_index": o.get("object_index"),
            "is_face": o.get("is_face", False),
            "person_name": o.get("person_name"),
            "face_known": o.get("face_known"),
            "face_score": o.get("face_score"),
            "face_dict_error": o.get("face_dict_error"),
        }
        for o in objects
    ]
    log.info(
        f"vision_objects — resposta crua do vision.py: total_detected={data.get('total_detected')} "
        f"faces={json.dumps(face_summary, ensure_ascii=False)}"
    )
    if not objects:
        return {
            "total_detected": 0,
            "image_width": data.get("image_width"), "image_height": data.get("image_height"),
            "answer": "Não detectei nenhum rosto ou objeto reconhecível nessa imagem.",
        }

    prompt = args.get("prompt", req.input)
    history = _CURRENT_HISTORY.get()
    answer = await _answer_vision_with_llm(objects, prompt, history)

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
        "objects": lean_objects,
        # Resposta final do turno — ver AUTO_FINISH_TOOLS no loop principal.
        # Já foi formulada pelo modelo multimodal olhando as imagens de
        # verdade e a pergunta real do usuário; não há segunda chamada ao
        # modelo de texto para reformular isso.
        "answer": answer,
    }

# _tool_alpha_code foi REMOVIDO — virou servidor MCP (alpha_code em
# MCP_SKILLS).

# name -> (description, [(field, type, required)], async executor)
# Só as tools NATIVAS ficam aqui agora — as que exigem estado/acoplamento
# demais com o próprio loop pra virar um servidor MCP genérico. memory,
# search e alpha_code saíram daqui: são descobertas
# dinamicamente via MCP_SKILLS (ver _run_tool_loop).
TOOLS: dict[str, dict[str, Any]] = {
    "vision_objects": {
        "description": (
            "Looks at the image attached to the current request (face recognition + object "
            "segmentation) and has the multimodal model directly answer the user's question "
            "about it — pass the user's actual question as `prompt`. Calling this tool ENDS "
            "the turn: its result is sent straight back to the user as the final response, you "
            "do not get to add or rephrase anything after it. Only works if an image was sent."
        ),
        "fields": [("prompt", "string", False)],
        "executor": _tool_vision_objects,
    },
    "finish": {
        "description": "Ends the cycle and delivers the final response to the user. Call this tool once you already have everything you need (or if no other tool is required).",
        "fields": [("response", "string", True)],
        "executor": None,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# Tools schema (formato OpenAI function-calling) — enviado ao backend de LLM,
# que embute as definições na chat template do modelo.
# ══════════════════════════════════════════════════════════════════════════════

_JSON_SCHEMA_TYPES = {"string": "string", "number": "number"}

def _build_native_tools_schema() -> list[dict]:
    """Converte o TOOLS registry (só as nativas: vision_objects, finish)
    para a lista `tools` no formato OpenAI."""
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


def _mcp_tool_to_openai_schema(tool) -> dict:
    """Converte uma Tool do MCP (name, description, inputSchema) direto
    pro formato OpenAI function-calling — substitui a conversão manual de
    campo-por-campo que o _build_native_tools_schema fazia, já que aqui o
    schema já vem pronto (JSON Schema) do próprio servidor MCP."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Seleção de skill (etapa 1 da descoberta em duas etapas) — escolhe quais
# servidores MCP são relevantes pro pedido, usando só os resumos leves de
# MCP_SKILLS, ANTES de conectar em qualquer servidor ou carregar tools reais.
# ══════════════════════════════════════════════════════════════════════════════

async def _select_skills(user_input: str) -> list[str]:
    """Pergunta ao LLM local quais skills (servidores MCP) parecem
    relevantes pro pedido do usuário, usando só ~1 linha de resumo por
    skill (barato em tokens). Retorna a lista de nomes escolhidos — pode
    ser vazia (pedido não precisa de nenhuma tool MCP, só vision/finish).
    """
    if not MCP_SKILLS:
        return []
    skills_list = "\n".join(f"- {name}: {s.skill_summary}" for name, s in MCP_SKILLS.items())
    log.debug(f"[_select_skills] skills disponíveis: {list(MCP_SKILLS.keys())}")
    log.debug(f"[_select_skills] input do usuário: {user_input!r}")
    try:
        raise NotImplementedError(
            "Seleção de skills sem backend (llama-server removido; aguardando substituição)."
        )
    except Exception as e:
        log.warning(f"_select_skills falhou, liberando todas as skills como fallback: {e}")
        return list(MCP_SKILLS.keys())


async def _load_mcp_tools(skill_names: list[str]) -> tuple[list[dict], dict[str, tuple[ClientSession, str]]]:
    """Etapa 2: conecta (ou reusa) as sessões MCP das skills escolhidas,
    lista as tools reais de cada uma, e devolve:
      - o schema combinado no formato OpenAI (pra somar com as tools nativas)
      - um dispatch map: nome da tool -> (sessão MCP, nome original na sessão)
        usado no loop pra saber pra qual sessão encaminhar cada tool_call.
    """
    schema: list[dict] = []
    dispatch: dict[str, tuple[ClientSession, str]] = {}
    log.debug(f"[_load_mcp_tools] conectando nas skills: {skill_names}")
    for skill_name in skill_names:
        try:
            t0 = time.perf_counter()
            session = await _get_mcp_session(skill_name)
            result = await session.list_tools()
            log.debug(
                f"[_load_mcp_tools] skill '{skill_name}': {len(result.tools)} tool(s) "
                f"em {(time.perf_counter()-t0)*1000:.0f}ms -> {[t.name for t in result.tools]}"
            )
        except Exception as e:
            log.warning(f"MCP: falha ao carregar tools da skill '{skill_name}': {type(e).__name__}: {e}")
            continue
        for tool in result.tools:
            schema.append(_mcp_tool_to_openai_schema(tool))
            dispatch[tool.name] = (session, tool.name)

    log.debug(f"[_load_mcp_tools] tools MCP disponíveis nesta rodada: {list(dispatch.keys())}")
    return schema, dispatch


# ══════════════════════════════════════════════════════════════════════════════
# LLM call helper (tool-calling nativo)
# ══════════════════════════════════════════════════════════════════════════════

# Número de retentativas do orquestrador ao chamar LLM.py — separado
# do retry interno do próprio LLM.py (que retrya o OpenRouter). Este
# retry cobre falhas transitórias de rede/5xx entre o orchestrator e o
# LLM.py.
ORCHESTRATOR_LLM_RETRIES = 4
ORCHESTRATOR_LLM_BACKOFF_S = 2.0


async def _llm_chat(messages: list[dict], tools: Optional[list[dict]] = None,
                     temperature: float = 0.0, max_tokens: Optional[int] = None) -> dict:
    """
    Chama o LLM.py (porta 4003) via seu endpoint real de tool-calling,
    /chat/tools, e retorna a mensagem completa (content + tool_calls).

    LLM.py não expõe um /v1/chat/completions no formato OpenAI puro.
    /chat/tools recebe {"messages": [...], "tools": [...], ...} (schema
    ToolUseRequest) e devolve {"message": {...}, ...} (schema
    ToolUseResponse), não {"choices": [{"message": {...}}]}.

    Inclui retry com backoff para 503 — cobre falhas transitórias do
    backend de LLM caso o retry interno do LLM.py (3 tentativas) não seja
    suficiente.
    """
    payload: dict = {"messages": messages, "temperature": temperature}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    last_error: Optional[Exception] = None
    for attempt in range(ORCHESTRATOR_LLM_RETRIES + 1):
        try:
            r = await state.llm_client.post("/chat/tools", json=payload)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            last_error = e
            log.warning(
                f"LLM.py (porta 4003): erro de rede ({type(e).__name__}), "
                f"tentativa {attempt + 1}/{ORCHESTRATOR_LLM_RETRIES + 1}."
            )
            if attempt < ORCHESTRATOR_LLM_RETRIES:
                await asyncio.sleep(ORCHESTRATOR_LLM_BACKOFF_S * (attempt + 1))
                continue
            raise

        if r.status_code == 200:
            data = r.json()
            if data.get("too_large"):
                raise RuntimeError(f"Contexto excede o limite do modelo: {data.get('usage')}")
            return data["message"]

        # 503: falha transitória do backend — retry com backoff progressivo.
        if r.status_code == 503:
            last_error = httpx.HTTPStatusError(
                f"503 Service Unavailable", request=r.request, response=r,
            )
            wait = ORCHESTRATOR_LLM_BACKOFF_S * (attempt + 1)
            log.warning(
                f"LLM.py retornou 503, "
                f"tentativa {attempt + 1}/{ORCHESTRATOR_LLM_RETRIES + 1}, "
                f"aguardando {wait:.0f}s antes de retentar..."
            )
            if attempt < ORCHESTRATOR_LLM_RETRIES:
                await asyncio.sleep(wait)
                continue

        # Qualquer outro erro: não retry, propaga diretamente
        r.raise_for_status()

    # Se chegou aqui, todas as tentativas falharam
    raise last_error  # type: ignore


def _tools_system_prompt(think_instruction: Optional[str] = None, lang_hint: Optional[str] = None,
                          mcp_tool_names: Optional[dict[str, str]] = None) -> str:
    """`mcp_tool_names`: nome da tool MCP -> descrição, já carregadas pela
    etapa 2 da descoberta (_load_mcp_tools) para ESTA rodada específica —
    somadas às tools nativas (TOOLS) na listagem final do prompt."""
    lang_names = {
        "pt": "Portuguese (Brazil)", "en": "English", "es": "Spanish",
        "fr": "French", "de": "German", "it": "Italian", "ja": "Japanese",
        "zh": "Chinese",
    }
    reply_lang = lang_names.get((lang_hint or "").lower(), None)
    lines = [
        "You are AVA, a helpful assistant. You have access to a set of tools/functions.",
        "",
        "Always reply in the same language the user wrote their message in"
        + (f" — for this conversation that is {reply_lang} ({lang_hint})." if reply_lang else ".")
        + " Never switch languages mid-conversation unless the user does.",
        "",
        "You have live tool access (web search, memory, local files, etc.) — you are NOT "
        "limited to a static training cutoff and you DO have a way to get current "
        "information when a tool is available for it. Never tell the user you \"don't have "
        "real-time access\", that your \"knowledge stops at [some date]\", or similar generic "
        "disclaimers — that is false when you have tools and is especially false after you "
        "have already called one. If a tool's result was thin, generic, or didn't contain the "
        "specific fact needed, say exactly that (e.g. \"I searched but only found homepage "
        "links, not the actual headlines\") instead of falling back to a canned refusal that "
        "ignores what the tool actually returned.",
        "",
        "Call a tool whenever you need information or need to take an action. You may also "
        "respond with plain text — reasoning, a plan, a clarifying remark — without calling "
        "a tool; in that case you will simply be prompted again on the next turn, so use it "
        "to think out loud if needed.",
        "",
        "BEFORE calling a tool again, read the content of the most recent role=\"tool\" message "
        "in the conversation and check: does it already contain enough information to answer "
        "the user, even partially? If yes, stop searching and use it — synthesize the answer "
        "from what you already have instead of chasing a more perfect result. If a result is "
        "irrelevant, empty, or off-topic, do NOT just reword the same query and try again — "
        "that rarely fixes an irrelevant result. Instead: (a) try a genuinely different angle "
        "(different tool, different entity, different assumption) at most once, or (b) call "
        "`finish` and tell the user honestly that you could not find reliable/current "
        "information on this, rather than guessing or looping. Never call the same tool with "
        "a near-identical query more than twice in a row.",
        "",
        "When a tool returns MULTIPLE results (e.g. `search` returns a list), do NOT pick just the "
        "first one. Synthesize across all of them: identify the 2–3 distinct items that best match "
        "the user's question, and compose a single coherent answer that mentions each one briefly "
        "with its source. Picking the top result and discarding the rest is a failure mode.",
        "",
        "When answering about 'news' or 'today', present a brief digest of the top items found — "
        "do NOT deep-dive into a single article unless the user asked for one specific topic. "
        "Each item in the digest should carry: (a) the headline, (b) 1–2 sentences of what it is "
        "about, (c) the source name. Cite source URLs at the end.",
        "",
        "Never paraphrase a snippet loosely — if you are translating, keep proper nouns (race names, "
        "event names) in the original language to avoid mistranslation. If unsure about a date or "
        "temporal relation in the source, omit it rather than guessing.",

        "To deliver your final answer to the user, call the `finish` tool with the complete "
        "response text as its `response` argument. The task is not done until you call `finish`.",
        "",
        f"Today is {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.",
        "",
        "Available tools:",
    ]
    for name, spec in TOOLS.items():
        lines.append(f"- {name}: {spec['description']}")
    for name, description in (mcp_tool_names or {}).items():
        lines.append(f"- {name}: {description}")
    if think_instruction:
        lines.append("")
        lines.append(think_instruction)
    return "\n".join(lines)


async def _force_finish(history: list[dict], eid: str) -> str:
    """
    Rede de segurança do guarda-corpo anti-loop: chamada quando o modelo
    insiste na mesma tool repetidas vezes sem nunca chamar `finish`. Faz
    UMA última chamada ao LLM sem `tools` (nesse modo ele não tem como
    devolver tool_calls, então é obrigado a responder em texto livre)
    pedindo explicitamente uma resposta final com o que já foi coletado
    no histórico. Se isso falhar por qualquer motivo, cai
    num texto fixo — nunca deixamos o loop rodar indefinidamente.
    """
    nudge = {
        "role": "user",
        "content": (
            "You've been trying the same tool repeatedly without concluding. Stop here: "
            "using only the information already gathered above, give me your best final "
            "answer now in plain text. If it's incomplete, say so plainly instead of "
            "searching more."
        ),
    }
    try:
        message = await _llm_chat(history + [nudge], tools=None, temperature=0.3)
        text = (message.get("content") or "").strip()
        if text:
            return text
    except Exception as e:
        log.warning(f"[{eid[:8]}] _force_finish: chamada de fechamento falhou: {e}")
    return (
        "Não consegui reunir informação suficiente para responder com confiança a essa "
        "pergunta depois de várias tentativas de busca — os resultados retornados não "
        "foram relevantes o bastante. Posso tentar de outro jeito se você me der mais "
        "contexto ou uma fonte específica."
    )


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
      2. Se NÃO houver tool_calls: o texto é tratado como resposta final
         direta (equivalente a um "finish" implícito) e o loop termina ali —
         NÃO re-chamamos o LLM com o histórico inalterado, pois isso geraria
         mensagens role="assistant" consecutivas, violando a alternância
         estrita user/assistant/tool exigida pelo chat template do modelo.
      3. Se houver tool_calls: cada uma é executada e o resultado volta ao
         histórico como mensagem role="tool" (tool_call_id correspondente).
      4. O loop também termina quando a tool "finish" é chamada explicitamente.
         Fora isso, não há limite de rodadas — EXCETO um guarda-corpo anti-loop:
         se a mesma tool for chamada REPEAT_HARD_LIMIT vezes seguidas sem
         `finish` (sinal de que o modelo não está "absorvendo" o resultado
         retornado e só reformulando a mesma chamada), o orquestrador força
         o encerramento via `_force_finish` em vez de deixar o loop rodar
         indefinidamente.
    """
    depth, think_instruction = await _think_instruction(req.input)

    # ── Descoberta de tools em duas etapas ──
    # Etapa 1: escolhe as skills (servidores MCP) relevantes pra este pedido,
    # usando só os resumos leves — sem conectar em nada ainda.
    chosen_skills = await _select_skills(req.input)
    
    # Etapa 2: conecta só nessas skills e carrega as tools reais delas.
    mcp_schema, mcp_dispatch = await _load_mcp_tools(chosen_skills)
    mcp_descriptions = {name: spec["function"]["description"] for spec in mcp_schema
                         for name in [spec["function"]["name"]]}

    system_prompt = _tools_system_prompt(think_instruction, lang_hint=req.lang, mcp_tool_names=mcp_descriptions)
    tools_schema = _build_native_tools_schema() + mcp_schema
    log.info(f"[{eid[:8]}] Tools oferecidas ao modelo nesta rodada: "
             f"{[t['function']['name'] for t in tools_schema]}")

    user_content = req.input
    if req.image_base64:
        # Sem isso, o LLM não tem nenhum sinal no histórico de que uma
        # imagem foi de fato anexada a ESTE request — só a descrição
        # genérica da tool vision_objects ("only works if an image was
        # sent"), que não é uma confirmação. Resultado: o modelo assume
        # que não há imagem e pede pro usuário reenviar, mesmo com
        # req.image_base64 preenchido.
        user_content = (
            f"{req.input}\n\n"
            "[An image was attached to this message. Use the vision_objects "
            "tool to see it before answering — do not ask the user to send "
            "it again.]"
        )

    history: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    step_results: list[StepResult] = []
    final_response = ""
    turn = 0

    # ── Guarda-corpo anti-repetição ──────────────────────────────────────
    # O modelo às vezes não "entende" que um resultado de tool já é
    # suficiente (ou já é irrelevante e reformular a query não vai ajudar)
    # e fica preso chamando a mesma tool em loop (visto em produção: 7+
    # chamadas seguidas de `search` sem nunca chamar `finish`). Em vez de
    # só confiar em instrução de prompt, rastreamos a sequência de tools
    # chamadas e intervimos ativamente:
    #   - a partir de REPEAT_NUDGE_AT chamadas seguidas da MESMA tool,
    #     injetamos um lembrete visível pro modelo (role="tool" sintético)
    #     apontando explicitamente o padrão, para forçar reavaliação;
    #   - em REPEAT_HARD_LIMIT chamadas seguidas da MESMA tool, paramos o
    #     loop nós mesmos e pedimos uma última resposta ao modelo usando
    #     só o que já foi coletado (sem permitir nova tool_call), evitando
    #     o loop indefinido que derrubou a sessão no log.
    REPEAT_NUDGE_AT = 3
    REPEAT_HARD_LIMIT = 5
    last_tool_name: Optional[str] = None
    same_tool_streak = 0

    # Tools cujo resultado já É a resposta final ao usuário — nenhuma
    # segunda chamada ao modelo de texto acontece depois delas. Hoje só
    # vision_objects: o modelo multimodal já respondeu diretamente à
    # pergunta do usuário usando os crops (ver _answer_vision_with_llm) —
    # só pronto pro PRÓXIMO turno, sem gerar resposta pra este.
    AUTO_FINISH_TOOLS = {"vision_objects"}

    while True:
        # ── Passo 1: LLM responde, podendo incluir tool_calls estruturadas ──
        message = await _llm_chat(history, tools=tools_schema, temperature=0.3)
        tool_calls = message.get("tool_calls") or []
        log.debug(
            f"[{eid[:8]}] Turn {turn}: content={message.get('content')!r} "
            f"tool_calls={[c.get('function', {}).get('name') for c in tool_calls]}"
        )

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        history.append(assistant_msg)

        if not tool_calls:
            # Modelo respondeu sem chamar nenhuma tool: tratamos como resposta
            # final direta (equivalente a um "finish" implícito).
            #
            # Importante: NÃO dá para simplesmente re-chamar o LLM de novo com
            # o mesmo histórico (sem nova mensagem de user/tool no meio) —
            # isso produziria duas mensagens role="assistant" consecutivas no
            # histórico, o que viola a alternância estrita user/assistant/tool
            # exigida pelo chat template do modelo e derruba a chamada
            # seguinte com 400 Bad Request no backend de LLM.
            final_response = message.get("content") or ""
            step_results.append(StepResult(
                step=turn, executor="llm", action="(resposta direta, sem tool_calls)",
                success=True, result=final_response,
            ))
            log.debug(f"[{eid[:8]}] Turn {turn}: resposta sem tool_calls, finalizando")
            break

        finished = False
        for call in tool_calls:
            fn = call.get("function", {}) or {}
            tool_name = fn.get("name", "")
            call_id = call.get("id", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}

            is_native = tool_name in TOOLS
            is_mcp = tool_name in mcp_dispatch
            log.debug(
                f"[{eid[:8]}] tool_call recebido: name={tool_name!r} args={args} "
                f"is_native={is_native} is_mcp={is_mcp}"
            )
            if not is_native and not is_mcp:
                log.warning(f"[{eid[:8]}] Tool desconhecida '{tool_name}' — o modelo pediu uma tool "
                            f"que não está no schema desta rodada (tools_schema atual: "
                            f"{[t['function']['name'] for t in tools_schema]})")
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

            # ── Executa a tool: nativa (executor Python direto) ou MCP
            # (despachada pra sessão correspondente via call_tool) ──
            t0 = time.perf_counter()
            _CURRENT_HISTORY.set(history)
            try:
                if is_native:
                    result = await asyncio.wait_for(
                        TOOLS[tool_name]["executor"](args, req),
                        timeout=EXECUTOR_TIMEOUTS.get(tool_name, 60.0),
                    )
                else:
                    session, mcp_tool_name = mcp_dispatch[tool_name]
                    mcp_result = await asyncio.wait_for(
                        session.call_tool(mcp_tool_name, arguments=args),
                        timeout=EXECUTOR_TIMEOUTS.get(tool_name, 60.0),
                    )
                    # Conteúdo MCP vem como lista de blocos (TextContent,
                    # ImageContent, ...) — junta os textuais num único
                    # texto, no formato que _result_to_text já sabe tratar.
                    result = "\n".join(
                        block.text for block in mcp_result.content
                        if hasattr(block, "text")
                    ) or str(mcp_result.content)
                success, err = True, None
            except Exception as e:
                result, success, err = None, False, f"{type(e).__name__}: {e}"
            lat = round((time.perf_counter() - t0) * 1000, 2)

            # ── Log bruto da resposta da tool (debug) ──
            if success:
                log.info(
                    f"[{eid[:8]}] TOOL RESULT '{tool_name}' (call_id={call_id}, "
                    f"{lat}ms) args={json.dumps(args, ensure_ascii=False)} "
                    f"result={json.dumps(result, ensure_ascii=False, default=str)}"
                )
            else:
                log.info(
                    f"[{eid[:8]}] TOOL RESULT '{tool_name}' (call_id={call_id}, "
                    f"{lat}ms) args={json.dumps(args, ensure_ascii=False)} FAILED err={err}"
                )

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

            # ── Auto-finish: tools cuja resposta já é final (vision_objects) ──
            # Evita a segunda chamada ao modelo de texto — o resultado do
            # modelo multimodal (campo "answer") vira a resposta do turno
            # direto, sem gerar nada agora.
            if tool_name in AUTO_FINISH_TOOLS and success and isinstance(result, dict) and result.get("answer"):
                final_response = result["answer"]
                finished = True
                break

            # ── Atualiza a sequência de repetição da mesma tool ──
            if tool_name == last_tool_name:
                same_tool_streak += 1
            else:
                last_tool_name = tool_name
                same_tool_streak = 1

            if same_tool_streak == REPEAT_NUDGE_AT:
                log.warning(
                    f"[{eid[:8]}] '{tool_name}' chamada {same_tool_streak}x seguidas "
                    "sem finish — injetando lembrete anti-loop."
                )
                history.append({
                    "role": "tool", "tool_call_id": call_id,
                    "content": (
                        f"SYSTEM NOTICE: you have called `{tool_name}` {same_tool_streak} times "
                        "in a row without calling `finish`. Stop and re-read the results above "
                        "carefully — either they already answer the question well enough to "
                        "respond now, or repeating this tool is not going to fix it. Do not call "
                        f"`{tool_name}` again with a reworded query; call `finish` with your best "
                        "answer, being explicit about what you could and couldn't confirm."
                    ),
                })
            elif same_tool_streak >= REPEAT_HARD_LIMIT:
                log.warning(
                    f"[{eid[:8]}] '{tool_name}' chamada {same_tool_streak}x seguidas — "
                    "forçando encerramento do loop (guarda-corpo anti-loop)."
                )
                finished = True
                final_response = await _force_finish(history, eid)
                step_results.append(StepResult(
                    step=turn, executor="orchestrator",
                    action="anti-loop hard stop", success=True, result=final_response,
                ))
                break

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
        return await _call_memory_tool("memory_read", {
            "query": req.query, "top_k": req.top_k, "min_score": req.min_score,
            "session_id": req.session_id, "strategy": "auto",
        })
    except Exception as e:
        raise HTTPException(502, f"Memory falhou: {e}")

@app.post("/memory/write")
async def memory_write(req: MemoryWriteRequest):
    try:
        return await _call_memory_tool("memory_write", {
            "text": req.text, "source": req.source, "confidence": req.confidence,
            "forgettable": req.forgettable, "ttl_days": req.ttl_days,
            "action": req.action, "memory_id": req.memory_id,
        })
    except Exception as e:
        raise HTTPException(502, f"Memory falhou: {e}")

@app.post("/memory/write-batch")
async def memory_write_batch(reqs: list[MemoryWriteRequest]):
    try:
        return await _call_memory_tool("memory_write_batch", {
            "items": [
                {
                    "text": r.text, "source": r.source, "confidence": r.confidence,
                    "forgettable": r.forgettable, "ttl_days": r.ttl_days,
                    "action": r.action, "memory_id": r.memory_id,
                }
                for r in reqs
            ],
        })
    except Exception as e:
        raise HTTPException(502, f"Memory falhou: {e}")

@app.post("/search")
async def search(query: str, max_results: int = 5, search_pdfs: bool = False, topic: str = "general"):
    try:
        r = await state.search_client.post("/search", json={
            "query": query, "max_results": max_results, "use_cache": True,
            "search_pdfs": search_pdfs, "topic": topic if topic in ("general", "news") else "general",
        })
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(502, f"Search falhou: {e}")

@app.post("/read-url")
async def read_url(url: str):
    try:
        r = await state.search_client.post("/extract", json={"urls": [url], "extract_depth": "basic"})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(502, f"Extract falhou: {e}")

@app.post("/chat")
async def chat(message: str, voice: str = "M1", lang: str = "pt", tts: bool = True):
    try:
        r = await state.llm_client.post("/chat", json={"message": message, "voice": voice, "lang": lang, "max_turns": 10, "tts": tts})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(502, f"Chat falhou: {e}")


# ── Status / utilitários ─────────────────────────────────────────────────────

@app.get("/status")
async def status():
    checks = {}
    cfg = {
        "search": (state.search_client, HEALTH_PATHS["search"]),
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

    # "memory" não tem endpoint HTTP — sua saúde é checada chamando a
    # própria tool MCP `memory_status` (conecta lazily se ainda não conectado).
    try:
        mem_status = await _call_memory_tool("memory_status", {})
        checks["memory"] = {"healthy": mem_status is not None, "status_code": None, "mcp": True}
    except Exception as e:
        checks["memory"] = {"healthy": False, "status_code": None, "mcp": True, "error": str(e)}

    return {
        "orchestrator": "ok", "architecture": "tool-calling + MCP (two-stage discovery)",
        "services": checks,
        "native_tools": list(TOOLS.keys()),
        "mcp_skills": {name: skill.skill_summary for name, skill in MCP_SKILLS.items()},
        "mcp_connected": list(state.mcp_sessions.keys()),
    }

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    try:
        return await _call_memory_tool("memory_clear_session", {"session_id": session_id})
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