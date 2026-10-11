from __future__ import annotations

"""
mcp_manager.py — Controle de servidores MCP: registro (REST) + executor.
==========================================================================

O usuário cadastra servidores MCP (stdio ou http) via API REST; este módulo
é o ÚNICO dono do ciclo de vida deles:

  1. REGISTRO — tabela `mcp_servers` (SQLite WAL, ava_mcp_servers.db):
     nome, transport, command/args/env (stdio) ou url (http), summary e
     enabled. Idempotente: re-registrar o mesmo nome atualiza a config
     (e derruba a sessão viva, se houver, pra forçar reconexão limpa).

  2. SYNC — POST /servers/{name}/sync: conecta (spawn sob demanda), roda
     list_tools() e grava CADA tool na memória de grafos de tools
     (ToolsDB, via memory_api REST /tools/register/batch). O nome usado
     no grafo é o GRAPH NAME `f"{servidor}__{tool}"` — único entre
     servidores (ToolsDB tem unicidade global de name_key; sem isso dois
     servidores expondo "search" se sobreporiam no grafo).

  3. SNIPPETS — o "skill.md" da tool: o texto que mostra ao modelo o que
     ela realmente faz e como usar. No sync, snippet = description que o
     próprio servidor MCP devolveu; o usuário pode reescrever a qualquer
     momento via PUT /tools/{graph_name}/snippet — a ToolsDB é
     re-registrada (doc_hash muda → re-embed; queries "learned" são
     preservadas pelo memory).

  4. EXECUÇÃO — este módulo é o EXECUTANTE: mantém as sessões MCP vivas
     (AsyncExitStack por servidor, spawn lazy na primeira chamada) e o
     orchestrator só faz proxy REST (POST /servers/{name}/call). Se uma
     chamada falhar, a sessão é considerada podre e descartada (próxima
     chamada respawna o processo stdio do zero).

Descoberta em duas etapas (consumida pelo orchestrator):
  Etapa 1 — POST /skills/select: proxy do /tools/select da memory_api
  (kNN semântico + PPR no grafo de co-uso das tools) e mapeia tools →
  servidores. NADA é conectado aqui.
  Etapa 2 — GET /servers/{name}/tools: schemas reais + snippets lidos do
  SQLite local (sem conexão MCP) — o orchestrator monta o payload OpenAI
  com esses dados; a execução real acontece no /call.

Rotas:
  GET    /health
  POST   /servers                      (registrar/atualizar servidor)
  POST   /servers/install              (instalação assistida de MCP npm: resolve pacote,
                                        registra `npx -y <pkg>` e sincroniza)
  GET    /servers                      (listar)
  GET    /servers/{name}               (detalhe)
  DELETE /servers/{name}               (remover + remover tools do grafo)
  POST   /servers/{name}/sync          (list_tools → grafo + cache local)
  POST   /servers/{name}/test          (conecta e conta tools; não grava nada)
  GET    /servers/{name}/tools         (etapa 2: schemas + snippets)
  POST   /servers/{name}/call          (executa tool; proxy do orchestrator)
  GET    /skills                       (etapa 1a: servidores habilitados)
  POST   /skills/select                (etapa 1b: seleção via grafo de tools)
  GET    /tools                        (todas as tools conhecidas)
  PUT    /tools/{graph_name}/snippet   (reescreve o skill.md de uma tool)

Porta 4008 (env MCP_MANAGER_PORT). Memory em MEMORY_API_URL (default
http://localhost:3000) — compartilhada com orchestrator/LLM.py.

Exemplo de uso:

curl -X POST http://localhost:4008/servers -H 'Content-Type: application/json' -d '{
  "name": "playwright",
  "skill_summary": "Automação de navegador (navegar, clicar, extrair conteúdo)",
  "transport": "stdio",
  "command": "npx",
  "args": ["-y", "@playwright/mcp@latest", "--cdp-endpoint", "http://localhost:9222"],
  "env": {"DISPLAY": ":0"}
}'

curl -X POST http://localhost:4008/servers/playwright/sync 
*SEMPRE que cadastras um novo MCP, ou atualiza o snippet de uma tool, ou muda a config do servidor (re-register), ou remove um servidor (DELETE) — para manter o grafo de tools atualizado.*

Instalação assistida (pacote npm — valida no registry, registra e sincroniza):

curl -X POST http://localhost:4008/servers/install -H 'Content-Type: application/json' -d '{
  "package": "@playwright/mcp@latest",
  "skill_summary": "Automação de navegador",
  "args": ["--cdp-endpoint", "http://localhost:9222"]
}'
"""

import asyncio
import contextlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── MCP (Model Context Protocol) — spawns de servidores stdio/http ──
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

MCP_MANAGER_PORT = int(os.getenv("MCP_MANAGER_PORT", "4008"))
MEMORY_API_URL   = os.getenv("MEMORY_API_URL", "http://localhost:3000")

# Banco local do manager (registry de servidores + cache de tools/schemas).
# Vive no mesmo diretório de DBs da memory (Modules/memory/db/) por
# convenção de empacotamento — mas é propriedade EXCLUSIVA deste processo.
_DB_DIR  = Path(__file__).resolve().parent / "memory" / "db"
MCP_DB_PATH = str(_DB_DIR / "ava_mcp_servers.db")

# Timeouts: sync/test conversam com um subprocesso recém-spawnado (cold start
# pode ser lento — npx baixa pacote etc.); call usa o timeout que o
# orchestrator pedir por request (default generoso abaixo). O HANDSHAKE de
# conexão (spawn + initialize) tem teto próprio (CONNECT_TIMEOUT) — servidor
# que não fala MCP não pode pendurar o handler nem travar a connect_lock.
CONNECT_TIMEOUT = float(os.getenv("MCP_CONNECT_TIMEOUT", "45.0"))
SYNC_TIMEOUT   = 90.0
REGISTER_TIMEOUT = 180.0   # batch register na memory: cada tool é embedada
DEFAULT_CALL_TIMEOUT = 120.0
# Servidor cuja conexão/call falhou fica fora da /skills/select por este
# tempo (oferecer 24 tools de um playwright sem CDP só distrai o modelo).
# Expiração é automática — a próxima seleção volta a tentar. Servidores
# always_on nunca são excluídos por isso.
UNHEALTHY_TTL_S = float(os.getenv("MCP_UNHEALTHY_TTL_S", "300"))

# Nome de servidor: sem "__" (é o separador do graph_name) e sem pontos.
_SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

logging.basicConfig(
    level=logging.DEBUG if os.environ.get("AVA_DEBUG") else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ava.mcp_manager")


def _graph_name(server: str, tool: str) -> str:
    return f"{server}__{tool}"


def _split_graph_name(graph_name: str) -> tuple[str, str]:
    """'playwright__browser_navigate' → ('playwright', 'browser_navigate').
    Levanta ValueError se não houver prefixo de servidor."""
    server, sep, tool = graph_name.partition("__")
    if not sep or not server or not tool:
        raise ValueError(f"graph_name inválido (esperado '<servidor>__<tool>'): {graph_name!r}")
    return server, tool


# ══════════════════════════════════════════════════════════════════════════════
# Data Models
# ══════════════════════════════════════════════════════════════════════════════

class ServerRegisterRequest(BaseModel):
    name: str
    skill_summary: str = ""
    # ── stdio (default): o manager spawna o subprocesso do servidor ──
    transport: Literal["stdio", "http"] = "stdio"
    command: Optional[str] = None
    args: list[str] = []
    env: dict[str, str] = {}
    # ── http: conecta a um servidor MCP JÁ RODANDO (streamable-http) ──
    url: Optional[str] = None
    enabled: bool = True
    # ── capability de base: tools deste servidor entram no grafo como
    # "core" (sempre oferecidas ao modelo, independentemente do kNN) e o
    # servidor nunca é exclído da seleção por marcação de unhealthy ──
    always_on: bool = False
    # Exemplos de gatilho por tool (chave = nome da tool EXPOSTA pelo
    # servidor, ex.: "web_search") — viram vetores próprios no grafo e
    # melhoram o recall do kNN pra queries que não parecem com a descrição.
    examples: dict[str, list[str]] = {}

class ServerCallRequest(BaseModel):
    tool: str
    arguments: dict[str, Any] = {}
    timeout: float = Field(DEFAULT_CALL_TIMEOUT, gt=0, le=3600,
                           description="Teto de execução da tool (segundos)")

class SkillsSelectRequest(BaseModel):
    query: str
    top_k: int = 8
    min_score: float = 0.35

class SnippetUpdateRequest(BaseModel):
    snippet: str

class ServerInstallRequest(BaseModel):
    """Instalação assistida de servidor MCP npm: resolve o pacote no
    registry, deriva o binário, registra como stdio `npx -y` e sincroniza
    — o usuário só passa o nome do pacote."""
    package: str
    name: Optional[str] = None          # default: derivado do nome do pacote
    skill_summary: str = ""
    args: list[str] = []                # argumentos extras ao binário (ex.: --cdp-endpoint ...)
    env: dict[str, str] = {}
    enabled: bool = True
    always_on: bool = False
    examples: dict[str, list[str]] = {}


# ══════════════════════════════════════════════════════════════════════════════
# Global State & Lifespan
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MCPState:
    db: sqlite3.Connection = field(default=None)
    # Escritas no SQLite são raras (registro/sync) mas o check_same_thread=False
    # do connect exige serialização explícita — mesmo padrão do ToolsDB.
    db_lock: threading.Lock = field(default_factory=threading.Lock)
    memory: httpx.AsyncClient = field(default=None)
    # ── executor: sessões MCP vivas por servidor. Spawn LAZY (só conecta na
    # primeira chamada/sync/test) — os subprocessos stdio ficam presos no
    # AsyncExitStack do servidor e morrem com ele. ──
    stacks: dict[str, AsyncExitStack] = field(default_factory=dict)
    sessions: dict[str, ClientSession] = field(default_factory=dict)
    # Uma lock por servidor pra evitar spawn duplicado em chamadas concorrentes.
    connect_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    # Geração por servidor: incrementada a cada _drop_session (delete,
    # re-register, sessão podre). Um connect em voo que terminar DEPOIS de um
    # drop compara sua geração e se descarta, em vez de cachear uma sessão
    # com config antiga (corrida re-register × connect).
    gens: dict[str, int] = field(default_factory=dict)
    # Evento de parada do keeper task de cada servidor (_drop_session → set).
    stop_events: dict[str, asyncio.Event] = field(default_factory=dict)
    # Keeper task vivo por servidor — aguardado no shutdown pra não matar
    # subprocessos stdio em estado indefinido.
    keeper_tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    # Servidores com falha recente de conexão/call: name → até quando
    # (time.time()+TTL) ficar fora da /skills/select.
    unhealthy: dict[str, float] = field(default_factory=dict)

state = MCPState()


def _mark_unhealthy(name: str, cause: str) -> None:
    until = time.time() + UNHEALTHY_TTL_S
    state.unhealthy[name] = until
    log.warning(f"MCP[{name}]: marcado UNHEALTHY por {UNHEALTHY_TTL_S:.0f}s ({cause})")


def _clear_unhealthy(name: str) -> None:
    if state.unhealthy.pop(name, None) is not None:
        log.info(f"MCP[{name}]: recovered — unhealthy removido")


def _is_unhealthy(name: str) -> bool:
    until = state.unhealthy.get(name)
    return until is not None and time.time() < until


def _open_db() -> sqlite3.Connection:
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(MCP_DB_PATH, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS mcp_servers (
            name          TEXT PRIMARY KEY,
            skill_summary TEXT NOT NULL DEFAULT '',
            transport     TEXT NOT NULL DEFAULT 'stdio',
            command       TEXT,
            args_json     TEXT NOT NULL DEFAULT '[]',
            env_json      TEXT NOT NULL DEFAULT '{}',
            url           TEXT,
            enabled       INTEGER NOT NULL DEFAULT 1,
            always_on     INTEGER NOT NULL DEFAULT 0,
            examples_json TEXT NOT NULL DEFAULT '{}',
            created_at    REAL NOT NULL,
            last_sync     REAL
        );
        CREATE TABLE IF NOT EXISTS mcp_tools (
            name              TEXT NOT NULL,
            server            TEXT NOT NULL,
            description_raw   TEXT NOT NULL DEFAULT '',
            snippet           TEXT NOT NULL DEFAULT '',
            input_schema_json TEXT NOT NULL DEFAULT '{}',
            core              INTEGER NOT NULL DEFAULT 0,
            updated_at        REAL NOT NULL,
            PRIMARY KEY (name, server),
            FOREIGN KEY (server) REFERENCES mcp_servers(name) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_mcp_tools_server ON mcp_tools(server);
    """)
    # Migração p/ DBs criados antes de always_on/core/examples (guardado por
    # PRAGMA table_info — ALTER em coluna existente daria erro).
    srv_cols = {r[1] for r in conn.execute("PRAGMA table_info(mcp_servers)").fetchall()}
    if "always_on" not in srv_cols:
        conn.execute("ALTER TABLE mcp_servers ADD COLUMN always_on INTEGER NOT NULL DEFAULT 0")
    if "examples_json" not in srv_cols:
        conn.execute("ALTER TABLE mcp_servers ADD COLUMN examples_json TEXT NOT NULL DEFAULT '{}'")
    tool_cols = {r[1] for r in conn.execute("PRAGMA table_info(mcp_tools)").fetchall()}
    if "core" not in tool_cols:
        conn.execute("ALTER TABLE mcp_tools ADD COLUMN core INTEGER NOT NULL DEFAULT 0")
    return conn


# ── DB helpers (todas sob db_lock) ──────────────────────────────────────────

def _db_upsert_server(req: ServerRegisterRequest) -> sqlite3.Row:
    now = time.time()
    examples = {k.strip(): [e.strip() for e in v if e.strip()]
                for k, v in (req.examples or {}).items() if v}
    with state.db_lock:
        state.db.execute("BEGIN")
        try:
            state.db.execute(
                "INSERT INTO mcp_servers (name, skill_summary, transport, command, "
                "args_json, env_json, url, enabled, always_on, examples_json, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET skill_summary=excluded.skill_summary, "
                "transport=excluded.transport, command=excluded.command, "
                "args_json=excluded.args_json, env_json=excluded.env_json, "
                "url=excluded.url, enabled=excluded.enabled, "
                "always_on=excluded.always_on, examples_json=excluded.examples_json",
                (req.name, req.skill_summary.strip(), req.transport, req.command,
                 json.dumps(req.args), json.dumps({k: str(v) for k, v in req.env.items()}),
                 req.url, int(req.enabled), int(req.always_on),
                 json.dumps(examples, ensure_ascii=False), now),
            )
            state.db.execute("COMMIT")
        except Exception:
            state.db.execute("ROLLBACK")
            raise
        return state.db.execute("SELECT * FROM mcp_servers WHERE name = ?", (req.name,)).fetchone()

def _db_get_server(name: str) -> Optional[sqlite3.Row]:
    with state.db_lock:
        return state.db.execute("SELECT * FROM mcp_servers WHERE name = ?", (name,)).fetchone()

def _db_list_servers(enabled_only: bool = False) -> list[sqlite3.Row]:
    q = "SELECT * FROM mcp_servers"
    if enabled_only:
        q += " WHERE enabled = 1"
    with state.db_lock:
        return state.db.execute(q + " ORDER BY name").fetchall()

def _db_delete_server(name: str):
    with state.db_lock:
        state.db.execute("DELETE FROM mcp_tools WHERE server = ?", (name,))
        state.db.execute("DELETE FROM mcp_servers WHERE name = ?", (name,))

def _db_set_last_sync(name: str):
    with state.db_lock:
        state.db.execute("UPDATE mcp_servers SET last_sync = ? WHERE name = ?", (time.time(), name))

def _db_set_skill_summary(name: str, summary: str):
    with state.db_lock:
        state.db.execute("UPDATE mcp_servers SET skill_summary = ? WHERE name = ?", (summary, name))

def _db_upsert_tool(server: str, name: str, description_raw: str, snippet: str,
                    input_schema: dict, core: bool = False):
    with state.db_lock:
        state.db.execute(
            "INSERT INTO mcp_tools (name, server, description_raw, snippet, "
            "input_schema_json, core, updated_at) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(name, server) DO UPDATE SET description_raw=excluded.description_raw, "
            "snippet=excluded.snippet, input_schema_json=excluded.input_schema_json, "
            "core=excluded.core, updated_at=excluded.updated_at",
            (name, server, description_raw, snippet,
             json.dumps(input_schema or {}, ensure_ascii=False), int(core), time.time()),
        )

def _db_tools_for_server(server: str) -> list[sqlite3.Row]:
    with state.db_lock:
        return state.db.execute(
            "SELECT * FROM mcp_tools WHERE server = ? ORDER BY name", (server,)
        ).fetchall()

def _db_all_tools() -> list[sqlite3.Row]:
    with state.db_lock:
        return state.db.execute(
            "SELECT t.*, s.enabled AS server_enabled, s.skill_summary AS server_summary "
            "FROM mcp_tools t JOIN mcp_servers s ON s.name = t.server ORDER BY t.server, t.name"
        ).fetchall()

def _db_delete_tool(server: str, name: str):
    with state.db_lock:
        state.db.execute("DELETE FROM mcp_tools WHERE name = ? AND server = ?", (name, server))

def _db_get_tool(server: str, name: str) -> Optional[sqlite3.Row]:
    with state.db_lock:
        return state.db.execute(
            "SELECT * FROM mcp_tools WHERE name = ? AND server = ?", (name, server)
        ).fetchone()

def _db_update_snippet(server: str, name: str, snippet: str):
    with state.db_lock:
        state.db.execute(
            "UPDATE mcp_tools SET snippet = ?, updated_at = ? WHERE name = ? AND server = ?",
            (snippet, time.time(), name, server),
        )


# ── Runtime MCP (executor) ──────────────────────────────────────────────────

async def _drop_session(name: str) -> None:
    """Derruba a sessão/subprocesso do servidor (se vivo). Usado quando a
    config muda, o servidor é removido, ou uma chamada falhou (sessão
    provavelmente podre — a próxima respawna do zero). O bump de geração
    invalida connects em voo que começaram ANTES deste drop. O fechamento
    real do subprocesso é responsabilidade do keeper task (ver
    _session_keeper) — o evento `stop` é o sinal."""
    state.gens[name] = state.gens.get(name, 0) + 1
    stack = state.stacks.pop(name, None)
    state.sessions.pop(name, None)
    state.connect_locks.pop(name, None)
    ev = state.stop_events.pop(name, None)
    if ev is not None:
        ev.set()


async def _session_keeper(name: str, row: sqlite3.Row, ready: asyncio.Event,
                          err: list, stop: asyncio.Event) -> None:
    """Task DONA da sessão do servidor: entra nos contextos MCP (spawn
    stdio ou conexão http), inicializa, cachear a sessão e segura o
    subprocesso vivo até `stop` ser setado. TODA a vida do contexto (enter
    E exit) acontece NESTE task: o transporte MCP é anyio e proíbe sair de
    um cancel scope em task diferente da que entrou (RuntimeError
    "cancel scope in a different task") — fechar a stack no request task
    ou no shutdown quebraria. Por isso o padrão keeper."""
    stack = AsyncExitStack()
    try:
        if row["transport"] == "http":
            if not row["url"]:
                raise RuntimeError("transport='http' sem url configurada")
            log.info(f"MCP[{name}]: conectando via streamable-http em {row['url']}")
            # mcp 2.x: o gerador produz a tupla (read, write) direto.
            read, write = await stack.enter_async_context(streamable_http_client(row["url"]))
        else:
            if not row["command"]:
                raise RuntimeError("transport='stdio' sem command configurado")
            args = json.loads(row["args_json"] or "[]")
            env = json.loads(row["env_json"] or "{}")
            params = StdioServerParameters(
                command=row["command"], args=args, env=env or None,
            )
            log.info(f"MCP[{name}]: spawn stdio `{row['command']} {' '.join(args)}`")
            read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        state.stacks[name] = stack
        state.sessions[name] = session
        ready.set()
        await stop.wait()   # subprocesso vivo até drop/shutdown
    except asyncio.CancelledError:
        raise
    except Exception as e:
        err.append(f"{type(e).__name__}: {e}")
        ready.set()   # desbloqueia o _get_session esperando
    finally:
        state.sessions.pop(name, None)
        state.stacks.pop(name, None)
        state.keeper_tasks.pop(name, None)
        state.stop_events.pop(name, None)
        try:
            await stack.aclose()   # MESMO task do enter — seguro
        except Exception:
            log.debug(f"[{name}] falha ao fechar stack do keeper (ignorado)", exc_info=True)

async def _get_session(name: str, allow_disabled: bool = False) -> ClientSession:
    """Retorna a ClientSession do servidor, conectando (spawn) na primeira
    vez — via keeper task próprio. O handshake inteiro tem teto de
    CONNECT_TIMEOUT: servidor que não fala MCP não pode pendurar o handler
    (e com ele a connect_lock, que bloquearia TODAS as futuras /call, /sync
    e /test desse servidor)."""
    gen = state.gens.get(name, 0)
    if name in state.sessions:
        return state.sessions[name]
    row = _db_get_server(name)
    if row is None:
        raise HTTPException(404, f"Servidor MCP '{name}' não registrado")
    if not row["enabled"] and not allow_disabled:
        raise HTTPException(409, f"Servidor MCP '{name}' está desabilitado")

    lock = state.connect_locks.setdefault(name, asyncio.Lock())
    async with lock:
        if name in state.sessions:   # outro request venceu a corrida
            return state.sessions[name]
        t0 = time.perf_counter()
        ready, stop = asyncio.Event(), asyncio.Event()
        err: list[str] = []
        state.stop_events[name] = stop
        task = asyncio.create_task(_session_keeper(name, row, ready, err, stop))
        state.keeper_tasks[name] = task
        try:
            await asyncio.wait_for(ready.wait(), timeout=CONNECT_TIMEOUT)
        except asyncio.TimeoutError:
            # Cancelar (não só sinalizar): o keeper está preso DENTRO do
            # enter_async_context e não vai olhar o `stop` — a cancelamento
            # é entregue NELE, então o unwind/stack.aclose() roda no task
            # certo e mata o subprocesso (sem zumbi de `sleep` eterno).
            stop.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(asyncio.shield(task), timeout=10.0)
            _mark_unhealthy(name, f"timeout de conexão ({CONNECT_TIMEOUT:.0f}s)")
            raise HTTPException(
                502, f"Timeout ao conectar no servidor MCP '{name}' "
                     f"({CONNECT_TIMEOUT:.0f}s) — spawn/handshake não completou"
            ) from None
        if err:
            stop.set()   # keeper falhou antes do ready — garante a limpeza
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            _mark_unhealthy(name, f"falha de conexão: {err[0]}")
            raise HTTPException(502, f"Falha ao conectar no servidor MCP '{name}': {err[0]}") from None
        # Corrida re-register/delete × connect: a config mudou enquanto
        # conectávamos — a sessão recém-nascida já está obsoleta; descarta.
        if state.gens.get(name, 0) != gen:
            await _drop_session(name)
            raise HTTPException(409, f"Config do servidor '{name}' mudou durante a conexão — repita a chamada")
        _clear_unhealthy(name)
        log.info(f"MCP[{name}]: conectado em {(time.perf_counter()-t0)*1000:.0f}ms")
        return state.sessions[name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Iniciando MCP Manager (registro + executor de servidores MCP)...")
    state.db = _open_db()
    state.memory = httpx.AsyncClient(
        base_url=MEMORY_API_URL, timeout=httpx.Timeout(REGISTER_TIMEOUT, connect=5.0),
        limits=httpx.Limits(max_keepalive_connections=2, max_connections=4),
    )
    try:
        r = await state.memory.get("/status", timeout=5.0)
        log.info(f"  ✓ memory_api OK ({r.status_code})" if r.status_code == 200 else f"  ⚠ memory_api {r.status_code} — syncs vão falhar até voltar")
    except Exception:
        log.warning("  ✗ memory_api OFFLINE — /sync não conseguirá gravar no grafo de tools")
    n = state.db.execute("SELECT COUNT(*) FROM mcp_servers").fetchone()[0]
    log.info(f"MCP Manager pronto — {n} servidor(es) registrado(s): "
             f"{[r['name'] for r in state.db.execute('SELECT name FROM mcp_servers ORDER BY name').fetchall()]}")
    yield
    # Derruba sessões vivas E connects em voo (keepers ainda não "ready") —
    # depois espera cada keeper fechar o próprio subprocesso (com teto).
    for name in list(set(state.sessions) | set(state.keeper_tasks)):
        await _drop_session(name)
    if state.keeper_tasks:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.gather(*state.keeper_tasks.values(), return_exceptions=True),
                timeout=10.0,
            )
    if state.memory is not None:
        await state.memory.aclose()
    if state.db is not None:
        state.db.close()
    log.info("MCP Manager encerrado")


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI Application
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="AVA MCP Manager", version="1.0.0",
    description="Registro REST + executor de servidores MCP (tools com snippets no grafo de tools)",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    n = state.db.execute("SELECT COUNT(*) FROM mcp_servers").fetchone()[0] if state.db is not None else 0
    return {"status": "ok", "servers": n, "connected": list(state.sessions.keys())}


@app.get("/status")
async def status():
    return await health()


# ── Registro de servidores ──────────────────────────────────────────────────

@app.post("/servers")
async def register_server(req: ServerRegisterRequest):
    """Registra (ou atualiza) um servidor MCP. stdio exige `command`; http
    exige `url`. Re-registrar o mesmo nome atualiza a config e derruba a
    sessão viva — a próxima chamada/sync reconecta com a config nova."""
    name = req.name.strip()
    if not _SERVER_NAME_RE.match(name) or "__" in name:
        raise HTTPException(400, "Nome de servidor inválido: use [A-Za-z0-9_-], sem '__' (é o separador do graph_name das tools)")
    if req.transport == "stdio" and not (req.command or "").strip():
        raise HTTPException(400, "transport='stdio' exige 'command'")
    if req.transport == "http" and not (req.url or "").strip().startswith(("http://", "https://")):
        raise HTTPException(400, "transport='http' exige 'url' (http:// ou https://)")
    existed = _db_get_server(name) is not None
    row = _db_upsert_server(req)
    if existed:
        await _drop_session(name)
        log.info(f"MCP: servidor '{name}' atualizado ({req.transport})")
    else:
        log.info(f"MCP: servidor '{name}' registrado ({req.transport}, enabled={req.enabled})")
    return {"registered": True, "updated": existed, "server": _server_dict(row)}


def _server_dict(row: sqlite3.Row) -> dict:
    return {
        "name": row["name"], "skill_summary": row["skill_summary"],
        "transport": row["transport"], "command": row["command"],
        "args": json.loads(row["args_json"] or "[]"),
        "env": json.loads(row["env_json"] or "{}"),
        "url": row["url"], "enabled": bool(row["enabled"]),
        "always_on": bool(row["always_on"]),
        "examples": json.loads(row["examples_json"] or "{}"),
        "unhealthy": _is_unhealthy(row["name"]),
        "created_at": row["created_at"], "last_sync": row["last_sync"],
        "tools": state.db.execute(
            "SELECT COUNT(*) FROM mcp_tools WHERE server = ?", (row["name"],)
        ).fetchone()[0],
    }


@app.get("/servers")
async def list_servers():
    return {"servers": [_server_dict(r) for r in _db_list_servers()]}


@app.get("/servers/{name}")
async def get_server(name: str):
    row = _db_get_server(name)
    if row is None:
        raise HTTPException(404, f"Servidor MCP '{name}' não registrado")
    return {"server": _server_dict(row)}


@app.delete("/servers/{name}")
async def delete_server(name: str):
    """Remove o servidor, suas tools do cache local E do grafo de tools
    (ToolsDB, via memory DELETE /tools/{graph_name}). Sessão viva é derrubada
    (subprocesso stdio morto junto). Tools cujo DELETE no grafo falhou
    (memory offline) ficam ÓRFÃS no ToolsDB — reportadas em
    `orphaned_in_graph`; o grafo não tem como removê-las depois (re-register
    não as conhece). Com a memory de volta, re-registre o servidor e faça
    /sync + DELETE de novo, ou apague as tools à mão na memory_api."""
    row = _db_get_server(name)
    if row is None:
        raise HTTPException(404, f"Servidor MCP '{name}' não registrado")
    tools = _db_tools_for_server(name)
    removed = 0
    orphaned: list[str] = []
    for t in tools:
        graph = _graph_name(name, t["name"])
        try:
            r = await state.memory.delete(f"/tools/{graph}", timeout=30.0)
            r.raise_for_status()
            removed += 1
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                removed += 1   # já não existia no grafo — ok
            else:
                orphaned.append(graph)
                log.warning(f"MCP: falha ao remover tool '{graph}' do grafo: {e.response.status_code} {e.response.text[:200]}")
        except Exception as e:
            orphaned.append(graph)
            log.warning(f"MCP: falha ao remover tool '{graph}' do grafo: {type(e).__name__}: {e}")
        _db_delete_tool(name, t["name"])
    _db_delete_server(name)
    await _drop_session(name)
    if orphaned:
        log.warning(f"MCP: servidor '{name}' removido, mas {len(orphaned)} tool(s) ficaram ÓRFÃS no ToolsDB: {orphaned}")
    log.info(f"MCP: servidor '{name}' removido ({len(tools)} tool(s), {removed} apagadas do grafo)")
    return {"deleted": True, "name": name,
            "tools_removed_from_graph": removed, "orphaned_in_graph": orphaned}


# ── Sync (list_tools → grafo de tools + cache local) ────────────────────────

@app.post("/servers/{name}/sync")
async def sync_server(name: str):
    """Conecta no servidor, roda list_tools() e grava cada tool na memória
    de grafos de tools (ToolsDB, via memory /tools/register/batch) com o
    snippet como descrição — snippet inicial = description do próprio MCP.
    Tools que sumiram do servidor são removidas do grafo e do cache.
    Idempotente: tool inalterada não é re-embedada (doc_hash na ToolsDB)."""
    row = _db_get_server(name)
    if row is None:
        raise HTTPException(404, f"Servidor MCP '{name}' não registrado")
    session = await _get_session(name, allow_disabled=True)
    try:
        result = await asyncio.wait_for(session.list_tools(), timeout=SYNC_TIMEOUT)
    except Exception as e:
        await _drop_session(name)
        _mark_unhealthy(name, f"list_tools falhou: {type(e).__name__}")
        raise HTTPException(502, f"list_tools falhou no servidor '{name}': {type(e).__name__}: {e}") from e

    live = {t.name: t for t in result.tools}
    prev_rows = {t["name"]: t for t in _db_tools_for_server(name)}
    always_on = bool(row["always_on"])
    try:
        examples_by_tool: dict[str, list[str]] = json.loads(row["examples_json"] or "{}")
    except Exception:
        examples_by_tool = {}
    tools_out: list[dict] = []
    to_register: list[dict] = []
    for tool in live.values():
        raw = (tool.description or "").strip()
        snippet = raw or f"Tool '{tool.name}' do servidor MCP '{name}'."
        prev = prev_rows.get(tool.name)
        if prev is not None:
            # Snippet customizado (PUT /snippet): usuário escreveu um
            # skill.md diferente da description crua do servidor — PRESERVA
            # no re-sync (só a description_raw acompanha o servidor).
            custom = (prev["snippet"] or "").strip() != (prev["description_raw"] or "").strip()
            if custom and (prev["snippet"] or "").strip():
                snippet = prev["snippet"]
        # mcp 2.x: Tool.input_schema (alias camelCase "inputSchema" só na serialização)
        try:
            _db_upsert_tool(name, tool.name, raw, snippet, tool.input_schema or {}, core=always_on)
        except sqlite3.IntegrityError:
            # Corrida rara: o servidor foi DELETADO durante o await do
            # list_tools — as rows filhas já foram removidas (FK cascade).
            log.warning(f"MCP: sync de '{name}' interrompido — servidor removido durante o sync")
            raise HTTPException(409, f"Servidor '{name}' foi removido durante o sync")
        tools_out.append({"name": tool.name, "graph_name": _graph_name(name, tool.name), "snippet": snippet})
        # TODAS as tools vão no batch (não só as alteradas): o doc_hash na
        # ToolsDB já evita re-embed de tool inalterada, e o caminho "unchanged"
        # de lá aplica set_core — necessário quando o usuário flipa always_on
        # e re-sincroniza. Enviar tudo é ~grátis e mantém core/exemplos vivos.
        to_register.append({
            "name": _graph_name(name, tool.name), "description": snippet,
            "core": always_on,
            "examples": examples_by_tool.get(tool.name, []),
        })

    # Stale: tools que estavam no cache mas sumiram do servidor — saem do
    # grafo (DELETE) e do cache. 404 do memory = já não existia, ok.
    stale = [t["name"] for t in _db_tools_for_server(name) if t["name"] not in live]
    for tname in stale:
        graph = _graph_name(name, tname)
        try:
            r = await state.memory.delete(f"/tools/{graph}", timeout=30.0)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                log.warning(f"MCP: falha ao remover tool obsoleta '{graph}' do grafo: {e.response.status_code}")
        except Exception as e:
            log.warning(f"MCP: falha ao remover tool obsoleta '{graph}' do grafo: {type(e).__name__}: {e}")
        _db_delete_tool(name, tname)

    # Registro no grafo (uma chamada batch; embeddings podem demorar —
    # REGISTER_TIMEOUT cobre). Só o que mudou (novo/alterado) é enviado —
    # re-sync idempotente é zero trabalho pra ToolsDB. Falha de memory NÃO
    # desfaz o cache local: re-sync depois re-registra.
    if to_register:
        try:
            r = await state.memory.post("/tools/register/batch", json={
                "tools": to_register,
                "prune_missing": False,   # NUNCA prune global: há tools de outros servidores
            }, timeout=REGISTER_TIMEOUT)
            r.raise_for_status()
        except Exception as e:
            raise HTTPException(
                502, f"Sync de '{name}' leu {len(tools_out)} tool(s), mas o registro no grafo de tools "
                     f"falhou (memory_api offline?): {type(e).__name__}: {e}"
            ) from e

    summary = (row["skill_summary"] or "").strip()
    if not summary and live:
        names = ", ".join(sorted(live)[:8])
        summary = f"Servidor MCP '{name}' — tools: {names}" + ("…" if len(live) > 8 else "")
        _db_set_skill_summary(name, summary)
    _db_set_last_sync(name)
    log.info(f"MCP: sync de '{name}': {len(tools_out)} tool(s), {len(to_register)} nova(s)/alterada(s) no grafo, {len(stale)} removida(s)")
    return {
        "synced": len(tools_out), "registered": len(to_register), "removed": len(stale),
        "skill_summary": summary, "tools": tools_out,
    }


@app.post("/servers/{name}/test")
async def test_server(name: str):
    """Health check: conecta (spawn real) + list_tools, SEM gravar nada.
    Erro volta como 502 com a causa — é o debug de config do usuário."""
    row = _db_get_server(name)
    if row is None:
        raise HTTPException(404, f"Servidor MCP '{name}' não registrado")
    session = await _get_session(name, allow_disabled=True)
    try:
        result = await asyncio.wait_for(session.list_tools(), timeout=SYNC_TIMEOUT)
    except Exception as e:
        await _drop_session(name)
        raise HTTPException(502, f"Servidor '{name}' conectou mas list_tools falhou: {type(e).__name__}: {e}") from e
    return {"ok": True, "server": name, "tools": len(result.tools), "names": [t.name for t in result.tools]}


# ── Instalação assistida (pacotes npm) ──────────────────────────────────────

# Uma instalação global por vez — `npm install -g`/`npm view` concorrentes
# podem corromper o cache do npm.
_install_lock = asyncio.Lock()

async def _run_cmd(cmd: list[str], timeout: float) -> tuple[int, str]:
    """Roda um subprocesso com teto de tempo; stdout+stderr joined."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        with contextlib.suppress(Exception):
            proc.kill()
        raise RuntimeError(f"`{' '.join(cmd)}` excedeu {timeout:.0f}s") from None
    return proc.returncode or 0, out.decode(errors="replace").strip()

def _npm_package_from_spec(spec: str) -> str:
    """'@playwright/mcp@0.0.80' → '@playwright/mcp' (spec com versão é aceita
    e repassada intacta ao npx; aqui extraímos só o NOME pra validar no
    registry e derivar o `name` do servidor)."""
    # Escopados (@org/pkg@ver): só o ÚLTIMO '@' separa versão.
    if spec.startswith("@"):
        base, sep, _ver = spec.rpartition("@")
        return base if sep and base != "@" else spec
    return spec.split("@", 1)[0]

async def _resolve_npm_package(package: str) -> dict:
    """Consulta o registry (`npm view`) e devolve {name, version, bin}.
    Levanta RuntimeError com mensagem amigável se o pacote não existir
    ou não expor binário (um MCP precisa de executável pro stdio)."""
    rc, out = await _run_cmd(
        ["npm", "view", package, "--json"], timeout=60.0,
    )
    if rc != 0:
        raise RuntimeError(f"Pacote npm '{package}' não encontrado no registry: {out[:400]}")
    try:
        meta = json.loads(out)
    except json.JSONDecodeError:
        raise RuntimeError(f"Resposta inválida do npm para '{package}': {out[:400]}") from None
    name = meta.get("name", _npm_package_from_spec(package))
    bin_field = meta.get("bin")
    if isinstance(bin_field, dict):
        bin_name = next(iter(bin_field), None)
    elif isinstance(bin_field, str):
        bin_name = name.rsplit("/", 1)[-1]
    else:
        bin_name = None
    if not bin_name:
        raise RuntimeError(
            f"Pacote '{name}' (v{meta.get('version', '?')}) não expõe 'bin' — "
            f"não é executável via stdio. Registre manualmente com o command correto."
        )
    return {"name": name, "version": meta.get("version", ""), "bin": bin_name}


@app.post("/servers/install")
async def install_server(req: ServerInstallRequest):
    """Instalação assistida de servidor MCP npm: valida o pacote no registry,
    deriva o binário e registra como stdio `npx -y <pkg>` (o spawn lazy do
    manager resolve o download no primeiro uso — sem estado extra de
    instalação) e roda /sync pra tools entrarem no grafo. Idempotente:
    pacote já registrado com o MESMO spec só re-sincroniza. O spec pode
    vir com versão ('@playwright/mcp@latest')."""
    spec = (req.package or "").strip()
    if not spec or " " in spec or spec.startswith("-"):
        raise HTTPException(400, "package inválido (ex.: '@playwright/mcp' ou '@playwright/mcp@0.0.80')")
    npm_name = _npm_package_from_spec(spec)
    pkg_name = npm_name.rsplit("/", 1)[-1]
    server_name = (req.name or "").strip() or pkg_name.replace("@", "").replace("/", "-")
    if not _SERVER_NAME_RE.match(server_name) or "__" in server_name:
        raise HTTPException(400, f"Nome de servidor derivado inválido: '{server_name}' — passe 'name' explícito")

    async with _install_lock:
        existing = _db_get_server(server_name)
        npx_args = ["-y", spec]
        if existing is not None:
            if existing["command"] == "npx" and json.loads(existing["args_json"] or "[]") == npx_args:
                log.info(f"MCP install: '{server_name}' já registrado com {spec} — re-sync apenas")
            else:
                raise HTTPException(409, f"Servidor '{server_name}' já existe com config diferente — DELETE antes ou registre manualmente")
        else:
            meta = await _resolve_npm_package(npm_name)
            log.info(f"MCP install: {npm_name} v{meta['version']} → bin '{meta['bin']}' → servidor '{server_name}' (npx {spec})")
            row = _db_upsert_server(ServerRegisterRequest(
                name=server_name, skill_summary=req.skill_summary,
                transport="stdio", command="npx", args=npx_args,
                env=req.env, enabled=req.enabled, always_on=req.always_on,
                examples=req.examples,
            ))
            log.info(f"MCP: servidor '{server_name}' registrado via install ({spec})")

        sync = await sync_server(server_name)
        return {
            "installed": True, "name": server_name, "package": spec,
            "transport": "stdio", "command": "npx", "args": npx_args,
            "synced": sync["synced"],
            "skill_summary": sync["skill_summary"],
            "tools": sync["tools"],
        }


# ── Etapa 2: schemas + snippets (do cache, sem conexão MCP) ─────────────────

@app.get("/servers/{name}/tools")
async def server_tools(name: str):
    """Tools do servidor com schema real e snippet (o 'skill.md' que vira a
    descrição pro LLM). Lê do cache local — NÃO conecta MCP; se a config
    mudou no servidor, rode /sync. Servidor nunca sincado → lista vazia."""
    if _db_get_server(name) is None:
        raise HTTPException(404, f"Servidor MCP '{name}' não registrado")
    return {
        "server": name,
        "tools": [
            {
                "name": t["name"],
                "graph_name": _graph_name(name, t["name"]),
                "description": t["description_raw"],
                "snippet": t["snippet"],
                "input_schema": json.loads(t["input_schema_json"] or "{}"),
            }
            for t in _db_tools_for_server(name)
        ],
    }


@app.get("/tools")
async def all_tools():
    rows = _db_all_tools()
    return {
        "total": len(rows),
        "tools": [
            {
                "graph_name": _graph_name(r["server"], r["name"]),
                "server": r["server"], "name": r["name"],
                "snippet": r["snippet"], "description": r["description_raw"],
                "server_enabled": bool(r["server_enabled"]),
            }
            for r in rows
        ],
    }


@app.put("/tools/{graph_name}/snippet")
async def update_snippet(graph_name: str, req: SnippetUpdateRequest):
    """Reescreve o skill.md de uma tool. Grava no cache local E re-registra
    no grafo (ToolsDB): a nova descrição muda o doc_hash → re-embed; as
    queries 'learned' (uso real) são preservadas pelo memory."""
    try:
        server, tool = _split_graph_name(graph_name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    row = _db_get_tool(server, tool)
    if row is None:
        raise HTTPException(404, f"Tool '{graph_name}' não encontrada (rode /servers/{server}/sync)")
    snippet = req.snippet.strip()
    if not snippet:
        raise HTTPException(400, "snippet vazio")
    _db_update_snippet(server, tool, snippet)
    # Preserva o core atual da tool (bug anterior: re-register forçava
    # core=False e zerava o flag de capability de base).
    srv = _db_get_server(server)
    core = bool(row["core"]) if srv is None else bool(row["core"]) or bool(srv["always_on"])
    try:
        r = await state.memory.post("/tools/register", json={
            "name": graph_name, "description": snippet, "core": core,
        }, timeout=REGISTER_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        # Cache local já está atualizado; o grafo ficou pra trás — erro
        # explícito pro caller retryar (o re-register é idempotente).
        raise HTTPException(502, f"Snippet salvo localmente, mas falhou re-registrar no grafo: {type(e).__name__}: {e}") from e
    log.info(f"MCP: snippet de '{graph_name}' atualizado ({len(snippet)} chars)")
    return {"updated": True, "graph_name": graph_name, "snippet": snippet}


# ── Etapa 1: seleção de servidores via grafo de tools ───────────────────────

@app.get("/skills")
async def skills():
    """Servidores habilitados com seus summaries — insumo barato da etapa 1
    (também usado como fallback quando o grafo está indisponível)."""
    return {
        "servers": [
            {"name": r["name"], "skill_summary": r["skill_summary"],
             "transport": r["transport"], "enabled": bool(r["enabled"])}
            for r in _db_list_servers(enabled_only=True)
        ]
    }


@app.post("/skills/select")
async def skills_select(req: SkillsSelectRequest):
    """ETAPA 1 da descoberta: pergunta ao grafo de tools (memory
    /tools/select — kNN semântico sobre doc + exemplos + queries aprendidas,
    com expansão PPR no grafo de co-uso) quais tools atendem a query, e
    mapeia as matches para os SERVIDORES donos delas. Zero conexão MCP.
    Sem matches → skills vazio (pedido não precisa de nenhuma tool MCP)."""
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "query vazia")
    try:
        r = await state.memory.post("/tools/select", json={
            "query": query, "top_k": req.top_k, "min_score": req.min_score,
            "include_related": True,
        }, timeout=45.0)
        r.raise_for_status()
        sel = r.json()
    except Exception as e:
        raise HTTPException(502, f"Seleção no grafo de tools falhou (memory_api offline?): {type(e).__name__}: {e}") from e

    matches = sel.get("tools", [])
    servers = _db_list_servers(enabled_only=True)
    # Unhealthy (falha de conexão recente) sai da seleção — oferecer 24 tools
    # de um servidor morto distrai o modelo. EXCETO always_on: capacidade de
    # base (busca web) melhor disponível e falhando do que sumida do schema.
    enabled = {row["name"] for row in servers
               if bool(row["always_on"]) or not _is_unhealthy(row["name"])}
    summaries = {row["name"]: row["skill_summary"] for row in servers}
    by_server: dict[str, list[str]] = {}
    for m in matches:
        try:
            server, _tool = _split_graph_name(m["name"])
        except ValueError:
            continue   # tool sem prefixo de servidor (não é MCP) — ignora
        if server not in enabled:
            continue
        by_server.setdefault(server, []).append(m["name"])

    return {
        "skills": [
            {"name": s, "skill_summary": summaries.get(s, ""), "tools": tools}
            for s, tools in by_server.items()
        ],
        "matched_tools": [m["name"] for m in matches],
        "top1_score": sel.get("top1_score"),
        "low_confidence": sel.get("low_confidence"),
    }


# ── Execução (proxy do orchestrator) ────────────────────────────────────────

@app.post("/servers/{name}/call")
async def server_call(name: str, req: ServerCallRequest):
    """Executa uma tool no servidor (spawn lazy da sessão na primeira
    chamada). Sessão que falhar é descartada — próxima chamada respawna o
    subprocesso do zero. Blocos textuais do resultado são joined; tool que
    sinaliza erro volta com is_error=true (o texto continua indo ao modelo,
    que precisa vê-lo pra reagir). AVISO: o registro de servidores stdio
    permite executar comandos arbitrários — o serviço segue a postura de
    rede aberta do resto do stack (0.0.0.0, sem auth); exponha a porta
    apenas no ambiente confiável."""
    session = await _get_session(name)
    try:
        result = await asyncio.wait_for(
            session.call_tool(req.tool, arguments=req.arguments),
            timeout=req.timeout,
        )
    except Exception as e:
        # Sem _mark_unhealthy aqui: timeout de tool longa não significa
        # servidor morto — unhealthy é só pra falha de CONEXÃO (_get_session).
        await _drop_session(name)
        raise HTTPException(502, f"call_tool '{req.tool}' falhou no servidor '{name}': {type(e).__name__}: {e}") from e
    text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
    if not text:
        # Só blocos não-textuais (imagem etc.) ou vazio — repr Python não
        # diz nada pro modelo; descreve os tipos dos blocos.
        kinds = sorted({getattr(b, "type", "desconhecido") for b in result.content})
        text = (f"(a tool devolveu {len(result.content)} bloco(s) não-textual(is): "
                f"{', '.join(kinds)})") if result.content else ""
    # mcp 2.x: o campo é is_error (alias camelCase "isError" só na serialização).
    return {"result": text, "is_error": bool(result.is_error)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcp_manager:app", host="0.0.0.0", port=MCP_MANAGER_PORT, log_level="info")
