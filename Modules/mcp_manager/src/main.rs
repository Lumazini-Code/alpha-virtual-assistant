//! mcp_manager — Controle de servidores MCP: registro (REST) + executor.
//! =====================================================================
//!
//! Port 1:1 do `mcp_manager.py`. A lógica, as rotas, as mensagens, o schema
//! SQLite (ava_mcp_servers.db) e os formatos JSON são os mesmos do original.
//!
//! ÚNICA adição em relação ao Python: a inicialização dos servidores MCP
//! stdio (spawn) é compatível com Linux E Windows — ver `build_stdio_command`
//! (resolução automática de `npx`/`node`/etc. pelo PATH E pelas instalações
//! bem-conhecidas do usuário — nvm/volta/fnm/npm-global — em ambas as
//! plataformas, ambiente herdado por plataforma, sem janela de console no
//! Windows, grupo de processos no Unix).
//!
//! Rotas:
//!   GET    /health
//!   POST   /servers                      (registrar/atualizar servidor)
//!   POST   /servers/install              (instalação assistida de MCP npm)
//!   GET    /servers                      (listar)
//!   GET    /servers/{name}               (detalhe)
//!   DELETE /servers/{name}               (remover + remover tools do grafo)
//!   POST   /servers/{name}/sync          (list_tools → grafo + cache local)
//!   POST   /servers/{name}/test          (conecta e conta tools; não grava nada)
//!   GET    /servers/{name}/tools         (etapa 2: schemas + snippets)
//!   POST   /servers/{name}/call          (executa tool; proxy do orchestrator)
//!   GET    /skills                       (etapa 1a: servidores habilitados)
//!   POST   /skills/select                (etapa 1b: seleção via grafo de tools)
//!   GET    /tools                        (todas as tools conhecidas)
//!   PUT    /tools/{graph_name}/snippet   (reescreve o skill.md de uma tool)
//!
//! Porta 4008 (env MCP_MANAGER_PORT). Memory em MEMORY_API_URL (default
//! http://localhost:3000).

use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::{FromRequest, Path, Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use indexmap::IndexMap;
use regex::Regex;
use rmcp::model::{CallToolRequestParam, RawContent};
use rmcp::service::{Peer, RunningService};
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{RoleClient, ServiceExt};
use rusqlite::{params, Connection, ErrorCode, OptionalExtension};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokio::sync::{watch, Mutex as AsyncMutex};
use tokio::task::{AbortHandle, JoinHandle};
use tracing::{debug, info, warn};

// ══════════════════════════════════════════════════════════════════════════════
// Configuration
// ══════════════════════════════════════════════════════════════════════════════

// Timeouts: sync/test conversam com um subprocesso recém-spawnado (cold start
// pode ser lento — npx baixa pacote etc.); call usa o timeout que o
// orchestrator pedir por request. O HANDSHAKE de conexão (spawn + initialize)
// tem teto próprio (CONNECT_TIMEOUT) — servidor que não fala MCP não pode
// pendurar o handler nem travar a connect_lock.
const SYNC_TIMEOUT: f64 = 90.0;
const REGISTER_TIMEOUT: f64 = 180.0; // batch register na memory: cada tool é embedada
const DEFAULT_CALL_TIMEOUT: f64 = 120.0;

// Nome de servidor: sem "__" (é o separador do graph_name) e sem pontos.
static SERVER_NAME_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Za-z0-9][A-Za-z0-9_-]*$").unwrap());

struct Config {
    port: u16,
    memory_api_url: String,
    db_dir: PathBuf,
    db_path: PathBuf,
    connect_timeout: f64,
    unhealthy_ttl_s: f64,
}

fn env_parse<T: FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<T>().ok())
        .unwrap_or(default)
}

impl Config {
    fn from_env() -> Self {
        // Banco local do manager (registry de servidores + cache de
        // tools/schemas). Equivalente ao `Path(__file__).parent / "db"`
        // do Python: relativo ao diretório do executável.
        let base = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."));
        let db_dir = base.join("Modules").join("mcp_manager").join("db");
        let db_path = db_dir.join("ava_mcp_servers.db");
        Config {
            port: env_parse("MCP_MANAGER_PORT", 4008u16),
            memory_api_url: std::env::var("MEMORY_API_URL")
                .unwrap_or_else(|_| "http://localhost:3000".to_string()),
            db_dir,
            db_path,
            connect_timeout: env_parse("MCP_CONNECT_TIMEOUT", 45.0f64),
            // Servidor cuja conexão/call falhou fica fora da /skills/select por
            // este tempo. Servidores always_on nunca são excluídos por isso.
            unhealthy_ttl_s: env_parse("MCP_UNHEALTHY_TTL_S", 300.0f64),
        }
    }
}

fn graph_name(server: &str, tool: &str) -> String {
    format!("{server}__{tool}")
}

/// 'playwright__browser_navigate' → ('playwright', 'browser_navigate').
/// Erro se não houver prefixo de servidor.
fn split_graph_name(graph_name: &str) -> Result<(String, String), String> {
    match graph_name.split_once("__") {
        Some((server, tool)) if !server.is_empty() && !tool.is_empty() => {
            Ok((server.to_string(), tool.to_string()))
        }
        _ => Err(format!(
            "graph_name inválido (esperado '<servidor>__<tool>'): {}",
            py_repr(graph_name)
        )),
    }
}

fn py_repr(s: &str) -> String {
    if s.contains('\'') && !s.contains('"') {
        format!("\"{}\"", s.replace('\\', "\\\\"))
    } else {
        format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
    }
}

fn py_list(items: &[String]) -> String {
    format!(
        "[{}]",
        items.iter().map(|n| py_repr(n)).collect::<Vec<_>>().join(", ")
    )
}

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn truncate_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

// ══════════════════════════════════════════════════════════════════════════════
// Erros HTTP (FastAPI HTTPException → {"detail": ...})
// ══════════════════════════════════════════════════════════════════════════════

enum ApiError {
    Http(StatusCode, String),
    /// Exceção não tratada → 500 "Internal Server Error" (como o FastAPI).
    Internal(String),
}

fn http_err(code: u16, msg: impl Into<String>) -> ApiError {
    ApiError::Http(
        StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        msg.into(),
    )
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            ApiError::Http(code, detail) => (code, Json(json!({ "detail": detail }))).into_response(),
            ApiError::Internal(msg) => {
                tracing::error!("Exceção não tratada: {msg}");
                (StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error").into_response()
            }
        }
    }
}

impl From<rusqlite::Error> for ApiError {
    fn from(e: rusqlite::Error) -> Self {
        ApiError::Internal(format!("sqlite3.Error: {e}"))
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::Http(code, d) => write!(f, "{}: {}", code.as_u16(), d),
            ApiError::Internal(m) => write!(f, "{m}"),
        }
    }
}

type ApiResult = Result<Json<Value>, ApiError>;

/// Extrator JSON que devolve 422 em qualquer falha de corpo (como o FastAPI).
struct ApiJson<T>(T);

impl<S, T> FromRequest<S> for ApiJson<T>
where
    S: Send + Sync,
    T: DeserializeOwned,
{
    type Rejection = ApiError;
    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(v)) => Ok(ApiJson(v)),
            Err(rej) => Err(http_err(422, rej.body_text())),
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Data Models
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
enum Transport {
    #[serde(rename = "stdio")]
    Stdio,
    #[serde(rename = "http")]
    Http,
}

impl Transport {
    fn as_str(&self) -> &'static str {
        match self {
            Transport::Stdio => "stdio",
            Transport::Http => "http",
        }
    }
}

fn d_true() -> bool {
    true
}
fn d_stdio() -> Transport {
    Transport::Stdio
}
fn d_call_timeout() -> f64 {
    DEFAULT_CALL_TIMEOUT
}
fn d_top_k() -> i64 {
    8
}
fn d_min_score() -> f64 {
    0.35
}

#[derive(Deserialize, Clone)]
struct ServerRegisterRequest {
    name: String,
    #[serde(default)]
    skill_summary: String,
    // ── stdio (default): o manager spawna o subprocesso do servidor ──
    #[serde(default = "d_stdio")]
    transport: Transport,
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: IndexMap<String, String>,
    // ── http: conecta a um servidor MCP JÁ RODANDO (streamable-http) ──
    #[serde(default)]
    url: Option<String>,
    #[serde(default = "d_true")]
    enabled: bool,
    // ── capability de base: tools entram no grafo como "core" e o servidor
    // nunca é excluído da seleção por marcação de unhealthy ──
    #[serde(default)]
    always_on: bool,
    // Exemplos de gatilho por tool (chave = nome da tool EXPOSTA pelo servidor).
    #[serde(default)]
    examples: IndexMap<String, Vec<String>>,
}

#[derive(Deserialize)]
struct ServerCallRequest {
    tool: String,
    #[serde(default)]
    arguments: Map<String, Value>,
    /// Teto de execução da tool (segundos); 0 < timeout <= 3600.
    #[serde(default = "d_call_timeout")]
    timeout: f64,
}

#[derive(Deserialize)]
struct SkillsSelectRequest {
    query: String,
    #[serde(default = "d_top_k")]
    top_k: i64,
    #[serde(default = "d_min_score")]
    min_score: f64,
}

#[derive(Deserialize)]
struct SnippetUpdateRequest {
    snippet: String,
}

/// Instalação assistida de servidor MCP npm: resolve o pacote no registry,
/// deriva o binário, registra como stdio `npx -y` e sincroniza.
#[derive(Deserialize)]
struct ServerInstallRequest {
    package: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    skill_summary: String,
    #[allow(dead_code)]
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: IndexMap<String, String>,
    #[serde(default = "d_true")]
    enabled: bool,
    #[serde(default)]
    always_on: bool,
    #[serde(default)]
    examples: IndexMap<String, Vec<String>>,
}

// ══════════════════════════════════════════════════════════════════════════════
// Global State
// ══════════════════════════════════════════════════════════════════════════════

/// Equivalente do `asyncio.Event` (nível, não borda): `set()` fica setado.
#[derive(Clone)]
struct Event(Arc<watch::Sender<bool>>);

impl Event {
    fn new() -> Self {
        Event(Arc::new(watch::channel(false).0))
    }
    fn set(&self) {
        self.0.send_replace(true);
    }
    async fn wait(&self) {
        let mut rx = self.0.subscribe();
        let _ = rx.wait_for(|v| *v).await;
    }
}

/// Keeper task vivo por servidor: `done` sinaliza o fim (depois que o
/// subprocesso foi fechado) — substitui o `await task` do Python.
struct KeeperHandle {
    #[allow(dead_code)]
    abort: AbortHandle,
    done: Event,
}

#[derive(Default)]
struct Runtime {
    // Sessões MCP vivas por servidor. Spawn LAZY (só conecta na primeira
    // chamada/sync/test). O dono real do subprocesso é o keeper task.
    sessions: IndexMap<String, Peer<RoleClient>>,
    // Uma lock por servidor pra evitar spawn duplicado em chamadas concorrentes.
    connect_locks: HashMap<String, Arc<AsyncMutex<()>>>,
    // Geração por servidor: incrementada a cada drop_session. Um connect em voo
    // que terminar DEPOIS de um drop compara sua geração e se descarta.
    gens: HashMap<String, u64>,
    // Evento de parada do keeper task de cada servidor (drop_session → set).
    stop_events: HashMap<String, Event>,
    // Keeper task vivo por servidor — aguardado no shutdown.
    keeper_tasks: HashMap<String, KeeperHandle>,
    // Servidores com falha recente de conexão/call: name → até quando
    // (time.time()+TTL) ficar fora da /skills/select.
    unhealthy: HashMap<String, f64>,
    // Watchdog da memory_api (nasce quando o startup não a encontrou).
    memory_watchdog_task: Option<JoinHandle<()>>,
}

struct AppState {
    cfg: Config,
    // Escritas no SQLite serializadas por um lock explícito (db_lock).
    db: Mutex<Connection>,
    memory: reqwest::Client,
    rt: Mutex<Runtime>,
    // Uma instalação global por vez — `npm view` concorrentes podem corromper o cache.
    install_lock: AsyncMutex<()>,
}

impl AppState {
    fn db(&self) -> MutexGuard<'_, Connection> {
        self.db.lock().unwrap_or_else(|p| p.into_inner())
    }
    fn rt(&self) -> MutexGuard<'_, Runtime> {
        self.rt.lock().unwrap_or_else(|p| p.into_inner())
    }
}

fn mark_unhealthy(st: &AppState, name: &str, cause: &str) {
    let until = now_ts() + st.cfg.unhealthy_ttl_s;
    st.rt().unhealthy.insert(name.to_string(), until);
    warn!(
        "MCP[{name}]: marcado UNHEALTHY por {:.0}s ({cause})",
        st.cfg.unhealthy_ttl_s
    );
}

fn clear_unhealthy(st: &AppState, name: &str) {
    let removed = st.rt().unhealthy.remove(name).is_some();
    if removed {
        info!("MCP[{name}]: recovered — unhealthy removido");
    }
}

fn is_unhealthy(st: &AppState, name: &str) -> bool {
    match st.rt().unhealthy.get(name) {
        Some(until) => now_ts() < *until,
        None => false,
    }
}

// ── Cliente HTTP da memory_api ──────────────────────────────────────────────

enum MemErr {
    Status { code: u16, text: String, msg: String },
    Other { label: &'static str, msg: String },
}

impl std::fmt::Display for MemErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemErr::Status { msg, .. } => write!(f, "HTTPStatusError: {msg}"),
            MemErr::Other { label, msg } => write!(f, "{label}: {msg}"),
        }
    }
}

fn mem_label(e: &reqwest::Error) -> &'static str {
    if e.is_timeout() {
        "TimeoutException"
    } else if e.is_connect() {
        "ConnectError"
    } else if e.is_decode() {
        "JSONDecodeError"
    } else {
        "RequestError"
    }
}

fn mem_other(e: reqwest::Error) -> MemErr {
    MemErr::Other { label: mem_label(&e), msg: e.to_string() }
}

impl AppState {
    fn mem_url(&self, path: &str) -> String {
        format!("{}{}", self.cfg.memory_api_url.trim_end_matches('/'), path)
    }

    /// raise_for_status(): qualquer resposta não-2xx vira erro.
    async fn mem_check(resp: reqwest::Response) -> Result<reqwest::Response, MemErr> {
        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        let url = resp.url().to_string();
        let text = resp.text().await.unwrap_or_default();
        Err(MemErr::Status {
            code: status.as_u16(),
            text,
            msg: format!("{status} for url '{url}'"),
        })
    }

    async fn mem_get_status(&self, timeout: f64) -> Result<u16, reqwest::Error> {
        let r = self
            .memory
            .get(self.mem_url("/status"))
            .timeout(Duration::from_secs_f64(timeout))
            .send()
            .await?;
        Ok(r.status().as_u16())
    }

    async fn mem_delete(&self, path: &str, timeout: f64) -> Result<(), MemErr> {
        let r = self
            .memory
            .delete(self.mem_url(path))
            .timeout(Duration::from_secs_f64(timeout))
            .send()
            .await
            .map_err(mem_other)?;
        Self::mem_check(r).await?;
        Ok(())
    }

    async fn mem_post_ok(&self, path: &str, body: &Value, timeout: f64) -> Result<(), MemErr> {
        let r = self
            .memory
            .post(self.mem_url(path))
            .json(body)
            .timeout(Duration::from_secs_f64(timeout))
            .send()
            .await
            .map_err(mem_other)?;
        Self::mem_check(r).await?;
        Ok(())
    }

    async fn mem_post_json(&self, path: &str, body: &Value, timeout: f64) -> Result<Value, MemErr> {
        let r = self
            .memory
            .post(self.mem_url(path))
            .json(body)
            .timeout(Duration::from_secs_f64(timeout))
            .send()
            .await
            .map_err(mem_other)?;
        let r = Self::mem_check(r).await?;
        r.json::<Value>().await.map_err(mem_other)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// SQLite (registry de servidores + cache de tools) — todas sob db_lock
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
struct ServerRow {
    name: String,
    skill_summary: String,
    transport: String,
    command: Option<String>,
    args_json: String,
    env_json: String,
    url: Option<String>,
    enabled: bool,
    always_on: bool,
    examples_json: String,
    created_at: f64,
    last_sync: Option<f64>,
}

const SERVER_COLS: &str = "name, skill_summary, transport, command, args_json, env_json, url, \
                           enabled, always_on, examples_json, created_at, last_sync";

impl ServerRow {
    fn from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Self> {
        Ok(ServerRow {
            name: r.get(0)?,
            skill_summary: r.get(1)?,
            transport: r.get(2)?,
            command: r.get(3)?,
            args_json: r.get(4)?,
            env_json: r.get(5)?,
            url: r.get(6)?,
            enabled: r.get::<_, i64>(7)? != 0,
            always_on: r.get::<_, i64>(8)? != 0,
            examples_json: r.get(9)?,
            created_at: r.get(10)?,
            last_sync: r.get(11)?,
        })
    }
}

#[derive(Clone)]
struct ToolRow {
    name: String,
    server: String,
    description_raw: String,
    snippet: String,
    input_schema_json: String,
    core: bool,
    // só preenchido por db_all_tools (JOIN com mcp_servers)
    server_enabled: bool,
}

const TOOL_COLS: &str = "name, server, description_raw, snippet, input_schema_json, core";

impl ToolRow {
    fn from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Self> {
        Ok(ToolRow {
            name: r.get(0)?,
            server: r.get(1)?,
            description_raw: r.get(2)?,
            snippet: r.get(3)?,
            input_schema_json: r.get(4)?,
            core: r.get::<_, i64>(5)? != 0,
            server_enabled: false,
        })
    }
}

fn open_db(cfg: &Config) -> rusqlite::Result<Connection> {
    let _ = std::fs::create_dir_all(&cfg.db_dir);
    let conn = Connection::open(&cfg.db_path)?;
    conn.query_row("PRAGMA journal_mode=WAL", [], |r| r.get::<_, String>(0))?;
    conn.execute_batch("PRAGMA synchronous=NORMAL; PRAGMA foreign_keys = ON;")?;
    conn.execute_batch(
        "
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
        ",
    )?;
    // Migração p/ DBs criados antes de always_on/core/examples (guardado por
    // PRAGMA table_info — ALTER em coluna existente daria erro).
    let table_cols = |table: &str| -> rusqlite::Result<Vec<String>> {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(1))?;
        rows.collect()
    };
    let srv_cols = table_cols("mcp_servers")?;
    if !srv_cols.iter().any(|c| c == "always_on") {
        conn.execute_batch("ALTER TABLE mcp_servers ADD COLUMN always_on INTEGER NOT NULL DEFAULT 0")?;
    }
    if !srv_cols.iter().any(|c| c == "examples_json") {
        conn.execute_batch("ALTER TABLE mcp_servers ADD COLUMN examples_json TEXT NOT NULL DEFAULT '{}'")?;
    }
    let tool_cols = table_cols("mcp_tools")?;
    if !tool_cols.iter().any(|c| c == "core") {
        conn.execute_batch("ALTER TABLE mcp_tools ADD COLUMN core INTEGER NOT NULL DEFAULT 0")?;
    }
    Ok(conn)
}

impl AppState {
    fn db_upsert_server(&self, req: &ServerRegisterRequest) -> rusqlite::Result<ServerRow> {
        let now = now_ts();
        let mut examples: IndexMap<String, Vec<String>> = IndexMap::new();
        for (k, v) in &req.examples {
            if v.is_empty() {
                continue;
            }
            examples.insert(
                k.trim().to_string(),
                v.iter()
                    .map(|e| e.trim())
                    .filter(|e| !e.is_empty())
                    .map(String::from)
                    .collect(),
            );
        }
        let db = self.db();
        db.execute_batch("BEGIN")?;
        let res = db
            .execute(
                "INSERT INTO mcp_servers (name, skill_summary, transport, command, \
                 args_json, env_json, url, enabled, always_on, examples_json, created_at) \
                 VALUES (?,?,?,?,?,?,?,?,?,?,?) \
                 ON CONFLICT(name) DO UPDATE SET skill_summary=excluded.skill_summary, \
                 transport=excluded.transport, command=excluded.command, \
                 args_json=excluded.args_json, env_json=excluded.env_json, \
                 url=excluded.url, enabled=excluded.enabled, \
                 always_on=excluded.always_on, examples_json=excluded.examples_json",
                params![
                    req.name,
                    req.skill_summary.trim(),
                    req.transport.as_str(),
                    req.command,
                    serde_json::to_string(&req.args).unwrap_or_else(|_| "[]".into()),
                    serde_json::to_string(&req.env).unwrap_or_else(|_| "{}".into()),
                    req.url,
                    req.enabled as i64,
                    req.always_on as i64,
                    serde_json::to_string(&examples).unwrap_or_else(|_| "{}".into()),
                    now,
                ],
            )
            .and_then(|_| db.execute_batch("COMMIT"));
        if let Err(e) = res {
            let _ = db.execute_batch("ROLLBACK");
            return Err(e);
        }
        db.query_row(
            &format!("SELECT {SERVER_COLS} FROM mcp_servers WHERE name = ?"),
            [&req.name],
            ServerRow::from_row,
        )
    }

    fn db_get_server(&self, name: &str) -> rusqlite::Result<Option<ServerRow>> {
        self.db()
            .query_row(
                &format!("SELECT {SERVER_COLS} FROM mcp_servers WHERE name = ?"),
                [name],
                ServerRow::from_row,
            )
            .optional()
    }

    fn db_list_servers(&self, enabled_only: bool) -> rusqlite::Result<Vec<ServerRow>> {
        let mut q = format!("SELECT {SERVER_COLS} FROM mcp_servers");
        if enabled_only {
            q.push_str(" WHERE enabled = 1");
        }
        q.push_str(" ORDER BY name");
        let db = self.db();
        let mut stmt = db.prepare(&q)?;
        let rows = stmt.query_map([], ServerRow::from_row)?;
        rows.collect()
    }

    fn db_delete_server(&self, name: &str) -> rusqlite::Result<()> {
        let db = self.db();
        db.execute("DELETE FROM mcp_tools WHERE server = ?", [name])?;
        db.execute("DELETE FROM mcp_servers WHERE name = ?", [name])?;
        Ok(())
    }

    fn db_set_last_sync(&self, name: &str) -> rusqlite::Result<()> {
        self.db()
            .execute("UPDATE mcp_servers SET last_sync = ? WHERE name = ?", params![now_ts(), name])?;
        Ok(())
    }

    fn db_set_skill_summary(&self, name: &str, summary: &str) -> rusqlite::Result<()> {
        self.db()
            .execute("UPDATE mcp_servers SET skill_summary = ? WHERE name = ?", params![summary, name])?;
        Ok(())
    }

    fn db_upsert_tool(
        &self,
        server: &str,
        name: &str,
        description_raw: &str,
        snippet: &str,
        input_schema: &Map<String, Value>,
        core: bool,
    ) -> rusqlite::Result<()> {
        self.db().execute(
            "INSERT INTO mcp_tools (name, server, description_raw, snippet, \
             input_schema_json, core, updated_at) VALUES (?,?,?,?,?,?,?) \
             ON CONFLICT(name, server) DO UPDATE SET description_raw=excluded.description_raw, \
             snippet=excluded.snippet, input_schema_json=excluded.input_schema_json, \
             core=excluded.core, updated_at=excluded.updated_at",
            params![
                name,
                server,
                description_raw,
                snippet,
                serde_json::to_string(input_schema).unwrap_or_else(|_| "{}".into()),
                core as i64,
                now_ts(),
            ],
        )?;
        Ok(())
    }

    fn db_tools_for_server(&self, server: &str) -> rusqlite::Result<Vec<ToolRow>> {
        let db = self.db();
        let mut stmt = db.prepare(&format!(
            "SELECT {TOOL_COLS} FROM mcp_tools WHERE server = ? ORDER BY name"
        ))?;
        let rows = stmt.query_map([server], ToolRow::from_row)?;
        rows.collect()
    }

    fn db_all_tools(&self) -> rusqlite::Result<Vec<ToolRow>> {
        let db = self.db();
        let mut stmt = db.prepare(
            "SELECT t.name, t.server, t.description_raw, t.snippet, t.input_schema_json, t.core, \
             s.enabled AS server_enabled \
             FROM mcp_tools t JOIN mcp_servers s ON s.name = t.server ORDER BY t.server, t.name",
        )?;
        let rows = stmt.query_map([], |r| {
            let mut t = ToolRow::from_row(r)?;
            t.server_enabled = r.get::<_, i64>(6)? != 0;
            Ok(t)
        })?;
        rows.collect()
    }

    fn db_delete_tool(&self, server: &str, name: &str) -> rusqlite::Result<()> {
        self.db()
            .execute("DELETE FROM mcp_tools WHERE name = ? AND server = ?", params![name, server])?;
        Ok(())
    }

    fn db_get_tool(&self, server: &str, name: &str) -> rusqlite::Result<Option<ToolRow>> {
        self.db()
            .query_row(
                &format!("SELECT {TOOL_COLS} FROM mcp_tools WHERE name = ? AND server = ?"),
                params![name, server],
                ToolRow::from_row,
            )
            .optional()
    }

    fn db_update_snippet(&self, server: &str, name: &str, snippet: &str) -> rusqlite::Result<()> {
        self.db().execute(
            "UPDATE mcp_tools SET snippet = ?, updated_at = ? WHERE name = ? AND server = ?",
            params![snippet, now_ts(), name, server],
        )?;
        Ok(())
    }

    fn db_count_servers(&self) -> rusqlite::Result<i64> {
        self.db().query_row("SELECT COUNT(*) FROM mcp_servers", [], |r| r.get(0))
    }

    fn db_count_tools(&self, server: &str) -> rusqlite::Result<i64> {
        self.db()
            .query_row("SELECT COUNT(*) FROM mcp_tools WHERE server = ?", [server], |r| r.get(0))
    }
}

fn parse_json_or(s: &str, default: &str) -> Value {
    let src = if s.is_empty() { default } else { s };
    serde_json::from_str(src).unwrap_or_else(|_| serde_json::from_str(default).unwrap())
}

fn parse_args(s: &str) -> Vec<String> {
    serde_json::from_str(if s.is_empty() { "[]" } else { s }).unwrap_or_default()
}

fn parse_env(s: &str) -> IndexMap<String, String> {
    serde_json::from_str(if s.is_empty() { "{}" } else { s }).unwrap_or_default()
}

// ══════════════════════════════════════════════════════════════════════════════
// Runtime MCP (executor) — spawn de servidores stdio/http
// ══════════════════════════════════════════════════════════════════════════════

/// Reescreve o `--cdp-endpoint` NO MOMENTO do spawn.
///
/// O endpoint registrado no DB (`http://localhost:9222`) aponta para o Chrome
/// de depuração NO HOST. Dentro do container no Windows/WSL2, `localhost` é o
/// PRÓPRIO container — o docker-compose.windows.yml injeta
/// `AVA_CDP_ENDPOINT=http://host.docker.internal:9222` e este helper troca a
/// URL aqui. No Linux (host network) o env não existe → no-op.
fn resolve_cdp_endpoint(args: &[String]) -> Vec<String> {
    let endpoint = std::env::var("AVA_CDP_ENDPOINT").unwrap_or_default();
    let endpoint = endpoint.trim();
    if endpoint.is_empty() {
        return args.to_vec();
    }
    let mut out: Vec<String> = Vec::new();
    let mut pending = false; // --cdp-endpoint esperando a URL no arg seguinte
    for a in args {
        if pending {
            out.push(endpoint.to_string());
            pending = false;
        } else if a == "--cdp-endpoint" {
            out.push(a.clone());
            pending = true;
        } else if a.starts_with("--cdp-endpoint=") {
            out.push(format!("--cdp-endpoint={endpoint}"));
        } else if a.contains("://localhost:9222") {
            out.push(
                a.replace("http://localhost:9222", endpoint)
                    .replace("https://localhost:9222", endpoint),
            );
        } else {
            out.push(a.clone());
        }
    }
    out
}

// ── Compatibilidade Linux/Windows no spawn dos servidores MCP stdio ─────────
//
// (ÚNICA adição em relação ao mcp_manager.py.) O SDK Python do MCP fazia isto
// por baixo dos panos; aqui fica explícito e por plataforma:
//   • Variáveis de ambiente herdadas: apenas uma lista segura POR PLATAFORMA
//     (+ o `env` do servidor por cima) — o filho NÃO herda o ambiente inteiro.
//   • Windows: `npx` é um shim `.cmd` — resolvido via PATH/PATHEXT (CreateProcess
//     não o acha pelo nome puro); sem janela de console.
//   • `npx`/`node` fora do PATH (manager iniciado pelo desktop/tray, sem shell —
//     nvm/volta/fnm/npm-global não aparecem no PATH herdado): resolução
//     automática varrendo instalações bem-conhecidas por plataforma, e o
//     diretório do binário resolvido é injetado à frente do PATH herdado pelo
//     filho (processos neto também acham o runtime).
//   • Unix: o filho vira líder de grupo de processos, para o encerramento
//     derrubar a árvore toda (npx → node), não só o wrapper.

#[cfg(windows)]
const DEFAULT_INHERITED_ENV_VARS: &[&str] = &[
    "APPDATA",
    "HOMEDRIVE",
    "HOMEPATH",
    "LOCALAPPDATA",
    "PATH",
    "PATHEXT",
    "COMSPEC",
    "PROCESSOR_ARCHITECTURE",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "USERNAME",
    "USERPROFILE",
];

#[cfg(not(windows))]
const DEFAULT_INHERITED_ENV_VARS: &[&str] = &["HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER"];

fn default_environment() -> IndexMap<String, String> {
    let mut env = IndexMap::new();
    for key in DEFAULT_INHERITED_ENV_VARS {
        let Ok(value) = std::env::var(key) else { continue };
        // Funções de shell exportadas (`() { ...`) não são herdadas.
        if value.starts_with("()") {
            continue;
        }
        env.insert((*key).to_string(), value);
    }
    env
}

// ── Resolução automática de executáveis (npx/node/uvx/…) ────────────────────
//
// O usuário registra servidores com `command: "npx"` — sem caminho absoluto.
// O manager pode rodar iniciado pelo desktop/tray, fora de um shell, onde o
// PATH nem sempre contém nvm/volta/fnm/npm-global. A resolução tenta, nesta
// ordem:
//   1. caminho explícito (contém separador `/` ou `\`) → usado como está;
//   2. PATH do processo via `which` (Windows respeita PATHEXT → acha npx.cmd);
//   3. fallback: varredura de instalações bem-conhecidas POR PLATAFORMA
//      (nvm, volta, fnm, asdf, npm global, nodejs oficial, scoop,
//      chocolatey…), da versão mais nova para a mais velha.
// O resultado é cacheado por nome de comando (a varredura roda 1× por
// processo) e o diretório do binário resolvido é injetado à frente do PATH
// herdado pelo filho — ver `build_stdio_command`.

static RESOLVED_EXECUTABLES: LazyLock<Mutex<HashMap<String, String>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn resolve_executable(command: &str) -> String {
    // Caminho explícito (absoluto ou relativo) não precisa de resolução.
    if command.contains('/') || command.contains('\\') {
        return command.to_string();
    }
    if let Ok(cache) = RESOLVED_EXECUTABLES.lock() {
        if let Some(hit) = cache.get(command) {
            return hit.clone();
        }
    }
    let resolved = resolve_executable_uncached(command);
    if let Ok(mut cache) = RESOLVED_EXECUTABLES.lock() {
        cache.insert(command.to_string(), resolved.clone());
    }
    resolved
}

fn resolve_executable_uncached(command: &str) -> String {
    // 1) PATH do processo (via crate `which`, válido nas duas plataformas).
    if let Ok(p) = which::which(command) {
        return p.to_string_lossy().into_owned();
    }
    // 2) Instalações bem-conhecidas (o PATH do processo não cobriu — caso
    //    comum quando o manager é iniciado pelo desktop/tray, sem shell).
    for dir in well_known_command_dirs() {
        for candidate in command_suffixes(command) {
            let p = dir.join(&candidate);
            if is_executable_file(&p) {
                info!(
                    "MCP spawn: '{command}' não está no PATH — resolvido via instalação conhecida: {}",
                    p.display()
                );
                return p.to_string_lossy().into_owned();
            }
        }
    }
    // 3) Não achou em lugar nenhum — devolve o nome puro e deixa o erro de
    //    spawn acontece com a mensagem nativa (comportamento anterior).
    command.to_string()
}

/// Sufixos tentados na varredura de fallback. Windows: `npx` vive como shim
/// `.cmd` (e ferramentas podem ser `.exe`/`.bat`); Unix: nome exato.
#[cfg(windows)]
fn command_suffixes(command: &str) -> Vec<String> {
    vec![
        command.to_string(),
        format!("{command}.cmd"),
        format!("{command}.exe"),
        format!("{command}.bat"),
    ]
}

#[cfg(not(windows))]
fn command_suffixes(command: &str) -> Vec<String> {
    vec![command.to_string()]
}

fn is_executable_file(p: &std::path::Path) -> bool {
    match std::fs::metadata(p) {
        Ok(md) if md.is_file() => {
            #[cfg(not(windows))]
            {
                use std::os::unix::fs::PermissionsExt;
                md.permissions().mode() & 0o111 != 0
            }
            #[cfg(windows)]
            {
                true
            }
        }
        _ => false,
    }
}

/// Diretórios onde os gerenciadores de Node/npx comuns instalam os binários —
/// usados quando o comando não está no PATH do processo.
fn well_known_command_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    let env_dir = |name: &str| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty())
    };
    let home = env_dir("HOME").or_else(|| env_dir("USERPROFILE"));

    #[cfg(windows)]
    {
        if let Some(appdata) = env_dir("APPDATA") {
            dirs.push(appdata.join("npm")); // npm global: npx.cmd vive aqui
            dirs.push(appdata.join("nvm")); // nvm-windows: versão ativa
            push_version_dirs(&mut dirs, &appdata.join("nvm"), ""); // todas as versões
        }
        if let Some(d) = env_dir("NVM_HOME") {
            dirs.push(d.clone());
            push_version_dirs(&mut dirs, &d, "");
        }
        if let Some(d) = env_dir("NVM_SYMLINK") {
            dirs.push(d);
        }
        if let Some(local) = env_dir("LOCALAPPDATA") {
            push_version_dirs(
                &mut dirs,
                &local.join("fnm").join("node-versions"),
                "installation",
            );
            dirs.push(local.join("Volta").join("bin")); // shims volta
            push_version_dirs(
                &mut dirs,
                &local
                    .join("Volta")
                    .join("tools")
                    .join("image")
                    .join("node"),
                "",
            );
        }
        if let Some(d) = env_dir("ProgramFiles") {
            dirs.push(d.join("nodejs"));
        }
        if let Some(d) = env_dir("ProgramFiles(x86)") {
            dirs.push(d.join("nodejs"));
        }
        if let Some(home) = &home {
            dirs.push(home.join("scoop").join("apps").join("nodejs").join("current"));
            dirs.push(home.join("scoop").join("shims"));
        }
        if let Some(d) = env_dir("ProgramData") {
            dirs.push(d.join("chocolatey").join("bin"));
        }
    }

    #[cfg(not(windows))]
    {
        if let Some(home) = &home {
            push_version_dirs(
                &mut dirs,
                &home.join(".nvm").join("versions").join("node"),
                "bin",
            );
            push_version_dirs(
                &mut dirs,
                &home.join(".local").join("share").join("fnm").join("node-versions"),
                "installation/bin",
            );
            push_version_dirs(
                &mut dirs,
                &home.join(".fnm").join("node-versions"),
                "installation/bin",
            );
            dirs.push(home.join(".volta").join("bin"));
            dirs.push(home.join(".asdf").join("shims"));
            dirs.push(home.join(".local").join("bin"));
            dirs.push(home.join("bin"));
        }
        #[cfg(target_os = "macos")]
        {
            dirs.push(PathBuf::from("/opt/homebrew/bin"));
            dirs.push(PathBuf::from("/usr/local/opt/node/bin"));
        }
        dirs.push(PathBuf::from("/usr/local/bin"));
        dirs.push(PathBuf::from("/usr/bin"));
        dirs.push(PathBuf::from("/opt/node/bin"));
        dirs.push(PathBuf::from("/opt/nodejs/bin"));
    }

    dirs
}

/// Para gerenciadores de versão (nvm/fnm/volta): empurra `base/<versão>[/<sub>]`
/// para cada subdiretório de `base` cujo nome começa com `v` ou dígito, da
/// versão mais nova para a mais velha.
fn push_version_dirs(out: &mut Vec<PathBuf>, base: &std::path::Path, sub: &str) {
    let Ok(rd) = std::fs::read_dir(base) else { return };
    let mut versions: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| {
                        n.starts_with('v') || n.chars().next().is_some_and(|c| c.is_ascii_digit())
                    })
                    .unwrap_or(false)
        })
        .collect();
    versions.sort_by(|a, b| {
        let ka = a.file_name().map(version_key).unwrap_or_default();
        let kb = b.file_name().map(version_key).unwrap_or_default();
        kb.cmp(&ka) // mais nova primeiro
    });
    for v in versions {
        let mut d = v;
        if !sub.is_empty() {
            d.push(sub);
        }
        out.push(d);
    }
}

/// Chave de ordenação de versão ("v22.11.0" → [22, 11, 0]) para ordenar as
/// instalações dos gerenciadores de versão da mais nova para a mais velha.
fn version_key(name: &std::ffi::OsStr) -> Vec<u64> {
    name.to_string_lossy()
        .trim_start_matches('v')
        .split(['.', '-'])
        .map(|seg| {
            seg.chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse()
                .unwrap_or(0)
        })
        .collect()
}

fn build_stdio_command(
    command: &str,
    args: &[String],
    env: &IndexMap<String, String>,
) -> process_wrap::tokio::TokioCommandWrap {
    let resolved = resolve_executable(command);
    let mut cmd = tokio::process::Command::new(&resolved);
    cmd.args(args);
    // env = {**default_environment(), **server.env}
    let mut merged = default_environment();
    for (k, v) in env {
        merged.insert(k.clone(), v.clone());
    }
    // Se o executável foi resolvido (fora do PATH), injeta o diretório dele à
    // frente do PATH herdado — processos neto (npx → node → binários do
    // pacote) precisam achar o runtime também, mesmo sem shell de origem.
    if (resolved.contains('/') || resolved.contains('\\')) && resolved != command {
        if let Some(bin_dir) = std::path::Path::new(&resolved).parent() {
            let bin_dir = bin_dir.to_path_buf();
            let sep = if cfg!(windows) { ';' } else { ':' };
            let already = merged
                .get("PATH")
                .map(|existing| {
                    existing
                        .split(sep)
                        .any(|p| std::path::Path::new(p.trim_matches('"')) == bin_dir)
                })
                .unwrap_or(false);
            if !already {
                let new_val = match merged.get("PATH") {
                    Some(existing) => format!("{}{}{}", bin_dir.to_string_lossy(), sep, existing),
                    None => bin_dir.to_string_lossy().into_owned(),
                };
                merged.insert("PATH".to_string(), new_val);
            }
        }
    }
    cmd.env_clear();
    cmd.envs(merged.iter());
    cmd.kill_on_drop(true);
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    #[allow(unused_mut)]
    let mut wrap = process_wrap::tokio::TokioCommandWrap::from(cmd);
    #[cfg(unix)]
    {
        wrap.wrap(process_wrap::tokio::ProcessGroup::leader());
    }
    wrap
}

/// Derruba a sessão/subprocesso do servidor (se vivo). Usado quando a config
/// muda, o servidor é removido, ou uma chamada falhou (sessão provavelmente
/// podre — a próxima respawna do zero). O bump de geração invalida connects em
/// voo que começaram ANTES deste drop. O fechamento real do subprocesso é
/// responsabilidade do keeper task — o evento `stop` é o sinal.
fn drop_session(st: &AppState, name: &str) {
    let mut rt = st.rt();
    *rt.gens.entry(name.to_string()).or_insert(0) += 1;
    rt.sessions.shift_remove(name);
    rt.connect_locks.remove(name);
    if let Some(ev) = rt.stop_events.remove(name) {
        ev.set();
    }
}

/// Equivalente do `finally` do keeper: limpa o estado do servidor e sinaliza
/// `done`. Roda também quando o task é abortado (o guard é dropado junto com o
/// future, mesmo que ele nunca tenha chegado a ser polled).
struct KeeperGuard {
    st: Arc<AppState>,
    name: String,
    done: Event,
    popped: bool,
}

impl KeeperGuard {
    fn pop_state(&mut self) {
        if self.popped {
            return;
        }
        self.popped = true;
        let mut rt = self.st.rt();
        rt.sessions.shift_remove(&self.name);
        rt.keeper_tasks.remove(&self.name);
        rt.stop_events.remove(&self.name);
    }
}

impl Drop for KeeperGuard {
    fn drop(&mut self) {
        self.pop_state();
        self.done.set();
    }
}

type Service = RunningService<RoleClient, ()>;

/// Task DONA da sessão do servidor: abre o transporte MCP (spawn stdio ou
/// conexão http), inicializa, cacheia a sessão e segura o subprocesso vivo até
/// `stop` ser setado. Toda a vida do contexto (abrir E fechar) acontece NESTE
/// task — mesmo padrão keeper do original.
async fn session_keeper(
    st: Arc<AppState>,
    name: String,
    row: ServerRow,
    ready: Event,
    err: Arc<Mutex<Vec<String>>>,
    stop: Event,
    mut guard: KeeperGuard,
) {
    let mut service: Option<Service> = None;
    let result: Result<(), String> = async {
        let svc: Service = if row.transport == "http" {
            let url = row
                .url
                .clone()
                .filter(|u| !u.is_empty())
                .ok_or_else(|| "RuntimeError: transport='http' sem url configurada".to_string())?;
            info!("MCP[{name}]: conectando via streamable-http em {url}");
            let transport = StreamableHttpClientTransport::from_uri(url);
            ().serve(transport)
                .await
                .map_err(|e| format!("ClientInitializeError: {e}"))?
        } else {
            let command = row
                .command
                .clone()
                .filter(|c| !c.is_empty())
                .ok_or_else(|| "RuntimeError: transport='stdio' sem command configurado".to_string())?;
            let args = resolve_cdp_endpoint(&parse_args(&row.args_json));
            let env = parse_env(&row.env_json);
            info!("MCP[{name}]: spawn stdio `{} {}`", command, args.join(" "));
            let transport = TokioChildProcess::new(build_stdio_command(&command, &args, &env))
                .map_err(|e| format!("OSError: {e}"))?;
            ().serve(transport)
                .await
                .map_err(|e| format!("ClientInitializeError: {e}"))?
        };
        st.rt().sessions.insert(name.clone(), svc.peer().clone());
        service = Some(svc);
        ready.set();
        stop.wait().await; // subprocesso vivo até drop/shutdown
        Ok::<(), String>(())
    }
    .await;

    if let Err(e) = result {
        err.lock().unwrap_or_else(|p| p.into_inner()).push(e);
        ready.set(); // desbloqueia o get_session esperando
    }
    // finally:
    guard.pop_state();
    if let Some(svc) = service {
        if let Err(e) = svc.cancel().await {
            debug!("[{name}] falha ao fechar sessão do keeper (ignorado): {e}");
        }
    }
    // `guard` dropado aqui → done.set()
}

/// Retorna a sessão do servidor, conectando (spawn) na primeira vez — via
/// keeper task próprio. O handshake inteiro tem teto de CONNECT_TIMEOUT:
/// servidor que não fala MCP não pode pendurar o handler (e com ele a
/// connect_lock, que bloquearia TODAS as futuras /call, /sync e /test).
async fn get_session(
    st: &Arc<AppState>,
    name: &str,
    allow_disabled: bool,
) -> Result<Peer<RoleClient>, ApiError> {
    let generation = st.rt().gens.get(name).copied().unwrap_or(0);
    let existing = st.rt().sessions.get(name).cloned();
    if let Some(s) = existing {
        return Ok(s);
    }
    let row = st
        .db_get_server(name)?
        .ok_or_else(|| http_err(404, format!("Servidor MCP '{name}' não registrado")))?;
    if !row.enabled && !allow_disabled {
        return Err(http_err(409, format!("Servidor MCP '{name}' está desabilitado")));
    }

    let lock = st
        .rt()
        .connect_locks
        .entry(name.to_string())
        .or_insert_with(|| Arc::new(AsyncMutex::new(())))
        .clone();
    let _held = lock.lock().await;
    let existing = st.rt().sessions.get(name).cloned();
    if let Some(s) = existing {
        // outro request venceu a corrida
        return Ok(s);
    }
    let t0 = Instant::now();
    let (ready, stop, done) = (Event::new(), Event::new(), Event::new());
    let err: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let guard = KeeperGuard {
        st: st.clone(),
        name: name.to_string(),
        done: done.clone(),
        popped: false,
    };
    let abort = {
        let mut rt = st.rt();
        rt.stop_events.insert(name.to_string(), stop.clone());
        let handle = tokio::spawn(session_keeper(
            st.clone(),
            name.to_string(),
            row,
            ready.clone(),
            err.clone(),
            stop.clone(),
            guard,
        ));
        let abort = handle.abort_handle();
        rt.keeper_tasks.insert(
            name.to_string(),
            KeeperHandle { abort: abort.clone(), done: done.clone() },
        );
        abort
    };

    let connect_timeout = Duration::from_secs_f64(st.cfg.connect_timeout);
    if tokio::time::timeout(connect_timeout, ready.wait()).await.is_err() {
        // Cancelar (não só sinalizar): o keeper está preso DENTRO do handshake
        // e não vai olhar o `stop` — o abort derruba o future, o que fecha o
        // subprocesso (sem zumbi de `sleep` eterno).
        stop.set();
        abort.abort();
        let _ = tokio::time::timeout(Duration::from_secs(10), done.wait()).await;
        mark_unhealthy(st, name, &format!("timeout de conexão ({:.0}s)", st.cfg.connect_timeout));
        return Err(http_err(
            502,
            format!(
                "Timeout ao conectar no servidor MCP '{name}' ({:.0}s) — spawn/handshake não completou",
                st.cfg.connect_timeout
            ),
        ));
    }
    let first_err = err.lock().unwrap_or_else(|p| p.into_inner()).first().cloned();
    if let Some(first_err) = first_err {
        stop.set(); // keeper falhou antes do ready — garante a limpeza
        done.wait().await;
        mark_unhealthy(st, name, &format!("falha de conexão: {first_err}"));
        return Err(http_err(
            502,
            format!("Falha ao conectar no servidor MCP '{name}': {first_err}"),
        ));
    }
    // Corrida re-register/delete × connect: a config mudou enquanto
    // conectávamos — a sessão recém-nascida já está obsoleta; descarta.
    let current_gen = st.rt().gens.get(name).copied().unwrap_or(0);
    if current_gen != generation {
        drop_session(st, name);
        return Err(http_err(
            409,
            format!("Config do servidor '{name}' mudou durante a conexão — repita a chamada"),
        ));
    }
    clear_unhealthy(st, name);
    info!("MCP[{name}]: conectado em {:.0}ms", t0.elapsed().as_secs_f64() * 1000.0);
    let session = st.rt().sessions.get(name).cloned();
    session.ok_or_else(|| ApiError::Internal(format!("KeyError: '{name}'")))
}

// ══════════════════════════════════════════════════════════════════════════════
// Watchdog da memory_api
// ══════════════════════════════════════════════════════════════════════════════

/// Vigia a memory_api: quando ela VOLTA depois de um período offline,
/// re-sincroniza os servidores habilitados — as tools reentram no grafo sem
/// intervenção manual (o register é idempotente; o doc_hash evita re-embed de
/// tool inalterada).
///
/// Motivação: o docker-compose sobe o manager ANTES da memory — o check
/// one-shot do startup via `✗ OFFLINE`, ninguém re-tentava. Também cobre a
/// memory cair e voltar no meio da sessão.
async fn memory_watchdog(st: Arc<AppState>, poll_s: f64) {
    let mut was_offline = true; // só nasce quando o startup não achou a memory
    loop {
        tokio::time::sleep(Duration::from_secs_f64(poll_s)).await;
        let online = matches!(st.mem_get_status(5.0).await, Ok(200));
        if !online {
            was_offline = true;
            continue;
        }
        if !was_offline {
            continue;
        }
        was_offline = false;
        let servers: Vec<String> = match st.db_list_servers(true) {
            Ok(rows) => rows.into_iter().map(|r| r.name).collect(),
            Err(e) => {
                warn!("MCP watchdog: falha ao listar servidores: {e}");
                continue;
            }
        };
        info!("MCP watchdog: memory_api de volta — re-sincronizando {}", py_list(&servers));
        for name in &servers {
            match sync_server_inner(&st, name).await {
                Ok(summary) => {
                    info!(
                        "MCP watchdog: sync de '{name}' ok — {} tool(s)",
                        summary["synced"].as_i64().unwrap_or(0)
                    );
                }
                // Falha isolada não impede os demais — a próxima detecção de
                // volta (memory fora→dentro) re-tenta tudo.
                Err(e) => warn!("MCP watchdog: sync de '{name}' falhou: HTTPException: {e}"),
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Rotas
// ══════════════════════════════════════════════════════════════════════════════

async fn health(State(st): State<Arc<AppState>>) -> ApiResult {
    let n = st.db_count_servers()?;
    let connected: Vec<String> = st.rt().sessions.keys().cloned().collect();
    Ok(Json(json!({ "status": "ok", "servers": n, "connected": connected })))
}

// ── Registro de servidores ──────────────────────────────────────────────────

fn server_dict(st: &AppState, row: &ServerRow) -> Result<Value, ApiError> {
    Ok(json!({
        "name": row.name,
        "skill_summary": row.skill_summary,
        "transport": row.transport,
        "command": row.command,
        "args": parse_json_or(&row.args_json, "[]"),
        "env": parse_json_or(&row.env_json, "{}"),
        "url": row.url,
        "enabled": row.enabled,
        "always_on": row.always_on,
        "examples": parse_json_or(&row.examples_json, "{}"),
        "unhealthy": is_unhealthy(st, &row.name),
        "created_at": row.created_at,
        "last_sync": row.last_sync,
        "tools": st.db_count_tools(&row.name)?,
    }))
}

/// Registra (ou atualiza) um servidor MCP. stdio exige `command`; http exige
/// `url`. Re-registrar o mesmo nome atualiza a config e derruba a sessão viva —
/// a próxima chamada/sync reconecta com a config nova.
async fn register_server(
    State(st): State<Arc<AppState>>,
    ApiJson(req): ApiJson<ServerRegisterRequest>,
) -> ApiResult {
    let name = req.name.trim().to_string();
    if !SERVER_NAME_RE.is_match(&name) || name.contains("__") {
        return Err(http_err(
            400,
            "Nome de servidor inválido: use [A-Za-z0-9_-], sem '__' (é o separador do graph_name das tools)",
        ));
    }
    if req.transport == Transport::Stdio && req.command.as_deref().unwrap_or("").trim().is_empty() {
        return Err(http_err(400, "transport='stdio' exige 'command'"));
    }
    if req.transport == Transport::Http {
        let u = req.url.as_deref().unwrap_or("").trim();
        if !(u.starts_with("http://") || u.starts_with("https://")) {
            return Err(http_err(400, "transport='http' exige 'url' (http:// ou https://)"));
        }
    }
    let existed = st.db_get_server(&name)?.is_some();
    let row = st.db_upsert_server(&req)?;
    if existed {
        drop_session(&st, &name);
        info!("MCP: servidor '{name}' atualizado ({})", req.transport.as_str());
    } else {
        info!(
            "MCP: servidor '{name}' registrado ({}, enabled={})",
            req.transport.as_str(),
            if req.enabled { "True" } else { "False" }
        );
    }
    Ok(Json(json!({
        "registered": true, "updated": existed, "server": server_dict(&st, &row)?
    })))
}

async fn list_servers(State(st): State<Arc<AppState>>) -> ApiResult {
    let mut servers = Vec::new();
    for r in st.db_list_servers(false)? {
        servers.push(server_dict(&st, &r)?);
    }
    Ok(Json(json!({ "servers": servers })))
}

async fn get_server(State(st): State<Arc<AppState>>, Path(name): Path<String>) -> ApiResult {
    let row = st
        .db_get_server(&name)?
        .ok_or_else(|| http_err(404, format!("Servidor MCP '{name}' não registrado")))?;
    Ok(Json(json!({ "server": server_dict(&st, &row)? })))
}

/// Remove o servidor, suas tools do cache local E do grafo de tools (ToolsDB,
/// via memory DELETE /tools/{graph_name}). Sessão viva é derrubada (subprocesso
/// stdio morto junto). Tools cujo DELETE no grafo falhou (memory offline) ficam
/// ÓRFÃS no ToolsDB — reportadas em `orphaned_in_graph`.
async fn delete_server(State(st): State<Arc<AppState>>, Path(name): Path<String>) -> ApiResult {
    if st.db_get_server(&name)?.is_none() {
        return Err(http_err(404, format!("Servidor MCP '{name}' não registrado")));
    }
    let tools = st.db_tools_for_server(&name)?;
    let mut removed = 0usize;
    let mut orphaned: Vec<String> = Vec::new();
    for t in &tools {
        let graph = graph_name(&name, &t.name);
        match st.mem_delete(&format!("/tools/{graph}"), 30.0).await {
            Ok(()) => removed += 1,
            Err(MemErr::Status { code, text, .. }) => {
                if code == 404 {
                    removed += 1; // já não existia no grafo — ok
                } else {
                    orphaned.push(graph.clone());
                    warn!(
                        "MCP: falha ao remover tool '{graph}' do grafo: {code} {}",
                        truncate_chars(&text, 200)
                    );
                }
            }
            Err(e) => {
                orphaned.push(graph.clone());
                warn!("MCP: falha ao remover tool '{graph}' do grafo: {e}");
            }
        }
        st.db_delete_tool(&name, &t.name)?;
    }
    st.db_delete_server(&name)?;
    drop_session(&st, &name);
    if !orphaned.is_empty() {
        warn!(
            "MCP: servidor '{name}' removido, mas {} tool(s) ficaram ÓRFÃS no ToolsDB: {}",
            orphaned.len(),
            py_list(&orphaned)
        );
    }
    info!(
        "MCP: servidor '{name}' removido ({} tool(s), {removed} apagadas do grafo)",
        tools.len()
    );
    Ok(Json(json!({
        "deleted": true, "name": name,
        "tools_removed_from_graph": removed, "orphaned_in_graph": orphaned,
    })))
}

// ── Sync (list_tools → grafo de tools + cache local) ────────────────────────

async fn list_tools_with_timeout(
    session: &Peer<RoleClient>,
) -> Result<rmcp::model::ListToolsResult, (&'static str, String)> {
    let fut = session.list_tools(Default::default());
    match tokio::time::timeout(Duration::from_secs_f64(SYNC_TIMEOUT), fut).await {
        Err(_) => Err(("TimeoutError", String::new())),
        Ok(Err(e)) => Err(("McpError", e.to_string())),
        Ok(Ok(r)) => Ok(r),
    }
}

/// Conecta no servidor, roda list_tools() e grava cada tool na memória de
/// grafos de tools (ToolsDB, via memory /tools/register/batch) com o snippet
/// como descrição — snippet inicial = description do próprio MCP. Tools que
/// sumiram do servidor são removidas do grafo e do cache. Idempotente.
async fn sync_server_inner(st: &Arc<AppState>, name: &str) -> Result<Value, ApiError> {
    let row = st
        .db_get_server(name)?
        .ok_or_else(|| http_err(404, format!("Servidor MCP '{name}' não registrado")))?;
    let session = get_session(st, name, true).await?;
    let result = match list_tools_with_timeout(&session).await {
        Ok(r) => r,
        Err((label, msg)) => {
            drop_session(st, name);
            mark_unhealthy(st, name, &format!("list_tools falhou: {label}"));
            return Err(http_err(
                502,
                format!("list_tools falhou no servidor '{name}': {label}: {msg}"),
            ));
        }
    };

    let mut live: IndexMap<String, rmcp::model::Tool> = IndexMap::new();
    for t in result.tools {
        live.insert(t.name.to_string(), t);
    }
    let prev_rows: HashMap<String, ToolRow> = st
        .db_tools_for_server(name)?
        .into_iter()
        .map(|t| (t.name.clone(), t))
        .collect();
    let always_on = row.always_on;
    let examples_by_tool: IndexMap<String, Vec<String>> =
        serde_json::from_str(if row.examples_json.is_empty() { "{}" } else { &row.examples_json })
            .unwrap_or_default();
    let mut tools_out: Vec<Value> = Vec::new();
    let mut to_register: Vec<Value> = Vec::new();
    for tool in live.values() {
        let tool_name = tool.name.to_string();
        let raw = tool.description.as_deref().unwrap_or("").trim().to_string();
        let mut snippet = if raw.is_empty() {
            format!("Tool '{tool_name}' do servidor MCP '{name}'.")
        } else {
            raw.clone()
        };
        if let Some(prev) = prev_rows.get(&tool_name) {
            // Snippet customizado (PUT /snippet): usuário escreveu um skill.md
            // diferente da description crua do servidor — PRESERVA no re-sync
            // (só a description_raw acompanha o servidor).
            let custom = prev.snippet.trim() != prev.description_raw.trim();
            if custom && !prev.snippet.trim().is_empty() {
                snippet = prev.snippet.clone();
            }
        }
        if let Err(e) = st.db_upsert_tool(name, &tool_name, &raw, &snippet, &tool.input_schema, always_on) {
            // Corrida rara: o servidor foi DELETADO durante o await do
            // list_tools — as rows filhas já foram removidas (FK cascade).
            if let rusqlite::Error::SqliteFailure(f, _) = &e {
                if f.code == ErrorCode::ConstraintViolation {
                    warn!("MCP: sync de '{name}' interrompido — servidor removido durante o sync");
                    return Err(http_err(409, format!("Servidor '{name}' foi removido durante o sync")));
                }
            }
            return Err(e.into());
        }
        tools_out.push(json!({
            "name": tool_name, "graph_name": graph_name(name, &tool_name), "snippet": snippet,
        }));
        // TODAS as tools vão no batch (não só as alteradas): o doc_hash na
        // ToolsDB já evita re-embed de tool inalterada, e o caminho "unchanged"
        // de lá aplica set_core — necessário quando o usuário flipa always_on e
        // re-sincroniza.
        to_register.push(json!({
            "name": graph_name(name, &tool_name), "description": snippet,
            "core": always_on,
            "examples": examples_by_tool.get(&tool_name).cloned().unwrap_or_default(),
        }));
    }

    // Stale: tools que estavam no cache mas sumiram do servidor — saem do grafo
    // (DELETE) e do cache. 404 do memory = já não existia, ok.
    let stale: Vec<String> = st
        .db_tools_for_server(name)?
        .into_iter()
        .filter(|t| !live.contains_key(&t.name))
        .map(|t| t.name)
        .collect();
    for tname in &stale {
        let graph = graph_name(name, tname);
        match st.mem_delete(&format!("/tools/{graph}"), 30.0).await {
            Ok(()) => {}
            Err(MemErr::Status { code, .. }) => {
                if code != 404 {
                    warn!("MCP: falha ao remover tool obsoleta '{graph}' do grafo: {code}");
                }
            }
            Err(e) => warn!("MCP: falha ao remover tool obsoleta '{graph}' do grafo: {e}"),
        }
        st.db_delete_tool(name, tname)?;
    }

    // Registro no grafo (uma chamada batch; embeddings podem demorar —
    // REGISTER_TIMEOUT cobre). Falha de memory NÃO desfaz o cache local:
    // re-sync depois re-registra.
    if !to_register.is_empty() {
        let body = json!({
            "tools": to_register,
            "prune_missing": false, // NUNCA prune global: há tools de outros servidores
        });
        if let Err(e) = st.mem_post_ok("/tools/register/batch", &body, REGISTER_TIMEOUT).await {
            return Err(http_err(
                502,
                format!(
                    "Sync de '{name}' leu {} tool(s), mas o registro no grafo de tools \
                     falhou (memory_api offline?): {e}",
                    tools_out.len()
                ),
            ));
        }
    }

    let mut summary = row.skill_summary.trim().to_string();
    if summary.is_empty() && !live.is_empty() {
        let mut sorted: Vec<&String> = live.keys().collect();
        sorted.sort();
        let names = sorted.iter().take(8).map(|s| s.as_str()).collect::<Vec<_>>().join(", ");
        summary = format!("Servidor MCP '{name}' — tools: {names}");
        if live.len() > 8 {
            summary.push('…');
        }
        st.db_set_skill_summary(name, &summary)?;
    }
    st.db_set_last_sync(name)?;
    info!(
        "MCP: sync de '{name}': {} tool(s), {} nova(s)/alterada(s) no grafo, {} removida(s)",
        tools_out.len(),
        to_register.len(),
        stale.len()
    );
    Ok(json!({
        "synced": tools_out.len(), "registered": to_register.len(), "removed": stale.len(),
        "skill_summary": summary, "tools": tools_out,
    }))
}

async fn sync_server(State(st): State<Arc<AppState>>, Path(name): Path<String>) -> ApiResult {
    Ok(Json(sync_server_inner(&st, &name).await?))
}

/// Health check: conecta (spawn real) + list_tools, SEM gravar nada. Erro volta
/// como 502 com a causa — é o debug de config do usuário.
async fn test_server(State(st): State<Arc<AppState>>, Path(name): Path<String>) -> ApiResult {
    if st.db_get_server(&name)?.is_none() {
        return Err(http_err(404, format!("Servidor MCP '{name}' não registrado")));
    }
    let session = get_session(&st, &name, true).await?;
    let result = match list_tools_with_timeout(&session).await {
        Ok(r) => r,
        Err((label, msg)) => {
            drop_session(&st, &name);
            return Err(http_err(
                502,
                format!("Servidor '{name}' conectou mas list_tools falhou: {label}: {msg}"),
            ));
        }
    };
    Ok(Json(json!({
        "ok": true, "server": name, "tools": result.tools.len(),
        "names": result.tools.iter().map(|t| t.name.to_string()).collect::<Vec<_>>(),
    })))
}

// ── Instalação assistida (pacotes npm) ──────────────────────────────────────

/// Roda um subprocesso com teto de tempo; stdout+stderr joined.
async fn run_cmd(cmd: &[&str], timeout: f64) -> Result<(i32, String), String> {
    let mut c = tokio::process::Command::new(cmd[0]);
    c.args(&cmd[1..])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    let child = c.spawn().map_err(|e| format!("FileNotFoundError: {e}"))?;
    match tokio::time::timeout(Duration::from_secs_f64(timeout), child.wait_with_output()).await {
        Err(_) => Err(format!("`{}` excedeu {:.0}s", cmd.join(" "), timeout)),
        Ok(Err(e)) => Err(format!("OSError: {e}")),
        Ok(Ok(out)) => {
            let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
            text.push_str(&String::from_utf8_lossy(&out.stderr));
            Ok((out.status.code().unwrap_or(-1), text.trim().to_string()))
        }
    }
}

/// '@playwright/mcp@0.0.80' → '@playwright/mcp' (spec com versão é aceita e
/// repassada intacta ao npx; aqui extraímos só o NOME pra validar no registry e
/// derivar o `name` do servidor).
fn npm_package_from_spec(spec: &str) -> String {
    // Escopados (@org/pkg@ver): só o ÚLTIMO '@' separa versão. Num escopado SEM
    // versão ('@playwright/mcp') o único '@' é o do INÍCIO do scope — base vazia
    // NÃO é separador de versão; nesse caso o spec inteiro é o nome.
    if spec.starts_with('@') {
        return match spec.rfind('@') {
            Some(i) if !spec[..i].is_empty() => spec[..i].to_string(),
            _ => spec.to_string(),
        };
    }
    spec.split('@').next().unwrap_or("").to_string()
}

/// Nome de servidor derivado do pacote npm. Escopado ganha o scope como
/// prefixo: '@playwright/mcp' → 'playwright-mcp'. '.' não é permitido em nome
/// de servidor — vira '-'.
fn server_name_from_spec(npm_name: &str) -> String {
    if let Some(rest) = npm_name.strip_prefix('@') {
        if let Some((scope, pkg)) = rest.split_once('/') {
            if !scope.is_empty() && !pkg.is_empty() {
                return format!("{scope}-{pkg}").replace('.', "-");
            }
        }
    }
    npm_name.replace('.', "-")
}

struct NpmMeta {
    version: String,
    bin: String,
}

/// Consulta o registry (`npm view`) e devolve {name, version, bin}. Erro com
/// mensagem amigável se o pacote não existir ou não expor binário.
async fn resolve_npm_package(package: &str) -> Result<NpmMeta, String> {
    let (rc, out) = run_cmd(&["npm", "view", package, "--json"], 60.0).await?;
    if rc != 0 {
        return Err(format!(
            "Pacote npm '{package}' não encontrado no registry: {}",
            truncate_chars(&out, 400)
        ));
    }
    let meta: Value = serde_json::from_str(&out).map_err(|_| {
        format!("Resposta inválida do npm para '{package}': {}", truncate_chars(&out, 400))
    })?;
    let obj = meta
        .as_object()
        .ok_or_else(|| "AttributeError: resposta do npm não é um objeto JSON".to_string())?;
    let name = obj
        .get("name")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_else(|| npm_package_from_spec(package));
    let bin_name: Option<String> = match obj.get("bin") {
        Some(Value::Object(m)) => m.keys().next().cloned(),
        Some(Value::String(_)) => Some(name.rsplit('/').next().unwrap_or("").to_string()),
        _ => None,
    };
    let version = obj.get("version").and_then(|v| v.as_str()).unwrap_or("").to_string();
    match bin_name.filter(|b| !b.is_empty()) {
        Some(bin) => Ok(NpmMeta { version, bin }),
        None => Err(format!(
            "Pacote '{name}' (v{}) não expõe 'bin' — não é executável via stdio. \
             Registre manualmente com o command correto.",
            if version.is_empty() { "?" } else { &version }
        )),
    }
}

/// Instalação assistida de servidor MCP npm: valida o pacote no registry,
/// deriva o binário e registra como stdio `npx -y <pkg>` (o spawn lazy do
/// manager resolve o download no primeiro uso) e roda /sync pra tools entrarem
/// no grafo. Idempotente: pacote já registrado com o MESMO spec só re-sincroniza.
async fn install_server(
    State(st): State<Arc<AppState>>,
    ApiJson(req): ApiJson<ServerInstallRequest>,
) -> ApiResult {
    let spec = req.package.trim().to_string();
    if spec.is_empty() || spec.contains(' ') || spec.starts_with('-') {
        return Err(http_err(
            400,
            "package inválido (ex.: '@playwright/mcp' ou '@playwright/mcp@0.0.80')",
        ));
    }
    let npm_name = npm_package_from_spec(&spec);
    let derived = req.name.as_deref().unwrap_or("").trim().to_string();
    let server_name = if derived.is_empty() { server_name_from_spec(&npm_name) } else { derived };
    if !SERVER_NAME_RE.is_match(&server_name) || server_name.contains("__") {
        return Err(http_err(
            400,
            format!("Nome de servidor derivado inválido: '{server_name}' — passe 'name' explícito"),
        ));
    }

    let _install = st.install_lock.lock().await;
    let existing = st.db_get_server(&server_name)?;
    let npx_args = vec!["-y".to_string(), spec.clone()];
    match existing {
        Some(ex) => {
            if ex.command.as_deref() == Some("npx") && parse_args(&ex.args_json) == npx_args {
                info!("MCP install: '{server_name}' já registrado com {spec} — re-sync apenas");
            } else {
                return Err(http_err(
                    409,
                    format!(
                        "Servidor '{server_name}' já existe com config diferente — DELETE antes ou registre manualmente"
                    ),
                ));
            }
        }
        None => {
            // Erro de resolução do pacote NÃO é HTTPException no original →
            // exceção não tratada (500), comportamento preservado.
            let meta = resolve_npm_package(&npm_name).await.map_err(ApiError::Internal)?;
            info!(
                "MCP install: {npm_name} v{} → bin '{}' → servidor '{server_name}' (npx {spec})",
                meta.version, meta.bin
            );
            st.db_upsert_server(&ServerRegisterRequest {
                name: server_name.clone(),
                skill_summary: req.skill_summary.clone(),
                transport: Transport::Stdio,
                command: Some("npx".to_string()),
                args: npx_args.clone(),
                env: req.env.clone(),
                url: None,
                enabled: req.enabled,
                always_on: req.always_on,
                examples: req.examples.clone(),
            })?;
            info!("MCP: servidor '{server_name}' registrado via install ({spec})");
        }
    }

    let sync = sync_server_inner(&st, &server_name).await?;
    Ok(Json(json!({
        "installed": true, "name": server_name, "package": spec,
        "transport": "stdio", "command": "npx", "args": npx_args,
        "synced": sync["synced"],
        "skill_summary": sync["skill_summary"],
        "tools": sync["tools"],
    })))
}

// ── Etapa 2: schemas + snippets (do cache, sem conexão MCP) ─────────────────

/// Tools do servidor com schema real e snippet (o 'skill.md' que vira a
/// descrição pro LLM). Lê do cache local — NÃO conecta MCP.
async fn server_tools(State(st): State<Arc<AppState>>, Path(name): Path<String>) -> ApiResult {
    if st.db_get_server(&name)?.is_none() {
        return Err(http_err(404, format!("Servidor MCP '{name}' não registrado")));
    }
    let tools: Vec<Value> = st
        .db_tools_for_server(&name)?
        .into_iter()
        .map(|t| {
            json!({
                "name": t.name,
                "graph_name": graph_name(&name, &t.name),
                "description": t.description_raw,
                "snippet": t.snippet,
                "input_schema": parse_json_or(&t.input_schema_json, "{}"),
            })
        })
        .collect();
    Ok(Json(json!({ "server": name, "tools": tools })))
}

async fn all_tools(State(st): State<Arc<AppState>>) -> ApiResult {
    let rows = st.db_all_tools()?;
    let tools: Vec<Value> = rows
        .iter()
        .map(|r| {
            json!({
                "graph_name": graph_name(&r.server, &r.name),
                "server": r.server, "name": r.name,
                "snippet": r.snippet, "description": r.description_raw,
                "server_enabled": r.server_enabled,
            })
        })
        .collect();
    Ok(Json(json!({ "total": rows.len(), "tools": tools })))
}

/// Reescreve o skill.md de uma tool. Grava no cache local E re-registra no
/// grafo (ToolsDB): a nova descrição muda o doc_hash → re-embed; as queries
/// 'learned' (uso real) são preservadas pelo memory.
async fn update_snippet(
    State(st): State<Arc<AppState>>,
    Path(graph): Path<String>,
    ApiJson(req): ApiJson<SnippetUpdateRequest>,
) -> ApiResult {
    let (server, tool) = split_graph_name(&graph).map_err(|e| http_err(400, e))?;
    let row = st.db_get_tool(&server, &tool)?.ok_or_else(|| {
        http_err(404, format!("Tool '{graph}' não encontrada (rode /servers/{server}/sync)"))
    })?;
    let snippet = req.snippet.trim().to_string();
    if snippet.is_empty() {
        return Err(http_err(400, "snippet vazio"));
    }
    st.db_update_snippet(&server, &tool, &snippet)?;
    // Preserva o core atual da tool (re-register não pode zerar o flag de
    // capability de base).
    let core = match st.db_get_server(&server)? {
        None => row.core,
        Some(srv) => row.core || srv.always_on,
    };
    let body = json!({ "name": graph, "description": snippet, "core": core });
    if let Err(e) = st.mem_post_ok("/tools/register", &body, REGISTER_TIMEOUT).await {
        // Cache local já está atualizado; o grafo ficou pra trás — erro
        // explícito pro caller retryar (o re-register é idempotente).
        return Err(http_err(
            502,
            format!("Snippet salvo localmente, mas falhou re-registrar no grafo: {e}"),
        ));
    }
    info!("MCP: snippet de '{graph}' atualizado ({} chars)", snippet.chars().count());
    Ok(Json(json!({ "updated": true, "graph_name": graph, "snippet": snippet })))
}

// ── Etapa 1: seleção de servidores via grafo de tools ───────────────────────

/// Servidores habilitados com seus summaries — insumo barato da etapa 1.
async fn skills(State(st): State<Arc<AppState>>) -> ApiResult {
    let servers: Vec<Value> = st
        .db_list_servers(true)?
        .into_iter()
        .map(|r| {
            json!({
                "name": r.name, "skill_summary": r.skill_summary,
                "transport": r.transport, "enabled": r.enabled,
            })
        })
        .collect();
    Ok(Json(json!({ "servers": servers })))
}

/// ETAPA 1 da descoberta: pergunta ao grafo de tools (memory /tools/select —
/// kNN semântico + expansão PPR no grafo de co-uso) quais tools atendem a
/// query, e mapeia as matches para os SERVIDORES donos delas. Zero conexão MCP.
async fn skills_select(
    State(st): State<Arc<AppState>>,
    ApiJson(req): ApiJson<SkillsSelectRequest>,
) -> ApiResult {
    let query = req.query.trim().to_string();
    if query.is_empty() {
        return Err(http_err(400, "query vazia"));
    }
    let body = json!({
        "query": query, "top_k": req.top_k, "min_score": req.min_score,
        "include_related": true,
    });
    let sel = st.mem_post_json("/tools/select", &body, 45.0).await.map_err(|e| {
        http_err(502, format!("Seleção no grafo de tools falhou (memory_api offline?): {e}"))
    })?;

    let matches: Vec<Value> = sel
        .get("tools")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let servers = st.db_list_servers(true)?;
    // Unhealthy (falha de conexão recente) sai da seleção — oferecer 24 tools de
    // um servidor morto distrai o modelo. EXCETO always_on: capacidade de base
    // melhor disponível e falhando do que sumida do schema.
    let enabled: Vec<&str> = servers
        .iter()
        .filter(|r| r.always_on || !is_unhealthy(&st, &r.name))
        .map(|r| r.name.as_str())
        .collect();
    let summaries: HashMap<&str, &str> = servers
        .iter()
        .map(|r| (r.name.as_str(), r.skill_summary.as_str()))
        .collect();
    let mut by_server: IndexMap<String, Vec<String>> = IndexMap::new();
    let mut matched_tools: Vec<String> = Vec::new();
    for m in &matches {
        let mname = m
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ApiError::Internal("KeyError: 'name'".into()))?
            .to_string();
        matched_tools.push(mname.clone());
        let Ok((server, _tool)) = split_graph_name(&mname) else {
            continue; // tool sem prefixo de servidor (não é MCP) — ignora
        };
        if !enabled.contains(&server.as_str()) {
            continue;
        }
        by_server.entry(server).or_default().push(mname);
    }

    let skills: Vec<Value> = by_server
        .iter()
        .map(|(s, tools)| {
            json!({
                "name": s,
                "skill_summary": summaries.get(s.as_str()).copied().unwrap_or(""),
                "tools": tools,
            })
        })
        .collect();
    Ok(Json(json!({
        "skills": skills,
        "matched_tools": matched_tools,
        "top1_score": sel.get("top1_score").cloned().unwrap_or(Value::Null),
        "low_confidence": sel.get("low_confidence").cloned().unwrap_or(Value::Null),
    })))
}

// ── Execução (proxy do orchestrator) ────────────────────────────────────────

/// Executa uma tool no servidor (spawn lazy da sessão na primeira chamada).
/// Sessão que falhar é descartada — próxima chamada respawna o subprocesso do
/// zero. Blocos textuais do resultado são joined; tool que sinaliza erro volta
/// com is_error=true (o texto continua indo ao modelo).
///
/// AVISO: o registro de servidores stdio permite executar comandos arbitrários —
/// o serviço segue a postura de rede aberta do resto do stack (0.0.0.0, sem
/// auth); exponha a porta apenas no ambiente confiável.
async fn server_call(
    State(st): State<Arc<AppState>>,
    Path(name): Path<String>,
    ApiJson(req): ApiJson<ServerCallRequest>,
) -> ApiResult {
    if !(req.timeout > 0.0 && req.timeout <= 3600.0) {
        return Err(http_err(422, "timeout: deve estar entre 0 (exclusivo) e 3600"));
    }
    let session = get_session(&st, &name, false).await?;
    let call = session.call_tool(CallToolRequestParam {
        name: req.tool.clone().into(),
        arguments: Some(req.arguments.clone()),
    });
    let outcome = tokio::time::timeout(Duration::from_secs_f64(req.timeout), call).await;
    let result = match outcome {
        Ok(Ok(r)) => r,
        Err(_) => {
            // Sem mark_unhealthy aqui: timeout de tool longa não significa servidor
            // morto — unhealthy é só pra falha de CONEXÃO (get_session).
            drop_session(&st, &name);
            return Err(http_err(
                502,
                format!("call_tool '{}' falhou no servidor '{name}': TimeoutError: ", req.tool),
            ));
        }
        Ok(Err(e)) => {
            drop_session(&st, &name);
            return Err(http_err(
                502,
                format!("call_tool '{}' falhou no servidor '{name}': McpError: {e}", req.tool),
            ));
        }
    };
    let mut text = result
        .content
        .iter()
        .filter_map(|b| match &b.raw {
            RawContent::Text(t) => Some(t.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() && !result.content.is_empty() {
        // Só blocos não-textuais (imagem etc.) ou vazio — descreve os tipos dos blocos.
        let mut kinds: Vec<String> = result
            .content
            .iter()
            .map(|b| {
                serde_json::to_value(b)
                    .ok()
                    .and_then(|v| v.get("type").and_then(|t| t.as_str()).map(String::from))
                    .unwrap_or_else(|| "desconhecido".to_string())
            })
            .collect();
        kinds.sort();
        kinds.dedup();
        text = format!(
            "(a tool devolveu {} bloco(s) não-textual(is): {})",
            result.content.len(),
            kinds.join(", ")
        );
    }
    Ok(Json(json!({ "result": text, "is_error": result.is_error.unwrap_or(false) })))
}

// ══════════════════════════════════════════════════════════════════════════════
// Application / Lifespan
// ══════════════════════════════════════════════════════════════════════════════

fn app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/status", get(health))
        .route("/servers", post(register_server).get(list_servers))
        .route("/servers/install", post(install_server))
        .route("/servers/{name}", get(get_server).delete(delete_server))
        .route("/servers/{name}/sync", post(sync_server))
        .route("/servers/{name}/test", post(test_server))
        .route("/servers/{name}/tools", get(server_tools))
        .route("/servers/{name}/call", post(server_call))
        .route("/tools", get(all_tools))
        .route("/tools/{graph_name}/snippet", put(update_snippet))
        .route("/skills", get(skills))
        .route("/skills/select", post(skills_select))
        .with_state(state)
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

async fn wait_all(events: Vec<Event>) {
    let mut handles = Vec::new();
    for ev in events {
        handles.push(tokio::spawn(async move { ev.wait().await }));
    }
    for h in handles {
        let _ = h.await;
    }
}

#[tokio::main]
async fn main() {
    let debug_on = std::env::var("AVA_DEBUG").map(|v| !v.is_empty()).unwrap_or(false);
    tracing_subscriber::fmt()
        .with_max_level(if debug_on { tracing::Level::DEBUG } else { tracing::Level::INFO })
        .with_target(false)
        .init();

    let cfg = Config::from_env();
    info!("Iniciando MCP Manager (registro + executor de servidores MCP)...");
    let db = match open_db(&cfg) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Falha ao abrir o banco {}: {e}", cfg.db_path.display());
            std::process::exit(1);
        }
    };
    let memory = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .read_timeout(Duration::from_secs_f64(REGISTER_TIMEOUT))
        .pool_max_idle_per_host(2)
        .build()
        .expect("falha ao criar cliente HTTP");
    let port = cfg.port;
    let st = Arc::new(AppState {
        cfg,
        db: Mutex::new(db),
        memory,
        rt: Mutex::new(Runtime::default()),
        install_lock: AsyncMutex::new(()),
    });

    let memory_ok = match st.mem_get_status(5.0).await {
        Ok(code) => {
            let ok = code == 200;
            if ok {
                info!("  ✓ memory_api OK ({code})");
            } else {
                info!("  ⚠ memory_api {code} — syncs vão falhar até voltar");
            }
            ok
        }
        Err(_) => {
            warn!("  ✗ memory_api OFFLINE — /sync não conseguirá gravar no grafo de tools");
            false
        }
    };
    if !memory_ok {
        // A memory sobe DEPOIS no docker-compose (corrida de startup) — o
        // watchdog re-sincroniza os servidores assim que ela responder.
        let h = tokio::spawn(memory_watchdog(st.clone(), 10.0));
        st.rt().memory_watchdog_task = Some(h);
    }
    let names: Vec<String> = st
        .db_list_servers(false)
        .map(|rows| rows.into_iter().map(|r| r.name).collect())
        .unwrap_or_default();
    info!(
        "MCP Manager pronto — {} servidor(es) registrado(s): {}",
        st.db_count_servers().unwrap_or(0),
        py_list(&names)
    );

    let listener = match tokio::net::TcpListener::bind(("0.0.0.0", port)).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Falha ao abrir 0.0.0.0:{port}: {e}");
            std::process::exit(1);
        }
    };
    info!("Escutando em http://0.0.0.0:{port}");
    if let Err(e) = axum::serve(listener, app(st.clone()))
        .with_graceful_shutdown(shutdown_signal())
        .await
    {
        tracing::error!("Servidor HTTP terminou com erro: {e}");
    }

    // ── shutdown (pós-yield do lifespan) ──
    // Watchdog antes do mais: pode estar no meio de um sync (spawn stdio).
    let wd = st.rt().memory_watchdog_task.take();
    if let Some(h) = wd {
        h.abort();
        let _ = h.await;
    }
    // Derruba sessões vivas E connects em voo (keepers ainda não "ready") —
    // depois espera cada keeper fechar o próprio subprocesso (com teto).
    let names: Vec<String> = {
        let rt = st.rt();
        let mut set: Vec<String> = rt.sessions.keys().cloned().collect();
        for k in rt.keeper_tasks.keys() {
            if !set.contains(k) {
                set.push(k.clone());
            }
        }
        set
    };
    for name in &names {
        drop_session(&st, name);
    }
    let dones: Vec<Event> = st.rt().keeper_tasks.values().map(|k| k.done.clone()).collect();
    if !dones.is_empty() {
        let _ = tokio::time::timeout(Duration::from_secs(10), wait_all(dones)).await;
        // Cede um instante ao runtime: o encerramento do filho (kill do grupo de
        // processos) roda em task própria e precisa concluir antes de sairmos.
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    info!("MCP Manager encerrado");
}
