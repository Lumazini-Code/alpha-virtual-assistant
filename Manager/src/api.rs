// Alpha AI (AVA) — Servidor REST exposto em localhost:9001
// Copyright (C) 2026  Felipe lumazini
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//! Servidor REST exposto em localhost:9001.
//!
//! Endpoints:
//!   GET  /status          -> status atual do docker
//!   POST /docker/start    -> inicia o ambiente docker
//!   POST /docker/stop     -> encerra o ambiente docker
//!   POST /browser/start   -> abre o Chrome NORMAL do usuário (perfil
//!                            padrão, SEM modo depuração) apenas se ele
//!                            não estiver aberto — o @playwright/mcp
//!                            conecta via extensão. Processo do host,
//!                            detached: sobrevive ao tray; só
//!                            /browser/stop encerra a instância do tray.
//!   POST /browser/stop    -> encerra o Chrome aberto pelo tray (controle
//!                            manual; o Chrome do usuário não é tocado)
//!   GET  /browser/status  -> Chrome aberto no host? + pid rastreado
//!   POST /transcript/start -> inicia o transcript de voz no host (mic +
//!                            ASR + ingest no orchestrator). Também é
//!                            AUTO-iniciado quando o container carrega.
//!   POST /transcript/stop  -> encerra o transcript de voz
//!   GET  /transcript/status -> estado do transcript + liveness por pid
//!
//! ── MCP manager EMBUTIDO ──────────────────────────────────────────────────
//! O mcp_manager (antigo binário standalone :4008) é agora um serviço interno
//! (crate::mcp_server): suas rotas (/health, /servers, /skills, /tools, ...)
//! são MERGED no MESMO listener :9001 abaixo. O auth middleware do mcp fica
//! no router dele (as rotas próprias do tray continuam sem auth). Consumidores:
//!   - orchestrator.py (MCP_MANAGER_URL → http://127.9.12.22:9001)
//!   - install.sh (registro de servers MCP via :9001)
//!   - GET /mcp/status (tray): reporta "embutido" — o /mcp/start do
//!     install.sh vira no-op idempotente.

use std::sync::Arc;
use std::time::Duration;

use crate::process_manager;
use crate::state::SharedState;
use crate::transcript;
use axum::{
    body::{to_bytes, Body},
    extract::{Query, Request, State},
    http::{
        header,
        header::{CACHE_CONTROL, CONTENT_TYPE},
        HeaderName, HeaderValue, StatusCode,
    },
    response::{IntoResponse, Json, Response},
    routing::{any, get, post},
    Router,
};
use serde::{Deserialize, Serialize};

const API_PORT: u16 = 9001;

/// Host de bind da API/WebUI. Mesma resolução usada pelo serve(): env
/// (carregada de Modules/.env) → default 127.9.12.22 (loopback dedicado).
pub fn bind_host() -> String {
    std::env::var("AVA_BIND_HOST")
        .ok()
        .map(|h| h.trim().to_string())
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "127.9.12.22".to_string())
}

/// URL que a janela nativa (`ava-web`) deve carregar. 0.0.0.0/:: não são
/// endereços de destino válidos para o WebView: normaliza para loopback.
pub fn base_url() -> String {
    let host = bind_host();
    let host = match host.as_str() {
        "0.0.0.0" | "::" | "[::]" => "127.0.0.1".to_string(),
        other => other.to_string(),
    };
    format!("http://{host}:{API_PORT}")
}

/// URL da API de memória (:3000) — planos, pessoas/vozes.
pub fn memory_url() -> String {
    std::env::var("AVA_MEMORY_URL")
        .ok()
        .map(|url| url.trim().trim_end_matches('/').to_string())
        .filter(|url| !url.is_empty())
        .unwrap_or_else(|| format!("http://{}:3000", bind_host()))
}

/// URL do visualizador de grafo (:8777) — link do painel e health check.
pub fn graph_url() -> String {
    std::env::var("AVA_GRAPH_URL")
        .ok()
        .map(|url| url.trim().trim_end_matches('/').to_string())
        .filter(|url| !url.is_empty())
        .unwrap_or_else(|| format!("http://{}:8777", bind_host()))
}

pub async fn serve(
    state: SharedState,
    ui_tx: tokio::sync::mpsc::UnboundedSender<crate::UiCommand>,
) -> anyhow::Result<()> {
    // ── Env: Modules/.env → processo ─────────────────────────────────────
    // Antes, o supervisor injetava env no binário mcp_manager filho; agora
    // o MCP é embutido — então o PRÓPRIO processo carrega o Modules/.env
    // no ambiente (uma vez, no boot, BEFORE o init do mcp_server). Isso
    // alimenta: auth middleware (AVA_INTERNAL_TOKEN), MEMORY_API_URL do
    // Config e o env herdado pelos servidores MCP stdio spawnados
    // (PLAYWRIGHT_MCP_EXTENSION_TOKEN etc.). Vars já presentes no processo
    // têm prioridade (o shell do usuário pode sobrescrever).
    load_env_file(&state.project_root.join("Modules").join(".env"));

    // ── MCP manager EMBUTIDO ────────────────────────────────────────────────
    // Boot do serviço interno (abre o registry SQLite, watchdog da memory).
    // Falha de init (banco corrompido, etc.) NÃO derruba a API do tray: as
    // rotas MCP ficam fora e o log registra — o tray continua gerenciando
    // docker/transcript. (Antes: o binário standalone morria e o tray
    // re-espawnava em 30s; o equivalente aqui é o restart do tray inteiro.)
    let mcp_state: Option<Arc<crate::mcp_server::AppState>> =
        match crate::mcp_server::init(&state.project_root).await {
            Ok(st) => {
                let mut mcp = state.mcp.lock().await;
                mcp.status = crate::state::McpStatus::Running;
                Some(st)
            }
            Err(e) => {
                tracing::error!(
                    "MCP manager embutido NÃO subiu (rotas MCP fora do ar): {e:#}"
                );
                let mut mcp = state.mcp.lock().await;
                mcp.status = crate::state::McpStatus::Crashed;
                None
            }
        };
    // Guarda o handle da runtime no estado: /mcp/stop e o auto-stop do
    // container precisam dele para derrubar as sessões MCP (shutdown_all).
    if let Some(ref st) = mcp_state {
        let _ = state.mcp_server.set(st.clone());
    }
    let mcp_router = match mcp_state {
        Some(st) => crate::mcp_server::app(st),
        None => axum::Router::new(),
    };

    let app = Router::new()
        // ── WebUI local (janela ava-web) ───────────────────────────────────
        // Servida pela MESMA origem da API: sem CORS para /status,
        // /docker/*, etc. O front ainda fala com o orchestrator via
        // /orch/* (proxy abaixo), que injeta o token interno no servidor.
        .route("/", get(get_webui))
        .route("/index.html", get(get_webui))
        .route("/favicon.ico", get(get_favicon))
        .route("/stats", get(get_stats))
        .route("/graph/status", get(get_graph_status))
        .route("/docker/logs", get(get_docker_logs))
        .route("/orch", any(proxy_orchestrator))
        .route("/orch/{*path}", any(proxy_orchestrator))
        .route("/memory", any(proxy_memory))
        .route("/memory/{*path}", any(proxy_memory))
        .route("/mcpapi", any(proxy_mcp))
        .route("/mcpapi/{*path}", any(proxy_mcp))
        .route("/status", get(get_status))
        .route("/docker/start", post(post_docker_start))
        .route("/docker/stop", post(post_docker_stop))
        .route("/browser/start", post(post_browser_start))
        .route("/browser/stop", post(post_browser_stop))
        .route("/browser/status", get(get_browser_status))
        .route("/mcp/start", post(post_mcp_start))
        .route("/mcp/stop", post(post_mcp_stop))
        .route("/mcp/status", get(get_mcp_status))
        .route("/transcript/start", post(post_transcript_start))
        .route("/transcript/stop", post(post_transcript_stop))
        .route("/transcript/status", get(get_transcript_status))
        // Chamado pelo transcribe (C++) na deteção da wake word: abre a
        // janela do tray direto na tela de chat. Fire-and-forget.
        .route("/ui/chat-open", post(post_ui_chat_open))
        .with_state(state)
        .merge(mcp_router)
        // Sender da UI para o /ui/chat-open (não vai no AppState — o canal
        // nasce no main e é clonado aqui; Extension evita mudar o AppState).
        .layer(axum::Extension(ui_tx));

    let addr = format!("{}:{API_PORT}", bind_host());
    tracing::info!("API REST escutando em http://{addr}");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// Carrega `KEY=VALUE` do Modules/.env no processo (sem sobrescrever vars
/// já setadas). Parser minimalista: ignora linhas vazias/comentário,
/// reconhece `export KEY=...` e remove aspas duplas/simples.
///
/// Fonte única: install.sh grava os tokens aqui (chmod 600) — o tray
/// pre-standalone lidos no boot. Erros de leitura não derrubam a API
/// (o fail-open do auth middleware cobre: sem token, aceita a menos que
/// AVA_REQUIRE_AUTH=1).
fn load_env_file(path: &std::path::Path) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };
    let mut loaded = 0usize;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").map(str::trim).unwrap_or(line);
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty() || !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            continue;
        }
        let mut value = value.trim().to_string();
        if (value.starts_with('"') && value.ends_with('"') && value.len() >= 2)
            || (value.starts_with('\'') && value.ends_with('\'') && value.len() >= 2)
        {
            value = value[1..value.len() - 1].to_string();
        }
        // Vars process env têm prioridade (sobrescrita manual deliberada).
        if std::env::var_os(key).is_none() {
            unsafe { std::env::set_var(key, &value) };
            loaded += 1;
        }
    }
    tracing::info!("Modules/.env: {loaded} variável(eis) carregada(s) no ambiente.");
}

// ════════════════════════════════════════════════════════════════════════
// DTOs
// ════════════════════════════════════════════════════════════════════════

#[derive(Serialize)]
struct StatusResponse {
    docker: DockerStatusDto,
}

#[derive(Serialize)]
struct DockerStatusDto {
    status: String,
    pid: Option<u32>,
    idle_seconds: Option<u64>,
}

#[derive(Serialize)]
struct SimpleResponse {
    ok: bool,
    message: String,
}

#[derive(Serialize)]
struct BrowserStartResponse {
    ok: bool,
    message: String,
    pid: Option<u32>,
}

#[derive(Serialize)]
struct BrowserStatusResponse {
    /// Chrome aberto no host (QUALQUER instância, a do usuário incluída)?
    running: bool,
    /// PID rastreado — Some só se o Chrome foi aberto pelo próprio tray.
    pid: Option<u32>,
}

// ════════════════════════════════════════════════════════════════════════
// Handlers
// ════════════════════════════════════════════════════════════════════════

async fn get_status(State(state): State<SharedState>) -> impl IntoResponse {
    // try_lock: o /status é polado pela TUI e pelo health check — se o
    // Mutex do docker estiver retido (start/stop demorando, ou qualquer
    // bug que retenha o lock), responde "Ocupado" na hora em vez de
    // enfileirar. Um lock preso nunca mais pode congelar o painel de
    // status (e, com ele, o fluxo inteiro da TUI).
    let docker_dto = match state.docker.try_lock() {
        Ok(docker) => DockerStatusDto {
            status: docker.status.label().to_string(),
            pid: docker.pid,
            idle_seconds: docker.last_activity.map(|t| t.elapsed().as_secs()),
        },
        Err(_busy) => DockerStatusDto {
            status: "Ocupado".to_string(),
            pid: None,
            idle_seconds: None,
        },
    };

    let resp = StatusResponse { docker: docker_dto };

    Json(resp)
}

async fn post_docker_start(State(state): State<SharedState>) -> impl IntoResponse {
    match process_manager::start_docker(&state).await {
        Ok(()) => Json(SimpleResponse {
            ok: true,
            message: "Docker iniciado.".into(),
        }),
        Err(e) => Json(SimpleResponse {
            ok: false,
            message: e.to_string(),
        }),
    }
}

async fn post_docker_stop(State(state): State<SharedState>) -> impl IntoResponse {
    match process_manager::stop_docker(&state).await {
        Ok(()) => Json(SimpleResponse {
            ok: true,
            message: "Docker encerrado.".into(),
        }),
        Err(e) => Json(SimpleResponse {
            ok: false,
            message: e.to_string(),
        }),
    }
}

async fn post_browser_start(State(state): State<SharedState>) -> impl IntoResponse {
    match process_manager::start_browser(&state).await {
        Ok(info) => Json(BrowserStartResponse {
            ok: true,
            message: if info.already_running {
                "Chrome já estava aberto — reaproveitado.".into()
            } else {
                "Chrome aberto (modo normal — o Playwright MCP conecta via extensão).".into()
            },
            pid: info.pid,
        }),
        Err(e) => Json(BrowserStartResponse {
            ok: false,
            message: e.to_string(),
            pid: None,
        }),
    }
}

async fn post_browser_stop(State(state): State<SharedState>) -> impl IntoResponse {
    match process_manager::stop_browser(&state).await {
        Ok(killed) => Json(SimpleResponse {
            ok: true,
            message: if killed {
                "Chrome aberto pelo tray encerrado.".into()
            } else {
                "Nada a encerrar — o tray não abriu nenhum Chrome.".into()
            },
        }),
        Err(e) => Json(SimpleResponse {
            ok: false,
            message: e.to_string(),
        }),
    }
}

async fn get_browser_status(State(state): State<SharedState>) -> impl IntoResponse {
    let (running, pid) = process_manager::browser_status(&state).await;

    Json(BrowserStatusResponse { running, pid })
}

// ════════════════════════════════════════════════════════════════════════
// MCP manager — agora EMBUTIDO no tray (sem binário standalone)
// ════════════════════════════════════════════════════════════════════════
// O antigo supervisor (Manager/src/mcp_manager.rs) foi removido: o serviço
// roda dentro do processo do tray (mcp_server.rs). Os endpoints /mcp/* do
// tray existem por compatibilidade — install.sh chama /mcp/start no boot e a
// UI do tray mostra o estado. Start é no-op idempotente; stop derruba as
// sessões MCP vivas (keepers + subprocessos stdio) e o watchdog da memory.

#[derive(Serialize)]
struct McpStatusResponse {
    /// "não instalado" | "Inativo" | "Iniciando..." | "Ativo" | "Encerrando..."
    /// | "Crash — reiniciando..."
    status: String,
    pid: Option<u32>,
    /// O mcp_manager é embutido: sempre "instalado".
    installed: bool,
    /// Enquanto ESTE handler responde, o processo (que serve o /health do
    /// MCP no mesmo listener) está vivo — liveness auto-evidente.
    healthy: bool,
    binary: String,
}

async fn post_mcp_start(State(state): State<SharedState>) -> impl IntoResponse {
    let was_crashed = {
        let mcp = state.mcp.lock().await;
        mcp.status == crate::state::McpStatus::Crashed
    };
    // No-op idempotente: o serviço é o próprio processo. Se um init anterior
    // falhou (Crashed), a "re-ativação" real é o restart do tray inteiro.
    if was_crashed {
        Json(SimpleResponse {
            ok: false,
            message: "MCP manager embutido não subiu no boot — reinicie o tray para \
                      tentar de novo (restart do processo inteiro)."
                .into(),
        })
    } else {
        tracing::info!("POST /mcp/start — no-op (MCP manager embutido, sempre ativo)");
        Json(SimpleResponse {
            ok: true,
            message: "MCP manager embutido no tray — sempre ativo (nada a iniciar).".into(),
        })
    }
}

async fn post_mcp_stop(State(state): State<SharedState>) -> impl IntoResponse {
    // Encerra as sessões MCP vivas (sessões stdio + keeper tasks) e o
    // watchdog da memory — o equivalente graceful do SIGTERM que o tray
    // mandava ao binário standalone. O serviço em si permanece ativo (é o
    // próprio processo): /health e rotas continuam respondendo.
    let mcp_state = state.mcp_server.get().cloned();
    let mut mcp = state.mcp.lock().await;
    let Some(mcp_state) = mcp_state else {
        return Json(SimpleResponse {
            ok: true,
            message: "MCP manager não subiu no boot — nada a encerrar.".into(),
        });
    };
    crate::mcp_server::shutdown_all(&mcp_state).await;
    mcp.status = crate::state::McpStatus::Stopped;
    Json(SimpleResponse {
        ok: true,
        message: "Sessões MCP encerradas (serviço embutido permanece ativo).".into(),
    })
}

async fn get_mcp_status(State(state): State<SharedState>) -> impl IntoResponse {
    let mcp = state.mcp.lock().await;
    Json(McpStatusResponse {
        status: mcp.status.label().to_string(),
        pid: None,
        installed: true,
        healthy: true,
        binary: "embutido no tray (mcp_server.rs)".into(),
    })
}

// ════════════════════════════════════════════════════════════════════════
// Transcript (HOST) — pipeline de voz supervisionado pelo tray
// ════════════════════════════════════════════════════════════════════════

#[derive(Serialize)]
struct TranscriptStatusResponse {
    /// "não instalado" | "Inativo" | "Iniciando..." | "Ativo" |
    /// "Encerrando..." | "Crash — reiniciando..."
    status: String,
    pid: Option<u32>,
    /// Binário existe (build CMake do módulo de transcrição)?
    installed: bool,
    /// Processo vivo agora (liveness por pid)?
    alive: bool,
    binary: String,
}

async fn post_transcript_start(State(state): State<SharedState>) -> impl IntoResponse {
    match transcript::start(&state).await {
        Ok(()) => Json(SimpleResponse {
            ok: true,
            message: "Transcript iniciado.".into(),
        }),
        Err(e) => Json(SimpleResponse {
            ok: false,
            message: e.to_string(),
        }),
    }
}

async fn post_transcript_stop(State(state): State<SharedState>) -> impl IntoResponse {
    match transcript::stop(&state).await {
        Ok(()) => Json(SimpleResponse {
            ok: true,
            message: "Transcript encerrado.".into(),
        }),
        Err(e) => Json(SimpleResponse {
            ok: false,
            message: e.to_string(),
        }),
    }
}

async fn get_transcript_status(State(state): State<SharedState>) -> impl IntoResponse {
    let (status, pid, installed, alive) = transcript::transcript_snapshot(&state).await;
    let binary = { state.transcript.lock().await.binary.display().to_string() };

    Json(TranscriptStatusResponse {
        status: status.label().to_string(),
        pid,
        installed,
        alive,
        binary,
    })
}

// ════════════════════════════════════════════════════════════════════════
// UI — abertura automática da tela de chat na wake word
// ════════════════════════════════════════════════════════════════════════

/// Chamado pelo transcribe (C++) assim que a wake word é detectada — antes
/// mesmo da transcrição terminar. Envia `UiCommand::ShowChat` para a thread
/// da UI (eframe): abre/restore a janela já na tela de chat.
///
/// Fire-and-forget: o transcribe NÃO deve esperar por repaint de UI. Se a
/// mpsc falhar (UI morta), só logamos — o turno de voz continua normal.
async fn post_ui_chat_open(
    axum::Extension(ui_tx): axum::Extension<
        tokio::sync::mpsc::UnboundedSender<crate::UiCommand>,
    >,
) -> impl IntoResponse {
    if let Err(e) = ui_tx.send(crate::UiCommand::ShowChat) {
        tracing::warn!("ui/chat-open: falha ao enviar ShowChat (UI fechada?): {e}");
    } else {
        tracing::info!("ui/chat-open: wake word detectada — abrindo tela de chat");
    }
    Json(SimpleResponse {
        ok: true,
        message: "chat-open enviado".into(),
    })
}

// ════════════════════════════════════════════════════════════════════════
// WebUI local — página + proxy do orchestrator
// ════════════════════════════════════════════════════════════════════════

/// Serve o `Manager/index.html`. Lê do disco a cada request (editar o HTML
/// e recarregar a janela já mostra a mudança, sem recompilar); se o
/// arquivo não existir no layout de instalação, usa a cópia embutida no
/// binário.
async fn get_webui(State(state): State<SharedState>) -> Response {
    let path = state.project_root.join("Manager").join("index.html");
    let html = match tokio::fs::read_to_string(&path).await {
        Ok(html) => html,
        Err(e) => {
            tracing::debug!(
                "webui: {} indisponível ({e}) — usando index.html embutido",
                path.display()
            );
            include_str!("../index.html").to_string()
        }
    };

    let mut response = html.into_response();
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("text/html; charset=utf-8"),
    );
    // Dev local: sem cache, o reload da WebView sempre pega o arquivo atual.
    response
        .headers_mut()
        .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

/// Ícone do app — evita o 401 do fallback do router MCP no /favicon.ico.
async fn get_favicon() -> Response {
    const ICON: &[u8] = include_bytes!("../assets/icon.png");
    let mut response = Response::new(Body::from(ICON));
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("image/png"));
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("public, max-age=86400"),
    );
    response
}

/// CPU/RAM/GPU do host para o painel de status do WebUI.
async fn get_stats() -> Response {
    match tokio::task::spawn_blocking(crate::stats::collect).await {
        Ok(stats) => Json(stats).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("falha ao coletar stats: {e}") })),
        )
            .into_response(),
    }
}

/// O grafo (:8777) é um app separado no host/container. O front só precisa
/// saber se está no ar para mostrar o link; um TCP connect resolve sem
/// depender de CORS.
async fn get_graph_status() -> Response {
    let url = graph_url();
    let online = match reqwest::Url::parse(&url) {
        Ok(parsed) => {
            let host = parsed.host_str().unwrap_or("127.0.0.1");
            let port = parsed.port_or_known_default().unwrap_or(80);
            matches!(
                tokio::time::timeout(
                    Duration::from_millis(500),
                    tokio::net::TcpStream::connect((host, port)),
                )
                .await,
                Ok(Ok(_))
            )
        }
        Err(_) => false,
    };
    Json(serde_json::json!({ "online": online, "url": url })).into_response()
}

#[derive(Debug, Deserialize)]
struct LogsQuery {
    tail: Option<usize>,
}

/// Snapshot das últimas linhas do container (painel Ctrl+D do WebUI).
async fn get_docker_logs(Query(query): Query<LogsQuery>) -> Response {
    let logs = crate::docker_logs::tail(query.tail.unwrap_or(200)).await;
    Json(logs).into_response()
}

/// Proxy same-origin do front (`/orch/*`) para o orchestrator.
///
/// Evita CORS e mantém o AVA_INTERNAL_TOKEN fora do JavaScript: o
/// navegador fala só com :9001, e este handler injeta o Bearer no
/// servidor. Suporta streaming (SSE do /conversation/stream) — o corpo
/// do upstream é repassado como stream, sem buffer.
async fn proxy_orchestrator(State(state): State<SharedState>, req: Request) -> Response {
    let base = crate::chat::orchestrator_url(&state).await;
    proxy_upstream(state, req, "/orch", base).await
}

/// Proxy same-origin da API de memória (:3000): planos, pessoas/vozes.
async fn proxy_memory(State(state): State<SharedState>, req: Request) -> Response {
    proxy_upstream(state, req, "/memory", memory_url()).await
}

/// Proxy same-origin das rotas do MCP manager EMBUTIDO. O front chama
/// `/mcpapi/servers`; este handler reescreve para `/servers` no MESMO
/// listener :9001 e injeta o Bearer — as rotas MCP continuam protegidas.
async fn proxy_mcp(State(state): State<SharedState>, req: Request) -> Response {
    proxy_upstream(state, req, "/mcpapi", base_url()).await
}

async fn proxy_upstream(state: SharedState, req: Request, prefix: &str, base: String) -> Response {
    let uri = req.uri().clone();
    let suffix = uri.path().strip_prefix(prefix).unwrap_or(uri.path());
    let suffix = if suffix.is_empty() { "/" } else { suffix };
    let upstream = match uri.query() {
        Some(query) => format!("{base}{suffix}?{query}"),
        None => format!("{base}{suffix}"),
    };

    let (parts, body) = req.into_parts();
    let body = match to_bytes(body, 64 * 1024 * 1024).await {
        Ok(body) => body,
        Err(e) => return proxy_error(StatusCode::BAD_REQUEST, format!("corpo inválido: {e}")),
    };

    let method = reqwest::Method::from_bytes(parts.method.as_str().as_bytes())
        .unwrap_or(reqwest::Method::GET);
    let mut headers = reqwest::header::HeaderMap::new();
    for (name, value) in parts.headers.iter() {
        if is_hop_by_hop(name.as_str()) || name == axum::http::header::CONTENT_LENGTH {
            continue;
        }
        let (Ok(name), Ok(value)) = (
            reqwest::header::HeaderName::from_bytes(name.as_str().as_bytes()),
            reqwest::header::HeaderValue::from_bytes(value.as_bytes()),
        ) else {
            continue;
        };
        headers.append(name, value);
    }

    // O front não recebe o token interno; quem fala com o orchestrator é o
    // Manager. Se o cliente já mandou um Authorization (settings do app),
    // respeita — útil para apontar para outro orchestrator.
    if !headers.contains_key(reqwest::header::AUTHORIZATION) {
        if let Ok(token) = crate::chat::auth_header(&state).await {
            if let Ok(value) = reqwest::header::HeaderValue::from_str(&token) {
                headers.insert(reqwest::header::AUTHORIZATION, value);
            }
        }
    }

    let upstream_response = match proxy_client()
        .request(method, &upstream)
        .headers(headers)
        .body(reqwest::Body::from(body))
        .send()
        .await
    {
        Ok(response) => response,
        Err(e) => {
            tracing::warn!("proxy /orch: falha ao conectar em {upstream}: {e}");
            return proxy_error(
                StatusCode::BAD_GATEWAY,
                format!("orchestrator indisponível: {e}"),
            );
        }
    };

    let status = StatusCode::from_u16(upstream_response.status().as_u16())
        .unwrap_or(StatusCode::BAD_GATEWAY);
    let upstream_headers = upstream_response.headers().clone();
    let mut response = Response::new(Body::from_stream(upstream_response.bytes_stream()));
    *response.status_mut() = status;

    let out_headers = response.headers_mut();
    for (name, value) in upstream_headers.iter() {
        if is_hop_by_hop(name.as_str()) || name == axum::http::header::CONTENT_LENGTH {
            continue;
        }
        let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(name.as_str().as_bytes()),
            HeaderValue::from_bytes(value.as_bytes()),
        ) else {
            continue;
        };
        out_headers.append(name, value);
    }

    response
}

/// Client do proxy: connect timeout curto (orchestrator é local), sem
/// timeout total — o SSE fica aberto enquanto a janela existir.
fn proxy_client() -> &'static reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new())
    })
}

fn proxy_error(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({ "error": message.into() }))).into_response()
}

fn is_hop_by_hop(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
            | "host"
    )
}
