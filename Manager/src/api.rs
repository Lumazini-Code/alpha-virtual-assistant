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

use crate::mcp_manager;
use crate::process_manager;
use crate::state::SharedState;
use crate::transcript;
use axum::{
    extract::State,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::Serialize;

const API_PORT: u16 = 9001;

pub async fn serve(
    state: SharedState,
    ui_tx: tokio::sync::mpsc::UnboundedSender<crate::UiCommand>,
) -> anyhow::Result<()> {
    let app = Router::new()
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
        // Sender da UI para o /ui/chat-open (não vai no AppState — o canal
        // nasce no main e é clonado aqui; Extension evita mudar o AppState).
        .layer(axum::Extension(ui_tx));

    let bind_host = std::env::var("AVA_BIND_HOST").unwrap_or_else(|_| "127.9.12.22".to_string());
    let addr = format!("{bind_host}:{API_PORT}");
    tracing::info!("API REST escutando em http://{addr}");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
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
// MCP manager (HOST) — ativação + supervisão do executável standalone
// ════════════════════════════════════════════════════════════════════════

#[derive(Serialize)]
struct McpStatusResponse {
    /// "não instalado" | "Inativo" | "Iniciando..." | "Ativo" | "Encerrando..."
    /// | "Crash — reiniciando..."
    status: String,
    pid: Option<u32>,
    /// Executável encontrado na raiz do projeto (ou AVA_MCP_MANAGER_BIN)?
    installed: bool,
    /// GET /health (:4008) respondeu 2xx agora?
    healthy: bool,
    binary: String,
}

async fn post_mcp_start(State(state): State<SharedState>) -> impl IntoResponse {
    match mcp_manager::start(&state).await {
        Ok(()) => Json(SimpleResponse {
            ok: true,
            message: "MCP manager iniciado.".into(),
        }),
        Err(e) => Json(SimpleResponse {
            ok: false,
            message: e.to_string(),
        }),
    }
}

async fn post_mcp_stop(State(state): State<SharedState>) -> impl IntoResponse {
    match mcp_manager::stop(&state).await {
        Ok(()) => Json(SimpleResponse {
            ok: true,
            message: "MCP manager encerrado.".into(),
        }),
        Err(e) => Json(SimpleResponse {
            ok: false,
            message: e.to_string(),
        }),
    }
}

async fn get_mcp_status(State(state): State<SharedState>) -> impl IntoResponse {
    let (status, pid, installed, healthy) = mcp_manager::mcp_snapshot(&state).await;
    let binary = { state.mcp.lock().await.binary.display().to_string() };

    Json(McpStatusResponse {
        status: status.label().to_string(),
        pid,
        installed,
        healthy,
        binary,
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
