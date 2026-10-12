//! Servidor REST exposto em localhost:9001.
//!
//! Endpoints:
//!   GET  /status          -> status atual do docker
//!   POST /docker/start    -> inicia o ambiente docker
//!   POST /docker/stop     -> encerra o ambiente docker
//!   POST /browser/start   -> copia o perfil do Chrome e abre o Chrome em
//!                            modo depuração (CDP :9222) — usado pelo
//!                            orchestrator antes de tools do Playwright.
//!                            O Chrome fica ABERTO depois (processo do host,
//!                            detached) — só /browser/stop fecha.
//!   POST /browser/stop    -> encerra o Chrome de depuração (controle manual)
//!   GET  /browser/status  -> estado do Chrome + liveness do CDP

use crate::process_manager;
use crate::state::SharedState;
use axum::{
    extract::State,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::Serialize;

const API_PORT: u16 = 9001;

pub async fn serve(state: SharedState) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/status", get(get_status))
        .route("/docker/start", post(post_docker_start))
        .route("/docker/stop", post(post_docker_stop))
        .route("/browser/start", post(post_browser_start))
        .route("/browser/stop", post(post_browser_stop))
        .route("/browser/status", get(get_browser_status))
        .with_state(state);

    let addr = format!("0.0.0.0:{API_PORT}");
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
    /// URL do CDP para o --cdp-endpoint do Playwright MCP.
    cdp_url: Option<String>,
    pid: Option<u32>,
    profile_dir: String,
}

#[derive(Serialize)]
struct BrowserStatusResponse {
    running: bool,
    pid: Option<u32>,
    cdp_alive: bool,
    cdp_url: String,
    profile_dir: String,
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
                "Chrome CDP já estava ativo — reaproveitado.".into()
            } else {
                // Nota sobre a cópia do perfil (pode ser inconsistente se o
                // Chrome real estava aberto durante a cópia).
                "Chrome aberto em modo depuração (cópia do perfil). Se o Chrome \
                 real estava aberto, alguns dados da cópia podem estar \
                 inconsistentes — feche-o antes de novas sessões."
                    .into()
            },
            cdp_url: Some(info.cdp_url),
            pid: info.pid,
            profile_dir: info.profile_dir.display().to_string(),
        }),
        Err(e) => Json(BrowserStartResponse {
            ok: false,
            message: e.to_string(),
            cdp_url: None,
            pid: None,
            profile_dir: String::new(),
        }),
    }
}

async fn post_browser_stop(State(state): State<SharedState>) -> impl IntoResponse {
    match process_manager::stop_browser(&state).await {
        Ok(()) => Json(SimpleResponse {
            ok: true,
            message: "Chrome de depuração encerrado.".into(),
        }),
        Err(e) => Json(SimpleResponse {
            ok: false,
            message: e.to_string(),
        }),
    }
}

async fn get_browser_status(State(state): State<SharedState>) -> impl IntoResponse {
    let (running, pid, cdp_alive, profile_dir) = process_manager::browser_status(&state).await;

    let port = { state.browser.lock().await.cdp_port };

    Json(BrowserStatusResponse {
        running,
        pid,
        cdp_alive,
        cdp_url: format!("http://localhost:{port}"),
        profile_dir: profile_dir.display().to_string(),
    })
}
