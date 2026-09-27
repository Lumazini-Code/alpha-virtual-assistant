//! Servidor REST exposto em localhost:9001.
//!
//! Endpoints:
//!   GET  /status        -> status atual do docker
//!   POST /docker/start     -> inicia o ambiente docker
//!   POST /docker/stop       -> encerra o ambiente docker

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

// ════════════════════════════════════════════════════════════════════════
// Handlers
// ════════════════════════════════════════════════════════════════════════

async fn get_status(State(state): State<SharedState>) -> impl IntoResponse {
    let docker = state.docker.lock().await;

    let resp = StatusResponse {
        docker: DockerStatusDto {
            status: docker.status.label().to_string(),
            pid: docker.pid,
            idle_seconds: docker.last_activity.map(|t| t.elapsed().as_secs()),
        },
    };

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
