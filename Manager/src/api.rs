//! Servidor REST exposto em localhost:9001.
//!
//! Endpoints:
//!   GET  /status        -> status atual do llama-server e do docker
//!   POST /llama/start    -> inicia (ou troca) o llama-server { "model": "...", "mmproj_used": true }
//!   POST /llama/stop      -> encerra o llama-server
//!   POST /docker/start     -> inicia o ambiente docker
//!   POST /docker/stop       -> encerra o ambiente docker
//!   GET  /models              -> lista os modelos .gguf disponíveis em ./Models

use crate::models::scan_models;
use crate::process_manager;
use crate::state::SharedState;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};

const API_PORT: u16 = 9001;

pub async fn serve(state: SharedState) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/status", get(get_status))
        .route("/models", get(get_models))
        .route("/llama/start", post(post_llama_start))
        .route("/llama/stop", post(post_llama_stop))
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
    llama: LlamaStatusDto,
    docker: DockerStatusDto,
}

#[derive(Serialize)]
struct LlamaStatusDto {
    status: String,
    pid: Option<u32>,
    model: Option<String>,
    mmproj: Option<String>,
    port: u16,
    idle_seconds: Option<u64>,
}

#[derive(Serialize)]
struct DockerStatusDto {
    status: String,
    pid: Option<u32>,
    idle_seconds: Option<u64>,
}

#[derive(Deserialize)]
struct LlamaStartRequest {
    /// Caminho absoluto ou relativo do .gguf a carregar.
    model: String,
    /// Se true, tenta usar o mmproj encontrado junto ao modelo.
    #[serde(default)]
    mmproj_used: bool,
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
    let llama = state.llama.lock().await;
    let docker = state.docker.lock().await;

    let resp = StatusResponse {
        llama: LlamaStatusDto {
            status: llama.status.label().to_string(),
            pid: llama.pid,
            model: llama.model_path.clone(),
            mmproj: llama.mmproj_path.clone(),
            port: llama.port,
            idle_seconds: llama.last_activity.map(|t| t.elapsed().as_secs()),
        },
        docker: DockerStatusDto {
            status: docker.status.label().to_string(),
            pid: docker.pid,
            idle_seconds: docker.last_activity.map(|t| t.elapsed().as_secs()),
        },
    };

    Json(resp)
}

async fn get_models(State(state): State<SharedState>) -> impl IntoResponse {
    match scan_models(&state.models_dir) {
        Ok(models) => Json(models).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(SimpleResponse {
                ok: false,
                message: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn post_llama_start(
    State(state): State<SharedState>,
    Json(req): Json<LlamaStartRequest>,
) -> impl IntoResponse {
    // Resolve o mmproj automaticamente a partir da pasta de modelos,
    // igual ao find_mmproj() do script original — só usa se mmproj_used = true.
    let mmproj = if req.mmproj_used {
        scan_models(&state.models_dir)
            .ok()
            .and_then(|models| {
                models
                    .into_iter()
                    .find(|m| m.path == req.model)
                    .and_then(|m| m.mmproj_path)
            })
    } else {
        None
    };

    match process_manager::start_llama(&state, &req.model, mmproj.as_deref()).await {
        Ok(()) => (
            StatusCode::OK,
            Json(SimpleResponse {
                ok: true,
                message: "llama-server iniciado (ou já estava ativo com este modelo).".into(),
            }),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(SimpleResponse {
                ok: false,
                message: e.to_string(),
            }),
        ),
    }
}

async fn post_llama_stop(State(state): State<SharedState>) -> impl IntoResponse {
    match process_manager::stop_llama(&state).await {
        Ok(()) => Json(SimpleResponse {
            ok: true,
            message: "llama-server encerrado.".into(),
        }),
        Err(e) => Json(SimpleResponse {
            ok: false,
            message: e.to_string(),
        }),
    }
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