//! Supervisão do MCP manager NO HOST.
//!
//! Plano: o mcp_manager.py (hoje dentro do container, porta 4008) será
//! portado para um EXECUTÁVEL RUST STANDALONE que roda no computador do
//! usuário — assim os servidores MCP (stdio e http) ganham acesso real à
//! máquina do usuário (npx/node/uvx do host, filesystem, Chrome de
//! depuração :9222 etc.), em vez de viver presos no container.
//!
//! Este módulo é o placeholder de ativação + supervisão no tray:
//!   - discover():      no boot, vê se o binário existe e se o manager
//!     já está de pé (health :4008) — reaproveita.
//!   - start():         spawn do executável (cwd = raiz do projeto),
//!     aguarda o /health responder.
//!   - stop():          encerra o processo (graceful → kill).
//!   - health_check():  tick periódico — Running sem /health vira Crashed;
//!     Crashed é re-espawnado (restart-on-fail).
//!
//! Enquanto o executável não existir (find_mcp_manager_binary em
//! state.rs), TUDO aqui é inerte: start responde erro claro de "não
//! instalado" e o health check não faz nada — nenhum spawn fantasma.

use crate::state::{McpStatus, SharedState};
use anyhow::{bail, Context, Result};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::process::Command;

/// Espera pelo /health após o spawn (o port Rust deve abrir rápido; 30s
/// cobre cold start com registry/DB).
const HEALTH_WAIT_SECS: u64 = 30;
/// Timeout de cada GET /health.
const HEALTH_TIMEOUT_SECS: u64 = 3;
/// Tenta o /health a cada 500ms durante o boot.
const HEALTH_POLL_MS: u64 = 500;

/// GET /health responde 200 com {"status":"ok"}? (qualquer 2xx vale —
/// o corpo só interessa ao manager; aqui basta o liveness HTTP.)
pub async fn healthy(state: &SharedState) -> bool {
    let url = { state.mcp.lock().await.health_url.clone() };
    let mut builder = reqwest::Client::builder()
        .timeout(Duration::from_secs(HEALTH_TIMEOUT_SECS));
    if let Ok(token) = std::env::var("AVA_INTERNAL_TOKEN") {
        if !token.trim().is_empty() {
            builder = builder.default_headers({
                let mut h = reqwest::header::HeaderMap::new();
                if let Ok(v) = reqwest::header::HeaderValue::from_str(&format!("Bearer {}", token.trim())) {
                    h.insert(reqwest::header::AUTHORIZATION, v);
                }
                h
            });
        }
    }
    let Ok(client) = builder.build() else { return false };
    matches!(
        client.get(&url).send().await,
        Ok(resp) if resp.status().is_success()
    )
}

/// Descoberta no boot: binário existe + manager já de pé (ex.: tray
/// reiniciou sem derrubar o processo) → Running. Binário ausente →
/// NotInstalled (inerte).
///
/// ORDEM IMPORTANTE: o guard de `state.mcp` NUNCA pode estar vivo quando
/// `healthy()` roda — ela re-locka o mesmo mutex (tokio Mutex não é
/// reentrante → deadlock na mesma task). Ler os campos primeiro, soltar
/// o guard, e só então fazer o health check.
pub async fn discover(state: &SharedState) {
    let binary_exists = { state.mcp.lock().await.binary.exists() };
    let alive = healthy(state).await;

    let mut mcp = state.mcp.lock().await;
    if !binary_exists {
        if mcp.status != McpStatus::NotInstalled {
            mcp.status = McpStatus::NotInstalled;
        }
        return;
    }
    if alive {
        mcp.status = McpStatus::Running;
        // pid desconhecido (spawn externo ao tray) — o stop via /mcp/stop
        // usa o health como "teto" e o kill por pid só quando o próprio
        // tray spawnou (pid Some).
        tracing::info!("MCP manager: já estava de pé (/health OK) — reaproveitando.");
    }
}

/// Ativa o MCP manager no host. Idempotente: Running (ou /health OK)
/// retorna sem tocar em nada. Binário ausente → erro claro (é o
/// placeholder falando: "port o executável e coloque-o na raiz do projeto").
pub async fn start(state: &SharedState) -> Result<()> {
    let binary = {
        let mcp = state.mcp.lock().await;
        mcp.binary.clone()
    };

    if !binary.exists() {
        bail!(
            "Executável do MCP manager não encontrado (procurado: {} — ou defina \
             AVA_MCP_MANAGER_BIN). O port Rust standalone ainda não foi instalado.",
            binary.display()
        );
    }

    // Já de pé (spawn externo ou chamada repetida)? Não relança.
    if healthy(state).await {
        let mut mcp = state.mcp.lock().await;
        mcp.status = McpStatus::Running;
        tracing::info!("MCP manager: /health já responde — nada a fazer.");
        return Ok(());
    }

    {
        let mut mcp = state.mcp.lock().await;
        mcp.status = McpStatus::Starting;
    }

    let cwd = state.project_root.clone();
    tracing::info!("MCP manager: spawn {} (cwd {})", binary.display(), cwd.display());

    let child = Command::new(&binary)
        .current_dir(&cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(false)
        .spawn()
        .with_context(|| format!("Falha ao iniciar MCP manager ({})", binary.display()))?;
    let pid = child.id(); // Option<u32> — None se o filho morrer já na saída
    std::mem::forget(child); // detached: sobrevive ao tray (igual docker/browser)

    // Espera o /health abrir (o port pode carregar DB/registry no boot).
    let deadline = Instant::now() + Duration::from_secs(HEALTH_WAIT_SECS);
    loop {
        if healthy(state).await {
            let mut mcp = state.mcp.lock().await;
            mcp.status = McpStatus::Running;
            mcp.pid = pid;
            tracing::info!("MCP manager ativo: pid {pid:?}, health {} OK", mcp.health_url);
            return Ok(());
        }
        if Instant::now() >= deadline {
            let mut mcp = state.mcp.lock().await;
            mcp.status = McpStatus::Crashed;
            mcp.pid = pid;
            bail!(
                "MCP manager não respondeu /health em {HEALTH_WAIT_SECS}s (pid {pid:?}) — \
                 checar log do processo; o health checker tentará re-espawnar."
            );
        }
        tokio::time::sleep(Duration::from_millis(HEALTH_POLL_MS)).await;
    }
}

/// Encerra o MCP manager. Kill por pid quando sabemos qual é (spawn do
/// próprio tray); fallback: pkill pelo caminho do binário — cobre o caso
/// "spawn externo ao tray / tray reiniciou e perdeu o pid".
pub async fn stop(state: &SharedState) -> Result<()> {
    let (pid, binary) = {
        let mcp = state.mcp.lock().await;
        (mcp.pid, mcp.binary.clone())
    };

    {
        let mut mcp = state.mcp.lock().await;
        mcp.status = McpStatus::Stopping;
    }

    if let Some(pid) = pid {
        let result = if cfg!(windows) {
            Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .await
        } else {
            Command::new("kill")
                .arg(pid.to_string())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .await
        };
        let killed = result.map(|s| s.success()).unwrap_or(false);
        tracing::debug!("Kill por pid {pid}: {killed}");
    }

    // Fallback pelo caminho do binário (não cobre binário ausente).
    if !binary.as_os_str().is_empty() {
        if cfg!(windows) {
            if let Some(name) = binary.file_name().and_then(|n| n.to_str()) {
                let _ = Command::new("taskkill")
                    .args(["/IM", name, "/F"])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .await;
            }
        } else {
            let _ = Command::new("pkill")
                .arg("-f")
                .arg(&binary)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .await;
        }
    }

    // Pergunta ao /health: se ainda responde, algo não morreu — log para
    // investigação, mas o estado encerra (o health checker re-verifica).
    let still_alive = healthy(state).await;
    if still_alive {
        tracing::warn!("MCP manager: /health ainda responde após o stop — re-verificação no próximo tick.");
    }

    {
        let mut mcp = state.mcp.lock().await;
        mcp.status = McpStatus::Stopped;
        mcp.pid = None;
    }

    tracing::info!("MCP manager encerrado.");
    Ok(())
}

/// Tick periódico (30s, mesmo loop do health check do docker):
///   Running sem /health → Crashed;
///   Crashed → re-spawn (é o "backoff" do placeholder: 1 tentativa por
///   tick; se o binário sumiu, cai em NotInstalled).
pub async fn health_check(state: &SharedState) {
    let (status, binary) = {
        let mcp = state.mcp.lock().await;
        (mcp.status.clone(), mcp.binary.clone())
    };

    match status {
        McpStatus::Running => {
            if !healthy(state).await {
                tracing::warn!("MCP manager: /health parou de responder — marcando como CRASHED.");
                let mut mcp = state.mcp.lock().await;
                mcp.status = McpStatus::Crashed;
            }
        }
        McpStatus::Crashed => {
            if !binary.exists() {
                tracing::warn!("MCP manager: binário sumiu — estado → NOT_INSTALLED.");
                let mut mcp = state.mcp.lock().await;
                mcp.status = McpStatus::NotInstalled;
                mcp.pid = None;
                return;
            }
            // Se nem /health tem e nem binário sumiu, tenta de novo (1x por tick).
            tracing::info!("MCP manager: Crashed — tentando re-espawn (restart-on-fail).");
            if let Err(e) = start(state).await {
                tracing::error!("Re-espawn do MCP manager falhou: {e:#}");
            }
        }
        McpStatus::Starting | McpStatus::Stopping => {
            // start/stop em voo tem o próprio timeout — nada a ver aqui.
        }
        McpStatus::Stopped | McpStatus::NotInstalled => {
            // Inativo de propósito (ou placeholder inerte) — não tocamos.
        }
    }
}

/// Snapshot para a API/UI: (status, pid, binário existe, /health vivo).
pub async fn mcp_snapshot(state: &SharedState) -> (McpStatus, Option<u32>, bool, bool) {
    let (status, pid, binary) = {
        let mcp = state.mcp.lock().await;
        (mcp.status.clone(), mcp.pid, mcp.binary.clone())
    };
    let installed = binary.exists();
    let healthy_now = installed && healthy(state).await;
    (status, pid, installed, healthy_now)
}
