// Alpha AI (AVA) — Supervisão do MCP manager NO HOST
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

use crate::platform::NoWindow;
use crate::state::{McpStatus, SharedState};
use anyhow::{bail, Context, Result};
use std::path::Path;
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

/// Variáveis que o mcp_manager consome do Modules/.env (mesma fonte dos
/// demais serviços): token de auth interna + URL do memory_api + porta.
/// PLAYWRIGHT_MCP_EXTENSION_TOKEN é repassado ao @playwright/mcp
/// (--extension): com o token no ambiente do servidor, a extensão
/// "Playwright MCP Bridge" aprova a conexão sozinha — sem diálogo
/// manual a cada chamada.
const MCP_CHILD_ENV_KEYS: &[&str] = &[
    "AVA_INTERNAL_TOKEN",
    "MEMORY_API_URL",
    "MCP_MANAGER_PORT",
    "PLAYWRIGHT_MCP_EXTENSION_TOKEN",
];

/// Lê o Modules/.env do projeto e devolve APENAS as variáveis que o
/// mcp_manager consome. O tray pode ter sido lançado sem essas vars
/// exportadas — o .env é a mesma fonte que os demais serviços do host usam.
/// Valores aqui SOBRESCREVEM o ambiente herdado. Arquivo ausente → vazio.
/// (Espelho de load_env_for_child em transcript.rs.)
fn load_env_for_child(project_root: &Path) -> Vec<(String, String)> {
    let path = project_root.join("Modules").join(".env");
    let Ok(content) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let k = k.trim();
        if MCP_CHILD_ENV_KEYS.contains(&k) {
            let v = v.trim().trim_matches('"').to_string();
            if !v.is_empty() {
                out.push((k.to_string(), v));
            }
        }
    }
    out
}

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

    // Auth interna: o mcp_manager precisa do AVA_INTERNAL_TOKEN para falar
    // com o memory_api (/tools/select etc.) — sem ele o seletor de skills
    // volta 401→502 e o orquestrador cai no fallback de liberar todas as
    // tools. O tray pode ter sido lançado sem as vars exportadas, então
    // lemos do Modules/.env (fonte única dos serviços do host).
    let child_env = load_env_for_child(&cwd);
    let injected: Vec<&str> = child_env.iter().map(|(k, _)| k.as_str()).collect();
    tracing::info!("MCP manager: env injetado do Modules/.env: {:?}", injected);

    // Redirect do stdout/stderr do filho → logs/mcp_manager.stderr.log
    // (append). O binário loga por conta própria em logs/mcp_manager.log
    // (tracing); este redirect cobre o que ESCAPA do tracing: panics antes
    // do init, prints avulsos e o caso de o arquivo de log do próprio
    // filho não abrir. Sem isso, uma morte do filho não deixava NENHUMA
    // evidência — foi o que aconteceu no incidente de 26/10: o manager
    // morreu e o stderr ia para /dev/null (Stdio::null).
    let log_dir = cwd.join("logs");
    let _ = std::fs::create_dir_all(&log_dir);
    let open_capture = || {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_dir.join("mcp_manager.stderr.log"))
    };
    let (stdout, stderr) = match (open_capture(), open_capture()) {
        (Ok(o), Ok(e)) => (Stdio::from(o), Stdio::from(e)),
        (o, e) => {
            tracing::warn!(
                "MCP manager: redirect de logs falhou (out={o:?} err={e:?}) — filho roda sem captura"
            );
            (Stdio::null(), Stdio::null())
        }
    };

    let mut cmd = Command::new(&binary);
    cmd.current_dir(&cwd)
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr)
        .no_window()
        .kill_on_drop(false);
    for (k, v) in child_env {
        cmd.env(k, v);
    }
    let child = cmd
        .spawn()
        .with_context(|| format!("Falha ao iniciar MCP manager ({})", binary.display()))?;
    let pid = child.id(); // Option<u32> — None se o filho morrer já na saída

    // Reap do filho: task bloquante no wait(). Enquanto o tray vive, o fim
    // do filho é coletado — nada de zumbi; se o TRAY morrer primeiro, o
    // filho fica órfão e o init reaps (detached: sobrevive ao tray, igual
    // docker/browser). O antigo std::mem::forget deixava um <defunct>
    // PERMANENTE por ciclo de crash/respawn — 3 acumulados em 3 minutos no
    // incidente de 26/10 (cada auto-stop criava um que ninguém nunca coletava).
    {
        let mut child = child;
        tokio::task::spawn_blocking(move || {
            let _ = child.wait();
        });
    }

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
                .no_window()
                .status()
                .await
        } else {
            Command::new("kill")
                .arg(pid.to_string())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .no_window()
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
                    .no_window()
                    .status()
                    .await;
            }
        } else {
            let _ = Command::new("pkill")
                .arg("-f")
                .arg(&binary)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .no_window()
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
