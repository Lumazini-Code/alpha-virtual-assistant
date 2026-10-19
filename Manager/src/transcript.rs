//! Supervisão do transcript (pipeline de voz) NO HOST.
//!
//! O `Modules/transcribe/transcript` é o binário C++ do pipeline de voz:
//!   PortAudio (microfone) → TEN VAD → ASR streaming (nemotron 0.6B) →
//!   speaker embedding CAM++ → POST /voice/ingest no orchestrator.
//!
//! Ele precisa rodar NO HOST (abre o microfone via PortAudio — o container
//! não tem device de áudio) e só faz sentido com o orchestrator de pé.
//! Este módulo o supervisiona como o mcp_manager.rs faz com o manager MCP:
//!
//!   - discover():      no boot, vê se o binário existe e se o processo já
//!     está vivo (pid) — reaproveita sem re-espawnar.
//!   - start():         spawn detached do executável (cwd = pasta do
//!     binário: os paths dos modelos são relativos a
//!     "../Models/..."), injetando AVA_INTERNAL_TOKEN /
//!     ORCHESTRATOR_PORT do Modules/.env quando existir.
//!   - stop():          encerra o processo (kill por pid → pkill fallback).
//!   - health_check():  tick periódico — Running sem pid vivo → Crashed;
//!     Crashed COM o container de pé → re-espawn.
//!
//! O transcript não expõe /health (é um processo de terminal): a liveness
//! aqui é o pid continuar existindo. Sem binário → NotInstalled (inerte:
//! start responde erro claro, health check não faz nada — nenhum spawn
//! fantasma).

use crate::state::{ProcStatus, SharedState, TranscriptStatus};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;

/// Após o spawn, quanto tempo esperamos antes de declarar o processo
/// "estável" — cobre morte instantânea (modelo ausente, sem microfone,
/// lib quebrada). O health checker assume daí em diante.
const STABILITY_CHECK_MS: u64 = 2000;

/// Variáveis que o transcript consome do ambiente (lidas do Modules/.env
/// pelo load_env_for_child). O main.cpp usa std::getenv para ambas.
const CHILD_ENV_KEYS: &[&str] = &["AVA_INTERNAL_TOKEN", "ORCHESTRATOR_PORT"];

/// O processo ainda está vivo? O transcript não tem /health — liveness é
/// o pid existir. `kill -0` (POSIX) / tasklist (Windows): sinal nulo só
/// testa a existência. Nota: só enxerga processos do mesmo usuário —
/// suficiente, pois o tray é quem spawn (ou o mesmo usuário rodou à mão).
pub async fn pid_alive(pid: u32) -> bool {
    #[cfg(target_os = "windows")]
    {
        Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .await
            .map(|out| {
                // A linha do processo só aparece se o filtro casou.
                String::from_utf8_lossy(&out.stdout).contains(&pid.to_string())
            })
            .unwrap_or(false)
    }
    #[cfg(not(target_os = "windows"))]
    {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .map(|s| s.success())
            .unwrap_or(false)
    }
}

/// Lê o Modules/.env do projeto e devolve APENAS as variáveis que o
/// transcript consome (token de auth interna + porta do orchestrator).
/// O tray pode ter sido lançado sem essas vars exportadas — o .env é a
/// mesma fonte que os demais serviços do host usam. Valores aqui
/// SOBRESCREVEM o ambiente herdado (configuração explícita ganha).
/// Arquivo ausente → lista vazia (o filho herda o ambiente como está).
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
        if CHILD_ENV_KEYS.contains(&k) {
            let v = v.trim().trim_matches('"').to_string();
            if !v.is_empty() {
                out.push((k.to_string(), v));
            }
        }
    }
    out
}

/// Descoberta no boot: binário existe + processo já vivo (ex.: tray
/// reiniciou sem derrubar o transcript) → Running. Binário ausente →
/// NotInstalled (inerte).
pub async fn discover(state: &SharedState) {
    let (binary_exists, pid) = {
        let t = state.transcript.lock().await;
        (t.binary.exists(), t.pid)
    };
    let alive = match pid {
        Some(p) => pid_alive(p).await,
        None => false,
    };

    let mut t = state.transcript.lock().await;
    if !binary_exists {
        if t.status != TranscriptStatus::NotInstalled {
            t.status = TranscriptStatus::NotInstalled;
        }
        return;
    }
    if alive {
        t.status = TranscriptStatus::Running;
        tracing::info!(
            "Transcript: processo já vivo (pid {:?}) — reaproveitando.",
            t.pid
        );
    }
}

/// Ativa o transcript no host. Idempotente: pid vivo (spawn externo ou
/// chamada repetida) reaproveita sem relançar. Binário ausente → erro
/// claro apontando o build (`./Modules/transcribe/make.sh`).
pub async fn start(state: &SharedState) -> Result<()> {
    let (binary, pid) = {
        let t = state.transcript.lock().await;
        (t.binary.clone(), t.pid)
    };

    if !binary.exists() {
        bail!(
            "Executável do transcript não encontrado (procurado: {} — ou defina \
             AVA_TRANSCRIPT_BIN). Compile o módulo de voz com ./Modules/transcribe/make.sh.",
            binary.display()
        );
    }

    // Já vivo (spawn externo ao tray, tray reiniciou, chamada repetida)?
    if let Some(p) = pid {
        if pid_alive(p).await {
            let mut t = state.transcript.lock().await;
            t.status = TranscriptStatus::Running;
            tracing::info!("Transcript: pid {p} já vivo — nada a fazer.");
            return Ok(());
        }
    }

    {
        let mut t = state.transcript.lock().await;
        t.status = TranscriptStatus::Starting;
    }

    // cwd = pasta do binário: o main.cpp resolve os modelos como
    // "../Models/*.onnx|gguf" — relativos a Modules/transcribe/.
    let cwd: PathBuf = binary
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| state.project_root.clone());

    let envs = load_env_for_child(&state.project_root);
    tracing::info!(
        "Transcript: spawn {} (cwd {}, env extras: {})",
        binary.display(),
        cwd.display(),
        envs.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>().join(",")
    );

    let mut cmd = Command::new(&binary);
    cmd.current_dir(&cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(false);
    for (k, v) in &envs {
        cmd.env(k, v);
    }
    let child = cmd
        .spawn()
        .with_context(|| format!("Falha ao iniciar transcript ({})", binary.display()))?;
    let pid = child.id(); // Option<u32> — None se o filho morrer já na saída
    std::mem::forget(child); // detached: sobrevive ao tray (igual docker/browser)

    let Some(pid) = pid else {
        // Filho morreu antes de nos dar um pid — certamente crash imediato.
        let mut t = state.transcript.lock().await;
        t.status = TranscriptStatus::Crashed;
        t.pid = None;
        bail!("Transcript morreu imediatamente após o spawn — checar dependências (microfone, modelos).");
    };

    // Espera curta de estabilidade: morreu nos primeiros segundos (modelo
    // ausente, sem /dev/snd...) → Crashed agora, em vez de deixar o health
    // checker descobrir no próximo tick.
    tokio::time::sleep(Duration::from_millis(STABILITY_CHECK_MS)).await;
    let stable = pid_alive(pid).await;

    {
        let mut t = state.transcript.lock().await;
        t.pid = Some(pid);
        t.status = if stable {
            TranscriptStatus::Running
        } else {
            TranscriptStatus::Crashed
        };
    }

    if !stable {
        bail!(
            "Transcript não sobreviveu aos primeiros {STABILITY_CHECK_MS}ms (pid {pid}) — \
             checar microfone/modelos; o health checker tentará re-espawnar."
        );
    }

    tracing::info!("Transcript ativo: pid {pid}.");
    Ok(())
}

/// Encerra o transcript. Kill por pid quando sabemos qual é (spawn do
/// próprio tray); fallback: pkill pelo caminho do binário — cobre o caso
/// "spawn externo ao tray / tray reiniciou e perdeu o pid".
pub async fn stop(state: &SharedState) -> Result<()> {
    let (pid, binary) = {
        let t = state.transcript.lock().await;
        (t.pid, t.binary.clone())
    };

    {
        let mut t = state.transcript.lock().await;
        t.status = TranscriptStatus::Stopping;
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

    {
        let mut t = state.transcript.lock().await;
        t.status = TranscriptStatus::Stopped;
        t.pid = None;
    }

    tracing::info!("Transcript encerrado.");
    Ok(())
}

/// Tick periódico (30s, mesmo loop do health check do docker/mcp):
///   Running sem pid vivo → Crashed;
///   Crashed → re-spawn — APENAS com o container de pé (o transcript só
///   tem propósito com o orchestrator respondendo /voice/ingest; sem
///   container é ruído de curl error no log).
pub async fn health_check(state: &SharedState) {
    let (status, binary) = {
        let t = state.transcript.lock().await;
        (t.status.clone(), t.binary.clone())
    };

    match status {
        TranscriptStatus::Running => {
            let alive = match pid_of(state).await {
                Some(p) => pid_alive(p).await,
                None => false,
            };
            if !alive {
                tracing::warn!("Transcript: processo parou de existir — marcando como CRASHED.");
                let mut t = state.transcript.lock().await;
                t.status = TranscriptStatus::Crashed;
            }
        }
        TranscriptStatus::Crashed => {
            if !binary.exists() {
                tracing::warn!("Transcript: binário sumiu — estado → NOT_INSTALLED.");
                let mut t = state.transcript.lock().await;
                t.status = TranscriptStatus::NotInstalled;
                t.pid = None;
                return;
            }
            // Re-espawn só com o container de pé.
            let docker_running = {
                let docker = state.docker.lock().await;
                docker.status == ProcStatus::Running
            };
            if !docker_running {
                tracing::debug!("Transcript crashed, mas docker não está Running — aguardando container.");
                return;
            }
            tracing::info!("Transcript: Crashed — tentando re-espawn (restart-on-fail).");
            if let Err(e) = start(state).await {
                tracing::error!("Re-espawn do transcript falhou: {e:#}");
            }
        }
        TranscriptStatus::Starting | TranscriptStatus::Stopping => {
            // start/stop em voo tem o próprio timeout — nada a ver aqui.
        }
        TranscriptStatus::Stopped | TranscriptStatus::NotInstalled => {
            // Inativo de propósito (ou binário inexistente) — não tocamos.
        }
    }
}

/// Snapshot para a API/UI: (status, pid, binário existe, processo vivo).
pub async fn transcript_snapshot(state: &SharedState) -> (TranscriptStatus, Option<u32>, bool, bool) {
    let (status, pid, binary) = {
        let t = state.transcript.lock().await;
        (t.status.clone(), t.pid, t.binary.clone())
    };
    let installed = binary.exists();
    let alive = match pid {
        Some(p) => installed && pid_alive(p).await,
        None => false,
    };
    (status, pid, installed, alive)
}

/// pid corrente (quando o próprio tray spawnou).
async fn pid_of(state: &SharedState) -> Option<u32> {
    state.transcript.lock().await.pid
}
