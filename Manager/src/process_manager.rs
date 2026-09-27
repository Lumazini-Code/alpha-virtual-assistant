//! Gerenciamento do processo filho: docker-start.
//!
//! O docker continua sendo chamado via subprocess, exatamente como no
//! script original (docker-start.bat / docker-start.sh).

use crate::state::{ProcStatus, SharedState};
use anyhow::{bail, Context, Result};
use std::process::Stdio;
use std::time::Instant;
use tokio::process::Command;

// ════════════════════════════════════════════════════════════════════════
// DESCOBERTA: detecta processos que já estão rodando
// ════════════════════════════════════════════════════════════════════════

/// Descobre se o Docker já está rodando.
async fn discover_docker(state: &SharedState) {
    tracing::info!("Verificando se Docker já está em execução...");

    // Procura o docker-compose.yml perto do script docker-start
    let compose_file = state
        .docker_start_script
        .parent()
        .map(|p| p.join("docker-compose.yml"));

    let compose_file = match compose_file {
        Some(f) if f.exists() => f,
        _ => {
            tracing::debug!("docker-compose.yml não encontrado, pulando detecção do Docker.");
            return;
        }
    };

    // Roda `docker compose ps` para verificar se há containers rodando
    let output = Command::new("docker")
        .args([
            "compose",
            "-f",
            compose_file.to_str().unwrap_or("docker-compose.yml"),
            "ps",
            "--status",
            "running",
            "--format",
            "{{.Name}}",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await;

    match output {
        Ok(out) if out.status.success() => {
            let running_containers = String::from_utf8_lossy(&out.stdout);
            let count = running_containers
                .lines()
                .filter(|l| !l.trim().is_empty())
                .count();

            if count > 0 {
                tracing::info!("Docker detectado! {} container(s) rodando.", count);

                let mut docker = state.docker.lock().await;
                docker.status = ProcStatus::Running;
                docker.pid = None; // Não temos um PID único para o compose
                docker.last_activity = Some(Instant::now());
            } else {
                tracing::info!("Docker compose presente, mas nenhum container rodando.");
            }
        }
        Ok(out) => {
            tracing::debug!(
                "docker compose ps falhou: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        Err(e) => {
            tracing::debug!("Não foi possível executar docker: {e}");
        }
    }
}

/// Ponto de entrada: descobre todos os processos que já estão rodando.
///
/// Chame isso na inicialização da aplicação, antes de iniciar a GUI ou API.
/// Isso permite que o gerenciador "se conecte" a instâncias existentes,
/// útil quando a aplicação é reiniciada sem derrubar os processos.
pub async fn discover_running_processes(state: &SharedState) {
    tracing::info!("═══ Descobrindo processos em execução... ═══");

    discover_docker(state).await;

    let docker = state.docker.lock().await;

    tracing::info!("═══ Estado descoberto: docker={} ═══", docker.status.label());
}

// ════════════════════════════════════════════════════════════════════════
// Docker: start / stop (continua chamando docker-start.sh / .bat)
// ════════════════════════════════════════════════════════════════════════

pub async fn start_docker(state: &SharedState) -> Result<()> {
    {
        let docker = state.docker.lock().await;
        if docker.status == ProcStatus::Running {
            tracing::info!("Docker já está em execução, ignorando pedido.");
            return Ok(());
        }
    }

    let script = &state.docker_start_script;
    if !script.exists() {
        let mut docker = state.docker.lock().await;
        docker.status = ProcStatus::Stopped;
        bail!("Script docker-start não encontrado: {}", script.display());
    }

    {
        let mut docker = state.docker.lock().await;
        docker.status = ProcStatus::Starting;
    }

    let script_dir = script.parent().unwrap_or(script);

    let child = if cfg!(target_os = "windows") {
        Command::new(script)
            .args(["up"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .current_dir(script_dir)
            .spawn()
            .context("Falha ao iniciar docker-start.bat")?
    } else {
        Command::new("bash")
            .arg(script)
            .args(["up"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .current_dir(script_dir)
            .spawn()
            .context("Falha ao iniciar docker-start.sh")?
    };

    let pid = child.id();
    std::mem::forget(child);

    let mut docker = state.docker.lock().await;
    docker.status = ProcStatus::Running;
    docker.pid = pid;
    docker.last_activity = Some(Instant::now());

    tracing::info!("Docker iniciado.");
    Ok(())
}

pub async fn stop_docker(state: &SharedState) -> Result<()> {
    {
        let docker = state.docker.lock().await;
        if docker.status != ProcStatus::Running {
            return Ok(());
        }
    }

    {
        let mut docker = state.docker.lock().await;
        docker.status = ProcStatus::Stopping;
    }

    let script = &state.docker_start_script;
    let script_dir = script.parent().unwrap_or(script);

    // Se o script não existe, tenta usar docker compose diretamente.
    let result = if script.exists() {
        if cfg!(target_os = "windows") {
            Command::new(script)
                .args(["down"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .current_dir(script_dir)
                .status()
                .await
        } else {
            Command::new("bash")
                .arg(script)
                .args(["down"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .current_dir(script_dir)
                .status()
                .await
        }
    } else {
        // Fallback: chama docker compose diretamente.
        tracing::warn!("Script não encontrado, usando docker compose diretamente...");
        Command::new("docker")
            .args(["compose", "down"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .current_dir(script_dir)
            .status()
            .await
    };

    if let Err(e) = result {
        tracing::warn!("Falha ao rodar comando de 'down' do docker: {e}");
    }

    let mut docker = state.docker.lock().await;
    docker.status = ProcStatus::Stopped;
    docker.pid = None;
    docker.last_activity = None;

    tracing::info!("Docker encerrado.");
    Ok(())
}

#[allow(dead_code)]
pub async fn touch_docker_activity(state: &SharedState) {
    let mut docker = state.docker.lock().await;
    if docker.status == ProcStatus::Running {
        docker.last_activity = Some(Instant::now());
    }
}

// ════════════════════════════════════════════════════════════════════════
// Health check: verifica periodicamente se os processos ainda estão vivos
// ════════════════════════════════════════════════════════════════════════

/// Verifica se os containers Docker ainda estão rodando.
/// Retorna true se ainda estão vivos, false se morreram.
pub async fn health_check_docker(state: &SharedState) -> bool {
    let is_running = {
        let docker = state.docker.lock().await;
        docker.status == ProcStatus::Running
    };

    if !is_running {
        return false;
    }

    // Reusa a lógica de descoberta
    let compose_file = state
        .docker_start_script
        .parent()
        .map(|p| p.join("docker-compose.yml"));

    let compose_file = match compose_file {
        Some(f) if f.exists() => f,
        _ => return true, // Não conseguimos verificar, assume que está ok
    };

    let output = Command::new("docker")
        .args([
            "compose",
            "-f",
            compose_file.to_str().unwrap_or("docker-compose.yml"),
            "ps",
            "--status",
            "running",
            "--quiet",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await;

    match output {
        Ok(out) if out.status.success() => {
            let has_running = !String::from_utf8_lossy(&out.stdout).trim().is_empty();
            if !has_running {
                tracing::warn!("Containers Docker pararam!");
                let mut docker = state.docker.lock().await;
                docker.status = ProcStatus::Stopped;
                docker.pid = None;
                docker.last_activity = None;
                return false;
            }
            true
        }
        _ => true, // Não conseguiu verificar, assume ok
    }
}
