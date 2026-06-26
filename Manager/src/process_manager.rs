//! Gerenciamento dos processos filhos: llama-server e docker-start.
//!
//! Substitui a lógica do `llamaManager.py` (start/stop/status do llama-server)
//! diretamente em Rust. O docker continua sendo chamado via subprocess,
//! exatamente como no script original (docker-start.bat / docker-start.sh).

use crate::state::{ProcStatus, SharedState};
use anyhow::{bail, Context, Result};
use std::path::Path;
use std::process::Stdio;
use std::time::Instant;
use tokio::process::Command;
use tokio::net::TcpStream;

/// Porta fixa do llama-server, igual ao script Python original.
const LLAMA_PORT: u16 = 2001;
const LLAMA_HOST: &str = "127.0.0.1";

// ════════════════════════════════════════════════════════════════════════
// Parâmetros do llama-server (equivalentes a TEXT_PARAMS / VISION_PARAMS)
// ════════════════════════════════════════════════════════════════════════

/// Monta a lista de argumentos de linha de comando para o llama-server,
/// espelhando TEXT_PARAMS / VISION_PARAMS do script Python.
fn build_llama_args(model_path: &str, mmproj_path: Option<&str>) -> Vec<String> {
    let mut args: Vec<String> = vec!["--model".into(), model_path.into()];

    let is_vision = mmproj_path.is_some();
    if let Some(mmproj) = mmproj_path {
        args.push("--mmproj".into());
        args.push(mmproj.into());
    }

    let ctx_size = "8192";
    let (batch_size, ubatch_size, threads_http, parallel) = if is_vision {
        ("1024", "256", "2", "1")
    } else {
        ("2048", "512", "4", "1")
    };

    args.extend(str_pairs(&[
        ("--ctx-size", ctx_size),
        ("--batch-size", batch_size),
        ("--ubatch-size", ubatch_size),
        ("--fit", "off"),
        ("--gpu-layers", "999"),
        ("--split-mode", "layer"),
        ("--cache-type-k", "q4_0"),
        ("--cache-type-v", "q4_0"),
        ("--threads", "4"),
        ("--threads-batch", "4"),
        ("--threads-http", threads_http),
        ("--parallel", parallel),
        ("--prio", "2"),
        ("--poll", "50"),
        ("--temp", if is_vision { "0.70" } else { "0.80" }),
        ("--top-k", "40"),
        ("--top-p", "0.95"),
        ("--min-p", "0.05"),
        ("--repeat-penalty", "1.05"),
        ("--host", "0.0.0.0"),
        ("--port", &LLAMA_PORT.to_string()),
    ]));

    args.push("--flash-attn".into());
    args.push("on".into());
    args.push("--cont-batching".into());
    args.push("--cache-prompt".into());
    args.push("--mmap".into());
    if is_vision {
        args.push("--mmproj-offload".into());
        args.push("--image-max-tokens".into());
        args.push("1024".into());
    }

    args
}

fn str_pairs(pairs: &[(&str, &str)]) -> Vec<String> {
    pairs
        .iter()
        .flat_map(|(k, v)| vec![k.to_string(), v.to_string()])
        .collect()
}

// ════════════════════════════════════════════════════════════════════════
// DESCOBERTA: detecta processos que já estão rodando
// ════════════════════════════════════════════════════════════════════════

/// Verifica se algo está escutando na porta do llama-server.
async fn is_port_listening(host: &str, port: u16) -> bool {
    let addr = format!("{host}:{port}");
    TcpStream::connect(&addr).await.is_ok()
}

/// Tenta obter informações do modelo carregado via API do llama-server.
/// Retorna o caminho do modelo se conseguir, ou None se falhar.
async fn query_llama_model_info() -> Option<String> {
    let url = format!("http://{LLAMA_HOST}:{LLAMA_PORT}/props");
    
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .ok()?;

    let resp = client.get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }

    let body: serde_json::Value = resp.json().await.ok()?;
    
    // A API do llama-server retorna algo como:
    // { "model": "/path/to/model.gguf", "mmproj": "/path/to/mmproj.gguf", ... }
    let model_path = body.get("model")?.as_str()?.to_string();
    
    // Tenta pegar mmproj também (pode não existir)
    let _mmproj = body.get("mmproj").and_then(|v| v.as_str()).map(|s| s.to_string());

    Some(model_path)
}

/// Tenta descobrir o PID de um processo pelo nome (multiplataforma).
/// Retorna o primeiro PID encontrado ou None.
async fn find_pid_by_name(name: &str) -> Option<u32> {
    let output = if cfg!(target_os = "windows") {
        Command::new("tasklist")
            .args(["/FI", &format!("IMAGENAME eq {name}"), "/FO", "CSV", "/NH"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .await
            .ok()?
    } else {
        Command::new("pgrep")
            .args(["-x", name])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .await
            .ok()?
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    
    if cfg!(target_os = "windows") {
        // Formato CSV: "nome.exe","PID","..."
        stdout
            .lines()
            .next()
            .and_then(|line| {
                let parts: Vec<&str> = line.split(',').collect();
                parts.get(1)?.trim_matches('"').parse().ok()
            })
    } else {
        // pgrep retorna apenas o PID (um por linha)
        stdout.lines().next()?.trim().parse().ok()
    }
}

/// Descobre se o llama-server já está rodando e atualiza o estado.
/// Descobre se o llama-server já está rodando e atualiza o estado.
async fn discover_llama_server(state: &SharedState) {
    tracing::info!("Verificando se llama-server já está em execução...");

    if !is_port_listening(LLAMA_HOST, LLAMA_PORT).await {
        tracing::info!("Porta {} não está em uso, llama-server não está rodando.", LLAMA_PORT);
        return;
    }

    tracing::info!("Porta {} está em uso, tentando identificar o modelo...", LLAMA_PORT);

    // Tenta obter info do modelo via API
    let model_path = query_llama_model_info().await;
    
    // ✅ CORRIGIDO: evita closure assíncrono usando um `if` direto
    let pid = find_pid_by_name("llama-server").await;
    let pid = if pid.is_some() {
        pid
    } else if cfg!(windows) {
        find_pid_by_name("llama-server.exe").await
    } else {
        None
    };

    let mut llama = state.llama.lock().await;
    llama.status = ProcStatus::Running;
    llama.pid = pid;
    llama.model_path = model_path;
    llama.port = LLAMA_PORT;
    llama.last_activity = Some(Instant::now());

    if let Some(ref model) = llama.model_path {
        tracing::info!(
            "llama-server detectado! PID: {:?}, Modelo: {}",
            llama.pid,
            model
        );
    } else {
        tracing::info!(
            "llama-server detectado na porta {} (não foi possível identificar o modelo)",
            LLAMA_PORT
        );
    }
}

/// Descobre se o Docker (ambiente vulkan) já está rodando.
async fn discover_docker(state: &SharedState) {
    tracing::info!("Verificando se Docker já está em execução...");

    // Procura o docker-compose.yml perto do script docker-start
    let compose_file = state.docker_start_script.parent()
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
            "-f", compose_file.to_str().unwrap_or("docker-compose.yml"),
            "--profile", "vulkan",
            "ps",
            "--status", "running",
            "--format", "{{.Name}}",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await;

    match output {
        Ok(out) if out.status.success() => {
            let running_containers = String::from_utf8_lossy(&out.stdout);
            let count = running_containers.lines().filter(|l| !l.trim().is_empty()).count();
            
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

/// ✅ PONTO DE ENTRADA: Descobre todos os processos que já estão rodando.
/// 
/// Chame isso na inicialização da aplicação, antes de iniciar a GUI ou API.
/// Isso permite que o gerenciador "se conecte" a instâncias existentes,
/// útil quando a aplicação é reiniciada sem derrubar os processos.
pub async fn discover_running_processes(state: &SharedState) {
    tracing::info!("═══ Descobrindo processos em execução... ═══");
    
    discover_llama_server(state).await;
    discover_docker(state).await;
    
    let llama = state.llama.lock().await;
    let docker = state.docker.lock().await;
    
    tracing::info!(
        "═══ Estado descoberto: llama-server={}, docker={} ═══",
        llama.status.label(),
        docker.status.label()
    );
}

// ════════════════════════════════════════════════════════════════════════
// llama-server: start / stop / swap
// ════════════════════════════════════════════════════════════════════════

/// Inicia o llama-server para o modelo informado.
pub async fn start_llama(state: &SharedState, model_path: &str, mmproj_path: Option<&str>) -> Result<()> {
    {
        let llama = state.llama.lock().await;
        if llama.status == ProcStatus::Running {
            if llama.model_path.as_deref() == Some(model_path) {
                tracing::info!("llama-server já está rodando com este modelo, ignorando pedido.");
                return Ok(());
            }
        }
    }

    stop_llama(state).await.ok();

    if !Path::new(model_path).exists() {
        bail!("Modelo não encontrado: {model_path}");
    }

    {
        let mut llama = state.llama.lock().await;
        llama.status = ProcStatus::Starting;
    }

    let args = build_llama_args(model_path, mmproj_path);

    tracing::info!(
        "Iniciando llama-server: {} {}",
        state.llama_server_bin.display(),
        args.join(" ")
    );

    let child = Command::new(&state.llama_server_bin)
        .args(&args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(false)
        .spawn()
        .context("Falha ao iniciar llama-server. Verifique o caminho do binário.")?;

    let pid = child.id();
    std::mem::forget(child);

    let mut llama = state.llama.lock().await;
    llama.status = ProcStatus::Running;
    llama.pid = pid;
    llama.model_path = Some(model_path.to_string());
    llama.mmproj_path = mmproj_path.map(|s| s.to_string());
    llama.port = LLAMA_PORT;
    llama.last_activity = Some(Instant::now());

    Ok(())
}

/// Encerra o llama-server, se estiver rodando.
pub async fn stop_llama(state: &SharedState) -> Result<()> {
    let pid = {
        let mut llama = state.llama.lock().await;
        if llama.status != ProcStatus::Running {
            return Ok(());
        }
        llama.status = ProcStatus::Stopping;
        llama.pid.take()
    };

    if let Some(pid) = pid {
        kill_process(pid).await?;
    } else {
        // ✅ Se não temos PID (processo descoberto, não iniciado por nós),
        // tentamos matar pelo nome
        kill_process_by_name("llama-server").await;
        #[cfg(windows)]
        kill_process_by_name("llama-server.exe").await;
    }

    let mut llama = state.llama.lock().await;
    llama.status = ProcStatus::Stopped;
    llama.pid = None;
    llama.model_path = None;
    llama.mmproj_path = None;
    llama.last_activity = None;

    tracing::info!("llama-server encerrado.");
    Ok(())
}

/// Atualiza o timestamp de última atividade do llama-server.
#[allow(dead_code)]
pub async fn touch_llama_activity(state: &SharedState) {
    let mut llama = state.llama.lock().await;
    if llama.status == ProcStatus::Running {
        llama.last_activity = Some(Instant::now());
    }
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
            .args(["--profile", "vulkan", "up"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .current_dir(script_dir)
            .spawn()
            .context("Falha ao iniciar docker-start.bat")?
    } else {
        Command::new("bash")
            .arg(script)
            .args(["--profile", "vulkan", "up"])
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

    tracing::info!("Docker iniciado (perfil vulkan).");
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

    // Usa o caminho do estado para o down (funciona mesmo se não fomos nós quem iniciou)
    let script = &state.docker_start_script;
    let script_dir = script.parent().unwrap_or(script);

    // Se o script não existe, tenta usar docker compose diretamente
    let result = if script.exists() {
        if cfg!(target_os = "windows") {
            Command::new(script)
                .args(["--profile", "vulkan", "down"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .current_dir(script_dir)
                .status()
                .await
        } else {
            Command::new("bash")
                .arg(script)
                .args(["--profile", "vulkan", "down"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .current_dir(script_dir)
                .status()
                .await
        }
    } else {
        // ✅ Fallback: chama docker compose diretamente
        tracing::warn!("Script não encontrado, usando docker compose diretamente...");
        Command::new("docker")
            .args([
                "compose",
                "--profile", "vulkan",
                "down",
            ])
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
// Util: matar processo por PID ou nome, multiplataforma
// ════════════════════════════════════════════════════════════════════════

#[cfg(unix)]
async fn kill_process(pid: u32) -> Result<()> {
    use nix::sys::signal::{kill, Signal};
    use nix::unistd::Pid;

    let nix_pid = Pid::from_raw(pid as i32);
    let _ = kill(nix_pid, Signal::SIGTERM);

    for _ in 0..10 {
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        if kill(nix_pid, None).is_err() {
            return Ok(());
        }
    }

    tracing::warn!("Encerramento gracioso falhou, forçando kill -9 no PID {pid}");
    let _ = kill(nix_pid, Signal::SIGKILL);
    Ok(())
}

#[cfg(unix)]
async fn kill_process_by_name(name: &str) {
    let _ = Command::new("pkill")
        .args(["-x", name])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await;
}

#[cfg(windows)]
async fn kill_process(pid: u32) -> Result<()> {
    let _ = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await;
    Ok(())
}

#[cfg(windows)]
async fn kill_process_by_name(name: &str) {
    let _ = Command::new("taskkill")
        .args(["/IM", name, "/T", "/F"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await;
}

// ════════════════════════════════════════════════════════════════════════
// Health check: verifica periodicamente se os processos ainda estão vivos
// ════════════════════════════════════════════════════════════════════════

/// Verifica se o llama-server ainda está respondendo.
/// Se não estiver, atualiza o estado para Stopped.
/// Retorna true se ainda está vivo, false se morreu.
pub async fn health_check_llama(state: &SharedState) -> bool {
    let is_running = {
        let llama = state.llama.lock().await;
        llama.status == ProcStatus::Running
    };

    if !is_running {
        return false;
    }

    // Verifica se a porta ainda está aberta
    if !is_port_listening(LLAMA_HOST, LLAMA_PORT).await {
        tracing::warn!("llama-server parou de responder (porta fechada)!");
        let mut llama = state.llama.lock().await;
        llama.status = ProcStatus::Stopped;
        llama.pid = None;
        llama.model_path = None;
        llama.last_activity = None;
        return false;
    }

    true
}

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
    let compose_file = state.docker_start_script.parent()
        .map(|p| p.join("docker-compose.yml"));
    
    let compose_file = match compose_file {
        Some(f) if f.exists() => f,
        _ => return true, // Não conseguimos verificar, assume que está ok
    };

    let output = Command::new("docker")
        .args([
            "compose",
            "-f", compose_file.to_str().unwrap_or("docker-compose.yml"),
            "--profile", "vulkan",
            "ps",
            "--status", "running",
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