// Alpha AI (AVA) — Gerenciamento do processo filho: docker-start
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
//! Gerenciamento do processo filho: docker-start.
//!
//! O docker continua sendo chamado via subprocess, exatamente como no
//! script original (docker-start.bat / docker-start.sh).

use crate::mcp_manager;
use crate::platform::NoWindow;
use crate::state::{McpStatus, ProcStatus, SharedState, TranscriptStatus};
use crate::transcript;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Instant;
use tokio::process::Command;

// ════════════════════════════════════════════════════════════════════════
// DESCOBERTA: detecta processos que já estão rodando
// ════════════════════════════════════════════════════════════════════════

/// `--env-file` para os comandos `docker compose` chamados DIRETO pelo
/// tray (descoberta, health check).
///
/// O compose resolve a interpolação (`${CRAWL4AI_API_TOKEN:?...}`) a
/// partir do shell + `.env` da RAIZ do projeto — e `Modules/.env` (onde
/// o install.sh grava os tokens) NÃO é lido automaticamente. Sem isso,
/// `docker compose ps` falha de interpolação MESMO com o stack de pé e
/// os checks reportam estado errado.
fn compose_env_args(state: &SharedState) -> Vec<std::ffi::OsString> {
    let env_file = state.project_root.join("Modules").join(".env");
    if env_file.exists() {
        vec!["--env-file".into(), env_file.into()]
    } else {
        Vec::new()
    }
}

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
    let mut compose = Command::new("docker");
    compose.arg("compose");
    for arg in compose_env_args(state) {
        compose.arg(arg);
    }
    compose.args([
        "-f",
        compose_file.to_str().unwrap_or("docker-compose.yml"),
        "ps",
        "--status",
        "running",
        "--format",
        "{{.Name}}",
    ]);
    compose.no_window();
    let output = compose
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await;

    let mut containers_up = false;
    match output {
        Ok(out) if out.status.success() => {
            let running_containers = String::from_utf8_lossy(&out.stdout);
            let count = running_containers
                .lines()
                .filter(|l| !l.trim().is_empty())
                .count();

            if count > 0 {
                tracing::info!("Docker detectado! {} container(s) rodando.", count);
                containers_up = true;

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

    // Container já carregado (tray reiniciou sem derrubar o docker) →
    // auto-inicializa os serviços do HOST que dependem dele. Mesma
    // semântica do start_docker: quem sobe o container também sobe o
    // transcript de voz e o MCP manager, sem diferença para o usuário.
    if containers_up {
        spawn_auto_start_host_services(state);
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
    // Status "Running" (já estava de pé) → só toca o timer de inatividade
    // e retorna. UM ÚNICO lock: a versão antiga re-lockava o Mutex com o
    // guard da checagem ainda vivo → self-deadlock no tokio::sync::Mutex;
    // o handler do /docker/start travava pra sempre RETENDO o lock e o
    // tray inteiro congelava (/status, stop, tudo). "Running" stale é
    // comum (containers mortos por fora não atualizam o estado) — o
    // health_check_docker corrige no tick seguinte; aqui basta não
    // travar nunca.
    let mut docker = state.docker.lock().await;
    if docker.status == ProcStatus::Running {
        // JÁ estava rodando: apenas resetamos o timer de inatividade.
        // A TUI chama /docker/start a cada pergunta — é exatamente isso
        // que "atividade" significa. Sem este touch, o idle_watcher
        // matava o docker 45 min após o START mesmo em uso contínuo
        // (touch_docker_activity existia mas nunca era chamado).
        docker.last_activity = Some(Instant::now());
        tracing::info!("Docker já estava em execução — timer de inatividade resetado.");
        return Ok(());
    }
    if docker.status == ProcStatus::Starting {
        // `up -d` em andamento (chamada concorrente da TUI): NÃO spawne
        // um segundo compose — espere o primeiro terminar. O chamador
        // recebe Ok e o health check/start seguinte confirma de fato.
        tracing::info!("Docker já está sendo iniciado — ignorando chamada repetida.");
        return Ok(());
    }
    drop(docker);   // solta antes das seções abaixo (cada uma re-locka)

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

    // `up -d` (detached): o compose EXITA quando os containers sobem.
    // Sem o -d, `docker compose up` ANEXA e roda pra sempre.
    //
    // Esperamos a SAÍDA (em vez de spawn + mem::forget): `up -d` num
    // stack já construído leva segundos, e só assim o /docker/start
    // reporta a verdade. Antes marcávamos Running às cegas — o compose
    // podia morrer na interpolação (env ausente) e a TUI/Manager exibiam
    // "iniciado" com NENHUM container de pé. Imagens novas baixando
    // podem estourar o timeout de 15s da TUI — o aviso é best-effort e
    // o health check corrige o estado no tick seguinte.
    let compose = if cfg!(target_os = "windows") {
        Command::new(script)
            .args(["up", "-d"])
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .current_dir(script_dir)
            .no_window()
            .output()
            .await
            .context("Falha ao executar docker-start.bat")?
    } else {
        Command::new("bash")
            .arg(script)
            .args(["up", "-d"])
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .current_dir(script_dir)
            .no_window()
            .output()
            .await
            .context("Falha ao executar docker-start.sh")?
    };

    if !compose.status.success() {
        // Falha REAL do compose (env ausente, daemon fora do ar, erro de
        // build): estado volta a Stopped e o erro vai para a TUI.
        let detail = String::from_utf8_lossy(&compose.stderr);
        let tail: String = detail
            .lines()
            .filter(|l| !l.trim().is_empty())
            .rev()
            .take(3)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join(" | ");
        tracing::warn!("docker compose up falhou ({}): {tail}", compose.status);
        let mut docker = state.docker.lock().await;
        docker.status = ProcStatus::Stopped;
        docker.pid = None;
        docker.last_activity = None;
        bail!("docker compose up falhou — detalhe: {tail}");
    }

    let mut docker = state.docker.lock().await;
    docker.status = ProcStatus::Running;
    // `up -d` é detached: o processo do compose já saiu — não há um PID
    // único de "docker" para vigiar (a lógica é igual à da descoberta).
    docker.pid = None;
    docker.last_activity = Some(Instant::now());

    // Container carregado → auto-inicializa os serviços do HOST que
    // dependem dele (transcript de voz + MCP manager), em background —
    // o handler do /docker/start NÃO espera o boot deles.
    spawn_auto_start_host_services(state);

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
                .no_window()
                .status()
                .await
        } else {
            Command::new("bash")
                .arg(script)
                .args(["down"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .current_dir(script_dir)
                .no_window()
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
            .no_window()
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

    // Container derrubado de propósito → os serviços do host também saem
    // (em background — o stop deles não bloqueia a resposta do /docker/stop).
    spawn_auto_stop_host_services(state);

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
// Auto-inicialização dos serviços do HOST quando o container carrega
// ════════════════════════════════════════════════════════════════════════

/// Sobe, em background, os serviços do HOST que dependem do container:
///   - transcript de voz (mic → ASR → /voice/ingest no orchestrator);
///   - MCP manager (executável standalone no host).
///
/// Chamado sempre que o container é CARREGADO:
///   - start_docker() com sucesso;
///   - descoberta no boot com containers já rodando (tray reiniciou).
///
/// Ambos os starts são IDEMPOTENTES (reaproveitam processo vivo), então
/// chamar de novo é barato e sem efeito colateral. Nada aqui bloqueia o
/// chamador — o /docker/start responde sem esperar o boot dos serviços.
///
/// Nota: o Chrome NÃO entra aqui — por design ele é controle manual
/// (/browser/start), pois a janela deve ficar aberta ao usuário.
fn spawn_auto_start_host_services(state: &SharedState) {
    let state = state.clone();
    tokio::spawn(async move {
        // Gate: só com o container de pé — sem o orchestrator não há para
        // quem mandar embeddings (transcript) nem tools MCP funcionais.
        let docker_running = {
            let docker = state.docker.lock().await;
            docker.status == ProcStatus::Running
        };
        if !docker_running {
            tracing::debug!("Auto-init: docker não está Running — pulando serviços do host.");
            return;
        }

        // ── Voz é opt-in (AVA_VOICE, default 0) ──────────────────────────
        // O pipeline C++ de voz carrega o ASR nemotron GGUF (~473MB) + KWS +
        // CAM++ residentes no host — ~600MB de RAM. Com a flag em 0 (default)
        // o transcript NÃO sobe no auto-init (o /transcript/start manual
        // continua funcionando; e o mcp_manager sobe independente disso).
        if voice_enabled(&state).await {
            match transcript::start(&state).await {
                Ok(()) => tracing::info!("Auto-init: transcript de voz ativo."),
                Err(e) => tracing::warn!("Auto-init: transcript não subiu: {e:#}"),
            }
        } else {
            tracing::info!(
                "Auto-init: transcript PULADO (AVA_VOICE=0) — economia de ~600MB de RAM. \
                 Ligue a voz com AVA_VOICE=1 em Modules/.env (ou POST /transcript/start)."
            );
        }
        match mcp_manager::start(&state).await {
            Ok(()) => tracing::info!("Auto-init: MCP manager ativo."),
            Err(e) => tracing::warn!("Auto-init: MCP manager não subiu: {e:#}"),
        }
    });
}

/// Lê AVA_VOICE do Modules/.env (fonte única dos serviços do host — o tray
/// pode ter sido lançado sem a var exportada). Default 0 (voz desligada —
/// perfil econômico); "1"/"true"/"yes"/"on" liga.
async fn voice_enabled(state: &SharedState) -> bool {
    let path = state.project_root.join("Modules").join(".env");
    let Ok(content) = std::fs::read_to_string(&path) else {
        return false;
    };
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            if k.trim() == "AVA_VOICE" {
                let v = v.trim().trim_matches('"').to_ascii_lowercase();
                return matches!(v.as_str(), "1" | "true" | "yes" | "on");
            }
        }
    }
    false
}

/// Derruba, em background, os serviços do HOST quando o container cai —
/// o inverso do spawn_auto_start_host_services. Sem o orchestrator de pé,
/// nem o transcript (não tem para quem mandar /voice/ingest) nem o MCP
/// manager (tools sem o backend) têm propósito: de pé, só viram ruído.
///
/// Chamado sempre que o container SAI de Running:
///   - stop_docker() (manual ou idle watcher dos 45 min);
///   - health check detectando containers mortos / compose sem resposta.
///
/// Só encerra quem está de fato ativo (Running/Starting/Crashed):
/// Stopped/NotInstalled permanecem como estão — ex., não rebaixa
/// NotInstalled para Stopped (o que mentiria o "não instalado" na UI).
/// A re-subida é automática: o próximo start_docker re-invoca
/// spawn_auto_start_host_services.
fn spawn_auto_stop_host_services(state: &SharedState) {
    let state = state.clone();
    tokio::spawn(async move {
        // Transcript: para só se estava ativo (evita rebaixar NotInstalled).
        let transcript_stop = {
            let t = state.transcript.lock().await;
            matches!(
                t.status,
                TranscriptStatus::Running | TranscriptStatus::Starting | TranscriptStatus::Crashed
            )
        };
        if transcript_stop {
            match transcript::stop(&state).await {
                Ok(()) => tracing::info!("Auto-stop: transcript de voz encerrado."),
                Err(e) => tracing::warn!("Auto-stop: transcript não parou: {e:#}"),
            }
        }

        // MCP manager: mesma regra.
        let mcp_stop = {
            let m = state.mcp.lock().await;
            matches!(
                m.status,
                McpStatus::Running | McpStatus::Starting | McpStatus::Crashed
            )
        };
        if mcp_stop {
            match mcp_manager::stop(&state).await {
                Ok(()) => tracing::info!("Auto-stop: MCP manager encerrado."),
                Err(e) => tracing::warn!("Auto-stop: MCP manager não parou: {e:#}"),
            }
        }
    });
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

    let mut compose = Command::new("docker");
    compose.arg("compose");
    for arg in compose_env_args(state) {
        compose.arg(arg);
    }
    compose.args([
        "-f",
        compose_file.to_str().unwrap_or("docker-compose.yml"),
        "ps",
        "--status",
        "running",
        "--quiet",
    ]);
    // CWD explícito: SEM isto o compose herda o cwd do tray — que uma
    // reinstalação (dir apagado e recriado com o tray de pé) transforma em
    // um cwd órfão. O docker CLI chama getwd antes de ler o -f e morre com
    // "getwd: no such file or directory" 30s após CADA start — o health
    // tick interpretava como "containers caíram" e derrubava a supervisão
    // em loop (incidente de 26/10: docker "caía sozinho" sem NENHUM erro
    // nos logs do docker). O path absoluto do project_root resolve mesmo
    // que o dir tenha sido recriado (novo inode, mesmo caminho).
    compose.current_dir(state.project_root.clone());
    compose.no_window();
    let output = compose
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
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
                // Container caiu → derruba os serviços do host (em background).
                spawn_auto_stop_host_services(state);
                return false;
            }
            true
        }
        Ok(out) => {
            // compose ps FALHOU (daemon fora do ar, env de interpolação
            // ausente...): NÃO é "assume ok" — antes o estado ficava
            // "Running" mentindo para sempre. Marca Stopped; o próximo
            // /docker/start tenta de novo e agora enxerga o erro real.
            tracing::warn!(
                "docker compose ps falhou no health check ({}): {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            );
            let mut docker = state.docker.lock().await;
            docker.status = ProcStatus::Stopped;
            docker.pid = None;
            docker.last_activity = None;
            // Container sem resposta → mesmo tratamento de "caiu": derruba
            // os serviços do host (em background).
            spawn_auto_stop_host_services(state);
            false
        }
        Err(e) => {
            tracing::warn!("Não foi possível executar docker no health check: {e}");
            let mut docker = state.docker.lock().await;
            docker.status = ProcStatus::Stopped;
            docker.pid = None;
            docker.last_activity = None;
            spawn_auto_stop_host_services(state);
            false
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Browser: Chrome normal do usuário para o Playwright MCP (--extension)
// ════════════════════════════════════════════════════════════════════════
//
// O @playwright/mcp é registrado com `--extension` — ele NÃO lança browser
// nem usa CDP: conecta ao Chrome REAL do usuário via extensão "Playwright
// MCP Bridge" (instalada no perfil padrão). Este módulo só garante o
// Chrome aberto:
//
//   1. Chrome já rodando (qualquer instância, a do usuário incluída) →
//      não abre NADA — nem janela nova, nem modo depuração;
//   2. Chrome fechado → abre o Chrome NORMAL, com o perfil padrão do
//      usuário (sem --remote-debugging-port, sem --user-data-dir, sem
//      cópia de perfil);
//   3. Spawn detached: o Chrome vive INDEPENDENTE do tray/docker —
//      sobrevive ao fim das ações do modelo e ao stop do docker de
//      propósito (a janela fica aberta para o usuário).
//
// A autenticação da extensão fica por conta de
// PLAYWRIGHT_MCP_EXTENSION_TOKEN (Modules/.env → tray → mcp_manager →
// @playwright/mcp): com o token no ambiente do servidor, a extensão
// aprova a conexão sozinha, sem diálogo manual a cada chamada.

/// Dados de retorno do /browser/start.
pub struct BrowserStartInfo {
    pub pid: Option<u32>,
    pub already_running: bool,
}

/// Localiza o executável do Chrome. Override: AVA_CHROME_PATH.
fn find_chrome() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("AVA_CHROME_PATH") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }

    #[cfg(target_os = "windows")]
    {
        let mut candidates: Vec<PathBuf> = Vec::new();
        for var in ["ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"] {
            if let Some(base) = std::env::var_os(var) {
                candidates.push(
                    PathBuf::from(base).join(r"Google\Chrome\Application\chrome.exe"),
                );
            }
        }
        for c in candidates {
            if c.exists() {
                return Some(c);
            }
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        let path_env = std::env::var("PATH").unwrap_or_default();
        for dir in path_env.split(':') {
            for name in ["google-chrome", "google-chrome-stable"] {
                let p = Path::new(dir).join(name);
                if p.exists() {
                    return Some(p);
                }
            }
        }
    }

    None
}

/// O Chrome já está aberto? Vale QUALQUER instância (a do usuário
/// incluída) — com a extensão "Playwright MCP Bridge" no perfil padrão,
/// o Playwright MCP conecta nela do mesmo jeito; abrir outra janela só
/// atrapalharia. Falha aberta: detecção indisponível → reporta fechado
/// (o Chrome é single-instance, então um spawn redundante vira apenas
/// uma janela nova na instância existente).
async fn chrome_running() -> bool {
    let output = if cfg!(target_os = "windows") {
        Command::new("tasklist")
            .args(["/FI", "IMAGENAME eq chrome.exe", "/NH"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .no_window()
            .output()
            .await
    } else {
        // Linux: o processo do Google Chrome se chama "chrome";
        // macOS: "Google Chrome" (comm truncado a 15 chars — cabe).
        let name = if cfg!(target_os = "macos") {
            "Google Chrome"
        } else {
            "chrome"
        };
        Command::new("pgrep")
            .arg("-x")
            .arg(name)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .no_window()
            .output()
            .await
    };

    match output {
        Ok(out) if out.status.success() => {
            if cfg!(target_os = "windows") {
                // tasklist sai 0 mesmo sem match ("INFO: No tasks are
                // running...") — o nome no stdout é que decide.
                String::from_utf8_lossy(&out.stdout).contains("chrome.exe")
            } else {
                true
            }
        }
        _ => false,
    }
}

/// Garante o Chrome do host aberto — JANELA NORMAL, sem modo depuração.
///
/// Idempotente: se o Chrome já está rodando (qualquer instância), não
/// abre nada e apenas sincroniza o estado (mesma semântica do start_docker
/// "já estava ativo"). O Playwright MCP conecta via extensão, não via CDP.
pub async fn start_browser(state: &SharedState) -> Result<BrowserStartInfo> {
    if chrome_running().await {
        let mut browser = state.browser.lock().await;
        if browser.status != ProcStatus::Running {
            tracing::info!("Chrome já estava aberto — reaproveitando (sem nova janela).");
        }
        browser.status = ProcStatus::Running;
        return Ok(BrowserStartInfo {
            pid: browser.pid,
            already_running: true,
        });
    }

    let chrome = find_chrome()
        .context("Chrome não encontrado (defina AVA_CHROME_PATH ou instale o Google Chrome)")?;

    {
        let mut browser = state.browser.lock().await;
        browser.status = ProcStatus::Starting;
    }

    // Spawn DETACHED: não fazemos wait nem guardamos o Child (mem::forget)
    // — o Chrome tem que sobreviver ao tray e ao docker. Perfil PADRÃO do
    // usuário (é nele que vive a extensão do Playwright MCP) e NENHUMA
    // flag de depuração — só os antídotos contra wizard de primeira vez.
    let child = Command::new(&chrome)
        .args(["--no-first-run", "--no-default-browser-check"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .no_window()
        .spawn()
        .with_context(|| format!("Falha ao iniciar Chrome ({})", chrome.display()))?;
    let pid = child.id();
    std::mem::forget(child);

    {
        let mut browser = state.browser.lock().await;
        browser.status = ProcStatus::Running;
        browser.pid = pid;
    }

    tracing::info!("Chrome aberto: pid {pid:?}, perfil padrão (modo normal, sem CDP).");

    Ok(BrowserStartInfo {
        pid,
        already_running: false,
    })
}

/// Encerra APENAS o Chrome que o próprio tray abriu (pid rastreado).
///
/// Controle manual — nada disso roda automaticamente. O Chrome aberto
/// pelo USUÁRIO nunca é alvo: sem pid rastreado, o stop não mata nada.
/// Retorna true se matou a instância rastreada, false se não havia uma.
pub async fn stop_browser(state: &SharedState) -> Result<bool> {
    let pid = {
        let browser = state.browser.lock().await;
        browser.pid
    };

    let mut killed = false;
    if let Some(pid) = pid {
        let result = if cfg!(target_os = "windows") {
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
        killed = result.map(|s| s.success()).unwrap_or(false);
        tracing::debug!("Kill do Chrome rastreado (pid {pid}): {killed}");
    }

    let mut browser = state.browser.lock().await;
    browser.status = ProcStatus::Stopped;
    browser.pid = None;

    if killed {
        tracing::info!("Chrome aberto pelo tray encerrado.");
    } else {
        tracing::info!("Nenhum Chrome aberto pelo tray para encerrar.");
    }
    Ok(killed)
}

/// Snapshot do browser: o Chrome está aberto no host (qualquer instância)
/// e qual o pid rastreado (Some = instância aberta pelo próprio tray).
pub async fn browser_status(state: &SharedState) -> (bool, Option<u32>) {
    let running = chrome_running().await;
    let pid = {
        let browser = state.browser.lock().await;
        browser.pid
    };
    (running, pid)
}
