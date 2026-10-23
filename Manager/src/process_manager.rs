//! Gerenciamento do processo filho: docker-start.
//!
//! O docker continua sendo chamado via subprocess, exatamente como no
//! script original (docker-start.bat / docker-start.sh).

use crate::mcp_manager;
use crate::platform::NoWindow;
use crate::state::{McpStatus, ProcStatus, SharedState, TranscriptStatus};
use crate::transcript;
use anyhow::{bail, Context, Result};
use std::fs;
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
/// Nota: o Chrome de depuração NÃO entra aqui — por design ele é controle
/// manual (/browser/start), pois a janela deve ficar aberta ao usuário.
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

        match transcript::start(&state).await {
            Ok(()) => tracing::info!("Auto-init: transcript de voz ativo."),
            Err(e) => tracing::warn!("Auto-init: transcript não subiu: {e:#}"),
        }
        match mcp_manager::start(&state).await {
            Ok(()) => tracing::info!("Auto-init: MCP manager ativo."),
            Err(e) => tracing::warn!("Auto-init: MCP manager não subiu: {e:#}"),
        }
    });
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
// Browser: Chrome em modo depuração (CDP :9222) para o Playwright MCP
// ════════════════════════════════════════════════════════════════════════
//
// O mcp_manager registra o @playwright/mcp com `--cdp-endpoint
// http://localhost:9222` — ou seja, o Playwright dentro do container
// CONECTA a um Chrome do HOST. Este módulo garante esse Chrome:
//
//   1. Copia o perfil padrão do usuário (sem caches) para um
//      --user-data-dir próprio — requisito do Chrome ≥136, que recusa
//      CDP no perfil padrão
//   2. Spawn detached: o Chrome vive INDEPENDENTE do tray/docker —
//      sobrevive ao fim das ações do modelo e ao stop do docker de
//      propósito (a janela fica aberta para o usuário)

/// Dados de retorno do /browser/start.
pub struct BrowserStartInfo {
    pub cdp_url: String,
    pub pid: Option<u32>,
    pub profile_dir: PathBuf,
    pub already_running: bool,
}

/// Porta do CDP (default no state.rs; mesma porta do --cdp-endpoint no
/// mcp_manager.py). O valor efetivo vem de BrowserState.cdp_port.
const CDP_WAIT_SECS: u64 = 20;

/// Diretórios do perfil que NÃO valem copiar: caches enormes e voláteis
/// que o Chrome recria sozinho. Sem isso, a cópia arrastaria GBs
/// (File System/OPFS de sites + Service Worker sozinhos já passam de 3 GB
/// num perfil real). Logins/sessões vivem em Cookies, Local Storage e
/// IndexedDB — esses ficam.
const PROFILE_EXCLUDED_DIRS: &[&str] = &[
    "Cache",
    "Cache_Data",
    "Code Cache",
    "GPUCache",
    "GrShaderCache",
    "ShaderCache",
    "DawnWebGPUCache",
    "DawnGraphiteCache",
    "Service Worker",
    "Media Cache",
    "Crashpad",
    "CrashpadMetrics",
    "CrashpadMetrics-old",
    "component_crx_cache",
    "extensions_crx_cache",
    "OptimizationGuidePredictionModels",
    "BrowserMetrics",
    "BrowserMetrics-spdy",
    "File System",
    "Shared Dictionary",
];

/// Arquivos de lock/instância: copiá-los faria o Chrome da CÓPIA se achar
/// dono e recusar subir ("profile in use"). Symlinks idem — apontam para
/// a instância real.
const PROFILE_EXCLUDED_FILES: &[&str] = &[
    "SingletonLock",
    "SingletonCookie",
    "SingletonSocket",
    "SingletonPid",
    "lockfile",
    "LOCK",
    // Credenciais/sessões — NUNCA copiar para o perfil CDP (Plano 1 — T7).
    // Contêm cookies, senhas salvas, autofill e dados de cartão.
    "Cookies",
    "Cookies-journal",
    "Login Data",
    "Login Data-journal",
    "Web Data",
    "Web Data-journal",
];

/// Onde ficam os dados do Chrome do usuário ("User Data" — pai de
/// `Default/` e `Local State`).
fn chrome_user_data_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("LOCALAPPDATA")
            .map(|v| PathBuf::from(v).join(r"Google\Chrome\User Data"))
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config").join("google-chrome"))
    }
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

/// Cópia recursiva do perfil, pulando caches/locks/symlinks.
/// Retorna o total de bytes copiados.
fn copy_profile_dir(src: &Path, dst: &Path) -> std::io::Result<u64> {
    fs::create_dir_all(dst)?;
    let mut bytes = 0u64;

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let name = entry.file_name();
        let name_lossy = name.to_string_lossy();

        if file_type.is_symlink() {
            continue; // apontam para a instância real — não copiar
        }
        if file_type.is_dir() {
            if PROFILE_EXCLUDED_DIRS
                .iter()
                .any(|x| name_lossy.eq_ignore_ascii_case(x))
            {
                continue;
            }
            bytes += copy_profile_dir(&entry.path(), &dst.join(&name))?;
        } else {
            if PROFILE_EXCLUDED_FILES
                .iter()
                .any(|x| name_lossy.eq_ignore_ascii_case(x))
                || name_lossy.ends_with(".tmp")
            {
                continue;
            }
            fs::copy(entry.path(), dst.join(&name))?;
            bytes += entry.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }

    Ok(bytes)
}

/// Onde a cópia do perfil vive (junto ao executável do tray — na
/// instalação isso é <install>/cache/chrome-profile-cdp).
fn browser_profile_dir() -> PathBuf {
    let base = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("cache").join("chrome-profile-cdp")
}

/// CDP está no ar? Um TCP connect basta: o Chrome só abre a porta
/// quando o servidor DevTools está pronto.
async fn cdp_alive(port: u16) -> bool {
    tokio::net::TcpStream::connect(("127.9.12.22", port))
        .await
        .is_ok()
}

/// Garante um Chrome do host aberto em modo depuração (CDP :9222).
///
/// Idempotente: se o CDP já responde, sincroniza o estado e retorna
/// sem relançar nada (mesma semântica do start_docker "já estava ativo").
pub async fn start_browser(state: &SharedState) -> Result<BrowserStartInfo> {
    let port = state.browser.lock().await.cdp_port;

    if cdp_alive(port).await {
        let mut browser = state.browser.lock().await;
        if browser.status != ProcStatus::Running {
            tracing::info!("Chrome CDP já estava ativo (porta {port}) — reaproveitando.");
        }
        browser.status = ProcStatus::Running;
        return Ok(BrowserStartInfo {
            cdp_url: format!("http://127.9.12.22:{port}"),
            pid: browser.pid,
            profile_dir: browser.profile_dir.clone(),
            already_running: true,
        });
    }

    let chrome = find_chrome()
        .context("Chrome não encontrado (defina AVA_CHROME_PATH ou instale o Google Chrome)")?;

    let src = chrome_user_data_dir().context(
        "Diretório de dados do Chrome não encontrado — o Chrome já foi aberto alguma vez?",
    )?;
    let dst = browser_profile_dir();
    let dst_clone = dst.clone();
    let src_clone = src.clone();

    {
        let mut browser = state.browser.lock().await;
        browser.status = ProcStatus::Starting;
    }

    // Cópia em thread de blocking (pode ser dezenas de MB) — não trava
    // o runtime tokio.
    let copied = tokio::task::spawn_blocking(move || -> std::io::Result<u64> {
        let _ = fs::remove_dir_all(&dst_clone);
        let mut bytes = 0u64;

        // `Local State` na raiz do User Data (perfil default, prefs globais)
        let local_state = src_clone.join("Local State");
        if local_state.exists() {
            fs::create_dir_all(&dst_clone)?;
            fs::copy(&local_state, dst_clone.join("Local State"))?;
            bytes += local_state.metadata().map(|m| m.len()).unwrap_or(0);
        }

        // `Default/` — o perfil padrão de fato
        let default_dir = src_clone.join("Default");
        if !default_dir.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Perfil 'Default' não existe em {}", src_clone.display()),
            ));
        }
        bytes += copy_profile_dir(&default_dir, &dst_clone.join("Default"))?;
        Ok(bytes)
    })
    .await
    .map_err(|e| anyhow::anyhow!("spawn_blocking falhou: {e}"))?
    .map_err(|e| anyhow::anyhow!("cópia do perfil do Chrome falhou: {e}"))?;

    if copied == 0 {
        bail!("Perfil do Chrome vazio — nada foi copiado de {}", src.display());
    }

    tracing::info!(
        "Perfil copiado: {:.1} MB -> {} (caches excluídos)",
        copied as f64 / 1_048_576.0,
        dst.display()
    );

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // cookies/sessões dentro da cópia — só o dono lê
        let _ = fs::set_permissions(&dst, fs::Permissions::from_mode(0o700));
    }

    // Spawn DETACHED: não fazemos wait nem guardamos o Child (mem::forget)
    // — o Chrome tem que sobreviver ao tray e ao docker.
    let child = Command::new(&chrome)
        .arg(format!("--user-data-dir={}", dst.display()))
        .arg(format!("--remote-debugging-port={port}"))
        .args([
            "--no-first-run",
            "--no-default-browser-check",
            "--window-size=1280,900",
            "about:blank",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("Falha ao iniciar Chrome ({})", chrome.display()))?;
    let pid = child.id();
    std::mem::forget(child);

    // Espera o servidor DevTools abrir a porta
    let mut up = false;
    for _ in 0..(CDP_WAIT_SECS * 2) {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        if cdp_alive(port).await {
            up = true;
            break;
        }
    }

    {
        let mut browser = state.browser.lock().await;
        browser.status = if up {
            ProcStatus::Running
        } else {
            ProcStatus::Stopped
        };
        browser.pid = pid;
        browser.profile_dir = dst.clone();
    }

    if !up {
        bail!("Chrome não abriu o CDP na porta {port} em {CDP_WAIT_SECS}s.");
    }

    tracing::info!(
        "Chrome em modo depuração ativo: pid {pid:?}, CDP :{port}, perfil {}",
        dst.display()
    );

    Ok(BrowserStartInfo {
        cdp_url: format!("http://127.9.12.22:{port}"),
        pid,
        profile_dir: dst,
        already_running: false,
    })
}

/// Encerra o Chrome de depuração (controle MANUAL — nada disso roda
/// automaticamente; a janela deve ficar aberta para o usuário).
pub async fn stop_browser(state: &SharedState) -> Result<()> {
    let (pid, profile_dir) = {
        let browser = state.browser.lock().await;
        (browser.pid, browser.profile_dir.clone())
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
    }

    // Fallback preciso: mata QUALQUER processo que use o nosso
    // --user-data-dir. Cobre o caso "tray reiniciou e perdeu o pid" —
    // o spawn detached sobrevive ao tray, então isso é o caminho comum.
    if !killed && !profile_dir.as_os_str().is_empty() {
        let flag = format!("--user-data-dir={}", profile_dir.display());
        let result = if cfg!(target_os = "windows") {
            Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-Command",
                    &format!(
                        "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | Where-Object {{ $_.CommandLine -like '*{flag}*' }} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}"
                    ),
                ])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .no_window()
                .status()
                .await
        } else {
            Command::new("pkill")
                .arg("-f")
                .arg(flag)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .no_window()
                .status()
                .await
        };
        let matched = result.map(|s| s.success()).unwrap_or(false);
        tracing::debug!("Fallback de kill do Chrome: matched={matched}");
    }

    let mut browser = state.browser.lock().await;
    browser.status = ProcStatus::Stopped;
    browser.pid = None;

    tracing::info!("Chrome de depuração encerrado.");
    Ok(())
}

/// Snapshot do estado do browser + liveness do CDP.
pub async fn browser_status(state: &SharedState) -> (bool, Option<u32>, bool, PathBuf) {
    let (running, pid, port, profile_dir) = {
        let browser = state.browser.lock().await;
        (
            browser.status == ProcStatus::Running,
            browser.pid,
            browser.cdp_port,
            browser.profile_dir.clone(),
        )
    };
    let alive = cdp_alive(port).await;
    (running, pid, alive, profile_dir)
}
