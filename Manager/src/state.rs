//! Estado compartilhado da aplicação.
//!
//! Esse módulo guarda tudo que precisa ser visto tanto pela GUI (egui)
//! quanto pelo servidor REST (axum) quanto pelos watchers de inatividade.
//! Como GUI e servidor rodam em threads/tasks diferentes, tudo é protegido
//! por `Mutex` dentro de um `Arc` para podermos clonar referências livremente.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex as TokioMutex;

/// Status de um processo gerenciado (docker).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcStatus {
    Stopped,
    Running,
    /// Started ou Stopped fica nesse meio tempo (subindo/derrubando)
    Starting,
    Stopping,
}

impl ProcStatus {
    pub fn label(&self) -> &'static str {
        match self {
            ProcStatus::Stopped => "Inativo",
            ProcStatus::Running => "Ativo",
            ProcStatus::Starting => "Iniciando...",
            ProcStatus::Stopping => "Encerrando...",
        }
    }
}

/// Informações sobre o container Docker (ambiente vulkan/etc).
#[derive(Debug, Clone)]
pub struct DockerState {
    pub status: ProcStatus,
    pub pid: Option<u32>,
    pub last_activity: Option<Instant>,
}

impl Default for DockerState {
    fn default() -> Self {
        Self {
            status: ProcStatus::Stopped,
            pid: None,
            last_activity: None,
        }
    }
}

/// Porta padrão do CDP — mesma usada pelo @playwright/mcp no
/// mcp_manager.py (`--cdp-endpoint http://localhost:9222`).
pub const CDP_DEFAULT_PORT: u16 = 9222;

/// Informações sobre o Chrome em modo depuração (CDP :9222).
///
/// O Chrome é um processo DO HOST, spawnado detached — sobrevive ao
/// stop do docker e ao timer de inatividade de propósito: a intenção
/// é que a janela fique aberta para o usuário depois que o modelo
/// terminar as ações (o Playwright MCP conecta via --cdp-endpoint).
#[derive(Debug, Clone)]
pub struct BrowserState {
    pub status: ProcStatus,
    pub pid: Option<u32>,
    pub cdp_port: u16,
    /// Cópia do perfil padrão do usuário usada como --user-data-dir.
    pub profile_dir: PathBuf,
}

impl Default for BrowserState {
    fn default() -> Self {
        Self {
            status: ProcStatus::Stopped,
            pid: None,
            cdp_port: CDP_DEFAULT_PORT,
            profile_dir: PathBuf::new(),
        }
    }
}

/// Porta padrão do MCP manager no host (a mesma que ele serve dentro do
/// container hoje — :4008).
pub const MCP_MANAGER_DEFAULT_PORT: u16 = 4008;

/// Status do MCP manager. Difere do ProcStatus por ter `NotInstalled`
/// (binário do executável Rust ainda não existe — placeholder inerte) e
/// `Crashed` (estava Running e o health check o perdeu).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpStatus {
    NotInstalled,
    Stopped,
    Starting,
    Running,
    Stopping,
    Crashed,
}

impl McpStatus {
    pub fn label(&self) -> &'static str {
        match self {
            McpStatus::NotInstalled => "não instalado",
            McpStatus::Stopped => "Inativo",
            McpStatus::Starting => "Iniciando...",
            McpStatus::Running => "Ativo",
            McpStatus::Stopping => "Encerrando...",
            McpStatus::Crashed => "Crash — reiniciando...",
        }
    }
}

/// Estado do MCP manager no HOST. A intenção: os servidores MCP (stdio e
/// http) passam a rodar NO computador do usuário (acesso real ao
/// filesystem, npx/node/uvx do host, Chrome de depuração, etc.) — o
/// mcp_manager.py atual mora no container e vai ser portado para um
/// executável Rust standalone supervisionado aqui.
#[derive(Debug, Clone)]
pub struct McpManagerState {
    pub status: McpStatus,
    pub pid: Option<u32>,
    /// Caminho do executável (vazio = não encontrado).
    pub binary: PathBuf,
    /// URL de health check (GET /health — o manager responde 200 {"status":"ok"}).
    pub health_url: String,
}

impl Default for McpManagerState {
    fn default() -> Self {
        Self {
            status: McpStatus::NotInstalled,
            pid: None,
            binary: PathBuf::new(),
            health_url: format!("http://localhost:{MCP_MANAGER_DEFAULT_PORT}/health"),
        }
    }
}

/// Status do transcript (pipeline de voz) no HOST. Mesma forma do
/// McpStatus: NotInstalled (binário do build CMake não existe) e Crashed
/// (estava Running e o health check o perdeu).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptStatus {
    NotInstalled,
    Stopped,
    Starting,
    Running,
    Stopping,
    Crashed,
}

impl TranscriptStatus {
    pub fn label(&self) -> &'static str {
        match self {
            TranscriptStatus::NotInstalled => "não instalado",
            TranscriptStatus::Stopped => "Inativo",
            TranscriptStatus::Starting => "Iniciando...",
            TranscriptStatus::Running => "Ativo",
            TranscriptStatus::Stopping => "Encerrando...",
            TranscriptStatus::Crashed => "Crash — reiniciando...",
        }
    }
}

/// Estado do transcript no HOST — binário C++ (Modules/transcribe) que
/// abre o microfone (PortAudio) e alimenta o orchestrator (/voice/ingest).
/// Não pode viver no container: precisa de device de áudio do host.
#[derive(Debug, Clone)]
pub struct TranscriptState {
    pub status: TranscriptStatus,
    pub pid: Option<u32>,
    /// Caminho do executável (o candidato default, mesmo quando ausente —
    /// o campo `status == NotInstalled` é que diz se ele existe de fato).
    pub binary: PathBuf,
}

impl Default for TranscriptState {
    fn default() -> Self {
        Self {
            status: TranscriptStatus::NotInstalled,
            pid: None,
            binary: PathBuf::new(),
        }
    }
}

/// Estado raiz da aplicação. Uma única instância, compartilhada via `Arc`.
pub struct AppState {
    pub docker: Arc<TokioMutex<DockerState>>,
    pub browser: Arc<TokioMutex<BrowserState>>,
    /// MCP manager: processo do HOST supervisionado pelo tray (o mcp_manager
    /// vira um executável standalone Rust; enquanto ele não existe, o estado
    /// fica em NotInstalled e nada é spawnado).
    pub mcp: Arc<TokioMutex<McpManagerState>>,
    /// Transcript (voz): processo do HOST supervisionado pelo tray —
    /// auto-iniciado junto com o container (same lifecycle do mcp).
    pub transcript: Arc<TokioMutex<TranscriptState>>,
    pub docker_start_script: PathBuf,
    /// Diretório raiz do projeto (onde ficam docker-compose.yml, o binário
    /// do MCP manager, etc.) — cwd dos processos supervisionados.
    pub project_root: PathBuf,
}

pub type SharedState = Arc<AppState>;

/// ✅ Função helper para resolver caminhos relativos ao executável.
/// Se não conseguir determinar o caminho do exe, fallback para CWD.
fn resolve_path(relative: &str) -> PathBuf {
    let base = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));

    base.join(relative)
        .canonicalize()
        .unwrap_or_else(|_| base.join(relative))
}

/// ✅ Detecta o nome correto do script docker-start (.sh ou .bat)
fn find_docker_start_script() -> PathBuf {
    // Tentar .sh primeiro (Linux/macOS), depois .bat (Windows)
    let candidates = if cfg!(windows) {
        vec!["../docker-start.bat", "../docker-start.sh"]
    } else {
        vec!["../docker-start.sh"]
    };

    for candidate in candidates {
        let path = resolve_path(candidate);
        if path.exists() {
            tracing::info!("Script docker encontrado: {}", path.display());
            return path;
        }
    }

    // Fallback: retorna o padrão para o SO atual (pode não existir, mas
    // dá um erro claro depois quando tentar executar)
    let fallback = if cfg!(windows) {
        resolve_path("../docker-start.bat")
    } else {
        resolve_path("../docker-start.sh")
    };

    tracing::warn!(
        "Nenhum script docker-start encontrado, usando fallback: {}",
        fallback.display()
    );
    fallback
}

/// ✅ Detecta o binário do MCP manager (executável Rust standalone no HOST).
///
/// Ordem de resolução:
///   1. env AVA_MCP_MANAGER_BIN — caminho explícito (absoluto ou relativo
///      à raiz do projeto);
///   2. `mcp_manager` (Linux) / `mcp_manager.exe` (Windows) na RAIZ do
///      projeto — mesmo lugar do docker-compose.yml (nome default que o
///      build do port Rust deve produzir);
///   3. `mcp-manager` idem (nome com hífen também é aceito).
///
/// Não encontrado → retorno vazio: o tray mantém o estado NotInstalled e
/// a supervisão fica INERTE (start responde "não instalado", health check
/// não faz nada). Nenhum crash, nenhum spawn fantasma.
fn find_mcp_manager_binary() -> PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.to_path_buf()));

    if let Some(env_path) = std::env::var_os("AVA_MCP_MANAGER_BIN") {
        let p = PathBuf::from(env_path);
        let p = if p.is_absolute() {
            p
        } else {
            let root = exe_dir
                .clone()
                .map(|d| d.join("..").canonicalize().unwrap_or(d))
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
            root.join(p)
        };
        tracing::info!("MCP manager (env AVA_MCP_MANAGER_BIN): {}", p.display());
        return p;
    }

    let names: &[&str] = if cfg!(windows) {
        &["mcp_manager.exe", "mcp-manager.exe"]
    } else {
        &["mcp_manager", "mcp-manager"]
    };

    // Diretórios candidatos, em ordem:
    //   1. exe_dir/.. — layout INSTALADO (ex.: <projeto>/bin/ava-tray → raiz)
    //   2. exe_dir/../.. — layout de BUILD do cargo (Manager/target/debug →
    //      raiz do projeto, que é onde docker-compose.yml e o binário do
    //      port vivem durante o desenvolvimento)
    //   3. exe_dir — exe e binário lado a lado
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(d) = &exe_dir {
        dirs.push(d.join("..").canonicalize().unwrap_or_else(|_| d.clone()));
        dirs.push(d.join("../..").canonicalize().unwrap_or_else(|_| d.clone()));
        dirs.push(d.clone());
    }
    dirs.push(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    for dir in &dirs {
        for name in names {
            let p = dir.join(name);
            if p.exists() {
                tracing::info!("Binário MCP manager encontrado: {}", p.display());
                return p;
            }
        }
    }

    // Não existe — placeholder default (o /mcp/status avisa "não instalado")
    let fallback = dirs[0].join(names[0]);
    tracing::info!(
        "Binário MCP manager não encontrado (procurado: {:?} em {:?}) — supervisão inerte até o port Rust existir.",
        names,
        dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>()
    );
    fallback
}

/// ✅ Detecta o binário do transcript (pipeline de voz C++ no HOST).
///
/// Ordem de resolução:
///   1. env AVA_TRANSCRIPT_BIN — caminho explícito (absoluto ou relativo
///      à raiz do projeto);
///   2. `Modules/transcribe/transcript` (Linux) / `transcript.exe`
///      (Windows) — saída do build CMake do módulo de transcrição
///      (`./Modules/transcribe/make.sh`);
///   3. `transcript` direto nos mesmos diretórios candidatos do MCP
///      manager (exe lado a lado / raiz do projeto).
///
/// Não encontrado → retorno aponta o candidato default: o estado fica em
/// NotInstalled e a supervisão é INERTE (nada é spawnado, sem crash).
fn find_transcript_binary() -> PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.to_path_buf()));

    if let Some(env_path) = std::env::var_os("AVA_TRANSCRIPT_BIN") {
        let p = PathBuf::from(env_path);
        let p = if p.is_absolute() {
            p
        } else {
            let root = exe_dir
                .clone()
                .map(|d| d.join("..").canonicalize().unwrap_or(d))
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
            root.join(p)
        };
        tracing::info!("Transcript (env AVA_TRANSCRIPT_BIN): {}", p.display());
        return p;
    }

    let names: &[&str] = if cfg!(windows) {
        &["transcript.exe"]
    } else {
        &["transcript"]
    };

    // Diretórios candidatos (mesma lógica do MCP manager):
    //   1. exe_dir/../.. — layout de BUILD do cargo (Manager/target/debug →
    //      raiz do projeto, onde Modules/transcribe/ vive no desenvolvimento)
    //   2. exe_dir/..    — layout INSTALADO (ex.: <projeto>/bin/ava-tray → raiz)
    //   3. exe_dir       — exe e binário lado a lado
    //   4. cwd
    // Dentro de cada um, procuramos em Modules/transcribe/ (layout do build
    // CMake) e na própria raiz.
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(d) = &exe_dir {
        dirs.push(d.join("../..").canonicalize().unwrap_or_else(|_| d.clone()));
        dirs.push(d.join("..").canonicalize().unwrap_or_else(|_| d.clone()));
        dirs.push(d.clone());
    }
    dirs.push(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let subdirs = [PathBuf::from("Modules").join("transcribe"), PathBuf::from(".")];
    for dir in &dirs {
        for sub in &subdirs {
            for name in names {
                let p = dir.join(sub).join(name);
                if p.exists() {
                    tracing::info!("Binário transcript encontrado: {}", p.display());
                    return p;
                }
            }
        }
    }

    let fallback = dirs[0].join("Modules").join("transcribe").join(names[0]);
    tracing::info!(
        "Binário transcript não encontrado (procurado {names:?} em Modules/transcribe/) — \
         supervisão inerte até o build do módulo de transcrição existir."
    );
    fallback
}

pub fn new_shared_state() -> SharedState {
    tracing::info!(
        "Diretório do executável: {}",
        std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."))
            .display()
    );

    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    let project_root = exe_dir.join("..").canonicalize().unwrap_or_else(|_| exe_dir.clone());
    let mcp_binary = find_mcp_manager_binary();
    let transcript_binary = find_transcript_binary();

    // Health check aponta para a porta do manager no HOST (env
    // AVA_MCP_MANAGER_URL sobrescreve — ex.: porta diferente no port).
    let mcp_port = std::env::var("AVA_MCP_MANAGER_PORT")
        .ok()
        .and_then(|v| v.trim().parse::<u16>().ok())
        .unwrap_or(MCP_MANAGER_DEFAULT_PORT);

    Arc::new(AppState {
        docker: Arc::new(TokioMutex::new(DockerState::default())),
        browser: Arc::new(TokioMutex::new(BrowserState::default())),
        mcp: Arc::new(TokioMutex::new(McpManagerState {
            status: if mcp_binary.exists() {
                McpStatus::Stopped
            } else {
                McpStatus::NotInstalled
            },
            pid: None,
            binary: mcp_binary,
            health_url: format!("http://localhost:{mcp_port}/health"),
        })),
        transcript: Arc::new(TokioMutex::new(TranscriptState {
            status: if transcript_binary.exists() {
                TranscriptStatus::Stopped
            } else {
                TranscriptStatus::NotInstalled
            },
            pid: None,
            binary: transcript_binary,
        })),
        docker_start_script: find_docker_start_script(),
        project_root,
    })
}
