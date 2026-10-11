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

/// Estado raiz da aplicação. Uma única instância, compartilhada via `Arc`.
pub struct AppState {
    pub docker: Arc<TokioMutex<DockerState>>,
    pub browser: Arc<TokioMutex<BrowserState>>,
    pub docker_start_script: PathBuf,
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

pub fn new_shared_state() -> SharedState {
    tracing::info!(
        "Diretório do executável: {}",
        std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."))
            .display()
    );

    Arc::new(AppState {
        docker: Arc::new(TokioMutex::new(DockerState::default())),
        browser: Arc::new(TokioMutex::new(BrowserState::default())),
        docker_start_script: find_docker_start_script(),
    })
}
