//! Estado compartilhado da aplicação.
//!
//! Esse módulo guarda tudo que precisa ser visto tanto pela GUI (egui)
//! quanto pelo servidor REST (axum) quanto pelos watchers de inatividade.
//! Como GUI e servidor rodam em threads/tasks diferentes, tudo é protegido
//! por `Mutex` dentro de um `Arc` para podermos clonar referências livremente.

use std::path::{PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex as TokioMutex;
use crate::process_manager::{SharedLlamaLog, new_llama_log};


/// Status de um processo gerenciado (llama-server ou docker).
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

/// Informações sobre o llama-server atualmente em execução (se houver).
#[derive(Debug, Clone)]
pub struct LlamaState {
    pub status: ProcStatus,
    pub pid: Option<u32>,
    pub model_path: Option<String>,
    pub mmproj_path: Option<String>,
    pub port: u16,
    /// Último instante em que houve atividade (requisição na API do llama-server
    /// ou ação manual). Usado pelo watcher de 15 minutos.
    pub last_activity: Option<Instant>,
}

impl Default for LlamaState {
    fn default() -> Self {
        Self {
            status: ProcStatus::Stopped,
            pid: None,
            model_path: None,
            mmproj_path: None,
            port: 2001,
            last_activity: None,
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

/// Um modelo .gguf encontrado na pasta ./Models, pronto para ser exibido
/// na tela de seleção.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub path: String,
    pub mmproj_path: Option<String>,
    pub is_multimodal: bool,
    pub size_mb: u64,
}

/// Estado raiz da aplicação. Uma única instância, compartilhada via `Arc`.
pub struct AppState {
    pub llama: Arc<TokioMutex<LlamaState>>,
    pub docker: Arc<TokioMutex<DockerState>>,
    pub models_dir: PathBuf,
    pub llama_server_bin: PathBuf,
    pub docker_start_script: PathBuf,
    
    /// Ringbuffer de log do llama-server (adicionar este campo!)
    pub llama_log: SharedLlamaLog,
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
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));

    tracing::info!("Diretório do executável: {}", exe_dir.display());

    Arc::new(AppState {
        llama: Arc::new(TokioMutex::new(LlamaState::default())),
        docker: Arc::new(TokioMutex::new(DockerState::default())),
        models_dir: resolve_path("../Modules/Models"),
        llama_server_bin: resolve_path("../Modules/llama-cpp/llama-server"),
        docker_start_script: find_docker_start_script(),
        llama_log: new_llama_log(),
    })
}