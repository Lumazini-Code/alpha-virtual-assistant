// Alpha AI (AVA) — Estado compartilhado da aplicação
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

/// Informações sobre o Chrome do host.
///
/// O Chrome é um processo DO HOST, spawnado detached — sobrevive ao
/// stop do docker e ao timer de inatividade de propósito: a intenção
/// é que a janela fique aberta para o usuário depois que o modelo
/// terminar as ações (o Playwright MCP conecta via extensão no perfil
/// padrão — sem CDP, sem modo depuração).
#[derive(Debug, Clone)]
pub struct BrowserState {
    pub status: ProcStatus,
    /// PID do Chrome aberto PELO TRAY. None = nada aberto pelo tray
    /// (o Chrome do usuário pode estar rodando por conta própria).
    pub pid: Option<u32>,
}

impl Default for BrowserState {
    fn default() -> Self {
        Self {
            status: ProcStatus::Stopped,
            pid: None,
        }
    }
}

/// Status do MCP manager EMBUTIDO. O serviço roda dentro do próprio
/// processo do tray (crate::mcp_server): sem processo externo, os únicos
/// estados possíveis referem-se ao init no boot e ao /mcp/stop.
/// `NotInstalled` é mantido como estado inicial inerte padrão (Default).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpStatus {
    NotInstalled,
    Stopped,
    Running,
    Crashed,
}

impl McpStatus {
    pub fn label(&self) -> &'static str {
        match self {
            McpStatus::NotInstalled => "não instalado",
            McpStatus::Stopped => "Inativo",
            McpStatus::Running => "Ativo",
            McpStatus::Crashed => "Crash — não subiu no boot",
        }
    }
}

/// Estado do MCP manager EMBUTIDO. O serviço roda DENTRO do processo do
/// tray (crate::mcp_server — ex-binário standalone :4008): sem binário
/// externo, sem pid, sem health check externo. O campo `status` só reporta
/// o resultado do init no boot (Running / Crashed) e o efeito do /mcp/stop
/// (sessões MCP encerradas). Os servidores MCP (stdio e http) rodam no
/// computador do usuário com acesso real ao filesystem/npx/node — igual ao
/// standalone, agora sem processo extra.
#[derive(Debug, Clone)]
pub struct McpManagerState {
    pub status: McpStatus,
}

impl Default for McpManagerState {
    fn default() -> Self {
        Self {
            status: McpStatus::NotInstalled,
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
/// abre o microfone (miniaudio) e alimenta o orchestrator (/voice/ingest).
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
    /// MCP manager EMBUTIDO: estado do serviço + o handle da sua runtime
    /// (Some após o init bem-sucedido — /mcp/stop derruba as sessões dele).
    pub mcp: Arc<TokioMutex<McpManagerState>>,
    /// Runtime do mcp_server embutido — Some se o init do boot foi OK
    /// (OnceLock porque o SharedState é Arc: o set acontece UMA vez, no
    /// boot da API; leitores (/mcp/stop, auto-stop, UI) consultam depois).
    pub mcp_server: std::sync::OnceLock<Arc<crate::mcp_server::AppState>>,
    /// Transcript (voz): processo do HOST supervisionado pelo tray —
    /// auto-iniciado junto com o container (same lifecycle do mcp).
    pub transcript: Arc<TokioMutex<TranscriptState>>,
    pub docker_start_script: PathBuf,
    /// Diretório raiz do projeto (onde ficam docker-compose.yml, o binário
    /// do MCP manager, etc.) — cwd dos processos supervisionados.
    pub project_root: PathBuf,
    /// Grafo embutido (worker Python no container) — consumido pelo WebUI.
    pub graph: Arc<crate::graph::GraphService>,
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

/// ✅ Resolve a RAIZ do projeto (onde ficam docker-compose.yml, Modules/, o
/// banco do MCP manager etc.).
///
/// Layouts suportados, em ordem:
///   1. INSTALADO — exe dentro do projeto (ex.: <raiz>/ava-tray ou
///      <raiz>/bin/ava-tray → 1 ou 2 níveis acima);
///   2. DEV (cargo) — <raiz>/Manager/target/{debug,release}/ava-tray → 3
///      níveis acima.
///
/// O teste é Contents-Based (docker-compose.yml existe), não conta de
/// nível fixa. Fallback: exe_dir (comportamento inerte como antes).
fn find_project_root(exe_dir: &std::path::Path) -> PathBuf {
    let dir_ok = |d: &std::path::Path| d.join("docker-compose.yml").is_file();
    let mut candidate = exe_dir.to_path_buf();
    for _ in 0..3 {
        candidate = match candidate.parent() {
            Some(p) => p.to_path_buf(),
            None => break,
        };
        if dir_ok(&candidate) {
            return candidate.canonicalize().unwrap_or(candidate);
        }
    }
    // Sem docker-compose.yml nenhum nível acima — mantém o comportamento
    // antigo (exe_dir/..) para o caso de layout de zip temporário do dev.
    let fallback = exe_dir.join("..");
    tracing::warn!(
        "docker-compose.yml não encontrado em até 3 níveis acima do exe — \
         project_root fallback: {}",
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

    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    let project_root = find_project_root(&exe_dir);
    let transcript_binary = find_transcript_binary();

    Arc::new(AppState {
        docker: Arc::new(TokioMutex::new(DockerState::default())),
        browser: Arc::new(TokioMutex::new(BrowserState::default())),
        mcp: Arc::new(TokioMutex::new(McpManagerState {
            status: McpStatus::Stopped,
        })),
        mcp_server: std::sync::OnceLock::new(),
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
        graph: Arc::new(crate::graph::GraphService::new(project_root.clone())),
        project_root,
    })
}
