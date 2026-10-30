// Alpha AI (AVA) — Tray — gerenciador de processos do docker
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
//! AVA Tray — gerenciador de processos do docker.
//!
//! Fica em segundo plano na bandeja do sistema, expondo uma API REST em
//! localhost:9001 para iniciar/parar o ambiente docker sob demanda. A
//! janela de status só aparece quando o usuário clica no ícone da bandeja.
//!
//! Arquitetura de threads: o event loop winit (eframe) EXIGE a main thread
//! (winit 0.29 faz panic fora dela), então quem vive na main thread é a
//! janela; o tray icon (GTK no Linux) roda numa thread secundária com o
//! próprio loop de eventos e conversa com a UI via canal mpsc.

mod api;
mod chat;
mod idle_watcher;
mod mcp_server;
mod platform;
mod process_manager;
mod state;
mod transcript;
mod ui;

use eframe::egui;
use state::new_shared_state;
use tokio::sync::mpsc;
use tray_icon::{
    menu::{Menu, MenuEvent, MenuItem},
    TrayIconBuilder, TrayIconEvent,
};

/// Comandos da thread do tray para a UI (eframe, na main thread).
/// A janela vive UMA única vez — "fechar" só esconde; abrir/quit são
/// comandos processados pela própria UI no repaint.
#[derive(Debug)]
pub enum UiCommand {
    Show,
    /// Abre a janela direto na tela de chat (wake word detectada —
    /// POST /ui/chat-open do transcribe C++).
    ShowChat,
    Quit,
    /// Evento do Conversation Hub do orchestrator (delta/tool_call/done de um
    /// turno voice/chat). A UI consome no repaint para atualizar o chat.
    ChatEvent(chat::ConversationEvent),
}

/// Raiz do projeto = pai do diretório do executável — mesma regra do
/// state.rs::new_shared_state (…/&lt;raiz&gt;/Manager/ava-tray → raiz).
fn project_root() -> Option<std::path::PathBuf> {
    let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    Some(exe_dir.join(".."))
}

/// Abre (append) &lt;raiz&gt;/logs/&lt;name&gt;. None se não der — nesses casos o log
/// segue só no stderr e NADA aqui pode derrubar o boot.
fn open_log_file(name: &str) -> Option<std::fs::File> {
    let dir = project_root()?.join("logs");
    std::fs::create_dir_all(&dir).ok()?;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(name))
        .ok()
}

/// Logging duplo: stderr (comportamento original — visível quando o tray é
/// lançado de terminal em debug) + logs/ava-tray.log (append, ANSI off).
///
/// O arquivo é obrigatório, não luxo: o tray é lançado pelo instalador com
/// stdout/stderr descartados (nohup >/dev/null) e pelo autostart sem console
/// algum — sem captura em arquivo, a causa de qualquer morte se perdia para
/// sempre (incidente do mcp_manager em 26/10: morreu ~2 min após o boot e
/// não sobrou UMA linha para diagnosticar).
fn install_logging() {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let stderr_layer = tracing_subscriber::fmt::layer();
    let file_layer = open_log_file("ava-tray.log").map(|file| {
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(std::sync::Arc::new(file))
    });
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ava_tray=info".into()),
        )
        .with(stderr_layer)
        .with(file_layer)
        .init();
}

/// Panics também vão para o log em arquivo: o hook roda ANTES do abort
/// (o crate compila com panic=abort) — sem ele, a mensagem do panic só
/// existiria no stderr, que não existe no modo autostart/GUI.
fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let backtrace = std::backtrace::Backtrace::force_capture();
        tracing::error!("PANIC: {info}\n{backtrace}");
        default_hook(info);
    }));
}

fn main() -> anyhow::Result<()> {
    // No Wayland o winit 0.29 ignora set_visible(false), então a janela não
    // some ao "fechar para a tray". Força o backend X11 (XWayland) no Linux.
    // Precisa rodar antes de qualquer thread ser criada (set_var não é
    // thread-safe) e antes do eframe iniciar. Para desfazer em debug:
    // AVA_TRAY_ALLOW_WAYLAND=1.
    #[cfg(target_os = "linux")]
    if std::env::var_os("AVA_TRAY_ALLOW_WAYLAND").is_none() {
        #[allow(unused_unsafe)]
        unsafe {
            std::env::set_var("WINIT_UNIX_BACKEND", "x11");
            // winit 0.29 escolhe Wayland se WAYLAND_DISPLAY existir; sem ele
            // cai para X11 (XWayland). GDK/GTK (tray) também seguem o X11.
            std::env::remove_var("WAYLAND_DISPLAY");
            std::env::set_var("GDK_BACKEND", "x11");
        }
    }

    install_panic_hook();
    install_logging();

    // Ancora o cwd do processo na raiz do projeto. O tray herda o cwd de
    // quem o lança (instalador: $INSTALL_DIR; autostart: $HOME) — e uma
    // reinstalação apaga/recria esse diretório com o tray de pé, deixando
    // o cwd ÓRFÃO. Todo subprocesso herda esse cwd e o docker CLI, p.ex.,
    // morre com "getwd: no such file or directory" antes de ler o -f
    // (foi o que derrubava a supervisão do docker em loop 30s após cada
    // start — 26/10). O current_dir explícito no health_check_docker
    // cobre o caso "dir recriado"; este set cobre o boot em qualquer cwd.
    if let Some(root) = project_root() {
        let _ = std::env::set_current_dir(&root);
    }

    // ── Runtime tokio compartilhado ────────────────────────────────────────
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let rt_handle = rt.handle().clone();

    let shared_state = new_shared_state();

    // ✅ Descobre processos que já estão rodando ANTES de tudo
    // Isso permite que a aplicação "se conecte" a instâncias do
    // docker que foram iniciadas anteriormente.
    rt_handle.block_on(process_manager::discover_running_processes(&shared_state));

    // Transcript (voz, host): descoberta em background — nada aqui pode
    // atrasar a subida da tray (o liveness é só um pid check).
    {
        let state = shared_state.clone();
        rt.spawn(async move {
            transcript::discover(&state).await;
        });
    }

    // ── Canal UiCommand: tray/API → UI (eframe, main thread) ──────────────
    // Criado ANTES da API: o endpoint /ui/chat-open precisa do cmd_tx para
    // pedir à UI que abra a janela na tela de chat (wake word detectada).
    // Unbounded: /ui/chat-open é fire-and-forget — nunca pode bloquear a
    // resposta HTTP (o transcribe não pode esperar por repaint de UI).
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<UiCommand>();

    // Sobe a API REST (axum) em background.
    {
        let state = shared_state.clone();
        let tx = cmd_tx.clone();
        rt.spawn(async move {
            if let Err(e) = api::serve(state, tx).await {
                tracing::error!("API REST encerrou com erro: {e}");
            }
        });
    }

    // Sobe o watcher de inatividade em background.
    {
        let state = shared_state.clone();
        rt.spawn(async move {
            idle_watcher::run(state).await;
        });
    }

    // Sobe o health checker periódico em background.
    {
        let state = shared_state.clone();
        rt.spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                process_manager::health_check_docker(&state).await;
                // MCP manager: embutido no processo (mcp_server.rs) — sem
                // health check externo (não há processo para "perder").
                // Transcript (host): Running sem pid → Crashed → respawn
                // (com o container de pé).
                transcript::health_check(&state).await;
            }
        });
    }

    // ── Tray icon numa thread secundária (GTK tem loop próprio) ───────────
    // Os handlers de evento enviam UiCommand pela mpsc; a UI (main thread,
    // presa no event loop winit) os consome no repaint.
    // (O canal cmd_tx/cmd_rx já foi criado acima — a API também envia.)

    // ── Task de SSE: consome o Conversation Hub do orchestrator ────────────
    // Espelha os turnos voice/chat para a tela de chat da UI via cmd_tx.
    {
        let state = shared_state.clone();
        let tx = cmd_tx.clone();
        rt.spawn(async move {
            let url = format!("{}/conversation/stream", chat::orchestrator_url(&state).await);
            let mut retries: u32 = 0;
            loop {
                let token = match chat::auth_header(&state).await {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::warn!("chat: token ausente — SSE não conectado: {e}");
                        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                        continue;
                    }
                };
                let client = reqwest::Client::new();
                let resp = client
                    .get(&url)
                    .header("Authorization", &token)
                    .send()
                    .await;
                match resp {
                    Ok(resp) if resp.status().is_success() => {
                        tracing::info!("chat: SSE conectado ao Conversation Hub");
                        retries = 0;
                        let mut resp = resp;
                        let mut buf: Vec<u8> = Vec::new();
                        // chunk() não precisa de stream traits — apenas reqwest.
                        while let Some(chunk) = resp.chunk().await.unwrap_or_default() {
                            buf.extend_from_slice(&chunk);
                            // Frames SSE são separados por \n\n — processa todos completos.
                            while let Some(pos) = find_frame_end(&buf) {
                                let frame_bytes: Vec<u8> = buf.drain(..pos + 2).collect();
                                let frame = String::from_utf8_lossy(&frame_bytes);
                                if let Some(event) = chat::parse_sse_frame(&frame) {
                                    let _ = tx.send(UiCommand::ChatEvent(event));
                                }
                            }
                        }
                        tracing::warn!("chat: SSE encerrou — reconectando");
                    }
                    Ok(resp) => {
                        tracing::warn!(
                            "chat: SSE falhou com status {} — retry em breve",
                            resp.status()
                        );
                    }
                    Err(e) => {
                        tracing::warn!("chat: SSE não conectou: {e}");
                    }
                }
                retries += 1;
                let delay = std::time::Duration::from_secs((5u64 * retries as u64).min(30));
                tokio::time::sleep(delay).await;
            }
        });
    }

    #[cfg(target_os = "linux")]
    {
        let tx = cmd_tx;
        std::thread::Builder::new()
            .name("tray-gtk".into())
            .spawn(move || run_tray_gtk(tx))
            .map_err(|e| anyhow::anyhow!("Falha ao criar thread do tray: {e}"))?;
    }
    #[cfg(not(target_os = "linux"))]
    {
        let tx = cmd_tx;
        std::thread::Builder::new()
            .name("tray".into())
            .spawn(move || run_tray(tx))
            .map_err(|e| anyhow::anyhow!("Falha ao criar thread do tray: {e}"))?;
    }

    // ── Main thread = eframe (winit exige main thread) ─────────────────────
    // Roda até a UI fechar de verdade (menu "Sair" do tray → quit_requested
    // → close permitido → run_native retorna → processo encerra e a thread
    // do tray morre junto).
    run_ui(shared_state, rt_handle, cmd_rx)
}

/// Tray no Linux: init + loop GTK na thread do tray. O loop roda pra
/// sempre — quem encerra o processo é a main thread (eframe) ao sair.
/// O TrayIcon PRECISA ficar vivo durante o gtk::main (drop remove o ícone).
#[cfg(target_os = "linux")]
fn run_tray_gtk(tx: mpsc::UnboundedSender<UiCommand>) {
    if let Err(e) = gtk::init() {
        tracing::error!("Falha ao inicializar GTK na thread do tray: {e}");
        return;
    }
    let _tray_icon = build_tray(tx);
    gtk::main();
}

/// Tray fora do Linux: sem loop GTK; mantém a thread viva com sleep.
#[cfg(not(target_os = "linux"))]
fn run_tray(tx: mpsc::UnboundedSender<UiCommand>) {
    let _tray_icon = build_tray(tx);
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

/// Monta o ícone + menu do tray e registra os handlers de evento
/// (enviam UiCommand pela mpsc). Retorna o TrayIcon — PRECISA ser
/// guardado vivo: no Windows/Linux o ícone some se o objeto for dropado.
fn build_tray(tx: mpsc::UnboundedSender<UiCommand>) -> TrayIconData {
    let menu = Menu::new();
    let open_item = MenuItem::new("Abrir", true, None);
    let quit_item = MenuItem::new("Sair", true, None);
    menu.append(&open_item).ok();
    menu.append(&quit_item).ok();

    let open_id = open_item.id().clone();
    let quit_id = quit_item.id().clone();

    let icon = load_tray_icon();

    let tray_icon = TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_tooltip("AVA — Gerenciador de Processos")
        .with_icon(icon)
        .build()
        .expect("falha ao criar o ícone de tray");

    // Clique simples no ícone (abre a janela direto).
    {
        let tx = tx.clone();
        TrayIconEvent::set_event_handler(Some(move |_event| {
            let _ = tx.send(UiCommand::Show);
        }));
    }

    // Menu de contexto (botão direito → Abrir / Sair).
    {
        let tx = tx;
        MenuEvent::set_event_handler(Some(move |event: MenuEvent| {
            if event.id == open_id {
                let _ = tx.send(UiCommand::Show);
            } else if event.id == quit_id {
                let _ = tx.send(UiCommand::Quit);
            }
        }));
    }

    tray_icon
}

type TrayIconData = tray_icon::TrayIcon;

/// Roda o eframe na MAIN thread (janela oculta no startup). Retorna só
/// quando a UI fecha de verdade (menu "Sair" do tray ou erro fatal).
fn run_ui(
    state: state::SharedState,
    rt_handle: tokio::runtime::Handle,
    cmd_rx: mpsc::UnboundedReceiver<UiCommand>,
) -> anyhow::Result<()> {
    // Debug: AVA_TRAY_START_VISIBLE=1 faz a janela nascer visível (testes
    // de automação do fluxo de close sem depender do clique no tray).
    let start_visible = std::env::var("AVA_TRAY_START_VISIBLE").is_ok();
    tracing::debug!("run_ui: start_visible={start_visible}");
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([360.0, 480.0])
            .with_resizable(true)
            .with_title("AVA — Gerenciador de Processos")
            // nasce oculta: só aparece quando o usuário pede via tray
            .with_visible(start_visible)
            // não aparece na barra de tarefas (o ícone fica só na tray)
            .with_taskbar(false),
        ..Default::default()
    };

    eframe::run_native(
        "AVA Tray",
        native_options,
        Box::new(move |_cc| Ok(Box::new(ui::TrayApp::new(state, rt_handle, cmd_rx)))),
    )
    .map_err(|e| anyhow::anyhow!("Erro na janela: {e}"))
}

/// Procura o fim de um frame SSE ("\n\n") no buffer. Retorna o índice do
/// primeiro byte do separador (tal que frame = buf[..pos+2]).
fn find_frame_end(buf: &[u8]) -> Option<usize> {
    buf.windows(2)
        .position(|w| w == b"\n\n")
}

fn load_tray_icon() -> tray_icon::Icon {
    const ICON_BYTES: &[u8] = include_bytes!("../assets/icon.png");

    match image::load_from_memory(ICON_BYTES) {
        Ok(img) => {
            let img = img.into_rgba8();
            let (w, h) = img.dimensions();
            tray_icon::Icon::from_rgba(img.into_raw(), w, h)
                .expect("ícone de tray inválido")
        }
        Err(_) => {
            let size = 16u32;
            let mut rgba = vec![0u8; (size * size * 4) as usize];
            for px in rgba.chunks_mut(4) {
                px.copy_from_slice(&[90, 200, 100, 255]);
            }
            tray_icon::Icon::from_rgba(rgba, size, size).expect("fallback de ícone inválido")
        }
    }
}