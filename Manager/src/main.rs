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
mod idle_watcher;
mod process_manager;
mod state;
mod ui;

use eframe::egui;
use state::new_shared_state;
use std::sync::mpsc;
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
    Quit,
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

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ava_tray=info".into()),
        )
        .init();

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

    // Sobe a API REST (axum) em background.
    {
        let state = shared_state.clone();
        rt.spawn(async move {
            if let Err(e) = api::serve(state).await {
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
            }
        });
    }

    // ── Tray icon numa thread secundária (GTK tem loop próprio) ───────────
    // Os handlers de evento enviam UiCommand pela mpsc; a UI (main thread,
    // presa no event loop winit) os consome no repaint.
    let (cmd_tx, cmd_rx) = mpsc::channel::<UiCommand>();

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
fn run_tray_gtk(tx: mpsc::Sender<UiCommand>) {
    if let Err(e) = gtk::init() {
        tracing::error!("Falha ao inicializar GTK na thread do tray: {e}");
        return;
    }
    let _tray_icon = build_tray(tx);
    gtk::main();
}

/// Tray fora do Linux: sem loop GTK; mantém a thread viva com sleep.
#[cfg(not(target_os = "linux"))]
fn run_tray(tx: mpsc::Sender<UiCommand>) {
    let _tray_icon = build_tray(tx);
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

/// Monta o ícone + menu do tray e registra os handlers de evento
/// (enviam UiCommand pela mpsc). Retorna o TrayIcon — PRECISA ser
/// guardado vivo: no Windows/Linux o ícone some se o objeto for dropado.
fn build_tray(tx: mpsc::Sender<UiCommand>) -> TrayIconData {
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
    cmd_rx: mpsc::Receiver<UiCommand>,
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