//! AVA Tray — gerenciador de processos para llama-server e docker.
//!
//! Fica em segundo plano na bandeja do sistema, expondo uma API REST em
//! localhost:9001 para iniciar/parar os processos sob demanda. A janela
//! de status só aparece quando o usuário clica no ícone da bandeja.

mod api;
mod idle_watcher;
mod models;
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

/// Mensagens internas para acordar/abrir a janela a partir de eventos
/// do tray (clique no ícone ou no item de menu "Abrir").
enum TrayMessage {
    OpenWindow,
    Quit,
}

fn main() -> anyhow::Result<()> {
    
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ava_tray=info".into()),
        )
        .init();

    // ── GTK: obrigatório no Linux antes de criar o tray icon ──────────────
    #[cfg(target_os = "linux")]
    {
        gtk::init()
            .map_err(|e| anyhow::anyhow!("Falha ao inicializar GTK: {e}"))?;
    }

    // ── Runtime tokio compartilhado ────────────────────────────────────────
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let rt_handle = rt.handle().clone();

    let shared_state = new_shared_state();

    // ✅ Descobre processos que já estão rodando ANTES de tudo
    // Isso permite que a aplicação "se conecte" a instâncias do
    // llama-server ou docker que foram iniciadas anteriormente.
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
                process_manager::health_check_llama(&state).await;
                process_manager::health_check_docker(&state).await;
            }
        });
    }

    // ── Tray icon ───────────────────────────────────────────────────────────
    let (tray_tx, tray_rx) = mpsc::channel::<TrayMessage>();

    let menu = Menu::new();
    let open_item = MenuItem::new("Abrir", true, None);
    let quit_item = MenuItem::new("Sair", true, None);
    menu.append(&open_item).ok();
    menu.append(&quit_item).ok();

    let open_id = open_item.id().clone();
    let quit_id = quit_item.id().clone();

    let icon = load_tray_icon();

    let _tray_icon = TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_tooltip("AVA — Gerenciador de Processos")
        .with_icon(icon)
        .build()?;

    // Eventos de clique simples no ícone (abre a janela direto).
    {
        let tx = tray_tx.clone();
        TrayIconEvent::set_event_handler(Some(move |_event| {
            let _ = tx.send(TrayMessage::OpenWindow);
        }));
    }

    // Eventos do menu de contexto (botão direito -> Abrir / Sair).
    {
        let tx = tray_tx.clone();
        MenuEvent::set_event_handler(Some(move |event: MenuEvent| {
            if event.id == open_id {
                let _ = tx.send(TrayMessage::OpenWindow);
            } else if event.id == quit_id {
                let _ = tx.send(TrayMessage::Quit);
            }
        }));
    }

    // ── Loop principal ──────────────────────────────────────────────────────
    loop {
        #[cfg(target_os = "linux")]
        while gtk::events_pending() {
            gtk::main_iteration();
        }

        match tray_rx.try_recv() {
            Ok(TrayMessage::OpenWindow) => {
                open_window(shared_state.clone(), rt_handle.clone())?;
            }
            Ok(TrayMessage::Quit) => {
                tracing::info!("Encerrando AVA Tray...");
                break;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                std::thread::sleep(std::time::Duration::from_millis(30));
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
        }
    }

    Ok(())
}

/// Abre a janela de status (bloqueante até o usuário fechar).
fn open_window(state: state::SharedState, rt_handle: tokio::runtime::Handle) -> anyhow::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([360.0, 480.0])
            .with_resizable(true)
            .with_title("AVA — Gerenciador de Processos"),
        ..Default::default()
    };

    eframe::run_native(
        "AVA Tray",
        native_options,
        Box::new(move |_cc| Ok(Box::new(ui::TrayApp::new(state, rt_handle)))),
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