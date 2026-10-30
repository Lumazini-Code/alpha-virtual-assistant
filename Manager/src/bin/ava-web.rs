// Alpha AI (AVA) — janela nativa do WebUI (ava-web)
// Copyright (C) 2026  Felipe lumazini
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. See <https://www.gnu.org/licenses/>.
//! Janela nativa com WebView para o WebUI local do Manager.
//!
//! Roda como processo SEPARADO do tray (`ava-tray --` não envolve o eframe/
//! winit aqui). O tray abre este binário pelo menu "Abrir aplicativo", e o
//! encerra no "Sair" via canal de controle em loopback (`app_ipc`).
//!
//! A página é servida pela API do tray (:9001) — o carregamento é same-origin
//! e o front conversa com o orchestrator pelo proxy `/orch` do Manager. Ou
//! seja: nenhum CORS e nenhum token interno exposto ao JavaScript.
//!
//! Linux: usa WebKitGTK via wry (runtime `libwebkit2gtk-4.1-0`; para
//! compilar, `libwebkit2gtk-4.1-dev`). Windows: usa WebView2 (runtime do
//! Edge, presente no Windows 10/11) e exige `WebView2Loader.dll` ao lado do
//! executável no target `windows-gnu`.

#[path = "../app_ipc.rs"]
mod app_ipc;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context};
#[cfg(target_os = "linux")]
use tao::platform::unix::WindowExtUnix;
use tao::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoopBuilder},
    window::{Icon, WindowBuilder},
};
use wry::{WebContext, WebViewBuilder};

/// Eventos que o listener de controle (thread) manda para o event loop.
#[derive(Debug, Clone, Copy)]
enum ControlEvent {
    Focus,
    Quit,
}

fn main() -> anyhow::Result<()> {
    let root = project_root();
    install_logging(&root);

    let (url, data_dir) = parse_args(&root)?;
    tracing::info!(
        "ava-web iniciando: url={url} data_dir={}",
        data_dir.display()
    );

    // Single-instance: quem chega depois apenas foca a janela existente.
    let Some(listener) = app_ipc::try_bind() else {
        tracing::info!("ava-web já está aberto — enviando focus");
        let _ = app_ipc::send(app_ipc::ControlCommand::Focus);
        return Ok(());
    };

    // Nome fixo do app/janela no Linux: o GTK deriva o WM_CLASS (X11) e o
    // app_id (Wayland) dele. Sem isso o valor é "ava-web"/"Ava-web" e a barra
    // de tarefas não casa com alpha-ai.desktop — cai no ícone genérico.
    #[cfg(target_os = "linux")]
    {
        glib::set_prgname(Some("alpha-ai"));
        glib::set_application_name("Alpha AI");
    }

    let event_loop = EventLoopBuilder::<ControlEvent>::with_user_event().build();
    // Depois do GTK inicializado (pelo event loop) e antes da janela nascer:
    // `prgname` define a parte "instance" do WM_CLASS (alpha-ai) e o GTK
    // capitaliza a parte "class" (Alpha-ai). Fixamos a class em minúsculas
    // para casar exatamente com StartupWMClass=alpha-ai no .desktop.
    #[cfg(target_os = "linux")]
    gtk::gdk::set_program_class("alpha-ai");
    let proxy = event_loop.create_proxy();

    let icon = load_icon();
    let window = WindowBuilder::new()
        .with_title("Alpha AI")
        .with_inner_size(LogicalSize::new(1280.0, 860.0))
        .with_min_inner_size(LogicalSize::new(900.0, 600.0))
        .with_resizable(true)
        // Nasce oculta e só aparece depois do WebView: evita uma janela
        // vazia piscando enquanto o GTK realiza a janela.
        .with_visible(false)
        .with_window_icon(icon)
        .build(&event_loop)
        .context("falha ao criar a janela nativa")?;

    let mut context = WebContext::new(Some(data_dir));
    let builder = WebViewBuilder::new_with_web_context(&mut context)
        .with_url(&url)
        // Cadastro de locutores grava pelo microfone da própria WebView.
        // Só o microfone é concedido; todo o resto segue o fluxo padrão.
        .with_permission_handler(|kind| match kind {
            wry::PermissionKind::Microphone => wry::PermissionResponse::Allow,
            _ => wry::PermissionResponse::Default,
        })
        // Links do chat (target=_blank) vão para o navegador padrão, não
        // abrem uma segunda WebView dentro do app.
        .with_new_window_req_handler(|url, _features| {
            if let Err(e) = webbrowser::open(&url) {
                tracing::warn!("falha ao abrir link externo {url}: {e}");
            }
            wry::NewWindowResponse::Deny
        })
        .with_on_page_load_handler(|event, url| {
            let state = match event {
                wry::PageLoadEvent::Started => "started",
                wry::PageLoadEvent::Finished => "finished",
            };
            tracing::info!("webui: load {state} {url}");
        });

    #[cfg(target_os = "linux")]
    let _webview = {
        use wry::WebViewBuilderExtUnix;
        let vbox = window
            .default_vbox()
            .ok_or_else(|| anyhow!("janela sem container GTK padrão"))?;
        builder
            .build_gtk(vbox)
            .context("falha ao criar WebKitGTK")?
    };
    #[cfg(not(target_os = "linux"))]
    let _webview = builder.build(&window).context("falha ao criar WebView2")?;

    window.set_visible(true);

    // Só começa a aceitar focus/quit depois que a janela existe.
    app_ipc::spawn_listener(listener, move |command| {
        let event = match command {
            app_ipc::ControlCommand::Focus => ControlEvent::Focus,
            app_ipc::ControlCommand::Quit => ControlEvent::Quit,
        };
        let _ = proxy.send_event(event);
    });

    event_loop.run(move |event, _target, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            }
            | Event::UserEvent(ControlEvent::Quit) => {
                tracing::info!("ava-web encerrando");
                *control_flow = ControlFlow::Exit;
            }
            Event::UserEvent(ControlEvent::Focus) => {
                window.set_minimized(false);
                window.set_visible(true);
                window.set_focus();
            }
            _ => {}
        }
    });
}

/// `--url <url>` e `--data-dir <path>`; env como fallback para execução
/// manual em desenvolvimento.
fn parse_args(root: &Path) -> anyhow::Result<(String, PathBuf)> {
    let host = std::env::var("AVA_BIND_HOST")
        .ok()
        .map(|h| h.trim().to_string())
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "127.9.12.22".to_string());
    let host = match host.as_str() {
        "0.0.0.0" | "::" | "[::]" => "127.0.0.1".to_string(),
        other => other.to_string(),
    };
    let mut url = std::env::var("AVA_WEBUI_URL").unwrap_or_else(|_| format!("http://{host}:9001/"));
    let mut data_dir = root.join("cache").join("ava-web");

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--url" => {
                url = args.next().ok_or_else(|| anyhow!("--url exige um valor"))?;
            }
            "--data-dir" => {
                data_dir = PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow!("--data-dir exige um valor"))?,
                );
            }
            other => anyhow::bail!("argumento desconhecido: {other}"),
        }
    }

    std::fs::create_dir_all(&data_dir)
        .with_context(|| format!("falha ao criar {}", data_dir.display()))?;
    Ok((url, data_dir))
}

/// Raiz do projeto: sobe do diretório do executável procurando o
/// docker-compose.yml (mesma ideia do state.rs do tray). Sem ele, usa o cwd
/// — o tray já ancora o cwd na raiz ao spawnar.
fn project_root() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(Path::to_path_buf);
        for _ in 0..5 {
            let Some(current) = dir else { break };
            if current.join("docker-compose.yml").is_file() {
                return current;
            }
            dir = current.parent().map(Path::to_path_buf);
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn install_logging(root: &Path) {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let logs = root.join("logs");
    let _ = std::fs::create_dir_all(&logs);
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(logs.join("ava-web.log"))
        .ok();

    let stderr_layer = tracing_subscriber::fmt::layer();
    let file_layer = file.map(|file| {
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(Arc::new(file))
    });

    let _ = tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ava_web=info".into()),
        )
        .with(stderr_layer)
        .with(file_layer)
        .try_init();
}

fn load_icon() -> Option<Icon> {
    const ICON_BYTES: &[u8] = include_bytes!("../../assets/icon.png");
    let image = image::load_from_memory(ICON_BYTES)
        .map_err(|e| tracing::warn!("ícone inválido em assets/icon.png: {e}"))
        .ok()?
        .into_rgba8();
    let (width, height) = image.dimensions();
    Icon::from_rgba(image.into_raw(), width, height)
        .map_err(|e| tracing::warn!("ícone rejeitado por assets/icon.png: {e}"))
        .ok()
}
