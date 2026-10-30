// Alpha AI (AVA) — lançador da janela ava-web a partir do tray
// Copyright (C) 2026  Felipe lumazini
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. See <https://www.gnu.org/licenses/>.
//! Abre/foca/encerra o processo `ava-web` (janela nativa com WebView).
//!
//! O binário fica ao lado do `ava-tray` (mesmo diretório), então a regra
//! de resolução é simples e independente do diretório de instalação. O
//! WebUI é servido pela própria API do tray (:9001), portanto o tray
//! precisa estar no ar — ele está, já que este código roda nele.

use std::path::PathBuf;

use crate::{api, app_ipc};

/// Janela já aberta? Só foca. Caso contrário, spawna `ava-web`.
pub fn open_or_focus() {
    if app_ipc::send(app_ipc::ControlCommand::Focus) {
        tracing::debug!("ava-web já estava aberto — focado");
        return;
    }

    let Some(binary) = binary_path() else {
        tracing::error!("ava-web não encontrado ao lado do executável do tray");
        return;
    };

    let url = format!("{}/", api::base_url());
    let mut command = std::process::Command::new(&binary);
    command
        .arg("--url")
        .arg(&url)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    // App GUI: no Windows o console do .exe não pode piscar a cada clique.
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }

    match command.spawn() {
        Ok(child) => tracing::info!(
            "ava-web iniciado (pid {}, url {url}, binário {})",
            child.id(),
            binary.display()
        ),
        Err(e) => tracing::error!("falha ao iniciar ava-web ({}): {e}", binary.display()),
    }
}

/// Pede para a janela fechar (usado no "Sair" do tray). Não falha se ela
/// nunca foi aberta.
pub fn request_quit() {
    let _ = app_ipc::send(app_ipc::ControlCommand::Quit);
}

fn binary_path() -> Option<PathBuf> {
    let name = if cfg!(target_os = "windows") {
        "ava-web.exe"
    } else {
        "ava-web"
    };
    let dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    let path = dir.join(name);
    path.is_file().then_some(path)
}
