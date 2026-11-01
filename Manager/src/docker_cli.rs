// Alpha AI (AVA) — acesso ao Docker CLI nas duas rotas
// Copyright (C) 2026  Felipe lumazini
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. See <https://www.gnu.org/licenses/>.
//! Acesso cross-platform ao Docker CLI.
//!
//! Linux: `docker` direto (comportamento histórico).
//! Windows:
//!   - Rota A (Docker Desktop): `docker` está no PATH -> usa direto.
//!   - Rota B (Docker Engine no WSL): não existe `docker` no Windows. Nesse
//!     caso os comandos são executados como
//!     `wsl -d Debian-slim --cd <projeto> -- docker ...`.
//!
//! Sem isso, descoberta/health/logs/grafo do Manager falham com
//! "program not found" mesmo com o stack de pé dentro do WSL.

use std::path::Path;
use std::sync::OnceLock;

use tokio::process::Command;

use crate::platform::NoWindow;

#[cfg(windows)]
fn docker_in_path() -> bool {
    std::env::var_os("PATH")
        .map(|paths| {
            std::env::split_paths(&paths).any(|dir| {
                dir.join("docker.exe").is_file()
                    || dir.join("docker.cmd").is_file()
                    || dir.join("docker.bat").is_file()
            })
        })
        .unwrap_or(false)
}

#[cfg(windows)]
fn docker_daemon_ok() -> bool {
    if !docker_in_path() {
        return false;
    }
    let mut cmd = std::process::Command::new("docker");
    cmd.args(["version", "--format", "{{.Server.Version}}"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    cmd.creation_flags(CREATE_NO_WINDOW);
    matches!(cmd.status(), Ok(status) if status.success())
}

/// Descobre a distro WSL somente quando o daemon do Docker do host não
/// responde. Resultado cacheado: a decisão não muda durante a vida do tray.
#[cfg(windows)]
fn detect_wsl_distro() -> Option<String> {
    // Rota A: Docker Desktop respondendo -> não usa WSL.
    if docker_daemon_ok() {
        return None;
    }

    let distro = std::env::var("AVA_WSL_DISTRO").unwrap_or_else(|_| "Debian-slim".to_string());
    let mut cmd = std::process::Command::new("wsl");
    cmd.args(["-d", &distro, "--", "docker", "--version"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    cmd.creation_flags(CREATE_NO_WINDOW);

    match cmd.status() {
        Ok(status) if status.success() => Some(distro),
        _ => None,
    }
}

#[cfg(not(windows))]
fn detect_wsl_distro() -> Option<String> {
    None
}

fn wsl_distro() -> Option<&'static String> {
    static DISTRO: OnceLock<Option<String>> = OnceLock::new();
    DISTRO.get_or_init(detect_wsl_distro).as_ref()
}

/// Converte C:\Users\x\projeto -> /mnt/c/Users/x/projeto.
/// Também remove o prefixo verbatim `\\?\` que o canonicalize do Windows cria.
fn to_wsl_path(path: &Path) -> Option<String> {
    let raw = path.to_string_lossy();
    let raw_str: &str = raw.as_ref();
    let s = raw_str.strip_prefix(r"\\?\").unwrap_or(raw_str);
    let bytes = s.as_bytes();
    if bytes.len() >= 2 && bytes[1] == b':' {
        let drive = (bytes[0] as char).to_ascii_lowercase();
        let rest = s[2..].replace('\\', "/");
        return Some(format!("/mnt/{drive}{rest}"));
    }
    Some(s.replace('\\', "/"))
}

/// Cria o `Command` correto para `docker` conforme a rota.
///
/// `project_root` é usado para `current_dir` (rota A) ou `--cd` (rota B),
/// permitindo argumentos relativos como `-f docker-compose.yml` e
/// `--env-file Modules/.env` funcionarem igual nas duas rotas.
pub fn command(project_root: Option<&Path>) -> Command {
    if let Some(distro) = wsl_distro() {
        let mut cmd = Command::new("wsl");
        cmd.arg("-d").arg(distro.as_str());
        if let Some(root) = project_root {
            if let Some(wsl_root) = to_wsl_path(root) {
                cmd.arg("--cd").arg(wsl_root);
            }
        }
        cmd.arg("--").arg("docker");
        cmd.no_window();
        cmd
    } else {
        let mut cmd = Command::new("docker");
        if let Some(root) = project_root {
            cmd.current_dir(root);
        }
        cmd.no_window();
        cmd
    }
}
