// Alpha AI (AVA) — snapshot dos logs do container para o WebUI
// Copyright (C) 2026  Felipe lumazini
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. See <https://www.gnu.org/licenses/>.
//! `docker logs --tail N` executado NO HOST (o tray é processo do host).
//!
//! A TUI antiga mantinha `docker logs -f` aberto e mostrava uma janela
//! deslizante. No WebUI o painel faz polling deste snapshot a cada ~2s —
//! mais simples, sem processo pendurado e sem um stream SSE para manter.

use std::process::Stdio;
use std::time::Duration;

use serde::Serialize;

const MAX_TAIL: usize = 2000;

#[derive(Debug, Clone, Serialize)]
pub struct DockerLogs {
    pub container: String,
    pub available: bool,
    pub lines: Vec<String>,
    pub error: Option<String>,
}

/// Últimas `tail` linhas do container (default `ava`, override por
/// `AVA_CONTAINER_NAME`). Nunca falha: erro vira `available=false` + texto.
pub async fn tail(tail: usize) -> DockerLogs {
    let container = std::env::var("AVA_CONTAINER_NAME")
        .ok()
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "ava".to_string());
    let tail = tail.clamp(1, MAX_TAIL).to_string();

    let mut command = tokio::process::Command::new("docker");
    command
        .args(["logs", "--tail", &tail, &container])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(target_os = "windows")]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }

    let output = match tokio::time::timeout(Duration::from_secs(12), command.output()).await {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => {
            return DockerLogs {
                container,
                available: false,
                lines: Vec::new(),
                error: Some(format!("falha ao executar docker: {e}")),
            }
        }
        Err(_) => {
            return DockerLogs {
                container,
                available: false,
                lines: Vec::new(),
                error: Some("docker logs não respondeu em 12s".into()),
            }
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut lines: Vec<String> = stdout.lines().map(str::to_string).collect();
    let error = if output.status.success() {
        None
    } else {
        let detail = stderr.trim();
        Some(if detail.is_empty() {
            format!("docker logs saiu com {}", output.status)
        } else {
            detail.to_string()
        })
    };

    // stderr do CLI geralmente é o próprio log do container em stderr
    // (docker preserva os dois streams). Anexa para não perder linhas.
    if output.status.success() && !stderr.trim().is_empty() {
        lines.extend(stderr.lines().map(str::to_string));
    }

    DockerLogs {
        container,
        available: output.status.success(),
        lines,
        error,
    }
}
