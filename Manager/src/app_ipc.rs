// Alpha AI (AVA) — canal de controle local entre o tray e a janela ava-web
// Copyright (C) 2026  Felipe lumazini
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. See <https://www.gnu.org/licenses/>.
//! Canal de controle local (loopback) entre `ava-tray` e `ava-web`.
//!
//! Por que existe: o `ava-web` é um processo separado (o event loop do
//! eframe/winit do tray não pode ser compartilhado com GTK/WebKit). O tray
//! precisa então de uma forma barata de:
//!   - descobrir se a janela já está aberta e apenas focá-la;
//!   - pedir para ela fechar quando o usuário escolhe "Sair" no tray.
//!
//! O protocolo é uma linha de texto ("focus" ou "quit") num TCP preso em
//! 127.0.0.1. Não há token porque só processos locais alcançam o loopback
//! e os comandos são inofensivos (focar/fechar a própria janela).

use std::io::{BufRead, BufReader, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::time::Duration;

/// Porta padrão do canal de controle. Fora das faixas já usadas pelo
/// ambiente (9000/9001, 9100 do transcribe, 9222 do Chrome, 8777 do grafo).
pub const DEFAULT_CONTROL_PORT: u16 = 9123;

/// Endereço do canal: sempre loopback, com override por env (útil em dev
/// para rodar dois Managers na mesma máquina).
pub fn control_addr() -> SocketAddr {
    let port = std::env::var("AVA_WEBUI_CONTROL_PORT")
        .ok()
        .and_then(|p| p.trim().parse::<u16>().ok())
        .unwrap_or(DEFAULT_CONTROL_PORT);
    SocketAddr::from((Ipv4Addr::LOCALHOST, port))
}

/// Comando aceito pelo listener do `ava-web`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlCommand {
    Focus,
    Quit,
}

impl ControlCommand {
    #[allow(dead_code)] // usado só pelo binário ava-web
    fn from_line(line: &str) -> Option<Self> {
        match line.trim() {
            "focus" => Some(Self::Focus),
            "quit" => Some(Self::Quit),
            _ => None,
        }
    }

    fn as_line(self) -> &'static str {
        match self {
            Self::Focus => "focus",
            Self::Quit => "quit",
        }
    }
}

/// Envia um comando para a janela. `false` = ninguém escutando (janela
/// fechada). Timeout curto: o tray roda isso na thread do GTK.
pub fn send(command: ControlCommand) -> bool {
    let Ok(mut stream) = TcpStream::connect_timeout(&control_addr(), Duration::from_millis(150))
    else {
        return false;
    };
    let _ = stream.set_write_timeout(Some(Duration::from_millis(150)));
    stream
        .write_all(command.as_line().as_bytes())
        .and_then(|_| stream.write_all(b"\n"))
        .is_ok()
}

/// Tenta reservar o canal. `None` = outra instância já está aberta.
#[allow(dead_code)] // usado só pelo binário ava-web
pub fn try_bind() -> Option<TcpListener> {
    TcpListener::bind(control_addr()).ok()
}

/// Atende comandos até o processo morrer. `handler` roda numa thread
/// separada; o `ava-web` só repassa o comando para o event loop do tao.
#[allow(dead_code)] // usado só pelo binário ava-web
pub fn spawn_listener<F>(listener: TcpListener, handler: F)
where
    F: Fn(ControlCommand) + Send + 'static,
{
    let _ = std::thread::Builder::new()
        .name("ava-web-ipc".into())
        .spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let mut line = String::new();
                if BufReader::new(stream).read_line(&mut line).is_ok() {
                    if let Some(command) = ControlCommand::from_line(&line) {
                        handler(command);
                    }
                }
            }
        });
}
