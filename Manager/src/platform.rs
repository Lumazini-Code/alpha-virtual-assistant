// Alpha AI (AVA) — Compatibilidade multiplataforma para spawn de processos pela tray
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
//! Compatibilidade multiplataforma para spawn de processos pela tray.
//!
//! A tray é um app GUI: no Windows, todo processo de console que ela
//! spawn (`taskkill`, `docker`, um `.bat`, o `powershell`, etc.) abre uma
//! janela preta de console que pisca a cada start/stop/health check — a
//! menos que o processo nasça com `CREATE_NO_WINDOW`. No Linux/macOS não
//! há janela alguma e a chamada é um no-op, mas o código continua igual
//! nos dois lados (sem `#[cfg]` espalhado pelos módulos).

use tokio::process::Command;

/// Flag do Windows: cria o processo SEM janela de console.
/// (CREATE_NO_WINDOW — mesma constante de winbase.h, 0x08000000.)
#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Extensão para todo `Command` que a tray spawn: no Windows aplica
/// `CREATE_NO_WINDOW`; fora do Windows não faz nada.
pub trait NoWindow {
    /// Marca o spawn para não abrir janela de console (Windows) / no-op.
    fn no_window(&mut self) -> &mut Self;
}

#[cfg(target_os = "windows")]
impl NoWindow for Command {
    fn no_window(&mut self) -> &mut Self {
        use std::os::windows::process::CommandExt;
        self.creation_flags(CREATE_NO_WINDOW);
        self
    }
}

#[cfg(not(target_os = "windows"))]
impl NoWindow for Command {
    fn no_window(&mut self) -> &mut Self {
        self
    }
}
