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
