// Alpha AI (AVA) — estatísticas do host para o painel do WebUI
// Copyright (C) 2026  Felipe lumazini
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. See <https://www.gnu.org/licenses/>.
//! CPU/RAM/GPU do HOST (o tray roda no host, não no container).
//!
//! A TUI antiga usava `systeminformation` no processo Node; aqui o mesmo
//! dado vem do `sysinfo`. O `System` fica vivo entre chamadas porque a
//! carga de CPU é calculada por delta. O `Gpus` NÃO é `Send` (guardava um
//! ponteiro NVML), então ele vive numa thread dedicada que publica um
//! snapshot imutável — sem expor `Gpus` para o runtime tokio.

use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use serde::Serialize;
use sysinfo::{Gpus, System};

#[derive(Debug, Clone, Serialize, Default)]
pub struct GpuStats {
    pub has_gpu: bool,
    pub vendor: String,
    pub gpu_pct: f32,
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct SystemStats {
    pub cpu_pct: f32,
    pub ram_pct: f32,
    pub ram_used_gb: f64,
    pub ram_total_gb: f64,
    pub gpu: GpuStats,
}

struct Collector {
    system: Option<System>,
}

static COLLECTOR: OnceLock<Mutex<Collector>> = OnceLock::new();
static GPU_STATS: OnceLock<Mutex<GpuStats>> = OnceLock::new();

/// Snapshot atualizado por uma thread própria (Gpus não é Send).
fn gpu_snapshot() -> GpuStats {
    let cell = GPU_STATS.get_or_init(|| {
        // A thread atualiza a célula; até o primeiro refresh, os defaults
        // (has_gpu=false) são o fallback.
        let _ = thread::Builder::new().name("gpu-stats".into()).spawn(|| {
            let Ok(mut gpus) = Gpus::new() else {
                tracing::debug!("stats: GPU indisponível no sysinfo");
                return;
            };
            loop {
                gpus.refresh(true);
                let next = gpus
                    .first()
                    .map(|device| {
                        let vendor = device.vendor().unwrap_or("").to_ascii_lowercase();
                        let normalized = if vendor.contains("nvidia") {
                            "NVIDIA"
                        } else if vendor.contains("amd") || vendor.contains("ati") {
                            "AMD"
                        } else {
                            "N/A"
                        };
                        GpuStats {
                            has_gpu: true,
                            vendor: normalized.to_string(),
                            gpu_pct: device.usage().unwrap_or(0.0),
                            vram_used_mb: device.used_memory().unwrap_or(0) / (1024 * 1024),
                            vram_total_mb: device.total_memory().unwrap_or(0) / (1024 * 1024),
                        }
                    })
                    .unwrap_or_default();
                if let Some(cell) = GPU_STATS.get() {
                    if let Ok(mut current) = cell.lock() {
                        *current = next;
                    }
                }
                thread::sleep(Duration::from_secs(2));
            }
        });
        Mutex::new(GpuStats::default())
    });
    cell.lock().map(|gpu| gpu.clone()).unwrap_or_default()
}

/// Coleta síncrona — chamada dentro de `spawn_blocking` na API.
pub fn collect() -> SystemStats {
    let collector = COLLECTOR.get_or_init(|| {
        Mutex::new(Collector {
            system: System::new().ok(),
        })
    });
    let mut collector = collector
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let (cpu_pct, ram_pct, ram_used_gb, ram_total_gb) = match collector.system.as_mut() {
        Some(system) => {
            system.refresh_cpu_usage();
            system.refresh_memory();
            let cpu = system.global_cpu_usage();
            let used = system.used_memory();
            let total = system.total_memory();
            (
                cpu,
                if total > 0 {
                    ((used as f64 / total as f64) * 100.0) as f32
                } else {
                    0.0
                },
                used as f64 / (1024.0 * 1024.0 * 1024.0),
                total as f64 / (1024.0 * 1024.0 * 1024.0),
            )
        }
        None => (0.0, 0.0, 0.0, 0.0),
    };

    SystemStats {
        cpu_pct,
        ram_pct,
        ram_used_gb,
        ram_total_gb,
        gpu: gpu_snapshot(),
    }
}
