// Alpha AI (AVA) — hook de estatísticas do sistema (CPU/RAM) na TUI
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
import { useEffect, useRef, useState } from "react";
import si from "systeminformation";
import type { SystemStats } from "../lib/types.js";

const EMPTY_STATS: SystemStats = {
  totalCpuPct: 0,
  sysRamPct: 0,
  sysRamGb: 0,
  sysRamTotGb: 0,
  gpu: { hasGpu: false, vendor: "N/A", gpuPct: 0, vramUsedMb: 0, vramTotMb: 0 },
};

async function collect(): Promise<SystemStats> {
  const [cpu, mem, graphics] = await Promise.all([
    si.currentLoad(),
    si.mem(),
    si.graphics(),
  ]);

  const gpu = graphics.controllers.find(
    (c) => (c.vram ?? 0) > 0 && (c.utilizationGpu ?? undefined) !== undefined,
  ) ?? graphics.controllers[0];

  const vendorRaw = (gpu?.vendor ?? "").toLowerCase();
  const vendor: SystemStats["gpu"]["vendor"] = vendorRaw.includes("nvidia")
    ? "NVIDIA"
    : vendorRaw.includes("amd") || vendorRaw.includes("ati")
      ? "AMD"
      : "N/A";

  return {
    totalCpuPct: cpu.currentLoad,
    sysRamPct: (mem.active / mem.total) * 100,
    sysRamGb: mem.active / 1024 ** 3,
    sysRamTotGb: mem.total / 1024 ** 3,
    gpu: {
      hasGpu: Boolean(gpu),
      vendor,
      gpuPct: gpu?.utilizationGpu ?? 0,
      vramUsedMb: gpu?.memoryUsed ?? 0,
      vramTotMb: gpu?.vram ?? 0,
    },
  };
}

/**
 * Substitui o polling de collect_metrics() do Textual (a cada N segundos).
 * `paused` congela a coleta (sem setState → sem redraw do Ink enquanto um
 * subprocesso interativo ocupa o terminal — ver SettingsPanel).
 */
export function useSystemStats(intervalMs = 2000, paused = false): SystemStats {
  const [stats, setStats] = useState<SystemStats>(EMPTY_STATS);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      if (pausedRef.current) return;
      try {
        const next = await collect();
        if (!cancelled) setStats(next);
      } catch {
        // mantém o último valor conhecido em caso de erro pontual
      }
    };
    tick();
    const id = setInterval(tick, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [intervalMs]);

  return stats;
}
