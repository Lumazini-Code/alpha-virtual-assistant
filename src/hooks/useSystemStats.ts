import { useEffect, useState } from "react";
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

/** Substitui o polling de collect_metrics() do Textual (a cada N segundos). */
export function useSystemStats(intervalMs = 2000): SystemStats {
  const [stats, setStats] = useState<SystemStats>(EMPTY_STATS);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
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
