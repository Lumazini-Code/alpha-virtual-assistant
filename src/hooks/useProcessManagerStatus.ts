import { useEffect, useState } from "react";
import { API_BASE } from "../lib/constants.js";

// Espelha StatusResponse do api.rs.
export interface ProcessManagerStatus {
  llama: {
    status: string;
    pid: number | null;
    model: string | null;
    mmproj: string | null;
    port: number;
    idleSeconds: number | null;
    mode: "text" | "multimodal" | string;
    mainModel: string | null;
  };
  docker: {
    status: string;
    pid: number | null;
    idleSeconds: number | null;
  };
}

/** Faz polling de GET /status na API Rust (porta 9001). */
export function useProcessManagerStatus(intervalMs = 3000): ProcessManagerStatus | null {
  const [status, setStatus] = useState<ProcessManagerStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const res = await fetch(`${API_BASE}/status`);
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled) return;
        setStatus({
          llama: {
            status: data.llama.status,
            pid: data.llama.pid ?? null,
            model: data.llama.model ?? null,
            mmproj: data.llama.mmproj ?? null,
            port: data.llama.port,
            idleSeconds: data.llama.idle_seconds ?? null,
            mode: data.llama.mode,
            mainModel: data.llama.main_model ?? null,
          },
          docker: {
            status: data.docker.status,
            pid: data.docker.pid ?? null,
            idleSeconds: data.docker.idle_seconds ?? null,
          },
        });
      } catch {
        // mantém o último status conhecido em caso de falha pontual
      }
    };
    tick();
    const id = setInterval(tick, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [intervalMs]);

  return status;
}
