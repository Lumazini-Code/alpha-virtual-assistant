import { useEffect, useState } from "react";
import { API_BASE } from "../lib/constants.js";

// Fatia docker do StatusResponse do api.rs (o app não usa mais o
// llama-server local, então só o estado do Docker interessa).
export interface ProcessManagerStatus {
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
