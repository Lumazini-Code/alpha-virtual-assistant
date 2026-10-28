// Alpha AI (AVA) — hook do status do tray/process manager (docker)
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

/**
 * Faz polling de GET /status na API Rust (porta 9001). `paused` congela
 * as atualizações de estado (sem isso a TUI redesenharia e corromperia a
 * tela de um subprocesso interativo — ver SettingsPanel).
 */
export function useProcessManagerStatus(
  intervalMs = 3000,
  paused = false,
): ProcessManagerStatus | null {
  const [status, setStatus] = useState<ProcessManagerStatus | null>(null);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      if (pausedRef.current) return;
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
