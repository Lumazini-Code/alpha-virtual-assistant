import { useEffect, useRef, useState } from "react";

export type ServiceStatus = boolean | null; // true = online, false = offline, null = desconhecido

/**
 * Ping periódico em um serviço HTTP (default: 10s) — devolve o estado
 * online/offline para a UI (link do grafo, por exemplo). Null enquanto
 * não houver resposta. `paused` congela o polling (usado quando a TUI
 * entrega o terminal a um subprocesso — ver SettingsPanel).
 */
export function useServiceStatus(
  url: string,
  intervalMs = 10000,
  paused = false,
): ServiceStatus {
  const [online, setOnline] = useState<ServiceStatus>(null);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  useEffect(() => {
    if (paused) return;
    let cancelled = false;

    const ping = async () => {
      // enquanto a TUI está suspensa (subprocesso no terminal), não pinga
      if (pausedRef.current) return;
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 2000);
      try {
        const res = await fetch(url, { signal: controller.signal });
        if (!cancelled) setOnline(res.ok);
      } catch {
        if (!cancelled) setOnline(false);
      } finally {
        clearTimeout(timer);
      }
    };

    ping();
    const id = setInterval(ping, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [url, intervalMs, paused]);

  return online;
}
