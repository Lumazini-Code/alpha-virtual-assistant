import { useEffect, useRef, useState } from "react";
import { ORCHESTRATOR_URL } from "../lib/constants.js";
import { authHeaders } from "../lib/auth.js";

/**
 * Polling leve do projeto ativo (GET /projects no orchestrator, default
 * 10s) — alimenta o badge "◆ Projeto: X" do StatusPanel. Falha/offline →
 * null (badge some; nunca derruba a UI). `paused` congela o polling
 * (padrão dos hooks da TUI — terminal entregue a subprocesso).
 */
export function useActiveProject(
  intervalMs = 10000,
  paused = false,
): string | null {
  const [active, setActive] = useState<string | null>(null);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  useEffect(() => {
    if (paused) return;
    let cancelled = false;

    const poll = async () => {
      if (pausedRef.current) return;
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 2000);
      try {
        const res = await fetch(`${ORCHESTRATOR_URL}/projects`, {
          signal: controller.signal,
          headers: authHeaders({}),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as { active?: string };
        if (!cancelled) setActive(data.active ?? "general");
      } catch {
        if (!cancelled) setActive(null);
      } finally {
        clearTimeout(timer);
      }
    };

    poll();
    const id = setInterval(poll, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [intervalMs, paused]);

  return active;
}
