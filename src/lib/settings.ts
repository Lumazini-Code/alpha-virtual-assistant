import { ORCHESTRATOR_URL } from "./constants.js";
import { logger } from "./logger.js";
import { authHeaders } from "./auth.js";

/**
 * Cliente das preferências da TUI — GET/PUT /settings/* no orchestrator.
 * O orchestrator persiste em resource/settings.json e a mudança surte efeito
 * IMEDIATO no próximo turno (sem restart) — mesmo contrato da persona.
 *
 * Hoje a única chave é o Memory Watch (vigia de confiança do LLM.py): quando
 * ligado, o orchestrator injeta `watch_confidence=true` nos requests de chat
 * (proxy /chat no modo CHAT e /chat/stream no modo TOOLS do /execute).
 */

export interface WatchConfidenceInfo {
  /** Estado atual (persistido ou default). */
  enabled: boolean;
  /** Default de fábrica (para exibir na UI). */
  default: boolean;
}

// Erros de negócio chegam como HTTP 4xx {"error": detail} ou {"detail": ...}
async function errorDetail(res: Response): Promise<string> {
  try {
    const j = (await res.json()) as { error?: string; detail?: string };
    return j.error ?? j.detail ?? `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
}

/** GET /settings/watch-confidence — estado atual do Memory Watch. */
export async function getWatchConfidence(): Promise<WatchConfidenceInfo> {
  const res = await fetch(`${ORCHESTRATOR_URL}/settings/watch-confidence`, {
    method: "GET",
    headers: authHeaders({}),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const data = (await res.json()) as { enabled?: boolean; default?: boolean };
  return {
    enabled: Boolean(data.enabled),
    default: Boolean(data.default),
  };
}

/** PUT /settings/watch-confidence — liga/desliga o Memory Watch (imediato). */
export async function setWatchConfidence(enabled: boolean): Promise<WatchConfidenceInfo> {
  const res = await fetch(`${ORCHESTRATOR_URL}/settings/watch-confidence`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders({}),
    },
    body: JSON.stringify({ enabled }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const data = (await res.json()) as { enabled?: boolean };
  const state = Boolean(data.enabled ?? enabled);
  logger.info("settings: memory watch alterado", { enabled: state });
  return { enabled: state, default: false };
}
