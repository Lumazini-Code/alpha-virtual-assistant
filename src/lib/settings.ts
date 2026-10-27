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

// ── Esforço de pensamento (reasoning.effort) ────────────────────────────────
// "auto" = não envia o campo (o orchestrator usa settings.json/env/nativo).
// O orchestrator resolve a precedência e injeta no tool loop do /execute.

export type ReasoningEffort = "auto" | "low" | "medium" | "high" | "none";

export interface ReasoningEffortInfo {
  /** Esforço atual (null do backend vira "auto"). */
  effort: ReasoningEffort;
  /** Default de fábrica (para exibir na UI) — sempre "auto" hoje. */
  default: ReasoningEffort;
}

/** GET /settings/reasoning-effort — esforço atual do raciocínio. */
export async function getReasoningEffort(): Promise<ReasoningEffortInfo> {
  const res = await fetch(`${ORCHESTRATOR_URL}/settings/reasoning-effort`, {
    method: "GET",
    headers: authHeaders({}),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const data = (await res.json()) as { effort?: string | null; default?: string | null };
  const toEnum = (v: string | null | undefined): ReasoningEffort =>
    v === "low" || v === "medium" || v === "high" || v === "none" ? v : "auto";
  return { effort: toEnum(data.effort), default: toEnum(data.default) };
}

/** PUT /settings/reasoning-effort — grava o esforço (imediato no próximo turno). */
export async function setReasoningEffort(effort: ReasoningEffort): Promise<ReasoningEffortInfo> {
  const res = await fetch(`${ORCHESTRATOR_URL}/settings/reasoning-effort`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders({}),
    },
    body: JSON.stringify({ effort: effort === "auto" ? null : effort }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const data = (await res.json()) as { effort?: string | null };
  logger.info("settings: reasoning effort alterado", { effort });
  return { effort: effort, default: "auto" };
}
