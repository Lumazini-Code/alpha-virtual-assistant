import { ORCHESTRATOR_URL } from "./constants.js";
import { logger } from "./logger.js";
import { authHeaders } from "./auth.js";

/**
 * Cliente da persona (1ª linha do system prompt) — GET/PUT /persona no
 * orchestrator. O orchestrator persiste em resource/persona.json e troca
 * surte efeito IMEDIATO no próximo turno (sem restart). A cláusula de tools
 * continua sempre anexada pelo próprio orchestrator.
 */

export interface PersonaInfo {
  /** Persona ativa (persistida ou default). */
  persona: string;
  /** Persona padrão de fábrica (para exibir "default: …" na UI). */
  default: string;
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

/** GET /persona — persona ativa + default. */
export async function getPersona(): Promise<PersonaInfo> {
  const res = await fetch(`${ORCHESTRATOR_URL}/persona`, {
    method: "GET",
    headers: authHeaders({}),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const data = (await res.json()) as { persona?: string; default?: string };
  return {
    persona: data.persona ?? "",
    default: data.default ?? "",
  };
}

/** PUT /persona — troca a persona ativa (mín. 1 char). Persistida e imediata. */
export async function setPersona(text: string): Promise<PersonaInfo> {
  const res = await fetch(`${ORCHESTRATOR_URL}/persona`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders({}),
    },
    body: JSON.stringify({ persona: text }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const data = (await res.json()) as { persona?: string };
  const persona = data.persona ?? text;
  logger.info("persona: salva", { length: persona.length });
  // O PUT não devolve o default — refetch é desnecessário (o default não
  // muda); a UI só usa `default` no primeiro carregamento.
  return { persona, default: "" };
}
