// Alpha AI (AVA) — cliente da persona (GET/PUT /persona no orchestrator)
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
