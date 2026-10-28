// Alpha AI (AVA) — cliente REST dos planos de execução (endpoints /plans na memory)
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
import { MEMORY_API_URL } from "./constants.js";
import { logger } from "./logger.js";
import { authHeaders } from "./auth.js";

/**
 * Cliente da API REST de memória (porta 3000) — os endpoints de planos.
 * NUNCA tocamos no ava_plans.db direto: toda escrita vai via API, que
 * versiona (update) e normaliza o nome. O grafo (8777) polla o SQLite da
 * memory por conta própria, então a gravação já "sincroniza" sozinha.
 */

export interface PlanEntry {
  plan_id: number;
  name: string;
  description: string;
  version: number;
  steps: Array<Record<string, unknown>>;
  corrections: Record<string, string>;
  created_at: number;
  updated_at: number;
}

interface PlanUpdateResponse {
  plan_id: number;
  version: number;
  updated: boolean;
}

// Erros de negócio (MemoryToolError) chegam como HTTP 400 {"error": detail}
async function errorDetail(res: Response): Promise<string> {
  try {
    const j = (await res.json()) as { error?: string; detail?: string };
    return j.error ?? j.detail ?? `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
}

/** POST /plans/list — todos os planos (com steps para contagem). */
export async function listPlans(): Promise<PlanEntry[]> {
  const headers = new Headers({
    "Content-Type": "application/json",
    ...authHeaders({}),
  });
  const res = await fetch(`${MEMORY_API_URL}/plans/list`, {
    method: "POST",
    headers,
    body: JSON.stringify({}),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const data = (await res.json()) as { total?: number; plans?: PlanEntry[] };
  return data.plans ?? [];
}

/**
 * PATCH /plans/update — renomeia e/ou edita a descrição (identificado por
 * plan_id). Campos undefined são preservados pela API; renomear para o
 * nome atual (normalizado) é no-op. Cada update incrementa a versão.
 */
export async function updatePlan(
  planId: number,
  fields: { name?: string; description?: string },
): Promise<PlanUpdateResponse> {
  const body: Record<string, unknown> = { plan_id: planId };
  if (fields.name !== undefined) body.new_name = fields.name;
  if (fields.description !== undefined) body.description = fields.description;

  const res = await fetch(`${MEMORY_API_URL}/plans/update`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders({}),
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const data = (await res.json()) as PlanUpdateResponse;
  logger.info("plans: plano atualizado", { planId, version: data.version });
  return data;
}
