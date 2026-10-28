// Alpha AI (AVA) — autenticação interna da TUI para as APIs locais (AVA_INTERNAL_TOKEN)
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
import { readFileSync } from "node:fs";
import { envFilePath } from "./paths.js";

/**
 * Autenticação interna da TUI para as APIs locais (orchestrator, memory,
 * MCP manager — ver constants.ts).
 *
 * Mesma resolução de token do Manager (Rust) e do install.sh:
 *   1. AVA_INTERNAL_TOKEN no ambiente (tem prioridade);
 *   2. AVA_INTERNAL_TOKEN gravado em Modules/.env.
 *
 * authHeaders(base) retorna os headers base + Authorization: Bearer <token>
 * quando o token está disponível; sem token, retorna os headers base sem
 * Authorization (as APIs locais podem rodar sem auth configurada).
 */

let cachedToken: string | null = null;
let cacheResolved = false;

/** Resolve o token interno: env > Modules/.env. Cacheia após a 1ª leitura. */
export function internalToken(): string | null {
  if (cacheResolved) return cachedToken;
  cacheResolved = true;

  const envToken = process.env.AVA_INTERNAL_TOKEN;
  if (envToken && envToken.trim()) {
    cachedToken = envToken.trim();
    return cachedToken;
  }

  try {
    const envContent = readFileSync(envFilePath(), "utf-8");
    const match = envContent.match(/^AVA_INTERNAL_TOKEN=(.+)$/m);
    if (match) {
      // Mesmo trim/clean de aspas feito pelo Manager e pelo install.sh.
      const tok = match[1].trim().replace(/^["']|["']$/g, "");
      if (tok) {
        cachedToken = tok;
      }
    }
  } catch {
    // Modules/.env ausente — sem token.
  }
  return cachedToken;
}

/**
 * Headers de autenticação para fetch das APIs internas. Recebe headers base
 * (ex: {"Content-Type": "application/json"}) e adiciona Authorization quando
 * houver token — os callers usam spread: `...authHeaders({})`.
 */
export function authHeaders(base: Record<string, string> = {}): Record<string, string> {
  const token = internalToken();
  if (!token) return { ...base };
  return { ...base, Authorization: `Bearer ${token}` };
}
