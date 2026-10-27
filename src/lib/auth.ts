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
