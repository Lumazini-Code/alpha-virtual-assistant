import { MCP_MANAGER_URL } from "./constants.js";

/**
 * Cliente da API REST do mcp_manager (porta 4008) — instalação assistida de
 * pacote npm (POST /servers/install), registro de servidores externos
 * (POST /servers: transport "http" p/ streamable-http rodando em qualquer
 * URL http(s) — externa ou localhost — e "stdio" p/ comando custom) e
 * remoção (DELETE /servers/{name}). O playwright é nativo do container
 * (Dockerfile) e é filtrado da lista — nunca deve ser gerenciado
 * (instalado/removido) pela TUI.
 */

export const NATIVE_SERVERS = new Set(["playwright", "playwright-mcp", "websearch"]);

export interface MCPServerEntry {
  name: string;
  skill_summary: string;
  transport: string;
  command: string | null;
  args: string[];
  url: string | null;
  enabled: boolean;
  always_on: boolean;
  tools: number;
}

interface InstallResponse {
  installed: boolean;
  name: string;
  package: string;
  synced: number;
  skill_summary: string;
  tools: Array<{ name: string; graph_name: string; snippet: string }>;
}

interface RegisterResponse {
  registered: boolean;
  updated: boolean;
}

interface SyncResponse {
  synced: number;
  registered: number;
  removed: number;
}

async function errorDetail(res: Response): Promise<string> {
  try {
    const j = (await res.json()) as { error?: string; detail?: string };
    const d = j.error ?? j.detail;
    if (typeof d === "string") return d;
    if (d != null) return JSON.stringify(d);
    return `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
}

/** GET /servers — servidores gerenciáveis (nativos como o playwright saem). */
export async function listMCPServers(): Promise<MCPServerEntry[]> {
  const res = await fetch(`${MCP_MANAGER_URL}/servers`);
  if (!res.ok) throw new Error(await errorDetail(res));
  const data = (await res.json()) as { servers: MCPServerEntry[] };
  return (data.servers ?? []).filter((s) => !NATIVE_SERVERS.has(s.name));
}

/**
 * POST /servers/install — instalação assistida de pacote npm: o manager
 * valida no registry, registra como stdio `npx -y <pkg>` e sincroniza as
 * tools no grafo. Pode demorar (npm view + sync com embeddings).
 */
export async function installMCPServer(req: {
  package: string;
  name?: string;
  skill_summary?: string;
}): Promise<{ name: string; synced: number }> {
  const res = await fetch(`${MCP_MANAGER_URL}/servers/install`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const data = (await res.json()) as InstallResponse;
  return { name: data.name, synced: data.synced };
}

/**
 * Registro manual de servidores que não vêm de pacote npm:
 *  - transport "http": streamable-http em URL externa OU local
 *    (http://localhost:4010 — ex.: websearch) — nada é spawnado.
 *  - transport "stdio": comando custom (python/uvx/node…) com args e env.
 *  - transporte "stdio" com comando `npx -y <pacote>` NÃO use aqui —
 *    prefira installMCPServer, que valida o pacote no registry npm.
 */
export async function registerMCPServer(req: {
  name: string;
  transport: "http" | "stdio";
  url?: string;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  skill_summary?: string;
}): Promise<{ name: string; synced: number }> {
  const res = await fetch(`${MCP_MANAGER_URL}/servers`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  await res.json(); // { registered, updated }
  const sync = await syncMCPServer(req.name);
  return { name: req.name, synced: sync.synced };
}

export async function syncMCPServer(
  name: string,
): Promise<{ synced: number; registered: number; removed: number }> {
  const res = await fetch(`${MCP_MANAGER_URL}/servers/${encodeURIComponent(name)}/sync`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as SyncResponse;
}

/** DELETE /servers/{name} — remove o servidor e as tools do grafo. */
export async function deleteMCPServer(name: string): Promise<void> {
  const res = await fetch(`${MCP_MANAGER_URL}/servers/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await errorDetail(res));
}

// ── Helpers de cadastro (espelham as regras do backend mcp_manager) ──────

/** Mesma regra do `_SERVER_NAME_RE` no backend: [A-Za-z0-9_-], sem "__"
 * (o "__" é o separador do graph_name das tools). */
export function isValidServerName(name: string): boolean {
  return /^[A-Za-z0-9][A-Za-z0-9_-]*$/.test(name) && !name.includes("__");
}

/** Nome derivado da URL: "https://api.exemplo.com:8080/mcp" →
 * "api-exemplo-com-8080"; "http://localhost:4010" → "localhost-4010". */
export function serverNameFromUrl(url: string): string {
  const rest = url.replace(/^https?:\/\//i, "");
  const authority = (rest.split(/[/?#]/, 1)[0] ?? "").trim();
  return authority
    .toLowerCase()
    .replace(/[^a-z0-9-]/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** Nome derivado do comando stdio: "/usr/bin/uvx" → "uvx". */
export function serverNameFromCommand(command: string): string {
  const base = command.trim().split(/[\\/]/).pop() ?? "";
  return (
    base
      .toLowerCase()
      .replace(/[^a-z0-9-]/g, "-")
      .replace(/-+/g, "-")
      .replace(/^-+|-+$/g, "")
  );
}

/** Split estilo shell: separa por espaço respeitando aspas simples/duplas
 * (as aspas NÃO entram no argumento final). "" → []. */
export function splitArgs(text: string): string[] {
  const out: string[] = [];
  const re = /"([^"]*)"|'([^']*)'|(\S+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    out.push(m[1] ?? m[2] ?? m[3] ?? "");
  }
  return out;
}
