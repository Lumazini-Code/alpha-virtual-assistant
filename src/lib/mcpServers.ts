import { MCP_MANAGER_URL } from "./constants.js";

/**
 * Cliente da API REST do mcp_manager (porta 4008) — instalação assistida
 * (POST /servers/install) e remoção de servidores (DELETE /servers/{name}).
 * O playwright é nativo do container (Dockerfile) e é filtrado da lista —
 * nunca deve ser gerenciado (instalado/removido) pela TUI.
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
}): Promise<InstallResponse> {
  const res = await fetch(`${MCP_MANAGER_URL}/servers/install`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as InstallResponse;
}

/** DELETE /servers/{name} — remove o servidor e as tools do grafo. */
export async function deleteMCPServer(name: string): Promise<void> {
  const res = await fetch(`${MCP_MANAGER_URL}/servers/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await errorDetail(res));
}
