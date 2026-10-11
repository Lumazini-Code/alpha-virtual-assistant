// Equivalente às constantes do topo do UI.py original.
// Em Ink as cores viram hex direto na prop `color` dos componentes <Text>.

export const API_BASE = process.env.AVA_API_BASE ?? "http://localhost:9001";
export const ORCHESTRATOR_URL =
  process.env.AVA_ORCHESTRATOR_URL ?? "http://localhost:9000";

// Visualizador do grafo Hebbiano (graph_activation.py — roda no container
// ava com network_mode: host, então localhost:8777 funciona da máquina host
// quando o docker está de pé).
export const GRAPH_VIEWER_URL =
  process.env.AVA_GRAPH_URL ?? "http://localhost:8777";

// API REST de memória (planos: POST /plans/list, PATCH /plans/update).
export const MEMORY_API_URL =
  process.env.AVA_MEMORY_URL ?? "http://localhost:3000";

// MCP Manager (porta 4008 — instalação/remoção/registry de servidores MCP).
export const MCP_MANAGER_URL =
  process.env.AVA_MCP_MANAGER_URL ?? "http://localhost:4008";

export const COLORS = {
  blue: "#1243E4",
  gray: "#555555",
  green: "#4CAF7D",
  amber: "#E4A012",
  red: "#E45012",
  white: "#CCCCCC",
  dim: "#333333",
} as const;

export const TAGLINE = "Any model. Every tool. Zero limits.";
