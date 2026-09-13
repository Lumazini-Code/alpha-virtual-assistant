// Equivalente às constantes do topo do UI.py original.
// Em Ink as cores viram hex direto na prop `color` dos componentes <Text>.

export const API_BASE = process.env.AVA_API_BASE ?? "http://localhost:9001";
export const ORCHESTRATOR_URL =
  process.env.AVA_ORCHESTRATOR_URL ?? "http://localhost:9000";

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
