// Equivalente às constantes do topo do UI.py original.
// Em Ink as cores viram hex direto na prop `color` dos componentes <Text>.

export const API_BASE = process.env.AVA_API_BASE ?? "http://localhost:9001";
export const ORCHESTRATOR_URL =
  process.env.AVA_ORCHESTRATOR_URL ?? "http://localhost:9000";

// Modelo usado para o auto-start de llama-server a cada pergunta (ver
// lib/processManager.ts). Equivalente ao `chosen["model"]` que o UI.py
// original pegava do model_selector antes do primeiro app.run().
export const DEFAULT_MODEL = process.env.AVA_MODEL ?? "";
export const DEFAULT_MMPROJ_USED = process.env.AVA_MMPROJ_USED === "true";

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
