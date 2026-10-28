// Alpha AI (AVA) — constantes compartilhadas da TUI (URLs, paths, cores)
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

// Equivalente às constantes do topo do UI.py original.
// Em Ink as cores viram hex direto na prop `color` dos componentes <Text>.

// ── Descoberta do host de bind (Modules/.env) ─────────────────────────────
// O install.sh grava AVA_BIND_HOST no Modules/.env — fonte da verdade do
// endereço dos serviços. A TUI carrega o valor daqui em vez de manter um
// IP fixo: trocar o bind no .env basta para realinhar TUI e serviços.
//
// projectRoot vem de lib/paths.js: o resolve() por import.meta.url desaba
// em "/" no binário compilado (bun build --compile → $bunfs virtual) e o
// .env nunca era lido — o fallback 127.9.12.22 mascarava isso enquanto o
// bind padrão não mudasse.
const FALLBACK_HOST = "127.9.12.22";

let bindHost = FALLBACK_HOST;
try {
  const envContent = readFileSync(envFilePath(), "utf-8");
  const match = envContent.match(/^AVA_BIND_HOST=(.+)$/m);
  if (match) {
    bindHost = match[1].trim();
  }
} catch {
  // Modules/.env ausente — mantém o fallback padrão.
}

export const API_BASE = process.env.AVA_API_BASE ?? `http://${bindHost}:9001`;
export const ORCHESTRATOR_URL =
  process.env.AVA_ORCHESTRATOR_URL ?? `http://${bindHost}:9000`;

// Visualizador do grafo Hebbiano (graph_activation.py — roda no container
// ava com network_mode: host, então localhost:8777 funciona da máquina host
// quando o docker está de pé).
export const GRAPH_VIEWER_URL =
  process.env.AVA_GRAPH_URL ?? `http://${bindHost}:8777`;

// API REST de memória (planos: POST /plans/list, PATCH /plans/update).
export const MEMORY_API_URL =
  process.env.AVA_MEMORY_URL ?? `http://${bindHost}:3000`;

// MCP Manager (porta 4008 — instalação/remoção/registry de servidores MCP).
export const MCP_MANAGER_URL =
  process.env.AVA_MCP_MANAGER_URL ?? `http://${bindHost}:4008`;

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

// ── Extensões suportadas para /attach ────────────────────────────────────────

export const IMAGE_EXTS = new Set<string>([
  "png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "avif", "svg",
]);

export const DOC_EXTS = new Set<string>([
  "txt", "md", "markdown", "pdf", "docx", "doc", "rtf",
  "py", "js", "ts", "tsx", "jsx", "json", "yaml", "yml",
  "toml", "ini", "cfg", "conf", "html", "htm", "css",
  "csv", "xml", "sql", "sh", "bash", "zsh", "fish",
  "rs", "go", "java", "kt", "kts", "swift",
  "cpp", "cc", "cxx", "hpp", "h", "c",
  "rb", "php", "pl", "pm", "lua", "r",
  "scala", "sc", "clj", "cljs", "ex", "exs",
  "dart", "ml", "mli", "fs", "fsi", "fsx",
  "vim", "dockerfile", "makefile", "gradle", "sbt",
]);
