// Alpha AI (AVA) — tipos compartilhados da TUI (mensagens, planos CoT, settings)
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
export type StepStatus = "pending" | "running" | "done" | "error";

export interface CoTStep {
  step: number;
  executor: string;
  action: string;
  dependsOn: number[];
  status: StepStatus;
  latencyMs?: number;
  error?: string;
}

export interface CoTPlanState {
  steps: CoTStep[];
  fromCache: boolean;
  dynamic: boolean; // true = /execute (tool loop, sem plano prévio via evento "plan")
}

export type ChatRole = "user" | "assistant" | "system";

export interface ChatMessageState {
  id: string;
  role: ChatRole;
  text: string;
  reasoning?: string;
  plan?: CoTPlanState;
  streaming: boolean;
  error?: string;
  imagePath?: string;
}

/**
 * Um frame SSE já parseado: `event: <name>\n` + N linhas `data:`
 * concatenadas de volta com "\n", igual _sse() no orchestrator.py monta.
 * `raw` é sempre string; quem consome decide se faz JSON.parse ou usa
 * como texto puro, dependendo do evento (ver handleEvent).
 */
export interface SseFrame {
  event: string;
  raw: string;
}

// ── Payloads exatos por evento (orchestrator.py) ──────────────────────────
// /execute (tool loop) e /code (alpha_code) compartilham nomes de evento
// mas alguns campos diferem — ver comentários.

export interface MetaPayload {
  execution_id: string;
  session_id: string;
  route?: "alpha_code";
  routed_directly?: boolean;
}

// /execute: {"step": turn, "tool": tool_name}
// /code:    {"step": ev_step, "tool": name, "arguments": {...}}
export interface ToolCallPayload {
  step: number | null;
  tool: string;
  arguments?: Record<string, unknown>;
}

// Só existe em /code (alpha_code manda o próprio "step_start")
export interface StepStartPayload {
  step: number;
  executor: string;
  action: string;
}

// /execute: StepResult completo (step, executor, action, success, result, error, retries, latency_ms)
// /code:    subset (step, executor, success, latency_ms, error)
export interface StepDonePayload {
  step: number;
  executor: string;
  action?: string;
  success: boolean;
  result?: unknown;
  error?: string | null;
  retries?: number;
  latency_ms?: number;
}

// Só em /code
export interface PlanPayload {
  steps?: Array<{
    step?: number;
    id?: number;
    executor?: string;
    name?: string;
    action?: string;
    depends_on?: number[];
  }>;
  [key: string]: unknown;
}

export interface ErrorPayload {
  error: string;
  fatal?: boolean;
}

// /execute done
export interface ExecuteDonePayload {
  execution_id: string;
  final_response: string;
  steps: StepDonePayload[];
  total_latency_ms: number;
  errors: string[];
}

// /code done
export interface CodeDonePayload {
  execution_id: string;
  session_id: string;
  final_response: string;
  steps_executed: number;
  tools_called: number;
  tokens_used: number;
  files_changed: string[];
  total_latency_ms: number;
  errors: string[];
  route: "alpha_code";
  routed_directly: boolean;
}

export interface GpuStats {
  hasGpu: boolean;
  vendor: "NVIDIA" | "AMD" | "N/A";
  gpuPct: number;
  vramUsedMb: number;
  vramTotMb: number;
}

export interface SystemStats {
  totalCpuPct: number;
  sysRamPct: number;
  sysRamGb: number;
  sysRamTotGb: number;
  gpu: GpuStats;
}
