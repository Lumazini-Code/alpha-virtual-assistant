import { API_BASE } from "./constants.js";
import { getChosenModel } from "./modelState.js";

// Espelha SimpleResponse do api.rs: { ok: bool, message: string }.
// IMPORTANTE: /docker/start, /docker/stop e /llama/stop sempre respondem
// HTTP 200, mesmo em falha — o sucesso real vem em `ok`, não no status
// HTTP. Só /llama/start devolve 500 de verdade em erro. Por isso sempre
// lemos o corpo e checamos `ok` explicitamente, em vez de confiar só em
// res.ok.
interface SimpleResponseDto {
  ok: boolean;
  message: string;
}

async function postAndCheck(path: string, body?: unknown): Promise<SimpleResponseDto> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    ...(body !== undefined && {
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  });
  const data = (await res.json().catch(() => null)) as SimpleResponseDto | null;
  if (!res.ok || !data || !data.ok) {
    throw new Error(data?.message ?? `HTTP ${res.status}`);
  }
  return data;
}

export interface EnsureProcessesResult {
  llamaOk: boolean;
  dockerOk: boolean;
  warning?: string;
}

/**
 * Dispara /llama/start e /docker/start no process manager Rust (porta
 * 9001) antes de cada pergunta. Como o backend já ignora a chamada se
 * o processo correspondente estiver ativo (mensagem "já estava ativo
 * com este modelo"), repetir isso a cada pergunta é seguro e barato —
 * substitui a necessidade de rodar start_processes() uma única vez no
 * bloco __main__ do UI.py original.
 *
 * Best-effort: nunca lança — falha aqui não impede a pergunta de seguir
 * para o orchestrator (mesmo comportamento do start_processes() original,
 * que só logava aviso se o Docker falhasse).
 */
export async function ensureProcessesStarted(): Promise<EnsureProcessesResult> {
  const { model, mmprojUsed } = getChosenModel();
  let llamaOk = true;
  let dockerOk = true;
  let warning: string | undefined;

  if (model) {
    try {
      await postAndCheck("/llama/start", { model, mmproj_used: mmprojUsed });
    } catch (err) {
      llamaOk = false;
      warning = `llama-server: ${err instanceof Error ? err.message : err}`;
    }
  }

  try {
    await postAndCheck("/docker/start");
  } catch (err) {
    dockerOk = false;
    const msg = `docker: ${err instanceof Error ? err.message : err}`;
    warning = warning ? `${warning} | ${msg}` : msg;
  }

  return { llamaOk, dockerOk, warning };
}
