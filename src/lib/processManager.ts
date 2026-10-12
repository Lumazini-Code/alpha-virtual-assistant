import { API_BASE } from "./constants.js";

// Espelha SimpleResponse do api.rs: { ok: bool, message: string }.
// IMPORTANTE: /docker/start e /docker/stop sempre respondem HTTP 200,
// mesmo em falha — o sucesso real vem em `ok`, não no status HTTP. Por
// isso sempre lemos o corpo e checamos `ok` explicitamente, em vez de
// confiar só em res.ok.
interface SimpleResponseDto {
  ok: boolean;
  message: string;
}

// Teto para toda chamada ao process manager (tray :9001). Sem isso, um
// tray congelado/travado prende o fetch para sempre e a pergunta nunca
// sai do TUI (bug real: /docker/start pendurado silenciava o submit).
// /docker/start costuma responder em ms — o tray spawnando o compose é
// fire-and-forget; 15s é folga ampla. Override via env.
const TRAY_TIMEOUT_MS = Number(process.env.AVA_TRAY_TIMEOUT_MS ?? 15_000);

async function postAndCheck(path: string, body?: unknown): Promise<SimpleResponseDto> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TRAY_TIMEOUT_MS);
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      signal: controller.signal,
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
  } catch (err) {
    if (controller.signal.aborted) {
      throw new Error(
        `process manager não respondeu ${path} em ${Math.round(TRAY_TIMEOUT_MS / 1000)}s — seguindo sem ele (best-effort)`,
      );
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

export interface EnsureDockerResult {
  dockerOk: boolean;
  warning?: string;
}

/**
 * Dispara /docker/start no process manager Rust (porta 9001) antes de
 * cada pergunta. Como o backend já ignora a chamada se o processo
 * correspondente estiver ativo (mensagem "já estava ativo"), repetir
 * isso a cada pergunta é seguro e barato.
 *
 * O app não usa mais llama-server local: o modelo é resolvido pelo
 * orchestrator, então aqui só garante o container Docker do orchestrator.
 *
 * Best-effort: nunca lança — falha aqui não impede a pergunta de seguir
 * para o orchestrator.
 */
export async function ensureDockerStarted(): Promise<EnsureDockerResult> {
  let dockerOk = true;
  let warning: string | undefined;

  try {
    await postAndCheck("/docker/start");
  } catch (err) {
    dockerOk = false;
    warning = `docker: ${err instanceof Error ? err.message : err}`;
  }

  return { dockerOk, warning };
}
