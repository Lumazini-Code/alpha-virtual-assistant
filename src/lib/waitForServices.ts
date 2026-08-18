import { API_BASE, ORCHESTRATOR_URL } from "./constants.js";
import { logger } from "./logger.js";

// Tempo máximo total esperando os dois serviços ficarem de pé antes de
// desistir e deixar a pergunta seguir mesmo assim (mesmo espírito
// best-effort do resto do processManager.ts — nunca trava o app pra
// sempre, só registra no log que não deu tempo).
const WAIT_TIMEOUT_MS = Number(process.env.AVA_WAIT_TIMEOUT_MS ?? 30_000);
const POLL_INTERVAL_MS = Number(process.env.AVA_WAIT_INTERVAL_MS ?? 1_000);
const PROBE_TIMEOUT_MS = 3_000;

export interface ServicesReadyResult {
  dockerReady: boolean;
  orchestratorReady: boolean;
  timedOut: boolean;
  elapsedMs: number;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchWithTimeout(url: string, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

/** GET /status no process manager Rust (porta 9001) e confere docker.status. */
async function isDockerReady(): Promise<boolean> {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/status`, PROBE_TIMEOUT_MS);
    if (!res.ok) return false;
    const data = (await res.json()) as { docker?: { status?: string } };
    const status = String(data?.docker?.status ?? "").toLowerCase();
    return status.includes("run") || status.includes("ativo") || status.includes("ok");
  } catch (err) {
    logger.debug("waitForServices: docker ainda não respondeu em /status", {
      error: err instanceof Error ? err.message : String(err),
    });
    return false;
  }
}

/**
 * O orchestrator não expõe um endpoint de health dedicado, então uma
 * requisição simples na raiz basta: não importa o status HTTP devolvido
 * (mesmo 404 prova que o servidor já está aceitando conexões na porta).
 * Só uma falha de rede (conexão recusada, timeout) conta como "ainda não
 * subiu".
 */
async function isOrchestratorReady(): Promise<boolean> {
  try {
    await fetchWithTimeout(ORCHESTRATOR_URL, PROBE_TIMEOUT_MS);
    return true;
  } catch (err) {
    logger.debug("waitForServices: orchestrator ainda não responde", {
      error: err instanceof Error ? err.message : String(err),
    });
    return false;
  }
}

/**
 * Faz polling de docker (via process manager) e orchestrator até os dois
 * responderem, ou até estourar WAIT_TIMEOUT_MS. Chamar antes de enviar
 * a pergunta ao orchestrator, depois de ensureProcessesStarted() ter
 * disparado o /docker/start.
 */
export async function waitForServicesReady(
  onWaiting?: (elapsedMs: number, dockerReady: boolean, orchestratorReady: boolean) => void,
): Promise<ServicesReadyResult> {
  const start = Date.now();
  let dockerReady = false;
  let orchestratorReady = false;

  logger.info("waitForServices: aguardando docker + orchestrator subirem");

  while (Date.now() - start < WAIT_TIMEOUT_MS) {
    [dockerReady, orchestratorReady] = await Promise.all([
      dockerReady || isDockerReady(),
      orchestratorReady || isOrchestratorReady(),
    ]);

    if (dockerReady && orchestratorReady) {
      const elapsedMs = Date.now() - start;
      logger.info("waitForServices: docker e orchestrator prontos", { elapsedMs });
      return { dockerReady, orchestratorReady, timedOut: false, elapsedMs };
    }

    onWaiting?.(Date.now() - start, dockerReady, orchestratorReady);
    await sleep(POLL_INTERVAL_MS);
  }

  const elapsedMs = Date.now() - start;
  logger.warn("waitForServices: timeout esperando serviços subirem", {
    dockerReady,
    orchestratorReady,
    elapsedMs,
  });
  return { dockerReady, orchestratorReady, timedOut: true, elapsedMs };
}
