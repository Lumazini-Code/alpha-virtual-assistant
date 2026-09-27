import { appendFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

// Como o Ink ocupa o terminal inteiro, console.log/console.error não são
// utilizáveis para depuração — eles corrompem o layout da TUI. Por isso
// todo log de depuração vai para um arquivo em disco, salvo direto na
// pasta onde o comando foi executado (cwd).
export const LOG_FILE = join(process.cwd(), "debug.log");

// Log bruto (sem timestamp/nível) do container Docker, no mesmo formato
// que o UI.py original grava em ava_docker.log: uma linha por linha de
// "docker logs -f", sem nenhuma formatação — só pra poder abrir o
// arquivo depois e ver exatamente o que o container produziu.
export const DOCKER_LOG_FILE = join(process.cwd(), "docker.log");

let ready = false;

/**
 * Prepara o arquivo de log, truncando (limpando) qualquer conteúdo de
 * uma execução anterior. Deve ser chamada uma única vez, o mais cedo
 * possível no boot do processo (cli.tsx) — antes de qualquer chamada a
 * `logger.*` — para que cada início do app comece com um log zerado.
 */
export function initLogger(): void {
  try {
    writeFileSync(LOG_FILE, "", { encoding: "utf-8" });
    ready = true;
  } catch {
    // Sem permissão de escrita, disco cheio, etc. — o app segue
    // funcionando normalmente, só sem log em disco.
    ready = false;
  }
}

let dockerLogReady = false;

/**
 * Trunca o arquivo de log do Docker. Chamar uma única vez no boot
 * (cli.tsx), junto com initLogger() — assim, igual ao debug.log, cada
 * início do app começa com o docker.log zerado.
 */
export function initDockerLogFile(): void {
  try {
    writeFileSync(DOCKER_LOG_FILE, "", { encoding: "utf-8" });
    dockerLogReady = true;
  } catch {
    dockerLogReady = false;
  }
}

/**
 * Grava uma linha crua vinda de "docker logs -f" no arquivo dedicado —
 * sem timestamp, sem nível, exatamente como a linha chegou. Espelha
 * _docker_log_file.write(...) do UI.py original.
 */
export function appendDockerLogLine(line: string): void {
  if (!dockerLogReady) return;
  try {
    appendFileSync(DOCKER_LOG_FILE, line.endsWith("\n") ? line : `${line}\n`, {
      encoding: "utf-8",
    });
  } catch {
    // best-effort — igual ao try/except vazio do original
  }
}

type LogLevel = "debug" | "info" | "warn" | "error";

function write(level: LogLevel, message: string, meta?: unknown): void {
  if (!ready) return;
  const ts = new Date().toISOString();
  const metaStr = meta !== undefined ? ` ${safeStringify(meta)}` : "";
  try {
    appendFileSync(LOG_FILE, `[${ts}] [${level.toUpperCase()}] ${message}${metaStr}\n`, {
      encoding: "utf-8",
    });
  } catch {
    // best-effort — um problema ao gravar log nunca deve derrubar o app
  }
}

function safeStringify(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export const logger = {
  debug: (message: string, meta?: unknown) => write("debug", message, meta),
  info: (message: string, meta?: unknown) => write("info", message, meta),
  warn: (message: string, meta?: unknown) => write("warn", message, meta),
  error: (message: string, meta?: unknown) => write("error", message, meta),
};
