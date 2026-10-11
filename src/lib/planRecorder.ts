import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { logger } from "./logger.js";

export interface PlanRecorderResult {
  exitCode: number | null;
  error?: string;
}

/** Raiz do projeto — resolve tanto com `tsx src/cli.tsx` (src/lib → raiz)
 * quanto com o build `dist/cli.js` (dist → raiz). */
export function projectRoot(): string {
  const here = dirname(fileURLToPath(import.meta.url)); // …/src/lib ou …/dist
  return resolve(here, "..", "..");
}

/** Python do projeto: venv primeiro (tem httpx p/ o plan_recorder),
 * fallback para o python3 do sistema. */
export function resolvePython(): string {
  const venv = join(projectRoot(), ".venv", "bin", "python");
  return existsSync(venv) ? venv : "python3";
}

/**
 * Roda o scripts/plan_recorder.py interativamente no terminal atual
 * (stdio "inherit") — quem chama deve ANTES: setRawMode(false) +
 * stdin.pause() e DEPOIS: stdin.resume() + setRawMode(true), para o
 * script dono do stdin (ver SettingsPanel).
 */
export function spawnPlanRecorder(
  urlInicial: string | null,
): Promise<PlanRecorderResult> {
  return new Promise((resolvePromise) => {
    const script = join(projectRoot(), "scripts", "plan_recorder.py");
    const args = [script];
    if (urlInicial && urlInicial.trim()) args.push(urlInicial.trim());
    const python = resolvePython();

    logger.info("spawnPlanRecorder: iniciando", { python, args, cwd: projectRoot() });
    const child = spawn(python, args, {
      stdio: "inherit",
      cwd: projectRoot(),
    });

    child.on("error", (err) => {
      logger.error("spawnPlanRecorder: falha ao iniciar", { message: err.message });
      resolvePromise({ exitCode: null, error: err.message });
    });
    child.on("close", (code) => {
      logger.info("spawnPlanRecorder: finalizado", { exitCode: code });
      resolvePromise({ exitCode: code });
    });
  });
}
