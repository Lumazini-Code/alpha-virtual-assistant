import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Raiz do projeto, robusta a TODOS os modos de execução da TUI:
 *
 *   - dev (`bun run dev` / tsx):      import.meta.url = …/src/lib/paths.ts
 *   - build JS (`node dist/cli.js`):  import.meta.url = …/dist/cli.js
 *   - binário único (bun compile):    import.meta.url = file:///$bunfs/root/…
 *                                     ← filesystem VIRTUAL do bun; o path
 *                                     real no disco só em process.execPath
 *                                     (ex.: <raiz>/tui/alpha-ai)
 *
 * Dois incidentes que moldaram esta função (26/10):
 *   1. No binário compilado, o resolve() por import.meta.url desaba em "/"
 *     (/$bunfs/root → "/" ) — a TUI lia "/Modules/.env" e ia SEM token para
 *     o orchestrator (401 em /projects e /execute).
 *   2. E "/" era pior do que "não existe": uma máquina com um /Modules
 *     órfão na raiz (só com Models/, sem .env) fazia o candidato "/" CASAR
 *     no marcador e esconder o candidato certo. Por isso o marcador é o
 *     ARQUIVO Modules/.env, não o diretório.
 *
 * Ordem dos candidatos: o do execPath vem PRIMEIRO (é o único que aponta
 * para o disco real no bun-compile — mesma resolução do project_root() do
 * Manager em Rust). Nos outros modos (dev/node) o execPath é o binário do
 * bun/node e o candidato simplesmente não casa, passando para o
 * import.meta.url, que é quem resolve nesses modos.
 */

let cachedRoot: string | null = null;

export function projectRoot(): string {
  if (cachedRoot) return cachedRoot;

  const candidates: string[] = [];
  try {
    // Bun compile: exe em <raiz>/tui/alpha-ai → <raiz>; exe direto na raiz
    // (<raiz>/alpha-ai) → o segundo candidato cobre.
    const exeDir = path.dirname(process.execPath);
    candidates.push(path.resolve(exeDir, ".."), exeDir);
  } catch {
    // sem execPath — improvável; segue.
  }
  try {
    const here = path.dirname(fileURLToPath(import.meta.url));
    // …/src/lib → raiz (dev) | …/dist → raiz (build JS)
    candidates.push(path.resolve(here, "..", ".."), path.resolve(here, ".."));
  } catch {
    // import.meta.url indisponível — segue para os candidatos abaixo.
  }
  candidates.push(process.cwd());

  for (const c of candidates) {
    if (c && existsSync(path.join(c, "Modules", ".env"))) {
      cachedRoot = c;
      return c;
    }
  }

  // Nenhum candidato com Modules/.env (instalação excética / dev sem env):
  // devolve o primeiro candidato "real" (não-virtual, existente) para os
  // callers falharem com erro legível em vez de ler dentro do $bunfs.
  cachedRoot =
    candidates.find((c) => c && !c.includes("$bunfs") && existsSync(c)) ??
    process.cwd();
  return cachedRoot;
}

/** Caminho absoluto do Modules/.env da instalação atual. */
export function envFilePath(): string {
  return path.join(projectRoot(), "Modules", ".env");
}
