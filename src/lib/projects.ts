import { ORCHESTRATOR_URL } from "./constants.js";
import { logger } from "./logger.js";
import { authHeaders } from "./auth.js";

/**
 * Cliente dos projetos (escopos de memória) — /projects no orchestrator.
 *
 * O orchestrator persiste o registry em resource/projects.json e reflete
 * criar/ativar/desativar/apagar no processo da memory (Modules/memory/
 * memory_api.py → /project/*): o switch abre/ativa o diretório do escopo
 * (Modules/memory/db/<slug>/ — LT, grafo Hebbiano, planos, arquivos
 * indexados, KG-RAG). ST, dicionários, pessoas e RAG de tools ficam
 * GLOBAIS. Com um projeto ativo, /read funde projeto + geral; /write grava
 * no escopo do projeto.
 */

export interface ProjectEntry {
  slug: string;
  name: string;
  created_at: number;
  last_activated_at?: number;
}

export interface ProjectsInfo {
  /** Slug do projeto ativo ("general" = modo geral). */
  active: string;
  projects: ProjectEntry[];
}

// Erros de negócio chegam como HTTP 4xx {"error": detail} ou {"detail": ...}
async function errorDetail(res: Response): Promise<string> {
  try {
    const j = (await res.json()) as { error?: string; detail?: string };
    return j.error ?? j.detail ?? `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
}

/** GET /projects — registry + projeto ativo. */
export async function listProjects(): Promise<ProjectsInfo> {
  const res = await fetch(`${ORCHESTRATOR_URL}/projects`, {
    method: "GET",
    headers: authHeaders({}),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const data = (await res.json()) as { active?: string; projects?: ProjectEntry[] };
  return {
    active: data.active ?? "general",
    projects: Array.isArray(data.projects) ? data.projects : [],
  };
}

/** POST /projects — cria (nome → slug) e ativa na hora. */
export async function createProject(name: string): Promise<ProjectsInfo & { project: ProjectEntry }> {
  const res = await fetch(`${ORCHESTRATOR_URL}/projects`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders({}) },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const data = (await res.json()) as {
    active?: string;
    project?: ProjectEntry;
  };
  logger.info("projects: projeto criado", { slug: data.project?.slug });
  return {
    active: data.active ?? "general",
    projects: [],
    project: data.project as ProjectEntry,
  };
}

/** POST /projects/{slug}/activate — troca o escopo ativo. */
export async function activateProject(slug: string): Promise<string> {
  const res = await fetch(`${ORCHESTRATOR_URL}/projects/${encodeURIComponent(slug)}/activate`, {
    method: "POST",
    headers: authHeaders({}),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const data = (await res.json()) as { active?: string };
  logger.info("projects: projeto ativado", { slug });
  return data.active ?? slug;
}

/** POST /projects/deactivate — volta ao modo geral (projeto permanece). */
export async function deactivateProject(): Promise<string> {
  const res = await fetch(`${ORCHESTRATOR_URL}/projects/deactivate`, {
    method: "POST",
    headers: authHeaders({}),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const data = (await res.json()) as { active?: string };
  logger.info("projects: modo geral");
  return data.active ?? "general";
}

/** DELETE /projects/{slug} — apaga o projeto E o diretório do escopo
 * (destrutivo e irreversível: LT, grafo, planos, arquivos indexados). */
export async function deleteProject(slug: string): Promise<string> {
  const res = await fetch(`${ORCHESTRATOR_URL}/projects/${encodeURIComponent(slug)}`, {
    method: "DELETE",
    headers: authHeaders({}),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const data = (await res.json()) as { active?: string };
  logger.info("projects: projeto apagado", { slug });
  return data.active ?? "general";
}
