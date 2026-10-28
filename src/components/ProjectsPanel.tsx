// Alpha AI (AVA) — painel de projetos (troca do escopo ativo de memória)
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
import React, { useCallback, useEffect, useState } from "react";
import { Box, Text, useInput } from "ink";
import SelectInput from "ink-select-input";
import TextInput from "ink-text-input";
import { COLORS } from "../lib/constants.js";
import { logger } from "../lib/logger.js";
import {
  listProjects,
  createProject,
  activateProject,
  deactivateProject,
  deleteProject,
  type ProjectEntry,
} from "../lib/projects.js";

type View = "menu" | "create" | "delete-select" | "delete-confirm";

function truncate(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

interface Props {
  onClose: () => void;
  /** Slug do projeto ativo mudou (o StatusPanel/Badge podem reagir). */
  onActiveChange?: (slug: string) => void;
}

/**
 * ◆ Painel de projetos (Ctrl+P ou /projects) — escopos de memória.
 *
 * Cada projeto tem um conjunto PRÓPRIO de artefatos de conteúdo na memory
 * (LT + grafo Hebbiano + planos + arquivos indexados + KG-RAG em
 * Modules/memory/db/<slug>/). Ativar um projeto faz o /read fundir o escopo
 * do projeto com o GERAL; escritas ficam no escopo do projeto. ST,
 * dicionários, pessoas e RAG de tools são globais (compartilhados).
 *
 * Operações refletem no orchestrator (/projects, registry + proxy) que
 * repassa à memory (/project/*). Criar JÁ ATIVA (fluxo natural: criou →
 * entrou). Apagar é destrutivo (remove o diretório do escopo inteiro) e
 * pede confirmação.
 */
export function ProjectsPanel({ onClose, onActiveChange }: Props) {
  const [view, setView] = useState<View>("menu");
  const [projects, setProjects] = useState<ProjectEntry[]>([]);
  const [active, setActive] = useState<string>("general");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<ProjectEntry | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const info = await listProjects();
      setProjects(info.projects);
      setActive(info.active);
    } catch (err) {
      const m = err instanceof Error ? err.message : String(err);
      logger.warn("projects: GET /projects falhou", { message: m });
      setMsg(`orchestrator offline ou sem resposta: ${m}`);
    } finally {
      setLoading(false);
    }
  }, []);

  // Carrega ao abrir o painel.
  useEffect(() => {
    void load();
  }, [load]);

  // Esc fecha um nível (menu → painel fecha; sub-views → menu). Operações
  // em voo não são interrompíveis.
  useInput((input, key) => {
    if (!key.escape) return;
    if (busy) return;
    if (view === "menu") onClose();
    else {
      setView("menu");
      setNewName("");
      setDeleteTarget(null);
      setMsg(null);
    }
  });

  const applyActive = useCallback(
    (slug: string) => {
      setActive(slug);
      onActiveChange?.(slug);
    },
    [onActiveChange],
  );

  const doActivate = useCallback(
    async (slug: string) => {
      setBusy(true);
      setMsg(null);
      try {
        const a = await activateProject(slug);
        applyActive(a);
        setMsg(`projeto '${slug}' ativado — leitura mescla projeto + geral`);
      } catch (err) {
        const m = err instanceof Error ? err.message : String(err);
        logger.warn("projects: activate falhou", { message: m });
        setMsg(`falha ao ativar '${slug}': ${m}`);
      } finally {
        setBusy(false);
        void load();
      }
    },
    [applyActive, load],
  );

  const doDeactivate = useCallback(async () => {
    setBusy(true);
    setMsg(null);
    try {
      const a = await deactivateProject();
      applyActive(a);
      setMsg("modo geral — escrevendo fora de qualquer projeto");
    } catch (err) {
      const m = err instanceof Error ? err.message : String(err);
      logger.warn("projects: deactivate falhou", { message: m });
      setMsg(`falha ao desativar: ${m}`);
    } finally {
      setBusy(false);
      void load();
    }
  }, [applyActive, load]);

  const doCreate = useCallback(async () => {
    const name = newName.trim();
    if (!name) return;
    setBusy(true);
    setMsg(null);
    try {
      const created = await createProject(name);
      applyActive(created.active);
      setNewName("");
      setMsg(`projeto '${created.project.slug}' criado e ativado`);
      setView("menu");
      await load();
    } catch (err) {
      const m = err instanceof Error ? err.message : String(err);
      logger.warn("projects: create falhou", { message: m });
      setMsg(`falha ao criar: ${m}`);
      setBusy(false);
    }
  }, [newName, applyActive, load]);

  const doDelete = useCallback(
    async (project: ProjectEntry) => {
      setBusy(true);
      setMsg(null);
      try {
        const a = await deleteProject(project.slug);
        applyActive(a);
        setMsg(
          `projeto '${project.slug}' APAGADO (escopo removido da memory — irreversível)`,
        );
        setDeleteTarget(null);
        setView("menu");
        await load();
      } catch (err) {
        const m = err instanceof Error ? err.message : String(err);
        logger.warn("projects: delete falhou", { message: m });
        setMsg(`falha ao apagar '${project.slug}': ${m}`);
        setBusy(false);
      }
    },
    [applyActive, load],
  );

  // ── Itens do menu: cada projeto vira uma ação (ativar/desativar) ─────────
  const menuItems: { label: string; value: string | null }[] = [];
  for (const p of projects) {
    const isActive = p.slug === active;
    menuItems.push({
      label: `${isActive ? "◆" : "◇"} ${p.slug} — ${p.name}${isActive ? "  (ATIVO)" : ""}`,
      value: `activate:${p.slug}`,
    });
  }
  if (active !== "general") {
    menuItems.push({ label: "⌂  Voltar ao modo geral (desativar projeto)", value: "deactivate" });
  }
  menuItems.push({ label: "＋  Criar novo projeto", value: "create" });
  menuItems.push({ label: "🗑  Apagar projeto…", value: "delete" });
  menuItems.push({ label: "↩  Fechar (Esc)", value: "close" });

  const deleteItems: { label: string; value: ProjectEntry | null }[] = [
    ...projects.map((p) => ({
      label: `◇ ${p.slug} — ${p.name}`,
      value: p,
    })),
    { label: "↩  Voltar (Esc)", value: null as ProjectEntry | null },
  ];

  return (
    <Box flexDirection="column" borderStyle="round" borderColor={COLORS.amber} paddingX={1}>
      <Text bold color={COLORS.amber}>
        ◆ Projetos — escopos de memória (escopo ativo: {loading ? "…" : active})
      </Text>

      {msg && (
        <Box marginTop={1}>
          <Text color={COLORS.dim}>{msg}</Text>
        </Box>
      )}

      {view === "menu" && (
        <Box flexDirection="column" marginTop={1}>
          {loading ? (
            <Text color={COLORS.dim}>carregando projetos…</Text>
          ) : (
            <SelectInput
              items={menuItems}
              onSelect={(item) => {
                if (busy) return;
                const v = String(item.value);
                if (v === "close") onClose();
                else if (v === "create") {
                  setNewName("");
                  setMsg(null);
                  setView("create");
                } else if (v === "delete") {
                  setMsg(null);
                  setView("delete-select");
                } else if (v === "deactivate") void doDeactivate();
                else if (v.startsWith("activate:")) {
                  const slug = v.slice("activate:".length);
                  if (slug === active) {
                    setMsg(`'${slug}' já é o projeto ativo`);
                  } else {
                    void doActivate(slug);
                  }
                }
              }}
            />
          )}
          <Box marginTop={1}>
            <Text color={COLORS.dim}>
              {"  "}Ativar funde a memória do projeto com a geral; tudo que for gravado entra
              no escopo do projeto.
            </Text>
          </Box>
        </Box>
      )}

      {view === "create" && (
        <Box flexDirection="column" marginTop={1}>
          <Text>Nome do projeto (Enter cria e ativa):</Text>
          <TextInput
            value={newName}
            onChange={setNewName}
            onSubmit={() => {
              if (!busy) void doCreate();
            }}
            placeholder="ex.: Dissertação IA"
          />
          <Box marginTop={1}>
            <Text color={COLORS.dim}>
              {"  "}O slug do diretório é derivado do nome (minúsculas, sem acentos/espaços).
            </Text>
          </Box>
        </Box>
      )}

      {view === "delete-select" && (
        <Box flexDirection="column" marginTop={1}>
          <Text color={COLORS.red}>Apagar qual projeto? (remove o escopo inteiro)</Text>
          <SelectInput
            items={deleteItems}
            onSelect={(item) => {
              if (busy) return;
              if (item.value === null) {
                setView("menu");
              } else {
                setDeleteTarget(item.value);
                setView("delete-confirm");
              }
            }}
          />
        </Box>
      )}

      {view === "delete-confirm" && deleteTarget && (
        <Box flexDirection="column" marginTop={1}>
          <Text color={COLORS.red}>
            ⚠ Apagar '{deleteTarget.slug}' REMOVE TODA a memória do projeto (LT, grafo,
            planos, arquivos indexados) — irreversível.
          </Text>
          <SelectInput
            items={[
              { label: "✗  Não — voltar", value: false },
              { label: `✓  Sim, apagar '${truncate(deleteTarget.slug, 40)}'`, value: true },
            ]}
            onSelect={(item) => {
              if (busy) return;
              if (item.value) void doDelete(deleteTarget);
              else {
                setDeleteTarget(null);
                setView("menu");
              }
            }}
          />
        </Box>
      )}
    </Box>
  );
}
