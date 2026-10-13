import React, { useCallback, useEffect, useRef, useState } from "react";
import { Box, Text, useInput, useStdin } from "ink";
import SelectInput from "ink-select-input";
import TextInput from "ink-text-input";
import { COLORS } from "../lib/constants.js";
import { logger } from "../lib/logger.js";
import { listPlans, updatePlan, type PlanEntry } from "../lib/plans.js";
import { spawnPlanRecorder } from "../lib/planRecorder.js";
import {
  listMCPServers,
  installMCPServer,
  registerMCPServer,
  deleteMCPServer,
  isValidServerName,
  serverNameFromUrl,
  serverNameFromCommand,
  splitArgs,
  type MCPServerEntry,
} from "../lib/mcpServers.js";

type View = "menu" | "plans" | "edit" | "record" | "mcp" | "mcp-install" | "mcp-delete";
type MCPInstallKind = "npm" | "http" | "stdio";
type MCPInstallStage =
  | "type" // escolha: pacote npm / URL externa / comando stdio
  | "target" // pkg npm | url http(s) | comando do executável
  | "args" // (stdio) argumentos do comando
  | "name" // (http/stdio) nome do servidor (Enter aceita sugestão)
  | "summary" // descrição opcional → Enter registra
  | "installing"
  | "done"
  | "error";
type MCPDeletePhase = "list" | "confirm" | "deleting" | "done" | "error";
type EditStage = "name" | "description" | "saving" | "saved" | "error";
type RecordPhase = "input" | "running" | "result";

function truncate(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

interface Props {
  onClose: () => void;
  /** Congela os hooks de polling da TUI enquanto o plan_recorder.py
   * ocupa o terminal (stdio inherit). */
  onSuspendChange: (suspended: boolean) => void;
}

/**
 * ⚙ Área de configurações da TUI (Ctrl+S ou /settings).
 *
 * Fluxo de gravação de plano: o terminal é ENTREGUE ao
 * scripts/plan_recorder.py (interativo) — antes do spawn, pausamos o
 * stdin da TUI (raw mode off + pause) e congelamos os hooks de polling
 * (onSuspendChange(true)) para não corromper a tela do script; ao sair,
 * retomamos tudo e mostramos o resultado.
 *
 * Fluxo de edição: nome/descrição são gravados via PATCH /plans/update na
 * API do memory (NUNCA no ava_plans.db direto). O grafo (8777) polla o
 * SQLite por conta própria — a gravação já aparece lá em ~1s.
 */
export function SettingsPanel({ onClose, onSuspendChange }: Props) {
  const [view, setView] = useState<View>("menu");
  const [plans, setPlans] = useState<PlanEntry[]>([]);
  const [plansLoading, setPlansLoading] = useState(false);
  const [plansError, setPlansError] = useState<string | null>(null);

  const [editing, setEditing] = useState<PlanEntry | null>(null);
  const [editStage, setEditStage] = useState<EditStage>("name");
  const [editName, setEditName] = useState("");
  const [editDesc, setEditDesc] = useState("");
  const [editError, setEditError] = useState<string | null>(null);
  const [savedVersion, setSavedVersion] = useState<number | null>(null);

  const [recordUrl, setRecordUrl] = useState("");
  const [recordPhase, setRecordPhase] = useState<RecordPhase>("input");
  const [recordResult, setRecordResult] = useState<string | null>(null);

  // ── MCP (instalação/remoção via mcp_manager :4008) ──────────────────────
  const [mcpServers, setMcpServers] = useState<MCPServerEntry[]>([]);
  const [mcpLoading, setMcpLoading] = useState(false);
  const [mcpError, setMcpError] = useState<string | null>(null);
  const [mcpInstallKind, setMcpInstallKind] = useState<MCPInstallKind>("npm");
  const [mcpInstallStage, setMcpInstallStage] = useState<MCPInstallStage>("type");
  const [mcpTarget, setMcpTarget] = useState(""); // pacote npm | url | comando
  const [mcpArgs, setMcpArgs] = useState(""); // args do comando (stdio)
  const [mcpName, setMcpName] = useState(""); // nome do servidor (vazio = deriva)
  const [mcpSummary, setMcpSummary] = useState("");
  const [mcpInstallResult, setMcpInstallResult] = useState<string | null>(null);
  const [mcpDeleting, setMcpDeleting] = useState<MCPServerEntry | null>(null);
  const [mcpDeletePhase, setMcpDeletePhase] = useState<MCPDeletePhase>("list");
  const [mcpDeleteMsg, setMcpDeleteMsg] = useState<string | null>(null);

  const backTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const { stdin, isRawModeSupported, setRawMode } = useStdin() as {
    stdin?: { pause?: () => void; resume?: () => void };
    isRawModeSupported?: boolean;
    setRawMode?: (isRawMode: boolean) => void;
  };

  useEffect(() => {
    return () => {
      if (backTimerRef.current) clearTimeout(backTimerRef.current);
    };
  }, []);

  const loadPlans = useCallback(async () => {
    setPlansLoading(true);
    setPlansError(null);
    try {
      const entries = await listPlans();
      setPlans(entries);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.warn("settings: /plans/list falhou", { message: msg });
      setPlansError(`memory offline ou sem resposta: ${msg}`);
    } finally {
      setPlansLoading(false);
    }
  }, []);

  const enterPlans = useCallback(() => {
    setView("plans");
    void loadPlans();
  }, [loadPlans]);

  const startEditing = useCallback((plan: PlanEntry) => {
    setEditing(plan);
    setEditName(plan.name);
    setEditDesc(plan.description);
    setEditStage("name");
    setEditError(null);
    setSavedVersion(null);
    setView("edit");
  }, []);

  // ── Gravação (plan_recorder.py) ──────────────────────────────────────────
  const startRecording = useCallback(async () => {
    setRecordPhase("running");
    setRecordResult(null);
    onSuspendChange(true);
    try {
      if (isRawModeSupported && setRawMode) setRawMode(false);
    } catch {
      // stdin não-TTY: segue com stdio inherit mesmo assim
    }
    try {
      stdin?.pause?.();
    } catch {
      // idem
    }
    // deixa o Ink terminar o último redraw antes do script escrever
    await new Promise((r) => setTimeout(r, 150));

    const result = await spawnPlanRecorder(recordUrl);

    try {
      stdin?.resume?.();
    } catch {
      // idem
    }
    try {
      if (isRawModeSupported && setRawMode) setRawMode(true);
    } catch {
      // idem
    }
    onSuspendChange(false);

    let msg: string;
    if (result.error) {
      msg = `✗ falha ao iniciar plan_recorder.py: ${result.error}`;
    } else if (result.exitCode === 0) {
      msg = "✓ gravação finalizada — plano enviado ao memory (deve aparecer no grafo em ~1s)";
    } else if (result.exitCode === 130) {
      msg = "✗ gravação cancelada — nada foi gravado";
    } else {
      msg = `✗ plan_recorder.py saiu com código ${result.exitCode} (veja a saída acima)`;
    }
    logger.info("settings: plan_recorder finalizado", { ...result });
    setRecordResult(msg);
    setRecordPhase("result");
    void loadPlans(); // lista por trás já fica atualizada
  }, [recordUrl, onSuspendChange, isRawModeSupported, setRawMode, stdin, loadPlans]);

  // ── MCP: listar / instalar / remover (via mcp_manager :4008) ────────────
  const loadMCPServers = useCallback(async () => {
    setMcpLoading(true);
    setMcpError(null);
    try {
      setMcpServers(await listMCPServers());
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.warn("settings: GET /servers falhou (mcp_manager)", { message: msg });
      setMcpError(`mcp_manager offline ou sem resposta: ${msg}`);
    } finally {
      setMcpLoading(false);
    }
  }, []);

  const enterMcp = useCallback(() => {
    setView("mcp");
    void loadMCPServers();
  }, [loadMCPServers]);

  const resetInstallWizard = useCallback(() => {
    setMcpInstallKind("npm");
    setMcpInstallStage("type");
    setMcpTarget("");
    setMcpArgs("");
    setMcpName("");
    setMcpSummary("");
    setMcpInstallResult(null);
  }, []);

  const beginInstall = useCallback((kind: MCPInstallKind) => {
    setMcpInstallKind(kind);
    setMcpInstallStage("target");
    setMcpTarget("");
    setMcpArgs("");
    setMcpName("");
    setMcpSummary("");
    setMcpInstallResult(null);
  }, []);

  /**
   * Submissão final: valida o cadastro conforme o tipo escolhido e
   * despacha para o mcp_manager (:4008).
   *   npm   → POST /servers/install (backend deriva o nome do pacote)
   *   http  → POST /servers {transport:"http", url}  (externa ou localhost)
   *   stdio → POST /servers {transport:"stdio", command, args}
   * Nos dois casos manuais, o nome vem do usuário (ou derivado da
   * URL/comando) e o sync de tools roda logo em seguida.
   */
  const submitInstall = useCallback(async () => {
    setMcpInstallStage("installing");
    setMcpInstallResult(null);
    try {
      let result: { name: string; synced: number };
      if (mcpInstallKind === "npm") {
        const pkg = mcpTarget.trim();
        if (!pkg) {
          setMcpInstallStage("target");
          return;
        }
        result = await installMCPServer({
          package: pkg,
          ...(mcpSummary.trim() ? { skill_summary: mcpSummary.trim() } : {}),
        });
      } else {
        const name =
          mcpName.trim() ||
          (mcpInstallKind === "http" ? serverNameFromUrl(mcpTarget) : serverNameFromCommand(mcpTarget));
        if (!isValidServerName(name)) {
          setMcpInstallResult(
            "nome inválido — use apenas letras, números, hífen ou underline (sem '__')",
          );
          setMcpInstallStage("error");
          return;
        }
        if (mcpInstallKind === "http") {
          result = await registerMCPServer({
            name,
            transport: "http",
            url: mcpTarget.trim(),
            ...(mcpSummary.trim() ? { skill_summary: mcpSummary.trim() } : {}),
          });
        } else {
          result = await registerMCPServer({
            name,
            transport: "stdio",
            command: mcpTarget.trim(),
            args: splitArgs(mcpArgs),
            ...(mcpSummary.trim() ? { skill_summary: mcpSummary.trim() } : {}),
          });
        }
      }
      logger.info("settings: MCP registrado", { kind: mcpInstallKind, ...result });
      setMcpInstallResult(
        `✓ ${result.name} registrado — ${result.synced} tool(s) sincronizadas no grafo`,
      );
      setMcpInstallStage("done");
      void loadMCPServers();
      // volta pra lista após um instante, pra leitura do resultado
      if (backTimerRef.current) clearTimeout(backTimerRef.current);
      backTimerRef.current = setTimeout(() => {
        resetInstallWizard();
        setView("mcp");
      }, 2500);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.warn("settings: falha ao registrar MCP", { kind: mcpInstallKind, message: msg });
      setMcpInstallResult(msg);
      setMcpInstallStage("error");
    }
  }, [mcpInstallKind, mcpTarget, mcpName, mcpArgs, mcpSummary, loadMCPServers, resetInstallWizard]);

  /** Validação por etapa — em erro, cai direto em "error" (Esc volta). */
  const advanceFromTarget = useCallback(() => {
    const t = mcpTarget.trim();
    if (!t) return;
    if (mcpInstallKind === "npm") {
      setMcpInstallStage("summary");
      return;
    }
    if (mcpInstallKind === "http") {
      if (!/^https?:\/\//i.test(t) || /\s/.test(t)) {
        setMcpInstallResult("URL inválida — comece com http:// ou https:// (sem espaços)");
        setMcpInstallStage("error");
        return;
      }
      setMcpInstallStage("name");
      return;
    }
    // stdio
    setMcpInstallStage("args");
  }, [mcpInstallKind, mcpTarget]);

  /** Enter no campo de nome (http/stdio): vazio deriva da URL/comando. */
  const advanceFromName = useCallback(() => {
    const t = mcpName.trim();
    if (t && !isValidServerName(t)) {
      setMcpInstallResult(
        "nome inválido — use apenas letras, números, hífen ou underline (sem '__')",
      );
      setMcpInstallStage("error");
      return;
    }
    setMcpInstallStage("summary");
  }, [mcpName]);

  const confirmDelete = useCallback(async () => {
    if (!mcpDeleting) return;
    const name = mcpDeleting.name;
    setMcpDeletePhase("deleting");
    try {
      await deleteMCPServer(name);
      logger.info("settings: MCP removido", { name });
      setMcpDeleteMsg(`✓ ${name} removido (tools apagadas do grafo)`);
      setMcpDeletePhase("done");
      void loadMCPServers();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.warn("settings: DELETE /servers/{name} falhou", { message: msg });
      setMcpDeleteMsg(msg);
      setMcpDeletePhase("error");
    }
  }, [mcpDeleting, loadMCPServers]);

  // ── Salvar edição (nome/descrição) via API do memory ────────────────────
  const saveEdit = useCallback(async () => {
    if (!editing) return;
    const nextName = editName.trim();
    const nameChanged = nextName !== editing.name;
    const descChanged = editDesc !== editing.description;
    if (!nextName) {
      setEditError("o nome não pode ficar vazio");
      setEditStage("name");
      return;
    }
    if (!nameChanged && !descChanged) {
      setView("plans"); // nada a salvar
      return;
    }
    setEditStage("saving");
    try {
      const res = await updatePlan(editing.plan_id, {
        ...(nameChanged ? { name: nextName } : {}),
        ...(descChanged ? { description: editDesc } : {}),
      });
      logger.info("settings: plano salvo", { planId: editing.plan_id, version: res.version });
      setSavedVersion(res.version);
      setEditStage("saved");
      // volta pra lista (atualizada) após um instante, pra leitura do "salvo"
      if (backTimerRef.current) clearTimeout(backTimerRef.current);
      backTimerRef.current = setTimeout(() => {
        setView("plans");
        void loadPlans();
      }, 1400);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.warn("settings: PATCH /plans/update falhou", { message: msg });
      setEditError(msg);
      setEditStage("error");
    }
  }, [editing, editName, editDesc, loadPlans]);

  // ── Navegação por teclado (Esc volta um nível / fecha) ───────────────────
  useInput((input, key) => {
    if (!key.escape || recordPhase === "running") return;
    if (view === "menu") onClose();
    else if (view === "plans") setView("menu");
    else if (view === "edit") enterPlans();
    else if (view === "record") {
      setView("menu");
      setRecordPhase("input");
      setRecordResult(null);
    } else if (view === "mcp") setView("menu");
    else if (view === "mcp-install") {
      if (mcpInstallStage === "installing") return; // não interrompe instalação
      if (mcpInstallStage === "type") {
        setView("mcp");
        setMcpInstallResult(null);
      } else {
        resetInstallWizard();
      }
    } else if (view === "mcp-delete") {
      setView("mcp");
      setMcpDeletePhase("list");
      setMcpDeleting(null);
      setMcpDeleteMsg(null);
    }
  });

  // ── Itens de menu / lista ────────────────────────────────────────────────
  const menuItems = [
    { label: "▶  Gravar novo plano (plan_recorder.py)", value: "record" },
    { label: "≡  Planos existentes — ver/editar", value: "plans" },
    { label: "⤓  Servidores MCP — instalar/gerenciar", value: "mcp" },
    { label: "↩  Fechar (Esc)", value: "close" },
  ];

  const planItems = [
    ...plans.map((p) => ({
      label:
        `${p.name} · v${p.version} · ${p.steps?.length ?? 0} passo(s)` +
        (p.description ? ` — ${truncate(p.description, 42)}` : ""),
      value: p,
    })),
    { label: "↩  Voltar (Esc)", value: null as PlanEntry | null },
  ];

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor={COLORS.blue}
      paddingX={1}
    >
      <Text bold color={COLORS.blue}>⚙ Configurações</Text>

      {view === "menu" && (
        <Box flexDirection="column" marginTop={1}>
          <SelectInput
            items={menuItems}
            onSelect={(item) => {
              if (item.value === "record") {
                setRecordUrl("");
                setRecordPhase("input");
                setRecordResult(null);
                setView("record");
              } else if (item.value === "plans") {
                enterPlans();
              } else if (item.value === "mcp") {
                enterMcp();
              } else {
                onClose();
              }
            }}
          />
        </Box>
      )}

      {view === "plans" && (
        <Box flexDirection="column" marginTop={1}>
          <Text color={COLORS.gray}>
            ↑/↓ navega · Enter edita nome/descrição · Esc volta
          </Text>
          {plansLoading && <Text color={COLORS.dim}>carregando planos…</Text>}
          {plansError && <Text color={COLORS.red}>✗ {plansError}</Text>}
          {!plansLoading && !plansError && plans.length === 0 && (
            <Text color={COLORS.dim}>nenhum plano ainda — grave um no menu acima</Text>
          )}
          {!plansLoading && plans.length > 0 && (
            <SelectInput
              items={planItems}
              onSelect={(item) => {
                if (item.value) startEditing(item.value);
                else setView("menu");
              }}
            />
          )}
        </Box>
      )}

      {view === "edit" && editing && (
        <Box flexDirection="column" marginTop={1}>
          <Text color={COLORS.gray}>
            plano #{editing.plan_id} (v{editing.version}, {editing.steps?.length ?? 0} passos)
          </Text>
          {editStage === "name" && (
            <>
              <Text>nome: </Text>
              <TextInput
                value={editName}
                onChange={setEditName}
                onSubmit={() => setEditStage("description")}
                placeholder="nome do plano"
              />
              <Text color={COLORS.dim}>Enter p/ descrição · Esc descarta</Text>
            </>
          )}
          {(editStage === "description" || editStage === "saving" || editStage === "saved" || editStage === "error") && (
            <>
              <Text color={COLORS.dim}>nome: {editName}</Text>
              {editStage === "description" && (
                <>
                  <Text>descrição: </Text>
                  <TextInput
                    value={editDesc}
                    onChange={setEditDesc}
                    onSubmit={() => void saveEdit()}
                    placeholder="descrição do plano (Enter salva)"
                  />
                  <Text color={COLORS.dim}>Enter salva (via /plans/update) · Esc descarta</Text>
                </>
              )}
              {editStage === "saving" && <Text color={COLORS.amber}>salvando…</Text>}
              {editStage === "saved" && (
                <Text color={COLORS.green}>✓ salvo — versão {savedVersion} (grafo atualiza em ~1s)</Text>
              )}
              {editStage === "error" && <Text color={COLORS.red}>✗ {editError}</Text>}
            </>
          )}
        </Box>
      )}

      {view === "record" && (
        <Box flexDirection="column" marginTop={1}>
          <Text color={COLORS.gray}>
            o terminal será entregue ao plan_recorder.py (a TUI volta quando
            você finalizar com 'done'/'cancel' lá no gravador)
          </Text>
          {recordPhase === "input" && (
            <>
              <Text>URL inicial (opcional): </Text>
              <TextInput
                value={recordUrl}
                onChange={setRecordUrl}
                onSubmit={() => void startRecording()}
                placeholder="https://… (Enter vazio começa sem abrir URL)"
              />
              <Text color={COLORS.dim}>Enter inicia · Esc volta</Text>
            </>
          )}
          {recordPhase === "running" && (
            <Text color={COLORS.amber}>
              ▶ plan_recorder.py em execução — TUI congelada; volte ao gravador no terminal
            </Text>
          )}
          {recordPhase === "result" && recordResult && (
            <Text color={recordResult.startsWith("✓") ? COLORS.green : COLORS.red}>{recordResult}</Text>
          )}
        </Box>
      )}

      {view === "mcp" && (
        <Box flexDirection="column" marginTop={1}>
          <Text color={COLORS.gray}>
            servidores MCP · Esc volta
          </Text>
          {mcpLoading && <Text color={COLORS.dim}>carregando servidores…</Text>}
          {mcpError && <Text color={COLORS.red}>✗ {mcpError}</Text>}
          {!mcpLoading && !mcpError && mcpServers.length === 0 && (
            <Text color={COLORS.dim}>nenhum servidor adicionado — use "Adicionar servidor MCP"</Text>
          )}
          {mcpServers.length > 0 && (
            <SelectInput<string | MCPServerEntry>
              items={[
                { label: "⤓  Adicionar servidor MCP (npm · URL · stdio)", value: "install" },
                ...mcpServers.map((s) => ({
                  label: `☐  ${s.name} · ${s.transport} · ${s.tools} tool(s)` + (s.enabled ? "" : " · desabilitado"),
                  value: s,
                })),
                { label: "↩  Voltar (Esc)", value: "back" },
              ]}
              onSelect={(item) => {
                if (item.value === "install") {
                  resetInstallWizard();
                  setView("mcp-install");
                } else if (item.value !== "back") {
                  setMcpDeleting(item.value as MCPServerEntry);
                  setMcpDeletePhase("confirm");
                  setMcpDeleteMsg(null);
                  setView("mcp-delete");
                }
              }}
            />
          )}
          {mcpServers.length === 0 && !mcpLoading && !mcpError && (
            <SelectInput<string>
              items={[
                { label: "⤓  Adicionar servidor MCP (npm · URL · stdio)", value: "install" },
                { label: "↩  Voltar (Esc)", value: "back" },
              ]}
              onSelect={(item) => {
                if (item.value === "install") {
                  resetInstallWizard();
                  setView("mcp-install");
                }
              }}
            />
          )}
        </Box>
      )}

      {view === "mcp-install" && (
        <Box flexDirection="column" marginTop={1}>
          {mcpInstallStage === "type" && (
            <>
              <Text>tipo de servidor: </Text>
              <SelectInput<MCPInstallKind>
                items={[
                  { label: "⤓  Pacote npm (npx -y <pkg> stdio)", value: "npm" as const },
                  { label: "🌐  Servidor externo — URL http(s), ex. http://localhost:4010", value: "http" as const },
                  { label: "⚙   Comando local (stdio) — ex. python / uvx / node", value: "stdio" as const },
                ]}
                onSelect={(item) => beginInstall(item.value)}
              />
              <Text color={COLORS.dim}>Esc volta</Text>
            </>
          )}
          {mcpInstallStage === "target" && (
            <>
              {mcpInstallKind === "npm" && (
                <>
                  <Text>pacote npm: </Text>
                  <TextInput
                    value={mcpTarget}
                    onChange={setMcpTarget}
                    onSubmit={advanceFromTarget}
                    placeholder="@org/mcp-servidor (ou @org/pkg@versão)"
                  />
                  <Text color={COLORS.dim}>Enter p/ descrição · Esc volta</Text>
                </>
              )}
              {mcpInstallKind === "http" && (
                <>
                  <Text>URL do servidor: </Text>
                  <TextInput
                    value={mcpTarget}
                    onChange={setMcpTarget}
                    onSubmit={advanceFromTarget}
                    placeholder="https://… ou http://localhost:4010/mcp"
                  />
                  <Text color={COLORS.dim}>Enter p/ nome · Esc volta</Text>
                </>
              )}
              {mcpInstallKind === "stdio" && (
                <>
                  <Text>comando: </Text>
                  <TextInput
                    value={mcpTarget}
                    onChange={setMcpTarget}
                    onSubmit={advanceFromTarget}
                    placeholder="python / uvx / node / npx …"
                  />
                  <Text color={COLORS.dim}>Enter p/ argumentos · Esc volta</Text>
                </>
              )}
            </>
          )}
          {mcpInstallStage === "args" && (
            <>
              <Text color={COLORS.dim}>comando: {mcpTarget}</Text>
              <Text>argumentos: </Text>
              <TextInput
                value={mcpArgs}
                onChange={setMcpArgs}
                onSubmit={() => setMcpInstallStage("name")}
                placeholder="--flag valor … (aspas agrupam; Enter vazio segue)"
              />
              <Text color={COLORS.dim}>Enter p/ nome · Esc volta</Text>
            </>
          )}
          {mcpInstallStage === "name" && (
            <>
              <Text color={COLORS.dim}>
                {mcpInstallKind === "http" ? `url: ${mcpTarget}` : `comando: ${mcpTarget}${mcpArgs ? ` ${mcpArgs}` : ""}`}
              </Text>
              <Text>nome do servidor: </Text>
              <TextInput
                value={mcpName}
                onChange={setMcpName}
                onSubmit={advanceFromName}
                placeholder={
                  mcpInstallKind === "http"
                    ? `${serverNameFromUrl(mcpTarget)} (Enter aceita)`
                    : `${serverNameFromCommand(mcpTarget)} (Enter aceita)`
                }
              />
              <Text color={COLORS.dim}>Enter p/ descrição · Esc volta</Text>
            </>
          )}
          {(mcpInstallStage === "summary" || mcpInstallStage === "installing" || mcpInstallStage === "done" || mcpInstallStage === "error") && (
            <>
              <Text color={COLORS.dim}>
                {mcpInstallKind === "npm" && `pacote: ${mcpTarget}`}
                {mcpInstallKind === "http" && `url: ${mcpTarget} · nome: ${mcpName || serverNameFromUrl(mcpTarget)}`}
                {mcpInstallKind === "stdio" &&
                  `comando: ${mcpTarget}${mcpArgs ? ` ${mcpArgs}` : ""} · nome: ${mcpName || serverNameFromCommand(mcpTarget)}`}
              </Text>
              {mcpInstallStage === "summary" && (
                <>
                  <Text>descrição (opcional): </Text>
                  <TextInput
                    value={mcpSummary}
                    onChange={setMcpSummary}
                    onSubmit={() => void submitInstall()}
                    placeholder="para que serve (Enter registra)"
                  />
                  <Text color={COLORS.dim}>Enter registra + sincroniza · Esc volta</Text>
                </>
              )}
              {mcpInstallStage === "installing" && (
                <Text color={COLORS.amber}>
                  ▶ registrando e sincronizando (sync com embeddings pode demorar)…
                </Text>
              )}
              {mcpInstallStage === "done" && mcpInstallResult && (
                <Text color={COLORS.green}>{mcpInstallResult}</Text>
              )}
              {mcpInstallStage === "error" && mcpInstallResult && (
                <Text color={COLORS.red}>✗ {mcpInstallResult}</Text>
              )}
            </>
          )}
        </Box>
      )}

      {view === "mcp-delete" && mcpDeleting && (
        <Box flexDirection="column" marginTop={1}>
          <Text color={COLORS.gray}>
            {mcpDeleting.name} · {mcpDeleting.transport} · {mcpDeleting.tools} tool(s)
            {mcpDeleting.skill_summary ? ` — ${truncate(mcpDeleting.skill_summary, 42)}` : ""}
          </Text>
          {mcpDeletePhase === "confirm" && (
            <SelectInput
              items={[
                { label: "✗  Remover (servidor + tools do grafo)", value: true as const },
                { label: "↩  Cancelar (Esc)", value: false as const },
              ]}
              onSelect={(item) => {
                if (item.value) void confirmDelete();
                else {
                  setView("mcp");
                  setMcpDeletePhase("list");
                  setMcpDeleting(null);
                }
              }}
            />
          )}
          {mcpDeletePhase === "deleting" && <Text color={COLORS.amber}>▶ removendo…</Text>}
          {mcpDeletePhase === "done" && mcpDeleteMsg && (
            <Text color={COLORS.green}>{mcpDeleteMsg}</Text>
          )}
          {mcpDeletePhase === "error" && mcpDeleteMsg && (
            <Text color={COLORS.red}>✗ {mcpDeleteMsg}</Text>
          )}
        </Box>
      )}
    </Box>
  );
}
