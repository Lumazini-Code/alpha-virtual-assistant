import React, { useCallback, useEffect, useRef, useState } from "react";
import { Box, Text, useInput, useStdin } from "ink";
import SelectInput from "ink-select-input";
import TextInput from "ink-text-input";
import { COLORS } from "../lib/constants.js";
import { logger } from "../lib/logger.js";
import {
  getReasoningEffort,
  setReasoningEffort,
  type ReasoningEffort,
} from "../lib/settings.js";
import { listPlans, updatePlan, type PlanEntry } from "../lib/plans.js";
import { spawnPlanRecorder } from "../lib/planRecorder.js";
import {
  listVoices,
  updateVoiceDescription,
  deleteVoice,
  registerVoice,
  type VoiceEntry,
} from "../lib/voices.js";
import { recordMic } from "../lib/voiceRecorder.js";
import { getPersona, setPersona } from "../lib/persona.js";
import { getWatchConfidence, setWatchConfidence } from "../lib/settings.js";
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

/** Rótulos do esforço de pensamento (menu de settings + mensagens). */
const REASONING_LABELS: Record<ReasoningEffort, string> = {
  auto: "AUTO (nativo do modelo)",
  low: "BAIXO (respostas mais rápidas)",
  medium: "MÉDIO (equilíbrio)",
  high: "ALTO (raciocínio profundo)",
  none: "NENHUM (sem thinking)",
};

type View = "menu" | "plans" | "edit" | "record" | "mcp" | "mcp-install" | "mcp-delete"
  | "voices" | "person-card" | "voice-edit" | "persona";
type VoiceRecordStage =
  | "name" // nome da pessoa
  | "description" // descrição opcional (Enter inicia a gravação)
  | "recording" // mic aberto — countdown
  | "uploading" // embedding (transcribe) + /voice-dict/write
  | "result" // sucesso + opção de gravar outro exemplo
  | "error";
type VoiceEditStage =
  | "menu" // ações sobre a pessoa
  | "description" // TextInput da descrição
  | "saving"
  | "saved"
  | "confirm-delete"
  | "deleting"
  | "delete-done"
  | "error";
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

/** Duração da gravação do cadastro de voz (~5s de fala é o mínimo que o
 * CAM++ produz embedding confiável — SPK_MIN_MS=600ms, ideal 3-6s). */
const RECORD_SECONDS = 5;

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

  // ── Cards de pessoa (PeopleDB unificado via memory :3000 + /voice/register
  // via orchestrator) — o cadastro de voz é a etapa final da criação do card ─
  const [voices, setVoices] = useState<VoiceEntry[]>([]);
  const [voicesLoading, setVoicesLoading] = useState(false);
  const [voicesError, setVoicesError] = useState<string | null>(null);
  const [voiceEditing, setVoiceEditing] = useState<VoiceEntry | null>(null);
  const [voiceEditStage, setVoiceEditStage] = useState<VoiceEditStage>("menu");
  const [voiceEditDesc, setVoiceEditDesc] = useState("");
  const [voiceEditMsg, setVoiceEditMsg] = useState<string | null>(null);
  const [recName, setRecName] = useState("");
  const [recDesc, setRecDesc] = useState("");
  const [recStage, setRecStage] = useState<VoiceRecordStage>("name");
  const [recMsg, setRecMsg] = useState<string | null>(null);
  const [recCountdown, setRecCountdown] = useState(0);

  const recInFlightRef = useRef(false);
  const recCountdownTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Persona (system prompt editável via GET/PUT /persona no orchestrator) ─
  const [persona, setPersonaState] = useState("");
  const [personaDefault, setPersonaDefault] = useState("");
  const [personaValue, setPersonaValue] = useState("");
  const [personaStage, setPersonaStage] = useState<"edit" | "saving" | "saved" | "error">("edit");
  const [personaMsg, setPersonaMsg] = useState<string | null>(null);

  // ── Memory Watch (vigia de confiança do LLM.py — toggle via orchestrator) ─
  // null = ainda não carregado (GET no mount do painel).
  const [watchEnabled, setWatchEnabled] = useState<boolean | null>(null);
  const [watchBusy, setWatchBusy] = useState(false);
  const [watchMsg, setWatchMsg] = useState<string | null>(null);

  // ── Esforço de pensamento (reasoning.effort — ciclo via orchestrator) ─────
  const [reasoningEffort, setReasoningEffortState] = useState<ReasoningEffort | null>(null);
  const [reasoningBusy, setReasoningBusy] = useState(false);
  const [reasoningMsg, setReasoningMsg] = useState<string | null>(null);

  const backTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const { stdin, isRawModeSupported, setRawMode } = useStdin() as {
    stdin?: { pause?: () => void; resume?: () => void };
    isRawModeSupported?: boolean;
    setRawMode?: (isRawMode: boolean) => void;
  };

  useEffect(() => {
    return () => {
      if (backTimerRef.current) clearTimeout(backTimerRef.current);
      if (recCountdownTimerRef.current) clearInterval(recCountdownTimerRef.current);
    };
  }, []);

  // Carrega o estado do Memory Watch e do reasoning effort quando o painel
  // abre (2 GETs leves).
  useEffect(() => {
    let cancelled = false;
    getWatchConfidence()
      .then((info) => {
        if (!cancelled) setWatchEnabled(info.enabled);
      })
      .catch((err) => {
        const msg = err instanceof Error ? err.message : String(err);
        logger.warn("settings: GET /settings/watch-confidence falhou", { message: msg });
        if (!cancelled) setWatchMsg(`orchestrator offline ou sem resposta: ${msg}`);
      });
    getReasoningEffort()
      .then((info) => {
        if (!cancelled) setReasoningEffortState(info.effort);
      })
      .catch((err) => {
        const msg = err instanceof Error ? err.message : String(err);
        logger.warn("settings: GET /settings/reasoning-effort falhou", { message: msg });
        if (!cancelled) setReasoningMsg(`orchestrator offline ou sem resposta: ${msg}`);
      });
    return () => {
      cancelled = true;
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

  /** Liga/desliga o Memory Watch (toggle imediato via PUT no orchestrator).
   * O flag vale já no próximo turno — o orchestrator injeta watch_confidence
   * nos requests de chat ao LLM.py. */
  const toggleWatch = useCallback(async () => {
    if (watchBusy) return;
    const next = !(watchEnabled ?? false);
    setWatchBusy(true);
    setWatchMsg(null);
    try {
      const info = await setWatchConfidence(next);
      setWatchEnabled(info.enabled);
      setWatchMsg(
        info.enabled
          ? "✓ Memory Watch ativado — o vigia de confiança já vale no próximo turno"
          : "✓ Memory Watch desativado — já vale no próximo turno",
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.warn("settings: PUT /settings/watch-confidence falhou", { message: msg });
      setWatchMsg(`✗ ${msg}`);
    } finally {
      setWatchBusy(false);
    }
  }, [watchBusy, watchEnabled]);

  /** Cicla o esforço de pensamento: auto → baixo → médio → alto → nenhum →
   * auto (PUT imediato — vale no próximo turno, sem restart). "auto" não
   * envia o campo: o orchestrator usa settings.json/env/nativo. */
  const cycleReasoningEffort = useCallback(async () => {
    if (reasoningBusy) return;
    const order: ReasoningEffort[] = ["auto", "low", "medium", "high", "none"];
    const next = order[(order.indexOf(reasoningEffort ?? "auto") + 1) % order.length];
    setReasoningBusy(true);
    setReasoningMsg(null);
    try {
      await setReasoningEffort(next);
      setReasoningEffortState(next);
      setReasoningMsg(`✓ esforço de pensamento: ${REASONING_LABELS[next]}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.warn("settings: PUT /settings/reasoning-effort falhou", { message: msg });
      setReasoningMsg(`✗ falhou: ${msg}`);
    } finally {
      setReasoningBusy(false);
    }
  }, [reasoningBusy, reasoningEffort]);

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

  // ── Cards de pessoa: listar / criar (card + gravação de voz) / editar ──
  const loadVoices = useCallback(async () => {
    setVoicesLoading(true);
    setVoicesError(null);
    try {
      setVoices(await listVoices());
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.warn("settings: GET /voice-dict/list falhou", { message: msg });
      setVoicesError(`memory offline ou sem resposta: ${msg}`);
    } finally {
      setVoicesLoading(false);
    }
  }, []);

  const enterVoices = useCallback(() => {
    setView("voices");
    void loadVoices();
  }, [loadVoices]);

  // ── Persona: carregar (GET) / salvar (PUT /persona) ──────────────────────
  const enterPersona = useCallback(async () => {
    setView("persona");
    setPersonaStage("edit");
    setPersonaMsg(null);
    try {
      const info = await getPersona();
      setPersonaState(info.persona);
      setPersonaDefault(info.default);
      setPersonaValue(info.persona);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.warn("settings: GET /persona falhou", { message: msg });
      setPersonaStage("error");
      setPersonaMsg(`orchestrator offline ou sem resposta: ${msg}`);
    }
  }, []);

  const savePersona = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) {
      setPersonaStage("error");
      setPersonaMsg("a persona não pode ficar vazia");
      return;
    }
    setPersonaStage("saving");
    setPersonaMsg(null);
    try {
      const info = await setPersona(trimmed);
      setPersonaState(info.persona);
      setPersonaValue(info.persona);
      setPersonaStage("saved");
      setPersonaMsg("persona salva — vale a partir do próximo turno");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.warn("settings: PUT /persona falhou", { message: msg });
      setPersonaStage("error");
      setPersonaMsg(msg);
    }
  }, []);

  const startVoiceEdit = useCallback((voice: VoiceEntry) => {
    setVoiceEditing(voice);
    setVoiceEditDesc(voice.description);
    setVoiceEditStage("menu");
    setVoiceEditMsg(null);
    setView("voice-edit");
  }, []);

  const saveVoiceEdit = useCallback(async () => {
    if (!voiceEditing) return;
    setVoiceEditStage("saving");
    try {
      await updateVoiceDescription(voiceEditing.person_id, voiceEditDesc);
      logger.info("settings: descrição de voz salva", { personId: voiceEditing.person_id });
      setVoiceEditMsg("✓ descrição salva");
      setVoiceEditStage("saved");
      void loadVoices();
      if (backTimerRef.current) clearTimeout(backTimerRef.current);
      backTimerRef.current = setTimeout(() => {
        setVoiceEditing(null);
        setVoiceEditStage("menu");
        setVoiceEditMsg(null);
        setView("voices");
      }, 1200);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.warn("settings: PATCH /voice-dict/{id} falhou", { message: msg });
      setVoiceEditMsg(msg);
      setVoiceEditStage("error");
    }
  }, [voiceEditing, voiceEditDesc, loadVoices]);

  const confirmVoiceDelete = useCallback(async () => {
    if (!voiceEditing) return;
    setVoiceEditStage("deleting");
    try {
      await deleteVoice(voiceEditing.person_id);
      logger.info("settings: voz removida", { personId: voiceEditing.person_id });
      setVoiceEditMsg(`✓ ${voiceEditing.person_name} removida (embeddings apagados do DB + FAISS)`);
      setVoiceEditStage("delete-done");
      void loadVoices();
      if (backTimerRef.current) clearTimeout(backTimerRef.current);
      backTimerRef.current = setTimeout(() => {
        setVoiceEditing(null);
        setVoiceEditStage("menu");
        setVoiceEditMsg(null);
        setView("voices");
      }, 1800);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.warn("settings: DELETE /voice-dict/{id} falhou", { message: msg });
      setVoiceEditMsg(msg);
      setVoiceEditStage("error");
    }
  }, [voiceEditing, loadVoices]);

  /** Etapa final da criação do card: grava RECORD_SECONDS de microfone (host)
   * e envia o PCM em base64 ao orchestrator (/voice/register → transcribe/CAM++ →
   * memory /voice-dict/write). O embedding de voz fica vinculado ao card da
   * pessoa (person_id) junto com nome/descrição já preenchidos. Gravação roda
   * em processo filho com stdout capturado — nada toca no terminal da TUI. */
  const startVoiceRecording = useCallback(async () => {
    if (recInFlightRef.current) return;
    recInFlightRef.current = true;
    setRecStage("recording");
    setRecMsg(null);
    setRecCountdown(RECORD_SECONDS);
    if (recCountdownTimerRef.current) clearInterval(recCountdownTimerRef.current);
    recCountdownTimerRef.current = setInterval(() => {
      setRecCountdown((c) => (c > 1 ? c - 1 : 1));
    }, 1000);

    const res = await recordMic(RECORD_SECONDS);
    if (recCountdownTimerRef.current) clearInterval(recCountdownTimerRef.current);

    if (!res.ok || !res.audioB64) {
      recInFlightRef.current = false;
      setRecMsg(`✗ ${res.error ?? "gravação falhou"}`);
      setRecStage("error");
      return;
    }

    setRecStage("uploading");
    setRecMsg(null);
    try {
      const result = await registerVoice({
        person_name: recName.trim(),
        description: recDesc.trim(),
        audio_b64: res.audioB64,
        sample_rate: 16000,
      });
      const who = result.new_person
        ? "pessoa nova cadastrada"
        : "exemplo adicional (reconhecimento fica mais preciso)";
      const where = result.memory_id ? ` — descrição linkada à memória LT #${result.memory_id}` : "";
      setRecMsg(`✓ embedding gravado (id ${result.embedding_id ?? "?"}) — ${who}${where}`);
      setRecStage("result");
      void loadVoices();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.warn("settings: POST /voice/register falhou", { message: msg });
      setRecMsg(`✗ ${msg}`);
      setRecStage("error");
    } finally {
      recInFlightRef.current = false;
    }
  }, [recName, recDesc, loadVoices]);

  const resetVoiceWizard = useCallback(() => {
    setRecName("");
    setRecDesc("");
    setRecStage("name");
    setRecMsg(null);
    setRecCountdown(0);
  }, []);

  /** Abre o wizard de criação de card de pessoa (voz embutida no cadastro). */
  const startPersonCard = useCallback(() => {
    resetVoiceWizard();
    setView("person-card");
  }, [resetVoiceWizard]);

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
    } else if (view === "voices") setView("menu");
    else if (view === "persona") {
      if (personaStage === "saving") return;
      setView("menu");
    } else if (view === "person-card") {
      // gravação/upload em andamento não são interrompíveis
      if (recStage === "recording" || recStage === "uploading") return;
      resetVoiceWizard();
      setView("voices");
    } else if (view === "voice-edit") {
      if (voiceEditStage === "saving" || voiceEditStage === "deleting") return;
      if (voiceEditStage === "description" || voiceEditStage === "confirm-delete") {
        setVoiceEditStage("menu");
        setVoiceEditMsg(null);
      } else {
        setVoiceEditing(null);
        setVoiceEditStage("menu");
        setVoiceEditMsg(null);
        setView("voices");
      }
    }
  });

  // ── Itens de menu / lista ────────────────────────────────────────────────
  const menuItems = [
    { label: "▶  Gravar novo plano (plan_recorder.py)", value: "record" },
    { label: "≡  Planos existentes — ver/editar", value: "plans" },
    { label: "👤  Cards de Pessoa — cadastro unificado (voz + dados)", value: "voices" },
    { label: "⤓  Servidores MCP — instalar/gerenciar", value: "mcp" },
    { label: "✎  Persona — editar o system prompt", value: "persona" },
    {
      label:
        `👁  Memory Watch (recuperação contínua de memória): ` +
        (watchEnabled === null ? "…" : watchEnabled ? "ATIVADO" : "DESATIVADO"),
      value: "watch",
    },
    {
      label:
        `💭  Esforço de pensamento: ` +
        (reasoningEffort === null ? "…" : REASONING_LABELS[reasoningEffort]),
      value: "reasoning",
    },
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
              } else if (item.value === "voices") {
                enterVoices();
              } else if (item.value === "persona") {
                void enterPersona();
              } else if (item.value === "watch") {
                void toggleWatch();
              } else if (item.value === "reasoning") {
                void cycleReasoningEffort();
              } else {
                onClose();
              }
            }}
          />
          {watchBusy && <Text color={COLORS.amber}>▶ alternando Memory Watch…</Text>}
          {watchMsg && (
            <Text
              color={
                watchMsg.startsWith("✓") ? COLORS.green : watchMsg.startsWith("✗") ? COLORS.red : COLORS.dim
              }
            >
              {watchMsg}
            </Text>
          )}
          {reasoningBusy && <Text color={COLORS.amber}>▶ alternando esforço de pensamento…</Text>}
          {reasoningMsg && (
            <Text
              color={
                reasoningMsg.startsWith("✓") ? COLORS.green : reasoningMsg.startsWith("✗") ? COLORS.red : COLORS.dim
              }
            >
              {reasoningMsg}
            </Text>
          )}
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

      {view === "persona" && (
        <Box flexDirection="column" marginTop={1}>
          <Text color={COLORS.gray}>
            persona: 1ª linha do system prompt (o orchestrator anexa a
            cláusula de tools). Enter salva · Esc volta
          </Text>
          {personaStage === "error" && <Text color={COLORS.red}>✗ {personaMsg}</Text>}
          {personaStage === "saved" && <Text color={COLORS.green}>✓ {personaMsg}</Text>}
          {personaStage === "saving" && <Text color={COLORS.dim}>salvando…</Text>}
          {personaStage === "edit" && (
            <>
              <Text>persona: </Text>
              <TextInput
                value={personaValue}
                onChange={setPersonaValue}
                onSubmit={(value) => void savePersona(value)}
              />
              <Text color={COLORS.dim}>
                atual: {truncate(persona, 64)}
              </Text>
              {personaDefault && persona !== personaDefault && (
                <Text color={COLORS.dim}>default: {truncate(personaDefault, 64)}</Text>
              )}
            </>
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

      {view === "voices" && (
        <Box flexDirection="column" marginTop={1}>
          <Text color={COLORS.gray}>
            dicionário de voz (CAM++) · ↑/↓ navega · Enter gerencia · Esc volta
          </Text>
          {voicesLoading && <Text color={COLORS.dim}>carregando vozes…</Text>}
          {voicesError && <Text color={COLORS.red}>✗ {voicesError}</Text>}
          <SelectInput<VoiceEntry | string>
            items={[
              { label: "＋  Criar card de pessoa (grava ~5s de voz no microfone)", value: "record" },
              ...voices.map((v) => ({
                label:
                  `${v.person_name} · ${v.examples_count} exemplo(s)` +
                  (v.description ? ` — ${truncate(v.description, 42)}` : ""),
                value: v,
              })),
              { label: "↺  Recarregar", value: "reload" },
              { label: "↩  Voltar (Esc)", value: "back" },
            ]}
            onSelect={(item) => {
              if (item.value === "record") {
                startPersonCard();
              } else if (item.value === "reload") {
                void loadVoices();
              } else if (typeof item.value !== "string") {
                startVoiceEdit(item.value);
              }
            }}
          />
          {!voicesLoading && !voicesError && voices.length === 0 && (
            <Text color={COLORS.dim}>nenhum card de pessoa — crie o primeiro acima</Text>
          )}
        </Box>
      )}

      {view === "person-card" && (
        <Box flexDirection="column" marginTop={1}>
          <Text color={COLORS.gray}>
            cadastro unificado do card de pessoa: nome + descrição + voz.
            o embedding (CAM++ 192 dims) é gerado on-demand pelo módulo
            transcribe (host) e persistido na memory via /voice-dict/write
          </Text>
          {recStage === "name" && (
            <>
              <Text>nome do card: </Text>
              <TextInput
                value={recName}
                onChange={setRecName}
                onSubmit={() => {
                  if (recName.trim()) setRecStage("description");
                }}
                placeholder="ex.: eu / Felipe / Ana"
              />
              <Text color={COLORS.dim}>Enter p/ descrição · Esc volta</Text>
            </>
          )}
          {recStage === "description" && (
            <>
              <Text color={COLORS.dim}>card: {recName}</Text>
              <Text>descrição do card (opcional): </Text>
              <TextInput
                value={recDesc}
                onChange={setRecDesc}
                onSubmit={() => void startVoiceRecording()}
                placeholder={`quem é (Enter GRAVA ${RECORD_SECONDS}s de voz e fecha o card)`}
              />
              <Text color={COLORS.dim}>Enter inicia a gravação de voz (etapa final do card) — fale continuamente · Esc volta</Text>
            </>
          )}
          {recStage === "recording" && (
            <Text color={COLORS.amber}>
              ● gravando a voz do card… fale naturalmente ({recCountdown}s restantes) — Esc desabilitado
            </Text>
          )}
          {recStage === "uploading" && (
            <Text color={COLORS.amber}>▶ gerando embedding (transcribe/CAM++) e salvando o card de pessoa na memory…</Text>
          )}
          {recStage === "result" && recMsg && (
            <>
              <Text color={COLORS.green}>{recMsg}</Text>
              <Text color={COLORS.dim}>
                Esc volta · crie o mesmo card de novo p/ adicionar outro
                exemplo de voz (recomendado: ambientes/altura de voz diferentes)
              </Text>
            </>
          )}
          {recStage === "error" && recMsg && (
            <Text color={COLORS.red}>{recMsg}</Text>
          )}
        </Box>
      )}

      {view === "voice-edit" && voiceEditing && (
        <Box flexDirection="column" marginTop={1}>
          <Text color={COLORS.gray}>
            {`#${voiceEditing.person_id} ${voiceEditing.person_name} · ${voiceEditing.examples_count} exemplo(s) · ${voiceEditing.access_count} acesso(s) · source ${voiceEditing.source}`}
          </Text>
          {voiceEditStage === "menu" && (
            <SelectInput
              items={[
                { label: "✎  Editar descrição", value: "edit" },
                { label: `✗  Remover "${voiceEditing.person_name}" + embeddings (DB + FAISS)`, value: "delete" },
                { label: "↩  Voltar (Esc)", value: "back" },
              ]}
              onSelect={(item) => {
                if (item.value === "edit") {
                  setVoiceEditDesc(voiceEditing.description);
                  setVoiceEditStage("description");
                } else if (item.value === "delete") {
                  setVoiceEditMsg(null);
                  setVoiceEditStage("confirm-delete");
                }
              }}
            />
          )}
          {voiceEditStage === "description" && (
            <>
              <Text>descrição: </Text>
              <TextInput
                value={voiceEditDesc}
                onChange={setVoiceEditDesc}
                onSubmit={() => void saveVoiceEdit()}
                placeholder="quem é essa pessoa (Enter salva)"
              />
              <Text color={COLORS.dim}>{`Enter salva (PATCH /voice-dict/${voiceEditing.person_id}) · Esc volta`}</Text>
            </>
          )}
          {voiceEditStage === "saving" && <Text color={COLORS.amber}>salvando…</Text>}
          {voiceEditStage === "saved" && voiceEditMsg && (
            <Text color={COLORS.green}>{voiceEditMsg}</Text>
          )}
          {voiceEditStage === "confirm-delete" && (
            <>
              <Text color={COLORS.red}>
                {`remover "${voiceEditing.person_name}" e ${voiceEditing.examples_count} embedding(s)?`}
              </Text>
              <Text color={COLORS.dim}>a memória de longo prazo linkada (se houver) NÃO é removida</Text>
              <SelectInput
                items={[
                  { label: "✗  Sim, remover", value: true as const },
                  { label: "↩  Cancelar (Esc)", value: false as const },
                ]}
                onSelect={(item) => {
                  if (item.value) void confirmVoiceDelete();
                  else {
                    setVoiceEditStage("menu");
                    setVoiceEditMsg(null);
                  }
                }}
              />
            </>
          )}
          {voiceEditStage === "deleting" && <Text color={COLORS.amber}>▶ removendo…</Text>}
          {voiceEditStage === "delete-done" && voiceEditMsg && (
            <Text color={COLORS.green}>{voiceEditMsg}</Text>
          )}
          {voiceEditStage === "error" && voiceEditMsg && (
            <Text color={COLORS.red}>✗ {voiceEditMsg}</Text>
          )}
        </Box>
      )}
    </Box>
  );
}
