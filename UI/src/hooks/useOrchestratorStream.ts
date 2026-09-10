import { useCallback, useRef, useState, type MutableRefObject } from "react";
import { existsSync, readFileSync } from "node:fs";
import { ORCHESTRATOR_URL } from "../lib/constants.js";
import { ensureDockerStarted } from "../lib/processManager.js";
import { waitForServicesReady } from "../lib/waitForServices.js";
import { logger } from "../lib/logger.js";
import { parseSseStream, tryParseJson } from "../lib/sse.js";
import type {
  ChatMessageState,
  CoTStep,
  ErrorPayload,
  PlanPayload,
  StepDonePayload,
  StepStartPayload,
  ToolCallPayload,
} from "../lib/types.js";

function newAssistantMessage(id: string): ChatMessageState {
  return { id, role: "assistant", text: "", streaming: true };
}

export function useOrchestratorStream() {
  const [messages, setMessages] = useState<ChatMessageState[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [codeMode, setCodeMode] = useState(false);
  const [pendingImagePath, setPendingImagePathState] = useState<string | null>(null);
  const sessionIdRef = useRef<string | undefined>(undefined);
  const pendingImagePathRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const setPendingImagePath = useCallback((path: string | null) => {
    pendingImagePathRef.current = path;
    setPendingImagePathState(path);
  }, []);

  const updateLast = useCallback(
    (fn: (msg: ChatMessageState) => ChatMessageState) => {
      setMessages((prev) => {
        if (prev.length === 0) return prev;
        const next = [...prev];
        next[next.length - 1] = fn(next[next.length - 1]);
        return next;
      });
    },
    [],
  );

  const submit = useCallback(
    async (input: string) => {
      if (isStreaming || !input.trim()) return;

      const userMsg: ChatMessageState = {
        id: crypto.randomUUID(),
        role: "user",
        text: input,
        streaming: false,
        imagePath: pendingImagePathRef.current ?? undefined,
      };
      const aiMsg = newAssistantMessage(crypto.randomUUID());
      setMessages((prev) => [...prev, userMsg, aiMsg]);
      setIsStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      // Auto-start do container docker a cada pergunta. O process manager
      // Rust (porta 9001) já ignora a chamada se o processo correspondente
      // estiver ativo, então repetir isso aqui é seguro e barato. O app
      // não usa mais llama-server local — o modelo é resolvido pelo
      // orchestrator. Best-effort: um aviso aqui não impede a pergunta de
      // seguir para o orchestrator.
      logger.info("submit: pergunta recebida", { codeMode, hasImage: !!pendingImagePathRef.current });

      const procResult = await ensureDockerStarted();
      if (procResult.warning) {
        logger.warn("submit: ensureProcessesStarted retornou aviso", { warning: procResult.warning });
        updateLast((m) => ({ ...m, error: procResult.warning }));
      }

      // Só dispara a pergunta depois que docker e orchestrator responderem
      // de verdade — /docker/start acima só dispara o boot, não espera ele
      // terminar. Sem isso, a primeira pergunta de uma sessão fria costuma
      // cair em "Orchestrator offline" porque o /execute chega antes dos
      // serviços estarem prontos.
      updateLast((m) => ({ ...m, text: "aguardando docker e orchestrator subirem…" }));
      const readyResult = await waitForServicesReady((elapsedMs, dockerReady, orchestratorReady) => {
        logger.debug("submit: ainda aguardando serviços", { elapsedMs, dockerReady, orchestratorReady });
      });
      if (readyResult.timedOut) {
        const timeoutMsg = `timeout aguardando docker/orchestrator subirem (docker=${readyResult.dockerReady}, orchestrator=${readyResult.orchestratorReady})`;
        logger.warn(`submit: ${timeoutMsg}`);
        updateLast((m) => ({
          ...m,
          error: procResult.warning ? `${procResult.warning} | ${timeoutMsg}` : timeoutMsg,
        }));
      }
      updateLast((m) => ({ ...m, text: "" }));

      // Corpos exatos dos Pydantic models ExecuteRequest / AlphaCodeRequest
      const endpoint = codeMode ? "/code" : "/execute";
      const imagePath = pendingImagePathRef.current;

      // A conversão para base64 acontece AQUI, no cliente (a TUI roda na
      // máquina do usuário, não no docker/orchestrator) — é por isso que
      // esse comando funciona independente de onde o orchestrator está
      // rodando: ele nunca recebe um caminho de arquivo, só os bytes já
      // codificados. Reconfere a existência do arquivo na hora de ler
      // (não só no /image) — entre o /image e essa pergunta pode ter
      // passado bastante tempo (inclusive a espera de docker/orchestrator
      // acima), então um caminho válido na hora do /image pode ter sido
      // movido/apagado nesse meio tempo.
      let imageBase64: string | null = null;
      if (imagePath) {
        if (!existsSync(imagePath)) {
          logger.warn("submit: imagem anexada não existe mais no disco", { imagePath });
          updateLast((m) => ({
            ...m,
            error: `Imagem não encontrada no envio: ${imagePath} (verifique se o arquivo ainda existe)`,
          }));
        } else {
          try {
            imageBase64 = readFileSync(imagePath).toString("base64");
            // NUNCA logar imageBase64 (nem um trecho) — só metadados.
            logger.info("submit: imagem convertida para base64", {
              imagePath,
              bytesBase64: imageBase64.length,
            });
          } catch (err) {
            const message = err instanceof Error ? err.message : "erro desconhecido";
            logger.warn("submit: falha ao ler/converter imagem para base64", { imagePath, message });
            updateLast((m) => ({
              ...m,
              error: `Falha ao ler a imagem para envio: ${imagePath} (${message})`,
            }));
          }
        }
      }

      const body = codeMode
        ? {
            task: input,
            session_id: sessionIdRef.current ?? null,
            project_dir: null,
            max_steps: 25,
            temperature: 0.3,
            model_override: null,
            stream: true,
          }
        : {
            input,
            session_id: sessionIdRef.current ?? null,
            voice: "M1",
            lang: "pt",
            tts: true,
            image_base64: imageBase64,
            search_pdfs: false,
            stream: true,
          };

      // A imagem é anexada só a essa pergunta (equivalente a "um turno") —
      // depois disso, novas perguntas vão sem imagem até um novo /image.
      if (imagePath) setPendingImagePath(null);

      // Aviso visível (não só no log): o endpoint /code (modo código) não
      // tem campo image_base64 no Pydantic model, então uma imagem anexada
      // nesse modo seria descartada em silêncio. Melhor avisar aqui do
      // que deixar o usuário achando que a imagem foi enviada.
      if (imageBase64 && codeMode) {
        logger.warn("submit: imagem anexada foi descartada — /code não suporta image_base64", {
          imagePath,
        });
        updateLast((m) => ({
          ...m,
          error: "Imagem anexada foi ignorada: modo código (/code) não suporta imagens. Use o modo normal (Ctrl+K) para perguntas com /image.",
        }));
      }

      // Log da requisição SEM o base64 — só metadados. safeStringify do
      // logger faria JSON.stringify(body) inteiro (potencialmente vários
      // MB de base64 dentro de debug.log) se passássemos "body" direto.
      logger.info("submit: enviando pergunta ao orchestrator", {
        endpoint,
        url: `${ORCHESTRATOR_URL}${endpoint}`,
        body: { ...body, image_base64: imageBase64 ? `<${imageBase64.length} chars omitidos>` : null },
      });

      try {
        const res = await fetch(`${ORCHESTRATOR_URL}${endpoint}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        if (!res.ok || !res.body) {
          throw new Error(`HTTP ${res.status}`);
        }

        for await (const frame of parseSseStream(res.body)) {
          logger.debug("submit: frame SSE recebido", { event: frame.event });
          handleFrame(frame, updateLast, sessionIdRef);
        }

        logger.info("submit: stream concluído");
      } catch (err) {
        const message = err instanceof Error ? err.message : "erro desconhecido";
        logger.error("submit: falha ao falar com o orchestrator", { message });
        updateLast((m) => ({
          ...m,
          streaming: false,
          error: `Orchestrator offline — ${ORCHESTRATOR_URL} (${message})`,
        }));
      } finally {
        setIsStreaming(false);
        abortRef.current = null;
      }
    },
    [isStreaming, codeMode, updateLast],
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const clear = useCallback(() => {
    setMessages([]);
    sessionIdRef.current = undefined;
  }, []);

  return {
    messages,
    isStreaming,
    codeMode,
    setCodeMode,
    submit,
    cancel,
    clear,
    pendingImagePath,
    setPendingImagePath,
  };
}

/**
 * Traduz cada frame SSE em uma atualização da última mensagem, seguindo
 * exatamente o que orchestrator.py manda em _execute_stream_generator e
 * _alpha_code_stream_generator:
 *
 *   meta        JSON  {execution_id, session_id, ...}
 *   plan        JSON  (só /code)                       PlanPayload
 *   tool_call   JSON  {step, tool, arguments?}          ToolCallPayload
 *   step_start  JSON  (só /code)                        StepStartPayload
 *   step_done   JSON  {step, executor, success, ...}    StepDonePayload
 *   result      texto puro (saída intermediária de uma tool)
 *   reasoning   texto puro (só /code)
 *   delta       texto puro — a resposta final INTEIRA (não incremental)
 *   error       JSON  {error, fatal?}
 *   done        JSON  ExecuteDonePayload | CodeDonePayload
 */
function handleFrame(
  frame: { event: string; raw: string },
  updateLast: (fn: (m: ChatMessageState) => ChatMessageState) => void,
  sessionIdRef: MutableRefObject<string | undefined>,
) {
  switch (frame.event) {
    case "meta": {
      const data = tryParseJson<{ session_id?: string }>(frame.raw);
      if (typeof data === "object" && data.session_id) {
        sessionIdRef.current = data.session_id;
      }
      break;
    }

    case "plan": {
      const data = tryParseJson<PlanPayload>(frame.raw);
      if (typeof data !== "object") break;
      const steps: CoTStep[] = (data.steps ?? []).map((s) => ({
        step: safeStep(s.step ?? s.id),
        executor: String(s.executor ?? s.name ?? ""),
        action: String(s.action ?? ""),
        dependsOn: s.depends_on ?? [],
        status: "pending",
      }));
      updateLast((m) => ({
        ...m,
        plan: { steps, fromCache: false, dynamic: false },
      }));
      break;
    }

    // /execute: {"step": turn, "tool": name} — sem step_start explícito,
    // então tool_call já cria a linha direto em "running".
    case "tool_call": {
      const data = tryParseJson<ToolCallPayload>(frame.raw);
      if (typeof data !== "object") break;
      const stepNum = safeStep(data.step);
      updateLast((m) => {
        const existing = m.plan?.steps ?? [];
        if (existing.some((s) => s.step === stepNum)) return m;
        const argsPreview = data.arguments
          ? JSON.stringify(data.arguments).slice(0, 90)
          : "";
        const newStep: CoTStep = {
          step: stepNum,
          executor: data.tool,
          action: argsPreview,
          dependsOn: [],
          status: "running",
        };
        return {
          ...m,
          plan: {
            steps: [...existing, newStep],
            fromCache: m.plan?.fromCache ?? false,
            dynamic: true,
          },
        };
      });
      break;
    }

    // Só existe em /code — reforça/atualiza a mesma linha criada por tool_call.
    case "step_start": {
      const data = tryParseJson<StepStartPayload>(frame.raw);
      if (typeof data !== "object") break;
      const stepNum = safeStep(data.step);
      updateLast((m) => {
        const existing = m.plan?.steps ?? [];
        const idx = existing.findIndex((s) => s.step === stepNum);
        if (idx === -1) {
          const newStep: CoTStep = {
            step: stepNum,
            executor: data.executor,
            action: data.action,
            dependsOn: [],
            status: "running",
          };
          return {
            ...m,
            plan: {
              steps: [...existing, newStep],
              fromCache: m.plan?.fromCache ?? false,
              dynamic: true,
            },
          };
        }
        const next = [...existing];
        next[idx] = { ...next[idx], action: data.action, status: "running" };
        return { ...m, plan: m.plan && { ...m.plan, steps: next } };
      });
      break;
    }

    case "step_done": {
      const data = tryParseJson<StepDonePayload>(frame.raw);
      if (typeof data !== "object") break;
      const stepNum = safeStep(data.step);
      updateLast((m) => ({
        ...m,
        plan: m.plan && {
          ...m.plan,
          steps: m.plan.steps.map((s) =>
            s.step === stepNum
              ? {
                  ...s,
                  status: data.success ? "done" : "error",
                  latencyMs: data.latency_ms,
                  error: data.error ?? undefined,
                }
              : s,
          ),
        },
      }));
      break;
    }

    // Saída intermediária de uma tool — texto puro. Não é a resposta final
    // (essa vem em "delta"), mas ajuda a acompanhar o progresso.
    case "result": {
      // Deixado como no-op de exibição por padrão para não poluir o chat
      // com saídas intermediárias de tool; descomente se quiser mostrá-las:
      // updateLast((m) => ({ ...m, reasoning: (m.reasoning ?? "") + `\n${frame.raw}` }));
      break;
    }

    case "reasoning": {
      // texto puro, só em /code
      updateLast((m) => ({ ...m, reasoning: (m.reasoning ?? "") + frame.raw }));
      break;
    }

    // A resposta final inteira, mandada de uma vez só (apesar do nome).
    case "delta": {
      updateLast((m) => ({ ...m, text: frame.raw }));
      break;
    }

    case "error": {
      const data = tryParseJson<ErrorPayload>(frame.raw);
      const message = typeof data === "object" ? data.error : frame.raw;
      updateLast((m) => ({ ...m, streaming: false, error: message }));
      break;
    }

    case "done": {
      updateLast((m) => ({ ...m, streaming: false }));
      break;
    }

    default:
      break;
  }
}

function safeStep(val: unknown): number {
  if (typeof val === "number") return val;
  if (typeof val === "string") {
    const n = parseInt(val, 10);
    return Number.isNaN(n) ? 0 : n;
  }
  return 0;
}
