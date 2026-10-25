import React, { useState } from "react";
import { Box, Text, useApp, useInput } from "ink";
import { existsSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { Logo } from "./components/Logo.js";
import { StatusPanel } from "./components/StatusPanel.js";
import { ChatMessage } from "./components/ChatMessage.js";
import { InputBar } from "./components/InputBar.js";
import { DockerLogPanel } from "./components/DockerLogPanel.js";
import { SettingsPanel } from "./components/SettingsPanel.js";
import { ProjectsPanel } from "./components/ProjectsPanel.js";
import { useSystemStats } from "./hooks/useSystemStats.js";
import { useDockerLogs } from "./hooks/useDockerLogs.js";
import { useOrchestratorStream } from "./hooks/useOrchestratorStream.js";
import { useProcessManagerStatus } from "./hooks/useProcessManagerStatus.js";
import { useServiceStatus } from "./hooks/useServiceStatus.js";
import { useActiveProject } from "./hooks/useActiveProject.js";
import { ProcessStatusLine } from "./components/ProcessStatusLine.js";
import { COLORS, GRAPH_VIEWER_URL } from "./lib/constants.js";

export function App() {
  const { exit } = useApp();
  const [suspended, setSuspended] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showProjects, setShowProjects] = useState(false);
  const stats = useSystemStats(2000, suspended);
  const procStatus = useProcessManagerStatus(3000, suspended);
  const graphOnline = useServiceStatus(GRAPH_VIEWER_URL, 10000, suspended);
  // NEW (projetos): badge do escopo ativo (polling leve, 10s) — some no
  // modo geral e quando o orchestrator está offline. `projectOverride`
  // reflete na hora o switch feito no painel /projects (o polling está
  // congelado enquanto o painel está aberto); ao fechar o painel, o polling
  // retoma e volta a mandar.
  const polledProject = useActiveProject(10000, suspended || showProjects);
  const [projectOverride, setProjectOverride] = useState<string | null>(null);
  const activeProject = projectOverride ?? polledProject;
  const docker = useDockerLogs(suspended);
  const {
    messages,
    isStreaming,
    codeMode,
    setCodeMode,
    submit,
    clear,
    pendingImagePath,
    setPendingImagePath,
  } = useOrchestratorStream();

  const [showDocker, setShowDocker] = useState(false);
  const [imageError, setImageError] = useState<string | null>(null);

  // Comando "/image <caminho>" — anexa uma imagem local à PRÓXIMA pergunta
  // enviada. O caminho é lido e convertido para base64 aqui no cliente
  // (useOrchestratorStream), na hora do envio; o orchestrator nunca lê
  // esse arquivo do disco, só recebe os bytes já em base64. Habilita a
  // tool vision_objects no /execute. Qualquer outro texto vira pergunta normal.
  const handleSubmit = (value: string) => {
    const match = value.match(/^\/image\s+(.+)$/i);
    if (match) {
      const rawPath = match[1].trim().replace(/^["']|["']$/g, "");
      const fullPath = resolve(rawPath);
      if (!existsSync(fullPath) || !statSync(fullPath).isFile()) {
        setImageError(`Arquivo não encontrado: ${fullPath}`);
        return;
      }
      setImageError(null);
      setPendingImagePath(fullPath);
      return;
    }
    if (value.trim().toLowerCase() === "/settings") {
      setShowSettings(true);
      return;
    }
    // NEW (projetos): /projects abre o painel de escopos de memória.
    if (value.trim().toLowerCase() === "/projects") {
      setShowProjects(true);
      return;
    }
    submit(value);
  };

  // Equivalente aos BINDINGS do Textual (ctrl+k toggle code, ctrl+d docker,
  // ctrl+s settings, ctrl+p projetos, ctrl+l clear...)
  useInput((input, key) => {
    if (key.ctrl && input === "c") {
      exit();
      return;
    }
    if (key.ctrl && input === "k") {
      setCodeMode((prev) => !prev);
      return;
    }
    if (key.ctrl && input === "d") {
      setShowDocker((prev) => !prev);
      return;
    }
    if (key.ctrl && input === "s") {
      setShowSettings((prev) => !prev);
      return;
    }
    if (key.ctrl && input === "p") {
      setShowProjects((prev) => !prev);
      return;
    }
    if (key.ctrl && input === "l") {
      clear();
      return;
    }
  });

  return (
    <Box flexDirection="column" width="100%">
      <Logo />
      <StatusPanel stats={stats} activeProject={activeProject} />
      <Box marginTop={1}>
        <ProcessStatusLine status={procStatus} graphOnline={graphOnline} />
      </Box>

      <Box flexDirection="column" marginY={1}>
        {messages.length === 0 ? (
          <Text color={COLORS.dim}>
            {"  "}(/settings também lista/edita planos · /projects gerencia escopos de memória)
          </Text>
        ) : (
          messages.map((m) => <ChatMessage key={m.id} message={m} />)
        )}
      </Box>

      {showDocker && (
        <Box marginBottom={1}>
          <DockerLogPanel lines={docker.lines} status={docker.status} />
        </Box>
      )}

      {showSettings && (
        <Box marginBottom={1}>
          <SettingsPanel
            onClose={() => setShowSettings(false)}
            onSuspendChange={setSuspended}
          />
        </Box>
      )}

      {showProjects && (
        <Box marginBottom={1}>
          <ProjectsPanel
            onClose={() => {
              setShowProjects(false);
              setProjectOverride(null);
            }}
            onActiveChange={(slug) => setProjectOverride(slug)}
          />
        </Box>
      )}

      {imageError && (
        <Box marginBottom={1}>
          <Text color={COLORS.red}>⚠ {imageError}</Text>
        </Box>
      )}

      {pendingImagePath && (
        <Box marginBottom={1}>
          <Text color={COLORS.amber}>
            🖼 imagem anexada à próxima pergunta: {pendingImagePath}
          </Text>
        </Box>
      )}

      <InputBar
        disabled={isStreaming || suspended}
        codeMode={codeMode}
        focus={!showSettings && !showProjects && !suspended}
        onSubmit={handleSubmit}
      />
    </Box>
  );
}
