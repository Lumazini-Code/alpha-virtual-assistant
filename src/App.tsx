import React, { useState } from "react";
import { Box, Text, useApp, useInput } from "ink";
import { existsSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { Logo } from "./components/Logo.js";
import { StatusPanel } from "./components/StatusPanel.js";
import { ChatMessage } from "./components/ChatMessage.js";
import { InputBar } from "./components/InputBar.js";
import { DockerLogPanel } from "./components/DockerLogPanel.js";
import { useSystemStats } from "./hooks/useSystemStats.js";
import { useDockerLogs } from "./hooks/useDockerLogs.js";
import { useOrchestratorStream } from "./hooks/useOrchestratorStream.js";
import { useProcessManagerStatus } from "./hooks/useProcessManagerStatus.js";
import { ProcessStatusLine } from "./components/ProcessStatusLine.js";
import { COLORS } from "./lib/constants.js";

export function App() {
  const { exit } = useApp();
  const stats = useSystemStats();
  const procStatus = useProcessManagerStatus();
  const docker = useDockerLogs();
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
    submit(value);
  };

  // Equivalente aos BINDINGS do Textual (ctrl+k toggle code, ctrl+d docker, ctrl+l clear...)
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
    if (key.ctrl && input === "l") {
      clear();
      return;
    }
  });

  return (
    <Box flexDirection="column" width="100%">
      <Logo />
      <StatusPanel stats={stats} />
      <Box marginTop={1}>
        <ProcessStatusLine status={procStatus} />
      </Box>

      <Box flexDirection="column" marginY={1}>
        {messages.length === 0 ? (
          <Text color={COLORS.dim}>
            Digite uma mensagem para começar. Ctrl+K alterna modo código,
            Ctrl+D mostra/esconde logs do Docker, Ctrl+L limpa o chat.
            {"\n"}Use /image caminho/para/arquivo.png para anexar uma
            imagem à próxima pergunta (recurso de vision).
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

      <InputBar disabled={isStreaming} codeMode={codeMode} onSubmit={handleSubmit} />
    </Box>
  );
}
