// Alpha AI (AVA) — TUI principal do cliente (composição dos painéis e atalhos de teclado)
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
import React, { useState, useCallback } from "react";
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
import { FilePickerModal } from "./components/FilePickerModal.js";
import { useSystemStats } from "./hooks/useSystemStats.js";
import { useDockerLogs } from "./hooks/useDockerLogs.js";
import { useOrchestratorStream } from "./hooks/useOrchestratorStream.js";
import { useProcessManagerStatus } from "./hooks/useProcessManagerStatus.js";
import { useServiceStatus } from "./hooks/useServiceStatus.js";
import { useActiveProject } from "./hooks/useActiveProject.js";
import { ProcessStatusLine } from "./components/ProcessStatusLine.js";
import { COLORS, GRAPH_VIEWER_URL } from "./lib/constants.js";
import {
  isImageFile,
  isDocumentFile,
  imageToBase64,
  extractTextContent,
  type AttachedFile,
} from "./lib/fileAttach.js";

export function App() {
  const { exit } = useApp();
  const [suspended, setSuspended] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showProjects, setShowProjects] = useState(false);
  const stats = useSystemStats(2000, suspended);
  const procStatus = useProcessManagerStatus(3000, suspended);
  const graphOnline = useServiceStatus(GRAPH_VIEWER_URL, 10000, suspended);
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
  // Pensamento (💭) expandido globalmente — alterna com ctrl+e. Colapsado,
  // o box mostra só o trecho mais recente do raciocínio (igual estilo ao
  // box de tool calls).
  const [reasoningExpanded, setReasoningExpanded] = useState(false);
  const [attachError, setAttachError] = useState<string | null>(null);
  const [showFilePicker, setShowFilePicker] = useState(false);
  const [pendingAttachment, setPendingAttachment] = useState<AttachedFile | null>(null);

  // Handler para processar arquivo selecionado (via picker ou comando)
  const handleAttachFile = useCallback(async (rawPath: string) => {
    const fullPath = resolve(rawPath);
    if (!existsSync(fullPath) || !statSync(fullPath).isFile()) {
      setAttachError(`Arquivo não encontrado: ${fullPath}`);
      return;
    }
    setAttachError(null);

    if (isImageFile(fullPath)) {
      // Fluxo imagem: converter para base64
      const { base64, mime } = imageToBase64(fullPath);
      setPendingAttachment({ kind: "image", base64, path: fullPath, mime });
      // Também setar no hook para compatibilidade com useOrchestratorStream
      setPendingImagePath(fullPath);
      return;
    }

    if (isDocumentFile(fullPath)) {
      // Fluxo documento: leitura direta (com markitdown para PDF/DOCX/HTML)
      try {
        const content = await extractTextContent(fullPath);
        setPendingAttachment({
          kind: "document",
          path: fullPath,
          content,
        });
      } catch (err) {
        setAttachError(`Falha ao ler documento: ${err instanceof Error ? err.message : String(err)}`);
      }
      return;
    }

    setAttachError(`Tipo de arquivo não suportado: ${fullPath}`);
  }, []);

  // Handler para múltiplos arquivos (drag-and-drop / paste)
  const handleAttachFiles = useCallback(async (rawPaths: string[]) => {
    for (const rawPath of rawPaths) {
      await handleAttachFile(rawPath);
      // Pequeno delay para não sobrecarregar se muitos arquivos
      await new Promise((r) => setTimeout(r, 50));
    }
  }, [handleAttachFile]);

  // Comando "/attach <caminho>" — anexa imagem ou documento à próxima pergunta
  // Imagem: segue fluxo antigo (base64 no image_base64 do /execute)
  // Documento: leitura direta do arquivo (markitdown p/ PDF/DOCX/HTML) e
  // injeção do conteúdo completo no prompt do turno.
  const handleSubmit = useCallback(async (value: string) => {
    const trimmed = value.trim();

    // /attach <caminho>
    const attachMatch = trimmed.match(/^\/attach\s+(.+)$/i);
    if (attachMatch) {
      const rawPath = attachMatch[1].trim().replace(/^["']|["']$/g, "");
      await handleAttachFile(rawPath);
      return;
    }

    // /attach sem argumento -> abre file picker
    if (trimmed.toLowerCase() === "/attach") {
      setShowFilePicker(true);
      return;
    }

    if (trimmed.toLowerCase() === "/settings") {
      setShowSettings(true);
      return;
    }

    if (trimmed.toLowerCase() === "/projects") {
      setShowProjects(true);
      return;
    }

    // Se há documento anexado, enviar o conteúdo como document_content
    // (o modelo recebe, mas a tela mostra só o texto da pergunta)
    let documentContent: string | undefined;
    if (pendingAttachment?.kind === "document" && trimmed) {
      documentContent = pendingAttachment.content;
    }

    submit(value, documentContent);

    // Anexo é one-shot: limpa após envio
    setPendingAttachment(null);
    setPendingImagePath(null);
  }, [handleAttachFile, pendingAttachment, submit]);

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
    if (key.ctrl && input === "e") {
      // Expande/colapsa o box de pensamento (💭) das mensagens.
      setReasoningExpanded((prev) => !prev);
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
            {"  "}(/settings também lista/edita planos · /projects gerencia escopos de memória · /attach para anexar arquivo)
          </Text>
        ) : (
          messages.map((m) => (
            <ChatMessage key={m.id} message={m} reasoningExpanded={reasoningExpanded} />
          ))
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

      {attachError && (
        <Box marginBottom={1}>
          <Text color={COLORS.red}>⚠ {attachError}</Text>
        </Box>
      )}

      {pendingAttachment && (
        <Box marginBottom={1}>
          <Text color={COLORS.amber}>
            {pendingAttachment.kind === "image"
              ? `🖼 imagem anexada à próxima pergunta: ${pendingAttachment.path}`
              : `📄 documento anexado: ${pendingAttachment.path}`}
          </Text>
        </Box>
      )}

      <FilePickerModal
        isOpen={showFilePicker}
        onClose={() => setShowFilePicker(false)}
        onSelect={handleAttachFile}
      />

      <InputBar
        disabled={isStreaming || suspended}
        codeMode={codeMode}
        focus={!showSettings && !showProjects && !suspended && !showFilePicker}
        onSubmit={handleSubmit}
        onAttachFiles={handleAttachFiles}
      />
    </Box>
  );
}