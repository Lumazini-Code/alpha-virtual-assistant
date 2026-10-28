// Alpha AI (AVA) — renderização de mensagem do chat (markdown simplificado p/ terminal)
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
import React from "react";
import { Box, Text } from "ink";
import Spinner from "ink-spinner";
import { COLORS } from "../lib/constants.js";
import { CoTPlan } from "./CoTPlan.js";
import type { ChatMessageState } from "../lib/types.js";

/**
 * Renderização Markdown simplificada, focada no essencial para terminal:
 * cabeçalhos, bullets, **negrito**, `code`. Cobre o caso de uso comum de
 * _append_markdown_lines/_append_inline sem a complexidade completa de
 * LaTeX do original (que pode ser adicionada depois se for realmente usada).
 */
function MarkdownLine({ line }: { line: string }) {
  const heading = line.match(/^(#{1,6})\s+(.*)/);
  if (heading) {
    const level = heading[1].length;
    return (
      <Text bold color={level <= 2 ? "#FFFFFF" : COLORS.blue}>
        {heading[2]}
      </Text>
    );
  }

  const bullet = line.match(/^(\s*)([-*+]|\d+\.)\s+(.*)/);
  if (bullet) {
    const marker = /\d/.test(bullet[2][0]) ? bullet[2] : "•";
    return (
      <Text color={COLORS.white}>
        <Text color={COLORS.blue}>{marker} </Text>
        {bullet[3]}
      </Text>
    );
  }

  // negrito e código inline básicos — Ink não faz rich markup por string,
  // então quebramos manualmente em segmentos
  const segments = line.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean);
  return (
    <Text color={COLORS.white}>
      {segments.map((seg, i) => {
        if (seg.startsWith("**") && seg.endsWith("**")) {
          return (
            <Text key={i} bold>
              {seg.slice(2, -2)}
            </Text>
          );
        }
        if (seg.startsWith("`") && seg.endsWith("`")) {
          return (
            <Text key={i} color="#A8D8A8" bold>
              {seg.slice(1, -1)}
            </Text>
          );
        }
        return seg;
      })}
    </Text>
  );
}

/**
 * Box do processo de pensamento do modelo — mesma linguagem visual do
 * CoTPlan (box de tool calls): borda round, header com contagem, conteúdo
 * dim/itálico. Expansível com ctrl+e:
 *   - colapsado: só o TRECHO MAIS RECENTE do raciocínio (o thinking cresce
 *     na cauda — a cabeça já é velha; a cauda é o que o modelo está
 *     decidindo AGORA), com indicador de quantos chars ficaram fora;
 *   - expandido: texto completo.
 */
function ReasoningBox({ reasoning, expanded }: { reasoning: string; expanded: boolean }) {
  const RECENT = 160;
  const compact = !expanded && reasoning.length > RECENT;
  const body = compact ? "… " + reasoning.slice(-RECENT) : reasoning;

  return (
    <Box flexDirection="column" borderStyle="round" borderColor={COLORS.gray} paddingX={1} marginY={1}>
      <Text color={COLORS.gray} bold>
        💭 Pensamento — {reasoning.length} chars
        {compact ? `  [ctrl+e expande · ${reasoning.length - RECENT} chars ocultos]` : "  [ctrl+e colapsa]"}
      </Text>
      <Text color={COLORS.gray} italic>
        {body}
      </Text>
    </Box>
  );
}

export function ChatMessage({ message, reasoningExpanded = false }: {
  message: ChatMessageState;
  /** Global (ctrl+e): true = texto completo; false = trecho mais recente. */
  reasoningExpanded?: boolean;
}) {
  const isUser = message.role === "user";
  const lines = message.text.split("\n");

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text color={isUser ? COLORS.blue : COLORS.green} bold>
        {isUser ? "› você" : "◆ alpha"}
        {message.streaming && (
          <>
            {" "}
            <Spinner type="dots" />
          </>
        )}
      </Text>

      {message.imagePath && (
        <Box paddingLeft={2}>
          <Text color={COLORS.dim}>🖼 {message.imagePath}</Text>
        </Box>
      )}

      {message.reasoning && (
        <ReasoningBox reasoning={message.reasoning} expanded={reasoningExpanded} />
      )}

      {message.plan && <CoTPlan plan={message.plan} />}

      <Box flexDirection="column" paddingLeft={2}>
        {lines.map((line, i) => (
          <MarkdownLine key={i} line={line} />
        ))}
      </Box>

      {message.error && (
        <Box paddingLeft={2}>
          <Text color={COLORS.red}>⚠ {message.error}</Text>
        </Box>
      )}
    </Box>
  );
}
