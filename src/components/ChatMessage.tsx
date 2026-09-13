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

export function ChatMessage({ message }: { message: ChatMessageState }) {
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
        <Box paddingLeft={2} marginBottom={0}>
          <Text color={COLORS.dim} italic>
            {message.reasoning}
          </Text>
        </Box>
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
