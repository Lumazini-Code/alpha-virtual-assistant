import React from "react";
import { Box, Text } from "ink";
import { COLORS } from "../lib/constants.js";

interface Props {
  lines: string[];
  status: "waiting" | "streaming" | "unavailable";
  visibleLines?: number;
}

export function DockerLogPanel({ lines, status, visibleLines = 12 }: Props) {
  if (status === "waiting") {
    return (
      <Box borderStyle="round" borderColor={COLORS.dim} paddingX={1}>
        <Text color={COLORS.gray}>Aguardando container subir…</Text>
      </Box>
    );
  }

  if (status === "unavailable") {
    return (
      <Box borderStyle="round" borderColor={COLORS.red} paddingX={1}>
        <Text color={COLORS.red}>
          Nenhum método de leitura de logs funcionou (container não encontrado).
        </Text>
      </Box>
    );
  }

  // Ink não tem scroll — mostra sempre a "janela" das últimas N linhas.
  const tail = lines.slice(-visibleLines);

  return (
    <Box flexDirection="column" borderStyle="round" borderColor={COLORS.dim} paddingX={1}>
      <Text color={COLORS.gray} bold>
        docker logs -f ava
      </Text>
      {tail.length === 0 ? (
        <Text color={COLORS.dim}>(sem saída ainda)</Text>
      ) : (
        tail.map((line, i) => (
          <Text key={i} color={COLORS.white} wrap="truncate-end">
            {line}
          </Text>
        ))
      )}
    </Box>
  );
}
