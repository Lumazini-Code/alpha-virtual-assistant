// Alpha AI (AVA) — painel das últimas N linhas do log do Docker
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
