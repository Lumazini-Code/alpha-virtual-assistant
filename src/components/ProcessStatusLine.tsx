import React from "react";
import { Box, Text } from "ink";
import { COLORS } from "../lib/constants.js";
import type { ProcessManagerStatus } from "../hooks/useProcessManagerStatus.js";

function statusColor(status: string): string {
  const s = status.toLowerCase();
  if (s.includes("run") || s.includes("ativo") || s.includes("ok")) return COLORS.green;
  if (s.includes("start") || s.includes("iniciando")) return COLORS.amber;
  if (s.includes("error") || s.includes("erro")) return COLORS.red;
  return COLORS.gray;
}

export function ProcessStatusLine({ status }: { status: ProcessManagerStatus | null }) {
  if (!status) {
    return <Text color={COLORS.dim}>process manager: conectando…</Text>;
  }

  const modelName = status.llama.model?.split(/[\\/]/).pop();

  return (
    <Box>
      <Text color={statusColor(status.llama.status)}>
        ● llama {status.llama.status}
      </Text>
      {modelName && (
        <Text color={COLORS.dim}>
          {" "}
          ({modelName}, {status.llama.mode}
          {status.llama.mmproj ? "+mmproj" : ""})
        </Text>
      )}
      <Text color={COLORS.gray}>   </Text>
      <Text color={statusColor(status.docker.status)}>● docker {status.docker.status}</Text>
    </Box>
  );
}
