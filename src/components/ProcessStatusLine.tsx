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

  return (
    <Box>
      <Text color={statusColor(status.docker.status)}>● docker {status.docker.status}</Text>
    </Box>
  );
}
