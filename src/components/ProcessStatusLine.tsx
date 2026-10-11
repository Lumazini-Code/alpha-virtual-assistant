import React from "react";
import { Box, Text } from "ink";
import { COLORS, GRAPH_VIEWER_URL } from "../lib/constants.js";
import { terminalLink } from "../lib/terminalLink.js";
import type { ProcessManagerStatus } from "../hooks/useProcessManagerStatus.js";
import type { ServiceStatus } from "../hooks/useServiceStatus.js";

function statusColor(status: string): string {
  const s = status.toLowerCase();
  if (s.includes("run") || s.includes("ativo") || s.includes("ok")) return COLORS.green;
  if (s.includes("start") || s.includes("iniciando")) return COLORS.amber;
  if (s.includes("error") || s.includes("erro")) return COLORS.red;
  return COLORS.gray;
}

// Canto da linha de status: link clicável (OSC 8 — ctrl+click abre o
// navegador) para o visualizador do grafo Hebbiano (graph_activation.py,
// porta 8777). Offline = dim com a dica de como subir o servidor.
function GraphLink({ online }: { online: ServiceStatus }) {
  const host = GRAPH_VIEWER_URL.replace(/^https?:\/\//, "");
  if (online === null) {
    return <Text color={COLORS.dim}>grafo: verificando…</Text>;
  }
  if (online) {
    return (
      <Text color={COLORS.green}>
        🔗 {terminalLink(`Visualizar Grafo de memória`, GRAPH_VIEWER_URL)}
      </Text>
    );
  }
  return (
    <Text color={COLORS.dim}>🔗 grafo offline (python graph_activation.py)</Text>
  );
}

export function ProcessStatusLine({
  status,
  graphOnline,
}: {
  status: ProcessManagerStatus | null;
  graphOnline: ServiceStatus;
}) {
  if (!status) {
    return (
      <Box>
        <Text color={COLORS.dim}>process manager: conectando…</Text>
        <Text> · </Text>
        <GraphLink online={graphOnline} />
      </Box>
    );
  }

  return (
    <Box>
      <Text color={statusColor(status.docker.status)}>● docker {status.docker.status}</Text>
      <Text color={COLORS.dim}> · </Text>
      <GraphLink online={graphOnline} />
    </Box>
  );
}
