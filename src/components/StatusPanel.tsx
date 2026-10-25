import React from "react";
import { Box, Text } from "ink";
import { COLORS } from "../lib/constants.js";
import type { SystemStats } from "../lib/types.js";

function bar(pct: number, width = 8): string {
  const filled = Math.round((pct / 100) * width);
  return "█".repeat(filled) + "░".repeat(width - filled);
}

function colorForPct(pct: number): string {
  if (pct >= 85) return COLORS.red;
  if (pct >= 60) return COLORS.amber;
  return COLORS.green;
}

function truncateProject(slug: string, n = 18): string {
  return slug.length > n ? `Projeto: ${slug.slice(0, n - 1)}…` : `Projeto: ${slug}`;
}

function Metric({ label, pct, detail }: { label: string; pct: number; detail: string }) {
  return (
    <Box marginRight={3}>
      <Text color={COLORS.gray}>{label} </Text>
      <Text color={colorForPct(pct)}>
        {bar(pct)} {pct.toFixed(0)}%
      </Text>
      <Text color={COLORS.dim}> {detail}</Text>
    </Box>
  );
}

export function StatusPanel({ stats, activeProject }: { stats: SystemStats; activeProject?: string | null }) {
  const { gpu } = stats;
  return (
    <Box borderStyle="round" borderColor={COLORS.dim} paddingX={1}>
      {activeProject && activeProject !== "general" && (
        // NEW (projetos): badge do escopo ativo — leitura mescla o escopo do
        // projeto com o geral; escritas vão para o projeto. some no modo geral.
        <Text bold color={COLORS.amber}>◆ {truncateProject(activeProject)} </Text>
      )}
      <Metric label="CPU" pct={stats.totalCpuPct} detail="" />
      <Metric
        label="RAM"
        pct={stats.sysRamPct}
        detail={`${stats.sysRamGb.toFixed(1)}/${stats.sysRamTotGb.toFixed(1)}GB`}
      />
      {gpu.hasGpu ? (
        <>
          <Metric label={`GPU(${gpu.vendor})`} pct={gpu.gpuPct} detail="" />
          <Metric
            label="VRAM"
            pct={gpu.vramTotMb > 0 ? (gpu.vramUsedMb / gpu.vramTotMb) * 100 : 0}
            detail={`${(gpu.vramUsedMb / 1024).toFixed(1)}/${(gpu.vramTotMb / 1024).toFixed(1)}GB`}
          />
        </>
      ) : (
        <Text color={COLORS.dim}>GPU: N/A</Text>
      )}
    </Box>
  );
}
