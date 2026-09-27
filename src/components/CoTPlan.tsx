import React from "react";
import { Box, Text } from "ink";
import Spinner from "ink-spinner";
import { COLORS } from "../lib/constants.js";
import type { CoTPlanState, CoTStep } from "../lib/types.js";

function StepIcon({ status }: { status: CoTStep["status"] }) {
  switch (status) {
    case "running":
      return (
        <Text color={COLORS.amber}>
          <Spinner type="dots" />
        </Text>
      );
    case "done":
      return <Text color={COLORS.green}>✓</Text>;
    case "error":
      return <Text color={COLORS.red}>✗</Text>;
    default:
      return <Text color={COLORS.gray}>○</Text>;
  }
}

function StepColor(status: CoTStep["status"]): string {
  if (status === "running") return COLORS.amber;
  if (status === "done") return COLORS.green;
  if (status === "error") return COLORS.red;
  return COLORS.gray;
}

function PlanRow({ step }: { step: CoTStep }) {
  const deps = step.dependsOn.length ? `  deps:[${step.dependsOn.join(",")}]` : "";
  const latency = step.latencyMs !== undefined ? ` ${step.latencyMs.toFixed(0)}ms` : "";
  const errorSuffix = step.error ? ` — ${step.error}` : "";

  return (
    <Box flexDirection="column" paddingLeft={3} marginBottom={0}>
      <Box>
        <StepIcon status={step.status} />
        <Text color={StepColor(step.status)}>
          {" "}
          Step {step.step} [{step.executor}]{deps}
          {latency}
          {errorSuffix}
        </Text>
      </Box>
      {step.status !== "done" && step.action && (
        <Box paddingLeft={2}>
          <Text color={COLORS.dim}>
            {step.action.length > 90 ? step.action.slice(0, 90) + "…" : step.action}
          </Text>
        </Box>
      )}
    </Box>
  );
}

export function CoTPlan({ plan }: { plan: CoTPlanState }) {
  const headerLabel = plan.dynamic ? "🔧 Tool calls" : "🧠 CoT";
  const cacheTag = plan.fromCache ? "  [cache]" : "";

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor={COLORS.amber}
      paddingX={1}
      marginY={1}
    >
      <Text color={COLORS.amber} bold>
        {headerLabel} — {plan.steps.length} step(s){cacheTag}
      </Text>
      {plan.steps.map((s) => (
        <PlanRow key={s.step} step={s} />
      ))}
    </Box>
  );
}
