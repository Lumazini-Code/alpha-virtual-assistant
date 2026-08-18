import React, { useEffect, useState } from "react";
import { Box, Text } from "ink";
import SelectInput from "ink-select-input";
import TextInput from "ink-text-input";
import { COLORS, API_BASE } from "../lib/constants.js";
import { fetchAvailableModels, type DiscoveredModel } from "../lib/models.js";

interface Props {
  onChoose: (model: string, mmprojUsed: boolean) => void;
  onSkip: () => void;
}

type Phase = "loading" | "list" | "manual";

export function ModelSelector({ onChoose, onSkip }: Props) {
  const [phase, setPhase] = useState<Phase>("loading");
  const [models, setModels] = useState<DiscoveredModel[]>([]);
  const [manualPath, setManualPath] = useState("");

  useEffect(() => {
    let cancelled = false;
    fetchAvailableModels().then((found) => {
      if (cancelled) return;
      setModels(found);
      setPhase(found.length > 0 ? "list" : "manual");
    });
    return () => {
      cancelled = true;
    };
  }, []);

  if (phase === "loading") {
    return (
      <Box>
        <Text color={COLORS.gray}>Procurando modelos em {API_BASE}/models…</Text>
      </Box>
    );
  }

  if (phase === "manual") {
    return (
      <Box flexDirection="column">
        <Text color={COLORS.amber}>
          Não encontrei modelos via API (rota /models indisponível ou vazia).
        </Text>
        <Text color={COLORS.gray}>
          Digite o caminho do .gguf manualmente (ou deixe em branco para pular
          o auto-start de llama-server e usar só o que já estiver ativo):
        </Text>
        <Box borderStyle="round" borderColor={COLORS.blue} paddingX={1} marginTop={1}>
          <Text color={COLORS.blue}>▸ </Text>
          <TextInput
            value={manualPath}
            onChange={setManualPath}
            onSubmit={(val) => {
              const trimmed = val.trim();
              if (!trimmed) {
                onSkip();
                return;
              }
              onChoose(trimmed, false);
            }}
            placeholder="/caminho/para/modelo.gguf"
          />
        </Box>
      </Box>
    );
  }

  const items = [
    ...models.map((m) => ({
      key: m.path,
      label: `${m.name}${m.hasMmproj ? "  [multimodal]" : ""}`,
      value: m,
    })),
    { key: "__manual__", label: "(digitar caminho manualmente)", value: null },
  ];

  return (
    <Box flexDirection="column">
      <Text color={COLORS.gray} bold>
        Escolha o modelo:
      </Text>
      <SelectInput
        items={items}
        onSelect={(item) => {
          if (item.value === null) {
            setPhase("manual");
            return;
          }
          onChoose(item.value.path, item.value.hasMmproj);
        }}
      />
    </Box>
  );
}
