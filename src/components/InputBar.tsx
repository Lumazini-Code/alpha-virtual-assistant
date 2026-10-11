import React, { useState } from "react";
import { Box, Text } from "ink";
import TextInput from "ink-text-input";
import { COLORS } from "../lib/constants.js";

interface Props {
  disabled: boolean;
  codeMode: boolean;
  /** false = outra área da TUI tem o foco (settings aberto / TUI
   * suspensa) — o TextInput ignora o stdin e não rouba teclas. */
  focus?: boolean;
  onSubmit: (value: string) => void;
}

export function InputBar({ disabled, codeMode, focus = true, onSubmit }: Props) {
  const [value, setValue] = useState("");

  const handleSubmit = (val: string) => {
    if (!val.trim() || disabled) return;
    onSubmit(val);
    setValue("");
  };

  return (
    <Box
      borderStyle="round"
      borderColor={focus ? (codeMode ? COLORS.amber : COLORS.blue) : COLORS.dim}
      paddingX={1}
    >
      <Text color={focus ? (codeMode ? COLORS.amber : COLORS.blue) : COLORS.dim}>
        {codeMode ? "code▸ " : "▸ "}
      </Text>
      <TextInput
        value={value}
        onChange={setValue}
        onSubmit={handleSubmit}
        focus={focus}
        placeholder={disabled ? "aguardando resposta…" : "digite sua mensagem"}
      />
    </Box>
  );
}
