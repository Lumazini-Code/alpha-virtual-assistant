import React, { useState } from "react";
import { Box, Text } from "ink";
import TextInput from "ink-text-input";
import { COLORS } from "../lib/constants.js";

interface Props {
  disabled: boolean;
  codeMode: boolean;
  onSubmit: (value: string) => void;
}

export function InputBar({ disabled, codeMode, onSubmit }: Props) {
  const [value, setValue] = useState("");

  const handleSubmit = (val: string) => {
    if (!val.trim() || disabled) return;
    onSubmit(val);
    setValue("");
  };

  return (
    <Box borderStyle="round" borderColor={codeMode ? COLORS.amber : COLORS.blue} paddingX={1}>
      <Text color={codeMode ? COLORS.amber : COLORS.blue}>
        {codeMode ? "code▸ " : "▸ "}
      </Text>
      <TextInput
        value={value}
        onChange={setValue}
        onSubmit={handleSubmit}
        placeholder={disabled ? "aguardando resposta…" : "digite sua mensagem"}
      />
    </Box>
  );
}
