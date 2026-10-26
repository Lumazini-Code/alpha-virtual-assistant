import React, { useEffect, useState } from "react";
import { Box, Text } from "ink";
import { FilePicker } from "ink-file-picker";
import { COLORS } from "../lib/constants.js";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (path: string) => void;
  startDir?: string;
}

export function FilePickerModal({ isOpen, onClose, onSelect, startDir = process.cwd() }: Props) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    return () => setMounted(false);
  }, []);

  if (!isOpen || !mounted) return null;

  return (
    <Box flexDirection="column" width="100%" height="100%" borderStyle="round" borderColor={COLORS.blue} padding={1}>
      <Text color={COLORS.blue}>◆ Selecionar arquivo para anexar (ESC para cancelar, ↑/↓ para navegar, Enter para selecionar)</Text>
      <FilePicker
        initialPath={startDir}
        onSelect={(paths) => {
          if (paths.length > 0) {
            onSelect(paths[0]);
            onClose();
          }
        }}
        onCancel={onClose}
        showHidden={false}
      />
    </Box>
  );
}