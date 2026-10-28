// Alpha AI (AVA) — modal de seleção de arquivos para anexar ao turno
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