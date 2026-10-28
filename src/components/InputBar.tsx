// Alpha AI (AVA) — barra de entrada da TUI (texto, anexos, comandos e histórico)
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
import React, { useState, useEffect, useRef } from "react";
import { Box, Text, useInput } from "ink";
import TextInput from "ink-text-input";
import { COLORS } from "../lib/constants.js";

interface Props {
  disabled: boolean;
  codeMode: boolean;
  /** false = outra área da TUI tem o foco (settings aberto / TUI
   * suspensa) — o TextInput ignora o stdin e não rouba teclas. */
  focus?: boolean;
  onSubmit: (value: string) => void;
  /** Callback quando arquivos são detectados no paste (drag-and-drop) */
  onAttachFiles?: (paths: string[]) => void;
}

/**
 * Detecta se uma string parece um caminho de arquivo válido.
 * Aceita caminhos absolutos, relativos, com ~, e caminhos entre aspas.
 */
function extractFilePaths(text: string): string[] {
  const lines = text.split(/\r?\n/);
  const paths: string[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    // Remove aspas envolventes
    const unquoted = trimmed.replace(/^["']|["']$/g, "");

    // Heurística: parece caminho se contém separador ou é caminho absoluto/~
    const looksLikePath =
      unquoted.includes("/") ||
      unquoted.includes("\\") ||
      unquoted.startsWith("~") ||
      /^[A-Za-z]:[\\/]/.test(unquoted); // Windows C:\path

    if (looksLikePath) {
      paths.push(unquoted);
    }
  }

  return paths;
}

export function InputBar({
  disabled,
  codeMode,
  focus = true,
  onSubmit,
  onAttachFiles,
}: Props) {
  const [value, setValue] = useState("");
  const pasteBufferRef = useRef<string>("");

  const handleSubmit = (val: string) => {
    if (!val.trim() || disabled) return;
    onSubmit(val);
    setValue("");
  };

  // Bracketed paste mode detection: \x1b[200~ ... \x1b[201~
  // Quando o terminal suporta bracketed paste, ele envia esses códigos
  // antes e depois do conteúdo colado.
  useInput((input, key) => {
    if (!focus || disabled) return;

    // Início do bracketed paste
    if (input === "\x1b[200~") {
      pasteBufferRef.current = "";
      return;
    }

    // Fim do bracketed paste - processa o buffer
    if (input === "\x1b[201~") {
      const paths = extractFilePaths(pasteBufferRef.current);
      if (paths.length > 0 && onAttachFiles) {
        onAttachFiles(paths);
      }
      pasteBufferRef.current = "";
      return;
    }

    // Acumula conteúdo durante o paste
    if (pasteBufferRef.current !== "") {
      pasteBufferRef.current += input;
    }
  });

  // Fallback: detecta paste "bruto" (sem bracketed paste mode)
  // quando muito texto chega de uma vez no onChange
  const handleChange = (val: string) => {
    setValue(val);

    // Heurística: se o valor cresceu muito de uma vez (paste),
    // e não estávamos no bracketed paste mode, tenta extrair paths
    if (
      val.length > value.length + 50 && // salto grande = provável paste
      !pasteBufferRef.current &&
      onAttachFiles
    ) {
      const paths = extractFilePaths(val);
      if (paths.length > 0) {
        // Remove os paths do input (só mantém o que o usuário digitou)
        let cleaned = val;
        for (const p of paths) {
          cleaned = cleaned.replace(p, "").replace(/^\s+|\s+$/g, "");
        }
        setValue(cleaned);
        onAttachFiles(paths);
      }
    }
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
        onChange={handleChange}
        onSubmit={handleSubmit}
        focus={focus}
        placeholder={disabled
          ? "aguardando resposta…"
          : codeMode
            ? "digite código (Ctrl+K para voltar)"
            : "digite sua mensagem (/attach ou arraste arquivos aqui)"}
      />
    </Box>
  );
}
