// Alpha AI (AVA) — hook de streaming dos logs do Docker
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
import { useEffect, useRef, useState } from "react";

import { execa } from "execa";
import { appendDockerLogLine, logger } from "../lib/logger.js";

const CONTAINER_NAME = "ava";
const MAX_LINES = 200; // Ink não tem scroll nativo -> janela deslizante manual

async function waitForContainer(
  timeoutMs = 30_000,
  intervalMs = 1000,
): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const { stdout } = await execa("docker", [
        "ps",
        "-q",
        "-f",
        `name=${CONTAINER_NAME}`,
      ]);
      if (stdout.trim()) return true;
    } catch {
      // ignora e tenta de novo
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  return false;
}

/**
 * Substitui _try_docker_logs / _try_tail_log_file / _wait_for_container.
 * Mantém só as últimas MAX_LINES linhas em estado (janela deslizante),
 * já que o Ink não tem um widget de scroll tipo o RichLog do Textual.
 *
 * `paused` congela a JANELA VISÍVEL (sem setState → sem redraw do Ink,
 * que corromperia um subprocesso interativo no terminal — ver
 * SettingsPanel). O stream continua vivo e o docker.log continua
 * recebendo as linhas; só a tela da TUI fica parada.
 */
export function useDockerLogs(paused = false) {
  const [lines, setLines] = useState<string[]>([]);
  const [status, setStatus] = useState<
    "waiting" | "streaming" | "unavailable"
  >("waiting");
  const bufferRef = useRef<string>("");
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  useEffect(() => {
    let cancelled = false;
    let subprocess: ReturnType<typeof execa> | null = null;

    const start = async () => {
      const up = await waitForContainer();
      if (cancelled) return;
      if (!up) {
        setStatus("unavailable");
        return;
      }

      try {
        subprocess = execa("docker", ["logs", "-f", CONTAINER_NAME], {
          all: true, // equivalente a stderr=subprocess.STDOUT unificado
        });
        // O stream NUNCA é awaited: quando a TUI fecha, o cleanup manda
        // SIGTERM no `docker logs -f` → exit 143 → a promise do execa
        // rejeita. Sem este catch, vira unhandled rejection (ExecaError
        // impresso no fechamento da TUI). Fix silencioso: só registra no
        // debug.log — o status da TUI não muda.
        subprocess.catch((err: unknown) => {
          const message = err instanceof Error ? err.message : String(err);
          logger.debug("useDockerLogs: stream de docker logs encerrado", { message });
        });
        setStatus("streaming");

        subprocess.all?.on("data", (chunk: Buffer) => {
          bufferRef.current += chunk.toString("utf-8");
          const parts = bufferRef.current.split("\n");
          bufferRef.current = parts.pop() ?? "";
          if (parts.length === 0) return;
          // Grava cada linha em docker.log igual ao original (ava_docker.log)
          // — cópia bruta e completa do log do container, fora da janela
          // deslizante que o painel da TUI mostra.
          for (const part of parts) appendDockerLogLine(part);
          if (pausedRef.current) return; // janela congelada — só o arquivo registra
          setLines((prev) => [...prev, ...parts].slice(-MAX_LINES));
        });
      } catch {
        setStatus("unavailable");
      }
    };

    start();
    return () => {
      cancelled = true;
      subprocess?.kill();
    };
  }, []);

  return { lines, status };
}
