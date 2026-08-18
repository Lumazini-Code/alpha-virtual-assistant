import { useEffect, useRef, useState } from "react";
import { execa } from "execa";
import { appendDockerLogLine } from "../lib/logger.js";

const CONTAINER_NAME = "ava-vulkan";
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
 */
export function useDockerLogs() {
  const [lines, setLines] = useState<string[]>([]);
  const [status, setStatus] = useState<
    "waiting" | "streaming" | "unavailable"
  >("waiting");
  const bufferRef = useRef<string>("");

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
