import { useCallback, useState } from "react";
import { API_BASE } from "../lib/constants.js";

interface ChosenModel {
  model: string;
  mmproj: boolean;
}

type ProcessStage = "idle" | "starting-llama" | "starting-docker" | "ready" | "error";

export function useProcessManagerApi() {
  const [stage, setStage] = useState<ProcessStage>("idle");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const startProcesses = useCallback(async (chosen: ChosenModel) => {
    setErrorMsg(null);
    setStage("starting-llama");
    try {
      const llamaRes = await fetch(`${API_BASE}/llama/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: chosen.model,
          mmproj: chosen.mmproj,
        }),
      });
      if (!llamaRes.ok) throw new Error(`llama/start HTTP ${llamaRes.status}`);
    } catch (err) {
      setStage("error");
      setErrorMsg(
        `Falha ao iniciar o llama-server: ${
          err instanceof Error ? err.message : err
        }`,
      );
      return false;
    }

    setStage("starting-docker");
    try {
      const dockerRes = await fetch(`${API_BASE}/docker/start`, {
        method: "POST",
      });
      if (!dockerRes.ok) throw new Error(`docker/start HTTP ${dockerRes.status}`);
    } catch (err) {
      // No original, falha do Docker só loga aviso — não derruba o app.
      setErrorMsg(
        `Aviso ao contatar a API do Docker: ${
          err instanceof Error ? err.message : err
        }`,
      );
    }

    setStage("ready");
    return true;
  }, []);

  return { stage, errorMsg, startProcesses };
}
