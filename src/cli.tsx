#!/usr/bin/env node
import React, { useState } from "react";
import { render } from "ink";
import { App } from "./App.js";
import { ModelSelector } from "./components/ModelSelector.js";
import { setChosenModel } from "./lib/modelState.js";
import { initLogger, initDockerLogFile, logger, LOG_FILE, DOCKER_LOG_FILE } from "./lib/logger.js";

// Limpa (trunca) os logs de depuração a cada início do app — precisa
// acontecer antes de qualquer outro import/código que possa logar.
initLogger();
initDockerLogFile();
logger.info("alpha-ai-tui: iniciando", { logFile: LOG_FILE, dockerLogFile: DOCKER_LOG_FILE });

function Root() {
  const [modelChosen, setModelChosen] = useState(false);

  if (!modelChosen) {
    return (
      <ModelSelector
        onChoose={(model, mmprojUsed) => {
          setChosenModel(model, mmprojUsed);
          setModelChosen(true);
        }}
        onSkip={() => setModelChosen(true)}
      />
    );
  }

  // A partir daqui, o App já dispara /llama/start + /docker/start (com o
  // modelo escolhido acima) automaticamente a cada pergunta — ver
  // lib/processManager.ts e useOrchestratorStream.ts.
  return <App />;
}

render(<Root />);
