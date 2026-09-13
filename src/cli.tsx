#!/usr/bin/env node
import React from "react";
import { render } from "ink";
import { App } from "./App.js";
import { initLogger, initDockerLogFile, logger, LOG_FILE, DOCKER_LOG_FILE } from "./lib/logger.js";

// Limpa (trunca) os logs de depuração a cada início do app — precisa
// acontecer antes de qualquer outro import/código que possa logar.
initLogger();
initDockerLogFile();
logger.info("alpha-ai-tui: iniciando", { logFile: LOG_FILE, dockerLogFile: DOCKER_LOG_FILE });

// O app não usa mais llama-server local (nem seleção de modelo .gguf):
// o modelo é resolvido pelo orchestrator. O App dispara /docker/start
// automaticamente a cada pergunta — ver lib/processManager.ts e
// useOrchestratorStream.ts.
render(<App />);
