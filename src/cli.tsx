#!/usr/bin/env node
// Alpha AI (AVA) — entry point do cliente TUI (bootstrap de logs + render do App)
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
