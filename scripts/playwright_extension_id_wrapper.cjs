// Alpha AI (AVA) — troca o ID da extensão Playwright em runtime
// Copyright (C) 2026  Felipe lumazini
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Carregado via NODE_OPTIONS=--require=... no processo do @playwright/mcp.
// O Playwright tem o ID da extensão oficial hardcoded em
// playwright-core/lib/coreBundle.js (e tools/utils/extension.js). Como o
// instalador Linux empacota a extensão patcheada com a NOSSA chave, o ID
// muda; este hook substitui o literal no momento do require, sem forkar o
// pacote nem editar o cache do npx.
'use strict';

const fs = require('fs');
const Module = require('module');

const OLD_ID = 'mmlmfjhmonkocbjadbfplnigmagldckm';
const NEW_ID = (process.env.PLAYWRIGHT_MCP_EXTENSION_ID || '').trim();

if (NEW_ID && NEW_ID !== OLD_ID) {
  const original = Module._extensions['.js'];
  Module._extensions['.js'] = function patchedLoader(mod, filename) {
    if (filename.includes('playwright-core')) {
      try {
        const source = fs.readFileSync(filename, 'utf8');
        if (source.includes(OLD_ID)) {
          mod._compile(source.split(OLD_ID).join(NEW_ID), filename);
          return;
        }
      } catch {
        // Deixa o loader original tratar/emitir o erro real.
      }
    }
    return original(mod, filename);
  };
}
