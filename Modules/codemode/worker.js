// Alpha AI (AVA) — Code Mode worker (Node)
// Copyright (C) 2026  Felipe lumazini
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Protocolo (linhas JSON):
//   stdin  (1ª linha): {code, catalog, limits}
//   stdout: {type:"tool_call", id, name, input}
//   stdin : {type:"tool_result", id, result|error}
//   stdout: {type:"final", output, toolCalls, logs} | {type:"error", error}
//
// O worker NÃO expõe require/process/fs/network ao código do modelo. O `vm`
// é isolamento de contexto (o container é a fronteira de segurança real).

"use strict";

const vm = require("vm");
const readline = require("readline");

let catalog = [];
let maxCalls = 50;
let timeoutMs = 30_000;
let callSeq = 0;
let callCount = 0;
const pending = new Map();
const callMeta = new Map();
const callLog = [];
const logs = [];
let started = false;

function send(message) {
  process.stdout.write(JSON.stringify(message) + "\n");
}

function format(value) {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch (_error) {
    return String(value);
  }
}

function makeTools(path) {
  const target = function () {};
  return new Proxy(target, {
    get(_target, property) {
      if (typeof property !== "string") return undefined;
      return makeTools(path ? `${path}.${property}` : property);
    },
    apply(_target, _thisArg, args) {
      const input = args.length > 0 && args[0] !== undefined ? args[0] : {};
      return callTool(path, input);
    },
  });
}

function callTool(name, input) {
  if (!name) return Promise.reject(new Error("tool path is empty"));
  if (callCount >= maxCalls) {
    return Promise.reject(new Error(`tool call limit exceeded (${maxCalls})`));
  }
  callCount += 1;
  const id = ++callSeq;
  const entry = { id, name, status: "running" };
  callLog.push(entry);
  callMeta.set(id, entry);
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    send({ type: "tool_call", id, name, input });
  });
}

function resolveToolResult(message) {
  const waiter = pending.get(message.id);
  const meta = callMeta.get(message.id);
  if (!waiter) return;
  pending.delete(message.id);
  if (meta) meta.status = message.error ? "error" : "completed";
  if (message.error) waiter.reject(new Error(String(message.error)));
  else waiter.resolve(message.result);
}

function search(query) {
  const terms = String(query || "")
    .toLowerCase()
    .split(/[^a-z0-9_]+/)
    .filter(Boolean);
  if (terms.length === 0) return [];
  const scored = catalog
    .map((entry) => {
      const haystack = `${entry.name} ${entry.display} ${entry.description}`.toLowerCase();
      return { entry, score: terms.filter((term) => haystack.includes(term)).length };
    })
    .filter((item) => item.score > 0)
    .sort((left, right) => right.score - left.score || left.entry.name.localeCompare(right.entry.name))
    .slice(0, 12)
    .map((item) => ({
      path: item.entry.display,
      name: item.entry.name,
      signature: item.entry.signature,
      description: item.entry.description,
    }));
  return scored;
}

function makeConsole() {
  const record = (level) => (...args) => {
    logs.push(`${level}: ${args.map(format).join(" ")}`);
  };
  return { log: record("log"), info: record("info"), warn: record("warn"), error: record("error") };
}

async function run(message) {
  catalog = Array.isArray(message.catalog) ? message.catalog : [];
  maxCalls = Number(message.limits && message.limits.maxToolCalls) || 50;
  timeoutMs = Number(message.limits && message.limits.timeoutMs) || 30_000;

  const context = vm.createContext({
    tools: makeTools(""),
    search,
    console: makeConsole(),
    fetch: () => Promise.reject(new Error("network is disabled in Code Mode")),
    setTimeout: undefined,
    setInterval: undefined,
    clearTimeout: undefined,
    clearInterval: undefined,
  });

  const source = `(async () => {\n${String(message.code || "")}\n})()`;
  try {
    const script = new vm.Script(source, { filename: "codemode.js" });
    const value = await script.runInContext(context, { timeout: timeoutMs });
    const output = await value;
    send({
      type: "final",
      output: output === undefined ? "undefined" : format(output),
      toolCalls: callLog,
      logs,
    });
  } catch (error) {
    send({
      type: "error",
      error: String(error && error.stack ? error.stack : error),
      toolCalls: callLog,
      logs,
    });
  }
  process.exit(0);
}

const reader = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
reader.on("line", (line) => {
  let message;
  try {
    message = JSON.parse(line);
  } catch (error) {
    send({ type: "error", error: `invalid protocol message: ${error}` });
    process.exit(1);
    return;
  }
  if (!started) {
    started = true;
    run(message);
    return;
  }
  if (message.type === "tool_result") resolveToolResult(message);
});

process.on("uncaughtException", (error) => {
  send({ type: "error", error: String(error && error.stack ? error.stack : error) });
  process.exit(1);
});
process.on("unhandledRejection", (error) => {
  send({ type: "error", error: String(error && error.stack ? error.stack : error) });
  process.exit(1);
});
