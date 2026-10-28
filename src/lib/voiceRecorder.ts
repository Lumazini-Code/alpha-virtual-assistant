// Alpha AI (AVA) — gravação de microfone no host (cadastro de locutores)
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
import { spawn, spawnSync } from "node:child_process";
import { logger } from "./logger.js";

/**
 * Gravação de microfone NO HOST (lado TUI) para o cadastro de locutores.
 *
 * Grava PCM cru (s16le mono 16kHz — mesmo formato que o orchestrator espera
 * em /voice/register) durante `seconds` e devolve o áudio em base64.
 *
 * A gravação roda em processo FILHO com stdout capturado — nada toca no
 * terminal da TUI (o stdin da TUI permanece intocado).
 *
 * Gravadores, na ordem de preferência:
 *   Linux — arecord (ALSA) → sox/rec → ffmpeg
 *   Windows — ffmpeg (dshow, sem seleção de device: usa o default)
 */

export interface MicRecordResult {
  ok: boolean;
  audioB64?: string;
  error?: string;
}

const SAMPLE_RATE = 16000;
const CHANNELS = 1;

function commandExists(cmd: string): boolean {
  try {
    const probe = process.platform === "win32"
      ? spawnSync("where", [cmd], { stdio: "ignore" })
      : spawnSync("which", [cmd], { stdio: "ignore" });
    return probe.status === 0;
  } catch {
    return false;
  }
}

/** Linha de comando do gravador: {cmd, args} que grava `seconds` de PCM
 * s16le mono 16kHz NO STDOUT (pipe:0 não é usado; stdin é ignorado). */
function resolveRecorder(seconds: number): { cmd: string; args: string[] } | null {
  const secs = String(seconds);
  if (process.platform === "win32") {
    if (commandExists("ffmpeg")) {
      return {
        cmd: "ffmpeg",
        args: ["-hide_banner", "-loglevel", "error", "-f", "dshow", "-i", "audio=Microphone",
               "-t", secs, "-f", "s16le", "-ar", String(SAMPLE_RATE), "-ac", String(CHANNELS), "pipe:1"],
      };
    }
    return null;
  }
  if (commandExists("arecord")) {
    return {
      cmd: "arecord",
      // Formato ALSA: S16_LE (arecord rejeita a variante ffmpeg "s16le").
      args: ["-q", "-f", "S16_LE", "-c", String(CHANNELS), "-r", String(SAMPLE_RATE),
             "-t", "raw", "-d", secs],
    };
  }
  if (commandExists("sox")) {
    return {
      cmd: "sox",
      args: ["-q", "-d", "-t", "s16", "-r", String(SAMPLE_RATE), "-c", String(CHANNELS),
             "-e", "signed-integer", "-b", "16", "-t", "s16", "-r", String(SAMPLE_RATE),
             "-c", String(CHANNELS), "-", "-", "trim", "0", secs],
    };
  }
  if (commandExists("rec")) {
    return {
      cmd: "rec",
      args: ["-q", "-r", String(SAMPLE_RATE), "-c", String(CHANNELS), "-b", "16",
             "-e", "signed-integer", "-t", "s16", "-", "trim", "0", secs],
    };
  }
  if (commandExists("ffmpeg")) {
    return {
      cmd: "ffmpeg",
      args: ["-hide_banner", "-loglevel", "error", "-f", "alsa", "-i", "default",
             "-t", secs, "-f", "s16le", "-ar", String(SAMPLE_RATE), "-ac", String(CHANNELS), "pipe:1"],
    };
  }
  return null;
}

/**
 * Grava `seconds` de microfone e devolve o PCM (s16le mono 16kHz) em base64.
 * `ok=false` quando não há gravador disponível ou o processo falha.
 */
export async function recordMic(seconds: number): Promise<MicRecordResult> {
  const spec = resolveRecorder(seconds);
  if (!spec) {
    const msg = process.platform === "win32"
      ? "nenhum gravador disponível (instale ffmpeg)"
      : "nenhum gravador disponível (instale arecord, sox ou ffmpeg)";
    logger.warn("voiceRecorder: gravação falhou — sem ferramenta", { platform: process.platform });
    return { ok: false, error: msg };
  }

  return new Promise<MicRecordResult>((resolve) => {
    let child;
    try {
      child = spawn(spec.cmd, spec.args, {
        // stdin ignorado (não toca no stdin da TUI); stderr capturado p/ erro.
        stdio: ["ignore", "pipe", "pipe"],
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.warn("voiceRecorder: falha ao spawnar gravador", { cmd: spec.cmd, message: msg });
      return resolve({ ok: false, error: msg });
    }

    const chunks: Buffer[] = [];
    let stderr = "";
    let settled = false;
    const finish = (r: MicRecordResult) => {
      if (settled) return;
      settled = true;
      resolve(r);
    };

    child.stdout?.on("data", (c: Buffer) => chunks.push(c));
    child.stderr?.on("data", (c: Buffer) => {
      stderr = (stderr + c.toString()).slice(-500); // mantém o final
    });
    child.on("error", (err) => {
      logger.warn("voiceRecorder: erro no gravador", { cmd: spec.cmd, message: err.message });
      finish({ ok: false, error: err.message });
    });
    child.on("close", (code) => {
      const pcm = Buffer.concat(chunks);
      if (code !== 0) {
        const msg = stderr.trim() || `gravador (${spec.cmd}) saiu com código ${code}`;
        logger.warn("voiceRecorder: gravação falhou", { cmd: spec.cmd, code, stderr: msg });
        return finish({ ok: false, error: msg });
      }
      // ~2 bytes/amostra: mesmo piso do orchestrator (0.5s mínimo de fala).
      const minBytes = SAMPLE_RATE * 2 * 0.5;
      if (pcm.length < minBytes) {
        return finish({
          ok: false,
          error: `áudio muito curto (${(pcm.length / (SAMPLE_RATE * 2)).toFixed(1)}s) — fale durante a gravação`,
        });
      }
      logger.info("voiceRecorder: gravação ok", {
        cmd: spec.cmd, seconds, bytes: pcm.length,
      });
      finish({ ok: true, audioB64: pcm.toString("base64") });
    });
  });
}
