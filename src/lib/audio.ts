import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { logger } from "./logger.js";

/**
 * Playback de áudio NO HOST (lado TUI) — modo tts_output=client com
 * STREAMING PROGRESSIVO: o orchestrator sintetiza por frase conforme a
 * resposta é gerada e emite chunks de PCM cru (s16le mono, base64) nos
 * eventos SSE `audio`. Este módulo abre UM player persistente e escreve
 * os chunks no stdin dele → fala começa durante a geração, sem gaps
 * entre frases.
 *
 * Players de stream (escrevem PCM pelo stdin):
 *   Linux   — pacat --raw (Pulse) → aplay -t raw (ALSA) → ffplay
 *   Windows — ffplay (se instalado); sem ffplay, fallback por chunk
 *
 * Fallback (sem player de stdin): cada chunk vira um WAV temporário tocado
 * via PowerShell Media.SoundPlayer — frases chegam progressivas, com pausa
 * pequena entre elas.
 *
 * Ciclo por turno: audio_start → audio* → audio_end (stdin.end — o player
 * exita sozinho ao drenar). Cancel (Ctrl+C): mata o player imediatamente.
 */

interface AudioStreamMeta {
  mime?: string;
  format?: string;   // "s16le"
  channels?: number; // 1
  sampleRate?: number;
}

// Estado global do stream de áudio do turno corrente.
let playerChild: ChildProcess | null = null;
let playerKind: "pacat" | "aplay" | "ffplay" | "fallback" | null = null;
let sampleRate = 44100;
let streamBroken = false;      // player morreu no meio (stdin EPIPE) — degrada
let streamEnded = true;        // audio_end já chegou / não há stream ativo
// Fallback (per-chunk): fila sequencial + último processo em play (p/ cancel).
let fallbackQueue: Promise<void> = Promise.resolve();
let fallbackLastChild: ChildProcess | null = null;

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

/** Player de stream (stdin PCM cru) para a plataforma, na ordem de
 * preferência. null = usar fallback por chunk. */
function resolveStreamPlayer(): { cmd: string; args: string[] } | null {
  const rate = String(sampleRate);
  if (process.platform === "win32") {
    if (commandExists("ffplay")) {
      return {
        cmd: "ffplay",
        args: ["-loglevel", "quiet", "-nodisp", "-autoexit",
               "-f", "s16le", "-ar", rate, "-ac", "1", "-i", "pipe:0"],
      };
    }
    return null;   // SoundPlayer não lê stdin — fallback por chunk
  }
  if (commandExists("pacat")) {
    return { cmd: "pacat", args: ["--raw", `--rate=${rate}`, "--channels=1", "--format=s16le"] };
  }
  if (commandExists("aplay")) {
    return { cmd: "aplay", args: ["-q", "-t", "raw", "-f", "S16_LE", "-c", "1", "-r", rate] };
  }
  if (commandExists("ffplay")) {
    return {
      cmd: "ffplay",
      args: ["-loglevel", "quiet", "-nodisp", "-autoexit",
             "-f", "s16le", "-ar", rate, "-ac", "1", "-i", "pipe:0"],
    };
  }
  return null;
}

/** audio_start — abre o player persistente do turno (ou modo fallback). */
export function startAudioStream(meta: AudioStreamMeta): void {
  cancelAudioStream();   // higiene: stream anterior (sem audio_end) morre aqui
  streamBroken = false;
  streamEnded = false;
  sampleRate = meta.sampleRate && meta.sampleRate > 0 ? meta.sampleRate : 44100;

  const spec = resolveStreamPlayer();
  if (spec) {
    try {
      playerChild = spawn(spec.cmd, spec.args, { stdio: ["pipe", "ignore", "ignore"] });
      playerKind = spec.cmd as typeof playerKind;
      playerChild.stdin?.on("error", (err) => {
        // Player morreu no meio (device removido, SIGKILL externo…) —
        // o turno segue; chunks subsequentes são descartados com aviso único.
        if (!streamBroken) {
          streamBroken = true;
          logger.warn("audio: player de stream morreu — resto do áudio descartado", {
            player: spec.cmd, message: err.message,
          });
        }
      });
      logger.info("audio: stream aberto no host", { player: spec.cmd, sampleRate });
      return;
    } catch (err) {
      logger.warn("audio: falha ao abrir player de stream — usando fallback por chunk", {
        message: err instanceof Error ? err.message : String(err),
      });
      playerChild = null;
    }
  }
  playerKind = "fallback";
  logger.info("audio: sem player de stdin — fallback por chunk (pausas entre frases)", {
    platform: process.platform,
  });
}

/** audio — chunk PCM cru base64; escreve no stdin do player (ou fallback). */
export function pushAudioChunk(audioB64: string): void {
  if (!audioB64 || streamEnded) return;
  const pcm = Buffer.from(audioB64, "base64");
  if (pcm.length === 0) return;

  if (playerKind && playerKind !== "fallback" && playerChild?.stdin) {
    if (streamBroken) return;
    playerChild.stdin.write(pcm);
    return;
  }

  // Fallback: WAV temporário por chunk (PowerShell SoundPlayer), fila
  // sequencial — frases progressivas com pausa pequena entre elas.
  if (playerKind === "fallback") {
    const rate = sampleRate;
    fallbackQueue = fallbackQueue
      .then(() => playPcmViaFallback(pcm, rate))
      .catch((err) => {
        logger.warn("audio: falha no playback por chunk", {
          message: err instanceof Error ? err.message : String(err),
        });
      });
  }
}

/** audio_end — fecha o stdin: o player drena e exita sozinho. */
export function endAudioStream(): void {
  streamEnded = true;
  if (playerChild?.stdin && !playerChild.stdin.destroyed && playerChild.stdin.writable) {
    playerChild.stdin.end();
  }
  playerChild = null;
  playerKind = null;
}

/** Cancel (Ctrl+C / erro fatal): mata o player imediatamente. */
export function cancelAudioStream(): void {
  streamEnded = true;
  if (playerChild) {
    try {
      playerChild.kill(process.platform === "win32" ? undefined : "SIGKILL");
    } catch {
      /* já morreu — ignora */
    }
    playerChild = null;
  }
  if (fallbackLastChild) {
    try {
      fallbackLastChild.kill();
    } catch {
      /* ignora */
    }
    fallbackLastChild = null;
  }
  playerKind = null;
}

/** Fim do fetch do turno: se o orchestrator encerrou limpo (audio_end
 * chegou), não faz nada — o player drena o que falta sozinho. Se o stream
 * morreu no meio (sem audio_end), mata o player pendurado. */
export function finishAudioStream(): void {
  if (!streamEnded) {
    cancelAudioStream();
  }
}

// ── Fallback: chunk PCM → WAV temporário → PowerShell SoundPlayer ───────────

function pcmToWav(pcm: Buffer, rate: number): Buffer {
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + pcm.length, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);            // PCM chunk size
  header.writeUInt16LE(1, 20);             // PCM format
  header.writeUInt16LE(1, 22);             // mono
  header.writeUInt32LE(rate, 24);
  header.writeUInt32LE(rate * 2, 28);      // byte rate (16-bit)
  header.writeUInt16LE(2, 32);             // block align
  header.writeUInt16LE(16, 34);            // bits
  header.write("data", 36);
  header.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([header, pcm]);
}

async function playPcmViaFallback(pcm: Buffer, rate: number): Promise<void> {
  if (process.platform === "win32") {
    const dir  = mkdtempSync(join(tmpdir(), "alpha-ai-tts-"));
    const file = join(dir, "fala.wav");
    writeFileSync(file, pcmToWav(pcm, rate));
    const ps = spawn(
      "powershell",
      ["-NoProfile", "-NonInteractive", "-Command",
       `(New-Object Media.SoundPlayer '${file.replace(/'/g, "''")}').PlaySync()`],
      { stdio: "ignore", windowsHide: true },
    );
    fallbackLastChild = ps;
    await new Promise<void>((resolve) => ps.on("close", () => resolve()));
    rmSync(dir, { recursive: true, force: true });
    return;
  }
  // POSIX sem pacat/aplay/ffplay: wav por arquivo via paplay se houver.
  if (commandExists("paplay")) {
    const dir  = mkdtempSync(join(tmpdir(), "alpha-ai-tts-"));
    const file = join(dir, "fala.wav");
    writeFileSync(file, pcmToWav(pcm, rate));
    const p = spawn("paplay", [file], { stdio: "ignore" });
    fallbackLastChild = p;
    await new Promise<void>((resolve) => p.on("close", () => resolve()));
    rmSync(dir, { recursive: true, force: true });
    return;
  }
  logger.warn("audio: nenhum player disponível no host (pacat/aplay/ffplay/paplay) — áudio descartado");
}
