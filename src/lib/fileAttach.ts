import { readFileSync } from "node:fs";
import { extname } from "node:path";
import { IMAGE_EXTS, DOC_EXTS } from "./constants.js";

export type AttachedFile =
  | { kind: "image"; base64: string; path: string; mime: string }
  | { kind: "document"; path: string; content: string };

/** Detecta se é imagem pela extensão */
export function isImageFile(path: string): boolean {
  const ext = extname(path).slice(1).toLowerCase();
  return IMAGE_EXTS.has(ext);
}

/** Detecta se é documento indexável pela extensão */
export function isDocumentFile(path: string): boolean {
  const ext = extname(path).slice(1).toLowerCase();
  return DOC_EXTS.has(ext);
}

/** Converte imagem para base64 (lógica existente do useOrchestratorStream) */
export function imageToBase64(path: string): { base64: string; mime: string } {
  const buf = readFileSync(path);
  const base64 = buf.toString("base64");
  const mime = detectImageMime(base64);
  return { base64, mime };
}

/** Sniff MIME por magic bytes (copiado do orchestrator) */
function detectImageMime(b64: string): string {
  const payload = b64.split(",", 1)[-1];
  try {
    const head = Buffer.from(payload.slice(0, 64), "base64").subarray(0, 12);
    if (head.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]))) return "image/png";
    if (head.subarray(0, 3).equals(Buffer.from([0xFF, 0xD8, 0xFF]))) return "image/jpeg";
    if (head.subarray(0, 6).equals(Buffer.from("GIF87a")) || head.subarray(0, 6).equals(Buffer.from("GIF89a"))) return "image/gif";
    if (head.subarray(0, 4).equals(Buffer.from("RIFF")) && head.subarray(8, 12).equals(Buffer.from("WEBP"))) return "image/webp";
  } catch {}
  return "image/jpeg";
}

/**
 * Extrai texto de arquivo usando leitura direta.
 * - Texto puro (code, md, txt, json, etc.): leitura UTF-8 direta (rápido).
 * - PDF/DOCX/HTML/PPTX/XLSX: usa markitdown (Python) via subprocesso, com
 *   timeout de 30s e fallback para UTF-8 se indisponível.
 */
export async function extractTextContent(path: string): Promise<string> {
  const ext = extname(path).slice(1).toLowerCase();
  const MARKITDOWN_EXTS = new Set(["pdf", "docx", "doc", "html", "htm", "pptx", "xlsx", "xls"]);

  // Arquivos de texto puro: leitura direta
  if (!MARKITDOWN_EXTS.has(ext)) {
    try {
      return readFileSync(path, "utf-8");
    } catch {
      return readFileSync(path).toString("utf-8");
    }
  }

  // PDF/DOCX/HTML/etc.: tentar markitdown
  // python path: .venv/bin/python (Linux/macOS) ou .venv/Scripts/python.exe (Windows)
  const isWin = process.platform === "win32";
  const pythonPath = process.env.AVA_PYTHON_PATH ||
    (isWin ? ".venv/Scripts/python.exe" : ".venv/bin/python");

  const { spawn } = await import("node:child_process");
  const MARKITDOWN_TIMEOUT_MS = 30_000;

  return new Promise((resolve) => {
    const chunks: string[] = [];
    const proc = spawn(pythonPath, [
      "-c",
      `import sys\nfrom markitdown import MarkItDown\nmd = MarkItDown()\nresult = md.convert(sys.argv[1])\nprint(result.text_content if hasattr(result, "text_content") else result.text, end=")`,
      path,
    ], { stdio: ["ignore", "pipe", "pipe"] });

    const timer = setTimeout(() => {
      proc.kill("SIGKILL");
      fallbackRead(path, resolve, "markitdown excedeu timeout");
    }, MARKITDOWN_TIMEOUT_MS);

    proc.stdout.on("data", (d) => chunks.push(d.toString()));
    proc.stderr.on("data", (d) => {
      // Log não-fatal: markitdown pode cuspir warnings (ex.: PDF sem OCR)
      console.error(`markitdown stderr: ${d.toString().trim()}`);
    });

    proc.on("close", (code) => {
      clearTimeout(timer);
      if (code === 0) {
        const text = chunks.join("");
        if (text.trim()) resolve(text);
        else fallbackRead(path, resolve, "markitdown retornou conteúdo vazio");
      } else {
        fallbackRead(path, resolve, `markitdown exit code ${code}`);
      }
    });

    proc.on("error", (err) => {
      clearTimeout(timer);
      fallbackRead(path, resolve, err.message);
    });
  });
}

/** Fallback: leitura UTF-8 direta (pode produzir lixo para binários, mas não quebra o fluxo). */
function fallbackRead(path: string, resolve: (s: string) => void, reason: string): void {
  console.error(`fileAttach: markitdown falhou (${reason}), lendo como UTF-8`);
  try {
    resolve(readFileSync(path, "utf-8"));
  } catch {
    resolve(readFileSync(path).toString("utf-8"));
  }
}

