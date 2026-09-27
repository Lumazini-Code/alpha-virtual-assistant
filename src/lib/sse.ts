import type { SseFrame } from "./types.js";

/**
 * Desfaz exatamente o que `_sse(event, data)` monta no orchestrator.py:
 *
 *   event: <name>\n
 *   data: <linha1>\n
 *   data: <linha2>\n   (se o payload original tinha múltiplas linhas)
 *   \n                  (frame termina em linha em branco)
 *
 * Cada frame é devolvido com `raw` = as linhas de "data:" já rejuntadas
 * com "\n" (reconstituindo o payload original, que pode ser uma string
 * simples ou um JSON serializado numa linha só).
 */
export async function* parseSseStream(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<SseFrame> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const flushFrame = function* (block: string): Generator<SseFrame> {
    if (!block.trim()) return;
    let eventName = "message";
    const dataLines: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) {
        eventName = line.slice("event:".length).trim();
      } else if (line.startsWith("data:")) {
        // remove só o primeiro espaço após "data:", igual convenção SSE
        dataLines.push(line.slice("data:".length).replace(/^ /, ""));
      }
    }
    if (dataLines.length > 0) {
      yield { event: eventName, raw: dataLines.join("\n") };
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      yield* flushFrame(block);
    }
  }

  // eventual último frame sem \n\n final
  if (buffer.trim()) {
    yield* flushFrame(buffer);
  }
}

/** Tenta JSON.parse; se falhar, devolve a string crua (payloads tipo delta/reasoning/result). */
export function tryParseJson<T>(raw: string): T | string {
  try {
    return JSON.parse(raw) as T;
  } catch {
    return raw;
  }
}
