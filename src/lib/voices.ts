import { MEMORY_API_URL, ORCHESTRATOR_URL } from "./constants.js";
import { logger } from "./logger.js";
import { authHeaders } from "./auth.js";

/**
 * Cliente do dicionário de locutores (cards de pessoa + embeddings de voz).
 *
 * - Listagem/edição/remoção vão DIRETO na API do memory (:3000,
 *   /voice-dict/*) — os embeddings ficam no DB + FAISS da memory.
 * - O cadastro (gravação de voz) passa pelo orchestrator (:9000,
 *   /voice/register), que gera o embedding via transcribe (CAM++ /
 *   /speaker/embed) e grava via /voice-dict/write na memory — a descrição
 *   replica para a memória de longo prazo (source="voice_dict").
 */

/** Card de pessoa do dicionário de voz (espelha VoiceDictEntry da memory). */
export interface VoiceEntry {
  person_id: number;
  person_name: string;
  description: string;
  likes?: string;
  religion?: string;
  additional_info?: string;
  source: string;
  confidence: number;
  memory_id?: number | null;
  examples_count: number;
  created_at: number;
  access_count: number;
}

/** Resposta de POST /voice/register (orchestrator). */
export interface VoiceRegisterResult {
  stored: boolean;
  reason?: string;
  person_id?: number | null;
  embedding_id?: number | null;
  memory_id?: number | null;
  new_person?: boolean;
}

interface VoiceRegisterInput {
  person_name: string;
  description: string;
  audio_b64: string;
  sample_rate: number;
}

// Erros de negócio chegam como HTTP 4xx {"error": detail} ou {"detail": ...}
async function errorDetail(res: Response): Promise<string> {
  try {
    const j = (await res.json()) as { error?: string; detail?: string };
    return j.error ?? j.detail ?? `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
}

/** GET /voice-dict/list — todos os cards de pessoa (com contagem de exemplos).
 * A memory (PeopleDB unificado) responde {"total", "people"} — lê `people`;
 * `voices` fica como fallback por compatibilidade com versões antigas da API. */
export async function listVoices(): Promise<VoiceEntry[]> {
  const res = await fetch(`${MEMORY_API_URL}/voice-dict/list`, {
    method: "GET",
    headers: authHeaders({}),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const data = (await res.json()) as {
    total?: number;
    people?: VoiceEntry[];
    voices?: VoiceEntry[];
  };
  return data.people ?? data.voices ?? [];
}

/** PATCH /voice-dict/{id} — edita a descrição do card (campos não enviados são preservados). */
export async function updateVoiceDescription(
  personId: number,
  description: string,
): Promise<VoiceEntry> {
  const res = await fetch(`${MEMORY_API_URL}/voice-dict/${personId}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders({}),
    },
    body: JSON.stringify({ description }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const data = (await res.json()) as { entry?: VoiceEntry };
  logger.info("voices: descrição atualizada", { personId });
  return data.entry ?? (data as unknown as VoiceEntry);
}

/** DELETE /voice-dict/{id} — remove o card + todos os embeddings (DB e FAISS). */
export async function deleteVoice(personId: number): Promise<{ deleted: boolean }> {
  const res = await fetch(`${MEMORY_API_URL}/voice-dict/${personId}`, {
    method: "DELETE",
    headers: authHeaders({}),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const data = (await res.json()) as { deleted?: boolean };
  logger.info("voices: card removido", { personId });
  return { deleted: data.deleted ?? true };
}

/**
 * POST /voice/register — cadastro de locutor. Envia PCM cru (s16le mono,
 * base64) ao orchestrator; ele gera o embedding (transcribe/CAM++) e
 * persiste via /voice-dict/write na memory. Pode demorar (embedding
 * on-demand), por isso o timeout generoso.
 */
export async function registerVoice(input: VoiceRegisterInput): Promise<VoiceRegisterResult> {
  const res = await fetch(`${ORCHESTRATOR_URL}/voice/register`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders({}),
    },
    body: JSON.stringify({
      person_name: input.person_name,
      description: input.description,
      audio_b64: input.audio_b64,
      sample_rate: input.sample_rate,
    }),
    signal: AbortSignal.timeout(60_000),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const data = (await res.json()) as VoiceRegisterResult;
  logger.info("voices: embedding gravado", {
    personId: data.person_id,
    embeddingId: data.embedding_id,
    newPerson: data.new_person,
  });
  return data;
}
