import { API_BASE } from "./constants.js";

// Espelha exatamente o que scan_models() devolve em GET /models (api.rs).
// Campos confirmados pelo uso em post_llama_start: `path` e `mmproj_path`.
interface ModelInfoDto {
  path: string;
  mmproj_path?: string | null;
  [key: string]: unknown;
}

export interface DiscoveredModel {
  path: string;
  name: string;
  hasMmproj: boolean;
}

/**
 * GET /models na API Rust (api.rs) — lista os .gguf disponíveis em
 * ./Models. Em erro, a API devolve { ok: false, message } com HTTP 500;
 * aqui simplesmente tratamos como lista vazia e quem chamar cai para
 * entrada manual do caminho.
 */
export async function fetchAvailableModels(): Promise<DiscoveredModel[]> {
  try {
    const res = await fetch(`${API_BASE}/models`);
    if (!res.ok) return [];
    const data = (await res.json()) as unknown;
    if (!Array.isArray(data)) return [];

    return (data as ModelInfoDto[])
      .filter((m) => typeof m?.path === "string" && m.path.length > 0)
      .map((m) => ({
        path: m.path,
        name: baseName(m.path),
        hasMmproj: Boolean(m.mmproj_path),
      }));
  } catch {
    return [];
  }
}

function baseName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}
