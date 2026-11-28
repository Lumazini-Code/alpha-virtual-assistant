// Alpha AI (AVA) — Preview de HTML no Manager (host)
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
//! Preview de ARTEFATOS do usuário (HTML, PDF, imagens, modelos 3D,
//! documentos e mídia).
//!
//! O modelo grava o arquivo em disco com as mesmas tools de arquivo dos MCPs
//! (cwd da sessão). A tool `artifact_preview` só devolve `{path,title}`; o
//! orchestrator repassa o evento para a WebUI, que registra o caminho AQUI.
//! O Manager — processo do HOST — valida o cwd exatamente como
//! `mcp_server::resolve_call_cwd` e serve o artefato + assets irmãos direto
//! do disco. Nada de copiar o conteúdo para o histórico do LLM.
//!
//! Registro é POR SESSÃO (memória do tray): a lista da aba Artefatos mostra
//! apenas os arquivos publicados naquela conversa. Os arquivos em si ficam no
//! disco do usuário. O `kind` (html/pdf/image/model/document/media/code) é
//! inferido da extensão e devolvido à WebUI para escolher o renderizador.

use axum::{
    body::Body,
    extract::{ConnectInfo, Extension, Path as AxumPath},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

const MAX_PREVIEWS_PER_SESSION: usize = 40;

#[derive(Clone)]
struct PreviewEntry {
    id: String,
    session_id: String,
    path: PathBuf,
    /// CWD usado no registro: revalidar depois de trocar de pasta continua
    /// funcionando (o Manager já guardou o root, mas a WebUI reenvia o cwd).
    cwd: PathBuf,
    /// Pasta do artefato: assets irmãos são servidos a partir daqui.
    root: PathBuf,
    /// Tipo de renderização escolhido pela WebUI (ver `artifact_kind`).
    kind: String,
    title: String,
    description: String,
    created_ms: u64,
}

#[derive(Default)]
pub struct PreviewStore {
    sessions: HashMap<String, Vec<PreviewEntry>>,
    next_id: u64,
}

pub type PreviewState = Arc<Mutex<PreviewStore>>;

pub fn new_state() -> PreviewState {
    Arc::new(Mutex::new(PreviewStore::default()))
}

// ════════════════════════════════════════════════════════════════════════
// DTOs
// ════════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
pub struct RegisterRequest {
    pub session_id: String,
    #[serde(default)]
    pub cwd: Option<String>,
    pub path: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub description: String,
    /// Override opcional do renderizador (ex.: forçar `document` num `.txt`).
    #[serde(default)]
    pub kind: Option<String>,
}

#[derive(Serialize)]
struct PreviewSummary {
    id: String,
    title: String,
    description: String,
    kind: String,
    path: String,
    cwd: String,
    url: String,
    created_ms: u64,
    mtime_ms: Option<u64>,
    size: Option<u64>,
}

#[derive(Serialize)]
struct StatResponse {
    exists: bool,
    mtime_ms: Option<u64>,
    size: Option<u64>,
}

// ════════════════════════════════════════════════════════════════════════
// Handlers
// ════════════════════════════════════════════════════════════════════════

/// POST /preview/register — corpo `{session_id,cwd,path,title,description}`.
/// Resolve o caminho relativo ao cwd da sessão (mesma regra dos MCPs stdio)
/// e guarda a referência, nunca o conteúdo.
pub async fn register(
    Extension(store): Extension<PreviewState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Json(req): Json<RegisterRequest>,
) -> Response {
    if let Err(resp) = crate::api::require_loopback(addr) {
        return resp;
    }
    let session_id = req.session_id.trim().to_string();
    if !valid_session(&session_id) {
        return err(StatusCode::BAD_REQUEST, "session_id inválido");
    }
    let raw_path = req.path.trim();
    if raw_path.is_empty() {
        return err(StatusCode::BAD_REQUEST, "path vazio");
    }

    let cwd = match resolve_cwd(req.cwd.as_deref()) {
        Ok(path) => path,
        Err(message) => return err(StatusCode::UNPROCESSABLE_ENTITY, &message),
    };
    let candidate = if Path::new(raw_path).is_absolute() {
        PathBuf::from(raw_path)
    } else {
        cwd.join(raw_path)
    };
    let canonical = match std::fs::canonicalize(&candidate) {
        Ok(path) => path,
        Err(e) => {
            return err(
                StatusCode::UNPROCESSABLE_ENTITY,
                &format!("arquivo não encontrado: {} ({e})", candidate.display()),
            )
        }
    };
    // Cada chat tem um cwd por projeto, mas o modelo pode escrever artefatos
    // em qualquer lugar do home (ex.: ~/modelos-3d, ~/Downloads). Exigir só
    // o cwd fazia o registro falhar com 403 quando a mesma tool era chamada
    // de outro chat/projeto. O home é o limite externo; a allowlist de
    // extensões e o sandbox dos assets irmãos continuam valendo.
    let home = crate::api::host_home_dir();
    if !canonical.starts_with(&cwd) && !canonical.starts_with(&home) {
        return err(
            StatusCode::FORBIDDEN,
            "o artefato precisa estar dentro do cwd da sessão ou do home do usuário",
        );
    }
    if !canonical.is_file() {
        return err(StatusCode::UNPROCESSABLE_ENTITY, "path não é um arquivo");
    }
    let ext = canonical
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let detected_kind = match artifact_kind(&ext) {
        Some(kind) => kind,
        None => {
            return err(
                StatusCode::UNPROCESSABLE_ENTITY,
                "extensão não suportada; use html/pdf/imagem/modelo 3D/documento/mídia",
            )
        }
    };
    // O override só troca o renderizador — a extensão continua precisando
    // estar na allowlist (evita servir arquivos arbitrários do cwd).
    let kind = req
        .kind
        .as_deref()
        .map(str::trim)
        .filter(|k| !k.is_empty() && is_artifact_kind(k))
        .unwrap_or(detected_kind)
        .to_string();

    let default_title = canonical
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Preview")
        .to_string();
    let title = {
        let t = req.title.trim();
        if t.is_empty() {
            default_title
        } else {
            t.to_string()
        }
    };
    let root = canonical
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| cwd.clone());

    let mut guard = store.lock().unwrap_or_else(|p| p.into_inner());
    let id = next_id(&mut guard);
    let list = guard.sessions.entry(session_id.clone()).or_default();

    // Re-publicar o mesmo arquivo atualiza a entrada existente (a lista não
    // cresce a cada turno/refresh).
    if let Some(entry) = list.iter_mut().find(|e| e.path == canonical) {
        entry.title = title;
        entry.description = req.description.trim().to_string();
        entry.kind = kind;
        return Json(summary(entry)).into_response();
    }

    if list.len() >= MAX_PREVIEWS_PER_SESSION {
        list.remove(0);
    }
    let entry = PreviewEntry {
        id,
        session_id,
        path: canonical,
        root,
        cwd,
        kind,
        title,
        description: req.description.trim().to_string(),
        created_ms: now_ms(),
    };
    list.push(entry.clone());
    Json(summary(&entry)).into_response()
}

/// GET /preview/{session} — lista os artefatos publicados naquela sessão.
pub async fn list(
    Extension(store): Extension<PreviewState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    AxumPath(session): AxumPath<String>,
) -> Response {
    if let Err(resp) = crate::api::require_loopback(addr) {
        return resp;
    }
    let guard = store.lock().unwrap_or_else(|p| p.into_inner());
    let items: Vec<PreviewSummary> = guard
        .sessions
        .get(&session)
        .map(|list| list.iter().map(summary).collect())
        .unwrap_or_default();
    Json(items).into_response()
}

/// DELETE /preview/{session} — limpa o registro quando a conversa é apagada.
pub async fn clear_session(
    Extension(store): Extension<PreviewState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    AxumPath(session): AxumPath<String>,
) -> Response {
    if let Err(resp) = crate::api::require_loopback(addr) {
        return resp;
    }
    let mut guard = store.lock().unwrap_or_else(|p| p.into_inner());
    guard.sessions.remove(&session);
    Json(serde_json::json!({ "ok": true })).into_response()
}

/// GET /preview/{session}/{id}/ — serve o artefato registrado.
pub async fn get_document(
    Extension(store): Extension<PreviewState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    AxumPath((session, id)): AxumPath<(String, String)>,
) -> Response {
    if let Err(resp) = crate::api::require_loopback(addr) {
        return resp;
    }
    let Some(entry) = find_entry(&store, &session, &id) else {
        return err(StatusCode::NOT_FOUND, "preview não registrado");
    };
    serve_file(&entry, "").await
}

/// GET /preview/{session}/{id}/{*path} — serve assets irmãos do artefato.
pub async fn get_asset(
    Extension(store): Extension<PreviewState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    AxumPath((session, id, asset)): AxumPath<(String, String, String)>,
) -> Response {
    if let Err(resp) = crate::api::require_loopback(addr) {
        return resp;
    }
    let Some(entry) = find_entry(&store, &session, &id) else {
        return err(StatusCode::NOT_FOUND, "preview não registrado");
    };
    serve_file(&entry, &asset).await
}

/// GET /preview-stat/{session}/{id} — mtime/size para auto-refresh.
pub async fn get_stat(
    Extension(store): Extension<PreviewState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    AxumPath((session, id)): AxumPath<(String, String)>,
) -> Response {
    if let Err(resp) = crate::api::require_loopback(addr) {
        return resp;
    }
    let Some(entry) = find_entry(&store, &session, &id) else {
        return err(StatusCode::NOT_FOUND, "preview não registrado");
    };
    let (exists, mtime_ms, size) = stat_path(&entry.path);
    Json(StatResponse {
        exists,
        mtime_ms,
        size,
    })
    .into_response()
}

// ════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════

/// Extensões aceitas e o renderizador correspondente na WebUI. Manter em
/// sincronia com `_artifact_kind_for_path()` do alpha-mcp (o Manager manda).
fn artifact_kind(ext: &str) -> Option<&'static str> {
    match ext {
        "html" | "htm" => Some("html"),
        "pdf" => Some("pdf"),
        "png" | "jpg" | "jpeg" | "gif" | "webp" | "svg" | "bmp" | "avif" | "ico" => {
            Some("image")
        }
        "glb" | "gltf" | "stl" | "obj" => Some("model"),
        "md" | "markdown" | "txt" | "text" | "json" | "jsonl" | "csv" | "tsv" | "yaml"
        | "yml" | "xml" | "log" | "rst" => Some("document"),
        "mp4" | "webm" | "ogv" | "mov" | "mp3" | "wav" | "ogg" | "m4a" | "flac" => {
            Some("media")
        }
        "py" | "js" | "mjs" | "ts" | "tsx" | "jsx" | "css" | "rs" | "go" | "java" | "c"
        | "cpp" | "h" | "hpp" | "sql" | "sh" | "ps1" | "toml" | "ini" => Some("code"),
        _ => None,
    }
}

fn is_artifact_kind(kind: &str) -> bool {
    matches!(
        kind,
        "html" | "pdf" | "image" | "model" | "document" | "media" | "code"
    )
}

fn valid_session(session: &str) -> bool {
    !session.is_empty()
        && session.len() <= 128
        && session
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Mesma resolução do `mcp_server::resolve_call_cwd`: relativo ao default do
/// host (home do usuário), absoluto aceito, canonicalizado e precisa ser dir.
fn resolve_cwd(requested: Option<&str>) -> Result<PathBuf, String> {
    let default = crate::api::host_home_dir();
    let candidate = match requested.map(str::trim).filter(|p| !p.is_empty()) {
        Some(raw) => {
            let path = PathBuf::from(raw);
            if path.is_absolute() {
                path
            } else {
                default.join(path)
            }
        }
        None => default,
    };
    let canonical = std::fs::canonicalize(&candidate)
        .map_err(|e| format!("cwd inválido '{}': {e}", candidate.display()))?;
    if !canonical.is_dir() {
        return Err(format!("cwd '{}' não é um diretório", canonical.display()));
    }
    Ok(canonical)
}

fn find_entry(store: &PreviewState, session: &str, id: &str) -> Option<PreviewEntry> {
    let guard = store.lock().unwrap_or_else(|p| p.into_inner());
    guard
        .sessions
        .get(session)
        .and_then(|list| list.iter().find(|e| e.id == id).cloned())
}

fn summary(entry: &PreviewEntry) -> PreviewSummary {
    let (_, mtime_ms, size) = stat_path(&entry.path);
    PreviewSummary {
        id: entry.id.clone(),
        title: entry.title.clone(),
        description: entry.description.clone(),
        kind: entry.kind.clone(),
        path: entry.path.to_string_lossy().into_owned(),
        cwd: entry.cwd.to_string_lossy().into_owned(),
        // URL same-origin do Manager. O iframe usa caminho relativo na WebUI.
        url: format!("/preview/{}/{}/", entry.session_id, entry.id),
        created_ms: entry.created_ms,
        mtime_ms,
        size,
    }
}

fn stat_path(path: &Path) -> (bool, Option<u64>, Option<u64>) {
    match std::fs::metadata(path) {
        Ok(meta) if meta.is_file() => (
            true,
            meta.modified().ok().map(system_time_ms),
            Some(meta.len()),
        ),
        _ => (false, None, None),
    }
}

async fn serve_file(entry: &PreviewEntry, asset: &str) -> Response {
    let rel = asset.trim_start_matches('/');
    let target = if rel.is_empty() {
        entry.path.clone()
    } else {
        entry.root.join(rel)
    };
    let canonical = match std::fs::canonicalize(&target) {
        Ok(path) => path,
        Err(_) => return err(StatusCode::NOT_FOUND, "arquivo não encontrado"),
    };
    if !canonical.starts_with(&entry.root) {
        return err(StatusCode::FORBIDDEN, "asset fora da pasta do preview");
    }
    if !canonical.is_file() {
        return err(StatusCode::NOT_FOUND, "arquivo não encontrado");
    }
    let bytes = match tokio::fs::read(&canonical).await {
        Ok(bytes) => bytes,
        Err(_) => return err(StatusCode::NOT_FOUND, "arquivo não encontrado"),
    };
    let content_type = content_type_for(&canonical);
    // SVG e XML também podem executar script num documento de topo: recebem
    // o mesmo sandbox de origem opaca do HTML.
    let sandboxed = matches!(
        canonical
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
            .as_str(),
        "html" | "htm" | "svg" | "xml"
    );
    let mut response = Response::new(Body::from(bytes));
    if let Ok(value) = HeaderValue::from_str(content_type) {
        response.headers_mut().insert(header::CONTENT_TYPE, value);
    }
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-store, max-age=0"),
    );
    response.headers_mut().insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    // A origem do iframe é opaca (sandbox sem allow-same-origin). Liberar os
    // assets do próprio preview permite `fetch`/módulos ES para arquivos
    // irmãos; as APIs internas da WebUI continuam fora de alcance.
    response.headers_mut().insert(
        header::ACCESS_CONTROL_ALLOW_ORIGIN,
        HeaderValue::from_static("*"),
    );
    if sandboxed {
        // Sandbox também no acesso direto (abrir em nova janela): scripts
        // rodam, mas o documento fica em origem opaca e não alcança a API
        // interna da WebUI. Assets irmãos continuam carregando por markup.
        response.headers_mut().insert(
            header::CONTENT_SECURITY_POLICY,
            HeaderValue::from_static(
                "sandbox allow-scripts allow-forms allow-modals allow-popups \
                 allow-downloads; frame-ancestors 'self'",
            ),
        );
    }
    response
}

fn content_type_for(path: &Path) -> &'static str {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "html" | "htm" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" | "mjs" => "application/javascript; charset=utf-8",
        "json" | "map" => "application/json; charset=utf-8",
        "jsonl" => "application/x-ndjson; charset=utf-8",
        "md" | "markdown" => "text/markdown; charset=utf-8",
        "csv" => "text/csv; charset=utf-8",
        "tsv" => "text/tab-separated-values; charset=utf-8",
        "yaml" | "yml" => "text/yaml; charset=utf-8",
        "xml" => "application/xml; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "avif" => "image/avif",
        "bmp" => "image/bmp",
        "ico" => "image/x-icon",
        "glb" => "model/gltf-binary",
        "gltf" => "model/gltf+json",
        "stl" => "model/stl",
        // OBJLoader lê texto; text/plain evita downloads e mantém o CORS simples.
        "obj" => "text/plain; charset=utf-8",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        "ttf" => "font/ttf",
        "otf" => "font/otf",
        "wasm" => "application/wasm",
        "txt" | "text" | "log" | "rst" | "py" | "ts" | "tsx" | "jsx" | "rs" | "go"
        | "java" | "c" | "cpp" | "h" | "hpp" | "sql" | "sh" | "ps1" | "toml" | "ini" => {
            "text/plain; charset=utf-8"
        }
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "ogv" => "video/ogg",
        "mov" => "video/quicktime",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "m4a" => "audio/mp4",
        "flac" => "audio/flac",
        "pdf" => "application/pdf",
        _ => "application/octet-stream",
    }
}

fn next_id(store: &mut PreviewStore) -> String {
    store.next_id = store.next_id.saturating_add(1);
    format!("p{:x}{:x}", now_ms(), store.next_id)
}

fn now_ms() -> u64 {
    system_time_ms(SystemTime::now())
}

fn system_time_ms(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn err(status: StatusCode, message: &str) -> Response {
    (status, Json(serde_json::json!({ "error": message }))).into_response()
}

#[cfg(test)]
mod tests {
    use super::{artifact_kind, is_artifact_kind};

    #[test]
    fn known_extensions_map_to_renderers() {
        for (ext, expected) in [
            ("html", "html"), ("htm", "html"), ("pdf", "pdf"),
            ("png", "image"), ("svg", "image"), ("webp", "image"),
            ("glb", "model"), ("gltf", "model"), ("stl", "model"), ("obj", "model"),
            ("md", "document"), ("json", "document"), ("csv", "document"),
            ("mp4", "media"), ("mp3", "media"), ("py", "code"), ("rs", "code"),
        ] {
            assert_eq!(artifact_kind(ext), Some(expected), ".{ext}");
            assert!(is_artifact_kind(expected));
        }
    }

    #[test]
    fn unknown_extensions_are_rejected() {
        for ext in ["exe", "env", "dll", "pdfx", ""] {
            assert_eq!(artifact_kind(ext), None, ".{ext}");
        }
        assert!(!is_artifact_kind("execute"));
        assert!(!is_artifact_kind(""));
    }
}
