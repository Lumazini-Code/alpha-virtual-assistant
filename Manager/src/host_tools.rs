// Alpha AI (AVA) — Ferramentas de host expostas pelo Manager
// Copyright (C) 2026  Felipe lumazini
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. See <https://www.gnu.org/licenses/>.
//! Leitura, busca e shell NO HOST para as tools do `alpha-mcp`.
//!
//! O `alpha-mcp` roda em container e não vê o filesystem do usuário. Estas
//! rotas rodam no processo do tray (host), com o mesmo usuário do desktop,
//! e são chamadas via orchestrator (`/host/*` → `PROCESS_MANAGER_URL`).
//!
//! Segurança: TODAS as rotas exigem `Authorization: Bearer <AVA_INTERNAL_TOKEN>`
//! e falham fechadas se o token não estiver configurado. O shell é intencional
//! (o usuário pediu acesso ao host sem isolamento), mas só o token interno
//! consegue chamá-lo; a porta :9001 continua restrita ao loopback.

use axum::{
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, Instant},
};
use tokio::process::Command;

const MAX_READ_BYTES: u64 = 8 * 1024 * 1024;
// Base64 expande ~33%; 7 MiB de imagem viram ~9,3 M caracteres, dentro do
// limite de 10 M do campo `images[].data` no orchestrator.
const MAX_IMAGE_BYTES: u64 = 7 * 1024 * 1024;
const MAX_OUTPUT_CHARS: usize = 200_000;

// ════════════════════════════════════════════════════════════════════════
// Auth / paths
// ════════════════════════════════════════════════════════════════════════

fn require_token(headers: &HeaderMap) -> Result<(), Response> {
    let expected = std::env::var("AVA_INTERNAL_TOKEN")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    let Some(expected) = expected else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "error": "AVA_INTERNAL_TOKEN não configurado no Manager" })),
        )
            .into_response());
    };
    let supplied = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("")
        .trim();
    if constant_time_eq(supplied, &expected) {
        Ok(())
    } else {
        Err((
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": "token interno inválido" })),
        )
            .into_response())
    }
}

fn constant_time_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut diff = (a.len() ^ b.len()) as u8;
    let n = a.len().max(b.len());
    for i in 0..n {
        diff |= a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0);
    }
    diff == 0
}

fn expand_path(base: &Path, raw: &str) -> PathBuf {
    let raw = raw.trim();
    if raw.is_empty() {
        return base.to_path_buf();
    }
    if raw == "~" {
        return base.to_path_buf();
    }
    if let Some(rest) = raw.strip_prefix("~/").or_else(|| raw.strip_prefix("~\\")) {
        return base.join(rest);
    }
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

fn resolve_cwd(raw: Option<&str>) -> Result<PathBuf, Response> {
    let base = crate::api::host_home_dir();
    let candidate = expand_path(&base, raw.unwrap_or("."));
    let canonical = std::fs::canonicalize(&candidate).map_err(|e| {
        (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": format!("cwd inválido '{}': {e}", candidate.display()) })),
        )
            .into_response()
    })?;
    if !canonical.is_dir() {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": format!("cwd '{}' não é um diretório", canonical.display()) })),
        )
            .into_response());
    }
    Ok(canonical)
}

fn resolve_path(cwd: &Path, raw: &str) -> PathBuf {
    expand_path(cwd, raw)
}

fn truncate_chars(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let mut out: String = text.chars().take(limit).collect();
    out.push_str("\n[…saída truncada…]");
    out
}

/// Codificação base64 padrão (RFC 4648), sem depender de um crate extra.
fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(((data.len() + 2) / 3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// MIME pela assinatura binária — o mesmo conjunto que o orchestrator usa
/// para os anexos enviados pela WebUI.
fn detect_image_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else if bytes.starts_with(b"BM") {
        Some("image/bmp")
    } else {
        None
    }
}

// ════════════════════════════════════════════════════════════════════════
// DTOs
// ════════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
pub struct ReadFileRequest {
    pub path: String,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub offset: Option<usize>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Deserialize)]
pub struct ReadImageRequest {
    pub path: String,
    #[serde(default)]
    pub cwd: Option<String>,
}

#[derive(Deserialize)]
pub struct ListDirRequest {
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub cwd: Option<String>,
}

#[derive(Deserialize)]
pub struct GlobRequest {
    pub pattern: String,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub max_results: Option<usize>,
}

#[derive(Deserialize)]
pub struct GrepRequest {
    pub pattern: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub max_results: Option<usize>,
    #[serde(default)]
    pub fixed_strings: bool,
}

#[derive(Deserialize)]
pub struct ShellRequest {
    pub command: String,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub timeout_s: Option<u64>,
}

#[derive(Deserialize)]
pub struct WriteFileRequest {
    pub path: String,
    pub content: String,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub create_dirs: Option<bool>,
}

#[derive(Deserialize)]
pub struct EditFileRequest {
    pub path: String,
    pub old_string: String,
    pub new_string: String,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub replace_all: bool,
}

// ════════════════════════════════════════════════════════════════════════
// Handlers
// ════════════════════════════════════════════════════════════════════════

pub async fn read_file(
    headers: HeaderMap,
    Json(req): Json<ReadFileRequest>,
) -> Response {
    if let Err(resp) = require_token(&headers) {
        return resp;
    }
    let cwd = match resolve_cwd(req.cwd.as_deref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let path = resolve_path(&cwd, &req.path);
    let meta = match tokio::fs::metadata(&path).await {
        Ok(m) => m,
        Err(e) => return err(StatusCode::NOT_FOUND, format!("arquivo não encontrado: {e}")),
    };
    if !meta.is_file() {
        return err(StatusCode::UNPROCESSABLE_ENTITY, "path não é um arquivo");
    }
    if meta.len() > MAX_READ_BYTES {
        return err(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("arquivo grande demais ({} bytes; teto {MAX_READ_BYTES})", meta.len()),
        );
    }
    let bytes = match tokio::fs::read(&path).await {
        Ok(v) => v,
        Err(e) => return err(StatusCode::NOT_FOUND, format!("falha ao ler arquivo: {e}")),
    };
    if bytes.contains(&0) {
        return err(StatusCode::UNPROCESSABLE_ENTITY, "arquivo binário (contém NUL)");
    }
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();
    let offset = req.offset.unwrap_or(1).max(1);
    let limit = req.limit.unwrap_or(2000).clamp(1, 10_000);
    let start = (offset - 1).min(total);
    let end = (start + limit).min(total);
    let content = lines[start..end].join("\n");
    Json(json!({
        "path": path.to_string_lossy(),
        "offset": offset,
        "limit": limit,
        "total_lines": total,
        "truncated": end < total,
        "content": content,
    }))
    .into_response()
}

/// Lê uma imagem do host e devolve base64 puro para o orchestrator embutir
/// como bloco `image_url` na mensagem da tool `view_image`.
pub async fn read_image(
    headers: HeaderMap,
    Json(req): Json<ReadImageRequest>,
) -> Response {
    if let Err(resp) = require_token(&headers) {
        return resp;
    }
    let cwd = match resolve_cwd(req.cwd.as_deref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let path = resolve_path(&cwd, &req.path);
    let meta = match tokio::fs::metadata(&path).await {
        Ok(m) => m,
        Err(e) => return err(StatusCode::NOT_FOUND, format!("imagem não encontrada: {e}")),
    };
    if !meta.is_file() {
        return err(StatusCode::UNPROCESSABLE_ENTITY, "path não é um arquivo");
    }
    if meta.len() > MAX_IMAGE_BYTES {
        return err(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("imagem grande demais ({} bytes; teto {MAX_IMAGE_BYTES})", meta.len()),
        );
    }
    let bytes = match tokio::fs::read(&path).await {
        Ok(v) => v,
        Err(e) => return err(StatusCode::NOT_FOUND, format!("falha ao ler imagem: {e}")),
    };
    let mime = match detect_image_mime(&bytes) {
        Some(m) => m,
        None => return err(
            StatusCode::UNPROCESSABLE_ENTITY,
            "formato de imagem não suportado (use PNG, JPEG, GIF, WEBP ou BMP)",
        ),
    };
    Json(json!({
        "path": path.to_string_lossy(),
        "mime": mime,
        "size": bytes.len(),
        "base64": base64_encode(&bytes),
    }))
    .into_response()
}

pub async fn list_dir(
    headers: HeaderMap,
    Json(req): Json<ListDirRequest>,
) -> Response {
    if let Err(resp) = require_token(&headers) {
        return resp;
    }
    let cwd = match resolve_cwd(req.cwd.as_deref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let path = resolve_path(&cwd, req.path.as_deref().unwrap_or("."));
    let mut rd = match tokio::fs::read_dir(&path).await {
        Ok(v) => v,
        Err(e) => return err(StatusCode::NOT_FOUND, format!("diretório não encontrado: {e}")),
    };
    let mut entries = Vec::new();
    while let Ok(Some(entry)) = rd.next_entry().await {
        if entries.len() >= 2000 {
            break;
        }
        let meta = entry.metadata().await.ok();
        entries.push(json!({
            "name": entry.file_name().to_string_lossy(),
            "path": entry.path().to_string_lossy(),
            "is_dir": meta.as_ref().map(|m| m.is_dir()).unwrap_or(false),
            "size": meta.map(|m| m.len()).unwrap_or(0),
        }));
    }
    entries.sort_by(|a, b| {
        a["name"].as_str().unwrap_or("").cmp(b["name"].as_str().unwrap_or(""))
    });
    Json(json!({
        "path": path.to_string_lossy(),
        "count": entries.len(),
        "entries": entries,
    }))
    .into_response()
}

pub async fn glob_files(
    headers: HeaderMap,
    Json(req): Json<GlobRequest>,
) -> Response {
    if let Err(resp) = require_token(&headers) {
        return resp;
    }
    let cwd = match resolve_cwd(req.cwd.as_deref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let pattern = if Path::new(&req.pattern).is_absolute() {
        req.pattern.clone()
    } else {
        cwd.join(&req.pattern).to_string_lossy().into_owned()
    };
    let max = req.max_results.unwrap_or(500).clamp(1, 5000);
    let paths = match glob::glob(&pattern) {
        Ok(v) => v,
        Err(e) => return err(StatusCode::BAD_REQUEST, format!("pattern inválido: {e}")),
    };
    let mut out = Vec::new();
    for entry in paths.flatten() {
        let meta = std::fs::metadata(&entry).ok();
        out.push(json!({
            "path": entry.to_string_lossy(),
            "is_dir": meta.as_ref().map(|m| m.is_dir()).unwrap_or(false),
            "size": meta.map(|m| m.len()).unwrap_or(0),
        }));
        if out.len() >= max {
            break;
        }
    }
    Json(json!({ "pattern": pattern, "count": out.len(), "matches": out })).into_response()
}

pub async fn grep_files(
    headers: HeaderMap,
    Json(req): Json<GrepRequest>,
) -> Response {
    if let Err(resp) = require_token(&headers) {
        return resp;
    }
    let cwd = match resolve_cwd(req.cwd.as_deref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let target = resolve_path(&cwd, req.path.as_deref().unwrap_or("."));
    let max = req.max_results.unwrap_or(200).clamp(1, 2000);
    let fixed = req.fixed_strings;

    let rg = run_search("rg", &cwd, &target, &req.pattern, fixed, max).await;
    let (stdout, stderr, code) = match rg {
        Ok(v) => v,
        Err(_) => match run_search("grep", &cwd, &target, &req.pattern, fixed, max).await {
            Ok(v) => v,
            Err(e) => return err(StatusCode::BAD_GATEWAY, format!("busca indisponível: {e}")),
        },
    };
    if code > 1 {
        return err(
            StatusCode::BAD_GATEWAY,
            format!("busca falhou (código {code}): {}", truncate_chars(&stderr, 2000)),
        );
    }
    let mut matches: Vec<String> = stdout.lines().take(max).map(str::to_string).collect();
    let truncated = stdout.lines().count() > matches.len();
    matches.shrink_to_fit();
    Json(json!({
        "path": target.to_string_lossy(),
        "pattern": req.pattern,
        "count": matches.len(),
        "truncated": truncated,
        "matches": matches,
    }))
    .into_response()
}

pub async fn write_file(
    headers: HeaderMap,
    Json(req): Json<WriteFileRequest>,
) -> Response {
    if let Err(resp) = require_token(&headers) {
        return resp;
    }
    let cwd = match resolve_cwd(req.cwd.as_deref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let path = resolve_path(&cwd, &req.path);
    let existed = tokio::fs::metadata(&path).await.is_ok();
    let bytes = req.content.into_bytes();
    if bytes.len() as u64 > MAX_READ_BYTES {
        return err(StatusCode::PAYLOAD_TOO_LARGE, "conteúdo grande demais");
    }
    if req.create_dirs.unwrap_or(true) {
        if let Some(parent) = path.parent() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                return err(StatusCode::UNPROCESSABLE_ENTITY, format!("falha ao criar diretório: {e}"));
            }
        }
    }
    match tokio::fs::write(&path, &bytes).await {
        Ok(()) => Json(json!({
            "ok": true,
            "path": path.to_string_lossy(),
            "bytes": bytes.len(),
            "created": !existed,
        }))
        .into_response(),
        Err(e) => err(StatusCode::UNPROCESSABLE_ENTITY, format!("falha ao gravar arquivo: {e}")),
    }
}

pub async fn edit_file(
    headers: HeaderMap,
    Json(req): Json<EditFileRequest>,
) -> Response {
    if let Err(resp) = require_token(&headers) {
        return resp;
    }
    if req.old_string.is_empty() {
        return err(StatusCode::BAD_REQUEST, "old_string vazio");
    }
    let cwd = match resolve_cwd(req.cwd.as_deref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let path = resolve_path(&cwd, &req.path);
    let bytes = match tokio::fs::read(&path).await {
        Ok(v) => v,
        Err(e) => return err(StatusCode::NOT_FOUND, format!("arquivo não encontrado: {e}")),
    };
    if bytes.contains(&0) {
        return err(StatusCode::UNPROCESSABLE_ENTITY, "arquivo binário (contém NUL)");
    }
    let text = String::from_utf8_lossy(&bytes).into_owned();
    let count = text.matches(&req.old_string).count();
    if count == 0 {
        return err(StatusCode::UNPROCESSABLE_ENTITY, "old_string não encontrado no arquivo");
    }
    if count > 1 && !req.replace_all {
        return err(
            StatusCode::CONFLICT,
            format!("old_string aparece {count} vezes; use replace_all=true ou um trecho mais específico"),
        );
    }
    let replaced = if req.replace_all { count } else { 1 };
    let updated = if req.replace_all {
        text.replace(&req.old_string, &req.new_string)
    } else {
        text.replacen(&req.old_string, &req.new_string, 1)
    };
    match tokio::fs::write(&path, updated.as_bytes()).await {
        Ok(()) => Json(json!({
            "ok": true,
            "path": path.to_string_lossy(),
            "replacements": replaced,
            "bytes": updated.len(),
        }))
        .into_response(),
        Err(e) => err(StatusCode::UNPROCESSABLE_ENTITY, format!("falha ao gravar arquivo: {e}")),
    }
}

pub async fn shell(
    headers: HeaderMap,
    Json(req): Json<ShellRequest>,
) -> Response {
    if let Err(resp) = require_token(&headers) {
        return resp;
    }
    if req.command.trim().is_empty() {
        return err(StatusCode::BAD_REQUEST, "command vazio");
    }
    let cwd = match resolve_cwd(req.cwd.as_deref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let timeout_s = req.timeout_s.unwrap_or(60).clamp(1, 600);

    #[cfg(target_os = "windows")]
    let mut cmd = {
        let mut c = Command::new("cmd");
        c.args(["/C", &req.command]);
        c
    };
    #[cfg(not(target_os = "windows"))]
    let mut cmd = {
        let mut c = Command::new("bash");
        c.args(["-lc", &req.command]);
        c
    };
    cmd.current_dir(&cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let started = Instant::now();
    match tokio::time::timeout(Duration::from_secs(timeout_s), cmd.output()).await {
        Ok(Ok(out)) => {
            let stdout = truncate_chars(&String::from_utf8_lossy(&out.stdout), MAX_OUTPUT_CHARS);
            let stderr = truncate_chars(&String::from_utf8_lossy(&out.stderr), MAX_OUTPUT_CHARS);
            Json(json!({
                "ok": out.status.success(),
                "exit_code": out.status.code(),
                "timed_out": false,
                "duration_ms": started.elapsed().as_millis() as u64,
                "cwd": cwd.to_string_lossy(),
                "stdout": stdout,
                "stderr": stderr,
            }))
            .into_response()
        }
        Ok(Err(e)) => err(StatusCode::BAD_GATEWAY, format!("falha ao executar shell: {e}")),
        Err(_) => Json(json!({
            "ok": false,
            "exit_code": null,
            "timed_out": true,
            "duration_ms": started.elapsed().as_millis() as u64,
            "cwd": cwd.to_string_lossy(),
            "stdout": "",
            "stderr": format!("timeout após {timeout_s}s"),
        }))
        .into_response(),
    }
}

// ════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════

async fn run_search(
    bin: &str,
    cwd: &Path,
    target: &Path,
    pattern: &str,
    fixed: bool,
    max: usize,
) -> Result<(String, String, i32), std::io::Error> {
    let mut cmd = Command::new(bin);
    cmd.current_dir(cwd);
    if bin == "rg" {
        cmd.args([
            "--no-heading",
            "--line-number",
            "--color",
            "never",
            "--max-count",
            "500",
        ]);
        if fixed {
            cmd.arg("--fixed-strings");
        }
        cmd.arg("--").arg(pattern).arg(target);
    } else {
        cmd.args(["-R", "-n", "-I"]);
        if fixed {
            cmd.arg("-F");
        }
        cmd.arg("--").arg(pattern).arg(target);
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let out = cmd.output().await?;
    let stdout = String::from_utf8_lossy(&out.stdout)
        .lines()
        .take(max)
        .collect::<Vec<_>>()
        .join("\n");
    Ok((
        stdout,
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.code().unwrap_or(-1),
    ))
}

fn err(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(json!({ "error": message.into() }))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_pad_and_alphabet() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn detects_supported_image_mime() {
        assert_eq!(detect_image_mime(b"\x89PNG\r\n\x1a\nrest"), Some("image/png"));
        assert_eq!(detect_image_mime(b"\xff\xd8\xff\xe0rest"), Some("image/jpeg"));
        assert_eq!(detect_image_mime(b"GIF89arest"), Some("image/gif"));
        assert_eq!(detect_image_mime(b"BMrest"), Some("image/bmp"));
        let webp = b"RIFF\x00\x00\x00\x00WEBPVP8 ";
        assert_eq!(detect_image_mime(webp), Some("image/webp"));
        assert_eq!(detect_image_mime(b"not an image"), None);
    }
}
