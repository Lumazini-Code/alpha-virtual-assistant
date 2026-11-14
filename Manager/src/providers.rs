// Alpha AI (AVA) — Cadastro e leitura de provedores de IA
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
//! Catálogo de provedores e configuração persistida em `Modules/.env`.
//!
//! O catálogo (`assets/providers.json`) é gerado a partir do snapshot
//! models.dev do opencode por `scripts/build_provider_catalog.py`; só entram
//! provedores com endpoint OpenAI-compatible — o protocolo que `Modules/LLM.py`
//! fala. `opencode`/`opencode-go` (OpenCode Zen) ficam de fora.
//!
//! A WebUI usa estes dados na primeira execução e em Configurações. A escrita
//! é atômica e preserva o restante do arquivo, com permissão 0600 (o `.env`
//! contém as chaves e os tokens internos).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::state::SharedState;

/// Catálogo embutido no binário: a instalação não carrega `test/opencode`.
pub const CATALOG: &str = include_str!("../assets/providers.json");

/// Chaves que o cadastro gerencia. `OPENROUTER_API_KEY` é removida porque o
/// fallback antigo do instalador foi descontinuado.
const MANAGED_KEYS: &[&str] = &[
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_PROVIDER_LOCAL",
    "OPENROUTER_API_KEY",
    "AVA_LLM_PROVIDER",
];

#[derive(Debug, Clone, Serialize)]
pub struct Config {
    /// `true` quando há base URL e credencial utilizável (ou provider local).
    pub configured: bool,
    pub provider_id: Option<String>,
    pub base_url: String,
    pub model: String,
    pub key_set: bool,
    /// Dica mascarada (nunca a chave inteira).
    pub key_hint: Option<String>,
    pub local: bool,
}

#[derive(Debug, Deserialize)]
pub struct ConfigInput {
    #[serde(default)]
    pub provider_id: Option<String>,
    pub base_url: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub local: Option<bool>,
}

fn env_path(project_root: &Path) -> PathBuf {
    project_root.join("Modules").join(".env")
}

fn read_lines(path: &PathBuf) -> Vec<String> {
    std::fs::read_to_string(path)
        .map(|text| text.lines().map(str::to_string).collect())
        .unwrap_or_default()
}

fn key_of(line: &str) -> Option<&str> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let line = line.strip_prefix("export ").map(str::trim).unwrap_or(line);
    line.split_once('=').map(|(key, _)| key.trim())
}

fn read_value(path: &PathBuf, wanted: &str) -> String {
    read_lines(path)
        .iter()
        .filter_map(|line| {
            let line = line.trim();
            let line = line.strip_prefix("export ").map(str::trim).unwrap_or(line);
            let (key, value) = line.split_once('=')?;
            (key.trim() == wanted)
                .then(|| value.trim().trim_matches(|c| c == '"' || c == '\'').to_string())
        })
        .next_back()
        .unwrap_or_default()
}

/// URL local (localhost/loopback) não exige API key — mesmo critério do
/// instalador e de `Modules/LLM.py`.
pub fn url_is_local(base: &str) -> bool {
    let lower = base.trim().to_ascii_lowercase();
    ["http://localhost", "https://localhost", "http://127.", "https://127.",
     "http://[::1]", "https://[::1]"]
        .iter()
        .any(|prefix| lower.starts_with(prefix))
        || lower
            .split('/')
            .nth(2)
            .map(|host| host == "localhost" || host.ends_with(".local"))
            .unwrap_or(false)
}

pub fn read_config(state: &SharedState) -> Config {
    read_config_at(&state.project_root)
}

fn read_config_at(project_root: &Path) -> Config {
    let path = env_path(project_root);
    let base_url = read_value(&path, "LLM_BASE_URL");
    let model = read_value(&path, "LLM_MODEL");
    let key = {
        let primary = read_value(&path, "LLM_API_KEY");
        if primary.is_empty() {
            read_value(&path, "OPENROUTER_API_KEY")
        } else {
            primary
        }
    };
    let local = read_value(&path, "LLM_PROVIDER_LOCAL") == "1" || url_is_local(&base_url);
    let provider_id = {
        let saved = read_value(&path, "AVA_LLM_PROVIDER");
        (!saved.is_empty()).then_some(saved)
    };
    let key_hint = (!key.is_empty()).then(|| {
        let tail: String = key
            .chars()
            .rev()
            .take(4)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        format!("••••{tail}")
    });

    Config {
        configured: !base_url.is_empty() && (local || !key.is_empty()),
        provider_id,
        base_url,
        model,
        key_set: !key.is_empty(),
        key_hint,
        local,
    }
}

pub fn write_config(state: &SharedState, input: ConfigInput) -> Result<Config, String> {
    write_config_at(&state.project_root, input)
}

fn write_config_at(project_root: &Path, input: ConfigInput) -> Result<Config, String> {
    let path = env_path(project_root);
    let base_url = input.base_url.trim().trim_end_matches('/').to_string();
    if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
        return Err("O endereço base deve começar com http:// ou https://.".into());
    }
    if base_url.len() <= "https://".len() {
        return Err("Informe o endereço base completo da API.".into());
    }

    let api_key = input.api_key.unwrap_or_default().trim().to_string();
    let model = input
        .model
        .unwrap_or_default()
        .trim()
        .to_string();
    let local = input.local.unwrap_or(false) || url_is_local(&base_url);
    let existing_key = {
        let primary = read_value(&path, "LLM_API_KEY");
        if primary.is_empty() {
            read_value(&path, "OPENROUTER_API_KEY")
        } else {
            primary
        }
    };
    if !local && api_key.is_empty() && existing_key.is_empty() {
        return Err("Informe a API key do provedor (ou marque como provider local).".into());
    }
    // Em branco = mantém a chave já gravada (permite trocar só modelo/base).
    let key_to_write = if api_key.is_empty() {
        existing_key
    } else {
        api_key.clone()
    };

    let mut lines: Vec<String> = read_lines(&path)
        .into_iter()
        .filter(|line| {
            key_of(line)
                .map(|key| !MANAGED_KEYS.contains(&key))
                .unwrap_or(true)
        })
        .collect();

    lines.push(format!("LLM_BASE_URL={base_url}"));
    if !model.is_empty() {
        lines.push(format!("LLM_MODEL={model}"));
    }
    if !key_to_write.is_empty() {
        lines.push(format!("LLM_API_KEY={key_to_write}"));
    }
    if let Some(provider_id) = input.provider_id.as_deref().map(str::trim).filter(|p| !p.is_empty()) {
        lines.push(format!("AVA_LLM_PROVIDER={provider_id}"));
    }
    if local {
        lines.push("LLM_PROVIDER_LOCAL=1".into());
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("criar {}: {e}", parent.display()))?;
    }
    let mut content = lines.join("\n");
    content.push('\n');
    std::fs::write(&path, content).map_err(|e| format!("gravar {}: {e}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }

    // Mantém o processo atual coerente para consumidores que leem env.
    let updates = [
        ("LLM_BASE_URL", base_url.as_str()),
        ("LLM_MODEL", model.as_str()),
        ("LLM_API_KEY", key_to_write.as_str()),
        ("AVA_LLM_PROVIDER", input.provider_id.as_deref().unwrap_or("")),
    ];
    for (key, value) in updates {
        if !value.is_empty() {
            // SAFETY: chamado durante o handler, depois do boot single-threaded.
            unsafe { std::env::set_var(key, value) };
        }
    }
    if local {
        unsafe { std::env::set_var("LLM_PROVIDER_LOCAL", "1") };
    } else {
        unsafe { std::env::remove_var("LLM_PROVIDER_LOCAL") };
    }
    unsafe { std::env::remove_var("OPENROUTER_API_KEY") };

    Ok(read_config_at(project_root))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "ava-providers-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("Modules")).unwrap();
        root
    }

    #[test]
    fn saves_and_reads_provider_config() {
        let root = temp_root("save");
        let config = write_config_at(
            &root,
            ConfigInput {
                provider_id: Some("zai".into()),
                base_url: "https://api.z.ai/api/paas/v4/".into(),
                api_key: Some("sk-test-1234".into()),
                model: Some("glm-5.3-flash".into()),
                local: None,
            },
        )
        .unwrap();

        assert!(config.configured);
        assert_eq!(config.provider_id.as_deref(), Some("zai"));
        assert_eq!(config.base_url, "https://api.z.ai/api/paas/v4");
        assert_eq!(config.model, "glm-5.3-flash");
        assert_eq!(config.key_hint.as_deref(), Some("••••1234"));

        let content = std::fs::read_to_string(env_path(&root)).unwrap();
        assert!(content.contains("LLM_BASE_URL=https://api.z.ai/api/paas/v4"));
        assert!(!content.contains("OPENROUTER_API_KEY"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn local_provider_needs_no_key_and_preserves_other_lines() {
        let root = temp_root("local");
        let env = env_path(&root);
        std::fs::write(&env, "AVA_INTERNAL_TOKEN=abc
OPENROUTER_API_KEY=old
").unwrap();
        let config = write_config_at(
            &root,
            ConfigInput {
                provider_id: Some("custom".into()),
                base_url: "http://127.0.0.1:11434/v1".into(),
                api_key: None,
                model: Some("llama3".into()),
                local: None,
            },
        )
        .unwrap();
        assert!(config.configured);
        assert!(config.local);
        let content = std::fs::read_to_string(&env).unwrap();
        assert!(content.contains("AVA_INTERNAL_TOKEN=abc"));
        assert!(content.contains("LLM_PROVIDER_LOCAL=1"));
        assert!(!content.contains("OPENROUTER_API_KEY"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn blank_key_keeps_the_existing_credential() {
        let root = temp_root("keep");
        write_config_at(
            &root,
            ConfigInput {
                provider_id: Some("zai".into()),
                base_url: "https://api.z.ai/api/paas/v4".into(),
                api_key: Some("sk-keep-9999".into()),
                model: Some("glm-5.3-flash".into()),
                local: None,
            },
        )
        .unwrap();

        let config = write_config_at(
            &root,
            ConfigInput {
                provider_id: Some("zai".into()),
                base_url: "https://api.z.ai/api/paas/v4".into(),
                api_key: None,
                model: Some("glm-4.6".into()),
                local: None,
            },
        )
        .unwrap();

        assert!(config.configured);
        assert!(config.key_set);
        assert_eq!(config.model, "glm-4.6");
        let content = std::fs::read_to_string(env_path(&root)).unwrap();
        assert!(content.contains("LLM_API_KEY=sk-keep-9999"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn remote_provider_requires_a_key() {
        let root = temp_root("nokey");
        let result = write_config_at(
            &root,
            ConfigInput {
                provider_id: Some("openai".into()),
                base_url: "https://api.openai.com/v1".into(),
                api_key: None,
                model: None,
                local: None,
            },
        );
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(root);
    }
}