//! Descoberta de modelos GGUF disponíveis na pasta ./Models.
//!
//! Equivalente em Rust ao `find_mmproj()` do script Python original:
//! procura arquivos *.gguf, e para cada um tenta achar um "irmão" mmproj
//! (arquivo contendo "mmproj" no nome) na mesma pasta. Se houver mais de
//! um candidato, prioriza o que tiver "f16" no nome.

use crate::state::ModelInfo;
use std::path::Path;

/// Lista todos os modelos .gguf na pasta informada, detectando quais são
/// multimodais (têm um mmproj associado) e quais são texto puro.
///
/// Modelos cujo próprio nome contém "mmproj" são ignorados na listagem
/// principal (eles são "auxiliares", não modelos para carregar sozinhos).
pub fn scan_models(models_dir: &Path) -> anyhow::Result<Vec<ModelInfo>> {
    if !models_dir.exists() {
        anyhow::bail!("Pasta de modelos não encontrada: {}", models_dir.display());
    }

    // 1. Coleta todos os .gguf da pasta.
    let mut all_gguf: Vec<std::path::PathBuf> = Vec::new();
    for entry in std::fs::read_dir(models_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            all_gguf.push(path);
        }
    }

    // 2. Separa os mmproj e os drafts MTP dos modelos "principais".
    let mmproj_files: Vec<&std::path::PathBuf> = all_gguf
        .iter()
        .filter(|p| file_name_lower(p).contains("mmproj"))
        .collect();

    let draft_files: Vec<&std::path::PathBuf> = all_gguf
        .iter()
        .filter(|p| {
            let name = file_name_lower(p);
            !name.contains("mmproj") && (name.contains("mtp") || name.contains("draft"))
        })
        .collect();

    let model_files: Vec<&std::path::PathBuf> = all_gguf
        .iter()
        .filter(|p| {
            let name = file_name_lower(p);
            !name.contains("mmproj") && !name.contains("mtp") && !name.contains("draft")
        })
        .collect();

    // 3. Para cada modelo principal, tenta achar um mmproj companheiro.
    //    Critério simples (igual ao script original): qualquer mmproj na
    //    mesma pasta serve; prioriza o que tiver "f16" no nome.
    let mut result = Vec::new();
    let model_count = model_files.len();
    for model_path in model_files {
        let mmproj_path = pick_mmproj(model_path, &mmproj_files, model_count);
        let mtp_draft_path = pick_mtp_draft(&draft_files, model_path);

        let size_mb = std::fs::metadata(model_path)
            .map(|m| m.len() / (1024 * 1024))
            .unwrap_or(0);

        result.push(ModelInfo {
            name: model_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("desconhecido")
                .to_string(),
            path: model_path.display().to_string(),
            mtp_draft_path: mtp_draft_path.map(|p| p.display().to_string()),
            is_multimodal: mmproj_path.is_some(),
            mmproj_path: mmproj_path.map(|p| p.display().to_string()),
            size_mb,
        });
    }

    // Ordena por nome para a UI ficar estável/previsível.
    result.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(result)
}

fn file_name_lower(path: &Path) -> String {
    path.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase()
}

/// Tokens de quantização/formato reconhecidos ao final do stem de um GGUF
/// (ex.: "-Q4_K_M", "-F16", "-Q8_0", "-IQ4_XS"), usados para derivar o
/// "nome base" do modelo (sem a parte de quantização) e assim comparar
/// modelo <-> mmproj mesmo quando cada um usa uma quantização diferente
/// (ex.: modelo em Q4_K_M, mmproj em F16).
const QUANT_TOKENS: &[&str] = &[
    "f16", "f32", "fp16", "fp32", "bf16",
    "q2", "q3", "q4", "q5", "q6", "q8",
    "iq1", "iq2", "iq3", "iq4", "iq5", "iq6",
    "k", "m", "s", "l", "xs", "xxs", "xxxs",
    "0", "1",
];

/// Remove, do final do stem, tokens separados por '-'/'_' enquanto eles
/// forem reconhecidos como parte de um sufixo de quantização/formato.
/// "LFM2.5-VL-3B-Q4_K_M" -> "LFM2.5-VL-3B"
/// "mmproj-LFM2.5-VL-3B-F16" -> "mmproj-LFM2.5-VL-3B"
fn strip_trailing_quant(stem: &str) -> &str {
    let mut end = stem.len();
    loop {
        let prefix = &stem[..end];
        let last_sep = prefix.rfind(|c| c == '-' || c == '_');
        let Some(idx) = last_sep else { break };
        let token = &prefix[idx + 1..];
        if token.is_empty() {
            end = idx;
            continue;
        }
        if QUANT_TOKENS.contains(&token.to_lowercase().as_str()) {
            end = idx;
        } else {
            break;
        }
    }
    &stem[..end]
}

/// Escolhe o mmproj companheiro de `model_path` dentre `candidates`.
///
/// Casa pelo nome: deriva o "nome base" do modelo (sem sufixo de
/// quantização) e só considera candidatos cujo nome contenha esse nome
/// base — evita que um modelo de texto puro (ex. "LFM2.5-8B-A1B") acabe
/// herdando o mmproj de um modelo multimodal diferente que só por acaso
/// está na mesma pasta (ex. "mmproj-LFM2.5-VL-3B-F16"). Entre os
/// candidatos que combinam, prioriza o que tiver "f16" no nome.
///
/// Se nenhum candidato combinar pelo nome, só cai no fallback de "pega o
/// único mmproj disponível" quando isso é inequívoco: exatamente um
/// modelo e um mmproj na pasta.
fn pick_mmproj<'a>(
    model_path: &Path,
    candidates: &[&'a std::path::PathBuf],
    model_count: usize,
) -> Option<&'a std::path::PathBuf> {
    if candidates.is_empty() {
        return None;
    }

    let model_stem = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    let base_name = strip_trailing_quant(&model_stem).to_string();

    if !base_name.is_empty() {
        let matches: Vec<&'a std::path::PathBuf> = candidates
            .iter()
            .copied()
            .filter(|p| file_name_lower(p).contains(&base_name))
            .collect();

        if !matches.is_empty() {
            return matches
                .iter()
                .copied()
                .find(|p| file_name_lower(p).contains("f16"))
                .or_else(|| matches.first().copied());
        }
    }

    // Nenhum candidato combina pelo nome. Só assume pareamento "cego" se
    // for a única combinação possível na pasta (1 modelo + 1 mmproj) —
    // caso contrário, é mais seguro não marcar o modelo como multimodal.
    if candidates.len() == 1 && model_count == 1 {
        return candidates.first().copied();
    }

    None
}

/// Escolhe o draft MTP companheiro de `model_path` dentre `candidates`.
/// Prioriza o candidato cujo nome compartilha o "stem" do modelo principal
/// (mesmo critério do find_mtp_draft_model em process_manager.rs); na
/// ausência de um candidato assim, cai no primeiro disponível.
fn pick_mtp_draft<'a>(
    candidates: &[&'a std::path::PathBuf],
    model_path: &Path,
) -> Option<&'a std::path::PathBuf> {
    if candidates.is_empty() {
        return None;
    }

    let base_stem = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    if !base_stem.is_empty() {
        if let Some(matching) = candidates
            .iter()
            .find(|p| file_name_lower(p).contains(&base_stem))
        {
            return Some(matching);
        }
    }

    candidates.first().copied()
}