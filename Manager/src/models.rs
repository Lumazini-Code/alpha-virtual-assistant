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

    // 2. Separa os mmproj dos modelos "principais".
    let mmproj_files: Vec<&std::path::PathBuf> = all_gguf
        .iter()
        .filter(|p| file_name_lower(p).contains("mmproj"))
        .collect();

    let model_files: Vec<&std::path::PathBuf> = all_gguf
        .iter()
        .filter(|p| !file_name_lower(p).contains("mmproj"))
        .collect();

    // 3. Para cada modelo principal, tenta achar um mmproj companheiro.
    //    Critério simples (igual ao script original): qualquer mmproj na
    //    mesma pasta serve; prioriza o que tiver "f16" no nome.
    let mut result = Vec::new();
    for model_path in model_files {
        let mmproj_path = pick_mmproj(&mmproj_files);

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

fn pick_mmproj<'a>(candidates: &[&'a std::path::PathBuf]) -> Option<&'a std::path::PathBuf> {
    if candidates.is_empty() {
        return None;
    }
    candidates
        .iter()
        .find(|p| file_name_lower(p).contains("f16"))
        .copied()
        .or_else(|| candidates.first().copied())
}