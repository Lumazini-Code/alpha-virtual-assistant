#!/usr/bin/env python3
"""
verify_models.py — valida e repara os modelos ONNX/empacotados do Alpha AI.

Motivação: instalação com arquivo truncado/corrompido (download ou extração
de zip interrompida — caso real em Windows) faz o onnxManager estourar
INVALID_PROTOBUF e o serviço fica fora. Este script confere os arquivos
ANTES de subir a stack e, com --download, re-baixa o que estiver faltando
ou abaixo do tamanho esperado (fontes oficiais no Hugging Face).

Uso:
  python3 scripts/verify_models.py              # só valida (exit 0 ok / 78 se corrompido)
  python3 scripts/verify_models.py --download   # valida e re-baixa o corrompido/ausente
  python3 scripts/verify_models.py --quiet      # saída mínima (para o instalador)

Rodar na raiz do projeto (ou dentro do container: `docker compose exec ava
python3 scripts/verify_models.py --download` — o Models/ é bind mount).
Multiplataforma: stdlib only (sem dependências extras).
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# Raiz do projeto = pai do diretório deste script (scripts/).
ROOT = Path(__file__).resolve().parent.parent

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

# Cada entrada: caminho relativo à raiz, tamanho mínimo em bytes (arquivos
# reais têm folga acima disso) e URL oficial para re-download.
MODELS = [
    {
        "name": "Embedding model (multilingual-e5-small.onnx)",
        "path": "Modules/Models/multilingual-e5-small/multilingual-e5-small.onnx",
        "min_bytes": 200 * 1024 * 1024,     # real: ~235 MB
        "url": "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx",
    },
    {
        "name": "Embedding tokenizer (e5 tokenizer.json)",
        "path": "Modules/Models/multilingual-e5-small/tokenizer.json",
        "min_bytes": 10 * 1024 * 1024,      # real: ~17 MB
        "url": "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/tokenizer.json",
    },
    {
        "name": "Reranker model (ms-marco-MiniLM-L-6-v2.onnx)",
        "path": "Modules/Models/ms-marco-MiniLM-L-6-v2/ms-marco-MiniLM-L-6-v2.onnx",
        "min_bytes": 20 * 1024 * 1024,      # real: ~23 MB
        "url": "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model.onnx",
    },
    {
        "name": "Reranker tokenizer (ms-marco tokenizer.json)",
        "path": "Modules/Models/ms-marco-MiniLM-L-6-v2/tokenizer.json",
        "min_bytes": 500 * 1024,            # real: ~711 KB
        "url": "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json",
    },
    # Vision/ASR — checagem de tamanho (re-download manual; grandes demais
    # para baixar automaticamente aqui).
    {
        "name": "Depth (depth_anything_v2_small_FP32.onnx)",
        "path": "Modules/Models/depth_anything_v2_small_FP32.onnx",
        "min_bytes": 90 * 1024 * 1024,      # real: ~99 MB
        "url": None,
    },
]

EXIT_OK, EXIT_CONFIG = 0, 78   # 78 = EX_CONFIG: mesmo código do start.sh


def _log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg, flush=True)


def _download(url: str, dest: Path, quiet: bool) -> bool:
    """Baixa `url` → `dest` (atômico: escreve em .part e renomeia)."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "alpha-ai-verify-models/1.0",
        **({"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}),
    })
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as f:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                block = resp.read(1024 * 1024)
                if not block:
                    break
                f.write(block)
                done += len(block)
                if not quiet and total:
                    pct = 100 * done // total
                    print(f"\r  baixando {dest.name}: {pct}% ({done // (1024 * 1024)} MB)", end="", flush=True)
        if not quiet:
            print(flush=True)
        tmp.replace(dest)
        return True
    except Exception as e:
        _log(f"  ✗ download falhou: {e}", quiet)
        tmp.unlink(missing_ok=True)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Valida (e repara) os modelos ONNX do Alpha AI")
    ap.add_argument("--download", action="store_true", help="re-baixa arquivos ausentes/corrompidos")
    ap.add_argument("--quiet", action="store_true", help="saída mínima (instaladores)")
    args = ap.parse_args()
    quiet = args.quiet

    bad: list[dict] = []
    for m in MODELS:
        p = ROOT / m["path"]
        status = "ok"
        if not p.exists():
            status = "ausente"
        elif p.stat().st_size < m["min_bytes"]:
            status = f"corrompido ({p.stat().st_size} bytes < {m['min_bytes']})"
        _log(f"{'✓' if status == 'ok' else '✗'} {m['name']}: {status}", quiet)
        if status != "ok":
            bad.append(m)

    if not bad:
        _log("✓ todos os modelos íntegros", quiet)
        return EXIT_OK

    if not args.download:
        for m in bad:
            print(f"✗ {m['name']}: {m['path']}")
        print(
            "\nModelos inválidos/ausentes. Rode com --download para re-baixar os corrigíveis\n"
            "(ou reinstale o Alpha AI). Sem reparo, o onnxManager NÃO vai subir (exit 78).",
            file=sys.stderr,
        )
        return EXIT_CONFIG

    for m in bad:
        if not m["url"]:
            _log(f"⚠ {m['name']}: sem URL automática — baixe manualmente para {m['path']}", quiet)
            continue
        dest = ROOT / m["path"]
        _log(f"↓ re-baixando {m['name']} …", quiet)
        if _download(m["url"], dest, quiet):
            ok = dest.exists() and dest.stat().st_size >= m["min_bytes"]
            _log(f"  {'✓' if ok else '✗'} {dest.name}: {dest.stat().st_size} bytes", quiet)
            if not ok:
                return EXIT_CONFIG
        else:
            return EXIT_CONFIG

    _log("✓ modelos reparados", quiet)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
