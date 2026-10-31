#!/usr/bin/env python3
# Alpha AI (AVA) — empacota a extensão Playwright patcheada para o Linux
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gera o CRX assinado + metadados para instalação automática no Chrome Linux.

O Chrome oficial não aceita mais `--load-extension` (removido no Chrome 137)
nem uma extensão unpacked sem interação. No Linux, porém, o mecanismo de
"external extension" instala sozinho um CRX local declarado em
`/opt/google/chrome/extensions/<id>.json`.

Como não temos a chave privada da Microsoft, o CRX é assinado com uma chave
nossa — logo, tem um ID novo. O instalador grava esse ID no `.env` e o
wrapper `playwright_extension_id_wrapper.cjs` faz o @playwright/mcp usá-lo
sem fork do pacote.

Saídas (em ~/.local/share/alpha-ai):
  playwright-extension-crx-src/       fonte patcheada (sem manifest.key)
  playwright-extension-patched.crx    CRX assinado
  playwright-extension-key.pem        chave privada (reutilizada)
  playwright-extension-id             ID novo (para o wrapper/status)
  playwright-extension-token          token fixo (para Modules/.env)
  playwright-extension-external.json  JSON para /opt/google/chrome/extensions
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import struct
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import patch_playwright_extension as ppe  # noqa: E402

SHARE = Path.home() / ".local" / "share" / "alpha-ai"
SRC = SHARE / "playwright-extension-crx-src"
CRX = SHARE / "playwright-extension-patched.crx"
KEY = SHARE / "playwright-extension-key.pem"
ID_FILE = SHARE / "playwright-extension-id"
TOKEN_FILE = SHARE / "playwright-extension-token"
EXT_JSON = SHARE / "playwright-extension-external.json"
VERSION_FILE = SHARE / "playwright-extension-build"


def find_chrome() -> str | None:
    env = os.environ.get("AVA_CHROME_PATH", "").strip()
    if env and Path(env).is_file():
        return env
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    return None


def stable_token(explicit: str | None) -> str:
    if explicit:
        token = explicit.strip()
    elif TOKEN_FILE.is_file():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    else:
        token = secrets.token_urlsafe(32)
    if not token:
        raise SystemExit("token vazio")
    SHARE.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(token + "\n", encoding="utf-8")
    TOKEN_FILE.chmod(0o600)
    return token


def next_version(base: str) -> str:
    try:
        counter = int(VERSION_FILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        counter = 0
    counter = (counter % 65000) + 1
    VERSION_FILE.write_text(str(counter), encoding="utf-8")
    parts = [p for p in base.split(".") if p.isdigit()][:3]
    while len(parts) < 3:
        parts.append("0")
    return ".".join(parts + [str(counter)])


def download_store_extension(chrome: str) -> Path:
    """Baixa/extrai o CRX da Chrome Web Store para máquinas sem a extensão.

    O patch precisa do código-fonte da extensão. Se ela ainda não estiver no
    perfil (instalação limpa), baixamos o CRX oficial e extraímos o ZIP do
    CRX3 — assim o install.sh não depende de abrir a loja antes.
    """
    try:
        out = subprocess.check_output([chrome, "--version"], text=True, stderr=subprocess.STDOUT)
        match = re.search(r"(\d+\.\d+\.\d+\.\d+)", out)
        prod = match.group(1) if match else "120.0.0.0"
    except Exception:
        prod = "120.0.0.0"
    query = urllib.parse.urlencode(
        {
            "response": "redirect",
            "prodversion": prod,
            "acceptformat": "crx2,crx3",
            "x": f"id={ppe.EXTENSION_ID}&uc",
        }
    )
    url = "https://clients2.google.com/service/update2/crx?" + query
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=45).read()
    if data[:4] != b"Cr24":
        raise SystemExit("Chrome Web Store não devolveu um CRX válido")
    version, header_len = struct.unpack("<II", data[4:12])
    if version != 3:
        raise SystemExit(f"CRX versão {version} não suportada (esperado 3)")
    zip_bytes = data[12 + header_len :]
    dest = SHARE / "playwright-extension-store-src"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(dest)
    return dest


def extension_id_from_key(key: Path) -> str:
    try:
        der = subprocess.check_output(
            ["openssl", "rsa", "-in", str(key), "-pubout", "-outform", "DER"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"openssl indisponível para calcular o ID da extensão: {exc}")
    digest = hashlib.sha256(der).hexdigest()[:32]
    return "".join(chr(ord("a") + int(c, 16)) for c in digest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", help="token fixo (default: reutiliza/gera)")
    parser.add_argument("--source", type=Path, help="diretório da extensão original")
    args = parser.parse_args()

    SHARE.mkdir(parents=True, exist_ok=True)
    token = stable_token(args.token)

    chrome = find_chrome()
    if chrome is None:
        raise SystemExit("Google Chrome/Chromium não encontrado para empacotar o CRX")

    installed = args.source or ppe.find_installed_extension()
    if installed is None:
        print("Extensão não encontrada no perfil; baixando o CRX oficial da Web Store...")
        installed = download_store_extension(chrome)

    if SRC.exists():
        shutil.rmtree(SRC)
    shutil.copytree(installed, SRC)
    shutil.rmtree(SRC / "_metadata", ignore_errors=True)

    # Patch de comportamento (aba ativa, sem tab group, keepalive).
    background = SRC / "lib" / "background.mjs"
    background.write_text(
        ppe.patch_background(background.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    auth = SRC / "lib" / "ui" / "authToken.js"
    auth.write_text(
        ppe.patch_auth_token(auth.read_text(encoding="utf-8"), token),
        encoding="utf-8",
    )
    ppe.apply_branding(SRC)

    manifest_path = SRC / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # O CRX terá a NOSSA chave. A key da Microsoft aqui definiria outro ID
    # para a versão unpacked e não pode assinar este CRX.
    manifest.pop("key", None)
    # Sem update_url da Web Store: senão o Chrome valida a cópia contra a
    # assinatura da loja e marca DISABLE_CORRUPTED.
    manifest.pop("update_url", None)
    version = next_version(str(manifest.get("version", "0.4.0")))
    manifest["version"] = version
    manifest_path.write_text(
        json.dumps(manifest, indent=3, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    CRX.unlink(missing_ok=True)
    cmd = [
        chrome,
        "--headless=new",
        "--no-sandbox",
        "--no-message-box",
        f"--pack-extension={SRC}",
    ]
    if KEY.is_file():
        cmd.append(f"--pack-extension-key={KEY}")
    subprocess.run(cmd, check=True, cwd=str(SHARE), stdout=subprocess.DEVNULL)

    produced_crx = SRC.with_suffix(".crx")
    produced_pem = SRC.with_suffix(".pem")
    if not produced_crx.is_file():
        raise SystemExit(f"Chrome não gerou o CRX esperado em {produced_crx}")
    shutil.move(str(produced_crx), CRX)
    CRX.chmod(0o644)
    if produced_pem.is_file():
        shutil.move(str(produced_pem), KEY)
        KEY.chmod(0o600)

    ext_id = extension_id_from_key(KEY)
    ID_FILE.write_text(ext_id + "\n", encoding="utf-8")
    EXT_JSON.write_text(
        json.dumps(
            {"external_crx": str(CRX), "external_version": version},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    EXT_JSON.chmod(0o644)

    print(f"ID={ext_id}")
    print(f"TOKEN={token}")
    print(f"VERSION={version}")
    print(f"CRX={CRX}")
    print(f"EXTERNAL_JSON={EXT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
