#!/usr/bin/env python3
# Alpha AI (AVA) — remove o ID da extensão Playwright da blocklist do Chrome
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reinstala uma external extension removendo `extensions.external_uninstalls`.

Quando o Chrome remove/considera inválida uma extensão instalada via
external extension, ele grava o ID em `extensions.external_uninstalls` no
Preferences. A partir daí, mesmo que o JSON em /opt/google/chrome/extensions
aponte para um CRX novo, o Chrome não reinstala.

Este script:
  - exige que o Chrome esteja FECHADO (senão o Chrome sobrescreve o arquivo);
  - faz backup do Preferences;
  - remove o ID de `extensions.external_uninstalls`;
  - remove a entrada residual em `extensions.settings`.

Uso:
  python3 scripts/clear_playwright_extension_blocklist.py
  python3 scripts/clear_playwright_extension_blocklist.py --id <id>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def chrome_running() -> bool:
    try:
        if os.name == "nt":
            out = subprocess.check_output(
                ["tasklist", "/FI", "IMAGENAME eq chrome.exe", "/NH"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return "chrome.exe" in out.lower()
        out = subprocess.check_output(
            ["pgrep", "-x", "chrome"], text=True, stderr=subprocess.DEVNULL
        )
        if out.strip():
            return True
        out = subprocess.check_output(
            ["pgrep", "-x", "chromium"], text=True, stderr=subprocess.DEVNULL
        )
        return bool(out.strip())
    except Exception:
        return False


def preferences_paths() -> list[Path]:
    home = Path.home()
    if os.name == "nt":
        roots = [
            Path(os.environ.get("LOCALAPPDATA", str(home))) / "Google" / "Chrome" / "User Data",
            Path(os.environ.get("LOCALAPPDATA", str(home))) / "Chromium" / "User Data",
            Path(os.environ.get("LOCALAPPDATA", str(home))) / "Microsoft" / "Edge" / "User Data",
            Path(os.environ.get("LOCALAPPDATA", str(home))) / "BraveSoftware" / "Brave-Browser" / "User Data",
        ]
    else:
        roots = [
            home / ".config" / "google-chrome",
            home / ".config" / "chromium",
            home / ".config" / "microsoft-edge",
            home / ".config" / "BraveSoftware" / "Brave-Browser",
        ]
    paths: list[Path] = []
    for root in roots:
        if root.is_dir():
            paths.extend(root.glob("*/Preferences"))
    return paths


def default_extension_id() -> str:
    id_file = Path.home() / ".local" / "share" / "alpha-ai" / "playwright-extension-id"
    try:
        return id_file.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", default=default_extension_id(), help="ID da extensão")
    parser.add_argument(
        "--force",
        action="store_true",
        help="edita mesmo com o Chrome aparentemente aberto (NÃO recomendado)",
    )
    args = parser.parse_args()

    ext_id = args.id.strip()
    if not ext_id:
        print("ID vazio: informe --id ou gere o playwright-extension-id primeiro.", file=sys.stderr)
        return 1

    if chrome_running() and not args.force:
        print(
            "O Chrome está aberto. Feche TODAS as janelas/processos do Chrome e rode de novo.\n"
            "Com ele aberto, o Chrome sobrescreve o Preferences ao sair e a blocklist volta.",
            file=sys.stderr,
        )
        return 2

    changed = 0
    for prefs in preferences_paths():
        try:
            data = json.loads(prefs.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"ignorando {prefs}: {exc}", file=sys.stderr)
            continue

        extensions = data.setdefault("extensions", {})
        uninstalls = extensions.get("external_uninstalls")
        removed = False
        if isinstance(uninstalls, list) and ext_id in uninstalls:
            uninstalls.remove(ext_id)
            removed = True
        settings = extensions.get("settings", {})
        if isinstance(settings, dict) and ext_id in settings:
            del settings[ext_id]
            removed = True
        if not removed:
            continue

        backup = prefs.with_name(f"Preferences.ava-bak-{int(time.time())}")
        try:
            shutil.copy2(prefs, backup)
        except OSError:
            backup = None

        tmp = prefs.with_name("Preferences.ava-tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        shutil.copymode(prefs, tmp)
        os.replace(tmp, prefs)
        changed += 1
        print(f"blocklist limpa: {prefs}")
        if backup:
            print(f"backup: {backup}")

    if not changed:
        print(f"ID {ext_id} não estava blocklistado em nenhum perfil.")
        return 0

    print("Pronto. Abra o Chrome; o external extension deve ser instalado no boot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
