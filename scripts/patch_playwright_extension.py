#!/usr/bin/env python3
# Alpha AI (AVA) — patch local da extensão Playwright MCP
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gera uma cópia PATCHED da extensão oficial do Playwright (Chrome/Edge).

Motivo
------
Com `PLAYWRIGHT_MCP_EXTENSION_TOKEN`, o `connect.html` chama
`connectToTab` SEM `tab`. O background então usa `sender.tab` — a própria
aba do connect page — como página controlada. Resultado: o MCP nasce numa
aba nova (chrome-extension://.../connect.html) e não na aba ativa real.

Além disso, a extensão oficial move as abas controladas para um tab group
"Playwright · <client>". Este patch:

  1. quando não há `tab` explícito, escolhe a última aba ativa real
     (maior `lastAccessed`, na janela em foco, excluindo páginas internas
     da extensão e URLs não debugáveis);
  2. NÃO cria/move abas para o tab group do Playwright;
  3. ignora mudanças de groupId (para não desanexar abas que o usuário
     mover entre grupos);
  4. no disconnect, NÃO desagrupa abas do usuário.

A cópia é gerada FORA do repositório, em:
  ~/.local/share/alpha-ai/playwright-extension

Instalação (manual, uma vez):
  1. chrome://extensions → Developer mode
  2. Desative/remova a "Playwright Extension" da Chrome Web Store
     (as duas têm o mesmo ID; não podem coexistir)
  3. "Load unpacked" → selecione a pasta gerada
  4. Reinicie a sessão do playwright no mcp_manager/tray

O token em `PLAYWRIGHT_MCP_EXTENSION_TOKEN` continua valendo: a extensão
unpacked usa a mesma `key` do manifest e, portanto, o mesmo ID/armazenamento.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

EXTENSION_ID = "mmlmfjhmonkocbjadbfplnigmagldckm"
if os.name == "nt":
    DEST = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "alpha-ai" / "playwright-extension"
else:
    DEST = Path.home() / ".local" / "share" / "alpha-ai" / "playwright-extension"

if os.name == "nt":
    _local = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    PROFILE_ROOTS = [
        _local / "Google" / "Chrome" / "User Data",
        _local / "Chromium" / "User Data",
        _local / "BraveSoftware" / "Brave-Browser" / "User Data",
        _local / "Microsoft" / "Edge" / "User Data",
    ]
else:
    PROFILE_ROOTS = [
        Path.home() / ".config" / "google-chrome",
        Path.home() / ".config" / "chromium",
        Path.home() / ".config" / "BraveSoftware" / "Brave-Browser",
        Path.home() / ".config" / "microsoft-edge",
    ]


def find_installed_extension() -> Path | None:
    matches: list[tuple[tuple[int, ...], Path]] = []
    for root in PROFILE_ROOTS:
        if not root.is_dir():
            continue
        for manifest in root.glob(f"*/Extensions/{EXTENSION_ID}/*/manifest.json"):
            try:
                version = tuple(
                    int(p) if p.isdigit() else 0
                    for p in manifest.parent.name.split(".")
                )
                data = json.loads(manifest.read_text(encoding="utf-8"))
                if data.get("manifest_version") != 3:
                    continue
                matches.append((version, manifest.parent))
            except Exception:
                continue
    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def insert_after_open(text: str, pattern: str, snippet: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise SystemExit(f"patch: ponto de inserção não encontrado: {pattern}")
    indent = match.group(1)
    payload = "".join(f"{indent}\t{line}\n" if line else "\n" for line in snippet.splitlines())
    return text[: match.end()] + payload + text[match.end() :]


def insert_before_method(text: str, signature: str, snippet: str) -> str:
    pos = text.find(signature)
    if pos < 0:
        raise SystemExit(f"patch: método não encontrado: {signature}")
    line_start = text.rfind("\n", 0, pos) + 1
    indent = text[line_start:pos]
    payload = "".join(f"{indent}{line}\n" if line else "\n" for line in snippet.splitlines())
    return text[:line_start] + payload + text[line_start:]


def patch_connect_tab(text: str) -> str:
    signature = "async _connectTab(selectorTabId, tab, clientName) {"
    start = text.find(signature)
    if start < 0:
        raise SystemExit("patch: _connectTab não encontrado")
    end = text.find("\n\tasync _releaseConnectPage", start)
    if end < 0:
        raise SystemExit("patch: fim de _connectTab não encontrado")

    block = text[start:end]
    # O selectedTab deixa de ser sempre o connect page; tab explícito (UI)
    # continua tendo prioridade. Sem tab, escolhe a aba ativa real.
    block = block.replace(
        signature + "\n",
        signature
        + "\n\t\tconst selectedTab = tab ?? await this._pickAutoTab(await chrome.tabs.get(selectorTabId).catch(() => undefined)); // AVA_PATCHED\n",
        1,
    )
    block = block.replace("tab.id", "selectedTab.id")
    block = block.replace("tab.windowId", "selectedTab.windowId")
    block = block.replace(
        "ConnectedTabGroup(connection, tab,", "ConnectedTabGroup(connection, selectedTab,"
    )
    return text[:start] + block + text[end:]


def patch_background(text: str) -> str:
    if "AVA_PATCHED" in text:
        return text

    # 1. _onMessage não deve substituir "sem tab" pelo connect page.
    old_call = """\t\t\tcase "connectToTab": {
\t\t\t\tconst selectedTab = message.tab ?? sender.tab;
\t\t\t\tthis._connectTab(sender.tab.id, selectedTab, message.clientName)"""
    new_call = """\t\t\tcase "connectToTab": {
\t\t\t\t// AVA_PATCHED: deixa _connectTab resolver a aba ativa real.
\t\t\t\tthis._connectTab(sender.tab.id, message.tab, message.clientName)"""
    if old_call not in text:
        raise SystemExit("patch: bloco connectToTab não encontrado")
    text = text.replace(old_call, new_call, 1)

    # Método auxiliar antes de _connectTab.
    helper = '''async _pickAutoTab(selectorTab) {
\t// AVA_PATCHED: com token, o connect page virava a aba controlada.
\t// Usa a aba ativa real mais recente, excluindo páginas da extensão.
\ttry {
\t\tconst tabs = await chrome.tabs.query({});
\t\tconst candidates = tabs.filter((tab) =>
\t\t\ttab.id !== undefined &&
\t\t\ttab.id !== selectorTab?.id &&
\t\t\t!isNonDebuggableUrl(tab.url) &&
\t\t\t!String(tab.url || "").startsWith("chrome-extension://")
\t\t);
\t\tif (!candidates.length) return selectorTab;
\t\tlet windowId = selectorTab?.windowId;
\t\ttry {
\t\t\tconst win = await chrome.windows.getLastFocused();
\t\t\tif (win?.id !== undefined) windowId = win.id;
\t\t} catch {}
\t\tconst sameWindow = candidates.filter((tab) => tab.windowId === windowId);
\t\tconst pool = sameWindow.length ? sameWindow : candidates;
\t\tpool.sort((a, b) => (b.lastAccessed || 0) - (a.lastAccessed || 0));
\t\treturn pool[0] || selectorTab;
\t} catch {
\t\treturn selectorTab;
\t}
}'''
    text = insert_before_method(
        text, "async _connectTab(selectorTabId, tab, clientName) {", helper
    )
    text = patch_connect_tab(text)

    # 2. Não criar/mover abas para o tab group do Playwright.
    text = insert_after_open(
        text,
        r"^(\s*)async _addTabToGroup\(tabId\) \{\n",
        "// AVA_PATCHED: sem separação por tab group.\n"
        "this._groupTabIds.add(tabId);\n"
        "return;",
    )

    # 3. Mudanças de groupId não devem desanexar a aba controlada.
    text = insert_after_open(
        text,
        r"^(\s*)_onTabGroupChanged\(tabId, tab\) \{\n",
        "// AVA_PATCHED: ignora mudanças de grupo.\nreturn;",
    )

    # 4. No disconnect, não desagrupar abas que pertencem ao usuário.
    text = insert_after_open(
        text,
        r"^(\s*)_onConnectionClose\(\) \{\n",
        "// AVA_PATCHED: nunca desagrupa abas do usuário.\nthis._groupTabIds.clear();",
    )

    # 5. O connect page é fechado após conectar (para não deixar uma aba
    #    extra); o keepalive que ele fazia no service worker passa a ser
    #    feito aqui, senão o MV3 pode suspender o worker e derrubar o WS.
    if "new PlaywrightExtension();" in text:
        text = text.replace(
            "new PlaywrightExtension();",
            "new PlaywrightExtension();\n\n"
            "// AVA_PATCHED: keepalive do service worker sem a aba do connect page.\n"
            "setInterval(() => { chrome.runtime.getPlatformInfo().catch(() => {}); }, 20000);",
            1,
        )

    return text


def patch_auth_token(text: str, token: str) -> str:
    """Fixa o token da extensão para o valor gerado pelo instalador.

    Sem isso, a extensão patcheada (novo ID) geraria outro token e o
    PLAYWRIGHT_MCP_EXTENSION_TOKEN antigo deixaria de casar.
    """
    if "AVA_PATCHED_TOKEN" in text:
        return text
    pattern = re.compile(
        r"var getOrCreateAuthToken = \(\) => \{\n"
        r"(?:.*\n)*?"
        r"\};",
        re.MULTILINE,
    )
    replacement = (
        "var getOrCreateAuthToken = () => {\n"
        "\t// AVA_PATCHED_TOKEN: token fixo definido pelo instalador.\n"
        f"\treturn {json.dumps(token)};\n"
        "};"
    )
    patched, n = pattern.subn(replacement, text, count=1)
    if n != 1:
        raise SystemExit("patch: getOrCreateAuthToken não encontrado em authToken.js")
    return patched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token",
        default=os.environ.get("PLAYWRIGHT_MCP_EXTENSION_TOKEN", ""),
        help="token fixo a embutir na extensão (default: env ou mantém o gerado)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEST,
        help=f"diretório de saída (default: {DEST})",
    )
    args = parser.parse_args()

    installed = find_installed_extension()
    if installed is None:
        print(
            "Extensão Playwright não encontrada nos perfis conhecidos.\n"
            f"Esperado: ~/.config/<browser>/<profile>/Extensions/{EXTENSION_ID}/<versao>/",
            file=sys.stderr,
        )
        return 1

    print(f"Origem: {installed}")
    if args.out.exists():
        shutil.rmtree(args.out)
    shutil.copytree(installed, args.out)
    # Metadados de verificação da versão da Web Store não valem para a cópia
    # patcheada; mantê-los faria o Chrome marcar a extensão como corrompida.
    shutil.rmtree(args.out / "_metadata", ignore_errors=True)

    background = args.out / "lib" / "background.mjs"
    patched = patch_background(background.read_text(encoding="utf-8"))
    background.write_text(patched, encoding="utf-8")

    if args.token:
        auth = args.out / "lib" / "ui" / "authToken.js"
        auth.write_text(
            patch_auth_token(auth.read_text(encoding="utf-8"), args.token),
            encoding="utf-8",
        )
        print("Token fixo embutido em lib/ui/authToken.js")

    print(f"Patch aplicado em: {args.out}")
    print()
    print("Instalação da extensão patcheada:")
    print("  1. Abra chrome://extensions e ative o Developer mode")
    print("  2. REMOVA a 'Playwright Extension' da Chrome Web Store")
    print("     (mesmo ID; as duas não podem coexistir)")
    print("  3. Clique em 'Load unpacked' e selecione:")
    print(f"       {args.out}")
    if args.token:
        print("  4. O token fixo já está embutido na extensão e deve ser copiado")
        print("     para PLAYWRIGHT_MCP_EXTENSION_TOKEN em Modules/.env:")
        print(f"       {args.token}")
    else:
        print("  4. Abra chrome-extension://" + EXTENSION_ID + "/status.html")
        print("     e copie o token exibido")
        print("  5. Atualize PLAYWRIGHT_MCP_EXTENSION_TOKEN em Modules/.env")
    print("  6. Reinicie a sessão do playwright no mcp_manager/tray")
    print()
    print("O connect page ainda abre por um instante (é o handshake da")
    print("extensão), mas é fechado logo após conectar — a aba controlada")
    print("passa a ser a última aba ativa real, sem navegar do zero.")
    print("AVISO: isso remove o isolamento por tab group e dá ao MCP acesso")
    print("direto à aba ativa. Use apenas na sua máquina/perfil confiável.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
