"""
Alpha-code — Patch Tools
========================
apply_patch: formato SEARCH/REPLACE inspirado no Aider.
Mais robusto que str_replace para múltiplas edições em um arquivo.

Formato do patch:
```
<<<SEARCH
def foo():
    return False
===
def foo():
    return validate_token()
>>>
```

Múltiplos blocos SEARCH/REPLACE podem ser aplicados em sequência.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import httpx
import os

from alpha_code.schemas import StepContext, ToolResult
from alpha_code.tools.registry import Tool, schema_function, param

log = logging.getLogger("ava.alpha_code.patch_tools")


# Reuso do scraping_client para write back
_DEFAULT_SCRAPE_URL = os.environ.get("ALPHA_SCRAPE_URL", "http://localhost:3005")
_DEFAULT_TOKEN = os.environ.get("ALPHA_SCRAPE_TOKEN", "")

_client: Optional[httpx.AsyncClient] = None


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=_DEFAULT_SCRAPE_URL,
            timeout=httpx.Timeout(60.0, connect=5.0),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {_DEFAULT_TOKEN}"} if _DEFAULT_TOKEN else {}),
            },
        )
    return _client


async def _read_file(file_path: str) -> str:
    client = await _get_client()
    r = await client.post("/read-file", json={"file_path": file_path})
    if r.status_code >= 400:
        raise RuntimeError(f"read-file failed: {r.status_code} {r.text[:300]}")
    return r.json().get("content", "")


async def _write_file(file_path: str, content: str) -> dict:
    client = await _get_client()
    r = await client.post("/write-file", json={
        "file_path": file_path,
        "content": content,
        "create_parents": False,
        "overwrite": True,
    })
    if r.status_code >= 400:
        raise RuntimeError(f"write-file failed: {r.status_code} {r.text[:300]}")
    return r.json()


# ── Patch parser ─────────────────────────────────────────────────────────────

SEARCH_REPLACEMENT_PATTERN = re.compile(
    r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
    re.DOTALL,
)


def _parse_patch(patch_text: str) -> list[tuple[str, str]]:
    """
    Aceita dois formatos:
      1. Padrão Aider com <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE
      2. Padrão curto com <<<SEARCH ... === ... >>>
    Retorna lista de (search, replace).
    """
    blocks: list[tuple[str, str]] = []

    # Formato 1 (Aider)
    matches = list(SEARCH_REPLACEMENT_PATTERN.finditer(patch_text))
    if matches:
        for m in matches:
            search = m.group(1)
            replace = m.group(2)
            blocks.append((search, replace))
        return blocks

    # Formato 2 (curto)
    pattern2 = re.compile(
        r"<<<SEARCH\n(.*?)\n===\n(.*?)\n>>>",
        re.DOTALL,
    )
    matches = list(pattern2.finditer(patch_text))
    if matches:
        for m in matches:
            blocks.append((m.group(1), m.group(2)))
        return blocks

    return []


def _apply_blocks(content: str, blocks: list[tuple[str, str]]) -> tuple[str, int, list[str]]:
    """
    Aplica blocos em sequência. Retorna (novo_content, total_replaces, erros).
    Cada SEARCH deve ser único no estado atual do conteúdo (após replaces anteriores).
    """
    errors: list[str] = []
    total = 0
    current = content
    for i, (search, replace) in enumerate(blocks):
        if not search:
            errors.append(f"bloco {i+1}: SEARCH vazio")
            continue
        if search == replace:
            errors.append(f"bloco {i+1}: SEARCH == REPLACE (no-op)")
            continue
        count = current.count(search)
        if count == 0:
            errors.append(
                f"bloco {i+1}: SEARCH não encontrado. "
                f"Verifique whitespace/indentação (pode ter mudado em bloco anterior)."
            )
            continue
        if count > 1:
            errors.append(
                f"bloco {i+1}: SEARCH aparece {count} vezes. "
                f"Inclua contexto suficiente para match único."
            )
            continue
        current = current.replace(search, replace, 1)
        total += 1
    return current, total, errors


# ════════════════════════════════════════════════════════════════════════════
# Tool: apply_patch
# ════════════════════════════════════════════════════════════════════════════

class ApplyPatchTool(Tool):
    name = "apply_patch"
    description = (
        "Aplica um patch no formato SEARCH/REPLACE a um arquivo. "
        "Mais robusto que str_replace para múltiplas edições. "
        "Formato (use exatamente):\\n"
        "```\\n<<<<<<< SEARCH\\n<código atual>\\n=======\\n<novo código>\\n>>>>>>> REPLACE\\n```\\n"
        "Pode incluir múltiplos blocos. Cada SEARCH deve ser único no arquivo. "
        "Atenção ao whitespace — copie exatamente o conteúdo atual."
    )

    def schema(self) -> dict:
        return schema_function(
            name=self.name,
            description=self.description,
            properties={
                "file_path": param("string", "Caminho do arquivo relativo ao BASE_DIR"),
                "patch": param(
                    "string",
                    "Patch no formato SEARCH/REPLACE (veja description para formato exato).",
                ),
            },
            required=["file_path", "patch"],
        )

    async def execute(self, args: dict, ctx: StepContext) -> ToolResult:
        file_path = args.get("file_path", "")
        patch_text = args.get("patch", "")
        if not file_path:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error="file_path é obrigatório",
            )
        if not patch_text:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error="patch é obrigatório",
            )

        # 1. Parse blocos
        blocks = _parse_patch(patch_text)
        if not blocks:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error=(
                    "Nenhum bloco SEARCH/REPLACE encontrado. "
                    "Use o formato: <<<<<<< SEARCH\\n...\\n=======\\n...\\n>>>>>>> REPLACE"
                ),
            )

        # 2. Lê conteúdo atual
        try:
            content = await _read_file(file_path)
        except Exception as e:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error=f"Falha ao ler arquivo: {e}",
            )

        # 3. Aplica blocos
        new_content, total, errors = _apply_blocks(content, blocks)
        if total == 0:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error="Nenhum bloco aplicado. " + " | ".join(errors),
            )

        # 4. Escreve de volta
        try:
            data = await _write_file(file_path, new_content)
        except Exception as e:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error=f"Falha ao escrever arquivo: {e}",
            )

        # 5. Reporta sucesso + warnings
        out_parts = [
            f"Patch aplicado em {file_path}: {total}/{len(blocks)} bloco(s) substituído(s).",
            f"bytes_written={data.get('bytes_written', 0)}",
        ]
        if errors and total < len(blocks):
            out_parts.append("Avisos: " + " | ".join(errors))

        return ToolResult(
            tool_call_id="",
            tool_name=self.name,
            success=True,
            output="\n".join(out_parts),
            data={
                "blocks_applied": total,
                "blocks_total": len(blocks),
                "warnings": errors,
                "new_hash": data.get("file_hash", ""),
            },
        )


def register_all(registry) -> None:
    registry.register(ApplyPatchTool())
