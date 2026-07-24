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

# Parser tolerante (fallback quando os dois formatos rígidos acima não batem).
# Aceita variações comuns que modelos geram na prática:
#   - CRLF (\r\n) em vez de \n
#   - Número variável de caracteres nos marcadores (3 a 10x '<'/'='/'>')
#   - Label "SEARCH"/"REPLACE" opcional, com espaços variáveis ao redor
#   - Espaço/tab sobrando no fim da linha do marcador
#   - Texto extra depois de "REPLACE" na mesma linha (ex: crase de code fence)
LENIENT_PATTERN = re.compile(
    r"^[ \t]*<{3,}[ \t]*(?:SEARCH)?[ \t]*\r?\n"
    r"(.*?)\r?\n"
    r"^[ \t]*={3,}[ \t]*\r?\n"
    r"(.*?)\r?\n"
    r"^[ \t]*>{3,}[ \t]*(?:REPLACE)?.*$",
    re.DOTALL | re.MULTILINE,
)


def _parse_patch(patch_text: str) -> list[tuple[str, str]]:
    """
    Aceita três formatos, do mais estrito ao mais tolerante:
      1. Padrão Aider com <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE
      2. Padrão curto com <<<SEARCH ... === ... >>>
      3. Fallback tolerante: qualquer variação razoável dos marcadores acima
         (CRLF, labels/espaços opcionais, nº variável de </=/>). Isso existe
         porque já vimos o modelo gerar um patch que nenhum dos dois formatos
         rígidos reconhecia e o apply_patch falhava inteiro (0 blocos) sem
         nenhuma pista de por quê — o log agora mostra o patch bruto nesse
         caso pra dar pra diagnosticar de vez.
    Retorna lista de (search, replace).
    """
    blocks: list[tuple[str, str]] = []

    # Normaliza CRLF cedo — ajuda até os formatos 1 e 2 abaixo, que já
    # assumem \n puro.
    normalized = patch_text.replace("\r\n", "\n")

    # Formato 1 (Aider)
    matches = list(SEARCH_REPLACEMENT_PATTERN.finditer(normalized))
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
    matches = list(pattern2.finditer(normalized))
    if matches:
        for m in matches:
            blocks.append((m.group(1), m.group(2)))
        return blocks

    # Formato 3 (tolerante) — último recurso antes de desistir
    matches = list(LENIENT_PATTERN.finditer(normalized))
    if matches:
        for m in matches:
            blocks.append((m.group(1), m.group(2)))
        return blocks

    # Nada bateu em nenhum dos 3 formatos — loga o patch bruto (truncado)
    # pra dar pra ver na próxima falha exatamente o que o modelo gerou,
    # em vez de só saber que "0 blocos" foram encontrados.
    log.warning(f"apply_patch: nenhum formato reconhecido. patch_text={patch_text[:1000]!r}")
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
    # ATENÇÃO: mantenha esta description <= 160 chars (incluindo os literais
    # "\n" e "```"). agent.py::_compress_schema corta em max_desc_chars=160
    # antes de enviar ao Groq — uma description mais longa é truncada no meio
    # (já aconteceu: cortava bem no meio do bloco de exemplo, sem nunca chegar
    # a mostrar 'file_path' sendo usado, o que contribuía para o modelo gerar
    # tool calls só com 'patch' e esquecer o 'file_path' obrigatório).
    description = (
        "Edita um arquivo via SEARCH/REPLACE. SEMPRE informe file_path + patch juntos. "
        "patch: <<<<<<< SEARCH\\n<atual>\\n=======\\n<novo>\\n>>>>>>> REPLACE"
    )

    def schema(self) -> dict:
        return schema_function(
            name=self.name,
            description=self.description,
            properties={
                "file_path": param(
                    "string",
                    "Caminho do arquivo relativo ao BASE_DIR. Obrigatório em toda chamada.",
                ),
                "patch": param(
                    "string",
                    "Blocos SEARCH/REPLACE. Cada SEARCH deve ser único no arquivo (inclua contexto).",
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