"""
Alpha-code — File Tools
========================
Tools que operam arquivos via scraping_client (porta 3005).

Reuso do serviço já existente:
  - POST /list-files   ← lista entradas em dir (glob)
  - POST /read-file    ← lê arquivo (já existia)
  - POST /write-file   ← escreve arquivo (NOVO)
  - POST /str-replace  ← replace string (NOVO)
  - POST /execute      ← roda comando no BASE_DIR
  - POST /stat         ← stat de path

Todas as tools herdam de Tool e expõem schema() + execute().
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

from alpha_code.schemas import StepContext, ToolResult
from alpha_code.tools.registry import Tool, schema_function, param

log = logging.getLogger("ava.alpha_code.file_tools")


# ── HTTP client ─────────────────────────────────────────────────────────────

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


async def _post(path: str, payload: dict) -> dict:
    """POST helper que levanta exceção em status não-2xx."""
    client = await _get_client()
    r = await client.post(path, json=payload)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text[:500])
        except Exception:
            detail = r.text[:500]
        raise RuntimeError(f"scraping_client {path} → {r.status_code}: {detail}")
    return r.json()


async def close_client() -> None:
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


# ════════════════════════════════════════════════════════════════════════════
# TOOLS
# ════════════════════════════════════════════════════════════════════════════

class ListFilesTool(Tool):
    name = "list_files"
    description = (
        "Lista arquivos e diretórios em um path do projeto. "
        "Retorna nome, tipo (arquivo/dir), tamanho e mtime. "
        "Use para descobrir a estrutura do projeto antes de buscar símbolos específicos."
    )

    def schema(self) -> dict:
        return schema_function(
            name=self.name,
            description=self.description,
            properties={
                "path": param("string", "Caminho relativo ao BASE_DIR. Default '.'", default="."),
                "pattern": param("string", "Glob pattern. Default '*'", default="*"),
                "recursive": param("boolean", "Buscar recursivamente. Default true", default=True),
                "max_entries": param("integer", "Máximo de entradas. Default 500", default=500),
                "include_hidden": param("boolean", "Incluir arquivos ocultos (.git, etc). Default false", default=False),
            },
            required=[],
        )

    async def execute(self, args: dict, ctx: StepContext) -> ToolResult:
        payload = {
            "path": args.get("path", "."),
            "pattern": args.get("pattern", "*"),
            "recursive": args.get("recursive", True),
            "max_entries": args.get("max_entries", 500),
            "include_hidden": args.get("include_hidden", False),
        }
        data = await _post("/list-files", payload)
        entries = data.get("entries", [])
        # compacta para o LLM: lines curtas
        lines = []
        for e in entries[:200]:
            kind = "D" if e.get("is_dir") else "F"
            size = e.get("size", 0)
            lines.append(f"[{kind}] {e['path']}  ({size}B)")
        out = "\n".join(lines)
        if len(entries) > 200:
            out += f"\n... ({len(entries) - 200} entradas omitidas)"
        if data.get("truncated"):
            out += "\n[AVISO: lista truncada, use max_entries maior]"
        return ToolResult(
            tool_call_id="",
            tool_name=self.name,
            success=True,
            output=out,
            data={"total": data.get("total", 0), "truncated": data.get("truncated", False)},
        )


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Lê o conteúdo de um arquivo de texto. "
        "Sempre prefira ler apenas arquivos relevantes. "
        "Retorna conteúdo + hash + tamanho."
    )

    def schema(self) -> dict:
        return schema_function(
            name=self.name,
            description=self.description,
            properties={
                "file_path": param("string", "Caminho do arquivo relativo ao BASE_DIR"),
                "encoding": param("string", "Encoding. Default utf-8", default="utf-8"),
            },
            required=["file_path"],
        )

    async def execute(self, args: dict, ctx: StepContext) -> ToolResult:
        file_path = args.get("file_path", "")
        if not file_path:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error="file_path é obrigatório",
            )
        data = await _post("/read-file", {"file_path": file_path})
        content = data.get("content", "")
        truncated = data.get("truncated", False)
        out = content
        if truncated:
            out += "\n\n[AVISO: arquivo truncado em 500_000 caracteres]"
        return ToolResult(
            tool_call_id="",
            tool_name=self.name,
            success=True,
            output=out,
            data={
                "size": data.get("size", 0),
                "file_hash": data.get("file_hash", ""),
                "modified": data.get("modified", ""),
                "truncated": truncated,
            },
        )


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Cria ou sobrescreve um arquivo com o conteúdo dado. "
        "CUIDADO: sobrescreve conteúdo existente sem perguntar. "
        "Para criar arquivo novo, use create_parents=true. "
        "Para evitar perdas, prefira str_replace para edições."
    )

    def schema(self) -> dict:
        return schema_function(
            name=self.name,
            description=self.description,
            properties={
                "file_path": param("string", "Caminho do arquivo relativo ao BASE_DIR"),
                "content": param("string", "Conteúdo completo do arquivo"),
                "create_parents": param("boolean", "Criar diretórios pais se não existirem. Default true", default=True),
                "overwrite": param("boolean", "Sobrescrever se existir. Default true", default=True),
            },
            required=["file_path", "content"],
        )

    async def execute(self, args: dict, ctx: StepContext) -> ToolResult:
        data = await _post("/write-file", {
            "file_path": args["file_path"],
            "content": args["content"],
            "create_parents": args.get("create_parents", True),
            "overwrite": args.get("overwrite", True),
        })
        return ToolResult(
            tool_call_id="",
            tool_name=self.name,
            success=True,
            output=f"Arquivo escrito: {data['file_path']} ({data['bytes_written']} bytes, created={data['created']})",
            data=data,
        )


class StrReplaceTool(Tool):
    name = "str_replace"
    description = (
        "Substitui uma string por outra em um arquivo. "
        "FALHA se old_str não existir, ou se aparecer >1 vez e replace_all=false. "
        "Inclua contexto suficiente (linhas ao redor) para garantir match único."
    )

    def schema(self) -> dict:
        return schema_function(
            name=self.name,
            description=self.description,
            properties={
                "file_path": param("string", "Caminho do arquivo relativo ao BASE_DIR"),
                "old_str": param("string", "Texto a substituir (inclua contexto suficiente para match único)"),
                "new_str": param("string", "Novo texto"),
                "replace_all": param("boolean", "Substituir todas as ocorrências. Default false", default=False),
            },
            required=["file_path", "old_str", "new_str"],
        )

    async def execute(self, args: dict, ctx: StepContext) -> ToolResult:
        data = await _post("/str-replace", {
            "file_path": args["file_path"],
            "old_str": args["old_str"],
            "new_str": args["new_str"],
            "replace_all": args.get("replace_all", False),
        })
        return ToolResult(
            tool_call_id="",
            tool_name=self.name,
            success=True,
            output=f"Substituição feita em {data['file_path']}: {data['replacements']} ocorrência(s). hash={data['new_hash'][:8]}",
            data=data,
        )


class RunCommandTool(Tool):
    name = "run_command"
    description = (
        "Executa um comando shell no diretório do projeto (BASE_DIR do scraping_client). "
        "Use para: build, lint, formatador, install de deps, git. "
        "EVITE: rm -rf, mkfs, format (bloqueado). "
        "Timeout padrão 30s."
    )

    def schema(self) -> dict:
        return schema_function(
            name=self.name,
            description=self.description,
            properties={
                "command": param("string", "Comando shell a executar"),
                "timeout": param("number", "Timeout em segundos (máx 30). Default 30", default=30.0),
            },
            required=["command"],
        )

    async def execute(self, args: dict, ctx: StepContext) -> ToolResult:
        command = args.get("command", "").strip()
        if not command:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error="command vazio",
            )
        # blocklist básico extra (o scraping_client já tem o dele)
        cmd_l = command.lower()
        for pat in [r"\brm\s+-rf\s+/", r"\bmkfs\.", r"\bdd\s+if="]:
            import re
            if re.search(pat, cmd_l):
                return ToolResult(
                    tool_call_id="", tool_name=self.name, success=False,
                    error=f"Comando bloqueado por policy: {command}",
                )

        data = await _post("/execute", {
            "command": command,
            "timeout": min(args.get("timeout", 30.0), 30.0),
        })
        success = data.get("exit_code", -1) == 0 and not data.get("timed_out", False)
        # Compose output truncating if huge
        stdout = (data.get("stdout") or "")[:50_000]
        stderr = (data.get("stderr") or "")[:10_000]
        out_parts = []
        if stdout:
            out_parts.append(f"STDOUT:\n{stdout}")
        if stderr:
            out_parts.append(f"STDERR:\n{stderr}")
        out_parts.append(f"exit_code: {data.get('exit_code')}")
        if data.get("timed_out"):
            out_parts.append("[TIMEOUT]")
        return ToolResult(
            tool_call_id="",
            tool_name=self.name,
            success=success,
            output="\n".join(out_parts),
            data=data,
        )


class StatTool(Tool):
    name = "stat"
    description = "Verifica existência/tamanho/tipo de um path sem ler conteúdo."

    def schema(self) -> dict:
        return schema_function(
            name=self.name,
            description=self.description,
            properties={"path": param("string", "Caminho relativo ao BASE_DIR")},
            required=["path"],
        )

    async def execute(self, args: dict, ctx: StepContext) -> ToolResult:
        data = await _post("/stat", {"path": args["path"]})
        return ToolResult(
            tool_call_id="",
            tool_name=self.name,
            success=True,
            output=str(data),
            data=data,
        )


# ── Registry helper ──────────────────────────────────────────────────────────

def register_all(registry) -> None:
    """Registra todas as file tools no registry dado."""
    registry.register(ListFilesTool())
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(StrReplaceTool())
    registry.register(RunCommandTool())
    registry.register(StatTool())
