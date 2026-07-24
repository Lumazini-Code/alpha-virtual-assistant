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


class ScrapeClientError(Exception):
    """Erro do scraping_client que preserva o status_code original.

    Antes tudo virava RuntimeError genérico, e quem chamava _post não tinha
    como distinguir "old_str não bateu" (422, recuperável, o modelo só
    precisa reler o arquivo e tentar de novo) de um erro de verdade (500,
    403, etc). Isso fazia o registry.py logar até esses casos esperados
    como "Tool crashed" (log.exception), o que é ruidoso e enganoso.
    """

    def __init__(self, path: str, status_code: int, detail: str):
        self.path = path
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"scraping_client {path} → {status_code}: {detail}")


async def _post(path: str, payload: dict) -> dict:
    """POST helper que levanta ScrapeClientError em status não-2xx."""
    client = await _get_client()
    r = await client.post(path, json=payload)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text[:500])
        except Exception:
            detail = r.text[:500]
        raise ScrapeClientError(path, r.status_code, detail)
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
        "Lista arquivos/dirs de um path do projeto: nome, tipo, tamanho, mtime. "
        "Use antes de buscar símbolos específicos."
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
        try:
            data = await _post("/read-file", {"file_path": file_path})
        except ScrapeClientError as e:
            if e.status_code == 404:
                # Path não existe — comum quando o modelo erra o path (ex:
                # usou um path chutado ou um identificador interno de outra
                # tool, como o "arquivo#chunk" do semantic_search). É
                # recuperável: o loop ReAct deve rodar list_files e tentar de
                # novo, não é um crash de verdade. Sem isso, todo 404 virava
                # log.exception("crashed") no registry.py — ruidoso e
                # enganoso pro mesmo motivo já documentado em StrReplaceTool.
                return ToolResult(
                    tool_call_id="", tool_name=self.name, success=False,
                    error=f"Arquivo não encontrado: {file_path}. Use list_files para confirmar o path exato.",
                )
            raise
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
        "Cria/sobrescreve um arquivo com o conteúdo dado. CUIDADO: sobrescreve "
        "sem perguntar. Prefira apply_patch pra editar sem perder conteúdo."
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
        "Substitui uma string por outra em um arquivo. FALHA se old_str não "
        "existir ou aparecer >1x (replace_all=false). Inclua contexto pro match único."
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
        try:
            data = await _post("/str-replace", {
                "file_path": args["file_path"],
                "old_str": args["old_str"],
                "new_str": args["new_str"],
                "replace_all": args.get("replace_all", False),
            })
        except ScrapeClientError as e:
            if e.status_code == 422:
                # old_str não bateu com o conteúdo atual — esperado e
                # recuperável, não é um crash. Devolve ToolResult normal
                # (success=False) pro loop ReAct decidir reler o arquivo,
                # em vez de propagar e virar log.exception("crashed").
                return ToolResult(
                    tool_call_id="", tool_name=self.name, success=False,
                    error=e.detail,
                )
            if e.status_code == 409:
                return ToolResult(
                    tool_call_id="", tool_name=self.name, success=False,
                    error=e.detail,
                )
            # Demais status codes (403, 404 de path real, 500...) seguem
            # sendo tratados como erro genérico pelo registry.py.
            raise
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
        "Roda comando shell no BASE_DIR do projeto. Use pra build/lint/deps/git. "
        "EVITE rm -rf/mkfs/format (bloqueado). Timeout 30s."
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
    """Registra todas as file tools no registry dado.

    NOTA: StrReplaceTool NÃO é registrada de propósito. apply_patch
    (patch_tools.py) cobre o mesmo caso de uso com múltiplos blocos por
    chamada e erro detalhado por bloco (qual bloco falhou, 0x vs Nx
    ocorrências) — é um superconjunto estrito. Manter as duas registradas
    dava ao modelo duas ferramentas sobrepostas para "editar arquivo", o
    que piora a escolha de tool e não trouxe benefício real em troca. A
    classe StrReplaceTool continua definida (e com o tratamento de erro
    422/409 corrigido) caso algum outro fluxo precise dela diretamente.
    """
    registry.register(ListFilesTool())
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(RunCommandTool())
    registry.register(StatTool())