"""
Alpha-code — Git Tools
=====================
Operações git via subprocess. Roda git no diretório do projeto alvo.

Segurança: apenas operações de leitura + commit em branch atual.
git push é bloqueado (exigiria autenticação e é irreversível).
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from alpha_code.schemas import StepContext, ToolResult
from alpha_code.tools.registry import Tool, schema_function, param

log = logging.getLogger("ava.alpha_code.git_tools")

PROJECT_ROOT = Path(os.environ.get("ALPHA_PROJECT_ROOT", "/home/z/my-project")).resolve()
GIT_TIMEOUT = 15.0


def _git_bin() -> str:
    git = shutil.which("git")
    if not git:
        raise RuntimeError("git não encontrado no PATH")
    return git


def _resolve_cwd(path: Optional[str]) -> Path:
    """Resolve path relativo ao PROJECT_ROOT. Garante que está dentro."""
    if not path or path == ".":
        return PROJECT_ROOT
    p = (PROJECT_ROOT / path).resolve()
    try:
        p.relative_to(PROJECT_ROOT)
    except ValueError:
        raise ValueError(f"Path fora do projeto: {path}")
    return p


async def _run_git(args: list[str], cwd: Path) -> tuple[int, str, str]:
    """Roda git subprocess e retorna (exit_code, stdout, stderr)."""
    git = _git_bin()
    cmd = [git] + args
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=GIT_TIMEOUT)
        return (
            proc.returncode or 0,
            stdout_b.decode("utf-8", errors="replace"),
            stderr_b.decode("utf-8", errors="replace"),
        )
    except asyncio.TimeoutExpired:
        return -1, "", "git timeout"
    except FileNotFoundError:
        return -1, "", "git não encontrado"


# ════════════════════════════════════════════════════════════════════════════
# TOOLS
# ════════════════════════════════════════════════════════════════════════════

class GitStatusTool(Tool):
    name = "git_status"
    description = "Mostra status do repositório git (arquivos modificados/novos/deletados)."

    def schema(self) -> dict:
        return schema_function(
            name=self.name,
            description=self.description,
            properties={
                "path": param("string", "Subdiretório. Default '.'", default="."),
            },
            required=[],
        )

    async def execute(self, args: dict, ctx: StepContext) -> ToolResult:
        try:
            cwd = _resolve_cwd(args.get("path", "."))
        except ValueError as e:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False, error=str(e)
            )
        code, out, err = await _run_git(["status", "--short"], cwd)
        success = code == 0
        out_text = out.strip() if out.strip() else "(working tree clean)"
        return ToolResult(
            tool_call_id="", tool_name=self.name, success=success,
            output=out_text, error=(err if not success else None),
            data={"exit_code": code},
        )


class GitDiffTool(Tool):
    name = "git_diff"
    description = (
        "Mostra diff das mudanças não-commitadas. "
        "Use para revisar o que foi alterado antes de commitar."
    )

    def schema(self) -> dict:
        return schema_function(
            name=self.name,
            description=self.description,
            properties={
                "path": param("string", "Subdiretório. Default '.'", default="."),
                "staged": param("boolean", "Mostrar diff de staged changes (--cached). Default false", default=False),
                "file": param("string", "Apenas um arquivo. Default: todos", default=""),
            },
            required=[],
        )

    async def execute(self, args: dict, ctx: StepContext) -> ToolResult:
        try:
            cwd = _resolve_cwd(args.get("path", "."))
        except ValueError as e:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False, error=str(e)
            )
        git_args = ["diff"]
        if args.get("staged", False):
            git_args.append("--cached")
        if args.get("file"):
            git_args.append("--")
            git_args.append(args["file"])
        code, out, err = await _run_git(git_args, cwd)
        success = code == 0
        out_text = out.strip() if out.strip() else "(no diff)"
        # truncate p/ não estourar contexto
        if len(out_text) > 20_000:
            out_text = out_text[:20_000] + "\n... [truncado]"
        return ToolResult(
            tool_call_id="", tool_name=self.name, success=success,
            output=out_text, error=(err if not success else None),
            data={"exit_code": code},
        )


class GitCommitTool(Tool):
    name = "git_commit"
    description = (
        "Adiciona todas as mudanças (git add -A) e faz commit com a mensagem dada. "
        "NÃO faz push (use push externamente se necessário). "
        "Use para criar checkpoint após completar uma tarefa."
    )

    def schema(self) -> dict:
        return schema_function(
            name=self.name,
            description=self.description,
            properties={
                "message": param("string", "Mensagem de commit (concisa, imperativa)"),
                "path": param("string", "Subdiretório. Default '.'", default="."),
            },
            required=["message"],
        )

    async def execute(self, args: dict, ctx: StepContext) -> ToolResult:
        message = args.get("message", "").strip()
        if not message:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error="message é obrigatório",
            )
        try:
            cwd = _resolve_cwd(args.get("path", "."))
        except ValueError as e:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False, error=str(e)
            )

        # git add -A
        c1, o1, e1 = await _run_git(["add", "-A"], cwd)
        if c1 != 0:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                output=o1, error=f"git add failed: {e1}",
            )

        # git commit -m
        c2, o2, e2 = await _run_git(["commit", "-m", message], cwd)
        # exit_code 0 = OK, 1 = nothing to commit (não é erro real)
        success = c2 == 0 or "nothing to commit" in (e2 + o2).lower() or "no changes" in (e2 + o2).lower()
        out_text = (o2 + ("\n" + e2 if e2 else "")).strip()
        return ToolResult(
            tool_call_id="", tool_name=self.name, success=success,
            output=out_text or "(nothing to commit)",
            data={"exit_code": c2, "added_exit_code": c1},
        )


class GitLogTool(Tool):
    name = "git_log"
    description = "Mostra últimos commits do repositório (hash, autor, mensagem)."

    def schema(self) -> dict:
        return schema_function(
            name=self.name,
            description=self.description,
            properties={
                "path": param("string", "Subdiretório. Default '.'", default="."),
                "max_count": param("integer", "Número de commits. Default 10", default=10),
            },
            required=[],
        )

    async def execute(self, args: dict, ctx: StepContext) -> ToolResult:
        try:
            cwd = _resolve_cwd(args.get("path", "."))
        except ValueError as e:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False, error=str(e)
            )
        n = min(args.get("max_count", 10), 30)
        code, out, err = await _run_git(
            ["log", f"-{n}", "--pretty=format:%h | %an | %ad | %s", "--date=short"],
            cwd,
        )
        success = code == 0
        out_text = out.strip() if out.strip() else "(no commits)"
        return ToolResult(
            tool_call_id="", tool_name=self.name, success=success,
            output=out_text, error=(err if not success else None),
            data={"exit_code": code},
        )


class GitBranchTool(Tool):
    name = "git_branch"
    description = (
        "Lista branches ou cria nova branch. "
        "mode='list' (default) lista todas. mode='create' cria nova branch."
    )

    def schema(self) -> dict:
        return schema_function(
            name=self.name,
            description=self.description,
            properties={
                "path": param("string", "Subdiretório. Default '.'", default="."),
                "mode": param("string", "list | create. Default list", default="list"),
                "name": param("string", "Nome da nova branch (apenas mode=create)", default=""),
            },
            required=[],
        )

    async def execute(self, args: dict, ctx: StepContext) -> ToolResult:
        try:
            cwd = _resolve_cwd(args.get("path", "."))
        except ValueError as e:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False, error=str(e)
            )
        mode = args.get("mode", "list")
        if mode == "create":
            name = args.get("name", "").strip()
            if not name:
                return ToolResult(
                    tool_call_id="", tool_name=self.name, success=False,
                    error="name é obrigatório para mode=create",
                )
            # valida nome simples (sem shell injection)
            if not all(c.isalnum() or c in "-_/" for c in name):
                return ToolResult(
                    tool_call_id="", tool_name=self.name, success=False,
                    error="nome de branch inválido (use apenas [a-zA-Z0-9_-/])",
                )
            code, out, err = await _run_git(["checkout", "-b", name], cwd)
        else:
            code, out, err = await _run_git(["branch", "--list"], cwd)
        success = code == 0
        out_text = out.strip() if out.strip() else "(no branches)"
        return ToolResult(
            tool_call_id="", tool_name=self.name, success=success,
            output=out_text, error=(err if not success else None),
            data={"exit_code": code},
        )


def register_all(registry) -> None:
    registry.register(GitStatusTool())
    registry.register(GitDiffTool())
    registry.register(GitCommitTool())
    registry.register(GitLogTool())
    registry.register(GitBranchTool())
