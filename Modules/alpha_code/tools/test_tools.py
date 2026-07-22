"""
Alpha-code — Test Tools
======================
Detecta test runner do projeto e executa via scraping_client:3005/execute.

Detecção:
  - pytest   (pyproject.toml / pytest.ini / setup.cfg com [tool:pytest])
  - jest     (package.json com jest em devDependencies)
  - vitest   (package.json com vitest em devDependencies)
  - cargo t  (Cargo.toml)
  - go test  (go.mod)
  - dotnet   (*.csproj ou *.sln)
  - maven    (pom.xml)

Fallback: roda `pytest` e tenta `npm test`.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import httpx

from alpha_code.schemas import StepContext, ToolResult
from alpha_code.tools.registry import Tool, schema_function, param
from alpha_code.tools.file_tools import _post as _scrape_post

log = logging.getLogger("ava.alpha_code.test_tools")


# ════════════════════════════════════════════════════════════════════════════
# Tool: run_tests
# ════════════════════════════════════════════════════════════════════════════

class RunTestsTool(Tool):
    name = "run_tests"
    description = (
        "Executa testes do projeto. Detecta automaticamente o runner (pytest/jest/vitest/cargo/go). "
        "Em caso de falha, retorna stdout completo com linhas de erro para você corrigir. "
        "Use após fazer mudanças para validar."
    )

    def schema(self) -> dict:
        return schema_function(
            name=self.name,
            description=self.description,
            properties={
                "command": param(
                    "string",
                    "Comando explícito. Se vazio, detecta automaticamente pelo projeto.",
                    default="",
                ),
                "timeout": param("number", "Timeout em segundos (máx 30). Default 30", default=30.0),
                "target": param(
                    "string",
                    "Alvo específico (ex: 'tests/test_foo.py::test_bar' ou 'src/__tests__/foo.test.ts'). Default: todos.",
                    default="",
                ),
            },
            required=[],
        )

    async def execute(self, args: dict, ctx: StepContext) -> ToolResult:
        timeout = min(args.get("timeout", 30.0), 30.0)
        explicit = args.get("command", "").strip()
        target = args.get("target", "").strip()

        # Se comando explícito, usa direto
        if explicit:
            command = explicit
            if target:
                command = f"{command} {target}"
        else:
            # Detecta runner
            command = await _detect_runner(target)
            if command is None:
                return ToolResult(
                    tool_call_id="", tool_name=self.name, success=False,
                    error=(
                        "Não foi possível detectar o runner de testes. "
                        "Use o parâmetro 'command' para especificar manualmente (ex: 'pytest', 'npm test')."
                    ),
                )

        # Executa via scraping_client
        try:
            data = await _scrape_post("/execute", {"command": command, "timeout": timeout})
        except Exception as e:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error=f"scraping_client /execute falhou: {e}",
            )

        exit_code = data.get("exit_code", -1)
        timed_out = data.get("timed_out", False)
        success = exit_code == 0 and not timed_out

        stdout = (data.get("stdout") or "")[:50_000]
        stderr = (data.get("stderr") or "")[:10_000]

        # Tenta extrair número de failures do pytest/jest para o LLM
        summary = _parse_test_summary(stdout, stderr, exit_code)

        out_parts = [f"$ {command}"]
        if stdout:
            out_parts.append(f"STDOUT:\n{stdout}")
        if stderr:
            out_parts.append(f"STDERR:\n{stderr}")
        out_parts.append(f"exit_code: {exit_code}")
        if timed_out:
            out_parts.append("[TIMEOUT]")
        if summary:
            out_parts.append(f"[SUMMARY] {summary}")

        return ToolResult(
            tool_call_id="",
            tool_name=self.name,
            success=success,
            output="\n".join(out_parts),
            data={
                "exit_code": exit_code,
                "timed_out": timed_out,
                "summary": summary,
            },
        )


# ════════════════════════════════════════════════════════════════════════════
# Tool: run_linter
# ════════════════════════════════════════════════════════════════════════════

class RunLinterTool(Tool):
    name = "run_linter"
    description = (
        "Executa linter do projeto. Detecta ruff (Python), eslint (JS/TS), golangci-lint. "
        "Retorna warnings/errors para você corrigir."
    )

    def schema(self) -> dict:
        return schema_function(
            name=self.name,
            description=self.description,
            properties={
                "command": param("string", "Comando explícito. Default: detecta automaticamente.", default=""),
                "timeout": param("number", "Timeout em segundos. Default 30", default=30.0),
                "fix": param("boolean", "Aplicar fixes automáticos (--fix). Default false", default=False),
            },
            required=[],
        )

    async def execute(self, args: dict, ctx: StepContext) -> ToolResult:
        timeout = min(args.get("timeout", 30.0), 30.0)
        explicit = args.get("command", "").strip()
        fix = args.get("fix", False)

        if explicit:
            command = explicit
        else:
            # Detecta linter
            command = await _detect_linter(fix)
            if command is None:
                return ToolResult(
                    tool_call_id="", tool_name=self.name, success=False,
                    error="Não foi possível detectar linter. Use 'command' para especificar.",
                )

        try:
            data = await _scrape_post("/execute", {"command": command, "timeout": timeout})
        except Exception as e:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error=f"scraping_client /execute falhou: {e}",
            )

        exit_code = data.get("exit_code", -1)
        # Linter geralmente retorna != 0 quando encontra issues (não é erro fatal)
        stdout = (data.get("stdout") or "")[:50_000]
        stderr = (data.get("stderr") or "")[:10_000]

        out_parts = [f"$ {command}"]
        if stdout:
            out_parts.append(f"STDOUT:\n{stdout}")
        if stderr:
            out_parts.append(f"STDERR:\n{stderr}")
        out_parts.append(f"exit_code: {exit_code}")

        return ToolResult(
            tool_call_id="",
            tool_name=self.name,
            success=exit_code == 0,
            output="\n".join(out_parts),
            data={"exit_code": exit_code},
        )


# ── Detecção helpers ─────────────────────────────────────────────────────────


async def _scrape_list_files(path: str = ".") -> list[str]:
    """Lista nomes de arquivos no BASE_DIR via scraping_client."""
    try:
        data = await _scrape_post("/list-files", {
            "path": path, "pattern": "*",
            "recursive": False, "max_entries": 100,
            "include_hidden": False,
        })
        return [e["name"] for e in data.get("entries", []) if e.get("is_file")]
    except Exception:
        return []


async def _detect_runner(target: str = "") -> Optional[str]:
    """Detecta runner de testes pelo projeto."""
    files = await _scrape_list_files(".")

    has_pytest_config = any(
        f in files for f in ("pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini")
    )
    has_pkg_json = "package.json" in files
    has_cargo = "Cargo.toml" in files
    has_go_mod = "go.mod" in files
    has_csproj = any(f.endswith(".csproj") or f.endswith(".sln") for f in files)
    has_pom = "pom.xml" in files

    # Python
    if has_pytest_config:
        cmd = "pytest -v"
        if target:
            cmd += f" {target}"
        return cmd
    # JS/TS
    if has_pkg_json:
        # tenta detectar vitest vs jest
        try:
            data = await _scrape_post("/read-file", {"file_path": "package.json"})
            content = data.get("content", "")
            if "vitest" in content:
                cmd = "npx vitest run"
            elif "jest" in content:
                cmd = "npx jest"
            else:
                cmd = "npm test"
            if target:
                cmd += f" {target}"
            return cmd
        except Exception:
            return "npm test"
    # Rust
    if has_cargo:
        return "cargo test" + (f" {target}" if target else "")
    # Go
    if has_go_mod:
        return "go test ./..." + (f" -run {target}" if target else "")
    # .NET
    if has_csproj:
        return "dotnet test"
    # Java
    if has_pom:
        return "mvn test"
    return None


async def _detect_linter(fix: bool = False) -> Optional[str]:
    files = await _scrape_list_files(".")
    has_pyproject = "pyproject.toml" in files
    has_pkg_json = "package.json" in files
    has_eslint_config = any(
        f in files for f in (".eslintrc.js", ".eslintrc.json", ".eslintrc.yml", ".eslintrc")
    )
    has_go_mod = "go.mod" in files
    has_cargo = "Cargo.toml" in files

    if has_pyproject:
        # tenta ruff primeiro
        return f"ruff check{' --fix' if fix else ''} ."
    if has_pkg_json or has_eslint_config:
        return f"npx eslint{' --fix' if fix else ''} ."
    if has_go_mod:
        return "golangci-lint run ./..."
    if has_cargo:
        return "cargo clippy"
    return None


def _parse_test_summary(stdout: str, stderr: str, exit_code: int) -> str:
    """Extrai contagem de pytest/jest/vitest do output."""
    import re
    # pytest: "===== 3 failed, 5 passed in 1.2s ====="
    m = re.search(r"(\d+) failed.*?(\d+) passed", stdout)
    if m:
        return f"pytest: {m.group(1)} failed, {m.group(2)} passed"
    m = re.search(r"(\d+) passed", stdout)
    if m:
        return f"pytest: {m.group(1)} passed"
    # jest: "Tests: 2 failed, 5 passed, 7 total"
    m = re.search(r"Tests:\s+(\d+) failed.*?(\d+) passed.*?(\d+) total", stdout)
    if m:
        return f"jest: {m.group(1)} failed, {m.group(2)} passed, {m.group(3)} total"
    # vitest: "Tests  2 failed | 5 passed (7)"
    m = re.search(r"Tests\s+(\d+) failed.*?(\d+) passed", stdout)
    if m:
        return f"vitest: {m.group(1)} failed, {m.group(2)} passed"
    # cargo: "test result: FAILED. 1 passed; 2 failed"
    m = re.search(r"test result:.*?(\d+) passed.*?(\d+) failed", stdout)
    if m:
        return f"cargo: {m.group(1)} passed, {m.group(2)} failed"
    return ""


def register_all(registry) -> None:
    registry.register(RunTestsTool())
    registry.register(RunLinterTool())
