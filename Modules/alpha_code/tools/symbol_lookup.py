"""
Alpha-code — Symbol Lookup (tree-sitter)
========================================
Busca por classes, funções e métodos em Python/JS/TS.

Não exige LSP — usa tree-sitter direto, parse local.
Fallback: se tree-sitter não estiver instalado, faz regex fallback.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Optional

from alpha_code.schemas import StepContext, ToolResult
from alpha_code.tools.registry import Tool, schema_function, param

log = logging.getLogger("ava.alpha_code.symbol_lookup")

PROJECT_ROOT = Path(os.environ.get("ALPHA_PROJECT_ROOT", "/home/z/my-project")).resolve()

# Tenta importar tree-sitter e linguagens
try:
    from tree_sitter import Language, Parser
    TS_AVAILABLE = True
except ImportError:
    TS_AVAILABLE = False
    log.info("tree-sitter não instalado — symbol_lookup usará regex fallback")

# Tenta importar linguagens específicas (tree-sitter-languages)
try:
    from tree_sitter_languages import get_language
    LANGS_AVAILABLE = True
except ImportError:
    LANGS_AVAILABLE = False
    log.info("tree-sitter-languages não instalado — symbol_lookup limitado")

# Queries por linguagem (simplified — captura nome do nó)
LANG_BY_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
}


# ════════════════════════════════════════════════════════════════════════════
# Tool: symbol_lookup
# ════════════════════════════════════════════════════════════════════════════

class SymbolLookupTool(Tool):
    name = "symbol_lookup"
    description = (
        "Busca definição de símbolos (classes, funções, métodos) por nome. "
        "Retorna arquivo:linha onde cada símbolo é definido. "
        "Use para localizar onde uma classe/função é definida antes de editá-la."
    )

    def schema(self) -> dict:
        return schema_function(
            name=self.name,
            description=self.description,
            properties={
                "name": param("string", "Nome do símbolo (classe, função, método)"),
                "kind": param(
                    "string",
                    "Filtrar por tipo: class | function | method | any. Default any",
                    default="any",
                ),
                "file_type": param(
                    "string",
                    "Extensão (ex: 'py', 'js', 'ts'). Default: busca em todas as suportadas.",
                    default="",
                ),
                "path": param("string", "Subdiretório. Default '.'", default="."),
                "max_results": param("integer", "Máximo de resultados. Default 20", default=20),
            },
            required=["name"],
        )

    async def execute(self, args: dict, ctx: StepContext) -> ToolResult:
        name = args.get("name", "").strip()
        if not name:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error="name é obrigatório",
            )

        kind = args.get("kind", "any").lower()
        file_type = (args.get("file_type") or "").strip().lstrip(".")
        path = args.get("path", ".")
        max_results = min(args.get("max_results", 20), 50)

        target = (PROJECT_ROOT / path).resolve()
        try:
            target.relative_to(PROJECT_ROOT)
        except ValueError:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error=f"Path fora do projeto: {path}",
            )

        # Coleta arquivos
        if file_type:
            exts = [f".{file_type}"]
        else:
            exts = list(LANG_BY_EXT.keys())

        files: list[Path] = []
        for ext in exts:
            files.extend(target.rglob(f"*{ext}"))
        # filter exclusions
        excluded = {".git", "node_modules", "__pycache__", ".venv", "venv",
                    "dist", "build", ".tox", ".eggs", ".mypy_cache", ".pytest_cache"}
        filtered = []
        for f in files:
            try:
                rel_parts = f.resolve().relative_to(PROJECT_ROOT).parts
            except Exception:
                continue
            if any(p in excluded for p in rel_parts):
                continue
            if f.stat().st_size > 500 * 1024:
                continue
            filtered.append(f)
        files = filtered[:300]

        if not files:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=True,
                output="Nenhum arquivo encontrado.",
                data={"count": 0},
            )

        # Busca símbolos em cada arquivo
        results: list[dict] = []
        for f in files:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            try:
                rel = str(f.resolve().relative_to(PROJECT_ROOT))
            except Exception:
                rel = str(f)

            ext = f.suffix.lower()
            lang_name = LANG_BY_EXT.get(ext)
            if not lang_name:
                continue

            matches = []
            if TS_AVAILABLE and LANGS_AVAILABLE:
                try:
                    matches = _find_symbols_ts(content, lang_name, name, kind)
                except Exception as e:
                    log.debug(f"tree-sitter falhou em {rel}: {e}")
                    matches = _find_symbols_regex(content, lang_name, name, kind)
            else:
                matches = _find_symbols_regex(content, lang_name, name, kind)

            for m in matches:
                results.append({
                    "path": rel,
                    "line": m["line"],
                    "kind": m["kind"],
                    "name": m["name"],
                    "snippet": m["snippet"],
                })
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break

        if not results:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=True,
                output=f"Símbolo '{name}' não encontrado.",
                data={"count": 0},
            )

        lines = []
        for r in results:
            lines.append(
                f"{r['path']}:{r['line']}  [{r['kind']}] {r['name']}\n  {r['snippet']}"
            )
        out = "\n\n".join(lines)
        return ToolResult(
            tool_call_id="",
            tool_name=self.name,
            success=True,
            output=out,
            data={"count": len(results), "truncated": len(results) >= max_results},
        )


# ── Parsers ─────────────────────────────────────────────────────────────────

def _find_symbols_ts(content: str, lang: str, name: str, kind: str) -> list[dict]:
    """Usa tree-sitter para encontrar definições de símbolos."""
    language = get_language(lang)
    parser = Parser()
    parser.language = language
    tree = parser.parse(content.encode("utf-8"))

    # queries simples por linguagem
    queries = {
        "python": "(class_definition name: (identifier) @name) (function_definition name: (identifier) @name)",
        "javascript": "(class_declaration name: (identifier) @name) (function_declaration name: (identifier) @name) (method_definition name: (property_identifier) @name)",
        "typescript": "(class_declaration name: (type_identifier) @name) (function_declaration name: (identifier) @name) (method_definition name: (property_identifier) @name)",
        "go": "(function_declaration name: (identifier) @name) (method_declaration name: (field_identifier) @name) (type_declaration name: (type_identifier) @name)",
        "rust": "(function_item name: (identifier) @name) (struct_item name: (type_identifier) @name) (impl_item type: (type_identifier) @name)",
    }
    query_str = queries.get(lang, "")
    if not query_str:
        return []

    query = language.query(query_str)
    captures = query.captures(tree.root_node)

    results: list[dict] = []
    lines = content.splitlines()

    for node, capture_name in captures:
        symbol_name = content[node.start_byte:node.end_byte]
        # Determine kind from parent node
        parent = node.parent
        kind_str = "function"
        if parent:
            if parent.type == "class_definition" or parent.type == "class_declaration":
                kind_str = "class"
            elif parent.type == "method_definition" or parent.type == "method_declaration":
                kind_str = "method"
            elif parent.type == "function_definition" or parent.type == "function_declaration":
                kind_str = "function"

        if symbol_name != name:
            continue
        if kind != "any" and kind_str != kind:
            continue

        line_no = node.start_point[0] + 1
        snippet = lines[node.start_point[0]] if node.start_point[0] < len(lines) else ""
        results.append({
            "line": line_no,
            "kind": kind_str,
            "name": symbol_name,
            "snippet": snippet.strip()[:200],
        })
    return results


# Regex fallback simples
PY_PATTERNS = {
    "class": re.compile(r"^\s*class\s+(\w+)", re.MULTILINE),
    "function": re.compile(r"^\s*def\s+(\w+)", re.MULTILINE),
    "method": re.compile(r"^\s+(?:async\s+)?def\s+(\w+)", re.MULTILINE),
}

JS_PATTERNS = {
    "class": re.compile(r"^\s*class\s+(\w+)", re.MULTILINE),
    "function": re.compile(r"^\s*function\s+(\w+)|^\s*const\s+(\w+)\s*=\s*(?:async\s*)?\(", re.MULTILINE),
    "method": re.compile(r"^\s+(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{", re.MULTILINE),
}


def _find_symbols_regex(content: str, lang: str, name: str, kind: str) -> list[dict]:
    """Fallback regex quando tree-sitter indisponível."""
    patterns = PY_PATTERNS if lang == "python" else JS_PATTERNS
    if lang not in ("python", "javascript", "typescript"):
        return []

    results: list[dict] = []
    lines = content.splitlines()
    kinds_to_check = [kind] if kind != "any" else list(patterns.keys())

    for k in kinds_to_check:
        pat = patterns.get(k)
        if not pat:
            continue
        for m in pat.finditer(content):
            groups = m.groups()
            sym = next((g for g in groups if g), None)
            if not sym or sym != name:
                continue
            line_no = content[:m.start()].count("\n") + 1
            snippet = lines[line_no - 1] if line_no - 1 < len(lines) else ""
            results.append({
                "line": line_no,
                "kind": k,
                "name": sym,
                "snippet": snippet.strip()[:200],
            })
    return results


def register_all(registry) -> None:
    try:
        registry.register(SymbolLookupTool())
    except Exception as e:
        log.warning(f"SymbolLookupTool não registrada: {e}")
