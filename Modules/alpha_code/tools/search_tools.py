"""
Alpha-code — Search Tools
=========================
search_code: busca em código via ripgrep subprocess.
semantic_search: busca semântica via embeddings onnx:2002 + rerank.

Não chama scraping_client — roda ripgrep localmente (mais rápido para buscas).
Se o projeto alvo está no host do scraping_client (porta 3005), a busca roda
neste mesmo host. Se o alpha_code está em outro container, precisa de volume
montado ou de expor search via scraping_client (futuro).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import httpx

from alpha_code.schemas import StepContext, ToolResult
from alpha_code.tools.registry import Tool, schema_function, param

log = logging.getLogger("ava.alpha_code.search_tools")


# ── Configuração ────────────────────────────────────────────────────────────

# Diretório raiz do projeto alvo (default: mesmo do scraping_client via env).
PROJECT_ROOT = Path(os.environ.get("ALPHA_PROJECT_ROOT", "/home/z/my-project")).resolve()

# Limites
MAX_MATCHES = 50
MAX_FILE_SIZE_KB = 500
DEFAULT_FILE_TYPES = [
    "py", "js", "ts", "tsx", "jsx", "go", "rs", "java", "kt", "rb",
    "php", "c", "cpp", "h", "hpp", "cs", "swift", "sql", "sh", "yaml", "yml",
    "json", "toml", "md",
]

# Ignora estes diretórios (similar ao .gitignore padrão)
DEFAULT_GLOBS = [
    "*/.git/*", "*/node_modules/*", "*/__pycache__/*", "*/.venv/*",
    "*/venv/*", "*/dist/*", "*/build/*", "*/.tox/*", "*/.eggs/*",
    "*/.mypy_cache/*", "*/.pytest_cache/*", "*/.ruff_cache/*",
]

ONNX_EMBED_URL = os.environ.get("ALPHA_ONNX_URL", "http://localhost:2002")


def _check_rg() -> str:
    """Retorna path do ripgrep ou raise."""
    rg = shutil.which("rg")
    if not rg:
        raise RuntimeError(
            "ripgrep (rg) não encontrado. Instale: apt-get install ripgrep ou brew install ripgrep"
        )
    return rg


# ════════════════════════════════════════════════════════════════════════════
# Tool: search_code (ripgrep)
# ════════════════════════════════════════════════════════════════════════════

class SearchCodeTool(Tool):
    name = "search_code"
    description = (
        "Busca por padrão (regex ou literal) em arquivos do projeto usando ripgrep. "
        "Retorna matches com arquivo:linha:conteúdo. "
        "Ex: search_code({\"pattern\": \"def authenticate\", \"file_type\": \"py\"})"
    )

    def schema(self) -> dict:
        return schema_function(
            name=self.name,
            description=self.description,
            properties={
                "pattern": param("string", "Padrão regex ou literal a buscar"),
                "file_type": param(
                    "string",
                    "Extensão de arquivo para filtrar (ex: 'py', 'js', 'ts'). Default: todos os arquivos de código.",
                    default="",
                ),
                "path": param(
                    "string",
                    "Subdiretório relativo ao projeto onde buscar. Default: raiz.",
                    default=".",
                ),
                "max_matches": param("integer", "Máximo de matches. Default 50", default=50),
                "literal": param("boolean", "Se true, trata pattern como literal (sem regex). Default false", default=False),
            },
            required=["pattern"],
        )

    async def execute(self, args: dict, ctx: StepContext) -> ToolResult:
        pattern = args.get("pattern", "").strip()
        if not pattern:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error="pattern vazio",
            )

        file_type = (args.get("file_type") or "").strip().lstrip(".")
        path = args.get("path", ".")
        max_matches = min(args.get("max_matches", 50), MAX_MATCHES)
        literal = args.get("literal", False)

        target = (PROJECT_ROOT / path).resolve()
        try:
            target.relative_to(PROJECT_ROOT)
        except ValueError:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error=f"Path fora do projeto: {path}",
            )
        if not target.exists():
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error=f"Path não existe: {path}",
            )

        rg = _check_rg()
        cmd = [
            rg, "--json",
            "--max-count", str(max_matches),
            "--max-filesize", f"{MAX_FILE_SIZE_KB}K",
        ]
        for g in DEFAULT_GLOBS:
            cmd += ["--glob", f"!{g}"]
        if file_type:
            cmd += ["--type", file_type]
        if literal:
            cmd += ["--fixed-strings"]
        cmd += [pattern, str(target)]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        except asyncio.TimeoutExpired:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error="ripgrep timeout (30s)",
            )
        except FileNotFoundError:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error="ripgrep não encontrado",
            )

        # parse --json output
        matches = []
        try:
            for line in stdout.decode("utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "match":
                    data = obj.get("data", {})
                    path_str = data.get("path", {}).get("text", "")
                    # tenta tornar path relativo ao PROJECT_ROOT
                    try:
                        rel = str(Path(path_str).resolve().relative_to(PROJECT_ROOT))
                    except Exception:
                        rel = path_str
                    line_no = data.get("line_number", 0)
                    content = data.get("lines", {}).get("text", "").rstrip("\n")
                    matches.append({
                        "path": rel,
                        "line": line_no,
                        "content": content[:300],
                    })
        except Exception as e:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error=f"Erro parse ripgrep output: {e}",
            )

        if not matches:
            return ToolResult(
                tool_call_id="",
                tool_name=self.name,
                success=True,
                output="Nenhum match encontrado.",
                data={"count": 0},
            )

        # formata output para LLM
        lines = [f"{m['path']}:{m['line']}:  {m['content']}" for m in matches[:max_matches]]
        out = "\n".join(lines)
        if len(matches) > max_matches:
            out += f"\n... ({len(matches) - max_matches} matches omitidos)"
        return ToolResult(
            tool_call_id="",
            tool_name=self.name,
            success=True,
            output=out,
            data={"count": len(matches), "truncated": len(matches) >= max_matches},
        )


# ════════════════════════════════════════════════════════════════════════════
# Tool: semantic_search (embeddings + rerank via onnx:2002)
# ════════════════════════════════════════════════════════════════════════════

class SemanticSearchTool(Tool):
    name = "semantic_search"
    description = (
        "Busca semântica em código: encontra arquivos/símbolos por significado, não por string exata. "
        "Ex: semantic_search({\"query\": \"onde é implementada a autenticação JWT\"}). "
        "Usa embeddings multilingual-e5-small + cross-encoder rerank via onnx_serving:2002."
    )

    def schema(self) -> dict:
        return schema_function(
            name=self.name,
            description=self.description,
            properties={
                "query": param("string", "Pergunta em linguagem natural sobre o que buscar"),
                "path": param("string", "Subdiretório onde buscar. Default '.'", default="."),
                "max_results": param("integer", "Número máximo de resultados. Default 10", default=10),
            },
            required=["query"],
        )

    async def execute(self, args: dict, ctx: StepContext) -> ToolResult:
        query = args.get("query", "").strip()
        if not query:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error="query vazio",
            )

        path = args.get("path", ".")
        max_results = min(args.get("max_results", 10), 20)

        target = (PROJECT_ROOT / path).resolve()
        try:
            target.relative_to(PROJECT_ROOT)
        except ValueError:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error=f"Path fora do projeto: {path}",
            )

        # 1. Coleta arquivos de código
        files: list[Path] = []
        for ext in DEFAULT_FILE_TYPES:
            files.extend(target.rglob(f"*.{ext}"))
        # filter out glob exclusions
        excluded_parts = {".git", "node_modules", "__pycache__", ".venv", "venv",
                           "dist", "build", ".tox", ".eggs", ".mypy_cache",
                           ".pytest_cache", ".ruff_cache"}
        filtered = []
        for f in files:
            try:
                rel_parts = f.resolve().relative_to(PROJECT_ROOT).parts
            except Exception:
                continue
            if any(part in excluded_parts for part in rel_parts):
                continue
            if f.stat().st_size > MAX_FILE_SIZE_KB * 1024:
                continue
            filtered.append(f)
            if len(filtered) >= 300:
                break
        files = filtered

        if not files:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=True,
                output="Nenhum arquivo de código encontrado.",
                data={"count": 0},
            )

        # 2. Lê conteúdo (limita tamanho por arquivo p/ não estourar embedding)
        docs: list[tuple[str, str]] = []
        for f in files:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            # quebra em chunks de ~1500 chars
            chunks = [content[i:i+1500] for i in range(0, len(content), 1500)]
            try:
                rel = str(f.resolve().relative_to(PROJECT_ROOT))
            except Exception:
                rel = str(f)
            for i, chunk in enumerate(chunks):
                docs.append((f"{rel}#{i}", chunk))

        if not docs:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=True,
                output="Arquivos vazios.",
                data={"count": 0},
            )

        try:
            async with httpx.AsyncClient(base_url=ONNX_EMBED_URL, timeout=60.0) as client:
                # 3. Embed query
                r_q = await client.post("/v1/embed", json={"texts": [f"query: {query}"]})
                r_q.raise_for_status()
                query_emb = r_q.json()["embeddings"][0]

                # 4. Embed passages (batch 64)
                passages = [f"passage: {d[1]}" for d in docs]
                all_embs = []
                for i in range(0, len(passages), 64):
                    batch = passages[i:i+64]
                    r_p = await client.post("/v1/embed", json={"texts": batch})
                    r_p.raise_for_status()
                    all_embs.extend(r_p.json()["embeddings"])

                # 5. Cosine similarity (e5 já é normalizado, mas por segurança normalizamos)
                import numpy as np
                q = np.array(query_emb, dtype=np.float32)
                q = q / (np.linalg.norm(q) + 1e-9)
                mat = np.array(all_embs, dtype=np.float32)
                norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
                mat = mat / norms
                sims = (mat @ q).tolist()

                # top 25 para reranking
                top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:25]

                # 6. Rerank com cross-encoder
                candidates = [docs[i][1] for i in top_idx]
                r_r = await client.post("/v1/rerank", json={"query": query, "candidates": candidates, "top_k": max_results})
                if r_r.status_code == 200:
                    ranked = r_r.json().get("ranked", [])
                else:
                    # sem reranker → usa só sims
                    ranked = [{"index": i, "score": sims[i], "text": candidates[i]} for i in top_idx[:max_results]]

                # 7. Monta resultado
                out_lines = []
                for r in ranked[:max_results]:
                    orig_idx = top_idx[r["index"]] if "index" in r else r.get("orig_idx", 0)
                    doc_path, doc_text = docs[orig_idx]
                    snippet = doc_text[:400].replace("\n", " ")
                    out_lines.append(f"[score={r.get('score', 0):.3f}] {doc_path}\n  {snippet}")
                out = "\n\n".join(out_lines)
                return ToolResult(
                    tool_call_id="",
                    tool_name=self.name,
                    success=True,
                    output=out,
                    data={"count": len(ranked), "files_indexed": len(files)},
                )

        except httpx.HTTPError as e:
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error=f"onnx_serving indisponível: {e}",
            )
        except Exception as e:
            log.exception("semantic_search crash")
            return ToolResult(
                tool_call_id="", tool_name=self.name, success=False,
                error=f"{type(e).__name__}: {e}",
            )


def register_all(registry) -> None:
    registry.register(SearchCodeTool())
    try:
        registry.register(SemanticSearchTool())
    except Exception as e:
        log.warning(f"SemanticSearchTool não registrada: {e}")
