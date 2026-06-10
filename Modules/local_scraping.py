"""
AVA Local Scraping Service — Port 3003
========================================
Searches, reads, and indexes local files on the user's machine.

Flow:
  1. Receives natural language query about a file
  2. Uses llama-server model to generate a search command
  3. Executes search in the Alpha execution directory
     (directly or via REST API client on the user's machine)
  4. Returns file content + path
  5. Multiple matches → returns list for user choice
  6. Already indexed → hash comparison → reindex if changed
  7. Indexed info saved to Memory API for semantic retrieval

REST API Client (optional):
  If CLIENT_API_URL is set, search commands are sent to a client
  running on the user's machine. The client executes the command
  in the Alpha directory and returns results.

  Expected client endpoints:
    POST /execute  {"command": "...", "working_dir": "..."} → {"stdout": "...", "exit_code": 0}
    POST /read-file {"file_path": "..."} → {"content": "...", "size": N, "modified": "..."}
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:2001")
LLAMA_TIMEOUT_S  = 30.0

MEMORY_API_URL   = os.environ.get("MEMORY_API_URL", "http://localhost:3001")
MEMORY_TIMEOUT_S = 5.0

# Alpha execution directory — base path for file searches
ALPHA_DIR = os.environ.get(
    "ALPHA_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)

# REST API client on the user's machine (optional)
# If set, commands are forwarded to this client instead of executed locally
CLIENT_API_URL = "http://localhost:3005"  # Set to None to disable and execute
CLIENT_TIMEOUT_S = 15.0

# Index storage
INDEX_DIR  = os.path.join(os.path.dirname(__file__), "local_scraping_index")
INDEX_FILE = os.path.join(INDEX_DIR, "index.json")

MAX_SEARCH_DEPTH  = 6
MAX_FILE_SIZE     = 10 * 1024 * 1024   # 10 MB
MAX_SEARCH_RESULTS = 50
MAX_CONTENT_LENGTH = 500_000            # 500 KB — truncate content beyond this

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [LOCAL-SCRAPING] %(message)s",
)
log = logging.getLogger("ava.local_scraping")

# ── File type support detection ───────────────────────────────────────────────

TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".csv",
    ".log", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".sh",
    ".bat", ".ps1", ".html", ".htm", ".css", ".xml", ".sql", ".r",
    ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", ".rb", ".php",
    ".swift", ".kt", ".scala", ".lua", ".pl", ".ex", ".exs", ".dart",
    ".tex", ".bib", ".rtf", ".env", ".gitignore", ".dockerfile",
    ".makefile", ".cmake", ".gradle", ".properties", ".resx",
}

HAS_PDF = HAS_DOCX = HAS_XLSX = False

try:
    from PyPDF2 import PdfReader
    HAS_PDF = True
except ImportError:
    pass

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    pass

try:
    import openpyxl
    HAS_XLSX = True
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic Models
# ══════════════════════════════════════════════════════════════════════════════

class ScrapeRequest(BaseModel):
    query: str = Field(..., description="Nome ou descrição do arquivo a buscar")
    search_path: Optional[str] = Field(None, description="Caminho base (padrão: ALPHA_DIR)")
    force_reindex: bool = Field(False, description="Forçar reindexação")
    session_id: Optional[str] = None

class ChooseRequest(BaseModel):
    query: str = Field(..., description="Query original da busca")
    file_path: str = Field(..., description="Caminho completo do arquivo escolhido")
    force_reindex: bool = Field(False, description="Forçar reindexação")
    session_id: Optional[str] = None

class FileMatch(BaseModel):
    file_path: str
    name: str
    extension: str
    size: str
    modified: str

class ScrapeResponse(BaseModel):
    content: Optional[str] = None
    file_path: Optional[str] = None
    was_reindexed: bool = False
    hash_match: bool = True
    multiple_matches: bool = False
    matches: Optional[list[FileMatch]] = None
    message: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# State & Lifespan
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AppState:
    llama_client:  httpx.AsyncClient = field(default=None)
    memory_client: httpx.AsyncClient = field(default=None)
    client_api:    httpx.AsyncClient = field(default=None)  # REST API client on user's machine
    index: dict    = field(default_factory=dict)            # file_path → index entry

state = AppState()


def _load_index_from_disk() -> dict:
    """Load the file index from disk."""
    if not os.path.exists(INDEX_FILE):
        return {}
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Falha ao carregar índice: {e}")
        return {}


def _save_index_to_disk(index: dict):
    """Persist the file index to disk."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    try:
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"Falha ao salvar índice: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"AVA Local Scraping Service iniciando — ALPHA_DIR={ALPHA_DIR}")

    # Load index from disk
    state.index = _load_index_from_disk()
    log.info(f"Índice carregado: {len(state.index)} arquivos indexados")

    # Initialize llama-server client
    state.llama_client = httpx.AsyncClient(
        base_url=LLAMA_SERVER_URL,
        timeout=httpx.Timeout(LLAMA_TIMEOUT_S),
        limits=httpx.Limits(max_keepalive_connections=2, max_connections=4),
    )

    # Initialize memory client
    state.memory_client = httpx.AsyncClient(
        base_url=MEMORY_API_URL,
        timeout=httpx.Timeout(MEMORY_TIMEOUT_S),
        limits=httpx.Limits(max_keepalive_connections=2, max_connections=4),
    )

    # Initialize REST API client (optional)
    if CLIENT_API_URL:
        state.client_api = httpx.AsyncClient(
            base_url=CLIENT_API_URL,
            timeout=httpx.Timeout(CLIENT_TIMEOUT_S),
            limits=httpx.Limits(max_keepalive_connections=2, max_connections=4),
        )
        log.info(f"REST API client configurado: {CLIENT_API_URL}")
    else:
        log.info("REST API client não configurado — usando execução local direta")

    # Probe llama-server
    try:
        r = await state.llama_client.get("/health", timeout=5.0)
        log.info(f"llama-server: {'OK' if r.status_code == 200 else f'{r.status_code}'}")
    except Exception:
        log.warning("llama-server não acessível — search command generation usará fallback heurístico")

    log.info("AVA Local Scraping Service pronto")
    yield

    # Shutdown
    for c in (state.llama_client, state.memory_client, state.client_api):
        if c:
            await c.aclose()
    # Persist index
    _save_index_to_disk(state.index)
    log.info("AVA Local Scraping Service encerrado")


app = FastAPI(
    title="AVA Local Scraping",
    version="1.0.0",
    description="Local file search, read & indexing service",
    lifespan=lifespan,
)


# ══════════════════════════════════════════════════════════════════════════════
# Search Command Generation (via model)
# ══════════════════════════════════════════════════════════════════════════════

_SEARCH_PROMPT = """<|system|>
You are a file search command generator. Given a user's natural language request about a local file, generate a single find command to locate the file on disk.

Rules:
1. Output ONLY the find command — no explanations, no markdown, no backticks
2. Always start with: find .
3. Use -iname for case-insensitive name matching with wildcard (*)
4. Always add -type f to restrict to files only
5. Use -maxdepth 6 to limit search depth
6. Use wildcards (*) around partial names
7. If the user mentions a file extension, include it in the pattern
8. Never use pipes (|), redirections (>), or command substitutions ($())

Examples:
"leia o arquivo relatório de vendas" → find . -maxdepth 6 -iname "*relatório*vendas*" -type f
"read the file report.pdf" → find . -maxdepth 6 -iname "*report*.pdf" -type f
"abra o documento notas" → find . -maxdepth 6 -iname "*notas*" -type f
"show me the config file" → find . -maxdepth 6 -iname "*config*" -type f
"leia o arquivo txt dados" → find . -maxdepth 6 -iname "*dados*.txt" -type f
"read the excel spreadsheet budget" → find . -maxdepth 6 -iname "*budget*.xlsx" -type f
<|end|>
<|user|>
{query}
<|end|>
<|assistant|"""


async def _generate_search_command(query: str) -> str:
    """
    Use the model to generate a find command from a natural language query.
    Falls back to heuristic extraction if the model is unavailable.
    """
    prompt = _SEARCH_PROMPT.format(query=query)

    try:
        r = await state.llama_client.post(
            "/completion",
            json={
                "prompt":       prompt,
                "max_tokens":   80,
                "temperature":  0.1,
                "top_p":        0.9,
                "stream":       False,
                "cache_prompt": True,
            },
        )
        r.raise_for_status()
        cmd = r.json().get("content", "").strip()

        # Validate the generated command
        if _validate_find_command(cmd):
            log.info(f"Search command gerado pelo modelo: {cmd}")
            return cmd
        else:
            log.warning(f"Search command inválido do modelo: {cmd} — usando fallback")
    except Exception as e:
        log.warning(f"Modelo falhou ao gerar search command: {e} — usando fallback")

    # Fallback: heuristic extraction
    return _heuristic_search_command(query)


def _validate_find_command(cmd: str) -> bool:
    """Validate that a command is a safe find command."""
    cmd = cmd.strip()
    if not cmd.startswith("find "):
        return False
    dangerous = ["|", ">", "<", "$", "`", "&", ";", "\n", "rm ", "delete", "exec"]
    if any(d in cmd for d in dangerous):
        return False
    return True


def _heuristic_search_command(query: str) -> str:
    """
    Fallback: extract search terms heuristically and build a find command.
    """
    # Remove common Portuguese/English verbs and articles
    stop_words = {
        "leia", "ler", "abra", "abrir", "mostre", "mostrar", "encontre",
        "buscar", "procure", "carregue", "read", "open", "show", "find",
        "load", "get", "fetch", "display", "the", "a", "o", "os", "as",
        "arquivo", "arquivos", "file", "files", "documento", "document",
        "de", "do", "da", "dos", "das", "em", "no", "na", "called",
        "chamado", "chamada", "nome", "named", "conteúdo", "content",
        "que", "that", "with", "com", "para", "for", "and", "e",
    }

    # Extract extension hints
    ext_pattern = re.compile(r"\.(pdf|txt|csv|xlsx|docx|json|md|py|js|ts|html|xml|yaml|yml|log|rtf|tex)\b", re.I)
    ext_match = ext_pattern.search(query)
    extension = ext_match.group(1).lower() if ext_match else None

    # Clean and tokenize
    words = re.findall(r"[a-zA-ZÀ-ÿ0-9_]+", query)
    keywords = [w for w in words if w.lower() not in stop_words and len(w) > 1]

    if not keywords:
        keywords = [w for w in words if len(w) > 1]
    if not keywords:
        return "find . -maxdepth 6 -type f"

    # Build pattern
    pattern = "*".join(keywords)
    if extension:
        pattern = f"*{pattern}*.{extension}"
    else:
        pattern = f"*{pattern}*"

    cmd = f'find . -maxdepth 6 -iname "{pattern}" -type f'
    log.info(f"Search command heurístico: {cmd}")
    return cmd


def _extract_pattern_from_find(cmd: str) -> str:
    """Extract the -iname glob pattern from a find command."""
    match = re.search(r'-iname\s+"([^"]+)"', cmd)
    if match:
        return match.group(1)
    match = re.search(r"-iname\s+'([^']+)'", cmd)
    if match:
        return match.group(1)
    # Fallback: try to get any quoted pattern
    match = re.search(r'"(\*[^"]+\*)"', cmd)
    if match:
        return match.group(1)
    return "*"


# ══════════════════════════════════════════════════════════════════════════════
# File Search Execution
# ══════════════════════════════════════════════════════════════════════════════

async def _execute_search(cmd: str, search_path: str) -> list[str]:
    """
    Execute a search command and return list of matching file paths.
    Uses REST API client if configured, otherwise executes locally.
    """
    if state.client_api:
        return await _execute_search_via_client(cmd, search_path)
    else:
        return await _execute_search_locally(cmd, search_path)


async def _execute_search_via_client(cmd: str, search_path: str) -> list[str]:
    """
    Busca via client no host.
    O client roda no BASE_DIR do host — ignora o search_path do container.
    Retorna caminhos que o client consegue ler depois.
    """
    try:
        r = await state.client_api.post(
            "/execute",
            json={"command": cmd, "working_dir": search_path},
        )
        r.raise_for_status()
        data = r.json()
        if data.get("exit_code", 1) != 0:
            log.warning(f"Client command failed: {data.get('stderr', '')}")
            return []

        stdout = data.get("stdout", "")
        lines = [line.strip() for line in stdout.strip().split("\n") if line.strip()]

        # Os paths são relativos ao BASE_DIR do host
        # Transforma ./file.txt em paths que o client entende para read-file
        results = []
        for line in lines:
            if line.startswith("./"):
                line = line[2:]
            if line:
                results.append(line)

        log.info(f"Client encontrou {len(results)} arquivo(s)")
        return results

    except httpx.HTTPStatusError as e:
        log.warning(f"Client HTTP {e.response.status_code}: {e.response.text[:200]}")
        return await _execute_search_locally(cmd, search_path)
    except Exception as e:
        log.warning(f"Client falhou: {e}")
        return await _execute_search_locally(cmd, search_path)



async def _execute_search_locally(cmd: str, search_path: str) -> list[str]:
    """Execute search command locally via subprocess."""
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15.0,
            cwd=search_path,
        )
        if proc.returncode != 0:
            log.warning(f"find command exit {proc.returncode}: {proc.stderr[:200]}")
            return []
        paths = [line.strip() for line in proc.stdout.strip().split("\n") if line.strip()]
        return _filter_valid_paths(paths, search_path)
    except subprocess.TimeoutExpired:
        log.warning("find command timeout (15s)")
        return []
    except Exception as e:
        log.warning(f"find command falhou: {e}")
        return []


def _filter_valid_paths(paths: list[str], search_path: str) -> list[str]:
    """Filter paths to only include real files within the search directory."""
    base = Path(search_path).resolve()
    valid = []
    for p in paths[:MAX_SEARCH_RESULTS]:
        try:
            fp = Path(p)
            if not fp.is_absolute():
                fp = base / p
            fp = fp.resolve()
            if fp.is_file() and str(fp).startswith(str(base)):
                valid.append(str(fp))
        except Exception:
            continue
    return valid


async def _search_via_pathlib(pattern: str, search_path: str) -> list[str]:
    """
    Cross-platform file search using pathlib (fallback for Windows or
    when find command fails).
    """
    base = Path(search_path).resolve()
    results = []
    try:
        for p in base.rglob(pattern):
            if p.is_file():
                try:
                    depth = len(p.relative_to(base).parts) - 1
                    if depth <= MAX_SEARCH_DEPTH:
                        if p.stat().st_size <= MAX_FILE_SIZE:
                            results.append(str(p.resolve()))
                            if len(results) >= MAX_SEARCH_RESULTS:
                                break
                except (ValueError, OSError):
                    continue
    except PermissionError:
        pass
    return results


# ══════════════════════════════════════════════════════════════════════════════
# File Reading
# ══════════════════════════════════════════════════════════════════════════════


async def _read_file(file_path: str) -> tuple[str, bool]:
    """
    Lê arquivo. Se client disponível, pede pro host ler.
    Senão, lê localmente.
    """
    if state.client_api:
        try:
            r = await state.client_api.post(
                "/read-file",
                json={"file_path": file_path},
            )
            r.raise_for_status()
            data = r.json()
            return data["content"], True
        except httpx.HTTPStatusError as e:
            log.warning(f"Client read-file HTTP {e.response.status_code}")
        except Exception as e:
            log.warning(f"Client read-file falhou: {e}")

    return _read_file_locally(file_path)




def _read_file_locally(file_path: str) -> tuple[str, bool]:
    """Read file content from local filesystem."""
    path = Path(file_path)

    if not path.exists():
        return f"[Arquivo não encontrado: {file_path}]", False
    if not path.is_file():
        return f"[Caminho não é um arquivo: {file_path}]", False
    if path.stat().st_size > MAX_FILE_SIZE:
        return f"[Arquivo muito grande: {path.stat().st_size} bytes (máx: {MAX_FILE_SIZE})]", False

    ext = path.suffix.lower()

    # ── Plain text files ──
    if ext in TEXT_EXTENSIONS or ext == "":
        return _read_text_file(path)

    # ── PDF ──
    if ext == ".pdf":
        if HAS_PDF:
            return _read_pdf_file(path)
        return "[PDF: instale PyPDF2 — pip install PyPDF2]", False

    # ── DOCX ──
    if ext == ".docx":
        if HAS_DOCX:
            return _read_docx_file(path)
        return "[DOCX: instale python-docx — pip install python-docx]", False

    # ── XLSX ──
    if ext == ".xlsx":
        if HAS_XLSX:
            return _read_xlsx_file(path)
        return "[XLSX: instale openpyxl — pip install openpyxl]", False

    # ── CSV ──
    if ext == ".csv":
        return _read_csv_file(path)

    # ── JSON ──
    if ext == ".json":
        return _read_json_file(path)

    # ── Try as plain text for unknown extensions ──
    return _read_text_file(path, fallback=True)


def _read_text_file(path: Path, fallback: bool = False) -> tuple[str, bool]:
    """Read a plain text file."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(MAX_CONTENT_LENGTH)
        # Check for binary content
        if "\x00" in content[:4096]:
            return f"[Arquivo binário: {path.suffix or 'sem extensão'} — não é possível exibir como texto]", False
        if len(content) == MAX_CONTENT_LENGTH:
            content += "\n\n[... conteúdo truncado ...]"
        return content, True
    except Exception as e:
        if fallback:
            return f"[Não foi possível ler o arquivo: {e}]", False
        return str(e), False


def _read_pdf_file(path: Path) -> tuple[str, bool]:
    """Read text from a PDF file using PyPDF2."""
    try:
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append(f"--- Página {i + 1} ---\n{text}")
        content = "\n\n".join(pages)
        if len(content) > MAX_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH] + "\n\n[... conteúdo truncado ...]"
        return content, True
    except Exception as e:
        return f"[Erro ao ler PDF: {e}]", False


def _read_docx_file(path: Path) -> tuple[str, bool]:
    """Read text from a DOCX file using python-docx."""
    try:
        doc = DocxDocument(str(path))
        content = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if len(content) > MAX_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH] + "\n\n[... conteúdo truncado ...]"
        return content, True
    except Exception as e:
        return f"[Erro ao ler DOCX: {e}]", False


def _read_xlsx_file(path: Path) -> tuple[str, bool]:
    """Read text from an XLSX file using openpyxl."""
    try:
        wb = openpyxl.load_workbook(str(path), read_only=True)
        parts = []
        for ws in wb.worksheets:
            parts.append(f"--- Aba: {ws.title} ---")
            for row in ws.iter_rows(values_only=True):
                line = "\t".join(str(c) if c is not None else "" for c in row)
                parts.append(line)
            if sum(len(p) for p in parts) > MAX_CONTENT_LENGTH:
                break
        wb.close()
        content = "\n".join(parts)
        if len(content) > MAX_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH] + "\n\n[... conteúdo truncado ...]"
        return content, True
    except Exception as e:
        return f"[Erro ao ler XLSX: {e}]", False


def _read_csv_file(path: Path) -> tuple[str, bool]:
    """Read a CSV file as formatted text."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            lines = []
            for row in reader:
                lines.append("\t".join(row))
                if sum(len(l) for l in lines) > MAX_CONTENT_LENGTH:
                    break
            content = "\n".join(lines)
            if len(content) > MAX_CONTENT_LENGTH:
                content = content[:MAX_CONTENT_LENGTH] + "\n\n[... conteúdo truncado ...]"
            return content, True
    except Exception as e:
        return f"[Erro ao ler CSV: {e}]", False


def _read_json_file(path: Path) -> tuple[str, bool]:
    """Read and pretty-print a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        content = json.dumps(data, indent=2, ensure_ascii=False)
        if len(content) > MAX_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH] + "\n\n[... conteúdo truncado ...]"
        return content, True
    except Exception as e:
        return f"[Erro ao ler JSON: {e}]", False


# ══════════════════════════════════════════════════════════════════════════════
# Index Management (hash-based change detection)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return ""


def _get_file_metadata(file_path: str) -> dict:
    """Get file metadata for indexing."""
    p = Path(file_path)
    try:
        stat = p.stat()
        return {
            "size":     stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }
    except Exception:
        return {"size": 0, "modified": ""}


def _check_index(file_path: str) -> Optional[dict]:
    """Check if a file is in the index and return its entry."""
    return state.index.get(file_path)


def _update_index(file_path: str, force: bool = False) -> tuple[bool, bool]:
    """
    Update the index for a file.
    Returns (was_reindexed, hash_match).
      was_reindexed = True if the file was (re)indexed in this call
      hash_match    = True if the hash matched the stored hash
    """
    current_hash = _compute_file_hash(file_path)
    entry = state.index.get(file_path)

    if not force and entry and entry.get("hash") == current_hash:
        # File unchanged — no reindex needed
        return False, True

    # File changed or new — update index
    meta = _get_file_metadata(file_path)
    state.index[file_path] = {
        "file_id":     entry.get("file_id", str(uuid.uuid4())) if entry else str(uuid.uuid4()),
        "hash":        current_hash,
        "size":        meta["size"],
        "modified":    meta["modified"],
        "indexed_at":  datetime.now().isoformat(),
        "extension":   Path(file_path).suffix.lower(),
    }
    _save_index_to_disk(state.index)

    hash_match = (entry is not None and entry.get("hash") == current_hash)
    return True, hash_match


# ══════════════════════════════════════════════════════════════════════════════
# Memory Integration
# ══════════════════════════════════════════════════════════════════════════════

async def _save_to_memory(file_path: str, content: str, hash_match: bool = True):
    """
    Save indexed file content to the Memory API for semantic retrieval.
    """
    if not content or not file_path:
        return

    fname = Path(file_path).name

    try:
        await state.memory_client.post(
            "/write",
            json={
                "text":       f"[ARQUIVO INDEXADO] {file_path}: {content}",
                "source":     "local_scraping",
                "confidence": 1.0 if hash_match else 0.9,
            },
        )
        log.info(f"Arquivo '{fname}' salvo na memória (hash_match={hash_match})")
    except Exception as e:
        log.warning(f"Falha ao salvar arquivo na memória: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Core Scrape Logic
# ══════════════════════════════════════════════════════════════════════════════
def _format_file_matches(file_paths: list[str]) -> list[FileMatch]:
    """Formata paths em FileMatch objects."""
    matches = []
    for fp in file_paths:
        p = Path(fp)
        name = p.name

        # Tenta obter metadata — pode falhar se for path do host
        try:
            stat = p.stat()
            size_str = _human_readable_size(stat.st_size)
            modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
        except Exception:
            size_str = "?"
            modified = "?"

        matches.append(FileMatch(
            file_path=fp,
            name=name,
            extension=p.suffix.lower(),
            size=size_str,
            modified=modified,
        ))
    return matches


def _human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"



async def _do_scrape(query: str, search_path: str, force_reindex: bool) -> ScrapeResponse:
    # Step 1: Generate search command
    cmd = await _generate_search_command(query)

    # Step 2: Execute search
    file_paths = await _execute_search(cmd, search_path)

    # Fallback: pathlib se find retornou nada E não tem client
    if not file_paths and not state.client_api:
        pattern = _extract_pattern_from_find(cmd)
        log.info(f"find retornou 0 — tentando pathlib com '{pattern}'")
        file_paths = await _search_via_pathlib(pattern, search_path)

    # Fallback: busca fuzzy
    if not file_paths:
        file_paths = await _search_fuzzy(query, search_path)

    # Sem resultados
    if not file_paths:
        return ScrapeResponse(
            content=None,
            message=f"Nenhum arquivo encontrado para: '{query}'",
        )

    # Múltiplos matches
    if len(file_paths) > 1:
        matches = _format_file_matches(file_paths)
        return ScrapeResponse(
            multiple_matches=True,
            matches=matches,
            message=f"Encontrados {len(matches)} arquivos. Escolha qual deseja ler.",
        )

    # Arquivo único — lê e indexa
    file_path = file_paths[0]
    return await _read_and_index(file_path, force_reindex)




async def _read_and_index(file_path: str, force_reindex: bool) -> ScrapeResponse:
    """
    Lê arquivo, verifica/_atualiza índice, salva na memória.
    file_path pode ser um caminho relativo (do client) ou absoluto (local).
    """
    # Lê conteúdo
    content, success = await _read_file(file_path)
    if not success:
        return ScrapeResponse(
            content=content,
            file_path=file_path,
            was_reindexed=False,
            hash_match=False,
            message=f"Erro ao ler arquivo: {content}",
        )

    # Hash do conteúdo
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    # Verifica índice
    was_reindexed = False
    hash_match = True

    existing = state.index.get(file_path)
    if existing:
        if existing.get("hash") == content_hash and not force_reindex:
            hash_match = True
            was_reindexed = False
        else:
            was_reindexed = True
            hash_match = existing.get("hash") == content_hash
    else:
        was_reindexed = False

    # Atualiza índice
    state.index[file_path] = {
        "file_id":    existing.get("file_id", str(uuid.uuid4())) if existing else str(uuid.uuid4()),
        "hash":       content_hash,
        "size":       len(content),
        "modified":   datetime.now().isoformat(),
        "indexed_at": datetime.now().isoformat(),
        "extension":  Path(file_path).suffix.lower(),
    }
    _save_index_to_disk(state.index)

    # Salva na memória (conteúdo completo)
    try:
        await state.memory_client.post(
            "/indexed-file/write",
            json={
                "file_path":   file_path,
                "file_name":   Path(file_path).name,
                "extension":   Path(file_path).suffix.lower(),
                "content":     content,
                "file_hash":   content_hash,
                "size":        len(content),
                "modified":    datetime.now().isoformat(),
                "source":      "local_scraping",
                "confidence":  1.0 if hash_match else 0.9,
                "force_reindex": force_reindex,
            },
        )
        log.info(f"Arquivo '{file_path}' salvo na memória")
    except Exception as e:
        log.warning(f"Falha ao salvar na memória: {e}")

    return ScrapeResponse(
        content=content,
        file_path=file_path,
        was_reindexed=was_reindexed,
        hash_match=hash_match,
    )

# ══════════════════════════════════════════════════════════════════════════════
# API Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/scrape", response_model=ScrapeResponse)
async def scrape(req: ScrapeRequest):
    """
    Search and read a local file on the user's machine.

    Flow:
      1. Model generates a search command from the query
      2. Command is executed (directly or via REST API client)
      3. Returns file content + path
      4. If multiple matches → returns list for user choice
      5. If file already indexed → hash comparison → reindex if changed
      6. Indexed info saved to memory
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query vazio")

    search_path = req.search_path or ALPHA_DIR
    if not os.path.isdir(search_path):
        raise HTTPException(status_code=400, detail=f"Caminho de busca inválido: {search_path}")

    t0 = time.perf_counter()
    result = await _do_scrape(query, search_path, req.force_reindex)
    latency = round((time.perf_counter() - t0) * 1000, 2)

    if result.multiple_matches:
        log.info(f"Scrape em {latency}ms — {len(result.matches)} matches para: {query[:60]}")
    elif result.content:
        log.info(f"Scrape em {latency}ms — arquivo lido: {result.file_path} (reindexed={result.was_reindexed}, hash_match={result.hash_match})")
    else:
        log.info(f"Scrape em {latency}ms — sem resultados para: {query[:60]}")

    return result


@app.post("/choose", response_model=ScrapeResponse)
async def choose(req: ChooseRequest):
    """
    Choose a specific file when multiple matches were found.
    Reads the chosen file, checks/updates index, saves to memory.
    """
    file_path = req.file_path.strip()
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path vazio")

    # Validate the file exists and is within allowed directory
    search_path = ALPHA_DIR
    try:
        resolved = Path(file_path).resolve()
        base = Path(search_path).resolve()
        if not str(resolved).startswith(str(base)):
            raise HTTPException(status_code=403, detail="Caminho fora do diretório permitido")
        if not resolved.is_file():
            raise HTTPException(status_code=404, detail=f"Arquivo não encontrado: {file_path}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Caminho inválido: {e}")

    t0 = time.perf_counter()
    result = await _read_and_index(file_path, req.force_reindex)
    latency = round((time.perf_counter() - t0) * 1000, 2)

    log.info(f"Choose em {latency}ms — {file_path} (reindexed={result.was_reindexed}, hash_match={result.hash_match})")
    return result


@app.get("/status")
async def status():
    """Health check and service status."""
    llama_ok = False
    memory_ok = False
    client_ok = None

    try:
        r = await state.llama_client.get("/health", timeout=2.0)
        llama_ok = r.status_code == 200
    except Exception:
        pass

    try:
        r = await state.memory_client.get("/status", timeout=2.0)
        memory_ok = r.status_code == 200
    except Exception:
        pass

    if state.client_api:
        try:
            r = await state.client_api.get("/status", timeout=2.0)
            client_ok = r.status_code == 200
        except Exception:
            client_ok = False

    return {
        "service":         "AVA Local Scraping",
        "status":          "ok",
        "alpha_dir":       ALPHA_DIR,
        "client_api_url":  CLIENT_API_URL,
        "client_api_ok":   client_ok,
        "llama_server":    llama_ok,
        "memory_api":      memory_ok,
        "indexed_files":   len(state.index),
        "supported_types": {
            "text": True,
            "pdf":  HAS_PDF,
            "docx": HAS_DOCX,
            "xlsx": HAS_XLSX,
        },
    }


@app.get("/indexed")
async def list_indexed(file_path: Optional[str] = None):
    """
    List indexed files, or check if a specific file is indexed.
    Optionally filter by file_path to check indexing status and hash.
    """
    if file_path:
        entry = state.index.get(file_path)
        if entry:
            # Check if file still exists and compute current hash
            current_hash = _compute_file_hash(file_path)
            meta = _get_file_metadata(file_path)
            return {
                "file_path":   file_path,
                "indexed":     True,
                "hash_match":  current_hash == entry.get("hash"),
                "current_hash": current_hash,
                "indexed_hash": entry.get("hash"),
                "index_entry":  entry,
                "current_meta": meta,
            }
        return {"file_path": file_path, "indexed": False}

    # List all indexed files
    files = []
    for fp, entry in state.index.items():
        files.append({
            "file_path":  fp,
            "file_id":    entry.get("file_id"),
            "extension":  entry.get("extension"),
            "size":       entry.get("size"),
            "modified":   entry.get("modified"),
            "indexed_at": entry.get("indexed_at"),
        })
    return {"total": len(files), "files": files}


@app.delete("/index/{file_id}")
async def delete_index(file_id: str):
    """Remove a specific file from the index by its file_id."""
    to_remove = None
    for fp, entry in state.index.items():
        if entry.get("file_id") == file_id:
            to_remove = fp
            break

    if not to_remove:
        raise HTTPException(status_code=404, detail=f"file_id '{file_id}' não encontrado no índice")

    del state.index[to_remove]
    _save_index_to_disk(state.index)
    log.info(f"Índice removido para: {to_remove}")
    return {"removed": True, "file_path": to_remove, "file_id": file_id}


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3003, log_level="info")
    
    
    