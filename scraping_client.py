"""
Alpha Host Client — Port 3005
==============================
Runs on the HOST machine.
Uses the CURRENT WORKING DIRECTORY for everything.
Ignores container paths — always reads from where it's running.

Usage:
    cd /anywhere/with/your/files
    python alpha-client.py
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import os
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Configuration ──────────────────────────────────────────────────────────────

CLIENT_TOKEN = os.environ.get("CLIENT_TOKEN", "")
MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_CMD_TIME  = 30.0

# ── Configuração do repo-sombra de versionamento (auto-commit de cada edição) ──
# NÃO usa o .git real do usuário (se existir) — GIT_DIR aponta pra uma pasta
# separada dentro de BASE_DIR/.ava, com GIT_WORK_TREE = BASE_DIR. Isso dá
# histórico/rollback de tudo que o agente faz, sem sujar (ou depender d)o
# repositório de trabalho do próprio usuário.
AVA_GIT_DIR      = None  # setado após BASE_DIR ser resolvido, ver abaixo
GIT_AUTHOR_NAME  = os.environ.get("AVA_GIT_NAME",  "AVA Alpha Code")
GIT_AUTHOR_EMAIL = os.environ.get("AVA_GIT_EMAIL", "alpha-code@ava.local")
GIT_TIMEOUT      = 10.0

# ── Configuração do fuzzy match no str-replace ──
FUZZY_THRESHOLD_DEFAULT = 0.85
# Acima disso, pula a etapa de fuzzy (custo O(linhas × tamanho do old_str))
# e vai direto pra normalized/exact — protege contra arquivos gigantes.
FUZZY_MAX_CONTENT_LINES = 20_000

# O ÚNICO caminho que importa — onde o script está rodando
BASE_DIR = os.path.abspath(os.getcwd())
ALPHA_DIR = os.path.dirname(os.path.abspath(__file__))
# NOTA: o nome "scraping_cwd.dll" é enganoso — é um arquivo de TEXTO puro
# (só contém o BASE_DIR), não uma DLL de verdade. Mantido como estava para
# não quebrar quem já lê esse caminho (ex.: orchestrator.py), mas o
# makedirs abaixo evita um crash no boot se "resource/" ainda não existir —
# antes isso derrubava o processo inteiro com FileNotFoundError antes mesmo
# do logging ser configurado.
_resource_dir = os.path.join(ALPHA_DIR, "resource")
os.makedirs(_resource_dir, exist_ok=True)
with open(os.path.join(_resource_dir, "scraping_cwd.dll"), "w") as f:
    f.write(BASE_DIR)
    
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ALPHA-CLIENT] %(message)s",
)
log = logging.getLogger("alpha_client")

log.info(f"BASE_DIR:  {BASE_DIR}")
log.info(f"Platform:  {platform.system()} {platform.release()}")
log.info(f"Home:      {Path.home()}")
log.info(f"CWD:       {os.getcwd()}")

AVA_GIT_DIR = os.path.join(BASE_DIR, ".ava", "git")


def _git_env() -> dict:
    env = os.environ.copy()
    env["GIT_DIR"] = AVA_GIT_DIR
    env["GIT_WORK_TREE"] = BASE_DIR
    # commits automáticos não devem depender de GPG configurado na máquina
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


def _git_run(*args: str, timeout: float = GIT_TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=BASE_DIR,
        env=_git_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _ensure_git_shadow_repo() -> None:
    """
    Garante que existe um repositório git dedicado ao AVA em BASE_DIR/.ava/git,
    com work-tree = BASE_DIR. Independente de existir (ou não) um .git real do
    usuário na mesma pasta — os dois nunca se tocam.
    """
    try:
        if not os.path.isdir(AVA_GIT_DIR):
            os.makedirs(os.path.dirname(AVA_GIT_DIR), exist_ok=True)
            r = _git_run("init", "--quiet")
            if r.returncode != 0:
                log.warning(f"git init (shadow) falhou: {r.stderr.strip()}")
                return
            # identidade local, isolada — não usa/mexe no git config global do usuário
            _git_run("config", "user.name", GIT_AUTHOR_NAME)
            _git_run("config", "user.email", GIT_AUTHOR_EMAIL)
            _git_run("config", "commit.gpgsign", "false")
            _git_run("config", "tag.gpgsign", "false")
            # autocrlf: normaliza CRLF->LF no repo mas mantém o arquivo do jeito
            # que o SO grava — evita diffs gigantes só por causa de line-ending
            autocrlf = "true" if platform.system() == "Windows" else "input"
            _git_run("config", "core.autocrlf", autocrlf)
            # nunca deixa um hook do usuário (ou de outro repo) travar um commit automático
            _git_run("config", "core.hooksPath", os.devnull)
            log.info(f"Shadow git repo inicializado em {AVA_GIT_DIR}")

            gitignore = Path(BASE_DIR) / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text(
                    "__pycache__/\n*.pyc\n.venv/\nvenv/\nnode_modules/\n"
                    ".env\n*.log\n.DS_Store\n.ava/\n",
                    encoding="utf-8",
                )
            # garante que o próprio .ava/ nunca é rastreado dentro do work-tree
            elif ".ava/" not in gitignore.read_text(encoding="utf-8"):
                with open(gitignore, "a", encoding="utf-8") as f:
                    f.write("\n.ava/\n")
        else:
            # repo já existe — só confirma que não há hook interferindo
            _git_run("config", "core.hooksPath", os.devnull)
    except Exception as e:
        log.warning(f"Falha ao inicializar shadow git repo: {e}")


def _git_commit(rel_path: str, action: str) -> None:
    """
    Faz `git add` + `git commit` do arquivo alterado no repo-sombra.
    Nunca levanta exceção — falha de git não deve derrubar write_file/str_replace.
    """
    try:
        add = _git_run("add", "--", rel_path)
        if add.returncode != 0:
            log.warning(f"git add falhou para {rel_path}: {add.stderr.strip()}")
            return
        commit = _git_run(
            "commit", "--quiet", "--no-verify", "-m", f"{action}: {rel_path}"
        )
        # exit code 1 sem stderr relevante == "nothing to commit" (conteúdo idêntico)
        if commit.returncode not in (0, 1):
            log.warning(f"git commit falhou para {rel_path}: {commit.stderr.strip()}")
    except Exception as e:
        log.warning(f"Erro ao commitar {rel_path} no shadow repo: {e}")


_ensure_git_shadow_repo()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve(path: str) -> Path:
    """
    Resolve qualquer caminho relativo ao BASE_DIR.
    Ignora caminhos absolutos do container (/app/alpha, /root, etc).
    Extrai apenas o nome do arquivo ou caminho relativo útil.
    """
    if not path or path == ".":
        return Path(BASE_DIR)

    # Caminho relativo — resolve a partir do BASE_DIR
    if not path.startswith("/"):
        return Path(BASE_DIR) / path

    # Caminho absoluto do container — extrai apenas o que importa
    # /app/alpha/pasta/arquivo.txt → pasta/arquivo.txt
    # /root/arquivo.txt → arquivo.txt
    # /home/user/Alpha/pasta/arquivo.txt → pasta/arquivo.txt
    p = Path(path)

    # Tenta encontrar o sufixo relativo que existe no BASE_DIR
    parts = p.parts
    for i in range(len(parts) - 1, -1, -1):
        candidate = Path(BASE_DIR) / Path(*parts[i:])
        if candidate.exists():
            return candidate

    # Não encontrou — usa só o nome do arquivo no BASE_DIR
    if p.name:
        return Path(BASE_DIR) / p.name

    return Path(BASE_DIR)


def _validate(resolved: Path) -> Path:
    """Garante que o caminho resolvido está dentro do BASE_DIR."""
    try:
        resolved.resolve().relative_to(Path(BASE_DIR).resolve())
        return resolved
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail=f"Fora do BASE_DIR: {resolved} (BASE_DIR={BASE_DIR})",
        )


async def _auth_check(authorization: Optional[str] = Header(None)):
    if not CLIENT_TOKEN:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization required")
    token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
    if token != CLIENT_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")


# ── Models ─────────────────────────────────────────────────────────────────────

class ExecuteRequest(BaseModel):
    command:     str
    working_dir: str   = "."     # ignorado — sempre usa BASE_DIR
    timeout:     float = MAX_CMD_TIME

class ExecuteResponse(BaseModel):
    stdout:    str
    stderr:    str
    exit_code: int
    timed_out: bool = False

class ReadFileRequest(BaseModel):
    file_path: str              # caminho do container — extrai filename
    encoding:  str   = "utf-8"
    max_size:  int   = MAX_FILE_SIZE

class ReadFileResponse(BaseModel):
    content:    str
    file_path:  str             # caminho original do container
    size:       int
    modified:   str
    file_hash:  str
    truncated:  bool = False

class StatRequest(BaseModel):
    path: str

class StatResponse(BaseModel):
    path:      str
    exists:    bool
    is_file:   bool
    is_dir:    bool
    size:      Optional[int]  = None
    modified:  Optional[str]  = None
    file_hash: Optional[str]  = None
    extension: Optional[str]  = None


# ── NOVOS: write / list / str_replace (para alpha_code) ──────────────────────

class WriteFileRequest(BaseModel):
    file_path:      str
    content:        str
    encoding:       str   = "utf-8"
    create_parents: bool  = True
    overwrite:      bool  = True
    # alpha_code.py manda `force=True` nas 3 estratégias de revert
    # pós-syntax-error (_validate_edit_and_revert). Sem este campo o
    # pydantic descartava silenciosamente o valor — funcionava por
    # acidente (overwrite já é True por padrão), mas "force" documentado
    # aqui deixa explícito que ele existe e, se enviado, tem precedência.
    force:          Optional[bool] = None

class WriteFileResponse(BaseModel):
    file_path: str
    bytes_written: int
    created:   bool
    modified:  str

class ListFilesRequest(BaseModel):
    path:         str   = "."
    pattern:      str   = "*"
    recursive:    bool  = True
    max_entries:  int   = 500
    include_hidden: bool = False

class FileEntry(BaseModel):
    path:      str
    name:      str
    is_file:   bool
    is_dir:    bool
    size:      Optional[int]  = None
    modified:  Optional[str]  = None

class ListFilesResponse(BaseModel):
    path:    str
    entries: list[FileEntry]
    total:   int
    truncated: bool = False

class StrReplaceRequest(BaseModel):
    file_path:       str
    old_str:         str
    new_str:         str
    replace_all:     bool            = False
    expected_hash:   Optional[str]   = None   # sha256 devolvido pelo último read_file — protege contra edição concorrente
    # ge=0.5 é intencional, não só validação de faixa: abaixo disso o fuzzy
    # vira "aceita qualquer trecho parecido", o que é perigoso porque o
    # match errado ainda é aplicado silenciosamente (sem erro) — só o
    # `strategy_used="fuzzy"` na resposta denuncia que não foi exato.
    fuzzy_threshold: float = Field(FUZZY_THRESHOLD_DEFAULT, ge=0.5, le=1.0)

class StrReplaceResponse(BaseModel):
    file_path:     str
    replacements:   int
    new_hash:       str
    modified:       str
    strategy_used:  str    # "exact" | "normalized" | "fuzzy" — qual camada resolveu o match

class ReplaceLinesRequest(BaseModel):
    file_path:     str
    start_line:    int   # 1-indexed (inclusive)
    end_line:      int   # 1-indexed (inclusive)
    new_content:   str
    expected_hash: Optional[str] = None   # sha256 devolvido pelo último read_file — protege contra edição concorrente

class ReplaceLinesResponse(BaseModel):
    file_path:      str
    lines_replaced: int
    new_hash:       str
    modified:       str
    
# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Alpha Host Client", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/status")
async def status():
    # Lista o que tem no BASE_DIR para debug
    try:
        contents = os.listdir(BASE_DIR)[:20]
    except Exception:
        contents = []
    return {
        "service":   "Alpha Host Client",
        "version":   "3.0.0",
        "base_dir":  BASE_DIR,
        "platform":  f"{platform.system()} {platform.release()}",
        "home":      str(Path.home()),
        "contents":  contents,
    }


@app.get("/health")
async def health():
    # alpha_code.py's /health handler faz GET /health neste serviço para
    # decidir "scraping_client: ok/unreachable" — sem esta rota, o
    # alpha_code SEMPRE reporta status degradado mesmo com tudo funcionando,
    # porque só existia /status aqui (404 != 200). Alias simples e barato.
    return {"status": "ok", "service": "alpha_host_client", "base_dir": BASE_DIR}


@app.post("/execute", response_model=ExecuteResponse, dependencies=[Depends(_auth_check)])
async def execute_command(req: ExecuteRequest):
    """
    Executa comando no BASE_DIR. working_dir é IGNORADO.
    """
    log.info(f"/execute: cmd='{req.command[:100]}' dir={BASE_DIR}")

    # Segurança
    cmd_l = req.command.lower()
    for pat in [r"\brm\s+-rf\s+/", r"\bdd\s+if=", r"\bmkfs\.",
                r"\bshutdown\b", r"\breboot\b",
                r"\bwget\b.*\|\s*sh", r"\bcurl\b.*\|\s*sh"]:
        if re.search(pat, cmd_l):
            raise HTTPException(status_code=403, detail="Comando bloqueado")

    try:
        proc = subprocess.run(
            req.command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=min(req.timeout, MAX_CMD_TIME),
            cwd=BASE_DIR,              # ← SEMPRE BASE_DIR
        )
        log.info(f"/execute: exit={proc.returncode} stdout_lines={len(proc.stdout.strip().split(chr(10))) if proc.stdout.strip() else 0}")
        return ExecuteResponse(
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return ExecuteResponse(stdout="", stderr="Timeout", exit_code=-1, timed_out=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/read-file", response_model=ReadFileResponse, dependencies=[Depends(_auth_check)])
async def read_file(req: ReadFileRequest):
    """
    Lê arquivo. Extrai o nome/caminho relativo do caminho do container
    e resolve a partir do BASE_DIR.
    """
    host_path = _resolve(req.file_path)
    log.info(f"/read-file: container='{req.file_path}' → host='{host_path}'")

    _validate(host_path)

    if not host_path.exists():
        raise HTTPException(status_code=404, detail=f"Não encontrado: {host_path}")
    if not host_path.is_file():
        raise HTTPException(status_code=400, detail=f"Não é arquivo: {host_path}")

    size = host_path.stat().st_size
    if size > req.max_size:
        raise HTTPException(status_code=413, detail=f"Muito grande: {size}")

    modified = datetime.fromtimestamp(host_path.stat().st_mtime).isoformat()

    sha256 = hashlib.sha256()
    truncated = False

    try:
        with open(host_path, "rb") as f:
            raw = f.read()
        sha256.update(raw)

        for enc in [req.encoding, "utf-8-sig", "latin-1", "cp1252"]:
            try:
                content = raw.decode(enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            content = raw.decode("utf-8", errors="replace")

        if len(content) > 500_000:
            content = content[:500_000]
            truncated = True

    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Sem permissão: {host_path}")

    return ReadFileResponse(
        content=content,
        file_path=req.file_path,
        size=size,
        modified=modified,
        file_hash=sha256.hexdigest(),
        truncated=truncated,
    )


@app.post("/stat", response_model=StatResponse, dependencies=[Depends(_auth_check)])
async def stat_path(req: StatRequest):
    host_path = _resolve(req.path)
    log.info(f"/stat: '{req.path}' → '{host_path}'")

    _validate(host_path)

    if not host_path.exists():
        return StatResponse(path=req.path, exists=False, is_file=False, is_dir=False)

    st = host_path.stat()
    result = StatResponse(
        path=req.path,
        exists=True,
        is_file=host_path.is_file(),
        is_dir=host_path.is_dir(),
        size=st.st_size if host_path.is_file() else None,
        modified=datetime.fromtimestamp(st.st_mtime).isoformat(),
        extension=host_path.suffix.lower() if host_path.is_file() else None,
    )

    if host_path.is_file() and st.st_size <= MAX_FILE_SIZE:
        try:
            sha256 = hashlib.sha256()
            with open(host_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            result.file_hash = sha256.hexdigest()
        except Exception:
            pass

    return result


# ── NOVOS endpoints: write / list / str_replace ───────────────────────────────

@app.post("/create-file", response_model=WriteFileResponse, dependencies=[Depends(_auth_check)])
@app.post("/write-file", response_model=WriteFileResponse, dependencies=[Depends(_auth_check)])
async def create_file(req: WriteFileRequest):
    """
    Cria/sobrescreve arquivo dentro do BASE_DIR.
    Cria diretórios pais se create_parents=True.

    Registrada em DOIS paths: alpha_code.py (agente ReAct) chama
    POST /write-file para todo write_file e para as 3 estratégias de
    revert pós-syntax-error — nenhuma delas chamava /create-file. Sem
    este alias, toda escrita e todo revert automático retornava 404 e
    o agente via isso como falha de rede/permissão, não como "rota
    errada". /create-file continua registrada por compatibilidade com
    quem já chamava esse nome.
    """
    host_path = _resolve(req.file_path)
    log.info(f"/create-file: container='{req.file_path}' → host='{host_path}'")
    _validate(host_path)

    existed = host_path.exists() and host_path.is_file()

    if host_path.exists() and not host_path.is_file():
        raise HTTPException(status_code=400, detail=f"Não é arquivo: {host_path}")

    effective_overwrite = req.force if req.force is not None else req.overwrite
    if host_path.exists() and not effective_overwrite:
        raise HTTPException(status_code=409, detail=f"Arquivo já existe (overwrite=False): {host_path}")

    if req.create_parents:
        host_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = req.content.encode(req.encoding)
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"Conteúdo muito grande: {len(data)} bytes")
        with open(host_path, "wb") as f:
            f.write(data)
        sha = hashlib.sha256(data).hexdigest()
        mtime = datetime.fromtimestamp(host_path.stat().st_mtime).isoformat()

        _git_commit(req.file_path, "write_file (novo)" if not existed else "write_file")

        return WriteFileResponse(
            file_path=req.file_path,
            bytes_written=len(data),
            created=not existed,
            modified=mtime,
        )
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Sem permissão: {host_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/list-files", response_model=ListFilesResponse, dependencies=[Depends(_auth_check)])
async def list_files(req: ListFilesRequest):
    """
    Lista arquivos/dirs dentro de um path do BASE_DIR.
    Usa glob pattern (default: *).
    """
    base = _resolve(req.path)
    log.info(f"/list-files: container='{req.path}' → host='{base}'")
    _validate(base)

    if not base.exists():
        raise HTTPException(status_code=404, detail=f"Path não existe: {req.path}")
    if not base.is_dir():
        raise HTTPException(status_code=400, detail=f"Não é diretório: {req.path}")

    if req.recursive:
        iterator = base.rglob(req.pattern)
    else:
        iterator = base.glob(req.pattern)

    entries: list[FileEntry] = []
    truncated = False
    for p in iterator:
        # skip hidden if not requested
        try:
            # SEMPRE retorna o caminho relativo à raiz do projeto (BASE_DIR).
            # Se listar a pasta "Modules", retorna "Modules/engine.py" em vez
            # de só "engine.py". Isso impede que o agente se perca na hierarquia.
            rel = p.relative_to(Path(BASE_DIR))
        except ValueError:
            continue
        if not req.include_hidden and any(part.startswith(".") for part in rel.parts if part):
            continue

        if len(entries) >= req.max_entries:
            truncated = True
            break

        try:
            st = p.stat()
            entries.append(FileEntry(
                path=str(rel),
                name=p.name,
                is_file=p.is_file(),
                is_dir=p.is_dir(),
                size=st.st_size if p.is_file() else None,
                modified=datetime.fromtimestamp(st.st_mtime).isoformat(),
            ))
        except (PermissionError, OSError):
            continue

    return ListFilesResponse(
        path=req.path,
        entries=entries,
        total=len(entries),
        truncated=truncated,
    )


class AmbiguousMatch(Exception):
    def __init__(self, count: int, hint: str = ""):
        self.count = count
        self.hint = hint


class NoMatch(Exception):
    def __init__(self, closest: str = ""):
        self.closest = closest


def _normalize_line(line: str) -> str:
    # colapsa espaços/tabs internos e remove espaço nas pontas — não mexe em maiúsculas
    return re.sub(r"[ \t]+", " ", line.strip())


def _split_lines(text: str) -> list[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _line_offsets(content: str) -> list[int]:
    """Offset (em caracteres) de início de cada linha de `content`."""
    offsets = [0]
    for line in content.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _windows(content_lines: list[str], n: int):
    for start in range(len(content_lines) - n + 1):
        yield start, start + n


def _find_normalized(content: str, old_str: str) -> list[tuple[int, int]]:
    """Match exato após normalizar espaços/quebras de linha. Retorna spans (linha_ini, linha_fim)."""
    content_lines = _split_lines(content)
    old_lines = [_normalize_line(l) for l in _split_lines(old_str)]
    n = len(old_lines)
    if n == 0 or n > len(content_lines):
        return []
    norm_content = [_normalize_line(l) for l in content_lines]
    matches = []
    for start, end in _windows(norm_content, n):
        if norm_content[start:end] == old_lines:
            matches.append((start, end))
    return matches


def _find_fuzzy(content: str, old_str: str, threshold: float) -> tuple[list[tuple[int, int]], float]:
    """Janela deslizante (mesmo nº de linhas do old_str) com maior similaridade via difflib."""
    content_lines_raw = content.splitlines(keepends=True)
    n = len(old_str.splitlines(keepends=True)) or 1
    if n > len(content_lines_raw):
        return [], 0.0
    best_ratio = 0.0
    best_spans: list[tuple[int, int]] = []
    for start, end in _windows(content_lines_raw, n):
        window = "".join(content_lines_raw[start:end])
        ratio = difflib.SequenceMatcher(None, window, old_str).ratio()
        if ratio > best_ratio + 1e-9:
            best_ratio = ratio
            best_spans = [(start, end)]
        elif abs(ratio - best_ratio) <= 1e-9 and ratio >= threshold:
            best_spans.append((start, end))
    if best_ratio < threshold:
        return [], best_ratio
    return best_spans, best_ratio


def _apply_line_span(content: str, span: tuple[int, int], new_str: str) -> str:
    lines = content.splitlines(keepends=True)
    start, end = span
    return "".join(lines[:start]) + new_str + "".join(lines[end:])


def _resolve_replacement(content: str, old_str: str, new_str: str, replace_all: bool, threshold: float):
    """
    Cadeia de fallback (nesta ordem, por pedido explícito): fuzzy -> normalizado -> exato.
    Retorna (novo_conteudo, nº_de_substituicoes, estrategia_usada).
    """
    content_lines = _split_lines(content)

    # 1) FUZZY — só roda em arquivos de tamanho razoável (custo O(linhas × old_str))
    if len(content_lines) <= FUZZY_MAX_CONTENT_LINES:
        spans, ratio = _find_fuzzy(content, old_str, threshold)
        if len(spans) == 1:
            return _apply_line_span(content, spans[0], new_str), 1, "fuzzy"
        if len(spans) > 1 and not replace_all:
            raise AmbiguousMatch(len(spans), hint=f"fuzzy ratio={ratio:.2f}")
        if len(spans) > 1 and replace_all:
            new_content = content
            for span in sorted(spans, reverse=True):
                new_content = _apply_line_span(new_content, span, new_str)
            return new_content, len(spans), "fuzzy"

    # 2) NORMALIZADO — mesmo texto ignorando diferenças de espaço/indentação/EOL
    spans = _find_normalized(content, old_str)
    if len(spans) == 1:
        return _apply_line_span(content, spans[0], new_str), 1, "normalized"
    if len(spans) > 1:
        if not replace_all:
            raise AmbiguousMatch(len(spans))
        new_content = content
        for span in sorted(spans, reverse=True):
            new_content = _apply_line_span(new_content, span, new_str)
        return new_content, len(spans), "normalized"

    # 3) EXATO — comportamento original, agora como último recurso
    count = content.count(old_str)
    if count == 0:
        raise NoMatch(closest=_closest_snippet(content, old_str))
    if count > 1 and not replace_all:
        raise AmbiguousMatch(count)
    new_content = content.replace(old_str, new_str) if replace_all else content.replace(old_str, new_str, 1)
    return new_content, (count if replace_all else 1), "exact"


def _closest_snippet(content: str, old_str: str, context_lines: int = 2) -> str:
    """Pra mensagem de erro: mostra o trecho mais parecido, ajuda o LLM a se corrigir no retry."""
    spans, ratio = _find_fuzzy(content, old_str, threshold=0.0)
    if not spans:
        return ""
    start, end = spans[0]
    lines = content.splitlines(keepends=True)
    lo = max(0, start - context_lines)
    hi = min(len(lines), end + context_lines)
    snippet = "".join(lines[lo:hi])
    return f"(similaridade {ratio:.0%}) {snippet[:500]}"


@app.post("/str-replace", response_model=StrReplaceResponse, dependencies=[Depends(_auth_check)])
async def str_replace(req: StrReplaceRequest):
    """
    Substitui old_str por new_str em arquivo do BASE_DIR.
    Tenta, nesta ordem: fuzzy match -> match normalizado (espaços/EOL) -> match exato.
    Falha se nenhuma camada achar exatamente 1 ocorrência (ou >1 sem replace_all=True).
    """
    if not req.old_str:
        raise HTTPException(status_code=400, detail="old_str não pode ser vazio")
    if req.old_str == req.new_str:
        raise HTTPException(status_code=400, detail="old_str == new_str")

    host_path = _resolve(req.file_path)
    log.info(f"/str-replace: container='{req.file_path}' → host='{host_path}'")
    _validate(host_path)

    if not host_path.exists():
        raise HTTPException(status_code=404, detail=f"Não encontrado: {host_path}")
    if not host_path.is_file():
        raise HTTPException(status_code=400, detail=f"Não é arquivo: {host_path}")

    try:
        raw = host_path.read_bytes()
        # tenta utf-8 primeiro, fallback latin-1
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                content = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            content = raw.decode("utf-8", errors="replace")

        # ── Proteção contra edição concorrente/arquivo desatualizado ──
        # Se o agente mandar o hash que recebeu no último read_file, confere
        # antes de tentar qualquer match — evita aplicar um replace "que só
        # por acidente" bate em cima de um arquivo que já mudou.
        if req.expected_hash:
            current_hash = hashlib.sha256(raw).hexdigest()
            if current_hash != req.expected_hash:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Arquivo mudou desde a última leitura (hash esperado="
                        f"{req.expected_hash[:12]}…, atual={current_hash[:12]}…). "
                        "Releia o arquivo (read-file) antes de tentar de novo."
                    ),
                )

        try:
            new_content, replacements, strategy = _resolve_replacement(
                content, req.old_str, req.new_str, req.replace_all, req.fuzzy_threshold,
            )
        except NoMatch as e:
            detail = (
                "old_str não encontrado no arquivo (nem por match exato, "
                "normalizado ou fuzzy). Isso normalmente significa que o "
                "conteúdo mudou desde a última leitura. Releia o arquivo "
                "(read-file) antes de tentar de novo."
            )
            if e.closest:
                detail += f" Trecho mais parecido encontrado: {e.closest}"
            raise HTTPException(status_code=422, detail=detail)
        except AmbiguousMatch as e:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"old_str corresponde a {e.count} trechos diferentes"
                    f"{' (' + e.hint + ')' if e.hint else ''}. "
                    "Use replace_all=true ou torne old_str mais específico "
                    "(inclua mais linhas de contexto ao redor)."
                ),
            )

        new_bytes = new_content.encode("utf-8")
        host_path.write_bytes(new_bytes)
        sha = hashlib.sha256(new_bytes).hexdigest()
        mtime = datetime.fromtimestamp(host_path.stat().st_mtime).isoformat()

        _git_commit(req.file_path, f"str_replace ({strategy})")

        return StrReplaceResponse(
            file_path=req.file_path,
            replacements=replacements,
            new_hash=sha,
            modified=mtime,
            strategy_used=strategy,
        )
    except HTTPException:
        raise
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Sem permissão: {host_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/replace-lines", response_model=ReplaceLinesResponse, dependencies=[Depends(_auth_check)])
async def replace_lines(req: ReplaceLinesRequest):
    """
    Substitui um intervalo de linhas (1-indexed, inclusivo) por um novo conteúdo.
    Útil para o agente não precisar reproduzir old_str exato via str-replace,
    apenas referenciando os números de linha que viu no output do read_file.
    """
    if req.start_line < 1 or req.end_line < req.start_line:
        raise HTTPException(status_code=400, detail="Intervalo de linhas inválido (start_line deve ser >= 1 e <= end_line)")

    host_path = _resolve(req.file_path)
    log.info(f"/replace-lines: container='{req.file_path}' → host='{host_path}' (lines {req.start_line}-{req.end_line})")
    _validate(host_path)

    if not host_path.exists():
        raise HTTPException(status_code=404, detail=f"Não encontrado: {host_path}")
    if not host_path.is_file():
        raise HTTPException(status_code=400, detail=f"Não é arquivo: {host_path}")

    try:
        raw = host_path.read_bytes()
        # tenta utf-8 primeiro, fallback latin-1
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                content = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            content = raw.decode("utf-8", errors="replace")

        # ── Proteção contra edição concorrente/arquivo desatualizado ──
        if req.expected_hash:
            current_hash = hashlib.sha256(raw).hexdigest()
            if current_hash != req.expected_hash:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Arquivo mudou desde a última leitura (hash esperado="
                        f"{req.expected_hash[:12]}…, atual={current_hash[:12]}…). "
                        "Releia o arquivo (read-file) antes de tentar de novo."
                    ),
                )

        lines = content.splitlines(keepends=True)
        
        # Validar limites do arquivo
        if req.start_line > len(lines) + 1:
            raise HTTPException(
                status_code=400, 
                detail=f"start_line ({req.start_line}) fora do limite (arquivo tem {len(lines)} linhas)"
            )

        # Ajustar para índice 0 do Python
        start_idx = req.start_line - 1
        end_idx = min(req.end_line, len(lines))

        # Construir o novo conteúdo do arquivo
        new_str = req.new_content
        
        # Se o novo conteúdo não for vazio e não terminar com quebra de linha,
        # mas houver linhas após a substituição, precisamos adicionar \n
        # para não mesclar com a próxima linha
        if new_str and not new_str.endswith(("\n", "\r")) and end_idx < len(lines):
            new_str += "\n"

        new_lines_list = lines[:start_idx] + [new_str] + lines[end_idx:]
        new_content = "".join(new_lines_list)

        new_bytes = new_content.encode("utf-8")
        host_path.write_bytes(new_bytes)
        sha = hashlib.sha256(new_bytes).hexdigest()
        mtime = datetime.fromtimestamp(host_path.stat().st_mtime).isoformat()

        _git_commit(req.file_path, f"replace_lines ({req.start_line}-{req.end_line})")

        return ReplaceLinesResponse(
            file_path=req.file_path,
            lines_replaced=end_idx - start_idx,
            new_hash=sha,
            modified=mtime,
        )
    except HTTPException:
        raise
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Sem permissão: {host_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entrypoint ───────────────────────────────────────────────────────────────



if __name__ == "__main__":
    import uvicorn
    log.info(f"═══════════════════════════════════════════")
    log.info(f"  Alpha Host Client v3.0")
    log.info(f"  Porta:    {3005}")
    log.info(f"  BASE_DIR: {BASE_DIR}")
    log.info(f"═══════════════════════════════════════════")
    uvicorn.run(app, host="0.0.0.0", port=3005, log_level="info")