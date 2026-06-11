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

CLIENT_PORT  = "3005"
CLIENT_TOKEN = os.environ.get("CLIENT_TOKEN", "")
MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_CMD_TIME  = 30.0

# O ÚNICO caminho que importa — onde o script está rodando
BASE_DIR = os.path.abspath(os.environ.get("BASE_DIR", os.getcwd()))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ALPHA-CLIENT] %(message)s",
)
log = logging.getLogger("alpha_client")

log.info(f"BASE_DIR:  {BASE_DIR}")
log.info(f"Platform:  {platform.system()} {platform.release()}")
log.info(f"Home:      {Path.home()}")
log.info(f"CWD:       {os.getcwd()}")


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


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    log.info(f"═══════════════════════════════════════════")
    log.info(f"  Alpha Host Client v3.0")
    log.info(f"  Porta:    {CLIENT_PORT}")
    log.info(f"  BASE_DIR: {BASE_DIR}")
    log.info(f"═══════════════════════════════════════════")
    uvicorn.run(app, host="0.0.0.0", port=CLIENT_PORT, log_level="info")