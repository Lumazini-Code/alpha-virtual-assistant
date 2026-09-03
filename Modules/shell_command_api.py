"""
AVA Shell Command API — execução de comandos de sistema pré-programados
=========================================================================

Este serviço concentra a única parte do pipeline AVA que precisa tocar
diretamente no sistema operacional via subprocess: hoje, abrir e derrubar
o processo do Chrome real do usuário com debugging remoto.

Por que separar isso do browser_agent.py?
    O WebAgent conecta no Chrome via CDP (Chrome DevTools Protocol) — ele
    não precisa ser o mesmo processo que abriu o Chrome, só precisa
    alcançar a porta de debug. Extraindo o "abrir o Chrome" pra um serviço
    HTTP próprio, o browser_agent.py deixa de fazer subprocess.Popen
    diretamente e passa a REQUISITAR esta API. Isso também documenta e
    concentra num único lugar tudo que mexe com processos do SO.

Segurança — isto NÃO é um "execute qualquer shell command" genérico:
    Só existem comandos PRÉ-PROGRAMADOS no COMMAND_REGISTRY abaixo. A
    requisição manda o NOME do comando + parâmetros com valores permitidos
    (porta, diretório de profile) — nunca uma linha de comando livre. O
    argv é sempre montado aqui dentro como LISTA (nunca shell=True com
    string interpolada), e qualquer parâmetro que não esteja no schema do
    comando é rejeitado com 400. Adicionar um comando novo é adicionar uma
    entrada no registro — não abre superfície pra execução arbitrária.
"""

from __future__ import annotations

import os
import time
import shutil
import socket
import asyncio
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


API_PORT = _env_int("AGENT_SHELL_API_PORT", 4005)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [ShellCommandAPI] %(message)s")
log = logging.getLogger("ava.shell_command_api")


# ── Descoberta do Chrome/profile na máquina onde ESTE serviço roda ────────
# (Movido de browser_agent.py — quem abre o processo real agora é este
# serviço, então é ele quem precisa localizar o binário/profile.)

def find_chrome_binary() -> str:
    candidates = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]
    for name in candidates:
        path = shutil.which(name)
        if path:
            return path
    raise FileNotFoundError("Nenhum binário de Chrome/Chromium encontrado no PATH desta máquina.")


def find_original_user_data_dir() -> Path:
    home = Path.home()
    candidates = [
        home / ".config" / "google-chrome",
        home / ".config" / "chromium",
        home / "Library" / "Application Support" / "Google" / "Chrome",
        home / "AppData" / "Local" / "Google" / "Chrome" / "User Data",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Nenhum diretório de profile do Chrome encontrado automaticamente nesta máquina.")


def get_or_create_profile_copy(copy_suffix: str = "-alphaai") -> str:
    original = find_original_user_data_dir()
    copy_path = original.parent / f"{original.name}{copy_suffix}"

    if copy_path.exists():
        return str(copy_path)

    log.info(f"Copiando profile de {original} para {copy_path} (só acontece uma vez)...")
    try:
        shutil.copytree(original, copy_path, ignore=shutil.ignore_patterns("Cache", "Code Cache", "GPUCache"))
    except shutil.Error as e:
        log.warning(f"Alguns arquivos não puderam ser copiados ({e}). Continuando.")
    except FileExistsError:
        pass

    return str(copy_path)


def is_port_open(port: int, host: str = "localhost") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


# ── Processos abertos por este serviço (pra poder derrubar depois) ─────────

_RUNNING_PROCESSES: dict[str, subprocess.Popen] = {}


# ── Catálogo de comandos pré-programados ───────────────────────────────────

@dataclass
class CommandSpec:
    description: str
    params: dict[str, Any] = field(default_factory=dict)  # nome -> default
    handler: Callable[[dict], dict] = None


def _cmd_launch_chrome_debug(params: dict) -> dict:
    port = int(params["port"])
    profile_directory = str(params["profile_directory"])
    key = f"chrome_debug:{port}"

    if is_port_open(port):
        log.info(f"Chrome já está rodando na porta {port}, reaproveitando.")
        return {"already_running": True, "port": port}

    chrome_path = find_chrome_binary()
    user_data_dir = get_or_create_profile_copy()

    argv = [
        chrome_path,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={user_data_dir}",
        f"--profile-directory={profile_directory}",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    log.info(f"Executando: {' '.join(argv)}")
    process = subprocess.Popen(argv)
    _RUNNING_PROCESSES[key] = process

    for _ in range(30):
        if is_port_open(port):
            break
        time.sleep(0.5)
    else:
        process.terminate()
        _RUNNING_PROCESSES.pop(key, None)
        raise RuntimeError("Chrome não abriu a porta de debug a tempo.")

    return {"already_running": False, "port": port, "pid": process.pid}


def _cmd_kill_chrome_debug(params: dict) -> dict:
    port = int(params["port"])
    key = f"chrome_debug:{port}"
    process = _RUNNING_PROCESSES.pop(key, None)
    if process is None:
        return {"killed": False, "reason": "processo não foi aberto por este serviço (ou já foi encerrado)."}
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
    return {"killed": True, "port": port}


def _cmd_is_port_open(params: dict) -> dict:
    port = int(params["port"])
    return {"port": port, "open": is_port_open(port)}


COMMAND_REGISTRY: dict[str, CommandSpec] = {
    "launch_chrome_debug": CommandSpec(
        description="Abre (ou reaproveita) o Chrome real do usuário com debugging remoto. Copia o profile original na 1ª vez.",
        params={"port": 9222, "profile_directory": "Default"},
        handler=_cmd_launch_chrome_debug,
    ),
    "kill_chrome_debug": CommandSpec(
        description="Derruba o processo do Chrome aberto por 'launch_chrome_debug' nesta porta (só funciona se foi este serviço quem o abriu).",
        params={"port": 9222},
        handler=_cmd_kill_chrome_debug,
    ),
    "is_port_open": CommandSpec(
        description="Verifica se uma porta TCP local está aberta (ex.: checar se o debug do Chrome já subiu).",
        params={"port": 9222},
        handler=_cmd_is_port_open,
    ),
}


# ── Modelos + app ────────────────────────────────────────────────────────

class CommandRequest(BaseModel):
    command: str = Field(..., description="Nome do comando pré-programado (ver GET /commands).")
    params: dict[str, Any] = Field(default_factory=dict, description="Só os parâmetros aceitos pelo comando — ver CommandSpec.params.")


class CommandResponse(BaseModel):
    command: str
    success: bool
    result: Any = None
    error: Optional[str] = None


_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="shell-command")


async def _run_handler(handler: Callable[[dict], dict], params: dict) -> dict:
    """
    launch_chrome_debug faz polling bloqueante (até ~15s) esperando a porta
    abrir — roda num executor pra não travar o loop de eventos e permitir
    outras requisições (ex.: /status) em paralelo.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, handler, params)


app = FastAPI(title="AVA Shell Command API")


@app.post("/command", response_model=CommandResponse)
async def run_command(req: CommandRequest):
    spec = COMMAND_REGISTRY.get(req.command)
    if spec is None:
        raise HTTPException(
            status_code=400,
            detail=f"Comando desconhecido: '{req.command}'. Disponíveis: {sorted(COMMAND_REGISTRY)}",
        )

    unknown = set(req.params) - set(spec.params)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Parâmetro(s) não reconhecido(s) para '{req.command}': {sorted(unknown)}. Aceitos: {sorted(spec.params)}",
        )

    merged_params = {**spec.params, **req.params}

    try:
        result = await _run_handler(spec.handler, merged_params)
    except Exception as e:
        log.exception(f"Falha ao executar comando '{req.command}'")
        return CommandResponse(command=req.command, success=False, error=str(e))

    return CommandResponse(command=req.command, success=True, result=result)


@app.get("/commands")
async def list_commands():
    return {
        name: {"description": spec.description, "params": spec.params}
        for name, spec in COMMAND_REGISTRY.items()
    }


@app.get("/status")
async def status():
    return {
        "running_processes": {
            k: v.pid for k, v in _RUNNING_PROCESSES.items() if v.poll() is None
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("shell_command_api:app", host="0.0.0.0", port=API_PORT, log_level="info")
