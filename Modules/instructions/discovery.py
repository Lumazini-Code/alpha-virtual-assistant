# Alpha AI (AVA) — Descoberta de AGENTS.md e instruções configuradas
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import difflib
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .source import REMOVED, Source

FILENAME = "AGENTS.md"


@dataclass(frozen=True)
class InstructionFile:
    path: str
    content: str


def _read(path: Path) -> Optional[InstructionFile]:
    try:
        resolved = path.resolve()
        if not resolved.is_file():
            return None
        return InstructionFile(str(resolved), resolved.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None


def _project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return start


def _ancestors(start: Path, stop: Path) -> list[Path]:
    directories: list[Path] = []
    current = start
    while True:
        directories.append(current)
        if current == stop or current.parent == current:
            break
        current = current.parent
    return directories


def _split_env(name: str) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    values: list[str] = []
    for chunk in raw.replace(",", os.pathsep).split(os.pathsep):
        chunk = chunk.strip()
        if chunk:
            values.append(chunk)
    return values


def _default_dirs() -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    dirs = [root, Path.cwd()]
    return list(dict.fromkeys(dirs))


def discover_files(start: Optional[Path] = None) -> list[InstructionFile]:
    """Lê AGENTS.md global, do projeto e arquivos configurados.

    Ordem: global, arquivos explícitos, diretórios (mais próximo → raiz).
    Caminhos inacessíveis (ex.: host path dentro do container) são ignorados.
    ``start`` permite usar o cwd do turno na execução nativa.
    """
    candidates: list[Path] = []

    global_file = os.getenv("AVA_GLOBAL_AGENTS_MD", "").strip()
    if global_file:
        candidates.append(Path(global_file).expanduser())
    else:
        appdata = os.getenv("APPDATA")
        if os.name == "nt" and appdata:
            candidates.append(Path(appdata) / "alpha-ai" / FILENAME)
        else:
            candidates.append(Path.home() / ".config" / "alpha-ai" / FILENAME)

    for pattern in _split_env("AVA_INSTRUCTION_FILES"):
        expanded = os.path.expanduser(pattern)
        if any(char in expanded for char in "*?["):
            candidates.extend(Path(path) for path in sorted(glob.glob(expanded, recursive=True)))
        else:
            candidates.append(Path(expanded))

    if start is not None:
        try:
            resolved = start.expanduser().resolve()
        except (OSError, ValueError):
            resolved = None
        dirs = [resolved] if resolved is not None else []
    else:
        dirs = [Path(value).expanduser() for value in _split_env("AVA_INSTRUCTIONS_DIRS")] or _default_dirs()
    for raw in dirs:
        try:
            current = raw.resolve()
        except (OSError, ValueError):
            continue
        if current.is_file():
            candidates.append(current)
            continue
        if not current.is_dir():
            continue
        try:
            if current == Path.home().resolve():
                # Home é o diretório padrão: usa só o AGENTS.md global.
                continue
        except (OSError, RuntimeError):
            pass
        stop = _project_root(current)
        for directory in _ancestors(current, stop):
            candidates.append(directory / FILENAME)

    files: list[InstructionFile] = []
    seen: set[str] = set()
    for candidate in candidates:
        file = _read(candidate)
        if file is None or file.path in seen:
            continue
        seen.add(file.path)
        files.append(file)
    return files


def _render(files: list[InstructionFile]) -> str:
    return "\n\n".join(f"Instructions from: {file.path}\n{file.content}" for file in files)


def discover_near(path: Path) -> list[InstructionFile]:
    """AGENTS.md no diretório do arquivo lido e seus ancestrais até a raiz."""
    try:
        start = path.resolve()
    except (OSError, ValueError):
        return []
    if not start.is_dir():
        start = start.parent
    root = _project_root(start)
    files: list[InstructionFile] = []
    seen: set[str] = set()
    for directory in _ancestors(start, root):
        file = _read(directory / FILENAME)
        if file is None or file.path in seen:
            continue
        seen.add(file.path)
        files.append(file)
    return files


def _render_change(previous: list[InstructionFile], current: list[InstructionFile]) -> str:
    before = {file.path: file.content for file in previous}
    after = {file.path: file.content for file in current}
    parts: list[str] = []

    for path in after:
        if path not in before:
            parts.append(f"New instructions apply from:\nInstructions from: {path}\n{after[path]}")

    for path in before:
        if path in after:
            continue
        parts.append(f"The instructions from {path} no longer apply.")

    for path in after:
        if path not in before or before[path] == after[path]:
            continue
        diff = "".join(
            difflib.unified_diff(
                before[path].splitlines(keepends=True),
                after[path].splitlines(keepends=True),
                fromfile=path,
                tofile=path,
                n=3,
            )
        )
        replacement = f"Instructions from: {path}\n{after[path]}"
        if diff:
            text = f"The instructions from {path} changed. Here's the diff:\n```diff\n{diff}```"
            parts.append(text if len(text) < len(replacement) else replacement)
        else:
            parts.append(replacement)

    return "\n\n".join(parts)


def make_source(files: list[InstructionFile] | None = None) -> Source:
    """Uma fonte única para todos os AGENTS.md descobertos."""

    def read() -> list[dict[str, str]] | Any:
        current = files if files is not None else discover_files()
        if not current:
            return REMOVED
        return [{"path": file.path, "content": file.content} for file in current]

    def decode(value: Any) -> list[InstructionFile]:
        if not isinstance(value, list):
            return []
        return [
            InstructionFile(str(item["path"]), str(item["content"]))
            for item in value
            if isinstance(item, dict) and "path" in item and "content" in item
        ]

    return Source(
        key="core/instructions",
        read=read,
        initial=lambda value: _render(decode(value)),
        changed=lambda previous, current: _render_change(decode(previous), decode(current)),
        removed=lambda previous: (
            "Previously loaded instructions no longer apply: "
            + ", ".join(file.path for file in decode(previous))
            + "."
            if decode(previous)
            else "Previously loaded instructions no longer apply."
        ),
    )


def load_sources(start: Optional[Path] = None) -> list[Source]:
    return [make_source(discover_files(start))]
