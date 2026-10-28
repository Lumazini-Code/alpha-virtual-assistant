# Alpha AI (AVA) — Leitor de Agent Skills (SKILL.md) p/ o orchestrator
# Copyright (C) 2026  Felipe lumazini
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
agent_skills.py — Leitor de Agent Skills (SKILL.md) p/ o orchestrator
=====================================================================

Skills no padrão "Agent Skills" (o mesmo do CLI `skills` do npm / skills.sh):
cada skill é uma PASTA com um SKILL.md que começa com frontmatter YAML
(`name` + `description`). Instalação canônica do CLI:
  - PROJETO:   `.agents/skills/` + lock `skills-lock.json` na raiz do projeto;
  - GLOBAL:    `~/.agents/skills/` (mesmo layout flat) + lock
               `~/.agents/.skill-lock.json` (ou $XDG_STATE_HOME/skills/ —
               variante do próprio CLI).
Este módulo NÃO instala nada do CLI `skills` — só LÊ o que está no disco
(o usuário instala/atualiza com `npx skills add [-g]` no host; os mounts ro
do compose fazem o container enxergar na hora). Ele CRIA um tipo especial:
skills escritas PELO AGENTE em runtime (`.agents/skills-created/` — dir
separado do store do CLI, que nunca as reconcilia/limpa).

Responsabilidades (tudo stdlib, sem dependência nova):
  - scan_agent_skills(dirs): descobre as skills (frontmatter + doc_hash) em
    uma lista de diretórios com prioridade — a primeira pasta com um dado
    nome vence (dirs de projeto sobrepõem os globais, que entram por último);
  - read_skill_body / resolve_skill_file: leitura usada pelas tools nativas
    `use_skill` / `skill_file` do orchestrator, com guarda de path traversal
    (symlinks resolvidos via os.path.realpath) e recusa de binário;
  - create_agent_skill: cria skill do AGENTE (SKILL.md + arquivos extras)
    com validação de nome/description, recusa de duplicata (salvo overwrite)
    e escrita atômica — usado pelo POST /skills/create do orchestrator;
  - load_lockfile: proveniência (`owner/repo`) de AMBOS os lockfiles — só
    informativa (GET /skills do orchestrator).

O ÍNDICE SEMÂNTICO (kNN + PPR + queries aprendidas) dessas skills vive no
ToolsDB da memory (mesma tabela das MCP tools, sob o prefixo "agentskill__":
ver memory_ops_agent_skills.py e o bloco "Agent Skills" no orchestrator.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SKILL_MD_NAME = "SKILL.md"
LOCKFILE_NAME = "skills-lock.json"        # projeto (raiz do projeto)
GLOBAL_LOCKFILE_NAME = ".skill-lock.json"  # global (install -g: ~/.agents/)


@dataclass(frozen=True)
class AgentSkill:
    name: str
    description: str
    dir: Path      # pasta da skill (contém o SKILL.md)
    path: Path     # caminho do SKILL.md
    doc_hash: str  # sha256 do conteúdo bruto do SKILL.md (hex)


# ── Frontmatter (subconjunto YAML — sem dependência) ──────────────────────────
# Só o que skills reais usam: `key: value` (com aspas opcional), scalars de
# bloco (`|`, `>` com modificadores), comentários de linha inteira e linhas
# vazias. Qualquer construção YAML mais exótica é ignorada linha a linha —
# uma skill que use algo fora disso ainda entra no índice (o nome cai para o
# nome da pasta), só perde o frontmatter.

_BLOCK_SCALARS = frozenset({"|", ">", "|-", ">-", "|+", ">+"})


def _unquote(value: str) -> str:
    """Remove aspas envolventes e comentário no fim de linha (` # ...`)."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    for i, ch in enumerate(value):
        if ch == "#" and i > 0 and value[i - 1] in " \t":
            return value[:i].rstrip()
    return value


def _finish_block(block: list[str], kind: str) -> str:
    # indentação do bloco = a da primeira linha de conteúdo não vazia (YAML
    # tira esse prefixo de TODAS as linhas do scalar — não só um espaço)
    indent = ""
    for l in block:
        if l.strip():
            indent = l[: len(l) - len(l.lstrip())]
            break
    if indent:
        block = [l[len(indent):] if l.startswith(indent) else l.lstrip() for l in block]
    if kind == "|":
        return "\n".join(block).strip("\n")
    return " ".join(l.strip() for l in block if l.strip()).strip()


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Separa o frontmatter YAML (primeiro bloco `--- ... ---`) do corpo.
    Sem frontmatter (ou bloco nunca fechado) → ({}, texto inteiro)."""
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end = idx
            break
    if end is None:
        return {}, text
    meta: dict[str, str] = {}
    body = "\n".join(lines[end + 1:])
    current_key: Optional[str] = None
    block_kind = ""
    block: list[str] = []
    for raw in lines[1:end]:
        if current_key is not None:
            # dentro de um scalar de bloco: linha vazia ou indentada continua
            if raw.strip() == "" or raw[:1] in (" ", "\t"):
                block.append(raw[1:] if raw[:1] == " " else raw)
                continue
            meta[current_key] = _finish_block(block, block_kind)
            current_key, block_kind, block = None, "", []
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in raw:
            continue
        key, _, value = raw.partition(":")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if value in _BLOCK_SCALARS:
            current_key, block_kind, block = key, value[0], []
            continue
        meta[key] = _unquote(value)
    if current_key is not None:
        meta[current_key] = _finish_block(block, block_kind)
    return meta, body


# ── Descoberta ────────────────────────────────────────────────────────────────

def load_lockfile(skill_dir: Path) -> dict[str, str]:
    """Lockfile do CLI `skills` (`npx skills add`) — proveniência por skill.
    Procura, nessa ordem (o primeiro que existir vence):
      1. <dir>/skills-lock.json        — projeto (raiz do projeto)
      2. <dir>/.skill-lock.json
      3. <dir-pai>/skills-lock.json
      4. <dir-pai>/.skill-lock.json    — GLOBAL (install -g: ~/.agents/)
      5. <dir-avô>/skills-lock.json    — projeto canônico (.agents/skills ↦ raiz)
      6. $XDG_STATE_HOME/skills/.skill-lock.json — variante global do CLI
    Ausente/corrompido → {} (proveniência é só informativa)."""
    candidates = [
        skill_dir / LOCKFILE_NAME,
        skill_dir / GLOBAL_LOCKFILE_NAME,
        skill_dir.parent / LOCKFILE_NAME,
        skill_dir.parent / GLOBAL_LOCKFILE_NAME,
        skill_dir.parent.parent / LOCKFILE_NAME,
    ]
    xdg = os.environ.get("XDG_STATE_HOME")
    if xdg:
        candidates.append(Path(xdg) / "skills" / GLOBAL_LOCKFILE_NAME)
    for f in candidates:
        if not f.is_file():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        out: dict[str, str] = {}
        for name, entry in (data.get("skills") or {}).items():
            if isinstance(entry, dict):
                src = str(entry.get("source") or "").strip()
                typ = str(entry.get("sourceType") or "").strip()
                out[str(name)] = f"{typ}:{src}" if src else ""
        return out
    return {}


def scan_agent_skills(dirs) -> tuple[list[AgentSkill], dict[str, str]]:
    """Descobre skills nos diretórios (a ordem da lista é prioridade: a
    primeira pasta com um dado `name` vence — dirs de PROJETO sobrepõem os
    GLOBAIS, que entram por último). Cada diretório carrega o SEU lockfile
    (projeto: `skills-lock.json` na raiz; global: `~/.agents/.skill-lock.json`)
    e a proveniência é mesclada sem sobrescrever o que o dir anterior já deu.
    Ignora pastas ocultas/__dunder__, pastas sem SKILL.md e nomes inválidos
    p/ nome de grafo (com `/` ou `\\`, começando com `.`). Devolve (skills,
    proveniência mesclada do lockfile)."""
    seen: dict[str, AgentSkill] = {}
    lock: dict[str, str] = {}
    for raw in dirs:
        root = Path(raw).expanduser()
        if not root.is_dir():
            continue
        for name, src in load_lockfile(root).items():
            lock.setdefault(name, src)
        for child in sorted(p for p in root.iterdir() if p.is_dir()):
            if child.name.startswith((".", "__")):
                continue
            skill_md = child / SKILL_MD_NAME
            if not skill_md.is_file():
                continue
            try:
                raw_text = skill_md.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            meta, _ = parse_frontmatter(raw_text)
            name = str(meta.get("name") or child.name).strip()
            if not name or "/" in name or "\\" in name or name.startswith("."):
                continue
            if name in seen:
                continue
            try:
                doc_hash = hashlib.sha256(skill_md.read_bytes()).hexdigest()
            except OSError:
                continue
            seen[name] = AgentSkill(
                name=name,
                description=str(meta.get("description") or "").strip(),
                dir=child,
                path=skill_md,
                doc_hash=doc_hash,
            )
    return list(seen.values()), lock


# ── Leitura (tools use_skill / skill_file) ────────────────────────────────────

def read_skill_body(path: Path, max_chars: int) -> str:
    """Corpo do SKILL.md (sem frontmatter), truncado em `max_chars` com
    marcador explícito (o modelo é orientado a usar skill_file p/ detalhes)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    _, body = parse_frontmatter(text)
    body = body.strip()
    if len(body) > max_chars:
        body = (
            body[:max_chars].rstrip()
            + f"\n\n[... SKILL.md truncated at {max_chars} chars — use skill_file "
              "to read specific sections (e.g. 'references/<file>.md') ...]"
        )
    return body


def resolve_skill_file(skill_dir: Path, rel_path: str, max_chars: int) -> str:
    """Lê um arquivo DENTRO da pasta da skill (references/, assets/... —
    caminho relativo à pasta). Guarda de traversal: o caminho resolvido
    (realpath — symlinks incluídos) TEM que continuar sob a pasta real da
    skill. Recusa binário (byte NUL nos primeiros 4 KB) e trunca em
    `max_chars` com marcador."""
    if not rel_path or rel_path.strip() in ("", ".", "/"):
        raise ValueError("caminho vazio")
    base_real = Path(os.path.realpath(skill_dir))
    target_real = Path(os.path.realpath(base_real / rel_path))
    if target_real != base_real and base_real not in target_real.parents:
        raise ValueError(f"caminho escapa da pasta da skill: {rel_path!r}")
    if not target_real.is_file():
        raise FileNotFoundError(f"arquivo não encontrado dentro da skill: {rel_path!r}")
    with open(target_real, "rb") as f:
        head = f.read(4096)
    if b"\x00" in head:
        raise ValueError(f"arquivo binário (recusado): {rel_path!r}")
    text = target_real.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + f"\n\n[... truncated at {max_chars} chars ...]"
    return text


# ── Criação (skill escrita PELO AGENTE — POST /skills/create) ─────────────────
# Skills criadas em runtime pelo próprio modelo vão para um diretório SEPARADO
# do store do CLI (`.agents/skills-created/` — o `.gitignore` já cobre via
# `.agents`): o CLI `skills` só reconcilia o que está no lock dele e nunca
# toca aqui; o GET /skills distingue a origem pelo dir. Prioridade no scan:
# projeto CLI > skills-created > global (uma skill criada SOMBREIA a global
# de mesmo nome, mas nunca a de projeto — a rota recusa nesse caso).

class SkillCreateError(Exception):
    """Recusa de criação (validação, duplicata). `status` sugere o HTTP —
    a rota do orchestrator converte em HTTPException(status, detail)."""

    def __init__(self, detail: str, status: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status = status


SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MIN_SKILL_DESCRIPTION_CHARS = 10   # description é o driver da seleção semântica
MAX_SKILL_DESCRIPTION_CHARS = 500
MAX_SKILL_BODY_CHARS        = 96_000   # corpo do SKILL.md (frontmatter à parte)
MAX_SKILL_EXTRA_FILES       = 20       # arquivos extras por skill
MAX_SKILL_EXTRA_FILE_CHARS  = 64_000   # por arquivo extra


def slugify_skill_name(raw: str) -> str:
    """Nome vindo do modelo → slug determinístico (mesmo espírito do
    `_slugify` dos projetos, mas kebab-case — convenção do ecossistema
    skills.sh: hyperframes-cli, reverse-engineer-anything). Acentos NFKD
    decompõem ("relatório" → "relatorio") antes da limpeza. Vazio/inválido
    após a limpeza → '' (caller recusa)."""
    text = unicodedata.normalize("NFKD", (raw or "").strip().lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    slug = re.sub(r"[^a-z0-9_-]+", "-", text)
    slug = re.sub(r"-{2,}", "-", slug).strip("-_")
    return slug if SKILL_NAME_RE.match(slug or "") else ""


def _clean_extra_file_path(rel: str) -> Path:
    """Valida o caminho de um arquivo extra (relativo, sem traversal,
    reservado SKILL.md fora). Devolve o relativo normalizado (POSIX)."""
    p = (rel or "").strip().replace("\\", "/")
    if not p:
        raise SkillCreateError("caminho de arquivo extra vazio")
    if p.startswith("/") or re.match(r"^[A-Za-z]:", p):
        raise SkillCreateError(f"caminho absoluto não permitido: {rel!r}")
    parts = p.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise SkillCreateError(f"caminho inválido (traversal): {rel!r}")
    if parts[-1].lower() == SKILL_MD_NAME.lower() and len(parts) == 1:
        raise SkillCreateError("'SKILL.md' é o playbook — use `content`")
    return Path(*parts)


def create_agent_skill(
    root,
    name: str,
    description: str,
    content: str,
    files: Optional[dict[str, str]] = None,
    overwrite: bool = False,
) -> Path:
    """Cria a skill do AGENTE `<root>/<slug>/SKILL.md` (+ arquivos extras) e
    devolve a PASTA da skill. Validações: slug ([a-z0-9_-], kebab-case),
    description com conteúdo útil (a seleção semântica usa SÓ ela), corpo
    não-vazio, arquivos extras de texto com caminho relativo seguro. Escrita
    atômica: os extras primeiro, SKILL.md por último (tmp + os.replace) —
    uma skill nunca aparece no scan pela metade. Recusa duplicata no MESMO
    dir salvo `overwrite=True` (colisão com dir de prioridade MAIOR é da
    rota — aqui só enxerga o próprio root)."""
    slug = slugify_skill_name(name)
    if not slug:
        raise SkillCreateError(
            f"nome inválido: {name!r} — use letras/números/-/_ (kebab-case, "
            "começando com letra ou número)"
        )
    desc = (description or "").strip()
    if len(desc) < MIN_SKILL_DESCRIPTION_CHARS:
        raise SkillCreateError(
            "description muito curta — a seleção semântica usa SÓ ela: diga "
            "O QUE a skill faz e QUANDO usar (1-2 linhas)"
        )
    if len(desc) > MAX_SKILL_DESCRIPTION_CHARS:
        desc = desc[:MAX_SKILL_DESCRIPTION_CHARS].rstrip()
    body = (content or "").strip()
    if not body:
        raise SkillCreateError("content vazio — o corpo do playbook é obrigatório")
    if len(body) > MAX_SKILL_BODY_CHARS:
        raise SkillCreateError(
            f"content muito grande ({len(body)} chars; máx {MAX_SKILL_BODY_CHARS})"
        )

    extras: list[tuple[Path, str]] = []
    for rel, text in (files or {}).items():
        if not isinstance(text, str):
            raise SkillCreateError(f"arquivo extra {rel!r} deve ser texto (str)")
        rel_path = _clean_extra_file_path(str(rel))
        if not text.strip():
            raise SkillCreateError(f"arquivo extra vazio: {rel!r}")
        if len(text) > MAX_SKILL_EXTRA_FILE_CHARS:
            raise SkillCreateError(
                f"arquivo extra muito grande ({rel!r}: {len(text)} chars; "
                f"máx {MAX_SKILL_EXTRA_FILE_CHARS})"
            )
        extras.append((rel_path, text))
    if len(extras) > MAX_SKILL_EXTRA_FILES:
        raise SkillCreateError(
            f"máximo de {MAX_SKILL_EXTRA_FILES} arquivos extras (recebi {len(extras)})"
        )

    root_real = Path(os.path.realpath(root))
    skill_dir = root_real / slug
    if os.path.islink(skill_dir):
        raise SkillCreateError(f"symlink no lugar da skill: {skill_dir}")
    if skill_dir.exists():
        if not skill_dir.is_dir():
            raise SkillCreateError(f"o caminho existe e não é pasta: {skill_dir}")
        if not overwrite:
            raise SkillCreateError(
                f"skill '{slug}' já existe (reenvie com overwrite=true para substituir)",
                status=409,
            )
        shutil.rmtree(skill_dir)
    try:
        skill_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise SkillCreateError(f"falha ao criar pasta da skill: {e}")

    def _atomic_write(target: Path, text: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)

    # Extras ANTES do SKILL.md: sem SKILL.md a pasta é invisível pro scan —
    # uma falha no meio nunca expõe skill pela metade.
    base_real = Path(os.path.realpath(skill_dir))
    for rel_path, text in extras:
        target = skill_dir / rel_path
        if base_real not in Path(os.path.realpath(target)).parents:
            raise SkillCreateError(f"caminho escapa da pasta da skill: {rel_path!r}")
        _atomic_write(target, text)
    front = (
        "---\n"
        f"name: {slug}\n"
        "description: |-\n"
        + "".join(f"  {line}\n" for line in desc.splitlines())
        + "---\n\n"
    )
    _atomic_write(skill_dir / SKILL_MD_NAME, front + body + "\n")
    return skill_dir
