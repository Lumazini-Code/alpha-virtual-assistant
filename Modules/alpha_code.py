from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
import posixpath
from typing import Any, AsyncGenerator, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ══════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════

load_dotenv()

SCRAPING_URL = os.environ.get("SCRAPING_URL", "http://localhost:3005")
CLIENT_TOKEN = os.environ.get("CLIENT_TOKEN", "")

# llama-server local — único backend de inferência. Sem Groq, não há mais
# múltiplos modelos para rotacionar por rate limit (RPM/RPD/TPM/TPD): é um
# único modelo local, então essas preocupações simplesmente não se aplicam.
LLAMA_HOST = os.environ.get("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "2001"))
LLAMA_URL  = f"http://{LLAMA_HOST}:{LLAMA_PORT}"

# memory.py — serviço de memória (indexed-file/*, entre outros). Toda leitura
# de arquivo (read_file) passa a ser espelhada aqui: primeiro gravada via
# /indexed-file/write (indexada pelo caminho ABSOLUTO do arquivo, para nunca
# confundir dois arquivos de mesmo nome em pastas diferentes) e então
# recuperada de volta via /indexed-file/read para servir de fonte do
# conteúdo entregue ao modelo.
MEMORY_HOST = os.environ.get("MEMORY_HOST", "127.0.0.1")
MEMORY_PORT = int(os.environ.get("MEMORY_PORT", "3001"))
MEMORY_URL  = f"http://{MEMORY_HOST}:{MEMORY_PORT}"
MEMORY_CHUNK_TOP_K = int(os.environ.get("MEMORY_CHUNK_TOP_K", "5"))
# Acima deste tamanho, read_file NUNCA injeta o conteúdo completo do arquivo
# junto — só os trechos relevantes (chunks). Um arquivo grande inteiro
# facilmente estoura o contexto do LFM2.5-8B-A1B sozinho; a partir daqui o
# modelo precisa navegar por queries sucessivas em vez de receber tudo de
# uma vez. ~40k chars ≈ 10k tokens é uma margem segura para deixar espaço
# pro resto do prompt (histórico, outros arquivos em cache).
MEMORY_LARGE_FILE_CHARS = int(os.environ.get("MEMORY_LARGE_FILE_CHARS", "40000"))
# Quando o arquivo é grande, pedimos mais chunks que o normal — cada uma é
# pequena (CHUNK_SIZE=500 chars no memory.py) e sem o arquivo inteiro há
# espaço de sobra para várias.
MEMORY_LARGE_FILE_TOP_K = int(os.environ.get("MEMORY_LARGE_FILE_TOP_K", "12"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ALPHA-CODE] %(message)s",
)
log = logging.getLogger("ava.alpha_code")

# ══════════════════════════════════════════════════════════════════════════
# Inference parameters (per step kind)
# ══════════════════════════════════════════════════════════════════════════
# Mantidos apenas os parâmetros de geração por tipo de chamada do pipeline
# (execution/summary). Não há mais lista de modelos, fallback chains,
# ModelHealth, RATE_LIMITS ou cooldowns — tudo isso só existia para
# gerenciar os limites de uso do Groq.



TEMPERATURE_BY_KIND: dict[str, float] = {
    "execution": 0.3,
    "summary":   0.4,
}


# ══════════════════════════════════════════════════════════════════════════
# HTTP Clients
# ══════════════════════════════════════════════════════════════════════════

_llama_client: Optional[httpx.AsyncClient] = None
_scrape_client: Optional[httpx.AsyncClient] = None
_memory_client: Optional[httpx.AsyncClient] = None


def _get_llama_client() -> httpx.AsyncClient:
    global _llama_client
    if _llama_client is None or _llama_client.is_closed:
        _llama_client = httpx.AsyncClient(
            base_url=LLAMA_URL,
            timeout=httpx.Timeout(600.0, connect=5.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=6),
            headers={"Content-Type": "application/json"},
        )
    return _llama_client


def _get_scrape_client() -> httpx.AsyncClient:
    global _scrape_client
    if _scrape_client is None or _scrape_client.is_closed:
        headers = {"Authorization": f"Bearer {CLIENT_TOKEN}"} if CLIENT_TOKEN else {}
        _scrape_client = httpx.AsyncClient(
            base_url=SCRAPING_URL,
            timeout=httpx.Timeout(60.0, connect=5.0),
            headers=headers,
        )
    return _scrape_client


def _get_memory_client() -> httpx.AsyncClient:
    global _memory_client
    if _memory_client is None or _memory_client.is_closed:
        _memory_client = httpx.AsyncClient(
            base_url=MEMORY_URL,
            timeout=httpx.Timeout(30.0, connect=5.0),
            headers={"Content-Type": "application/json"},
        )
    return _memory_client


# ══════════════════════════════════════════════════════════════════════════
# memory.py — indexed-file write/read
# ══════════════════════════════════════════════════════════════════════════
# Toda leitura real de arquivo (cache miss em read_file) é espelhada para o
# serviço de memória: primeiro um /indexed-file/write (grava o conteúdo
# indexado pelo caminho ABSOLUTO), depois um /indexed-file/read por esse
# mesmo caminho para obter o conteúdo de volta. `file_path` é sempre o
# caminho absoluto — é essa chave, e não o nome do arquivo, que diferencia
# dois arquivos de mesmo nome em pastas diferentes no armazenamento.
#
# Ambas as chamadas são melhor-esforço: se o memory.py estiver fora do ar,
# o read_file continua funcionando normalmente com o conteúdo já obtido do
# scraping_client — indexar na memória nunca pode quebrar a leitura do
# arquivo em si.

async def _memory_index_file(abs_path: str, content: str) -> None:
    """Grava o conteúdo COMPLETO de `abs_path` no indexed-file store do
    memory.py, indexado pelo seu caminho absoluto."""
    if not abs_path or not content:
        return
    try:
        client = _get_memory_client()
        file_hash = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()
        r = await client.post("/indexed-file/write", json={
            "file_path": abs_path,
            "file_name": posixpath.basename(abs_path) or abs_path,
            "extension": posixpath.splitext(abs_path)[1],
            "content":   content,
            "file_hash": file_hash,
            "size":      len(content),
            "source":    "alpha_code",
        })
        r.raise_for_status()
    except Exception as e:
        log.warning(f"memory /indexed-file/write falhou para '{abs_path}': {e}")


async def _memory_read_relevant_chunks(
    abs_path: str,
    query: str,
    top_k: int = MEMORY_CHUNK_TOP_K,
    include_full_content: bool = True,
) -> tuple[Optional[str], list[dict]]:
    """Pede ao memory.py, restrito às chunks do PRÓPRIO arquivo `abs_path`
    (nunca comparado com outros arquivos do índice), as `top_k` chunks mais
    relevantes para `query` — que é o objetivo do step atual (ou a `query`
    explícita passada pelo modelo em read_file), usado como proxy de "a
    pesquisa" do modelo.

    `include_full_content=False` pede ao memory.py para NÃO devolver o
    conteúdo completo do arquivo em cada entrada — usado para arquivos
    grandes, onde só as chunks interessam e mandar o arquivo inteiro de
    volta pela rede (e depois pro contexto do modelo) seria desperdício.

    Retorna (conteúdo_completo_do_arquivo, chunks). `conteúdo_completo` vem
    None quando não há match, a chamada falha, ou `include_full_content` é
    False — nesses dois primeiros casos o chamador deve cair de volta para
    o conteúdo que já tinha do scraping_client. `chunks` é uma lista de
    {"chunk_text", "score", "chunk_index", "char_start", "char_end"}, mais
    relevante primeiro; vem vazia junto com o miss/falha.
    """
    if not abs_path or not query:
        return None, []
    try:
        client = _get_memory_client()
        r = await client.post("/indexed-file/read", json={
            "file_path":             abs_path,
            "query":                 query,
            "top_k":                 top_k,
            "include_full_content":  include_full_content,
        })
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        if not results:
            return None, []
        full_content = results[0].get("content") or None
        chunks = [
            {
                "chunk_text":  e.get("chunk_text"),
                "score":       e.get("score"),
                "chunk_index": e.get("chunk_index"),
                "chunk_id":    e.get("chunk_id"),
                "char_start":  e.get("char_start"),
                "char_end":    e.get("char_end"),
            }
            for e in results
            if e.get("chunk_text")
        ]
        return full_content, chunks
    except Exception as e:
        log.warning(f"memory /indexed-file/read (chunks) falhou para '{abs_path}': {e}")
        return None, []


# ══════════════════════════════════════════════════════════════════════════
# llama-server — chamada local de inferência
# ══════════════════════════════════════════════════════════════════════════
# Único backend de inferência do pipeline. Sem rotação de modelos, sem
# rate-limit tracking, sem fallback chain: havia só um modelo local rodando
# no llama-server. Os retries aqui cobrem apenas falhas transitórias de
# rede/5xx do processo local — não esgotamento de cota, que não existe mais.

LLAMA_MAX_RETRIES = 2
LLAMA_RETRY_BACKOFF_S = 2.0


async def _call_llama(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    sid: Optional[str] = None,
    grammar: Optional[str] = None,
) -> dict:
    """
    Chama o llama-server local (endpoint OpenAI-compatible
    /v1/chat/completions). Retorna o JSON de resposta.

    Faz um pequeno número de retries apenas para falhas transitórias de rede
    ou 5xx — não existe mais lógica de rate limit / rotação de modelo, já
    que há um único modelo local servido pelo llama-server.

    MODO GRAMMAR (otimização extrema):
      Se `grammar` (string GBNF) for fornecido, o payload NÃO inclui `tools`/
      `tool_choice` — desabilita function-calling nativo do OpenAI, que
      consome ~500-1000 tokens de schemas por request e é a fonte #1 de JSON
      malformado em modelos locais. Em vez disso, a grammar força o output a
      seguir um formato JSON compacto (definido pelo caller), e a resposta
      vem em `choices[0].message.content` — não em `message.tool_calls`.
      `tools` é IGNORADO mesmo se fornecido (mutuamente exclusivo).
    """
    client = _get_llama_client()
    payload: dict = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # ── Modo grammar vs. modo tool-calling nativo (mutuamente exclusivos) ──
    # Quando grammar está presente, não enviamos `tools` — o llama-server
    # aplica a grammar no sampler e o output é puro texto em `content`.
    if grammar:
        payload["grammar"] = grammar
    elif tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    tag = f"[{sid[:8]}] " if sid else ""
    last_error: Optional[Exception] = None

    for attempt in range(LLAMA_MAX_RETRIES + 1):
        try:
            log.info(f"{tag}Calling llama-server (attempt {attempt + 1}/{LLAMA_MAX_RETRIES + 1})")
            r = await client.post("/v1/chat/completions", json=payload)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            last_error = e
            log.error(f"{tag}llama-server network error: {type(e).__name__}: {e}")
            if attempt < LLAMA_MAX_RETRIES:
                await asyncio.sleep(LLAMA_RETRY_BACKOFF_S)
                continue
            raise RuntimeError(
                f"llama-server unreachable at {LLAMA_URL} — make sure it is running: {e}"
            ) from e

        if r.status_code >= 500:
            log.warning(f"{tag}llama-server {r.status_code} (attempt {attempt + 1}/{LLAMA_MAX_RETRIES + 1})")
            if attempt < LLAMA_MAX_RETRIES:
                await asyncio.sleep(LLAMA_RETRY_BACKOFF_S)
                continue
            raise RuntimeError(f"llama-server {r.status_code} persistente: {r.text[:500]}")

        if r.status_code != 200:
            detail = r.text[:500]
            raise RuntimeError(f"llama-server {r.status_code}: {detail}")

        return r.json()

    raise RuntimeError(f"llama-server: falhou após {LLAMA_MAX_RETRIES + 1} tentativa(s): {last_error}")



# ══════════════════════════════════════════════════════════════════════════
# System-reminder wrapping (adapted from OpenClaude's planMode.ts)
# ══════════════════════════════════════════════════════════════════════════
#
# OpenClaude wraps every synthetically-injected "user" message (plan-mode
# instructions, mode reminders) in a `<system-reminder>` tag before adding it
# to the conversation. The point isn't styling — it's disambiguation: without
# a marker, a message with role "user" reads to the model exactly like
# something the human typed, and it will treat a nudge like "you've read 3
# files in a row, make an edit now" as a new instruction from the person
# rather than as tooling talking to it. That distinction matters more here
# than in a chat UI, because our injected messages are corrective/steering
# (read-loop nudges, iteration warnings) — the model needs to know they came
# from the harness, not from the user overriding the original task.
def _wrap_reminder(content: str) -> str:
    return f"<system-reminder>\n{content}\n</system-reminder>"


# ══════════════════════════════════════════════════════════════════════════
# Destructive command safety gate (adapted from OpenClaude's Auto Mode rule:
# "Auto mode is not a license to destroy. Anything that deletes data or
# modifies shared or production systems still needs explicit user
# confirmation.")
# ══════════════════════════════════════════════════════════════════════════
#
# alpha_code runs autonomously end-to-end with no human in the loop per
# tool call (unlike Claude Code, which asks for per-command approval in the
# terminal). execute_command is the one tool that can reach past the
# project directory entirely — a normal edit is bounded by str_replace's
# file-path argument, but a shell command is not. Before running anything
# that looks destructive, block it and require the model to escalate to the
# user via ask_user_question instead of just doing it.
_DESTRUCTIVE_COMMAND_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\brm\s+-[a-z]*r[a-z]*f|\brm\s+-[a-z]*f[a-z]*r", re.IGNORECASE), "recursive forced delete (rm -rf)"),
    (re.compile(r"\bgit\s+push\s+.*--force|\bgit\s+push\s+.*-f\b", re.IGNORECASE), "force push (can overwrite remote history)"),
    (re.compile(r"\bgit\s+reset\s+--hard", re.IGNORECASE), "hard reset (discards uncommitted work)"),
    (re.compile(r"\bgit\s+clean\s+-[a-z]*d[a-z]*f|\bgit\s+clean\s+-[a-z]*f[a-z]*d", re.IGNORECASE), "git clean -df (deletes untracked files)"),
    (re.compile(r"\bdrop\s+(table|database|schema)\b", re.IGNORECASE), "SQL DROP (irreversible data loss)"),
    (re.compile(r"\btruncate\s+table\b", re.IGNORECASE), "SQL TRUNCATE (irreversible data loss)"),
    (re.compile(r"\bmkfs\b|\bdd\s+if=.*of=/dev/", re.IGNORECASE), "disk/filesystem-level destructive command"),
    (re.compile(r">\s*/dev/sd[a-z]"), "raw write to a disk device"),
    (re.compile(r"\bchmod\s+-R\s+777\s+/|\bchown\s+-R\s+.*\s+/(\s|$)", re.IGNORECASE), "recursive permission change at filesystem root"),
    (re.compile(r"\bkubectl\s+delete\b", re.IGNORECASE), "kubectl delete (can remove live cluster resources)"),
    (re.compile(r"\bdocker\s+(system\s+prune|volume\s+prune|rmi)\b.*-f\b|\bdocker\s+(system\s+prune|volume\s+prune)\b", re.IGNORECASE), "docker prune/remove (deletes containers, volumes, or images)"),
    (re.compile(r"\bnpm\s+publish\b|\byarn\s+publish\b|\btwine\s+upload\b|\bpip\s+.*upload\b", re.IGNORECASE), "package publish (public, hard to undo)"),
    (re.compile(r"\bterraform\s+destroy\b", re.IGNORECASE), "terraform destroy (tears down provisioned infrastructure)"),
]


def _check_destructive_command(command: str) -> Optional[str]:
    """Returns a human-readable reason if `command` matches a known
    destructive pattern, else None."""
    for pattern, reason in _DESTRUCTIVE_COMMAND_PATTERNS:
        if pattern.search(command):
            return reason
    return None


# ══════════════════════════════════════════════════════════════════════════
# Tool Definitions (English — for the model)
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a software engineering agent. You receive a task in natural language "
    "and execute it by editing files through the available tools.\n\n"
    "OUTPUT FORMAT — PYTHONIC FUNCTION CALL (grammar-enforced):\n"
    "You MUST respond with a SINGLE Python-style function call in this exact shape:\n"
    "  tool_name(arg1=\"value1\", arg2=\"value2\")\n"
    "This is your native tool-calling format — use it exactly as you would normally.\n"
    "Do NOT add any text before or after the call. Do NOT wrap it in markdown fences. "
    "Do NOT wrap it in a list or in <|tool_call_start|>/<|tool_call_end|> tokens — just the bare call.\n"
    "The output is validated by a grammar on the server — any deviation is rejected.\n"
    "Call exactly ONE tool per turn. After receiving the tool result, call the next tool "
    "or call `finish(summary=\"...\")` to end the step.\n\n"
    "AVAILABLE TOOLS:\n"
    "  - list_files(path=\"...\", pattern=\"...\", recursive=True|False)  — list directory contents "
    "(pass a directory, NOT a file). path defaults to \".\"; pattern and recursive are optional.\n"
    "  - read_file(file_path=\"...\", query=\"...\")   — read a file. file_path is required, "
    "query is optional. Small/medium files come back in full. LARGE files NEVER come back "
    "in full — you only get the most relevant excerpts, so `query` is how you steer which "
    "part of the file you see. Call it again with a different `query` on the same file_path "
    "to fetch a different section.\n"
    "  - create_file(file_path=\"...\", content=\"...\") — create a NEW file or overwrite an EMPTY file. "
    "Refused on non-empty files. Both args required.\n"
    "  - str_replace(file_path=\"...\", old_str=\"...\", new_str=\"...\", replace_all=True|False) — replace "
    "an exact text snippet in an existing file. replace_all is optional (default False).\n"
    "  - execute_command(command=\"...\") — run a shell command (destructive commands are blocked).\n"
    "  - web_search(query=\"...\")  — search the web for external info.\n"
    "  - ask_user_question(question=\"...\") — pause and ask the user a clarifying question.\n"
    "  - finish(summary=\"...\")      — stop calling tools and produce a text summary of what was done "
    "and which files were changed.\n\n"
    "ESCAPING IN STRING LITERALS (CRITICAL):\n"
    "- Newlines inside a string value MUST be written as \\n.\n"
    "- Double quotes inside a string value MUST be written as \\\".\n"
    "- Backslashes MUST be written as \\\\.\n"
    "- All string arguments use double quotes (\"...\"), same as Python double-quoted strings.\n"
    "- The grammar enforces these rules — a malformed call is impossible, but if your "
    "old_str/new_str does not match the file byte-for-byte, the edit will fail and "
    "you will have to retry. Copy snippets exactly from the read_file output.\n\n"
    "RULES:\n"
    "- ALWAYS use list_files/read_file to understand the code BEFORE editing.\n"
    "- Use `list_files` ONLY to list directories. To read file contents, use `read_file`.\n"
    "- NEVER guess or invent a file path. Use exactly the paths returned by list_files.\n"
    "- To edit an existing file with content, ALWAYS use `str_replace` (old_str must be "
    "an exact, unique snippet from the file). `create_file` ONLY works for new/empty files.\n"
    "- If a tool call fails, do NOT repeat the same call — read the error, adjust your approach, "
    "or call `finish` with an explanation.\n"
    "- Use `execute_command` to run tests, linters, builds, or git when it makes sense.\n"
    "- Use `web_search` when you need external information not present in the project's own files.\n"
    "- When the step is complete, call `finish` with a summary. Do NOT keep calling tools.\n\n"
    "PREFER ACTION, BUT ESCALATE REAL AMBIGUITY:\n"
    "- Make reasonable assumptions and proceed for routine decisions (naming, formatting, "
    "where to put a helper). Do not stop to ask about things you could resolve by reading the code.\n"
    "- Use `ask_user_question` ONLY when the step is genuinely ambiguous in a way that reading "
    "code cannot resolve. This pauses the whole task until the user answers, so use it sparingly.\n"
    "- Destructive or irreversible operations (force-push, hard reset, dropping data, deleting "
    "cloud/cluster resources, publishing packages) are blocked automatically — if one is truly "
    "required, use `ask_user_question` to get explicit confirmation before attempting it again.\n\n"
    "CRITICAL EFFICIENCY RULES — AVOID RE-READING FILES:\n"
    "- If a file's content is already shown in the conversation (in 'Previously read files' "
    "or a previous tool result), do NOT call read_file on it again. The cache returns a stub.\n"
    "- Read each file ONCE, then make ALL your edits. Do not read → edit → re-read → edit.\n"
    "- If you need to edit multiple parts of the same file, make multiple str_replace calls "
    "in sequence without re-reading between them.\n"
    "- EXCEPTION for LARGE files: if read_file tells you it only showed a few excerpts out of "
    "a much bigger file, and none of them cover what you need, calling read_file again on the "
    "SAME file_path with a different `query` is expected and correct — that is not the "
    "re-reading this rule warns against.\n\n"
    "CRITICAL str_replace RULES:\n"
    "- old_str MUST be an EXACT copy of text from the file, including ALL whitespace, "
    "indentation, and line breaks.\n"
    "- On a LARGE file where you only saw excerpts, old_str MUST be copied from one of the "
    "excerpts you actually saw — never guess or reconstruct text from a part of the file you "
    "have not been shown.\n"
    "- new_str MUST include proper line breaks (\\n in the JSON string) between lines.\n"
    "  NEVER concatenate multiple lines or imports into a single line without newlines.\n"
    "- Make SMALL, targeted replacements (1-5 lines) rather than replacing large blocks.\n"
    "- After each str_replace, the file is automatically validated for Python syntax. "
    "If a syntax error is detected, the edit is REVERTED and you must try again.\n"
    "- NEVER add duplicate imports. Check the existing imports in the file first.\n\n"
    "LARGE FILE EDITS:\n"
    "- For LARGE files where you only saw excerpts, use replace_chunk(chunk_id=N, new_content=...) "
    "instead of str_replace. The chunk_id is shown in the read_file output. Copy the chunk content "
    "you saw, modify the relevant lines, and pass the full new content. This avoids the need to "
    "reproduce old_str exactly.\n"
)

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories inside a project path. Use this to discover the project structure before reading or editing files. Pass a directory path, NOT a file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list. Use '.' for the project root.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to filter results, e.g. '*.py'. Default is '*'.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Search recursively into subdirectories. Default is true.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file in the project. Use this to understand existing code before making edits. The file_path must be relative to the project root. For small/medium files you get the full content back. For large files, only the most relevant excerpts are returned (never the full content) — use `query` to steer which excerpts you get; call read_file again with a different `query` on the same file_path to fetch other parts of it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path of the file to read, relative to the project root.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional. What you're looking for in this file (a function name, a class, a behavior, an error message). Steers which part of the file you get back — most useful (and, for large files, the ONLY way to target a specific section) when the file is too big to return in full.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a NEW file, or write to an existing file that is EMPTY. Does NOT work on existing files with content — for those, use str_replace instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path of the file to create, relative to the project root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full content to write to the file.",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": "Replace an exact text snippet (old_str) with new_str inside an existing file. old_str must appear exactly as-is in the file — use this for targeted edits instead of rewriting the entire file. Copy the exact text from a read_file result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path of the file to edit, relative to the project root.",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "The exact text snippet to find and replace. Must be unique in the file.",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "The replacement text.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences of old_str. Default is false.",
                    },
                },
                "required": ["file_path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Execute a shell command in the project directory. Use for running tests, builds, git operations, linters, or any validation commands. Commands that look destructive or irreversible (force-push, hard reset, dropping data, deleting cloud resources, publishing packages, etc.) are blocked — use ask_user_question to get explicit confirmation first if one is truly needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for external information (library docs, API references, error messages, package versions, best practices) needed to complete the task. Use only when the information isn't available by reading the project's own files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user_question",
            "description": "Pause the task and ask the user a clarifying question. Use ONLY for genuine ambiguity that reading code cannot resolve, or to confirm a destructive/irreversible action that was blocked. This suspends the whole task until the user replies, so it is expensive — never use it for something discoverable by reading the code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "A specific, self-contained question for the user. Include the relevant options or tradeoffs if there are any.",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
    "type": "function",
    "function": {
        "name": "replace_chunk",
        "description": "Replace the content of a specific chunk (identified by chunk_id) in a large file. Use this instead of str_replace when you only saw excerpts of a large file — reference the chunk_id from the read_file output instead of reproducing old_str.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "chunk_id": {"type": "integer", "description": "The chunk_id from the read_file output"},
                "new_content": {"type": "string", "description": "The new content for this chunk's character range"},
            },
            "required": ["file_path", "chunk_id", "new_content"],
        },
    },
}

]




USE_GRAMMAR_MODE: bool = True


TOOL_GRAMMAR: str = r''' 
root ::= ws call ws
 
call ::= "list_files(" ws list-args ws ")" | "read_file(" ws read-args ws ")" | "create_file(" ws create-args ws ")" | "str_replace(" ws str-args ws ")" | "execute_command(" ws exec-args ws ")" | "web_search(" ws web-args ws ")" | "ask_user_question(" ws ask-args ws ")" | "finish(" ws finish-args ws ")"
 
list-args   ::= ("path=" string (ws "," ws "pattern=" string)? (ws "," ws "recursive=" bool)?)?
 
read-args   ::= "file_path=" string (ws "," ws "query=" string)?
 
create-args ::= "file_path=" string ws "," ws "content=" string
 
str-args    ::= "file_path=" string ws "," ws "old_str=" string ws "," ws "new_str=" string (ws "," ws "replace_all=" bool)?
 
exec-args   ::= "command=" string
 
web-args    ::= "query=" string
 
ask-args    ::= "question=" string
 
finish-args ::= "summary=" string
 
string ::= "\"" ( [^"\\\n] | "\\" ["\\/bfnrtu] )* "\""
bool   ::= "True" | "False"
ws     ::= [ \t\n]*
'''


def _parse_tool_call_from_content(content: str) -> Optional[dict]:
    """
    Converte o `content` Pythônico gerado pelo modelo (sob grammar) em um
    tool_call sintético no formato OpenAI, mantendo compatibilidade com o
    resto do pipeline (_run_tool, tracking de loops, step_messages history,
    etc.) — a interface pública desta função (dict retornado) não mudou,
    só a sintaxe que ela sabe interpretar (Pythônica em vez de JSON-envelope).

    Formato esperado do content (imposto pela TOOL_GRAMMAR):
        tool_name(arg1="valor1", arg2="valor2")
    Ou seja: exatamente a sintaxe de chamada de função nativa do LFM2.5 —
    o mesmo formato que o model card documenta dentro de
    `<|tool_call_start|>[...]<|tool_call_end|>`, só que sem o wrapper de
    tokens especiais (não precisamos deles: nós mesmos parseamos o texto,
    não dependemos do parser nativo do llama-server).

    Formato de saída (igual ao que o llama-server retornaria em modo nativo):
        {
            "id": "call_<hex12>",
            "type": "function",
            "function": {"name": "<tool>", "arguments": "<json-string>"}
        }

    Retorna None se o content não for parseável. Sob grammar ativa, isso NÃO
    deveria acontecer — a grammar garante validade estrutural no nível do
    sampler. O None é tratado como fallback para o caminho antigo (tool_calls
    nativos) no _execute_step.

    Por que `ast.parse` em vez de regex: o content é, por construção da
    grammar, uma expressão Python válida de chamada de função com kwargs
    literais (strings/bools) — exatamente o que `ast.parse(..., mode="eval")`
    foi feito para interpretar. Isso nos dá escaping correto (\\n, \\", \\\\)
    de graça, sem regex frágil, e SEM usar `eval()`: `ast.literal_eval` só
    aceita literais (str/bool/num/None/list/dict/tuple), nunca executa código.

    Defensive parsing: a grammar já força a sintaxe pura, mas alguns
    templates de chat podem envolver o output em ```python ... ``` fences,
    em colchetes `[...]` (formato de lista do model card), ou nos tokens
    especiais `<|tool_call_start|>`/`<|tool_call_end|>`. Removemos tudo isso
    antes do ast.parse para tolerar esses casos.
    """
    if not content or not content.strip():
        return None

    text = content.strip()

    # Defensive: remove markdown fences (alguns templates adicionam apesar da grammar)
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    # Defensive: remove o wrapper de tokens especiais do model card do LFM2.5,
    # caso algum template os insira como texto literal apesar da grammar.
    text = text.replace("<|tool_call_start|>", "").replace("<|tool_call_end|>", "").strip()

    # Defensive: o model card do LFM2.5 mostra o call dentro de uma lista
    # `[tool(...)]` (permite múltiplas chamadas em paralelo). Nossa grammar
    # gera um call bare, mas se um template envolver em colchetes mesmo
    # assim, desembrulhamos aqui.
    if text.startswith("[") and text.endswith(")]"):
        text = text[1:-1].strip()

    try:
        node = ast.parse(text, mode="eval").body
    except (SyntaxError, ValueError):
        return None

    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None

    name = node.func.id
    if not name:
        return None

    args: dict = {}
    try:
        for kw in node.keywords:
            if kw.arg is None:
                # **kwargs expansion — não deveria ocorrer sob a grammar, ignora
                continue
            args[kw.arg] = ast.literal_eval(kw.value)
    except (ValueError, SyntaxError):
        return None

    # Sintetiza um tool_call_id (em modo nativo o llama-server gera isso;
    # em modo grammar a aplicação precisa gerar para casar role=tool no histórico)
    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        },
    }




# ══════════════════════════════════════════════════════════════════════════
# Tool Execution (scraping_client calls)
# ══════════════════════════════════════════════════════════════════════════

# ── Post-edit Python syntax validation ──────────────────────────────────────

def _check_python_syntax(content: str) -> Optional[str]:
    """Check Python syntax. Returns error message or None if OK."""
    try:
        compile(content, "<file>", "exec")
        return None
    except SyntaxError as e:
        line_info = f"line {e.lineno}" if e.lineno else "unknown line"
        return f"SyntaxError: {e.msg} ({line_info})"


# ── Near-match context for failed str_replace ───────────────────────────────
# Adapted from readEditContext.ts. That version streams a file from disk in
# 8KB chunks looking for an exact byte match and returns a small window of
# real content around it. We don't have direct filesystem access here (only
# the scraping_client HTTP API), so this operates on a string already in
# memory — but the goal is identical: when old_str doesn't match exactly,
# don't force the model to call read_file on the WHOLE file just to find out
# why. Most str_replace failures are a whitespace/indentation mismatch, not
# "this content doesn't exist" — so find where the text is CLOSE and hand
# back a few lines of real, current content around that spot. That's often
# enough to fix old_str character-by-character without another full read,
# which matters because "str_replace fails -> full re-read -> fails again"
# is one of the concrete ways this agent ends up looping without progress.
NEAR_MATCH_CONTEXT_LINES = 3
NEAR_MATCH_MIN_ANCHOR_LEN = 6


def _find_near_match_context(
    content: str, needle: str, context_lines: int = NEAR_MATCH_CONTEXT_LINES
) -> Optional[tuple[str, int]]:
    """
    Looks for a line of `needle` that appears (whitespace-insensitively) in
    `content`, and if found, returns (snippet, start_line) — start_line is
    1-based. Returns None if nothing in `needle` matches anywhere, which
    means old_str likely isn't just slightly off; it may not be in this
    file at all, and a full read_file is the right move.
    """
    if not needle.strip() or not content:
        return None

    file_lines = content.split("\n")
    stripped_file_lines = [ln.strip() for ln in file_lines]

    for needle_line in needle.split("\n"):
        candidate = needle_line.strip()
        if len(candidate) < NEAR_MATCH_MIN_ANCHOR_LEN:
            continue  # too short/generic (e.g. "}", "") to anchor a search
        for idx, fl in enumerate(stripped_file_lines):
            if candidate == fl or (len(candidate) >= 12 and candidate in fl):
                start = max(0, idx - context_lines)
                end = min(len(file_lines), idx + context_lines + 1)
                snippet = "\n".join(file_lines[start:end])
                return snippet, start + 1

    return None


async def _validate_edit_and_revert(
    client: httpx.AsyncClient,
    file_path: str,
    snapshot_content: str,
    file_cache: dict[str, str],
) -> tuple[bool, str]:
    """
    After a str_replace or write_file on a .py file, read the result and
    check for Python syntax errors. If broken, revert to the snapshot.
    Returns (is_valid, message).

    Revert strategy (in order):
      1. write-file with force=True (bypasses content check)
      2. str-replace the entire file content back to snapshot
      3. If both fail, log critical error and return the broken content
         so the model can see the damage and fix it.
    """
    # Only validate Python files
    if not file_path.lower().endswith(".py"):
        return True, ""

    # Read the file after the edit
    try:
        r = await client.post("/read-file", json={"file_path": file_path})
        if r.status_code != 200:
            return True, ""  # Can't read — assume OK
        new_content = r.json().get("content", "")
    except Exception:
        return True, ""  # Can't read — assume OK

    # Check syntax
    syntax_err = _check_python_syntax(new_content)
    if syntax_err is None:
        # File is valid — update cache
        _cache_file(file_cache, file_path, new_content)
        return True, ""

    # ── Syntax error — revert to snapshot ──────────────────────────────
    log.warning(f"Post-edit syntax error in '{file_path}': {syntax_err} — reverting")

    # Strategy 1: write-file with force=True
    reverted = False
    revert_detail = ""
    try:
        r2 = await client.post("/write-file", json={
            "file_path": file_path,
            "content": snapshot_content,
            "force": True,
        })
        if r2.status_code == 200:
            log.info(f"Reverted '{file_path}' to pre-edit snapshot (write-file)")
            reverted = True
        else:
            revert_detail = f"write-file returned {r2.status_code}: {r2.text[:200]}"
            log.warning(f"write-file revert failed for '{file_path}': {revert_detail}")
    except Exception as e:
        revert_detail = f"write-file exception: {e}"
        log.warning(f"write-file revert exception for '{file_path}': {e}")

    # Strategy 2: str-replace the entire broken content back to snapshot
    if not reverted and new_content.strip():
        try:
            # Use the first 200 chars as old_str to find a unique match
            # If the entire file is different, use the whole content
            r3 = await client.post("/str-replace", json={
                "file_path": file_path,
                "old_str": new_content,
                "new_str": snapshot_content,
            })
            if r3.status_code == 200:
                log.info(f"Reverted '{file_path}' to pre-edit snapshot (str-replace)")
                reverted = True
            else:
                revert_detail += f" | str-replace returned {r3.status_code}: {r3.text[:200]}"
                log.warning(f"str-replace revert failed for '{file_path}': {r3.text[:200]}")
        except Exception as e:
            revert_detail += f" | str-replace exception: {e}"
            log.warning(f"str-replace revert exception for '{file_path}': {e}")

    # Strategy 3: write-file WITHOUT force (some scraping_clients may not support it)
    if not reverted:
        try:
            r4 = await client.post("/write-file", json={
                "file_path": file_path,
                "content": snapshot_content,
            })
            if r4.status_code == 200:
                log.info(f"Reverted '{file_path}' to pre-edit snapshot (write-file no-force)")
                reverted = True
            else:
                revert_detail += f" | write-file(no-force) returned {r4.status_code}"
        except Exception as e:
            revert_detail += f" | write-file(no-force) exception: {e}"

    # Invalidate cache regardless — content changed
    file_cache.pop(file_path, None)

    if reverted:
        _cache_file(file_cache, file_path, snapshot_content)
        return False, (
            f"EDIT REVERTED: The edit caused a Python syntax error: {syntax_err}. "
            f"The file has been restored to its previous state.\n\n"
            f"COMMON CAUSES:\n"
            f"  - Missing line breaks (\\n) between imports or statements\n"
            f"  - Concatenated lines like 'from X import Yfrom Z import W'\n"
            f"  - Duplicate or missing imports\n"
            f"  - Broken indentation\n\n"
            f"WHAT TO DO:\n"
            f"  1. Run read_file on '{file_path}' to see the current (restored) content\n"
            f"  2. Make a SMALLER, more precise str_replace\n"
            f"  3. Ensure new_str has proper line breaks between every line\n"
            f"  4. Do NOT add duplicate imports"
        )
    else:
        log.error(f"ALL REVERT STRATEGIES FAILED for '{file_path}': {revert_detail}")
        return False, (
            f"EDIT REVERT FAILED: The edit caused a syntax error ({syntax_err}) "
            f"and ALL revert strategies failed ({revert_detail}).\n\n"
            f"CRITICAL: The file '{file_path}' is now BROKEN. You MUST fix it immediately:\n"
            f"  1. Run read_file on '{file_path}' to see the broken content\n"
            f"  2. Use create_file to replace the ENTIRE file with correct content\n"
            f"  3. The file should start with the original imports and end with the app routes\n"
            f"  4. Do NOT attempt another str_replace until the file is fixed"
        )


MAX_TOOL_OUTPUT_CHARS = 2000

# ── File content cache ─────────────────────────────────────────────────────────
# Caches file contents read by the agent so subsequent steps don't need to
# re-read the same file. This dramatically reduces the number of API calls.
#
# IMPORTANT: this cache is intentionally NOT a module-level global. It used to
# be (`_file_cache: dict[str, str] = {}` at module scope, cleared with
# `_file_cache.clear()` at the start of `_run_pipeline`), but since this is an
# async FastAPI server, concurrent requests interleave on the same event loop.
# Two tasks running at the same time shared the exact same dict, so:
#   - request B's `_file_cache.clear()` could wipe request A's cache mid-run
#   - request A could get "cache hits" containing file content read by an
#     unrelated request B (different project/session), causing the model to
#     believe it already knows a file's content when it doesn't
#   - str_replace then fails against the *real* file (old_str not found),
#     the model tries again with a slightly different snippet, that fails
#     too, and the step spends its whole iteration budget failing to
#     match content that was never actually correct — one of the ways this
#     agent could "read files in circles" without ever changing anything.
#
# Instead, each call to `_run_pipeline` creates its own local dict and passes
# it down through `_execute_step` -> `_run_tool` / `_validate_edit_and_revert`.
#
# Bounded like conversationCache.ts's ConversationCache: unbounded growth is
# a real risk here too — a task that touches hundreds of files in a large
# monorepo would otherwise keep every single one in memory (and re-inject
# candidates for the prompt) for the whole run. We don't need a separate
# monotonic clock like that file uses (its plain `accessOrder.size` bug is
# irrelevant here) because a Python dict already preserves insertion order:
# deleting and re-inserting a key moves it to the end, which gives us "most
# recently touched last" for free on both writes and cache hits.
FILE_CACHE_MAX_ENTRIES = 300


def _cache_file(cache: dict[str, str], path: str, content: str) -> None:
    """Store file content in the given cache, bumping it to most-recent."""
    if not content or not path:
        return
    if path in cache:
        del cache[path]
    cache[path] = content
    # Evict the least-recently-touched entries once we're over budget.
    while len(cache) > FILE_CACHE_MAX_ENTRIES:
        oldest_key = next(iter(cache))
        del cache[oldest_key]


def _get_cached_file(cache: dict[str, str], path: str) -> Optional[str]:
    """Get cached file content, bumping it to most-recent on hit."""
    if path not in cache:
        return None
    content = cache[path]
    del cache[path]
    cache[path] = content
    return content

# Path normalization helpers
def _norm(p: str) -> str:
    if not p:
        return ""
    p = p.strip()
    p = posixpath.normpath(p)
    if p == ".":
        return ""
    if p.startswith("./"):
        p = p[2:]
    return p.lower()


def _register_path(known_paths: dict[str, str], path: str) -> None:
    if not path:
        return
    key = _norm(path)
    pretty = posixpath.normpath(path)
    if pretty.startswith("./"):
        pretty = pretty[2:]
    known_paths[key] = pretty


def _resolve_path(known_paths: dict[str, str], requested: str) -> tuple[Optional[str], Optional[str]]:
    if not requested:
        return None, None
    if not known_paths:
        return requested, None

    key = _norm(requested)
    if key in known_paths:
        real = known_paths[key]
        requested_norm = posixpath.normpath(requested)
        if requested_norm.startswith("./"):
            requested_norm = requested_norm[2:]
        if real != requested_norm:
            return real, f"[path corrected from '{requested}' to '{real}']"
        return real, None

    # Check if it's a parent directory of known paths
    for known_key in known_paths:
        if known_key.startswith(key + "/"):
            return requested, None

    return None, None


async def _run_tool(
    name: str,
    args: dict,
    known_paths: dict[str, str],
    file_cache: dict[str, str],
    cache_shown_in_prompt: set[str],
    edit_failures: Optional[dict[str, int]] = None,
    search_query: str = "",
) -> tuple[bool, str]:
    """Executes a tool by calling the scraping_client. Returns (success, output).

    Em modo grammar, o nome da tool vem do JSON sintetizado pelo parser
    (_parse_tool_call_from_content), não do tool_calls nativo do llama-server.
    Os nomes aceitos incluem `create_file` (nome canônico) e `write_file`
    (alias mantido para compat com chamadas legadas e para o caminho fallback
    com tool_calls nativos). A tool virtual `finish` é interceptada pelo
    _execute_step antes de chegar aqui — se chegar, retorna sucesso sem ação.

    `search_query` é a ação do step atual (o objetivo que o modelo está
    tentando cumprir agora) — usada como proxy de "a pesquisa do modelo" ao
    pedir ao memory.py os trechos mais relevantes de um arquivo lido via
    `read_file` (ver `_memory_read_relevant_chunks`).
    """
    # Guarda defensiva: finish não deveria chegar aqui. O orchestrator
    # intercepta essa tool antes do loop de execução.
    if name == "finish":
        return True, "(finish handled by orchestrator — should not reach _run_tool)"
    client = _get_scrape_client()
    try:
        if name == "list_files":
            r = await client.post("/list-files", json={
                "path": args.get("path", "."),
                "pattern": args.get("pattern", "*"),
                "recursive": args.get("recursive", True),
            })
            if r.status_code == 400 and "não é diretório" in r.text.lower():
                bad_path = args.get("path", ".")
                return False, (
                    f"ERROR: '{bad_path}' is a file, not a directory. "
                    f"Use `read_file` to read file contents, not `list_files`."
                )
            r.raise_for_status()
            data = r.json()
            for e in data["entries"]:
                _register_path(known_paths, e["path"])
            lines = [f"{'[D]' if e['is_dir'] else '[F]'} {e['path']}" for e in data["entries"]]
            out = "\n".join(lines) or "(empty directory)"
            if data.get("truncated"):
                out += "\n[...list truncated...]"
            return True, out

        if name == "read_file":
            resolved, note = _resolve_path(known_paths, args["file_path"])
            if resolved is None:
                return False, (
                    f"Path '{args['file_path']}' not found in current project listing. "
                    f"Do NOT guess paths — run list_files to confirm the exact path."
                )
            # ── CACHE HIT: don't re-send full content, send a short stub ──
            # The full content is already injected at the top of THIS step's
            # prompt (see the "Previously read files" block built in
            # _execute_step), so echoing it again here would be a second
            # full copy for zero benefit — the same waste Claude Code's Read
            # tool fixes with its own dedup stub (`file_unchanged`) for
            # repeated same-range reads of an unmodified file.
            cached = _get_cached_file(file_cache, resolved)
            if cached is not None:
                log.info(f"read_file cache hit for '{resolved}' — skipping API call")
                if resolved in cache_shown_in_prompt:
                    # Content is already visible above in "Previously read
                    # files" — a stub is enough, avoids a second full copy.
                    body = (
                        f"[FILE UNCHANGED] '{resolved}' was already read earlier in this task "
                        f"and has not been modified since. Its content is shown under "
                        f"'Previously read files' at the top of this step's instructions — "
                        f"use that content directly instead of reading it again."
                    )
                else:
                    # It's cached but got trimmed out of the prompt injection
                    # for budget reasons — the model has NOT actually seen
                    # this content yet this step, so it needs the real thing.
                    body = f"[From cache — file unchanged since last read]\n\n{cached}"
                if note:
                    body = f"{note}\n\n{body}"
                return True, body
            # ── CACHE MISS: fetch from scraping_client ──
            r = await client.post("/read-file", json={"file_path": resolved})
            r.raise_for_status()
            data = r.json()
            raw_content = data["content"]

            # ── Memory mirror: grava no indexed-file store (chave = caminho
            # absoluto). O `resolved` aqui é o caminho absoluto no host (é o
            # que o scraping_client usa e o que known_paths registra), então
            # é essa string — não apenas o nome do arquivo — que vai para a
            # memória, evitando colisão entre arquivos homônimos em pastas
            # diferentes.
            await _memory_index_file(resolved, raw_content)

            # `query` explícita do modelo tem prioridade sobre a ação do
            # step (search_query) — deixa o modelo mirar exatamente o que
            # precisa, especialmente importante em arquivos grandes onde
            # diferentes chamadas de read_file no mesmo step podem querer
            # partes bem diferentes do arquivo.
            effective_query = (args.get("query") or search_query or "").strip()

            is_large_file = len(raw_content) > MEMORY_LARGE_FILE_CHARS

            if is_large_file:
                # ── Arquivo grande: NUNCA manda o conteúdo inteiro junto.
                # Só chunks — e pede pro memory.py nem devolver `content`
                # no payload (include_full_content=False), pra não gastar
                # banda/contexto à toa com algo que vamos descartar mesmo.
                mem_content, chunks = await _memory_read_relevant_chunks(
                    resolved, effective_query,
                    top_k=MEMORY_LARGE_FILE_TOP_K,
                    include_full_content=False,
                )
                if chunks:
                    lines = [
                        f"[FILE TOO LARGE — showing only the {len(chunks)} most relevant "
                        f"excerpts out of {len(raw_content)} chars total. The excerpts below "
                        f"are the ONLY parts of this file you can see and copy from right now.]",
                        "",
                        "esses são os trechos do código mais relevantes de acordo com a sua pesquisa:",
                        "",
                    ]
                    for i, c in enumerate(chunks, 1):
                        score = c.get("score")
                        pos = (
                            f", chars {c['char_start']}-{c['char_end']}"
                            if c.get("char_start") is not None else ""
                        )
                        score_str = f" (score {score}{pos})" if score is not None else ""
                        lines.append(f"--- trecho {i} [chunk_id={c['chunk_id']}, score {score_str}] ---")
                        lines.append(c["chunk_text"])
                        lines.append("")
                    lines.append(
                        "If none of these excerpts cover the part of the file you need to "
                        "edit, call read_file again on the SAME file_path with a more "
                        "specific `query` describing exactly what you're looking for "
                        "(e.g. a function name, a class, an error message) — this will "
                        "fetch a different set of excerpts from the same file. "
                        "For str_replace, old_str must be copied EXACTLY from an excerpt "
                        "you have actually seen — never guess text outside of what's shown."
                    )
                    body = "\n".join(lines)
                else:
                    # Sem query/sem match/memory fora do ar — não há como
                    # mostrar o arquivo inteiro (é grande demais), então
                    # pedimos explicitamente ao modelo para se orientar com
                    # uma query em vez de truncar arbitrariamente o começo
                    # do arquivo (que raramente é a parte relevante).
                    body = (
                        f"[FILE TOO LARGE — {len(raw_content)} chars, too big to show in "
                        f"full. Call read_file again on this same file_path with a `query` "
                        f"argument describing what you're looking for (a function name, a "
                        f"class, a specific behavior) to get the most relevant excerpts.]"
                    )
            else:
                # ── Arquivo cabe no contexto: comportamento normal — chunks
                # em destaque (se houver query) seguidas do conteúdo inteiro.
                mem_content, chunks = await _memory_read_relevant_chunks(resolved, effective_query)
                body_content = mem_content if mem_content is not None else raw_content

                if chunks:
                    highlight = [
                        "esses são os trechos do código mais relevantes de acordo com a sua pesquisa:",
                        "",
                    ]
                    for i, c in enumerate(chunks, 1):
                        score = c.get("score")
                        score_str = f" (score {score})" if score is not None else ""
                        highlight.append(f"--- trecho {i}{score_str} ---")
                        highlight.append(c["chunk_text"])
                        highlight.append("")
                    body = (
                        "\n".join(highlight)
                        + "--- conteúdo completo do arquivo ---\n"
                        + body_content
                    )
                else:
                    # Sem query, sem match acima do threshold, ou memory.py
                    # fora do ar — cai de volta para o conteúdo integral,
                    # sem o bloco de destaque.
                    body = body_content

            if data.get("truncated"):
                body += "\n\n[...content truncated...]"
            if note:
                body = f"{note}\n\n{body}"
            # Cache file content for subsequent steps
            _cache_file(file_cache, resolved, body)
            return True, body

        # Aceita tanto `create_file` (nome canônico no novo SYSTEM_PROMPT e na
        # grammar) quanto `write_file` (nome legado, mantido para compat com
        # chamadas que vierem do caminho fallback com tool_calls nativos).
        if name in ("create_file", "write_file"):
            resolved, note = _resolve_path(known_paths, args["file_path"])
            target_path = resolved if resolved is not None else args["file_path"]

            # Check if file exists and has content
            try:
                existing = await client.post("/read-file", json={"file_path": target_path})
                existing_has_content = existing.status_code == 200 and bool(
                    existing.json().get("content", "").strip()
                )
            except Exception:
                existing_has_content = False

            if existing_has_content:
                fail_count = (edit_failures or {}).get(target_path, 0)
                if fail_count >= 2:
                    log.warning(f"create_file unlocked for '{target_path}' after {fail_count} str_replace failures")
                    note = (note + "\n" if note else "") + \
                        f"[FALLBACK: create_file used after {fail_count} str_replace failures]"
                else:
                    return False, (
                        f"'{target_path}' already exists with content — create_file is not allowed "
                        f"here (it would overwrite the entire file). Use str_replace for targeted edits.\n\n"
                        f"HOW TO USE str_replace:\n"
                        f"  1. Run read_file on '{target_path}' to see current content\n"
                        f"  2. Copy an EXACT, UNIQUE snippet into old_str\n"
                        f"  3. Write the modified version in new_str\n"
                        f"  4. old_str cannot be empty — it must be text that exists in the file\n\n"
                        f"If you've tried str_replace 3+ times and it keeps failing, "
                        f"create_file will be unlocked automatically as a fallback."
                    )

            # Snapshot before edit for potential rollback on syntax error
            snapshot_content = None
            if target_path.lower().endswith(".py"):
                try:
                    snap_r = await client.post("/read-file", json={"file_path": target_path})
                    if snap_r.status_code == 200:
                        snapshot_content = snap_r.json().get("content", "")
                except Exception:
                    pass

            r = await client.post("/write-file", json={
                "file_path": target_path, "content": args["content"],
            })
            r.raise_for_status()
            data = r.json()
            _register_path(known_paths, data["file_path"])
            if edit_failures is not None and target_path in edit_failures:
                edit_failures.pop(target_path, None)
            # Update cache with new content
            _cache_file(file_cache, target_path, args["content"])
            msg = f"File written: {data['file_path']} ({data['bytes_written']} bytes, created={data['created']})"

            # Post-edit syntax validation for Python files
            if snapshot_content is not None:
                is_valid, val_msg = await _validate_edit_and_revert(client, target_path, snapshot_content, file_cache)
                if not is_valid:
                    if edit_failures is not None:
                        edit_failures[target_path] = edit_failures.get(target_path, 0) + 1
                    return False, f"{note}\n{msg}\n\n{val_msg}" if note else f"{msg}\n\n{val_msg}"

            return True, f"{note}\n{msg}" if note else msg

        if name == "str_replace":
            resolved, note = _resolve_path(known_paths, args["file_path"])
            if resolved is None:
                return False, (
                    f"Path '{args['file_path']}' not found in current project listing. "
                    f"Do NOT guess paths — run list_files to confirm the exact path."
                )

            old_str = args.get("old_str", "")
            new_str = args.get("new_str", "")

            if not old_str or not old_str.strip():
                if edit_failures is not None:
                    edit_failures[resolved] = edit_failures.get(resolved, 0) + 1
                return False, (
                    f"old_str cannot be empty. str_replace replaces an EXISTING snippet in the file. "
                    f"Steps:\n"
                    f"  1. Run read_file on '{resolved}' to see current content\n"
                    f"  2. Select a small snippet (1-5 lines) to modify\n"
                    f"  3. Copy that snippet EXACTLY into old_str\n"
                    f"  4. Write the modified version in new_str\n\n"
                    f"If you want to CREATE a new file, use create_file instead."
                )

            # Snapshot before edit for potential rollback on syntax error
            snapshot_content = None
            if resolved.lower().endswith(".py"):
                try:
                    snap_r = await client.post("/read-file", json={"file_path": resolved})
                    if snap_r.status_code == 200:
                        snapshot_content = snap_r.json().get("content", "")
                except Exception:
                    pass

            r = await client.post("/str-replace", json={
                "file_path":   resolved,
                "old_str":     old_str,
                "new_str":     new_str,
                "replace_all": args.get("replace_all", False),
            })

            if r.status_code != 200:
                if edit_failures is not None:
                    edit_failures[resolved] = edit_failures.get(resolved, 0) + 1
                fail_count = edit_failures.get(resolved, 0) if edit_failures else 0
                try:
                    err_detail = r.json().get("detail", r.text[:300])
                except Exception:
                    err_detail = r.text[:300]

                suggestion = ""
                if "não encontrado" in err_detail.lower() or "not found" in err_detail.lower():
                    # Try to hand back a small window of REAL, current
                    # content near where old_str almost matched, instead of
                    # unconditionally sending the model back to read_file
                    # (which, on a large file, is exactly the kind of extra
                    # round-trip that turns into a read-loop). Reuse the .py
                    # snapshot if we already fetched one above to avoid a
                    # second HTTP call.
                    near_content = snapshot_content
                    if near_content is None:
                        try:
                            fetch_r = await client.post("/read-file", json={"file_path": resolved})
                            if fetch_r.status_code == 200:
                                near_content = fetch_r.json().get("content", "")
                                # We paid for this fetch — keep it, respecting the cache's LRU cap.
                                _cache_file(file_cache, resolved, near_content)
                        except Exception:
                            near_content = None

                    near_match = (
                        _find_near_match_context(near_content, old_str)
                        if near_content is not None else None
                    )
                    if near_match:
                        snippet, start_line = near_match
                        suggestion = (
                            f"\n\nThe exact old_str was NOT found, but similar content exists "
                            f"near line {start_line} of '{resolved}':\n"
                            f"---\n{snippet}\n---\n"
                            f"Compare this to your old_str character-by-character (whitespace, "
                            f"indentation, line breaks) and retry with the corrected exact text. "
                            f"You do NOT need to call read_file — this IS the current content."
                        )
                    else:
                        suggestion = (
                            f"\n\nThe old_str does NOT exist in '{resolved}' — not even a close "
                            f"match was found. Run read_file on '{resolved}' NOW and copy the "
                            f"exact old_str."
                        )
                elif "várias" in err_detail.lower() or "multiple" in err_detail.lower():
                    suggestion = (
                        f"\n\nThe old_str appears MULTIPLE times. Include more context "
                        f"(lines before/after) to make it unique, or use replace_all=true."
                    )

                fallback_hint = ""
                if fail_count >= 3:
                    fallback_hint = (
                        f"\n\nYou've failed {fail_count}x on str_replace in this file. "
                        f"create_file is now unlocked as FALLBACK."
                    )

                return False, f"str_replace failed: {err_detail}{suggestion}{fallback_hint}"

            r.raise_for_status()
            data = r.json()
            if edit_failures is not None and resolved in edit_failures:
                edit_failures.pop(resolved, None)
            msg = f"{data['replacements']} replacement(s) in {data['file_path']}"
            # Invalidate cache — file content changed
            file_cache.pop(resolved, None)

            # Post-edit syntax validation for Python files
            if snapshot_content is not None:
                is_valid, val_msg = await _validate_edit_and_revert(client, resolved, snapshot_content, file_cache)
                if not is_valid:
                    if edit_failures is not None:
                        edit_failures[resolved] = edit_failures.get(resolved, 0) + 1
                    return False, f"{note}\n{msg}\n\n{val_msg}" if note else f"{msg}\n\n{val_msg}"

            return True, f"{note}\n{msg}" if note else msg

        if name == "execute_command":
            command = args.get("command", "")
            destructive_reason = _check_destructive_command(command)
            if destructive_reason:
                log.warning(f"Blocked destructive command ({destructive_reason}): {command[:200]}")
                return False, (
                    f"BLOCKED: this command was not executed because it looks destructive/"
                    f"irreversible ({destructive_reason}).\n\n"
                    f"Command: {command}\n\n"
                    f"If this is genuinely required to complete the task, call "
                    f"`ask_user_question` to get the user's explicit confirmation first. "
                    f"Do not retry this command, rephrase it, or attempt an equivalent "
                    f"workaround without that confirmation."
                )
            r = await client.post("/execute", json={"command": args["command"]})
            r.raise_for_status()
            data = r.json()
            out = f"exit_code={data['exit_code']}\nstdout:\n{data['stdout'][:3000]}\nstderr:\n{data['stderr'][:1500]}"
            # A shell command can write/modify/delete ANY file on disk —
            # git checkout, sed -i, a build script, a formatter, etc. — none
            # of which goes through write_file/str_replace, so nothing above
            # would invalidate those files' cache entries. Trusting stale
            # cached content after this point risks read_file returning
            # content that no longer matches the real file (Claude Code's
            # FileReadTool guards the equivalent case with a real mtime
            # check before trusting readFileState; we don't have filesystem
            # access here, so the safe default is to drop the whole cache).
            if file_cache:
                log.info(f"execute_command ran — invalidating {len(file_cache)} cached file(s), outcome unknown")
                file_cache.clear()
            return data["exit_code"] == 0, out

        if name == "web_search":
            # NOTE: assumes the scraping_client (port 3005) exposes a
            # POST /search endpoint accepting {"query": "..."} and returning
            # {"results": [{"title", "url", "snippet"}, ...]}. This mirrors
            # the file-operation endpoints already used above (/read-file,
            # /write-file, etc.) — verify/implement it on the scraping_client
            # side if it doesn't already exist, and adjust the field names
            # below to match its actual response shape.
            query = (args.get("query") or "").strip()
            if not query:
                return False, "ERROR: 'query' is required for web_search."
            r = await client.post("/search", json={"query": query})
            r.raise_for_status()
            data = r.json()
            results = data.get("results", data if isinstance(data, list) else [])
            if not results:
                return True, f"No results found for: {query}"
            lines = []
            for item in results[:8]:
                title = item.get("title", "")
                url = item.get("url", "")
                snippet = item.get("snippet", "") or item.get("content", "")
                lines.append(f"- {title}\n  {url}\n  {snippet[:300]}")
            return True, "\n".join(lines)

        if name == "replace_chunk":
            chunk_id = args["chunk_id"]
            new_content = args["new_content"]

            # Passo 1: buscar chunk no memory.py
            mem_client = _get_memory_client()
            r = await mem_client.get(f"/indexed-file/chunk/{chunk_id}")
            r.raise_for_status()
            chunk_data = r.json()

            # Passo 2: ler arquivo real do disco
            resolved, _ = _resolve_path(known_paths, args["file_path"])
            r2 = await client.post("/read-file", json={"file_path": resolved})
            r2.raise_for_status()
            file_data = r2.json()
            real_content = file_data["content"]

            # Passo 3: validar hash (arquivo não mudou desde indexação)
            real_hash = hashlib.sha256(real_content.encode()).hexdigest()
            if real_hash != chunk_data["file_hash"]:
                return False, (
                    f"File has changed since the chunk was created. "
                    f"Re-read the file with read_file to get fresh chunk IDs."
                )

            # Passo 4: substituir char_start:char_end por new_content
            char_start = chunk_data["char_start"]
            char_end = chunk_data["char_end"]
            new_file_content = real_content[:char_start] + new_content + real_content[char_end:]

            # Passo 5: escrever arquivo
            r3 = await client.post("/write-file", json={
                "file_path": resolved, "content": new_file_content, "force": True,
            })
            r3.raise_for_status()

            # Passo 6: re-indexar no memory.py
            await _memory_index_file(resolved, new_file_content)

            # Passo 7: invalidar cache
            file_cache.pop(resolved, None)
            _cache_file(file_cache, resolved, new_file_content)

            return True, f"Chunk {chunk_id} replaced. File re-indexed."
        
        return False, f"Unknown tool: {name}"
    
    except httpx.HTTPStatusError as e:
        return False, f"HTTP {e.response.status_code}: {e.response.text[:500]}"
    except httpx.ConnectError:
        return False, (
            f"scraping_client offline at {SCRAPING_URL} — make sure the client "
            f"is running on the user's machine (alpha-client.py)."
        )
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════
# Task Execution — single ReAct loop for the whole task
# ══════════════════════════════════════════════════════════════════════════

STEP_MAX_ITERATIONS = 10
STEP_STUCK_LIMIT = 3

# Read-loop breaker: catches the model calling read_file/list_files on
# different paths over and over without ever attempting an edit — a pattern
# the exact-signature stuck detector (STEP_STUCK_LIMIT) does not catch.
STEP_READ_ONLY_NUDGE = 3       # inject a soft reminder to switch to action
STEP_READ_ONLY_HARD_LIMIT = 5  # start refusing further reads, force an edit attempt

# Cached-file content injected into each step's prompt (see "Previously read
# files" below). Previously this was per-file only (6000 chars/file, no
# total cap), so a task that had read a dozen files would re-inject all of
# them, in full, into EVERY iteration of EVERY subsequent step — easily
# blowing past the TPM caps of the smaller execution-chain models (Claude
# Code's Read tool bounds this the same way, via maxSizeBytes/maxTokens on
# a single read; here the risk compounds because the cache is re-sent whole
# on every single model call, not just once).
CACHE_PROMPT_TOTAL_CHAR_BUDGET = 20000  # combined cap across all cached files shown
CACHE_PROMPT_PER_FILE_CHAR_CAP = 6000   # cap for any single file within that budget


# ── Anti-hallucination guard for finish/step-complete paths ──────────────────
# O modelo (LFM2.5-8B-A1B) tem a tendência de chamar `finish` ou produzir
# texto puro afirmando "gerei o arquivo X" sem nunca ter chamado create_file
# ou str_replace. Sem este guard, a resposta final engana o usuário: ele lê
# "Gerei o arquivo project.md" mas nenhum arquivo existe em disco.
#
# Este guard é aplicado em TODOS os caminhos que encerram um step:
#   1. Handler do `finish` (quando o modelo chama finish(summary=...) sob grammar)
#   2. Branch `else` com `tool_calls` vazio (quando o modelo retorna texto puro
#      que não é parseable como chamada de função Python)
#
# Em ambos os casos, se a action original ou o summary mencionam edição de
# arquivos mas edit_successes_this_step == 0, o finish é RECUSADO e um
# <system-reminder> é injetado forçando o modelo a realmente chamar
# create_file/str_replace. Limitado a 2 recusas por (sid, step_num) para
# evitar loop infinito — após 2 recusas, deixa passar e a Camada 2
# (pós-loop) marca o passo como FALHADO.
_EDIT_INTENT_KEYWORDS: tuple[str, ...] = (
    # PT
    "gerar", "gere", "gerou", "gerad", "criar", "crie", "criou", "criad",
    "escrever", "escreva", "escreveu", "salvar", "salve", "salvou",
    "modificar", "modificou", "atualizar", "atualize", "atualizou",
    "refatorar", "refatora", "refatorou", "editar", "edite", "editou",
    # EN
    "create", "created", "generate", "generated", "wrote", "write",
    "written", "modify", "modified", "update", "updated", "edit",
    "edited", "refactor", "refactored", "saved", "save",
)


def _text_claims_edit(*texts: str) -> bool:
    """Retorna True se qualquer um dos textos menciona criação/edição de
    arquivos. Bilingue (PT + EN) para cobrir prompts em ambos os idiomas."""
    for t in texts:
        if not t:
            continue
        tl = t.lower()
        if any(kw in tl for kw in _EDIT_INTENT_KEYWORDS):
            return True
    return False


def _refuse_hallucinated_finish(
    sid: str,
    step_num: int,
    task: str,
    step_result_text: str,
    edit_successes_this_step: int,
    tag: str,
    step_messages: list[dict],
    finish_kind: str = "finish",
) -> bool:
    """
    Verifica se o finish atual é uma alucinação (alega edição mas nenhuma
    edição foi feita neste passo). Se for E ainda houver budget de recusas
    (< 2), injeta um <system-reminder> forçando o modelo a chamar
    create_file/str_replace e retorna True — o caller deve fazer
    `step_result_text = ""; continue` para reiniciar o loop sem break.

    Retorna False se o finish deve ser aceito:
      - edit_successes_this_step > 0  (houve edição real, não é alucinação)
      - nem a task nem o summary mencionam edição (passo puramente de leitura)
      - budget de recusas esgotado (deixa passar; a Camada 2 pós-loop pega)

    `finish_kind`: "finish" (chamada finish via grammar) ou "plain-text"
    (modelo retornou texto puro sem tool_call parseable). Só afeta a mensagem
    de log; a lógica é idêntica.
    """
    if edit_successes_this_step > 0:
        return False

    if not _text_claims_edit(task, step_result_text):
        return False

    # Cache de recusas — atributo da função para persistir entre chamadas
    # sem poluir o escopo global do módulo.
    if not hasattr(_refuse_hallucinated_finish, "_refusals_cache"):
        _refuse_hallucinated_finish._refusals_cache = {}  # type: ignore[attr-defined]
    cache = _refuse_hallucinated_finish._refusals_cache  # type: ignore[attr-defined]
    key = (sid, step_num)
    refusals = cache.get(key, 0)

    if refusals >= 2:
        # Budget esgotado — deixa o finish passar. A Camada 2 (após o loop)
        # vai marcar o passo como FALHADO com "HALLUCINATION DETECTED".
        return False

    cache[key] = refusals + 1
    log.warning(
        f"{tag}Step {step_num}: REFUSING {finish_kind} — claims file edit "
        f"but edit_successes_this_step=0 (refusal #{refusals + 1}/2). "
        f"Forcing real create_file/str_replace call."
    )

    reminder = _wrap_reminder(
        f"You called finish(summary=\"{step_result_text[:300]}\"), but you have NOT "
        f"actually called create_file or str_replace in this step. The summary "
        f"claims a file was created/modified, but NO edit tool was invoked — the "
        f"file does NOT exist on disk.\n\n"
        f"You MUST call create_file (for a new file) or str_replace (for an existing "
        f"file) NOW with the actual full content, and only THEN call finish again. "
        f"Do NOT call finish until a real edit has succeeded and you have seen the "
        f"'File written:' or 'replacement(s) in' confirmation in the tool result."
    )
    step_messages.append({"role": "user", "content": reminder})
    return True


def _clear_finish_refusals(sid: str, step_num: int) -> None:
    """Limpa o cache de recusas quando o passo é concluído com sucesso
    (edição real ou passo puramente de leitura). Evita que contadores
    de passos anteriores vazem para passos novos do mesmo sid."""
    if not hasattr(_refuse_hallucinated_finish, "_refusals_cache"):
        return
    _refuse_hallucinated_finish._refusals_cache.pop((sid, step_num), None)  # type: ignore[attr-defined]


async def _execute_step(
    task: str,
    sid: str,
    known_paths: dict[str, str],
    files_changed: set[str],
    edit_failures: dict[str, int],
    file_cache: dict[str, str],
    resume_state: Optional[dict] = None,
    max_iterations: int = STEP_MAX_ITERATIONS,
) -> AsyncGenerator[dict, None]:
    """
    Executes the task using a single ReAct loop.
    Produces SSE events: step_start, model_choice, thinking, tool_call, tool_result,
    needs_input, step_done, step_error.

    resume_state: if the loop previously paused on an `ask_user_question` call
    (see the "needs_input" event / _SESSIONS), this carries everything needed
    to pick the ReAct loop back up instead of rebuilding the prompt from
    scratch — the already-built `step_messages` (with the user's answer
    appended as the pending tool result), and the loop counters as they stood
    right before the pause.
    """
    # step_num is a fixed constant, not derived from a plan — kept only so
    # SSE events keep the same "step" field consumers may already read.
    step_num = 1
    tag = f"[{sid[:8]}] " if sid else ""

    # Proxy de "a pesquisa do modelo" para a busca de chunks relevantes no
    # memory.py (ver `_memory_read_relevant_chunks`): a task, usada como proxy
    # de "a pesquisa do modelo". Calculado aqui em vez de dentro do `if not
    # resuming` porque precisa estar disponível também ao retomar um loop
    # pausado em `ask_user_question`.
    step_search_query = (task or "").strip()

    if resume_state is not None:
        step_messages: list[dict] = resume_state["step_messages"]
        cache_shown_in_prompt: set[str] = set(resume_state.get("cache_shown_in_prompt", []))
        iterations = resume_state.get("iterations", 0)
        tools_in_step = resume_state.get("tools_in_step", 0)
        edit_attempts_this_step = resume_state.get("edit_attempts_this_step", 0)
        edit_successes_this_step = resume_state.get("edit_successes_this_step", 0)
        read_streak_since_edit = resume_state.get("read_streak_since_edit", 0)
        last_tool_sig: Optional[str] = resume_state.get("last_tool_sig")
        repeat_count = resume_state.get("repeat_count", 0)
        step_result_text = ""
        step_error: Optional[str] = None
        log.info(f"{tag}▶ Resuming ReAct loop after ask_user_question answer")
        resuming = True
    else:
        resuming = False

    if not resuming:
        step_prompt = (
            f"═══ TASK ═══\n"
            f"{task}\n"
        )

        # Include cached file contents so the model doesn't need to re-read them.
        # Bounded by a total budget (not just a per-file cap) and biased toward
        # the most recently touched files, so a long task with many files read
        # doesn't re-inject all of them, in full, on every single iteration.
        #
        # `cache_shown_in_prompt` tracks exactly which files actually made it
        # into the text below (budget cuts can drop older ones) — it's threaded
        # into the tool loop so a later read_file cache-hit only points the
        # model at "shown above" when that's actually true.
        cache_shown_in_prompt = set()
        if file_cache:
            cache_lines = []
            budget_used = 0
            omitted = 0
            # Walk newest-touched-first (see _cache_file callers) so anything
            # dropped for budget reasons is the OLDEST, least-likely-relevant
            # content, not the file the model just read.
            for fpath, fcontent in reversed(list(file_cache.items())):
                if budget_used >= CACHE_PROMPT_TOTAL_CHAR_BUDGET:
                    omitted += 1
                    continue
                per_file_cap = min(
                    CACHE_PROMPT_PER_FILE_CHAR_CAP,
                    CACHE_PROMPT_TOTAL_CHAR_BUDGET - budget_used,
                )
                trunc = fcontent[:per_file_cap]
                if len(fcontent) > per_file_cap:
                    trunc += "\n[...file truncated...]"
                cache_lines.append(f"[Cached content of {fpath}]\n{trunc}")
                cache_shown_in_prompt.add(fpath)
                budget_used += len(trunc)
            if cache_lines:
                cache_lines.reverse()  # restore original read order for readability
                block = "\n\n".join(cache_lines)
                if omitted:
                    block += (
                        f"\n\n[...{omitted} older cached file(s) omitted to stay within budget "
                        f"— call read_file again if you need one of them...]"
                    )
                step_prompt += "\n\nPreviously read files (DO NOT re-read these — use the content below):\n" + block + "\n"

        step_prompt += (
            "\nCRITICAL INSTRUCTIONS:\n"
            "1. Work the task above using the available tools (list_files, read_file, create_file, "
            "str_replace, execute_command, ask_user_question) as needed. Call `finish` with a summary when done.\n"
            "2. As soon as the task is complete, respond in NATURAL TEXT (without calling any tool) "
            "with a brief, specific summary of what was done.\n"
            "3. If the task cannot be completed (e.g., file not found, tool error), respond in text explaining the problem.\n"
            "4. Do NOT re-read files that are already shown above in 'Previously read files'. "
            "The cache system will return '[From cache]' if you try — this wastes tokens. "
            "Use the content shown above directly to make your edits.\n"
            "5. PREFER ACTION over observation: if you already have the file content, make the edit. "
            "Do not read the file again to 'verify' before editing.\n"
            "══════════════════════════════════════════════"
        )

        # Circuit breaker: if a file has had multiple edit failures, tell the model
        if edit_failures:
            broken_files = [f for f, count in edit_failures.items() if count >= 2]
            if broken_files:
                step_prompt += (
                    "\n\n⚠️ CIRCUIT BREAKER: The following files have had multiple str_replace failures: "
                    + ", ".join(f"'{f}'" for f in broken_files)
                    + ". For these files, use create_file to rewrite the ENTIRE file instead of str_replace. "
                    "create_file is now UNLOCKED for these files as a fallback."
                )

        step_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": step_prompt},
        ]

        yield {
            "event": "step_start",
            "data": {"step": step_num, "total": 1, "action": task},
            "step": step_num,
        }
        log.info(f"{tag}▶ Step {step_num}/1: {task[:80]}")

        step_result_text = ""
        step_error = None
        iterations = 0
        tools_in_step = 0
        last_tool_sig = None
        repeat_count = 0

        # ── Read-loop tracking ──────────────────────────────────────────────
        # The old stuck-detector only caught the EXACT same tool called with the
        # EXACT same arguments 3x in a row. That misses the much more common
        # failure mode: the model calls list_files / read_file on a bunch of
        # *different* paths (or the same path with slightly different notes),
        # never repeating a signature, and therefore never triggers the
        # repeat-count breaker — while never actually attempting an edit. That
        # is the "reads files in circles without changing anything" behavior.
        # These counters track that pattern directly instead of relying on
        # exact-signature repetition.
        # create_file é o nome canônico; write_file é alias legado. Ambos contam
        # como edit_tool para o circuit breaker de read-loop.
        edit_attempts_this_step = 0     # create_file/write_file / str_replace calls attempted
        edit_successes_this_step = 0    # create_file/write_file / str_replace calls that succeeded
        read_streak_since_edit = 0      # consecutive read_file/list_files calls with no edit attempt yet

    # Aviso disparado a 80% do orçamento de iterações (mesma proporção do
    # STEP_WARNING_ITERATION=8/STEP_MAX_ITERATIONS=10 original), agora
    # escalado pelo max_iterations efetivo (vindo de req.max_steps).
    warning_iteration = max(1, int(max_iterations * 0.8))

    while iterations < max_iterations:
        iterations += 1

        yield {
            "event": "model_choice",
            "data": {
                "model": "llama-server (local)",
                "step_kind": "execution",
            },
            "step": step_num,
        }

        try:
            data = await _call_llama(
                messages=step_messages,
                tools=TOOLS if not USE_GRAMMAR_MODE else None,
                temperature=TEMPERATURE_BY_KIND["execution"],
                sid=sid,
                grammar=TOOL_GRAMMAR if USE_GRAMMAR_MODE else None,
            )
        except RuntimeError as e:
            step_error = str(e)
            log.error(f"{tag}llama-server call failed at step {step_num}: {e}")
            yield {"event": "step_error", "data": {"step": step_num, "error": step_error, "fatal": True}, "step": step_num}
            return

        # Parse response
        choices = data.get("choices", [])
        if not choices:
            step_error = f"Step {step_num}: llama-server returned no choices"
            yield {"event": "step_error", "data": {"step": step_num, "error": step_error, "fatal": True}, "step": step_num}
            return

        message = choices[0].get("message", {})
        content = (message.get("content") or "").strip()

        # ── MODO GRAMMAR: content é um JSON garantido pela grammar ───────────
        # O llama-server não retorna `tool_calls` nativos quando uma grammar é
        # aplicada — o output é puro texto em `content`, e a grammar força o
        # formato {"tool": "...", "args": {...}}. Parseamos e sintetizamos um
        # tool_call OpenAI-equivalente para manter compatibilidade com o resto
        # do pipeline (_run_tool, tracking de loops, step_messages history).
        synthetic_tc: Optional[dict] = None
        if USE_GRAMMAR_MODE and content:
            synthetic_tc = _parse_tool_call_from_content(content)

        if synthetic_tc is not None:
            # ── Tool virtual "finish": step completo ─────────────────────────
            # O modelo decidiu parar de chamar ferramentas e produzir um
            # resumo em texto. Equivalente ao "no tool_calls" do caminho
            # nativo, mas sob grammar é uma tool explícita — mais limpo.
            if synthetic_tc["function"]["name"] == "finish":
                try:
                    f_args = json.loads(synthetic_tc["function"]["arguments"])
                except json.JSONDecodeError:
                    f_args = {}
                step_result_text = (f_args.get("summary") or "").strip()
                # gpt-oss fallback: alguns modelos podem colocar o resumo em
                # 'reasoning' ao invés de preencher o campo summary da grammar.
                if not step_result_text:
                    reasoning = (message.get("reasoning") or "").strip()
                    if reasoning:
                        log.info(f"{tag}Step {step_num} finish summary was in 'reasoning' field — using it")
                        step_result_text = reasoning
                if not step_result_text:
                    step_result_text = (
                        f"(Step {step_num} completed via 'finish' tool but no summary was provided.)"
                    )

                # ── ANTI-HALLUCINATION GUARD (Camada 1) ─────────────────────
                # Se a action ou o summary mencionam edição mas nenhuma edição
                # foi feita (edit_successes_this_step == 0), recusa o finish,
                # injeta um reminder forçando create_file/str_replace, e dá
                # ao modelo uma nova iteração para se recuperar.
                if _refuse_hallucinated_finish(
                    sid, step_num, task,
                    step_result_text, edit_successes_this_step,
                    tag, step_messages, finish_kind="finish",
                ):
                    step_result_text = ""
                    continue  # ← próxima iteração do while, sem break

                # Aceito: limpa o cache de recusas e encerra o passo.
                _clear_finish_refusals(sid, step_num)
                step_messages.append({"role": "assistant", "content": step_result_text})
                break

            # Constrói tool_calls sintético no formato OpenAI para o histórico
            tool_calls = [synthetic_tc]
            # O assistant message no histórico precisa carregar tanto o
            # content (JSON que o modelo gerou, preserva a decisão) quanto o
            # tool_calls sintético (casa com a próxima role=tool message).
            step_messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            })
            # Emite o content como "thinking" para a UI acompanhar o raciocínio
            yield {"event": "thinking", "data": {"text": content, "plan_step": step_num}, "step": step_num}

        else:
            # ── Fallback: caminho antigo (tool_calls nativos do OpenAI) ───────
            # Só atinge este ramo se USE_GRAMMAR_MODE=False OU se a grammar
            # falhou (não deveria acontecer — grammar garante JSON válido).
            # Tratamos o caso de erro de grammar explicitamente para evitar
            # loop silencioso.
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                # Modelo retornou texto puro — step completo
                step_result_text = content
                # gpt-oss models may put the response in 'reasoning' and leave 'content' empty
                if not step_result_text:
                    reasoning = (message.get("reasoning") or "").strip()
                    if reasoning:
                        log.info(f"{tag}Step {step_num} content was in 'reasoning' field — using it")
                        step_result_text = reasoning
                if not step_result_text:
                    if USE_GRAMMAR_MODE:
                        # Grammar ativa mas content vazio = problema sério.
                        # Não é normal — força erro para diagnóstico.
                        step_error = (
                            f"Step {step_num}: grammar mode returned empty content. "
                            f"This should not happen — check that the llama-server supports "
                            f"the `grammar` parameter and that TOOL_GRAMMAR is valid GBNF."
                        )
                        log.error(f"{tag}{step_error}")
                    else:
                        step_error = f"Step {step_num}: model returned empty response (no tool calls and no content)"
                        log.warning(f"{tag}{step_error}")
                    break

                # ── ANTI-HALLUCINATION GUARD (Camada 1 — branch texto puro) ─
                # Mesmo guard do handler do `finish`, aplicado aqui porque o
                # modelo pode retornar texto puro afirmando "gerei o arquivo
                # X" sem nunca ter chamado create_file. Sem este guard, o
                # `break` abaixo aceitaria o finish alucinado e a Camada 2
                # (pós-loop) pegaria como rede de segurança — mas aí o modelo
                # perderia a chance de se recuperar dentro do loop.
                #
                # Caso real que motivou este guard: tarefa "gere um .md
                # descrevendo o projeto", o modelo listou arquivos e depois
                # retornou prosa afirmando ter gerado o .md, sem chamar
                # create_file. O guard dispara "REFUSING plain-text finish"
                # e força o modelo a chamar create_file de verdade.
                if _refuse_hallucinated_finish(
                    sid, step_num, task,
                    step_result_text, edit_successes_this_step,
                    tag, step_messages, finish_kind="plain-text",
                ):
                    step_result_text = ""
                    continue  # ← próxima iteração do while, sem break

                # Aceito: limpa o cache de recusas e encerra o passo.
                _clear_finish_refusals(sid, step_num)
                step_messages.append({"role": "assistant", "content": step_result_text})
                break

            # Add assistant message to history — strip any 'reasoning' field some
            # local model templates attach to the message, so it doesn't
            # accumulate as an unsupported field in the sent history.
            cleaned_msg = {k: v for k, v in message.items() if k != "reasoning"}
            step_messages.append(cleaned_msg)

            if content:
                yield {"event": "thinking", "data": {"text": content, "plan_step": step_num}, "step": step_num}

        # Execute each tool call
        for tc in tool_calls:
            fn = tc.get("function", {}) or {}
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}

            # Stuck detection
            sig = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
            if sig == last_tool_sig:
                repeat_count += 1
            else:
                repeat_count = 1
                last_tool_sig = sig

            if repeat_count >= STEP_STUCK_LIMIT:
                step_error = (
                    f"Step {step_num}: loop detected — tool '{name}' called "
                    f"{repeat_count}x consecutively with the same arguments."
                )
                log.warning(f"{tag}{step_error}")
                yield {"event": "step_error", "data": {"step": step_num, "error": step_error, "fatal": False}, "step": step_num}
                break

            tools_in_step += 1
            yield {"event": "tool_call", "data": {"name": name, "arguments": args, "plan_step": step_num}, "step": step_num}

            # ── ask_user_question pauses the whole pipeline ─────────────
            # Unlike every other tool, this one can't be answered by
            # _run_tool — the answer has to come from the human, on a
            # future request. We yield a `needs_input` event carrying
            # everything the caller needs to persist and later resume this
            # exact step (see _SESSIONS in _run_pipeline), then stop the
            # generator without a step_done. No str_replace/write_file/
            # execute_command happens after this point in the current turn.
            if name == "ask_user_question":
                question = (args.get("question") or "").strip() or (
                    "The agent needs more information to proceed but did not "
                    "provide a specific question."
                )
                log.info(f"{tag}Step {step_num} paused — asking user: {question[:200]}")
                yield {
                    "event": "needs_input",
                    "data": {
                        "step": step_num,
                        "question": question,
                        "pending_tool_call_id": tc.get("id", ""),
                    },
                    "step": step_num,
                }
                yield {
                    "event": "__resume_state__",
                    "data": {
                        "step_messages": step_messages,
                        "cache_shown_in_prompt": list(cache_shown_in_prompt),
                        "iterations": iterations,
                        "tools_in_step": tools_in_step,
                        "edit_attempts_this_step": edit_attempts_this_step,
                        "edit_successes_this_step": edit_successes_this_step,
                        "read_streak_since_edit": read_streak_since_edit,
                        "last_tool_sig": last_tool_sig,
                        "repeat_count": repeat_count,
                        "pending_tool_call_id": tc.get("id", ""),
                        "question": question,
                    },
                    "step": step_num,
                }
                return

            is_read_tool = name in ("read_file", "list_files")
            is_edit_tool = name in ("create_file", "write_file", "str_replace")

            t0 = time.perf_counter()
            if is_read_tool and edit_attempts_this_step == 0 and read_streak_since_edit >= STEP_READ_ONLY_HARD_LIMIT:
                # Hard block: the model has read files repeatedly this step
                # without ever attempting an edit. Refuse the read instead of
                # burning another iteration on it, and force a decision.
                success, output = False, (
                    f"BLOCKED: you've made {read_streak_since_edit} read-only tool calls "
                    f"({name}/list_files/read_file) in this step without attempting a single edit. "
                    f"You already have enough information from what you've read so far. "
                    f"Call str_replace or create_file NOW with your change, or — if the step genuinely "
                    f"requires no code change — respond in plain text explaining why."
                )
            else:
                success, output = await _run_tool(
                    name, args, known_paths, file_cache, cache_shown_in_prompt,
                    edit_failures, search_query=step_search_query,
                )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

            if is_edit_tool:
                edit_attempts_this_step += 1
                read_streak_since_edit = 0
                if success:
                    edit_successes_this_step += 1
            elif is_read_tool:
                read_streak_since_edit += 1

            if success and name in ("create_file", "write_file", "str_replace") and args.get("file_path"):
                files_changed.add(args["file_path"])

            yield {
                "event": "tool_result",
                "data": {
                    "success": success,
                    "output": output if success else "",
                    "error": "" if success else output,
                    "elapsed_ms": elapsed_ms,
                    "plan_step": step_num,
                },
                "step": step_num,
            }

            step_messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": name,
                "content": output[:MAX_TOOL_OUTPUT_CHARS],
            })

        if step_error:
            break

        # Soft nudge: read-only streak crossed the threshold but hasn't hit
        # the hard block yet — steer the model toward action early instead
        # of waiting for the generic iteration-count warning.
        if (
            edit_attempts_this_step == 0
            and read_streak_since_edit >= STEP_READ_ONLY_NUDGE
            and not step_result_text
        ):
            step_messages.append({
                "role": "user",
                "content": _wrap_reminder(
                    f"You've called read_file/list_files {read_streak_since_edit} times in a row "
                    f"this step without attempting an edit. If you already know what needs to "
                    f"change, call str_replace or create_file now instead of reading more files."
                ),
            })

        # Warning when close to iteration limit
        if iterations >= warning_iteration and not step_result_text:
            remaining = max_iterations - iterations
            step_messages.append({
                "role": "user",
                "content": _wrap_reminder(
                    f"WARNING: You have made {iterations} iterations on step {step_num}. "
                    f"Only {remaining} iteration(s) left. If you have gathered enough information, "
                    f"RESPOND NOW in natural text summarizing what was done. "
                    f"Do NOT call more tools unless strictly necessary."
                ),
            })

    # Handle iterations exhausted
    if not step_result_text and not step_error:
        if edit_successes_this_step > 0:
            # A real edit happened — this is genuine (if incomplete) progress.
            step_result_text = (
                f"(Step {step_num} partially completed — {tools_in_step} tool calls "
                f"in {iterations} iterations, including {edit_successes_this_step} successful edit(s), "
                f"but no final text response.)"
            )
            log.info(f"{tag}Step {step_num} PARTIAL (with edits): {tools_in_step} tools, "
                     f"{edit_successes_this_step} successful edits, in {iterations} iterations")
        elif edit_attempts_this_step > 0:
            # Edits were attempted but every single one failed (e.g. old_str
            # never matched). This is a real failure, not a success — don't
            # let it silently pass through as "done" to the next step.
            step_error = (
                f"Step {step_num} exhausted {max_iterations} iterations: "
                f"{edit_attempts_this_step} edit attempt(s) were made but ALL of them failed "
                f"(see tool errors above). No change was actually made to any file."
            )
            log.warning(f"{tag}Step {step_num} FAILED: {edit_attempts_this_step} edit attempts, all failed")
        elif tools_in_step > 0:
            # Only read_file/list_files/execute_command were called — the
            # step spent its whole budget "reading in circles" and never
            # even tried to make a change.
            step_error = (
                f"Step {step_num} exhausted {max_iterations} iterations making {tools_in_step} "
                f"read-only tool call(s) without ever attempting an edit (no str_replace or create_file "
                f"call was made). The step is stuck in a read loop, not actually progressing."
            )
            log.warning(f"{tag}Step {step_num} FAILED: {tools_in_step} read-only calls, zero edit attempts")
        else:
            step_error = f"Step {step_num} exhausted {max_iterations} iterations without a final response"

    # ── ANTI-HALLUCINATION: mesmo se o modelo chamou finish com um summary
    # não-vazio (passando pelo guard acima), se a action original do passo
    # exigia edição mas edit_successes_this_step == 0, marcamos o passo como
    # FALHADO. Isso pega o caso em que o modelo persistiu na alucinação por
    # 2 recusas seguidas e o guard eventualmente deixou passar.
    _EDIT_INTENT_KEYWORDS_FINAL = (
        "gerar", "gere", "criar", "crie", "escrever", "escreva",
        "salvar", "salve", "modificar", "atualizar", "refatorar",
        "create", "generate", "write", "modify", "update", "refactor",
    )
    action_required_edit = any(
        kw in task.lower() for kw in _EDIT_INTENT_KEYWORDS_FINAL
    )
    if (
        step_result_text
        and step_error is None
        and action_required_edit
        and edit_successes_this_step == 0
    ):
        step_error = (
            f"Step {step_num} HALLUCINATION DETECTED: the task required a file edit "
            f"(task='{task[:120]}'), the model reported completion in its finish summary, "
            f"but ZERO edits actually succeeded (edit_successes_this_step=0). The claimed file "
            f"modification does NOT exist on disk. Treating this as a failure rather than reporting "
            f"a false success to the user."
        )
        log.error(f"{tag}{step_error}")
        # Mantém o step_result_text visível no result para diagnóstico, mas
        # marca success=False via step_error estar setado.

    yield {
        "event": "step_done",
        "data": {
            "step":          step_num,
            "total":         1,
            "success":       step_error is None,
            "result":        step_result_text,
            "error":         step_error,
            "iterations":    iterations,
            "tools_called":  tools_in_step,
            "edit_attempts": edit_attempts_this_step,
            "edit_successes": edit_successes_this_step,
        },
        "step": step_num,
    }


# ══════════════════════════════════════════════════════════════════════════
# Main Execution Pipeline
# ══════════════════════════════════════════════════════════════════════════

class RunRequest(BaseModel):
    task:           str
    session_id:     Optional[str] = None
    project_dir:    Optional[str] = None
    max_steps:      int = Field(25, ge=1, le=100)
    temperature:    float = Field(0.3, ge=0.0, le=2.0)
    model_override: Optional[str] = None
    streaming:      bool = False
    context:        Optional[str] = None
    # Summary
    final_synthesis: bool = True
    # Resuming after a `needs_input` pause: pass the SAME session_id back
    # together with the user's answer to the pending ask_user_question.
    resume_answer:  Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════
# Paused-session store (for ask_user_question)
# ══════════════════════════════════════════════════════════════════════════
#
# In-memory only — fine for a single process, lost on restart. A paused
# task is everything `_run_pipeline` needs to pick the ReAct loop back up:
# the original task string, the mutable known_paths/files_changed/
# edit_failures/file_cache it was using, and the exact _execute_step
# resume_state (step_messages etc.) captured at the moment it called
# ask_user_question.
#
# Bounded and FIFO-evicted like FILE_CACHE_MAX_ENTRIES, so an orchestrator
# that starts tasks and never resumes them can't leak memory forever.
_SESSIONS: dict[str, dict] = {}
_SESSIONS_MAX_ENTRIES = 200


def _save_session(
    sid: str, task: str, known_paths: dict[str, str],
    files_changed: set[str], edit_failures: dict[str, int],
    file_cache: dict[str, str], tools_called_total: int,
    resume_payload: dict,
    max_iterations: int = STEP_MAX_ITERATIONS,
) -> None:
    if sid in _SESSIONS:
        del _SESSIONS[sid]
    _SESSIONS[sid] = {
        "task": task,
        "known_paths": known_paths, "files_changed": list(files_changed),
        "edit_failures": edit_failures, "file_cache": file_cache,
        "tools_called_total": tools_called_total,
        "resume_state": resume_payload,
        "max_iterations": max_iterations,
    }
    while len(_SESSIONS) > _SESSIONS_MAX_ENTRIES:
        oldest = next(iter(_SESSIONS))
        del _SESSIONS[oldest]
    log.info(f"[{sid[:8]}] Session saved — paused awaiting user input")


async def _drain_step(step_gen: AsyncGenerator[dict, None], out: dict) -> AsyncGenerator[dict, None]:
    """
    Forwards every event from an `_execute_step` generator except the
    internal `__resume_state__` marker (never part of the public SSE
    contract), and records into `out` whether the step paused
    (ask_user_question) or finished (step_done) — an async generator can't
    `return` a value, so the caller reads it back from `out` after this
    generator is exhausted.
    """
    async for evt in step_gen:
        if evt["event"] == "__resume_state__":
            out["resume_payload"] = evt["data"]
            out["paused"] = True
            continue
        if evt["event"] == "needs_input":
            out["paused"] = True
        if evt["event"] == "step_done":
            out["step_done"] = evt["data"]
        yield evt


async def _finalize_pipeline(
    step_result: Optional[dict], task: str, req: RunRequest,
    sid: str, tag: str, files_changed: set[str], tools_called_total: int,
) -> AsyncGenerator[dict, None]:
    """Step 3 of the pipeline: build the final answer (optionally via a
    synthesis call) and emit the `final` event. Shared by both the fresh-run
    and resume-after-pause paths so they converge on identical output.

    `step_result` is the single `step_done` event data from `_execute_step`
    (or None if the ReAct loop never produced one, e.g. a fatal error before
    completion)."""
    success = bool(step_result and step_result.get("success"))

    if success:
        final_answer = step_result.get("result", "")
        if not final_answer:
            final_answer = "Task executed, but no text response was produced."
    else:
        final_answer = "Task could not be completed."

    if success and req.final_synthesis:
        try:
            log.info(f"{tag}Generating final synthesis...")
            synthesis_prompt = (
                f"Original task: {task}\n\n"
                f"Result of the ReAct loop:\n"
                f"\n{step_result.get('result','')[:500]}\n"
            )
            synthesis_prompt += (
                "\nWrite a concise final response for the user, explaining "
                "what was done and which files were changed. Do not call any tools."
            )

            data = await _call_llama(
                messages=[
                    {"role": "system", "content": "You are a software engineering assistant. Summarize what was done concisely."},
                    {"role": "user",   "content": synthesis_prompt},
                ],
                temperature=TEMPERATURE_BY_KIND["summary"],
                sid=sid,
            )

            choices = data.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                if msg.get("content") and not msg.get("tool_calls"):
                    final_answer = msg["content"]
                    yield {"event": "final_synthesis", "data": {"text": final_answer}, "step": 0}
                    log.info(f"{tag}Final synthesis generated")
        except Exception as e:
            log.warning(f"{tag}Final synthesis failed: {e}")

    # ── ANTI-HALLUCINATION (camada final): se a resposta final menciona
    # criação/edição de arquivos mas files_changed está vazio, anexar um
    # AVISO EXPLÍCITO ao usuário. Última linha de defesa — pega casos em
    # que o guard do _execute_step não disparou (e.g. task não continha
    # keyword de edição, mas o modelo alucinou a criação mesmo assim).
    _CLAIM_KEYWORDS = (
        "gerar", "gere", "gerou", "criar", "crie", "criou",
        "escrever", "escreva", "escreveu", "salvar", "salve",
        "modifiquei", "atualizei", "refatorei", "editei",
        "i created", "i generated", "i wrote", "i modified",
        "i updated", "i edited", "i refactored", "i saved",
        "created the file", "generated the file", "wrote the file",
    )
    answer_lower = final_answer.lower()
    answer_claims_file_work = any(kw in answer_lower for kw in _CLAIM_KEYWORDS)
    actual_files = sorted(f for f in files_changed if f)
    if answer_claims_file_work and not actual_files:
        warning = (
            "\n\n⚠️ AVISO DE VERIFICAÇÃO: a resposta acima alega que arquivos foram "
            "criados/modificados, mas o pipeline não registrou NENHUMA alteração "
            "real em disco (files_changed=[]). É provável que o modelo tenha "
            "alucinado a criação do arquivo sem efetivamente chamar create_file "
            "ou str_replace. Verifique manualmente o sistema de arquivos antes "
            "de confiar no resumo. Reinicie a tarefa se necessário."
        )
        final_answer = final_answer.rstrip() + warning
        log.error(
            f"{tag}HALLUCINATION WARNING: final answer claims file work but "
            f"files_changed is empty. Appended user-visible warning."
        )

    yield {
        "event": "final",
        "data": {
            "answer":         final_answer,
            "files_changed":  actual_files,
            "tools_called":   tools_called_total,
            "steps_executed": 1 if step_result is not None else 0,
            "session_id":     sid,
            "success":        success,
            "error":          None if success else "No steps completed successfully",
            "cot_plan":       None,
        },
        "step": 0,
    }

    log.info(f"{tag}═══ Task complete: success={success}, files_changed={len(files_changed)}, tools_called={tools_called_total}")


async def _run_pipeline(req: RunRequest) -> AsyncGenerator[dict, None]:
    """
    Main pipeline — pure ReAct, no upfront planning phase:
      1. Run the task through a single ReAct sub-loop (`_execute_step`) —
         may PAUSE mid-loop on ask_user_question and return early; see
         `_SESSIONS`.
      2. Generate summary.

    No plan is ever generated by an LLM call — the whole task is executed
    as one ReAct loop, so there's nothing to index or resolve across steps.
    """
    sid = req.session_id or str(uuid.uuid4())
    tag = f"[{sid[:8]}] " if sid else ""

    # ── Resume path: continuing a task paused on ask_user_question ─────────
    if req.resume_answer is not None:
        session = _SESSIONS.pop(sid, None)
        if session is None:
            log.warning(f"{tag}resume_answer given but no paused session found")
            yield {
                "event": "final",
                "data": {
                    "answer": "", "files_changed": [], "tools_called": 0,
                    "steps_executed": 0, "session_id": sid, "success": False,
                    "error": (
                        f"No paused task found for session_id='{sid}' — it may already "
                        f"have been resumed, completed, or expired. Start a new task instead."
                    ),
                    "cot_plan": None,
                },
                "step": 0,
            }
            return

        task = session["task"]
        known_paths = session["known_paths"]
        files_changed = set(session["files_changed"])
        edit_failures = session["edit_failures"]
        file_cache = session["file_cache"]
        tools_called_total = session["tools_called_total"]
        max_iterations = session.get("max_iterations", STEP_MAX_ITERATIONS)

        resume_state = dict(session["resume_state"])
        resume_state["step_messages"] = resume_state["step_messages"] + [{
            "role": "tool",
            "tool_call_id": resume_state.get("pending_tool_call_id", ""),
            "name": "ask_user_question",
            "content": (req.resume_answer or "").strip() or "(no answer provided)",
        }]

        log.info(f"{tag}Resuming ReAct loop with user-provided answer")

        out: dict = {"paused": False, "resume_payload": None, "step_done": None}
        gen = _execute_step(
            task, sid, known_paths, files_changed, edit_failures, file_cache,
            resume_state=resume_state, max_iterations=max_iterations,
        )
        async for evt in _drain_step(gen, out):
            yield evt
        if out["paused"]:
            _save_session(
                sid, task, known_paths, files_changed,
                edit_failures, file_cache, tools_called_total,
                out["resume_payload"], max_iterations,
            )
            return
        step_result = out["step_done"]
        if step_result:
            tools_called_total += step_result.get("tools_called", 0)

        async for evt in _finalize_pipeline(step_result, task, req, sid, tag, files_changed, tools_called_total):
            yield evt
        return

    # ── Fresh path ───────────────────────────────────────────────────────
    log.info(f"{tag}═══ New task: {req.task[:100]}")
    log.info(f"{tag}max_steps={req.max_steps}")

    # Per-run file cache — local to this pipeline execution, NOT a module
    # global, so concurrent requests/sessions never see or clobber each
    # other's cached file contents (see comment above `_cache_file`).
    file_cache: dict[str, str] = {}

    # ── Run the whole task through a single ReAct sub-loop ─────────────────
    known_paths: dict[str, str] = {}
    edit_failures: dict[str, int] = {}
    files_changed: set[str] = set()
    tools_called_total = 0

    out = {"paused": False, "resume_payload": None, "step_done": None}
    gen = _execute_step(
        req.task, sid, known_paths, files_changed, edit_failures,
        file_cache, max_iterations=req.max_steps,
    )
    async for evt in _drain_step(gen, out):
        yield evt
    if out["paused"]:
        _save_session(
            sid, req.task, known_paths, files_changed,
            edit_failures, file_cache, tools_called_total,
            out["resume_payload"], req.max_steps,
        )
        return
    step_result = out["step_done"]
    if step_result:
        tools_called_total += step_result.get("tools_called", 0)

    # ── Generate summary ────────────────────────────────────────────────
    async for evt in _finalize_pipeline(step_result, req.task, req, sid, tag, files_changed, tools_called_total):
        yield evt


# ══════════════════════════════════════════════════════════════════════════
# FastAPI Application
# ══════════════════════════════════════════════════════════════════════════

app = FastAPI(title="AVA Alpha Code", version="2.1.0")


def _sse(event: str, data: Any, step: Optional[int] = None) -> str:
    payload: dict = {"event": event, "data": data, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if step is not None:
        payload["step"] = step
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/health")
async def health():
    llama_ok = False
    try:
        client = _get_llama_client()
        r = await client.get("/health", timeout=3.0)
        llama_ok = r.status_code == 200
    except Exception:
        pass

    scrape_ok = False
    try:
        client = _get_scrape_client()
        r = await client.get("/health", timeout=3.0)
        scrape_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok" if llama_ok and scrape_ok else "degraded",
        "service": "alpha_code",
        "llama_server": "ok" if llama_ok else "unreachable",
        "scraping_client": "ok" if scrape_ok else "unreachable",
    }


@app.post("/run/stream")
async def run_stream(req: RunRequest):
    async def gen():
        async for evt in _run_pipeline(req):
            yield _sse(evt["event"], evt["data"], evt.get("step"))
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/run")
async def run_sync(req: RunRequest):
    final: dict = {}
    cot_plan: Optional[list[dict]] = None
    cot_error: Optional[str] = None
    needs_input: Optional[dict] = None
    async for evt in _run_pipeline(req):
        if evt["event"] == "final":
            final = evt["data"]
        elif evt["event"] == "cot_plan":
            cot_plan = evt["data"].get("steps")
        elif evt["event"] == "cot_error":
            cot_error = evt["data"].get("error")
        elif evt["event"] == "needs_input":
            needs_input = evt["data"]

    sid = req.session_id or ""
    if needs_input is not None:
        # Task paused mid-step waiting on the user. Call /run again with the
        # SAME session_id and `resume_answer` set to the user's reply to
        # `question` to continue exactly where it left off.
        return {
            "success":        False,
            "status":         "needs_input",
            "question":       needs_input.get("question", ""),
            "answer":         "",
            "files_changed":  [],
            "steps_executed": 0,
            "tools_called":   0,
            "session_id":     sid,
            "cot_plan":       cot_plan,
            "cot_error":      cot_error,
        }

    return {
        "success":        final.get("success", False),
        "status":         "done",
        "answer":         final.get("answer", ""),
        "files_changed":  final.get("files_changed", []),
        "steps_executed": final.get("steps_executed", 0),
        "tools_called":   final.get("tools_called", 0),
        "session_id":     final.get("session_id", sid),
        "cot_plan":       final.get("cot_plan", cot_plan),
        "cot_error":      cot_error,
    }


if __name__ == "__main__":
    import uvicorn
    log.info("═══════════════════════════════════════")
    log.info("  AVA Alpha Code — port 4006")
    log.info(f"  LLAMA_URL: {LLAMA_URL}")
    log.info(f"  SCRAPING_URL: {SCRAPING_URL}")
    log.info("═══════════════════════════════════════")
    uvicorn.run(app, host="0.0.0.0", port=4006, log_level="info")