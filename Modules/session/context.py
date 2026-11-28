# Alpha AI (AVA) — Montagem do contexto de trabalho do turno
# Copyright (C) 2026  Felipe lumazini
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
from __future__ import annotations

"""
context.py — Constrói o contexto que vai ao modelo em cada turno.

Dois modos, escolhidos pelo usuário na WebUI:

* ``execution``      — histórico completo durável (comportamento antigo).
  Indicado para tarefas longas com muitas tool calls.
* ``conversational`` — janela de N turnos + orçamento de tokens; tool calls
  antigas são descartadas (o episódio guarda o que foi útil). Indicado para
  conversas com muitos turnos de pergunta-resposta.

O ledger bruto (SessionStore) continua guardando tudo. Este módulo só decide
o que entra no prompt — nunca apaga nada.
"""

from typing import Any, Optional

from .history import message_text, sanitize_for_model

MODE_CONVERSATIONAL = "conversational"
MODE_EXECUTION = "execution"
VALID_MODES = frozenset({MODE_CONVERSATIONAL, MODE_EXECUTION})


def normalize_mode(mode: Optional[str], default: str = MODE_CONVERSATIONAL) -> str:
    value = (mode or "").strip().lower()
    return value if value in VALID_MODES else default


def estimate_tokens(text: Any) -> int:
    """Estimativa barata e conservadora (~4 chars/token)."""
    return max(1, len(message_text(text)) // 4)


def _content_tokens(message: dict[str, Any]) -> int:
    total = estimate_tokens(message.get("content"))
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        total += estimate_tokens(fn.get("name")) + estimate_tokens(fn.get("arguments"))
    return total


def split_turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Agrupa mensagens em turnos (um turno começa em cada ``user``).

    Mensagens anteriores ao primeiro ``user`` (ex.: notificação de subagente
    ou retomada) ficam no primeiro turno para não serem perdidas.
    """
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "user" and current:
            turns.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        turns.append(current)
    return turns


def _strip_tools(turn: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove tool calls/resultados de um turno antigo, preservando o texto.

    O objetivo é manter o fio da conversa sem carregar o corpo das
    ferramentas. Se o assistente só tinha tool calls (sem texto), a mensagem
    é descartada; o resultado útil já está no episódio.
    """
    kept: list[dict[str, Any]] = []
    for message in turn:
        role = message.get("role")
        if role == "tool":
            continue
        if role == "assistant" and message.get("tool_calls"):
            text = message_text(message.get("content")).strip()
            if not text:
                continue
            kept.append({**message, "tool_calls": None, "content": text})
            continue
        kept.append(message)
    return kept


def _trim_to_budget(messages: list[dict[str, Any]], max_tokens: int) -> list[dict[str, Any]]:
    """Corta texto do início até caber no orçamento, sem quebrar o chat template."""
    if max_tokens <= 0:
        return messages
    total = sum(_content_tokens(m) for m in messages)
    if total <= max_tokens:
        return messages

    # Remove mensagens mais antigas primeiro, mas nunca a última (o turno
    # corrente precisa chegar inteiro ao modelo).
    out = list(messages)
    while out and total > max_tokens and len(out) > 1:
        removed = out.pop(0)
        total -= _content_tokens(removed)
    if total <= max_tokens:
        return out

    # Ainda passou: trunca o texto das mensagens mais antigas restantes.
    # Mensagens com blocos de mídia (image_url) são preservadas — recortar o
    # texto dela substituiria o content multi-parte por uma string e derrubaria
    # as imagens anexadas. O orçamento de imagens do request cuida do tamanho.
    excess = total - max_tokens
    for i, message in enumerate(out[:-1]):
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict)
            and part.get("type") not in (None, "text", "input_text")
            for part in content
        ):
            continue
        text = message_text(content)
        if not text:
            continue
        cut = min(len(text), max(0, excess * 4))
        if cut <= 0:
            break
        message = {**message, "content": text[: len(text) - cut].rstrip() + "…"}
        out[i] = message
        excess = max(0, excess - cut // 4)
        if excess <= 0:
            break
    return out


def build_working_context(
    messages: list[dict[str, Any]],
    *,
    mode: str = MODE_CONVERSATIONAL,
    max_turns: int = 6,
    max_tokens: int = 12000,
    keep_prior_tools: bool = False,
    keep_trailing_tools: bool = False,
    max_chars: int = 0,
) -> list[dict[str, Any]]:
    """Monta a lista de mensagens que vai ao modelo.

    ``keep_trailing_tools`` é usado quando o histórico é recarregado no meio
    de um turno (após tool results): aí as tool calls do turno corrente
    precisam permanecer para o modelo continuar.
    """
    mode = normalize_mode(mode)
    if mode == MODE_EXECUTION:
        return sanitize_for_model(
            messages,
            max_chars=max_chars,
            keep_trailing_tools=keep_trailing_tools,
        )

    turns = split_turns(messages)
    if max_turns > 0:
        turns = turns[-max_turns:]

    flattened: list[dict[str, Any]] = []
    last_index = len(turns) - 1
    for index, turn in enumerate(turns):
        keep_tools = keep_prior_tools or (keep_trailing_tools and index == last_index)
        flattened.extend(turn if keep_tools else _strip_tools(turn))

    flattened = _trim_to_budget(flattened, max_tokens)
    return sanitize_for_model(flattened, keep_trailing_tools=keep_trailing_tools)


def build_recall_turns(
    messages: list[dict[str, Any]],
    *,
    mode: str = MODE_CONVERSATIONAL,
    max_turns: int = 6,
) -> list[dict[str, Any]]:
    """Turnos de texto usados para deduplicar o recall (sem tool calls)."""
    turns = split_turns(messages)
    if normalize_mode(mode) == MODE_CONVERSATIONAL and max_turns > 0:
        turns = turns[-max_turns:]
    out: list[dict[str, Any]] = []
    for turn in turns:
        out.extend(_strip_tools(turn))
    return out
