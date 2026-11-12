# Alpha AI (AVA) — Leitura de histórico para o runner
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .store import SessionStore


def message_text(content: Any) -> str:
    """Extrai o texto de um content OpenAI-like (string ou lista de partes)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in (None, "text", "input_text"):
                text = part.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _merge_content(previous: Any, current: Any) -> Any:
    """Mescla conteúdo de duas mensagens do mesmo papel.

    Quando os dois lados são texto, concatena. Se algum lado tem partes
    multimídia, preserva as partes em ordem (o texto vira uma parte de texto).
    """
    if isinstance(previous, list) or isinstance(current, list):
        def parts(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [part for part in value if isinstance(part, dict)]
            text = message_text(value)
            return [{"type": "text", "text": text}] if text else []

        return [*parts(previous), *parts(current)]
    left = message_text(previous).strip()
    right = message_text(current).strip()
    if not left:
        return right
    if not right:
        return left
    return f"{left}\n\n{right}"


def _tool_call_ids(message: dict[str, Any]) -> list[str]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return []
    return [
        str(call.get("id"))
        for call in calls
        if isinstance(call, dict) and call.get("id") is not None
    ]


def sanitize_for_model(
    messages: list[dict[str, Any]],
    *,
    max_chars: int = 0,
    keep_trailing_tools: bool = False,
) -> list[dict[str, Any]]:
    """Prepara mensagens duráveis para o chat template.

    Regras:
      * mantém apenas ``user``/``assistant``/``tool`` (o system prompt atual
        é montado à parte; synthetic/compaction ficam para a compactação);
      * um ``assistant`` com ``tool_calls`` só entra se todos os resultados
        daquele turno estiverem presentes; caso contrário o grupo inteiro é
        descartado (turno interrompido);
      * ``tool`` órfão é descartado;
      * se a história terminar em ``tool`` (turno sem resposta final), o
        grupo final é descartado — exceto com ``keep_trailing_tools``, usado
        na recuperação de overflow no meio de um step;
      * mensagens consecutivas do mesmo papel são mescladas para preservar a
        alternância exigida pelo template;
      * ``max_chars`` (opcional) descarta as mensagens mais antigas até caber.
    """
    cleaned: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if role not in ("user", "assistant", "tool"):
            index += 1
            continue

        if role == "assistant" and message.get("tool_calls"):
            expected = _tool_call_ids(message)
            cursor = index + 1
            results: list[dict[str, Any]] = []
            seen: set[str] = set()
            while cursor < len(messages) and messages[cursor].get("role") == "tool":
                result = messages[cursor]
                result_id = str(result.get("tool_call_id"))
                if result_id in expected and result_id not in seen:
                    results.append(result)
                    seen.add(result_id)
                cursor += 1
            if expected and all(call_id in seen for call_id in expected):
                # O grupo só é válido se o turno continuou com uma resposta do
                # assistant. Se o próximo item é um user (turno interrompido)
                # ou o fim da história, o template não aceita tool -> user.
                probe = cursor
                while probe < len(messages) and messages[probe].get("role") not in (
                    "user", "assistant", "tool",
                ):
                    probe += 1
                next_role = messages[probe].get("role") if probe < len(messages) else None
                if next_role == "assistant" or (keep_trailing_tools and next_role is None):
                    cleaned.append(message)
                    cleaned.extend(results)
                    index = cursor
                    continue
                index = cursor
                continue
            # Turno interrompido: descarta a chamada e os resultados parciais.
            index = cursor if cursor > index else index + 1
            continue

        if role == "tool":
            # Órfão: não há assistant com tool_calls imediatamente antes.
            index += 1
            continue

        cleaned.append(message)
        index += 1

    # Um grupo de tool calls no fim não tem resposta do assistant; descarta,
    # exceto quando a chamada seguinte é justamente a resposta que vai ser
    # gerada (recuperação de overflow no meio do step).
    if cleaned and cleaned[-1].get("role") == "tool" and not keep_trailing_tools:
        last_assistant = next(
            (position for position in range(len(cleaned) - 1, -1, -1)
             if cleaned[position].get("role") == "assistant"),
            None,
        )
        if last_assistant is not None:
            del cleaned[last_assistant:]

    # Alternância: mescla mensagens consecutivas do mesmo papel.
    merged: list[dict[str, Any]] = []
    for message in cleaned:
        if (
            merged
            and merged[-1].get("role") == message.get("role")
            and message.get("role") in ("user", "assistant")
            and not merged[-1].get("tool_calls")
            and not message.get("tool_calls")
        ):
            merged[-1] = {
                **merged[-1],
                "content": _merge_content(merged[-1].get("content"), message.get("content")),
            }
            continue
        merged.append(message)

    # A história deve começar em um turno do usuário; um assistant inicial
    # (saudação/continuação) violaria a alternância depois do system prompt.
    while merged and merged[0].get("role") != "user":
        merged.pop(0)

    if max_chars > 0:
        total = sum(len(message_text(message.get("content"))) for message in merged)
        while merged and total > max_chars and len(merged) > 1:
            removed = merged.pop(0)
            total -= len(message_text(removed.get("content")))
        # A história deve começar em um turno do usuário.
        while merged and merged[0].get("role") != "user":
            removed = merged.pop(0)
            total -= len(message_text(removed.get("content")))

    return merged


@dataclass
class History:
    """O que o runner precisa para montar um request.

    ``initial`` será preenchido pelo sistema de instruções versionadas. Nesta
    fase fica vazio e o orchestrator continua usando o system prompt atual.
    """

    initial: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)


class SessionHistory:
    def __init__(self, store: SessionStore):
        self.store = store

    def load(self, session_id: str, *, boundary: str = "latest") -> History:
        return History(initial=[], messages=self.store.load_messages(session_id, boundary=boundary))
