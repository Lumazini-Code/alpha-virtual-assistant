# Alpha AI (AVA) — Tool registry unificado
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Registry de tools nativas e MCP.

Substitui os dicionários ad hoc do orchestrator por um único ponto com:
- definição OpenAI determinística;
- validação de entrada com mensagem amigável ao modelo;
- política allow/deny;
- hooks before/after;
- execução uniforme (nativa ou MCP).
"""

from .hooks import ToolHooks
from .permissions import ToolPolicy
from .registry import (
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSnapshot,
    ToolSpec,
)
from .validation import format_issues, validate

__all__ = [
    "ToolContext",
    "ToolHooks",
    "ToolPolicy",
    "ToolRegistry",
    "ToolResult",
    "ToolSnapshot",
    "ToolSpec",
    "format_issues",
    "validate",
]
