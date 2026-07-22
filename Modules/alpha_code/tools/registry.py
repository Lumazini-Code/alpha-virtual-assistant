"""
Alpha-code — Tool Registry
==========================
Centraliza registro de tools com schemas OpenAI function-calling.

Cada tool expõe:
  - schema(): dict  → schema no formato OpenAI (enviado ao Groq)
  - execute(args, ctx) -> ToolResult  → executa a tool

O agente chama registry.get_schemas() para montar o payload do /chat/tools,
e registry.dispatch(call, ctx) para executar.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from alpha_code.schemas import StepContext, ToolCall, ToolResult

log = logging.getLogger("ava.alpha_code.registry")


class Tool:
    """Base class para todas as tools do alpha_code."""

    name: str = ""
    description: str = ""

    def schema(self) -> dict:
        """Retorna schema no formato OpenAI function-calling."""
        raise NotImplementedError

    async def execute(self, args: dict, ctx: StepContext) -> ToolResult:
        raise NotImplementedError


class ToolRegistry:
    """
    Registry central de tools.

    Uso:
        registry = ToolRegistry()
        registry.register(MyTool())
        schemas = registry.get_schemas()
        result = await registry.dispatch(tool_call, ctx)
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> "ToolRegistry":
        if not tool.name:
            raise ValueError(f"Tool {tool.__class__.__name__} não tem name")
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name} já registrada")
        self._tools[tool.name] = tool
        log.info(f"Tool registrada: {tool.name}")
        return self

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def get_schemas(self) -> list[dict]:
        """Retorna lista de schemas no formato esperado pelo Groq tools=[]."""
        return [
            {"type": "function", "function": tool.schema()}
            for tool in self._tools.values()
        ]

    async def dispatch(self, call: ToolCall, ctx: StepContext) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                success=False,
                error=f"Tool desconhecida: {call.name}",
            )

        t0 = time.perf_counter()
        try:
            result = await tool.execute(call.arguments, ctx)
            if result.elapsed_ms == 0.0:
                result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            log.info(
                f"tool={call.name} success={result.success} "
                f"elapsed={result.elapsed_ms}ms out_len={len(result.output)}"
            )
            return result
        except Exception as e:
            elapsed = round((time.perf_counter() - t0) * 1000, 1)
            log.exception(f"Tool {call.name} crashed")
            return ToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                success=False,
                error=f"{type(e).__name__}: {e}",
                elapsed_ms=elapsed,
            )


# ── Helper para construir JSON schema simples ────────────────────────────────

def param(type: str, description: str, **kw) -> dict:
    """
    Atalho para criar uma propriedade JSON schema.
    O nome da propriedade é a chave no dict `properties` (passado por schema_function).
    """
    d = {"type": type, "description": description}
    d.update(kw)
    return d


def schema_function(
    name: str,
    description: str,
    properties: dict[str, dict],
    required: list[str],
) -> dict:
    """Monta schema completo no formato OpenAI function-calling."""
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
