# Alpha AI (AVA) — Tool `execute` do Code Mode
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from typing import Any, Optional

from ..tools.registry import ToolResult, ToolSnapshot, ToolSpec
from .catalog import Catalog
from .runtime import CodeModeRuntime

DESCRIPTION = """Run JavaScript in a confined Code Mode runtime to compose MCP tool calls.
MCP tools are listed in the Code Mode catalog in the system prompt; call them through the
`tools` object and await the result, e.g. `return await tools["server__tool"]({...})`.
The dotted alias `tools.server.tool(...)` also works. If a tool is not listed, use
`search("what you need")` to find its exact path. Return a value; the returned value is the
tool result. No imports, filesystem, network, timers, or `process` are available."""


def _resolve(name: str, snapshot: ToolSnapshot) -> Optional[str]:
    if name in snapshot.specs:
        return name
    if "." in name:
        alias = name.replace(".", "__", 1)
        if alias in snapshot.specs:
            return alias
    return None


def make_execute_spec(
    snapshot: ToolSnapshot,
    runtime: CodeModeRuntime,
    catalog: Catalog,
    *,
    timeout_ms: int = 30_000,
    max_tool_calls: int = 50,
    max_output_chars: int = 200_000,
) -> ToolSpec:
    async def execute(args: dict[str, Any], context: Any) -> ToolResult:
        code = str(args.get("code") or "")
        if not code.strip():
            return ToolResult(content="ERROR: `code` is empty", is_error=True)

        async def call_tool(name: str, payload: dict[str, Any]) -> ToolResult:
            resolved = _resolve(name, snapshot)
            if resolved is None:
                return ToolResult(
                    content=(
                        f'ERROR: tool "{name}" is not in the Code Mode catalog. '
                        'Use search("what you need") to find the exact path.'
                    ),
                    is_error=True,
                )
            if resolved == "execute":
                return ToolResult(content="ERROR: execute cannot call itself", is_error=True)
            return await snapshot.execute(resolved, payload, context)

        result = await runtime.run(
            code,
            call_tool,
            catalog.as_list(),
            timeout_ms=timeout_ms,
            max_tool_calls=max_tool_calls,
            max_output_chars=max_output_chars,
        )
        metadata = {"toolCalls": result.tool_calls}
        if result.error:
            return ToolResult(content=f"ERROR: {result.error}", metadata=metadata, is_error=True)
        output = result.output
        if result.logs:
            output = (output + "\n\n[console]\n" + "\n".join(result.logs[-50:])).strip()
        return ToolResult(output=output, metadata=metadata)

    return ToolSpec(
        name="execute",
        description=DESCRIPTION,
        input_schema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "JavaScript to run. Use `return` for the final value.",
                }
            },
            "required": ["code"],
        },
        execute=execute,
        parallel_safe=False,
        codemode=False,
        source="native",
    )
