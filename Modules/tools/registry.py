# Alpha AI (AVA) — Registry, snapshot e execução de tools
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .hooks import ToolHooks
from .permissions import ToolPolicy
from .validation import format_issues, validate


@dataclass
class ToolContext:
    session_id: str
    call_id: str
    request: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    output: Any = None
    content: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    is_error: bool = False


Executor = Callable[[dict[str, Any], ToolContext], Awaitable[Any]]


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    execute: Executor
    output_schema: Optional[dict[str, Any]] = None
    permission: Optional[str] = None
    parallel_safe: bool = False
    codemode: bool = True
    namespace: Optional[str] = None
    source: str = "native"
    always_on: bool = False


class ToolRegistry:
    """Registry por turno. O snapshot é imutável e capturado antes do loop."""

    def __init__(
        self,
        hooks: Optional[ToolHooks] = None,
        error_formatter: Optional[Callable[[BaseException], str]] = None,
    ):
        self._specs: dict[str, ToolSpec] = {}
        self.hooks = hooks or ToolHooks()
        self.error_formatter = error_formatter

    def register(self, spec: ToolSpec) -> None:
        if not spec.name:
            raise ValueError("tool sem nome")
        self._specs[spec.name] = spec

    def extend(self, specs: list[ToolSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def snapshot(self, policy: Optional[ToolPolicy] = None) -> "ToolSnapshot":
        policy = policy or ToolPolicy.from_env()
        specs = {
            name: spec
            for name, spec in self._specs.items()
            if policy.permits(name)
        }
        return ToolSnapshot(
            specs=dict(sorted(specs.items())),
            policy=policy,
            hooks=self.hooks,
            error_formatter=self.error_formatter,
        )

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._specs.get(name)


class ToolSnapshot:
    def __init__(
        self,
        specs: dict[str, ToolSpec],
        policy: ToolPolicy,
        hooks: ToolHooks,
        error_formatter: Optional[Callable[[BaseException], str]] = None,
    ):
        self.specs = specs
        self.policy = policy
        self.hooks = hooks
        self.error_formatter = error_formatter

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_schema or {"type": "object", "properties": {}},
                },
            }
            for spec in self.specs.values()
        ]

    def names(self) -> list[str]:
        return list(self.specs)

    def parallel_safe(self, name: str) -> bool:
        spec = self.specs.get(name)
        return bool(spec and spec.parallel_safe)

    def always_on_names(self) -> set[str]:
        return {name for name, spec in self.specs.items() if spec.always_on}

    async def execute(
        self,
        name: str,
        args: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        spec = self.specs.get(name)
        if spec is None:
            return ToolResult(
                content=f'ERROR: tool "{name}" is not available in this turn. Use a tool from the provided list.',
                is_error=True,
            )
        if not self.policy.permits(name):
            return ToolResult(
                content=f"ERROR: {self.policy.reason(name)}",
                is_error=True,
            )

        issues = validate(spec.input_schema, args)
        if issues:
            return ToolResult(content=format_issues(name, issues, args), is_error=True)

        try:
            rewritten = await self.hooks.run_before(name, args, context)
            issues = validate(spec.input_schema, rewritten)
            if issues:
                return ToolResult(content=format_issues(name, issues, rewritten), is_error=True)
            value = await spec.execute(rewritten, context)
            result = value if isinstance(value, ToolResult) else ToolResult(output=value)
            return await self.hooks.run_after(name, result, context)
        except Exception as error:
            content = (
                self.error_formatter(error)
                if self.error_formatter is not None
                else f"ERROR: {type(error).__name__}: {error}"
            )
            return ToolResult(content=content, is_error=True)


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)
