# Alpha AI (AVA) — Catálogo do Code Mode
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    display: str
    description: str
    signature: str
    namespace: str
    source: str = "mcp"

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "display": self.display,
            "description": self.description,
            "signature": self.signature,
            "namespace": self.namespace,
        }


def _display_name(name: str) -> str:
    if "__" in name:
        server, tool = name.split("__", 1)
        return f"{server}.{tool}"
    return name


def _signature(name: str, schema: Any) -> str:
    if not isinstance(schema, dict):
        return f"{_display_name(name)}()"
    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    required = schema.get("required")
    required = set(required) if isinstance(required, list) else set()
    parts: list[str] = []
    for field, value in properties.items():
        value = value if isinstance(value, dict) else {}
        kind = value.get("type") or "any"
        if isinstance(kind, list):
            kind = "|".join(str(item) for item in kind)
        optional = "" if field in required else "?"
        parts.append(f"{field}{optional}: {kind}")
    return f"{_display_name(name)}({', '.join(parts)})"


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class Catalog:
    def __init__(self, entries: Iterable[CatalogEntry]):
        self.entries = sorted(entries, key=lambda entry: (entry.namespace, entry.name))

    @classmethod
    def from_specs(cls, specs: Iterable[Any]) -> "Catalog":
        entries: list[CatalogEntry] = []
        for spec in specs:
            namespace = spec.namespace or _display_name(spec.name).split(".", 1)[0]
            entries.append(
                CatalogEntry(
                    name=spec.name,
                    display=_display_name(spec.name),
                    description=_clip(spec.description or "", 240),
                    signature=_signature(spec.name, spec.input_schema),
                    namespace=namespace,
                    source=getattr(spec, "source", "mcp"),
                )
            )
        return cls(entries)

    def as_list(self) -> list[dict[str, str]]:
        return [entry.as_dict() for entry in self.entries]

    def search(self, query: str, limit: int = 12) -> list[CatalogEntry]:
        terms = [term for term in re.split(r"[^a-z0-9_]+", (query or "").lower()) if term]
        if not terms:
            return []

        def score(entry: CatalogEntry) -> int:
            haystack = f"{entry.name} {entry.display} {entry.description}".lower()
            return sum(1 for term in terms if term in haystack)

        ranked = sorted(
            ((score(entry), entry) for entry in self.entries),
            key=lambda item: (-item[0], item[1].name),
        )
        return [entry for value, entry in ranked if value > 0][:limit]

    def render(self, max_entries: int = 80) -> str:
        if not self.entries:
            return (
                "## Code Mode\n\n"
                "No MCP tools are available to `execute` this turn. "
                "Use the native tools directly."
            )
        shown = self.entries[:max_entries]
        lines = [
            "## Code Mode",
            "",
            "MCP tools are only callable inside the `execute` tool, as JavaScript.",
            "Call them through the `tools` object and await the result:",
            "",
            "```js",
            'const result = await tools["server__tool"]({ arg: "value" });',
            "return result;",
            "```",
            "",
            "The dotted alias also works (`tools.server.tool(...)`). If a tool is not",
            "listed below, find it with `search(\"what you need\")`. Do not guess names.",
            "No imports, filesystem, network, timers, or `process` are available.",
            "",
            "### Catalog",
        ]
        namespace = None
        for entry in shown:
            if entry.namespace != namespace:
                namespace = entry.namespace
                lines.append(f"\n#### {namespace}")
            lines.append(f"- `{entry.signature}` — {entry.description}")
        if len(self.entries) > len(shown):
            lines.append(
                f"\n...and {len(self.entries) - len(shown)} more; use `search(query)` to find them."
            )
        return "\n".join(lines)


def search_payload(catalog: Catalog, query: str, limit: int = 12) -> str:
    return json.dumps(
        [
            {
                "path": entry.display,
                "name": entry.name,
                "signature": entry.signature,
                "description": entry.description,
            }
            for entry in catalog.search(query, limit)
        ],
        ensure_ascii=False,
    )
