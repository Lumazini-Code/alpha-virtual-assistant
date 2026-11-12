# Alpha AI (AVA) — Validação de argumentos de tools
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import json
from typing import Any

_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def validate(schema: Any, value: Any, path: str = "root") -> list[str]:
    """Valida um subconjunto de JSON Schema usado pelas tools.

    Cobre type, required, properties, additionalProperties, enum, items e
    minimum/maximum. Schemas desconhecidos são aceitos (forward compatible),
    como no OpenCode.
    """
    if not isinstance(schema, dict) or not schema:
        return []

    issues: list[str] = []
    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        accepted = tuple(_TYPES[item] for item in types if item in _TYPES)
        if accepted and not isinstance(value, accepted):
            issues.append(f"{path}: expected {expected}")
            return issues

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        issues.append(f"{path}: must be one of {json.dumps(enum, ensure_ascii=False)}")

    if isinstance(value, dict):
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    issues.append(f"{path}.{key}: required")
        for key, item in value.items():
            if key in properties:
                issues.extend(validate(properties[key], item, f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                issues.append(f"{path}.{key}: unexpected property")

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            issues.extend(validate(schema["items"], item, f"{path}[{index}]"))

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            issues.append(f"{path}: must be >= {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            issues.append(f"{path}: must be <= {maximum}")

    return issues


def format_issues(tool: str, issues: list[str], value: Any) -> str:
    details = issues[:5]
    if len(issues) > 5:
        details.append(f"...and {len(issues) - 5} more issue(s)")
    provided = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return (
        f'Invalid arguments for tool "{tool}":\n'
        + "\n".join(f"- {issue}" for issue in details)
        + f"\n\nArguments provided:\n{provided}\n\n"
        "Update the arguments and call the tool again."
    )
