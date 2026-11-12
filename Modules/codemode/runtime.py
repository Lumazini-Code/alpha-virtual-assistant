# Alpha AI (AVA) — Runtime do Code Mode (Node subprocess)
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

ToolCaller = Callable[[str, dict[str, Any]], Awaitable[Any]]


@dataclass
class CodeModeResult:
    output: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    error: Optional[str] = None


class CodeModeRuntime:
    """Executa JavaScript em um processo Node isolado.

    O processo não recebe `require`, `process`, `fs`, `fetch` ou timers. Cada
    chamada de tool é um round-trip JSON pelo stdin/stdout; o Python executa a
    tool pelo registry e devolve o resultado.
    """

    def __init__(
        self,
        node: Optional[str] = None,
        worker: Optional[str | Path] = None,
    ):
        self.node = node or os.getenv("AVA_NODE_BIN") or shutil.which("node") or shutil.which("nodejs")
        self.worker = Path(worker) if worker else Path(__file__).resolve().parent / "worker.js"
        self.available = bool(self.node and self.worker.is_file())

    async def run(
        self,
        code: str,
        execute_tool: ToolCaller,
        catalog: list[dict[str, Any]],
        *,
        timeout_ms: int = 30_000,
        max_tool_calls: int = 50,
        max_output_chars: int = 200_000,
    ) -> CodeModeResult:
        if not self.available:
            return CodeModeResult(error="Code Mode runtime is unavailable (node/worker.js not found)")

        try:
            process = await asyncio.create_subprocess_exec(
                self.node,
                str(self.worker),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=10 * 1024 * 1024,
            )
        except Exception as error:
            return CodeModeResult(error=f"failed to start Code Mode runtime: {type(error).__name__}: {error}")

        write_lock = asyncio.Lock()
        stderr_lines: list[str] = []
        tasks: set[asyncio.Task[Any]] = set()

        async def write(message: dict[str, Any]) -> None:
            if process.stdin is None:
                return
            payload = json.dumps(message, ensure_ascii=False, default=str) + "\n"
            async with write_lock:
                try:
                    process.stdin.write(payload.encode("utf-8"))
                    await process.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        async def drain_stderr() -> None:
            if process.stderr is None:
                return
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
                stderr_lines.append(line.decode("utf-8", errors="replace").rstrip())

        async def handle_call(message: dict[str, Any]) -> None:
            call_id = message.get("id")
            name = str(message.get("name") or "")
            payload = message.get("input")
            payload = payload if isinstance(payload, dict) else {}
            try:
                result = await execute_tool(name, payload)
                if getattr(result, "is_error", False):
                    await write({
                        "type": "tool_result",
                        "id": call_id,
                        "error": getattr(result, "content", None) or "tool failed",
                    })
                else:
                    value = getattr(result, "output", None)
                    if value is None:
                        value = getattr(result, "content", None)
                    await write({"type": "tool_result", "id": call_id, "result": value})
            except Exception as error:
                await write({
                    "type": "tool_result",
                    "id": call_id,
                    "error": f"{type(error).__name__}: {error}",
                })

        stderr_task = asyncio.create_task(drain_stderr())
        await write({
            "code": code,
            "catalog": catalog,
            "limits": {
                "timeoutMs": timeout_ms,
                "maxToolCalls": max_tool_calls,
            },
        })

        result = CodeModeResult()
        deadline = asyncio.get_running_loop().time() + timeout_ms / 1000.0
        try:
            while True:
                if process.stdout is None:
                    result.error = "Code Mode runtime has no stdout"
                    break
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    result.error = f"Code Mode timed out after {timeout_ms} ms"
                    break
                line = await asyncio.wait_for(process.stdout.readline(), timeout=remaining)
                if not line:
                    break
                try:
                    message = json.loads(line.decode("utf-8", errors="replace"))
                except ValueError:
                    continue
                kind = message.get("type")
                if kind == "tool_call":
                    task = asyncio.create_task(handle_call(message))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                elif kind == "final":
                    result.output = str(message.get("output") or "")
                    result.tool_calls = message.get("toolCalls") or []
                    result.logs = message.get("logs") or []
                    break
                elif kind == "error":
                    result.error = str(message.get("error") or "Code Mode failed")
                    result.tool_calls = message.get("toolCalls") or []
                    result.logs = message.get("logs") or []
                    break
        except asyncio.TimeoutError:
            result.error = f"Code Mode timed out after {timeout_ms} ms"
        except Exception as error:
            result.error = f"Code Mode runtime failed: {type(error).__name__}: {error}"
        finally:
            for task in list(tasks):
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if process.returncode is None:
                process.kill()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            await asyncio.gather(stderr_task, return_exceptions=True)

        if result.error and stderr_lines:
            result.error = result.error + "\n" + "\n".join(stderr_lines[-20:])
        if len(result.output) > max_output_chars:
            result.output = result.output[:max_output_chars] + "\n[…output truncated…]"
        return result
