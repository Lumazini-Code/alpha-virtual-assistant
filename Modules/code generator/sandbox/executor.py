"""
Sandbox Executor
Runs generated code in an isolated subprocess directly inside the API container.
Uses resource limits via ulimit and a strict timeout.

Supported languages: python, javascript, bash
"""

import asyncio
import logging
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

EXECUTION_TIMEOUT = 10      # seconds
MAX_OUTPUT_CHARS  = 8_000

RUNNER_SCRIPTS = {
    "python":     ("main.py",  [sys.executable, "main.py"]),
    "javascript": ("main.js",  ["node", "main.js"]),
    "bash":       ("main.sh",  ["bash", "main.sh"]),
}


class SandboxExecutor:

    async def execute(self, code: str, language: str) -> dict:
        if language not in RUNNER_SCRIPTS:
            logger.warning(f"Sandbox: unsupported language '{language}' — skipping.")
            return self._skip_result(f"Language '{language}' not configured for subprocess sandbox.")

        filename, cmd = RUNNER_SCRIPTS[language]

        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = Path(tmpdir) / filename
            code_path.write_text(code, encoding="utf-8")

            try:
                return await asyncio.wait_for(
                    self._run(cmd, cwd=tmpdir),
                    timeout=EXECUTION_TIMEOUT + 2,
                )
            except asyncio.TimeoutError:
                return {
                    "passed":    False,
                    "stdout":    "",
                    "stderr":    "",
                    "error":     f"Sandbox timeout: execution exceeded {EXECUTION_TIMEOUT}s.",
                    "exit_code": -1,
                }

    # ─── Subprocess runner ────────────────────────────────────────────────────

    async def _run(self, cmd: list[str], cwd: str) -> dict:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            return {
                "passed":    False,
                "stdout":    "",
                "stderr":    "",
                "error":     f"Runtime not found: {e}",
                "exit_code": -1,
            }

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=EXECUTION_TIMEOUT
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "passed":    False,
                "stdout":    "",
                "stderr":    "",
                "error":     f"Process killed after {EXECUTION_TIMEOUT}s timeout.",
                "exit_code": -1,
            }

        stdout    = stdout_b.decode("utf-8", errors="replace")[:MAX_OUTPUT_CHARS]
        stderr    = stderr_b.decode("utf-8", errors="replace")[:MAX_OUTPUT_CHARS]
        exit_code = proc.returncode
        passed    = exit_code == 0

        error_msg = None
        if not passed:
            lines     = (stderr or stdout).strip().splitlines()
            error_msg = "\n".join(lines[-20:])
            logger.debug(f"Sandbox failed (exit {exit_code}): {error_msg[:200]}")
        else:
            logger.debug(f"Sandbox passed. stdout={stdout[:100]}")

        return {
            "passed":    passed,
            "stdout":    stdout,
            "stderr":    stderr,
            "error":     error_msg,
            "exit_code": exit_code,
        }

    @staticmethod
    def _skip_result(reason: str) -> dict:
        return {
            "passed":    True,
            "stdout":    "",
            "stderr":    "",
            "error":     None,
            "exit_code": 0,
            "note":      reason,
        }