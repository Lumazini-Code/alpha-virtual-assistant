"""
Code Generator
Sends structured prompts to a running llama.cpp server (GGUF model)
via its OpenAI-compatible /v1/chat/completions endpoint.
Uses chat format — required for instruction-tuned models like Gemma.
"""

import logging
import os
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

LLAMA_CPP_BASE_URL  = os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080")
CHAT_ENDPOINT       = f"{LLAMA_CPP_BASE_URL}/v1/chat/completions"
HEALTH_ENDPOINT     = f"{LLAMA_CPP_BASE_URL}/health"

TEMPERATURE    = 0.2
TOP_P          = 0.95
REPEAT_PENALTY = 1.1
TIMEOUT        = 120

SYSTEM_PROMPT = """
You are a code generation engine.

Rules:
- Output only valid code.
- Be concise.
- Prefer minimal implementations.
- No tutorial-style code.
- No comments unless required.
- No docstrings unless requested.
- Avoid excessive error handling.
- Avoid unnecessary prints.
- Never output explanations.
"""

INTENT_INSTRUCTIONS = {
    "write":
        "Write concise {language} code for the following request:",

    "refactor":
        "Refactor this {language} code while preserving behavior. Keep it concise:",

    "fix":
        "Fix the following {language} code. Return only the corrected code:",
}


class CodeGenerator:
    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None

    async def connect(self):
        self.client = httpx.AsyncClient(timeout=TIMEOUT)
        try:
            resp = await self.client.get(HEALTH_ENDPOINT)
            resp.raise_for_status()
            logger.info(f"CodeGenerator: llama.cpp server healthy at {LLAMA_CPP_BASE_URL}")
        except Exception as e:
            logger.warning(f"CodeGenerator: llama.cpp server unreachable — {e}. Will retry on first request.")

    async def disconnect(self):
        if self.client:
            await self.client.aclose()

    # ─── Generation ───────────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        language: str,
        intent: str,
        context: dict,
        error_context: Optional[str] = None,
    ) -> str:
        messages = self._build_messages(
            prompt=prompt,
            language=language,
            intent=intent,
            context=context,
            error_context=error_context,
        )

        payload = {
            "messages":       messages,
            "temperature":    TEMPERATURE if intent != "fix" else 0.05,
            "top_p":          TOP_P,
            "repeat_penalty": REPEAT_PENALTY,
        }

        try:
            resp = await self.client.post(CHAT_ENDPOINT, json=payload)
            resp.raise_for_status()
            data = resp.json()
            code = data["choices"][0]["message"]["content"].strip()
            logger.info(f"CodeGenerator: received {len(code)} chars from llama.cpp.")
            return self._clean_output(code)
        except httpx.TimeoutException:
            raise RuntimeError("CodeGenerator: llama.cpp server timed out.")
        except Exception as e:
            raise RuntimeError(f"CodeGenerator: request failed — {e}")

    # ─── Message Builder ──────────────────────────────────────────────────────

    def _build_messages(
        self,
        prompt: str,
        language: str,
        intent: str,
        context: dict,
        error_context: Optional[str],
    ) -> list[dict]:
        # System message
        system_content = SYSTEM_PROMPT

        # Add preferences to system if available
        prefs = context.get("preferences", {})
        if prefs:
            pref_lines = "\n".join(f"  - {k}: {v}" for k, v in prefs.items())
            system_content += f"\nUser preferences:\n{pref_lines}\n"

        # User message
        instruction = INTENT_INSTRUCTIONS.get(intent, INTENT_INSTRUCTIONS["write"]).format(language=language)
        user_parts  = [instruction, f"\n{prompt}"]

        # Context sections
        sections = context.get("sections", [])
        if sections:
            user_parts.append("\n\nRelevant context (use as reference):")
            for section in sections[:6]:
                src     = section.get("source", "")
                content = section.get("content", "")
                user_parts.append(f"\n[{src.upper()}]\n{content}")

        # Error context (bug-fix mode)
        if error_context:
            user_parts.append(f"\n\nPrevious attempt failed with:\n{error_context}\nFix this error in your output.")

        return [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": "".join(user_parts)},
        ]

    # ─── Output Cleaning ──────────────────────────────────────────────────────

    def _clean_output(self, raw: str) -> str:
        """Strip markdown fences if the model added them despite instructions."""
        lines = raw.splitlines()

        # Drop opening fence line (```python, ```py, ```, etc.)
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        # Drop closing fence line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        return "\n".join(lines).strip()