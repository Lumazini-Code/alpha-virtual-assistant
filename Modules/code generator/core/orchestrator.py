"""
Core Orchestrator
Controls the full pipeline:
  input → semantic classification → memory → research → generation → sandbox → feedback loop
"""

import asyncio
import logging
from typing import Optional

from modules.semantic_classifier import SemanticClassifier
from modules.context_ranker import ContextRanker
from modules.memory_module import MemoryModule
from modules.research_module import ResearchModule
from modules.code_generator import CodeGenerator
from modules.confidence_scorer import ConfidenceScorer
from sandbox.executor import SandboxExecutor

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


class Orchestrator:
    def __init__(self):
        self.classifier   = SemanticClassifier()
        self.ranker       = ContextRanker()
        self.memory       = MemoryModule()
        self.research     = ResearchModule()
        self.generator    = CodeGenerator()
        self.scorer       = ConfidenceScorer()
        self.sandbox      = SandboxExecutor()

    async def startup(self):
        """Load all ONNX models and warm up the llama.cpp server connection."""
        logger.info("Starting up pipeline modules...")
        await asyncio.gather(
            self.classifier.load(),
            self.scorer.load(),
            self.memory.connect(),
            self.research.connect(),
            self.generator.connect(),
        )
        logger.info("All modules ready.")

    async def shutdown(self):
        await self.generator.disconnect()
        await self.memory.disconnect()

    # ─── Main Pipeline ────────────────────────────────────────────────────────

    async def run(self, payload: dict) -> dict:
        prompt    = payload["prompt"]
        language  = payload.get("language", "python")
        context   = payload.get("context_files", [])
        prefs     = payload.get("preferences", {})

        # 1. Semantic classification: write | refactor | fix
        intent = await self.classifier.classify(prompt)
        logger.info(f"Intent classified as: {intent}")

        # 2. Parallel fetch: memory + research
        memory_ctx, research_ctx = await asyncio.gather(
            self.memory.retrieve(prompt=prompt, language=language, intent=intent),
            self.research.query(topic=prompt, language=language),
        )

        # 3. Rank and merge all context sources (resolves contradictions)
        ranked_context = self.ranker.rank(
            user_input=prompt,
            memory=memory_ctx,
            research=research_ctx,
            inline_context=context,
            preferences={**memory_ctx.get("preferences", {}), **prefs},
        )

        # 4. Generation + sandbox loop with auto bug-fix escalation
        code, iterations, sandbox_passed = await self._generation_loop(
            prompt=prompt,
            language=language,
            intent=intent,
            context=ranked_context,
        )

        # 5. Score final output
        confidence = await self.scorer.score(
            code=code,
            language=language,
            prompt=prompt,
        )

        return {
            "intent": intent,
            "language": language,
            "code": code,
            "confidence_score": confidence,
            "sandbox_passed": sandbox_passed,
            "iterations": iterations,
            "sources_used": {
                "memory_hits": len(memory_ctx.get("entries", [])),
                "research_hits": len(research_ctx),
            },
        }

    # ─── Generation + Sandbox Loop ────────────────────────────────────────────

    async def _generation_loop(
        self,
        prompt: str,
        language: str,
        intent: str,
        context: dict,
    ) -> tuple[str, int, bool]:
        """
        Attempts to generate and validate code up to MAX_RETRIES times.
        On each failure, switches to bug-fix mode and feeds the error back
        into the generator as additional context.
        """
        current_intent = intent
        error_context: Optional[str] = None
        code = ""

        for iteration in range(1, MAX_RETRIES + 1):
            logger.info(f"Generation attempt {iteration}/{MAX_RETRIES} — mode: {current_intent}")

            code = await self.generator.generate(
                prompt=prompt,
                language=language,
                intent=current_intent,
                context=context,
                error_context=error_context,
            )

            # Pre-execution confidence gate
            pre_score = await self.scorer.score(code=code, language=language, prompt=prompt)
            if pre_score < 0.35:
                logger.warning(f"Pre-score too low ({pre_score:.2f}), retrying without sandbox.")
                current_intent = "fix"
                error_context = f"Previous attempt had low confidence score: {pre_score:.2f}. Improve correctness and edge-case coverage."
                continue

            # Sandbox execution
            sandbox_result = await self.sandbox.execute(code=code, language=language)

            if sandbox_result["passed"]:
                logger.info(f"Sandbox passed on iteration {iteration}.")
                return code, iteration, True

            # Failed — switch to bug-fix mode
            error_context = sandbox_result["error"]
            current_intent = "fix"
            logger.warning(f"Sandbox failed (iter {iteration}): {error_context[:120]}")

        # Exhausted retries — return last generated code with failure flag
        logger.error("Max retries reached. Returning last generated code with sandbox_passed=False.")
        return code, MAX_RETRIES, False
