"""
Confidence Scorer — ONNX Runtime
Scores generated code before sandbox execution.
Evaluates: syntactic validity, cyclomatic complexity, edge-case coverage,
style adherence, and semantic alignment with the original prompt.

Returns a float in [0.0, 1.0]. Scores below 0.35 trigger a retry.
"""

import ast
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

MODEL_PATH     = Path(__file__).parent.parent / "models" / "scorer" / "model.onnx"
TOKENIZER_PATH = Path(__file__).parent.parent / "models" / "scorer"

LOW_SCORE_THRESHOLD = 0.35


class ConfidenceScorer:
    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.tokenizer = None

    async def load(self):
        if MODEL_PATH.exists():
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            self.session = ort.InferenceSession(
                str(MODEL_PATH),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            try:
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH / "tokenizer.json"))
                logger.info("ConfidenceScorer: ONNX model loaded.")
            except Exception as e:
                logger.warning(f"ConfidenceScorer: tokenizer unavailable — using heuristic scorer. {e}")
                self.session = None
        else:
            logger.warning("ConfidenceScorer: ONNX model not found — using heuristic scorer.")

    async def score(self, code: str, language: str, prompt: str) -> float:
        """Returns confidence in [0.0, 1.0]."""
        if self.session and self.tokenizer:
            return self._score_onnx(code, prompt)
        return self._score_heuristic(code, language, prompt)

    # ─── ONNX Scoring ─────────────────────────────────────────────────────────

    def _score_onnx(self, code: str, prompt: str) -> float:
        combined = f"PROMPT: {prompt[:256]}\nCODE: {code[:512]}"
        enc  = self.tokenizer.encode(combined)
        ids  = np.array([enc.ids],            dtype=np.int64)
        mask = np.array([enc.attention_mask],  dtype=np.int64)
        out  = self.session.run(None, {"input_ids": ids, "attention_mask": mask})
        score = float(out[0][0])  # Expects a regression head output in [0,1]
        logger.debug(f"ConfidenceScorer (ONNX): {score:.3f}")
        return max(0.0, min(1.0, score))

    # ─── Heuristic Scoring ────────────────────────────────────────────────────

    def _score_heuristic(self, code: str, language: str, prompt: str) -> float:
        """
        Rule-based fallback. Combines multiple signals:
          1. Syntax validity (Python AST parse)
          2. Code non-emptiness
          3. Complexity penalty (too many nested ifs)
          4. Keyword alignment with prompt
          5. Presence of error handling
        """
        if not code or len(code.strip()) < 10:
            return 0.0

        score = 0.5  # Base

        # 1. Syntax check (Python only for now)
        if language == "python":
            try:
                ast.parse(code)
                score += 0.20
            except SyntaxError as e:
                logger.debug(f"Syntax error in generated code: {e}")
                score -= 0.25

        # 2. Reward reasonable length
        lines = code.strip().splitlines()
        if 5 <= len(lines) <= 300:
            score += 0.05
        elif len(lines) > 300:
            score -= 0.05

        # 3. Complexity penalty
        nesting = self._max_nesting_depth(lines)
        if nesting > 6:
            score -= 0.10

        # 4. Keyword alignment with prompt
        prompt_keywords = set(re.findall(r'\b\w{4,}\b', prompt.lower()))
        code_lower = code.lower()
        matched = sum(1 for kw in prompt_keywords if kw in code_lower)
        alignment = matched / max(len(prompt_keywords), 1)
        score += alignment * 0.15

        # 5. Error handling presence
        if language == "python" and ("try:" in code or "except" in code):
            score += 0.05
        elif language in ("javascript", "typescript") and "try {" in code:
            score += 0.05

        final = max(0.0, min(1.0, score))
        logger.debug(f"ConfidenceScorer (heuristic): {final:.3f}")
        return final

    @staticmethod
    def _max_nesting_depth(lines: list[str]) -> int:
        max_depth = 0
        for line in lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            depth = indent // 4
            max_depth = max(max_depth, depth)
        return max_depth
