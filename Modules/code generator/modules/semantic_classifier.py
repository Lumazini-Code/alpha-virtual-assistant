"""
Semantic Classifier — ONNX Runtime
Classifies user intent into: write | refactor | fix

Uses paraphrase-multilingual-MiniLM-L12-v2 as a feature extractor.
Intent is determined by cosine similarity against reference phrase embeddings,
NOT by logits — because MiniLM is an embedder, not a classifier.
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

INTENT_LABELS = ["write", "refactor", "fix"]

# Reference phrases per intent — the prompt embedding is compared against these.
# More phrases = more robust classification.
INTENT_REFERENCES = {
    "write": [
        "write a function", "create a class", "implement an algorithm",
        "build a module", "generate code", "escrever uma função",
        "criar um script", "implementar uma solução",
    ],
    "refactor": [
        "refactor this code", "improve code quality", "optimize performance",
        "simplify this function", "clean up the code", "reorganize the module",
        "refatorar o código", "melhorar a qualidade", "otimizar",
    ],
    "fix": [
        "fix this bug", "there is an error", "the code is broken",
        "correct this exception", "debug this crash", "not working",
        "corrigir o erro", "tem um bug", "está quebrando",
    ],
}

MODEL_PATH     = Path(__file__).parent.parent / "models" / "classifier" / "model.onnx"
TOKENIZER_PATH = Path(__file__).parent.parent / "models" / "classifier"


class SemanticClassifier:
    def __init__(self):
        self.session   = None
        self.tokenizer = None
        self.reference_embeddings: dict[str, np.ndarray] = {}

    async def load(self):
        """Load ONNX embedder and pre-compute reference embeddings."""
        if not MODEL_PATH.exists():
            logger.warning("SemanticClassifier: model not found — using keyword heuristic.")
            return
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 2
            self.session = ort.InferenceSession(
                str(MODEL_PATH),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH / "tokenizer.json"))

            # Pre-compute mean embedding per intent class
            for intent, phrases in INTENT_REFERENCES.items():
                embs = [self._embed(p) for p in phrases]
                embs = [e for e in embs if e is not None]
                if embs:
                    mean = np.mean(embs, axis=0)
                    self.reference_embeddings[intent] = mean / (np.linalg.norm(mean) + 1e-9)

            logger.info(f"SemanticClassifier: loaded. Reference intents: {list(self.reference_embeddings.keys())}")

        except Exception as e:
            logger.warning(f"SemanticClassifier: load failed ({e}) — using keyword heuristic.")
            self.session   = None
            self.tokenizer = None
            self.reference_embeddings = {}

    async def classify(self, prompt: str) -> str:
        if self.session and self.tokenizer and self.reference_embeddings:
            return self._classify_semantic(prompt)
        return self._classify_keywords(prompt)

    # ── Semantic classification via cosine similarity ─────────────────────────

    def _classify_semantic(self, prompt: str) -> str:
        prompt_emb = self._embed(prompt)
        if prompt_emb is None:
            return self._classify_keywords(prompt)

        best_intent = "write"
        best_score  = -1.0

        for intent, ref_emb in self.reference_embeddings.items():
            score = float(np.dot(prompt_emb, ref_emb))
            if score > best_score:
                best_score  = score
                best_intent = intent

        logger.debug(f"SemanticClassifier: '{best_intent}' (score={best_score:.3f})")
        return best_intent

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str):
        try:
            enc  = self.tokenizer.encode(text[:512])
            ids  = np.array([enc.ids],            dtype=np.int64)
            mask = np.array([enc.attention_mask],  dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}

            expected = {i.name for i in self.session.get_inputs()}
            if "token_type_ids" in expected:
                feed["token_type_ids"] = np.zeros_like(ids, dtype=np.int64)

            out = self.session.run(None, feed)
            # Mean-pool token embeddings (last hidden state)
            emb = out[0][0].mean(axis=0).astype(np.float32)
            return emb / (np.linalg.norm(emb) + 1e-9)
        except Exception as e:
            logger.warning(f"SemanticClassifier embed error: {e}")
            return None

    # ── Keyword fallback ──────────────────────────────────────────────────────

    KEYWORDS = {
        "fix":      ["fix", "bug", "error", "broken", "crash", "exception",
                     "not working", "corrigir", "erro", "quebrado"],
        "refactor": ["refactor", "improve", "optimize", "clean", "simplify",
                     "reorganize", "refatorar", "melhorar", "otimizar"],
        "write":    ["write", "create", "build", "implement", "generate",
                     "make", "escrever", "criar", "construir"],
    }

    def _classify_keywords(self, prompt: str) -> str:
        lower  = prompt.lower()
        scores = {intent: sum(1 for kw in kws if kw in lower)
                  for intent, kws in self.KEYWORDS.items()}
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "write"