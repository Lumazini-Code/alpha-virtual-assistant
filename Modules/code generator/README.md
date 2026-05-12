# CodeGen API

Modular AI code generation pipeline exposed as a REST API.

## Architecture

```
POST /generate
      │
      ▼
 SemanticClassifier (ONNX)          → intent: write | refactor | fix
      │
      ├──── MemoryModule (ONNX + SQLite)   → preferences + past bug fixes
      └──── ResearchModule (ONNX + index)  → docs + patterns
                    │
                    ▼
             ContextRanker                 → merges + deduplicates + conflict-resolves
                    │
                    ▼
            CodeGenerator                 → llama.cpp GGUF server (HTTP)
                    │
                    ▼
           ConfidenceScorer (ONNX)        → pre-flight quality gate
                    │
              score < 0.35? ──────────────► retry (mode=fix)
                    │
                    ▼
           SandboxExecutor (Docker)       → isolated container, resource-limited
                    │
              failed? ────────────────────► retry with error context (max 3)
                    │
                    ▼
              Return code + metadata
```

## Endpoints

| Method | Path                    | Description                                              |
|--------|-------------------------|----------------------------------------------------------|
| POST   | `/api/v1/generate`      | Submit a code generation request                         |
| POST   | `/api/v1/feedback`      | Send feedback to close the learning loop                 |
| POST   | `/api/v1/memory/preference` | Store a user preference                              |
| POST   | `/api/v1/research`      | Trigger a standalone research query                      |

---

## Quickstart

### 1. GGUF model

Place your GGUF model at `models/gguf/codegen.gguf`.
Recommended: any CodeLlama, DeepSeek-Coder, or Qwen2.5-Coder GGUF.

### 2. ONNX models

Place ONNX models at:
```
models/
  classifier/   model.onnx + tokenizer.json   (intent classifier)
  embedder/     model.onnx + tokenizer.json   (sentence embedder — shared)
  scorer/       model.onnx + tokenizer.json   (confidence scorer)
```

If models are absent, the system falls back to keyword heuristics automatically.

### 3. Run

```bash
docker compose up
```

API will be available at `http://localhost:8000`.
Interactive docs at `http://localhost:8000/docs`.

---

## Example Requests

### Generate code

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a binary search function with input validation",
    "language": "python"
  }'
```

Response:
```json
{
  "request_id": "abc-123",
  "status": "success",
  "intent": "write",
  "language": "python",
  "code": "def binary_search(arr, target):\n    ...",
  "confidence_score": 0.91,
  "sandbox_passed": true,
  "iterations": 1,
  "sources_used": { "memory_hits": 2, "research_hits": 3 }
}
```

### Store a preference

```bash
curl -X POST http://localhost:8000/api/v1/memory/preference \
  -H "Content-Type: application/json" \
  -d '{"key": "indent_style", "value": "4 spaces", "scope": "python"}'
```

### Submit feedback

```bash
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "abc-123",
    "accepted": false,
    "modifications": "def binary_search(arr: list, target: int) -> int:\n    ...",
    "notes": "Always add type hints"
  }'
```

---

## Design Decisions

### Why ONNX for small models?
No PyTorch/TensorFlow runtime needed. Fast cold start, low memory, CPU-optimised.

### Why llama.cpp for the LLM?
GGUF quantisation gives strong code quality at low VRAM. The OpenAI-compatible
HTTP server makes it a drop-in for future cloud LLM swaps.

### Why Docker sandbox?
Hard isolation: no network, read-only filesystem, memory + CPU caps, timeout kill.
The pipeline never trusts its own output.

### Why MAX_RETRIES = 3?
Empirically, if a model can't fix a bug in 3 attempts with full error context,
additional attempts yield diminishing returns and the error is likely
a misunderstood requirement — better to surface it to the user.

### Feedback loop
Every user correction (via `/feedback`) is embedded and stored. On the next
similar request, that corrected version surfaces as a high-weight memory entry,
nudging the generator toward the user's actual preferences.
