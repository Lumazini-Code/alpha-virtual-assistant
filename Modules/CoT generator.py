from __future__ import annotations

import time
import hashlib
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional
from collections import OrderedDict
from pathlib import Path

import json
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Configuração ───────────────────────────────────────────────────────────────

LLAMA_SERVER_URL = "http://localhost:8081"
LLAMA_TIMEOUT_S  = 60.0

MAX_TOKENS      = 256
TEMPERATURE     = 0.1
TOP_P           = 0.9
REPEAT_PENALTY  = 1.1

PLAN_CACHE_SIZE = 64
CACHE_HIT_SCORE = 0.92

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ava.cot")

# ── Módulos disponíveis no AVA ─────────────────────────────────────────────────

AVA_MODULES = {
    "llm":        "responder perguntas, explicar conceitos, gerar texto, análise geral",
    "memory":     "buscar ou gravar informações de longo prazo sobre o usuário",
    "search":     "buscar informações atuais na internet",
    "vision":     "analisar imagens, descrever cenas, ler texto em imagens",
    "tts":        "converter texto em fala, ajustar voz ou velocidade",
    "stt":        "transcrever áudio para texto",
    "commander":  "executar comandos do sistema operacional, abrir programas",
    "translator": "traduzir texto entre idiomas",
    "calculator": "realizar cálculos matemáticos complexos",
}

MODULES_BLOCK = "\n".join(f"  - {k}: {v}" for k, v in AVA_MODULES.items())

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are a planning module for an AI assistant called AVA.
Your ONLY job is to break down a user request into a sequence of concrete, actionable steps.

Available modules:
{MODULES_BLOCK}

Rules:
1. Each step must be specific and executable — never vague or descriptive
2. Each step must assign exactly one executor from the available modules
3. Steps must be ordered by dependency — earlier steps feed into later ones
4. Generate between 2 and 7 steps — no more
5. Steps must be in the SAME LANGUAGE as the user input
6. Never include explanations outside the JSON structure
7. If a step needs output from a previous step, reference it as "result of step N"

BAD example (vague):
  {{"action": "analyze the problem", "executor": "llm"}}

GOOD example (actionable):
  {{"action": "search recent RTX 4070 vs RX 7800 XT benchmarks", "executor": "search"}}
  {{"action": "retrieve user GPU budget from memory", "executor": "memory"}}
  {{"action": "recommend best GPU based on result of step 1 and result of step 2", "executor": "llm"}}
"""

# ── Grammar GBNF ───────────────────────────────────────────────────────────────

GRAMMAR_GBNF = r"""root ::= "{" ws "\"steps\"" ws ":" ws steps-array ws "}"
steps-array ::= "[" ws step ("," ws step)* ws "]"
step ::= "{" ws "\"step\"" ws ":" ws number ws "," ws "\"action\"" ws ":" ws string ws "," ws "\"executor\"" ws ":" ws executor ws "}"
executor ::= "\"llm\"" | "\"memory\"" | "\"search\"" | "\"vision\"" | "\"tts\"" | "\"stt\"" | "\"commander\"" | "\"translator\"" | "\"calculator\""
number ::= [0-9]+
string ::= "\"" ([^"\\] | "\\" .)* "\""
ws ::= [ \t\n]*
"""
# ── Cache LRU Jaccard ──────────────────────────────────────────────────────────

class PlanCache:
    def __init__(self, max_size: int = PLAN_CACHE_SIZE):
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._max_size = max_size

    def _tokens(self, text: str) -> set[str]:
        return set(text.lower().split())

    def _jaccard(self, a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _key(self, text: str) -> str:
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def get(self, query: str) -> Optional[dict]:
        query_tokens = self._tokens(query)
        best_score, best_plan = 0.0, None
        for entry in self._cache.values():
            score = self._jaccard(query_tokens, entry["tokens"])
            if score > best_score:
                best_score, best_plan = score, entry["plan"]
        if best_score >= CACHE_HIT_SCORE:
            log.info(f"Cache hit (jaccard={best_score:.3f}): {query[:50]}")
            return best_plan
        return None

    def put(self, query: str, plan: dict):
        key = self._key(query)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = {"tokens": self._tokens(query), "plan": plan}

    def clear(self):
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


# ── Modelos de request/response ────────────────────────────────────────────────

class PlanRequest(BaseModel):
    input:     str
    context:   Optional[str] = None
    max_steps: Optional[int] = None
    use_cache: bool = True

class PlanStep(BaseModel):
    step:     int
    action:   str
    executor: str

class PlanResponse(BaseModel):
    steps:      list[PlanStep]
    input:      str
    from_cache: bool
    latency_ms: float


# ── Estado global ──────────────────────────────────────────────────────────────

@dataclass
class AppState:
    http_client: httpx.AsyncClient = field(default=None)
    cache:       PlanCache         = field(default=None)

state = AppState()


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Iniciando AVA CoT API...")

    # Verifica llama-server
    async with httpx.AsyncClient(timeout=5.0) as probe:
        try:
            r = await probe.get(f"{LLAMA_SERVER_URL}/health")
            if r.status_code != 200:
                raise RuntimeError(f"llama-server retornou {r.status_code}")
            log.info(f"llama-server OK em {LLAMA_SERVER_URL}")
        except httpx.ConnectError:
            raise RuntimeError(
                f"llama-server não encontrado em {LLAMA_SERVER_URL}\n"
                f"Inicie com: llama-server -m Phi-3.5-mini-instruct-Q4_K_M.gguf --port 8081"
            )

    # Cliente HTTP persistente com keep-alive
    state.http_client = httpx.AsyncClient(
        base_url = LLAMA_SERVER_URL,
        timeout  = httpx.Timeout(LLAMA_TIMEOUT_S),
        limits   = httpx.Limits(max_keepalive_connections=4, max_connections=8),
    )
    state.cache = PlanCache(max_size=PLAN_CACHE_SIZE)

    log.info("CoT API pronta")
    yield

    await state.http_client.aclose()
    log.info("AVA CoT API encerrada")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="AVA CoT API", lifespan=lifespan)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_prompt(user_input: str, context: Optional[str], max_steps: Optional[int]) -> str:
    steps_hint   = f" Use no máximo {max_steps} passos." if max_steps else ""
    user_content = f"Plan the following request:{steps_hint}\n\n{user_input}"
    if context:
        user_content += f"\n\nRelevant context:\n{context}"
    return (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{user_content}<|end|>\n"
        f"<|assistant|>\n"
    )


async def _call_llama_server(prompt: str) -> str:
    payload = {
        "prompt":         prompt,
        "grammar":        GRAMMAR_GBNF,
        "max_tokens":     MAX_TOKENS,
        "temperature":    TEMPERATURE,
        "top_p":          TOP_P,
        "repeat_penalty": REPEAT_PENALTY,
        "stream":         False,
        "cache_prompt":   True,   # reutiliza KV cache do system prompt entre chamadas
    }

    try:
        response = await state.http_client.post("/completion", json=payload)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"llama-server erro {e.response.status_code}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="llama-server timeout")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="llama-server não acessível")

    return response.json().get("content", "").strip()


def _parse_and_validate(raw: str) -> list[PlanStep]:
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON inválido: {e}")

    raw_steps = result.get("steps", [])
    if not raw_steps:
        raise HTTPException(status_code=500, detail="Modelo retornou plano vazio")

    valid_executors = set(AVA_MODULES.keys())
    steps = []
    for i, s in enumerate(raw_steps, start=1):
        executor = s.get("executor", "llm")
        if executor not in valid_executors:
            executor = "llm"
        action = s.get("action", "").strip()
        if action:
            steps.append(PlanStep(step=i, action=action, executor=executor))

    if not steps:
        raise HTTPException(status_code=500, detail="Nenhum passo válido gerado")

    return steps


# ── POST /plan ─────────────────────────────────────────────────────────────────

@app.post("/plan", response_model=PlanResponse)
async def plan(req: PlanRequest):
    user_input = req.input.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="input vazio")

    t0 = time.perf_counter()

    if req.use_cache:
        cached = state.cache.get(user_input)
        if cached:
            return PlanResponse(
                steps      = [PlanStep(**s) for s in cached["steps"]],
                input      = user_input,
                from_cache = True,
                latency_ms = round((time.perf_counter() - t0) * 1000, 2),
            )

    prompt = _build_prompt(user_input, req.context, req.max_steps)
    raw    = await _call_llama_server(prompt)
    steps  = _parse_and_validate(raw)

    if req.use_cache:
        state.cache.put(user_input, {"steps": [s.model_dump() for s in steps]})

    latency = round((time.perf_counter() - t0) * 1000, 2)
    log.info(f"Plano em {latency}ms — {len(steps)} passos: {user_input[:60]}")

    return PlanResponse(
        steps      = steps,
        input      = user_input,
        from_cache = False,
        latency_ms = latency,
    )


# ── GET /status ────────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    try:
        r = await state.http_client.get("/health", timeout=2.0)
        llama_ok = r.status_code == 200
    except Exception:
        llama_ok = False

    return {
        "llama_server":  LLAMA_SERVER_URL,
        "llama_healthy": llama_ok,
        "cache_size":    state.cache.size,
        "cache_max":     PLAN_CACHE_SIZE,
        "modules":       list(AVA_MODULES.keys()),
    }


# ── DELETE /cache ──────────────────────────────────────────────────────────────

@app.delete("/cache")
async def clear_cache():
    state.cache.clear()
    return {"cleared": True}


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")