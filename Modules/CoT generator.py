from __future__ import annotations

import time
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Configuração ───────────────────────────────────────────────────────────────

LLAMA_SERVER_URL  = "http://localhost:2001"
LLAMA_TIMEOUT_S   = 60.0
MEMORY_API_URL    = "http://localhost:3001"
MEMORY_TIMEOUT_S  = 5.0

MAX_TOKENS     = 220   # margem para 7 steps com "local_scraping" (~14 chars)
TEMPERATURE    = 0.1
TOP_P          = 0.9
REPEAT_PENALTY = 1.1

CACHE_HIT_THRESHOLD = 0.92

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [COT] %(message)s")
log = logging.getLogger("ava.cot")


# ── Módulos disponíveis no AVA ─────────────────────────────────────────────────

AVA_MODULES: dict[str, str] = {
    "llm":            "responder perguntas, explicar conceitos, gerar texto, análise geral",
    "memory":         "buscar ou gravar informações de longo prazo sobre o usuário",
    "search":         "buscar informações atuais na internet",
    "vision":         "analisar imagens, descrever cenas, ler texto em imagens",
    "tts":            "converter texto em fala, ajustar voz ou velocidade",
    "stt":            "transcrever áudio para texto",
    "translator":     "traduzir texto entre idiomas",
    "calculator":     "realizar cálculos matemáticos complexos",
    "local_scraping": "buscar e ler arquivos locais no computador do usuário, indexar conteúdo de documentos locais",
    "deep_search":     "realizar pesquisas web automáticas avançadas com múltiplas etapas, agregando e resumindo informações de várias fontes online e aprendendo com base nelas",
}

MODULES_BLOCK = "\n".join(f"  - {k}: {v}" for k, v in AVA_MODULES.items())

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are a planning module.
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
8. Mark independent steps with depends_on:null; dependent steps list their dependencies
9. Use "local_scraping" (NOT "commander") when the user wants to READ or SEARCH a local file on their computer
10. Use "commander" only to OPEN/launch applications or execute OS commands — never to read file contents
11. When a file is read via local_scraping and the user asks about its content, the next step should use "llm" to analyze result of step N
12. When is asked to learn or research something, don't use search, instead use deep search to make a deep search on the topic and learn from it

Output format — output ONLY the inner content, starting from the first step object:
{{"step":1,"action":"...","executor":"llm","depends_on":null}},{{"step":2,"action":"...","executor":"search","depends_on":[1]}}]}}

Examples:
Input: "what GPU should I buy for gaming under R$2000"
Output: {{"step":1,"action":"search RTX 4060 RX 7600 benchmark price Brazil 2024","executor":"search","depends_on":null}},{{"step":2,"action":"retrieve user GPU preferences from memory","executor":"memory","depends_on":null}},{{"step":3,"action":"recommend GPU based on result of step 1 and result of step 2","executor":"llm","depends_on":[1,2]}}]}}

Input: "play some jazz music"
Output: {{"step":1,"action":"search jazz playlist Spotify","executor":"search","depends_on":null}},{{"step":2,"action":"open music player with result of step 1","executor":"commander","depends_on":[1]}}]}}

Input: "leia o arquivo relatório de vendas e me diga o total"
Output: {{"step":1,"action":"buscar e ler arquivo relatório de vendas","executor":"local_scraping","depends_on":null}},{{"step":2,"action":"analisar o conteúdo e informar o total de vendas com base em result of step 1","executor":"llm","depends_on":[1]}}]}}

Input: "read the local report file and compare with web search results"
Output: {{"step":1,"action":"search latest industry report data online","executor":"search","depends_on":null}},{{"step":2,"action":"buscar e ler arquivo de relatório local","executor":"local_scraping","depends_on":null}},{{"step":3,"action":"compare result of step 1 with result of step 2 and summarize key differences","executor":"llm","depends_on":[1,2]}}]}}

Input: "leia o arquivo notas.txt e grave as informações na memória"
Output: {{"step":1,"action":"buscar e ler arquivo notas.txt","executor":"local_scraping","depends_on":null}},{{"step":2,"action":"gravar na memória as informações de result of step 1","executor":"memory","depends_on":[1]}}]}}

Input: "traduza o conteúdo do arquivo contrato.docx para inglês"
Output: {{"step":1,"action":"buscar e ler arquivo contrato.docx","executor":"local_scraping","depends_on":null}},{{"step":2,"action":"traduzir result of step 1 para inglês","executor":"translator","depends_on":[1]}}]}}

Input: "leia o relatório financeiro e calcule a soma das despesas"
Output: {{"step":1,"action":"buscar e ler arquivo relatório financeiro","executor":"local_scraping","depends_on":null}},{{"step":2,"action":"calcular a soma das despesas em result of step 1","executor":"calculator","depends_on":[1]}}]}}
"""

# ── Grammar GBNF otimizada ─────────────────────────────────────────────────────

GRAMMAR_GBNF = r"""root        ::= steps-cont "]}"
steps-cont  ::= step ("," step)*
step        ::= "{\"step\":" step-num ",\"action\":" string ",\"executor\":" executor ",\"depends_on\":" depends "}"
step-num    ::= [1-7]
executor    ::= "\"llm\"" | "\"memory\"" | "\"search\"" | "\"vision\"" | "\"tts\"" | "\"stt\"" | "\"commander\"" | "\"translator\"" | "\"calculator\"" | "\"local_scraping\""
depends     ::= "null" | "[" step-num ("," step-num)* "]"
string      ::= "\"" ([^"\\] | "\\" .)* "\""
"""

JSON_PREFIX = '{"steps":['


# ── Modelos de request/response ────────────────────────────────────────────────

class PlanRequest(BaseModel):
    input:      str
    context:    Optional[str] = None
    max_steps:  Optional[int] = None
    use_cache:  bool = True
    cache_threshold: float = CACHE_HIT_THRESHOLD

class PlanStep(BaseModel):
    step:       int
    action:     str
    executor:   str
    depends_on: Optional[list[int]] = None

class PlanResponse(BaseModel):
    steps:         list[PlanStep]
    input:         str
    from_cache:    bool
    cache_score:   Optional[float] = None
    tokens_used:   Optional[int]   = None
    latency_ms:    float


# ── Estado global ──────────────────────────────────────────────────────────────

@dataclass
class AppState:
    llama_client:  httpx.AsyncClient = field(default=None)
    memory_client: httpx.AsyncClient = field(default=None)

state = AppState()


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    async with httpx.AsyncClient() as probe:
        for attempt in range(60):
            try:
                r = await probe.get(f"{LLAMA_SERVER_URL}/health", timeout=5)
                if r.status_code == 200:
                    log.info(f"[CoT] llama-server pronto após {attempt * 5}s")
                    break
            except Exception:
                pass
            log.info(f"[CoT] Aguardando llama-server... ({attempt+1}/60)")
            await asyncio.sleep(5)
        else:
            raise RuntimeError(f"llama-server não respondeu em {LLAMA_SERVER_URL}")

    state.llama_client = httpx.AsyncClient(
        base_url=LLAMA_SERVER_URL,
        timeout=httpx.Timeout(LLAMA_TIMEOUT_S),
        limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
    )
    state.memory_client = httpx.AsyncClient(
        base_url=MEMORY_API_URL,
        timeout=httpx.Timeout(MEMORY_TIMEOUT_S),
        limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
    )
    log.info("CoT API pronta")

    yield

    await state.llama_client.aclose()
    await state.memory_client.aclose()
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
        f" />\n{SYSTEM_PROMPT}<|end|>\n"
        f" />\n{user_content}<|end|>\n"
        f" />\n{JSON_PREFIX}"
    )


async def _call_llama(prompt: str) -> tuple[str, int]:
    payload = {
        "prompt":         prompt,
        "grammar":        GRAMMAR_GBNF,
        "max_tokens":     MAX_TOKENS,
        "temperature":    TEMPERATURE,
        "top_p":          TOP_P,
        "repeat_penalty": REPEAT_PENALTY,
        "stream":         False,
        "cache_prompt":   True,
    }
    try:
        response = await state.llama_client.post("/completion", json=payload)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"llama-server erro {e.response.status_code}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="llama-server timeout")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="llama-server não acessível")

    data        = response.json()
    raw         = data.get("content", "").strip()
    tokens_used = data.get("tokens_predicted", 0)
    return raw, tokens_used


def _parse_steps(raw: str) -> list[PlanStep]:
    valid_executors = set(AVA_MODULES.keys())
    full_json       = JSON_PREFIX + raw

    try:
        result    = json.loads(full_json)
        raw_steps = result.get("steps", [])
    except json.JSONDecodeError:
        log.warning("JSON incompleto do llama-server — tentando recuperação parcial")
        raw_steps = _recover_partial_steps(raw)
        if not raw_steps:
            raise HTTPException(status_code=500, detail="JSON inválido e sem steps recuperáveis")

    if not raw_steps:
        raise HTTPException(status_code=500, detail="Modelo retornou plano vazio")

    steps = []
    for i, s in enumerate(raw_steps, start=1):
        executor = s.get("executor", "llm")
        if executor not in valid_executors:
            log.warning(f"Executor inválido '{executor}' no step {i} — substituído por 'llm'")
            executor = "llm"
        action = s.get("action", "").strip()
        if not action:
            continue
        depends_on = s.get("depends_on")
        if isinstance(depends_on, list):
            depends_on = [d for d in depends_on if isinstance(d, int) and d < i]
            depends_on = depends_on or None
        elif depends_on is not None:
            depends_on = None
        steps.append(PlanStep(step=i, action=action, executor=executor, depends_on=depends_on))

    if not steps:
        raise HTTPException(status_code=500, detail="Nenhum step válido após validação")

    return steps


def _recover_partial_steps(raw: str) -> list[dict]:
    steps = []
    depth, start = 0, None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(raw[start:i + 1])
                    if "action" in obj and "executor" in obj:
                        steps.append(obj)
                except json.JSONDecodeError:
                    pass
    return steps


# ── Cache semântico via Memory API ─────────────────────────────────────────────

async def _cache_get(query: str, threshold: float) -> Optional[tuple[dict, float, int]]:
    try:
        r = await state.memory_client.post(
            "/cache/get",
            json={"query": query, "threshold": threshold},
        )
        r.raise_for_status()
        data = r.json()
        if data.get("hit"):
            return data["plan"], data["score"], data["cache_id"]
    except Exception as e:
        log.warning(f"cache_get falhou (continuando sem cache): {e}")
    return None


async def _cache_put(query: str, plan: dict):
    try:
        r = await state.memory_client.post(
            "/cache/put",
            json={"query": query, "plan": plan},
        )
        r.raise_for_status()
    except Exception as e:
        log.warning(f"cache_put falhou (plano não cacheado): {e}")


# ── POST /plan ─────────────────────────────────────────────────────────────────

@app.post("/plan", response_model=PlanResponse)
async def plan(req: PlanRequest):
    user_input = req.input.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="input vazio")

    t0 = time.perf_counter()

    if req.use_cache:
        cache_result = await _cache_get(user_input, req.cache_threshold)
        if cache_result:
            cached_plan, score, _ = cache_result
            steps = [PlanStep(**s) for s in cached_plan["steps"]]
            latency = round((time.perf_counter() - t0) * 1000, 2)
            log.info(f"Cache HIT em {latency}ms — score={score:.3f}: {user_input[:60]}")
            return PlanResponse(
                steps=steps, input=user_input, from_cache=True,
                cache_score=score, latency_ms=latency,
            )

    prompt      = _build_prompt(user_input, req.context, req.max_steps)
    raw, tokens = await _call_llama(prompt)
    steps       = _parse_steps(raw)

    if req.use_cache:
        plan_dict = {"steps": [s.model_dump() for s in steps]}
        await _cache_put(user_input, plan_dict)

    latency = round((time.perf_counter() - t0) * 1000, 2)
    log.info(f"Plano em {latency}ms — {len(steps)} steps | {tokens} tokens: {user_input[:60]}")

    return PlanResponse(
        steps=steps, input=user_input, from_cache=False,
        tokens_used=tokens, latency_ms=latency,
    )


# ── GET /status ────────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    llama_ok  = False
    memory_ok = False
    try:
        r = await state.llama_client.get("/health", timeout=2.0)
        llama_ok = r.status_code == 200
    except Exception:
        pass
    try:
        r = await state.memory_client.get("/status", timeout=2.0)
        memory_ok = r.status_code == 200
        memory_status = r.json().get("plan_cache", {}) if memory_ok else {}
    except Exception:
        memory_status = {}

    return {
        "llama_server":    LLAMA_SERVER_URL,
        "llama_healthy":   llama_ok,
        "memory_api":      MEMORY_API_URL,
        "memory_healthy":  memory_ok,
        "plan_cache":      memory_status,
        "modules":         list(AVA_MODULES.keys()),
        "cache_threshold": CACHE_HIT_THRESHOLD,
    }


# ── DELETE /cache ──────────────────────────────────────────────────────────────

@app.delete("/cache")
async def invalidate_cache():
    try:
        r = await state.memory_client.delete("/cache", timeout=10.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Falha ao invalidar cache: {e}")


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")