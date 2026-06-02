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

MAX_TOKENS     = 160   # margem para 7 steps verbosos sem whitespace (~100-140 tokens reais)
TEMPERATURE    = 0.1
TOP_P          = 0.9
REPEAT_PENALTY = 1.1

# Threshold de similaridade para aceitar um plano do cache semântico.
# Valor mais alto (ex: 0.95) = só aceita hits muito próximos = menos falsos positivos.
CACHE_HIT_THRESHOLD = 0.92

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ava.cot")


# ── Módulos disponíveis no AVA ─────────────────────────────────────────────────

AVA_MODULES: dict[str, str] = {
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
#
# Separado em bloco estático puro — nunca concatenado com dados dinâmicos —
# para maximizar o reuso do KV cache do llama-server (cache_prompt: true).
# O _build_prompt injeta os dados variáveis apenas no turno <|user|>.

SYSTEM_PROMPT = f"""You are a planning module..
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

Output format — output ONLY the inner content, starting from the first step object:
{{"step":1,"action":"...","executor":"llm","depends_on":null}},{{"step":2,...}}]}}

Examples:
Input: "what GPU should I buy for gaming under R$2000"
Output: {{"step":1,"action":"search RTX 4060 RX 7600 benchmark price Brazil 2024","executor":"search","depends_on":null}},{{"step":2,"action":"retrieve user GPU preferences from memory","executor":"memory","depends_on":null}},{{"step":3,"action":"recommend GPU based on result of step 1 and result of step 2","executor":"llm","depends_on":[1,2]}}]}}

Input: "play some jazz music"
Output: {{"step":1,"action":"search jazz playlist Spotify","executor":"search","depends_on":null}},{{"step":2,"action":"open music player with result of step 1","executor":"commander","depends_on":[1]}}]}}
"""

# ── Grammar GBNF otimizada ─────────────────────────────────────────────────────
#
# Melhorias em relação à versão anterior:
#
# 1. PREFIXO INJETADO NO PROMPT: o modelo começa a gerar a partir do primeiro
#    step object — pula os ~12 tokens fixos do envelope {"steps":[ que foram
#    pré-injetados no prompt. O grammar começa correspondentemente no meio do array.
#
# 2. SEM WHITESPACE ENTRE LITERAIS: ws removido entre campos fixos. Literais
#    contíguos são processados como uma única restrição no FSM do llama.cpp.
#
# 3. number RESTRITO A [1-7]: elimina o estado de "pode continuar com dígito?"
#    após o primeiro caractere. Um único token resolve o campo step.
#
# 4. depends_on TIPADO: null | array de dígitos. O orquestrador usa isso para
#    paralelizar steps independentes sem heurísticas extras.

GRAMMAR_GBNF = r"""root        ::= steps-cont "]}"
steps-cont  ::= step ("," step)*
step        ::= "{\"step\":" step-num ",\"action\":" string ",\"executor\":" executor ",\"depends_on\":" depends "}"
step-num    ::= [1-7]
executor    ::= "\"llm\"" | "\"memory\"" | "\"search\"" | "\"vision\"" | "\"tts\"" | "\"stt\"" | "\"commander\"" | "\"translator\"" | "\"calculator\""
depends     ::= "null" | "[" step-num ("," step-num)* "]"
string      ::= "\"" ([^"\\] | "\\" .)* "\""
"""

# Prefixo que é injetado no prompt — o modelo continua daqui.
# Recomposto junto com o output do modelo para formar o JSON final.
JSON_PREFIX = '{"steps":['


# ── Modelos de request/response ────────────────────────────────────────────────

class PlanRequest(BaseModel):
    input:      str
    context:    Optional[str] = None
    max_steps:  Optional[int] = None
    use_cache:  bool = True
    # threshold sobrescrevível por request (ex: domínios críticos usam 0.95)
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
    cache_score:   Optional[float] = None   # score de similaridade se from_cache=True
    tokens_used:   Optional[int]   = None   # tokens_predicted do llama-server
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
    # 1. Aguarda llama-server
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

    # 2. Inicializa clientes
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

    yield  # ← único yield

    # 3. Shutdown
    await state.llama_client.aclose()
    await state.memory_client.aclose()
    log.info("AVA CoT API encerrada")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="AVA CoT API", lifespan=lifespan)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_prompt(user_input: str, context: Optional[str], max_steps: Optional[int]) -> str:
    """
    Monta o prompt com o system prompt completamente estático no bloco <|system|>
    e os dados dinâmicos isolados no bloco <|user|>.

    Isso garante que o llama-server possa reutilizar o KV cache do system prompt
    entre todas as chamadas, independente do user input.

    O prompt termina com o JSON_PREFIX pré-injetado no bloco <|assistant|> —
    o modelo começa a gerar a partir do primeiro step object.
    """
    steps_hint   = f" Use no máximo {max_steps} passos." if max_steps else ""
    user_content = f"Plan the following request:{steps_hint}\n\n{user_input}"
    if context:
        user_content += f"\n\nRelevant context:\n{context}"

    return (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{user_content}<|end|>\n"
        f"<|assistant|>\n{JSON_PREFIX}"   # modelo continua daqui
    )


async def _call_llama(prompt: str) -> tuple[str, int]:
    """
    Chama o llama-server e retorna (raw_content, tokens_predicted).
    raw_content é o que o modelo gerou APÓS o JSON_PREFIX injetado.
    """
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

    data            = response.json()
    raw             = data.get("content", "").strip()
    tokens_used     = data.get("tokens_predicted", 0)
    return raw, tokens_used


def _parse_steps(raw: str) -> list[PlanStep]:
    """
    Reconstrói o JSON completo prefixando JSON_PREFIX ao output do modelo,
    parseia e valida os steps.

    Estratégia de fallback: se o JSON completo falhar (quantizações INT4 podem
    cortar o output), tenta parsear steps individuais que estejam completos.
    """
    valid_executors = set(AVA_MODULES.keys())
    full_json       = JSON_PREFIX + raw

    # Tentativa 1: parse normal do JSON completo
    try:
        result    = json.loads(full_json)
        raw_steps = result.get("steps", [])
    except json.JSONDecodeError:
        # Tentativa 2: o modelo cortou antes do fechamento — tenta recuperar
        # steps que já estão completos no output truncado
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
            # Filtra referências para steps que ainda não existem
            depends_on = [d for d in depends_on if isinstance(d, int) and d < i]
            depends_on = depends_on or None
        elif depends_on is not None:
            depends_on = None
        steps.append(PlanStep(step=i, action=action, executor=executor, depends_on=depends_on))

    if not steps:
        raise HTTPException(status_code=500, detail="Nenhum step válido após validação")

    return steps


def _recover_partial_steps(raw: str) -> list[dict]:
    """
    Tenta extrair steps completos de um JSON truncado.
    Procura por objetos {...} fechados no output.
    """
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
    """
    Busca no cache semântico da Memory API.
    Retorna (plan_dict, score, cache_id) se hit, None se miss.
    Falhas de rede retornam None silenciosamente — o CoT continua sem cache.
    """
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
    """
    Grava um plano no cache semântico da Memory API.
    Falhas de rede são logadas mas não propagadas.
    """
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
    """
    Gera um plano de execução para o input do usuário.

    Fluxo:
      1. Consulta o cache semântico na Memory API (se use_cache=True).
         Hit  → retorna imediatamente (~5ms, sem inferência).
         Miss → continua para o passo 2.

      2. Infere com o Phi-3.5-mini via llama-server.
         O prompt usa prefixo JSON pré-injetado para reduzir tokens gerados.
         Grammar GBNF garante output estruturado com depends_on para paralelismo.

      3. Cacheia o plano gerado na Memory API para requests futuros similares.

    O campo depends_on em cada step indica quais steps anteriores precisam
    terminar antes deste começar. Steps com depends_on=null podem ser
    executados em paralelo pelo orquestrador do AVA.
    """
    user_input = req.input.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="input vazio")

    t0 = time.perf_counter()

    # ── 1. Consulta cache semântico ────────────────────────────────────────────
    if req.use_cache:
        cache_result = await _cache_get(user_input, req.cache_threshold)
        if cache_result:
            cached_plan, score, _ = cache_result
            steps = [PlanStep(**s) for s in cached_plan["steps"]]
            latency = round((time.perf_counter() - t0) * 1000, 2)
            log.info(f"Cache HIT em {latency}ms — score={score:.3f}: {user_input[:60]}")
            return PlanResponse(
                steps       = steps,
                input       = user_input,
                from_cache  = True,
                cache_score = score,
                latency_ms  = latency,
            )

    # ── 2. Inferência ──────────────────────────────────────────────────────────
    prompt      = _build_prompt(user_input, req.context, req.max_steps)
    raw, tokens = await _call_llama(prompt)
    steps       = _parse_steps(raw)

    # ── 3. Cacheia o plano gerado ──────────────────────────────────────────────
    if req.use_cache:
        plan_dict = {"steps": [s.model_dump() for s in steps]}
        await _cache_put(user_input, plan_dict)

    latency = round((time.perf_counter() - t0) * 1000, 2)
    log.info(
        f"Plano em {latency}ms — {len(steps)} steps | "
        f"{tokens} tokens: {user_input[:60]}"
    )

    return PlanResponse(
        steps       = steps,
        input       = user_input,
        from_cache  = False,
        tokens_used = tokens,
        latency_ms  = latency,
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


# ── DELETE /cache — invalida todo o cache de planos ───────────────────────────

@app.delete("/cache")
async def invalidate_cache():
    """
    Proxy para DELETE /cache da Memory API.
    Chamar quando módulos do AVA mudarem ou o system prompt for atualizado.
    """
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
