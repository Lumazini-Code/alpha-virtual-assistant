from __future__ import annotations

import asyncio
import logging
import shlex
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ══════════════════════════════════════════════════════════════════════════════
# Configuração
# ══════════════════════════════════════════════════════════════════════════════

# Módulos auxiliares
COT_URL       = "http://localhost:3000"
MEMORY_URL    = "http://localhost:3001"
SEARCH_URL    = "http://localhost:3002"
TTS_URL       = "http://localhost:3004"

# Módulos principais
LLM_URL       = "http://localhost:4000"

# Timeouts por executor (segundos)
EXECUTOR_TIMEOUTS: dict[str, float] = {
    "llm":        60.0,
    "memory":      5.0,
    "search":     20.0,
    "vision":     30.0,
    "tts":        15.0,
    "stt":        30.0,
    "commander":  10.0,
    "translator": 20.0,
    "calculator":  5.0,
}

# Retry por executor: número máximo de tentativas em falhas de I/O
EXECUTOR_MAX_RETRIES: dict[str, int] = {
    "llm":        2,
    "memory":     3,
    "search":     2,
    "vision":     1,
    "tts":        2,
    "stt":        1,
    "commander":  1,   # comandos OS não são idempotentes — sem retry
    "translator": 2,
    "calculator": 3,
}

RETRY_BACKOFF_S   = 0.15   # espera entre tentativas
COT_TIMEOUT_S     = 30.0
MAX_CONTEXT_CHARS = 800    # tamanho máximo do contexto injetado no LLM final

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ava.orchestrator")


# ══════════════════════════════════════════════════════════════════════════════
# Modelos de dados
# ══════════════════════════════════════════════════════════════════════════════

class ExecuteRequest(BaseModel):
    input:      str
    session_id: Optional[str] = None   # se ausente, gerado automaticamente
    voice:      str  = "M1"
    lang:       str  = "pt"
    tts:        bool = True
    use_cache:  bool = True             # passa para o CoT
    strategy:   Literal["parallel", "sequential", "fail_fast"] = "parallel"

class StepResult(BaseModel):
    step:       int
    executor:   str
    action:     str
    success:    bool
    result:     Optional[Any]  = None
    error:      Optional[str]  = None
    retries:    int            = 0
    latency_ms: float          = 0.0

class ExecuteResponse(BaseModel):
    execution_id:   str
    input:          str
    session_id:     str
    final_response: str
    steps:          list[StepResult]
    plan_from_cache: bool
    total_latency_ms: float
    errors:         list[str]


# ══════════════════════════════════════════════════════════════════════════════
# Estado global
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AppState:
    cot_client:    httpx.AsyncClient = field(default=None)
    memory_client: httpx.AsyncClient = field(default=None)
    search_client: httpx.AsyncClient = field(default=None)
    tts_client:    httpx.AsyncClient = field(default=None)
    llm_client:    httpx.AsyncClient = field(default=None)

state = AppState()


# ══════════════════════════════════════════════════════════════════════════════
# Lifespan
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Iniciando AVA Orchestrator...")

    # Verifica serviços obrigatórios
    required = {
        "CoT API":    (COT_URL,    "/status"),
        "Memory API": (MEMORY_URL, "/status"),
        "LLM API":    (LLM_URL,    "/status"),
    }
    async with httpx.AsyncClient(timeout=5.0) as probe:
        for name, (url, path) in required.items():
            try:
                r = await probe.get(f"{url}{path}")
                if r.status_code == 200:
                    log.info(f"  ✓ {name} OK em {url}")
                else:
                    log.warning(f"  ⚠ {name} respondeu {r.status_code} em {url}")
            except httpx.ConnectError:
                log.warning(f"  ✗ {name} não acessível em {url} — continuando mesmo assim")

    def _make_client(base_url: str, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url = base_url,
            timeout  = httpx.Timeout(timeout),
            limits   = httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )

    state.cot_client    = _make_client(COT_URL,    COT_TIMEOUT_S)
    state.memory_client = _make_client(MEMORY_URL, EXECUTOR_TIMEOUTS["memory"])
    state.search_client = _make_client(SEARCH_URL, EXECUTOR_TIMEOUTS["search"])
    state.tts_client    = _make_client(TTS_URL,    EXECUTOR_TIMEOUTS["tts"])
    state.llm_client    = _make_client(LLM_URL,    EXECUTOR_TIMEOUTS["llm"])

    log.info("Orchestrator pronto")
    yield

    for client in (
        state.cot_client, state.memory_client, state.search_client,
        state.tts_client, state.llm_client,
    ):
        await client.aclose()
    log.info("AVA Orchestrator encerrado")


# ══════════════════════════════════════════════════════════════════════════════
# Adaptadores de executor
# ══════════════════════════════════════════════════════════════════════════════
#
# Cada adaptador recebe:
#   action  : str — a ação descrita no plano pelo CoT
#   context : dict[str, Any] — outputs de steps anteriores {step_N: resultado}
#   req     : ExecuteRequest — dados originais do request (session_id, voice, etc.)
#
# Cada adaptador retorna Any — o resultado bruto que será injetado no contexto.
# Em caso de falha, levanta Exception com mensagem descritiva.

async def _adapt_llm(action: str, context: dict, req: ExecuteRequest) -> str:
    """
    Chama a LLM principal (porta 4000).
    Injeta resultados de steps anteriores como contexto adicional no prompt.
    """
    context_block = _format_context_for_llm(action, context)
    message = f"{action}\n\n{context_block}" if context_block else action

    r = await state.llm_client.post("/chat", json={
        "message":    message,
        "voice":      req.voice,
        "lang":       req.lang,
        "max_turns":  10,
        "tts":        False,   # TTS é gerenciado pelo orquestrador no final
        "session_id": req.session_id,
    })
    r.raise_for_status()
    data = r.json()
    # Aceita "response", "text" ou "content" como campo de resposta
    return data.get("response") or data.get("text") or data.get("content") or str(data)


async def _adapt_memory(action: str, context: dict, req: ExecuteRequest) -> list[dict]:
    """
    Detecta se é leitura ou escrita pelo verbo da action.
    Verbos de escrita: gravar, salvar, registrar, armazenar, store, save, write, record.
    Todo o resto → leitura.
    """
    write_verbs = {"gravar", "salvar", "registrar", "armazenar",
                   "store", "save", "write", "record", "memorize"}
    first_word  = action.lower().split()[0] if action.split() else ""

    if first_word in write_verbs:
        # Escrita: usa o texto da action como conteúdo a gravar
        r = await state.memory_client.post("/write", json={
            "text":       action,
            "source":     "chat",
            "confidence": 1.0,
        })
        r.raise_for_status()
        return [r.json()]
    else:
        # Leitura: injeta session_id para busca contextual
        r = await state.memory_client.post("/read", json={
            "query":      action,
            "top_k":      5,
            "min_score":  0.0,
            "session_id": req.session_id,
            "strategy":   "auto",
        })
        r.raise_for_status()
        return r.json().get("results", [])


async def _adapt_search(action: str, context: dict, req: ExecuteRequest) -> list[dict]:
    """Busca na internet via Search API (porta 3002)."""
    r = await state.search_client.post("/search", json={
        "query":       action,
        "max_results": 3,
        "use_cache":   True,
    })
    r.raise_for_status()
    return r.json().get("results", [])


async def _adapt_tts(action: str, context: dict, req: ExecuteRequest) -> str:
    """
    Converte texto para fala.
    Se a action referencia um step anterior (ex: "speak result of step 1"),
    usa o texto do contexto correspondente em vez da action literal.
    """
    text = _resolve_action_text(action, context) or action
    r = await state.tts_client.post("/speak", json={
        "text":  text,
        "voice": req.voice,
        "lang":  req.lang,
    })
    r.raise_for_status()
    return "tts_ok"


async def _adapt_stt(action: str, context: dict, req: ExecuteRequest) -> str:
    """
    STT — não há input de áudio neste path (o orquestrador recebe texto).
    Retorna a própria action como passthrough; o módulo STT real é ativado
    upstream quando o frontend captura áudio.
    """
    log.info(f"STT step ignorado em execução de texto: {action}")
    return action


async def _adapt_commander(action: str, context: dict, req: ExecuteRequest) -> str:
    """
    Executa comandos do sistema operacional.
    Resolve referências a steps anteriores antes de executar.
    Comandos são executados com shlex.split para evitar shell injection.
    """
    resolved = _resolve_action_text(action, context) or action

    # Prefixos que indicam abertura de aplicativo
    open_prefixes = ("open ", "abrir ", "launch ", "start ", "iniciar ", "executar ")
    cmd_lower = resolved.lower()

    if any(cmd_lower.startswith(p) for p in open_prefixes):
        # Extrai o nome do programa e tenta abrir
        program = resolved.split(None, 1)[1] if len(resolved.split()) > 1 else resolved
        cmd = ["xdg-open", program]
    else:
        try:
            cmd = shlex.split(resolved)
        except ValueError as e:
            raise RuntimeError(f"Comando inválido: {e}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=8.0,
        )
        output = proc.stdout.strip() or proc.stderr.strip() or "ok"
        if proc.returncode != 0:
            raise RuntimeError(f"Saiu com código {proc.returncode}: {output}")
        return output
    except subprocess.TimeoutExpired:
        raise RuntimeError("Comando excedeu timeout de 8s")
    except FileNotFoundError:
        raise RuntimeError(f"Comando não encontrado: {cmd[0]}")


async def _adapt_translator(action: str, context: dict, req: ExecuteRequest) -> str:
    """
    Tradução via LLM principal com prompt especializado.
    O CoT gera actions como "translate result of step N to English".
    """
    text = _resolve_action_text(action, context) or action
    r = await state.llm_client.post("/chat", json={
        "message":   f"Translate the following accurately. Return ONLY the translation:\n\n{text}",
        "tts":       False,
        "max_turns": 1,
    })
    r.raise_for_status()
    data = r.json()
    return data.get("response") or data.get("text") or data.get("content") or str(data)


async def _adapt_calculator(action: str, context: dict, req: ExecuteRequest) -> str:
    """
    Cálculo via eval seguro ou LLM.
    Tenta eval Python para expressões simples; fallback para LLM.
    """
    resolved = _resolve_action_text(action, context) or action

    # Tenta extrair e avaliar expressão matemática simples
    import re
    expr_match = re.search(r"[\d\s\+\-\*\/\(\)\.\^%]+", resolved)
    if expr_match:
        expr = expr_match.group().strip().replace("^", "**")
        try:
            # eval restrito: apenas builtins matemáticos
            result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
            return str(result)
        except Exception:
            pass

    # Fallback: LLM resolve o cálculo
    r = await state.llm_client.post("/chat", json={
        "message":   f"Solve this calculation and return ONLY the numeric result:\n{resolved}",
        "tts":       False,
        "max_turns": 1,
    })
    r.raise_for_status()
    data = r.json()
    return data.get("response") or data.get("text") or str(data)


async def _adapt_vision(action: str, context: dict, req: ExecuteRequest) -> str:
    """
    Visão — placeholder. O módulo de visão (porta 4002) requer imagem como input,
    que não está disponível neste path de texto. Retorna aviso estruturado.
    """
    log.warning(f"Vision step sem imagem disponível: {action}")
    return f"[vision: imagem não disponível neste contexto — ação: {action}]"


# Mapa executor → adaptador
EXECUTOR_ADAPTERS = {
    "llm":        _adapt_llm,
    "memory":     _adapt_memory,
    "search":     _adapt_search,
    "tts":        _adapt_tts,
    "stt":        _adapt_stt,
    "commander":  _adapt_commander,
    "translator": _adapt_translator,
    "calculator": _adapt_calculator,
    "vision":     _adapt_vision,
}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers de contexto
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_action_text(action: str, context: dict[str, Any]) -> str:
    """
    Substitui referências "result of step N" pelo resultado real do contexto.
    Ex: "summarize result of step 1 and result of step 2"
        → "summarize <texto do step 1> and <texto do step 2>"
    """
    import re
    def replacer(m: re.Match) -> str:
        n   = int(m.group(1))
        key = f"step_{n}"
        val = context.get(key)
        if val is None:
            return m.group(0)
        return _result_to_text(val)

    return re.sub(r"result of step (\d+)", replacer, action, flags=re.IGNORECASE)


def _result_to_text(result: Any) -> str:
    """Converte qualquer resultado de executor para string injetável no prompt."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result[:MAX_CONTEXT_CHARS]
    if isinstance(result, list):
        # list[dict] — formata como bullet list
        parts = []
        for item in result[:5]:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("result") or str(item)
                source = item.get("source") or item.get("url") or ""
                parts.append(f"- {text[:300]}" + (f" ({source})" if source else ""))
            else:
                parts.append(f"- {str(item)[:300]}")
        return "\n".join(parts)[:MAX_CONTEXT_CHARS]
    if isinstance(result, dict):
        text = result.get("text") or result.get("content") or result.get("response") or str(result)
        return str(text)[:MAX_CONTEXT_CHARS]
    return str(result)[:MAX_CONTEXT_CHARS]


def _format_context_for_llm(action: str, context: dict[str, Any]) -> str:
    """
    Monta bloco de contexto para o LLM a partir dos outputs de steps anteriores.
    Só inclui steps que sejam referenciados na action ou todos se não houver referência.
    """
    import re
    referenced = set(int(m) for m in re.findall(r"result of step (\d+)", action, re.IGNORECASE))
    keys_to_include = (
        {f"step_{n}" for n in referenced} & set(context.keys())
        if referenced else set(context.keys())
    )
    if not keys_to_include:
        return ""
    parts = []
    for key in sorted(keys_to_include):
        n    = key.split("_")[1]
        text = _result_to_text(context[key])
        if text:
            parts.append(f"[Step {n} result]\n{text}")
    return "\n\n".join(parts)[:MAX_CONTEXT_CHARS]


# ══════════════════════════════════════════════════════════════════════════════
# Executor com retry
# ══════════════════════════════════════════════════════════════════════════════

async def _run_step_with_retry(
    step_num:  int,
    action:    str,
    executor:  str,
    context:   dict[str, Any],
    req:       ExecuteRequest,
) -> StepResult:
    """
    Executa um único step com retry automático em falhas de I/O.
    Falhas de lógica (ValueError, etc.) não fazem retry.
    """
    adapter     = EXECUTOR_ADAPTERS.get(executor)
    max_retries = EXECUTOR_MAX_RETRIES.get(executor, 1)

    if adapter is None:
        return StepResult(
            step=step_num, executor=executor, action=action,
            success=False, error=f"Executor desconhecido: {executor}",
        )

    t0      = time.perf_counter()
    retries = 0
    last_error: str = ""

    # Resolve referências na action antes de passar ao adaptador
    resolved_action = _resolve_action_text(action, context)

    while retries <= max_retries:
        try:
            result = await asyncio.wait_for(
                adapter(resolved_action, context, req),
                timeout=EXECUTOR_TIMEOUTS.get(executor, 30.0),
            )
            latency = round((time.perf_counter() - t0) * 1000, 2)
            log.info(
                f"  Step {step_num} [{executor}] OK em {latency}ms"
                + (f" (retry {retries})" if retries else "")
            )
            return StepResult(
                step=step_num, executor=executor, action=action,
                success=True, result=result, retries=retries, latency_ms=latency,
            )

        except (httpx.ConnectError, httpx.TimeoutException, asyncio.TimeoutError) as e:
            # Falhas de I/O — fazer retry
            last_error = f"{type(e).__name__}: {e}"
            retries   += 1
            if retries <= max_retries:
                log.warning(f"  Step {step_num} [{executor}] falhou (tentativa {retries}/{max_retries}): {last_error}")
                await asyncio.sleep(RETRY_BACKOFF_S * retries)
            continue

        except Exception as e:
            # Falhas de lógica — não fazer retry
            last_error = f"{type(e).__name__}: {e}"
            break

    latency = round((time.perf_counter() - t0) * 1000, 2)
    log.error(f"  Step {step_num} [{executor}] FALHOU após {retries} tentativas: {last_error}")
    return StepResult(
        step=step_num, executor=executor, action=action,
        success=False, error=last_error, retries=retries, latency_ms=latency,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DAG executor
# ══════════════════════════════════════════════════════════════════════════════

async def _execute_plan(
    steps:    list[dict],
    req:      ExecuteRequest,
    strategy: str,
) -> tuple[list[StepResult], dict[str, Any]]:
    """
    Executa o plano respeitando dependências declaradas em depends_on.

    Estratégias:
      parallel   — steps sem dependências pendentes rodam em asyncio.gather().
      sequential — mesmo com depends_on=null, executa um por vez (debug/teste).
      fail_fast  — primeiro step que falha interrompe toda a execução.

    Retorna (lista de StepResult, dict de contexto {step_N: resultado}).
    """
    # Indexa steps por número
    step_map: dict[int, dict] = {s["step"]: s for s in steps}
    results:  dict[int, StepResult] = {}
    context:  dict[str, Any]        = {}
    pending:  set[int]              = set(step_map.keys())

    while pending:
        # Identifica steps prontos para executar (todas as dependências satisfeitas)
        ready: list[int] = []
        for sn in sorted(pending):
            s    = step_map[sn]
            deps = s.get("depends_on") or []

            # Verifica se alguma dependência falhou (bloqueia este step)
            failed_deps = [d for d in deps if d in results and not results[d].success]
            if failed_deps:
                dep_errors = [results[d].error for d in failed_deps]
                results[sn] = StepResult(
                    step=sn, executor=s["executor"], action=s["action"],
                    success=False,
                    error=f"Bloqueado: dependências {failed_deps} falharam — {dep_errors}",
                )
                pending.discard(sn)
                continue

            # Verifica se todas as dependências já terminaram com sucesso
            if all(d in results and results[d].success for d in deps):
                ready.append(sn)

        if not ready:
            # Nenhum step pronto e ainda há pendentes → deadlock no DAG
            # (não deveria acontecer com planos válidos do CoT)
            for sn in list(pending):
                s = step_map[sn]
                results[sn] = StepResult(
                    step=sn, executor=s["executor"], action=s["action"],
                    success=False, error="Deadlock no DAG — dependências nunca satisfeitas",
                )
            break

        if strategy == "fail_fast":
            # Verifica se algum step anterior falhou
            if any(not r.success for r in results.values()):
                for sn in list(pending):
                    s = step_map[sn]
                    results[sn] = StepResult(
                        step=sn, executor=s["executor"], action=s["action"],
                        success=False, error="Abortado (fail_fast)",
                    )
                pending.clear()
                break

        if strategy == "sequential":
            # Executa um step por vez mesmo que haja múltiplos prontos
            ready = [ready[0]]

        # Executa steps prontos em paralelo
        log.info(f"Executando steps {ready} em paralelo...")
        tasks = [
            _run_step_with_retry(
                step_num = sn,
                action   = step_map[sn]["action"],
                executor = step_map[sn]["executor"],
                context  = context,
                req      = req,
            )
            for sn in ready
        ]
        batch_results: list[StepResult] = await asyncio.gather(*tasks)

        for step_result in batch_results:
            sn = step_result.step
            results[sn] = step_result
            pending.discard(sn)
            if step_result.success:
                context[f"step_{sn}"] = step_result.result

    return [results[sn] for sn in sorted(results.keys())], context


# ══════════════════════════════════════════════════════════════════════════════
# Agregador de resposta final
# ══════════════════════════════════════════════════════════════════════════════

async def _build_final_response(
    original_input: str,
    steps:          list[StepResult],
    context:        dict[str, Any],
    req:            ExecuteRequest,
) -> str:
    """
    Agrega todos os resultados numa resposta coerente.

    Lógica:
    - Se o último step com sucesso for do executor "llm", usa o resultado diretamente.
    - Caso contrário, manda tudo para o LLM sintetizar uma resposta final.
    - Se todos os steps falharam, retorna mensagem de erro estruturada.
    """
    successful = [s for s in steps if s.success]

    if not successful:
        errors = [s.error for s in steps if s.error]
        return f"Não foi possível completar a tarefa. Erros: {'; '.join(errors)}"

    # Se o último step bem-sucedido já é uma resposta do LLM, usa direto
    last = successful[-1]
    if last.executor == "llm" and isinstance(last.result, str) and len(last.result) > 20:
        return last.result

    # Há resultados de múltiplos steps para consolidar
    context_parts = []
    for s in successful:
        text = _result_to_text(s.result)
        if text:
            context_parts.append(f"[{s.executor.upper()} — step {s.step}]\n{text}")

    if not context_parts:
        return "Tarefa concluída sem resultado textual."

    # Chama o LLM para sintetizar
    combined_context = "\n\n".join(context_parts)[:MAX_CONTEXT_CHARS]
    synthesis_prompt = (
        f"Original request: {original_input}\n\n"
        f"Gathered information:\n{combined_context}\n\n"
        f"Based on the gathered information, provide a clear and direct response "
        f"to the original request. Respond in the same language as the request."
    )

    try:
        r = await state.llm_client.post("/chat", json={
            "message":    synthesis_prompt,
            "voice":      req.voice,
            "lang":       req.lang,
            "max_turns":  1,
            "tts":        False,
            "session_id": req.session_id,
        })
        r.raise_for_status()
        data = r.json()
        return data.get("response") or data.get("text") or data.get("content") or combined_context
    except Exception as e:
        log.warning(f"LLM final falhou, retornando contexto bruto: {e}")
        return combined_context


# ══════════════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="AVA Orchestrator", lifespan=lifespan)


# ── POST /execute ──────────────────────────────────────────────────────────────

@app.post("/execute", response_model=ExecuteResponse)
async def execute(req: ExecuteRequest):
    """
    Ponto de entrada principal do AVA.

    Fluxo completo:
      1. Gera um session_id se não fornecido.
      2. Chama o CoT API para obter o plano de execução.
      3. Constrói o DAG a partir dos campos depends_on.
      4. Executa o plano: steps independentes em paralelo, respeitando dependências.
         - Retry automático em falhas de I/O (configurable por executor).
         - Resultados de steps anteriores injetados nos subsequentes via context.
      5. Agrega a resposta final:
         - Se o último step for LLM, usa direto.
         - Caso contrário, sintetiza com o LLM.
      6. Envia para TTS se req.tts=True.
      7. Grava o turno na memória de curto prazo.
    """
    t0           = time.perf_counter()
    execution_id = str(uuid.uuid4())
    session_id   = req.session_id or str(uuid.uuid4())

    log.info(f"[{execution_id[:8]}] Executando: '{req.input[:80]}'")

    # ── 1. Obtém plano do CoT ──────────────────────────────────────────────────
    try:
        cot_r = await state.cot_client.post("/plan", json={
            "input":     req.input,
            "use_cache": req.use_cache,
        })
        cot_r.raise_for_status()
        plan_data      = cot_r.json()
        raw_steps      = plan_data.get("steps", [])
        plan_from_cache = plan_data.get("from_cache", False)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"CoT API falhou: {e}")

    if not raw_steps:
        raise HTTPException(status_code=500, detail="CoT retornou plano vazio")

    log.info(
        f"[{execution_id[:8]}] Plano: {len(raw_steps)} steps "
        f"{'(cache)' if plan_from_cache else '(novo)'}"
    )

    # ── 2. Executa o DAG ───────────────────────────────────────────────────────
    step_results, context = await _execute_plan(raw_steps, req, req.strategy)

    errors = [
        f"Step {s.step} [{s.executor}]: {s.error}"
        for s in step_results if not s.success
    ]

    # ── 3. Agrega resposta final ───────────────────────────────────────────────
    final_response = await _build_final_response(req.input, step_results, context, req)

    # ── 4. TTS (fire-and-forget, não bloqueia a resposta) ─────────────────────
    if req.tts and final_response:
        asyncio.create_task(_fire_tts(final_response, req))

    # ── 5. Grava turno na memória de curto prazo (fire-and-forget) ────────────
    asyncio.create_task(_save_turn_to_memory(req.input, final_response, session_id))

    total_latency = round((time.perf_counter() - t0) * 1000, 2)
    log.info(
        f"[{execution_id[:8]}] Concluído em {total_latency}ms — "
        f"{sum(1 for s in step_results if s.success)}/{len(step_results)} steps OK"
        + (f" | {len(errors)} erros" if errors else "")
    )

    return ExecuteResponse(
        execution_id     = execution_id,
        input            = req.input,
        session_id       = session_id,
        final_response   = final_response,
        steps            = step_results,
        plan_from_cache  = plan_from_cache,
        total_latency_ms = total_latency,
        errors           = errors,
    )


# ── Fire-and-forget helpers ────────────────────────────────────────────────────

async def _fire_tts(text: str, req: ExecuteRequest):
    """Envia texto para TTS sem bloquear a resposta HTTP."""
    try:
        await state.tts_client.post("/speak", json={
            "text":  text[:2000],
            "voice": req.voice,
            "lang":  req.lang,
        })
    except Exception as e:
        log.warning(f"TTS fire-and-forget falhou: {e}")


async def _save_turn_to_memory(user_input: str, assistant_response: str, session_id: str):
    """Salva o par (user, assistant) na memória de curto prazo."""
    try:
        await state.memory_client.post("/write_st", json={
            "session_id": session_id,
            "turns": [
                {"role": "user",      "content": user_input},
                {"role": "assistant", "content": assistant_response},
            ],
        })
    except Exception as e:
        log.warning(f"Gravação de turno na memória falhou: {e}")


# ── GET /status ────────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    checks: dict[str, bool] = {}
    probes = {
        "cot":    (state.cot_client,    "/status"),
        "memory": (state.memory_client, "/status"),
        "search": (state.search_client, "/status"),
        "tts":    (state.tts_client,    "/status"),
        "llm":    (state.llm_client,    "/status"),
    }
    for name, (client, path) in probes.items():
        try:
            r = await client.get(path, timeout=2.0)
            checks[name] = r.status_code == 200
        except Exception:
            checks[name] = False

    return {
        "services":  checks,
        "executors": list(EXECUTOR_ADAPTERS.keys()),
        "urls": {
            "cot":    COT_URL,
            "memory": MEMORY_URL,
            "search": SEARCH_URL,
            "tts":    TTS_URL,
            "llm":    LLM_URL,
        },
    }


# ── DELETE /cache — proxy para invalidar cache do CoT ─────────────────────────

@app.delete("/cache")
async def invalidate_cot_cache():
    """
    Invalida o cache de planos do CoT.
    Chamar após adicionar/remover módulos ou atualizar o system prompt.
    """
    try:
        r = await state.cot_client.delete("/cache", timeout=10.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Falha ao invalidar cache: {e}")


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("orchestrator:app", host="0.0.0.0", port=9000, log_level="info")
