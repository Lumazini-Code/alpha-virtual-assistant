"""
Alpha-code — Agent (ReAct Loop)
================================
State machine que executa tarefas de código via Groq tool use nativo.

Fluxo:
  1. system prompt com instruções + tools disponíveis
  2. Loop:
     a. Decide step kind (planning/editing/debugging/explaining/final_review)
     b. Model router escolhe modelo + reasoning_effort
     c. Chama LLM:4003 /chat/tools com messages + tools
     d. Se resposta tem tool_calls → executa cada via registry.dispatch
     e. Anexa tool results como role=tool messages
     f. Se resposta tem content sem tool_calls → finaliza
  3. Streaming: yields Event objects para SSE
  4. Criterios de parada:
     - LLM respondeu sem tool_calls (considerou pronto)
     - max_steps atingido
     - Erro fatal (LLM indisponível)
     - Token budget estourado após summarization
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import AsyncGenerator, Optional

import httpx

from alpha_code.context import ContextManager, SessionStore
from alpha_code.model_router import (
    MODEL_EDITOR,
    MODEL_TPM_LIMITS,
    ModelChoice,
    classify_step,
    max_tokens_for_kind,
    pick_chain,
    reasoning_effort_for,
)
from alpha_code.schemas import (
    Event,
    EventType,
    SessionState,
    StepContext,
    TaskRequest,
    ToolCall,
    ToolResult,
)
from alpha_code.tools.registry import ToolRegistry

log = logging.getLogger("ava.alpha_code.agent")


LLM_URL = os.environ.get("ALPHA_LLM_URL", "http://localhost:4003")


# ════════════════════════════════════════════════════════════════════════════
# Redução de tokens — categorias de tools por tipo de step
#
# Nomes 100% confirmados (contra file_tools.py, patch_tools.py,
# symbol_lookup.py, test_tools.py, e o próprio log de execução, que mostra
# "tool=search_code" e "tool=semantic_search" rodando de verdade):
#   file:   list_files, read_file, write_file, run_command, stat
#   patch:  apply_patch                          (patch_tools.py)
#   search: search_code, semantic_search          (search_tools.py, via log)
#   test:   run_tests, run_linter                 (test_tools.py)
#   symbol: symbol_lookup                         (symbol_lookup.py)
#
# "git" foi removida: alpha_code.py atual não importa nem chama
# register_git_tools — essa categoria de tool não existe mais no sistema.
# ════════════════════════════════════════════════════════════════════════════

TOOL_CATEGORIES: dict[str, tuple[str, ...]] = {
    # str_replace removida: não é mais registrada em file_tools.py
    # (apply_patch, na categoria "patch" abaixo, é o superconjunto usado
    # para edição — ver comentário em file_tools.py::register_all).
    "file": ("list_files", "read_file", "write_file", "run_command", "stat"),
    "search": ("search_code", "semantic_search"),
    "patch": ("apply_patch",),
    "test": ("run_tests", "run_linter"),
    "symbol": ("symbol_lookup",),
}

KIND_CATEGORIES: dict[str, tuple[str, ...]] = {
    "planning": ("file", "search", "symbol"),
    "editing": ("file", "patch", "search", "symbol"),
    "debugging": ("file", "search", "symbol", "test", "patch"),
    "explaining": ("file", "search", "symbol"),
    "final_review": ("test", "file"),
}

# Limites de tamanho de output de tools antes de guardar no histórico
# (item "paginação"/truncamento — sem acesso a tools/file_tools.py para
# implementar paginação real no read_file, isso trunca no lado do agente
# mantendo início+fim do conteúdo, o que já evita reenviar arquivos
# inteiros de novo a cada step subsequente).
TOOL_OUTPUT_MAX_CHARS = int(os.environ.get("ALPHA_TOOL_OUTPUT_MAX_CHARS", "4000"))
READ_FILE_MAX_CHARS = int(os.environ.get("ALPHA_READ_FILE_MAX_CHARS", "6000"))

# ════════════════════════════════════════════════════════════════════════════
# Timeout por tool call. ANTES: self.registry.dispatch(...) era chamado sem
# NENHUM timeout — se qualquer tool travasse (ex: run_command executando um
# comando que nunca retorna: dev server, watch mode, comando interativo
# esperando stdin), o await ficava parado pra sempre, sem log nenhum e sem
# possibilidade de recuperação (incidente 2026-07-22: sessão "travou" ~1min
# depois do rate limit já ter liberado — não era rate limit, era isso).
# Agora todo dispatch tem um teto; ao estourar, vira um ToolResult de erro
# normal e o loop ReAct continua (o modelo vê o erro e decide o que fazer).
# ════════════════════════════════════════════════════════════════════════════
TOOL_TIMEOUT_DEFAULT_SECONDS = int(os.environ.get("ALPHA_TOOL_TIMEOUT", "120"))
TOOL_TIMEOUT_OVERRIDES: dict[str, int] = {
    "run_command": int(os.environ.get("ALPHA_TOOL_TIMEOUT_RUN_COMMAND", "180")),
    "run_tests": int(os.environ.get("ALPHA_TOOL_TIMEOUT_RUN_TESTS", "300")),
    "run_linter": int(os.environ.get("ALPHA_TOOL_TIMEOUT_RUN_LINTER", "180")),
    "semantic_search": int(os.environ.get("ALPHA_TOOL_TIMEOUT_SEMANTIC_SEARCH", "60")),
}


def _tool_timeout_seconds(tool_name: str) -> int:
    return TOOL_TIMEOUT_OVERRIDES.get(tool_name, TOOL_TIMEOUT_DEFAULT_SECONDS)


def _schema_name(schema: dict) -> str:
    fn = schema.get("function", schema)
    return fn.get("name", "")


def _compress_schema(schema: dict, max_desc_chars: int = 160, max_param_desc_chars: int = 80) -> dict:
    """Encolhe descriptions verbosas de tool schemas antes de enviar ao Groq."""
    schema = json.loads(json.dumps(schema))  # cópia rasa segura
    fn = schema.get("function", schema)

    desc = fn.get("description")
    if isinstance(desc, str) and len(desc) > max_desc_chars:
        # ATENÇÃO: já vimos isso quebrar 2 tools (apply_patch e search_code)
        # cortando a description bem no meio do exemplo que mostrava o
        # parâmetro obrigatório sendo usado — o modelo aprendia um exemplo
        # incompleto e passava a esquecer esse parâmetro. Loga aqui pra
        # pegar isso na hora, em vez de só descobrir via "Groq rejeitou
        # tool call por schema mismatch" alguns dias depois.
        log.warning(
            f"_compress_schema: description de '{fn.get('name', '?')}' tem "
            f"{len(desc)} chars (limite {max_desc_chars}) e será truncada — "
            f"considere encurtar a description da tool no código-fonte."
        )
        fn["description"] = desc[:max_desc_chars].rsplit(" ", 1)[0] + "…"

    params = fn.get("parameters", {})
    props = params.get("properties", {}) if isinstance(params, dict) else {}
    for p in props.values():
        if not isinstance(p, dict):
            continue
        pd = p.get("description")
        if isinstance(pd, str) and len(pd) > max_param_desc_chars:
            p["description"] = pd[:max_param_desc_chars].rsplit(" ", 1)[0] + "…"
        p.pop("examples", None)

    return schema


def _truncate_output(text: str, max_chars: int) -> str:
    """Trunca output de tool mantendo início e fim (mais informativo que só cortar o fim)."""
    if not text or len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - max_chars
    return (
        f"{text[:half]}\n\n"
        f"[...{omitted} caracteres omitidos para economizar tokens...]\n\n"
        f"{text[-half:]}"
    )


# ════════════════════════════════════════════════════════════════════════════
# Preload de arquivos relevantes — roda 1x no início de sessões novas, ANTES
# do loop ReAct, pra evitar gastar os primeiros steps só em list_files/
# read_file às cegas (foi isso que consumiu os steps 0-2 no incidente
# 2026-07-21: list_files → read_file → só aí começou a editar).
#
# Estratégia (best-effort, NUNCA bloqueia o fluxo normal se falhar em
# qualquer etapa — pré-carregar é um adiantamento, não uma dependência;
# list_files/read_file continuam disponíveis no loop ReAct normalmente):
#   1. semantic_search (se registrada) — barato, sem custo de LLM.
#   2. Se não existir/não retornar nada, pergunta pro modelo mais barato
#      (MODEL_EDITOR, reasoning baixo) dado só a árvore de arquivos.
#   3. Lê os candidatos UM POR VEZ, truncando cada um e parando cedo se o
#      orçamento de tokens reservado pro preload estourar.
# ════════════════════════════════════════════════════════════════════════════

PRELOAD_ENABLED = os.environ.get("ALPHA_PRELOAD_ENABLED", "true").lower() not in ("0", "false", "no")
PRELOAD_MAX_FILES = int(os.environ.get("ALPHA_PRELOAD_MAX_FILES", "5"))
PRELOAD_TOKEN_BUDGET = int(os.environ.get("ALPHA_PRELOAD_TOKEN_BUDGET", "3000"))
_PRELOAD_CHARS_PER_TOKEN = 3.3


def _extract_paths_from_semantic_result(result: ToolResult) -> list[str]:
    """
    Extrai paths de um ToolResult de semantic_search sem assumir o schema
    exato — tools/search_tools.py não foi enviado para conferência (ao
    contrário de file_tools.py, que já validamos). Tenta primeiro chaves
    comuns em result.data; se não achar nada estruturado, tenta um parser
    permissivo do texto em result.output (linhas no estilo "[score] path"
    ou "[F] path (NB)", como o list_files já usa).

    Retorna [] se nada reconhecível — quem chama trata isso como "sem
    sugestão útil", não como erro (cai pro fallback de LLM).
    """
    data = result.data or {}
    for key in ("results", "matches", "files", "paths", "hits"):
        items = data.get(key)
        if isinstance(items, list) and items:
            paths = []
            for it in items:
                if isinstance(it, str):
                    paths.append(it)
                elif isinstance(it, dict):
                    p = it.get("path") or it.get("file_path") or it.get("file")
                    if p:
                        paths.append(p)
            if paths:
                return paths

    paths = []
    for line in (result.output or "").splitlines():
        line = line.strip()
        if not line:
            continue
        # Formato atual do semantic_search: "[score=..] arquivo=<path> (trecho N)".
        # Checa isso PRIMEIRO — o fallback abaixo (pegar o último token da
        # linha) quebrava aqui: o último token virava "N)" (ou, se ranked
        # vier vazio, a própria frase "Nenhum resultado relevante
        # encontrado." virava um "path" fake terminando em "encontrado.").
        m = re.search(r"arquivo=(\S+)", line)
        if m:
            paths.append(m.group(1))
            continue
        token = line.split()[-1] if " " in line else line
        token = token.strip("[]()")
        if "/" in token or ("." in token and " " not in token):
            paths.append(token)
    return paths


# ── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Você é o Alpha-code, um agente de software que edita e gera código em um projeto real.

PRINCÍPIOS:
1. PLANEJE antes de agir. Use search_code/symbol_lookup/list_files para entender o projeto antes de editar.
2. FAÇA MUDANÇAS MÍNIMAS. Prefira apply_patch a write_file.
3. SEMPRE valide: após editar, execute run_tests para verificar.
4. Se um teste falhar, leia o erro, corrija, reteste. Não diga "pronto" sem validar.
5. Use git_status para ver o que mudou. Faça git_commit quando uma tarefa for concluída.
6. Comente O QUE o código faz, não COMO (o código já mostra o como).
7. Se a tarefa for ambígua, faça suposições razoáveis e diga o que assumiu.
8. Não repita a mesma tool call com os mesmos args. Se falhou, mude a abordagem.
9. PATHS: use apenas paths que apareceram literalmente no output de list_files/search_code/semantic_search NESTA sessão. Nunca reutilize um path só porque ele aparece em algum trecho de código colado na tarefa do usuário (ex: comentários, logs, prints de árvore de arquivos) — esse texto pode estar desatualizado ou nem refletir a estrutura real do projeto. Se um read_file/list_files der 404, não tente variações de capitalização ou de prefixo por conta própria: rode list_files na pasta pai para confirmar o nome exato antes de tentar de novo.

PADRÃO DE TRABALHO:
- Para "corrija bug X": leia código → localize → entenda → corrija → teste → commit
- Para "adicione feature Y": planeje arquitetura → liste arquivos afetados → edite → teste → commit
- Para "refatore Z": mapeie usos → planeje → faça em passos pequenos → teste após cada passo
- Para "explique/documente": leia código → escreva explicação concisa → não precisa commitar

Responda SEMPRE em português (pt-BR), exceto código e identificadores.

Quando terminar a tarefa, responda SEM tool_calls — só com texto final resumindo o que fez.
"""


class Agent:
    """ReAct agent que executa uma tarefa de código."""

    def __init__(
        self,
        registry: ToolRegistry,
        session: SessionStore,
        context: ContextManager,
        max_steps: int = 25,
        model_override: Optional[str] = None,
        temperature: float = 0.3,
    ):
        self.registry = registry
        self.session = session
        self.context = context
        self.max_steps = max_steps
        self.model_override = model_override
        self.temperature = temperature
        self._client = httpx.AsyncClient(
            base_url=LLM_URL,
            timeout=httpx.Timeout(300.0, connect=10.0),
        )
        # Cache local dos schemas já comprimidos — a API do Groq é stateless
        # (não existe cache de tools entre requests HTTP no lado do
        # provider), então o ganho aqui é evitar recomprimir/reserializar os
        # schemas a cada step; a economia de tokens de fato vem de combinar
        # isso com o filtro por categoria em `_tools_for_kind`.
        self._all_schemas_cache: Optional[list[dict]] = None

    async def close(self) -> None:
        if not self._client.is_closed:
            await self._client.aclose()

    def _get_all_schemas(self) -> list[dict]:
        if self._all_schemas_cache is None:
            self._all_schemas_cache = [
                _compress_schema(s) for s in self.registry.get_schemas()
            ]
        return self._all_schemas_cache

    def _tools_for_kind(self, kind: str) -> list[dict]:
        all_schemas = self._get_all_schemas()
        wanted_categories = KIND_CATEGORIES.get(kind, tuple(TOOL_CATEGORIES.keys()))
        wanted_names = {n for cat in wanted_categories for n in TOOL_CATEGORIES.get(cat, ())}
        filtered = [s for s in all_schemas if _schema_name(s) in wanted_names]
        # Fallback de segurança: se os nomes assumidos em TOOL_CATEGORIES não
        # baterem com nada real (schemas.py/registry.py divergentes), manda
        # tudo — melhor gastar tokens a quebrar o agente por falta de tool.
        return filtered if filtered else all_schemas

    async def _llm_pick_files(self, task: str, file_listing: str) -> list[str]:
        """
        Fallback quando semantic_search não existe no registry ou não
        retornou nada útil: pergunta pro modelo mais barato (MODEL_EDITOR,
        reasoning baixo), dado só a árvore de arquivos (sem conteúdo) + a
        tarefa, uma lista de paths prováveis de serem relevantes.

        Nunca propaga exceção — se falhar por qualquer motivo (parse de
        JSON, LLM indisponível, etc.), retorna [] e o preload é
        simplesmente pulado; o loop ReAct normal (list_files/read_file)
        continua funcionando exatamente como antes.
        """
        prompt = (
            "Você recebe a árvore de arquivos de um projeto e uma tarefa. "
            "Responda APENAS com um JSON array de paths (no máximo "
            f"{PRELOAD_MAX_FILES}) mais prováveis de precisarem ser lidos "
            "ou editados para essa tarefa. Sem texto adicional, sem "
            "markdown, sem explicação — só o array JSON.\n\n"
            f"Tarefa: {task}\n\n"
            f"Arquivos:\n{file_listing[:4000]}"
        )
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "tools": [],
            "tool_choice": "auto",
            "model": MODEL_EDITOR,
            "temperature": 0.1,
            "reasoning_effort": "low",
            "max_tokens": 300,
            "allow_llama_fallback": False,
            "max_retries": 2,
        }
        try:
            r = await self._client.post("/chat/tools", json=payload, timeout=60.0)
            if r.status_code >= 400:
                log.debug(f"_llm_pick_files: LLM retornou {r.status_code}, pulando preload.")
                return []
            data = r.json()
            content = (data.get("message") or {}).get("content") or ""
            content = content.strip()
            if content.startswith("```"):
                content = content.strip("`")
                content = content.split("\n", 1)[-1] if "\n" in content else content
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return [p for p in parsed if isinstance(p, str)]
        except Exception as e:
            log.debug(f"_llm_pick_files falhou (não crítico, preload pulado): {e}")
        return []

    async def _preload_relevant_files(self, task: str) -> Optional[Event]:
        """
        Roda 1x no início de uma sessão NOVA (nunca em sessão retomada),
        ANTES do loop ReAct. Best-effort: qualquer falha em qualquer etapa
        apenas pula o preload — list_files/read_file continuam disponíveis
        normalmente pro agente pedir depois, exatamente como hoje.

        Ordem: semantic_search (se existir e retornar algo) → senão LLM
        barato dado a árvore de arquivos → lê os candidatos um por vez,
        truncando e respeitando PRELOAD_TOKEN_BUDGET.
        """
        if not PRELOAD_ENABLED:
            return None

        preload_ctx = StepContext(step_index=0, is_planning=True)
        candidate_paths: list[str] = []
        method: Optional[str] = None

        semantic_tool = self.registry.get("semantic_search")
        if semantic_tool is not None:
            try:
                call = ToolCall(
                    id="preload-semantic",
                    name="semantic_search",
                    arguments={"query": task, "top_k": PRELOAD_MAX_FILES},
                )
                result = await self.registry.dispatch(call, preload_ctx)
                if result.success:
                    candidate_paths = _extract_paths_from_semantic_result(result)
                    if candidate_paths:
                        method = "semantic_search"
            except Exception as e:
                log.debug(f"Preload: semantic_search falhou (não crítico): {e}")

        if not candidate_paths and self.registry.get("list_files") is not None:
            try:
                call = ToolCall(
                    id="preload-list",
                    name="list_files",
                    arguments={"recursive": True, "max_entries": 500},
                )
                listing = await self.registry.dispatch(call, preload_ctx)
                if listing.success and listing.output:
                    candidate_paths = await self._llm_pick_files(task, listing.output)
                    if candidate_paths:
                        method = "llm"
            except Exception as e:
                log.debug(f"Preload: list_files/_llm_pick_files falhou (não crítico): {e}")

        if not candidate_paths or self.registry.get("read_file") is None:
            return None

        candidate_paths = candidate_paths[:PRELOAD_MAX_FILES]
        preloaded_chunks: list[str] = []
        loaded_paths: list[str] = []
        used_tokens = 0

        for idx, path in enumerate(candidate_paths):
            if used_tokens >= PRELOAD_TOKEN_BUDGET:
                log.info(
                    f"Preload: orçamento de {PRELOAD_TOKEN_BUDGET} tokens atingido, "
                    f"parando em {len(loaded_paths)} arquivo(s) de {len(candidate_paths)} candidatos."
                )
                break
            try:
                call = ToolCall(id=f"preload-read-{idx}", name="read_file", arguments={"file_path": path})
                result = await self.registry.dispatch(call, preload_ctx)
            except Exception as e:
                log.debug(f"Preload: read_file({path}) falhou (não crítico): {e}")
                continue

            if not result.success or not result.output:
                continue

            content = _truncate_output(result.output, READ_FILE_MAX_CHARS)
            chunk_tokens = int(len(content) / _PRELOAD_CHARS_PER_TOKEN)
            if loaded_paths and used_tokens + chunk_tokens > PRELOAD_TOKEN_BUDGET:
                break  # já tem ao menos 1 arquivo — não estoura o orçamento por causa do próximo

            preloaded_chunks.append(f"### {path}\n```\n{content}\n```")
            used_tokens += chunk_tokens
            loaded_paths.append(path)
            await self.session.record_file_seen(path)

        if not preloaded_chunks:
            return None

        preload_msg = {
            "role": "user",
            "content": (
                f"[PRÉ-CARREGADO com base na tarefa, via {method} — arquivos "
                "prováveis de serem relevantes. Se precisar de outros "
                "arquivos, use list_files/read_file normalmente.]\n\n"
                + "\n\n".join(preloaded_chunks)
            ),
        }
        await self.session.add_message(preload_msg, kind="preload")

        return Event(
            event=EventType.PLAN,
            data={
                "note": "Arquivos pré-carregados com base na tarefa",
                "method": method,
                "files": loaded_paths,
                "tokens_used": used_tokens,
            },
            step=0,
        )

    async def run(self, task: str) -> AsyncGenerator[Event, None]:
        """
        Executa a tarefa, yield Event objects para streaming.

        Último event é sempre EventType.FINAL com answer + summary.
        """
        t_start = time.perf_counter()

        # 1. Inicializa messages com system prompt + tarefa do usuário
        if not self.session.state.messages:
            system_msg = {
                "role": "system",
                "content": SYSTEM_PROMPT
                + f"\n\nSessão: {self.session.session_id}\n"
                + (f"Projeto alvo: {self.session.project_dir}\n" if self.session.project_dir else ""),
            }
            await self.session.add_message(system_msg, kind="system")

            user_msg = {
                "role": "user",
                "content": f"# Tarefa\n\n{task}\n\nComece planejando. Use tools para entender o projeto antes de editar.",
            }
            await self.session.add_message(user_msg, kind="user_task")

            preload_event = await self._preload_relevant_files(task)
            if preload_event:
                yield preload_event
        else:
            yield Event(
                event=EventType.THINKING,
                data={"note": "Sessão retomada, continuando de onde parou."},
                step=self.session.state.steps_executed,
            )

        # 2. Loop principal ReAct
        last_error: Optional[str] = None
        final_answer: Optional[str] = None

        try:
            for step_idx in range(self.max_steps):
                # 2a. Compactação local (dedup + encolhimento de tool results
                # antigos) antes de checar se precisa de summarization via LLM —
                # frequentemente já é suficiente para não precisar chamar o LLM
                # só para resumir.
                await self.context.compact_history()
                summarized = await self.context.maybe_summarize()
                if summarized:
                    yield Event(
                        event=EventType.CONTEXT_BUDGET,
                        data={
                            "tokens_used": self.context.tokens_used(),
                            "tokens_remaining": self.context.tokens_remaining(),
                            "summarized": True,
                        },
                        step=step_idx,
                    )
                else:
                    if step_idx % 3 == 0:
                        yield Event(
                            event=EventType.CONTEXT_BUDGET,
                            data={
                                "tokens_used": self.context.tokens_used(),
                                "tokens_remaining": self.context.tokens_remaining(),
                                "summarized": False,
                            },
                            step=step_idx,
                        )

                # 2b. Decide kind do step
                kind = classify_step(
                    step_index=step_idx,
                    is_first_step=(step_idx == 0),
                    last_tool_result_error=last_error,
                    files_changed_count=len(self.session.state.files_changed),
                    user_task_lowercase=task.lower(),
                    max_steps=self.max_steps,
                )
                step_ctx = self.context.build_step_context(kind=kind)

                # 2c. Model router — retorna uma CADEIA ordenada de fallback,
                # não mais 1 modelo só. 3 dos 4 modelos Groq têm RPD=1000/dia;
                # antes, quando um batia rate limit, o código só esperava
                # (30/60/90s) e tentava de NOVO O MESMO MODELO — se o RPD
                # dele tivesse esgotado de verdade, isso nunca resolvia
                # sozinho. Agora rotaciona pro próximo candidato da cadeia,
                # que resolve na hora (os outros modelos têm cota própria).
                chain = pick_chain(step_ctx, override=self.model_override)
                yield Event(
                    event=EventType.MODEL_CHOICE,
                    data={
                        "model": chain[0].model,
                        "reasoning_effort": chain[0].reasoning_effort,
                        "temperature": chain[0].temperature,
                        "step_kind": kind,
                        "fallback_chain": [c.model for c in chain],
                    },
                    step=step_idx,
                )

                # 2d. Chama LLM, rotacionando pela cadeia em rate limit
                message = None
                model_used = ""
                fallback = False
                try:
                    message, model_used, fallback = await self._call_llm_chain(chain, step_idx, kind)
                except RuntimeError as e:
                    log.exception("LLM call falhou (cadeia de modelos esgotada)")
                    yield Event(
                        event=EventType.ERROR,
                        data={"error": f"LLM indisponível: {e}", "fatal": True},
                        step=step_idx,
                    )
                except Exception as e:
                    log.exception("LLM call falhou (erro inesperado)")
                    yield Event(
                        event=EventType.ERROR,
                        data={"error": f"LLM indisponível: {e}", "fatal": True},
                        step=step_idx,
                    )

                if message is None:
                    break  # erro fatal, sai do loop ReAct

                # 2e. Anexa resposta do assistant ao histórico
                await self.session.add_message(message, kind="assistant")
                await self.session.record_step(model_used, kind)

                # 2f. Se tem tool_calls, executa cada um
                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    # thinking content (se houver, normalmente vem como reasoning_content separado)
                    if message.get("content"):
                        yield Event(
                            event=EventType.THINKING,
                            data={"text": message["content"][:500]},
                            step=step_idx,
                        )

                    # executa cada tool call
                    for tc in tool_calls:
                        call = ToolCall.from_groq(tc)
                        yield Event(
                            event=EventType.TOOL_CALL,
                            data={
                                "id": call.id,
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                            step=step_idx,
                        )

                        ctx_for_tool = self.context.build_step_context(kind=kind)
                        tool_timeout = _tool_timeout_seconds(call.name)
                        t_tool_start = time.perf_counter()
                        try:
                            result = await asyncio.wait_for(
                                self.registry.dispatch(call, ctx_for_tool),
                                timeout=tool_timeout,
                            )
                        except asyncio.TimeoutError:
                            elapsed_ms = round((time.perf_counter() - t_tool_start) * 1000, 1)
                            log.error(
                                f"Step {step_idx}: tool '{call.name}' excedeu timeout de "
                                f"{tool_timeout}s — abortando esta chamada, sessão continua."
                            )
                            result = ToolResult(
                                tool_call_id=call.id,
                                tool_name=call.name,
                                success=False,
                                error=(
                                    f"Tool '{call.name}' excedeu o timeout de {tool_timeout}s "
                                    f"e foi abortada. Se era um comando de longa duração (servidor, "
                                    f"watch mode), rode em background ou com um teto de tempo explícito."
                                ),
                                elapsed_ms=elapsed_ms,
                            )

                        # track file changes
                        if call.name in ("write_file", "apply_patch"):
                            fp = call.arguments.get("file_path", "")
                            if fp:
                                await self.session.record_file_change(fp, action=call.name)
                        if call.name == "read_file":
                            fp = call.arguments.get("file_path", "")
                            if fp:
                                await self.session.record_file_seen(fp)

                        # anexa tool result como message — mas antes trunca
                        # outputs grandes (economia de tokens; read_file tem
                        # limite maior pois costuma ser o maior ofensor).
                        tool_msg = result.to_tool_message()
                        max_chars = READ_FILE_MAX_CHARS if call.name == "read_file" else TOOL_OUTPUT_MAX_CHARS
                        if isinstance(tool_msg.get("content"), str):
                            tool_msg["content"] = _truncate_output(tool_msg["content"], max_chars)
                        await self.session.add_message(tool_msg, kind="tool_result")
                        await self.session.record_tool_call(
                            tool_name=call.name,
                            args=call.arguments,
                            result_summary=result.output[:200] if result.output else (result.error or "")[:200],
                        )

                        yield Event(
                            event=EventType.TOOL_RESULT,
                            data={
                                "tool_call_id": call.id,
                                "tool_name": call.name,
                                "success": result.success,
                                "output": result.output[:2000] if result.output else "",
                                "error": result.error,
                                "elapsed_ms": result.elapsed_ms,
                            },
                            step=step_idx,
                        )

                        # Se erro, marca para próximo step ser debugging
                        if not result.success:
                            last_error = result.error or result.output[:200]
                        else:
                            last_error = None

                    continue  # próximo step do ReAct

                # 2g. Sem tool_calls → resposta final
                final_answer = message.get("content") or ""
                break

            # 3. Finaliza
            elapsed = (time.perf_counter() - t_start) * 1000

            if final_answer is None:
                final_answer = (
                    f"Tarefa interrompida após {self.max_steps} steps sem resposta final. "
                    f"Veja o log da sessão para detalhes."
                )

            yield Event(
                event=EventType.FINAL,
                data={
                    "answer": final_answer,
                    "session_id": self.session.session_id,
                    "steps_executed": self.session.state.steps_executed,
                    "tools_called": self.session.state.tools_called,
                    "tokens_used": self.session.state.tokens_used,
                    "files_changed": self.session.state.files_changed,
                    "model_steps": self.session.state.model_steps,
                    "elapsed_ms": round(elapsed, 1),
                    "success": True,
                },
                step=self.session.state.steps_executed,
            )

        except Exception as e:
            log.exception("Agent crashed")
            yield Event(
                event=EventType.ERROR,
                data={"error": f"{type(e).__name__}: {e}", "fatal": True},
            )
            yield Event(
                event=EventType.FINAL,
                data={
                    "answer": f"Erro fatal: {e}",
                    "session_id": self.session.session_id,
                    "steps_executed": self.session.state.steps_executed,
                    "tools_called": self.session.state.tools_called,
                    "tokens_used": self.session.state.tokens_used,
                    "elapsed_ms": round((time.perf_counter() - t_start) * 1000, 1),
                    "success": False,
                },
            )

    async def _call_llm_chain(
        self,
        chain: list[ModelChoice],
        step_idx: int,
        kind: str,
        max_full_passes: int = 3,
    ) -> tuple[dict, str, bool]:
        """
        Tenta os modelos de `chain` em ordem. Se um bater rate limit
        (429/RPD/TPD — detectado pela wording "429"/"rate limit"/"rate-limited"
        na RuntimeError que sobe de _call_llm), passa pro próximo candidato
        IMEDIATAMENTE, sem esperar — trocar de modelo é o que resolve rate
        limit de verdade aqui, já que 3 dos 4 modelos têm RPD=1000/dia
        independentes entre si.

        Erros que NÃO são de rate limit (contexto grande demais sem solução,
        tool call quebrado 3x seguidas, erro de rede) não têm nada a ver com
        qual modelo foi usado — propagam direto pro caller, sem rotacionar
        (rotacionar não ajudaria e só mascararia o erro real).

        Só espera (sleep com backoff) e refaz a cadeia inteira do zero se
        TODOS os candidatos estiverem rate-limited na mesma passada — situação
        rara (precisa os 4 modelos esgotados ao mesmo tempo), mas possível
        sob carga alta simultânea.
        """
        last_rate_limit_error: Optional[Exception] = None

        for full_pass in range(max_full_passes):
            for idx, candidate in enumerate(chain):
                try:
                    return await self._call_llm(choice=candidate, step_idx=step_idx, kind=kind)
                except RuntimeError as e:
                    err_msg = str(e)
                    is_rate_limit = (
                        "429" in err_msg
                        or "rate-limited" in err_msg
                        or "rate limit" in err_msg.lower()
                    )
                    if not is_rate_limit:
                        # Não é rate limit — trocar de modelo não resolveria.
                        raise
                    last_rate_limit_error = e
                    if idx < len(chain) - 1:
                        log.warning(
                            f"Step {step_idx}: {candidate.model} rate-limited "
                            f"(passada {full_pass+1}/{max_full_passes}). Rotacionando "
                            f"para {chain[idx+1].model}."
                        )
                    # último candidato da passada também rate-limited → cai
                    # pro sleep+retry da passada inteira, abaixo

            if full_pass < max_full_passes - 1:
                wait_s = 30.0 * (full_pass + 1)
                log.warning(
                    f"Step {step_idx}: todos os {len(chain)} modelos da cadeia "
                    f"({', '.join(c.model for c in chain)}) estão rate-limited "
                    f"(passada {full_pass+1}/{max_full_passes}). Esperando {wait_s}s "
                    f"antes de refazer a cadeia inteira."
                )
                await asyncio.sleep(wait_s)

        raise RuntimeError(
            f"Step {step_idx}: todos os modelos Groq "
            f"({', '.join(c.model for c in chain)}) estavam rate-limited após "
            f"{max_full_passes} passadas completas pela cadeia. "
            f"Última falha: {last_rate_limit_error}"
        )

    async def _pick_best_headroom_model(
        self, safe_limits: dict[str, int], avoid: str
    ) -> tuple[str, int]:
        """
        Escolhe o melhor modelo de fallback pra estourar de tamanho de
        contexto (too_large), considerando o estado REAL de rate limit —
        não só o TPM configurado estaticamente.

        Antes: escolhia sempre `max(safe_limits.items(), key=lambda kv: kv[1])`
        — ou seja, sempre o MESMO modelo (o de maior TPM configurado),
        mesmo que ele tivesse acabado de levar 2-3 429 seguidos e estivesse
        prestes a ser pré-bloqueado pelo LLM.py. Isso causava um incidente
        real: todo step que estourasse o limite martelava de novo o mesmo
        modelo já sangrando, encadeando esperas de dezenas de segundos por
        step (ver incidente 2026-07-23).

        Agora: consulta GET /rate-limit-status (LLM.py) e evita candidatos
        com `blocked=True` ou `daily_exhausted=True` — entre os que sobram,
        pega o de maior TPM configurado. Se a consulta falhar por qualquer
        motivo, ou se todos estiverem bloqueados, cai de volta no
        comportamento antigo (melhor tentar algo do que travar aqui).
        """
        candidates = {m: tpm for m, tpm in safe_limits.items() if m != avoid}
        if not candidates:
            return max(safe_limits.items(), key=lambda kv: kv[1])

        try:
            r = await self._client.get("/rate-limit-status")
            r.raise_for_status()
            status = r.json()
        except Exception as e:
            log.warning(
                f"Falha ao consultar /rate-limit-status ({e}); escolhendo "
                f"fallback só pelo TPM configurado, sem checar disponibilidade real."
            )
            return max(candidates.items(), key=lambda kv: kv[1])

        available = {
            m: tpm for m, tpm in candidates.items()
            if not status.get(m, {}).get("blocked", False)
            and not status.get(m, {}).get("daily_exhausted", False)
        }
        if not available:
            log.warning(
                "Todos os modelos candidatos de fallback estão bloqueados/"
                "esgotados agora — usando o de maior TPM mesmo assim."
            )
            return max(candidates.items(), key=lambda kv: kv[1])

        return max(available.items(), key=lambda kv: kv[1])

    async def _call_llm(self, choice: ModelChoice, step_idx: int, kind: str = "editing") -> tuple[dict, str, bool]:
        """
        Chama LLM.py:4003 /chat/tools com messages + tools (filtradas por kind).

        Estratégia anti-"request too large" (revisada):
          1. Estima tokens antes de enviar (heurística ~3.3 chars/token,
             já considerando só as tools relevantes ao kind do step)
          2. Se estourar o limite seguro do modelo, tenta reduzir o próprio
             conteúdo primeiro (compact_history / summarize) — trocar de
             modelo sozinho não ajuda, já que os limites reais de TPM são
             equivalentes entre os modelos nesta org (ver MODEL_TPM_LIMITS)
          3. Só troca de modelo se o alvo tiver headroom REAL maior
          4. Se LLM ainda retornar too_large=true, tenta de novo após
             compactar/sumarizar; em último caso, emergency_summarize()
        """
        SAFETY_FACTOR = 0.85
        safe_limits = {m: int(tpm * SAFETY_FACTOR) for m, tpm in MODEL_TPM_LIMITS.items()}

        estimated_tokens = self._estimate_request_tokens(choice, kind)
        safe_limit = safe_limits.get(choice.model, min(safe_limits.values()))

        if estimated_tokens > safe_limit:
            log.warning(
                f"Step {step_idx}: contexto estimado em {estimated_tokens} tokens "
                f"excede limite seguro de {choice.model} ({safe_limit}). "
                f"Tentando compactar histórico antes de trocar de modelo."
            )
            await self.context.compact_history()
            estimated_tokens = self._estimate_request_tokens(choice, kind)

            if estimated_tokens > safe_limit:
                await self.context.maybe_summarize()
                estimated_tokens = self._estimate_request_tokens(choice, kind)

            if estimated_tokens > safe_limit:
                best_model, best_limit = await self._pick_best_headroom_model(safe_limits, avoid=choice.model)
                if best_model != choice.model and best_limit > safe_limit * 1.15:
                    log.warning(
                        f"Step {step_idx}: ainda {estimated_tokens} tokens após compactação. "
                        f"Trocando para {best_model} (limite real {best_limit} > {safe_limit})."
                    )
                    choice = ModelChoice(
                        model=best_model,
                        # ATENÇÃO: só gpt-oss (120B/20B) suporta reasoning_effort —
                        # mandar isso pra um modelo Llama (best_model pode ser
                        # MODEL_DOC ou MODEL_FAST) faz a Groq rejeitar com
                        # "reasoning effort is not supported by this model".
                        reasoning_effort=reasoning_effort_for(best_model, choice.reasoning_effort or "low"),
                        temperature=choice.temperature,
                        max_tokens=choice.max_tokens,
                    )
                else:
                    log.warning(
                        f"Step {step_idx}: nenhum modelo tem headroom real maior que "
                        f"{choice.model} nesta org — mantendo o modelo e confiando na "
                        f"compactação já feita."
                    )

        # ── Tenta chamar com choice (possivelmente ajustado) ──────────────────
        result = await self._do_llm_request(choice, step_idx, kind)

        # ── Tool call JSON malformado — geralmente um glitch transitório de
        # geração do Groq, não um problema de tamanho de contexto. Antes,
        # QUALQUER 503 (que é como LLM.py embrulha todo erro do Groq,
        # incluindo este) derrubava a task inteira sem tentar de novo.
        # Agora: até 2 retries, o último com temperature=0 (reduz a chance
        # do modelo gerar JSON quebrado de novo).
        retry_count = 0
        while result.get("malformed_tool_call") and retry_count < 2:
            retry_count += 1
            log.warning(
                f"Step {step_idx}: Groq retornou tool call JSON malformado "
                f"(tentativa {retry_count}/2 de retry)."
            )
            retry_choice = choice if retry_count == 1 else ModelChoice(
                model=choice.model,
                reasoning_effort=choice.reasoning_effort,
                temperature=0.0,
                max_tokens=choice.max_tokens,
            )
            result = await self._do_llm_request(retry_choice, step_idx, kind)
            choice = retry_choice

        if result.get("malformed_tool_call"):
            raise RuntimeError(
                f"Step {step_idx}: Groq retornou tool call JSON malformado 3x seguidas "
                f"para {choice.model}. Desistindo — pode ser um problema momentâneo do "
                f"provider com esse modelo/prompt."
            )

        # ── Tool call com argumentos que não batem com o schema (ex: falta
        # 'file_path' obrigatório) — mesma estratégia de retry do JSON
        # malformado acima: até 2 tentativas, a última com temperature=0.
        # Antes disso era tratado como erro fatal genérico (ver incidente
        # 2026-07-22: RuntimeError não capturado matava a sessão inteira).
        retry_count = 0
        while result.get("invalid_tool_args") and retry_count < 2:
            retry_count += 1
            log.warning(
                f"Step {step_idx}: Groq rejeitou argumentos de tool call por schema "
                f"mismatch (tentativa {retry_count}/2 de retry)."
            )
            retry_choice = choice if retry_count == 1 else ModelChoice(
                model=choice.model,
                reasoning_effort=choice.reasoning_effort,
                temperature=0.0,
                max_tokens=choice.max_tokens,
            )
            result = await self._do_llm_request(retry_choice, step_idx, kind)
            choice = retry_choice

        if result.get("invalid_tool_args"):
            raise RuntimeError(
                f"Step {step_idx}: Groq rejeitou tool call por schema mismatch 3x "
                f"seguidas para {choice.model} (ex: propriedade obrigatória faltando). "
                f"Desistindo — pode ser um problema momentâneo do provider com esse "
                f"modelo/prompt, ou o schema da tool precisa ficar mais claro."
            )

        # ── Se retornou too_large, compacta/sumariza (não só troca de modelo) ─
        if result.get("too_large"):
            log.warning(f"Step {step_idx}: LLM retornou too_large para {choice.model}.")

            await self.context.compact_history()
            result = await self._do_llm_request(choice, step_idx, kind)

            if result.get("too_large"):
                best_model, best_limit = await self._pick_best_headroom_model(safe_limits, avoid=choice.model)
                if best_model != choice.model:
                    fallback_choice = ModelChoice(
                        model=best_model,
                        reasoning_effort=reasoning_effort_for(best_model, "low"),
                        temperature=0.3,
                        max_tokens=choice.max_tokens,
                    )
                    result = await self._do_llm_request(fallback_choice, step_idx, kind)
                    choice = fallback_choice

            if not result.get("too_large"):
                return result["message"], result["model"], result["fallback_used"]

            # ── Ainda too_large → summarização emergencial ───────────────────
            log.error(
                f"Step {step_idx}: ainda too_large mesmo após compactação/troca de modelo. "
                f"Forçando emergency_summarize."
            )
            try:
                summarized = await self.context.emergency_summarize()
            except Exception as e:
                # Falha DENTRO do emergency_summarize (ex: a própria chamada
                # de summarization ao LLM deu erro). Isso sim é "não
                # conseguimos reduzir o contexto" — cai no too_large final.
                log.exception(f"Step {step_idx}: emergency_summarize() falhou: {e}")
                summarized = False

            if summarized:
                try:
                    result = await self._do_llm_request(choice, step_idx, kind)
                except Exception as e:
                    # ATENÇÃO: isto NÃO é mais "emergency summarization
                    # failed" — o summarize funcionou; foi o retry do
                    # request real (com os tools/schema do step) que falhou,
                    # por um motivo possivelmente bem diferente de tamanho
                    # de contexto (ex: incidente 2026-07-23: Groq 400
                    # "Failed to call a function" após um corte de histórico
                    # deixar um tool result órfão — ver _ensure_tool_pairing
                    # em context.py). Antes isso era engolido aqui e o
                    # usuário via só "Context too large..." — mensagem
                    # totalmente errada pro problema real.
                    raise RuntimeError(
                        f"Step {step_idx}: emergency_summarize() reduziu o contexto com "
                        f"sucesso, mas o request seguinte falhou por outro motivo "
                        f"(não tamanho de contexto): {e}"
                    ) from e
                if not result.get("too_large"):
                    return result["message"], result["model"], result["fallback_used"]

            raise RuntimeError(
                f"Context too large for any Groq model even after summarization. "
                f"Estimated {estimated_tokens} tokens. Reduce task complexity or start new session."
            )

        return result["message"], result["model"], result["fallback_used"]

    def _estimate_request_tokens(self, choice: ModelChoice, kind: str = "editing") -> int:
        """
        Estima tokens do request atual (messages + tools schemas filtradas
        por kind). Usa json.dumps (mais fiel ao payload real que str()) e
        ~3.3 chars/token — código/JSON tende a ser mais denso em tokens que
        texto natural, então 4 chars/token subestimava o real (ver
        incidente 2026-07-21: estimado 7726, real ~10463).
        """
        CHARS_PER_TOKEN = 3.3
        messages_json = json.dumps(self.session.state.messages, ensure_ascii=False, default=str)
        tools_json = json.dumps(self._tools_for_kind(kind), ensure_ascii=False, default=str)
        return int((len(messages_json) + len(tools_json)) / CHARS_PER_TOKEN)

    async def _do_llm_request(self, choice: ModelChoice, step_idx: int, kind: str = "editing") -> dict:
        """Faz uma chamada HTTP real ao /chat/tools. Retorna dict com message/model/fallback_used/too_large/malformed_tool_call."""
        tools_schemas = self._tools_for_kind(kind)

        payload = {
            "messages": self.session.state.messages,
            "tools": tools_schemas,
            "tool_choice": "auto",
            "model": choice.model,
            "temperature": choice.temperature,
            "allow_llama_fallback": False,
            "max_retries": 3,
        }
        if choice.reasoning_effort:
            payload["reasoning_effort"] = choice.reasoning_effort
        if choice.max_tokens:
            payload["max_tokens"] = choice.max_tokens

        r = await self._client.post("/chat/tools", json=payload, timeout=600.0)

        if r.status_code == 429:
            # LLM.py já tenta internamente (max_retries=3) com backoff antes de
            # devolver 429 — se chegou até aqui, o backoff dele já se esgotou.
            raise RuntimeError(f"Groq rate-limited após retries: {r.text[:300]}")

        if r.status_code >= 400:
            # ATENÇÃO: LLM.py embrulha TODO erro do Groq (413 too-large, 400
            # tool-call JSON malformado, etc.) como 503 pro alpha_code — antes
            # isso era tratado como fatal na hora, o que tornava a lógica de
            # too_large em _call_llm código morto (nunca era alcançada) e
            # matava a task inteira em qualquer glitch transitório de geração.
            # Agora inspecionamos o `detail` pra decidir o que fazer.
            try:
                detail = (r.json() or {}).get("detail", "") or ""
            except Exception:
                detail = r.text[:300]
            detail_l = detail.lower()

            if any(kw in detail_l for kw in ("413", "too large", "tokens per minute", "tpm")):
                return {
                    "message": None, "model": choice.model, "fallback_used": False,
                    "too_large": True, "malformed_tool_call": False,
                }

            if "failed to parse tool call arguments" in detail_l:
                return {
                    "message": None, "model": choice.model, "fallback_used": False,
                    "too_large": False, "malformed_tool_call": True,
                    "invalid_tool_args": False,
                }

            if "tool call validation failed" in detail_l or "did not match schema" in detail_l:
                # Groq validou os argumentos do tool call contra o schema e
                # rejeitou (ex: propriedade obrigatória faltando, tipo errado).
                # Mesma categoria de glitch transitório de geração que o JSON
                # malformado acima — não é um erro de tamanho de contexto nem
                # um bug estrutural, então tratamos como retryable também.
                log.warning(f"Step {step_idx}: Groq rejeitou tool call por schema mismatch: {detail[:300]}")
                return {
                    "message": None, "model": choice.model, "fallback_used": False,
                    "too_large": False, "malformed_tool_call": False,
                    "invalid_tool_args": True,
                }

            # Erro genuinamente desconhecido/fatal — continua derrubando a task.
            raise RuntimeError(f"LLM /chat/tools → {r.status_code}: {detail or r.text[:300]}")

        data = r.json()
        return {
            "message": data["message"],
            "model": data.get("model", choice.model),
            "fallback_used": data.get("fallback_used", False),
            "too_large": data.get("too_large", False),
            "malformed_tool_call": False,
            "invalid_tool_args": False,
        }