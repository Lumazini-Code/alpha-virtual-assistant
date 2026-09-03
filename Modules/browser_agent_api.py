"""
AVA Browser Agent API — Orquestrador de navegação guiado por LLM com grammar
=============================================================================

Implementa o pipeline descrito no projeto:

    requisição (pedido base + lista de passos sugeridos)
        → TODO o contexto (pedido, passos, histórico heurístico, estrutura
          atual da página, descrição das funções disponíveis) é montado num
          prompt único a cada rodada — os "passos sugeridos" NUNCA são
          executados em ordem fixa, são só mais uma fonte de contexto que o
          modelo pode seguir, ignorar, pular ou reordenar.
        → [1] um modelo pequeno (LFM2.5 350M, rodando num llama-server local)
              decide a PRÓXIMA ação, respondendo em JSON estritamente restrito
              por uma grammar GBNF (sem tool calling — só grammar).
        → [2] a ação é executada via browser_agent.WebAgent (click/type/...)
              e um heurístico gera uma linha de histórico em português
              (ex.: 'Digitei "canal do usuário" em "Pesquisar" (id 7).')
        → [3] quando o modelo pequeno decide que a tarefa terminou, ele chama
              a função "finish" — que a grammar OBRIGA a vir com uma
              justificativa e uma lista de ids (da estrutura atual) que
              contêm a informação útil.
        → [4] esse "finish" é então validado por um modelo MAIOR, rodando
              noutro llama-server (porta 2002 por padrão), também com
              grammar própria (valid/reason/next_steps). Se ele invalidar,
              os "next_steps" sugeridos são adicionados ao CONTEXTO (de novo,
              não como ordem fixa) e o loop continua.
        → [5] uma vez confirmado, se sobrou conteúdo útil extraído (texto pra
              usar como fonte de informação), o PRÓPRIO modelo pequeno faz um
              resumo com perda mínima de detalhe. Antes de resumir, qualquer
              trecho que pareça código (detectado por regex — blocos
              ```cifrados```, `inline`, ou linhas com cara de código) é
              separado do texto; o resumo é feito só com o texto restante, e
              os trechos de código são reanexados ao final.

Este arquivo NÃO controla o browser diretamente: toda a extração de DOM e
execução de ações vem do browser_agent.py (Playwright + stealth). Aqui só
existe a camada de decisão (LLMs + grammar) e orquestração (loop, histórico,
confirmação, resumo) — igual ao vision.py, que não decide nada sozinho e
delega pro memory.py o que fazer com resultados ambíguos.

Segue o mesmo estilo arquitetural do vision.py: FastAPI + Pydantic + httpx,
API REST simples, configuração 100% via variável de ambiente.
"""

from __future__ import annotations

import os
import re
import json
import time
import asyncio
import logging
from functools import partial
from dataclasses import dataclass
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from browser_agent import WebAgent

# Carrega variáveis de um .env na pasta atual, se existir — mesmo padrão do
# vision.py. Opcional: sem python-dotenv, só usa o que já está no shell.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Configuração ────────────────────────────────────────────────────────────
# Tudo ajustável via variável de ambiente, sem tocar no código.

def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default

def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# Modelo pequeno (LFM2.5 350M) — decide cada ação, grammar-only.
SMALL_LLM_URL = _env_str("AGENT_SMALL_LLM_URL", "http://localhost:2002")
# Modelo maior — só confirma (ou rejeita) o "finish" do modelo pequeno.
BIG_LLM_URL = _env_str("AGENT_BIG_LLM_URL", "http://localhost:2002")
LLM_HTTP_TIMEOUT = _env_float("AGENT_LLM_HTTP_TIMEOUT", 60.0)

N_PREDICT_ACTION  = _env_int("AGENT_N_PREDICT_ACTION", 220)
N_PREDICT_CONFIRM = _env_int("AGENT_N_PREDICT_CONFIRM", 300)
N_PREDICT_SUMMARY = _env_int("AGENT_N_PREDICT_SUMMARY", 700)

# Parâmetros de sampling enviados em CADA /completion. Antes o cliente
# mandava "temperature": 0.0 fixo no payload — como o llama-server aplica
# os parâmetros do corpo da requisição por cima dos defaults de CLI, isso
# SOBRESCREVIA o --temp 0.1 usado pra subir o servidor e forçava greedy
# decoding puro. Sob a grammar (que já restringe bastante o vocabulário),
# greedy decoding é o cenário clássico pra travar num loop de repetição de
# token (o ">>>>>>>>>>>>>>>" visto nos logs). Os defaults abaixo replicam
# os flags de inicialização do servidor (--temp 0.1 --top-k 50
# --repeat-penalty 1.05); ajuste via env se subir os servidores com outros
# valores.
LLM_TEMPERATURE    = _env_float("AGENT_LLM_TEMPERATURE", 0.1)
LLM_TOP_K          = _env_int("AGENT_LLM_TOP_K", 50)
LLM_REPEAT_PENALTY = _env_float("AGENT_LLM_REPEAT_PENALTY", 1.05)

DEFAULT_MAX_ITERATIONS      = _env_int("AGENT_MAX_ITERATIONS", 25)
DEFAULT_MAX_FINISH_ATTEMPTS = _env_int("AGENT_MAX_FINISH_ATTEMPTS", 3)

MAX_ITEMS_PER_STRUCTURE    = _env_int("AGENT_MAX_ITEMS_PER_STRUCTURE", 150)
MAX_PROMPT_CHARS_STRUCTURE = _env_int("AGENT_MAX_PROMPT_CHARS_STRUCTURE", 6000)
MAX_ITEM_TEXT_CHARS        = _env_int("AGENT_MAX_ITEM_TEXT_CHARS", 140)

# browser_agent.WebAgent — Chrome real com debugging remoto.
BROWSER_DEBUG_PORT      = _env_int("AGENT_BROWSER_DEBUG_PORT", 9222)
BROWSER_PROFILE_DIR     = _env_str("AGENT_BROWSER_PROFILE_DIR", "Default")
BROWSER_BLOCK_RESOURCES = _env_bool("AGENT_BROWSER_BLOCK_RESOURCES", True)
AUTO_START_BROWSER      = _env_bool("AGENT_AUTO_START_BROWSER", True)

API_PORT = _env_int("AGENT_API_PORT", 4003)

# ── Logging ──────────────────────────────────────────────────────────────
# AGENT_LOG_LEVEL controla o nível (INFO por padrão mostra cada ação decidida
# e executada + resultado; DEBUG mostra também os prompts/respostas cruas
# completas dos LLMs, que podem ser grandes e conter o texto da página).
AGENT_LOG_LEVEL   = _env_str("AGENT_LOG_LEVEL", "INFO").upper()
# Trunca blobs grandes (prompts, conteúdo extraído) nos logs em nível INFO.
# Em DEBUG, os prompts completos são logados sem truncar.
AGENT_LOG_MAX_CHARS = _env_int("AGENT_LOG_MAX_CHARS", 400)

logging.basicConfig(level=AGENT_LOG_LEVEL, format="%(asctime)s [%(levelname)s] [BrowserAgent] %(message)s")
log = logging.getLogger("ava.browser_agent")


def _clip(text: str, limit: int = AGENT_LOG_MAX_CHARS) -> str:
    """Encurta texto para log, deixando claro quando foi cortado."""
    text = "" if text is None else str(text)
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…[+{len(text) - limit} chars]"


# ── Funções que o modelo pequeno pode executar (viram parte do prompt) ─────
# Fonte única de verdade: os nomes daqui viram os literais aceitos pela
# grammar da ação (ver _build_action_grammar). Adicionar uma função nova é
# só adicionar uma entrada aqui + um branch em _execute_action.

FUNCTION_DESCRIPTIONS: dict[str, str] = {
    "click":        "Clica em um elemento (link, botão, aba, item de menu...) pelo id.",
    "type":         "Digita um texto num campo. Requer item_id e value (o texto).",
    "clear":        "Limpa o conteúdo de um campo de texto. Requer item_id.",
    "press":        "Aperta uma tecla dentro de um elemento focado. Requer item_id e value (ex.: 'Enter', 'Escape', 'Tab').",
    "press_global": "Aperta uma tecla na página inteira, sem precisar de item específico. Requer value (a tecla).",
    "toggle":       "Alterna um checkbox/switch. Requer item_id.",
    "select":       "Seleciona uma opção num combobox/listbox/radio. Requer item_id e value (texto da opção).",
    "set_value":    "Define o valor numérico de um slider. Requer item_id e value (o número).",
    "expand":       "Expande um menu/árvore/combobox recolhido. Requer item_id.",
    "collapse":     "Recolhe um menu/árvore/combobox expandido. Requer item_id.",
    "submit":       "Envia um formulário ou confirma uma busca. Requer item_id.",
    "navigate":     "Segue o link (href) de um item. Requer item_id.",
    "close_dialog": "Fecha um diálogo/modal aberto. Requer item_id.",
    "wait":         "Espera alguns segundos antes de olhar a página de novo. Requer value (segundos, ex.: '1.5').",
    "open":         "Navega direto para uma URL, sem precisar clicar em nada. Requer value (a URL).",
    "search":       "Faz uma busca no Google e devolve os resultados orgânicos. Requer value (o termo buscado).",
    "finish": (
        "Encerra a tarefa porque ela JÁ foi cumprida (pela estrutura atual da página ou pelo "
        "histórico de ações). Requer 'justification' explicando o porquê, e 'useful_item_ids' "
        "com os ids (da estrutura atual) que contêm a informação pedida — pode ser uma lista "
        "vazia se a tarefa era só uma ação (ex.: 'abrir uma página'), sem informação pra extrair."
    ),
}

ACTION_FUNCTION_NAMES: list[str] = list(FUNCTION_DESCRIPTIONS.keys())

# Funções que operam sobre um item específico da última extração — sem
# item_id elas não têm como ser executadas, então isso é validado ANTES de
# chamar o WebAgent (em vez de deixar o erro estourar lá dentro).
FUNCTIONS_REQUIRING_ITEM_ID = {
    "click", "type", "clear", "press", "toggle", "select",
    "set_value", "expand", "collapse", "submit", "navigate", "close_dialog",
}


# ── Grammars GBNF (llama.cpp) — sem tool calling, só JSON restrito ─────────
# Bloco comum de primitivos JSON reaproveitado pelas 3 grammars abaixo.
_JSON_PRIMITIVES = r'''
string ::= "\"" char* "\""
char ::= [^"\\\x00-\x1f] | "\\" ["\\/bfnrt] | "\\u" hex hex hex hex
hex ::= [0-9a-fA-F]
ws ::= [ \t\n]*
'''

_ACTION_GRAMMAR_TEMPLATE = r'''
root ::= "{" ws "\"function\":" ws function ws "," ws "\"item_id\":" ws int-or-null ws "," ws "\"value\":" ws string-or-null ws "," ws "\"justification\":" ws string-or-null ws "," ws "\"useful_item_ids\":" ws int-array ws "}"
function ::= __FUNCTION_ALTERNATIVES__
int-or-null ::= "null" | int
int ::= "-"? [0-9]+
int-array ::= "[" ws "]" | "[" ws int (ws "," ws int)* ws "]"
string-or-null ::= "null" | string
''' + _JSON_PRIMITIVES


def _build_action_grammar() -> str:
    alternatives = " | ".join('"\\"%s\\""' % name for name in ACTION_FUNCTION_NAMES)
    return _ACTION_GRAMMAR_TEMPLATE.replace("__FUNCTION_ALTERNATIVES__", alternatives)


ACTION_GRAMMAR = _build_action_grammar()

CONFIRMATION_GRAMMAR = r'''
root ::= "{" ws "\"valid\":" ws bool ws "," ws "\"reason\":" ws string ws "," ws "\"next_steps\":" ws string-array ws "}"
bool ::= "true" | "false"
string-array ::= "[" ws "]" | "[" ws string (ws "," ws string)* ws "]"
''' + _JSON_PRIMITIVES

SUMMARY_GRAMMAR = r'''
root ::= "{" ws "\"summary\":" ws string ws "}"
''' + _JSON_PRIMITIVES


# ── Modelos Pydantic ────────────────────────────────────────────────────────

class TaskRequest(BaseModel):
    base_request: str = Field(..., description="O pedido base, em linguagem natural (ex.: 'abra o YouTube no meu canal').")
    steps: list[str] = Field(
        default_factory=list,
        description="Sugestões de como fazer, como CONTEXTO — não uma ordem fixa. O modelo pode pular, reordenar ou ignorar.",
    )
    max_iterations: Optional[int] = Field(None, description="Teto de rodadas ação↔estrutura antes de desistir.")
    max_finish_attempts: Optional[int] = Field(None, description="Teto de tentativas de 'finish' rejeitadas pelo revisor.")


class AgentAction(BaseModel):
    """Formato exato que a ACTION_GRAMMAR força o modelo pequeno a produzir."""
    function: str
    item_id: Optional[int] = None
    value: Optional[str] = None
    justification: Optional[str] = None
    useful_item_ids: list[int] = Field(default_factory=list)


class ConfirmationResult(BaseModel):
    """Formato exato que a CONFIRMATION_GRAMMAR força o modelo maior a produzir."""
    valid: bool
    reason: str
    next_steps: list[str] = Field(default_factory=list)


class TaskResult(BaseModel):
    success: bool
    justification: str = ""
    confirmation_reason: Optional[str] = None
    useful_items: list[dict] = Field(default_factory=list)
    summary: str = ""
    code_blocks: list[str] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)
    iterations: int = 0


# ── Cliente HTTP genérico para um llama-server com grammar GBNF ───────────

class LlamaGrammarClient:
    """
    Fala com o endpoint /completion de um llama-server (llama.cpp) passando
    `grammar` (GBNF) — não usa tool calling, o "function calling" aqui É a
    grammar: o servidor só deixa o modelo emitir tokens que fecham no JSON
    definido nela.
    """

    def __init__(self, base_url: str, timeout: float, name: str = "llm"):
        self.base_url = base_url.rstrip("/")
        self.name = name
        self._client = httpx.AsyncClient(timeout=timeout)

    async def complete(
        self, prompt: str, grammar: str, n_predict: int,
        temperature: float = LLM_TEMPERATURE,
        top_k: int = LLM_TOP_K,
        repeat_penalty: float = LLM_REPEAT_PENALTY,
    ) -> str:
        payload = {
            "prompt": prompt,
            "grammar": grammar,
            "n_predict": n_predict,
            "temperature": temperature,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
        }
        log.debug(f"[{self.name}] → POST {self.base_url}/completion (n_predict={n_predict}) prompt=\"{prompt}\"")
        log.info(f"[{self.name}] enviando prompt ({len(prompt)} chars, n_predict={n_predict}): \"{_clip(prompt)}\"")

        t0 = time.perf_counter()
        try:
            resp = await self._client.post(f"{self.base_url}/completion", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            log.error(f"[{self.name}] llama-server respondeu HTTP {e.response.status_code}: {_clip(e.response.text)}")
            raise
        except httpx.HTTPError as e:
            log.error(f"[{self.name}] falha de conexão com {self.base_url}: {e}")
            raise
        elapsed = time.perf_counter() - t0

        data = resp.json()
        content = data.get("content")
        if content is None:
            log.error(f"[{self.name}] resposta sem 'content' ({elapsed:.2f}s): {data}")
            raise RuntimeError(f"llama-server ({self.base_url}) respondeu sem 'content': {data}")

        log.debug(f"[{self.name}] ← resposta crua completa ({elapsed:.2f}s): \"{content}\"")
        log.info(f"[{self.name}] resposta recebida em {elapsed:.2f}s ({len(content)} chars): \"{_clip(content)}\"")
        return content

    async def ping(self) -> bool:
        try:
            resp = await self._client.get(f"{self.base_url}/health", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


# Mesmo com repeat_penalty, um modelo pequeno sob grammar pode travar
# emitindo o mesmo caractere várias vezes seguidas (ex.: ">>>>>>>>>>>>>>>").
# Detecta esse padrão pra tratar a rodada como resposta inválida, em vez de
# executar/aceitar um 'finish' com valor lixo.
_DEGENERATE_OUTPUT_RE = re.compile(r"(.)\1{4,}")


def _looks_degenerate(*texts: Optional[str]) -> bool:
    return any(t and _DEGENERATE_OUTPUT_RE.search(t) for t in texts)


_NEGATION_MARKERS_RE = re.compile(
    r"n[ãa]o (?:completou|cumpr[ie]|realizou|foi cumprida|foi concluída|atende)|not complet"
)


def _clean_step(step: str) -> Optional[str]:
    """
    Normaliza um item de 'next_steps' vindo do revisor antes de virar
    contexto pro modelo pequeno. Sem isso, fragmentos como ',' ou ' '
    (frequentes quando o revisor também degenera a resposta) entravam
    direto na lista de 'steps', inflando e sujando o prompt a cada rodada
    rejeitada — o que por sua vez contribuía pra degenerar o modelo pequeno
    ainda mais.
    """
    step = (step or "").strip(" \t\n,.;:-")
    return step if len(step) >= 4 else None


def _parse_json_object(text: str) -> dict:
    """
    Faz o parse do JSON emitido pelo modelo. Com a grammar, o retorno já
    deveria ser JSON puro — mas alguns llama-servers ecoam o prompt ou
    adicionam espaço/token solto antes/depois, então há um fallback que
    procura o primeiro objeto `{...}` balanceado no texto.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError(f"Nenhum JSON encontrado na resposta do modelo: {text!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"JSON malformado/incompleto na resposta do modelo: {text!r}")


# ── Detecção de código por regex (separado ANTES do resumo) ───────────────

_FENCED_CODE_RE = re.compile(r"```[a-zA-Z0-9+_-]*\n?(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]{2,300})`")
_CODE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"def\s|class\s|function\s|import\s|from\s+\S+\s+import|"
    r"const\s|let\s|var\s|public\s|private\s|static\s|#include|"
    r"</?[a-zA-Z][^>]*>|\$\(.*\)|SELECT\s.+FROM|<\?php|"
    r"[\w.]+\([^)]*\)\s*[{;]"
    r")",
    re.IGNORECASE,
)


def extract_code_blocks(text: str) -> tuple[str, list[str]]:
    """
    Separa trechos que parecem código do resto do texto, ANTES de resumir —
    exatamente como pedido: nada de perder um snippet de código dentro de um
    resumo com "perda mínima de detalhes" (resumo é pra prosa, não pra
    código). Detecta blocos ```cercados```, `trechos inline` e linhas soltas
    com cara de código (def/class/import/tags HTML/SQL/chamada de função
    seguida de `{`/`;`...). É heurístico por natureza — regex não é um
    parser de linguagem nenhuma, então a meta é "não perder o snippet",
    não "classificar a linguagem certinho".
    """
    code_blocks: list[str] = []

    def _pull(match: re.Match) -> str:
        code_blocks.append(match.group(1).strip())
        return " "

    remaining = _FENCED_CODE_RE.sub(_pull, text)
    remaining = _INLINE_CODE_RE.sub(_pull, remaining)

    kept_lines = []
    for line in remaining.splitlines():
        if _CODE_LINE_RE.match(line):
            code_blocks.append(line.strip())
        else:
            kept_lines.append(line)

    remaining_text = "\n".join(kept_lines).strip()
    return remaining_text, [c for c in code_blocks if c]


# ── Formatação de contexto pros prompts ─────────────────────────────────────

_ITEM_EXTRA_PROPS = (
    "value", "placeholder", "options", "checked", "selected",
    "expanded", "pressed", "url", "level", "disabled",
)


def format_items_for_prompt(items: list[dict], max_chars: int) -> str:
    if not items:
        return "(nenhum elemento extraído — a página pode estar vazia, carregando, ou em branco)"

    lines = []
    for it in items:
        parts = [f'id={it.get("id")}', str(it.get("role", "?"))]
        text = (it.get("text") or "")[:MAX_ITEM_TEXT_CHARS]
        if text:
            parts.append(f'"{text}"')
        for prop in _ITEM_EXTRA_PROPS:
            val = it.get(prop)
            if val not in (None, "", False):
                parts.append(f"{prop}={val}")
        actions = it.get("actions") or []
        parts.append(f'acoes=[{",".join(actions)}]')
        lines.append(" | ".join(parts))

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...(lista truncada por tamanho)"
    return text


def build_action_prompt(
    base_request: str, steps: list[str], history: list[str],
    items_text: str, iteration: int, max_iterations: int,
) -> str:
    steps_block = "\n".join(f"- {s}" for s in steps) if steps else "(nenhuma sugestão fornecida)"
    history_block = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(history)) if history else "(nenhuma ação executada ainda)"
    functions_block = "\n".join(f"- {name}: {desc}" for name, desc in FUNCTION_DESCRIPTIONS.items())

    return f"""Você controla um navegador para cumprir uma tarefa. Responda APENAS com o objeto JSON pedido — nenhum outro texto.

TAREFA:
{base_request}

CONTEXTO — SUGESTÕES DE COMO FAZER (não é uma ordem fixa nem uma sequência obrigatória; use como referência, pule ou reordene o que fizer sentido dado o que a página mostra agora):
{steps_block}

HISTÓRICO DO QUE JÁ FOI FEITO (rodada {iteration}/{max_iterations}):
{history_block}

ESTRUTURA ATUAL DA PÁGINA (id | role | "texto" | propriedades | ações possíveis):
{items_text}

FUNÇÕES DISPONÍVEIS:
{functions_block}

Escolha a PRÓXIMA função a executar. Use "finish" assim que a estrutura atual ou o histórico já mostrarem a tarefa cumprida — nesse caso preencha "justification" e, se houver informação útil para extrair, "useful_item_ids" com os ids correspondentes da estrutura atual (senão, lista vazia)."""


def build_confirmation_prompt(
    base_request: str, steps: list[str], history: list[str],
    justification: str, raw_content: str,
) -> str:
    steps_block = "\n".join(f"- {s}" for s in steps) if steps else "(nenhuma sugestão fornecida)"
    history_block = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(history)) if history else "(vazio)"
    content_block = raw_content.strip() or "(nenhum conteúdo foi separado como útil)"

    return f"""Você revisa se um agente de navegador realmente cumpriu a tarefa abaixo antes de encerrar. Responda APENAS com o objeto JSON pedido.

TAREFA ORIGINAL:
{base_request}

SUGESTÕES DE COMO FAZER (contexto, não ordem obrigatória):
{steps_block}

HISTÓRICO COMPLETO DE AÇÕES:
{history_block}

JUSTIFICATIVA DO AGENTE PARA ENCERRAR:
{justification}

CONTEÚDO QUE O AGENTE MARCOU COMO ÚTIL:
{content_block}

Avalie: isso realmente cumpre a tarefa original? Se sim, "valid": true e "next_steps" vazio. Se não, "valid": false, explique o motivo em "reason" e sugira em "next_steps" o que ainda falta fazer (esses passos serão adicionados como contexto para o agente continuar — não como uma ordem rígida)."""


def build_summary_prompt(text: str) -> str:
    return f"""Resuma o texto abaixo em português, com a MENOR perda de detalhe relevante possível. Responda APENAS com o objeto JSON pedido, sem nenhum outro texto.

TEXTO:
{text}"""


# ── Estado da aplicação ──────────────────────────────────────────────────

@dataclass
class AppState:
    agent: Optional[WebAgent] = None
    executor: Optional[ThreadPoolExecutor] = None
    small_llm: Optional[LlamaGrammarClient] = None
    big_llm: Optional[LlamaGrammarClient] = None


state = AppState()


def _create_agent() -> None:
    state.agent = WebAgent(
        port=BROWSER_DEBUG_PORT,
        profile_directory=BROWSER_PROFILE_DIR,
        block_resources=BROWSER_BLOCK_RESOURCES,
    )


async def _run_browser_sync(func, *args, **kwargs):
    """
    O WebAgent usa a API síncrona do Playwright, que fica presa à thread em
    que foi criada. Por isso TODA chamada a ele (criação inclusive) passa
    por este único executor de 1 thread — nunca pelo threadpool default do
    asyncio, que rodaria cada chamada numa thread diferente e quebraria o
    Playwright.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(state.executor, partial(func, *args, **kwargs))


async def _ensure_agent() -> WebAgent:
    if state.agent is None:
        await _run_browser_sync(_create_agent)
    if state.agent is None:
        raise RuntimeError("Falha ao iniciar o WebAgent (ver logs).")
    return state.agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="browser-agent")
    state.small_llm = LlamaGrammarClient(SMALL_LLM_URL, LLM_HTTP_TIMEOUT, name="small_llm")
    state.big_llm = LlamaGrammarClient(BIG_LLM_URL, LLM_HTTP_TIMEOUT, name="big_llm")

    if AUTO_START_BROWSER:
        try:
            await _ensure_agent()
            log.info("WebAgent iniciado no startup.")
        except Exception as e:
            log.error(f"Não foi possível iniciar o WebAgent no startup ({e}) — tentando de novo na 1ª tarefa.")

    yield

    try:
        if state.agent is not None:
            await _run_browser_sync(state.agent.close)
    except Exception as e:
        log.warning(f"Erro ao fechar o WebAgent no shutdown: {e}")

    if state.small_llm:
        await state.small_llm.aclose()
    if state.big_llm:
        await state.big_llm.aclose()
    if state.executor:
        state.executor.shutdown(wait=False)


app = FastAPI(title="AVA Browser Agent API", lifespan=lifespan)


# ── Execução de ações do navegador + histórico heurístico ─────────────────

async def _execute_action(agent: WebAgent, action: AgentAction, items_by_id: dict[int, dict]) -> tuple[dict, str]:
    """Executa a ação escolhida pelo modelo e devolve (resultado_bruto, descrição_heurística_em_pt)."""
    fn = action.function
    log.info(f"[ação] executando '{fn}' (item_id={action.item_id}, value={action.value!r})")

    if fn in FUNCTIONS_REQUIRING_ITEM_ID and action.item_id is None:
        log.warning(f"[ação] '{fn}' exige item_id e nenhum foi informado — ignorando.")
        return (
            {"success": False, "error": f"função '{fn}' exige item_id, e nenhum foi informado"},
            f"[erro] o modelo tentou '{fn}' sem informar item_id — ação ignorada.",
        )

    item = items_by_id.get(action.item_id) if action.item_id is not None else None
    item_text = (item.get("text") if item else None) or "?"

    try:
        if fn == "click":
            result = await _run_browser_sync(agent.click, action.item_id)
            desc = f'Cliquei em "{item_text}" (id {action.item_id}).'
        elif fn == "type":
            result = await _run_browser_sync(agent.type, action.item_id, action.value or "")
            desc = f'Digitei "{action.value}" em "{item_text}" (id {action.item_id}).'
        elif fn == "clear":
            result = await _run_browser_sync(agent.clear, action.item_id)
            desc = f'Limpei o campo "{item_text}" (id {action.item_id}).'
        elif fn == "press":
            key = action.value or "Enter"
            result = await _run_browser_sync(agent.press, action.item_id, key)
            desc = f'Apertei "{key}" em "{item_text}" (id {action.item_id}).'
        elif fn == "press_global":
            key = action.value or "Enter"
            result = await _run_browser_sync(agent.press_global, key)
            desc = f'Apertei "{key}" na página (tecla global, sem item específico).'
        elif fn == "toggle":
            result = await _run_browser_sync(agent.toggle, action.item_id)
            desc = f'Alternei "{item_text}" (id {action.item_id}).'
        elif fn == "select":
            result = await _run_browser_sync(agent.select, action.item_id, action.value or "")
            desc = f'Selecionei "{action.value}" em "{item_text}" (id {action.item_id}).'
        elif fn == "set_value":
            result = await _run_browser_sync(agent.set_value, action.item_id, action.value or "")
            desc = f'Defini o valor "{action.value}" em "{item_text}" (id {action.item_id}).'
        elif fn == "expand":
            result = await _run_browser_sync(agent.expand, action.item_id)
            desc = f'Expandi "{item_text}" (id {action.item_id}).'
        elif fn == "collapse":
            result = await _run_browser_sync(agent.collapse, action.item_id)
            desc = f'Recolhi "{item_text}" (id {action.item_id}).'
        elif fn == "submit":
            result = await _run_browser_sync(agent.submit, action.item_id)
            desc = f'Enviei "{item_text}" (id {action.item_id}).'
        elif fn == "navigate":
            result = await _run_browser_sync(agent.navigate, action.item_id)
            desc = f'Segui o link "{item_text}" (id {action.item_id}).'
        elif fn == "close_dialog":
            result = await _run_browser_sync(agent.close_dialog, action.item_id)
            desc = f'Fechei o diálogo "{item_text}" (id {action.item_id}).'
        elif fn == "wait":
            try:
                seconds = float(action.value) if action.value else 1.0
            except ValueError:
                seconds = 1.0
            await _run_browser_sync(agent.wait, seconds)
            result = {"success": True, "waited": seconds}
            desc = f"Esperei {seconds}s antes de olhar a página de novo."
        elif fn == "open":
            new_items = await _run_browser_sync(agent.open, action.value or "")
            result = {"success": True, "items_found": len(new_items)}
            desc = f'Abri a URL "{action.value}".'
        elif fn == "search":
            new_items = await _run_browser_sync(agent.search, action.value or "")
            result = {"success": True, "items_found": len(new_items)}
            desc = f'Busquei "{action.value}" no Google.'
        else:
            result = {"success": False, "error": f"função desconhecida: {fn}"}
            desc = f"[erro] função desconhecida '{fn}' — ação ignorada."
    except Exception as e:
        log.exception(f"[ação] exceção ao executar '{fn}' (item_id={action.item_id})")
        result = {"success": False, "error": str(e)}
        desc = f'Falha ao executar "{fn}" (id {action.item_id}): {e}'

    if isinstance(result, dict) and result.get("success") is False and "error" in result and fn not in ("click",):
        pass  # o texto de erro já foi embutido em `desc` acima, quando aplicável
    if isinstance(result, dict) and result.get("success") is False:
        desc += f" [FALHOU: {result.get('error')}]"

    log.info(f"[ação] resultado bruto de '{fn}': {result} — histórico: \"{desc}\"")
    return result, desc


async def _summarize_content(raw_content: str) -> tuple[str, list[str]]:
    """
    [5] do pipeline: separa código por regex, resume o texto restante com o
    próprio modelo pequeno (grammar de resumo), e devolve os dois pra serem
    remontados pelo chamador (summary + code_blocks).
    """
    text_without_code, code_blocks = extract_code_blocks(raw_content)
    log.info(f"[resumo] {len(code_blocks)} bloco(s) de código separados antes de resumir")
    if not text_without_code.strip():
        return "", code_blocks

    prompt = build_summary_prompt(text_without_code)
    try:
        raw = await state.small_llm.complete(prompt, SUMMARY_GRAMMAR, N_PREDICT_SUMMARY)
        summary = str(_parse_json_object(raw).get("summary", "")).strip()
    except Exception as e:
        log.warning(f"[resumo] falha ao resumir conteúdo útil ({e}) — devolvendo texto truncado sem resumir.")
        summary = text_without_code[:1500]

    return summary, code_blocks


# ── Loop principal: contexto → ação → (finish → confirmação) → repete ────

async def run_agent_task(req: TaskRequest) -> TaskResult:
    task_t0 = time.perf_counter()
    log.info(f"[task] iniciando: base_request=\"{req.base_request}\" steps={req.steps} "
             f"max_iterations={req.max_iterations or DEFAULT_MAX_ITERATIONS} "
             f"max_finish_attempts={req.max_finish_attempts or DEFAULT_MAX_FINISH_ATTEMPTS}")
    agent = await _ensure_agent()

    steps = list(req.steps)
    history: list[str] = []
    max_iterations = req.max_iterations or DEFAULT_MAX_ITERATIONS
    max_finish_attempts = req.max_finish_attempts or DEFAULT_MAX_FINISH_ATTEMPTS
    finish_attempts = 0
    iteration = 0
    real_actions_executed = 0  # quantas ações != 'finish' realmente rodaram no browser

    for iteration in range(1, max_iterations + 1):
        log.info(f"[task] === rodada {iteration}/{max_iterations} ===")
        items = await _run_browser_sync(agent.structure, MAX_ITEMS_PER_STRUCTURE)
        items_by_id = {it["id"]: it for it in items}
        items_text = format_items_for_prompt(items, MAX_PROMPT_CHARS_STRUCTURE)
        log.info(f"[task] estrutura da página extraída: {len(items)} itens")
        log.debug(f"[task] itens da estrutura: {items}")

        prompt = build_action_prompt(req.base_request, steps, history, items_text, iteration, max_iterations)

        try:
            raw = await state.small_llm.complete(prompt, ACTION_GRAMMAR, N_PREDICT_ACTION)
            parsed = _parse_json_object(raw)
            action = AgentAction(**parsed)
            log.info(f"[task] ação decidida pelo modelo pequeno: {parsed}")
        except Exception as e:
            log.warning(f"[task] resposta inválida do modelo de ação ({e}) — ignorando rodada.")
            history.append(f"[erro] resposta inválida do modelo de ação nesta rodada, ignorando ({e}).")
            continue

        if action.function not in ACTION_FUNCTION_NAMES:
            log.warning(f"[task] função desconhecida retornada pelo modelo: '{action.function}'")
            history.append(f"[erro] função desconhecida '{action.function}' retornada pelo modelo — ignorada.")
            continue

        if _looks_degenerate(action.value, action.justification):
            log.warning(f"[task] saída degenerada do modelo pequeno (padrão repetitivo) — ignorando rodada: "
                        f"value={action.value!r} justification={action.justification!r}")
            history.append("[erro] resposta do modelo com padrão repetitivo (ex.: '>>>>>>') — ação ignorada, tarefa continua.")
            continue

        if action.function != "finish":
            result, desc = await _execute_action(agent, action, items_by_id)
            history.append(desc)
            real_actions_executed += 1
            continue

        # --- finish: precisa justificar, e o modelo maior precisa confirmar ---
        finish_attempts += 1
        justification = (action.justification or "").strip()
        log.info(f"[task] modelo pequeno chamou 'finish' (tentativa {finish_attempts}/{max_finish_attempts}): "
                 f"justification=\"{justification}\" useful_item_ids={action.useful_item_ids}")
        if not justification:
            log.warning("[task] 'finish' sem justificativa — ignorando.")
            history.append("[erro] chamou 'finish' sem justificativa — ação ignorada, tarefa continua.")
            if finish_attempts >= max_finish_attempts:
                log.info(f"[task] abortando: {max_finish_attempts} tentativas de finish sem justificativa atingidas.")
                break
            continue

        useful_items = [items_by_id[i] for i in action.useful_item_ids if i in items_by_id]
        raw_content = "\n".join(it.get("text") or "" for it in useful_items).strip()
        log.info(f"[task] {len(useful_items)} item(ns) marcados como úteis "
                 f"({len(raw_content)} chars de conteúdo bruto): \"{_clip(raw_content)}\"")

        if real_actions_executed == 0:
            # Nenhuma ação real (open/click/type/...) rodou no browser ainda —
            # um 'finish' aqui não pode ser válido pra praticamente nenhuma
            # tarefa. Rejeita localmente sem gastar uma chamada do big_llm
            # (e sem correr o risco dele confirmar por engano, como
            # aconteceu com 'valid: true' + reason negando a conclusão).
            log.info("[task] 'finish' chamado sem nenhuma ação real executada — rejeitando localmente, sem consultar o big_llm.")
            confirmation = ConfirmationResult(
                valid=False,
                reason="nenhuma ação foi executada no navegador ainda — a tarefa não pode estar concluída.",
                next_steps=[],
            )
        else:
            confirm_prompt = build_confirmation_prompt(req.base_request, steps, history, justification, raw_content)
            try:
                confirm_raw = await state.big_llm.complete(confirm_prompt, CONFIRMATION_GRAMMAR, N_PREDICT_CONFIRM)
                confirm_parsed = _parse_json_object(confirm_raw)
                confirmation = ConfirmationResult(**confirm_parsed)
                log.info(f"[task] modelo maior confirmou: {confirm_parsed}")
            except Exception as e:
                log.warning(f"[task] resposta inválida do modelo de confirmação ({e}) — tratando 'finish' como inválido.")
                history.append(f"[erro] resposta inválida do modelo de confirmação ({e}) — tratando 'finish' como inválido.")
                confirmation = ConfirmationResult(valid=False, reason="resposta de confirmação malformada", next_steps=[])

            # Guard-rail de consistência: já vimos o big_llm devolver
            # "valid": true com uma "reason" que nega a própria conclusão
            # (ex.: "não completou a tarefa original"). A grammar garante o
            # formato, não a coerência entre os campos — então checamos aqui.
            if confirmation.valid and _NEGATION_MARKERS_RE.search(confirmation.reason.lower()):
                log.warning(f"[task] big_llm devolveu valid=true mas a reason nega a conclusão "
                            f"({confirmation.reason!r}) — tratando como inconsistente/inválido.")
                confirmation = ConfirmationResult(
                    valid=False,
                    reason=f"resposta inconsistente do revisor (valid=true, mas reason nega): {confirmation.reason}",
                    next_steps=confirmation.next_steps,
                )

        if confirmation.valid:
            summary, code_blocks = ("", [])
            if raw_content:
                summary, code_blocks = await _summarize_content(raw_content)
                log.info(f"[task] conteúdo resumido ({len(summary)} chars de resumo, "
                         f"{len(code_blocks)} bloco(s) de código preservados)")
            elapsed = time.perf_counter() - task_t0
            log.info(f"[task] CONCLUÍDA com sucesso em {elapsed:.2f}s, {iteration} rodada(s). "
                     f"reason=\"{confirmation.reason}\"")
            return TaskResult(
                success=True,
                justification=justification,
                confirmation_reason=confirmation.reason,
                useful_items=useful_items,
                summary=summary,
                code_blocks=code_blocks,
                history=history,
                iterations=iteration,
            )

        # Revisor rejeitou: as sugestões dele viram mais CONTEXTO (não uma
        # ordem obrigatória) e o loop continua normalmente.
        log.info(f"[task] finish REJEITADO pelo revisor: motivo=\"{confirmation.reason}\" "
                 f"next_steps={confirmation.next_steps}")
        history.append(
            f'[finish rejeitado] justificativa do agente: "{justification}" — motivo do revisor: "{confirmation.reason}".'
        )
        cleaned_next_steps = [s for s in (_clean_step(s) for s in confirmation.next_steps) if s]
        steps = steps + [s for s in cleaned_next_steps if s not in steps]

        if finish_attempts >= max_finish_attempts:
            log.info(f"[task] abortando: limite de {max_finish_attempts} tentativas de 'finish' rejeitadas atingido.")
            history.append(f"[abortado] limite de {max_finish_attempts} tentativas de 'finish' rejeitadas atingido.")
            break

    elapsed = time.perf_counter() - task_t0
    log.warning(f"[task] FALHOU (sem finish confirmado) após {elapsed:.2f}s, {iteration} rodada(s). "
                f"histórico completo: {history}")
    return TaskResult(
        success=False,
        justification="",
        confirmation_reason=None,
        useful_items=[],
        summary="",
        code_blocks=[],
        history=history,
        iterations=iteration,
    )


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/agent/task", response_model=TaskResult)
async def agent_task(req: TaskRequest):
    """
    Recebe o pedido base + a lista de passos sugeridos (contexto, não ordem
    fixa) e roda o loop completo até um 'finish' confirmado ou o teto de
    rodadas/tentativas ser atingido. A chamada fica bloqueada até terminar —
    é uma tarefa potencialmente longa (múltiplas rodadas de navegação real).
    """
    if state.small_llm is None or state.big_llm is None:
        raise HTTPException(status_code=503, detail="Clientes dos modelos (pequeno/maior) não inicializados.")
    log.info(f"[endpoint] POST /agent/task recebido: base_request=\"{req.base_request}\" "
             f"steps={req.steps} max_iterations={req.max_iterations} max_finish_attempts={req.max_finish_attempts}")
    try:
        result = await run_agent_task(req)
        log.info(f"[endpoint] POST /agent/task devolvendo: success={result.success} "
                 f"iterations={result.iterations} useful_items={len(result.useful_items)} "
                 f"summary_chars={len(result.summary)} code_blocks={len(result.code_blocks)}")
        log.debug(f"[endpoint] TaskResult completo: {result.model_dump()}")
        return result
    except Exception as e:
        log.exception("[endpoint] falha ao executar tarefa do agente")
        raise HTTPException(status_code=500, detail=f"Falha ao executar tarefa: {e}")


@app.get("/agent/status")
async def agent_status():
    small_ok = await state.small_llm.ping() if state.small_llm else False
    big_ok = await state.big_llm.ping() if state.big_llm else False
    status = {
        "browser_ready": state.agent is not None,
        "small_llm": {"url": SMALL_LLM_URL, "reachable": small_ok, "role": "decide cada ação (grammar de ação)"},
        "big_llm": {"url": BIG_LLM_URL, "reachable": big_ok, "role": "confirma/rejeita o finish (grammar de confirmação)"},
        "config": {
            "max_iterations_default": DEFAULT_MAX_ITERATIONS,
            "max_finish_attempts_default": DEFAULT_MAX_FINISH_ATTEMPTS,
            "max_items_per_structure": MAX_ITEMS_PER_STRUCTURE,
            "browser_debug_port": BROWSER_DEBUG_PORT,
            "browser_profile_directory": BROWSER_PROFILE_DIR,
        },
    }
    log.info(f"[endpoint] GET /agent/status → {status}")
    return status


@app.post("/agent/close")
async def agent_close():
    """Fecha o Chrome/Playwright da sessão atual (o único jeito de derrubar o browser, igual ao fechar() do browser_agent)."""
    log.info("[endpoint] POST /agent/close recebido")
    if state.agent is not None:
        await _run_browser_sync(state.agent.close)
        state.agent = None
        log.info("[endpoint] browser fechado")
    else:
        log.info("[endpoint] nenhum browser ativo para fechar")
    return {"closed": True}


@app.post("/agent/reset")
async def agent_reset():
    """Fecha (se existir) e abre uma sessão nova do browser."""
    log.info("[endpoint] POST /agent/reset recebido")
    if state.agent is not None:
        await _run_browser_sync(state.agent.close)
        state.agent = None
    await _ensure_agent()
    log.info("[endpoint] browser reiniciado")
    return {"reset": True}


# ── Entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("browser_agent_api:app", host="0.0.0.0", port=API_PORT, log_level="info")