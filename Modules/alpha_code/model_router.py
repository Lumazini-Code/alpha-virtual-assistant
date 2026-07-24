"""
Alpha-code — Model Router
========================
Policy function que decide qual CADEIA de modelos Groq tentar em cada step
(não é mais "escolhe 1 modelo" — é "escolhe a ordem de fallback"), pra
rotacionar automaticamente quando um modelo bate rate limit (RPM/RPD/TPD).

Não chama LLM nem mantém estado — é uma função Python pura baseada em
heurísticas. Quem detecta rate limit de verdade (via headers da Groq) e
decide trocar de candidato da cadeia é o agent.py, chamando o /chat/tools
do LLM.py pra cada candidato até um funcionar.

Modelos disponíveis no Groq (jul/2026):
  - openai/gpt-oss-120b     → planejamento, arquitetura, revisão (raciocínio complexo)
  - openai/gpt-oss-20b      → edição, tool calling, tarefas rápidas (low latency)
  - llama-3.3-70b-versatile → documentação, explicações, sumarização
  - llama-3.1-8b-instant    → alto volume: maior RPD/TPD de longe (14.4K/dia,
                              500K tokens/dia) — bom 2º candidato pra steps
                              de "editing" (o kind mais comum do loop)

Reasoning effort só existe nos modelos gpt-oss (Llama não suporta):
  - high   → planejamento, debugging complexo
  - medium → edição que envolve múltiplos arquivos
  - low    → edição trivial, tool calling direto
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from alpha_code.schemas import StepContext

log = logging.getLogger("ava.alpha_code.router")


# Modelos canônicos
MODEL_PLANNER = "openai/gpt-oss-120b"
MODEL_EDITOR = "openai/gpt-oss-20b"
MODEL_DOC = "llama-3.3-70b-versatile"
MODEL_FAST = "llama-3.1-8b-instant"

ALL_MODELS: tuple[str, ...] = (MODEL_PLANNER, MODEL_EDITOR, MODEL_DOC, MODEL_FAST)

# Só os modelos da família openai/gpt-oss suportam o parâmetro reasoning_effort
# na Groq. Passar isso pra um modelo Llama é rejeitado/ignorado — então
# qualquer ModelChoice para MODEL_DOC ou MODEL_FAST tem que ir com
# reasoning_effort=None, não importa o "nível" pedido pelo kind do step.
_REASONING_CAPABLE = {MODEL_PLANNER, MODEL_EDITOR}


# ════════════════════════════════════════════════════════════════════════════
# Limites reais (RPM/RPD/TPM/TPD) por modelo, direto do console da Groq
# (tier on_demand desta org, checado em 2026-07-22).
#
# ATENÇÃO: os valores anteriores (TPM=8000 igual pros 3 modelos, "chute"
# do incidente 2026-07-21) subestimavam o llama-3.3-70b (TPM real = 12000)
# e não tinham dado nenhum sobre RPD/TPD — que é onde a dor de verdade
# está: 3 dos 4 modelos têm RPD=1000/dia, então rodar uma sessão longa de
# ReAct SÓ com um deles esgota rápido. O llama-3.1-8b-instant se destaca
# com RPD=14400 e TPD=500000 — é o único com fôlego real pra alto volume,
# por isso entra primeiro na cadeia de steps de "editing" (o kind mais
# frequente do loop). Ajustável via env var caso os tiers mudem.
# ════════════════════════════════════════════════════════════════════════════

RATE_LIMITS: dict[str, dict[str, int]] = {
    MODEL_FAST:    {"rpm": 30, "rpd": 14_400, "tpm": 6_000,  "tpd": 500_000},
    MODEL_DOC:     {"rpm": 30, "rpd": 1_000,  "tpm": 12_000, "tpd": 100_000},
    MODEL_PLANNER: {"rpm": 30, "rpd": 1_000,  "tpm": 8_000,  "tpd": 200_000},
    MODEL_EDITOR:  {"rpm": 30, "rpd": 1_000,  "tpm": 8_000,  "tpd": 200_000},
}

MODEL_TPM_LIMITS: dict[str, int] = {
    MODEL_PLANNER: int(os.environ.get("ALPHA_TPM_120B", str(RATE_LIMITS[MODEL_PLANNER]["tpm"]))),
    MODEL_EDITOR: int(os.environ.get("ALPHA_TPM_20B", str(RATE_LIMITS[MODEL_EDITOR]["tpm"]))),
    MODEL_DOC: int(os.environ.get("ALPHA_TPM_70B", str(RATE_LIMITS[MODEL_DOC]["tpm"]))),
    MODEL_FAST: int(os.environ.get("ALPHA_TPM_8B", str(RATE_LIMITS[MODEL_FAST]["tpm"]))),
}


def reasoning_effort_for(model: str, level: Optional[str]) -> Optional[str]:
    """Aplica reasoning_effort só nos modelos que suportam (gpt-oss). Público
    — use isso sempre que montar um ModelChoice pra um modelo que não veio
    direto de pick_chain() (ex: troca de modelo por too_large em agent.py),
    senão a Groq rejeita a request com 'reasoning effort is not supported
    by this model' pros modelos Llama (MODEL_DOC, MODEL_FAST)."""
    return level if (level and model in _REASONING_CAPABLE) else None


# Alias interno — mantido só pra não duplicar código dentro deste arquivo.
_reasoning_for = reasoning_effort_for

# max_tokens de RESPOSTA (não de contexto) por tipo de step — steps que só
# fazem tool-calling não precisam de um teto alto; steps de revisão/
# explicação precisam de mais espaço para texto final.
_MAX_TOKENS_BY_KIND: dict[str, int] = {
    "planning": 900,
    "final_review": 700,
    "debugging": 500,
    "explaining": 1200,
    "editing": 400,
}


def max_tokens_for_kind(kind: str) -> int:
    return _MAX_TOKENS_BY_KIND.get(kind, 500)


@dataclass
class ModelChoice:
    model: str
    reasoning_effort: Optional[str] = None  # None = default do modelo
    temperature: float = 0.3
    max_tokens: Optional[int] = None  # None = usa default do LLM.py

    def __str__(self) -> str:
        return (
            f"{self.model}(reasoning={self.reasoning_effort}, "
            f"T={self.temperature}, max_tokens={self.max_tokens})"
        )


# ════════════════════════════════════════════════════════════════════════════
# Router
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# Cadeias de fallback por kind de step.
#
# Cada entrada é (modelo, reasoning_level_desejado) em ORDEM DE PREFERÊNCIA.
# O primeiro da lista é quem pick_chain() colocaria como escolha única antes
# desta mudança; os demais são pra onde o caller (agent.py) rotaciona quando
# o candidato atual bate rate limit (RPM/RPD/TPD) — troca de modelo é
# imediata e resolve na hora, ao contrário de esperar o reset do mesmo
# modelo esgotado.
#
# "editing" é o kind mais frequente do loop ReAct (a maioria dos steps), por
# isso MODEL_FAST entra logo depois do MODEL_EDITOR ali: dá um respiro de
# RPD/TPD enorme (14400/dia, 500k tokens/dia) pros steps de alto volume.
# Para planning/final_review/debugging complexo, a ordem prioriza qualidade
# de raciocínio (120B primeiro) já que esses steps são raros por sessão.
# ════════════════════════════════════════════════════════════════════════════

_TASK_CHAINS: dict[str, list[tuple[str, Optional[str]]]] = {
    "planning": [
        (MODEL_PLANNER, "high"), (MODEL_EDITOR, "medium"), (MODEL_DOC, None), (MODEL_FAST, None),
    ],
    "final_review": [
        (MODEL_PLANNER, "high"), (MODEL_EDITOR, "medium"), (MODEL_DOC, None), (MODEL_FAST, None),
    ],
    "debugging_hard": [
        (MODEL_PLANNER, "medium"), (MODEL_EDITOR, "low"), (MODEL_DOC, None), (MODEL_FAST, None),
    ],
    "debugging_simple": [
        (MODEL_EDITOR, "low"), (MODEL_FAST, None), (MODEL_DOC, None), (MODEL_PLANNER, "low"),
    ],
    "explaining": [
        (MODEL_DOC, None), (MODEL_PLANNER, None), (MODEL_EDITOR, None), (MODEL_FAST, None),
    ],
    "editing": [
        (MODEL_EDITOR, "low"), (MODEL_FAST, None), (MODEL_DOC, None), (MODEL_PLANNER, "low"),
    ],
}

# Mapeia o kind interno da cadeia pro kind usado em max_tokens_for_kind
# (debugging_hard/debugging_simple colapsam pra "debugging").
_MAX_TOKENS_KIND_ALIAS = {"debugging_hard": "debugging", "debugging_simple": "debugging"}

_TEMPERATURE_BY_KIND = {
    "planning": 0.2, "final_review": 0.2,
    "debugging_hard": 0.2, "debugging_simple": 0.2,
    "explaining": 0.4, "editing": 0.3,
}


def _select_kind(ctx: StepContext) -> str:
    if ctx.is_planning or ctx.is_final_review:
        return "planning" if ctx.is_planning else "final_review"
    if ctx.is_debug:
        # Antes: escalava com `files_touched >= 3 OR last_error` (bastava UM
        # dos dois). Como agora rotacionamos de modelo em vez de só trocar
        # de "tier" uma vez, manter a exigência das DUAS condições juntas
        # continua fazendo sentido — só escala pro modelo caro se for
        # debug realmente complicado.
        return "debugging_hard" if (ctx.files_touched >= 4 and ctx.last_error) else "debugging_simple"
    if ctx.is_explanation:
        return "explaining"
    return "editing"


def pick_chain(ctx: StepContext, override: Optional[str] = None) -> list[ModelChoice]:
    """
    Decide a cadeia de fallback (modelo + reasoning_effort + temperatura)
    para o step atual. O caller tenta chain[0]; se bater rate limit,
    tenta chain[1], e assim por diante.

    Args:
        ctx: StepContext com flags do step
        override: se não-None, ignora a cadeia e força só este modelo

    Ver _TASK_CHAINS acima para a ordem de cada kind.
    """
    if override:
        return [ModelChoice(model=override, temperature=0.3, max_tokens=max_tokens_for_kind("editing"))]

    kind = _select_kind(ctx)
    max_tok = max_tokens_for_kind(_MAX_TOKENS_KIND_ALIAS.get(kind, kind))
    temperature = _TEMPERATURE_BY_KIND[kind]

    return [
        ModelChoice(
            model=model,
            reasoning_effort=_reasoning_for(model, level),
            temperature=temperature,
            max_tokens=max_tok,
        )
        for model, level in _TASK_CHAINS[kind]
    ]


def pick_model(ctx: StepContext, override: Optional[str] = None) -> ModelChoice:
    """
    Shim de compatibilidade pro código que só pedia 1 modelo (sem rotação).
    Código novo deve preferir pick_chain(), que já traz os fallbacks
    ordenados pra rotacionar quando um modelo bate rate limit.
    """
    return pick_chain(ctx, override=override)[0]


# ════════════════════════════════════════════════════════════════════════════
# Classification helpers (decidem o "kind" do step)
# ════════════════════════════════════════════════════════════════════════════

def classify_step(
    step_index: int,
    is_first_step: bool,
    last_tool_result_error: Optional[str],
    files_changed_count: int,
    user_task_lowercase: str,
    max_steps: int,
) -> str:
    """
    Retorna kind: 'planning' | 'editing' | 'debugging' | 'explaining' | 'final_review'

    Heurísticas:
      - Primeiro step (step 0): planning (sempre)
      - Último step (>= max_steps - 3): final_review
      - Se última tool result teve erro: debugging
      - Se task pede "explique" / "documente" / "comente": explaining
      - Default: editing
    """
    if is_first_step:
        return "planning"

    # Detecta pedido de explicação
    explanation_keywords = ["explique", "explique o que", "documente", "adicione comentários",
                            "comente o código", "what does", "explain how", "write docs",
                            "gere readme", "gerar documentação"]
    if any(kw in user_task_lowercase for kw in explanation_keywords):
        return "explaining"

    # Detecta erro na última tool → debugging
    if last_tool_result_error:
        return "debugging"

    # Próximo do fim → final review
    if step_index >= max_steps - 3:
        return "final_review"

    # Default
    return "editing"
