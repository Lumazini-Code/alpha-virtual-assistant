"""
Alpha-code — Model Router
========================
Policy function que decide qual modelo Groq usar em cada step.

Não chama LLM — é uma função Python simples baseada em heurísticas.

Modelos disponíveis no Groq (set 2025):
  - openai/gpt-oss-120b   → planejamento, arquitetura, revisão (raciocínio complexo)
  - openai/gpt-oss-20b    → edição, tool calling, tarefas rápidas (low latency)
  - llama-3.3-70b-versatile → documentação, explicações, sumarização

Reasoning effort também é controlado:
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


# ════════════════════════════════════════════════════════════════════════════
# Limites reais de TPM (tokens-per-minute) por modelo, nesta organização/tier.
#
# ATENÇÃO: os valores anteriores (20b=8k, 120b=30k, 70b=15k, embutidos como
# comentário/estimativa no agent.py) eram um "chute" baseado em documentação
# genérica do Groq, não no limite real observado. Na prática (incidente
# 2026-07-21), o tier `on_demand` desta org tem TPM=8000 tanto para
# gpt-oss-120b quanto para gpt-oss-20b — ou seja, "promover" para o modelo
# maior quando o contexto estoura NÃO dá mais fôlego nenhum, só troca de
# fila. Ajustável via env var caso a Groq libere tiers diferentes por
# modelo para esta org no futuro.
# ════════════════════════════════════════════════════════════════════════════

MODEL_TPM_LIMITS: dict[str, int] = {
    MODEL_PLANNER: int(os.environ.get("ALPHA_TPM_120B", "8000")),
    MODEL_EDITOR: int(os.environ.get("ALPHA_TPM_20B", "8000")),
    MODEL_DOC: int(os.environ.get("ALPHA_TPM_70B", "8000")),
}

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

def pick_model(ctx: StepContext, override: Optional[str] = None) -> ModelChoice:
    """
    Decide modelo + reasoning_effort + temperatura para o step atual.

    Args:
        ctx: StepContext com flags do step
        override: se não-None, sempre usa este modelo (sem reasoning custom)

    Regras (em ordem de prioridade):
      1. Planejamento        → 120B reasoning=high
      2. Revisão final       → 120B reasoning=high
      3. Debug complexo      → 120B reasoning=medium  (>=3 arquivos OU tem erro anterior)
      4. Explicação/docs     → Llama 3.3 70B
      5. Edição simples      → 20B  reasoning=low
      6. Default             → 20B
    """
    if override:
        return ModelChoice(model=override, temperature=0.3, max_tokens=max_tokens_for_kind("editing"))

    # 1. Planejamento ou revisão final
    if ctx.is_planning or ctx.is_final_review:
        kind = "planning" if ctx.is_planning else "final_review"
        return ModelChoice(
            model=MODEL_PLANNER,
            reasoning_effort="high",
            temperature=0.2,
            max_tokens=max_tokens_for_kind(kind),
        )

    # 2. Debug complexo
    # Antes: escalava para o modelo grande com `files_touched >= 3 OR last_error`
    # (bastava UM dos dois). Como o modelo grande não tem mais TPM de verdade
    # que o pequeno nesta org (ver MODEL_TPM_LIMITS acima), escalar sem
    # necessidade só consome o mesmo orçamento de tokens em um modelo mais
    # caro/lento, sem ganho real. Agora exige as DUAS condições juntas.
    if ctx.is_debug:
        if ctx.files_touched >= 4 and ctx.last_error:
            return ModelChoice(
                model=MODEL_PLANNER,
                reasoning_effort="medium",
                temperature=0.2,
                max_tokens=max_tokens_for_kind("debugging"),
            )
        # debug simples — fica no modelo barato
        return ModelChoice(
            model=MODEL_EDITOR,
            reasoning_effort="low",
            temperature=0.2,
            max_tokens=max_tokens_for_kind("debugging"),
        )

    # 3. Explicação / docs
    if ctx.is_explanation:
        return ModelChoice(
            model=MODEL_DOC,
            reasoning_effort=None,  # Llama 3.3 não suporta reasoning_effort
            temperature=0.4,
            max_tokens=max_tokens_for_kind("explaining"),
        )

    # 4. Default: edição rápida
    return ModelChoice(
        model=MODEL_EDITOR,
        reasoning_effort="low",
        temperature=0.3,
        max_tokens=max_tokens_for_kind("editing"),
    )


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
