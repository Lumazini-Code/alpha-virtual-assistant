"""
AVA KG-RAG — Cliente LLM
Chama o backend de LLM do AVA como os outros microserviços.
Inclui: chat completion, geração JSON estruturada via schema.
"""

import json
import logging
from typing import Any

from config import LLM

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 12000


class LLMClient:
    """
    Wrapper sobre o backend de LLM (OpenAI-compatible REST API).
    O backend é STATELESS — cada chamada recebe o histórico completo.

    A implementação que falava com o llama-server local foi removida —
    o backend de substituição será plugado em `chat` (que sobe exceção
    até lá).
    """

    def __init__(self, base_url: str | None = None, timeout: float = 120.0):
        self._url = base_url
        self._timeout = timeout

    # ─── API pública ─────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = None,
        temperature: float = None,
        grammar: str = None,
        json_schema: dict = None,
    ) -> str:
        """
        Chamada genérica de chat completion.
        Retorna o texto gerado como string.
        """
        raise NotImplementedError(
            "LLMClient.chat sem backend (llama-server removido; aguardando substituição)."
        )

    def plan_domain(self, user_goal: str) -> dict:
        """
        Etapa 1: Planner — decompõe objetivo em sub-tópicos de pesquisa.
        Retorna dict com chave 'sub_topics': List[str].
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "Você é um Planejador de Domínio. Sua única tarefa é decompor "
                    "o objetivo fornecido em exatamente 3 a 5 sub-tópicos de pesquisa "
                    "específicos e acionáveis. Responda EXCLUSIVAMENTE em JSON válido "
                    "com a chave 'sub_topics' contendo uma lista de strings."
                ),
            },
            {
                "role": "user",
                "content": f"Objetivo: {user_goal}",
            },
        ]
        raw = self.chat(
            messages,
            temperature=LLM.planner_temp,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: tenta extrair JSON de dentro de markdown code block
            import re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Planner não retornou JSON válido: {raw!r}")

    def distill_triples(self, text_chunk: str) -> list[tuple[str, str, str]]:
        """
        Etapa 3: Distillation — extrai triplas (sujeito, relação, objeto) do texto.
        Retorna lista de tuplas.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "Você é um Extrator de Conhecimento. Analise o texto fornecido "
                    "e extraia até 10 relacionamentos semânticos relevantes. "
                    "Retorne EXCLUSIVAMENTE um array JSON de triplas: "
                    '[[\"Sujeito\", \"relação\", \"Objeto\"], ...] '
                    "Use termos concisos e específicos. Sem texto extra."
                ),
            },
            {
                "role": "user",
                "content": f"Texto:\n{text_chunk[:3000]}",  # Limite seguro por chunk
            },
        ]
        raw = self.chat(
            messages,
            temperature=0.1,
            max_tokens=512,
        )
        try:
            data = json.loads(raw)
            # Valida que é lista de listas/tuplas com 3 elementos
            return [tuple(t[:3]) for t in data if isinstance(t, (list, tuple)) and len(t) >= 3]
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("Falha ao parsear triplas: %s — raw: %r", e, raw[:200])
            return []



    def _compress_context(self, context: str) -> str:
        """
        Limita tamanho do contexto enviado ao LLM.
        Aproximação simples baseada em caracteres.
        """

        if len(context) <= MAX_CONTEXT_CHARS:
            return context

        logger.warning(
            "Contexto muito grande (%d chars). Compactando...",
            len(context)
        )

        return context[:MAX_CONTEXT_CHARS]
