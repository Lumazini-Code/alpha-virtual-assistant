"""
Alpha-code — Context Manager
===========================
SessionStore: persiste estado de sessão em JSONL append (recoverable).
TokenLedger: estima tokens usados, decide quando sumarizar.
Summarizer: pede ao LLM:4003 para sumarizar últimas N mensagens.

Não usa memory.py:3001 — esse módulo tem suas próprias necessidades
(messages OpenAI com tool_calls), que não casam bem com o chat API do memory.
Futuro: integrar via endpoint específico.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel

from alpha_code.schemas import SessionState, StepContext

log = logging.getLogger("ava.alpha_code.context")


# ── Configuração ────────────────────────────────────────────────────────────

SESSION_DIR = Path(os.environ.get(
    "ALPHA_SESSION_DIR",
    str(Path(__file__).parent.parent.parent / "memory" / "alpha_code_sessions"),
))
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# Limite default de contexto (gpt-oss-120b = 200k; reservamos 32k para resposta)
DEFAULT_CONTEXT_LIMIT = 168_000  # tokens

# Quando atingir X% do limite, dispara summarization.
# Baixado de 0.70 para 0.55: o TPM real por minuto (ver MODEL_TPM_LIMITS em
# model_router.py) é muito menor que DEFAULT_CONTEXT_LIMIT, então vale a
# pena sumarizar mais cedo em vez de deixar o contexto chegar perto do teto
# e só então descobrir que o request nem cabe na cota por minuto do modelo.
SUMMARIZE_THRESHOLD = float(os.environ.get("ALPHA_SUMMARIZE_THRESHOLD", "0.55"))

# Quantas mensagens manter após summarization (baixado de 6 para 4)
KEEP_RECENT_AFTER_SUMMARY = int(os.environ.get("ALPHA_KEEP_RECENT", "4"))

# ── Compactação leve de histórico (sem chamar o LLM) ────────────────────────
# Quantos tool results recentes ficam com o conteúdo completo; os demais são
# encolhidos (ou colapsados se forem duplicados) para economizar tokens sem
# remover mensagens da conversa (o que quebraria o protocolo tool_call_id).
COMPACT_KEEP_LAST_TOOL_RESULTS = int(os.environ.get("ALPHA_KEEP_LAST_TOOL_RESULTS", "3"))
COMPACT_TOOL_RESULT_MAX_CHARS = int(os.environ.get("ALPHA_COMPACT_MAX_CHARS", "500"))
COMPACT_MARKER = (
    "[resultado antigo compactado para economizar tokens — "
    "ver /session/{id}/log para o conteúdo completo]"
)

# LLM endpoint (LLM.py:4003) — para summarization
LLM_URL = os.environ.get("ALPHA_LLM_URL", "http://localhost:4003")


# ════════════════════════════════════════════════════════════════════════════
# SessionStore (JSONL append)
# ════════════════════════════════════════════════════════════════════════════

class SessionStore:
    """
    Persiste eventos de sessão em JSONL (1 linha por evento).

    Cada sessão tem:
      - {session_id}.jsonl   → log append de eventos (recoverable)
      - {session_id}.state.json → estado atual (overwrite)
    """

    def __init__(self, session_id: Optional[str] = None, project_dir: Optional[str] = None):
        self.session_id = session_id or f"sess-{uuid.uuid4().hex[:8]}"
        self.project_dir = project_dir
        self.state = SessionState(session_id=self.session_id, project_dir=project_dir)
        self._events_path = SESSION_DIR / f"{self.session_id}.jsonl"
        self._state_path = SESSION_DIR / f"{self.session_id}.state.json"
        self._lock = asyncio.Lock()

        # tenta carregar estado existente
        self._load_state()

    def _load_state(self) -> None:
        if self._state_path.exists():
            try:
                data = json.loads(self._state_path.read_text(encoding="utf-8"))
                self.state = SessionState(**data)
            except Exception as e:
                log.warning(f"Falha ao carregar state de {self.session_id}: {e}")

    async def _save_state(self) -> None:
        async with self._lock:
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(self.state.model_dump_json(indent=2), encoding="utf-8")
            tmp.replace(self._state_path)

    async def append_event(self, event: dict) -> None:
        """Append de evento ao JSONL."""
        async with self._lock:
            with open(self._events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    async def add_message(self, message: dict, kind: str = "unknown") -> None:
        """Adiciona message OpenAI ao state + persiste + append event."""
        self.state.messages.append(message)
        self.state.tokens_used += _estimate_tokens(message)
        await self.append_event({
            "ts": datetime.utcnow().isoformat(),
            "type": "message",
            "kind": kind,
            "message": message,
        })
        await self._save_state()

    async def record_tool_call(self, tool_name: str, args: dict, result_summary: str) -> None:
        """Persiste evento de tool call (sem incluir result completo — já está nas messages)."""
        self.state.tools_called += 1
        await self.append_event({
            "ts": datetime.utcnow().isoformat(),
            "type": "tool_call",
            "tool": tool_name,
            "args": args,
            "result_summary": result_summary,
        })
        await self._save_state()

    async def record_file_change(self, file_path: str, action: str) -> None:
        if file_path not in self.state.files_changed:
            self.state.files_changed.append(file_path)
        await self.append_event({
            "ts": datetime.utcnow().isoformat(),
            "type": "file_change",
            "file": file_path,
            "action": action,
        })
        await self._save_state()

    async def record_file_seen(self, file_path: str) -> None:
        if file_path not in self.state.files_seen:
            self.state.files_seen.append(file_path)
        await self._save_state()

    async def record_step(self, model_used: str, kind: str) -> None:
        self.state.steps_executed += 1
        self.state.model_steps.append(model_used)
        self.state.last_step_kind = kind
        await self._save_state()

    def get_events(self) -> list[dict]:
        """Lê todos os eventos (para endpoint /session/{id}/log)."""
        if not self._events_path.exists():
            return []
        events = []
        with open(self._events_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events

    async def clear(self) -> None:
        async with self._lock:
            if self._events_path.exists():
                self._events_path.unlink()
            if self._state_path.exists():
                self._state_path.unlink()
        self.state = SessionState(session_id=self.session_id, project_dir=self.project_dir)


# ════════════════════════════════════════════════════════════════════════════
# Token estimation (heuristic, no tokenizer dependency)
# ════════════════════════════════════════════════════════════════════════════

def _estimate_tokens(message: dict) -> int:
    """Heurística: 4 chars ≈ 1 token. Inclui overhead de tool_calls (~10 tokens por call)."""
    content = message.get("content") or ""
    n = len(content) // 4
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        fn = tc.get("function", {})
        args = fn.get("arguments", "")
        if isinstance(args, dict):
            args = json.dumps(args)
        n += len(args) // 4 + 10
    return max(n, 1)


def estimate_messages_tokens(messages: list[dict]) -> int:
    return sum(_estimate_tokens(m) for m in messages)


# ════════════════════════════════════════════════════════════════════════════
# Summarizer (chama LLM.py:4003 para sumarizar messages antigas)
# ════════════════════════════════════════════════════════════════════════════

class ContextManager:
    """
    Decide quando sumarizar e executa a summarization via LLM.py:4003.

    Estratégia:
      - Mantém SYSTEM + últimas KEEP_RECENT_AFTER_SUMMARY messages intactas
      - Substitui messages intermediárias por 1 message role=user com resumo
      - Resumo inclui: o que foi feito, arquivos vistos/alterados, próximos passos
    """

    def __init__(self, session: SessionStore, context_limit: int = DEFAULT_CONTEXT_LIMIT):
        self.session = session
        self.context_limit = context_limit
        self._client = httpx.AsyncClient(
            base_url=LLM_URL,
            timeout=httpx.Timeout(120.0, connect=5.0),
        )

    async def close(self) -> None:
        if not self._client.is_closed:
            await self._client.aclose()

    def tokens_used(self) -> int:
        return estimate_messages_tokens(self.session.state.messages)

    def tokens_remaining(self) -> int:
        return max(0, self.context_limit - self.tokens_used())

    def should_summarize(self) -> bool:
        return self.tokens_used() >= int(self.context_limit * SUMMARIZE_THRESHOLD)

    def build_step_context(self, kind: str = "editing") -> StepContext:
        """Constrói StepContext para o model_router baseado no estado atual."""
        last_messages = self.session.state.messages[-5:] if self.session.state.messages else []
        tool_history = [
            (m.get("tool_calls") or [{}])[0].get("function", {}).get("name", "")
            for m in last_messages
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        tool_history = [t for t in tool_history if t]

        is_debug = kind == "debugging"
        # Detecta erro na última tool result
        last_error = None
        for m in reversed(self.session.state.messages):
            if m.get("role") == "tool":
                content = m.get("content", "")
                if "error" in content.lower() or "fail" in content.lower():
                    last_error = content[:200]
                    break
            if m.get("role") == "assistant" and not m.get("tool_calls"):
                break

        return StepContext(
            step_index=self.session.state.steps_executed,
            is_planning=(kind == "planning"),
            is_debug=is_debug,
            is_final_review=(kind == "final_review"),
            is_explanation=(kind == "explaining"),
            files_touched=len(self.session.state.files_changed),
            tool_history=tool_history,
            last_error=last_error,
            tokens_remaining=self.tokens_remaining(),
        )

    async def compact_history(self) -> bool:
        """
        Compactação local (sem chamar o LLM), rodada a cada step antes de
        decidir sobre summarization/modelo. Duas coisas, sem remover
        nenhuma mensagem (removeria o pareamento tool_call_id ↔ tool result
        exigido pelo protocolo OpenAI):

          1. Dedup: se o MESMO conteúdo de tool result já apareceu antes
             (ex.: dois `list_files` seguidos no mesmo diretório), o mais
             antigo é colapsado para um marcador curto.
          2. Encolhimento: tool results grandes que não estão entre os
             últimos N (COMPACT_KEEP_LAST_TOOL_RESULTS) são truncados.

        Retorna True se alguma coisa foi alterada.
        """
        msgs = self.session.state.messages
        tool_indices = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
        if not tool_indices:
            return False

        keep_full = set(tool_indices[-COMPACT_KEEP_LAST_TOOL_RESULTS:])
        seen_content: dict[str, int] = {}
        changed = False

        for i in tool_indices:
            m = msgs[i]
            content = m.get("content") or ""

            if i in keep_full:
                seen_content.setdefault(content, i)
                continue

            already_compacted = content.startswith("[resultado antigo compactado")
            is_duplicate = content in seen_content
            is_large = len(content) > COMPACT_TOOL_RESULT_MAX_CHARS

            if already_compacted:
                continue

            if is_duplicate:
                m["content"] = COMPACT_MARKER
                changed = True
            elif is_large:
                m["content"] = content[:COMPACT_TOOL_RESULT_MAX_CHARS] + "\n" + COMPACT_MARKER
                changed = True

            seen_content.setdefault(content, i)

        if changed:
            self.session.state.tokens_used = self.tokens_used()
            await self.session._save_state()
            await self.session.append_event({
                "ts": datetime.utcnow().isoformat(),
                "type": "compaction",
                "tokens_after": self.tokens_used(),
            })
            log.info(f"Sessão {self.session.session_id}: histórico compactado. Tokens agora: {self.tokens_used()}")

        return changed

    async def emergency_summarize(self) -> bool:
        """
        Versão agressiva de maybe_summarize, usada como último recurso
        quando o Groq já recusou o request por too_large mesmo depois de
        compact_history() e maybe_summarize() normais. Ignora o threshold
        e sumariza tudo exceto a última mensagem.

        (Antes esse método era chamado em agent.py mas não existia aqui —
        qualquer fallback "too_large" duplo derrubava com AttributeError
        em vez de tentar se recuperar.)
        """
        msgs = self.session.state.messages
        system_msgs = [m for m in msgs if m.get("role") == "system"]
        non_system = [m for m in msgs if m.get("role") != "system"]

        if len(non_system) < 3:
            return False

        keep = non_system[-1:]
        to_summarize = non_system[:-1]

        log.warning(
            f"Sessão {self.session.session_id}: emergency_summarize de "
            f"{len(to_summarize)} mensagens ({self.tokens_used()} tokens)."
        )

        try:
            summary = await self._call_llm_summarize(to_summarize)
        except Exception as e:
            log.warning(f"Emergency summarization falhou: {e}")
            return False

        summary_msg = {
            "role": "user",
            "content": (
                "[CONTEXT SUMMARY EMERGENCIAL — substitui mensagens anteriores]\n\n"
                f"{summary}\n\n[FIM DO RESUMO]"
            ),
        }
        self.session.state.messages = system_msgs + [summary_msg] + keep
        self.session.state.tokens_used = self.tokens_used()
        await self.session._save_state()
        await self.session.append_event({
            "ts": datetime.utcnow().isoformat(),
            "type": "emergency_summarization",
            "summarized_count": len(to_summarize),
            "tokens_after": self.tokens_used(),
        })
        log.warning(f"Emergency summarization concluída. Tokens agora: {self.tokens_used()}")
        return True

    async def maybe_summarize(self) -> bool:
        """
        Se preciso, sumariza. Retorna True se sumarizou.
        """
        if not self.should_summarize():
            return False

        msgs = self.session.state.messages
        if len(msgs) < KEEP_RECENT_AFTER_SUMMARY + 4:
            return False  # poucas mensagens, não vale a pena

        # System message fica; intermediárias viram summary; últimas KEEP_RECENT ficam
        system_msgs = [m for m in msgs if m.get("role") == "system"]
        non_system = [m for m in msgs if m.get("role") != "system"]
        keep = non_system[-KEEP_RECENT_AFTER_SUMMARY:]
        to_summarize = non_system[:-KEEP_RECENT_AFTER_SUMMARY]

        if len(to_summarize) < 3:
            return False

        log.info(
            f"Sessão {self.session.session_id}: summarizing {len(to_summarize)} messages "
            f"({self.tokens_used()} tokens usados, limite {self.context_limit})"
        )

        try:
            summary = await self._call_llm_summarize(to_summarize)
        except Exception as e:
            log.warning(f"Summarization falhou: {e}")
            return False

        # Substitui mensagens antigas por 1 summary message
        summary_msg = {
            "role": "user",
            "content": (
                "[CONTEXT SUMMARY — substitui mensagens anteriores]\n\n"
                f"{summary}\n\n"
                "[FIM DO RESUMO — abaixo estão as mensagens recentes]"
            ),
        }

        self.session.state.messages = system_msgs + [summary_msg] + keep
        self.session.state.tokens_used = self.tokens_used()
        await self.session._save_state()
        await self.session.append_event({
            "ts": datetime.utcnow().isoformat(),
            "type": "summarization",
            "summarized_count": len(to_summarize),
            "tokens_before": self.tokens_used() + _estimate_tokens_summary(to_summarize),
            "tokens_after": self.tokens_used(),
        })
        log.info(f"Summarization concluída. Tokens agora: {self.tokens_used()}")
        return True

    async def _call_llm_summarize(self, messages: list[dict]) -> str:
        """Chama LLM.py:4003 /chat/tools para sumarizar mensagens antigas."""
        # Monta prompt de summarization
        formatted = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content") or ""
            if m.get("tool_calls"):
                tcs = m.get("tool_calls", [])
                calls = [f"{tc.get('function', {}).get('name', '')}({tc.get('function', {}).get('arguments', '')[:100]}...)"
                         for tc in tcs]
                content += f" [TOOL_CALL: {', '.join(calls)}]"
            formatted.append(f"[{role}] {content[:500]}")

        prompt = (
            "Você é um assistente de memória de um agente de código. "
            "Resuma o que foi feito até agora de forma CONCISA (máx 400 palavras):\n"
            "- Tarefas concluídas\n"
            "- Arquivos lidos/criados/editados\n"
            "- Decisões importantes\n"
            "- Erros encontrados (e se foram corrigidos)\n"
            "- Próximos passos pendentes\n\n"
            "Não inclua código, apenas descrição textual.\n\n"
            "Mensagens a resumir:\n" + "\n---\n".join(formatted)
        )

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "tools": [],
            "tool_choice": "auto",
            "model": "openai/gpt-oss-20b",  # modelo barato para summarization
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        r = await self._client.post("/chat/tools", json=payload)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message", {})
        content = msg.get("content") or ""
        if not content:
            raise RuntimeError("LLM retornou message vazia na summarization")
        return content


def _estimate_tokens_summary(messages: list[dict]) -> int:
    return estimate_messages_tokens(messages)
