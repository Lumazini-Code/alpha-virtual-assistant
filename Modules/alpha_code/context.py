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

# Tools cujo resultado NUNCA deve ser compactado/colapsado, mesmo fora da
# janela de "últimos N" — são o mapa estrutural do projeto (o que existe e
# onde), tipicamente pequeno (list_files aqui: ~850 chars), e o agente
# precisa continuar reconsultando o caminho REAL dos arquivos em qualquer
# step, não só nos primeiros. Compactar isso foi a causa real de um
# incidente (2026-07-22): list_files sumiu do contexto após poucos steps,
# e o agente passou ~20 steps alucinando paths de outro projeto em vez de
# usar a listagem real que já tinha visto.
STRUCTURAL_TOOL_NAMES = {"list_files", "stat"}

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
            tool_name = m.get("name", "")

            # Tools estruturais (list_files/stat) nunca são compactadas —
            # ver STRUCTURAL_TOOL_NAMES acima.
            if tool_name in STRUCTURAL_TOOL_NAMES:
                seen_content.setdefault(content, i)
                continue

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

    async def compact_for_budget(self, target_tokens: int) -> bool:
        """
        Compactação agressiva que visa trazer o total de tokens de MENSAGENS
        abaixo de target_tokens. Usada quando o contexto + tool schemas
        excedem o TPM do modelo e a compactação normal + summarization
        não foram suficientes.

        Diferente de compact_history() (que preserva os últimos N tool
        results intactos), este método também trunca tool results
        recentes e conteúdo de assistant se necessário para atingir o
        orçamento.

        Nunca toca mensagens de system nem tools estruturais
        (list_files/stat) — o agente precisa do mapa real do projeto.
        """
        msgs = self.session.state.messages
        changed = False

        # 1. Roda compactação normal primeiro (dedup + encolhe antigos)
        changed = await self.compact_history() or changed

        if self.tokens_used() <= target_tokens:
            return changed

        # 2. Trunca tool results que estavam na janela "keep full"
        #    (do mais antigo pro mais recente, exceto estruturais)
        tool_indices = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
        for i in tool_indices:
            if self.tokens_used() <= target_tokens:
                break
            m = msgs[i]
            content = m.get("content") or ""
            tool_name = m.get("name", "")

            # Nunca compacta tools estruturais
            if tool_name in STRUCTURAL_TOOL_NAMES:
                continue

            if len(content) > COMPACT_TOOL_RESULT_MAX_CHARS:
                m["content"] = content[:COMPACT_TOOL_RESULT_MAX_CHARS] + "\n" + COMPACT_MARKER
                changed = True
                # Recalcula a cada iteração para parar cedo
                self.session.state.tokens_used = self.tokens_used()

        if self.tokens_used() <= target_tokens:
            if changed:
                await self.session._save_state()
            return changed

        # 3. Trunca mensagens de assistant com thinking (conteúdo longo
        #    que não é tool_call — geralmente raciocínio/explicação)
        for m in msgs:
            if self.tokens_used() <= target_tokens:
                break
            if m.get("role") == "assistant" and not m.get("tool_calls"):
                content = m.get("content") or ""
                if len(content) > 300:
                    m["content"] = (
                        content[:300]
                        + "\n[...compactado para cabir no orçamento de TPM...]"
                    )
                    changed = True
                    self.session.state.tokens_used = self.tokens_used()

        self.session.state.tokens_used = self.tokens_used()
        if changed:
            await self.session._save_state()
            await self.session.append_event({
                "ts": datetime.utcnow().isoformat(),
                "type": "budget_compaction",
                "tokens_after": self.tokens_used(),
                "target": target_tokens,
            })
            log.info(
                f"Sessão {self.session.session_id}: compactação por orçamento de TPM. "
                f"Tokens: {self.tokens_used()} (alvo: {target_tokens})"
            )

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

        # Preserva a última ocorrência de tools estruturais (list_files/
        # stat), além da última mensagem — sem isso o agente perde de vez
        # o mapa real do projeto bem na hora mais crítica (contexto já no
        # limite), e passa a alucinar paths (ver STRUCTURAL_TOOL_NAMES).
        last_structural_idx: Optional[int] = None
        for idx, m in enumerate(non_system):
            if m.get("role") == "tool" and m.get("name") in STRUCTURAL_TOOL_NAMES:
                last_structural_idx = idx

        keep_idxs = {len(non_system) - 1}
        if last_structural_idx is not None:
            keep_idxs.add(last_structural_idx)
        keep_idxs = _ensure_tool_pairing(non_system, keep_idxs)

        keep = [m for i, m in enumerate(non_system) if i in keep_idxs]
        to_summarize = [m for i, m in enumerate(non_system) if i not in keep_idxs]

        log.warning(
            f"Sessão {self.session.session_id}: emergency_summarize de "
            f"{len(to_summarize)} mensagens ({self.tokens_used()} tokens)."
        )

        try:
            summary = await self._call_llm_summarize(to_summarize, model="groq/compound-mini")
        except Exception as e:
            log.warning(
                f"Emergency summarization via LLM falhou: {e}. "
                f"Aplicando truncamento forçado (sem LLM) como último recurso "
                f"para a sessão não morrer por causa de uma falha do provider "
                f"bem no passo que deveria salvá-la."
            )
            return await self._force_truncate(system_msgs, keep, to_summarize)

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

    async def _force_truncate(
        self, system_msgs: list[dict], keep: list[dict], dropped: list[dict]
    ) -> bool:
        """
        Último recurso dos últimos recursos: descarta mensagens antigas SEM
        chamar o LLM. Só é usado quando a própria chamada de summarization
        emergencial falha (ex: Groq 400/503) — ou seja, quando confiar de
        novo no LLM para se recuperar de um problema causado pelo LLM não é
        uma boa aposta. Não produz um resumo de qualidade nenhum, só garante
        que a sessão consiga continuar em vez de morrer com
        "Context too large... Reduce task complexity or start new session."
        """
        if not dropped:
            return False

        marker_msg = {
            "role": "user",
            "content": (
                "[CONTEXT TRUNCADO À FORÇA — a summarization emergencial via "
                "LLM falhou (erro do provider), então "
                f"{len(dropped)} mensagem(ns) antiga(s) foram descartadas SEM "
                "resumo para a sessão poder continuar. O histórico completo "
                f"ainda está em /session/{self.session.session_id}/log se "
                "precisar consultar o que foi feito antes.]"
            ),
        }
        self.session.state.messages = system_msgs + [marker_msg] + keep
        self.session.state.tokens_used = self.tokens_used()
        await self.session._save_state()
        await self.session.append_event({
            "ts": datetime.utcnow().isoformat(),
            "type": "forced_truncation",
            "dropped_count": len(dropped),
            "tokens_after": self.tokens_used(),
        })
        log.warning(
            f"Sessão {self.session.session_id}: truncamento forçado (sem LLM) "
            f"aplicado. Tokens agora: {self.tokens_used()}"
        )
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
        keep_from = max(0, len(non_system) - KEEP_RECENT_AFTER_SUMMARY)
        keep_idxs = _ensure_tool_pairing(non_system, set(range(keep_from, len(non_system))))
        keep = [m for i, m in enumerate(non_system) if i in keep_idxs]
        to_summarize = [m for i, m in enumerate(non_system) if i not in keep_idxs]

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

    async def _call_llm_summarize(
        self, messages: list[dict], model: str = "openai/gpt-oss-20b"
    ) -> str:
        """Chama LLM.py:4003 /chat/tools para sumarizar mensagens antigas.

        model: por padrão o mesmo modelo barato de sempre (usado pela
        summarization rotineira via maybe_summarize). emergency_summarize
        passa "groq/compound-mini" — TPM de 70K (vs ~6800-10200 dos modelos
        de chain normais) praticamente elimina o "too_large" bem no passo
        que é o último recurso antes de matar a sessão. O trade-off é RPD
        baixo (250/dia) nesse tier, por isso NÃO é o default: usar isso
        também no caminho rotineiro (maybe_summarize, que dispara toda vez
        que a sessão passa de 55% do contexto — pode ser várias vezes por
        sessão) esgotaria a cota diária rápido demais.
        """
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
            # Sem tools/tool_choice: isto é geração de texto puro (um resumo),
            # não precisa de function calling. Antes mandava tools=[] junto
            # com tool_choice="auto" — combinação instável que é candidata
            # direta a gerar o erro do Groq "400 - Failed to call a
            # function. Please adjust your prompt" bem na chamada que é o
            # último recurso antes de desistir da sessão inteira. Além
            # disso, os sistemas groq/compound* NEM suportam tools
            # customizadas — mandar isso quebraria a chamada de vez.
            "model": model,
            "temperature": 0.2,
            "max_tokens": 1024,
        }

        if model.startswith("groq/compound"):
            # Desliga TODAS as built-in tools (web_search, code_interpreter,
            # visit_website, browser_automation, wolfram_alpha). Sem isso o
            # compound pode decidir sozinho pesquisar na web ou rodar código
            # com o conteúdo do resumo (que inclui trechos de arquivos do
            # projeto do usuário) — imprevisível e desnecessário pra uma
            # tarefa que é só "resuma esse texto".
            payload["compound_custom"] = {"tools": {"enabled_tools": []}}

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


def _ensure_tool_pairing(non_system: list[dict], idxs: set[int]) -> set[int]:
    """
    Garante que, para toda mensagem role=tool mantida em idxs, a mensagem
    assistant (com tool_calls) que a originou também fique em idxs.

    Sem isso, cortar o histórico "a partir do índice X" pode manter um
    resultado de tool ÓRFÃO — sem o tool_call correspondente antes dele.
    Isso quebra o protocolo de mensagens OpenAI/Groq (um `tool` role sem o
    `assistant.tool_calls` que o originou é inválido) e pode fazer o Groq
    recusar o request inteiro. Suspeita forte de ser a causa real de um
    incidente em produção: um Groq 400 "Failed to call a function" logo
    depois de um emergency_summarize() que alterou o histórico.
    """
    result = set(idxs)
    for i in list(idxs):
        j = i
        while j >= 0 and non_system[j].get("role") == "tool":
            result.add(j)
            j -= 1
        if j >= 0:
            result.add(j)  # a mensagem assistant que originou os tool_calls
    return result