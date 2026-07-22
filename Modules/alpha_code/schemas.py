"""
Alpha-code — Schemas Pydantic
==============================
Modelos de request/response para a API FastAPI + tipos internos do agente.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union
from pydantic import BaseModel, Field


# ════════════════════════════════════════════════════════════════════════════
# API REQUEST / RESPONSE
# ════════════════════════════════════════════════════════════════════════════

class TaskRequest(BaseModel):
    """POST /run e /run/stream"""
    task: str = Field(..., min_length=1, description="Descrição da tarefa em linguagem natural")
    session_id: Optional[str] = Field(default=None, description="ID de sessão. None = gera novo.")
    project_dir: Optional[str] = Field(default=None, description="Diretório alvo. None = BASE_DIR do scraping_client.")
    max_steps: int = Field(default=25, ge=1, le=100, description="Limite de iterações ReAct (anti-loop infinito)")
    max_tokens_per_step: int = Field(default=4096, ge=256, le=32000)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    model_override: Optional[str] = Field(
        default=None,
        description="Força modelo Groq para TODOS os steps. None = roteamento adaptativo."
    )
    streaming: bool = Field(default=False, description="True = use /run/stream")


class TaskResult(BaseModel):
    """POST /run response"""
    session_id: str
    answer: str
    steps_executed: int
    tools_called: int
    tokens_used: int
    elapsed_ms: float
    model_steps: list[str] = Field(default_factory=list, description="Modelo usado em cada step")
    files_changed: list[str] = Field(default_factory=list)
    success: bool


# ════════════════════════════════════════════════════════════════════════════
# SSE EVENT TYPES (streaming)
# ════════════════════════════════════════════════════════════════════════════

class EventType(str, Enum):
    PLAN = "plan"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MODEL_CHOICE = "model_choice"
    CONTEXT_BUDGET = "context_budget"
    ERROR = "error"
    FINAL = "final"


class Event(BaseModel):
    """Evento SSE individual"""
    event: EventType
    data: dict = Field(default_factory=dict)
    step: Optional[int] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_sse(self) -> str:
        import json
        payload = {"event": self.event.value, "data": self.data, "step": self.step, "ts": self.timestamp}
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ════════════════════════════════════════════════════════════════════════════
# TOOL CALL / RESULT (interno)
# ════════════════════════════════════════════════════════════════════════════

class ToolCall(BaseModel):
    """Chamada de tool vinda do LLM (formato OpenAI function-calling)"""
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_groq(cls, raw: dict) -> "ToolCall":
        """Converte entrada do Groq tool_calls[0] → ToolCall."""
        import json as _json
        fn = raw.get("function", {})
        args_raw = fn.get("arguments", "{}")
        try:
            if isinstance(args_raw, str):
                args = _json.loads(args_raw) if args_raw else {}
            else:
                args = args_raw or {}
        except Exception:
            args = {"_raw_arguments": args_raw}
        return cls(id=raw.get("id", ""), name=fn.get("name", ""), arguments=args)


class ToolResult(BaseModel):
    """Resultado de executar uma tool"""
    tool_call_id: str
    tool_name: str
    success: bool
    output: str = Field(default="", description="Saída principal (texto)")
    data: dict = Field(default_factory=dict, description="Dados estruturados opcionais")
    error: Optional[str] = None
    elapsed_ms: float = 0.0

    def to_tool_message(self) -> dict:
        """Converte para message role=tool a enviar de volta ao LLM."""
        import json as _json
        content = self.error if not self.success and self.error else self.output
        if not content and self.data:
            content = _json.dumps(self.data, ensure_ascii=False, default=str)
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.tool_name,
            "content": content or "(no output)",
        }


# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════

class SessionState(BaseModel):
    """Estado persistido por sessão (JSONL append)"""
    session_id: str
    project_dir: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    messages: list[dict] = Field(default_factory=list, description="Histórico de messages OpenAI")
    steps_executed: int = 0
    tools_called: int = 0
    tokens_used: int = 0
    files_changed: list[str] = Field(default_factory=list)
    files_seen: list[str] = Field(default_factory=list)
    model_steps: list[str] = Field(default_factory=list)
    plan: Optional[list[str]] = None
    last_step_kind: Optional[str] = None  # planning | editing | debugging | explaining


# ════════════════════════════════════════════════════════════════════════════
# STEP CONTEXT (para model_router)
# ════════════════════════════════════════════════════════════════════════════

class StepContext(BaseModel):
    """Contexto do step atual — alimenta o model_router."""
    step_index: int = 0
    is_planning: bool = False
    is_debug: bool = False
    is_final_review: bool = False
    is_explanation: bool = False
    files_touched: int = 0
    tool_history: list[str] = Field(default_factory=list)
    last_error: Optional[str] = None
    tokens_remaining: int = 100_000
