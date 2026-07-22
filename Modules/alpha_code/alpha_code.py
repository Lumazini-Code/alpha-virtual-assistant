"""
AVA — Alpha-code (porta 4006)
============================
Servidor FastAPI do agente de código ReAct.

Endpoints:
  GET  /health                 → status + dependências
  GET  /stats                   → métricas de sessões
  GET  /tools                   → lista tools disponíveis (com schemas)
  POST /run                     → executa tarefa síncrona (JSON response)
  POST /run/stream              → executa tarefa com SSE streaming
  GET  /session/{id}/log        → eventos da sessão (JSONL)
  DELETE /session/{id}          → limpa sessão

Dependências externas:
  - LLM.py:4003                 → POST /chat/tools (Groq com tool use nativo)
  - scraping_client:3005        → files + execute + stat
  - onnxManager:2002            → /v1/embed + /v1/rerank (para semantic_search)

Variáveis de ambiente:
  ALPHA_SCRAPE_URL              → URL scraping_client (default http://localhost:3005)
  ALPHA_SCRAPE_TOKEN            → Bearer token do scraping_client (default vazio)
  ALPHA_LLM_URL                 → URL LLM.py (default http://localhost:4003)
  ALPHA_ONNX_URL                → URL onnxManager (default http://localhost:2002)
  ALPHA_PROJECT_ROOT            → raiz do projeto alvo (default /home/z/my-project)
  ALPHA_SESSION_DIR             → dir para persistir sessões
  GROQ_API_KEY                  → chave do Groq (necessária no LLM.py)
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from alpha_code.__init__ import __version__
from alpha_code.agent import Agent
from alpha_code.context import ContextManager, SessionStore, SESSION_DIR
from alpha_code.schemas import Event, EventType, TaskRequest, TaskResult
from alpha_code.tools.file_tools import close_client as close_file_client
from alpha_code.tools.file_tools import register_all as register_file_tools
from alpha_code.tools.patch_tools import register_all as register_patch_tools
from alpha_code.tools.search_tools import register_all as register_search_tools
from alpha_code.tools.symbol_lookup import register_all as register_symbol_tools
from alpha_code.tools.test_tools import register_all as register_test_tools
from alpha_code.tools.registry import ToolRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Alpha-code] %(message)s",
)
log = logging.getLogger("ava.alpha_code")

# ── Configuração ────────────────────────────────────────────────────────────

PORT = int(os.environ.get("ALPHA_PORT", "4006"))
LLM_URL = os.environ.get("ALPHA_LLM_URL", "http://localhost:4003")
SCRAPE_URL = os.environ.get("ALPHA_SCRAPE_URL", "http://localhost:3005")
ONNX_URL = os.environ.get("ALPHA_ONNX_URL", "http://localhost:2002")

# ── Global state ─────────────────────────────────────────────────────────────

_registry: Optional[ToolRegistry] = None
_active_sessions: dict[str, Agent] = {}


def _build_registry() -> ToolRegistry:
    """Constrói registry com todas as tools registradas."""
    r = ToolRegistry()
    register_file_tools(r)
    register_search_tools(r)
    register_patch_tools(r)
    register_test_tools(r)
    register_symbol_tools(r)
    return r


# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _registry
    log.info("═══════════════════════════════════════════")
    log.info(f"  Alpha-code v{__version__}")
    log.info(f"  Porta:        {PORT}")
    log.info(f"  LLM:          {LLM_URL}")
    log.info(f"  Scraping:     {SCRAPE_URL}")
    log.info(f"  ONNX:         {ONNX_URL}")
    log.info(f"  Session dir:  {SESSION_DIR}")
    log.info("═══════════════════════════════════════════")

    _registry = _build_registry()
    log.info(f"Registry inicializado: {len(_registry.names())} tools")
    for name in _registry.names():
        log.info(f"  - {name}")

    yield

    # cleanup
    log.info("Encerrando Alpha-code...")
    for agent in list(_active_sessions.values()):
        await agent.close()
    _active_sessions.clear()
    await close_file_client()


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AVA — Alpha-code",
    description="Agente de código ReAct integrado ao framework AVA",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints básicos ───────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Verifica serviço + dependências externas."""
    deps = {}
    async with httpx.AsyncClient(timeout=3.0) as c:
        # LLM
        try:
            r = await c.get(f"{LLM_URL}/health")
            deps["llm"] = "ok" if r.status_code == 200 else f"error:{r.status_code}"
        except Exception as e:
            deps["llm"] = f"down: {type(e).__name__}"
        # Scraping
        try:
            r = await c.get(f"{SCRAPE_URL}/status")
            deps["scraping"] = "ok" if r.status_code == 200 else f"error:{r.status_code}"
        except Exception as e:
            deps["scraping"] = f"down: {type(e).__name__}"
        # ONNX
        try:
            r = await c.get(f"{ONNX_URL}/health")
            deps["onnx"] = "ok" if r.status_code == 200 else f"error:{r.status_code}"
        except Exception as e:
            deps["onnx"] = f"down: {type(e).__name__}"

    all_ok = all(v == "ok" for v in deps.values())
    return {
        "service": "alpha-code",
        "version": __version__,
        "status": "ok" if all_ok else "degraded",
        "deps": deps,
        "tools_count": len(_registry.names()) if _registry else 0,
    }


@app.get("/tools")
async def list_tools():
    """Lista tools disponíveis com schemas (debug)."""
    if not _registry:
        raise HTTPException(status_code=503, detail="Registry não inicializado")
    return {
        "tools": [
            {"name": name, "schema": _registry.get(name).schema()}
            for name in _registry.names()
        ],
        "count": len(_registry.names()),
    }


@app.get("/stats")
async def stats():
    """Métricas: sessões ativas, sessões persistidas."""
    sessions_dir = SESSION_DIR
    persisted = list(sessions_dir.glob("*.state.json"))
    return {
        "active_sessions": len(_active_sessions),
        "persisted_sessions": len(persisted),
        "session_dir": str(sessions_dir),
        "tools_count": len(_registry.names()) if _registry else 0,
    }


# ── Run síncrono ─────────────────────────────────────────────────────────────

@app.post("/run", response_model=TaskResult)
async def run_task(req: TaskRequest):
    """Executa tarefa e retorna resultado final (síncrono)."""
    if not _registry:
        raise HTTPException(status_code=503, detail="Registry não inicializado")

    if req.streaming:
        raise HTTPException(
            status_code=400,
            detail="Use /run/stream para streaming=true"
        )

    session = SessionStore(session_id=req.session_id, project_dir=req.project_dir)
    ctx_mgr = ContextManager(session)
    agent = Agent(
        registry=_registry,
        session=session,
        context=ctx_mgr,
        max_steps=req.max_steps,
        model_override=req.model_override,
        temperature=req.temperature,
    )

    final_event: Optional[Event] = None
    async for ev in agent.run(req.task):
        if ev.event == EventType.FINAL:
            final_event = ev
            break

    await agent.close()
    await ctx_mgr.close()

    if final_event is None:
        return TaskResult(
            session_id=session.session_id,
            answer="Sem resposta final.",
            steps_executed=session.state.steps_executed,
            tools_called=session.state.tools_called,
            tokens_used=session.state.tokens_used,
            elapsed_ms=0,
            model_steps=session.state.model_steps,
            files_changed=session.state.files_changed,
            success=False,
        )

    d = final_event.data
    return TaskResult(
        session_id=d["session_id"],
        answer=d["answer"],
        steps_executed=d["steps_executed"],
        tools_called=d["tools_called"],
        tokens_used=d["tokens_used"],
        elapsed_ms=d["elapsed_ms"],
        model_steps=d["model_steps"],
        files_changed=d["files_changed"],
        success=d["success"],
    )


# ── Run streaming (SSE) ──────────────────────────────────────────────────────

@app.post("/run/stream")
async def run_task_stream(req: TaskRequest):
    """Executa tarefa com SSE streaming de events."""
    if not _registry:
        raise HTTPException(status_code=503, detail="Registry não inicializado")

    if not req.streaming:
        req.streaming = True

    session = SessionStore(session_id=req.session_id, project_dir=req.project_dir)
    ctx_mgr = ContextManager(session)
    agent = Agent(
        registry=_registry,
        session=session,
        context=ctx_mgr,
        max_steps=req.max_steps,
        model_override=req.model_override,
        temperature=req.temperature,
    )

    _active_sessions[session.session_id] = agent

    async def generator():
        try:
            async for ev in agent.run(req.task):
                yield ev.to_sse()
        finally:
            await agent.close()
            await ctx_mgr.close()
            _active_sessions.pop(session.session_id, None)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Session management ──────────────────────────────────────────────────────

@app.get("/session/{session_id}/log")
async def get_session_log(session_id: str):
    """Retorna eventos da sessão (JSONL)."""
    session = SessionStore(session_id=session_id)
    events = session.get_events()
    return {
        "session_id": session_id,
        "events": events,
        "count": len(events),
        "state": session.state.model_dump(),
    }


@app.get("/session/{session_id}/state")
async def get_session_state(session_id: str):
    """Retorna estado atual da sessão."""
    session = SessionStore(session_id=session_id)
    return session.state.model_dump()


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Limpa sessão (apaga JSONL + state)."""
    session = SessionStore(session_id=session_id)
    await session.clear()
    return {"cleared": True, "session_id": session_id}


@app.get("/sessions")
async def list_sessions():
    """Lista todas as sessões persistidas."""
    sessions = []
    for p in SESSION_DIR.glob("*.state.json"):
        try:
            import json
            data = json.loads(p.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id"),
                "created_at": data.get("created_at"),
                "steps_executed": data.get("steps_executed", 0),
                "tools_called": data.get("tools_called", 0),
                "files_changed_count": len(data.get("files_changed", [])),
            })
        except Exception:
            continue
    sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return {"sessions": sessions, "count": len(sessions)}


# ── Entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "alpha_code.alpha_code:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        reload=False,
    )
