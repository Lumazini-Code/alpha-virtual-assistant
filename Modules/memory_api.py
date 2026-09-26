"""
memory_rest_api.py — API REST (FastAPI) sobre a infraestrutura de memory.py
═══════════════════════════════════════════════════════════════════════════

Este módulo NÃO reimplementa nenhuma lógica de negócio. memory.py já foi
desenhado como uma camada de domínio agnóstica de transporte:
  - `startup()` / `shutdown()` inicializam/encerram o AppState global
    (SQLite + LanceDB + jobs em background) e já eram chamados pelo antigo
    `lifespan` do FastAPI, e hoje pelo MCP server.
  - cada handler (`write_memory`, `read_memory`, `indexed_file_write`, ...)
    já recebe/retorna os modelos Pydantic corretos.
  - `MemoryToolError` já é o equivalente ao antigo `HTTPException`, só que
    sem código de status HTTP — aqui é mapeado para 400.

Este arquivo só liga essa infraestrutura pronta em rotas REST convencionais,
usando os mesmos nomes de rota que o resto do projeto AVA já referencia
(vision.py → /visual-dict/*, alpha_code.py → /indexed-file/*, etc.).

Rodar:
    python memory_rest_api.py
ou:
    uvicorn memory_rest_api:app --host 127.0.0.1 --port 3000

Config via env vars (mesmo padrão MCP_HOST/MCP_PORT do memory_server.py):
    MEMORY_REST_HOST  (default: 127.0.0.1)
    MEMORY_REST_PORT  (default: 3000)
"""

from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

import memory as mem

log = logging.getLogger("memory_api")

REST_HOST = os.environ.get("MEMORY_REST_HOST", "0.0.0.0")
REST_PORT = int(os.environ.get("MEMORY_REST_PORT", "3000"))


# ── Lifespan — delega 100% para memory.startup()/shutdown() ───────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await mem.startup()
    yield
    await mem.shutdown()


app = FastAPI(
    title="AVA Memory REST API",
    description="Camada REST sobre a infraestrutura de memória do projeto AVA (memory.py).",
    version="1.0.0",
    lifespan=lifespan,
)


# ── MemoryToolError → HTTP 400, uma vez só, para todas as rotas ───────────────

@app.exception_handler(mem.MemoryToolError)
async def _memory_tool_error_handler(request: Request, exc: mem.MemoryToolError):
    return JSONResponse(status_code=400, content={"error": exc.detail})


@app.get("/")
async def root():
    return {"service": "ava-memory-rest", "status": "ok", "docs": "/docs"}


# ── Memória de longo / curto prazo ─────────────────────────────────────────────

@app.post("/write", response_model=mem.WriteResponse)
async def write(req: mem.WriteRequest):
    return await mem.write_memory(req)


@app.post("/write/batch", response_model=mem.WriteBatchResponse)
async def write_batch(req: mem.WriteBatchRequest):
    return await mem.write_memory_batch(req)


@app.post("/write-short-term", response_model=mem.WriteSTResponse)
async def write_short_term(req: mem.WriteSTRequest):
    return await mem.write_short_term(req)


@app.post("/read-short-term", response_model=mem.ReadSTResponse)
async def read_short_term(req: mem.ReadSTRequest):
    return await mem.read_short_term(req)


@app.post("/read", response_model=mem.ReadResponse)
async def read(req: mem.ReadRequest):
    return await mem.read_memory(req)


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    return await mem.clear_session(session_id)


# ── Arquivos indexados (local-scraping) ────────────────────────────────────────
# Rotas estáticas ANTES de /indexed-file/{file_id} para não colidirem no
# roteamento (FastAPI casa na ordem de declaração).

@app.post("/indexed-file/write", response_model=mem.IndexedFileWriteResponse)
async def indexed_file_write(req: mem.IndexedFileWriteRequest):
    return await mem.indexed_file_write(req)


@app.post("/indexed-file/read", response_model=mem.IndexedFileReadResponse)
async def indexed_file_read(req: mem.IndexedFileReadRequest):
    return await mem.indexed_file_read(req)


@app.get("/indexed-file/check", response_model=mem.IndexedFileCheckResponse)
async def indexed_file_check(file_path: str = Query(...)):
    return await mem.indexed_file_check(file_path)


@app.get("/indexed-file/list")
async def indexed_file_list():
    return await mem.indexed_file_list()


@app.get("/indexed-file/chunk/{chunk_id}")
async def get_chunk_by_id(chunk_id: int):
    return await mem.get_chunk_by_id(chunk_id)


@app.delete("/indexed-file/by-path")
async def indexed_file_delete_by_path(file_path: str = Query(...)):
    return await mem.indexed_file_delete_by_path(file_path)


@app.get("/indexed-file/{file_id}")
async def indexed_file_get(file_id: int):
    return await mem.indexed_file_get(file_id)


@app.delete("/indexed-file/{file_id}")
async def indexed_file_delete(file_id: int):
    return await mem.indexed_file_delete(file_id)


# ── Dicionário visual (vision.py — Tradução de Objetos) ────────────────────────

@app.post("/visual-dict/write", response_model=mem.VisualDictWriteResponse)
async def visual_dict_write(req: mem.VisualDictWriteRequest):
    return await mem.visual_dict_write(req)


@app.post("/visual-dict/read", response_model=mem.VisualDictReadResponse)
async def visual_dict_read(req: mem.VisualDictReadRequest):
    return await mem.visual_dict_read(req)


@app.get("/visual-dict/list")
async def visual_dict_list():
    return await mem.visual_dict_list()


@app.get("/visual-dict/{concept_id}")
async def visual_dict_get(concept_id: int):
    return await mem.visual_dict_get(concept_id)


@app.delete("/visual-dict/{concept_id}")
async def visual_dict_delete(concept_id: int):
    return await mem.visual_dict_delete(concept_id)


# ── Dicionário de rostos ────────────────────────────────────────────────────────

@app.post("/face-dict/write", response_model=mem.FaceDictWriteResponse)
async def face_dict_write(req: mem.FaceDictWriteRequest):
    return await mem.face_dict_write(req)


@app.post("/face-dict/read", response_model=mem.FaceDictReadResponse)
async def face_dict_read(req: mem.FaceDictReadRequest):
    return await mem.face_dict_read(req)


@app.get("/face-dict/list")
async def face_dict_list():
    return await mem.face_dict_list()


@app.get("/face-dict/{person_id}")
async def face_dict_get(person_id: int):
    return await mem.face_dict_get(person_id)


@app.patch("/face-dict/{person_id}")
async def face_dict_update(person_id: int, req: mem.FaceDictUpdateRequest):
    return await mem.face_dict_update(person_id, req)


@app.delete("/face-dict/{person_id}")
async def face_dict_delete(person_id: int):
    return await mem.face_dict_delete(person_id)


# ── Dicionário de voz ────────────────────────────────────────────────────────────

@app.post("/voice-dict/write", response_model=mem.VoiceDictWriteResponse)
async def voice_dict_write(req: mem.VoiceDictWriteRequest):
    return await mem.voice_dict_write(req)


@app.post("/voice-dict/read", response_model=mem.VoiceDictReadResponse)
async def voice_dict_read(req: mem.VoiceDictReadRequest):
    return await mem.voice_dict_read(req)


@app.get("/voice-dict/list")
async def voice_dict_list():
    return await mem.voice_dict_list()


@app.get("/voice-dict/{person_id}")
async def voice_dict_get(person_id: int):
    return await mem.voice_dict_get(person_id)


@app.patch("/voice-dict/{person_id}")
async def voice_dict_update(person_id: int, req: mem.VoiceDictUpdateRequest):
    return await mem.voice_dict_update(person_id, req)


@app.delete("/voice-dict/{person_id}")
async def voice_dict_delete(person_id: int):
    return await mem.voice_dict_delete(person_id)


# ── RAG de seleção de tools ──────────────────────────────────────────────────────

@app.post("/tools/register", response_model=mem.ToolRegisterResponse)
async def tools_register(req: mem.ToolRegisterRequest):
    return await mem.tools_register(req)


@app.post("/tools/register/batch", response_model=mem.ToolRegisterBatchResponse)
async def tools_register_batch(req: mem.ToolRegisterBatchRequest):
    return await mem.tools_register_batch(req)


@app.post("/tools/select", response_model=mem.ToolSelectResponse)
async def tools_select(req: mem.ToolSelectRequest):
    return await mem.tools_select(req)


@app.post("/tools/record-usage")
async def tools_record_usage(req: mem.ToolUsageRequest):
    return await mem.tools_record_usage(req)


@app.get("/tools/list")
async def tools_list():
    return await mem.tools_list()


@app.delete("/tools/{name}")
async def tools_delete(name: str):
    return await mem.tools_delete(name)


# ── Status ───────────────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    return await mem.status()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "memory_api:app",
        host=REST_HOST,
        port=REST_PORT,
        reload=False,
    )
