"""
AVA KG-RAG — Microserviço FastAPI (porta 4005)
Expõe o engine como REST API no padrão dos outros microserviços do AVA.

Rotas:
  POST /query           — Pipeline completo: busca web + KG + RAG + LLM
  GET  /stats           — Status do índice e KG
  DELETE /reset         — Limpa índice e KG (cuidado!)
  GET  /health          — Health check
"""

import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engine import AVAKnowledgeEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ava.kg_rag")

engine: AVAKnowledgeEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Iniciando AVA KG-RAG Service (porta 4005)...")
    engine = AVAKnowledgeEngine(lazy_models=False)
    logger.info("Service pronto. Stats: %s", engine.stats)
    yield
    engine.close()
    logger.info("AVA KG-RAG Service encerrado.")


app = FastAPI(
    title="AVA KG-RAG Service",
    description="Knowledge-Graph-Driven RAG com pesquisa web automática para o sistema AVA",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ─────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    text: str = Field(..., description="Pergunta ou objetivo — o engine pesquisa na internet automaticamente")

class QueryResponse(BaseModel):
    answer: str
    stats:  dict

class StatsResponse(BaseModel):
    vectors:  int
    kg_nodes: int
    kg_edges: int


# ─── Rotas ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "ava-kg-rag", "version": "2.0.0"}


@app.get("/stats", response_model=StatsResponse)
async def stats():
    s = engine.stats
    return StatsResponse(
        vectors  = s["vectors"],
        kg_nodes = s["kg"]["nodes"],
        kg_edges = s["kg"]["edges"],
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Executa o pipeline KG-RAG completo:
    1. Decompõe o objetivo em sub-tópicos (LLM)
    2. Pesquisa cada sub-tópico no DuckDuckGo
    3. Extrai texto das páginas encontradas
    4. Distila triplas semânticas (LLM)
    5. Indexa chunks com embeddings ONNX
    6. Recupera contexto via FAISS + Knowledge Graph
    7. Rerank cross-encoder + síntese final (LLM)
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Campo 'text' não pode ser vazio.")
    try:
        answer = await engine.query(req.text, verbose=False)
        return QueryResponse(answer=answer, stats=engine.stats)
    except Exception as e:
        logger.error("Erro na query: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/reset")
async def reset():
    """Remove todos os dados indexados (vetores + KG). IRREVERSÍVEL."""
    import os
    from pathlib import Path
    from config import FAISS_INDEX_PATH, FAISS_META_PATH, GRAPH_DB_PATH
    global engine
    engine.close()
    for p in [FAISS_INDEX_PATH, FAISS_META_PATH, GRAPH_DB_PATH]:
        if Path(str(p)).exists():
            os.remove(str(p))
    engine = AVAKnowledgeEngine(lazy_models=False)
    return {"status": "reset_ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("deep_search:app", host="0.0.0.0", port=4005, log_level="info", reload=False)
