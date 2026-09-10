"""
AVA Memory MCP Server
======================
Substitui a antiga API REST (FastAPI, porta 3001) do módulo de memória por um
servidor MCP (stdio), consumido pelo orchestrator como a skill "memory" (ver
`MCP_SKILLS["memory"]` em orchestrator.py).

Toda a lógica de negócio (SQLite, FAISS, embeddings, decay, dedup, chunking,
dicionário visual/facial etc.) permanece intacta em `Modules/memory.py` — este
arquivo é só a camada de transporte: em vez de rotas `@app.post(...)` do
FastAPI, expõe as mesmas operações como `@mcp.tool()`.

Rodando:
    python mcp_servers/memory_server.py

O orchestrator conecta a este processo via stdio (`stdio_client` +
`ClientSession`, ver `_get_mcp_session` em orchestrator.py) — não precisa
subir nenhuma porta HTTP.
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from mcp.server.fastmcp import FastMCP

# ── Resolução de path ──────────────────────────────────────────────────────
# memory.py mora em Modules/memory.py (relativo à raiz do projeto, que é o
# diretório pai deste arquivo, já que este script fica em mcp_servers/).
# Adicionamos os dois ao sys.path para que:
#   - `import memory_core` funcione (import direto de Modules/memory.py)
#   - as importações internas de memory.py (`from onnx_client import ...`,
#     `from modules.vector_store import ...`) continuem resolvendo do mesmo
#     jeito que resolviam quando memory.py rodava como script standalone.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULES_DIR  = PROJECT_ROOT / "Modules"

for p in (str(PROJECT_ROOT), str(MODULES_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import memory as memory_core  # noqa: E402  (Modules/memory.py)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [MemoryMCP] %(message)s")
log = logging.getLogger("ava.memory.mcp")


# ── Lifespan — inicializa/encerra o estado global de memory_core ──────────

@asynccontextmanager
async def lifespan(server: FastMCP):
    await memory_core.startup()
    try:
        yield {}
    finally:
        await memory_core.shutdown()


mcp = FastMCP("ava-memory", lifespan=lifespan)


def _err(e: Exception) -> str:
    """Normaliza mensagens de erro (equivalente ao `detail` do antigo
    HTTPException) para as tools devolverem algo legível ao modelo."""
    if isinstance(e, memory_core.MemoryToolError):
        return e.detail
    return str(e)


# ══════════════════════════════════════════════════════════════════════════
# Memória de longo/curto prazo — as tools que o MODELO decide chamar
# ══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def memory_write(text: str, source: str = "chat", confidence: float = 1.0) -> dict:
    """Grava um fato/informação na memória de longo prazo (persistente,
    buscável semanticamente). Ignora textos muito curtos ou duplicados
    (exatos ou semanticamente muito similares a algo já salvo)."""
    req = memory_core.WriteRequest(text=text, source=source, confidence=confidence)
    resp = await memory_core.write_memory(req)
    return resp.model_dump()


@mcp.tool()
async def memory_read(
    query: str,
    top_k: int = memory_core.TOP_K_READ,
    min_score: float = memory_core.READ_MIN_SCORE,
    session_id: Optional[str] = None,
    strategy: str = "auto",
) -> dict:
    """Busca semântica na memória: combina memória de longo prazo, curto
    prazo (turnos recentes da sessão), conhecimento (KG-RAG) e arquivos
    indexados localmente. `strategy` pode ser "auto", "expanded" ou "dual"."""
    req = memory_core.ReadRequest(
        query=query, top_k=top_k, min_score=min_score,
        session_id=session_id, strategy=strategy,
    )
    try:
        resp = await memory_core.read_memory(req)
    except memory_core.MemoryToolError as e:
        return {"error": _err(e)}
    return resp.model_dump()


# ══════════════════════════════════════════════════════════════════════════
# Curto prazo / sessão — usadas principalmente como bookkeeping interno do
# orchestrator (não decisão do modelo), mas expostas aqui também porque toda
# a lógica REST antiga virou MCP — não há mais servidor REST por trás.
# ══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def memory_write_short_term(session_id: str, turns: list[dict]) -> dict:
    """Grava um grupo de turnos (ex.: par pergunta/resposta) na memória de
    curto prazo de uma sessão. Cada item de `turns` é
    {"role": "user"|"assistant", "content": str}."""
    req = memory_core.WriteSTRequest(
        session_id=session_id,
        turns=[memory_core.Turn(**t) for t in turns],
    )
    resp = await memory_core.write_short_term(req)
    return resp.model_dump()


@mcp.tool()
async def memory_read_short_term(session_id: str, n_pairs: int = memory_core.ST_READ_DEFAULT_PAIRS) -> dict:
    """Leitura crua (sem busca semântica) dos N grupos de turnos mais
    recentes de uma sessão — útil para montar o histórico de conversa."""
    req = memory_core.ReadSTRequest(session_id=session_id, n_pairs=n_pairs)
    try:
        resp = await memory_core.read_short_term(req)
    except memory_core.MemoryToolError as e:
        return {"error": _err(e)}
    return resp.model_dump()


@mcp.tool()
async def memory_clear_session(session_id: str) -> dict:
    """Remove todos os turnos de curto prazo de uma sessão (DB + FAISS)."""
    return await memory_core.clear_session(session_id)


@mcp.tool()
async def memory_status() -> dict:
    """Retorna estatísticas gerais de todos os subsistemas de memória
    (longo/curto prazo, cache de planos, conhecimento, arquivos indexados,
    dicionário visual, dicionário de rostos)."""
    return await memory_core.status()


# ══════════════════════════════════════════════════════════════════════════
# Arquivos indexados (efeito colateral de local-scraping)
# ══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def indexed_file_write(
    file_path: str,
    file_name: str,
    content: str,
    extension: str = "",
    file_hash: str = "",
    size: int = 0,
    modified: str = "",
    source: str = "local_scraping",
    confidence: float = 1.0,
    force_reindex: bool = False,
) -> dict:
    """Indexa (ou reindexa) o conteúdo completo de um arquivo local, dividido
    em chunks embedados para busca semântica. Detecta arquivos inalterados
    via hash para evitar reindexação desnecessária."""
    req = memory_core.IndexedFileWriteRequest(
        file_path=file_path, file_name=file_name, extension=extension,
        content=content, file_hash=file_hash, size=size, modified=modified,
        source=source, confidence=confidence, force_reindex=force_reindex,
    )
    resp = await memory_core.indexed_file_write(req)
    return resp.model_dump()


@mcp.tool()
async def indexed_file_read(
    file_path: Optional[str] = None,
    query: Optional[str] = None,
    top_k: int = 5,
    min_score: float = memory_core.IF_MIN_SCORE,
    include_full_content: bool = True,
) -> dict:
    """Lê arquivo(s) indexado(s). Três modos: `file_path` sozinho (arquivo
    inteiro, lookup exato), `file_path` + `query` (chunks só desse arquivo,
    ranqueados por similaridade), ou só `query` (busca semântica entre todos
    os arquivos indexados)."""
    req = memory_core.IndexedFileReadRequest(
        file_path=file_path, query=query, top_k=top_k,
        min_score=min_score, include_full_content=include_full_content,
    )
    try:
        resp = await memory_core.indexed_file_read(req)
    except memory_core.MemoryToolError as e:
        return {"error": _err(e)}
    return resp.model_dump()


@mcp.tool()
async def indexed_file_check(file_path: str) -> dict:
    """Verifica se um arquivo já está indexado e devolve os hashes
    armazenados, para o chamador decidir se precisa reindexar."""
    try:
        resp = await memory_core.indexed_file_check(file_path)
    except memory_core.MemoryToolError as e:
        return {"error": _err(e)}
    return resp.model_dump()


@mcp.tool()
async def indexed_file_get(file_id: int) -> dict:
    """Retorna o conteúdo completo de um arquivo indexado pelo seu ID."""
    try:
        return await memory_core.indexed_file_get(file_id)
    except memory_core.MemoryToolError as e:
        return {"error": _err(e)}


@mcp.tool()
async def indexed_file_get_chunk(chunk_id: int) -> dict:
    """Retorna uma chunk específica de um arquivo indexado pelo seu ID."""
    try:
        return await memory_core.get_chunk_by_id(chunk_id)
    except memory_core.MemoryToolError as e:
        return {"error": _err(e)}


@mcp.tool()
async def indexed_file_delete(file_id: int) -> dict:
    """Remove um arquivo indexado e todos os seus chunks (DB + FAISS)."""
    try:
        return await memory_core.indexed_file_delete(file_id)
    except memory_core.MemoryToolError as e:
        return {"error": _err(e)}


@mcp.tool()
async def indexed_file_delete_by_path(file_path: str) -> dict:
    """Remove um arquivo indexado pelo caminho absoluto."""
    try:
        return await memory_core.indexed_file_delete_by_path(file_path)
    except memory_core.MemoryToolError as e:
        return {"error": _err(e)}


@mcp.tool()
async def indexed_file_list() -> dict:
    """Lista todos os arquivos indexados com metadados (sem conteúdo)."""
    return await memory_core.indexed_file_list()


# ══════════════════════════════════════════════════════════════════════════
# Dicionário Visual (módulo vision.py — "Tradução de Objetos")
# ══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def visual_dict_write(
    concept_name: str,
    description: str,
    embedding: list[float],
    source: str = "vision_pipeline",
    confidence: float = 1.0,
    link_to_memory: bool = True,
) -> dict:
    """Grava um exemplo de conceito visual (objeto). Se o conceito ainda não
    existe, cria-o (e opcionalmente grava a descrição também na memória de
    longo prazo); se já existe, só adiciona mais um embedding de exemplo."""
    req = memory_core.VisualDictWriteRequest(
        concept_name=concept_name, description=description, embedding=embedding,
        source=source, confidence=confidence, link_to_memory=link_to_memory,
    )
    try:
        resp = await memory_core.visual_dict_write(req)
    except memory_core.MemoryToolError as e:
        return {"error": _err(e)}
    return resp.model_dump()


@mcp.tool()
async def visual_dict_read(
    embedding: list[float],
    top_k: int = memory_core.VD_TOP_K,
    min_score: float = memory_core.VD_MIN_SCORE,
) -> dict:
    """Busca o(s) conceito(s) visual(is) mais próximo(s) de um embedding de
    crop. `ambiguous=True` indica que o chamador deve perguntar ao usuário
    em vez de assumir qual candidato é o certo."""
    try:
        resp = await memory_core.visual_dict_read(
            memory_core.VisualDictReadRequest(embedding=embedding, top_k=top_k, min_score=min_score)
        )
    except memory_core.MemoryToolError as e:
        return {"error": _err(e)}
    return resp.model_dump()


@mcp.tool()
async def visual_dict_list() -> dict:
    """Lista todos os conceitos visuais cadastrados."""
    return await memory_core.visual_dict_list()


@mcp.tool()
async def visual_dict_get(concept_id: int) -> dict:
    """Retorna os detalhes de um conceito visual pelo seu ID."""
    try:
        resp = await memory_core.visual_dict_get(concept_id)
    except memory_core.MemoryToolError as e:
        return {"error": _err(e)}
    return resp.model_dump()


@mcp.tool()
async def visual_dict_delete(concept_id: int) -> dict:
    """Remove um conceito visual e todos os seus embeddings (DB + FAISS)."""
    try:
        return await memory_core.visual_dict_delete(concept_id)
    except memory_core.MemoryToolError as e:
        return {"error": _err(e)}


# ══════════════════════════════════════════════════════════════════════════
# Dicionário de Rostos (módulo vision.py — reconhecimento facial)
# ══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def face_dict_write(
    person_name: str,
    embedding: list[float],
    description: str = "",
    source: str = "vision_pipeline",
    confidence: float = 1.0,
) -> dict:
    """Grava um exemplo de rosto de uma pessoa. Se a pessoa ainda não existe,
    cria-a; se já existe, adiciona mais um embedding de exemplo (e atualiza a
    descrição, se uma nova e não-vazia for enviada)."""
    req = memory_core.FaceDictWriteRequest(
        person_name=person_name, embedding=embedding, description=description,
        source=source, confidence=confidence,
    )
    try:
        resp = await memory_core.face_dict_write(req)
    except memory_core.MemoryToolError as e:
        return {"error": _err(e)}
    return resp.model_dump()


@mcp.tool()
async def face_dict_read(
    embedding: list[float],
    top_k: int = memory_core.FD_TOP_K,
    min_score: float = memory_core.FD_MIN_SCORE,
) -> dict:
    """Busca a(s) pessoa(s) mais próxima(s) de um embedding de rosto.
    `ambiguous=True` indica que o chamador deve perguntar ao usuário."""
    try:
        resp = await memory_core.face_dict_read(
            memory_core.FaceDictReadRequest(embedding=embedding, top_k=top_k, min_score=min_score)
        )
    except memory_core.MemoryToolError as e:
        return {"error": _err(e)}
    return resp.model_dump()


@mcp.tool()
async def face_dict_list() -> dict:
    """Lista todas as pessoas cadastradas no dicionário de rostos."""
    return await memory_core.face_dict_list()


@mcp.tool()
async def face_dict_get(person_id: int) -> dict:
    """Retorna os detalhes de uma pessoa pelo seu ID."""
    try:
        resp = await memory_core.face_dict_get(person_id)
    except memory_core.MemoryToolError as e:
        return {"error": _err(e)}
    return resp.model_dump()


@mcp.tool()
async def face_dict_update(person_id: int, description: str) -> dict:
    """Edita só a descrição de uma pessoa já cadastrada."""
    try:
        resp = await memory_core.face_dict_update(
            person_id, memory_core.FaceDictUpdateRequest(description=description)
        )
    except memory_core.MemoryToolError as e:
        return {"error": _err(e)}
    return resp.model_dump()


@mcp.tool()
async def face_dict_delete(person_id: int) -> dict:
    """Remove uma pessoa e todos os seus embeddings de rosto (DB + FAISS)."""
    try:
        return await memory_core.face_dict_delete(person_id)
    except memory_core.MemoryToolError as e:
        return {"error": _err(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
