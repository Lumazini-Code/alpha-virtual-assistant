"""
AVA Memory MCP Server
=====================
Substitui a antiga API REST (FastAPI, porta 3001) do módulo de memória por um
servidor MCP, consumido tanto pelo orchestrator (skill "memory", ver
`MCP_SKILLS["memory"]` em orchestrator.py) quanto pelo LLM.py (extração
automática de memórias + recall) — os DOIS processos precisam enxergar o
MESMO estado (SQLite/FAISS), então este servidor roda como um ÚNICO processo
de longa duração, e não é mais spawnado como subprocesso por quem o consome.

Toda a lógica de negócio (SQLite, FAISS, embeddings, decay, dedup, chunking,
grafo Hebbiano/PPR, dicionário visual/facial etc.) permanece intacta em
`Modules/memory.py` — este arquivo é só a camada de transporte: em vez de
rotas `@app.post(...)` do FastAPI, expõe as mesmas operações como
`@mcp.tool()`.

Transporte:
    Por padrão sobe via streamable-http (mesma topologia de serviço de rede
    que o antigo memory.py/FastAPI tinha, só que falando MCP em vez de REST
    puro) — é isso que permite orchestrator.py E LLM.py se conectarem como
    clientes independentes ao MESMO processo/estado, em vez de cada um subir
    sua própria cópia divergente do índice FAISS. Path default "/mcp",
    igual ao que `LLM.py` (MEMORY_MCP_URL) e `orchestrator.py`
    (MEMORY_MCP_URL) já esperam.

    Para debug local/standalone (sem outro processo dependendo do mesmo
    estado), ainda dá para rodar via stdio setando MCP_TRANSPORT=stdio.

Variáveis de ambiente:
    MCP_TRANSPORT      "streamable-http" (default) | "stdio" | "sse"
    MCP_HOST           host do listener HTTP (default: "0.0.0.0")
    MCP_PORT           porta do listener HTTP (default: 3001 — mesma porta
                       do antigo serviço REST de memória)
    MCP_HTTP_PATH      path do endpoint MCP (default: "/mcp")

Rodando (produção — serviço compartilhado):
    python mcp_servers/memory_server.py

Rodando (debug standalone via stdio):
    MCP_TRANSPORT=stdio python mcp_servers/memory_server.py
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Literal, Optional

# ── Resolução de path ──────────────────────────────────────────────────────
# memory.py mora em Modules/memory.py (relativo à raiz do projeto, que é o
# diretório pai deste arquivo, já que este script fica em mcp_servers/).
# Adicionamos os dois ao sys.path para que:
#   - `import memory` funcione (import direto de Modules/memory.py)
#   - as importações internas de memory.py (`from onnx_client import ...`,
#     `from modules.vector_store import ...`, `from config import ...`)
#     continuem resolvendo do mesmo jeito que resolviam quando memory.py
#     rodava como script standalone dentro de Modules/.
#
# O chdir para Modules/ é essencial: os caminhos de dados de memory.py são
# relativos ("./memory/ava_memory.db" etc.) e os outros serviços do projeto
# também rodam com cwd=Modules (ver start.sh) — assim o servidor MCP grava/lê
# exatamente nos mesmos bancos/índices que o restante do sistema.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULES_DIR  = PROJECT_ROOT / "Modules"

for p in (str(PROJECT_ROOT), str(MODULES_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(MODULES_DIR)

import memory as memory_core  # noqa: E402  (Modules/memory.py)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [MemoryMCP] %(message)s")
log = logging.getLogger("ava.memory.mcp")


# ── Import compatível com mcp 1.x e 2.x ────────────────────────────────────
# No mcp >= 2, FastMCP foi renomeado para MCPServer (mcp.server.mcpserver).
# A API usada aqui (name/lifespan no construtor, @mcp.tool() e .run()) é
# idêntica nas duas versões, então o fallback mantém compatibilidade com
# ambientes que ainda tenham `mcp<2` (como pede o requirements.txt).
try:
    from mcp.server.mcpserver import MCPServer  # mcp >= 2
    _MCP_V2 = True
except ImportError:  # mcp < 2
    from mcp.server.fastmcp import FastMCP as MCPServer
    _MCP_V2 = False


# ── Lifespan — inicializa/encerra o estado global de memory_core ──────────

@asynccontextmanager
async def lifespan(server: MCPServer):
    await memory_core.startup()
    try:
        yield {}
    finally:
        await memory_core.shutdown()


mcp = MCPServer("ava-memory", lifespan=lifespan)


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
async def memory_write(
    text: str,
    source: str = "chat",
    confidence: float = 1.0,
    forgettable: bool = True,
    ttl_days: Optional[float] = None,
    action: Literal["create", "update"] = "create",
    memory_id: Optional[int] = None,
) -> dict:
    """Grava um fato/informação na memória de longo prazo (persistente,
    buscável semanticamente). Ignora textos muito curtos ou duplicados
    (exatos ou semanticamente muito similares a algo já salvo).

    - forgettable: False marca a memória como não-esquecível — fica de fora
      do decaimento por inatividade (mas ainda pode ser removida por
      deleção explícita). Default True (comportamento anterior).
    - ttl_days: meia-vida específica desta memória, em dias. None usa o
      default global do serviço (DECAY_HALF_LIFE_DAYS).
    - action="update": em vez de criar um fato novo, corrige uma memória já
      gravada. Se `memory_id` vier preenchido, atualiza-a diretamente; se
      vier vazio, procura a memória mais parecida semanticamente (score
      >= UPDATE_SIM_THRESHOLD) e atualiza ela — se nenhuma candidata
      suficientemente parecida existir, o write falha com
      reason="update_target_not_found" em vez de criar um fato solto.

    Quando o texto novo cai numa faixa "possível correção" de um fato já
    existente (nem duplicata clara, nem claramente novo), o write é
    recusado com reason="possible_update:<score>" e a resposta traz
    candidate_id/candidate_text/candidate_score — reenvie com
    action="update" e memory_id=candidate_id se de fato for uma correção.
    """
    req = memory_core.WriteRequest(
        text=text, source=source, confidence=confidence,
        forgettable=forgettable, ttl_days=ttl_days,
        action=action, memory_id=memory_id,
    )
    resp = await memory_core.write_memory(req)
    return resp.model_dump()


@mcp.tool()
async def memory_write_batch(items: list[dict]) -> dict:
    """Grava várias memórias de longo prazo em uma única chamada, evitando N
    round-trips quando o chamador (ex.: um extrator LLM) tem um array de
    fatos para uma mesma dupla pergunta-resposta. Cada item de `items` aceita
    os mesmos campos de `memory_write` (text obrigatório; source, confidence,
    forgettable, ttl_days, action, memory_id opcionais). Um item que falha
    (ex.: duplicata) não impede os demais de serem processados — a resposta
    traz um resultado por item, na mesma ordem de entrada, mais
    `stored_count`/`total`."""
    req = memory_core.WriteBatchRequest(
        items=[memory_core.WriteRequest(**item) for item in items]
    )
    resp = await memory_core.write_memory_batch(req)
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
    (longo/curto prazo, conhecimento, arquivos indexados, dicionário
    visual, dicionário de rostos)."""
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


# ── Transporte ──────────────────────────────────────────────────────────────
# Default: streamable-http, escutando como serviço de rede compartilhado
# (ver docstring do módulo). MCP_TRANSPORT=stdio continua disponível para
# debug standalone (ex.: `mcp dev mcp_servers/memory_server.py`).

_TRANSPORT: Literal["stdio", "sse", "streamable-http"] = os.environ.get(
    "MCP_TRANSPORT", "streamable-http"
)  # type: ignore[assignment]
_HTTP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
_HTTP_PORT = int(os.environ.get("MCP_PORT", "3001"))
_HTTP_PATH = os.environ.get("MCP_HTTP_PATH", "/mcp")

if __name__ == "__main__":
    if _TRANSPORT == "stdio":
        log.info("Memory MCP: subindo via stdio (debug standalone)")
        mcp.run(transport="stdio")
    else:
        log.info(
            f"Memory MCP: subindo via {_TRANSPORT} em "
            f"http://{_HTTP_HOST}:{_HTTP_PORT}{_HTTP_PATH} "
            f"(serviço compartilhado — orchestrator.py e LLM.py conectam aqui)"
        )
        if _MCP_V2:
            # mcp >= 2: MCPServer.run aceita host/port/streamable_http_path
            # como kwargs repassados para run_streamable_http_async.
            mcp.run(
                transport=_TRANSPORT,
                host=_HTTP_HOST,
                port=_HTTP_PORT,
                streamable_http_path=_HTTP_PATH,
            )
        else:
            # mcp < 2 (FastMCP): host/port/path são configurados via
            # settings do construtor, não kwargs de .run().
            #
            # IMPORTANTE: decidimos qual API usar ANTES de chamar .run()
            # (com base em _MCP_V2, setado onde o import foi resolvido lá
            # em cima), em vez de tentar mcp.run(host=...) e capturar
            # TypeError aqui. .run() é uma chamada BLOQUEANTE que roda pelo
            # resto da vida do processo — se o fallback rodasse dentro de
            # um `except TypeError:`, o processo ficaria "dentro" desse
            # except para sempre, e o Python encadearia (__context__)
            # QUALQUER exceção futura durante o atendimento de requests a
            # essa TypeError original, poluindo todo traceback depois
            # disso, para sempre, mesmo sem relação nenhuma com host/port
            # (foi o que gerou o "TypeError: FastMCP.run() got an
            # unexpected keyword argument 'host'" reaparecendo em erros de
            # ASGI completamente não relacionados, bem mais tarde).
            mcp.settings.host = _HTTP_HOST
            mcp.settings.port = _HTTP_PORT
            mcp.settings.streamable_http_path = _HTTP_PATH
            mcp.run(transport=_TRANSPORT)