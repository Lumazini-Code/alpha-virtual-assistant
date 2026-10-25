from __future__ import annotations

"""
memory_ops_files.py — Handlers de arquivos indexados (local-scraping).

/indexed-file/write (chunking + embed em batches), /read (3 modos: path
exato, path+query por chunks, busca semântica global), /check, /get,
/chunk, /delete, /delete-by-path e /list.
"""

import asyncio
import hashlib
import sqlite3
from typing import Optional

import numpy as np

from memory_config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_DIM,
    GENERAL_SCOPE,
    IF_EMBED_BATCH_SIZE,
    IF_MAX_CONTENT_SIZE,
    encode_ref,
    is_general_ref,
    log,
    ref_real_id,
)
from memory_models import (
    IndexedFileCheckResponse,
    IndexedFileEntry,
    IndexedFileReadRequest,
    IndexedFileReadResponse,
    IndexedFileWriteRequest,
    IndexedFileWriteResponse,
    MemoryToolError,
)
from memory_state import state
from memory_vector import _chunk_text

# ══════════════════════════════════════════════════════════════════════════════
# NEW: Arquivos Indexados — /indexed-file/*
# ══════════════════════════════════════════════════════════════════════════════

def _if_backend_for_file_id(file_id: int):
    """Resolve (if_db, if_index, real_id, out_id) a partir de uma id de
    arquivo — offset-aware (projetos). Id composta (offset +10^12) → escopo
    geral. Id crua → escopo ativo primeiro; não achou → sonda o geral (o
    arquivo foi indexado em outro modo). `out_id`: id de resposta (crua para
    o escopo ativo; composta quando o dono é o geral com projeto ativo)."""
    gen = state.scopes.get(GENERAL_SCOPE)
    if is_general_ref(file_id):
        real = ref_real_id(file_id)
        if gen is None:
            return None
        row = gen.if_db.get_by_id(real)
        if row is None:
            return None
        return gen.if_db, gen.if_index, row, file_id
    row = state.if_db.get_by_id(file_id)
    if row is not None:
        return state.if_db, state.if_index, row, file_id
    if gen is not None:
        row = gen.if_db.get_by_id(file_id)
        if row is not None:
            return gen.if_db, gen.if_index, row, encode_ref(GENERAL_SCOPE, file_id)
    return None


def _if_backend_for_chunk_id(chunk_id: int):
    """Mesma resolução de _if_backend_for_file_id para ids de CHUNK (sequência
    própria de cada escopo). Retorna (if_db, real_chunk_id)."""
    gen = state.scopes.get(GENERAL_SCOPE)
    if is_general_ref(chunk_id):
        if gen is None:
            return None
        return gen.if_db, ref_real_id(chunk_id)
    return state.if_db, chunk_id


def _if_backend_for_path(file_path: str):
    """Resolve (if_db, if_index, row) por caminho — escopo ativo primeiro,
    geral depois (mesmo arquivo indexado em outro modo permanece no dono)."""
    row = state.if_db.get_by_path(file_path)
    if row is not None:
        return state.if_db, state.if_index, row
    gen = state.scopes.get(GENERAL_SCOPE)
    if gen is not None:
        row = gen.if_db.get_by_path(file_path)
        if row is not None:
            return gen.if_db, gen.if_index, row
    return None

async def indexed_file_write(req: IndexedFileWriteRequest):
    """
    Armazena o conteúdo COMPLETO de um arquivo indexado.

    Fluxo:
      1. Verifica se o arquivo já está indexado (por file_path)
      2. Se existe e file_hash é igual → sem reindexação necessária
      3. Se existe e file_hash difere → remove chunks antigos, reindexa
      4. Se não existe → insere e indexa

    O conteúdo é dividido em chunks, cada chunk é embedado e
    adicionado ao FAISS para busca semântica. O hash do arquivo
    é armazenado para detectar mudanças futuras.
    """
    content = req.content
    if not content or not content.strip():
        return IndexedFileWriteResponse(stored=False, reason="content_empty")

    if len(content) > IF_MAX_CONTENT_SIZE:
        return IndexedFileWriteResponse(
            stored=False,
            reason=f"content_too_large:{len(content)}>{IF_MAX_CONTENT_SIZE}",
        )

    content_hash = hashlib.sha256(content.encode()).hexdigest()
    loop = asyncio.get_event_loop()

    # ── Check existing (NEW projetos: dono por path — ativo primeiro, geral
    # depois. Arquivo já indexado em outro modo é ATUALIZADO no dono (mesma
    # identidade de path; evita duplicar o arquivo nos dois escopos). Arquivo
    # novo → cria no ESCOPO ATIVO. ──
    resolved = _if_backend_for_path(req.file_path)
    if resolved is not None:
        if_db, if_index, existing = resolved
    else:
        if_db, if_index, existing = state.if_db, state.if_index, None
    owner_is_general = if_db is not state.if_db

    def _out_fid(real_id: int) -> int:
        # Id de resposta: composta quando o dono é o geral com projeto ativo
        return encode_ref(GENERAL_SCOPE, real_id) if owner_is_general else real_id

    if existing and not req.force_reindex:
        if existing["file_hash"] == req.file_hash and existing["content_hash"] == content_hash:
            # Arquivo inalterado — nada a fazer
            log.info(f"Indexed file unchanged: {req.file_path} (hash_match=True)")
            return IndexedFileWriteResponse(
                stored=False,
                reason="unchanged",
                file_id=_out_fid(existing["id"]),
                chunks_created=0,
                was_reindexed=False,
                hash_match=True,
            )

    # ── Remove old chunks if re-indexing ──
    if existing:
        old_chunk_ids = if_db.get_chunk_ids_by_file(existing["id"])
        if old_chunk_ids:
            await loop.run_in_executor(None, if_index.remove_ids, set(old_chunk_ids))
            if_db.delete_chunks_by_file(existing["id"])
            log.info(
                f"Removed {len(old_chunk_ids)} old chunks for: {req.file_path} "
                f"[{'geral' if owner_is_general else 'ativo'}]"
            )

    # ── Chunk the content ──
    chunks = _chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        return IndexedFileWriteResponse(stored=False, reason="chunking_failed")

    # ── Embed chunks in batches ──
    chunk_texts = [c["text"] for c in chunks]
    all_embeddings = []

    for batch_start in range(0, len(chunk_texts), IF_EMBED_BATCH_SIZE):
        batch = chunk_texts[batch_start:batch_start + IF_EMBED_BATCH_SIZE]
        try:
            batch_embs = await state.embed_engine.embed_passages(batch)
            all_embeddings.append(batch_embs)
        except Exception as e:
            log.error(f"Embedding batch failed for {req.file_path}: {e}")
            return IndexedFileWriteResponse(
                stored=False,
                reason=f"embedding_failed:{e}",
            )

    embeddings = np.vstack(all_embeddings) if all_embeddings else np.empty((0, EMBED_DIM), dtype=np.float32)

    # ── Insert or update file record ──
    was_reindexed = existing is not None
    if existing:
        file_id = existing["id"]
        if_db.update_file(
            file_id=file_id,
            content=content,
            content_hash=content_hash,
            file_hash=req.file_hash,
            size=req.size or len(content),
            modified=req.modified,
        )
    else:
        file_id = if_db.insert_file(
            file_path=req.file_path,
            file_name=req.file_name,
            extension=req.extension,
            content=content,
            content_hash=content_hash,
            file_hash=req.file_hash,
            size=req.size or len(content),
            modified=req.modified,
            source=req.source,
            confidence=req.confidence,
        )

    # ── Insert chunks into DB ──
    if_db.insert_chunks(file_id, chunks)

    # ── Get chunk IDs (just inserted) ──
    chunk_rows = if_db.get_chunks_by_file(file_id)
    chunk_ids = [row["id"] for row in chunk_rows]

    if len(chunk_ids) != len(chunks):
        log.warning(
            f"Chunk ID mismatch: expected {len(chunks)}, got {len(chunk_ids)} "
            f"for {req.file_path}"
        )

    # ── Add embeddings to FAISS in batch ──
    if len(chunk_ids) == embeddings.shape[0]:
        await loop.run_in_executor(
            None,
            if_index.add_batch,
            embeddings,
            chunk_ids,
        )
    else:
        # Fallback: add one by one if sizes don't match
        log.warning("Chunk/embedding size mismatch — adding individually")
        for emb, cid in zip(embeddings, chunk_ids):
            await loop.run_in_executor(None, if_index.add, emb, cid)

    hash_match = (
        existing is not None
        and existing["file_hash"] == req.file_hash
        and existing["content_hash"] == content_hash
    )

    log.info(
        f"Indexed file: {req.file_path} → file_id={_out_fid(file_id)} "
        f"[{'geral' if owner_is_general else 'ativo'}] "
        f"chunks={len(chunks)} reindexed={was_reindexed} "
        f"hash_match={hash_match} content_size={len(content)}"
    )

    return IndexedFileWriteResponse(
        stored=True,
        reason="ok" if not was_reindexed else "reindexed",
        file_id=_out_fid(file_id),
        chunks_created=len(chunks),
        was_reindexed=was_reindexed,
        hash_match=hash_match,
    )


async def indexed_file_read(req: IndexedFileReadRequest):
    """
    Lê arquivos indexados. Três modos, mutuamente exclusivos:

      1. `file_path` sozinho → lookup EXATO por caminho absoluto (mesma
         chave usada em `/indexed-file/write`). Não passa pelo FAISS —
         é uma busca direta no SQLite por igualdade de `file_path`, o que
         garante que dois arquivos com o mesmo nome em pastas diferentes
         nunca sejam confundidos. Retorna o arquivo INTEIRO (1 resultado).

      2. `file_path` + `query` → mesmo lookup exato por caminho, mas em vez
         de devolver o arquivo inteiro de uma vez, ranqueia as chunks DESSE
         MESMO ARQUIVO (e só dele — não compara com outros arquivos do
         índice) contra `query` e devolve até `top_k` chunks mais
         relevantes, cada uma com seu `chunk_text` e `score`.

      3. Nem um nem outro, só `query` → busca semântica original entre
         TODOS os arquivos indexados, retornando até `top_k` arquivos
         cujos chunks tiveram similaridade acima do threshold.
    """
    file_path = (req.file_path or "").strip()
    query     = (req.query or "").strip()

    # ── Modo 1/2: caminho absoluto informado ──
    if file_path:
        # NEW (projetos): dono por path — ativo primeiro, geral depois
        resolved = _if_backend_for_path(file_path)
        if resolved is None:
            return IndexedFileReadResponse(results=[], file_path=file_path, query=query or None)
        if_db, if_index, row = resolved
        scope_tag = GENERAL_SCOPE if if_db is not state.if_db else None
        encode = scope_tag is not None

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, if_db.update_access, row["id"])

        # ── Modo 2: sem query → devolve o arquivo inteiro (comportamento original) ──
        if not query:
            entry = IndexedFileEntry(
                file_id      = encode_ref(scope_tag, row["id"]) if encode else row["id"],
                file_path    = row["file_path"],
                file_name    = row["file_name"],
                extension    = row["extension"],
                content      = row["content"],           # conteúdo COMPLETO
                file_hash    = row["file_hash"],
                content_hash = row["content_hash"],
                size         = row["size"],
                modified     = row["modified"],
                score        = 1.0,                       # lookup exato — sem score de similaridade
                confidence   = round(row["confidence"], 4),
                created_at   = row["created_at"],
                access_count = row["access_count"],
                source       = row["source"],
                chunk_text   = None,
                match_type   = "exact_path",
                scope        = scope_tag,
            )
            return IndexedFileReadResponse(results=[entry], file_path=file_path)

        # ── Modo 2b: com query → top_k chunks ranqueadas, restritas a este arquivo ──
        chunk_rows = if_db.get_chunks_by_file(row["id"])
        chunk_ids  = [c["id"] for c in chunk_rows]
        if not chunk_ids:
            return IndexedFileReadResponse(results=[], file_path=file_path, query=query)

        query_emb = await state.embed_engine.embed_query_one(query)
        ranked = await loop.run_in_executor(
            None, if_index.search_subset, query_emb, chunk_ids, req.top_k
        )
        ranked = [(cid, score) for cid, score in ranked if score >= req.min_score]
        if not ranked:
            return IndexedFileReadResponse(results=[], file_path=file_path, query=query)

        chunk_by_id = {c["id"]: c for c in chunk_rows}
        full_content = row["content"] if req.include_full_content else ""
        results = []
        for cid, score in ranked:
            chunk_row = chunk_by_id.get(cid)
            if chunk_row is None:
                continue
            results.append(IndexedFileEntry(
                file_id      = encode_ref(scope_tag, row["id"]) if encode else row["id"],
                file_path    = row["file_path"],
                file_name    = row["file_name"],
                extension    = row["extension"],
                content      = full_content,              # arquivo completo — vazio se include_full_content=False
                file_hash    = row["file_hash"],
                content_hash = row["content_hash"],
                size         = row["size"],
                modified     = row["modified"],
                score        = round(score, 4),
                confidence   = round(row["confidence"], 4),
                created_at   = row["created_at"],
                access_count = row["access_count"],
                source       = row["source"],
                chunk_text   = chunk_row["chunk_text"],   # a chunk específica ranqueada
                chunk_index  = chunk_row["chunk_index"],
                char_start   = chunk_row["char_start"],
                char_end     = chunk_row["char_end"],
                chunk_id     = encode_ref(scope_tag, chunk_row["id"]) if encode else chunk_row["id"],
                match_type   = "exact_path_chunks",
                scope        = scope_tag,
            ))
        return IndexedFileReadResponse(results=results, file_path=file_path, query=query)

    # ── Modo 3: busca semântica entre todos os arquivos (NEW projetos:
    # MESCLADA — busca no escopo ativo E no geral com os mesmos embeddings,
    # fundida por score, com tag de escopo e ids compostas p/ o geral) ──
    if not query:
        raise MemoryToolError("informe 'file_path' (leitura exata) ou 'query' (busca semântica)")

    loop = asyncio.get_event_loop()
    query_emb = await state.embed_engine.embed_query_one(query)

    gen = state.scopes.get(GENERAL_SCOPE)
    backends: list[tuple] = [(state.if_db, state.if_index, None)]
    if state.active_scope != GENERAL_SCOPE and gen is not None:
        backends.append((gen.if_db, gen.if_index, GENERAL_SCOPE))

    active_scopes = [(db, idx, tag) for db, idx, tag in backends if idx.total > 0]
    if not active_scopes:
        return IndexedFileReadResponse(results=[], query=query)

    # Busca em cada escopo (mesmos embeddings, batch nativo por escopo)
    per_scope_raw = await asyncio.gather(*(
        loop.run_in_executor(None, idx.search, query_emb, req.top_k * 2)
        for _, idx, _ in active_scopes
    ))

    # Filtra por score, agrupa por file_id (por escopo — sequências de id
    # não colidem entre escopos) e funde por score
    candidates: list[tuple] = []   # (score, fid, chunk_row, file_row, scope_tag)
    for (if_db, _, scope_tag), if_raw in zip(active_scopes, per_scope_raw):
        filtered = [(cid, score) for cid, score in if_raw if score >= req.min_score]
        if not filtered:
            continue
        # Busca chunk records
        chunk_rows = if_db.get_chunks_by_ids([cid for cid, _ in filtered])
        chunk_map = {row["id"]: row for row in chunk_rows}

        # Agrupa por file_id — melhor score por arquivo
        file_best: dict[int, tuple[float, sqlite3.Row]] = {}
        for cid, score in filtered:
            chunk_row = chunk_map.get(cid)
            if chunk_row is None:
                continue
            fid = chunk_row["file_id"]
            if fid not in file_best or score > file_best[fid][0]:
                file_best[fid] = (score, chunk_row)

        if not file_best:
            continue

        # Busca file records
        file_ids = list(file_best.keys())
        file_rows = if_db._conn.execute(
            f"SELECT * FROM indexed_files WHERE id IN ({','.join('?' * len(file_ids))})",
            file_ids,
        ).fetchall()
        file_map = {row["id"]: row for row in file_rows}

        for fid, (score, chunk_row) in file_best.items():
            file_row = file_map.get(fid)
            if file_row is None:
                continue
            candidates.append((score, fid, chunk_row, file_row, scope_tag))

    if not candidates:
        return IndexedFileReadResponse(results=[], query=query)

    candidates.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, fid, chunk_row, file_row, scope_tag in candidates:
        if_db = state.if_db if scope_tag is None else state.scopes[scope_tag].if_db
        encode = scope_tag is not None
        results.append(IndexedFileEntry(
            file_id      = encode_ref(scope_tag, fid) if encode else fid,
            file_path    = file_row["file_path"],
            file_name    = file_row["file_name"],
            extension    = file_row["extension"],
            content      = file_row["content"],          # conteúdo COMPLETO
            file_hash    = file_row["file_hash"],
            content_hash = file_row["content_hash"],
            size         = file_row["size"],
            modified     = file_row["modified"],
            score        = round(score, 4),
            confidence   = round(file_row["confidence"], 4),
            created_at   = file_row["created_at"],
            access_count = file_row["access_count"],
            source       = file_row["source"],
            chunk_text   = chunk_row["chunk_text"],       # chunk que deu match
            chunk_index  = chunk_row["chunk_index"],
            char_start   = chunk_row["char_start"],
            char_end     = chunk_row["char_end"],
            scope        = scope_tag,
        ))
        loop.run_in_executor(None, if_db.update_access, fid)

    return IndexedFileReadResponse(results=results[:req.top_k], query=query)


async def indexed_file_check(file_path: str):
    """
    Verifica se um arquivo está indexado e se o hash bate.

    Usado pelo local-scraping para decidir se precisa reindexar:
      - indexed=False  → arquivo nunca foi indexado
      - hash_match=True → já indexado e inalterado
      - hash_match=False → indexado mas arquivo mudou → reindexar

    NEW (projetos): dono por path — ativo primeiro, geral depois.
    """
    if not file_path:
        raise MemoryToolError("file_path vazio")

    resolved = _if_backend_for_path(file_path)
    if resolved is None:
        return IndexedFileCheckResponse(indexed=False)
    if_db, _, row = resolved

    chunks_count = if_db.count_chunks(row["id"])

    return IndexedFileCheckResponse(
        indexed            = True,
        file_id            = encode_ref(GENERAL_SCOPE, row["id"])
                             if if_db is not state.if_db else row["id"],
        stored_file_hash   = row["file_hash"],
        stored_content_hash = row["content_hash"],
        stored_modified    = row["modified"],
        chunks_count       = chunks_count,
        hash_match         = None,  # caller compara com o hash atual
    )


async def indexed_file_get(file_id: int):
    """
    Retorna o conteúdo completo de um arquivo indexado pelo seu ID.
    NEW (projetos): aceita id composta (offset = escopo geral).
    """
    resolved = _if_backend_for_file_id(file_id)
    if resolved is None:
        raise MemoryToolError(f"Arquivo indexado #{file_id} não encontrado")
    if_db, _, row, out_id = resolved

    chunks = if_db.get_chunks_by_file(row["id"])

    return {
        "file_id":      out_id,
        "file_path":    row["file_path"],
        "file_name":    row["file_name"],
        "extension":    row["extension"],
        "content":      row["content"],
        "content_hash": row["content_hash"],
        "file_hash":    row["file_hash"],
        "size":         row["size"],
        "modified":     row["modified"],
        "source":       row["source"],
        "confidence":   row["confidence"],
        "created_at":   row["created_at"],
        "access_count": row["access_count"],
        "chunks_count": len(chunks),
    }



async def get_chunk_by_id(chunk_id: int):
    """NEW (projetos): aceita id composta de chunk (offset = escopo geral)."""
    resolved = _if_backend_for_chunk_id(chunk_id)
    if resolved is None:
        raise MemoryToolError(f"Chunk #{chunk_id} não encontrado")
    if_db, real_id = resolved
    row = if_db._conn.execute(
        "SELECT c.*, f.file_path, f.file_hash, f.content "
        "FROM indexed_file_chunks c "
        "JOIN indexed_files f ON c.file_id = f.id "
        "WHERE c.id = ?", (real_id,)
    ).fetchone()
    if row is None:
        raise MemoryToolError(f"Chunk #{chunk_id} não encontrado")
    return {
        "chunk_id": chunk_id,  # ecoa a id pedida (crua ou composta)
        "file_path": row["file_path"],
        "file_hash": row["file_hash"],
        "char_start": row["char_start"],
        "char_end": row["char_end"],
        "chunk_text": row["chunk_text"],
        "file_content": row["content"],  # conteúdo completo para validação
    }


async def indexed_file_delete(file_id: int):
    """
    Remove um arquivo indexado e todos os seus chunks (DB + vetor).
    NEW (projetos): aceita id composta (offset = escopo geral).
    """
    resolved = _if_backend_for_file_id(file_id)
    if resolved is None:
        raise MemoryToolError(f"Arquivo indexado #{file_id} não encontrado")
    if_db, if_index, row, out_id = resolved
    real_id = row["id"]

    # Remove vectors first
    chunk_ids = if_db.get_chunk_ids_by_file(real_id)
    if chunk_ids:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, if_index.remove_ids, set(chunk_ids))

    # Delete from DB (cascades to chunks)
    deleted = if_db.delete_file(real_id)

    log.info(f"Indexed file deleted: #{real_id} ({len(chunk_ids)} chunks removed)")
    return {"deleted": deleted, "file_id": out_id, "chunks_removed": len(chunk_ids)}


async def indexed_file_delete_by_path(file_path: str):
    """
    Remove um arquivo indexado pelo caminho.
    NEW (projetos): dono por path — ativo primeiro, geral depois.
    """
    if not file_path:
        raise MemoryToolError("file_path vazio")

    resolved = _if_backend_for_path(file_path)
    if resolved is None:
        return {"deleted": 0, "file_path": file_path, "message": "not indexed"}

    if_db, _, row = resolved
    out_id = encode_ref(GENERAL_SCOPE, row["id"]) if if_db is not state.if_db else row["id"]

    # Reaproveita indexed_file_delete resolvendo pelo id REAL do dono
    return await indexed_file_delete(out_id)


async def indexed_file_list():
    """
    Lista todos os arquivos indexados com metadados.
    NEW (projetos): MESCLADA — arquivos do escopo ativo (ids cruas) + os do
    geral (ids compostas + scope="general") quando um projeto está ativo.
    """
    merged = state.active_scope != GENERAL_SCOPE
    gen = state.scopes.get(GENERAL_SCOPE)
    files = []
    for if_db, tag in (
        [(state.if_db, None)]
        + ([(gen.if_db, GENERAL_SCOPE)] if merged and gen is not None else [])
    ):
        encode = tag is not None
        for row in if_db.list_files():
            files.append({
                "file_id":      encode_ref(tag, row["id"]) if encode else row["id"],
                "file_path":    row["file_path"],
                "file_name":    row["file_name"],
                "extension":    row["extension"],
                "content_hash": row["content_hash"],
                "file_hash":    row["file_hash"],
                "size":         row["size"],
                "modified":     row["modified"],
                "created_at":   row["created_at"],
                "access_count": row["access_count"],
                "scope":        tag,
            })
    return {"total": len(files), "files": files}


__all__ = [
    '_if_backend_for_chunk_id',
    '_if_backend_for_file_id',
    '_if_backend_for_path',
    'get_chunk_by_id',
    'indexed_file_check',
    'indexed_file_delete',
    'indexed_file_delete_by_path',
    'indexed_file_get',
    'indexed_file_list',
    'indexed_file_read',
    'indexed_file_write',
]
