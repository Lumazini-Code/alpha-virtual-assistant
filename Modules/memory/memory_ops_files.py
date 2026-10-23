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
    IF_EMBED_BATCH_SIZE,
    IF_MAX_CONTENT_SIZE,
    log,
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

    # ── Check existing ──
    existing = state.if_db.get_by_path(req.file_path)

    if existing and not req.force_reindex:
        if existing["file_hash"] == req.file_hash and existing["content_hash"] == content_hash:
            # Arquivo inalterado — nada a fazer
            log.info(f"Indexed file unchanged: {req.file_path} (hash_match=True)")
            return IndexedFileWriteResponse(
                stored=False,
                reason="unchanged",
                file_id=existing["id"],
                chunks_created=0,
                was_reindexed=False,
                hash_match=True,
            )

    # ── Remove old chunks if re-indexing ──
    if existing:
        old_chunk_ids = state.if_db.get_chunk_ids_by_file(existing["id"])
        if old_chunk_ids:
            await loop.run_in_executor(None, state.if_index.remove_ids, set(old_chunk_ids))
            state.if_db.delete_chunks_by_file(existing["id"])
            log.info(f"Removed {len(old_chunk_ids)} old chunks for: {req.file_path}")

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
        state.if_db.update_file(
            file_id=file_id,
            content=content,
            content_hash=content_hash,
            file_hash=req.file_hash,
            size=req.size or len(content),
            modified=req.modified,
        )
    else:
        file_id = state.if_db.insert_file(
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
    state.if_db.insert_chunks(file_id, chunks)

    # ── Get chunk IDs (just inserted) ──
    chunk_rows = state.if_db.get_chunks_by_file(file_id)
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
            state.if_index.add_batch,
            embeddings,
            chunk_ids,
        )
    else:
        # Fallback: add one by one if sizes don't match
        log.warning("Chunk/embedding size mismatch — adding individually")
        for emb, cid in zip(embeddings, chunk_ids):
            await loop.run_in_executor(None, state.if_index.add, emb, cid)

    hash_match = (
        existing is not None
        and existing["file_hash"] == req.file_hash
        and existing["content_hash"] == content_hash
    )

    log.info(
        f"Indexed file: {req.file_path} → file_id={file_id} "
        f"chunks={len(chunks)} reindexed={was_reindexed} "
        f"hash_match={hash_match} content_size={len(content)}"
    )

    return IndexedFileWriteResponse(
        stored=True,
        reason="ok" if not was_reindexed else "reindexed",
        file_id=file_id,
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
        row = state.if_db.get_by_path(file_path)
        if row is None:
            return IndexedFileReadResponse(results=[], file_path=file_path, query=query or None)

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, state.if_db.update_access, row["id"])

        # ── Modo 2: sem query → devolve o arquivo inteiro (comportamento original) ──
        if not query:
            entry = IndexedFileEntry(
                file_id      = row["id"],
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
            )
            return IndexedFileReadResponse(results=[entry], file_path=file_path)

        # ── Modo 2b: com query → top_k chunks ranqueadas, restritas a este arquivo ──
        chunk_rows = state.if_db.get_chunks_by_file(row["id"])
        chunk_ids  = [c["id"] for c in chunk_rows]
        if not chunk_ids:
            return IndexedFileReadResponse(results=[], file_path=file_path, query=query)

        query_emb = await state.embed_engine.embed_query_one(query)
        ranked = await loop.run_in_executor(
            None, state.if_index.search_subset, query_emb, chunk_ids, req.top_k
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
                file_id      = row["id"],
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
                chunk_id = chunk_row["id"],
                match_type   = "exact_path_chunks",
            ))
        return IndexedFileReadResponse(results=results, file_path=file_path, query=query)

    # ── Modo 3: busca semântica entre todos os arquivos ──
    if not query:
        raise MemoryToolError("informe 'file_path' (leitura exata) ou 'query' (busca semântica)")

    if state.if_index.total == 0:
        return IndexedFileReadResponse(results=[], query=query)

    loop = asyncio.get_event_loop()
    query_emb = await state.embed_engine.embed_query_one(query)

    if_raw = await loop.run_in_executor(
        None, state.if_index.search, query_emb, req.top_k * 2
    )

    # Filtra por score
    filtered = [(cid, score) for cid, score in if_raw if score >= req.min_score]
    if not filtered:
        return IndexedFileReadResponse(results=[], query=query)

    # Busca chunk records
    chunk_ids = [cid for cid, _ in filtered]
    chunk_rows = state.if_db.get_chunks_by_ids(chunk_ids)
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
        return IndexedFileReadResponse(results=[], query=query)

    # Busca file records
    file_ids = list(file_best.keys())
    file_rows = state.if_db._conn.execute(
        f"SELECT * FROM indexed_files WHERE id IN ({','.join('?' * len(file_ids))})",
        file_ids,
    ).fetchall()
    file_map = {row["id"]: row for row in file_rows}

    results = []
    for fid, (score, chunk_row) in sorted(file_best.items(), key=lambda x: x[1][0], reverse=True):
        file_row = file_map.get(fid)
        if file_row is None:
            continue

        results.append(IndexedFileEntry(
            file_id      = fid,
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
        ))
        loop.run_in_executor(None, state.if_db.update_access, fid)

    return IndexedFileReadResponse(results=results[:req.top_k], query=query)


async def indexed_file_check(file_path: str):
    """
    Verifica se um arquivo está indexado e se o hash bate.

    Usado pelo local-scraping para decidir se precisa reindexar:
      - indexed=False  → arquivo nunca foi indexado
      - hash_match=True → já indexado e inalterado
      - hash_match=False → indexado mas arquivo mudou → reindexar
    """
    if not file_path:
        raise MemoryToolError("file_path vazio")

    row = state.if_db.get_by_path(file_path)
    if row is None:
        return IndexedFileCheckResponse(indexed=False)

    chunks_count = state.if_db.count_chunks(row["id"])

    return IndexedFileCheckResponse(
        indexed            = True,
        file_id            = row["id"],
        stored_file_hash   = row["file_hash"],
        stored_content_hash = row["content_hash"],
        stored_modified    = row["modified"],
        chunks_count       = chunks_count,
        hash_match         = None,  # caller compara com o hash atual
    )


async def indexed_file_get(file_id: int):
    """
    Retorna o conteúdo completo de um arquivo indexado pelo seu ID.
    """
    row = state.if_db.get_by_id(file_id)
    if row is None:
        raise MemoryToolError(f"Arquivo indexado #{file_id} não encontrado")

    chunks = state.if_db.get_chunks_by_file(file_id)

    return {
        "file_id":      row["id"],
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
    row = state.if_db._conn.execute(
        "SELECT c.*, f.file_path, f.file_hash, f.content "
        "FROM indexed_file_chunks c "
        "JOIN indexed_files f ON c.file_id = f.id "
        "WHERE c.id = ?", (chunk_id,)
    ).fetchone()
    if row is None:
        raise MemoryToolError(f"Chunk #{chunk_id} não encontrado")
    return {
        "chunk_id": row["id"],
        "file_path": row["file_path"],
        "file_hash": row["file_hash"],
        "char_start": row["char_start"],
        "char_end": row["char_end"],
        "chunk_text": row["chunk_text"],
        "file_content": row["content"],  # conteúdo completo para validação
    }


async def indexed_file_delete(file_id: int):
    """
    Remove um arquivo indexado e todos os seus chunks (DB + FAISS).
    """
    row = state.if_db.get_by_id(file_id)
    if row is None:
        raise MemoryToolError(f"Arquivo indexado #{file_id} não encontrado")

    # Remove FAISS vectors first
    chunk_ids = state.if_db.get_chunk_ids_by_file(file_id)
    if chunk_ids:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, state.if_index.remove_ids, set(chunk_ids))

    # Delete from DB (cascades to chunks)
    deleted = state.if_db.delete_file(file_id)

    log.info(f"Indexed file deleted: #{file_id} ({len(chunk_ids)} chunks removed)")
    return {"deleted": deleted, "file_id": file_id, "chunks_removed": len(chunk_ids)}


async def indexed_file_delete_by_path(file_path: str):
    """
    Remove um arquivo indexado pelo caminho.
    """
    if not file_path:
        raise MemoryToolError("file_path vazio")

    row = state.if_db.get_by_path(file_path)
    if row is None:
        return {"deleted": 0, "file_path": file_path, "message": "not indexed"}

    return await indexed_file_delete(row["id"])


async def indexed_file_list():
    """
    Lista todos os arquivos indexados com metadados.
    """
    rows = state.if_db.list_files()
    files = []
    for row in rows:
        files.append({
            "file_id":      row["id"],
            "file_path":    row["file_path"],
            "file_name":    row["file_name"],
            "extension":    row["extension"],
            "content_hash": row["content_hash"],
            "file_hash":    row["file_hash"],
            "size":         row["size"],
            "modified":     row["modified"],
            "created_at":   row["created_at"],
            "access_count": row["access_count"],
        })
    return {"total": len(files), "files": files}


__all__ = [
    'get_chunk_by_id',
    'indexed_file_check',
    'indexed_file_delete',
    'indexed_file_delete_by_path',
    'indexed_file_get',
    'indexed_file_list',
    'indexed_file_read',
    'indexed_file_write',
]
