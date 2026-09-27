from __future__ import annotations

"""
memory_ops_dicts.py — Handlers dos dicionários multimodais.

Dicionário Visual (conceitos/objetos — DINOv3), de Rostos (EdgeFace) e de
Voz (CAM++/3D-Speaker): write/read/list/get/update/delete. Todos os writes
normalizam o embedding, deduplicam por chave normalizada e podem replicar a
descrição na memória de longo prazo (_store_long_term_text) — o que dá ao
conceito/pessoa um nó no grafo Hebbiano.

Locks de escrita (_vd/_fd/_vo_write_lock) vivem aqui — mesmo padrão do
comentário em memory_ops_memory.
"""

import asyncio
import sqlite3
from typing import Optional

import numpy as np

from memory_config import (
    FD_AMBIGUOUS_MARGIN,
    FD_EMBED_DIM,
    VD_AMBIGUOUS_MARGIN,
    VD_EMBED_DIM,
    VO_AMBIGUOUS_MARGIN,
    VO_EMBED_DIM,
    log,
)
from memory_models import (
    FaceCandidate,
    FaceDictEntry,
    FaceDictReadRequest,
    FaceDictReadResponse,
    FaceDictWriteRequest,
    FaceDictWriteResponse,
    MemoryToolError,
    RelatedMemory,
    VisualDictCandidate,
    VisualDictEntry,
    VisualDictReadRequest,
    VisualDictReadResponse,
    VisualDictWriteRequest,
    VisualDictWriteResponse,
    VoiceCandidate,
    VoiceDictEntry,
    VoiceDictReadRequest,
    VoiceDictReadResponse,
    VoiceDictWriteRequest,
    VoiceDictWriteResponse,
)
from memory_ops_memory import _store_long_term_text
from memory_search import _dict_graph_related
from memory_state import state

# Locks de escrita dos dicionários (mesmo racional do comentário em
# memory_ops_memory.py — um asyncio.Lock por fluxo de escrita).
_vd_write_lock = asyncio.Lock()
_fd_write_lock = asyncio.Lock()
_vo_write_lock = asyncio.Lock()


# ── NEW: Dicionário Visual — endpoints usados pelo vision.py ──────────────────
#
# vision.py NÃO persiste nada localmente: ele roda o pipeline (depth →
# clustering → segmentação → embeddings) e manda cada embedding de crop pra
# cá. Este arquivo decide se é um conceito novo ou um exemplo a mais de um
# conceito já existente, e é quem guarda tudo (FAISS + SQLite + memória de
# longo prazo).

async def visual_dict_write(req: VisualDictWriteRequest):
    concept_name = req.concept_name.strip()
    description  = req.description.strip()
    if not concept_name:
        return VisualDictWriteResponse(stored=False, reason="empty_concept_name")
    if len(req.embedding) != VD_EMBED_DIM:
        raise MemoryToolError(f"embedding deve ter dimensão {VD_EMBED_DIM}, recebido {len(req.embedding)}")

    vec = np.asarray(req.embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm

    loop = asyncio.get_event_loop()

    async with _vd_write_lock:
        existing = state.vd_db.get_by_name(concept_name)
        new_concept = existing is None

        if existing is not None:
            concept_id = existing["id"]
            memory_id  = existing["memory_id"]
        else:
            memory_id = None
            if req.link_to_memory and description:
                _, _, memory_id, _ = await _store_long_term_text(
                    f"{concept_name}: {description}", "visual_dict", req.confidence,
                )
            try:
                concept_id = state.vd_db.insert_concept(
                    concept_name, description, req.source, req.confidence, memory_id,
                )
            except sqlite3.IntegrityError:
                # Corrida: outro write criou o mesmo concept_key nesse meio-tempo.
                existing = state.vd_db.get_by_name(concept_name)
                if existing is None:
                    raise
                concept_id, memory_id, new_concept = existing["id"], existing["memory_id"], False
            else:
                log.info(f"Visual-dict: novo conceito #{concept_id} criado — '{concept_name}'")

        embedding_id = state.vd_db.insert_embedding(concept_id)
        await loop.run_in_executor(None, state.vd_index.add, vec, embedding_id)

    log.info(
        f"Visual-dict: embedding #{embedding_id} gravado p/ conceito #{concept_id} "
        f"({'novo' if new_concept else 'exemplo adicional'})"
    )

    return VisualDictWriteResponse(
        stored=True, reason="ok", concept_id=concept_id,
        embedding_id=embedding_id, memory_id=memory_id, new_concept=new_concept,
    )


async def visual_dict_read(req: VisualDictReadRequest):
    if len(req.embedding) != VD_EMBED_DIM:
        raise MemoryToolError(f"embedding deve ter dimensão {VD_EMBED_DIM}, recebido {len(req.embedding)}")

    vec = np.asarray(req.embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm

    # sobre-amostra o kNN porque vários embeddings podem apontar pro mesmo
    # conceito (múltiplos exemplos) — precisamos deduplicar por concept_id
    hits = state.vd_index.search(vec, max(req.top_k * 4, 20))

    best_score_by_concept: dict[int, float] = {}
    for embedding_id, score in hits:
        concept_id = state.vd_db.get_concept_id_by_embedding(embedding_id)
        if concept_id is None:
            continue
        if concept_id not in best_score_by_concept or score > best_score_by_concept[concept_id]:
            best_score_by_concept[concept_id] = score

    ranked = sorted(best_score_by_concept.items(), key=lambda kv: kv[1], reverse=True)[:req.top_k]

    results: list[VisualDictCandidate] = []
    for concept_id, score in ranked:
        if score < req.min_score:
            continue
        row = state.vd_db.get_concept_by_id(concept_id)
        if row is None:
            continue
        state.vd_db.update_access(concept_id)
        results.append(VisualDictCandidate(
            concept_id=row["id"],
            concept_name=row["concept_name"],
            description=row["description"],
            score=score,
            confidence=row["confidence"],
            access_count=row["access_count"] + 1,
            memory_id=row["memory_id"],
        ))

    # ambíguo quando: nenhum resultado confiável, OU os dois melhores
    # candidatos estão muito próximos (o objeto pode ser qualquer um dos dois)
    ambiguous = (
        len(results) == 0
        or (len(results) > 1 and (results[0].score - results[1].score) < VD_AMBIGUOUS_MARGIN)
    )

    # ── NEW: busca por grafo a partir dos hits (memory_id = nó no grafo Hebbiano)
    related: list[RelatedMemory] = []
    if req.include_related:
        related = await _dict_graph_related(
            [(c.memory_id, c.score) for c in results], asyncio.get_event_loop()
        )

    return VisualDictReadResponse(results=results, ambiguous=ambiguous, related=related)


async def visual_dict_list():
    rows = state.vd_db.list_concepts()
    concepts = [
        VisualDictEntry(
            concept_id=row["id"],
            concept_name=row["concept_name"],
            description=row["description"],
            source=row["source"],
            confidence=row["confidence"],
            memory_id=row["memory_id"],
            examples_count=row["examples_count"],
            created_at=row["created_at"],
            access_count=row["access_count"],
        ).model_dump()
        for row in rows
    ]
    return {"total": len(concepts), "concepts": concepts}


async def visual_dict_get(concept_id: int):
    row = state.vd_db.get_concept_by_id(concept_id)
    if row is None:
        raise MemoryToolError(f"Conceito visual #{concept_id} não encontrado")
    return VisualDictEntry(
        concept_id=row["id"],
        concept_name=row["concept_name"],
        description=row["description"],
        source=row["source"],
        confidence=row["confidence"],
        memory_id=row["memory_id"],
        examples_count=state.vd_db.count_examples(concept_id),
        created_at=row["created_at"],
        access_count=row["access_count"],
    )


async def visual_dict_delete(concept_id: int):
    """Remove um conceito visual e todos os seus embeddings (DB + FAISS)."""
    row = state.vd_db.get_concept_by_id(concept_id)
    if row is None:
        raise MemoryToolError(f"Conceito visual #{concept_id} não encontrado")

    embedding_ids = state.vd_db.get_embedding_ids_by_concept(concept_id)
    if embedding_ids:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, state.vd_index.remove_ids, set(embedding_ids))

    deleted = state.vd_db.delete_concept(concept_id)
    log.info(f"Visual-dict: conceito #{concept_id} removido ({len(embedding_ids)} embeddings)")
    return {"deleted": deleted, "concept_id": concept_id, "embeddings_removed": len(embedding_ids)}


# ── NEW: Dicionário de Rostos — endpoints usados pelo vision.py ───────────────
#
# Mesma lógica do dicionário visual acima, adaptada pra reconhecimento
# facial: vision.py detecta+alinha o rosto, extrai o embedding (EdgeFace) e
# manda pra cá. Aqui decide-se se é uma pessoa nova ou mais um exemplo de
# uma pessoa já cadastrada.

async def face_dict_write(req: FaceDictWriteRequest):
    person_name = req.person_name.strip()
    if not person_name:
        return FaceDictWriteResponse(stored=False, reason="empty_person_name")

    if len(req.embedding) != FD_EMBED_DIM:
        raise MemoryToolError(f"embedding deve ter dimensão {FD_EMBED_DIM}, recebido {len(req.embedding)}")

    vec = np.asarray(req.embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm

    description = req.description.strip()
    loop = asyncio.get_event_loop()

    async with _fd_write_lock:
        existing = state.fd_db.get_by_name(person_name)
        new_person = existing is None

        memory_id: Optional[int] = None
        if existing is not None:
            person_id = existing["id"]
            memory_id = existing["memory_id"]
            # se veio uma descrição não-vazia num cadastro de exemplo adicional,
            # atualiza/completa a descrição já salva (permite corrigir depois)
            if description:
                state.fd_db.update_description(person_id, description)
                # NEW: pessoa já cadastrada SEM link + link pedido → "promove"
                if req.link_to_memory and memory_id is None:
                    stored, _, memory_id, _ = await _store_long_term_text(
                        f"{person_name}: {description}", "face_dict", req.confidence,
                    )
                    if stored:
                        state.fd_db.set_memory_id(person_id, memory_id)
                    else:
                        memory_id = None
        else:
            # NEW (opt-in): descrição também vira memória LT → nó no grafo Hebbiano
            if req.link_to_memory and description:
                stored, _, memory_id, _ = await _store_long_term_text(
                    f"{person_name}: {description}", "face_dict", req.confidence,
                )
                if not stored:
                    memory_id = None  # dedup/erro — pessoa cadastrada mesmo assim, sem link
            try:
                person_id = state.fd_db.insert_person(
                    person_name=person_name, description=description,
                    source=req.source, confidence=req.confidence, memory_id=memory_id,
                )
            except sqlite3.IntegrityError:
                # Corrida: outro write criou a mesma person_key nesse meio-tempo.
                existing = state.fd_db.get_by_name(person_name)
                if existing is None:
                    raise
                person_id, memory_id, new_person = existing["id"], existing["memory_id"], False

        embedding_id = state.fd_db.insert_embedding(person_id)
        await loop.run_in_executor(None, state.fd_index.add, vec, embedding_id)

    log.info(
        f"Face-dict: {'nova pessoa' if new_person else 'novo exemplo'} "
        f"'{person_name}' (person_id={person_id}, embedding_id={embedding_id})"
    )

    return FaceDictWriteResponse(
        stored=True, reason="ok", person_id=person_id,
        embedding_id=embedding_id, memory_id=memory_id, new_person=new_person,
    )


async def face_dict_read(req: FaceDictReadRequest):
    if len(req.embedding) != FD_EMBED_DIM:
        raise MemoryToolError(f"embedding deve ter dimensão {FD_EMBED_DIM}, recebido {len(req.embedding)}")

    vec = np.asarray(req.embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm

    # sobre-amostra o kNN porque vários embeddings podem apontar pra mesma
    # pessoa (múltiplos exemplos) — precisamos deduplicar por person_id
    hits = state.fd_index.search(vec, max(req.top_k * 4, 20))

    best_score_by_person: dict[int, float] = {}
    for embedding_id, score in hits:
        person_id = state.fd_db.get_person_id_by_embedding(embedding_id)
        if person_id is None:
            continue
        if person_id not in best_score_by_person or score > best_score_by_person[person_id]:
            best_score_by_person[person_id] = score

    ranked = sorted(best_score_by_person.items(), key=lambda kv: kv[1], reverse=True)[:req.top_k]

    results: list[FaceCandidate] = []
    for person_id, score in ranked:
        if score < req.min_score:
            continue
        row = state.fd_db.get_person_by_id(person_id)
        if row is None:
            continue
        state.fd_db.update_access(person_id)
        results.append(FaceCandidate(
            person_id=row["id"],
            person_name=row["person_name"],
            description=row["description"],
            score=score,
            confidence=row["confidence"],
            access_count=row["access_count"] + 1,
            memory_id=row["memory_id"],
        ))

    # ambíguo quando: ninguém bateu com confiança suficiente, OU os dois
    # melhores candidatos estão muito próximos (pode ser qualquer um dos dois)
    ambiguous = (
        len(results) == 0
        or (len(results) > 1 and (results[0].score - results[1].score) < FD_AMBIGUOUS_MARGIN)
    )

    # ── NEW: busca por grafo a partir dos hits (só rostos com memory_id)
    related: list[RelatedMemory] = []
    if req.include_related:
        related = await _dict_graph_related(
            [(c.memory_id, c.score) for c in results], asyncio.get_event_loop()
        )

    return FaceDictReadResponse(results=results, ambiguous=ambiguous, related=related)


async def face_dict_list():
    rows = state.fd_db.list_people()
    people = [
        FaceDictEntry(
            person_id=row["id"],
            person_name=row["person_name"],
            description=row["description"],
            source=row["source"],
            confidence=row["confidence"],
            memory_id=row["memory_id"],
            examples_count=row["examples_count"],
            created_at=row["created_at"],
            access_count=row["access_count"],
        ).model_dump()
        for row in rows
    ]
    return {"total": len(people), "people": people}


async def face_dict_get(person_id: int):
    row = state.fd_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")
    return FaceDictEntry(
        person_id=row["id"],
        person_name=row["person_name"],
        description=row["description"],
        source=row["source"],
        confidence=row["confidence"],
        memory_id=row["memory_id"],
        examples_count=state.fd_db.count_examples(person_id),
        created_at=row["created_at"],
        access_count=row["access_count"],
    )


async def face_dict_update(person_id: int, req: FaceDictUpdateRequest):
    """Edita só a descrição de uma pessoa já cadastrada, sem precisar
    mandar um novo embedding junto."""
    row = state.fd_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")
    state.fd_db.update_description(person_id, req.description.strip())
    row = state.fd_db.get_person_by_id(person_id)
    return FaceDictEntry(
        person_id=row["id"],
        person_name=row["person_name"],
        description=row["description"],
        source=row["source"],
        confidence=row["confidence"],
        memory_id=row["memory_id"],
        examples_count=state.fd_db.count_examples(person_id),
        created_at=row["created_at"],
        access_count=row["access_count"],
    )


async def face_dict_delete(person_id: int):
    """Remove uma pessoa e todos os seus embeddings de rosto (DB + FAISS)."""
    row = state.fd_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")

    embedding_ids = state.fd_db.get_embedding_ids_by_person(person_id)
    if embedding_ids:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, state.fd_index.remove_ids, set(embedding_ids))

    deleted = state.fd_db.delete_person(person_id)
    log.info(f"Face-dict: pessoa #{person_id} removida ({len(embedding_ids)} embeddings)")
    return {"deleted": deleted, "person_id": person_id, "embeddings_removed": len(embedding_ids)}


# ── NEW: Dicionário de Voz — endpoints usados pelo pipeline de áudio ────────────
#
# Mesma lógica do dicionário de rostos, adaptada pra identificação de locutor:
# o pipeline de áudio extrai o embedding da frase falada (CAM++ etc.) e manda
# pra cá. Aqui decide-se se é uma pessoa nova ou mais um exemplo de uma pessoa
# já cadastrada. DIFERENÇA em relação ao face-dict: a descrição de quem é a
# pessoa também é gravada na memória de longo prazo (source="voice_dict"),
# então ela é pesquisável pela leitura normal (/read) — igual ao link_to_memory
# do dicionário visual.

async def voice_dict_write(req: VoiceDictWriteRequest):
    person_name = req.person_name.strip()
    description = req.description.strip()
    if not person_name:
        return VoiceDictWriteResponse(stored=False, reason="empty_person_name")

    if len(req.embedding) != VO_EMBED_DIM:
        raise MemoryToolError(f"embedding deve ter dimensão {VO_EMBED_DIM}, recebido {len(req.embedding)}")

    vec = np.asarray(req.embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm

    loop = asyncio.get_event_loop()

    async with _vo_write_lock:
        existing = state.vo_db.get_by_name(person_name)
        new_person = existing is None

        if existing is not None:
            person_id = existing["id"]
            memory_id = existing["memory_id"]
            # se veio uma descrição não-vazia num cadastro de exemplo adicional,
            # atualiza/completa a descrição já salva
            if description:
                state.vo_db.update_description(person_id, description)
        else:
            memory_id = None
            if description:
                # ── descrição replicada na memória de longo prazo — mesma
                # política do visual-dict: só na criação do conceito/pessoa,
                # exemplos adicionais não duplicam a entrada de texto.
                stored, _, memory_id, _ = await _store_long_term_text(
                    f"{person_name}: {description}", "voice_dict", req.confidence,
                )
                if not stored:
                    memory_id = None  # dedup/erro — a pessoa ainda é cadastrada,
                                        # só sem o link pra memória
            try:
                person_id = state.vo_db.insert_person(
                    person_name=person_name, description=description,
                    source=req.source, confidence=req.confidence, memory_id=memory_id,
                )
            except sqlite3.IntegrityError:
                # Corrida: outro write criou a mesma person_key nesse meio-tempo.
                existing = state.vo_db.get_by_name(person_name)
                if existing is None:
                    raise
                person_id, memory_id, new_person = existing["id"], existing["memory_id"], False

        embedding_id = state.vo_db.insert_embedding(person_id)
        await loop.run_in_executor(None, state.vo_index.add, vec, embedding_id)

    log.info(
        f"Voice-dict: {'nova pessoa' if new_person else 'novo exemplo'} "
        f"'{person_name}' (person_id={person_id}, embedding_id={embedding_id}, memory_id={memory_id})"
    )

    return VoiceDictWriteResponse(
        stored=True, reason="ok", person_id=person_id,
        embedding_id=embedding_id, memory_id=memory_id, new_person=new_person,
    )


async def voice_dict_read(req: VoiceDictReadRequest):
    if len(req.embedding) != VO_EMBED_DIM:
        raise MemoryToolError(f"embedding deve ter dimensão {VO_EMBED_DIM}, recebido {len(req.embedding)}")

    vec = np.asarray(req.embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm

    # sobre-amostra o kNN porque vários embeddings podem apontar pra mesma
    # pessoa (múltiplos exemplos) — precisamos deduplicar por person_id
    hits = state.vo_index.search(vec, max(req.top_k * 4, 20))

    best_score_by_person: dict[int, float] = {}
    for embedding_id, score in hits:
        person_id = state.vo_db.get_person_id_by_embedding(embedding_id)
        if person_id is None:
            continue
        if person_id not in best_score_by_person or score > best_score_by_person[person_id]:
            best_score_by_person[person_id] = score

    ranked = sorted(best_score_by_person.items(), key=lambda kv: kv[1], reverse=True)[:req.top_k]

    results: list[VoiceCandidate] = []
    for person_id, score in ranked:
        if score < req.min_score:
            continue
        row = state.vo_db.get_person_by_id(person_id)
        if row is None:
            continue
        state.vo_db.update_access(person_id)
        results.append(VoiceCandidate(
            person_id=row["id"],
            person_name=row["person_name"],
            description=row["description"],
            score=score,
            confidence=row["confidence"],
            access_count=row["access_count"] + 1,
            memory_id=row["memory_id"],
        ))

    # ambíguo quando: ninguém bateu com confiança suficiente, OU os dois
    # melhores candidatos estão muito próximos (pode ser qualquer um dos dois)
    ambiguous = (
        len(results) == 0
        or (len(results) > 1 and (results[0].score - results[1].score) < VO_AMBIGUOUS_MARGIN)
    )

    # ── NEW: busca por grafo a partir dos hits
    related: list[RelatedMemory] = []
    if req.include_related:
        related = await _dict_graph_related(
            [(c.memory_id, c.score) for c in results], asyncio.get_event_loop()
        )

    return VoiceDictReadResponse(results=results, ambiguous=ambiguous, related=related)


async def voice_dict_list():
    rows = state.vo_db.list_people()
    people = [
        VoiceDictEntry(
            person_id=row["id"],
            person_name=row["person_name"],
            description=row["description"],
            source=row["source"],
            confidence=row["confidence"],
            memory_id=row["memory_id"],
            examples_count=row["examples_count"],
            created_at=row["created_at"],
            access_count=row["access_count"],
        ).model_dump()
        for row in rows
    ]
    return {"total": len(people), "people": people}


async def voice_dict_get(person_id: int):
    row = state.vo_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")
    return VoiceDictEntry(
        person_id=row["id"],
        person_name=row["person_name"],
        description=row["description"],
        source=row["source"],
        confidence=row["confidence"],
        memory_id=row["memory_id"],
        examples_count=state.vo_db.count_examples(person_id),
        created_at=row["created_at"],
        access_count=row["access_count"],
    )


async def voice_dict_update(person_id: int, req: VoiceDictUpdateRequest):
    """Edita só a descrição de uma pessoa já cadastrada, sem precisar
    mandar um novo embedding junto."""
    row = state.vo_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")
    state.vo_db.update_description(person_id, req.description.strip())
    row = state.vo_db.get_person_by_id(person_id)
    return VoiceDictEntry(
        person_id=row["id"],
        person_name=row["person_name"],
        description=row["description"],
        source=row["source"],
        confidence=row["confidence"],
        memory_id=row["memory_id"],
        examples_count=state.vo_db.count_examples(person_id),
        created_at=row["created_at"],
        access_count=row["access_count"],
    )


async def voice_dict_delete(person_id: int):
    """Remove uma pessoa e todos os seus embeddings de voz (DB + FAISS).

    A memória de longo prazo linkada (descrição gravada na criação) NÃO é
    removida — a remoção explícita lá continua sendo responsabilidade de
    quem gerencia `memories` (mesma política do visual-dict)."""
    row = state.vo_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")

    embedding_ids = state.vo_db.get_embedding_ids_by_person(person_id)
    if embedding_ids:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, state.vo_index.remove_ids, set(embedding_ids))

    deleted = state.vo_db.delete_person(person_id)
    log.info(f"Voice-dict: pessoa #{person_id} removida ({len(embedding_ids)} embeddings)")
    return {"deleted": deleted, "person_id": person_id, "embeddings_removed": len(embedding_ids)}


__all__ = [
    '_fd_write_lock',
    '_vd_write_lock',
    '_vo_write_lock',
    'face_dict_delete',
    'face_dict_get',
    'face_dict_list',
    'face_dict_read',
    'face_dict_update',
    'face_dict_write',
    'visual_dict_delete',
    'visual_dict_get',
    'visual_dict_list',
    'visual_dict_read',
    'visual_dict_write',
    'voice_dict_delete',
    'voice_dict_get',
    'voice_dict_list',
    'voice_dict_read',
    'voice_dict_update',
    'voice_dict_write',
]
