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
import json
import sqlite3
from typing import Optional

import numpy as np

from memory_config import (
    FD_AMBIGUOUS_MARGIN,
    FD_EMBED_DIM,
    PERSONA_FILE,
    PERSONA_MAX,
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
    IdentityEntry,
    IdentityUpdateRequest,
    MemoryToolError,
    PeopleCardEntry,
    PeopleCardUpdateRequest,
    PeopleCardWriteRequest,
    PeopleCardWriteResponse,
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
        existing = state.pd_db.get_by_name(person_name)
        new_person = existing is None

        memory_id: Optional[int] = None
        if existing is not None:
            person_id = existing["id"]
            memory_id = existing["memory_id"]
            # se veio uma descrição não-vazia num cadastro de exemplo adicional,
            # atualiza/completa a descrição já salva (permite corrigir depois)
            if description:
                state.pd_db.update_card(person_id, description)
                # NEW: pessoa já cadastrada SEM link + link pedido → "promove"
                if req.link_to_memory and memory_id is None:
                    stored, _, memory_id, _ = await _store_long_term_text(
                        f"{person_name}: {description}", "face_dict", req.confidence,
                    )
                    if stored:
                        state.pd_db.set_memory_id(person_id, memory_id)
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
                person_id = state.pd_db.insert_person(
                    person_name=person_name, description=description,
                    likes=req.likes.strip(), religion=req.religion.strip(),
                    additional_info=req.additional_info.strip(),
                    source=req.source, confidence=req.confidence, memory_id=memory_id,
                )
            except sqlite3.IntegrityError:
                # Corrida: outro write criou a mesma person_key nesse meio-tempo.
                existing = state.pd_db.get_by_name(person_name)
                if existing is None:
                    raise
                person_id, memory_id, new_person = existing["id"], existing["memory_id"], False

        embedding_id = state.pd_db.insert_embedding(person_id, "face")
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
        person_id = state.pd_db.get_person_id_by_embedding(embedding_id, "face")
        if person_id is None:
            continue
        if person_id not in best_score_by_person or score > best_score_by_person[person_id]:
            best_score_by_person[person_id] = score

    ranked = sorted(best_score_by_person.items(), key=lambda kv: kv[1], reverse=True)[:req.top_k]

    results: list[FaceCandidate] = []
    for person_id, score in ranked:
        if score < req.min_score:
            continue
        row = state.pd_db.get_person_by_id(person_id)
        if row is None:
            continue
        state.pd_db.update_access(person_id)
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
    rows = state.pd_db.list_people()
    people = [
        FaceDictEntry(
            person_id=row["id"],
            person_name=row["person_name"],
            description=row["description"],
            likes=row["likes"],
            religion=row["religion"],
            additional_info=row["additional_info"],
            source=row["source"],
            confidence=row["confidence"],
            memory_id=row["memory_id"],
            examples_count=row["face_examples_count"],
            created_at=row["created_at"],
            access_count=row["access_count"],
        ).model_dump()
        for row in rows
    ]
    return {"total": len(people), "people": people}


async def face_dict_get(person_id: int):
    row = state.pd_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")
    return FaceDictEntry(
        person_id=row["id"],
        person_name=row["person_name"],
        description=row["description"],
        likes=row["likes"],
        religion=row["religion"],
        additional_info=row["additional_info"],
        source=row["source"],
        confidence=row["confidence"],
        memory_id=row["memory_id"],
        examples_count=state.pd_db.count_examples(person_id, "face"),
        created_at=row["created_at"],
        access_count=row["access_count"],
    )


async def face_dict_update(person_id: int, req: FaceDictUpdateRequest):
    """Edita campos do card de pessoa já cadastrado, sem precisar mandar um
    novo embedding junto. Campos não enviados (None) NÃO são alterados."""
    row = state.pd_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")
    state.pd_db.update_card(
        person_id,
        description=req.description.strip() or None,
        likes=req.likes,
        religion=req.religion,
        additional_info=req.additional_info,
    )
    row = state.pd_db.get_person_by_id(person_id)
    return FaceDictEntry(
        person_id=row["id"],
        person_name=row["person_name"],
        description=row["description"],
        likes=row["likes"],
        religion=row["religion"],
        additional_info=row["additional_info"],
        source=row["source"],
        confidence=row["confidence"],
        memory_id=row["memory_id"],
        examples_count=state.pd_db.count_examples(person_id, "face"),
        created_at=row["created_at"],
        access_count=row["access_count"],
    )


async def face_dict_delete(person_id: int):
    """Remove um card de pessoa e todos os seus embeddings (DB + índices).

    Unificação: o card é único (PeopleDB), então deletar a pessoa remove
    os embeddings de rosto E de voz do banco (CASCADE) e dos dois índices
    vetoriais — sem isso, o índice de voz ficaria com vetores órfãos."""
    row = state.pd_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")

    face_ids  = state.pd_db.get_embedding_ids_by_person(person_id, "face")
    voice_ids = state.pd_db.get_embedding_ids_by_person(person_id, "voice")
    loop = asyncio.get_event_loop()
    if face_ids:
        await loop.run_in_executor(None, state.fd_index.remove_ids, set(face_ids))
    if voice_ids:
        await loop.run_in_executor(None, state.vo_index.remove_ids, set(voice_ids))

    deleted = state.pd_db.delete_person(person_id)
    log.info(
        f"Face-dict: card de pessoa #{person_id} removido "
        f"({len(face_ids)} emb. rosto, {len(voice_ids)} emb. voz)"
    )
    return {
        "deleted": deleted, "person_id": person_id,
        "embeddings_removed": len(face_ids) + len(voice_ids),
    }


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
        existing = state.pd_db.get_by_name(person_name)
        new_person = existing is None

        if existing is not None:
            person_id = existing["id"]
            memory_id = existing["memory_id"]
            # se veio uma descrição não-vazia num cadastro de exemplo adicional,
            # atualiza/completa a descrição já salva
            if description:
                state.pd_db.update_card(person_id, description)
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
                person_id = state.pd_db.insert_person(
                    person_name=person_name, description=description,
                    likes=req.likes.strip(), religion=req.religion.strip(),
                    additional_info=req.additional_info.strip(),
                    source=req.source, confidence=req.confidence, memory_id=memory_id,
                )
            except sqlite3.IntegrityError:
                # Corrida: outro write criou a mesma person_key nesse meio-tempo.
                existing = state.pd_db.get_by_name(person_name)
                if existing is None:
                    raise
                person_id, memory_id, new_person = existing["id"], existing["memory_id"], False

        embedding_id = state.pd_db.insert_embedding(person_id, "voice")
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
        person_id = state.pd_db.get_person_id_by_embedding(embedding_id, "voice")
        if person_id is None:
            continue
        if person_id not in best_score_by_person or score > best_score_by_person[person_id]:
            best_score_by_person[person_id] = score

    ranked = sorted(best_score_by_person.items(), key=lambda kv: kv[1], reverse=True)[:req.top_k]

    results: list[VoiceCandidate] = []
    for person_id, score in ranked:
        if score < req.min_score:
            continue
        row = state.pd_db.get_person_by_id(person_id)
        if row is None:
            continue
        state.pd_db.update_access(person_id)
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
    rows = state.pd_db.list_people()
    people = [
        VoiceDictEntry(
            person_id=row["id"],
            person_name=row["person_name"],
            description=row["description"],
            likes=row["likes"],
            religion=row["religion"],
            additional_info=row["additional_info"],
            source=row["source"],
            confidence=row["confidence"],
            memory_id=row["memory_id"],
            examples_count=row["voice_examples_count"],
            created_at=row["created_at"],
            access_count=row["access_count"],
        ).model_dump()
        for row in rows
    ]
    return {"total": len(people), "people": people}


async def voice_dict_get(person_id: int):
    row = state.pd_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")
    return VoiceDictEntry(
        person_id=row["id"],
        person_name=row["person_name"],
        description=row["description"],
        likes=row["likes"],
        religion=row["religion"],
        additional_info=row["additional_info"],
        source=row["source"],
        confidence=row["confidence"],
        memory_id=row["memory_id"],
        examples_count=state.pd_db.count_examples(person_id, "voice"),
        created_at=row["created_at"],
        access_count=row["access_count"],
    )


async def voice_dict_update(person_id: int, req: VoiceDictUpdateRequest):
    """Edita campos do card de pessoa já cadastrado, sem precisar mandar um
    novo embedding junto. Campos não enviados (None) NÃO são alterados."""
    row = state.pd_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")
    state.pd_db.update_card(
        person_id,
        description=req.description.strip() or None,
        likes=req.likes,
        religion=req.religion,
        additional_info=req.additional_info,
    )
    row = state.pd_db.get_person_by_id(person_id)
    return VoiceDictEntry(
        person_id=row["id"],
        person_name=row["person_name"],
        description=row["description"],
        likes=row["likes"],
        religion=row["religion"],
        additional_info=row["additional_info"],
        source=row["source"],
        confidence=row["confidence"],
        memory_id=row["memory_id"],
        examples_count=state.pd_db.count_examples(person_id, "voice"),
        created_at=row["created_at"],
        access_count=row["access_count"],
    )


async def voice_dict_delete(person_id: int):
    """Remove um card de pessoa e todos os seus embeddings (DB + índices).

    Unificação: o card é único (PeopleDB), então deletar a pessoa remove
    os embeddings de voz E de rosto do banco (CASCADE) e dos dois índices
    vetoriais — sem isso, o índice de rosto ficaria com vetores órfãos.

    A memória de longo prazo linkada (descrição gravada na criação) NÃO é
    removida — a remoção explícita lá continua sendo responsabilidade de
    quem gerencia `memories` (mesma política do visual-dict)."""
    row = state.pd_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")

    voice_ids = state.pd_db.get_embedding_ids_by_person(person_id, "voice")
    face_ids  = state.pd_db.get_embedding_ids_by_person(person_id, "face")
    loop = asyncio.get_event_loop()
    if voice_ids:
        await loop.run_in_executor(None, state.vo_index.remove_ids, set(voice_ids))
    if face_ids:
        await loop.run_in_executor(None, state.fd_index.remove_ids, set(face_ids))

    deleted = state.pd_db.delete_person(person_id)
    log.info(
        f"Voice-dict: card de pessoa #{person_id} removido "
        f"({len(voice_ids)} emb. voz, {len(face_ids)} emb. rosto)"
    )
    return {
        "deleted": deleted, "person_id": person_id,
        "embeddings_removed": len(voice_ids) + len(face_ids),
    }


# ── NEW: People Card — card textual de pessoa (sem embeddings) ────────────────
#
# O card unificado (PeopleDB) já centraliza os metadados ricos das pessoas
# (nome, descrição, gostos, religião, informações adicionais) — é a mesma
# tabela `people` que face-dict e voice-dict usam. Estes handlers expõem o
# card como recurso TEXTUAL puro: o MCP alpha-mcp (LLM, sem câmera/microfone)
# cria e edita pessoas pela conversa. O mesmo person_id continua recebendo
# embeddings dos pipelines de visão/áudio normalmente.
#
# O card ganha um nó de identidade no grafo Hebbiano (memória LT
# source="people_card", id guardado em memory_id) — é por ele que /read com
# person_id injeta semente no PPR e memórias sobre a pessoa vêm como "related".

_pc_write_lock = asyncio.Lock()  # guarda o check-then-insert no PeopleDB


def _people_card_entry(row: sqlite3.Row) -> PeopleCardEntry:
    return PeopleCardEntry(
        person_id=row["id"],
        person_name=row["person_name"],
        description=row["description"],
        likes=row["likes"],
        religion=row["religion"],
        additional_info=row["additional_info"],
        source=row["source"],
        confidence=row["confidence"],
        memory_id=row["memory_id"],
        face_examples_count=state.pd_db.count_examples(row["id"], "face"),
        voice_examples_count=state.pd_db.count_examples(row["id"], "voice"),
        created_at=row["created_at"],
        access_count=row["access_count"],
    )


async def _card_identity_memory(
    person_id: int, person_name: str, description: str,
    current: Optional[int],
) -> Optional[int]:
    """Cria/atualiza a memória de identidade LT do card ("Nome: descrição",
    source="people_card") e garante o memory_id atualizado no card.

    Card com memória existente → o texto é atualizado in-place (action=
    "update", mesmo memory_id — o nó de identidade permanece estável no
    grafo). Falha no update → mantém o vínculo atual (o nó antigo continua
    válido). Card sem memória → cria o nó e grava o id no card."""
    text = f"{person_name}: {description.strip() or 'sem descrição'}"
    if current is not None:
        stored, _, memory_id, _ = await _store_long_term_text(
            text, "people_card", 1.0, False,
            action="update", memory_id=current,
            internal=True,  # cards guardam id CRUA (o nó pode morar no geral)
        )
        if stored:
            return memory_id
        log.warning(
            f"people: update da memória de identidade do card #{person_id} "
            f"recusado — vínculo atual mantido"
        )
        return current
    stored, reason, memory_id, _ = await _store_long_term_text(
        text, "people_card", 1.0, False,
    )
    if not stored or memory_id is None:
        log.warning(
            f"people: falha ao criar memória de identidade do card "
            f"#{person_id} ({reason}) — card criado sem nó de identidade"
        )
        return None
    state.pd_db.set_memory_id(person_id, memory_id)
    return memory_id


async def people_card_write(req: PeopleCardWriteRequest):
    """Cria (action="create") ou edita (action="update") um card de pessoa
    pelo nome — fluxo textual puro, sem embedding.

    update: campos AUSENTES do JSON não alteram o card; campos enviados
    sobrescrevem (string vazia limpa o campo) — o POST explicita só o que
    mudou. O card é localizado por person_id (quando enviado) ou pelo nome.
    create: nome já cadastrado → stored=False, reason="duplicate" (o chamador
    decide editar o card existente em vez de criar um duplicado)."""
    person_name = req.person_name.strip()
    if not person_name:
        return PeopleCardWriteResponse(stored=False, reason="empty_person_name")

    async with _pc_write_lock:
        if req.action == "update":
            row = (state.pd_db.get_person_by_id(req.person_id)
                   if req.person_id is not None
                   else state.pd_db.get_by_name(person_name))
            if row is None:
                return PeopleCardWriteResponse(stored=False, reason="not_found")
            person_id = row["id"]
            # só os campos explicitamente enviados no JSON são alterados
            sent = set(req.model_fields_set)
            state.pd_db.update_card(
                person_id,
                description=req.description if "description" in sent else None,
                likes=req.likes if "likes" in sent else None,
                religion=req.religion if "religion" in sent else None,
                additional_info=req.additional_info if "additional_info" in sent else None,
            )
            if req.link_to_memory:
                row = state.pd_db.get_person_by_id(person_id)
                await _card_identity_memory(
                    person_id, row["person_name"], row["description"], row["memory_id"],
                )
            log.info(f"People-card: card #{person_id} ('{row['person_name']}') atualizado")
            return PeopleCardWriteResponse(
                stored=True, reason="ok", person_id=person_id,
                memory_id=state.pd_db.get_person_by_id(person_id)["memory_id"],
                new_person=False,
            )

        # create — nome já cadastrado é recusado (evita card duplicado)
        existing = state.pd_db.get_by_name(person_name)
        if existing is not None:
            return PeopleCardWriteResponse(
                stored=False, reason="duplicate", person_id=existing["id"],
                memory_id=existing["memory_id"], new_person=False,
            )

        memory_id = None
        if req.link_to_memory:
            # nó de identidade LT criado ANTES do insert (insert_person já
            # grava o memory_id) — o card ainda não existe, set_memory_id
            # não teria a quem aplicar.
            stored, reason, memory_id, _ = await _store_long_term_text(
                f"{person_name}: {req.description.strip() or 'sem descrição'}",
                "people_card", 1.0, False,
            )
            if not stored:
                memory_id = None  # dedup/erro — card criado mesmo assim, sem nó
        try:
            person_id = state.pd_db.insert_person(
                person_name=person_name,
                description=req.description.strip(),
                likes=req.likes.strip(),
                religion=req.religion.strip(),
                additional_info=req.additional_info.strip(),
                source="people_card", confidence=1.0, memory_id=memory_id,
            )
        except sqlite3.IntegrityError:
            # Corrida: outro write criou a mesma person_key nesse meio-tempo.
            existing = state.pd_db.get_by_name(person_name)
            if existing is None:
                raise
            return PeopleCardWriteResponse(
                stored=False, reason="duplicate", person_id=existing["id"],
                memory_id=existing["memory_id"], new_person=False,
            )

    log.info(
        f"People-card: card #{person_id} criado — '{person_name}' "
        f"(memory_id={memory_id})"
    )
    return PeopleCardWriteResponse(
        stored=True, reason="ok", person_id=person_id,
        memory_id=memory_id, new_person=True,
    )


async def people_card_update(person_id: int, req: PeopleCardUpdateRequest):
    """Edita campos do card de pessoa SEM embedding. Campos não enviados
    (None) NÃO são alterados. Renomear ou mudar a descrição atualiza a
    memória de identidade LT in-place (relink_memory=True)."""
    row = state.pd_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")
    state.pd_db.update_card(
        person_id,
        description=req.description,
        likes=req.likes,
        religion=req.religion,
        additional_info=req.additional_info,
        person_name=req.person_name.strip() if req.person_name else None,
    )
    row = state.pd_db.get_person_by_id(person_id)
    if req.relink_memory and ((req.person_name or "").strip() or req.description is not None):
        await _card_identity_memory(
            person_id, row["person_name"], row["description"], row["memory_id"],
        )
        row = state.pd_db.get_person_by_id(person_id)
    return _people_card_entry(row)


async def people_card_get(person_id: int):
    row = state.pd_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")
    return _people_card_entry(row)


async def people_card_list():
    rows = state.pd_db.list_people()
    people = [_people_card_entry(row).model_dump() for row in rows]
    return {"total": len(people), "people": people}


# ── NEW: Identity Card — card singleton da própria assistente (Alpha) ─────────
#
# Mesmo desenho dos people cards, aplicado à ALPHa: card textual em ava_people.db
# (tabela `identity`, UMA linha — id=1) + memória de identidade LT
# (source="identity", id guardado em identity.memory_id). É essa memória que
# dá à Alpha o nó no grafo Hebbiano e a aresta `identity_memory` no viewer
# (graph_activation.py) — por ela ela se liga a outras memórias LT e,
# transitivamente, às pessoas (cada uma com a sua).
#
# Seed no boot (identity_seed): se a tabela estiver vazia, o card nasce a
# partir de resource/persona.json — a frase de identidade que também é a 1ª
# linha do system prompt no orchestrator. Depois do seed os dois evoluem
# INDEPENDENTES (persona.json continua fonte do prompt; o card é o que vive
# no grafo). O seed é auto-curativo: card existe sem memória (ONNX fora do
# ar no boot anterior) → tenta criar o vínculo no boot seguinte.

_identity_lock = asyncio.Lock()   # guarda check-then-insert/update do singleton


def _identity_entry(row: sqlite3.Row) -> IdentityEntry:
    return IdentityEntry(
        name=row["name"],
        description=row["description"],
        roles=row["roles"],
        additional_info=row["additional_info"],
        memory_id=row["memory_id"],
        source=row["source"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _load_persona_identity() -> str:
    """Frase de identidade persistida em resource/persona.json (MESMO arquivo
    que o orchestrator usa como 1ª linha do system prompt). Ausente/corrompido
    → default — o seed nunca derruba o boot por causa de uma linha de prompt."""
    try:
        data = json.loads(PERSONA_FILE.read_text(encoding="utf-8"))
        p = str(data.get("persona") or "").strip()
        if p:
            return p[:PERSONA_MAX]
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning(f"Identity: falha ao ler {PERSONA_FILE} — usando default: {e}")
    return "You are Alpha, a helpful assistant"


async def _identity_memory(
    name: str, description: str, current: Optional[int],
) -> Optional[int]:
    """Cria/atualiza a memória de identidade LT do card ("Nome: descrição",
    source="identity") e garante o memory_id atualizado no card — espelha
    _card_identity_memory: update in-place (action="update", mesmo memory_id)
    mantém o nó estável no grafo; falha no update mantém o vínculo atual."""
    text = f"{name}: {description.strip() or 'sem descrição'}"
    if current is not None:
        stored, _, memory_id, _ = await _store_long_term_text(
            text, "identity", 1.0, False,
            action="update", memory_id=current,
            internal=True,  # card guarda id CRUA (a memória pode morar no geral)
        )
        if stored:
            return memory_id
        log.warning(
            f"identity: update da memória de identidade recusado — vínculo "
            f"atual mantido (#{current})"
        )
        return current
    stored, reason, memory_id, _ = await _store_long_term_text(
        text, "identity", 1.0, False,
    )
    if not stored or memory_id is None:
        log.warning(
            f"identity: falha ao criar memória de identidade ({reason}) — "
            f"card segue sem nó"
        )
        return None
    state.pd_db.set_identity_memory_id(memory_id)
    return memory_id


async def identity_seed() -> None:
    """Seed do card de identidade no BOOT (chamado por memory_state.startup).

    Tabela vazia → cria o card a partir de resource/persona.json e, na
    sequência, a memória de identidade LT. Card já existe SEM memória →
    só tenta criar o vínculo (auto-heal do boot em que o ONNX estava fora).
    Qualquer falha vira warning — o boot da memory NUNCA quebra por aqui."""
    try:
        row = state.pd_db.get_identity()
        if row is None:
            persona = _load_persona_identity()
            # Memória PRIMEIRO (insert já grava o memory_id — mesmo fluxo do
            # people_card create). O embedding pode ESTOURAR exceção (ONNX
            # serving ainda subindo no boot) — sem este try o insert nunca
            # rodaria e a tabela nasceria VAZIA. Agora: falha → card nasce
            # sem vínculo e o retry em background (memory_state.
            # _identity_seed_retry) cria o vínculo quando o ONNX responder.
            memory_id = None
            try:
                memory_id = await _identity_memory("Alpha", persona, None)
            except Exception as e:
                log.warning(
                    f"Identity: memória de identidade falhou no seed "
                    f"({type(e).__name__}: {e}) — card nasce sem vínculo"
                )
            state.pd_db.insert_identity(
                name="Alpha", description=persona, memory_id=memory_id,
            )
            log.info(
                f"Identity: card semeado da persona (memory_id={memory_id})"
            )
            return
        if row["memory_id"] is None:
            memory_id = await _identity_memory(
                row["name"], row["description"], None,
            )
            log.info(
                f"Identity: vínculo de memória criado no boot "
                f"(memory_id={memory_id})"
            )
    except Exception as e:
        log.warning(f"Identity: seed pulado ({type(e).__name__}: {e})")


async def identity_get() -> IdentityEntry:
    """Card de identidade da Alpha (singleton). Cria o card MINIMO na hora
    se o seed do boot ainda não rodou (ONNX fora do ar) — o vínculo de
    memória nasce no boot seguinte ou em identity_update."""
    row = state.pd_db.get_identity()
    if row is None:
        async with _identity_lock:
            row = state.pd_db.get_identity()
            if row is None:
                state.pd_db.insert_identity(name="Alpha")
                row = state.pd_db.get_identity()
    return _identity_entry(row)


async def identity_update(req: IdentityUpdateRequest) -> IdentityEntry:
    """Edita o card de identidade (singleton). Campos não enviados (None)
    NÃO são alterados. Mudar name/description regenera a memória de identidade
    LT in-place (relink_memory=True) — o nó permanece estável no grafo."""
    async with _identity_lock:
        row = state.pd_db.get_identity()
        if row is None:
            state.pd_db.insert_identity(name="Alpha")   # seed ausente → card mínimo
        sent = set(req.model_fields_set)
        state.pd_db.update_identity(
            name=req.name if "name" in sent else None,
            description=req.description if "description" in sent else None,
            roles=req.roles if "roles" in sent else None,
            additional_info=req.additional_info if "additional_info" in sent else None,
        )
        row = state.pd_db.get_identity()
        if req.relink_memory and (("name" in sent) or ("description" in sent)):
            await _identity_memory(row["name"], row["description"], row["memory_id"])
            row = state.pd_db.get_identity()
        log.info(
            f"Identity: card atualizado (memory_id={row['memory_id']}, "
            f"campos: {', '.join(sorted(sent - {'relink_memory'}))})"
        )
        return _identity_entry(row)


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
    'identity_get',
    'identity_seed',
    'identity_update',
    'people_card_get',
    'people_card_list',
    'people_card_update',
    'people_card_write',
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
