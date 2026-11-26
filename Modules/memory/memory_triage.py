# Alpha AI (AVA) — Triagem de memória: instrumentação do extrator
# Copyright (C) 2026  Felipe Lumazini
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
from __future__ import annotations

"""
memory_triage.py — Fase 1 do planejamento de triagem de memória.

Não decide nada e não muda a escrita de memórias: apenas registra, de forma
idempotente, (1) o que o LLM propôs, (2) o que o pipeline de escrita aplicou e
(3) os segmentos sobre os quais o extrator nunca se pronunciou. Esses três
estados são distintos — um segmento suprimido por orçamento NÃO é um
`DISCARD`, é `unlabeled` (PU learning, ver docs/MEMORY_TRIAGE_PHASE1.md §5).

Os embeddings NÃO são gravados: são determinísticos dado texto + versão do e5
e recriados no treino. Guardamos só `embed_model_version` e o que é caro
recomputar (`cross_encoder_score`, snapshot do candidato).
"""

import difflib
import hashlib
import json
import re
import time
from typing import Optional

from memory_config import (
    MEMORY_TRIAGE_CONTRASTIVE,
    MEMORY_TRIAGE_GATE,
    MEMORY_TRIAGE_GATE_WEIGHT,
    MEMORY_TRIAGE_GOLD_TARGET,
    MEMORY_TRIAGE_LOG,
    MEMORY_TRIAGE_LOG_TEXT,
    MEMORY_TRIAGE_MAX_TEXT_CHARS,
    MEMORY_TRIAGE_PAIR_TOP_K,
    MEMORY_TRIAGE_PROBE,
    MEMORY_TRIAGE_PROBE_MAX,
    MEMORY_TRIAGE_PROBE_RATE,
    MEMORY_TRIAGE_PROMPT_VERSION,
    MEMORY_TRIAGE_PU,
    MEMORY_TRIAGE_RETENTION_DAYS,
    MEMORY_TRIAGE_RENDER_LABELS,
    MEMORY_TRIAGE_SCHEMA_VERSION,
    MEMORY_TRIAGE_SEGMENT_MIN_CHARS,
    MEMORY_TRIAGE_SHADOW,
    MEMORY_TRIAGE_SYNTHETIC,
    MEMORY_TRIAGE_SYNTHETIC_MAX,
    MEMORY_TRIAGE_TARGET_SAMPLES,
    log,
)
from memory_models import (
    TriageGoldListResponse,
    TriageGoldRequest,
    TriageGoldResponse,
    TriageLabelsResponse,
    TriageLogRequest,
    TriageLogResponse,
    TriageReadiness,
    TriageStatsResponse,
)

# `memory_search` (numpy) e `memory_state` (LanceDB) são importados sob
# demanda: LLM.py importa só os helpers puros deste módulo, em background,
# e não deve pagar (nem exigir) as dependências do serviço de memória.

# Contadores do processo de memória (expostos em /status e /metrics).
_STATS = {
    "triage_logged_total": 0,
    "triage_alignment_exact": 0,
    "triage_alignment_fuzzy": 0,
    "triage_alignment_none": 0,
    "triage_log_failures": 0,
    "triage_suppressed_total": 0,
    "triage_probe_calls_total": 0,
    "triage_probe_segments_total": 0,
    "triage_contrastive_pairs_total": 0,
    "triage_gate_calls_total": 0,
    "triage_synthetic_total": 0,
    "triage_thinking_total": 0,
    "triage_gold_reviews_total": 0,
    "triage_recall_gap": 0.0,
}

# Sentinelus de candidate_id para segmentos sem candidato. SQLite não trata
# NULL como duplicado em UNIQUE; -1 mantém a idempotência também nos
# suprimidos (a coluna continua INTEGER e o export devolve None).
_NO_CANDIDATE = -1

_ALIGN_MIN_RATIO = 0.75
_DEICTIC_RE = re.compile(
    r"\b(?:eu|me|mim|minha|meu|ele|ela|isso|isto|aquilo|esse|essa|aqui|ali|"
    r"agora|ontem|hoje|amanhã|it|this|that|he|she|they|here|there|yesterday|today|tomorrow)\b",
    re.IGNORECASE,
)
_POLICY_RE = re.compile(
    r"^\s*(?:\?|oi|ol[aá]|obrigad[oa]|valeu|ok|beleza|ei|eps|pressa|socorro|help|hi|hello)\b|"
    r"\?\s*$",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})\b")
_NUMBER_RE = re.compile(r"\b(?:porta|vers[aã]o|v)?\s*[:=]?\s*(\d{2,5}(?:\.\d+)*)\b", re.IGNORECASE)
_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/)[^\s,;]+")
_SLOT_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _db_for(scope: Optional[str] = None):
    """Resolve o DB do escopo sem trocar o escopo ativo — o log roda em
    background durante o turno e mexer no ponteiro global corruptoria outras
    handlers (mesmo padrão de memory_episodes.py)."""
    from memory_state import state
    name = (scope or "").strip()
    if name:
        sc = state.scopes.get(name)
        if sc is not None:
            return sc.lt_db
    return state.scope.lt_db


def _sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", "replace")).hexdigest()


def _hash_text(text: str) -> str:
    return _sha1((text or "").strip().lower())


def _clip(text: str) -> str:
    text = (text or "").strip()
    if not MEMORY_TRIAGE_LOG_TEXT:
        return ""
    limit = max(0, int(MEMORY_TRIAGE_MAX_TEXT_CHARS))
    return text[:limit] if limit else text


def _normalize(text: str) -> str:
    """Lowercase, sem pontuação e sem espaços repetidos — usado só no
    alinhamento evidence↔segmento, nunca no texto gravado."""
    text = re.sub(r"[^\w\s]", " ", (text or "").lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def align_evidence(evidence: str, segments: list[str]) -> tuple[Optional[int], str]:
    """Alinha a evidence literal do LLM a um segmento do turno.

    Substring exata primeiro; se falhar,similaridade por SequenceMatcher com
    limiar 0.75. `none` NÃO é rótulo negativo: é segmento suprimido.
    """
    needle = _normalize(evidence)
    if not needle:
        return None, "none"
    normalized = [_normalize(s) for s in segments]
    for i, seg in enumerate(normalized):
        if seg and needle in seg:
            return i, "exact"
    best_i, best = None, 0.0
    for i, seg in enumerate(normalized):
        if not seg:
            continue
        ratio = difflib.SequenceMatcher(None, needle, seg).ratio()
        if ratio > best:
            best_i, best = i, ratio
    if best_i is not None and best >= _ALIGN_MIN_RATIO:
        return best_i, "fuzzy"
    return None, "none"


def split_segments(user_input: str, assistant_response: str) -> list[dict]:
    """Segmenta os dois lados da troca e anota role/tamanho/deixis."""
    from memory_search import _split_query_segments
    out: list[dict] = []
    for role, text in (("user", user_input), ("assistant", assistant_response)):
        for seg in _split_query_segments(text or ""):
            out.append({
                "index": len(out),
                "text": seg,
                "role": role,
                "n_chars": len(seg.strip()),
                "has_deictic": int(bool(_DEICTIC_RE.search(seg))),
                "hash": _sha1(seg.strip().lower()),
            })
    return out


def suppression_reason(seg: dict) -> str:
    """Causa provável da supressão. Política = regra correta do extrator;
    budget = viés a corrigir nos variants relaxados (§5 do planejamento)."""
    text = seg.get("text") or ""
    if seg.get("has_deictic") or _POLICY_RE.search(text) or seg.get("n_chars", 0) < 20:
        return "policy"
    return "budget"


def _infer_frame(text: str, fact_type: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ("prefere", "gosta", "prefiro", "prefers", "likes")):
        return "PREFERS"
    if re.search(r"\bv?\d+(?:\.\d+)+\b|\bvers[aã]o\b|\bversion\b", t):
        return "VERSION"
    if any(k in t for k in ("porta", "config", "caminho", "arquivo", "path", "port ")):
        return "CONFIG"
    if any(k in t for k in ("decidiu", "decisão", "decided", "vamos usar")):
        return "DECISION"
    if any(k in t for k in ("aconteceu", "evento", "install", "instalou")):
        return "EVENT"
    if fact_type in ("preference", "decision", "instruction"):
        return fact_type.upper()
    return "GENERIC"


def _infer_slots(segment_text: str, evidence: str, rendered_text: str) -> dict:
    """Rótulos fracos do renderizador (Fase 3): o que mudou entre o trecho
    literal e o texto escrito, mais números/caminhos do próprio segmento."""
    source = segment_text or evidence
    extra = rendered_text
    entity = ""
    if extra and _normalize(extra) != _normalize(source):
        entity = _clip(extra)[:80]
    values = _NUMBER_RE.findall(source) or _NUMBER_RE.findall(rendered_text or "")
    path = _PATH_RE.search(source)
    value = (path.group(0) if path else (values[0] if values else ""))
    date_match = _SLOT_DATE_RE.search(extra) or _SLOT_DATE_RE.search(source)
    return {
        "slot_entity": entity[:120],
        "slot_value": value[:120],
        "slot_date": date_match.group(1) if date_match else "",
    }


def gate_fields(result: dict) -> dict:
    """Traduz a resposta de um `/write` em rótulo LOCAL do gate cross-encoder.

    Não substitui `applied_label`: é sinal auxiliar com peso 0.6 (§4/§5.2.4).
    """
    reason = str(result.get("reason") or "")
    if reason.startswith("possible_update:"):
        label, why = "UPDATE", "gate_possible_update"
    elif "duplicate" in reason.lower():
        label, why = "NOOP", "gate_exact_duplicate"
    elif result.get("stored"):
        label, why = "ADD", "gate_stored"
    else:
        return {
            "gate_label": "", "gate_reason": reason[:80] or "gate_no_result",
            "gate_candidate_id": None, "gate_score": None,
        }
    cid = result.get("candidate_id")
    # candidate_score = bi-encoder, candidate_rerank_score = cross-encoder —
    # é o segundo que importa para a cabeça B (caro de recomputar).
    score = result.get("candidate_rerank_score")
    if score is None:
        score = result.get("candidate_score")
    return {
        "gate_label": label,
        "gate_reason": why,
        "gate_candidate_id": int(cid) if cid is not None else None,
        "gate_score": float(score) if score is not None else None,
    }


def is_probe_worthy(seg: dict) -> bool:
    """Segmento suprimido que vale um probe (§5.2.3): substancial, do usuário
    e sem dêixis/política — sondar 'estou com pressa' só gastaria API."""
    return (
        bool(MEMORY_TRIAGE_PROBE)
        and int(seg.get("n_chars") or 0) >= int(MEMORY_TRIAGE_SEGMENT_MIN_CHARS)
        and seg.get("role") == "user"
        and not int(seg.get("has_deictic") or 0)
        and suppression_reason(seg) == "budget"
    )


def synthetic_perturbations(text: str, evidence: str = "") -> list[dict]:
    """Perturbações determinísticas de um fato (§4). O teacher rotula; só
    aceitamos se o rótulo bater com o esperado. Nunca inventa dados do zero:
    parte sempre do texto/evidência emitidos pelo próprio extrator."""
    base = (evidence or text or "").strip()
    nums = re.findall(r"\b\d{2,5}(?:\.\d+)*\b", base)
    if not nums:
        return []
    n = nums[0]
    try:
        bumped = str(int(float(n)) + 1)
    except ValueError:
        bumped = n + "1"
    newer = base.replace(n, bumped, 1)
    return [
        {"kind": "version", "text": f"Agora é {bumped}. {newer}", "expected": "UPDATE"},
        {"kind": "conflict", "text": f"Na verdade é {bumped}; o de {n} era o antigo. {newer}", "expected": "CONFLICT"},
        {"kind": "noop", "text": f"{base} — continua em {n}.", "expected": "NOOP"},
        {"kind": "discard", "text": "Estou com pressa para testar isso.", "expected": "DISCARD"},
    ]


def probe_samples(ctx: dict, results: list[dict]) -> list[dict]:
    """Converte respostas de probes por segmento em samples rotuladas.

    `results`: [{"segment_index", "text", "keep": bool, "reason": str}].
    Cada probe é uma linha distinta por `teacher_variant`, para que o UNIQUE
    não colida com as amostras do turno base.
    """
    out: list[dict] = []
    segments = ctx.get("segments") or []
    for r in results:
        idx = int(r.get("segment_index") or 0)
        seg = next((s for s in segments if int(s["index"]) == idx), None)
        if seg is None:
            continue
        keep = bool(r.get("keep"))
        text = (r.get("text") or "").strip()
        row_ctx = {
            **ctx,
            "segments": [seg],
            "facts": [{
                "text": text,
                "evidence": "",
                "type": "fact",
                "validity": "temporal",
                "proposed_label": "KEEP" if keep else "DISCARD",
                "applied_label": "KEEP" if keep else ("DISCARD" if r.get("reason") else "unlabeled"),
                "applied_reason": str(r.get("reason") or "probe_no_output")[:80],
                "target_ids": [], "related_ids": [],
            }],
            "created_at": ctx.get("created_at") or time.time(),
        }
        built = build_samples(row_ctx)
        if built:
            if "weight" in ctx:
                built[0]["weight"] = float(ctx["weight"])
            out.append(built[0])
    _STATS["triage_probe_segments_total"] += len(out)
    return out


def build_samples(ctx: dict) -> list[dict]:
    """Constrói as linhas de `memory_triage_sample` a partir do estado do
    extrator. `ctx` é montado em LLM.py no hook entre validação de IDs e
    roteamento, completado com os resultados aplicados das escritas.

    Chaves de ctx: turn_id, session_id, scope, segments, facts, candidates,
    m_map, variant, propensity, origin, created_at, embed_model_version.
    """
    segments = ctx.get("segments") or []
    candidates = ctx.get("candidates") or []
    now = float(ctx.get("created_at") or time.time())
    variant = ctx.get("teacher_variant") or "base"
    propensity = float(ctx.get("propensity") or 1.0)
    origin = ctx.get("origin") or "live"
    embed_version = ctx.get("embed_model_version") or ""
    samples: list[dict] = []
    seen: set[tuple] = set()

    def _add(row: dict):
        key = (row["segment_hash"], _NO_CANDIDATE if row.get("candidate_id") is None else row["candidate_id"], variant)
        if key in seen:
            return
        seen.add(key)
        row.update({
            "turn_id": ctx.get("turn_id") or "",
            "session_id": ctx.get("session_id") or "",
            "scope": ctx.get("scope") or "",
            "teacher_variant": variant,
            "propensity": propensity,
            "origin": origin,
            "embed_model_version": embed_version,
            "created_at": now,
        })
        samples.append(row)

    for fact in ctx.get("facts") or []:
        if not isinstance(fact, dict):
            continue
        evidence = (fact.get("evidence") or "").strip()
        rendered = (fact.get("text") or "").strip()
        seg_i, aligned = align_evidence(evidence, [s["text"] for s in segments])
        targets = [int(x) for x in (fact.get("target_ids") or []) if isinstance(x, int)]
        related = [int(x) for x in (fact.get("related_ids") or []) if isinstance(x, int)]
        # Linhas de PAR (§3): todos os target/related + top-K por cosine.
        # `score` do /read é proxy do cosine; o cross_encoder_score real vem
        # do gate quando disponível. `_add` deduplica por
        # (segmento, candidato, variant).
        pair_candidates: list[Optional[dict]] = []
        for cid in targets + related:
            cand = next((c for c in candidates if c.get("id") == cid), None)
            if cand is not None and cand not in pair_candidates:
                pair_candidates.append(cand)
        for cand in sorted(
            candidates, key=lambda c: float(c.get("score") or 0.0), reverse=True
        )[:max(1, int(MEMORY_TRIAGE_PAIR_TOP_K))]:
            if cand not in pair_candidates:
                pair_candidates.append(cand)
        if not pair_candidates:
            pair_candidates.append(None)
        if seg_i is None:
            # Evidência não ancorada: não vira negativo. Continua como sample
            # suprimida, para entrar na fila de probes (§5.2).
            if not segments:
                continue
            seg = segments[0]
            aligned = "none"
        else:
            seg = segments[seg_i]

        slots = _infer_slots(seg["text"], evidence, rendered) if MEMORY_TRIAGE_RENDER_LABELS else {}
        frame = _infer_frame(rendered, fact.get("type") or "fact") if MEMORY_TRIAGE_RENDER_LABELS else ""
        render_source = ""
        if rendered:
            render_source = "verbatim" if _normalize(rendered) == _normalize(seg["text"]) else "llm"
        applied = fact.get("applied_label") or "unlabeled"
        for cand in pair_candidates:
            row = {
                "segment_index": int(seg["index"]),
                "segment_text": _clip(seg["text"]),
                "segment_hash": seg["hash"],
                "role": seg["role"],
                "n_chars": seg["n_chars"],
                "has_deictic": seg["has_deictic"],
                "candidate_id": (cand or {}).get("id"),
                "candidate_text_snapshot": _clip((cand or {}).get("text") or ""),
                "candidate_created_at": (cand or {}).get("created_at"),
                "cosine": (cand or {}).get("score"),
                "cross_encoder_score": fact.get("cross_encoder_score"),
                "used_in_answer": int(bool((cand or {}).get("used_in_answer"))),
                "rank": fact.get("rank"),
                "teacher_emitted": 1,
                "suppression_reason": "",
                "proposed_label": fact.get("proposed_label") or "",
                "proposed_targets": json.dumps(targets),
                "proposed_type": fact.get("type") or "fact",
                "proposed_validity": fact.get("validity") or "temporal",
                "applied_label": applied,
                "applied_reason": fact.get("applied_reason") or "",
                "frame_label": frame,
                "fragment_index": seg_i,
                "slot_entity": slots.get("slot_entity", ""),
                "slot_value": slots.get("slot_value", ""),
                "slot_date": slots.get("slot_date", ""),
                "rendered_text": _clip(rendered),
                "render_source": render_source,
                "evidence_aligned": aligned,
                "gate_label": fact.get("gate_label") or "",
                "gate_reason": fact.get("gate_reason") or "",
                "gate_candidate_id": fact.get("gate_candidate_id"),
                "gate_score": fact.get("gate_score"),
                "origin": origin,
                "weight": float(fact.get("weight") or 1.0),
            }
            _add(row)

    # Todo segmento sem evidence ancorada é suprimido — nunca DISCARD.
    emitted_hashes = {s["segment_hash"] for s in samples if s.get("teacher_emitted")}
    for seg in segments:
        if seg["hash"] in emitted_hashes:
            continue
        # Segmentos sem fato também geram LINHAS DE PAR (§3), todas com
        # applied_label='unlabeled' — candidatos a probe, nunca negativos.
        top = sorted(
            candidates, key=lambda c: float(c.get("score") or 0.0), reverse=True
        )[:max(1, int(MEMORY_TRIAGE_PAIR_TOP_K))] or [None]
        for cand in top:
            _add({
                "segment_index": int(seg["index"]),
                "segment_text": _clip(seg["text"]),
                "segment_hash": seg["hash"],
                "role": seg["role"],
                "n_chars": seg["n_chars"],
                "has_deictic": seg["has_deictic"],
                "candidate_id": (cand or {}).get("id"),
                "candidate_text_snapshot": _clip((cand or {}).get("text") or ""),
                "candidate_created_at": (cand or {}).get("created_at"),
                "cosine": (cand or {}).get("score"),
                "used_in_answer": int(bool((cand or {}).get("used_in_answer"))),
                "teacher_emitted": 0,
                "suppression_reason": suppression_reason(seg),
                "applied_label": "unlabeled",
                "applied_reason": "teacher_not_emitted",
                "evidence_aligned": "none",
                "weight": 0.0,
            })

    _STATS["triage_suppressed_total"] += sum(1 for s in samples if not s.get("teacher_emitted"))
    for s in samples:
        key = "triage_alignment_" + str(s.get("evidence_aligned") or "none")
        if key in _STATS:
            _STATS[key] += 1
    return samples


def _upsert(db, table: str, row: dict, conflict: list[str]) -> bool:
    cols = list(row)
    if not cols:
        return False
    placeholders = ",".join("?" for _ in cols)
    updates = ",".join(f"{c}=excluded.{c}" for c in cols if c not in conflict)
    sql = (
        f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT({','.join(conflict)}) DO UPDATE SET {updates}"
    )
    with db._lock:
        db._conn.execute(sql, [row[c] for c in cols])
    return True


_TURN_FIELDS = (
    "turn_id", "session_id", "scope", "created_at", "extractor_model",
    "fallback_model", "attempt_index", "prompt_version", "schema_version",
    "budget_variant", "propensity", "latency_ms", "user_hash", "assistant_hash",
    "n_facts", "n_segments", "n_suppressed", "n_candidates", "status",
    "episode_importance", "task_status",
)
_SAMPLE_FIELDS = (
    "turn_id", "session_id", "scope", "segment_index", "segment_text",
    "segment_hash", "role", "n_chars", "has_deictic", "candidate_id",
    "candidate_text_snapshot", "candidate_created_at", "cosine",
    "cross_encoder_score", "used_in_answer", "rank", "teacher_variant",
    "teacher_emitted", "suppression_reason", "propensity", "proposed_label",
    "proposed_targets", "proposed_type", "proposed_validity", "applied_label",
    "applied_reason", "frame_label", "fragment_index", "slot_entity",
    "slot_value", "slot_date", "rendered_text", "render_source",
    "evidence_aligned", "origin", "weight", "embed_model_version",
    "shadow_pred", "gate_label", "gate_reason", "gate_candidate_id",
    "gate_score", "created_at",
)


def _prune_retention(db) -> None:
    days = int(MEMORY_TRIAGE_RETENTION_DAYS)
    if days <= 0:
        return
    cutoff = time.time() - days * 86400
    with db._lock:
        db._conn.execute("DELETE FROM memory_triage_sample WHERE created_at < ?", (cutoff,))
        db._conn.execute("DELETE FROM memory_triage_turn WHERE created_at < ?", (cutoff,))


async def triage_log(req: TriageLogRequest) -> TriageLogResponse:
    """Upsert idempotente de turn + samples. Best-effort: nunca propaga
    exceção para o extrator."""
    if not MEMORY_TRIAGE_LOG:
        return TriageLogResponse(stored=False, reason="triage_disabled")
    turn_id = (req.turn.turn_id or "").strip()
    session_id = (req.turn.session_id or "").strip()
    if not turn_id or not session_id:
        return TriageLogResponse(stored=False, reason="missing_session_or_turn")
    try:
        db = _db_for(req.turn.scope)
        payload = req.turn.model_dump()
        payload["created_at"] = float(payload.get("created_at") or time.time())
        turn_row = {k: payload.get(k) for k in _TURN_FIELDS}
        _upsert(db, "memory_triage_turn", turn_row, ["session_id", "turn_id"])

        inserted = 0
        for s in req.samples:
            data = s.model_dump()
            # O modelo de sample não carrega turn/session/scope (vêm do turn
            # do mesmo payload); preenche aqui para não depender do chamador.
            data["turn_id"] = turn_id
            data["session_id"] = session_id
            data["scope"] = req.turn.scope or ""
            # list[int] vai como JSON determinístico para a coluna TEXT.
            if isinstance(data.get("proposed_targets"), list):
                data["proposed_targets"] = json.dumps(data["proposed_targets"])
            data["created_at"] = float(data.get("created_at") or payload["created_at"])
            data["candidate_id"] = data.get("candidate_id") or _NO_CANDIDATE
            row = {k: data.get(k) for k in _SAMPLE_FIELDS}
            if _upsert(db, "memory_triage_sample", row,
                       ["turn_id", "segment_hash", "candidate_id", "teacher_variant"]):
                inserted += 1
        _prune_retention(db)
        _STATS["triage_logged_total"] += 1
        return TriageLogResponse(stored=True, turn_id=turn_id, samples_logged=inserted)
    except Exception as e:
        _STATS["triage_log_failures"] += 1
        log.info(f"[MEMORY-TRIAGE] log falhou (não-bloqueante): {type(e).__name__}: {e}")
        return TriageLogResponse(stored=False, turn_id=turn_id, reason=f"{type(e).__name__}: {e}")


async def triage_labels(
    limit: int = 500,
    since: Optional[float] = None,
    scope: Optional[str] = None,
    origin: Optional[str] = None,
    variant: Optional[str] = None,
    include_unlabeled: bool = False,
) -> TriageLabelsResponse:
    """Exportação para treino. `include_unlabeled=0` EXCLUI as linhas; =1 as
    inclui marcadas como `unlabeled` — nunca convertidas em negativo (PU)."""
    db = _db_for(scope)
    clauses: list[str] = []
    params: list[object] = []
    if scope:
        clauses.append("scope = ?")
        params.append(scope)
    if since is not None:
        clauses.append("created_at >= ?")
        params.append(float(since))
    if origin:
        clauses.append("origin = ?")
        params.append(origin)
    if variant:
        clauses.append("teacher_variant = ?")
        params.append(variant)
    if not include_unlabeled:
        clauses.append("applied_label != 'unlabeled'")
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    cap = max(1, min(int(limit or 500), 10000))
    rows = db._conn.execute(
        f"SELECT * FROM memory_triage_sample{where} ORDER BY created_at DESC LIMIT ?",
        [*params, cap + 1],
    ).fetchall()
    samples = []
    for r in rows[:cap]:
        d = dict(r)
        d["candidate_id"] = None if d.get("candidate_id") in (None, _NO_CANDIDATE) else d["candidate_id"]
        samples.append(d)
    return TriageLabelsResponse(
        samples=samples, total=len(samples), has_more=len(rows) > cap,
    )


async def triage_gold_write(req: TriageGoldRequest) -> TriageGoldResponse:
    """Registra uma revisão humana. Gold set é IMUTÁVEL: não sobrescreve."""
    if not (req.reviewed_label or "").strip():
        return TriageGoldResponse(stored=False, sample_id=req.sample_id, reason="empty_label")
    db = _db_for(req.scope)
    try:
        with db._lock:
            exists = db._conn.execute(
                "SELECT 1 FROM memory_triage_gold WHERE sample_id = ?", (req.sample_id,)
            ).fetchone()
            if exists:
                return TriageGoldResponse(
                    stored=False, sample_id=req.sample_id, reason="immutable_already_reviewed"
                )
            sample = db._conn.execute(
                "SELECT origin FROM memory_triage_sample WHERE sample_id = ?", (req.sample_id,)
            ).fetchone()
            db._conn.execute(
                "INSERT INTO memory_triage_gold "
                "(sample_id, reviewed_label, reviewed_by, reviewed_at, notes, origin) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (req.sample_id, req.reviewed_label.strip(), req.reviewed_by or "",
                 time.time(), req.notes or "", (sample["origin"] if sample else "")),
            )
        _STATS["triage_gold_reviews_total"] += 1
        return TriageGoldResponse(stored=True, sample_id=req.sample_id)
    except Exception as e:
        log.info(f"[MEMORY-TRIAGE] gold write falhou: {type(e).__name__}: {e}")
        return TriageGoldResponse(stored=False, sample_id=req.sample_id, reason=str(e)[:80])


async def triage_gold_list(limit: int = 300,
                           scope: Optional[str] = None) -> TriageGoldListResponse:
    db = _db_for(scope)
    rows = db._conn.execute(
        "SELECT g.*, s.segment_text, s.role, s.proposed_label, s.applied_label, "
        "s.teacher_variant, s.origin, s.evidence_aligned, s.segment_hash "
        "FROM memory_triage_gold g LEFT JOIN memory_triage_sample s "
        "ON s.sample_id = g.sample_id ORDER BY g.reviewed_at DESC LIMIT ?",
        (max(1, min(int(limit or 300), 5000)),),
    ).fetchall()
    entries = [dict(r) for r in rows]
    return TriageGoldListResponse(entries=entries, total=len(entries))


async def triage_gold_candidates(limit: int = 300,
                                 scope: Optional[str] = None) -> TriageLabelsResponse:
    """Amostra estratificada para revisão: emitidas alinhadas + suprimidas
    substanciais + classes raras. Nunca devolve rótulos já revisados."""
    db = _db_for(scope)
    cap = max(1, min(int(limit or 300), 5000))
    clauses = [
        "(s.sample_id NOT IN (SELECT sample_id FROM memory_triage_gold))",
        "(s.teacher_emitted = 1 OR (s.teacher_emitted = 0 AND s.n_chars >= ?))",
    ]
    params: list[object] = [int(MEMORY_TRIAGE_SEGMENT_MIN_CHARS)]
    if scope:
        clauses.append("s.scope = ?")
        params.append(scope)
    rows = db._conn.execute(
        "SELECT s.* FROM memory_triage_sample s WHERE " + " AND ".join(clauses) +
        " ORDER BY s.teacher_emitted DESC, s.created_at DESC LIMIT ?",
        [*params, cap],
    ).fetchall()
    return TriageLabelsResponse(
        samples=[dict(r) for r in rows], total=len(rows),
    )


def _counts(db, sql: str, params: tuple = ()) -> dict[str, int]:
    try:
        return {str(r[0]): int(r[1]) for r in db._conn.execute(sql, params).fetchall()}
    except Exception:
        return {}


async def triage_stats(scope: Optional[str] = None) -> TriageStatsResponse:
    """Estatísticas do dataset + readiness (target configurável) para a WebUI."""
    db = _db_for(scope)
    where = " WHERE scope = ?" if scope else ""
    params: tuple = (scope,) if scope else ()
    one = lambda sql, p=params: int(db._conn.execute(sql, p).fetchone()[0] or 0)

    turns_total = one("SELECT COUNT(*) FROM memory_triage_turn" + where)
    samples_total = one("SELECT COUNT(*) FROM memory_triage_sample" + where)
    labelled = one(
        "SELECT COUNT(*) FROM memory_triage_sample" +
        (" WHERE scope = ? AND applied_label != 'unlabeled'" if scope else " WHERE applied_label != 'unlabeled'"),
    )
    unlabeled = samples_total - labelled
    emitted = one(
        "SELECT COUNT(*) FROM memory_triage_sample" +
        (" WHERE scope = ? AND teacher_emitted = 1" if scope else " WHERE teacher_emitted = 1"),
    )
    suppressed = samples_total - emitted
    render_labels = one(
        "SELECT COUNT(*) FROM memory_triage_sample" +
        (" WHERE scope = ? AND frame_label != ''" if scope else " WHERE frame_label != ''"),
    )

    suppression = _counts(db, (
        "SELECT suppression_reason, COUNT(*) FROM memory_triage_sample "
        "WHERE teacher_emitted = 0" + (" AND scope = ?" if scope else "") +
        " GROUP BY suppression_reason"
    ), params)
    alignment = _counts(db, (
        "SELECT evidence_aligned, COUNT(*) FROM memory_triage_sample" + where + " GROUP BY evidence_aligned"
    ), params)
    labels = _counts(db, (
        "SELECT applied_label, COUNT(*) FROM memory_triage_sample" + where + " GROUP BY applied_label"
    ), params)
    origins = _counts(db, (
        "SELECT origin, COUNT(*) FROM memory_triage_sample" + where + " GROUP BY origin"
    ), params)
    variants = _counts(db, (
        "SELECT teacher_variant, COUNT(*) FROM memory_triage_sample" + where + " GROUP BY teacher_variant"
    ), params)
    gate_labels = _counts(db, (
        "SELECT gate_label, COUNT(*) FROM memory_triage_sample" + where +
        " AND gate_label != '' GROUP BY gate_label"
    ) if scope else (
        "SELECT gate_label, COUNT(*) FROM memory_triage_sample "
        "WHERE gate_label != '' GROUP BY gate_label"
    ), params)
    gold_total = 0
    try:
        gold_total = int(db._conn.execute("SELECT COUNT(*) FROM memory_triage_gold").fetchone()[0] or 0)
    except Exception:
        pass

    # Taxa de emissão por posição no turno (início/meio/fim). Detecta o viés
    # de orçamento que esmaga os fatos do fim de respostas longas (§5.3).
    pos: dict[str, list[int]] = {k: [0, 0] for k in ("start", "middle", "end")}
    by_role: dict[str, list[int]] = {k: [0, 0] for k in ("user", "assistant")}
    by_size: dict[str, list[int]] = {k: [0, 0] for k in ("short", "medium", "long")}
    try:
        rows = db._conn.execute(
            "SELECT s.segment_index, s.teacher_emitted, t.n_segments, "
            "s.role, s.n_chars "
            "FROM memory_triage_sample s JOIN memory_triage_turn t "
            "ON t.session_id = s.session_id AND t.turn_id = s.turn_id" + where.replace("scope", "t.scope"),
            params,
        ).fetchall()
        for idx, emitted_flag, n_seg, role, n_chars in rows:
            n = max(1, int(n_seg or 1))
            frac = int(idx or 0) / n
            bucket = "start" if frac < 0.34 else ("middle" if frac < 0.67 else "end")
            flag = int(emitted_flag or 0)
            pos[bucket][0] += flag
            pos[bucket][1] += 1
            r = str(role or "user")
            by_role.setdefault(r, [0, 0])[0] += flag
            by_role[r][1] += 1
            size = "short" if int(n_chars or 0) < 40 else ("medium" if int(n_chars or 0) < 120 else "long")
            by_size[size][0] += flag
            by_size[size][1] += 1
    except Exception:
        pass
    rate = lambda d: {k: round(v[0] / v[1], 4) if v[1] else 0.0 for k, v in d.items()}
    emitted_rate = rate(pos)

    # recall_gap: |relaxado - normal| / relaxado, por turno pareado.
    try:
        variant_totals = _counts(db, (
            "SELECT budget_variant, SUM(n_facts) FROM memory_triage_turn" + where +
            " GROUP BY budget_variant"
        ), params)
        relaxed = float(variant_totals.get("relaxed", 0) or 0)
        normal = float(variant_totals.get("normal", 0) or 0)
        gap = abs(relaxed - normal) / relaxed if relaxed > 0 else 0.0
    except Exception:
        gap = 0.0
    _STATS["triage_recall_gap"] = round(gap, 4)

    target = int(MEMORY_TRIAGE_TARGET_SAMPLES)
    missing = max(0, target - labelled)
    readiness = TriageReadiness(
        target=target,
        labelled=labelled,
        progress=round(min(1.0, labelled / target), 4),
        ready=labelled >= target,
        missing=missing,
        reason=("dataset pronto para treinar a Fase 2" if labelled >= target
                else f"faltam {missing} rótulos para o alvo de {target}"),
    )

    return TriageStatsResponse(
        enabled=MEMORY_TRIAGE_LOG,
        turns_total=turns_total,
        samples_total=samples_total,
        labelled_total=labelled,
        unlabeled_total=unlabeled,
        teacher_emitted_total=emitted,
        suppressed_total=suppressed,
        suppressed_by_reason=suppression,
        alignment=alignment,
        label_counts=labels,
        origin_counts=origins,
        variant_counts=variants,
        gate_label_counts=gate_labels,
        emitted_rate_by_position=emitted_rate,
        emitted_rate_by_role=rate(by_role),
        emitted_rate_by_size=rate(by_size),
        render_labels_total=render_labels,
        recall_gap=round(gap, 4),
        gold_total=gold_total,
        gold_target=int(MEMORY_TRIAGE_GOLD_TARGET),
        gold_candidates_total=max(0, samples_total - gold_total),
        readiness=readiness,
        flags={
            "log": MEMORY_TRIAGE_LOG,
            "log_text": MEMORY_TRIAGE_LOG_TEXT,
            "probe": MEMORY_TRIAGE_PROBE,
            "probe_rate": MEMORY_TRIAGE_PROBE_RATE,
            "probe_max": MEMORY_TRIAGE_PROBE_MAX,
            "segment_probe": MEMORY_TRIAGE_PROBE,
            "contrastive": MEMORY_TRIAGE_CONTRASTIVE,
            "synthetic": MEMORY_TRIAGE_SYNTHETIC,
            "gate": MEMORY_TRIAGE_GATE,
            "gate_weight": MEMORY_TRIAGE_GATE_WEIGHT,
            "pair_top_k": MEMORY_TRIAGE_PAIR_TOP_K,
            "gold_target": MEMORY_TRIAGE_GOLD_TARGET,
            "synthetic_max": MEMORY_TRIAGE_SYNTHETIC_MAX,
            "pu": MEMORY_TRIAGE_PU,
            "render_labels": MEMORY_TRIAGE_RENDER_LABELS,
            "retention_days": MEMORY_TRIAGE_RETENTION_DAYS,
            "shadow": MEMORY_TRIAGE_SHADOW,
            "prompt_version": MEMORY_TRIAGE_PROMPT_VERSION,
            "schema_version": MEMORY_TRIAGE_SCHEMA_VERSION,
        },
        counters={k: v for k, v in _STATS.items()},
    )


async def triage_delete(scope: Optional[str] = None) -> dict:
    """Apaga o log (privacidade/retention). Sem scope = todos os escopos."""
    if scope:
        dbs = [_db_for(scope)]
    else:
        from memory_state import state
        dbs = [sc.lt_db for sc in list(state.scopes.values())] + [state.scope.lt_db]
    seen_ids: set[int] = set()
    deleted = 0
    for db in dbs:
        if id(db) in seen_ids:
            continue
        seen_ids.add(id(db))
        try:
            with db._lock:
                # Gold referencia sample_id — apaga primeiro para não deixar
                # órfãos apontando para linhas removidas.
                db._conn.execute(
                    "DELETE FROM memory_triage_gold WHERE sample_id IN "
                    "(SELECT sample_id FROM memory_triage_sample WHERE scope = ?)",
                    (scope or "",),
                ) if scope else db._conn.execute("DELETE FROM memory_triage_gold")
                if scope:
                    cur = db._conn.execute("DELETE FROM memory_triage_sample WHERE scope = ?", (scope,))
                    deleted += cur.rowcount
                    cur = db._conn.execute("DELETE FROM memory_triage_turn WHERE scope = ?", (scope,))
                    deleted += cur.rowcount
                else:
                    deleted += db._conn.execute("DELETE FROM memory_triage_sample").rowcount
                    deleted += db._conn.execute("DELETE FROM memory_triage_turn").rowcount
        except Exception as e:
            log.info(f"[MEMORY-TRIAGE] delete falhou: {type(e).__name__}: {e}")
    return {"deleted": deleted}


__all__ = [
    "align_evidence",
    "build_samples",
    "gate_fields",
    "is_probe_worthy",
    "probe_samples",
    "split_segments",
    "suppression_reason",
    "synthetic_perturbations",
    "triage_delete",
    "triage_gold_candidates",
    "triage_gold_list",
    "triage_gold_write",
    "triage_labels",
    "triage_log",
    "triage_stats",
    "_STATS",
]
