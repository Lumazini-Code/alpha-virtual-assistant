#!/usr/bin/env python3
# Alpha AI (AVA) — Replay de turnos duráveis para o dataset de triagem
# Copyright (C) 2026  Felipe Lumazini
"""Replay de turnos antigos pelo extrator, SEM gravar memórias.

Fase 1 §4: replay serve para aprender a distribuição de features, não para
rótulos confiáveis de UPDATE/CONFLICT. Cada turno reprocessado entra com
`origin=replay` e peso 0.4. O recorte de chain-of-thought entra separado com
`origin=probe_thinking`, peso 0.3 e `applied_label='unlabeled'` (nunca vira
rótulo aplicado). Os variants `replay`/`thinking` são somente leitura no
LLM.py — nenhuma memória ou episódio é escrito.

Uso:
    python scripts/triage_replay.py --limit 200
    python scripts/triage_replay.py --mode thinking --session <id>
    python scripts/triage_replay.py --dry-run --limit 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Modules"))

import httpx  # noqa: E402

from session import (  # noqa: E402
    SessionStore, get_task_state, message_text, split_turns,
)

LLM_URL = os.getenv("AVA_LLM_URL", "http://127.9.12.22:4003")


def _headers() -> dict:
    tok = os.getenv("AVA_INTERNAL_TOKEN", "").strip()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _final_answer(turn: list[dict]) -> str:
    """Última mensagem do assistente com texto final (sem tool_calls)."""
    answer = ""
    for m in turn:
        if m.get("role") != "assistant":
            continue
        if m.get("tool_calls"):
            continue
        content = message_text(m.get("content"))
        if content.strip():
            answer = content.strip()
    return answer


def _user_text(turn: list[dict]) -> str:
    for m in turn:
        if m.get("role") == "user":
            text = message_text(m.get("content")).strip()
            if text:
                return text
    return ""


def _thinking_cut(turn: list[dict], max_chars: int = 800) -> str:
    """Recorte MÍNIMO do CoT — nunca entra no log de produção, só no probe."""
    parts = []
    for m in turn:
        r = m.get("reasoning")
        if isinstance(r, str) and r.strip() and "$$" not in r:
            parts.append(r.strip())
    return ("\n".join(parts))[:max_chars].strip()


def _turn_seqs(store: SessionStore, sid: str) -> list[int]:
    seqs = []
    for e in store.entries(sid):
        if str(e.get("type") or "").startswith("compaction"):
            continue
        try:
            seqs.append(int(e["seq"]))
        except (KeyError, TypeError, ValueError):
            continue
    return seqs


def _build_payloads(store: SessionStore, sid: str, session: dict,
                    mode: str, min_chars: int) -> list[dict]:
    messages = [e["message"] for e in store.entries(sid)]
    turns = split_turns(messages)
    seqs = _turn_seqs(store, sid)
    out: list[dict] = []
    scope = session.get("project_id") or ""
    task_state = get_task_state(store, sid)
    for i, turn in enumerate(turns):
        user = _user_text(turn)
        answer = _final_answer(turn)
        if not user or len(user) < min_chars:
            continue
        seq = seqs[i] if i < len(seqs) else i
        base = {
            "user_input": user,
            "session_id": sid,
            **({"scope": scope} if scope and scope != "general" else {}),
            **({"task_state": task_state} if task_state else {}),
        }
        if mode in ("replay", "both") and answer:
            out.append({
                **base,
                "assistant_response": answer,
                "turn_id": f"replay:{sid}:{seq}",
                "probe": "replay",
                "propensity": 0.4,
            })
        if mode in ("thinking", "both"):
            cut = _thinking_cut(turn)
            if cut:
                out.append({
                    **base,
                    "assistant_response": cut,
                    "turn_id": f"probe-thinking:{sid}:{seq}",
                    "probe": "thinking",
                    "propensity": 0.3,
                })
    return out


async def _send(client: httpx.AsyncClient, payload: dict) -> tuple[bool, str]:
    try:
        r = await client.post("/memories/extract", json=payload)
        if r.status_code == 200:
            return True, ""
        return False, f"HTTP {r.status_code}: {r.text[:120]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm", default=LLM_URL, help="URL do LLM.py (:4003)")
    ap.add_argument("--limit", type=int, default=100, help="máximo de sessões")
    ap.add_argument("--session", default=None, help="reprocessa só esta sessão")
    ap.add_argument("--mode", choices=["replay", "thinking", "both"], default="both")
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--sample", type=float, default=1.0, help="fração de turnos (0-1)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="não reenvia turnos já presentes no log (retomada)")
    ap.add_argument("--memory", default=os.getenv("MEMORY_API_URL", "http://127.9.12.22:3000"))
    ap.add_argument("--dry-run", action="store_true", help="só imprime os payloads")
    args = ap.parse_args()

    store = SessionStore()
    if args.session:
        s = store.get(args.session)
        sessions = [s] if s else []
    else:
        sessions = [
            s for s in store.list_sessions(limit=max(1, args.limit), root_only=True)
            if not s.get("parent_id")
        ]

    payloads: list[dict] = []
    for s in sessions:
        payloads.extend(_build_payloads(store, s["id"], s, args.mode, args.min_chars))

    if args.sample < 1.0:
        payloads = [p for p in payloads if random.random() < args.sample]

    if args.skip_existing:
        existing: set[str] = set()
        try:
            async with httpx.AsyncClient(
                base_url=args.memory.rstrip("/"), headers=_headers(), timeout=30.0,
            ) as mem:
                for origin in ("replay", "probe_thinking"):
                    r = await mem.get("/triage/labels", params={
                        "origin": origin, "limit": 10000, "include_unlabeled": "true",
                    })
                    if r.status_code == 200:
                        existing.update(
                            str(s.get("turn_id") or "") for s in r.json().get("samples", [])
                        )
        except Exception as e:
            print(f"aviso: não foi possível consultar rótulos existentes: {e}", file=sys.stderr)
        before = len(payloads)
        payloads = [p for p in payloads if p["turn_id"] not in existing]
        print(f"triagem replay: {before - len(payloads)} já registrado(s) ignorado(s)")

    print(f"triagem replay: {len(sessions)} sessão(ões) → {len(payloads)} payload(s)")
    if args.dry_run:
        for p in payloads[:5]:
            print(json.dumps(p, ensure_ascii=False)[:600])
        return

    ok = 0
    fail = 0
    async with httpx.AsyncClient(
        base_url=args.llm.rstrip("/"), headers=_headers(),
        timeout=httpx.Timeout(120.0, connect=10.0),
    ) as client:
        for p in payloads:
            sent, err = await _send(client, p)
            if sent:
                ok += 1
            else:
                fail += 1
                print(f"falha {p['turn_id']}: {err}", file=sys.stderr)
            await asyncio.sleep(0)  # mantém o event loop responsivo
    print(f"triagem replay: {ok} enviado(s), {fail} falha(s)")
    print("Verifique com: GET /triage/labels?origin=replay (ou origin=probe_thinking)")


if __name__ == "__main__":
    asyncio.run(main())
