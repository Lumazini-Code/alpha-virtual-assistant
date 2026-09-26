# Patch: log de ativação PPR em tempo real

## 1. Adicione perto das outras constantes (ex.: junto do bloco "Hebbian graph")

```python
# ── NEW: log de ativação p/ visualização em tempo real ────────────────────────
ACTIVATION_LOG_PATH = "./memory/activation_log.jsonl"
ACTIVATION_LOG_MAX_IDS = 60   # não loga o rank inteiro se o subgrafo for gigante
```

## 2. Adicione esta função helper (pode ir logo acima de `_ppr_spread_on`)

```python
def _log_activation_sync(seed_ids: list[int], rank: dict[int, float], elapsed_ms: float):
    """Grava um evento de ativação PPR no log JSONL. Best-effort: qualquer
    falha de IO aqui NUNCA deve derrubar uma leitura de memória."""
    try:
        top = sorted(rank.items(), key=lambda kv: kv[1], reverse=True)[:ACTIVATION_LOG_MAX_IDS]
        line = json.dumps({
            "ts": time.time(),
            "seeds": seed_ids,
            "rank": {str(k): round(v, 4) for k, v in top},
            "elapsed_ms": round(elapsed_ms, 2),
        }, ensure_ascii=False)
        Path(ACTIVATION_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(ACTIVATION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        log.debug("activation log: falha ao gravar (ignorado)", exc_info=True)
```

## 3. Dentro de `_ppr_spread_on`, logo depois de calcular `rank` (antes de `seed_set = set(seed_ids)`)

Onde está hoje:

```python
    rank = await loop.run_in_executor(
        None, personalized_pagerank,
        adjacency, personalization, PPR_DAMPING, PPR_MAX_ITER, PPR_CONVERGENCE_EPS,
    )
    seed_set = set(seed_ids)
```

Vira:

```python
    rank = await loop.run_in_executor(
        None, personalized_pagerank,
        adjacency, personalization, PPR_DAMPING, PPR_MAX_ITER, PPR_CONVERGENCE_EPS,
    )
    # ── NEW: dispara log de ativação sem bloquear o event loop nem o /read ──
    asyncio.create_task(
        loop.run_in_executor(None, _log_activation_sync, seed_ids, rank, (time.perf_counter() - t0) * 1000.0)
    )
    seed_set = set(seed_ids)
```

Pronto — nenhuma outra mudança necessária. Toda vez que uma leitura de memória
disparar propagação PPR (via `_ppr_spread` / `_dict_graph_related`), uma linha
é anexada a `./memory/activation_log.jsonl` sem impacto perceptível na latência.
