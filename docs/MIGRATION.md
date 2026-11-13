# Migração e cutover

## Pré-requisitos

- Backup dos bancos antigos:
  ```bash
  cp Modules/memory/db/ava_memory.db     Modules/memory/db/ava_memory.db.bak
  cp Modules/memory/db/ava_short_term.db Modules/memory/db/ava_short_term.db.bak
  ```
- Parar o orchestrator/LLM antes de rodar a migração.

## 1. Dry-run da migração

```bash
python scripts/migrate_st_to_sessions.py --dry-run
```

Confira `grupos`, `sessões` e `mensagens`. O script é idempotente
(`metadata.st_group_id`).

## 2. Migração real

```bash
python scripts/migrate_st_to_sessions.py
```

Valide uma amostra:

```bash
python - <<'PY'
from Modules.session import SessionStore
s = SessionStore()
for session in s.list_sessions(10):
    print(session["id"], session["title"])
    for message in s.load_messages(session["id"])[-6:]:
        print(" ", message.get("role"), str(message.get("content"))[:80])
s.close()
PY
```

## 3. Flags

Todas as features novas estão **ligadas por padrão** (regra do projeto):

| Flag | Default | Efeito |
|------|---------|--------|
| `AVA_SESSION_STORE` | `1` | session store durável |
| `AVA_COMPACTION` | `1` | compactação de contexto |
| `AVA_INSTRUCTIONS` | `1` | AGENTS.md versionado |
| `AVA_TOOL_REGISTRY` | `1` | registry unificado |
| `AVA_CODEMODE` | `1` | Code Mode |
| `AVA_SKILL_CATALOG` | `1` | catálogo de skills |
| `AVA_SESSION_RUNNER` | `1` | recarga por step + output limit |
| `AVA_SUBAGENTS` | `1` | subagentes em background |

Rollback pontual: `AVA_<FLAG>=0`. O tool loop usa o session store como fonte
única; `AVA_SESSION_STORE=0` afeta apenas o modo CHAT do `LLM.py`.

## 4. Validação em produção

- `GET /status` — serviços e métricas.
- `GET /metrics` — Prometheus (tokens, cache ratio, tools, compactações).
- `GET /session/{id}/messages` — histórico persistido.
- `GET /session/{id}/usage` — tokens/custo por sessão.
- `GET /jobs` — subagentes/jobs.

## 5. Observação

Manter dual-write por um ciclo de release. Só remover o ST antigo quando:

- migração validada;
- `/metrics` sem regressão de latência;
- logs sem `session store` falhando;
- 2 releases com as flags ligadas.

## 6. Remoção posterior

Após a validação:

- remover `ShortTermDB`, `st_index`, `/write-short-term`, `/read-short-term`;
- remover `_save_turn`, `_recall_for_turn`, `_sanitize_st_turns` do caminho
  principal;
- remover `TOOLS`/`mcp_dispatch` antigos quando o registry estiver estável;
- atualizar README/AGENTS.md.

## Rollback

1. Restaurar os backups se necessário.
2. O orchestrator usa o session store como fonte única; não há mais rollback
   para o ST antigo no tool loop. `AVA_SESSION_STORE=0` afeta apenas o modo
   CHAT do `LLM.py`.
3. `AVA_SESSION_RUNNER=0` desliga recarga por step/continuação de output.
4. O banco `Modules/session/db/ava_sessions.db` pode ficar no lugar — ele é
   ignorado pelo git.
