# Session store (durável)

Log de eventos + store de sessões/mensagens inspirado no OpenCode. Ainda
**não substitui** o ST memory como fonte de leitura; nesta fase o
orchestrator faz dual-write.

## Configuração

Ativo por padrão. Para desligar:

```bash
export AVA_SESSION_STORE=0
# opcional: caminho do SQLite (default: Modules/session/db/ava_sessions.db)
export AVA_SESSION_DB_PATH=/caminho/ava_sessions.db
```

## Estrutura

- `db.py` — conexão SQLite, schema e transações.
- `events.py` — tipos de evento (`session.message.appended`, `session.step.*`...).
- `projector.py` — traduz evento em linha durável.
- `store.py` — API de sessão/mensagem/usage.
- `compaction.py` — estimativa de tokens, resumo e checkpoint (`keepID`).
- `history.py` — leitura para o runner (boundary por compactação).
- `ids.py` — `ses_`, `msg_`, `part_`.

## Endpoints de inspeção

Com o orchestrator rodando:

- `GET /sessions` — sessões recentes.
- `GET /session/{id}/messages` — histórico durável.
- `GET /session/{id}/usage` — tokens/custo acumulados.

## Compactação

Quando o request estimado passa do limiar, o histórico antigo é resumido e a
cauda recente é mantida verbatim:

- `AVA_COMPACTION=1` (default)
- `AVA_COMPACTION_CONTEXT_TOKENS` (default `128000`)
- `AVA_COMPACTION_BUFFER` (default `0.15`)
- `AVA_COMPACTION_KEEP_TOKENS` (default `15000`)

O resumo é gravado como uma mensagem `type=compaction` com
`metadata.keepID`; `SessionStore.entries()` reexpande como
`[resumo, ...cauda, ...mensagens novas]`. O resumo entra no system prompt via
`_with_compaction_summary()`, não como mensagem do histórico.

Compactação manual:

```bash
curl -X POST /session/<id>/compact
```

## Migração do ST antigo

```bash
python scripts/migrate_st_to_sessions.py --dry-run
python scripts/migrate_st_to_sessions.py
```

O script é idempotente: cada grupo migrado carrega `metadata.st_group_id`.

## Testes

```bash
python -m unittest discover -s tests/session -v
```

## Próximo passo

Trocar a leitura do ST por `SessionStore.load_messages()` e remover o
dedup/truncamento do caminho principal.
