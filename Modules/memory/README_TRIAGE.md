# memory_triage — instrumentação do extrator (Fase 1)

Companion de `memory_triage.py`. O planejamento completo está em
`docs/MEMORY_TRIAGE_PHASE1.md`; este arquivo registra o contrato implementado.

## O que este módulo faz

Não decide nada. Registra três estados distintos por segmento do turno:

| Estado | Coluna | Uso no treino |
|---|---|---|
| Proposto | `proposed_label` | divergência do teacher |
| Aplicado | `applied_label` | ground truth da cabeça B |
| Suprimido | `teacher_emitted=0`, `applied_label='unlabeled'` | desconhecido — **nunca** negativo |

`build_samples()` recebe o estado capturado no hook do `LLM.py` (segmentos,
fatos propostos/aplicados, candidatos) e produz as linhas. `triage_log()` faz
o upsert idempotente; `triage_labels()` exporta sem converter `unlabeled` em
negativo; `triage_stats()` devolve contagens, `recall_gap`,
`suppressed_by_reason`, `emitted_rate_by_position` e `readiness` (target
configurável, lido pela WebUI).

## Notas de implementação

- `candidate_id` ausente é gravado como `-1` (sentinel): SQLite não considera
  `NULL` duplicado em `UNIQUE`, então os segmentos suprimidos também precisam
  de chave de idempotência. A exportação devolve `null`.
- `proposed_targets` é JSON determinístico na coluna TEXT.
- Embeddings não são gravados — só `embed_model_version`; o texto do segmento
  pode ser desligado com `AVA_MEMORY_TRIAGE_LOG_TEXT=0` (hash-only).
- `frame_label`/`fragment_index`/slots são rótulos fracos inferidos de
  `evidence`↔`text` para o renderizador da Fase 3; `rendered_text` é apenas
  referência de paridade, não alvo de cópia token a token.
- `triage_delete(scope=...)` apaga o log por escopo (privacidade/retention).

## Flags

Ver tabela em `README.md` (todas opt-out, ligadas por padrão).

## Rotas

`POST /triage/log`, `GET /triage/labels`, `GET /triage/stats`,
`DELETE /triage/log` — todas no memory API `:3000`, protegidas pelo token
interno, proxeadas pela WebUI via `/memory/*`.
