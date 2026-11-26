# Triagem de memória — Fase 3: autoridade local, LLM só para unificar

Documento de planejamento. Continuação de `MEMORY_TRIAGE_PHASE1.md` e
`MEMORY_TRIAGE_PHASE2.md`.

**Objetivo:** tirar o LLM do caminho da **adição** de memórias. A rede passa a ter
autoridade sobre os casos fáceis — e, para `ADD`, também passa a *escrever* o
fato, sem gerar texto livre: ela produz um **plano estruturado** que um
renderizador determinístico converte em texto. O LLM fica apenas com o que uma
rede pequena não sabe fazer: **unificar** (`UPDATE`/`MERGE`/`CONFLICT`) e,
opcionalmente, narrativas complexas.

## 1. Decisões tomadas

| # | Decisão | Escolha |
|---|---|---|
| 1 | Quem escreve os fatos `ADD` | **rede planeja + renderizador determinístico escreve** — zero LLM |
| 2 | `UPDATE`/`MERGE`/`CONFLICT` | **sempre LLM** (reescrita exige modelo generativo) |
| 3 |Autoridade | por classe, progressiva: `DISCARD → NOOP → ADD → UPDATE` |
| 4 | `episode` / `task_state` | **podem ser gerados localmente** por template + regras, com LLM como fallback de alta incerteza (§8) |

O princípio: uma rede de 64 neuronios não faz geração livre, mas **escrever um
fato atômico é extração + preenchimento de slots**, não criação. O LLM que hoje
escreve o fato vira **professor** do renderizador na Fase 1; em produção ele
desaparece do caminho do `ADD`.

## 2. O que muda em relação à Fase 2

| | Fase 2 | Fase 3 |
|---|---|---|
| `AVA_MEMORY_TRIAGE_SHADOW` | `1` | `0` |
| Decisões da rede | gravadas em `shadow_pred`, nada aplicado | aplicadas conforme `AVA_TRIAGE_AUTHORITY` |
| `ADD` | LLM escreve | renderizador local escreve |
| `DISCARD`/`NOOP` | LLM decide | rede decide |
| `UPDATE`/`MERGE`/`CONFLICT` | LLM | LLM (inalterado) |
| `episode`/`task_state` | LLM | local por default, LLM em incerteza |
| Canary | 10% ao LLM | mantido, como detector de drift |

## 3. Plano de fato + renderizador

O `/v1/triage` (Fase 2) passa a devolver, por segmento aceito, um **plano**:

```json
{
  "segment_id": 3,
  "plan": {
    "fragment_index": 1,
    "frame": "VERSION",
    "slots": {
      "entity": "embeddings server",
      "version": "2002",
      "date": "2026-11-25",
      "span": "o servidor de embeddings roda na porta 2002"
    },
    "confidence": 0.91
  },
  "rendered_text": "O servidor de embeddings do usuário está na versão 2002."
}
```

O renderizador fica em `Modules/memory/memory_triage.py` (domínio da memória),
**não** no onnxManager. Ele é uma função pura e determinística:

```python
def render_fact(plan: dict, role: str, ts: float) -> tuple[str, str]:
    frame = plan["frame"]
    if plan["confidence"] < FRAME_MIN_CONF or frame not in TEMPLATES:
        return f"O usuário relatou: {plan['slots']['span']}.", "verbatim"
    text = TEMPLATES[frame].format(**plan["slots"])
    if role == "assistant" and not plan["confirmed"]:
        raise SuppressedFact()   # política: assistente não confirmado não vira memória
    return text, "rendered"
```

O texto resultante é passado para o roteamento **já existente**
(`_lt_payload()` L2243 → `memory_write_facts_batch()` L1234), inclusive com o
gate do cross-encoder na escrita. Nada de dedup é reinventado.

### 3.1 Inventário de frames

| Frame | Template (pt-BR) | Slots |
|---|---|---|
| `USES_TOOL` | `O usuário usa {entity}{detail}.` | entity, detail |
| `PREFERS` | `O usuário prefere {value}.` | value |
| `CONFIG` | `A configuração {key} do usuário é {value}.` | key, value |
| `VERSION` | `{entity} do usuário está na versão {version}.` | entity, version |
| `OWNERSHIP` | `O usuário possui {entity}{detail}.` | entity, detail |
| `DECISION` | `O usuário decidiu {action}.` | action, date |
| `LOCATION` | `{entity} fica em {place}.` | entity, place |
| `EVENT` | `Em {date}, {event}.` | date, event |
| `IDENTITY` | `O usuário se chama {name}.` | name |
| `INSTRUCTION` | `O usuário instruiu: {span}.` | span |
| `CONFIRMED` | `O usuário confirmou/adotou: {span}.` | span |
| `GENERIC` | `O usuário relatou: {span}.` | span |

`GENERIC` é o fail-safe: se a confiança do frame ou dos slots estiver baixa, cai
para ele. O fato continua sendo gravado — **nunca** volta ao LLM só porque o
renderizador ficou em dúvida. `source` marcas: `chat:triage-rendered`,
`chat:triage-verbatim`, `chat:triage-narrative`.

### 3.2 Cabeças adicionais (ainda no mesmo trunk)

| Cabeça | Saída | Treino |
|---|---|---|
| **E — fragmento** | índice do fragmento do segmento (≤ 4 opções, saídas do `_split_query_segments()`) | alinhamento `evidence`↔fragmento da Fase 1 |
| **F — frame** | uma das 12 classes acima | weak labeling do `text` do LLM + revisão no gold |
| **G — slots** | entidade/valor/versão/data por extração de um vocabulário fechado | diff entre `evidence` e `text` |

Custo: três softmaxes pequenos na inferência já existente (< 0,1 ms). Nenhuma
chamada HTTP nova.

A v1 não usa atenção token a token: o próprio `_split_query_segments()`
(`memory_search.py:196`) já entrega fragmentos quase atômicos. Se um fragmento
ficou longo demais na prática, a v2 expõe `token_embeddings` de
`EmbeddingModel._embed_batch()` (`onnxManager.py:362`, L372) e aprende spans de
início/fim sobre os tokens.

### 3.3 Canonicalização de entidades

O prompt do extrator hoje diz `CANONICAL ENTITIES: (none)` (`LLM.py:1917-1918`)
— sempre vazio. O renderizador recebe uma lista real construída a partir de
memórias existentes, cards de pessoa e dicionários sensoriais, e os slots de
entidade são escolhidos por **cópia** dessa lista (classificação sobre poucos
itens), não por geração. Datas relativas são normalizadas por código:
`created_at` do turno + regex (`ontem`, `hoje`, `semana passada`) → ISO.

## 4. De onde vêm os rótulos do renderizador

A Fase 1 já loga, por fato emitido, `evidence` e `text`. A diferença entre os
dois é a supervisão:

1. **fragmento**: o fragmento que melhor casa com `evidence` (já calculado no
   alinhamento exact/fuzzy da Fase 1);
2. **frame**: padrões fracos no `text` — "prefere"→`PREFERS`, "versão"/número→
   `VERSION`, "porta"/"config"→`CONFIG`, "decidiu"→`DECISION`, "ontem"→`EVENT`.
   Revisar 200-300 casos no gold e corrigir;
3. **slots**: o que o LLM acrescentou em relação ao `evidence` — troca de "eu"
   por "O usuário", data absoluta, nome canônico no lugar de pronome. Números,
   versões e caminhos são extraídos por regex do próprio segmento;
4. **texto esperado**: o `text` do LLM serve só de referência para a métrica de
   paridade (§5) — o renderizador não é treinado a copiá-lo token a token.

## 5. Qualidade sem LLM

Duas métricas, nenhuma é "parece com o texto do LLM":

1. **Paridade de recuperação**: para cada fato do gold, embuta a versão do LLM e
   a versão renderizada; consultas parafraseadas devem recuperar a renderizada
   no mesmo top-k. O consumidor final é o embedding, não o leitor — se recupera
   igual, a redação é suficiente.
2. **Gold humano**: 200-300 pares renderizados checados quanto a autocontenção,
   ausência de pronome solto, preservação de números/versões/caminhos e
   atomicidade (uma ideia por fato).

Se a paridade ficar abaixo de ~90%, ajustar o inventário de frames — nunca
terceirizar de volta ao LLM por preguiça de template.

## 6. Fluxo completo no `_extract_and_save_memories`

```
candidatos (existing)
  ↓
/v1/triage → decisões + planos
  ↓
particiona:
  ├─ aceitos (KEEP+ADD/NOOP/DISCARD confiantes)
  ├─ duros (MERGE/CONFLICT/UPDATE com baixa confiança ou abstain)
  └─ suprimidos → nada
  ↓
[aceitos ADD]  → render_fact() → memory_write_facts_batch()   ← zero LLM
[aceitos NOOP] → reforço do alvo (sem escrita)
[duros]        → chamada de decisão ao LLM com candidatos pré-restritos
  ↓
[episódio/task_state] → gerador local (§8) ou LLM se incerteza
  ↓
merge_task_state() no orchestrator (como hoje)
```

O roteamento proposed→aplicado da Fase 1 permanece intacto — é ele que impede
que um `UPDATE` local destrua história. `AVA_TRIAGE_AUTHORITY` lista as classes
com autoridade; o default de entrada é `DISCARD,ADD,NOOP`.

## 7. Rollout e degradação

| Ordem | Classe | Risco de errar |
|---|---|---|
| 1 | `DISCARD` | perder um fato — recuperável pelo turno seguinte |
| 2 | `NOOP` | nenhuma escrita, só reforço |
| 3 | `ADD` | ruído — decaimento e poda limpam |
| 4 | `UPDATE` | destrutivo — entra só com concordância altíssima no gold |
| — | `MERGE`/`CONFLICT` | nunca local |

| Falha | Comportamento |
|---|---|
| onnxManager fora / modelo ausente | caminho completo da Fase 2 (LLM decide e escreve) |
| `/v1/score` fora | pares viram `abstain` → LLM |
| LLM fora | ADD local continua gravando; união e narrativa ficam pendentes com aviso |
| frame/slots com baixa confiança | `GENERIC` — grava igual, sem LLM |
| todas as tentativas do LLM falharem | `_note_memory_failure()` (`orchestrator.py:2064`) avisa o turno seguinte |

## 8. Episódio e task state sem LLM

Aqui está a extensão que fecha o objetivo de "remover o LLM da extração". O
mesmo princípio do plano de fato se aplica à narrativa: **template + regras**,
com LLM só em incerteza alta.

### 8.1 Episódio local

`Modules/memory/memory_episodes.py` ganha `render_episode(turn)`:

| Campo | Geração local |
|---|---|
| `summary` |concatena os 2-3 `rendered_text` do turno com "O usuário pediu X; recebeu Y." |
| `intent` | frame dominante dos fatos (`DECISION`→"decidir", `CONFIG`→"configurar"…) |
| `outcome` | contagem de ações: "N fatos gravados, M confirmados, K descartados" |
| `tools` / `tool_results` | já vêm do `tool_context` do turno (sem LLM) |
| `entities` | união dos slots `entity` dos fatos renderizados |
| `decisions` | fatos com frame `DECISION`/`CONFIRMED` |
| `open_questions` | frases do usuário terminadas em `?` e não respondidas |
| `importance` | heurística: `0.3 + 0.1×n_facts + 0.2×tem_decisão + 0.1×usou_tool`, clamp 0..1 |

Escreve via `POST /episodes/write` existente, com `turn_id` já disponível.

### 8.2 Task state local

`Modules/session/task_state.py` ganha `infer_task_state(update)`:

| Campo | Regra |
|---|---|
| `goal` | se houver fato `DECISION`, usar `{action}`; senão manter o anterior (`merge_task_state` já preserva) |
| `status` | `done` se o texto contiver "pronto/concluí/ resolved"; `blocked` se "travado/erro"; senão `active` |
| `decisions` | fatos `DECISION`/`CONFIRMED` do turno |
| `open_questions` | perguntas não respondidas do usuário |
| `next_steps` | fragmentos com "vou/preciso de/depois" |
| `constraints` | fatos com frame `PREFERS`/`INSTRUCTION` |

### 8.3 Quando ainda chamar o LLM

Gate de incerteza — local por default (`AVA_TRIAGE_LOCAL_NARRATIVE=1`), LLM se:

- `importance ≥ 0.8` (turno estruturante);
- houve `MERGE`/`CONFLICT` no turno (a narrativa precisa refletir a união);
- `task_state.status` mudou pela primeira vez ou o `goal` é novo;
- confiança média dos planos < limiar;
- turno sem nenhum fato aceito **e** com tool calls relevantes (o LLM resume melhor).

Nesses casos, a chamada curta de narrativa do §6 continua existindo. Fora deles,
**zero chamada ao LLM** — incluindo turns triviais, que já tinham `depth=0` no
`_think_instruction()` (`orchestrator.py:1155`).

### 8.4 Histórico preservado

`episode` e `task_state` locais entram nos mesmos canais de sempre: episódio no
recall de `_recall_for_turn()` (L2737-2752) e task state no system prompt via
`format_task_state()`. Nenhum consumidor muda — só a origem do texto.

Marcar `source` nos metadados (`episode_source=local|llm`) para auditar e medir
paridade de recuperação também nesses casos.

## 9. Flags

Todas ligadas por padrão (opt-out `=0`), documentadas no README:

| Flag | Default | Efeito |
|---|---|---|
| `AVA_TRIAGE_AUTHORITY` | `DISCARD,ADD,NOOP` | classes com autoridade local |
| `AVA_TRIAGE_RENDERER` | `1` | renderizador determinístico para `ADD` |
| `AVA_TRIAGE_LOCAL_NARRATIVE` | `1` | episódio/task state locais com gate de incerteza |
| `AVA_TRIAGE_NARRATIVE_MIN_CONF` | `0.70` | confiança abaixo da qual volta ao LLM |
| `AVA_TRIAGE_FRAME_MIN_CONF` | `0.65` | confiança abaixo da qual usa `GENERIC` |
| `AVA_TRIAGE_CANARY` | `10` | % de turnos com LLM challenger |
| `AVA_TRIAGE_MODEL` | `v1` | versão do ONNX — rollback instantâneo |

`AVA_MEMORY_TRIAGE_SHADOW=0` é a mudança que efetiva a fase (vem da Fase 2).

## 10. Observabilidade

- `/metrics`: `triage_authority_accepted_total` por classe, `triage_rendered_total`,
  `triage_rendered_verbatim_total`, `triage_llm_calls_total`,
  `triage_narrative_local_total`, `triage_narrative_llm_total`,
  `triage_frame_low_conf_total`, `triage_canary_disagreement_total`.
- Token savings estimado por `session_usage`: comparar `prompt_tokens` da
  extração antes/depois da Fase 3.
- `/triage/stats`: distribuição de frames, slots faltando e paridade de
  recuperação (renderizado vs. LLM no gold).
- WebUI: nenhuma mudança obrigatória; o `source` do fato aparece na inspeção.

## 11. Testes

- Renderizador: todos os 12 frames com slots válidos; `GENERIC` em baixa confiança;
  role=assistant não confirmado → `SuppressedFact`.
- Paridade de recuperação no gold (renderizado vs. LLM, consultas parafraseadas).
- Autoridade: cada classe respeita `AVA_TRIAGE_AUTHORITY`; `MERGE`/`CONFLICT`
  nunca gravam sem LLM, mesmo com confiança 0.99.
- Episódio local: campos preenchidos, `importance` no clamp, gravação via
  `/episodes/write` com o mesmo `turn_id`.
- Task state local: `goal`/`status` preservados quando o turno não traz nada novo.
- Gate de narrativa: cada condição do §8.3 dispara a chamada ao LLM.
- Fallbacks: onnxManager fora → Fase 2; LLM fora → ADD local grava e avisa;
  `/v1/score` fora → `abstain`.
- Canary 10%: chamada extra e divergência logadas.
- `python -m unittest discover -s tests`.

## 12. Critérios de aceite

- **Zero chamada ao LLM em turns com apenas `ADD`/`DISCARD`/`NOOP`** (inclusive
  episódio e task state locais).
- ≥ 90% de paridade de recuperação no gold entre fato renderizado e fato do LLM.
- ≥ 60% das decisões do turno resolvidas sem o prompt completo de extração.
- redução ≥ 40% nos tokens de extração por turno (medido em `session_usage`).
- **zero** falso `UPDATE`/`MERGE` — por construção, essas classes ficam no LLM.
- episódio escrito em ≥ 99% dos turns; `task_state` coerente em 100%.
- nenhum aumento em correções do usuário do tipo "não era isso".
- rollback por flag em um turno, sem migração.

## 13. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Renderizador produce texto "robótico" | paridade de recuperação como métrica; gold humano; `GENERIC` bem formado |
| Slot errado perde número/versão | checagem de entidades do `evidence` no `rendered_text`; fallback `GENERIC` |
| Rede vira gargalo silencioso | canary 10% + correções do usuário como sinal de drift |
| Ruído de `ADD` locais | gate do cross-encoder na escrita + decaimento/poda |
| Narrativa local perde contexto | gate de incerteza (§8.3) devolve ao LLM nos turnos estruturantes |
| Drift do teacher | versionar `extractor_model` + retreinar por geração |
| Usuário desconfia da memória | `source` auditável + correção em 1 clique na Fase 4 |

## 14. Próxima fase

- **Fase 4**: conversa contínua — desacoplar sessão de escopo, tópicos como
  blocos, `task_state` como pilha. Depende da Fase 2 (cabeça D) e se beneficia
  da Fase 3 (custo por turno baixo o suficiente para uma thread sem fim).
