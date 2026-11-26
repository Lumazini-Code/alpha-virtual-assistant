# Triagem de memória — Fase 1: instrumentação

Documento de planejamento. Esta fase **não muda nenhum comportamento**: o extrator
atual continua decidindo e gravando. Ela apenas registra, de forma estruturada,
tudo o que ele decidiu, tudo o que ele **não** decidiu e tudo o que o sistema de
fato executou, para produzir o dataset que treinará a rede da Fase 2.

Contexto: a ideia é substituir gradualmente a decisão "o que vira memória / o que
descartar / o que unificar" (hoje inteiramente num LLM extrator) por uma rede
pequena e local que recebe embeddings e5-small já calculados. Na Fase 3 a rede
também **escreve** os fatos `ADD`, por um renderizador determinístico alimentado
por um plano estruturado (`MEMORY_TRIAGE_PHASE3.md`) — e esta fase precisa coletar
os rótulos que treinam esse renderizador. O LLM fica apenas com as unificações e
os casos duvidosos. A Fase 1 é o pré-requisito: sem rótulos confiáveis e sem
cobrir os segmentos que o extrator **suprimiu por economia de tokens**, a rede
aprenderia o conservadorismo do teacher em vez do valor da informação (ver §5).

## 0. Estado da implementação

Implementado nesta rodada (início da Fase 1):

- esquema `memory_triage_turn` / `memory_triage_sample` + índices em
  `Modules/memory/memory_db.py` (idempotente por `CREATE IF NOT EXISTS`);
  `candidate_id` sem candidato é gravado como `-1` para preservar a
  idempotência também dos suprimidos;
- domínio em `Modules/memory/memory_triage.py` (+ `README_TRIAGE.md`):
  alinhamento exato/fuzzy, segmentação com `role`, rótulos fracos do
  renderizador, `build_samples`, upsert, exportação PU, estatísticas e delete;
- hook no `LLM.py` entre a validação de IDs e o roteamento: `proposed_label`
  vem da `action` original, `applied_label` é corrigido com o resultado real
  das escritas (`possible_update` → `UPDATE`, duplicata exata → `NOOP`);
  falhas das três tentativas e JSON inválido também geram linha de turno;
- `turn_id` = `execution_id` do orchestrator, compartilhado com o episódio;
- probe `relaxed` (orçamento 20 fatos / 4000 tokens, sem blocks/episode/
  task_state, **sem gravar memórias**), disparado em ~8% dos turnos como
  chamada extra em background, com propensão registrada;
- rotas `/triage/log|labels|stats` e `DELETE /triage/log`, `/status` com
  `memory_triage`, métricas `ava_triage_*`;
- WebUI: barra **Dados** na lateral com progresso até
  `AVA_MEMORY_TRIAGE_TARGET_SAMPLES` e aviso one-shot quando o dataset fica
  pronto;
- `tests/test_memory_triage.py` (11 testes).

Segunda rodada — Fase 1 **completa** (todas as fontes do §4 e o gold do §7):

- **Gate cross-encoder** (§5.2.4): `dry_run` em `WriteRequest`/`WriteBatchRequest`
  executa dedup exata + cross-encoder e devolve a mesma resposta sem gravar
  nada (sem insert, vetor, arestas, blocos ou cards). Rótulo local em
  `gate_label`/`gate_reason`/`gate_candidate_id`/`gate_score`, peso 0.6.
  Os `reason` reais do código são `duplicate_exact`,
  `duplicate_semantic:...:rerank=...`, `possible_update:...:rerank=...` e `ok`,
  não os nomes simplificados do doc.
- **Linhas de par** (§3): um fato gera uma linha por alvo/relacionado + top-5
  por cosine; segmentos sem fato também geram linhas de par, todas com
  `applied_label='unlabeled'`.
- **Probes por segmento** (§5.2.3): segmentos suprimidos por orçamento,
  do usuário, com ≥20 chars e não-déiticos, até 3 por turno, num único lote
  no modelo extrator barato. `teacher_variant=segment_probe`.
- **Pares contrastivos** (§5.2.7): o mesmo segmento com e sem o cue
  "Lembre-se disso", em `contrastive_no_cue` / `contrastive_cue`, peso 0.8.
- **Sintético** (§4): perturbações determinísticas (versão, "na verdade",
  continuação, efêmero) rotuladas pelo teacher; só aceitas se baterem com o
  esperado, teto de 25% do batch e `AVA_MEMORY_TRIAGE_SYNTHETIC_RATE=0.05`,
  `origin=synthetic`, peso 0.3.
- **Thinkings** (§5.2.8): `probe=thinking` usa um recorte mínimo do CoT,
  `origin=probe_thinking`, `teacher_variant=probe_thinking`, peso 0.3 e
  `applied_label='unlabeled'` — nunca `applied`.
- **Replay** (§4): `scripts/triage_replay.py` reprocessa turnos do session
  store com `probe=replay`/`thinking`. Todos os variants de instrumentação
  são **somente leitura** — replay nunca reescreve memórias corrigidas.
- **Gold set** (§7): tabela `memory_triage_gold` imutável, amostra
  estratificada em `GET /triage/gold/candidates`, revisão via
  `POST /triage/gold`, consulta em `GET /triage/gold`, alvo
  `AVA_MEMORY_TRIAGE_GOLD_TARGET=200`. A calibração da Fase 2 deve usar
  **somente** estes rótulos, nunca os do teacher (§5.4).
- `/triage/stats` também expõe `emitted_rate_by_role`, `emitted_rate_by_size`,
  `gate_label_counts`, `gold_total` e os pesos/flags de cada fonte.

O job de retenção roda de forma lazy a cada `/triage/log` (não há scheduler
dedicado). Aplicação offline dos pesos de treino (IPW, over/under-sample e
Platt no gold) permanece na Fase 2 — aqui só coletamos os rótulos e a propensão.

## 1. Estado atual e pontos de captura

O fluxo relevante hoje:

1. `orchestrator.py:_extract_memories_via_llm()` (L2764) chama
   `POST /memories/extract` no `LLM.py` com `user_input`, `assistant_response`,
   `session_id`, `scope`, `task_state` e `tool_context`.
2. `LLM.py:memories_extract()` (L3601) delega para
   `LLM.py:_extract_and_save_memories()` (L1970).
3. Essa função coleta candidatos (`_collect_extraction_candidates()`, L1714),
   monta o prompt (`_build_extraction_user_prompt()`, L1857), chama o extrator
   numa cadeia de 3 tentativas (L2102-2176), faz o parse/validação dos IDs
   (L2178-2241) e roteia cada fato (L2262-2358).

**Hook principal:** entre a validação de IDs (L2241) e o roteamento (L2262).
Nesse ponto estão disponíveis: `facts_raw`, `candidates` (com `text`, `score`,
`match_type`, `used_in_answer`), `m_map` (M-n → ID real), `evidence`,
`rationale`, `blocks`, `task_state`, `episode`, o modelo escolhido e o índice da
tentativa que venceu.

**Segundo hook:** depois das escritas assíncronas (L2360-2365), para capturar o
resultado aplicado de `memory_write_facts_batch()`, `_memory_write_update()` e
`_resolve_possible_update()` (L1144).

**Terceiro hook (novo):** no alinhamento `evidence`→segmento (§3), para marcar
**todo** segmento como `teacher_emitted` ou `suppressed`. É o que evita que a
ausência de um fato seja interpretada como `DISCARD`.

### Proposto, aplicado e suprimido

O dataset tem três estados, não dois:

| Estado | Significado | Uso no treino |
|---|---|---|
| **Proposto** (`proposed_label`) | o que o LLM escreveu | mede divergência do teacher |
| **Aplicado** (`applied_label`) | o que o pipeline de escrita fez | ground truth da cabeça B |
| **Suprimido** (`teacher_emitted=0`) | segmento sobre o qual o teacher nunca se pronunciou | **desconhecido**, nunca negativo — ver §5 |

Treinar só no que o LLM emitiu clona seus erros; tratar o que ele não emitiu como
`DISCARD` clona sua economia de tokens. As duas coisas precisam ser registradas
separadamente.

| Caso no código | Proposto | Aplicado |
|---|---|---|
| `UPDATE`/`MERGE` sem alvo válido (L2307-2316) | `UPDATE`/`MERGE` | `ADD` (downgrade) |
| `MERGE` com vários alvos (L2319-2330) | `MERGE` em N | `UPDATE` no primeiro |
| `CONFLICT` (L2332-2350) | `CONFLICT` | `ADD` + update do antigo com prefixo |
| `NOOP` sem alvo (L2351-2355) | `NOOP` | `DISCARD` |
| Duplicata exata recusada | `ADD` | nada gravado |
| `possible_update:<score>` (L1144) | `ADD` | `UPDATE` no `candidate_id` |

### `turn_id`

`MemoryExtractionRequest` (L3588) não tem identificador de turno. Threading
necessário: `execution_id` (criado no `_execute_stream_generator` do
orchestrator) → `_extract_memories_via_llm()` → `MemoryExtractionRequest.turn_id`.
O mesmo `turn_id` já é a chave de `memory_episodes` (`UNIQUE(session_id,
turn_id)` em `memory_db.py:155`), o que permite ligar log, episódio e fatos.

## 2. Esquema do log

Tabelas em `Modules/memory/memory_db.py`, seguindo o padrão de migração leve
já usado em L164-176 (`PRAGMA table_info` + `ALTER TABLE`).

```sql
CREATE TABLE IF NOT EXISTS memory_triage_turn (
    turn_id           TEXT NOT NULL,
    session_id        TEXT NOT NULL,
    scope             TEXT NOT NULL DEFAULT '',
    created_at        REAL NOT NULL,
    extractor_model   TEXT NOT NULL,
    fallback_model    TEXT NOT NULL DEFAULT '',
    attempt_index     INTEGER NOT NULL DEFAULT 1,
    prompt_version    TEXT NOT NULL DEFAULT '',
    schema_version    TEXT NOT NULL DEFAULT '',
    budget_variant    TEXT NOT NULL DEFAULT 'normal', -- normal|relaxed|probe
    propensity        REAL NOT NULL DEFAULT 1.0,      -- P(variant | turno)
    latency_ms        REAL NOT NULL DEFAULT 0,
    user_hash         TEXT NOT NULL DEFAULT '',
    assistant_hash    TEXT NOT NULL DEFAULT '',
    n_facts           INTEGER NOT NULL DEFAULT 0,
    n_segments        INTEGER NOT NULL DEFAULT 0,
    n_suppressed      INTEGER NOT NULL DEFAULT 0,
    n_candidates      INTEGER NOT NULL DEFAULT 0,
    status            TEXT NOT NULL DEFAULT 'ok',  -- ok|all_attempts_failed|no_output
    episode_importance REAL DEFAULT NULL,
    task_status       TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (session_id, turn_id)
);

CREATE TABLE IF NOT EXISTS memory_triage_sample (
    sample_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id             TEXT NOT NULL,
    session_id          TEXT NOT NULL,
    scope               TEXT NOT NULL DEFAULT '',
    segment_index       INTEGER NOT NULL,
    segment_text        TEXT NOT NULL,
    segment_hash        TEXT NOT NULL,
    role                TEXT NOT NULL,             -- user|assistant
    n_chars             INTEGER NOT NULL,
    has_deictic         INTEGER NOT NULL DEFAULT 0,
    candidate_id        INTEGER DEFAULT NULL,
    candidate_text_snapshot TEXT NOT NULL DEFAULT '',
    candidate_created_at   REAL DEFAULT NULL,
    cosine              REAL DEFAULT NULL,
    cross_encoder_score REAL DEFAULT NULL,
    used_in_answer      INTEGER NOT NULL DEFAULT 0,
    rank                INTEGER DEFAULT NULL,
    teacher_variant     TEXT NOT NULL DEFAULT 'base',
      -- base|relaxed_budget|facts_only|segment_probe|cross_encoder|fallback|probe_thinking
    teacher_emitted     INTEGER NOT NULL DEFAULT 0,
    suppression_reason  TEXT NOT NULL DEFAULT '', -- policy|budget|unknown|no_candidate
    propensity          REAL NOT NULL DEFAULT 1.0,
    proposed_label      TEXT NOT NULL DEFAULT '',
    proposed_targets    TEXT NOT NULL DEFAULT '[]',
    proposed_type       TEXT NOT NULL DEFAULT 'fact',
    proposed_validity   TEXT NOT NULL DEFAULT 'temporal',
    applied_label       TEXT NOT NULL DEFAULT 'unlabeled',
    applied_reason      TEXT NOT NULL DEFAULT '',
    -- Rótulos do renderizador (Fase 3): derivados de evidence↔text
    frame_label         TEXT NOT NULL DEFAULT '',   -- inventário de 12 frames
    fragment_index      INTEGER DEFAULT NULL,       -- índice do fragmento alvo
    slot_entity         TEXT NOT NULL DEFAULT '',
    slot_value          TEXT NOT NULL DEFAULT '',
    slot_date           TEXT NOT NULL DEFAULT '',
    rendered_text       TEXT NOT NULL DEFAULT '',   -- texto do LLM, referência
    render_source       TEXT NOT NULL DEFAULT '',   -- rendered|verbatim|llm
    evidence_aligned    TEXT NOT NULL DEFAULT 'none', -- exact|fuzzy|none
    origin              TEXT NOT NULL DEFAULT 'live', -- live|replay|synthetic|human|probe
    weight              REAL NOT NULL DEFAULT 1.0,
    embed_model_version TEXT NOT NULL DEFAULT '',
    shadow_pred         TEXT NOT NULL DEFAULT '',     -- reservado para Fase 2
    created_at          REAL NOT NULL,
    UNIQUE (turn_id, segment_hash, candidate_id, teacher_variant)
);
CREATE INDEX IF NOT EXISTS idx_tri_sample_turn    ON memory_triage_sample(turn_id);
CREATE INDEX IF NOT EXISTS idx_tri_sample_label   ON memory_triage_sample(applied_label);
CREATE INDEX IF NOT EXISTS idx_tri_sample_origin  ON memory_triage_sample(origin);
CREATE INDEX IF NOT EXISTS idx_tri_sample_suppress ON memory_triage_sample(teacher_emitted);
CREATE INDEX IF NOT EXISTS idx_tri_sample_variant ON memory_triage_sample(teacher_variant);
```

Decisões de desenho:

- **Não armazenar embeddings** (384 floats) no SQLite: eles são determinísticos
  dado texto + versão do e5 e são recriados no treino. Guardar só
  `embed_model_version`.
- **Guardar `cross_encoder_score`**: é caro de recomputar e varia por par.
- **Guardar `candidate_text_snapshot`**: o candidato pode ser atualizado depois;
  sem o snapshot, pares de `UPDATE`/`CONFLICT` ficam irreproduzíveis.
- **Texto do segmento, não da conversa inteira**: o ledger já vive no session
  store. `AVA_MEMORY_TRIAGE_LOG_TEXT=0` troca por hash-only.
- **`applied_label='unlabeled'` é valor legítimo**: é o estado dos segmentos
  suprimidos. A exportação para treino precisa filtrar por ele explicitamente —
  nunca convertê-lo em `DISCARD` por default.
- **Idempotência** por `UNIQUE(turn_id, segment_hash, candidate_id,
  teacher_variant)`, porque `/memories/extract` já serve para reprocessar turnos
  antigos e agora há múltiplos variants por turno.

## 3. Alinhamento `evidence` → segmento

O `evidence` literal devolvido pelo LLM (L2286) é o que permite rotular sem
anotação humana:

1. rodar `_split_query_segments()` (`Modules/memory/memory_search.py:196`)
   sobre `user_input` e `assistant_response`, marcando o `role` de cada segmento;
2. normalizar (`lower`, sem pontuação/espaços extras) `evidence` e segmentos;
3. substring exata; se falhar, `difflib.SequenceMatcher` (já usado em
   `LLM.py:354`) com limiar ≈ 0.75;
4. se casar: `evidence_aligned = exact|fuzzy`, `teacher_emitted=1`, e a `action`
   vai para aquele segmento;
5. se não casar: `evidence_aligned = none`, `teacher_emitted=0`,
   `applied_label='unlabeled'`, `suppression_reason` preenchido quando possível.
   **Não é `DISCARD` e não entra no treino como negativo** — entra na fila de
   probes de §5 e nas métricas do teacher.

Métrica derivada importante: **taxa de evidência ancorada**. Se for baixa, o
próprio extrator está produzindo `evidence` vaga — corrigir isso antes de treinar.

**Rótulos do renderizador (Fase 3).** Para cada fato com `evidence` alinhada, o
par `(segment_text, evidence, text)` é supervisão suficiente para treinar as
cabeças E/F/G:

- `fragment_index` = fragmento do `_split_query_segments()` que melhor casa com
  `evidence`;
- `frame_label` = classe fraca inferida do `text` ("prefere"→`PREFERS`,
  "versão"/número→`VERSION`, "porta"/"config"→`CONFIG`, "decidiu"→`DECISION`),
  revisada no gold;
- `slot_entity`/`slot_value`/`slot_date` = o que o LLM acrescentou em relação à
  `evidence` (troca de "eu" por "O usuário", data absoluta, entidade canônica);
  números e caminhos vêm de regex sobre o próprio segmento;
- `rendered_text` guarda o `text` do LLM **só como referência de paridade** — o
  renderizador não é treinado a copiá-lo token a token.

Para as linhas de par: `target`/`related` já viram IDs reais via `m_map`; cada
fato gera uma linha por candidato relevante (top-5 por cosine + todos os
`target`/`related`), com o rótulo daquele par. Segmentos sem fato emitido também
geram linhas de par, mas com `applied_label='unlabeled'` — são justamente os
candidatos a probe.

## 4. Fontes do dataset

| Fonte | Geração | Papel | Peso |
|---|---|---|---|
| Tráfego ao vivo (base) | hook descrito acima, `teacher_variant=base` | distribuição real de decisões aplicadas | 1.0 |
| **Variant relaxado de orçamento** | `probe=relaxed`, mesma política, até 20 fatos, `facts_only`, `max_tokens≈4000` | recupera fatos suprimidos só por token | 1.0 para cabeça A |
| **Probe por segmento** | pergunta mínima sobre segmento `suppressed` + 1-3 candidatos | rótulo direto na unidade que a rede vê | 1.0 para cabeça A |
| **Gate cross-encoder** | tentativa de escrita do segmento cru; lê `reason` | rótulo local, sem custo de API | 0.6 |
| **Pares contrastivos "lembre-se disso"** | mesmo segmento com e sem cue de intenção | separa "valioso" de "suprimido por economia" | 0.8 |
| **Probe de thinkings** | recorte mínimo do CoT, `origin=probe_thinking` | descobre o que a política de produção proíbe gravar | 0.3, nunca `applied` |
| Replay do histórico durável | `/memories/extract` sobre turnos antigos | bootstrap e calibração de features | 0.4 |
| Sintético / contraste | perturbar fatos: versão, negação, "na verdade...", data, paráfrase, frase aleatória | cobrir `CONFLICT`/`MERGE`/`UPDATE`, raros | 0.3, `origin=synthetic`, teto 25% do batch |
| Sinais humanos | "lembre-se/disque isso", correção, `guide` manual, correção de escopo | âncora de alta precisão | 1.5 |
| Supervisão por uso (Fase 2+) | recuperado e usado = reforço; podado sem uso = negativo fraco | fecha o loop hebbiano | 0.2 |

Replay serve para aprender a distribuição de features, **não** para rótulos
confiáveis de `UPDATE`/`CONFLICT`: o candidato que o teacher veria hoje é
diferente do da época. Sempre com `origin=replay`.

Exemplo de sintético para "O usuário roda o embeddings server na porta 2002":

- "agora é 2003" → `UPDATE`
- "na verdade é 3000, o da 2002 era o antigo" → `CONFLICT`
- "o servidor de embeddings continua na 2002" → `NOOP`
- "o servidor de TTS é o supertonic" → `ADD`
- "estou com pressa" → `DISCARD`

O teacher rotula; aceitar só se o rótulo bater com o esperado.

## 5. Viés de supressão: o extrator é conservador por design

O extrator de produção **não emite todos os fatos que valem memória**. Isso é
intencional e tem duas causas distintas, que precisam ser tratadas de forma
diferente:

| Causa | Onde está | Queremos replicar na rede? |
|---|---|---|
| **Política** | prompt: estado momentâneo, conhecimento genérico, fala do assistente não confirmada (README L153, prompt ~L1514) | **Sim** — é a regra correta |
| **Orçamento** | `max_tokens: 1500` (L2117) para facts + blocks + episode + task_state; teto de 8 fatos (L2084, L1642); teto de candidatos | **Não** — é o viés a corrigir |

Relaxar tudo de uma vez pioraria o modelo: ele passaria a gravar "estou com
pressa" e sugestões do assistente. O contorno **relaxa só o orçamento**, mantém
a política intacta.

### 5.1 Separação de teachers por cabeça

| Cabeça | Fonte do rótulo | Razão |
|---|---|---|
| **A** (`KEEP`/`DISCARD`) | união dos fatos emitidos pelos variants relaxados + probes | alta cobertura; é a decisão que a rede vai baratear |
| **B** (ação no par) | `applied_label` do pipeline de produção | é a ação real, validada pelo cross-encoder e pelo `possible_update` |
| **C** (abstention) | divergência **entre variants** | é onde os teachers discordam que a rede deve escalar |
| **D** (escopo, Fase 2) | inalterada, self-supervised | — |

O conservadorismo fica confinado à cabeça B, onde é correto, e não contamina a A.

### 5.2 Táticas

1. **Logar todo segmento com `teacher_emitted`.** Segmentos sem `evidence`
   alinhada são `suppressed`, nunca `DISCARD`. Campos: `teacher_variant`,
   `suppression_reason`, `propensity`.
2. **Passada `facts-only` com orçamento relaxado.** O `/memories/extract` já é
   endpoint manual/debug (L3601). Adicionar `probe=true` que mantém todas as
   regras de política, troca "máximo 8 fatos" por "até 20, cubra os segmentos
   com conteúdo memorável", remove `blocks`/`episode`/`task_state` do schema e
   sobe `max_tokens` para ~4000. Elimina a competição por token com o episódio,
   que esmaga os fatos do fim de respostas longas.
3. **Probe por segmento suprimido.** Para cada `suppressed` substancial (≥ 20
   chars, role=user, não-déitico), uma chamada mínima: *"Este trecho contém
   informação memorável segundo as regras? Se sim, escreva o fato."* Com 1-3
   candidatos, no máximo 3 segmentos por turno.
4. **Gate cross-encoder como rotulador gratuito.** Tentar gravar o próprio
   segmento como fato e ler o `reason`: `exact_duplicate`→`NOOP`,
   `possible_update`→`UPDATE`, `stored`→`ADD`. Zero custo de API; o ruído do
   texto cru é representativo do que a rede verá na inferência.
5. **Randomizar o orçamento em 5-10% dos turnos.** Atribuir `normal` ou
   `relaxed` aleatoriamente e registrar a propensão. Dentro do grupo
   aleatorizado a supressão vira *missing completely at random*, permitindo
   estimar a distribuição verdadeira sem inventar rótulos.
6. **PU learning em vez de negativos ingênuos.** Emitidos = positivos;
   nunca julgados = *unlabeled*. Usar nnPU/Elkan-Noto, ou primeiro treinar um
   `worth_storing` com positivos confiantes + negativos **verificados** (os que
   o variant relaxado também descartou por política) e pseudo-rotular o resto.
7. **Pares contrastivos "lembre-se disso".** Para cada segmento suprimido por
   orçamento, repetir com o cue. Se o teacher emitir com o cue e não sem ele, há
   um par positivo/negativo que difere só pela intenção.
8. **Thinkings só no probe.** O CoT é a maior fonte de conteúdo que a política
   proíbe gravar. Entra como `origin=probe_thinking`, separado, e nunca como
   `applied_label`.

### 5.3 Medir antes de treinar

Rode normal vs. relaxado numa amostra e calcule:

```
recall_gap = |facts_relaxado − facts_normal| / |facts_relaxado|
```

Se der 30%, o teacher de produção perde 30% dos fatos memoráveis — esse número
justifica o custo dos probes. Estratificar por `role`, tamanho e **posição no
turno**: os segmentos do fim são sistematicamente suprimidos porque episode +
task_state consomem o orçamento primeiro.

`GET /triage/stats` deve expor `recall_gap`, `suppressed_total`,
`suppressed_by_reason` e `emitted_rate_by_position`.

### 5.4 Correção no treino

- over-sample dos positivos dos variants relaxados e under-sample dos
  `DISCARD`s do teacher de produção, para que o prior de treino corresponda ao
  deployment (rede barata → pode guardar mais);
- calibração (Platt) **no gold set revisado**, nunca nos rótulos do teacher —
  calibrar no teacher só reproduz o viés dele;
- `propensity` entra como peso (inverse propensity weighting);
- `prompt_version`, `schema_version` e `teacher_variant` versionados por linha.

### 5.5 Limites de custo

- variants relaxados em 5-10% dos turnos, não 100%;
- probes só em `suppressed` acima de limiar, no máximo 3 por turno;
- sempre no modelo extrator barato (`MEMORY_EXTRACTOR_MODEL`), nunca no principal;
- cross-encoder e regras de política são locais — usar à vontade.

## 6. Rede: um trunk, várias cabeças

Não são necessárias várias redes independentes. Formato alvo:

- **Trunk**: e5-small congelado (já no onnxManager `:2002`) → 384 dims + ~15
  features → densa 64.
- **Cabeça A (por segmento)**: `KEEP`/`DISCARD` + `type` + `validity` +
  `importance` (regressão). Multitask no mesmo trunk. Rótulos conforme §5.1.
- **Cabeça B (por par)**: `[emb_seg; emb_cand; emb_seg ⊙ emb_cand;
  |emb_seg − emb_cand|; cosine; cross_encoder_score; features]` → 6 logits
  (`ADD/NOOP/UPDATE/CONFLICT/MERGE/DISCARD`). Rótulos = `applied_label`.
- **Cabeça C (abstention)**: binária "devo escalar para o LLM?". Treinada com
  divergências entre variants e teacher×aplicado. É válvula de segurança.
- **Cabeça D (Fase 2)**: escopo do segmento, self-supervised pelo `scope` da
  memória gravada — sem anotação nova. O `scope` do turno é target, **nunca**
  feature (ver `MEMORY_TRIAGE_PHASE2.md` §5).
- **Cabeça E (Fase 3)**: seleção do fragmento alvo (≤ 4 opções do splitter).
- **Cabeça F (Fase 3)**: frame do fato, entre 12 classes (`PREFERS`, `VERSION`,
  `CONFIG`, `DECISION`, `EVENT`, `GENERIC`…).
- **Cabeça G (Fase 3)**: slots por cópia de um vocabulário fechado (entidade de
  memórias/cards/dicionários, valor, versão, data).

As cabeças E/F/G alimentam o renderizador determinístico da Fase 3. Estão no
mesmo trunk e custam três softmaxes pequenos por inferência.

Features sugeridas além do embedding: `role`, tamanho, posição no turno,
marcadores deíticos, cosine com o candidato, `cross_encoder_score`,
`used_in_answer`, idade do candidato, `rank`, `attempt_index`, `propensity`.

Tipos, em ordem de complexidade:

1. Regressão logística / SVM linear — baseline; mostra se o sinal está no
   embedding ou nas features.
2. **LightGBM / XGBoost** — aposta para a primeira versão da cabeça B: ganha de
   MLP com poucos milhares de exemplos e dá feature importance.
3. MLP pequeno em ONNX — quando o volume crescer; < 1 ms.
4. Siamese com triplet/contrastive — se `CONFLICT`/`MERGE` continuarem raros.
5. Fine-tune do cross-encoder ms-marco — máxima precisão relacional, só após
   milhares de rótulos; destilar de volta para o MLP depois.

Serviço: rota `/triage` **dentro do `onnxManager.py`**, ao lado do e5 e do
`/v1/score`, com lazy-load igual ao `_ensure_rerank_model()` (L642). Um
round-trip, embeddings não saem do processo.

## 7. Treino e validação

- Split **por conversa/escopo** (`group=session_id`), nunca por linha, para
  evitar vazamento de vocabulário. Validação final com split temporal.
- **Nunca rotular `applied_label='unlabeled'` como negativo**; excluir do
  supervisionado ou usar como unlabeled em PU learning (§5.4).
- Métricas: macro-F1, precision/recall por classe, e sobretudo **falso
  `UPDATE`/`MERGE`** (destrutivo) × falso `DISCARD` (recuperável). Reportar
  também recall da cabeça A contra o variant relaxado (é o cliente de §5.3).
- Calibração: ECE + curva cobertura×acurácia para a cabeça C, **no gold set**.
- **Gold set imutável**: 200-300 exemplos revisados, incluindo uma amostra
  estratificada de segmentos antes suprimidos — é a única defesa independente
  contra o viés do teacher.
- `attempt_index` entra no `weight`: rótulos do fallback (tentativas 2 e 3)
  são de menor qualidade.
- Falhas das 3 tentativas (L2170-2176) também são logadas (`status`): são os
  casos difíceis que a cabeça C deve escalar e hoje desaparecem sem traço.

## 8. Integração e contratos

Rotas novas em `Modules/memory/memory_api.py` (que só delega ao domínio):

| Rota | Uso |
|---|---|
| `POST /triage/log` | grava turn + samples, idempotente por variant |
| `GET /triage/labels?limit&since&scope&origin&variant&include_unlabeled` | exportação para o treino |
| `GET /triage/stats` | classes/origin/attempt + `recall_gap` + `suppressed_by_reason` |
| `DELETE /triage/log?scope=` | privacidade/retention |

Em `Modules/LLM.py`:

- adicionar `"memory_triage_log": ("POST", "/triage/log")` ao mapa
  `_MEMORY_TOOL_ROUTES` (L510) e chamar via `_call_memory_tool()` dentro de
  `asyncio.create_task()`. Best-effort: falha da API nunca derruba o turno.
- estender `MemoryExtractionRequest` (L3588) com `probe?: "relaxed"|"segment"|"thinking"`
  e `propensity?: float`. O variant `relaxed` altera prompt + schema +
  `max_tokens` **somente nessa chamada de debug**, nunca no caminho de produção.

`POST /triage/log` (payload resumido):

```json
{
  "turn": {
    "turn_id": "...", "session_id": "...", "scope": "",
    "extractor_model": "...", "fallback_model": "", "attempt_index": 1,
    "prompt_version": "v1", "schema_version": "v1",
    "budget_variant": "normal", "propensity": 1.0,
    "latency_ms": 812, "user_hash": "...", "assistant_hash": "...",
    "n_facts": 3, "n_segments": 11, "n_suppressed": 8,
    "n_candidates": 12, "status": "ok", "episode_importance": 0.6,
    "task_status": "active"
  },
  "samples": [
    {
      "segment_index": 0, "segment_text": "...", "segment_hash": "...",
      "role": "user", "n_chars": 48, "has_deictic": 0,
      "candidate_id": 17, "candidate_text_snapshot": "...",
      "cosine": 0.71, "cross_encoder_score": 0.83,
      "used_in_answer": 1, "rank": 1,
      "teacher_variant": "base", "teacher_emitted": 1,
      "suppression_reason": "", "propensity": 1.0,
      "proposed_label": "UPDATE", "proposed_targets": "[17]",
      "proposed_type": "fact", "proposed_validity": "temporal",
      "applied_label": "UPDATE", "applied_reason": "ok",
      "frame_label": "CONFIG", "fragment_index": 1,
      "slot_entity": "embeddings server", "slot_value": "2002", "slot_date": "",
      "rendered_text": "O usuário roda o embeddings server na porta 2002.",
      "render_source": "llm",
      "evidence_aligned": "exact", "origin": "live", "weight": 1.0
    },
    {
      "segment_index": 7, "segment_text": "...", "segment_hash": "...",
      "role": "assistant", "n_chars": 91, "has_deictic": 0,
      "teacher_variant": "base", "teacher_emitted": 0,
      "suppression_reason": "budget", "propensity": 1.0,
      "applied_label": "unlabeled", "evidence_aligned": "none",
      "origin": "live", "weight": 0.0
    }
  ]
}
```

Nova pasta de domínio: `Modules/memory/memory_triage.py` (com `README.md`),
exportada por `Modules/memory/memory.py` junto dos outros handlers.

## 9. Flags

Todas ligadas por padrão (regra do opt-out do projeto), documentadas no README:

| Flag | Default | Efeito |
|---|---|---|
| `AVA_MEMORY_TRIAGE_LOG` | `1` | instrumentação da extração |
| `AVA_MEMORY_TRIAGE_LOG_TEXT` | `1` | guardar texto do segmento (`0` = hash-only) |
| `AVA_MEMORY_TRIAGE_PROBE` | `1` | variants relaxados + probes por segmento |
| `AVA_MEMORY_TRIAGE_PROBE_RATE` | `0.08` | fração de turnos com orçamento randomizado |
| `AVA_MEMORY_TRIAGE_PROBE_MAX` | `3` | probes por turno |
| `AVA_MEMORY_TRIAGE_PU` | `1` | tratar `unlabeled` como unlabeled (não negativo) |
| `AVA_MEMORY_TRIAGE_RENDER_LABELS` | `1` | derivar frame/fragmento/slots de `evidence`↔`text` |
| `AVA_MEMORY_TRIAGE_RETENTION_DAYS` | `90` | poda do log |
| `AVA_MEMORY_TRIAGE_SHADOW` | `0` | reservado para Fase 2 |

Rollback pontual: `AVA_MEMORY_TRIAGE_LOG=0` ou `AVA_MEMORY_TRIAGE_PROBE=0`.
Esquema e rotas são aditivos.

## 10. Não-funcionais

- Payload limitado (8 fatos × 15 candidatos no base; 3 segmentos por probe).
  Zero impacto na latência do turno — tudo em `asyncio.create_task()`.
- Regex de segredo antes de gravar. CoT cru **nunca** entra no log de produção;
  no variant `probe_thinking` entra só um recorte mínimo, marcado
  `origin=probe_thinking`, e nunca vira `applied_label`.
- `/metrics`: `triage_logged_total`, `triage_alignment_exact|fuzzy|none`,
  `triage_log_failures`, `triage_suppressed_total`, `triage_probe_calls_total`,
  `triage_recall_gap`; `/status` com seção `memory_triage`.
- Auth interna já aplicada pelo `install_internal_auth()` do memory_api.

## 11. Testes

- Alinhamento `evidence`↔segmento: exato, fuzzy, nenhum, evidência entre dois
  segmentos, evidência parafraseada.
- Segmento sem `evidence` → `teacher_emitted=0` e `applied_label='unlabeled'`,
  **nunca** `DISCARD`.
- Upsert idempotente (mesmo turno + hash + variant reprocessado).
- Transformações proposed→applied da tabela da seção 1.
- Variant `relaxed`: política preservada, teto/c `max_tokens` alterados apenas
  na chamada de probe; produção idêntica.
- `recall_gap` calculado em fixture com fatos suprimidos por orçamento.
- Exportação PU: `include_unlabeled=0` exclui, `=1` marca como unlabeled — nunca
  converte em negativo.
- Derivação de rótulos do renderizador: `fragment_index` casa com `evidence`,
  `frame_label` reconhece os 12 frames, slots extraídos do segmento, e
  `rendered_text` igual ao `text` do LLM usado apenas como referência.
- Migração do esquema em banco antigo (sem as novas colunas/tabelas).
- Integração: fixture de turno → `/triage/log` → `/triage/labels`.
- `python -m unittest discover -s tests`.

## 12. Critérios de aceite

- ≥ 95% das extrações com linha em `memory_triage_turn`.
- ≥ 80% dos fatos emitidos com `evidence` ancorada (`exact` ou `fuzzy`).
- **100% dos segmentos** com `teacher_emitted` preenchido e nenhum `unlabeled`
  exportado como negativo.
- `recall_gap` medido por `role` e posição no turno.
- ≥ 80% dos fatos emitidos com `frame_label`, `fragment_index` e slots
  preenchidos (para treinar o renderizador da Fase 3).
- Distribuição de classes medida no variant relaxado (para decidir quanto
  sintético gerar).
- Latência do turno inalterada e nenhuma exceção de log vazando para o extrator.

## 13. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| **Viés de supressão** (rede herda a economia do LLM) | `teacher_emitted` + variants relaxados + PU learning + propensity |
| Relaxar política e gravar efêmero | variant relaxado muda **só** orçamento; política intacta |
| Teacher bias em geral | `applied_label` como ground truth + gold set + divergência |
| Evidência não literal | alinhamento fuzzy; não-ancorado vira `suppressed`, não negativo |
| Vazamento entre treino/teste | split por `session_id`/escopo + split temporal |
| Classes raras | sintético com `origin` marcado e teto de 25% |
| Drift do teacher | versionamento de prompt/modelo/variant/attempt + gold imutável |
| Seletividade: só turnos bem-sucedidos | logar também falhas das 3 tentativas |
| Custo dos probes | 5-10% dos turnos, ≤ 3 segmentos, modelo extrator barato |

## 14. Próximas fases (fora deste documento)

- **Fase 2**: rede local em shadow mode, métricas de concordância, limiares por
  classe e abstention, cabeça D de escopo.
- **Fase 3**: triagem assume os casos fáceis; as cabeças E/F/G alimentam o
  renderizador que escreve os `ADD` sem LLM; LLM fica com unificações,
  episódios de alta incerteza e duvidosos.
- **Fase 4**: conversa contínua — desacoplar sessão de escopo, tópicos como
  blocos, `task_state` como pilha.
