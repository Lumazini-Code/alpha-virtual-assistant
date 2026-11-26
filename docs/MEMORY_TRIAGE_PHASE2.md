# Triagem de memória — Fase 2: rede local em shadow mode

Documento de planejamento. Continuação de `MEMORY_TRIAGE_PHASE1.md`.

**Decisões já tomadas:**

1. Treino com `MLPClassifier` (scikit-learn, já no `requirements.txt`) e exportação
   ONNX direta com `onnx.helper` — sem dependência nova.
2. `MERGE` e `CONFLICT` **sempre** com o LLM para gerar o texto unificado, mesmo
   quando a rede está confiante. A rede escolhe *quem* unir; o LLM escreve o texto.
3. A cabeça D (escopo) entra já na Fase 2, em shadow.

Nesta fase a rede **ainda não tem autoridade**. Ela roda em paralelo ao extrator
e grava o que faria na coluna `shadow_pred` da Fase 1. O ground truth continua
sendo o LLM + o pipeline de escrita, até os critérios de saída do shadow (§8).

## 1. Componentes

| Peça | Arquivo | Papel |
|---|---|---|
| `/v1/triage` | `Modules/onnxManager.py` | embed → score → MLP em uma chamada |
| `TriageModel` | `Modules/onnxManager.py` | lazy-load do ONNX, inferência em pool de background |
| `TriageClient` | `Modules/onnx_client.py` | client httpx igual a `EmbeddingClient`/`RerankerClient` |
| `POST /triage/classify` | `Modules/memory/memory_api.py` | ponte para o `LLM.py` (que não fala ONNX direto) |
| Feature builder | `Modules/memory/memory_triage_features.py` | **único** lugar que constróe o vetor (treino e inferência) |
| Treino | `scripts/train_memory_triage.py` | labels → split → MLP → calibração → ONNX → gold |
| Roteamento | `Modules/memory/memory_triage.py` | aplica as regras da §7 |

O `LLM.py` continua falando somente com a API de memória: basta adicionar
`"memory_triage_classify": ("POST", "/triage/classify")` ao mapa
`_MEMORY_TOOL_ROUTES` (L510).

## 2. Contrato do `/v1/triage`

O endpoint aceita **textos ou embeddings já calculados** (o chamador escolhe),
para honrar a ideia original de enviar texto já encoded ao e5-small:

```json
{
  "segments": [
    {"id": 0, "text": "troquei para o Postgres 17", "role": "user"}
  ],
  "candidates": [
    {"id": 41, "text": "O usuário usa Postgres 16.", "embedding": [0.01, ...]}
  ],
  "embeddings": null,
  "compute_cross_encoder": true,
  "priority": 0
}
```

- Se `embeddings` vier preenchido, os vetores dos segmentos são usados direto
  (o `EmbeddingEngine` já tem cache LRU+TTL do lado query — `memory_engines.py:84`).
- Os embeddings dos candidatos preferencialmente vêm do índice LanceDB via
  `get_vectors()` (`memory_vector.py:416`, cache por ID) — não há custo de embed.
- `compute_cross_encoder=true` dispara `/v1/score` internamente para os pares.
  Se falhar, a decisão é marcada como `abstain` (não se substitui a feature
  por 0 — isso muda a distribuição vista no treino).
- `priority=0` sempre: a triagem é background e **nunca** usa as pools
  `*_hi_pool` do recall interativo (`onnxManager.py:729-734`).

Resposta (uma passada, tudo de uma vez):

```json
{
  "keep": [{"segment_id": 0, "p_keep": 0.94, "p_abstain": 0.08,
            "type": "fact", "validity": "temporal", "importance": 0.71,
            "plan": {"fragment_index": 1, "frame": "CONFIG", "confidence": 0.91,
                     "slots": {"key": "embeddings server", "value": "2002",
                               "span": "o servidor roda na porta 2002"}},
            "rendered_text": "A configuração embeddings server do usuário é 2002."}],
  "pairs": [{"segment_id": 0, "candidate_id": 41, "cosine": 0.78,
             "ce_score": 0.91, "logits": {"ADD": -3.1, "NOOP": -1.2,
             "UPDATE": 4.4, "CONFLICT": -2.0, "MERGE": -2.8, "DISCARD": -5.0}}],
  "scopes": [{"segment_id": 0, "scores": {"": 0.12, "projeto-x": 0.81}}],
  "model_version": "triage_v1",
  "latency_ms": 7.4
}
```

`plan` e `rendered_text` são produzidos pelas cabeças E/F/G (§4) e consumidos
pelo renderizador determinístico da Fase 3. Nesta fase eles são **apenas
registrados** no log — nenhum texto é gravado a partir deles.

## 3. Features

Builder único em `memory_triage_features.py`, usado por `train_memory_triage.py`
e por `/triage/classify`. Qualquer mudança aqui exige novo `model_version`.

| Feature | Origem | Dimensão |
|---|---|---|
| `emb_seg` | e5-small, prefixo `query:` (ou fornecido) | 384 |
| `emb_cand` | LanceDB `get_vectors()` ou e5 `passage:` | 384 |
| `emb_seg ⊙ emb_cand` | produto elementwise | 384 |
| `|emb_seg − emb_cand|` | diferença absoluta | 384 |
| `cosine` | dot product (vetores L2-normalizados) | 1 |
| `ce_score` | `/v1/score` | 1 |
| `used_in_answer`, `rank`, `idade_cand` | log da Fase 1 | 3 |
| `role`, `n_chars`, `has_deictic`, `position` | segmento | 4 |
| `attempt_index` | log da Fase 1 | 1 |

Os quatro blocos de 384 passam por uma projeção linear aprendida para 64 dims
cada antes da concatenação → entrada final ~80 dims. Com 2-3 mil exemplos isso
treina em segundos na CPU e infere em < 1 ms.

## 4. Cabeças

**A — por segmento:** `KEEP`/`DISCARD`, `type` (`fact|preference|decision|instruction`),
`validity` (`permanent|temporal`) e `importance` (regressão). Multitask no mesmo
trunk.

**B — por par:** 6 logits (`ADD/NOOP/UPDATE/CONFLICT/MERGE/DISCARD`) sobre a
concatenação dos blocos projetados + features escalares.

**C — abstention:** binária, "isso vai para o LLM?". Rótulo positivo = casos em
que o teacher divergiu do aplicado na Fase 1 ou que o próprio modelo errou no
gold. É a válvula de segurança, não um limiar fixo de softmax.

**D — escopo:** ver §5. Roda em shadow nesta fase; a escrita continua usando o
`scope` do request (`WriteRequest.scope`, `memory_models.py:241`).

**E — fragmento (Fase 3):** índice do fragmento alvo entre as saídas do
`_split_query_segments()` (≤ 4 opções). Treinado pelo `fragment_index` da Fase 1.

**F — frame (Fase 3):** uma das 12 classes do inventário (`PREFERS`, `VERSION`,
`CONFIG`, `DECISION`, `EVENT`, `GENERIC`…). Treinada pelo `frame_label` da Fase 1.

**G — slots (Fase 3):** entidade, valor, versão e data por **cópia** de um
vocabulário fechado — entidades de memórias/cards/dicionários, números e datas
extraídos do segmento. Treinada pelos campos `slot_*` da Fase 1.

E/F/G alimentam o renderizador em `Modules/memory/memory_triage.py`, que na
Fase 3 escreve os fatos `ADD` sem nenhuma chamada ao LLM. Aqui elas apenas
produzem o plano para auditoria.

## 5. Cabeça D: armadilha e desenho correto

O rótulo ingênuo ("escopo da memória gravada") não serve: hoje o escopo é
**determinado pelo chat**, nunca pelo conteúdo. Treinar com o escopo do turno
como target e o próprio turno como feature produziria uma cabeça perfeita e
inútil — ela só copiaria a entrada.

Desenho correto:

- **Target**: o escopo real da memória que o segmento gerou (do log da Fase 1).
- **Features permitidas**: `emb_seg`, embedding do **centroide** de cada escopo
  (média dos embeddings das memórias daquele escopo), similaridades, citações
  explícitas de projeto/pessoa no texto. **O `scope` do turno NÃO entra como
  feature** — ele é só o rótulo.
- **Negativos**: centroids dos outros escopos, priorizando hard negatives
  (escopos com vocabulário parecido com o segmento).
- **Formato**: score por `(segmento, escopo)`, não uma classe fechada — novos
  escopos entram sem retreino.

Validação no shadow: a cabeça deve **recuperar o escopo atual do chat** na
maioria dos turnos. Os desacordos são o produto mais interessante desta fase —
eles apontam conteúdo que "não pertence" ao projeto corrente, que é exatamente
o sinal necessário para a Fase 4 (conversa contínua com roteamento por conteúdo).
Nenhum desacordo altera escrita nesta fase; ele vira log + métrica.

## 6. Treino e exportação

Pipeline em `scripts/train_memory_triage.py`:

1. `GET /triage/labels` → DataFrame com `applied_label` como ground truth.
2. Features pelo builder compartilhado (§3).
3. Split `group=session_id` + split temporal; gold set imutável separado.
4. Baselines em ordem: regressão logística → `MLPClassifier(hidden_layer_sizes=(64,),
   activation="relu", solver="adam")` → (só se o gold não fechar) HistGradientBoosting.
5. **Calibração Platt (sigmoid), não isotônica**: a sigmoid é apenas mais um
   `MatMul + Add`, exportável junto; isotônica não é.
6. Temperatura por cabeça ajustada no gold para o limite de falso `UPDATE`/`MERGE`.
7. Exportação com `onnx.helper`: `MatMul`/`Gemm` + `Relu` + `Add` + `Sigmoid`/
   `Softmax`, lendo `coefs_` e `intercepts_` do MLP. `onnx` já está instalado.
8. Artefato versionado: `Modules/memory/models/triage_v1.onnx` + `triage_v1.json`
   (dimensões, limiares, hash do dataset, versão do extractor teacher).

`TriageModel` no onnxManager é **lazy** (como `_ensure_rerank_model()`, L642):
arquivo ausente/corrompido → log + degradação para "sem triagem", nunca
`_fatal_exit` no boot (que é reservado ao embed, L583-590). Pré-aquecer em
background no startup, sem bloquear.

## 7. Regras de roteamento (quem decide o quê)

| Decisão da rede | Ação na Fase 2 | Ação na Fase 3 |
|---|---|---|
| `KEEP` + `ADD`, confiante | somente registrado em `shadow_pred` | renderizador escreve → `memory_write_facts_batch` (zero LLM) |
| `DISCARD`, confiante | somente registrado | nada |
| `NOOP`, confiante | somente registrado | nada, reforça o alvo (recall implícito) |
| `UPDATE` / `MERGE` / `CONFLICT` | LLM, como hoje | **sempre LLM**, candidatos pré-restritos, só gera o texto unificado |
| `p_abstain` alto ou `ce_score` indisponível | caminho completo do extrator | caminho completo do extrator |
| `episode` + `task_state` | sempre LLM | local por template + regras, LLM só em incerteza (Fase 3 §8) |

Na Fase 2 **nenhuma** linha desta tabela é aplicada: o extrator de produção continua
decidindo e gravando. A tabela descreve o destino previsto, para que o shadow meça
o que *seria* feito.

Isso preserva a regra do README de que evidência literal é obrigatória e que
conteúdo gerado pelo assistente sem confirmação não vira memória. O LLM continua
sendo o autor de toda reescrita de texto; a rede só decide o destino.

## 8. Shadow mode e critérios de saída

Cada turno produz duas decisões. `shadow_pred` registra o que a rede teria feito;
nada é aplicado. Painel: `GET /triage/stats` com matriz de confusão teacher×rede
por classe, top-N desacordos para inspeção, e contagem de "custo que seria
evitado" (% de decisões aceitas no limiar).

Saída do shadow só quando:

- ≥ 95% de concordância em `DISCARD` e `ADD`;
- falso `UPDATE`/`MERGE` < 1% no gold set;
- cobertura ≥ 60% no limiar escolhido (60% das decisões não precisam de LLM);
- nenhuma regressão nas métricas atuais do extrator.

Após a saída, manter 10% dos turnos indo ao LLM como **challenger** para detectar
drift do teacher ou do e5.

## 9. Flags

Todas ligadas por padrão (opt-out `=0`), documentadas no README:

| Flag | Default | Efeito |
|---|---|---|
| `AVA_MEMORY_TRIAGE` | `1` | liga `/v1/triage` e o roteamento |
| `AVA_MEMORY_TRIAGE_SHADOW` | `1` | shadow (Fase 2); `0` na Fase 3 |
| `AVA_MEMORY_TRIAGE_MODEL` | `v1` | versão do ONNX — rollback instantâneo |
| `AVA_MEMORY_TRIAGE_CANARY` | `10` | % de turnos com LLM challenger |
| `AVA_MEMORY_TRIAGE_SCOPE` | `1` | cabeça D em shadow |
| `AVA_TRIAGE_PLAN` | `1` | cabeças E/F/G produzem o plano de fato no shadow |

Rollback por flag não exige reinstalar modelo: basta `AVA_MEMORY_TRIAGE=0`.

## 10. Observabilidade

- `/metrics` do onnxManager: `triage_calls`, `triage_latency_ms`, `triage_version`.
- `/metrics` do orchestrator: `triage_accepted_total`, `triage_escalated_total`,
  `triage_disagreement_total` por classe, `triage_scope_disagreement_total`.
- `/status`: seção `memory_triage` com modelo ativo, versão e modo shadow.
- `/triage/stats`: matriz de confusão + cobertura por limiar + acurácia da cabeça D.

## 11. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Train/serve skew | feature builder único compartilhado |
| Deriva do teacher (LLM muda de versão) | `extractor_model`/`prompt_version` versionados; um modelo por geração; gold como árbitro |
| `/v1/score` falha na inferência | marcar `abstain` — nunca substituir por 0 |
| Cold-start do lazy load | pré-aquecimento em background no boot |
| Cabeça D aprender o escopo do turno | escopo do turno **não é feature**, só target; hard negatives entre escopos |
| Concordância enganosa (LLM também erra) | gold set imutável como único árbitro independente |
| MLP exportado divergir do sklearn | teste de paridade: mesmas features, `assert_allclose` sklearn vs ONNX no gold |

## 12. Testes

- Paridade sklearn ↔ ONNX no gold (logits e probabilidades).
- `compute_cross_encoder=false` e falha do `/v1/score` → `abstain`, não 0.
- Cabeça D: recuperar escopo atual em fixtures; não usar `scope` do turno como
  feature (teste de leak: embaralhar o scope do request não pode mudar a saída).
- Cabeças E/F/G: `fragment_index` válido, `frame` dentro do inventário de 12,
  slots copiados do vocabulário fechado — nunca inventados.
- Plano e `rendered_text` presentes na resposta e gravados no log em shadow,
  sem efeito de escrita.
- Regras da §7: `MERGE`/`CONFLICT` nunca gravam sem passar pelo LLM.
- Rollback `AVA_MEMORY_TRIAGE=0` → extrator atual idêntico ao da Fase 1.
- `python -m unittest discover -s tests`.

## 13. Critérios de aceite

- `/v1/triage` responde em lote com um único round-trip e `priority=0`.
- Shadow roda em 100% dos turnos sem latência perceptível e sem escrita indevida.
- Relatório de concordância por classe + curva cobertura×acurácia disponíveis.
- Cabeça D produz scores por escopo e o relatório de desacordo existe.
- Plano E/F/G e `rendered_text` disponíveis para o renderizador da Fase 3.
- Critérios de saída do shadow (§8) medidos, mesmo que ainda não atingidos.

## 14. Fora do escopo

- **Fase 3**: triagem assume autoridade; o renderizador escreve os `ADD` sem LLM;
  LLM fica só com unificações, duvidosos e narrativas de alta incerteza.
- **Fase 4**: conversa contínua — desacoplar sessão de escopo, usar a cabeça D
  para rotear por conteúdo, tópicos como blocos, `task_state` como pilha.
