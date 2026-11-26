# Triagem de memória — Fase 4: conversa contínua

Documento de planejamento. Continuação de `MEMORY_TRIAGE_PHASE1.md` e
`MEMORY_TRIAGE_PHASE2.md`.

**Objetivo:** eliminar a necessidade de chats separados. A Alpha passa a ter uma
conversa contínua — como conversar com uma pessoa — em que mudança de assunto é
um tópico (bloco de memória), não uma nova sessão, e o destino de cada memória é
resolvido pelo conteúdo do turno (cabeça D da Fase 2), não pelo chat em que o
usuário estava.

## 1. Decisões tomadas

| # | Decisão | Escolha |
|---|---|---|
| 1 | Unidade da conversa | **Thread global única**; pessoa/voz é metadado e filtro, não fronteira de sessão |
| 2 | Visualização de tópico | **Thread inteira com lente de tópico**; nunca uma fatia isolada de histórico |
| 3 | Escopo inferido | **Aplicado na hora, reversível com um clique**; confirmação explícita só em empate |
| 4 | "Novo chat" | **Mantido, redefinido como "Novo tópico"**; reset explícito via compactação |

Justificativas resumidas:

1. O objetivo é "uma conversa contínua como com uma pessoa" — singular. Identidade
   de voz já entra como contexto (`_format_voice_context_block`, cards de pessoa),
   então não precisa de segunda sessão. `person_id` já existe no schema
   (`Modules/session/db.py:66`) para virar filtro depois, de forma aditiva.
2. Se a UI mostrasse só as mensagens do bloco enquanto o modelo vê a janela
   conversacional global, a tela e o prompt divergiriam — o usuário não
   conseguiria explicar por que a Alpha "lembrou" de algo de outro tópico.
3. Fronteira rígida de chat era o que impedia vazamento entre projetos. Com
   roteamento por conteúdo, o erro precisa ser **visível e desfeito**, não
   silencioso (README L157: "falha de memória é tratada como bug").
4. O botão é memória muscular; mas ele precisa parar de criar `sid` novo, senão
   nada mudou. A necessidade real por trás de "novo chat" é escapar de um
   contexto confuso — isso é compactação (`POST /session/{id}/compact`), que
   preserva o ledger.

## 2. Diagnóstico: o que está colado hoje

| Conceito | Onde está |
|---|---|
| **Chat** | sessão raiz `parent_id IS NULL` com `project_id` fixo — `SessionStore.create()` (`store.py:29-52`), `newChat()` (`index.html:734-739`) |
| **Escopo** | `req.project or session.project_id` (`_resolve_turn_project()`, `orchestrator.py:454`) → `_ensure_memory_scope()` (L1647) → `_CURRENT_SCOPE` (L5612, L6541); leitura mescla projeto+geral (`lt_views()`, `memory_search.py:106`) |
| **Tópico** | não existe; o mais próximo é um bloco de memória |

O "Novo chat" é o substituto manual de algo que o sistema deveria fazer sozinho:
detectar mudança de assunto e trocar de foco/escopo.

**Achado de código:** `Manager/index.html` define `pact()` duas vezes — L1288
(cria/abre chat no projeto) e L1295 (só ativa o projeto globalmente). A segunda
sobrescreve a primeira, então hoje trocar de projeto nunca cria o chat naquele
projeto. A migração precisa absorver esse bug em vez de reproduzi-lo.

## 3. O que já serve sem reconstruir

- `session` com `parent_id` e histórico durável (`db.py:62-82`), compactação
  (`Modules/session/compaction.py`) e modo conversacional (`context.py`:
  `split_turns`, `_strip_tools`, orçamento) — suportam uma thread de meses.
- `_build_recall_query()` (`orchestrator.py:2322`) para déiticos e
  `/episodes/search` em `_recall_for_turn()` (L2737-2752) — a autobiografia.
- Blocos com hierarquia, `gist`, `cues` e `guide` (`memory_blocks.py`),
  `propose_block_from_memories()` (L797) e `memory_search(block=..., expand=true)`.
- Cabeça D da Fase 2: score por `(segmento, escopo)` **sem** usar o escopo do
  turno como feature — exatamente o que o roteamento por conteúdo precisa.

## 4. Arquitetura alvo

| Conceito | Entidade | Regra |
|---|---|---|
| **Thread** | uma `session` raiz (`parent_id IS NULL`), `ses_main` | única; preserva a continuidade literal |
| **Escopo** | slug de projeto, resolvido por turno | cabeça D + LLM só em ambiguidade |
| **Tópico** | bloco `kind=topic\|episode` | unidade de "mudei de assunto" |
| **Fork** | `session.parent_id` | ramificar um assunto, nunca criar raiz paralela |
| **Pessoa** | `metadata.person_id` nas mensagens | filtro de recall, não separação de thread |

## 5. Fase 4a — preparação (sem mudar a UX)

Objetivo: criar as estruturas e a infraestrutura de correção antes de ligar
qualquer roteamento automático.

### 5.1 Arquivo de task state

`Modules/session/task_state.py` hoje grava com
`ON CONFLICT(session_id) DO UPDATE` (L69-79) — **um objeto único**, sobrescrito.
`merge_task_state()` (L94) preserva campos não repetidos, mas não guarda história.

Mudanças:

```sql
CREATE TABLE IF NOT EXISTS session_task_state_archive (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    task_id      TEXT NOT NULL,
    goal         TEXT NOT NULL,
    status       TEXT NOT NULL,
    decisions    TEXT NOT NULL DEFAULT '[]',
    open_questions TEXT NOT NULL DEFAULT '[]',
    next_steps   TEXT NOT NULL DEFAULT '[]',
    constraints  TEXT NOT NULL DEFAULT '[]',
    topic_id     TEXT DEFAULT NULL,
    time_created REAL NOT NULL,
    time_archived REAL NOT NULL,
    UNIQUE(session_id, task_id)
);
```

- `set_task_state()` continua gerenciando o objeto ativo (compatibilidade).
- Novo `archive_task_state(store, sid, topic_id)`: chamado quando o extrator
  devolve `status=done` ou um `goal` diferente do atual.
- `format_task_state()` (L111) injeta o ativo + até 3 recentes, respeitando
  `TASK_STATE_MAX_CHARS=1200` (L21).
- O arquivamento preferencialmente também promove o objetivo a bloco
  `kind=episode` (`propose_block_from_memories()`).

### 5.2 Metadados de tópico e origem do escopo

Em `Modules/session/db.py` / `Projector`: `metadata.topic_id` em toda mensagem do
turno (best-effort) e `scope_source` na sessão (`explicit|inferred|default`).
`scope_source` não altera `project_id` — é só auditoria.

### 5.3 Log e correção de escopo

Em `orchestrator.py`:

- `session_scope_event(turn_id, session_id, inferred, confirmed, final,
  corrected, created_at)` — nova tabela no store, com índice por sessão.
- `POST /session/{id}/scope/correct`: recebe `turn_id`, `scope` e `apply_recent`
  (default `true`). Regrava o `scope` das memórias escritas naquele turno:
  SQLite + índice vetorial + grafo do escopo de origem/destino. Opera por ID de
  memória, nunca por texto.
- A correção vira rótulo para a cabeça D (`origin=human`, peso 1.5 — ver Fase 1).

### 5.4 Migração `ava-chats` → thread

Em `Manager/index.html`:

1. ler `localStorage["ava-chats"]` (formato antigo, L729-758);
2. escolher a sessão mais recente como `ses_main` (ou criar);
3. importar cada chat antigo como bloco `kind=topic` via `/blocks/write`, com
   `cues` derivadas do título e `gist` do primeiro `user`;
4. gravar `localStorage["ava-thread"] = {sid, migrated: 1, version: 1}`;
5. **nunca apagar** `ava-chats`; manter leitura do formato antigo por uma versão.
6. flag de migração com dry-run: `POST /topics/migrate?dry_run=1` no orchestrator,
   relatando quantos chats virariam blocos e quantos conflitos de `gist` existem.

## 6. Fase 4b — backend: escopo por turno

### 6.1 Resolvedor

`_resolve_turn_project()` (L454) deixa de ser "request > sessão = identidade":
`req.project` vira **dica opcional**. Ordem:

1. se `req.project` explícito e válido → usar, `scope_source=explicit`;
2. se a cabeça D estiver disponível → `/triage/classify` (Fase 2) com o texto do
   turno, pegar scores por `(segmento, escopo)`;
3. top-1 acima de limiar → usar, `scope_source=inferred`, registrar evento;
4. empate (top-2 com margin < `AVA_SCOPE_CONFIRM_MARGIN`) → pergunta curta ao LLM
   com os dois candidatos; fallback `general`;
5. nenhum candidato → `general`, `scope_source=default`.

`_ensure_memory_scope()` (L1647) continua igual — com escopo resolvido ele chama
`/project/ensure` (abre sem trocar o ativo do processo), o que preserva turnos
concorrentes. `_CURRENT_SCOPE` continua sendo o canal para tools e extração;
muda só a origem do valor.

`session.project_id` passa a ser "último escopo usado" (dica fraca), nunca
identidade da thread.

### 6.2 Indicador e desfazer

O SSE `meta` (L6554) passa a incluir `scope` e `scope_source`. A WebUI mostra um
chip discreto no composer com o escopo inferido e duas ações: "geral" e "projeto
Y". Clicar chama `/session/{id}/scope/correct` com o `turn_id` do evento `meta`
mais recente. Nenhuma confirmação modal em decisões confiantes.

### 6.3 Flags

| Flag | Default | Efeito |
|---|---|---|
| `AVA_CONTINUOUS_SCOPE` | `1` | roteamento de escopo por conteúdo |
| `AVA_SCOPE_CONFIRM_MARGIN` | `0.10` | margem abaixo da qual pergunta ao LLM |
| `AVA_SCOPE_CORRECT_RECENT` | `1` | correção re-escopa memórias do turno |
| `AVA_TASK_ARCHIVE` | `1` | arquivamento de objetivos |

Todas ligadas por padrão (opt-out `=0`), documentadas no README.

## 7. Fase 4c — UI: tópicos como lente

### 7.1 Estado

`S.chats` vira `S.topics` (alias por uma versão). Cada item:

```js
{ id, block_id, gist, scope, topic_id, ms:[], busy:false, ctl:null,
  usage:null, jobs:[], unread:false, hydrated:false }
```

Sem `sid` próprio: a thread é uma só. `chat()` continua existindo como alias de
`topic()` para não reescrever tudo de uma vez.

### 7.2 Histórico

- `hydrate()` (L764-796) vira paginado:
  `GET /session/{sid}/messages?topic=<id>&before=<id>&limit=150`.
- `msgToUi()` (L798-819) ganha badge de tópico e mostra claramente quando a
  visualização é filtrada: "visualizando: Deploy cliente X · ver thread completa".
- O botão "ver thread completa" limpa o filtro sem sair da tela.
- `renderChats()` (L1412-1428) passa a listar `gist` de bloco + badge de escopo
  inferido, com o mesmo padrão visual `.pj`.

### 7.3 Ações

| Ação antiga | Nova ação |
|---|---|
| `#nw`, `#addc` (L2053) | "Novo tópico": arquiva o atual em bloco, seta `topic_id` novo, mantém sessão e janela |
| `/clear`, `clear()` (L1207, L1254) | "Compactar agora": `POST /session/{id}/compact` (já em L2092) |
| `pact(slug)` (L1288/L1295) | uma única definição: muda o escopo da thread atual, sem criar sessão |
| clique num chat | foca um tópico (filtro), `hydrate()` com `topic_id` |
| subagente virar chat (L2078-2083) | job/tópico filho no drawer, sem raiz nova |

### 7.4 Detecção de tópico

Sinal composto, sem LLM: queda de similaridade entre o turno e os últimos N,
entidades novas, `task_state.status=done` + goal diferente. Ao disparar:
`propose_block_from_memories()` + `block_write(kind="topic")` com `gist`, `cues`
e `guide`. A retomada é `memory_search(block="nome", expand=true)`.

## 8. Fase 4d — legado

- Modo antigo (chats = sessões raiz) permanece atrás de `AVA_CONTINUOUS_SCOPE=0`
  **e** de uma preferência local `ava-thread.legacy=1`.
- `ava-chats` nunca é apagado. Se o usuário reverter, os blocos importados viram
  chats novamente (dry-run antes).
- Remoção definitiva só depois de uma versão inteira sem regressions reportadas.

## 9. Observabilidade

- `/metrics`: `continuous_turns_total`, `scope_inferred_total`,
  `scope_conflict_asked_total`, `scope_corrected_total`, `topics_created_total`,
  `task_archived_total`.
- `/status`: seção `continuous_conversation` com `thread_id`, tópico ativo,
  escopo atual e `scope_source`.
- `GET /session/{id}/scope-events`: eventos de escopo para auditoria.
- `GET /topics`: blocos `kind=topic|episode` da thread, com `gist` e contagem de
  mensagens por `metadata.topic_id`.
- WebUI: badge do escopo inferido + chip de desfazer no composer.

## 10. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Vazamento entre projetos (escopo errado) | correção em 1 clique que re-escopa memórias; log de eventos; confirmação no empate |
| Tela e prompt divergirem | tópico é lente, nunca partição; selo explícito "visualizando..." |
| Thread longa demais | compactação + modo conversacional; paginação de `/messages` |
| Migração perder chats | dry-run, `ava-chats` preservado, rollback por flag |
| Subagentes virarem raiz | `parent_id` continua exclusivo de forks/jobs |
| `pact()` duplicado | unificar numa única definição na 4c |
| Task state arquivado crescer | TTL/poda do archive; no máximo 3 recentes no prompt |
| Pessoas diferentes na mesma thread | `metadata.person_id` como filtro de recall; perfil explícito se necessário |

## 11. Testes

- Migração: fixture com N chats → 1 thread + N blocos, com dry-run e sem perda.
- Resolvedor: dica explícita, inferência confiante, empate, fallback `general`.
- Correção de escopo: re-escopar memória recém-gravada (SQLite + vetor + grafo).
- Task state: arquivar ao `done` e ao goal novo; prompt com ativo + 3 recentes.
- UI: `hydrate()` paginado com e sem `topic_id`; "novo tópico" não cria `sid`.
- Subagente: `parent_id` preservado, nenhuma raiz nova.
- Rollback `AVA_CONTINUOUS_SCOPE=0` → comportamento da Fase 3 intacto.
- `python -m unittest discover -s tests`.

## 12. Critérios de aceite

- Uma única sessão raiz para a conversa; nenhum botão cria `sid` novo.
- Escopo inferido por conteúdo em ≥ 90% dos turnos sem correção manual (medido
  no shadow da Fase 2 + eventos da Fase 4).
- Tópico visível como lente, com caminho explícito para a thread completa.
- "Novo tópico" preserva histórico; reset é compactação, não apagamento.
- Migração reversível por flag e sem perda de `ava-chats`.
- Subagentes seguem em `parent_id`, fora da lista de tópicos raiz.

## 13. Dependências das fases anteriores

- **Fase 2**: cabeça D fornece os scores de escopo — sem ela, o 4b cai para
  dica explícita + LLM.
- **Fase 3**: triagem barata sustenta o volume maior de turnos de uma thread
  contínua.
- O 4a (preparação: arquivo de task state, metadados, correção de escopo,
  migração) é independente e pode ser feito antes das duas.
