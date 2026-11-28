# Alpha AI (AVA — Alpha Virtual Assistant)

> **Any model. Every tool. Zero limits.**

Assistente de IA pessoal: uma **WebUI nativa** em Rust/WebKit (`Manager/ava-web`) com TUI opcional em Node/Bun, um Manager em Rust na bandeja do sistema (com chat widget ativado por wake word), um backend Python de microserviços orquestrado via Docker, um **mcp_manager embutido no tray** e um **pipeline de voz em C++** — ambos rodando no host sob supervisão do tray. Inferência via **OpenRouter** (qualquer modelo), memória de longo prazo em **grafo hebbiano**, **voz** (wake word + ASR streaming + identificação de falante + TTS streaming), **busca web** como servidor MCP próprio e **tools MCP extensíveis** em runtime.

---

## Arquitetura

O sistema é dividido entre o **container Docker** (serviços Python sem acesso a microfone/Chrome/servidores MCP stdio) e o **host** (WebUI nativa `ava-web`, tray, mcp_manager Rust embutido e pipeline de voz C++ — coisas que precisam de microfone, navegador, filesystem e execução real na máquina do usuário). A TUI em Ink continua disponível como interface opcional.

```
┌──────────────────────────────── HOST ────────────────────────────────────────┐
│                                                                              │
│  WebUI nativa (Manager/ava-web, WebKitGTK/WebView2)  Manager Rust (ava-tray)│
│  comando: alpha-ai (chat, grafo, CWD dos MCPs)       tray + API REST :9001   │
│  TUI opcional (TypeScript/Ink, build Bun)            /docker /browser /mcp   │
│    │  POST :9001/docker/start (a cada pergunta)      /fs/cwd /status         │
│    │  POST :9000/execute (SSE, por turno, cwd opcional)                      │
│    │  + Bearer AVA_INTERNAL_TOKEN em tudo                                    │
│    ▼                                                     │                   │
│  ┌────────────── Container "ava" (Python, FastAPI) ──────┼────────────────┐  │
│  │  orchestrator.py :9000   engine de tool-calling (SSE) │                │  │
│  │  LLM.py :4003            inferência (OpenRouter)      │                │  │
│  │  memory_api.py :3000     memória (SQLite+LanceDB+grafo)│               │  │
│  │  TTS.py :3004            fala (Supertonic, streaming) │                │  │
│  │  onnxManager.py :2002    embeddings E5 + reranker ONNX │                │  │
│  │  graph_activation.py       worker do grafo (stdio, on-demand)│          │  │
│  └────────────────────────────────────────────────────────┘                │
│    │ :9001 no host (tray API + rotas MCP no MESMO listener)                     │
│    ▼                                                                        │
│  mcp_manager (EMBUTIDO no tray, :9001)  — ÚNICO executor de servidores MCP; │
│    stdio (npx/uvx) + http, no processo do tray (sem binário standalone)     │
│    │                                                                        │
│    ▼                                                                        │
│  transcript (C++, :9100 /health + /speaker/embed) — voz contínua no host:   │
│    miniaudio → TEN VAD → wake word (sherpa-onnx KWS) → ASR streaming       │
│    (nemotron 0.6B GGUF) → speaker embedding CAM++ → POST /voice/ingest     │
│                                                                             │
│  Chrome normal do usuário (perfil padrão) p/ Playwright MCP (--extension)   │
│                                                                             │
│  Stack de busca web: searxng :8081 · valkey :6379 · crawl4ai :11235         │
│    alpha-mcp :4010 · docker-socket-proxy (allowlist p/ crawl4ai)            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Fluxo de uma pergunta (texto)

1. **Entrada**: a **WebUI nativa (`ava-web`)**, a TUI opcional ou o chat da bandeja envia o turno ao orchestrator: `POST :9000/execute` (SSE), com `source` (`"tui"` / `"chat"`) e Bearer token interno. A WebUI envia também `cwd` — a pasta de trabalho escolhida para o projeto ativo, usada pelos MCPs stdio daquele turno.
2. **Pré-loop**: o orchestrator garante a sessão de memória, consome o cache de contexto de voz (se houver embedding de falante pré-ingestado) e faz o recall inicial de memórias.
3. **Inferência**: o histórico + tools vão ao **LLM.py** (:4003), que chama o OpenRouter com function-calling nativo.
4. **Tools**: o loop resolve as chamadas em duas frentes:
   - **Nativas** (`list_plans`, `run_plan`, `memory_search`, `vision_objects` quando habilitada) — registradas direto no orchestrator;
   - **MCP** — descoberta em duas etapas via mcp_manager :9001 (grafo de co-uso escolhe os servidores → carrega as tools reais → executa via REST). O `cwd` do turno vai junto na chamada; o tray valida o caminho no host e spawna/respawna o servidor stdio nele. Antes de tools de browser, o orchestrator pede o Chrome ao tray (que o abre apenas se não estiver aberto).
5. **Memória**: leituras/escritas no **grafo hebbiano** (memory_api :3000) durante o loop — recall, correção e extração de novas memórias.
6. **Resposta final**: o texto encerra o turno (não existe tool "finish") e dispara **TTS automático** quando o cliente pediu áudio. Todos os eventos (delta, tool_call, done) são espelhados no **Conversation Hub** (`GET /conversation/stream`, SSE), que alimenta o chat widget do Manager em tempo real.

### Fluxo de voz (integrado ao pipeline principal)

1. O binário `transcript` (C++, **no host**, supervisionado pelo tray) abre o microfone via miniaudio e roda o **VAD** (ten-vad).
2. A **wake word** ("Alpha", via sherpa-onnx KeywordSpotter) é detectada; com `WAKE_GATE_ASR=true` o ASR só abre depois dela — o áudio acumulado da frase (incluindo a wake word) é reenviado ao ASR de uma vez. Frases sem wake word não gastam nenhuma inferência.
3. O ASR streaming (nemotron 0.6B GGUF, cache-aware com lookahead) transcreve o comando; a wake word **abre o chat widget do Manager**.
4. O **speaker embedding CAM++ (192 dims)** é enviado ao orchestrator via `POST /voice/ingest` — fora do caminho crítico do turno. O orchestrator pré-busca a memória de voz (kNN no `voice_dict`) e injeta no próximo `/execute` com custo 0 ms.
5. Cadastro de voz: pela WebUI/TUI (Configurações → Vozes), o microfone é gravado N segundos (`arecord`/`parec`/`ffmpeg`, Linux) e o embedding é gerado via `POST /voice/register` → `POST /speaker/embed` (:9100) → gravado no `voice_dict` (memory :3000).
6. A resposta final dispara **TTS automático** (Supertonic :3004) quando habilitado — o modelo nunca chama TTS.

O único jeito de encerrar o turno é **responder em texto puro** — não existe tool de "finish".

---

## Memória — como a AVA lembra

A memória é o centro do projeto: não é um histórico de chat com busca por cima, e sim um sistema que **decide o que merece ser lembrado, guarda isso de forma estruturada, traz de volta na hora certa e se corrige com o uso**. Ela responde a três perguntas o tempo todo:

1. **O que vale a pena guardar?** — triagem pós-turno, não despejo de tudo o que foi dito.
2. **Como isso volta quando importa?** — recall automático antes de responder, mais consulta ativa durante o trabalho.
3. **Como continua confiável?** — duplicatas, correções, conflitos, esquecimento e reforço fazem parte do ciclo.

O mesmo mecanismo vale para as **ferramentas**: a AVA também "lembra" quais tools existem, quando cada uma costuma ser útil e o que aprendeu usando cada uma.

### As camadas (e por que não é só um banco de fatos)

| Camada | O que guarda | Para que serve |
|---|---|---|
| **Histórico durável** (`Modules/session`) | Cada mensagem, tool call e resultado, na ordem | Continuidade literal da conversa, retomada exata, compactação de contexto |
| **Memória de longo prazo (LT)** | Fatos, preferências, decisões e instruções, um por ideia | O que a AVA *sabe* sobre o usuário, o ambiente e o trabalho |
| **Grafo hebbiano** | Arestas tipadas entre memórias (e entre tools) | Associação indireta: o que foi ativado junto tende a ser lembrado junto |
| **Blocos de memória** | Ponteiros para grupos coesos de átomos, com `gist`, `guide` e hierarquia | Ocupar 1 slot de recall por assunto, não 1 por fato; o detalhe fica a uma expansão de distância |
| **Episódios de curta duração (ST)** | Turnos/resumos episódicos com TTL curto | Contexto episódico no caminho CHAT legado; o fluxo principal usa o histórico durável |
| **Coleções laterais** | Dicionários (objetos, rostos, vozes), cards de pessoa, identidade da Alpha, arquivos indexados, planos | Contexto sensorial e material de referência que também é recuperável |
| **Catálogo de tools (ToolsDB)** | Cada tool MCP/skill: descrição, exemplos, queries aprendidas, dicas de uso | Escolher *quais* ferramentas carregar em cada turno, em vez de despejar todas |

O histórico é literal e não é "curado"; a memória de longo prazo é curada e vive no grafo. É essa separação que permite responder "o que eu acabei de dizer?" sem confundir com "o que eu sei sobre você?" — e vice-versa.

### Memória comum de agentes vs. memória de grafo

Quase todo agente hoje tenta lembrar de uma destas formas: um arquivo de instruções, o histórico inteiro no contexto, uma busca vetorial sobre conversas ou uma tool que o modelo precisa chamar para salvar e consultar. Os nomes e as bibliotecas mudam, mas o padrão se repete — inclusive nos assistentes de código que usam `CLAUDE.md`/`AGENTS.md` e nas bibliotecas de memória que expõem um `add`/`search` para o modelo chamar. Todas funcionam até certo ponto. O problema aparece quando a memória cresce.

| Abordagem | Como guarda | Como lembra | Onde quebra |
|---|---|---|---|
| **`AGENTS.md` / `CLAUDE.md` / regras do projeto** | Texto plano versionado, carregado inteiro | Está sempre no prompt | Orçamento fixo pago em todo turno; crescer demais faz o arquivo competir com a tarefa; fato velho e fato novo pesam igual; sem relevância, confiança, escopo ou esquecimento |
| **Contexto longo / histórico completo** | Mensagens acumuladas | Tudo vai no prompt; compacta quando estoura | Janela finita; custo e diluição de atenção; informação no meio do contexto se perde; compactar resume e destrói detalhe; não atravessa sessões |
| **RAG vetorial sobre conversas** | Chunks + embeddings | Top-k por similaridade | Parecido não é relevante; duplicatas e contradições convivem; um fato antigo pode voltar com o mesmo peso do atual; não existe associação entre fatos |
| **Memória por tool (salvar/consultar)** | Registros + busca | O modelo decide quando gravar e quando buscar | A qualidade depende da disciplina do modelo naquele turno; ele gasta contexto decidindo o que salvar; esquecer de salvar é silencioso; correção e duplicata ficam por conta dele |
| **Grafo (AVA)** | Fatos curados + arestas tipadas + escopos | Extração automática pós-turno + recall por similaridade e propagação + poda/reforço | Tem custo de extração e embedding; a extração pode errar; precisa de calibração; não substitui um arquivo de regras |

**O primeiro gargalo: quem gerencia a memória?** Na memória por tool, o modelo ganha um segundo trabalho: perceber que algo importa, lembrar de chamar a ferramenta no meio da tarefa, escrever bem, não duplicar, e mais tarde achar e corrigir. Isso consome exatamente o recurso que ele deveria usar para resolver o problema — a memória de trabalho. E a gravação fica dependente da disciplina do modelo naquele turno: tarefa longa, modelo distraído ou troca de modelo e o fato simplesmente não é salvo. Na AVA, a gravação não é uma decisão dentro do loop: é uma passada separada, depois da resposta, com prompt próprio, candidatos próprios e schema próprio. O modelo principal não precisa "lembrar de lembrar" — ele só conversa e executa. A extração ainda pode falhar, mas a falha é registrada, tem fallback e reaparece como aviso no turno seguinte; não é um silêncio.

**O segundo gargalo: memória de trabalho é finita.** Um `AGENTS.md` não é infinito, e o contexto do modelo também não. Um arquivo é tudo-ou-nada: 20 linhas custam pouco; 20 mil linhas não cabem, e se coubessem diluiriam a atenção e empurrariam a tarefa para fora. É por isso que "manter um arquivo atualizado" não escala como memória: cada linha ocupa janela em todo turno — mesmo com cache de prompt, ela desloca a tarefa — e não existe recall seletivo: ou o arquivo inteiro entra, ou nada entra. A memória humana não funciona assim: ela não mantém todas as lembranças na consciência; ela **reconstrói** a lembrança certa a partir de uma pista. O grafo faz o mesmo em termos de engenharia: separa o **armazenamento** (barato, em repouso, que não ocupa prompt e na escala pessoal cresce sem drama) do **conjunto de trabalho** (pequeno, relevante, montado a cada turno). Guardar milhares de fatos não aumenta o prompt nem o que o modelo precisa ler; só o punhado que a pergunta atual ativa entra em cena. O arquivo continua sendo ótimo para o que ele faz bem — regras explícitas, convenções e restrições auditáveis. O que ele não deve ser é um banco de fatos.

**O que o grafo imita do cérebro.** A analogia não é decorativa; ela produz propriedades concretas:

- **Codificação e consolidação** → a extração pós-turno transforma a experiência em memória estável, fora do caminho crítico.
- **Associação hebbiana** → memórias que foram ativadas juntas ganham arestas; a força muda com o uso e com o desuso.
- **Recordação por pista + ativação espalhada** → a pergunta atual semeia alguns nós e o Personalized PageRank espalha a ativação pelas associações, trazendo o que não casou na busca literal.
- **Inibição lateral** → memórias redundantes são suprimidas para não ocupar o espaço das distintas.
- **Curva do esquecimento** → a confiança cai com a inatividade; o que nunca volta perde força e é podado.
- **Reconsolidação / potenciação de longo prazo** → cada recall reforça a memória usada, de forma saturante.
- **Poda sináptica** → memórias abaixo do piso e arestas fracas são removidas; o grafo não guarda lixo morto.
- **Contexto** → escopos separam o que pertence a um projeto do que é geral, como o cérebro usa o contexto para desambiguar lembranças.
- **Consolidação em background** → decaimento, poda, compactação e reforço rodam em jobs, não no meio da conversa.

**Onde a analogia para.** Não é um cérebro: não há emoção, corpo, replay episódico real nem consciência. A extração é um LLM e pode errar; o auto-link pode criar associação fraca; o decaimento pode apagar algo que ainda seria útil se nunca foi evocado. Por isso o sistema é conservador onde importa: evidência literal, gate por cross-encoder, correção com `update`/`conflict`, arestas estruturais que não decaem e um piso de confiança antes de apagar. A memória é tratada como um organismo que envelhece — não como um log imutável nem como uma promessa de onisciência.

**O que muda na prática.** A pergunta deixa de ser "quanto cabe no prompt?" e passa a ser "o que merece voltar agora?". O limite deixa de ser o armazenamento — que é barato e, na escala pessoal, cresce sem ocupar contexto — e passa a ser a curadoria do que merece ser gravado. O usuário pode contar detalhes de dez projetos, mudar de ferramenta, corrigir um fato de meses atrás: o armazenamento cresce, mas o custo por turno continua limitado ao que o recall ativa. É o oposto do arquivo: guarda-se muito mais do que caberia em qualquer `AGENTS.md`, mas lembra-se por vez apenas o recorte que a situação pede. E as ferramentas seguem a mesma lógica: um catálogo grande de servidores MCP não precisa caber no contexto — só os poucos schemas relevantes entram, e o resto permanece disponível na memória de tools.

### O que vira memória: a triagem pós-turno

Depois que a resposta é entregue, um **modelo extrator dedicado** analisa a dupla pergunta-resposta em background. Ele não vê a conversa inteira: vê a troca atual, um contexto anterior curto (só para resolver pronomes como "ele", "aquilo", "o arquivo") e uma lista de memórias existentes relevantes. A resposta ao usuário nunca espera por isso.

**Como os candidatos são escolhidos.** O extrator só pode comparar com o que recebe — o que não entra na lista é invisível para ele. A lista combina três rotas:

- memórias que foram **usadas para construir aquela resposta** (prioridade máxima — se o turno as utilizou, uma correção nelas é urgente);
- busca semântica por **sentença** da pergunta e da resposta, não só pelo texto inteiro — cada frase pode puxar um assunto diferente;
- expansão pelo **grafo** (associações indiretas já existentes).

A lista é deduplicada, ordenada por prioridade e limitada a até 24 memórias (ajustável entre 15 e 30). Cada memória recebe um apelido temporário (`M1`, `M2`…) e o modelo devolve os apelidos; o código converte de volta para os IDs reais. ID inventado ou fora do mapa é descartado — o modelo não tem como "alucinar" um destino. No máximo 8 fatos são considerados por turno, priorizando os mais úteis.

**Como o extrator decide.** Para cada fato candidato, ele escolhe uma ação:

| Ação | Quando acontece | Efeito |
|---|---|---|
| `ADD` | Informação nova e útil | Cria uma memória nova |
| `NOOP` | Já existe algo igual ou mais completo | Não gasta nada |
| `UPDATE` | O fato mudou, ou complementa o que já existe | Reescreve a memória-alvo com a versão atual |
| `CONFLICT` | Contradiz uma memória e não há como saber qual vale | Guarda os dois; o antigo fica marcado como em conflito |
| `MERGE` | Duas peças só fazem sentido juntas | Grava a versão unificada no primeiro alvo; os demais permanecem |
| `DISCARD` | Efêmero, irrelevante, conhecimento genérico ou fala do próprio assistente | Não é memória |

Cada fato sai em **terceira pessoa, autocontido e atômico** — legível sem a conversa, sem pronomes soltos, com nomes, números, versões e caminhos preservados. Datas relativas ("ontem") são convertidas para absolutas. Todo fato carrega um **trecho literal de evidência** da conversa, o que impede a memória de virar interpretação livre.

**O que não entra.** Estado momentâneo ("estou com pressa"), perguntas que não revelam fato, conhecimento que qualquer um pesquisaria e — importante — **conteúdo que o assistente gerou e o usuário não confirmou, adotou ou usou**. Sugestão do modelo não se transforma em memória do usuário por conta própria. Fatos ditos pelo usuário têm prioridade; algo vindo do assistente só vira memória se o usuário confirmou ou aplicou, e mesmo assim é registrado como decisão/fato do usuário.

**Permanente ou temporária.** Cada memória nasce como `permanent` (identidade, preferência, decisão estrutural) ou `temporal` (versão atual, local, status de projeto). Isso muda a confiança inicial e, principalmente, se ela **decai** com o tempo. O tipo (`fact`, `preference`, `decision`, `instruction`) orienta como ela é lida depois.

**Quando a extração falha, isso não é silencioso.** Há uma cadeia de tentativas (formato estruturado → JSON simples → modelo de fallback). Se todas falharem, nada é gravado e o turno seguinte recebe um aviso interno de que a memória do turno anterior não foi salva — o modelo pode agir de acordo em vez de fingir que lembrou. Falha de memória é tratada como bug, não como detalhe.

Além do extrator, a memória recebe entradas por outros caminhos: o usuário pedindo explicitamente para lembrar (tools de memória do `alpha-mcp`, sempre disponíveis), importação de arquivos, dicionários sensoriais, cards de pessoa e o card de identidade da própria Alpha. Todos passam pelo mesmo pipeline de escrita.

### O que acontece no momento da gravação

A qualidade começa antes do grafo: a escrita é conservadora por desenho.

- **Duplicata exata** é recusada na hora.
- **Quase-duplicata** passa por dois estágios: a busca vetorial apenas *sugere* um candidato (o bi-encoder é bom de recall, mas grosseiro); quem **decide** é um cross-encoder, que compara o texto novo com o candidato de verdade. Se for a mesma informação → descarta. Se for o mesmo assunto com informação nova → não grava um fato solto: devolve um `possible_update` com o candidato, e o chamador reenvia a escrita como `update`, substituindo a memória certa. Se for apenas parecido → é fato novo.
- **Atualização não é "apagar e recriar"**: o nó mantém o mesmo ID e as arestas antigas; o texto e o vetor são trocados, e o conteúdo novo acrescenta associações. Quem apontava para aquela memória continua apontando.
- **Toda memória nova nasce conectada.** Uma task de auto-link busca vizinhos semânticos e só cria aresta quando o cross-encoder confirma relação. Sem isso, o grafo começaria cheio de nós ilhados esperando uma leitura futura que talvez nunca aconteça.
- **A ordem narrativa é preservada.** Fatos extraídos da mesma troca são gravados em lote e encadeados com arestas de precedência temporal (antes → depois), para que o contexto de um episódio não se perca.
- **Relações explícitas têm tipo**: `updates`, `temporal_precedence`, `contradicts` e `co_activation`. Arestas de correção e contradição são estruturais e não decaem.
- **Escopos são fronteiras rígidas**: arestas nunca cruzam escopo; IDs de outro escopo são recusados em vez de criar ligações fantasmas.

### Escopos: memória geral × memória de projeto

Existe um escopo **geral** e um escopo por **projeto**. A regra é simples e aparece na prática:

- **Escrita** acontece no escopo do projeto ativo (ou no geral, quando não há projeto).
- **Leitura** mescla o projeto ativo com o geral — o que é pessoal/estrutural continua visível em qualquer projeto; o que é específico de um cliente/trabalho não vaza para os outros.
- Cada chat tem um projeto fixo (que pode ser o geral); chats diferentes podem rodar em paralelo sem trocar o escopo um do outro.
- Criar um projeto (pela WebUI ou pela tool `project_create`) já o ativa; apagar um projeto remove o diretório inteiro — memórias, grafo, planos, arquivos indexados e índices vetoriais.

Na prática: uma preferência sobre "respostas curtas" gravada no geral acompanha o usuário em todos os projetos; um detalhe sobre "o servidor do cliente X" fica no projeto X e não aparece quando ele trabalha no projeto Y.

### Como o passado volta: o recall de cada turno

O recall começa **antes do primeiro token** e roda em paralelo com a descoberta de ferramentas, para não custar tempo de resposta.

1. **A pergunta é preparada.** Se o input é curto ou deítico ("faz isso", "e agora?", "continua"), ele é enriquecido com os últimos turnos — sem isso a busca semântica erra o alvo.
2. **Uma leitura só consulta fontes diferentes**: memória de longo prazo por semelhança, associações do grafo, episódios, dicionários (objeto/rosto/voz) e arquivos indexados. Consultas compostas são divididas em sub-pedidos e os resultados fundidos — "quero fazer X usando Y para conseguir Z" não deixa X ou Z afogados pelo termo mais central.
3. **O grafo entra em ação.** Os hits diretos viram sementes de um Personalized PageRank: a ativação se espalha pelas arestas e traz memórias que não casaram na busca, mas estão associadas às que casaram. Essas entradas vêm marcadas como **evidência fraca**, separadas dos fatos diretos.
4. **Ruído é cortado em camadas.** Um piso de relevância mais rígido que o normal tira o que só passou de raspão; um cross-encoder reordena os candidatos principais; uma inibição lateral (MMR) remove memórias redundantes do bloco "related"; e um orçamento de caracteres limita o total. O objetivo é lembrar o suficiente, não despejar o grafo inteiro no prompt.
5. **O bloco entra no system prompt como memória própria**, com confiança, idade e relevância por item. A instrução é explícita: usar naturalmente, sinalizar quando algo pode estar desatualizado ("pelo que eu lembro…"), verificar com uma tool quando o assunto importa e **nunca mencionar notas, scores ou o mecanismo interno** ao usuário.
6. **A voz tem um atalho.** O embedding do falante é processado fora do caminho crítico e a memória pré-buscada é consumida em O(1); o contexto de "quem está falando" entra antes do recall, com dedup contra os fatos já recuperados.

**O histórico não vem da memória de longo prazo.** A conversa literal vem do session store; a memória LT é o que foi aprendido. O contexto enviado ao modelo é montado por `Modules/session/context.py` em dois modos escolhidos pelo usuário no composer (`Conversacional` / `Execução`, alternável com clique ou `Shift+Tab`):

- **Conversacional** (padrão): últimos 6 turnos + orçamento de tokens (`AVA_ST_WINDOW_MAX_TOKENS`, 12000), episódios recuperados, fatos, blocos e task state. Tool calls de turnos anteriores saem do prompt; o episódio guarda o que foi útil delas.
- **Execução**: histórico durável completo, com compactação por overflow — o comportamento antigo, indicado para tarefas longas com muitas tool calls.

Quando o histórico cresce demais no modo execução, a compactação resume os turnos antigos e mantém uma cauda recente verbatim — o bloco de memória recuperada continua estável durante todo o loop.

**Recall ativo durante o trabalho.** O modelo também pode consultar a memória no meio do loop com a tool nativa `memory_search`. É o mesmo pipeline, com uma query mais focada, devolvendo o mesmo bloco formatado. Quando não encontra nada, a resposta orienta o modelo a não repetir a mesma busca com outras palavras e a seguir com o que tem.

### Blocos de memória: um assunto, um slot

Quando um conjunto de fatos vive sendo lembrado junto — "o deploy do cliente X", um procedimento, uma pessoa — o sistema pode promovê-lo a **bloco**. O bloco é um **ponteiro**, não um resumo substituto: ele guarda um `gist` de 1–2 frases, aponta para os átomos originais e ocupa **um único item** no orçamento de recall. Os fatos continuam sendo a fonte da verdade; o bloco é a unidade que a memória de trabalho enxerga.

- **Consolidação em background.** Um job agrupa clusters densos de co-ativação e cadeias temporais, aplica um gate (mínimo de membros, peso, evocações) e pede ao LLM um nome, `gist`, `cues` e `guide`. O mesmo conjunto de membros nunca vira dois blocos: o upsert é idempotente por hash dos IDs e por sobreposição (Jaccard).
- **Recall em duas etapas.** A busca vetorial consulta os blocos junto com os átomos; um bloco que casa entra como semente do PPR e desce para os membros. O prompt recebe o bloco como uma linha; até 3 átomos entram como sub-bullets quando o orçamento permite. `memory_search(block="nome", expand=true)` devolve o bloco inteiro sob demanda.
- **Hierarquia.** Blocos irmãos com membros sobrepostos ganham um bloco-pai (`parent_block_id` + `level`). O pai não copia membros: aponta para os filhos. O recall para no nível que couber e desce só se precisar.
- **Guia do bloco (o "AGENTS.md" do bloco).** Cada bloco pode carregar um `guide`: instruções curtas de **como interagir** com aquelas memórias (o que verificar, tom, precedência, quando usar `run_plan`). É **advisory e escopado**: só entra no prompt quando o bloco é ativado, nunca sobrepõe o sistema, o AGENTS.md global/projeto ou os próprios átomos. O extrator e a consolidação geram o guia; `PUT /blocks/{id}/guide` permite edição manual e um guia `manual` nunca é sobrescrito automaticamente. O histórico de versões fica em `memory_block_guide_history`.
- **Comportamento.** `kind=behavior` agrupa um plano, as tools usadas e as dicas de uso; `_recall_plans` consulta esses blocos antes de despejar a lista de planos no prompt. `kind=person|project` reaproveita cards e dicionários existentes, sem criar um sistema paralelo.
- **Decaimento e poda.** A confiança do bloco decai com o tempo; membros de um bloco vivo decaem mais devagar. Bloco abaixo do piso é podado — **os átomos permanecem** e voltam ao decaimento normal. Blocos-pai são estruturais e não são podados enquanto tiverem filhos.
- **Observabilidade.** `GET /blocks`, `GET /blocks/{id}`, `POST /blocks/consolidate` e `PUT /blocks/{id}/guide`; `/status` expõe a seção `memory_blocks`; `/metrics` inclui contadores de recall, expansão, consolidação, poda e guias.

### Episódios e tarefa atual

- **Episódio por turno.** Todo turno gera um episódio: resumo, intenção, resultado, ferramentas usadas, resultados úteis, entidades, decisões e pendências. Ele é gravado em `memory_episodes` (SQLite) + vetor LanceDB e funciona como a memória autobiográfica do turno. O ledger bruto continua guardando o texto completo; o episódio é o destilado que entra no recall.
- **Bloco de arco.** Vários episódios de uma mesma tarefa podem ser consolidados em um bloco `kind=episode`, formando o arco da conversa. O átomo (episódio) continua sendo a fonte da verdade; o bloco é o ponteiro.
- **Task state explícito.** O objeto `session_task_state` guarda `goal`, `status`, `decisions`, `open_questions`, `constraints` e entra no system prompt como `## Current task`. O extrator atualiza depois de cada turno; `GET/PUT /session/{id}/task-state` permite inspeção e edição manual.
- **Custo controlado.** O extrator grava fatos, episódio e task state na mesma chamada; o modo conversacional troca o histórico completo por uma janela curta + episódios, reduzindo tokens sem perder o fio da conversa.

**Correção em voo (opt-in).** No caminho de streaming do `LLM.py`, um vigia de confiança acompanha os logprobs por frase. Se a geração fica insegura no meio de uma frase, ela é pausada, a memória é consultada com o trecho em andamento, um classificador decide se a evidência confirma, complementa ou contradiz, e a resposta é retomada já corrigida — sem que o usuário veja o bloco interno. É opt-in porque adiciona latência; ligado, evita que uma resposta longa continue teimosa depois de contrariar o que a memória já sabia.

### Manutenção: esquecer também é projeto

Memória que só cresce vira ruído. O sistema tem um ciclo de vida:

- **Decaimento por inatividade.** A confiança de uma memória esquecível cai com o tempo (meia-vida configurável, por memória ou global). Memórias marcadas como não esquecíveis — identidade, preferência estável — ficam fora do decaimento.
- **Reforço pelo uso.** Cada recall que traz uma memória aumenta sua confiança de forma saturante e atualiza o último acesso. O que é usado com frequência fica forte; o que nunca volta enfraquece. É o "use e melhore" na prática.
- **Poda.** Abaixo de um piso de confiança, a memória é removida — e as arestas que apontavam para ela também, para o grafo nunca referenciar um nó morto.
- **Arestas também esquecem.** Arestas comuns decaem e são podadas; arestas de correção/contradição não, porque a história de uma mudança de fato importa.
- **Episódios expiram** depois de um TTL curto; os índices vetoriais são compactados periodicamente.
- **Correção preserva a história.** Não existe "última escrita ganha" cego: um `UPDATE` é uma decisão com evidência; um `CONFLICT` guarda as duas versões e marca a antiga, em vez de destruir o que havia. O usuário pode mudar de ideia sem que a memória reescreva o passado silenciosamente.
- **O grafo aprende com a leitura.** Memórias que aparecem juntas em um recall criam/reforçam arestas de co-ativação; o subgrafo percorrido pelo PPR também é reforçado. Usar a memória a reorganiza.

O resultado é uma memória que envelhece: fatos estáveis permanecem, fatos temporários somem, associações fortes ficam e ligações casuais se desfazem.

### O outro lado do carro-chefe: a memória de tools

O mesmo problema aparece nas ferramentas: dezenas de servidores MCP, cada um com dezenas de tools, não cabem no contexto nem devem ser todos carregados em toda pergunta. A AVA trata tools como memória — e aprende com o uso.

**Catálogo.** Cada tool MCP (e cada Agent Skill) vira um registro no mesmo tipo de grafo, com nome, descrição real, exemplos de acionamento e um embedding. O registro é idempotente por hash: mudou a descrição, re-indexa; não mudou, não faz nada; o vetor sumiu do índice, re-embeda (self-heal); a tool desapareceu do servidor, é podada. A descrição em inglês continua sendo a fonte da verdade; os exemplos em português existem para que pedidos como "liste os arquivos da pasta" encontrem a tool certa.

**Seleção por turno, em duas etapas.** Primeiro, uma busca semântica no catálogo (com max-pooling por tool e cota por servidor — um servidor grande não pode ocupar todos os slots) mais uma expansão pelo grafo de co-uso escolhe **quais servidores** têm a ver com o pedido. Se a confiança do topo é baixa, a rede abre em vez de arriscar uma escolha só. Ferramentas "core" e servidores always-on entram sempre. Só então, na segunda etapa, o orchestrator carrega o **schema real** apenas dos servidores escolhidos, já priorizando as tools que justificaram a escolha — o teto de contexto nunca descarta justamente a tool relevante. Skills têm um trilho próprio e não competem por slots de servidor, mas compartilham o grafo: usar uma skill e uma tool no mesmo turno cria arestas cruzadas.

**Aprendizado depois do turno.** As tools realmente usadas, na ordem, viram arestas de co-uso e de sequência; o texto do pedido vira um **exemplo aprendido** daquelas tools (com dedup e limite). Na próxima vez, uma pergunta parecida chega mais perto da ferramenta certa. É por isso que "as novidades anunciadas na última MineCon" passa a puxar a busca web depois de algumas experiências bem-sucedidas, mesmo sem nenhuma regra escrita ligando um termo ao outro.

**Dicas de uso (Tool Tips).** O próprio modelo grava notas sobre o uso real das ferramentas — "falha com payload acima de X", "chame A antes de B". Elas são **aditivas** à descrição do MCP (nunca a substituem), aparecem só para as tools carregadas naquele turno, têm limite por tool e o modelo é quem cura: para gravar uma nova, remove a mais fraca. Duplicata semântica vira atualização em vez de gastar slot. É a memória operacional da ferramenta, escrita por quem a usa.

### Na prática

- **Primeira conversa:** "prefiro respostas curtas e uso Postgres 16". A extração grava duas memórias; a versão tende a ser marcada como temporal e a preferência como permanente. Na conversa seguinte, o bloco de recall já traz as duas antes de o usuário repetir qualquer coisa.
- **Correção:** "na verdade troquei para o Postgres 17". O fato novo não vira uma terceira memória: a memória de versão é atualizada no mesmo nó, com o texto novo, e a associação antiga permanece.
- **Projeto:** "o deploy do cliente X usa a porta 8443". Isso fica no escopo do projeto X; ao abrir o projeto Y, o recall não mistura os dois — mas a preferência por respostas curtas, que é geral, continua lá.
- **Ferramenta:** um pedido de planilha usa uma skill de Office; no turno seguinte, um pedido parecido é selecionado mais rápido, e uma dica prática gravada no primeiro uso evita repetir o mesmo erro no segundo.

### Onde isso vive

| Peça | Onde |
|---|---|
| API de memória | `Modules/memory/memory_api.py` (:3000): `/read`, `/write`, `/write/batch`, `/read-short-term`, `/tools/select`, `/tools/record-usage`, `/tools/tips`, `/project/*` |
| Pipeline de busca e grafo | `Modules/memory/memory_search.py`, `memory_graph.py`, `memory_ops_memory.py` |
| Extração | `Modules/LLM.py` → `/memories/extract` (`_MEMORY_EXTRACTOR_SYSTEM_PROMPT`, `_extract_and_save_memories`) |
| Recall e injeção no prompt | `orchestrator.py` → `_recall_for_turn`, `_format_recall_block`, `_tool_memory_search` |
| Histórico durável e compactação | `Modules/session/` |
| Modos de contexto (conversa × execução) | `Modules/session/context.py` |
| Task state explícito | `Modules/session/task_state.py` |
| Episódios por turno | `Modules/memory/memory_episodes.py` |
| Tools e dicas | `Modules/memory/memory_ops_tools.py`, `memory_ops_tips.py` |
| Visualização do grafo | `graph_activation.py` + WebUI do Manager |

---

## Capacidades

### Núcleo (orchestrator + LLM)
- **Tool-calling nativo** via chat template (sem grammar manual): o loop recebe `tool_calls` estruturados, executa e realimenta o histórico como `role="tool"`, sem limite artificial de rodadas.
- **Qualquer modelo do OpenRouter** — o modelo é resolvido pelo backend; trocar modelo = trocar uma chave/config, sem rebuild.
- **Tools nativas** (`TOOLS` registry no orchestrator): `memory_search` (recall pessoal antes de inventar), `list_plans` + `run_plan` (planos híbridos de browser), `view_image` (abre uma imagem do host e a injeta no contexto como bloco multimodal — permite ao modelo ler fotos dentro de pastas usando `host_glob`/`host_list_dir`) e `vision_objects` (gate por `VISION_OBJECTS_ENABLED`, default 0 — veja Visão abaixo).
- **Imagens direto no modelo principal** (default): imagens anexadas pela TUI (`/image`) ou pela WebUI (um ou vários arquivos) vão como blocos multi-parte `image_url` (MIME detectado por magic bytes) verbatim ao OpenRouter — sem tool, sem vision.py. O orçamento de imagens (`AVA_IMAGE_BUDGET_TRIGGER_BYTES`/`_TARGET_BYTES`) remove as mais antigas quando o total passa do gatilho, cobrindo todos os anexos do turno.
- **Planos híbridos (macro + modelo)** para navegação web: passos "macro" (seletores fixos de Playwright, com resultado esperado verificado) e passos "modelo" (o LLM julga a tela e extrai dados estruturados), com allowlist de domínios e teto de ações. Dados ausentes voltam como `not_found` explícito — nunca inventados.
- **Memory-watch**: monitor de correção de memória — se a nova resposta contradiz uma memória usada, classifica (confirmação/complementação/correção) e reescreve o prefill da retomada.
- **Extração pós-resposta** de memórias de longo prazo, forçada por JSON schema.
- **Conversation Hub**: SSE com todos os eventos dos turnos (`/conversation/stream`) + catch-up por id (`/conversation/recent`) — reconexão do chat widget do Manager sem perder eventos.
- **Persona editável em runtime**: `GET/PUT /persona` no orchestrator — a 1ª linha do system prompt vive em `resource/persona.json` e é trocada pela WebUI/TUI (Configurações → Persona) sem restart (a cláusula de tools é sempre anexada pelo orchestrator).

### Agent Skills (SKILL.md) — playbooks no padrão skills.sh
- Skills no padrão **Agent Skills** (o mesmo do CLI `skills` do npm: `npx skills add owner/repo`) são lidas **direto do disco** (mounts ro no container — `npx skills` roda no host, o container enxerga na hora). Os dois instaladores já fazem a instalação automática: etapa 3b instala o pacote hyperframes (`npx --yes skills add heygen-com/hyperframes -a universal -y` — `-a universal` grava só em `.agents/skills` (store canônico, sem linkar em `.claude`/`.cursor`/etc), `-y` pula os prompts, re-run é idempotente e falha (rede) é aviso, não morte) e a etapa 3c instala o **OfficeCLI** (`npm install -g @officecli/officecli`, prefix do usuário; no Linux com node de apt/pacman usa `as_root`) + a skill `officecli` baixada de `officecli.ai/SKILL.md` direto para o store (temp + `mv` — SKILL.md truncado não entra no índice). Duas origens instaladas pelo usuário + uma criada em runtime, mesma mecânica:
  - **Projeto**: `.agents/skills/` + lock `skills-lock.json` na raiz do repo;
  - **Global**: `npx skills add -g` instala em `~/.agents/skills/` + lock `~/.agents/.skill-lock.json` (ou `$XDG_STATE_HOME/skills/` — variante do próprio CLI), montado em `/root/.agents/skills` no container. **Skill de projeto vence global em nome duplicado** (ordem = prioridade em `AVA_SKILLS_DIRS`); a proveniência segue o lock de cada diretório.
  - **Windows**: `npx skills add -g` grava em `%USERPROFILE%\.agents\skills`. Na rota A (Docker Desktop) o caminho literal entra direto na interpolação; na rota B (Docker Engine dentro do WSL) o `docker-start.bat`/`install.ps1` traduz com `wslpath` (mesmo esquema do diretório do projeto) e exporta a var via **`WSLENV`** (`AVA_WIN_GLOBAL_SKILLS/w` — sem isso o `wsl.exe` não leva env do Windows pro Linux). O override Windows interpola no mount `/root/.agents-win/skills` e o `AVA_SKILLS_DIRS` do perfil já inclui os quatro diretórios; fallback `/tmp` quando a var não vem (compose "na mão") — mount vazio, nunca derruba o stack.
  - **Criadas pelo agente (runtime)**: `.agents/skills-created/` — dir **separado** do store do CLI (o `npx skills` só reconcilia o lock dele e nunca toca aqui). É o único mount de skills `rw`: o orchestrator escreve via **`POST /skills/create`** (disparado pela tool `skill_create` do alpha-mcp) e o bind mount mostra no host também. Prioridade no scan: **projeto CLI > skills-created > global** — uma skill criada SOMBREIA a global de mesmo nome (nunca a de projeto: a rota recusa com 409).
- **Índice semântico no ToolsDB**: cada skill vira uma linha `agentskill__<nome>` no mesmo grafo hebbiano das MCP tools (kNN + PPR + queries aprendidas pós-turno). O seletor de **servidores** MCP (`/tools/select`) **exclui** o prefixo — skills não competem por slots de servidores; a seleção de skills usa o endpoint próprio `POST /agent-skills/select` (memory :3000). Co-uso numa mesma lista de `record-usage` cria **arestas skill↔tool** cruzando os dois mundos.
- **Por turno**: `/execute` seleciona as skills casadas com o pedido (top-K, env `AVA_SKILLS_TOP_K`) e injeta `nome + descrição` no system prompt (progressive disclosure — o playbook completo não vai por padrão). O modelo carrega o corpo com as tools nativas **`use_skill(name)`** e lê referências internas com **`skill_file(name, path)`** (leitura com guarda de path traversal via realpath + recusa de binário, em `Modules/agent_skills.py`).
- **Criação em runtime (o agente cria skills)**: a tool **`skill_create`** (alpha-mcp) pede **`POST /skills/create`** ao orchestrator, que escreve `.agents/skills-created/<slug>/SKILL.md` (+ arquivos extras de texto, lidos depois com `skill_file`) e **re-sincroniza o índice semântico na hora** — a skill já entra na seleção da próxima pergunta, sem restart nem esperar o TTL do scan. Slug kebab-case com folding NFKD de acentos (`Relatórios XLSX` → `relatorios-xlsx`); a **description é o driver da seleção semântica** (mín. 10 chars — vaga = skill nunca selecionada); escrita atômica (extras primeiro, SKILL.md por último — o scan nunca vê skill pela metade). Colisão de nome: skill do **projeto CLI** (prioridade maior) → **409 sempre**; skill **já criada** → **409** salvo `overwrite=true` (substituição total da pasta); **global** de mesmo nome → cria e sombreia (nota na resposta). A tool **`skill_list`** (alpha-mcp) mostra o catálogo completo (`GET /skills`) com origem, proveniência e status de índice.
- **Sincronização**: boot + assinatura por turno (nome/hash/always-on mudou → re-registra via `/tools/register`, idempotente por `doc_hash` com self-heal; skill removida do disco → prune). Env: `AVA_SKILLS_DIRS` (default: projeto + skills-created + global, substitui a lista inteira), `AVA_ALWAYS_ON_AGENT_SKILLS` (core — entram em TODA seleção), `AVA_SKILLS_TOP_K`. Inspeção: `GET /skills` e `POST /skills/sync` no orchestrator. Nomenclatura: **Agent Skills** (SKILL.md) ≠ "skills" MCP (servidores do mcp_manager).

### Memória — resumo
A memória é descrita em detalhe na seção **[Memória — como a AVA lembra](#memória--como-a-ava-lembra)**. Em uma frase: histórico durável + memória de longo prazo curada + grafo hebbiano de memórias e tools, com triagem pós-turno, recall em camadas, correção com evidência, decaimento e reforço pelo uso.

- **Histórico de conversa durável** (`Modules/session`) para continuidade literal; **LT curada** (SQLite + LanceDB) para o que foi aprendido; **grafo hebbiano** para associação indireta; **episódios** de vida curta.
- **Dicionários sensoriais** integrados: `visual_dict` (significado de objetos), `face_dict` (rostos) e `voice_dict` (vozes) — consultados por similaridade e pesquisáveis pela leitura normal de memória.
- **Cards de pessoa** (rostos + vozes + descrição) e **card de identidade da Alpha** (`GET/PATCH /identity`) ancorados por um nó estável no grafo.
- **RAG de arquivos locais** (`indexed-file`, hash SHA-256, chunking com posição) e **planos persistidos** (`/plans/*`), ambos recuperáveis junto com as memórias.
- **Grafo de tools (ToolsDB)**: o mesmo mecanismo escolhe quais servidores MCP carregar por turno, aprende com o uso real e guarda dicas escritas pelo próprio modelo.
- Visualização **em tempo real** do grafo no WebUI do Manager (`graph_activation.py` + `vis-network`), com memórias, tools como hexágonos e planos como diamantes.

### Visão (vision.py) — desligada por padrão
O pipeline completo (**Depth Anything V2** → clusterização por camadas → **EdgeSAM** → crops → **DINOv3** → consulta ao dicionário visual, mais **YuNet** + **EdgeFace** para rostos, tudo ONNX/CPU) existe e funciona, mas **não sobe por default**: a forma como foi projetado para ser usado (agente totalmente autônomo com visão constante) ainda não é suportada. Com `VISION_OBJECTS_ENABLED=0` (default), `start.sh` não inicia vision.py, a tool `vision_objects` não é oferecida e as imagens vão embutidas direto no contexto do modelo principal. Com a flag em `1`, o fluxo antigo volta (face recognition + segmentação → crops → resposta do modelo multimodal).

### Voz
- **Opt-in**: `AVA_VOICE=1` em `Modules/.env` (o instalador pergunta o perfil — econômico deixa desligado). O ASR nemotron GGUF fica residente no host (~600 MB); desligado, o tray pula o transcript no auto-init e `POST /transcript/start` sobe manualmente.
- **Pipeline C++ no host** (transcribe): wake word + ASR streaming + speaker verification, com gate de ASR por wake word (nenhuma inferência gasta sem comando). Supervisionado pelo tray como um processo do host (liveness por pid, restart-on-fail).
- **Build do transcript**: `./Modules/transcribe/build.sh` compila `main.cpp` para Linux (nativo) e/ou Windows x64 (cross-compile MinGW) contra os binários pré-compilados em `linux/`/`windows/`; detalhes e pré-requisitos em `Modules/transcribe/README.md`.
- **Identificação de falante**: embeddings CAM++ pré-ingestados injetam "quem está falando" no turno — o assistente sabe diferenciar usuários cadastrados no `voice_dict`.
- **TTS** (Supertonic, :3004): pipeline estilo Piper (worker pool, fila de texto, heap de áudio ordenado, miniaudio 40 ms): `/speak`, `/stream` (síntese paralela por sentença em ordem), `/cancel`. Disparado automaticamente sobre a resposta final; o modelo nunca decide chamar TTS.
- **Cadastro de vozes pela WebUI/TUI** (Configurações → Vozes): gravação local de microfone, edição de descrição, listagem — tudo via `voice_dict` na memory API.

### Busca web (alpha-mcp)
O Deep Search standalone foi **removido** — a busca web agora é um **servidor MCP próprio** (`mcp_servers/alpha-mcp`, :4010). A tool `web_search` esconde o pipeline: **cache Valkey (TTL 6h) → SearXNG → dedup/blocklist SEO → BM25 → fetch em camadas (trafilatura estático, Crawl4AI com navegador para os difíceis) → chunking por headers markdown → BM25 final** — o modelo recebe só trechos com fontes `[n]`. O crawl4ai sobe **on-demand** via **docker-socket-proxy** com allowlist mínima (start/stop de container apenas) — sem socket do Docker exposto. Tools de memória completam o servidor: `memory_write_batch` (grava fatos permanentes com cadeia `temporal_precedence` p/ o grafo) e `memory_people_card_create/edit/list` (cards de pessoa) — falando direto com a memory API. E as tools de **criação duradoura** falam com o **orchestrator** (:9000, Bearer interno): `project_create` (cria e **ATIVA** um escopo de memória próprio — memórias LT, grafo e planos isolados; mesmo fluxo da WebUI/TUI: criou → entrou) + `project_list`; `skill_create` (cria skill SKILL.md em `.agents/skills-created/` com re-sync do índice na hora — ver **Agent Skills**) + `skill_list` (catálogo completo); `artifact_preview` (publica na aba Artefatos um arquivo já gravado no cwd da sessão — HTML, PDF, imagem, modelo 3D, documento ou mídia; o Manager valida e serve o arquivo + assets irmãos; nada de conteúdo no histórico do LLM); e `host_read_file`, `host_list_dir`, `host_glob`, `host_grep`, `host_write_file`, `host_edit`, `host_shell` (leitura, escrita, edição, busca e shell **no HOST** via orchestrator → Manager, sem o isolamento de filesystem do container). No Linux o `alpha-mcp` usa `network_mode: host` para alcançar o orchestrator/memory/Manager em `127.9.12.22`.

### Tools MCP (extensibilidade)
- **mcp_manager reescrito em Rust e EMBUTIDO no tray** (`Manager/src/mcp_server.rs`) rodando **NO HOST**, fora do container — para que os servidores MCP tenham acesso real à máquina do usuário (npx/node/uvx do host, filesystem, Chrome do usuário). Suas rotas são servidas no MESMO listener da API do tray (:9001) — não existe mais binário standalone nem processo supervisionado.
- O orchestrator é **cliente REST** dele: registro, sync, listagem de tools e **execução** (`/servers/{name}/call`) ficam todos lá; este processo nunca spawna servidores MCP nem segura sessões.
- **CWD por turno**: a WebUI envia `cwd` no `/execute`; o orchestrator repassa o valor em cada `/servers/{name}/call`. O tray valida o caminho no host (o orchestrator roda no container e não enxerga o filesystem do host) e aplica `current_dir` no spawn do servidor **stdio**. Sem seleção, o padrão é a pasta base do usuário (`HOME`/`USERPROFILE`). Trocar de pasta derruba e respawna a sessão stdio daquele servidor; servidores **http** ignoram o CWD.
- A pasta escolhida **persiste por projeto** em `cache/webui_cwd.json` (chave = slug do projeto ativo) e é restaurada entre sessões. Os endpoints `GET/PUT/DELETE /fs/cwd` do tray gerenciam esse estado e só aceitam conexões loopback.
- Servidores são registrados/sincronizados/removidos **em runtime** — inclusive pela **WebUI** e pela **TUI opcional** (instalação assistida de servidores remotos, comandos locais pip/uv e endpoints localhost), sem reiniciar nada.
- **Playwright MCP** pré-configurado com `--extension`: conecta ao **Chrome real do usuário** via extensão "Playwright MCP Bridge" (instalada no perfil padrão). O orchestrator pede o Chrome ao tray, que o abre **apenas se ele não estiver aberto** — em modo normal, sem CDP nem cópia de perfil — e **o Chrome fica aberto** após as ações (processo detached do host, sobrevive ao docker). Suas sessões/logins do Chrome valendo. A autenticação da extensão é automática: o token `PLAYWRIGHT_MCP_EXTENSION_TOKEN` (`Modules/.env`) é injetado pelo tray no mcp_manager e herdado pelo servidor MCP — sem diálogo de aprovação por conexão.
- **Boardview MCP** (`mcp_servers/boardview/`) pré-configurado: servidor Python (stdio, `python -m boardview_mcp`) para diagnósticos de placa a partir de um boardview **GenCAD `.cad`** — rails, curtos, caminhos de alimentação, vizinhança e cruzamento com medições do técnico. Os instaladores criam o venv `.venv-boardview`, instalam o pacote (`mcp` + `pillow`) e cadastram o servidor no mcp_manager. **Skills próprias da Alpha:** a fonte versionada é `skills/<nome>/SKILL.md` (inclui `skills/boardview/SKILL.md` e `skills/math-calculator/SKILL.md`); os instaladores copiam cada pasta para `.agents/skills/<nome>/`, o store canônico (gitignored) lido pelo scanner e carregado pela tool `skill`. O recurso `boardview://skill` serve o playbook a partir de `mcp_servers/boardview/SKILL.md`. `load_board` aceita o arquivo, a pasta que o contém (`.` = cwd escolhido na WebUI) ou um glob (`*.cad`), então a placa da pasta selecionada é encontrada sem depender de outra tool de listagem. Sem `python3`/`venv` ou rede, a etapa é pulada com aviso. Alternativa Docker: `./scripts/setup-docker-mcps.sh` builda uma imagem Alpine (~161 MB) e registra o servidor como HTTP com autostart sob demanda, montando `$HOME` no mesmo path dentro do container.
- **CadQuery MCP** (`mcp_servers/3d_modeling/`): servidor Python **stdio** para modelagem CAD paramétrica e peças de impressão 3D — sólidos declarativos, furos, booleanas, medição, validação de imprimibilidade e exportação STEP/STL/3MF/AMF/DXF, com manifesto reproduzível. Os instaladores criam o venv `.venv-cadquery` (preferência por Python 3.12/3.11), instalam `mcp` + `cadquery` e registram `cadquery-3d-printing` no mcp_manager (`always_on=false`). As exportações usam `CQ_WORKDIR` ou, se ausente, a pasta do projeto selecionada na WebUI; `export_object` nunca escreve fora dela. A skill `cadquery-mcp` sai de `skills/cadquery-mcp/SKILL.md` e é instalada pelo loop genérico de skills. Sem Python/venv ou rede, a etapa é pulada com aviso. Alternativa Docker: `./scripts/setup-docker-mcps.sh` builda a imagem (`python:3.12-slim` multi-stage, CadQuery 2.4/OCP 7.7, sem VTK/CasADi/Numba) e registra o servidor como HTTP com autostart; o CadQuery é carregado só na primeira tool (~41 MiB de RAM em idle).
- **Aba ativa vs. tab group**: por padrão, o auto-connect com token ancora o MCP na aba do `connect.html` e a extensão oficial isola as abas controladas no grupo `Playwright`. No **Linux**, o `install.sh` empacota e instala automaticamente uma cópia patcheada como **external extension** (CRX próprio, novo ID, token fixo) e usa um wrapper `NODE_OPTIONS` para o Playwright MCP aceitar o novo ID — a aba controlada passa a ser a última aba ativa real e não há tab group. Use `AVA_PLAYWRIGHT_PATCHED_EXTENSION=0` para voltar à extensão da loja. Se o Chrome blocklistar o ID (`extensions.external_uninstalls`), feche o Chrome e rode `python3 scripts/clear_playwright_extension_blocklist.py`. No **Windows**, o `install.ps1` imprime o passo a passo do patch manual (`python scripts\patch_playwright_extension.py` + *Load unpacked*), porque o Chrome não aceita CRX local/registro externo fora da Web Store. Sem o patch, a alternativa é remover o token do MCP e escolher a aba manualmente em **“Allow & select”** no `connect.html`.
- **Windows**: URLs de servidores http cadastradas como `localhost:PORTA` são reescritas para `host.docker.internal` no momento da conexão (porta e path preservados).

### Manager Rust (ava-tray)
- Ícone na **bandeja do sistema** (GTK/Linux, tray nativo no Windows) + janela de status (egui) com start/stop do docker, status do mcp_manager e do transcript.
- **Chat widget**: abre com a wake word "Alpha" (detectada pelo pipeline de voz) ou manualmente; envia turnos escritos ao orchestrator (`source="chat"`, sem TTS — a fala é papel da WebUI/TUI/voz) e consome o Conversation Hub (SSE) para renderizar a resposta em tempo real.
- API REST (:9001): `/status`, `/docker/start`, `/docker/stop`, `/browser/start`, `/browser/stop`, `/browser/status`, `/mcp/*` (start/stop/status do mcp_manager), `/transcript/*` (start/stop/status do pipeline de voz), `/fs/cwd` (GET/PUT/DELETE do CWD por projeto, loopback-only) e `POST /ui/chat-open` (wake word do transcribe → abre o chat widget).
- **Preview de artefatos** (`/preview/*`, `/preview-stat/*`, loopback-only): registro por sessão do caminho de um arquivo no cwd — HTML, PDF, imagem, modelo 3D (glb/gltf/stl/obj), documento (md/txt/json/csv/yaml/xml), mídia e código. Serve o arquivo e assets irmãos direto do disco e devolve `kind` para a WebUI escolher o renderizador; `/preview-stat` (mtime/size) dispara o recarregamento quando o arquivo muda. O Manager é o único que lê arquivos do host — orchestrator/alpha-mcp só transportam `{path,title,kind}`.
- **Supervisão de processos do host**: transcript (:9100) — discover → spawn → health check periódico → restart-on-fail. (O mcp_manager não é mais um processo: é embutido no tray.) Variáveis de auth/config lidas do `Modules/.env`.
- **Inicialização automática**: o stack sobe no boot do tray; a WebUI exibe um popup enquanto o ambiente está subindo. O Docker só é encerrado quando o usuário pede (`/docker/stop` ou a UI).
- **Chrome do usuário** sob demanda: aberto em **modo normal** (perfil padrão, sem `--remote-debugging-port`) apenas **se o Chrome não estiver aberto** — spawn detached, sem cópia de perfil e sem cache `chrome-profile-cdp`.

### WebUI (Manager/ava-web)
- Janela nativa com WebView (`ava-web`) carregando o WebUI servido pelo tray em `:9001`: chat com streaming SSE, grafo hebbiano embutido, projetos e configurações (servidores MCP, vozes, planos, Docker).
- **Anexos múltiplos**: `📎` aceita vários arquivos/imagens de uma vez (ou colagem/arrastar); `📂` anexa **todas as fotos/arquivos de uma pasta**. Cada imagem vira um bloco `image_url` e cada arquivo de texto é injetado como documento nomeado. Limites por turno: 12 anexos, 7 MiB por imagem e 1,5 MB por documento; bits excedentes são ignorados com aviso. O modelo também pode buscar fotos sozinho com `host_glob`/`host_list_dir` e abri-las com `view_image`.
- **Vários chats ao mesmo tempo**: cada chat é uma sessão raiz (projeto fixo, histórico, uso e subagentes próprios). A lista **Chats** fica abaixo de Projetos; chats que continuam respondendo em background mostram o ponto âmbar e o contador de subagentes. `Ctrl+T` abre um chat novo, `Ctrl+I` abre/fecha os detalhes.
- **Faixa de sessão na tela do chat**: projeto, sessão, tokens de entrada/saída, cache hit, custo e subagentes ativos. O botão *Detalhes* (ou `Ctrl+I`) abre o drawer com a sessão completa, os subagentes (clicáveis para abrir o chat filho) e ações (`compactar contexto`, `fechar conversa`).
- **Timeline do turno**: pensamento e execuções de tools intercalados; cada resposta mostra tokens de entrada/saída, cache hit, custo e latência.
- **Grafo de memória**: nós de memória/tool/plano/pessoa/identidade **e blocos de memória** (ids negativos, `contains`/`part_of`, guia e hierarquia); revelação em onda no primeiro carregamento; filtros de arestas fracas/`part_of`, rótulos sob demanda e filtro por tipo. O worker é pré-aquecido no boot (`AVA_GRAPH_PREWARM`, default 1) e o snapshot traz `rev` para não ser aplicado duas vezes.
- **Aba Artefatos (full-screen)**: a tool `artifact_preview` publica um arquivo já gravado no cwd (`html_preview` segue como alias); a aba escolhe o renderizador pelo `kind`: iframe sandboxed para HTML, visualizador de PDF, `<img>` para imagens, `<video>`/`<audio>` para mídia, markdown/texto para documentos e código, e um visualizador 3D local (three.js vendorizado, órbita/zoom) para glb/gltf/stl/obj. Lista os artefatos daquela sessão num menu expansível no topo esquerdo e recarrega sozinha quando o arquivo muda (polling de `mtime`/`size` em `/preview-stat`) ou quando o usuário volta de Chat/Grafo. Assets irmãos (texturas de GLTF, CSS/JS de HTML) são servidos pelo Manager.
- **Seletor de pasta de trabalho dos MCPs**: o botão `📁`/chip de pasta abre o **diálogo nativo do sistema** (GTK na thread principal no Linux; `rfd` nativo no Windows/macOS). A escolha é enviada no `cwd` de cada `/execute`.
- O CWD é **por projeto ativo** e persistido entre sessões em `cache/webui_cwd.json`. Ao trocar de projeto, a pasta salva daquele projeto é restaurada; sem pasta configurada, usa a base do usuário (`~`/`C:\Users\%USERNAME%`).
- A seleção nativa existe apenas dentro do `ava-web`; não há fallback de navegador de pastas em HTML.

### TUI (terminal, Ink/React) — opcional
- Chat com streaming SSE, planos de execução (CoT) renderizados, timeline de tool calls e linha de status dos processos.
- Painel de **status** (CPU/RAM/GPU via systeminformation), painel de **logs do Docker** ao vivo (`Ctrl+D`), painel de **settings** (`Ctrl+S` ou `/settings`) com gestão de **servidores MCP** (instalação assistida npm/http/stdio e remoção), **vozes** (cadastro por microfone, edição de descrição e remoção), **planos** (lista/edição versionada) e **persona** editável, links para o grafo.
- Comando `/image <caminho>` para anexar imagens à conversa (vão direto ao modelo multimodal).
- **Autenticação interna**: toda requisição leva `Authorization: Bearer <AVA_INTERNAL_TOKEN>` (token do `Modules/.env` gerado pelo install.sh; nunca logado).
- **Binário standalone** (~80 MB, cross-compile Bun) — não exige Node no usuário. O instalador atual aponta `alpha-ai` para o WebUI; a TUI fica disponível no release em `tui/alpha-ai`.

---

## Segurança

- **Token interno compartilhado** (`AVA_INTERNAL_TOKEN`, gerado pelo install.sh em `Modules/.env` com chmod 600): memory (:3000), orchestrator (:9000) e as rotas MCP do tray (:9001) exigem `Bearer` — middleware dedicado em todos os serviços internos.
- **Sem portas expostas à LAN**: no Windows, todas as portas do container são publicadas apenas em `127.9.12.22`; no Linux, `network_mode: host` mantém os serviços em localhost. O tray (:9001) nunca é publicado.
- **docker-socket-proxy**: o alpha-mcp não tem acesso direto ao socket do Docker — apenas allowlist mínima (listar/inspecionar containers e start/stop do crawl4ai). Sem exec, images, volumes, networks ou build. No Linux, o proxy é publicado só em `127.9.12.22:2375`.
- **Host tools** (`host_read_file/list/glob/grep/write/edit/shell` do alpha-mcp): executam no processo do Manager, como o usuário do desktop, e exigem `Authorization: Bearer <AVA_INTERNAL_TOKEN>`; sem token a rota falha fechada. O shell é intencionalmente sem isolamento de container — é o acesso ao host pedido pelo usuário.
- **crawl4ai com token** (`CRAWL4AI_API_TOKEN`): sem ele o servidor só escuta no loopback do container.
- **SSRF guard**: `POST /search` e `POST /read-url` (orchestrator) e o fetch do alpha-mcp resolvem o DNS e bloqueiam IP privado/loopback/link-local/metadata — a validação acompanha redirects.

---

## Estrutura do projeto

```
├── orchestrator.py          # Engine de tool-calling (FastAPI/SSE, :9000) + Conversation Hub
├── graph_activation.py      # Worker headless do grafo hebbiano (JSON via stdio; render no WebUI)
├── Modules/
│   ├── LLM.py               # Cliente OpenRouter + prompts + memory-watch (:4003)
│   ├── TTS.py               # Supertonic streaming (:3004)
│   ├── vision.py            # Depth/SAM/DINOv3/YuNet/EdgeFace (:4002) — desligado por padrão
│   ├── onnxManager.py       # E5 embed + ms-marco rerank, API única (:2002)
│   ├── onnx_client.py       # Cliente httpx do onnxManager (embed/rerank/score)
│   ├── internal_auth.py     # Bearer token interno compartilhado (middleware)
│   ├── mcp_manager/         # Registry + db do MCP manager (o CÓDIGO agora vive no tray: Manager/src/mcp_server.rs)
│   ├── memory/              # Memória: grafo, vetores, dicionários, planos
│   │   ├── memory.py        #   fachada que reexporta os submódulos (API plana)
│   │   ├── memory_api.py    #   FastAPI (:3000)
│   │   └── memory_*.py      #   config/db/graph/vector/search/ops/status…
│   ├── session/             # Session store durável: eventos, mensagens, usage
│   ├── instructions/        # AGENTS.md: descoberta, baseline e deltas versionados
│   ├── codemode/            # Tool execute: JS sandbox + catálogo de tools MCP
│   ├── skills/              # Catálogo de SKILL.md + carregamento sob demanda
│   ├── observability/       # Métricas em memória + render Prometheus
│   ├── tools/               # Registry unificado de tools (validação/permissões/hooks)
│   ├── transcribe/          # Pipeline de voz em C++ (integrado!) — NO HOST
│   │   ├── main.cpp         #   VAD + wake word + ASR streaming + CAM++ (:9100)
│   │   └── linux/ windows/  #   builds por plataforma
│   └── Models/              # Pesos ONNX/GGUF (gitignored; via Models.zip)
├── mcp_servers/alpha-mcp/   # alpha-mcp (SearXNG→BM25→Crawl4AI, :4010)
├── Manager/                 # ava-tray (Rust: axum + eframe + tray-icon, :9001) + ava-web (WebUI nativa)
│   └── src/                 #   api, chat (widget), transcript, mcp_manager…
├── src/                     # TUI opcional (TypeScript/Ink; build Bun)
│   └── lib/                 #   auth (Bearer), voiceRecorder, voices, mcpServers, plans…
├── resource/                # ícones, settings do SearXNG, config alpha-mcp
├── docker-compose.yml       # stack base (+ overrides .linux/.windows)
├── start.sh                 # supervisor do container (restart-on-fail por serviço)
├── install.sh / install.bat # instaladores (baixam o release, geram tokens, models)
└── scripts/                 # patch-cfonts (TUI), plan_recorder, patch/empacotamento da extensão Playwright
```

## Serviços e portas

| Serviço | Onde roda | Porta | Papel |
|---|---|---|---|
| orchestrator.py | container | 9000 | API de chat (SSE) + tool loop + Conversation Hub |
| ava-tray (Manager) | host | 9001 | Controle do docker/browser/mcp_manager + chat widget |
| memory_api.py | container | 3000 | Memória (grafo/vetores/dicionários/planos) |
| TTS.py | container | 3004 | Síntese e playback de fala (Supertonic) — **opt-in** (`AVA_TTS_ENABLED=1`) |
| LLM.py | container | 4003 | Inferência (OpenRouter) — mesclado no processo do orchestrator (`AVA_MERGED_SERVICES=1`) |
| onnxManager.py | container | 2002 | E5 embeddings + rerank (API única ONNX) |
| graph_activation.py | container | — | Worker do grafo (stdio, sob demanda via `docker exec`); renderização embutida no WebUI |
| **mcp_manager (Rust)** | **host (embutido no tray)** | 9001 | Registro + execução de servidores MCP (mesmo listener da API do tray) |
| **transcript (C++)** | **host** | 9100 | `/health` + `/speaker/embed` (voz contínua, VAD/ASR/CAM++) |
| searxng | container | 8081 | Meta-busca (loopback) — on-demand (`WEB_STACK_ON_DEMAND=1`: sobe na 1ª busca, dorme após 5 min) |
| valkey | container | 6379 | Cache da busca web (loopback) — on-demand junto com o searxng |
| crawl4ai | container | 11235 | Crawl com navegador (loopback, on-demand) |
| alpha-mcp | container | 4010 | Tool `web_search` (loopback) |
| docker-socket-proxy | container | — (rede interna) | Allowlist do socket Docker p/ alpha-mcp |
| Chrome do usuário | host | — | Aberto pelo tray em modo normal (só se já não estiver aberto) |

---

## Requisitos

- **Linux** (recomendado; `network_mode: host`, áudio via PulseAudio, GPU opcional via Vulkan `/dev/dri`) ou **Windows** (Docker Desktop + WSL2; portas publicadas em `127.9.12.22` no override).
- Docker Engine + plugin compose v2; arquitetura **x86_64**.
- ~2,5 GB livres além do Docker (pesos extraídos) e internet no primeiro boot (build + download do Supertonic no HF).
- Áudio e GPU são **opcionais** — tudo roda em CPU (ONNX Runtime). Otimizações de memória já embutidas (`MALLOC_ARENA_MAX`, threads limitados do ONNX/Supertonic): o stack rodou de ~4,5 GB para ~1,4 GB — e, com o **perfil econômico** do instalador (default), fica em **~500 MB** sem voz (LLM + grafo mesclados no processo do orchestrator, TTS/graph UI/voz desligados, web stack on-demand). Voz ligada: ~1,1 GB.
- Para voz: microfone + `arecord`/`parec`/`ffmpeg` no Linux (gravação de voz pela WebUI/TUI ainda não suportada no Windows — o pipeline C++ de voz contínua tem build para ambas as plataformas). A voz é **opt-in**: `AVA_VOICE=1` em `Modules/.env` (o tray pulando o transcript economiza ~600 MB de RAM no host; `POST /transcript/start` sobe manualmente sem reiniciar nada).

## Instalação

**Linux**
```bash
curl -fsSL -o /tmp/alpha-install.sh \
  https://raw.githubusercontent.com/Lumazini-Code/alpha-virtual-assistant/main/install.sh \
  && bash /tmp/alpha-install.sh
```
**Windows**
```bat
curl -fsSL -o "%TEMP%\install.bat" https://raw.githubusercontent.com/Lumazini-Code/alpha-virtual-assistant/main/install.bat -o "%TEMP%\install.ps1" https://raw.githubusercontent.com/Lumazini-Code/alpha-virtual-assistant/main/install.ps1 && call "%TEMP%\install.bat"
```

O instalador gera os tokens internos (`AVA_INTERNAL_TOKEN`, `CRAWL4AI_API_TOKEN`) em `Modules/.env` — o stack não sobe sem eles. O **provider de inferência é cadastrado na primeira execução da WebUI**: um catálogo baseado no models.dev (sem o OpenCode Zen) lista os provedores OpenAI-compatible, e a escolha grava `LLM_BASE_URL`/`LLM_API_KEY`/`AVA_LLM_PROVIDER` no `.env` antes de reiniciar o ambiente. O **modelo é escolhido na tela de chat**, no seletor sob a caixa de escrita (favoritos fixados no topo; lista agrupada por provedor no formato “Modelo - Provedor”), e é enviado por turno como override — trocar de modelo não reinicia o stack. O cadastro deprecated do OpenRouter saiu dos instaladores. No Linux o instalador cria o comando `alpha-ai` apontando para `Manager/ava-web`; no Windows, um wrapper `alpha-ai.cmd` que abre `Manager\ava-web.exe`.

**Modo teste (antes do release oficial)**: com a env var `ALPHA_RELEASE_ZIP` apontando para um zip local, o instalador pula o GitHub inteiro e instala desse zip — mesmo fluxo, mesma validação:

```bash
# Linux
export ALPHA_RELEASE_ZIP=/caminho/alpha-ai-linux.zip && ./install.sh
```
```powershell
# Windows
$env:ALPHA_RELEASE_ZIP = 'C:\caminho\alpha-ai-windows.zip'; .\install.ps1
```

## Uso

```bash
alpha-ai          # abre a WebUI nativa (Manager/ava-web)
```
Digite a pergunta no chat da WebUI (o ambiente sobe sozinho sob demanda) e use o botão `📁` para escolher a pasta de trabalho dos MCPs do projeto ativo. Ou simplesmente **fale "Alpha"** — o pipeline de voz detecta a wake word, transcreve o comando e abre o chat do Manager. Configurações, servidores MCP, vozes, planos, logs do Docker e o grafo ficam no próprio WebUI. A TUI opcional continua acessível em `tui/alpha-ai` (com `/image` e painéis de terminal).

Logs e manutenção:
```bash
cd ~/.local/opt/alpha-ai
./docker-start.sh logs -f ava    # logs do backend
./docker-start.sh down           # derruba o stack manualmente
curl -s localhost:9001/status    # tray: docker + mcp_manager + transcript
curl -s localhost:9001/browser/status   # Chrome do host
```

## Configuração (principais)

| Variável | Onde | Default | O que faz |
|---|---|---|---|
| `OPENROUTER_API_KEY` | `Modules/.env` | — | **Deprecated**: removida no próximo salvamento do cadastro; use `LLM_API_KEY` |
| `LLM_BASE_URL` | `Modules/.env` | runtime: OpenRouter | Provider cadastrado na WebUI (qualquer API OpenAI-compatible — tabela abaixo) |
| `LLM_API_KEY` | `Modules/.env` | — | API key do provider (opcional em provider local: localhost/127.0.0.1) |
| `LLM_MODEL` | `Modules/.env` | `z-ai/glm-5.3-flash:floor` | Modelo fallback (a WebUI envia o modelo escolhido por chat como override) |
| `AVA_LLM_PROVIDER` | `Modules/.env` | — | ID do provedor escolhido no catálogo (exibição na WebUI) |
| `AVA_INTERNAL_TOKEN` | `Modules/.env` | — | Bearer token interno (memory/orchestrator/mcp_manager/TUI) |
| `CRAWL4AI_API_TOKEN` | `Modules/.env` | — | Token do crawl4ai (falha de boot se ausente) |
| ~~`AVA_DOCKER_IDLE_MINUTES`~~ | tray | — | Removido: o Docker não é mais encerrado por inatividade; só por `/docker/stop` ou pela UI |
| `AVA_TTS_ENABLED` | `start.sh` | `0` | Liga o serviço de TTS (Supertonic, ~470 MB/worker na RAM). Desligado = fala não sintetiza |
| `AVA_VOICE` | tray | `0` | Liga o pipeline de voz no host (wake word + ASR, ~600 MB). Desligado = transcript não sobe no auto-init; `POST /transcript/start` sobe à mão |
| `AVA_MERGED_SERVICES` | orchestrator/start.sh | `1` | LLM (:4003) mesclado no processo do orchestrator (um interpreter Python). `0` volta ao modo standalone |
| `AVA_SESSION_STORE` | LLM.py (CHAT) | `1` | Fallback do modo CHAT; o orchestrator usa o session store sempre |
| `AVA_SESSION_DB_PATH` | orchestrator | `Modules/session/db/ava_sessions.db` | Caminho do SQLite de sessões/mensagens |
| `AVA_HISTORY_MAX_CHARS` | orchestrator | `0` | Teto opcional do histórico injetado; `0` deixa a compactação controlar o tamanho |
| `AVA_CONTEXT_MODE` | orchestrator | `conversational` | Modo padrão do composer; `execution` restaura o histórico completo |
| `AVA_ST_WINDOW_TURNS` | orchestrator | `6` | Turnos verbatim na janela conversacional |
| `AVA_ST_WINDOW_MAX_TOKENS` | orchestrator | `12000` | Orçamento de tokens da janela conversacional |
| `AVA_ST_KEEP_PRIOR_TOOLS` | orchestrator | `0` | `1` mantém tool calls de turnos anteriores na janela |
| `AVA_EPISODES` | memory | `1` | Liga a gravação/recuperação de episódios por turno |
| `EPISODE_TOP_K` | memory | `4` | Episódios recuperados por recall |
| `EPISODE_MIN_SCORE` | memory | `0.35` | Piso de similaridade para um episódio entrar no recall |
| `EPISODE_MAX_CHARS` | memory | `1200` | Teto do texto do episódio |
| `AVA_TASK_STATE` | orchestrator | `1` | Liga o objeto explícito de tarefa atual |
| `AVA_MEMORY_BLOCKS` | memory | `1` | Liga blocos de memória (ponteiros para grupos de átomos); `0` desliga sem apagar dados |
| `AVA_MEMORY_TRIAGE_LOG` | memory/LLM | `1` | Instrumenta a extração: registra o proposto, o aplicado e o suprimido em `memory_triage_*`; `0` desliga |
| `AVA_MEMORY_TRIAGE_LOG_TEXT` | memory | `1` | Guarda o texto dos segmentos; `0` = só hash |
| `AVA_MEMORY_TRIAGE_PROBE` | memory/LLM/orchestrator | `1` | Habilita o probe relaxado (orçamento maior, sem gravar memórias) |
| `AVA_MEMORY_TRIAGE_PROBE_RATE` | orchestrator | `0.08` | Fração de turnos com um probe relaxado extra (máx. `0.5`) |
| `AVA_MEMORY_TRIAGE_PROBE_MAX` | memory | `3` | Máximo de segmentos sondados por turno |
| `AVA_MEMORY_TRIAGE_PU` | memory | `1` | Trata `unlabeled` como não rotulado — nunca como negativo |
| `AVA_MEMORY_TRIAGE_RENDER_LABELS` | memory | `1` | Deriva frame/fragmento/slots de `evidence`↔`text` para a Fase 3 |
| `AVA_MEMORY_TRIAGE_RETENTION_DAYS` | memory | `90` | Poda do log; `0` desliga |
| `AVA_MEMORY_TRIAGE_SHADOW` | memory | `0` | Reservado para a Fase 2 |
| `AVA_MEMORY_TRIAGE_TARGET_SAMPLES` | memory | `2000` | Alvo de rótulos exibido na barra/notificação da WebUI |
| `AVA_MEMORY_TRIAGE_GATE` | memory/LLM | `1` | Gate cross-encoder local via `/write` com `dry_run` (rótulo `gate_label`, peso 0.6) |
| `AVA_MEMORY_TRIAGE_SEGMENT_PROBE` | LLM | `1` | Probes por segmento suprimido (≤3/turno, modelo extrator) |
| `AVA_MEMORY_TRIAGE_SEGMENT_MIN_CHARS` | memory/LLM | `20` | Tamanho mínimo do segmento sondado |
| `AVA_MEMORY_TRIAGE_CONTRASTIVE` | LLM | `1` | Pares "lembre-se disso" com/sem cue (peso 0.8) |
| `AVA_MEMORY_TRIAGE_SYNTHETIC` | LLM | `1` | Perturbações sintéticas aceitas só se o teacher confirmar |
| `AVA_MEMORY_TRIAGE_SYNTHETIC_RATE` | LLM | `0.05` | Fração de turnos com sintético (limite de custo) |
| `AVA_MEMORY_TRIAGE_SYNTHETIC_MAX` | LLM | `2` | Máximo de perturbações por turno (teto 25% do batch) |
| `AVA_MEMORY_TRIAGE_THINKING` | LLM | `1` | Habilita `probe=thinking` (CoT só como probe, nunca `applied`) |
| `AVA_MEMORY_TRIAGE_PAIR_TOP_K` | memory | `5` | Candidatos por linha de par (`target`/`related` + top-K) |
| `AVA_MEMORY_TRIAGE_GOLD_TARGET` | memory | `200` | Alvo do gold set revisado |
| `BLOCK_MIN_MEMBERS` | memory | `3` | Mínimo de átomos para consolidar um bloco |
| `BLOCK_MIN_DENSITY` | memory | `0.5` | Densidade mínima de co-ativação do cluster |
| `BLOCK_MIN_AVG_WEIGHT` | memory | `0.25` | Peso médio mínimo das arestas internas |
| `BLOCK_MIN_EVOCATIONS` | memory | `2` | Uso mínimo do cluster antes de consolidar |
| `BLOCK_TOP_K` | memory | `4` | Blocos semeáveis por leitura |
| `BLOCK_MIN_SCORE` | memory | `0.78` | Piso de similaridade para semear um bloco |
| `BLOCK_EXPAND_MAX_MEMBERS` | memory | `3` | Membros desdobrados por bloco no recall |
| `BLOCK_EXPAND_MAX_CHARS` | memory | `1200` | Orçamento de desdobramento por bloco |
| `BLOCK_MEMBER_DECAY_FACTOR` | memory | `0.5` | Fator de desaceleração do decaimento de membros de bloco |
| `BLOCK_PRUNE_THRESHOLD` | memory | `0.15` | Confiança abaixo da qual o bloco é podado |
| `BLOCK_CONSOLIDATE_INTERVAL_S` | memory | `900` | Intervalo do job de consolidação |
| `BLOCK_PARENT_MIN_CHILDREN` | memory | `3` | Filhos necessários para formar um bloco-pai |
| `BLOCK_PARENT_OVERLAP` | memory | `0.3` | Sobreposição (Jaccard) para agrupar irmãos sob um pai |
| `BLOCK_GUIDE_MAX_CHARS` | memory | `1200` | Tamanho máximo do guia do bloco |
| `BLOCK_GUIDE_MAX_BULLETS` | memory | `6` | Máximo de bullets no guia |
| `BLOCK_GUIDE_INJECT_MIN_LEVEL` | memory | `2` | Nível a partir do qual o guia entra no prompt |
| `AVA_COMPACTION` | orchestrator | `1` | Liga a compactação automática (resumo + cauda verbatim) |
| `AVA_COMPACTION_CONTEXT_TOKENS` | orchestrator | `128000` | Janela assumida para o gatilho de compactação |
| `AVA_COMPACTION_BUFFER` | orchestrator | `0.15` | Fração da janela mantida livre antes de compactar |
| `AVA_COMPACTION_KEEP_TOKENS` | orchestrator | `15000` | Tokens recentes mantidos verbatim |
| `AVA_INSTRUCTIONS` | orchestrator | `1` | Liga a descoberta/injeção de AGENTS.md e instruções configuradas |
| `AVA_INSTRUCTIONS_DIRS` | orchestrator | repo + CWD | Diretórios onde procurar AGENTS.md (separados por `:` ou vírgula) |
| `AVA_INSTRUCTION_FILES` | orchestrator | — | Arquivos/globs extras de instruções |
| `AVA_GLOBAL_AGENTS_MD` | orchestrator/tray | `~/.config/alpha-ai/AGENTS.md` (Linux/macOS) ou `%APPDATA%\alpha-ai\AGENTS.md` (Windows) | Arquivo global de instruções |
| `AVA_TOOL_REGISTRY` | orchestrator | `1` | Registry unificado de tools (validação, permissões, hooks) |
| `AVA_TOOL_ALLOW` / `AVA_TOOL_DENY` | orchestrator | — | Padrões glob de allow/deny de tools (ex.: `browser__*`) |
| `AVA_ALWAYS_ON_MCP_SERVERS` | orchestrator | `alpha-mcp` | Servidores MCP sempre carregados e nunca descartados pelo teto de tools |
| `AVA_GRAPH_PREWARM` | tray | `1` | Pré-aquece o worker do grafo no boot (snapshot em cache ao abrir a aba Grafo); `0` desliga |
| `AVA_MCP_STARTUP_SYNC` | tray | `1` | Re-sincroniza as tools dos servidores MCP no boot quando a memory_api já está online (necessário após reset/recriação do ToolsDB); `0` desliga |
| `AVA_CODEMODE` | orchestrator | `1` | Expõe a tool `execute` e move tools MCP para o catálogo JS |
| `AVA_CODEMODE_TIMEOUT_MS` | orchestrator | `180000` | Timeout de uma execução Code Mode (3 min) |
| `AVA_CODEMODE_MAX_TOOL_CALLS` | orchestrator | `0` | Máximo de chamadas de tool dentro de um `execute`; `0` = sem limite |
| `AVA_EXECUTE_MAX_STEPS` | orchestrator | `0` | Teto opcional de passos do loop ReAct; `0` = sem limite (o turno termina quando o modelo responde sem tools) |
| `AVA_CODEMODE_MAX_OUTPUT_CHARS` | orchestrator | `200000` | Teto de saída de uma execução Code Mode |
| `AVA_SKILL_CATALOG` | orchestrator | `1` | Injeta o catálogo completo de skills e usa a tool `skill` |
| `AVA_SKILL_ALLOW` / `AVA_SKILL_DENY` | orchestrator | — | Filtro glob de skills no catálogo e no carregamento |
| `AVA_IMAGE_BUDGET_TRIGGER_BYTES` | orchestrator | `26214400` (25 MiB) | Gatilho para remover as imagens mais antigas do request |
| `AVA_IMAGE_BUDGET_TARGET_BYTES` | orchestrator | `15728640` (15 MiB) | Alvo após o corte de imagens |
| `AVA_SESSION_RUNNER` | orchestrator | `1` | Recarrega histórico do store por step e trata `finish_reason=length` |
| `AVA_SUBAGENTS` | orchestrator | `1` | Liga a tool `subagent` e jobs em background |
| `AVA_SUBAGENT_MAX` | orchestrator | `3` | Máximo de subagentes simultâneos por sessão pai |
| `AVA_MAX_CONCURRENT_TURNS` | orchestrator | `8` | Teto de turnos concorrentes no processo (`0` desliga o limite) |
| `AVA_MCP_MAX_SESSIONS` | tray/mcp_manager | `16` | Teto de sessões MCP vivas (chave servidor + cwd); a mais antiga é derrubada |
| `AVA_MCP_EXCLUSIVE_SERVERS` | tray/mcp_manager | `playwright,playwright-mcp` | Servidores cujas chamadas são serializadas (automação de browser) |
| `ONNX_EMBED_HI_WORKERS` / `ONNX_RERANK_HI_WORKERS` | onnxManager | `1` / `1` | Workers dedicados ao recall interativo (não competem com indexação em background) |
| `WEB_STACK_ON_DEMAND` | alpha-mcp | `1` | searxng + valkey sobem na 1ª `web_search` e dormem após 5 min sem busca (`WEB_STACK_IDLE_STOP_S`) |
| `AVA_CHROME_PATH` | tray | autodetect | Caminho do Chrome (aberto pelo tray em modo normal) |
| `PLAYWRIGHT_MCP_EXTENSION_TOKEN` | `Modules/.env` | — | Token da extensão "Playwright MCP Bridge" (injetado no @playwright/mcp; conexão sem aprovação manual) |
| `AVA_PLAYWRIGHT_PATCHED_EXTENSION` | `install.sh` | `1` | Linux: empacota/instala a extensão Playwright patcheada (aba ativa, sem tab group) como external CRX. `0` usa a extensão da Chrome Web Store |
| `AVA_CDP_SERVERS` | orchestrator | `playwright` | Servidores MCP que usam o Chrome do host |
| `MCP_MANAGER_URL` | orchestrator | `localhost:9001` | Onde o mcp_manager está (embutido no tray) |
| `TRANSCRIBE_HTTP_URL` | orchestrator | `localhost:9100` | Mini HTTP do transcript (host) p/ `/speaker/embed` |
| `VISION_OBJECTS_ENABLED` | orchestrator | `0` | Liga o fluxo vision_objects (sobe vision.py à parte) |
| `MCP_CALL_TIMEOUT` | orchestrator | 180 s | Teto de execução de tools MCP sem override |
| `AVA_LOCALHOST_REWRITE` | compose (Windows) | `host.docker.internal` | Reescrita de URLs localhost de servidores MCP http |
| `AVA_API_BASE` / `AVA_ORCHESTRATOR_URL` | WebUI / TUI / Manager | :9001 / :9000 | Endpoints dos componentes |
| `AVA_TTS_OUTPUT` | TUI | `client` | Onde a fala toca: `client` (a TUI toca no host via pacat/aplay/ffplay) ou `device` (container, via PulseAudio) |
| `ONNX_PROVIDERS` | compose | `CPUExecutionProvider` | Provedores do ONNX Runtime |

**CWD dos MCPs stdio**: não é variável de ambiente — é escolhido pela WebUI (`📁`) e persistido por projeto em `cache/webui_cwd.json`. Sem seleção, o padrão é a pasta base do usuário (`HOME`/`USERPROFILE`). O mcp_manager mantém uma sessão por par **(servidor, cwd)**: chats em pastas diferentes coexistem sem derrubar um ao outro, limitados por `AVA_MCP_MAX_SESSIONS` (LRU). Servidores `http` compartilham uma sessão por nome. O tray expõe `GET/PUT/DELETE /fs/cwd` (loopback-only) para ler/gravar esse estado.

### Triagem de memória — dataset para a rede local (Fase 1)

O extrator continua decidindo e gravando normalmente; agora ele **também**
registra cada decisão em `memory_triage_turn` / `memory_triage_sample`: o que
o LLM propôs (`proposed_label`), o que o pipeline de escrita aplicou
(`applied_label`) e os segmentos sobre os quais ele não se pronunciou
(`teacher_emitted=0`, `applied_label='unlabeled'`). Suprimido por orçamento
**não** é `DISCARD` — é desconhecido, e a exportação nunca o converte em
negativo (PU learning). A Fase 1 pertence ao planejamento em
`docs/MEMORY_TRIAGE_PHASE1.md`.

Rotas no memory API (proxeadas pela WebUI via `/memory/*` e pelo tray):
`POST /triage/log` (idempotente por turno+segmento+candidato+variant),
`GET /triage/labels?limit&since&scope&origin&variant&include_unlabeled`,
`GET /triage/stats` (contagens, `recall_gap`, `suppressed_by_reason`,
`emitted_rate_by_position`/`role`/`size`, `gate_label_counts`, `gold_total` e
`readiness`), `GET /triage/gold/candidates`, `POST /triage/gold` (imutável),
`GET /triage/gold` e `DELETE /triage/log?scope=` (privacidade).

`POST /memories/extract` aceita `turn_id`, `probe` e `propensity`. Variants:
`relaxed` (só orçamento), `segment`, `contrastive`, `synthetic`, `replay` e
`thinking`. **Todos os variants de instrumentação são somente leitura** —
nenhum grava memórias ou episódios. Cerca de 8% dos turnos recebem um probe
relaxado extra em background. O gate cross-encoder usa `POST /write` com
`dry_run: true`: mesma decisão de dedup/rerank, zero mutação.

Replay offline: `python scripts/triage_replay.py --limit 200` (use `--dry-run`
para inspecionar, `--mode thinking` para o recorte de CoT). As revisões do gold
set são feitas em `GET /triage/gold/candidates` → `POST /triage/gold`; a
calibração da Fase 2 deve usar **somente** esses rótulos.

A barra **Dados** na barra lateral da WebUI mostra o progresso até
`AVA_MEMORY_TRIAGE_TARGET_SAMPLES` **rótulos efetivos** e um aviso one-shot
aparece quando o dataset fica pronto. Replay conta no dataset, mas pelo peso
da fonte (§4): `live=1.0`, `probe=1.0`, `contrastivo=0.8`, `replay=0.4`,
`sintético/thinking=0.3`. `readiness.labelled` é a soma dos pesos arredondada;
`readiness.labelled_raw` é a contagem bruta de linhas, e
`labelled_by_origin` separa ao vivo de replay no tooltip. `unlabeled` nunca
entra como negativo. As métricas `ava_triage_*` e o bloco `memory_triage` de
`/status` expõem o resto.

### Providers de inferência (OpenAI-compatible)

Todos os endpoints que requisitam LLM diretamente (`/chat`, `/chat/stream`,
`/chat/tools`, `/memories/extract` + extrator pós-turno e classificador do
confidence-watch) falam com **um provider ativo por vez**, escolhido no
cadastro da WebUI e persistido em `Modules/.env` — trocar de provider =
Configurações → *Provedor de IA* (ou a tela de primeira execução), que salva a
env e reinicia o ambiente (sem rebuild, sem código). O catálogo
(`Manager/assets/providers.json`) é gerado do models.dev por
`scripts/build_provider_catalog.py` e só lista endpoints `chat-completions`.
O **modelo** é independente do provider: a WebUI mantém um seletor por chat
(barra sob a caixa de escrita, com favoritos e agrupamento por provedor) e
envia `model` em cada `/execute`; `Modules/LLM.py` aceita esse override sem
alterar `LLM_MODEL`. Qualquer API que implemente `POST /v1/chat/completions`
também serve via "Outro provedor OpenAI-compatible":

| Provider | `LLM_BASE_URL` | `LLM_API_KEY` | Exemplo de `LLM_MODEL` |
|---|---|---|---|
| OpenRouter | `https://openrouter.ai/api/v1` | obrigatória | `z-ai/glm-5.3-flash:floor` |
| Ollama | `http://localhost:11434/v1` | não precisa | `llama3.1:8b-instruct` |
| vLLM | `http://localhost:8000/v1` | opcional | nome do modelo servido |
| llama.cpp server | `http://localhost:8080/v1` | não precisa | nome do modelo servido |
| LM Studio | `http://localhost:1234/v1` | não precisa | nome do modelo servido |
| Groq | `https://api.groq.com/openai/v1` | obrigatória | `llama-3.3-70b-versatile` |
| DeepSeek | `https://api.deepseek.com/v1` | obrigatória | `deepseek-chat` |
| Together | `https://api.together.xyz/v1` | obrigatória | `meta-llama/Llama-3-70b-chat-hf` |

- **Capability gating**: campos que o backend não aceita (`logprobs`,
  `reasoning`, `response_format`, bloco `provider`) recebem 400 → strip
  automático + reenvio, e ficam omitidos nos requests seguintes (aviso único
  no log). O confidence-watch se desliga sozinho em backend sem logprobs.
- **Modelos extras**: `MEMORY_EXTRACTOR_MODEL`/`MEMORY_EXTRACTOR_FALLBACK_MODEL`
  já são env; em provider ≠ OpenRouter o default do extrator vira o próprio
  `LLM_MODEL`.
- **Key opcional**: provider local não exige key (o cadastro da WebUI pergunta
  a base URL primeiro e só cobra a key se o provider não for local; `/health`
  reporta `llm_key: "not-required"`). `LLM_PROVIDER_LOCAL=1` marca como local
  qualquer outra url (ex.: servidor de inferência na LAN — a WebUI oferece a
  opção quando a key é deixada em branco).

## Roadmap

- [ ] Empacotar o release automatizado (CI com build do `ava-tray` + `ava-web` + TUI opcional)
- [ ] Gravação de voz no Windows pela WebUI/TUI (dshow exige nome de device localizado)
- [ ] Modo "agente autônomo" com visão constante (reabilitar vision.py por padrão)
- [ ] Autoupdate do `install.sh` (diff entre releases)
- [ ] Suporte a GPU via ONNX Runtime (CUDA/ROCm) além de CPU

## Stack técnica

| Camada | Tecnologias |
|---|---|
| Orquestração | Python 3.11, FastAPI, httpx, SSE |
| IA/ML | OpenRouter (LLM), ONNX Runtime (E5, ms-marco, DINOv3, EdgeSAM, Depth Anything V2, YuNet, EdgeFace, nemotron-ASR, sherpa-onnx KWS, CAM++), Supertonic (TTS), LanceDB, SQLite |
| Voz | C++ (miniaudio, ten-vad, sherpa-onnx, GGUF streaming) — host |
| Tools | MCP (mcp_manager próprio em Rust, **embutido no tray**, com CWD por turno/projeto), Playwright MCP (extensão), SearXNG + Valkey + Crawl4AI + docker-socket-proxy |
| Manager / WebUI | Rust, axum, tokio, eframe/egui, tray-icon; `ava-web` com WebKitGTK/WebView2 |
| TUI (opcional) | TypeScript, Ink/React, Bun (single-binary cross-compile) |
| Infra | Docker Compose (base + overrides Linux/Windows) |
