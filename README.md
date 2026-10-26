# Alpha AI (AVA — Alpha Virtual Assistant)

> **Any model. Every tool. Zero limits.**

Assistente de IA pessoal: uma TUI reativa em Node/Bun, um Manager em Rust na bandeja do sistema (com chat widget ativado por wake word), um backend Python de microserviços orquestrado via Docker, um **mcp_manager standalone em Rust** e um **pipeline de voz em C++** — ambos rodando no host sob supervisão do tray. Inferência via **OpenRouter** (qualquer modelo), memória de longo prazo em **grafo hebbiano**, **voz** (wake word + ASR streaming + identificação de falante + TTS streaming), **busca web** como servidor MCP próprio e **tools MCP extensíveis** em runtime.

---

## Arquitetura

O sistema é dividido entre o **container Docker** (serviços Python sem acesso a microfone/Chrome/servidores MCP stdio) e o **host** (TUI, tray, mcp_manager Rust e pipeline de voz C++ — coisas que precisam de microfone, navegador e execução real na máquina do usuário).

```
┌──────────────────────────────── HOST ────────────────────────────────────────┐
│                                                                              │
│  TUI (TypeScript/Ink, binário standalone via Bun)   Manager Rust (ava-tray)  │
│  comando: alpha-ai                                  tray + UI egui + chat    │
│    │  POST :9001/docker/start (a cada pergunta)     widget; REST API :9001   │
│    │  POST :9000/execute (SSE, por turno)           /docker /browser /mcp    │
│    │                                                /status                  │
│    │  + Bearer AVA_INTERNAL_TOKEN em tudo                                    │
│    ▼                                                     │                   │
│  ┌────────────── Container "ava" (Python, FastAPI) ──────┼────────────────┐  │
│  │  orchestrator.py :9000   engine de tool-calling (SSE) │                │  │
│  │  LLM.py :4003            inferência (OpenRouter)      │                │  │
│  │  memory_api.py :3000     memória (SQLite+LanceDB+grafo)│               │  │
│  │  TTS.py :3004            fala (Supertonic, streaming) │                │  │
│  │  onnxManager.py :2002    embeddings E5 + reranker ONNX │                │  │
│  │  graph_activation.py :8777  visualização do grafo (SSE)│               │  │
│  └────────────────────────────────────────────────────────┘                │
│    │ localhost:4008 (Linux host network) / host.docker.internal (Windows)   │
│    ▼                                                                        │
│  mcp_manager (Rust standalone, :4008)  — ÚNICO executor de servidores MCP;  │
│    stdio (npx/uvx) + http, supervisionado pelo tray (restart-on-fail)      │
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

1. **Entrada**: a TUI ou o **chat widget do Manager** envia o turno ao orchestrator: `POST :9000/execute` (SSE), com `source` (`"tui"` / `"chat"`) e Bearer token interno.
2. **Pré-loop**: o orchestrator garante a sessão de memória, consome o cache de contexto de voz (se houver embedding de falante pré-ingestado) e faz o recall inicial de memórias.
3. **Inferência**: o histórico + tools vão ao **LLM.py** (:4003), que chama o OpenRouter com function-calling nativo.
4. **Tools**: o loop resolve as chamadas em duas frentes:
   - **Nativas** (`list_plans`, `run_plan`, `memory_search`, `vision_objects` quando habilitada) — registradas direto no orchestrator;
   - **MCP** — descoberta em duas etapas via mcp_manager :4008 (grafo de co-uso escolhe os servidores → carrega as tools reais → executa via REST). Antes de tools de browser, o orchestrator pede o Chrome ao tray (que o abre apenas se não estiver aberto).
5. **Memória**: leituras/escritas no **grafo hebbiano** (memory_api :3000) durante o loop — recall, correção e extração de novas memórias.
6. **Resposta final**: o texto encerra o turno (não existe tool "finish") e dispara **TTS automático** quando o cliente pediu áudio. Todos os eventos (delta, tool_call, done) são espelhados no **Conversation Hub** (`GET /conversation/stream`, SSE), que alimenta o chat widget do Manager em tempo real.

### Fluxo de voz (integrado ao pipeline principal)

1. O binário `transcript` (C++, **no host**, supervisionado pelo tray) abre o microfone via miniaudio e roda o **VAD** (ten-vad).
2. A **wake word** ("Alpha", via sherpa-onnx KeywordSpotter) é detectada; com `WAKE_GATE_ASR=true` o ASR só abre depois dela — o áudio acumulado da frase (incluindo a wake word) é reenviado ao ASR de uma vez. Frases sem wake word não gastam nenhuma inferência.
3. O ASR streaming (nemotron 0.6B GGUF, cache-aware com lookahead) transcreve o comando; a wake word **abre o chat widget do Manager**.
4. O **speaker embedding CAM++ (192 dims)** é enviado ao orchestrator via `POST /voice/ingest` — fora do caminho crítico do turno. O orchestrator pré-busca a memória de voz (kNN no `voice_dict`) e injeta no próximo `/execute` com custo 0 ms.
5. Cadastro de voz: pela TUI (Settings → Vozes), o microfone é gravado N segundos (`arecord`/`parec`/`ffmpeg`, Linux) e o embedding é gerado via `POST /voice/register` → `POST /speaker/embed` (:9100) → gravado no `voice_dict` (memory :3000).
6. A resposta final dispara **TTS automático** (Supertonic :3004) quando habilitado — o modelo nunca chama TTS.

O único jeito de encerrar o turno é **responder em texto puro** — não existe tool de "finish".

---

## Capacidades

### Núcleo (orchestrator + LLM)
- **Tool-calling nativo** via chat template (sem grammar manual): o loop recebe `tool_calls` estruturados, executa e realimenta o histórico como `role="tool"`, sem limite artificial de rodadas.
- **Qualquer modelo do OpenRouter** — o modelo é resolvido pelo backend; trocar modelo = trocar uma chave/config, sem rebuild.
- **Tools nativas** (`TOOLS` registry no orchestrator): `memory_search` (recall pessoal antes de inventar), `list_plans` + `run_plan` (planos híbridos de browser) e `vision_objects` (gate por `VISION_OBJECTS_ENABLED`, default 0 — veja Visão abaixo).
- **Imagens direto no modelo principal** (default): imagens anexadas pela TUI (`/image`) vão como blocos multi-parte `image_url` (MIME detectado por magic bytes) verbatim ao OpenRouter — sem tool, sem vision.py.
- **Planos híbridos (macro + modelo)** para navegação web: passos "macro" (seletores fixos de Playwright, com resultado esperado verificado) e passos "modelo" (o LLM julga a tela e extrai dados estruturados), com allowlist de domínios e teto de ações. Dados ausentes voltam como `not_found` explícito — nunca inventados.
- **Memory-watch**: monitor de correção de memória — se a nova resposta contradiz uma memória usada, classifica (confirmação/complementação/correção) e reescreve o prefill da retomada.
- **Extração pós-resposta** de memórias de longo prazo, forçada por JSON schema.
- **Conversation Hub**: SSE com todos os eventos dos turnos (`/conversation/stream`) + catch-up por id (`/conversation/recent`) — reconexão do chat widget do Manager sem perder eventos.
- **Persona editável em runtime**: `GET/PUT /persona` no orchestrator — a 1ª linha do system prompt vive em `resource/persona.json` e é trocada pela TUI (Settings → Persona) sem restart (a cláusula de tools é sempre anexada pelo orchestrator).

### Agent Skills (SKILL.md) — playbooks no padrão skills.sh
- Skills no padrão **Agent Skills** (o mesmo do CLI `skills` do npm: `npx skills add owner/repo`) são lidas **direto do disco** (mounts ro no container — `npx skills` roda no host, o container enxerga na hora). Os dois instaladores já fazem a instalação automática: etapa 3b instala o pacote hyperframes (`npx --yes skills add heygen-com/hyperframes -a universal -y` — `-a universal` grava só em `.agents/skills` (store canônico, sem linkar em `.claude`/`.cursor`/etc), `-y` pula os prompts, re-run é idempotente e falha (rede) é aviso, não morte) e a etapa 3c instala o **OfficeCLI** (`npm install -g @officecli/officecli`, prefix do usuário; no Linux com node de apt/pacman usa `as_root`) + a skill `officecli` baixada de `officecli.ai/SKILL.md` direto para o store (temp + `mv` — SKILL.md truncado não entra no índice). Duas origens, mesma mecânica:
  - **Projeto**: `.agents/skills/` + lock `skills-lock.json` na raiz do repo;
  - **Global**: `npx skills add -g` instala em `~/.agents/skills/` + lock `~/.agents/.skill-lock.json` (ou `$XDG_STATE_HOME/skills/` — variante do próprio CLI), montado em `/root/.agents/skills` no container. **Skill de projeto vence global em nome duplicado** (ordem = prioridade em `AVA_SKILLS_DIRS`); a proveniência segue o lock de cada diretório.
  - **Windows**: `npx skills add -g` grava em `%USERPROFILE%\.agents\skills`. Na rota A (Docker Desktop) o caminho literal entra direto na interpolação; na rota B (Docker Engine dentro do WSL) o `docker-start.bat`/`install.ps1` traduz com `wslpath` (mesmo esquema do diretório do projeto) e exporta a var via **`WSLENV`** (`AVA_WIN_GLOBAL_SKILLS/w` — sem isso o `wsl.exe` não leva env do Windows pro Linux). O override Windows interpola no mount `/root/.agents-win/skills` e o `AVA_SKILLS_DIRS` do perfil já inclui os três diretórios; fallback `/tmp` quando a var não vem (compose "na mão") — mount vazio, nunca derruba o stack.
- **Índice semântico no ToolsDB**: cada skill vira uma linha `agentskill__<nome>` no mesmo grafo hebbiano das MCP tools (kNN + PPR + queries aprendidas pós-turno). O seletor de **servidores** MCP (`/tools/select`) **exclui** o prefixo — skills não competem por slots de servidores; a seleção de skills usa o endpoint próprio `POST /agent-skills/select` (memory :3000). Co-uso numa mesma lista de `record-usage` cria **arestas skill↔tool** cruzando os dois mundos.
- **Por turno**: `/execute` seleciona as skills casadas com o pedido (top-K, env `AVA_SKILLS_TOP_K`) e injeta `nome + descrição` no system prompt (progressive disclosure — o playbook completo não vai por padrão). O modelo carrega o corpo com as tools nativas **`use_skill(name)`** e lê referências internas com **`skill_file(name, path)`** (leitura com guarda de path traversal via realpath + recusa de binário, em `Modules/agent_skills.py`).
- **Sincronização**: boot + assinatura por turno (nome/hash/always-on mudou → re-registra via `/tools/register`, idempotente por `doc_hash` com self-heal; skill removida do disco → prune). Env: `AVA_SKILLS_DIRS` (default: projeto + global, substitui a lista inteira), `AVA_ALWAYS_ON_AGENT_SKILLS` (core — entram em TODA seleção), `AVA_SKILLS_TOP_K`. Inspeção: `GET /skills` e `POST /skills/sync` no orchestrator. Nomenclatura: **Agent Skills** (SKILL.md) ≠ "skills" MCP (servidores do mcp_manager).

### Memória — grafo hebbiano (o "hipocampo" sintético)
- Curto prazo (SQLite) + longo prazo (LanceDB vetorial) + **grafo de memórias e tools** com arestas que se **reforçam por co-uso e co-ativação** (regra hebbiana), com auto-link na escrita e decay podando o que esquece.
- Busca semântica com **E5 multilingual** (ONNX, onnxManager :2002) + **rerank cross-encoder** (ms-marco MiniLM) + **Personalized PageRank** com spread de ativação + MMR (inibição lateral).
- **Grafo de tools (ToolsDB)**: cada tool MCP tem um snippet (o "skill.md" dela); o selector de servidores usa kNN + PPR no grafo de co-uso para escolher **quais** servidores carregar por rodada (economia de contexto com servidores de 20+ tools), com **cota de 4 tools por servidor** (um servidor grande não monopoliza o kNN), **exemplos de uso aprendidos** pós-turno (a query vira vetor "learned") e **self-heal** de vetores perdidos (re-embeda pelo doc_hash).
- **Dicionários sensoriais** integrados: `visual_dict` (significado de objetos), `face_dict` (rostos) e `voice_dict` (vozes cadastradas pela TUI) — embeddings consultados por kNN, texto pesquisável pela leitura normal de memória.
- **Cards de pessoa (`people-card`)**: unificam rostos + vozes em um card textual (descrição, likes, religião, infos extras) com **memória de identidade** no grafo (nó estável que ancora as arestas) — criados/editados pelo alpha-mcp (`memory_people_card_*`) e lidos como memória normal.
- **Card de identidade da Alpha (`GET/PATCH /identity`)**: card singleton com quem a própria assistente é (nome, descrição, papéis/valores, infos extras), **semeado no boot** a partir de `resource/persona.json` e ligado à sua **memória de identidade LT** — no viewer vira o nó ★ dourado `a{id}` com aresta `identity_memory`; editado pelo alpha-mcp (`memory_identity_*`).
- **RAG de arquivos locais (`indexed-file`)**: cadastro com hash SHA-256 (reindexa só o que mudou), chunking 500/100 com posição (`char_start/end`) e leitura em 3 modos — arquivo inteiro por path, chunks daquele arquivo ranqueados pela query, ou busca semântica global.
- **Planos persistidos** (`/plans/*`) com CRUD pela TUI e pelo `plan_recorder`.
- Visualização **em tempo real** do grafo em `graph_activation.py` (:8777): polling de diff do SQLite (~1 s) com nós de memória, **tools como hexágonos** (`t{id}`) e **planos como diamantes** (`p{id}`); arestas nascendo, sendo reforçadas e podadas por decay ao vivo.

### Visão (vision.py) — desligada por padrão
O pipeline completo (**Depth Anything V2** → clusterização por camadas → **EdgeSAM** → crops → **DINOv3** → consulta ao dicionário visual, mais **YuNet** + **EdgeFace** para rostos, tudo ONNX/CPU) existe e funciona, mas **não sobe por default**: a forma como foi projetado para ser usado (agente totalmente autônomo com visão constante) ainda não é suportada. Com `VISION_OBJECTS_ENABLED=0` (default), `start.sh` não inicia vision.py, a tool `vision_objects` não é oferecida e as imagens vão embutidas direto no contexto do modelo principal. Com a flag em `1`, o fluxo antigo volta (face recognition + segmentação → crops → resposta do modelo multimodal).

### Voz
- **Opt-in**: `AVA_VOICE=1` em `Modules/.env` (o instalador pergunta o perfil — econômico deixa desligado). O ASR nemotron GGUF fica residente no host (~600 MB); desligado, o tray pula o transcript no auto-init e `POST /transcript/start` sobe manualmente.
- **Pipeline C++ no host** (transcribe): wake word + ASR streaming + speaker verification, com gate de ASR por wake word (nenhuma inferência gasta sem comando). Supervisionado pelo tray como um processo do host (liveness por pid, restart-on-fail).
- **Identificação de falante**: embeddings CAM++ pré-ingestados injetam "quem está falando" no turno — o assistente sabe diferenciar usuários cadastrados no `voice_dict`.
- **TTS** (Supertonic, :3004): pipeline estilo Piper (worker pool, fila de texto, heap de áudio ordenado, miniaudio 40 ms): `/speak`, `/stream` (síntese paralela por sentença em ordem), `/cancel`. Disparado automaticamente sobre a resposta final; o modelo nunca decide chamar TTS.
- **Cadastro de vozes pela TUI** (Settings → Vozes): gravação local de microfone, edição de descrição, listagem — tudo via `voice_dict` na memory API.

### Busca web (alpha-mcp)
O Deep Search standalone foi **removido** — a busca web agora é um **servidor MCP próprio** (`mcp_servers/alpha-mcp`, :4010). A tool `web_search` esconde o pipeline: **cache Valkey (TTL 6h) → SearXNG → dedup/blocklist SEO → BM25 → fetch em camadas (trafilatura estático, Crawl4AI com navegador para os difíceis) → chunking por headers markdown → BM25 final** — o modelo recebe só trechos com fontes `[n]`. O crawl4ai sobe **on-demand** via **docker-socket-proxy** com allowlist mínima (start/stop de container apenas) — sem socket do Docker exposto. Tools de memória completam o servidor: `memory_write_batch` (grava fatos permanentes com cadeia `temporal_precedence` p/ o grafo) e `memory_people_card_create/edit/list` (cards de pessoa) — todos falando direto com a memory API.

### Tools MCP (extensibilidade)
- **mcp_manager reescrito em Rust** (`Modules/mcp_manager`, :4008) e rodando **NO HOST** — fora do container — para que os servidores MCP tenham acesso real à máquina do usuário (npx/node/uvx do host, filesystem, Chrome do usuário). O tray o supervisiona: descobre o binário no boot, faz spawn, espera o `/health`, e re-espawna em caso de crash (restart-on-fail).
- O orchestrator é **cliente REST** dele: registro, sync, listagem de tools e **execução** (`/servers/{name}/call`) ficam todos lá; este processo nunca spawna servidores MCP nem segura sessões.
- Servidores são registrados/sincronizados/removidos **em runtime** — inclusive pela **TUI** (instalação assistida de servidores remotos, comandos locais pip/uv e endpoints localhost), sem reiniciar nada.
- **Playwright MCP** pré-configurado com `--extension`: conecta ao **Chrome real do usuário** via extensão "Playwright MCP Bridge" (instalada no perfil padrão). O orchestrator pede o Chrome ao tray, que o abre **apenas se ele não estiver aberto** — em modo normal, sem CDP nem cópia de perfil — e **o Chrome fica aberto** após as ações (processo detached do host, sobrevive ao docker). Suas sessões/logins do Chrome valendo. A autenticação da extensão é automática: o token `PLAYWRIGHT_MCP_EXTENSION_TOKEN` (`Modules/.env`) é injetado pelo tray no mcp_manager e herdado pelo servidor MCP — sem diálogo de aprovação por conexão.
- **Windows**: URLs de servidores http cadastradas como `localhost:PORTA` são reescritas para `host.docker.internal` no momento da conexão (porta e path preservados).

### Manager Rust (ava-tray)
- Ícone na **bandeja do sistema** (GTK/Linux, tray nativo no Windows) + janela de status (egui) com start/stop do docker, status do mcp_manager e do transcript.
- **Chat widget**: abre com a wake word "Alpha" (detectada pelo pipeline de voz) ou manualmente; envia turnos escritos ao orchestrator (`source="chat"`, sem TTS — a fala é papel da TUI/voz) e consome o Conversation Hub (SSE) para renderizar a resposta em tempo real.
- API REST (:9001): `/status`, `/docker/start`, `/docker/stop`, `/browser/start`, `/browser/stop`, `/browser/status`, `/mcp/*` (start/stop/status do mcp_manager), `/transcript/*` (start/stop/status do pipeline de voz) e `POST /ui/chat-open` (wake word do transcribe → abre o chat widget).
- **Supervisão de processos do host**: mcp_manager (:4008) e transcript (:9100) — discover → spawn → health check periódico → restart-on-fail. Variáveis de auth/config lidas do `Modules/.env` e injetadas nos filhos.
- **Timer de inatividade**: `docker down` após 45 min sem perguntas (configurável via `AVA_DOCKER_IDLE_MINUTES`).
- **Chrome do usuário** sob demanda: aberto em **modo normal** (perfil padrão, sem `--remote-debugging-port`) apenas **se o Chrome não estiver aberto** — spawn detached, sem cópia de perfil e sem cache `chrome-profile-cdp`.

### TUI (terminal, Ink/React)
- Chat com streaming SSE, planos de execução (CoT) renderizados, timeline de tool calls e linha de status dos processos.
- Painel de **status** (CPU/RAM/GPU via systeminformation), painel de **logs do Docker** ao vivo (`Ctrl+D`), painel de **settings** (`Ctrl+S` ou `/settings`) com gestão de **servidores MCP** (instalação assistida npm/http/stdio e remoção), **vozes** (cadastro por microfone, edição de descrição e remoção), **planos** (lista/edição versionada) e **persona** editável, links para o grafo.
- Comando `/image <caminho>` para anexar imagens à conversa (vão direto ao modelo multimodal).
- **Autenticação interna**: toda requisição leva `Authorization: Bearer <AVA_INTERNAL_TOKEN>` (token do `Modules/.env` gerado pelo install.sh; nunca logado).
- **Binário standalone** (~80 MB, cross-compile Bun) — não exige Node no usuário.

---

## Segurança

- **Token interno compartilhado** (`AVA_INTERNAL_TOKEN`, gerado pelo install.sh em `Modules/.env` com chmod 600): memory (:3000), orchestrator (:9000) e mcp_manager (:4008) exigem `Bearer` — middleware dedicado em todos os serviços internos.
- **Sem portas expostas à LAN**: no Windows, todas as portas do container são publicadas apenas em `127.9.12.22`; no Linux, `network_mode: host` mantém os serviços em localhost. O tray (:9001) nunca é publicado.
- **docker-socket-proxy**: o alpha-mcp não tem acesso direto ao socket do Docker — apenas allowlist mínima (listar/inspecionar containers e start/stop do crawl4ai). Sem exec, images, volumes, networks ou build.
- **crawl4ai com token** (`CRAWL4AI_API_TOKEN`): sem ele o servidor só escuta no loopback do container.
- **SSRF guard**: `POST /search` e `POST /read-url` (orchestrator) e o fetch do alpha-mcp resolvem o DNS e bloqueiam IP privado/loopback/link-local/metadata — a validação acompanha redirects.

---

## Estrutura do projeto

```
├── orchestrator.py          # Engine de tool-calling (FastAPI/SSE, :9000) + Conversation Hub
├── graph_activation.py      # Visualizador do grafo hebbiano em tempo real (SSE, :8777)
├── Modules/
│   ├── LLM.py               # Cliente OpenRouter + prompts + memory-watch (:4003)
│   ├── TTS.py               # Supertonic streaming (:3004)
│   ├── vision.py            # Depth/SAM/DINOv3/YuNet/EdgeFace (:4002) — desligado por padrão
│   ├── onnxManager.py       # E5 embed + ms-marco rerank, API única (:2002)
│   ├── onnx_client.py       # Cliente httpx do onnxManager (embed/rerank/score)
│   ├── internal_auth.py     # Bearer token interno compartilhado (middleware)
│   ├── mcp_manager/         # MCP manager reescrito em RUST, standalone no host (:4008)
│   ├── memory/              # Memória: grafo, vetores, dicionários, planos
│   │   ├── memory.py        #   fachada que reexporta os submódulos (API plana)
│   │   ├── memory_api.py    #   FastAPI (:3000)
│   │   └── memory_*.py      #   config/db/graph/vector/search/ops/status…
│   ├── transcribe/          # Pipeline de voz em C++ (integrado!) — NO HOST
│   │   ├── main.cpp         #   VAD + wake word + ASR streaming + CAM++ (:9100)
│   │   └── linux/ windows/  #   builds por plataforma
│   └── Models/              # Pesos ONNX/GGUF (gitignored; via Models.zip)
├── mcp_servers/alpha-mcp/   # alpha-mcp (SearXNG→BM25→Crawl4AI, :4010)
├── Manager/                 # ava-tray (Rust: axum + eframe + tray-icon, :9001)
│   └── src/                 #   api, chat (widget), transcript, mcp_manager, idle_watcher…
├── src/                     # TUI (TypeScript/Ink; build Bun)
│   └── lib/                 #   auth (Bearer), voiceRecorder, voices, mcpServers, plans…
├── resource/                # ícones, settings do SearXNG, config alpha-mcp
├── docker-compose.yml       # stack base (+ overrides .linux/.windows)
├── start.sh                 # supervisor do container (restart-on-fail por serviço)
├── install.sh / install.bat # instaladores (baixam o release, geram tokens, models)
└── scripts/                 # patch-cfonts (build TUI), plan_recorder (gravação de planos)
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
| graph_activation.py | container | 8777 | Grafo em tempo real (SSE, diff polling) — **opt-in** (`AVA_GRAPH_UI=1`) |
| **mcp_manager (Rust)** | **host** | 4008 | Registro + execução de servidores MCP (supervisionado pelo tray) |
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
- Para voz: microfone + `arecord`/`parec`/`ffmpeg` no Linux (gravação de voz na TUI ainda não suportada no Windows — o pipeline C++ de voz contínua tem build para ambas as plataformas). A voz é **opt-in**: `AVA_VOICE=1` em `Modules/.env` (o tray pulando o transcript economiza ~600 MB de RAM no host; `POST /transcript/start` sobe manualmente sem reiniciar nada).

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

O instalador pede o provider de inferência (`LLM_BASE_URL` — Enter = OpenRouter; `LLM_API_KEY` só se o provider não for local) e gera os tokens internos (`AVA_INTERNAL_TOKEN`, `CRAWL4AI_API_TOKEN`) em `Modules/.env` — o stack não sobe sem eles.

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
alpha-ai          # abre a TUI
```
Digite a pergunta no terminal (o ambiente sobe sozinho sob demanda). Ou simplesmente **fale "Alpha"** — o pipeline de voz detecta a wake word, transcreve o comando e abre o chat widget do Manager. `/image <caminho>` anexa uma imagem; `?`/painéis para status, logs do Docker, settings (servidores MCP, vozes) e o grafo.

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
| `OPENROUTER_API_KEY` | `Modules/.env` | — | Chave do OpenRouter (gerada pelo install.sh; fallback de `LLM_API_KEY`) |
| `LLM_BASE_URL` | `Modules/.env` | OpenRouter | Provider de inferência — qualquer API OpenAI-compatible (tabela abaixo) |
| `LLM_API_KEY` | `Modules/.env` | `OPENROUTER_API_KEY` | API key do provider (opcional em provider local: localhost/127.0.0.1) |
| `LLM_MODEL` | `Modules/.env` | `z-ai/glm-5.3-flash:floor` | Modelo principal do provider ativo |
| `AVA_INTERNAL_TOKEN` | `Modules/.env` | — | Bearer token interno (memory/orchestrator/mcp_manager/TUI) |
| `CRAWL4AI_API_TOKEN` | `Modules/.env` | — | Token do crawl4ai (falha de boot se ausente) |
| `AVA_DOCKER_IDLE_MINUTES` | tray | 15 | Minutos de inatividade antes do `docker down` (a pergunta seguinte reergue o stack sozinho) |
| `AVA_TTS_ENABLED` | `start.sh` | `0` | Liga o serviço de TTS (Supertonic, ~470 MB/worker na RAM). Desligado = fala não sintetiza |
| `AVA_GRAPH_UI` | orchestrator/start.sh | `0` | Liga o visualizador do grafo (:8777, ~80 MB) — dentro do processo do orchestrator no modo mesclado |
| `AVA_VOICE` | tray | `0` | Liga o pipeline de voz no host (wake word + ASR, ~600 MB). Desligado = transcript não sobe no auto-init; `POST /transcript/start` sobe à mão |
| `AVA_MERGED_SERVICES` | orchestrator/start.sh | `1` | LLM (:4003) e grafo (:8777) mesclados no processo do orchestrator (um interpreter Python). `0` volta ao modo standalone |
| `WEB_STACK_ON_DEMAND` | alpha-mcp | `1` | searxng + valkey sobem na 1ª `web_search` e dormem após 5 min sem busca (`WEB_STACK_IDLE_STOP_S`) |
| `AVA_CHROME_PATH` | tray | autodetect | Caminho do Chrome (aberto pelo tray em modo normal) |
| `PLAYWRIGHT_MCP_EXTENSION_TOKEN` | `Modules/.env` | — | Token da extensão "Playwright MCP Bridge" (injetado no @playwright/mcp; conexão sem aprovação manual) |
| `AVA_CDP_SERVERS` | orchestrator | `playwright` | Servidores MCP que usam o Chrome do host |
| `MCP_MANAGER_URL` | orchestrator | `localhost:4008` | Onde o mcp_manager está |
| `TRANSCRIBE_HTTP_URL` | orchestrator | `localhost:9100` | Mini HTTP do transcript (host) p/ `/speaker/embed` |
| `VISION_OBJECTS_ENABLED` | orchestrator | `0` | Liga o fluxo vision_objects (sobe vision.py à parte) |
| `MCP_CALL_TIMEOUT` | orchestrator | 180 s | Teto de execução de tools MCP sem override |
| `AVA_LOCALHOST_REWRITE` | compose (Windows) | `host.docker.internal` | Reescrita de URLs localhost de servidores MCP http |
| `AVA_API_BASE` / `AVA_ORCHESTRATOR_URL` | TUI / Manager | :9001 / :9000 | Endpoints dos componentes |
| `AVA_TTS_OUTPUT` | TUI | `client` | Onde a fala toca: `client` (a TUI toca no host via pacat/aplay/ffplay) ou `device` (container, via PulseAudio) |
| `ONNX_PROVIDERS` | compose | `CPUExecutionProvider` | Provedores do ONNX Runtime |

### Providers de inferência (OpenAI-compatible)

Todos os endpoints que requisitam LLM diretamente (`/chat`, `/chat/stream`,
`/chat/tools`, `/memories/extract` + extrator pós-turno e classificador do
confidence-watch) falam com **um provider ativo por vez**, escolhido por env
em `Modules/.env` — trocar de provider = mudar a env e reiniciar (sem rebuild,
sem código). Qualquer API que implemente `POST /v1/chat/completions` serve:

| Provider | `LLM_BASE_URL` | `LLM_API_KEY` | Exemplo de `LLM_MODEL` |
|---|---|---|---|
| OpenRouter (default) | `https://openrouter.ai/api/v1` | obrigatória (`OPENROUTER_API_KEY` funciona como fallback) | `z-ai/glm-5.3-flash:floor` |
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
- **Key opcional**: provider local não exige key (o `install.sh`/`install.ps1`
  pedem o `LLM_BASE_URL` primeiro e só cobram a key se o provider não for
  local; `/health` reporta `llm_key: "not-required"`). `LLM_PROVIDER_LOCAL=1`
  marca como local qualquer outra url (ex.: servidor de inferência na LAN —
  o próprio instalador oferece a opção quando a key é deixada em branco).

## Roadmap

- [ ] Empacotar o release automatizado (CI com build do `ava-tray` + `mcp_manager` + binários da TUI)
- [ ] Gravação de voz no Windows pela TUI (dshow exige nome de device localizado)
- [ ] Modo "agente autônomo" com visão constante (reabilitar vision.py por padrão)
- [ ] Autoupdate do `install.sh` (diff entre releases)
- [ ] Suporte a GPU via ONNX Runtime (CUDA/ROCm) além de CPU

## Stack técnica

| Camada | Tecnologias |
|---|---|
| Orquestração | Python 3.11, FastAPI, httpx, SSE |
| IA/ML | OpenRouter (LLM), ONNX Runtime (E5, ms-marco, DINOv3, EdgeSAM, Depth Anything V2, YuNet, EdgeFace, nemotron-ASR, sherpa-onnx KWS, CAM++), Supertonic (TTS), LanceDB, SQLite |
| Voz | C++ (miniaudio, ten-vad, sherpa-onnx, GGUF streaming) — host |
| Tools | MCP (mcp_manager próprio em Rust, standalone no host), Playwright MCP (extensão), SearXNG + Valkey + Crawl4AI + docker-socket-proxy |
| Manager | Rust, axum, tokio, eframe/egui, tray-icon |
| TUI | TypeScript, Ink/React, Bun (single-binary cross-compile) |
| Infra | Docker Compose (base + overrides Linux/Windows) |
