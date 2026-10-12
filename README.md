# Alpha AI (AVA — Alpha Virtual Assistant)

> **Any model. Every tool. Zero limits.**

Assistente de IA pessoal de **terminal**: uma TUI reativa em Node/Bun, um backend Python de microserviços orquestrado via Docker, e um gerenciador de processos em Rust na bandeja do sistema. Inferência via **OpenRouter** (qualquer modelo), memória de longo prazo em **grafo hebbiano**, **visão computacional** (segmentação + reconhecimento de objetos/rostos), **voz** (TTS streaming), **busca web** com pipeline próprio e **tools MCP extensíveis** em runtime.

---

## Arquitetura

```
┌──────────────────────────────── HOST ─────────────────────────────────┐
│                                                                       │
│  TUI  (TypeScript/Ink, binário standalone via Bun)                    │
│  comando: alpha-ai                                                    │
│    │  POST :9001/docker/start   POST :9001/browser/start              │
│    │  (a cada pergunta)         (antes de tools de browser)           │
│    ▼                                                                  │
│  Manager Rust (ava-tray) — bandeja do sistema + API REST :9001        │
│    • sobe/derruba o stack Docker (docker-start.sh/.bat)               │
│    • timer de inatividade: docker down após 45 min sem perguntas      │
│    • abre Chrome do host em modo depuração (CDP :9222)                │
│    │                                                                  │
│    ▼ docker compose                                                   │
│  ┌──────────────────── Container "ava" (Python, network host) ──────┐ │
│  │  orchestrator.py :9000   engine de tool-calling (SSE)            │ │
│  │  LLM.py :4003            inferência conversacional (OpenRouter)  │ │
│  │  memory_api.py :3000     memória (SQLite + LanceDB + grafo)      │ │
│  │  TTS.py :3004            fala (Supertonic, streaming)            │ │
│  │  vision.py :4002         visão (depth, SAM, DINOv3, rostos)      │ │
│  │  onnxManager / LLM.py    embeddings E5 + reranker (ONNX local)   │ │
│  │  mcp_manager.py :4008    registro + execução de servidores MCP   │ │
│  │  graph_activation.py :8777  visualização do grafo em tempo real  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  Stack de busca web:  searxng :8081 · valkey :6379 · crawl4ai :11235  │
│                       websearch-mcp :4010                             │
│  Chrome de depuração (perfil copiado, CDP :9222) p/ Playwright MCP    │
└───────────────────────────────────────────────────────────────────────┘
```

### Fluxo de uma pergunta

1. A TUI garante o ambiente no ar: `POST :9001/docker/start` no Manager (idempotente — se já está rodando, só reseta o timer de inatividade).
2. A TUI abre o stream da resposta: `POST :9000` (SSE) no orchestrator.
3. O orchestrator envia o histórico + tools ao **LLM.py** (:4003), que chama o OpenRouter com function-calling nativo.
4. Tool calls são executadas em loop: **nativas** (ex.: `vision_objects`) ou **MCP** (proxy REST via mcp_manager :4008 — antes de tools de browser, o orchestrator pede o Chrome de depuração ao tray).
5. Memórias são lidas/escritas na **memória de grafo** (memory_api :3000) durante o loop — recall, correção e extração de novas memórias.
6. O texto final da resposta dispara **TTS automático** (o modelo nunca chama TTS) e volta por SSE para a TUI, que renderiza streaming, planos (CoT), tool calls e status.

O único jeito de encerrar o turno é **responder em texto puro** — não existe tool de "finish".

---

## Capacidades

### Núcleo (orchestrator + LLM)
- **Tool-calling nativo** via chat template (sem grammar manual): o loop recebe `tool_calls` estruturados, executa e realimenta o histórico como `role="tool"`, sem limite artificial de rodadas.
- **Qualquer modelo do OpenRouter** — o modelo é resolvido pelo backend; trocar modelo = trocar uma chave/config, sem rebuild.
- **Planos híbridos (macro + modelo)** para navegação web: passos "macro" (seletores fixos de Playwright) e passos "modelo" (o LLM julga a tela), com allowlist de domínios e teto de ações.
- **Memory-watch**: monitor de correção de memória — se a nova resposta contradiz uma memória usada, classifica (confirmação/complementação/correção) e reescreve o prefill da retomada.
- **Extração pós-resposta** de memórias de longo prazo, forçada por JSON schema.

### Memória — grafo hebbiano (o "hipocampo" sintético)
- Curto prazo (SQLite) + longo prazo (LanceDB vetorial) + **grafo de memórias e tools** com arestas que se **reforçam por co-uso e co-ativação** (regra hebbiana).
- Busca semântica com **E5 multilingual** (ONNX) + **rerank cross-encoder** (ms-marco MiniLM) + **Personalized PageRank** com spread de ativação + MMR (inibição lateral).
- **Grafo de tools (ToolsDB)**: cada tool MCP tem um snippet (o "skill.md" dela); o selector de servidores usa kNN + PPR no grafo de co-uso para escolher **quais** servidores carregar por rodada (economia de contexto com servidores de 20+ tools).
- **Dicionários sensoriais** integrados: `visual_dict` (significado de objetos), `face_dict` (rostos) e `voice_dict` (vozes) — embeddings consultados por kNN, texto pesquisável pela leitura normal de memória.
- **Planos persistidos** (`/plans/*`) com CRUD pela TUI.
- Visualização **em tempo real** do grafo (nós/arestas nascendo, sendo reforçados e podados por decay) em `graph_activation.py` (:8777).

### Visão (vision.py — "tradução de objetos")
Pipeline sem LLM no meio: **Depth Anything V2** (profundidade) → clusterização por camadas → amostragem de pontos → **EdgeSAM** (segmentação class-agnostic) → crops → **DINOv3** (embeddings) → consulta ao dicionário visual (kNN). Mais: **YuNet** (detecção de rostos) + **EdgeFace** (embeddings faciais) para reconhecimento, tudo ONNX/CPU. Nenhuma decisão é tomada aqui — crops + candidatos voltam ao orchestrator.

### Voz (TTS)
- **Supertonic** com pipeline estilo Piper (worker pool, fila de texto, heap de áudio ordenado, miniaudio 40 ms): `/speak`, `/stream` (síntese paralela por sentença em ordem), `/cancel`. Disparado automaticamente sobre a resposta final.

### Busca web (websearch-mcp)
Uma única tool exposta ao modelo (`web_search`) esconde o pipeline: **cache Valkey → SearXNG → dedup/blocklist SEO → BM25 → fetch em camadas (trafilatura estático, Crawl4AI com navegador para os difíceis) → chunking por headers markdown → BM25 final** — o modelo recebe só trechos com fontes `[n]`.

### Tools MCP (extensibilidade)
- **mcp_manager.py** (:4008) é o único processo que spawna servidores MCP e mantém sessões — o orchestrator é cliente REST dele. Servidores são registrados/sincronizados/removidos **em runtime** (inclusive pela TUI, sem reiniciar nada).
- **Playwright MCP** pré-configurado com `--cdp-endpoint http://localhost:9222`: o orchestrator pede o Chrome ao tray, que **copia o perfil padrão do usuário** (sem caches) e abre o Chrome do host em modo depuração — **o Chrome fica aberto** após as ações (processo detached do host, sobrevive ao docker). Suas sessões/logins do Chrome valendo.
- O próprio mcp_manager resolve dependências de servidores (gateway de registry, instalação/remoção, sync).

### Manager Rust (ava-tray)
- Ícone na **bandeja do sistema** (GTK/Linux, tray nativo no Windows) + janela de status (egui) com start/stop do docker.
- API REST (:9001): `/status`, `/docker/start`, `/docker/stop`, `/browser/start`, `/browser/stop`, `/browser/status`.
- **Timer de inatividade**: `docker down` após 45 min sem perguntas (configurável via `AVA_DOCKER_IDLE_MINUTES`).
- **Chrome de depuração** sob demanda: cópia do perfil (cookies/sessões preservados, caches excluídos — 3,9 GB viram centenas de MB) em `<install>/cache/chrome-profile-cdp`, spawn detached com `--remote-debugging-port=9222`.

### TUI (terminal, Ink/React)
- Chat com streaming SSE, planos de execução (CoT) renderizados, timeline de tool calls.
- Painel de **status** (CPU/RAM/GPU via systeminformation), painel de **logs do Docker** ao vivo, painel de **settings**, gestão de **servidores MCP**, links para o grafo.
- Comando `/image <caminho>` para anexar imagens à conversa (pipeline de visão).
- **Binário standalone** (~80 MB, cross-compile Bun) — não exige Node no usuário.

---

## Estrutura do projeto

```
├── orchestrator.py          # Engine de tool-calling (FastAPI/SSE, :9000)
├── graph_activation.py      # Visualizador do grafo hebbiano (SSE, :8777)
├── Modules/
│   ├── LLM.py               # Cliente OpenRouter + prompts + memory-watch
│   ├── TTS.py               # Supertonic streaming (:3004)
│   ├── vision.py            # Depth/SAM/DINOv3/YuNet/EdgeFace (:4002)
│   ├── onnxManager.py       # Embeddings E5 + reranker (ONNX)
│   ├── mcp_manager.py       # Registro/execução de servidores MCP (:4008)
│   ├── memory/              # Memória: grafo, vetores, dicionários, planos
│   │   ├── memory.py        #   fachada de compatibilidade
│   │   ├── memory_api.py    #   FastAPI (:3000)
│   │   └── memory_*.py      #   config/db/graph/vector/search/ops…
│   ├── transcribe/          # Voz em C++ (ASR + wake word + VAD) ⚠ see below
│   └── Models/              # Pesos ONNX/GGUF (gitignored; via Models.zip)
├── mcp_servers/websearch/   # websearch-mcp (SearXNG→BM25→Crawl4AI)
├── Manager/                 # ava-tray (Rust: axum + eframe + tray-icon)
├── src/                     # TUI (TypeScript/Ink; build Bun)
├── resource/                # ícones, settings do SearXNG, config websearch
├── docker-compose.yml       # stack completa (+ overrides .linux/.windows)
├── install.sh / install.bat # instaladores (baixam o release, models, etc.)
└── scripts/                 # patch-cfonts (build TUI), plan_recorder
```

## Serviços e portas

| Serviço | Porta | Papel |
|---|---|---|
| orchestrator.py | 9000 | API de chat (SSE) + tool loop |
| ava-tray (Manager) | 9001 | Controle do docker + browser CDP |
| memory_api.py | 3000 | Memória (grafo/vetores/dicionários/planos) |
| TTS.py | 3004 | Síntese e playback de fala |
| LLM.py | 4003 | Inferência (OpenRouter) |
| vision.py | 4002 | Visão computacional |
| mcp_manager.py | 4008 | Registro + execução de tools MCP |
| graph_activation.py | 8777 | Grafo em tempo real (SSE) |
| searxng | 8081 | Meta-busca (loopback) |
| valkey | 6379 | Cache da busca web (loopback) |
| crawl4ai | 11235 | Crawl com navegador (loopback) |
| websearch-mcp | 4010 | Tool `web_search` (loopback) |
| Chrome CDP | 9222 | Browser de depuração (host) |

---

## Requisitos

- **Linux** (recomendado; `network_mode: host`, áudio via PulseAudio, GPU opcional via Vulkan `/dev/dri`) ou **Windows** (Docker Desktop + WSL2; portas publicadas no override).
- Docker Engine + plugin compose v2; arquitetura **x86_64**.
- ~2,5 GB livres além do Docker (pesos extraídos) e internet no primeiro boot (build + download do Supertonic no HF).
- Áudio e GPU são **opcionais** — tudo roda em CPU (ONNX Runtime).

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
O instalador: valida prereqs → baixa o último release do GitHub → extrai em `~/.local/opt/alpha-ai` (Linux) / `%LOCALAPPDATA%\Programs\AlphaAI` (Windows) → extrai `Modules/Models.zip` (pesos ONNX) → pede e grava a `OPENROUTER_API_KEY` em `Modules/.env` (chmod 600) → build + `up` do stack e espera o healthcheck → cria o comando `alpha-ai` → registra o auto-start do tray → abre a TUI.

## Uso

```bash
alpha-ai          # abre a TUI
```
Digite a pergunta no terminal (o ambiente sobe sozinho sob demanda). `/image <caminho>` anexa uma imagem; `?`/painéis para status, logs do Docker, settings e servidores MCP. `alpha-ai` é um symlink/binário que fala com o tray (:9001) e com o orchestrator (:9000).

Logs e manutenção:
```bash
cd ~/.local/opt/alpha-ai
./docker-start.sh logs -f ava    # logs do backend
./docker-start.sh down           # derruba o stack manualmente
curl -s localhost:9001/browser/status   # Chrome de depuração
```

## Configuração (principais)

| Variável | Onde | Default | O que faz |
|---|---|---|---|
| `OPENROUTER_API_KEY` | `Modules/.env` | — | Chave do OpenRouter (lida no import — configurar antes do primeiro start) |
| `AVA_DOCKER_IDLE_MINUTES` | tray | 45 | Minutos de inatividade antes do `docker down` |
| `AVA_CHROME_PATH` | tray | autodetect | Caminho do Chrome p/ modo depuração |
| `AVA_CDP_SERVERS` | orchestrator | `playwright` | Servidores MCP que usam o Chrome CDP |
| `AVA_API_BASE` / `AVA_ORCHESTRATOR_URL` | TUI | :9001 / :9000 | Endpoints dos componentes |
| `ONNX_PROVIDERS` | compose | `CPUExecutionProvider` | Provedores do ONNX Runtime |

---

## ⚠ Sobre o módulo de transcrição

`Modules/transcribe/` contém um pipeline de **voz em C++** **funcional e testado** de forma isolada: ASR streaming (**nemotron-3.5-asr** via GGUF, cache-aware com lookahead configurável), **wake word** (sherpa-onnx KeywordSpotter zipformer), **speaker verification** (3D Speaker CAM++) e **VAD** (ten-vad), com backends Vulkan/CPU. **Porém ele ainda NÃO está integrado ao projeto principal** — roda como binário de teste independente (`transcript`), fora do container e fora da TUI. A integração (voz contínua + wake word + identificação de falante no assistente) está no roadmap.

## Roadmap

- [ ] Integrar `Modules/transcribe` (voz contínua, wake word, diarização por falante)
- [ ] Empacotar o release automatizado (CI com build do `ava-tray.exe` + binários da TUI)
- [ ] Autoupdate do `install.sh` (diff entre releases)
- [ ] Suporte a GPU via ONNX Runtime (CUDA/ROCm) além de CPU

## Stack técnica

| Camada | Tecnologias |
|---|---|
| Orquestração | Python 3.11, FastAPI, httpx, SSE |
| IA/ML | OpenRouter (LLM), ONNX Runtime (E5, ms-marco, DINOv3, EdgeSAM, Depth Anything V2, YuNet, EdgeFace), Supertonic (TTS), LanceDB, SQLite |
| Tools | MCP (mcp_manager próprio), Playwright MCP (CDP), SearXNG + Valkey + Crawl4AI |
| Manager | Rust, axum, tokio, eframe/egui, tray-icon |
| TUI | TypeScript, Ink/React, Bun (single-binary cross-compile) |
| Infra | Docker Compose (base + overrides Linux/Windows) |
