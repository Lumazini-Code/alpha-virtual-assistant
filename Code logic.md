


# AVA — Padrão de Uso das Portas de API REST (0.0.0.0)

## Mapeamento de Portas

### 0–1024: Não permitido (Linux bloqueia)

---

### 2000–2999: Conexões Llama Server (backends de inferência)

| Porta | Serviço | Modelo | Descrição |
|-------|---------|--------|-----------|
| 2001  | LLM llama-server | Text-only (configurável via `Aimodel.dll`) | Inferência conversacional principal |
| 2004  | VQA llama-server | Qwen3VL-2B-Instruct-Q4_K_M | Inferência multimodal (visão) |

> **Nota:** O `llamaManager.py` gerencia o ciclo de vida desses processos.  
> Modelos textuais usam o preset `TEXT_PARAMS` (porta 8080 por default, sobrescrita pela config).  
> Modelos multimodais usam o preset `VISION_PARAMS` (porta 8081 por default) e anexam `--mmproj` automaticamente.

---

### 3000–3999: Módulos Auxiliares

| Porta | Serviço | Descrição |
|-------|---------|-----------|
| 3000  | CoT Generator | Geração de planos de execução com cache semântico |
| 3001  | Memory API | Memória de longo prazo (LT), curto prazo (ST), cache de planos, base de conhecimento |
| 3002  | Search API | Busca web (DuckDuckGo) + PDF + reranking cross-encoder |
| 3004  | TTS API | Text-to-Speech (Supertonic) com playback via miniaudio |

---

### 4000–4999: Módulos Principais

| Porta | Serviço | Descrição |
|-------|---------|-----------|
| 4002  | VQA (Vision) | Análise de imagens via Qwen3VL |
| 4003  | LLM Chat | API conversacional com memória + TTS + detecção de idioma |
| 4005  | Deep Search | KG-RAG com pesquisa web automática |

---

### 9000–9999: Conexão Backend–Frontend

| Porta | Serviço | Descrição |
|-------|---------|-----------|
| 9000  | Orchestrator | Roteamento inteligente (ONNX + heurísticas) + execução de planos |

---

## Módulos Auxiliares — Referência de API

---

### API de Memória (porta 3001)

#### Busca na memória

```python
import httpx

# Busca simples (sem contexto de sessão)
result = httpx.post("http://localhost:3001/read", json={
    "query": "preferências de comunicação do usuário",
    "top_k": 5,
    "min_score": 0.3,
    "strategy": "none"        # "auto" | "expanded" | "dual" | "none"
})

# Busca contextual (com histórico da sessão)
result = httpx.post("http://localhost:3001/read", json={
    "query": "sobre o que estávamos conversando?",
    "top_k": 5,
    "min_score": 0.3,
    "session_id": "abc-123",
    "strategy": "auto"        # Classifica automaticamente: expanded ou dual
})

for r in result.json()["results"]:
    print(f"[{r['memory_type']}] {r['score']:.4f} — {r['text']}")
```

**Campos de resposta por tipo:**
- `long_term`: id, text, score, confidence, created_at, access_count, source
- `short_term`: id, text, score, session_id, turns (lista de Turn: role + content)
- `knowledge`: id, text, score, source (chunks do VectorStore/KG-RAG)

**Estratégias de busca contextual:**
| Estratégia | Descrição |
|-----------|-----------|
| `auto` | Classifica por comprimento/stop-words → escolhe `expanded` ou `dual` |
| `expanded` | Um embedding (contexto + query concatenados) |
| `dual` | Dois embeddings separados (query e contexto) com fusão ponderada |
| `none` | Ignora session_id, busca simples |

---

#### Escrita na memória de longo prazo

```python
result = httpx.post("http://localhost:3001/write", json={
    "text": "eu gosto de maracujá",
    "source": "user_explicit",   # "chat" | "user_explicit" | "system" | "orchestrator"
    "confidence": 1.0
})
for r in result.json()["results"]:
    print(f"{r['stored']} — {r['reason']}")
# Retorno: {"stored": true/false, "reason": "ok"|"duplicate_exact"|"duplicate_semantic:0.95"|"too_short", "memory_id": 42}
```

---

#### Escrita na memória de curto prazo (turnos de conversa)

```python
result = httpx.post("http://localhost:3001/write_st", json={
    "session_id": "abc-123",
    "turns": [
        {"role": "user",      "content": "Qual a capital da França?"},
        {"role": "assistant", "content": "A capital da França é Paris."}
    ]
})
print(result.json())
# {"stored": true, "reason": "ok", "turn_ids": [15]}
```

---

#### Cache de planos (CoT)

```python
# Verificar cache
result = httpx.post("http://localhost:3001/cache/get", json={
    "query": "qual a melhor GPU pra rodar modelos de 256B?",
    "threshold": 0.92         # Similaridade mínima para hit (default: 0.92)
})
print(result.json())
# Hit:  {"hit": true, "plan": {"steps": [...]}, "score": 0.95, "cache_id": 7, "hit_count": 3}
# Miss: {"hit": false, "plan": null, "score": null, "cache_id": null}

# Gravar plano no cache
result = httpx.post("http://localhost:3001/cache/put", json={
    "query": "qual a melhor GPU pra rodar modelos de 256B?",
    "plan": {"steps": [{"step":1, "action":"...", "executor":"search", "depends_on":null}]}
})

# Invalidar todo o cache
result = httpx.delete("http://localhost:3001/cache")

# Remover entrada específica
result = httpx.delete("http://localhost:3001/cache/7")

# Status
result = httpx.get("http://localhost:3001/status")
```

---

#### Limpar sessão

```python
result = httpx.delete("http://localhost:3001/session/abc-123")
# {"cleared": 3, "session_id": "abc-123"}
```

---

### Gerador de CoT (porta 3000)

```python
import httpx

result = httpx.post("http://localhost:3000/plan", json={
    "input": "qual a melhor GPU pra rodar modelos de 256B?",
    "context": "Felipe tem orçamento de R$2000 e usa Ubuntu",
    "use_cache": True,
    "cache_threshold": 0.92,   # Sobrescreve o threshold padrão
    "max_steps": 5             # Limita número de passos (2–7)
}, timeout=45)

plan = result.json()
for step in plan["steps"]:
    deps = step.get("depends_on") or []
    print(f"[{step['executor']}] step {step['step']}. {step['action']} (deps: {deps})")

# Resposta completa:
# {
#   "steps": [
#     {"step":1, "action":"search RTX 4060 benchmark price", "executor":"search", "depends_on": null},
#     {"step":2, "action":"retrieve GPU preferences from memory", "executor":"memory", "depends_on": null},
#     {"step":3, "action":"recommend GPU based on result of step 1 and result of step 2", "executor":"llm", "depends_on": [1,2]}
#   ],
#   "input": "qual a melhor GPU...",
#   "from_cache": false,
#   "cache_score": null,
#   "tokens_used": 87,
#   "latency_ms": 823.45
# }
```

**Executores disponíveis no CoT:**
`llm`, `memory`, `search`, `vision`, `tts`, `stt`, `commander`, `translator`, `calculator`

**Cache de planos:** O CoT consulta `/cache/get` na Memory API antes de inferir. Hit → retorna em ~5ms. Miss → infere e grava via `/cache/put`.

**Invalidação de cache:**
```python
httpx.delete("http://localhost:3000/cache")   # Proxy para DELETE /cache da Memory API
```

---

### API de Busca na Internet (porta 3002)

```python
import httpx

BASE_URL = "http://localhost:3002"

# Busca simples (apenas snippets)
r = httpx.post(f"{BASE_URL}/search", json={
    "query":       "como instalar o llama.cpp no Ubuntu com suporte CUDA?",
    "max_results": 5,
    "use_cache":   True,
    "search_pdfs": False
}, timeout=30.0)

# Busca com PDFs (download + extração + reranking)
r = httpx.post(f"{BASE_URL}/search", json={
    "query":       "transformer architecture paper",
    "max_results": 5,
    "use_cache":   True,
    "search_pdfs": True
}, timeout=30.0)

for result in r.json()["results"]:
    print(f"  {'[PDF]' if result['from_pdf'] else '[WEB]'} "
          f"Score: {result['score']:.4f} | "
          f"Fonte: {result['source'][:80]}")
    print(f"  Trecho: {result['text'][:200]}...")
    if result['from_pdf']:
        print(f"  Chunk: {result['pdf_chunk']}")

# Limpar cache
httpx.delete(f"{BASE_URL}/cache")
```

**Pipeline de busca:**
1. Extração de keywords (YAKE)
2. Busca DuckDuckGo (texto + PDF se ativado)
3. Download de PDFs em paralelo (até 20MB cada)
4. Extração de chunks com `pdfplumber` (~400 palavras, 80 de sobreposição)
5. Reranking com cross-encoder ONNX (ms-marco-MiniLM-L-6-v2)
6. Normalização e merge final dos scores

---

### API de Text-to-Speech (porta 3004)

```python
import httpx

API_URL = "http://127.0.0.1:3004"

# Sintetizar texto completo (divide em chunks automaticamente)
r = httpx.post(f"{API_URL}/speak", json={
    "text":  "Olá, este é um teste da API de Text-to-Speech.",
    "voice": "M1",           # M1–M5, F1–F5
    "lang":  "pt",
    "speed": 1.0             # 0.5–2.0
}, timeout=30.0)
# {"duration_s": 3.2, "latency_ms": 450.0, "text": "...", "chunks": 2}

# Streaming — para textos longos, sintetiza em paralelo por sentença
r = httpx.post(f"{API_URL}/stream", json={
    "text":  "Texto longo que será dividido em sentenças e sintetizado em paralelo...",
    "voice": "F1",
    "lang":  "pt"
}, timeout=30.0)
# {"sentences": 5, "total_duration_s": 12.4, "latency_ms": 1800.0}

# Cancelar fala em andamento
r = httpx.post(f"{API_URL}/cancel")
# {"cancelled": true}

# Listar vozes disponíveis
r = httpx.get(f"{API_URL}/voices")
# {"voices": ["M1","M2","M3","M4","M5","F1","F2","F3","F4","F5"], "default": "M1"}

# Status
r = httpx.get(f"{API_URL}/status")
```

**Normalização automática de texto:**
- Remove Markdown, LaTeX, HTML, emojis
- Converte símbolos (R$ → "reais", % → "por cento", × → "por")
- Interpreta LaTeX (`\frac{a}{b}` → "a sobre b", `\sqrt{x}` → "raiz de x")
- Respeita abreviações e reticências no split

---

## Módulos Principais — Referência de API

---

### API LLM Principal (porta 4003)

```python
import httpx

# Chat síncrono
r = httpx.post("http://0.0.0.0:4003/chat", json={
    "message":    "Qual é meu nome?",
    "session_id": "sessao-42",   # Para memória de curto prazo
    "voice":      "F1",          # Voz TTS (null = padrão do config)
    "lang":       null,          # null = detectado automaticamente via langdetect
    "max_turns":  10,            # Limite de contexto recuperado da memória
    "tts":        True           # Dispara TTS após gerar
}, timeout=120.0)
# {"response": "Seu nome é Felipe.", "session_id": "sessao-42", "lang": "pt", "elapsed": 2.34}

# Chat com streaming (SSE)
r = httpx.post("http://0.0.0.0:4003/chat/stream", json={
    "message":    "Explique redes neurais",
    "session_id": "sessao-42",
    "voice":      "F1",
    "lang":       null,
    "max_turns":  10,
    "tts":        True
}, timeout=120.0)
# Retorna Server-Sent Events: data: {"delta": "Redes"} data: {"delta": " neurais"} ...
# Ao final: data: {"done": true, "elapsed": 5.12}

# Histórico
r = httpx.get("http://0.0.0.0:4003/history", params={"session_id": "sessao-42", "last_n": 20})

# Limpar histórico
r = httpx.delete("http://0.0.0.0:4003/history", json={"confirm": True, "session_id": "sessao-42"})

# Health check
r = httpx.get("http://0.0.0.0:4003/health")
# {"api": "ok", "llama_server": "ok"|"down"}
```

**Integrações automáticas:**
- Memória LT: grava fatos relevantes via `POST /write` (fire-and-forget)
- Memória ST: grava turnos via `POST /write_st` (fire-and-forget)
- TTS: dispara `POST /speak` ou streaming por chunk (fire-and-forget)
- Detecção de idioma: `langdetect` no input do usuário

---

### API VQA / Vision (porta 4002)

```python
import httpx

# Descrição de imagem (síncrono)
r = httpx.post("http://0.0.0.0:4002/describe", json={
    "img_path": "/home/user/foto.png",
    "prompt":   "Descreva a imagem detalhadamente."
}, timeout=60.0)
# {"result": "A imagem mostra um gato laranja sentado em..."}

# Descrição com streaming (SSE)
r = httpx.post("http://0.0.0.0:4002/describe/stream", json={
    "img_path": "/home/user/foto.png",
    "prompt":   "O que está escrito nesta imagem?"
}, timeout=60.0)
# data: {"delta": "Na imagem"} data: {"delta": " está escrito..."} ...

# Health check
r = httpx.get("http://0.0.0.0:4002/health")
# {"status": "ok"}
```

**Modelo:** Qwen3VL-2B-Instruct via llama-server na porta 2004 com `--mmproj`.

---

### API Deep Search / KG-RAG (porta 4005)

```python
import httpx

# Pesquisa profunda (pipeline completo)
r = httpx.post("http://0.0.0.0:4005/query", json={
    "text": "Quais são as causas da Revolução Industrial e seus impactos sociais?"
}, timeout=90.0)
# {"answer": "A Revolução Industrial teve como causas...", "stats": {"vectors": 42, "kg_nodes": 87, "kg_edges": 156}}

# Estatísticas do índice
r = httpx.get("http://0.0.0.0:4005/stats")
# {"vectors": 42, "kg_nodes": 87, "kg_edges": 156}

# Health check
r = httpx.get("http://0.0.0.0:4005/health")
# {"status": "ok", "service": "ava-kg-rag", "version": "2.0.0"}

# Reset completo (IRREVERSÍVEL)
r = httpx.delete("http://0.0.0.0:4005/reset")
# {"status": "reset_ok"}
```

**Pipeline KG-RAG:**
1. Decompõe o objetivo em sub-tópicos (LLM)
2. Pesquisa cada sub-tópico no DuckDuckGo
3. Extrai texto das páginas encontradas
4. Distila triplas semânticas (LLM)
5. Indexa chunks com embeddings ONNX
6. Recupera contexto via FAISS + Knowledge Graph
7. Rerank cross-encoder + síntese final (LLM)

---

## Orchestrator (porta 9000) — Roteamento + Execução Unificada

### POST /execute — Endpoint principal

```python
import httpx

r = httpx.post("http://0.0.0.0:9000/execute", json={
    "input":       "Pesquise preços de RTX 4060 e me recomende uma",
    "session_id":  null,           # Gerado automaticamente se null
    "voice":       "M1",
    "lang":        "pt",
    "tts":         True,
    "use_cache":   True,
    "image_path":  null,           # Para queries de visão
    "search_pdfs": False,
    "strategy":    "parallel"      # "parallel" | "sequential" | "fail_fast"
}, timeout=120.0)
```

**Resposta:**
```json
{
    "execution_id": "a1b2c3d4",
    "input": "Pesquise preços de RTX 4060 e me recomende uma",
    "session_id": "auto-generated-uuid",
    "final_response": "Com base na pesquisa, a RTX 4060...",
    "steps": [
        {"step":1, "executor":"search", "action":"search RTX 4060 price", "success":true, "result":[...], "retries":0, "latency_ms":2340.5},
        {"step":2, "executor":"llm", "action":"recommend GPU based on result of step 1", "success":true, "result":"Com base...", "retries":0, "latency_ms":3200.1}
    ],
    "plan_from_cache": false,
    "total_latency_ms": 5840.6,
    "errors": [],
    "route": "cot",
    "route_confidence": 0.72,
    "route_method": "heuristic",
    "routed_directly": false
}
```

### POST /classify — Apenas classificação de rota

```python
r = httpx.post("http://0.0.0.0:9000/classify", json={
    "text":       "traduza isso para inglês",
    "image_path": null,
    "lang":       "pt"
})
# {"route": "translator", "confidence": 0.85, "method": "onnx_deberta+minilm+heuristic",
#  "all_scores": {"llm": 0.05, "search": 0.02, "translator": 0.85, ...}, "needs_cot": false, "latency_ms": 12.3}
```

### Rotas diretas (proxy para módulos)

```python
# Chat direto (bypassa orquestração)
r = httpx.post("http://0.0.0.0:9000/chat", params={"message": "Olá", "voice": "M1", "lang": "pt", "tts": True})

# Busca direta
r = httpx.post("http://0.0.0.0:9000/search", params={"query": "RTX 4060", "max_results": 5, "search_pdfs": False})

# Visão direta
r = httpx.post("http://0.0.0.0:9000/vision", json={"img_path": "/path/to/img.png", "prompt": "Descreva"})

# Deep search direto
r = httpx.post("http://0.0.0.0:9000/deep-search", json={"text": "pesquisa profunda sobre IA"})

# Memória direta
r = httpx.post("http://0.0.0.0:9000/memory/read", json={"query": "minhas preferências", "top_k": 5, "min_score": 0.3})
r = httpx.post("http://0.0.0.0:9000/memory/write", json={"text": "gosto de rock", "source": "chat", "confidence": 1.0})
```

### Rotas auxiliares do Orchestrator

```python
# Status geral de todos os serviços
r = httpx.get("http://0.0.0.0:9000/status")
# {"orchestrator": "ok", "internal_router": {"deberta_loaded": true, "minilm_loaded": true},
#  "services": {"cot": {"healthy": true}, "memory": {"healthy": true}, ...},
#  "executors": ["llm","memory","search","deep_search","vision","tts","stt","commander","translator","calculator"]}

# Invalidar cache de planos
r = httpx.delete("http://0.0.0.0:9000/cache")

# Limpar sessão
r = httpx.delete("http://0.0.0.0:9000/session/abc-123")

# Cancelar TTS
r = httpx.post("http://0.0.0.0:9000/tts/cancel")

# Listar vozes TTS
r = httpx.get("http://0.0.0.0:9000/tts/voices")
```

---

## Roteador Interno (ONNX + Heurísticas)

O Orchestrator possui um roteador de intent interno que combina 3 métodos:

1. **DeBERTa (ONNX)** — Modelo fine-tuned para classificação de 11 rotas (se disponível)
2. **MiniLM (ONNX)** — Similaridade por protótipos semânticos (fallback semântico)
3. **Heurísticas** — Regex + palavras-chave (sempre disponível)

**Rotas disponíveis e seus executores:**

| Rota | Executor | Gatilho típico |
|------|----------|---------------|
| `llm` | LLM Chat | Conversação, perguntas, explicações |
| `search` | Search API | "procure", "busque", "pesquise" |
| `memory_read` | Memory | "o que você sabe sobre mim", "lembra" |
| `memory_write` | Memory | "lembrar", "salvar", "gravar" |
| `vision` | VQA | "imagem", "foto", "descreva a" |
| `deep_search` | Deep Search | "pesquisa profunda", "investigue" |
| `calculator` | eval/LLM | "quanto é", "calcule" |
| `commander` | subprocess | "abrir", "open", "launch" |
| `translator` | LLM (prompt) | "traduz", "translate" |
| `tts` | TTS | "fale", "speak" |
| `cot` | CoT + multi-step | Queries complexas/multi-etapas |

**Lógica de decisão:**
- Se confiança ≥ threshold → roteamento direto (1 step, sem CoT)
- Se confiança < threshold ou rota = `cot` → pipeline CoT (multi-step com DAG de dependências)
- Steps com `depends_on: null` são executados em paralelo

**Fusão de scores:** `deberta×0.5 + minilm×0.2 + heuristic×0.3` (ajusta conforme modelos disponíveis)

---

## Llama Manager (CLI)

Gerencia instâncias do llama-server com detecção automática de modelos multimodais.

```bash
# Iniciar modelo de texto
python llama_manager.py start ./Models/Phi-3.5-mini-instruct-Q4_K_M.gguf

# Iniciar modelo de visão (auto-detecta mmproj no mesmo diretório)
python llama_manager.py start ./Models/mmpro-Qwen3V-2B-Instruct-Q8_0.gguf

# Parar modelo
python llama_manager.py stop ./Models/Phi-3.5-mini-instruct-Q4_K_M.gguf

# Verificar status
python llama_manager.py status ./Models/Phi-3.5-mini-instruct-Q4_K_M.gguf
```

**Detecção automática:**
- Lê metadata GGUF via biblioteca `gguf` (fallback: heurística por nome de arquivo)
- Se arquitetura for `llava`, `qwen2vl`, `minicpmv`, etc. → modo multimodal
- Busca arquivo `*mmproj*.gguf` no mesmo diretório automaticamente
- PIDs e configs salvos em `/tmp/ava_llama_pids/`



### Arquitetura

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (porta 9000)                 │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│              ORCHESTRATOR (porta 9000)                   │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Internal Router (ONNX DeBERTa + MiniLM + Heur) │    │
│  └────────┬──────────────────────────┬─────────────┘    │
│           │ Direct Route             │ CoT Pipeline     │
└─────┬─────┴──────────┬───────────────┴──────┬───────────┘
      │                │                      │
┌─────▼─────┐  ┌──────▼──────┐  ┌────────────▼────────────┐
│ AUX MODS  │  │ MAIN MODS   │  │    LLAMA SERVERS        │
│           │  │             │  │                         │
│ CoT :3000 │  │ LLM :4003   │  │ Text   :2001            │
│ Mem :3001 │  │ VQA :4002   │  │ Vision :2004            │
│ Srch:3002 │  │ DpS :4005   │  │ CoT    :8081            │
│ TTS :3004 │  │             │  │                         │
└───────────┘  └─────────────┘  └─────────────────────────┘
```

---

## Módulos

| Módulo | Porta | Arquivo | Descrição |
|--------|-------|---------|-----------|
| **Orchestrator** | 9000 | `orchestrator.py` | Roteamento inteligente + execução de planos |
| **CoT Generator** | 3000 | `CoT generator.py` | Geração de planos de execução (Phi-3.5-mini) |
| **Memory API** | 3001 | `memory.py` | Memória LT + ST + cache de planos + conhecimento |
| **Search API** | 3002 | `Search_api.py` | Busca web + PDF + reranking cross-encoder |
| **TTS API** | 3004 | `TTS.py` | Text-to-Speech com Supertonic + miniaudio |
| **LLM Chat** | 4003 | `LLM.py` | Chat conversacional com memória + TTS |
| **VQA / Vision** | 4002 | `VQA.py` | Análise de imagens (Qwen3VL) |
| **Deep Search** | 4005 | `deep_search.py` | KG-RAG com pesquisa web automática |
| **Llama Manager** | — | `llamaManager.py` | CLI para gerenciar instâncias llama-server |

---

## Funcionalidades

### 🧠 Roteamento Inteligente
- **3 camadas de classificação**: DeBERTa fine-tuned (ONNX) → MiniLM protótipos (ONNX) → Heurísticas regex
- Fusão ponderada de scores: `deberta×0.5 + minilm×0.2 + heuristic×0.3`
- 11 rotas: `llm`, `search`, `memory_read`, `memory_write`, `vision`, `deep_search`, `calculator`, `commander`, `translator`, `tts`, `cot`
- Roteamento direto (1 step) ou pipeline CoT (multi-step com DAG)

### 🔗 Planejamento Multi-Etapa (CoT)
- Geração de planos com 2–7 steps via Phi-3.5-mini
- Grammar GBNF para output JSON estruturado
- Dependências explícitas (`depends_on`) para paralelismo
- Cache semântico via Memory API (~5ms em hit vs ~800ms em miss)
- Recuperação de steps parciais em caso de truncamento

### 💾 Memória Persistente
- **Longo prazo**: SQLite + FAISS (embeddings e5-small) com decay de confiança
- **Curto prazo**: Turnos de conversa por sessão com TTL de 24h
- **Conhecimento**: VectorStore/KG-RAG integrado
- **Cache de planos**: Cache semântico para o CoT (threshold 0.92)
- Busca contextual com estratégias: `auto`, `expanded`, `dual`, `none`
- Deduplicação exata e semântica

### 🔍 Busca na Internet
- DuckDuckGo para snippets e PDFs
- Download e extração de PDFs em paralelo (até 20MB)
- Chunking com sobreposição (~400 palavras, 80 overlap)
- Reranking com cross-encoder ONNX (ms-marco-MiniLM-L-6-v2)
- Extração de keywords via YAKE
- Cache TTL (3600s, 128 entradas)

### 🗣️ Text-to-Speech
- Engine Supertonic com pool de 3 workers
- 10 vozes: M1–M5 (masculinas), F1–F5 (femininas)
- Playback via miniaudio com buffer de 40ms
- Heap ordenado por sequência para paralelismo
- Normalização automática: Markdown, LaTeX, HTML, emojis, símbolos PT-BR
- Cancelamento imediato de fala em andamento

### 💬 Chat Conversacional
- Inferência via llama-server (OpenAI-compatible API)
- Detecção automática de idioma (langdetect)
- Histórico persistido via memória de curto prazo
- Streaming SSE com TTS em tempo real (chunk a chunk)
- Gravação automática de fatos na memória de longo prazo

### 👁️ Visão / VQA
- Qwen3VL-2B-Instruct com projeção multimodal
- Descrição de imagens (síncrono e streaming)
- Suporte a qualquer formato de imagem via base64

### 📚 Deep Search / KG-RAG
- Pipeline completo: decomposição → busca → extração → triplas → indexação → RAG
- Knowledge Graph + FAISS para recuperação
- Reranking com cross-encoder
- Síntese final via LLM

### 🖥️ Gerenciamento de Modelos
- Detecção automática de modelos multimodais via metadata GGUF
- Anexação automática de `--mmproj` para modelos de visão
- Parâmetros otimizados para texto vs visão
- Gerenciamento de PIDs e logs

---

## Instalação

### Pré-requisitos

- Python 3.10+
- Vulkan SDK ou CUDA (para utilizar a GPU na inferência)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) compilado com suporte CUDA ou vulkan

### Dependências Python

```bash
pip install fastapi uvicorn httpx pydantic numpy faiss-cpu onnxruntime tokenizers gguf langdetect yake pdfplumber ddgs miniaudio supertonic

```

### Modelos Necessários

Coloque os modelos na pasta `./Models/`:

| Modelo | Uso | Download |
|--------|-----|----------|
| `Qwen3VL-2B-Instruct-Q4_K_M.gguf` | VQA/Vision | HuggingFace |
| `mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf` | Projeção visual | HuggingFace |
| Modelo de texto principal (configurável) | LLM Chat | Configurável via `Aimodel.dll` |
| `DebertaV2ForSequenceClassification/` | Router (ONNX) | Fine-tuned |
| `ms-marco-MiniLM-L-6-v2/` | Cross-encoder + Router | HuggingFace |
| `multilingual-e5-small/` | Embeddings de memória | HuggingFace |

---

## Uso

### 1. Iniciar os servidores Llama

```bash
# Modelo de texto (LLM Chat)
python llama_manager.py start ./Models/seu-modelo-texto.gguf

# Modelo de visão (VQA)
python llama_manager.py start ./Models/Qwen3VL-2B-Instruct-Q4_K_M.gguf

# Modelo CoT (Phi-3.5-mini)
python llama_manager.py start ./Models/Phi-3.5-mini-instruct-Q4_K_M.gguf
```

### 2. Iniciar os microserviços

```bash
# Em terminais separados (ou via systemd/supervisor):
uvicorn memory:app       --host 0.0.0.0 --port 3001   # Memória
uvicorn "Search_api:app" --host 0.0.0.0 --port 3002   # Busca
uvicorn "CoT generator:app" --host 0.0.0.0 --port 3000 # CoT
uvicorn TTS:app          --host 0.0.0.0 --port 3004   # TTS
uvicorn LLM:app          --host 0.0.0.0 --port 4003   # LLM Chat
uvicorn VQA:app          --host 0.0.0.0 --port 4002   # Visão
uvicorn deep_search:app  --host 0.0.0.0 --port 4005   # Deep Search
uvicorn orchestrator:app --host 0.0.0.0 --port 9000   # Orchestrator
```

### 3. Fazer requisições

```python
import httpx

# Requisição unificada via Orchestrator
r = httpx.post("http://0.0.0.0:9000/execute", json={
    "input": "Pesquise preços de RTX 4060 e recomende uma para jogos",
    "voice": "F1",
    "lang":  "pt",
    "tts":   True
}, timeout=120)

print(r.json()["final_response"])
```

### 4. Classificar intent sem executar

```python
r = httpx.post("http://0.0.0.0:9000/classify", json={"text": "traduza para inglês"})
print(r.json()["route"])  # "translator"
```

---

## Configuração do LLM Chat

O módulo `LLM.py` lê configurações de arquivos na pasta `resource/`:

| Arquivo | Conteúdo |
|---------|----------|
| `resource/username.dll` | Nome do usuário |
| `resource/VoiceModel.dll` | Voz TTS padrão (ex: "F1") |
| `resource/ctxConfig.dll` | Nome do contexto a usar |
| `resource/Aimodel.dll` | Caminho do modelo GGUF |
| `resource/SearchCfg.dll` | Configuração de busca |
| `Ctxbin/{nome}.bin` | System prompt do assistente |
| `CfgModels/{modelo}.json` | Configurações específicas do modelo |

---

## Estrutura de Diretórios

```
AVA/
├── orchestrator.py          # Orchestrator principal (porta 9000)
├── CoT generator.py         # CoT Generator (porta 3000)
├── memory.py                # Memory API (porta 3001)
├── Search_api.py            # Search API (porta 3002)
├── TTS.py                   # TTS API (porta 3004)
├── LLM.py                   # LLM Chat (porta 4003)
├── VQA.py                   # Vision/QA (porta 4002)
├── deep_search.py           # KG-RAG (porta 4005)
├── llamaManager.py          # CLI para gerenciar llama-servers
├── engine.py                # Engine do KG-RAG (usado por deep_search.py)
├── config.py                # Configurações do KG-RAG
├── modules/
│   └── vector_store.py      # VectorStore para conhecimento
├── Models/
│   ├── DebertaV2ForSequenceClassification/
│   ├── ms-marco-MiniLM-L-6-v2/
│   ├── multilingual-e5-small/
│   ├── Phi-3.5-mini-instruct-Q4_K_M.gguf
│   ├── Qwen3VL-2B-Instruct-Q4_K_M.gguf
│   └── mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf
├── memory/                  # Dados persistidos (SQLite + FAISS)
├── logs/                    # Logs dos módulos
├── resource/                # Configs do LLM Chat
└── CfgModels/               # Configs por modelo
```

---

## Variáveis de Ambiente

| Variável | Default | Descrição |
|----------|---------|-----------|
| `AVA_MODELS_DIR` | `./Models` | Diretório dos modelos ONNX do Router |
| `ROUTER_CONFIDENCE` | `0.65` | Threshold de confiança para roteamento direto |
| `ONNX_PROVIDERS` | `CUDAExecutionProvider,CPUExecutionProvider` | Providers ONNX Runtime |
| `LLAMA_SERVER_PATH` | `llama-server` | Caminho do binário llama-server |
| `LLAMA_PID_DIR` | `/tmp/ava_llama_pids` | Diretório dos arquivos PID |

---

## Health Checks

Todos os módulos expõem endpoints de saúde:

```bash
curl http://0.0.0.0:9000/status     # Orchestrator (inclui status de todos os serviços)
curl http://0.0.0.0:3000/status     # CoT Generator
curl http://0.0.0.0:3001/status     # Memory API
curl http://0.0.0.0:3002/status     # Search API
curl http://0.0.0.0:3004/status     # TTS API
curl http://0.0.0.0:4003/health     # LLM Chat
curl http://0.0.0.0:4002/health     # VQA
curl http://0.0.0.0:4005/health     # Deep Search
```

---

## Fluxo de Execução Típico

```
Usuário: "Pesquise preços de RTX 4060 e recomende uma"
    │
    ▼
Orchestrator (POST /execute)
    │
    ├── Router classifica: "cot" (conf: 0.72, método: heuristic)
    │
    ▼
CoT Generator (POST /plan)
    │
    ├── Cache miss → Inferência Phi-3.5-mini
    │
    ├── Plano gerado:
    │   Step 1: [search] "search RTX 4060 price Brazil 2024" (depends: null)
    │   Step 2: [memory] "retrieve GPU preferences" (depends: null)
    │   Step 3: [llm] "recommend GPU based on result of step 1 and 2" (depends: [1,2])
    │
    ├── Plano cacheado na Memory API
    │
    ▼
Execução do Plano
    │
    ├── Steps 1 e 2 em PARALELO
    │   ├── Search API → snippets reranked
    │   └── Memory API → preferências do usuário
    │
    ├── Step 3 (após 1 e 2 completarem)
    │   └── LLM Chat → resposta final com contexto
    │
    ├── TTS disparado em background
    ├── Turno salvo na memória ST
    ├── Fato salvo na memória LT
    │
    ▼
Resposta final ao usuário
```

---

## Licença

Projeto licenciado em Apache 2.0