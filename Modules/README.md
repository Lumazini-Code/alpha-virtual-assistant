# AVA KG-RAG — Knowledge Graph RAG

Implementação local do pipeline **KG-RAG (Knowledge-Graph-Driven RAG)**, adaptada ao stack do AVA.

---

## Stack

| Componente | Implementação | Porta/Arquivo |
|---|---|---|
| LLM | llama-server (REST) | `localhost:4003` |
| Embeddings | ONNX Runtime + multilingual-e5-small | `models/` |
| Reranker | ONNX Runtime + ms-marco-MiniLM-L-6-v2 | `models/` |
| Knowledge Graph | NetworkX + SQLite | `storage/knowledge_graph.db` |
| Vector Store | FAISS IndexFlatIP | `storage/faiss.index` |
| Ingestão | PyMuPDF + python-docx | local |
| API | FastAPI + Uvicorn | `localhost:4005` |

---

## Arquitetura do Pipeline

```
[Objetivo do Usuário]
         │
         ▼
  1. PLANNER (LLM via llama-server)
     Decompõe o objetivo em sub-tópicos usando GBNF grammar
     → Retorna JSON: {"sub_topics": [...]}
         │
         ▼
  2. INGESTOR (PDF / DOCX / TXT local)
     PyMuPDF para PDFs, python-docx para DOCX
     → Lista de (texto_da_seção, metadados)
         │
         ▼
  3. DISTILLER (LLM via llama-server)
     Extrai triplas semânticas: (Sujeito, relação, Objeto)
     → [["Chlorella", "produz", "O₂"], ...]
         │
         ▼
  4. SEMANTIC CHUNKER + KNOWLEDGE GRAPH
     ├─ Chunker: distância cosseno entre embeddings de sentenças
     │   Quebra no percentil 95° de distância (fronteiras reais)
     └─ KG: insere triplas no NetworkX + SQLite (persistente)
         │
         ▼
  5. EMBEDDINGS (ONNX Runtime)
     multilingual-e5-small com:
     - Prefixo "passage:" para chunks
     - Mean pooling com attention mask (correto)
     - L2 normalização → compatível com FAISS IndexFlatIP
         │
         ▼
  6. MEMÓRIA DE LONGO PRAZO
     ├─ FAISS: IndexFlatIP com vetores normalizados
     │   Deduplicação SHA-256 (padrão AVA)
     └─ SQLite: nós e arestas persistem entre sessões
         │
         ▼
  7. RETRIEVAL + REASONING
     ├─ Dense: FAISS top-15 por similaridade cosseno
     ├─ Graph: vizinhança 1-hop dos nós relacionados
     ├─ Rerank: cross-encoder ms-marco-MiniLM → top-4
     └─ LLM: síntese final com contexto enriquecido
```

---

## Correções em relação ao Blueprint original

| Problema no Blueprint | Solução implementada |
|---|---|
| `llama-cpp-python` (bindings) | `httpx` → `localhost:4003` (padrão AVA) |
| `AutoTokenizer` (transformers) | `tokenizers` Rust (sem overhead) |
| CLS token sem mean pooling | Mean pooling correto com attention mask |
| Web scraping (Playwright) | Ingestão local PDF/DOCX (caso de uso AVA) |
| GraphML sem persistência | SQLite com autocommit (padrão AVA) |
| JSON mode sem garantia | GBNF grammar explícita |
| `outputs[0][0][0]` errado | L2-normalized mean pooling |

---

## Instalação

```bash
# 1. Dependências
pip install -r requirements.txt --break-system-packages

# 2. Baixar e exportar modelos ONNX
python download_models.py

# Opcional: quantização INT8 (mais rápido no CPU)
python download_models.py --quantize
```

> Se o `multilingual-e5-small` já existir no AVA (Memory API), aponte
> `EMBEDDING_ONNX` em `config.py` para o caminho existente.

---

## Uso

### CLI

```bash
# Indexar um laudo arbóreo
python cli.py ingest /path/to/laudo_arboreo.pdf

# Indexar diretório inteiro
python cli.py ingest /path/to/laudos/

# Query
python cli.py query "Quais árvores apresentam risco de queda?"

# Modo interativo
python cli.py interactive

# Estatísticas
python cli.py stats
```

### Microserviço REST (porta 4005)

```bash
python api.py
```

```bash
# Ingerir documento
curl -X POST http://localhost:4005/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/documento.pdf"}'

# Query
curl -X POST http://localhost:4005/query \
  -H "Content-Type: application/json" \
  -d '{"text": "Qual é o estado fitossanitário das árvores?", "use_planner": true}'

# Stats
curl http://localhost:4005/stats
```

### Python direto

```python
from engine import AVAKnowledgeEngine

engine = AVAKnowledgeEngine()

# Ingestão
engine.ingest("laudo_parque_municipal.pdf")

# Query
resposta = engine.query("Quais espécies apresentam risco iminente?")
print(resposta)

engine.close()
```

---

## Configuração

Edite `config.py` para ajustar:

```python
# Endereços dos microserviços AVA
LLM_API_URL = "http://localhost:4003/v1/chat/completions"

# Caminhos dos modelos
EMBEDDING_ONNX = "models/multilingual-e5-small/model.onnx"

# Parâmetros de retrieval
RETRIEVAL.top_k_dense  = 15   # Candidatos FAISS antes do reranking
RETRIEVAL.top_k_final  = 4    # Após cross-encoder
RETRIEVAL.graph_hops   = 1    # Profundidade do KG (1 = vizinhos diretos)
RETRIEVAL.sem_chunk_pct= 95.0 # Percentil para quebra semântica
```

---

## Estrutura de Arquivos

```
ava_kg_rag/
├── config.py              # Configuração central
├── engine.py              # Orquestrador principal (7 etapas)
├── api.py                 # Microserviço FastAPI (porta 4005)
├── cli.py                 # Interface de linha de comando
├── download_models.py     # Setup dos modelos ONNX
├── requirements.txt
├── modules/
│   ├── llm_client.py       # Wrapper llama-server REST + GBNF
│   ├── embedding_engine.py # ONNX + mean pooling correto
│   ├── reranker.py         # Cross-encoder ONNX
│   ├── knowledge_graph.py  # NetworkX + SQLite persistente
│   ├── semantic_chunker.py # Chunking por distância cosseno
│   ├── vector_store.py     # FAISS + SHA-256 dedup
│   └── document_ingestor.py# PDF/DOCX/TXT local
└── storage/               # Criado automaticamente
    ├── knowledge_graph.db
    ├── faiss.index
    └── faiss_meta.json
```
