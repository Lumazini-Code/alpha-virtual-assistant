"""
AVA KG-RAG — Configuração Central
Adapta o blueprint KG-RAG ao stack existente do AVA.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ─── Caminhos ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(exist_ok=True)

GRAPH_DB_PATH      = STORAGE_DIR / "knowledge_graph.db"
FAISS_INDEX_PATH   = STORAGE_DIR / "faiss.index"
FAISS_META_PATH    = STORAGE_DIR / "faiss_meta.json"

# ─── Endpoints do AVA (microserviços existentes) ──────────────────────────────
LLM_API_URL        = "http://localhost:2001/v1/chat/completions"   # llama-server
MEMORY_API_URL     = "http://localhost:3001"                        # Memory RAG
TTS_API_URL        = "http://localhost:4003"                        # TTS (mesma porta, rota diferente)

# ─── Modelos ONNX (mesmos usados no AVA) ─────────────────────────────────────
EMBEDDING_ONNX     = "Models/multilingual-e5-small/multilingual-e5-small.onnx"
EMBEDDING_TOKENIZER= "Models/multilingual-e5-small/tokenizer.json"
RERANKER_ONNX      = "Models/ms-marco-MiniLM-L-6-v2/ms-marco-MiniLM-L-6-v2.onnx"
RERANKER_TOKENIZER = "Models/ms-marco-MiniLM-L-6-v2/tokenizer.json"

# ─── Parâmetros de retrieval ──────────────────────────────────────────────────
@dataclass
class RetrievalConfig:
    top_k_dense: int    = 15    # Candidatos iniciais do FAISS
    top_k_final: int    = 4     # Após reranking cross-encoder
    graph_hops: int     = 1     # Graus de vizinhança no grafo (1 = apenas vizinhos diretos)
    sem_chunk_pct: float= 95.0  # Percentil de distância para quebra semântica
    embed_dim: int      = 384   # Dimensão do multilingual-e5-small
    sim_threshold: float= 0.92  # Threshold dedup (igual ao Memory API)

# ─── Parâmetros do LLM ────────────────────────────────────────────────────────
@dataclass
class LLMConfig:
    model: str          = "local"   # Ignorado — llama-server decide o modelo carregado
    max_tokens: int     = 1024
    temperature: float  = 0.3       # Baixo: queremos respostas factuais, não criativas
    planner_temp: float = 0.1       # Ainda mais baixo para o JSON planner
    stream: bool        = False

@dataclass
class WebConfig:
    max_results_per_query: int   = 5      # URLs retornadas por sub-tópico
    max_chars_per_page: int      = 8000   # Trunca páginas longas (evita overflow de contexto)
    timeout_seconds: float       = 15.0   # Timeout por request HTTP
    delay_between_requests: float= 3.0    # Delay entre queries DDG (evita rate limit)

RETRIEVAL = RetrievalConfig()
LLM       = LLMConfig()
WEB       = WebConfig()
