from __future__ import annotations

import re
import time
import json
import sqlite3
import hashlib
import asyncio
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal

import heapq

import numpy as np
import pyarrow as pa
import lancedb
import scipy.sparse as sp
from pydantic import BaseModel

# ── MODIFIED: Import from onnx_client instead of local ONNX ───────────────────
from onnx_client import EmbeddingClient, RerankerClient, DEFAULT_ONNX_BASE_URL

# ── MODIFIED: Import VectorStore from vector_store (graceful fallback) ────────
try:
    from modules.vector_store import VectorStore, VectorEntry as VSVectorEntry
    _VS_AVAILABLE = True
except ImportError as _imp_err:
    _VS_AVAILABLE = False
    VectorStore = None          # type: ignore[assignment,misc]
    VSVectorEntry = None        # type: ignore[assignment,misc]

# ── Configuração ───────────────────────────────────────────────────────────────

ONNX_SERVING_URL = DEFAULT_ONNX_BASE_URL

# ── Prefixos E5 (multilingual-e5-small/base/large e primos) ─────────────────
# Modelos da família E5 são treinados com prefixos textuais fixos — sem eles,
# o embedding de QUALQUER texto colapsa para uma região estreita do espaço,
# e cosine similarity entre frases sem nenhuma relação semântica sai
# artificialmente alta (~0.85-0.93), quebrando dedup e busca por igual.
# Ref.: model card intfloat/multilingual-e5-small.
# Se um dia trocar pra um modelo que não é da família E5, ajuste isso pra "".
EMBED_QUERY_PREFIX   = "query: "
EMBED_PASSAGE_PREFIX = "passage: "

# Longo prazo
DB_PATH           = "./memory/ava_memory.db"
FAISS_INDEX_PATH  = "./memory/ava_memory.index"
FAISS_ID_MAP_PATH = "./memory/ava_id_map.npy"

# Curto prazo
ST_DB_PATH           = "./memory/ava_short_term.db"
ST_FAISS_INDEX_PATH  = "./memory/ava_short_term.index"
ST_FAISS_ID_MAP_PATH = "./memory/ava_short_term_id_map.npy"

# Knowledge (Vector Store / KG-RAG)
VS_DB_PATH           = "./memory/ava_kg_chunks.db"
VS_FAISS_INDEX_PATH  = "./memory/ava_kg_vectors.index"
VS_FAISS_ID_MAP_PATH = "./memory/ava_kg_vectors_id_map.npy"
VS_MIN_SCORE         = 0.70

# ── NEW: Indexed Files (local-scraping) ────────────────────────────────────────
IF_DB_PATH           = "./memory/ava_indexed_files.db"
IF_FAISS_INDEX_PATH  = "./memory/ava_indexed_files.index"
IF_FAISS_ID_MAP_PATH = "./memory/ava_indexed_files_id_map.npy"
IF_MIN_SCORE         = 0.75
IF_MAX_CONTENT_SIZE  = 500_000    # 500 KB — mesmo limite do local-scraping
IF_MAX_CHUNKS        = 2000       # máx chunks por arquivo
CHUNK_SIZE           = 500        # chars por chunk
CHUNK_OVERLAP        = 100        # chars de sobreposição entre chunks
IF_EMBED_BATCH_SIZE  = 64         # chunks por batch de embedding

# ── NEW: Dicionário Visual (módulo vision.py — "Tradução de Objetos") ─────────
# Armazena, por conceito visual, N embeddings de exemplo (DINOv3) + a
# descrição textual ("significado") do objeto. A descrição também é gravada
# na memória de longo prazo (mesma tabela `memories`), então ela é
# recuperável pela leitura normal (/read) igual a qualquer outra memória.
VD_DB_PATH           = "./memory/ava_visual_dict.db"
VD_FAISS_INDEX_PATH  = "./memory/ava_visual_dict.index"
VD_FAISS_ID_MAP_PATH = "./memory/ava_visual_dict_id_map.npy"
VD_EMBED_DIM         = 384     # DINOv3 ViT-S/16 (ajustar se trocar de encoder)
VD_MIN_SCORE         = 0.55    # similaridade mínima p/ considerar candidato válido
VD_TOP_K             = 5
VD_AMBIGUOUS_MARGIN  = 0.05    # se score#1 - score#2 < margem → resposta ambígua

# ── NEW: Dicionário de Rostos (módulo vision.py — reconhecimento facial) ──────
# Mesmo padrão do dicionário visual acima, só que pra pessoas: por pessoa,
# guarda N embeddings de exemplo (EdgeFace) — permite reconhecer o mesmo
# rosto em ângulos/luz diferentes. Fica em banco/índice separados porque a
# dimensão do embedding e os thresholds de similaridade de rosto costumam
# ser bem diferentes dos de objeto genérico (DINOv3).
FD_DB_PATH           = "./memory/ava_face_dict.db"
FD_FAISS_INDEX_PATH  = "./memory/ava_face_dict.index"
FD_FAISS_ID_MAP_PATH = "./memory/ava_face_dict_id_map.npy"
FD_EMBED_DIM         = 512     # EdgeFace — ajustar se a saída do seu .onnx for outra dim
FD_MIN_SCORE         = 0.42    # cosine similarity — modelos estilo ArcFace/EdgeFace costumam
                                # precisar de threshold mais baixo que embeddings genéricos
                                # (DINOv3 usa 0.55); CALIBRE em cima dos seus próprios exemplos
                                # antes de confiar nisso em produção
FD_TOP_K             = 3
FD_AMBIGUOUS_MARGIN  = 0.05

# ── NEW: Dicionário de Voz (embeddings de locutor — ex.: CAM++/3D-Speaker) ────
# Mesmo padrão do dicionário de rostos, só que pra identificação de locutor a
# partir de embeddings extraídos de áudio: por pessoa, guarda N embeddings de
# exemplo (um por frase falada registrada — permite reconhecer a mesma voz em
# condições/ângulos de captação diferentes). A descrição de quem é a pessoa é
# REPLICADA na memória de longo prazo (igual ao link_to_memory do dicionário
# visual), então ela é recuperável pela leitura normal (/read).
VO_DB_PATH           = "./memory/ava_voice_dict.db"
VO_FAISS_INDEX_PATH  = "./memory/ava_voice_dict.index"
VO_FAISS_ID_MAP_PATH = "./memory/ava_voice_dict_id_map.npy"
VO_EMBED_DIM         = 192     # CAM++ (3D-Speaker) — ajustar se a saída do seu .onnx for outra dim
VO_MIN_SCORE         = 0.45    # cosine similarity — CALIBRE em cima dos seus próprios exemplos
                                # antes de confiar nisso em produção (threshold de voz varia
                                # bastante com microfone, ruído e distância da boca)
VO_TOP_K             = 3
VO_AMBIGUOUS_MARGIN  = 0.05

# ── NEW: LanceDB (substitui FAISS + .npy de id-map) ───────────────────────────
# Um único diretório LanceDB guarda os vetores de TODOS os stores (longo prazo,
# curto prazo, arquivos indexados, dicionários visual/rosto/voz, tools). Vetores
# de dimensões diferentes não cabem na mesma coluna FixedSizeList, então existe
# UMA tabela por dimensão (`vectors_384`, `vectors_512`, `vectors_192`…) e a
# coluna `memory_type` separa os stores dentro dela (filtro nativo, pré-filtro).
# Os caminhos FAISS_*_PATH / *_ID_MAP_PATH acima passam a ser usados SÓ para a
# migração automática one-shot dos índices antigos (ver MemoryIndex).
LANCE_DIR          = "./memory/ava_lance"
LANCE_TABLE_PREFIX = "vectors"

MT_LONG_TERM    = "long_term"
MT_SHORT_TERM   = "short_term"
MT_INDEXED_FILE = "indexed_file"
MT_VISUAL_DICT  = "visual_dict"
MT_FACE_DICT    = "face_dict"
MT_VOICE_DICT   = "voice_dict"
MT_TOOL         = "tool"

# ── NEW: retrieval de tools (RAG de tool selection) ───────────────────────────
# Recall é mais crítico que precisão aqui: uma tool que não aparece não pode ser
# chamada — por isso os thresholds são BEM mais permissivos que os do /read
# (READ_MIN_SCORE_STRICT = 0.85).
TOOLS_DB_PATH               = "./memory/ava_tools.db"
TOOLS_TOP_K                 = 8      # tools "primary" por seleção
TOOLS_MIN_SCORE             = 0.45   # cosine mínimo p/ uma tool entrar como primary
TOOLS_LOW_CONFIDENCE_SCORE  = 0.55   # top-1 abaixo disso → confiança baixa → rede mais larga
TOOLS_FALLBACK_TOP_K        = 12     # quantas tools devolver (sem threshold) se confiança baixa
TOOLS_MAX_RELATED           = 4      # tools extras trazidas só pelo grafo (co-uso/sequência)
TOOLS_MAX_EXAMPLES          = 10     # exemplos de query fornecidos no cadastro, por tool
TOOLS_MAX_LEARNED_EXAMPLES  = 20     # queries reais aprendidas via tools_record_usage, por tool
TOOLS_LEARN_DEDUP_SCORE     = 0.95   # query aprendida ≥ isso de outra já existente → não duplica
TOOLS_USAGE_MAX_SEQUENCE    = 12     # teto de tools por registro de uso (anti-O(n²))

# ── NEW: busca por grafo nos dicionários (rosto/objeto/voz) ───────────────────
DICT_GRAPH_MAX_RELATED = 3

EMBED_DIM            = 384
READ_MIN_SCORE       = 0.83
# ── Thresholds recalibrados (ver diagnóstico) ────────────────────────────────
# multilingual-e5-small tem anisotropia alta: MEDIMOS cosine médio de ~0.78
# (chegando a ~0.86) entre frases SEM NENHUMA relação semântica (idiomas e
# domínios completamente diferentes). Os valores antigos (DEDUP=0.92,
# UPDATE=0.75) estavam dentro/abaixo desse piso de ruído — por isso qualquer
# fato novo batia como "possible_update"/"duplicate_semantic" contra
# qualquer fato antigo. Subimos os dois pra ficarem acima do piso medido, e
# a decisão final na faixa "parecido" passa a ser confirmada pelo reranker
# cross-encoder (RERANK_DUPLICATE_SCORE/RERANK_UPDATE_SCORE abaixo), que não
# sofre do mesmo problema de anisotropia — os thresholds de cosine aqui viram
# só um filtro de RECALL (candidatos a checar), não mais a decisão final.
DEDUP_THRESHOLD      = 0.95
TOP_K_READ           = 5
DECAY_HALF_LIFE_DAYS = 90
DECAY_JOB_INTERVAL_S = 3600

# ── Hebbian graph / PPR retrieval ──────────────────────────────────────────
# Grafo de co-ativação sobre as memórias de longo prazo, com recuperação por
# Personalized PageRank. NENHUM valor hardcodado na lógica abaixo — tudo é
# lido daqui pra poder ser tunado depois sem tocar no código.
EDGE_DECAY_HALF_LIFE_DAYS    = 90     # pode diferir de DECAY_HALF_LIFE_DAYS; começar igual
EDGE_PRUNE_THRESHOLD         = 0.05   # arestas abaixo desse peso são deletadas
EDGE_LEARNING_RATE           = 0.15   # lr do crescimento saturante hebbiano
EDGE_MIN_SCORE_TO_LINK       = 0.80   # ambas as memórias precisam disso p/ criar aresta NOVA
LTP_ACCESS_BOOST             = 0.05   # boost de confiança por leitura (saturante)
PPR_DAMPING                  = 0.85
PPR_MAX_ITER                 = 20
PPR_CONVERGENCE_EPS          = 1e-4
PPR_MIN_NEIGHBORS_PER_HOP    = 20     # teto de vizinhos por nó (SQLite barato)
PPR_MIN_ACTIVATION_REINFORCE = 0.05   # threshold de elegibilidade p/ reforço (Opção B)
PPR_SPREAD_WEIGHT            = 0.4    # peso do score PPR vs score direto do índice vetorial
MMR_LAMBDA                   = 0.7    # 1.0 = pura relevância, 0.0 = pura diversidade

# ── NEW: log de ativação p/ visualização em tempo real (live_activation_server.py) ──
ACTIVATION_LOG_PATH          = "./memory/activation_log.jsonl"
ACTIVATION_LOG_MAX_IDS       = 60     # não loga o rank inteiro se o subgrafo for gigante
HEBBIAN_LINK_MAX_IDS         = 12     # safeguard: teto de ids linkados por /read (anti-O(n²))

# ── NEW: expansão ADAPTATIVA do grafo (substitui os 2 hops fixos) ─────────────
# BFS priorizado por peso acumulado: sempre expande o nó de maior peso de
# caminho (produto dos pesos de aresta ao longo do melhor caminho até ele).
# Para quando (a) o número de nós carregados chega ao teto, ou (b) o melhor nó
# restante tem peso de caminho abaixo do mínimo.
PPR_EXPAND_MAX_NODES         = 200
PPR_EXPAND_MIN_PATH_WEIGHT   = 0.05

# ── NEW: MMR — fallback quando NÃO existe aresta entre dois candidatos ────────
# Nesse caso a "similaridade" passa a ser a cosine direta entre os embeddings
# (em vez de 0.0). Cosines de embeddings quaisquer raramente ficam perto de 0,
# então MMR_COSINE_FLOOR desconta esse piso: sim = max(0, (cos-floor)/(1-floor)).
# 0.0 = usa a cosine crua.
MMR_COSINE_FLOOR             = 0.0

# ── NEW: arestas tipadas/direcionadas ─────────────────────────────────────────
# Convenção de direção (memory_id_a → memory_id_b) nos tipos dirigidos:
#   temporal_precedence : a veio ANTES de b
#   updates             : a é a versão NOVA que substitui b
# Tipos simétricos guardam o par normalizado (min, max), como antes.
EDGE_CO_ACTIVATION = "co_activation"
EDGE_TEMPORAL      = "temporal_precedence"
EDGE_CONTRADICTS   = "contradicts"
EDGE_UPDATES       = "updates"
EDGE_TYPES          = (EDGE_CO_ACTIVATION, EDGE_TEMPORAL, EDGE_CONTRADICTS, EDGE_UPDATES)
EDGE_DIRECTED_TYPES = frozenset({EDGE_TEMPORAL, EDGE_UPDATES})
# Fator aplicado ao peso da aresta ao PROPAGAR no PPR: (a→b, b→a).
#   contradicts: (0,0) — não propaga ativação por contradição (o MMR ainda a usa);
#   updates: de a memória ANTIGA pra NOVA propaga forte (1.0), da nova pra antiga quase nada.
EDGE_PPR_FACTORS = {
    EDGE_CO_ACTIVATION: (1.0, 1.0),
    EDGE_TEMPORAL:      (0.6, 0.3),
    EDGE_CONTRADICTS:   (0.0, 0.0),
    EDGE_UPDATES:       (0.1, 1.0),
}
# Fator do peso da aresta como "similaridade" no MMR (redundância entre candidatos).
EDGE_MMR_FACTORS = {
    EDGE_CO_ACTIVATION: 1.0,
    EDGE_TEMPORAL:      0.3,
    EDGE_CONTRADICTS:   1.0,   # mesmo assunto, versões conflitantes → não mostrar as duas
    EDGE_UPDATES:       1.0,
}
# Arestas estruturais (correção/contradição) não sofrem decay nem poda por inatividade.
EDGE_TYPES_NO_DECAY    = frozenset({EDGE_CONTRADICTS, EDGE_UPDATES})
# Peso inicial de arestas criadas de forma EXPLÍCITA (link_to no /write) — o
# lr hebbiano (0.15) é fraco demais pra um vínculo declarado.
EDGE_EXPLICIT_LINK_WEIGHT = 0.8
EDGE_EXPLICIT_TYPES       = frozenset({EDGE_CONTRADICTS, EDGE_UPDATES})

# ── NEW (solução 2): faixa de similaridade "provável correção" ────────────────
# Entre UPDATE_SIM_THRESHOLD e DEDUP_THRESHOLD, um texto novo não é nem uma
# duplicata clara (>= DEDUP_THRESHOLD, rejeitada) nem algo totalmente
# diferente (< UPDATE_SIM_THRESHOLD, vira memória nova). Nessa faixa, o
# write é recusado com reason="possible_update:<score>" e a memória mais
# próxima (candidate_id/candidate_text/candidate_score) volta na resposta
# — cabe a quem chamou (o extrator LLM) decidir se reenvia o /write com
# action="update" e memory_id=candidate_id, ou se era mesmo um fato novo.
UPDATE_SIM_THRESHOLD = 0.87

# ── NEW: confirmação por reranker cross-encoder ───────────────────────────────
# Bi-encoder (embedding + cosine) é só triagem/recall aqui — decide se existe
# um candidato "parecido o bastante pra vale a pena checar", não se É de fato
# duplicata/atualização. Quem decide isso é o cross-encoder (ms-marco-MiniLM
# via /v1/score), que compara os dois textos diretamente e não tem o mesmo
# problema de anisotropia do espaço de embeddings. Scores do /v1/score são
# sigmoid-normalizados em [0, 1] (ver onnx_client.RerankerClient.score).
RERANK_DUPLICATE_SCORE = 0.90   # cross-encoder >= isso → duplicate_semantic
RERANK_UPDATE_SCORE     = 0.55   # cross-encoder >= isso (e < duplicate) → possible_update
# abaixo de RERANK_UPDATE_SCORE → o bi-encoder deu falso positivo (anisotropia/
# domínio compartilhado); grava como fato novo mesmo assim.

ST_TTL_HOURS          = 24.0
ST_CLEANUP_INTERVAL_S = 1800

# ── Otimização de tokens (reduzir payload enviado ao LLM) ─────────────────────
# Evita erros 413 Payload Too Large no Groq quando o /read é chamado múltiplas
# vezes pelo pipeline CoT → LLM (5 passos × N entradas × ~1KB cada).
#
# Estratégia em 3 camadas:
#   1. Thresholds mais seletivos para o /read (combinação de 4 fontes)
#   2. Teto de caracteres por entrada individual (trunca preservando palavra)
#   3. Teto global de caracteres no resultado final (corta entradas de menor score)
READ_TOP_K_FINAL        = 3        # era TOP_K_READ (5) — menos entradas no /read
READ_TOTAL_MAX_CHARS    = 2400     # teto global de chars retornados pelo /read
READ_LT_MAX_CHARS       = 600      # teto por entrada de memória de longo prazo
READ_ST_MAX_CHARS       = 350      # teto menor para curto prazo (texto denso)
READ_VS_MAX_CHARS       = 500      # teto para chunks de conhecimento (KG-RAG)
READ_IF_MAX_CHARS       = 500      # teto para chunks de arquivos indexados
READ_MIN_SCORE_STRICT   = 0.85     # mais seletivo que READ_MIN_SCORE (0.83)
IF_MIN_SCORE_READ       = 0.82     # threshold p/ /read (era min(0.83,0.75)=0.75)
                                   # corrige bug que retornava chunks demais de IF

# ── Parâmetros de busca contextual ─────────────────────────────────────────────
QUERY_SHORT_WORDS     = 6
QUERY_AMBIGUOUS_RATIO = 0.55
CONTEXT_MAX_CHARS     = 1200
CONTEXT_TURNS_FETCH   = 6
DUAL_CONTEXT_WEIGHT   = 0.35

# ── NEW: Decomposição de query composta (busca por segmento, sem LLM) ─────────
# Quando a query tem vários "sub-pedidos" numa frase só (ex.: "quero fazer X
# usando Y para conseguir Z"), embedar a frase inteira dilui a atenção
# semântica e nenhum sub-tópico fica bem representado no vetor final. Em vez
# de pedir a um LLM para gerar sub-queries, a query é quebrada por regras
# heurísticas (pontuação, vírgula, gerúndio, "para <verbo>") e cada pedaço é
# embedado/buscado separadamente — puro cosine similarity, sem custo de LLM.
SEGMENT_TRIGGER_WORDS = 12   # frases com até isso não valem a pena segmentar
SEGMENT_MIN_WORDS     = 3    # fragmento menor que isso é remendado no vizinho
SEGMENT_MAX_COUNT     = 4    # teto de segmentos por query (custo de embed/batch)

# ── /read_st — leitura crua do short-term (sem busca semântica) ───────────────
# Usada pelo LLM para montar o histórico recente da conversa como contexto,
# sem passar pelo pipeline de embeddings/scoring do /read.
ST_READ_DEFAULT_PAIRS = 5

_STOP_WORDS: frozenset[str] = frozenset({
    "o","a","os","as","um","uma","uns","umas","de","do","da","dos","das",
    "em","no","na","nos","nas","por","para","com","que","se","é","são",
    "foi","isso","esse","essa","eu","tu","ele","ela","nós","vocês","eles",
    "elas","me","te","nos","como","mas","mais","já","não","sim","tem",
    "ter","ser","fazer","ir","vou","vai","aqui","lá","também","então",
    "quando","onde","porque","qual","quais",
    "the","a","an","is","are","was","were","be","been","being","have",
    "has","had","do","does","did","will","would","could","should","may",
    "might","shall","can","to","of","in","on","at","by","for","with",
    "about","this","that","it","he","she","we","they","i","you","and",
    "or","but","so","if","my","your","his","her","our","their","what",
    "how","when","where","why","which","who","then","there","here","also",
})

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [Memory] %(message)s")
log = logging.getLogger("ava.memory")

if not _VS_AVAILABLE:
    log.warning("vector_store module not available — knowledge search disabled")
else:
    log.info("vector_store module found — knowledge search enabled")


# ── Modelos de request/response — memória ─────────────────────────────────────

class Turn(BaseModel):
    role:    Literal["user", "assistant"]
    content: str

class WriteRequest(BaseModel):
    text:        str
    source:      str   = "chat"
    confidence:  float = 1.0
    # "esquecível?" — quando False, a memória fica isenta do decay por
    # inatividade (apply_decay a ignora), mas ainda pode ser removida por
    # deleção explícita. Default True preserva o comportamento anterior.
    forgettable: bool  = True
    # ── NEW (solução 3): meia-vida específica desta memória, em dias.
    # Quando None, apply_decay cai para o default global
    # (DECAY_HALF_LIFE_DAYS). Ex.: ttl_days=7 para "viagem essa semana",
    # deixado em branco para memórias sem prazo específico.
    ttl_days:    Optional[float] = None
    # ── NEW (solução 2): "criar" (default) ou "atualizar/corrigir" uma
    # memória existente. Em action="update":
    #   - se memory_id vier preenchido, atualiza diretamente essa memória;
    #   - se memory_id vier vazio, procura a memória de longo prazo mais
    #     semelhante (score >= UPDATE_SIM_THRESHOLD) e atualiza ela; se
    #     nenhuma candidata suficientemente parecida existir, o write falha
    #     com reason="update_target_not_found" em vez de criar uma memória
    #     nova (evita que uma correção vire um fato solto).
    action:      Literal["create", "update"] = "create"
    memory_id:   Optional[int] = None
    # ── NEW (arestas tipadas): grava como memória NOVA já ligada a uma
    # existente por uma aresta tipada. Complementa o fluxo "possible_update":
    # em vez de action="update" (sobrescreve o texto), o extrator pode manter as
    # duas memórias e declarar a relação:
    #   updates             → a nova substitui `link_to` (aresta nova → antiga)
    #   contradicts         → a nova conflita com `link_to` (simétrica)
    #   temporal_precedence → `link_to` veio ANTES da nova (aresta link_to → nova)
    #   co_activation       → associação simples
    # Quando link_to + link_type vêm preenchidos, a faixa "possible_update" NÃO
    # bloqueia o write (o chamador já decidiu que é um fato distinto); duplicata
    # exata/semântica (>= DEDUP_THRESHOLD) continua sendo recusada.
    link_to:     Optional[int] = None
    link_type:   Optional[Literal["co_activation", "temporal_precedence", "contradicts", "updates"]] = None

class WriteBatchRequest(BaseModel):
    # ── NEW (solução 1): grava várias memórias em uma única chamada,
    # evitando N round-trips HTTP/MCP quando o extrator LLM devolve um
    # array de fatos para uma mesma dupla pergunta-resposta.
    items: list[WriteRequest]
    # ── NEW (arestas tipadas): quando True, memórias gravadas consecutivamente
    # neste batch recebem aresta temporal_precedence (anterior → seguinte), na
    # ordem do array. Default False — a ordem do array só é sinal temporal
    # quando o extrator garante isso.
    sequential: bool = False

class WriteSTRequest(BaseModel):
    session_id: str
    turns:      list[Turn]

class ReadRequest(BaseModel):
    query:      str
    top_k:      int   = TOP_K_READ
    min_score:  float = READ_MIN_SCORE
    session_id: Optional[str] = None
    strategy:   str = "auto"

class WriteResponse(BaseModel):
    stored:    bool
    reason:    str
    memory_id: Optional[int] = None
    # ── NEW (solução 2): só populados quando reason começa com
    # "possible_update:" — a memória de longo prazo mais parecida
    # encontrada na faixa [UPDATE_SIM_THRESHOLD, DEDUP_THRESHOLD), para o
    # chamador decidir se reenvia como action="update".
    candidate_id:    Optional[int]   = None
    candidate_text:  Optional[str]   = None
    candidate_score: Optional[float] = None
    # ── NEW: score do reranker cross-encoder que confirmou/descartou o
    # candidato do bi-encoder (ver RERANK_UPDATE_SCORE/RERANK_DUPLICATE_SCORE).
    # None quando não houve candidato a checar.
    candidate_rerank_score: Optional[float] = None

class WriteBatchResponse(BaseModel):
    # ── NEW (solução 1): um WriteResponse por item de entrada, na mesma
    # ordem de WriteBatchRequest.items.
    results: list[WriteResponse]
    stored_count: int
    total: int

class WriteSTResponse(BaseModel):
    stored:   bool
    reason:   str
    turn_ids: list[int] = []

class ReadSTRequest(BaseModel):
    session_id: str
    n_pairs:    int = ST_READ_DEFAULT_PAIRS

class ReadSTResponse(BaseModel):
    session_id:     str
    turns:          list[Turn]
    pairs_returned: int

class MemoryEntry(BaseModel):
    id:           int
    text:         str
    score:        float
    confidence:   float
    created_at:   float
    access_count: int
    memory_type:  Literal["long_term", "short_term", "knowledge", "indexed_file"]
    session_id:   Optional[str]        = None
    turns:        Optional[list[Turn]] = None
    source:       Optional[str]        = None
    # ── NEW: esquecível — só populado para memory_type == "long_term" ──
    forgettable:  Optional[bool]       = None
    # ── NEW (solução 3): meia-vida por memória, em dias — só populado
    # para memory_type == "long_term" quando definida (senão usa o
    # default global DECAY_HALF_LIFE_DAYS).
    ttl_days:     Optional[float]      = None
    # ── NEW (Hebbian graph): "primary" = hit direto (FAISS/VS/IF/ST);
    # "related" = alcançado SÓ via propagação PPR no grafo de co-ativação —
    # deixa o LLM downstream distinguir fato diretamente relevante de
    # contexto associativo ──
    match_type:   Literal["primary", "related"] = "primary"
    # ── NEW: Indexed file metadata ──
    file_path:    Optional[str]        = None
    file_name:    Optional[str]        = None
    extension:    Optional[str]        = None
    content_hash: Optional[str]        = None
    file_hash:    Optional[str]        = None

class ReadResponse(BaseModel):
    results:  list[MemoryEntry]
    query:    str
    strategy: str


# ── NEW: memória alcançada pelo grafo a partir de um hit de dicionário ────────

class RelatedMemory(BaseModel):
    memory_id: int
    text:      str
    score:     float     # PPR_SPREAD_WEIGHT * rank PPR — comparável só entre itens desta lista


# ── NEW: Modelos de request/response — arquivos indexados ─────────────────────

class IndexedFileWriteRequest(BaseModel):
    """Store a complete indexed file with full content, hash, and auto-chunking."""
    file_path:    str
    file_name:    str
    extension:    str   = ""
    content:      str
    file_hash:    str   = ""     # SHA-256 do arquivo original no disco
    size:         int   = 0
    modified:     str   = ""     # data de modificação ISO
    source:       str   = "local_scraping"
    confidence:   float = 1.0
    force_reindex: bool = False

class IndexedFileWriteResponse(BaseModel):
    stored:         bool
    reason:         str
    file_id:        Optional[int] = None
    chunks_created: int  = 0
    was_reindexed:  bool = False
    hash_match:     bool = True

class IndexedFileReadRequest(BaseModel):
    # ── NEW: leitura exata por caminho absoluto ──
    # Quando `file_path` é enviado, a busca semântica (via `query`) é
    # ignorada — o lookup é feito diretamente por igualdade de caminho
    # (mesma chave usada em `/indexed-file/write`), garantindo que dois
    # arquivos com o mesmo nome em pastas diferentes nunca se confundam.
    # `query` continua obrigatório apenas quando `file_path` não é enviado.
    file_path: Optional[str] = None
    query:     Optional[str] = None
    top_k:     int   = 5
    min_score: float = IF_MIN_SCORE
    # ── NEW: quando False, as entradas retornadas vêm com `content=""` —
    # útil no modo "file_path + query" para arquivos grandes, onde o
    # chamador só quer as chunks e não o payload inteiro do arquivo.
    # Não afeta o modo "file_path" sozinho (arquivo inteiro), que sempre
    # devolve `content` preenchido — é o próprio propósito desse modo.
    include_full_content: bool = True

class IndexedFileEntry(BaseModel):
    file_id:      int
    file_path:    str
    file_name:    str
    extension:    str
    content:      str                   # conteúdo completo do arquivo
    file_hash:    str
    content_hash: str
    size:         int
    modified:     str
    score:        float
    confidence:   float
    created_at:   float
    access_count: int
    source:       str = "local_scraping"
    chunk_text:   Optional[str] = None  # chunk específico que deu match
    # ── NEW: posição da chunk dentro do arquivo — ajuda o chamador a se
    # orientar sem precisar do arquivo inteiro (útil pra arquivos grandes) ──
    chunk_index:  Optional[int] = None
    char_start:   Optional[int] = None
    char_end:     Optional[int] = None
    chunk_id: Optional[int] = None 
    # ── NEW: "exact_path" (lookup direto por file_path) ou "semantic" (busca vetorial) ──
    match_type:   str = "semantic"

class IndexedFileReadResponse(BaseModel):
    results:   list[IndexedFileEntry]
    query:     Optional[str] = None
    file_path: Optional[str] = None

class IndexedFileCheckResponse(BaseModel):
    indexed:          bool
    hash_match:       Optional[bool]   = None
    file_id:          Optional[int]    = None
    stored_file_hash: Optional[str]    = None
    stored_content_hash: Optional[str] = None
    stored_modified:  Optional[str]    = None
    chunks_count:     Optional[int]    = None


# ── NEW: Modelos de request/response — Dicionário Visual (vision.py) ──────────
#
# O módulo vision.py (pipeline de "Tradução de Objetos") faz a segmentação,
# extração de crops e geração de embeddings (DINOv3) de cada objeto detectado
# numa imagem. Ele NÃO guarda nenhum estado — apenas manda o embedding (+ meta)
# pra cá via HTTP, e este arquivo é o único responsável por persistir o
# "dicionário" (embeddings no FAISS + texto/significado no SQLite).
#
# A descrição textual de cada conceito também é replicada na tabela
# `memories` (longo prazo), então ela aparece normalmente em qualquer
# chamada de /read — não é preciso um endpoint de leitura separado pra isso.

class VisualDictWriteRequest(BaseModel):
    concept_name:   str                 # nome curto do objeto/conceito, ex.: "caneca azul"
    description:    str                 # "significado" textual — o que é, contexto, uso etc.
    embedding:      list[float]         # embedding do crop (DINOv3), normalizado ou não
    source:         str   = "vision_pipeline"
    confidence:     float = 1.0
    # quando True (padrão), a descrição também é gravada como memória de
    # longo prazo normal (pesquisável via /read). Só é feito na primeira vez
    # que o conceito é criado — exemplos adicionais do mesmo conceito não
    # duplicam a entrada de texto, só adicionam mais um vetor de embedding.
    link_to_memory: bool  = True

class VisualDictWriteResponse(BaseModel):
    stored:       bool
    reason:       str
    concept_id:   Optional[int] = None
    embedding_id: Optional[int] = None
    memory_id:    Optional[int] = None   # id em `memories`, se link_to_memory=True
    new_concept:  bool = False           # True se um conceito novo foi criado agora

class VisualDictReadRequest(BaseModel):
    embedding: list[float]        # embedding do crop consultado (DINOv3)
    top_k:     int   = VD_TOP_K
    min_score: float = VD_MIN_SCORE
    # ── NEW: expande os hits pelo grafo Hebbiano (memórias LT relacionadas).
    # Desligue em loops por-frame se a latência importar.
    include_related: bool = True

class VisualDictCandidate(BaseModel):
    concept_id:   int
    concept_name: str
    description:  str
    score:        float
    confidence:   float
    access_count: int
    memory_id:    Optional[int] = None

class VisualDictReadResponse(BaseModel):
    results:   list[VisualDictCandidate]
    ambiguous: bool     # True → nenhum candidato confiável / candidatos muito próximos;
                         # quem chama (vision.py / orquestrador) deve perguntar ao usuário
    related:   list[RelatedMemory] = []   # NEW: memórias vizinhas no grafo dos hits com memory_id

class VisualDictEntry(BaseModel):
    concept_id:    int
    concept_name:  str
    description:   str
    source:        str
    confidence:    float
    memory_id:     Optional[int] = None
    examples_count: int
    created_at:    float
    access_count:  int


# ── NEW: Modelos de request/response — dicionário de rostos ───────────────────

class FaceDictWriteRequest(BaseModel):
    person_name: str                 # nome da pessoa, ex.: "eu" / "Fulano"
    embedding:   list[float]         # embedding do rosto alinhado (EdgeFace)
    description: str   = ""          # quem é essa pessoa (relação, contexto etc.)
    source:      str   = "vision_pipeline"
    confidence:  float = 1.0
    # ── NEW: replica a descrição na memória de longo prazo (source="face_dict"),
    # o que dá ao rosto um nó no grafo Hebbiano (necessário pra busca por
    # grafo funcionar em rostos). Default False = comportamento anterior: a
    # descrição do rosto NÃO entra no /read geral. Também "promove" uma pessoa
    # já cadastrada sem link, se vier com description.
    link_to_memory: bool = False

class FaceDictWriteResponse(BaseModel):
    stored:       bool
    reason:       str
    person_id:    Optional[int] = None
    embedding_id: Optional[int] = None
    memory_id:    Optional[int] = None   # NEW: id em `memories`, se linkado
    new_person:   bool = False        # True se a pessoa foi criada agora

class FaceDictReadRequest(BaseModel):
    embedding: list[float]        # embedding do rosto consultado (EdgeFace)
    top_k:     int   = FD_TOP_K
    min_score: float = FD_MIN_SCORE
    include_related: bool = True   # NEW: expansão por grafo (só p/ rostos com memory_id)

class FaceCandidate(BaseModel):
    person_id:    int
    person_name:  str
    description:  str
    score:        float
    confidence:   float
    access_count: int
    memory_id:    Optional[int] = None   # NEW

class FaceDictReadResponse(BaseModel):
    results:   list[FaceCandidate]
    ambiguous: bool     # True → nenhum candidato confiável / candidatos muito próximos
    related:   list[RelatedMemory] = []   # NEW

class FaceDictEntry(BaseModel):
    person_id:      int
    person_name:    str
    description:    str
    source:         str
    confidence:     float
    memory_id:      Optional[int] = None   # NEW
    examples_count: int
    created_at:     float
    access_count:   int


# ── NEW: Modelos de request/response — dicionário de voz ──────────────────────

class VoiceDictWriteRequest(BaseModel):
    person_name: str                 # nome da pessoa, ex.: "eu" / "Fulano"
    embedding:   list[float]         # embedding do locutor (CAM++ etc.), normalizado ou não
    description: str   = ""          # quem é essa pessoa (relação, contexto etc.) —
                                     # também gravada na memória de longo prazo
    source:      str   = "voice_pipeline"
    confidence:  float = 1.0

class VoiceDictWriteResponse(BaseModel):
    stored:       bool
    reason:       str
    person_id:    Optional[int] = None
    embedding_id: Optional[int] = None
    memory_id:    Optional[int] = None   # id em `memories`, se a descrição foi linkada
    new_person:   bool = False           # True se a pessoa foi criada agora

class VoiceDictReadRequest(BaseModel):
    embedding: list[float]        # embedding do locutor consultado
    top_k:     int   = VO_TOP_K
    min_score: float = VO_MIN_SCORE
    include_related: bool = True   # NEW: expansão por grafo Hebbiano

class VoiceCandidate(BaseModel):
    person_id:    int
    person_name:  str
    description:  str
    score:        float
    confidence:   float
    access_count: int
    memory_id:    Optional[int] = None

class VoiceDictReadResponse(BaseModel):
    results:   list[VoiceCandidate]
    ambiguous: bool     # True → nenhum candidato confiável / candidatos muito próximos
    related:   list[RelatedMemory] = []   # NEW

class VoiceDictEntry(BaseModel):
    person_id:      int
    person_name:    str
    description:    str
    source:         str
    confidence:     float
    memory_id:      Optional[int] = None
    examples_count: int
    created_at:     float
    access_count:   int

class VoiceDictUpdateRequest(BaseModel):
    description: str


# ── NEW: Modelos — retrieval de tools (tool selection dinâmica) ───────────────

class ToolRegisterRequest(BaseModel):
    name:        str
    description: str
    # queries reais/representativas que deveriam disparar esta tool. Cada
    # exemplo vira um vetor PRÓPRIO (max-pooling por tool na busca) — evita o
    # problema de descrições genéricas parecidas demais entre tools.
    examples:    list[str] = []
    # tools "core" entram em TODA seleção, independente do score
    core:        bool = False

class ToolRegisterBatchRequest(BaseModel):
    tools: list[ToolRegisterRequest]
    # remove do índice as tools que não estão nesta lista (sincroniza com o
    # registry do orquestrador no startup)
    prune_missing: bool = False

class ToolRegisterResponse(BaseModel):
    name:    str
    tool_id: int
    status:  Literal["created", "updated", "unchanged"]
    vectors: int

class ToolRegisterBatchResponse(BaseModel):
    results: list[ToolRegisterResponse]
    pruned:  list[str] = []

class ToolSelectRequest(BaseModel):
    query:           str
    top_k:           int   = TOOLS_TOP_K
    min_score:       float = TOOLS_MIN_SCORE
    include_related: bool  = True

class ToolMatch(BaseModel):
    name:        str
    description: str
    score:       float
    match_type:  Literal["primary", "related", "core"]

class ToolSelectResponse(BaseModel):
    tools:          list[ToolMatch]
    top1_score:     float
    low_confidence: bool

class ToolUsageRequest(BaseModel):
    # tools realmente chamadas para atender `query`, NA ORDEM da chamada
    tools_used: list[str]
    query:      str = ""


# ── MODIFIED: EmbeddingEngine now delegates to EmbeddingClient ────────────────

class EmbeddingEngine:
    """
    Motor de embeddings via ONNX Serving API.

    IMPORTANTE (E5): toda chamada precisa dizer se o texto é uma QUERY (algo
    que vai ser usado para *buscar*) ou uma PASSAGE (algo que vai ser
    *armazenado* e depois encontrado por uma query). Os métodos genéricos
    antigos (embed/embed_one/embed_batch_two) foram removidos de propósito —
    cada call site abaixo precisa escolher explicitamente `_query`/`_passage`,
    pra não reintroduzir por engano uma chamada sem prefixo.
    """

    def __init__(self, base_url: str = ONNX_SERVING_URL):
        self._client = EmbeddingClient(base_url=base_url)
        log.info(f"EmbeddingEngine carregado — via API: {base_url}")

    @staticmethod
    def _prefixed(texts: list[str], prefix: str) -> list[str]:
        return [f"{prefix}{t}" for t in texts]

    # ── lado "documento" (o que fica gravado/indexado) ──────────────────────

    async def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        return await self._client.embed(self._prefixed(texts, EMBED_PASSAGE_PREFIX))

    async def embed_passage_one(self, text: str) -> np.ndarray:
        result = await self.embed_passages([text])
        return result[0]

    # ── lado "busca" (o que vai comparar contra o índice) ───────────────────

    async def embed_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        return await self._client.embed(self._prefixed(texts, EMBED_QUERY_PREFIX))

    async def embed_query_one(self, text: str) -> np.ndarray:
        result = await self.embed_queries([text])
        return result[0]

    async def embed_query_two(self, text_a: str, text_b: str) -> tuple[np.ndarray, np.ndarray]:
        results = await self.embed_queries([text_a, text_b])
        return results[0], results[1]

    @property
    def client(self) -> EmbeddingClient:
        return self._client


class RerankEngine:
    """
    Motor de rerank via ONNX Serving API (/v1/score) — usado para CONFIRMAR
    duplicata/atualização quando o bi-encoder (cosine) encontra um candidato
    na faixa "parecido", em vez de decidir só pelo cosine bruto (que sofre de
    anisotropia em modelos multilíngues pequenos como o multilingual-e5-small
    — ver comentário perto de DEDUP_THRESHOLD/UPDATE_SIM_THRESHOLD).
    """

    def __init__(self, base_url: str = ONNX_SERVING_URL):
        self._client = RerankerClient(base_url=base_url)

    async def score_one(self, text_a: str, text_b: str) -> float:
        """Score cross-encoder sigmoid [0,1] de quão relacionados text_a e
        text_b são — não confundir com a cosine do bi-encoder."""
        scores = await self._client.score(text_a, [text_b])
        return float(scores[0]) if len(scores) else 0.0

    @property
    def client(self) -> RerankerClient:
        return self._client


# ── NEW: grafo de arestas tipadas (mixin reutilizado por MemoryDB e ToolsDB) ───
#
# Tabela de arestas com PK (memory_id_a, memory_id_b, edge_type). Tipos
# SIMÉTRICOS guardam o par normalizado (min, max); tipos DIRIGIDOS (ver
# EDGE_DIRECTED_TYPES) guardam origem em memory_id_a e destino em memory_id_b.
# Só `co_activation` é reforçado pela ativação do PPR e só os tipos fora de
# EDGE_TYPES_NO_DECAY sofrem decay/poda.

class EdgeGraphMixin:
    EDGE_TABLE = "memory_edges"     # sobrescrito em ToolsDB

    def _edge_table_ddl(self) -> str:
        return f"""
            CREATE TABLE IF NOT EXISTS {self.EDGE_TABLE} (
                memory_id_a        INTEGER NOT NULL,
                memory_id_b        INTEGER NOT NULL,
                edge_type          TEXT    NOT NULL DEFAULT 'co_activation',
                weight             REAL    NOT NULL DEFAULT 0.0,
                coactivation_count INTEGER NOT NULL DEFAULT 0,
                last_coactivated   REAL    NOT NULL,
                PRIMARY KEY (memory_id_a, memory_id_b, edge_type)
            )"""

    def _create_edge_table(self):
        T = self.EDGE_TABLE
        cols = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({T})")}
        if cols and "edge_type" not in cols:
            # Migração one-shot: o schema antigo tinha PK (a, b) e nenhum tipo.
            # SQLite não altera PK — recria a tabela; toda aresta antiga vira
            # co_activation (que era a única semântica que existia).
            self._conn.execute("BEGIN")
            try:
                self._conn.execute("DROP INDEX IF EXISTS idx_edges_a")
                self._conn.execute("DROP INDEX IF EXISTS idx_edges_b")
                self._conn.execute(f"ALTER TABLE {T} RENAME TO {T}_legacy")
                self._conn.execute(self._edge_table_ddl())
                self._conn.execute(
                    f"INSERT INTO {T} (memory_id_a, memory_id_b, edge_type, weight, "
                    f"coactivation_count, last_coactivated) "
                    f"SELECT memory_id_a, memory_id_b, 'co_activation', weight, "
                    f"coactivation_count, last_coactivated FROM {T}_legacy"
                )
                self._conn.execute(f"DROP TABLE {T}_legacy")
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            log.info(f"Grafo [{T}]: schema migrado p/ arestas tipadas (legado → co_activation)")
        else:
            self._conn.execute(self._edge_table_ddl())
        self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{T}_a ON {T}(memory_id_a)")
        self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{T}_b ON {T}(memory_id_b)")

    def get_neighbors(self, memory_id: int, limit: int) -> list[sqlite3.Row]:
        """Arestas de um nó (dos dois lados), ordenadas por peso. Cada linha
        traz `edge_type` — quem consome decide a direção efetiva."""
        return self._conn.execute(
            f"SELECT * FROM {self.EDGE_TABLE} WHERE memory_id_a = ? OR memory_id_b = ? "
            f"ORDER BY weight DESC LIMIT ?",
            (memory_id, memory_id, limit),
        ).fetchall()

    def get_edges_for_ids(self, ids: list[int]) -> list[sqlite3.Row]:
        """Todas as arestas com QUALQUER extremidade em `ids`."""
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        return self._conn.execute(
            f"SELECT * FROM {self.EDGE_TABLE} "
            f"WHERE memory_id_a IN ({ph}) OR memory_id_b IN ({ph})",
            list(ids) + list(ids),
        ).fetchall()

    def upsert_edge(
        self, id_a: int, id_b: int, lr: float, now: float,
        edge_type: str = EDGE_CO_ACTIVATION,
        initial_weight: Optional[float] = None,
    ):
        """Cria ou reforça a aresta (id_a, id_b, edge_type). Tipos simétricos
        normalizam a ordem (min/max); tipos dirigidos preservam id_a → id_b.
        Crescimento saturante: w_new = w_old + lr * (1 - w_old)."""
        if id_a == id_b:
            return
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"edge_type inválido: {edge_type!r}")
        if edge_type in EDGE_DIRECTED_TYPES:
            a, b = id_a, id_b
        else:
            a, b = min(id_a, id_b), max(id_a, id_b)
        if initial_weight is None:
            initial_weight = EDGE_EXPLICIT_LINK_WEIGHT if edge_type in EDGE_EXPLICIT_TYPES else lr
        T = self.EDGE_TABLE
        with self._lock:
            row = self._conn.execute(
                f"SELECT weight FROM {T} WHERE memory_id_a = ? AND memory_id_b = ? AND edge_type = ?",
                (a, b, edge_type),
            ).fetchone()
            if row is None:
                self._conn.execute(
                    f"INSERT INTO {T} (memory_id_a, memory_id_b, edge_type, weight, "
                    f"coactivation_count, last_coactivated) VALUES (?, ?, ?, ?, 1, ?)",
                    (a, b, edge_type, initial_weight, now),
                )
            else:
                w_old = row["weight"]
                w_new = w_old + lr * (1.0 - w_old)
                self._conn.execute(
                    f"UPDATE {T} SET weight = ?, coactivation_count = coactivation_count + 1, "
                    f"last_coactivated = ? WHERE memory_id_a = ? AND memory_id_b = ? AND edge_type = ?",
                    (w_new, now, a, b, edge_type),
                )

    def reinforce_edges_batch(
        self,
        pairs_with_activation: list[tuple[int, int, float]],
        lr: float,
        min_activation: float,
        now: float,
    ) -> int:
        """Reforço em lote (Step 8 / Opção B) — SÓ atualiza arestas
        `co_activation` que já existem; criar aresta nova é trabalho do Step 4,
        e reforçar temporal/contradicts/updates por co-ativação seria tratar
        correção como associação. `pairs_with_activation` =
        [(id_a, id_b, activation)] com activation = ppr_rank[i] * ppr_rank[j].
        Atualização: w += lr * (activation * w) * (1 - w). Uma transação."""
        if not pairs_with_activation:
            return 0
        rows = [
            (lr, act, now, min(a, b), max(a, b), act, min_activation)
            for a, b, act in pairs_with_activation
            if act >= min_activation
        ]
        if not rows:
            return 0
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.executemany(
                    f"UPDATE {self.EDGE_TABLE} SET "
                    f"weight = MIN(1.0, weight + ? * (? * weight) * (1.0 - weight)), "
                    f"coactivation_count = coactivation_count + 1, "
                    f"last_coactivated = ? "
                    f"WHERE memory_id_a = ? AND memory_id_b = ? AND edge_type = 'co_activation' "
                    f"AND ? >= ?",
                    rows,
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return len(rows)

    def decay_and_prune_edges(self, half_life_days: float, prune_threshold: float) -> int:
        """Decay das arestas (mesma matemática de apply_decay) e poda das que
        caem abaixo de prune_threshold. Tipos em EDGE_TYPES_NO_DECAY ficam de
        fora dos dois passos. Retorna quantas foram podadas."""
        now = time.time()
        T = self.EDGE_TABLE
        skip = tuple(EDGE_TYPES_NO_DECAY)
        ph = ",".join("?" * len(skip)) or "''"
        with self._lock:
            rows = self._conn.execute(
                f"SELECT memory_id_a, memory_id_b, edge_type, weight, last_coactivated "
                f"FROM {T} WHERE edge_type NOT IN ({ph})", skip,
            ).fetchall()
            updates = []
            for row in rows:
                days_idle    = (now - row["last_coactivated"]) / 86400.0
                decay_factor = 0.5 ** (days_idle / half_life_days)
                updates.append((
                    row["weight"] * decay_factor,
                    row["memory_id_a"], row["memory_id_b"], row["edge_type"],
                ))
            if updates:
                self._conn.executemany(
                    f"UPDATE {T} SET weight = ? "
                    f"WHERE memory_id_a = ? AND memory_id_b = ? AND edge_type = ?",
                    updates,
                )
            cur = self._conn.execute(
                f"DELETE FROM {T} WHERE weight < ? AND edge_type NOT IN ({ph})",
                (prune_threshold, *skip),
            )
            return cur.rowcount

    def delete_edges_for_memory(self, memory_id: int):
        """Remove todas as arestas de um nó (de qualquer tipo) — o grafo nunca
        mantém aresta apontando pra linha inexistente."""
        with self._lock:
            self._conn.execute(
                f"DELETE FROM {self.EDGE_TABLE} WHERE memory_id_a = ? OR memory_id_b = ?",
                (memory_id, memory_id),
            )

    def count_edges(self) -> int:
        return self._conn.execute(f"SELECT COUNT(*) FROM {self.EDGE_TABLE}").fetchone()[0]


# ── Banco de dados de longo prazo ──────────────────────────────────────────────

class MemoryDB(EdgeGraphMixin):
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # ITEM 4 da revisão: esta conexão é usada tanto pelo event loop quanto
        # por threads do executor (endpoints e jobs em background). O módulo
        # sqlite3 não serializa automaticamente chamadas concorrentes vindas
        # de threads diferentes sobre a MESMA conexão — este lock garante que
        # cada statement de escrita rode de forma atômica em relação aos
        # outros.
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                text          TEXT    NOT NULL,
                text_hash     TEXT    NOT NULL UNIQUE,
                source        TEXT    NOT NULL DEFAULT 'chat',
                confidence    REAL    NOT NULL DEFAULT 1.0,
                created_at    REAL    NOT NULL,
                last_accessed REAL    NOT NULL,
                access_count  INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_confidence    ON memories(confidence);
            CREATE INDEX IF NOT EXISTS idx_last_accessed ON memories(last_accessed);
        """)
        # ── NEW: grafo Hebbiano com arestas tipadas (cria/migra memory_edges)
        self._create_edge_table()
        # Migração leve: bancos criados antes do campo "esquecível?" existir
        # não têm a coluna — adiciona com default 1 (esquecível, comportamento
        # anterior) sem quebrar instalações já em uso.
        cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(memories)")}
        if "forgettable" not in cols:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN forgettable INTEGER NOT NULL DEFAULT 1"
            )
        # ── NEW (solução 3): mesma migração leve de sempre — bancos criados
        # antes de ttl_days existir não têm a coluna; adiciona como NULL
        # (== "usar o default global DECAY_HALF_LIFE_DAYS"), sem quebrar
        # instalações já em uso.
        if "ttl_days" not in cols:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN ttl_days REAL DEFAULT NULL"
            )

    def insert(
        self,
        text: str,
        source: str,
        confidence: float,
        forgettable: bool = True,
        ttl_days: Optional[float] = None,
    ) -> int:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO memories (text, text_hash, source, confidence, forgettable, "
                "ttl_days, created_at, last_accessed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (text, text_hash, source, confidence, int(forgettable), ttl_days, now, now),
            )
            return cur.lastrowid

    def exists_exact(self, text: str) -> bool:
        h = hashlib.sha256(text.encode()).hexdigest()
        return self._conn.execute(
            "SELECT 1 FROM memories WHERE text_hash = ?", (h,)
        ).fetchone() is not None

    def get_by_id(self, memory_id: int) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()

    def get_by_ids(self, ids: list[int]) -> list[sqlite3.Row]:
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        return self._conn.execute(
            f"SELECT * FROM memories WHERE id IN ({ph})", ids
        ).fetchall()

    def update_access(self, memory_id: int):
        try:
            with self._lock:
                # ── NEW (LTP): reforço saturante de confiança a cada
                # acesso — "use it and improve it". Compõe com o decay:
                # o decay multiplica o que quer que a confiança seja; o
                # reforço puxa pra cima no acesso, o decay puxa pra baixo
                # com o tempo. Caminho único — nenhum outro lugar incrementa
                # access_count de LT.
                self._conn.execute(
                    "UPDATE memories SET access_count = access_count + 1, last_accessed = ?, "
                    "confidence = confidence + ? * (1 - confidence) WHERE id = ?",
                    (time.time(), LTP_ACCESS_BOOST, memory_id),
                )
        except sqlite3.OperationalError:
            pass

    def update(
        self,
        memory_id: int,
        text: str,
        source: Optional[str]      = None,
        confidence: Optional[float] = None,
        forgettable: Optional[bool] = None,
        ttl_days: Optional[float]   = None,
        ttl_days_set: bool          = False,
    ) -> bool:
        """
        Atualiza o texto (e opcionalmente source/confidence/forgettable/
        ttl_days) de uma memória já existente, em vez de inserir uma nova
        — usado pelo fluxo action="update" (solução 2 da revisão), quando
        o extrator identifica que um texto novo é uma correção de um fato
        já gravado, e não um fato adicional.

        `text_hash` é recalculado a partir do novo texto para que a
        dedup exata (`exists_exact`) continue funcionando corretamente
        depois da correção. `created_at` não é alterado — só o conteúdo,
        a confiança e o "relógio" de acesso.

        `ttl_days_set` distingue "não mexer no ttl_days atual" (default)
        de "setar ttl_days para None explicitamente" (chamador passou
        ttl_days=None de propósito) — sem isso não daria pra diferenciar
        os dois casos só olhando `ttl_days is None`.
        """
        row = self.get_by_id(memory_id)
        if row is None:
            return False

        new_text        = text.strip()
        new_text_hash   = hashlib.sha256(new_text.encode()).hexdigest()
        new_source      = source if source is not None else row["source"]
        new_confidence  = confidence if confidence is not None else row["confidence"]
        new_forgettable = int(forgettable) if forgettable is not None else row["forgettable"]
        new_ttl_days    = ttl_days if ttl_days_set else row["ttl_days"]
        now = time.time()

        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE memories SET text = ?, text_hash = ?, source = ?, "
                    "confidence = ?, forgettable = ?, ttl_days = ?, last_accessed = ? "
                    "WHERE id = ?",
                    (
                        new_text, new_text_hash, new_source, new_confidence,
                        new_forgettable, new_ttl_days, now, memory_id,
                    ),
                )
            except sqlite3.IntegrityError:
                # O texto novo já é idêntico (mesmo text_hash) a OUTRA
                # memória existente — não faz sentido duplicar o hash.
                return False
        return True

    def apply_decay(self, half_life_days: float):
        now  = time.time()
        with self._lock:
            # forgettable = 0 → memória marcada como "não esquecível":
            # fica de fora do decay por completo, mesmo que fique muito
            # tempo sem ser acessada.
            rows = self._conn.execute(
                "SELECT id, confidence, last_accessed, ttl_days FROM memories "
                "WHERE confidence > 0.01 AND forgettable = 1"
            ).fetchall()
            updates = []
            for row in rows:
                # ── NEW (solução 3): ttl_days por linha tem prioridade
                # sobre o half_life_days global quando presente (> 0);
                # cai para o default global quando ausente/NULL.
                row_ttl = row["ttl_days"]
                effective_half_life = row_ttl if (row_ttl is not None and row_ttl > 0) else half_life_days
                days_idle    = (now - row["last_accessed"]) / 86400.0
                decay_factor = 0.5 ** (days_idle / effective_half_life)
                updates.append((row["confidence"] * decay_factor, row["id"]))
            if updates:
                self._conn.executemany("UPDATE memories SET confidence = ? WHERE id = ?", updates)
                # ── NEW (Hebbian graph): antes de apagar as memórias
                # expiradas, apaga as arestas que apontam pra elas — o
                # grafo nunca mantém aresta pra linha morta.
                expired = self._conn.execute(
                    "SELECT id FROM memories WHERE confidence < 0.01"
                ).fetchall()
                for expired_row in expired:
                    self._conn.execute(
                        "DELETE FROM memory_edges WHERE memory_id_a = ? OR memory_id_b = ?",
                        (expired_row["id"], expired_row["id"]),
                    )
                self._conn.execute("DELETE FROM memories WHERE confidence < 0.01")
                log.info(f"Decay aplicado em {len(updates)} memórias de longo prazo")

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]


# ── Banco de dados de curto prazo ──────────────────────────────────────────────

class ShortTermDB:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()  # ITEM 4 — ver comentário em MemoryDB
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS turn_groups (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT    NOT NULL,
                turns_json    TEXT    NOT NULL,
                embed_text    TEXT    NOT NULL,
                created_at    REAL    NOT NULL,
                last_accessed REAL    NOT NULL,
                access_count  INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_st_session  ON turn_groups(session_id);
            CREATE INDEX IF NOT EXISTS idx_st_accessed ON turn_groups(last_accessed);
        """)

    def insert(self, session_id: str, turns: list[Turn], embed_text: str) -> int:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO turn_groups (session_id, turns_json, embed_text, created_at, last_accessed) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, json.dumps([t.model_dump() for t in turns]), embed_text, now, now),
            )
            return cur.lastrowid

    def get_by_ids(self, ids: list[int]) -> list[sqlite3.Row]:
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        return self._conn.execute(
            f"SELECT * FROM turn_groups WHERE id IN ({ph})", ids
        ).fetchall()

    def get_recent_turns_text(self, session_id: str, n_groups: int) -> list[str]:
        rows = self._conn.execute(
            "SELECT turns_json FROM turn_groups "
            "WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, n_groups),
        ).fetchall()
        lines: list[str] = []
        for row in reversed(rows):
            for t in json.loads(row["turns_json"]):
                lines.append(f"{t['role']}: {t['content']}")
        return lines

    def get_recent_turn_groups(self, session_id: str, n_groups: int) -> tuple[list[dict], int]:
        """
        Leitura crua (sem busca semântica) das N duplas pergunta-resposta mais
        recentes de uma sessão, em ordem cronológica (mais antiga primeiro).
        Usada pelo /read_st — contexto de conversa direto, sem embeddings.
        """
        rows = self._conn.execute(
            "SELECT turns_json FROM turn_groups "
            "WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, n_groups),
        ).fetchall()
        turns: list[dict] = []
        for row in reversed(rows):
            turns.extend(json.loads(row["turns_json"]))
        return turns, len(rows)

    def update_access(self, group_id: int):
        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE turn_groups SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                    (time.time(), group_id),
                )
        except sqlite3.OperationalError:
            pass

    def expire_old(self, ttl_hours: float) -> int:
        cutoff = time.time() - ttl_hours * 3600
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM turn_groups WHERE last_accessed < ?", (cutoff,)
            )
            removed = cur.rowcount
        if removed:
            log.info(f"Curto prazo: {removed} grupos expirados (TTL={ttl_hours}h)")
        return removed

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM turn_groups").fetchone()[0]

    def get_ids_by_session(self, session_id: str) -> list[int]:
        rows = self._conn.execute(
            "SELECT id FROM turn_groups WHERE session_id = ?", (session_id,)
        ).fetchall()
        return [row["id"] for row in rows]

    def delete_by_session(self, session_id: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM turn_groups WHERE session_id = ?", (session_id,)
            )
            return cur.rowcount


# ── NEW: Banco de dados de arquivos indexados ──────────────────────────────────

class IndexedFilesDB:
    """
    Armazena o conteúdo COMPLETO dos arquivos indexados pelo local-scraping,
    com hash para detectar mudanças e chunks para busca semântica via FAISS.
    """

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()  # ITEM 4 — ver comentário em MemoryDB
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS indexed_files (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path     TEXT    NOT NULL UNIQUE,
                file_name     TEXT    NOT NULL,
                extension     TEXT    NOT NULL DEFAULT '',
                content       TEXT    NOT NULL,
                content_hash  TEXT    NOT NULL,
                file_hash     TEXT    NOT NULL DEFAULT '',
                size          INTEGER NOT NULL DEFAULT 0,
                modified      TEXT    NOT NULL DEFAULT '',
                source        TEXT    NOT NULL DEFAULT 'local_scraping',
                confidence    REAL    NOT NULL DEFAULT 1.0,
                created_at    REAL    NOT NULL,
                last_accessed REAL    NOT NULL,
                access_count  INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_if_path      ON indexed_files(file_path);
            CREATE INDEX IF NOT EXISTS idx_if_file_hash ON indexed_files(file_hash);

            CREATE TABLE IF NOT EXISTS indexed_file_chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id     INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text  TEXT    NOT NULL,
                char_start  INTEGER NOT NULL DEFAULT 0,
                char_end    INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (file_id) REFERENCES indexed_files(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_ifc_file_id ON indexed_file_chunks(file_id);
        """)

    # ── File operations ──

    def insert_file(
        self,
        file_path:   str,
        file_name:   str,
        extension:   str,
        content:     str,
        content_hash: str,
        file_hash:   str,
        size:        int,
        modified:    str,
        source:      str,
        confidence:  float,
    ) -> int:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO indexed_files "
                "(file_path, file_name, extension, content, content_hash, file_hash, "
                "size, modified, source, confidence, created_at, last_accessed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (file_path, file_name, extension, content, content_hash, file_hash,
                 size, modified, source, confidence, now, now),
            )
            return cur.lastrowid

    def update_file(
        self,
        file_id:     int,
        content:     str,
        content_hash: str,
        file_hash:   str,
        size:        int,
        modified:    str,
    ):
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE indexed_files SET content=?, content_hash=?, file_hash=?, "
                "size=?, modified=?, last_accessed=? WHERE id=?",
                (content, content_hash, file_hash, size, modified, now, file_id),
            )

    def get_by_path(self, file_path: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM indexed_files WHERE file_path = ?", (file_path,)
        ).fetchone()

    def get_by_id(self, file_id: int) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM indexed_files WHERE id = ?", (file_id,)
        ).fetchone()

    def update_access(self, file_id: int):
        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE indexed_files SET access_count = access_count + 1, "
                    "last_accessed = ? WHERE id = ?",
                    (time.time(), file_id),
                )
        except sqlite3.OperationalError:
            pass

    def delete_file(self, file_id: int) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM indexed_files WHERE id = ?", (file_id,))
            return cur.rowcount

    def count_files(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()[0]

    def list_files(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT id, file_path, file_name, extension, content_hash, file_hash, "
            "size, modified, created_at, access_count FROM indexed_files "
            "ORDER BY last_accessed DESC"
        ).fetchall()

    # ── Chunk operations ──

    def insert_chunks(self, file_id: int, chunks: list[dict]):
        """Insert multiple chunks for a file. chunks = [{text, index, char_start, char_end}]"""
        rows = [
            (file_id, c["index"], c["text"], c["char_start"], c["char_end"])
            for c in chunks
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT INTO indexed_file_chunks (file_id, chunk_index, chunk_text, char_start, char_end) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def get_chunks_by_file(self, file_id: int) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM indexed_file_chunks WHERE file_id = ? ORDER BY chunk_index",
            (file_id,),
        ).fetchall()

    def get_chunk_ids_by_file(self, file_id: int) -> list[int]:
        rows = self._conn.execute(
            "SELECT id FROM indexed_file_chunks WHERE file_id = ?", (file_id,)
        ).fetchall()
        return [row["id"] for row in rows]

    def count_chunks(self, file_id: Optional[int] = None) -> int:
        if file_id:
            return self._conn.execute(
                "SELECT COUNT(*) FROM indexed_file_chunks WHERE file_id = ?", (file_id,)
            ).fetchone()[0]
        return self._conn.execute(
            "SELECT COUNT(*) FROM indexed_file_chunks"
        ).fetchone()[0]

    def delete_chunks_by_file(self, file_id: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM indexed_file_chunks WHERE file_id = ?", (file_id,)
            )
            return cur.rowcount

    def get_chunks_by_ids(self, chunk_ids: list[int]) -> list[sqlite3.Row]:
        if not chunk_ids:
            return []
        ph = ",".join("?" * len(chunk_ids))
        return self._conn.execute(
            f"SELECT * FROM indexed_file_chunks WHERE id IN ({ph})", chunk_ids
        ).fetchall()

    def get_total_chunks(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM indexed_file_chunks").fetchone()[0]


# ── NEW: Banco de dados do Dicionário Visual ───────────────────────────────────

class VisualDictDB:
    """
    Persiste os "conceitos visuais" do módulo vision.py: um conceito é um
    objeto/tipo de objeto (ex.: "caneca azul", "controle remoto da TV"),
    com uma descrição textual e N embeddings de exemplo (um por crop
    registrado — permite reconhecer o mesmo conceito em ângulos/condições
    de luz diferentes).

    Os embeddings em si moram no FAISS (`vd_index`); aqui só ficam o texto
    e o mapeamento embedding_id → concept_id.
    """

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()  # ITEM 4 — ver comentário em MemoryDB
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS visual_concepts (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_name  TEXT    NOT NULL,
                concept_key   TEXT    NOT NULL UNIQUE,   -- nome normalizado (lower/strip)
                description   TEXT    NOT NULL,
                source        TEXT    NOT NULL DEFAULT 'vision_pipeline',
                confidence    REAL    NOT NULL DEFAULT 1.0,
                memory_id     INTEGER,                    -- FK lógica p/ memories.id
                created_at    REAL    NOT NULL,
                last_accessed REAL    NOT NULL,
                access_count  INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_vc_key ON visual_concepts(concept_key);

            CREATE TABLE IF NOT EXISTS visual_concept_embeddings (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_id INTEGER NOT NULL,
                created_at REAL    NOT NULL,
                FOREIGN KEY (concept_id) REFERENCES visual_concepts(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_vce_concept ON visual_concept_embeddings(concept_id);
        """)

    @staticmethod
    def _normalize_key(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip().lower())

    # ── Concept operations ──

    def insert_concept(
        self, concept_name: str, description: str, source: str,
        confidence: float, memory_id: Optional[int],
    ) -> int:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO visual_concepts "
                "(concept_name, concept_key, description, source, confidence, memory_id, "
                "created_at, last_accessed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (concept_name, self._normalize_key(concept_name), description, source,
                 confidence, memory_id, now, now),
            )
            return cur.lastrowid

    def get_by_name(self, concept_name: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM visual_concepts WHERE concept_key = ?",
            (self._normalize_key(concept_name),),
        ).fetchone()

    def get_concept_by_id(self, concept_id: int) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM visual_concepts WHERE id = ?", (concept_id,)
        ).fetchone()

    def update_access(self, concept_id: int):
        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE visual_concepts SET access_count = access_count + 1, "
                    "last_accessed = ? WHERE id = ?",
                    (time.time(), concept_id),
                )
        except sqlite3.OperationalError:
            pass

    def delete_concept(self, concept_id: int) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM visual_concepts WHERE id = ?", (concept_id,))
            return cur.rowcount

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM visual_concepts").fetchone()[0]

    def list_concepts(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT c.*, "
            "(SELECT COUNT(*) FROM visual_concept_embeddings e WHERE e.concept_id = c.id) "
            "AS examples_count "
            "FROM visual_concepts c ORDER BY c.last_accessed DESC"
        ).fetchall()

    # ── Embedding-row operations (mapeamento embedding_id → concept_id) ──

    def insert_embedding(self, concept_id: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO visual_concept_embeddings (concept_id, created_at) VALUES (?, ?)",
                (concept_id, time.time()),
            )
            return cur.lastrowid

    def get_concept_id_by_embedding(self, embedding_id: int) -> Optional[int]:
        row = self._conn.execute(
            "SELECT concept_id FROM visual_concept_embeddings WHERE id = ?",
            (embedding_id,),
        ).fetchone()
        return row["concept_id"] if row else None

    def get_embedding_ids_by_concept(self, concept_id: int) -> list[int]:
        rows = self._conn.execute(
            "SELECT id FROM visual_concept_embeddings WHERE concept_id = ?", (concept_id,)
        ).fetchall()
        return [r["id"] for r in rows]

    def count_examples(self, concept_id: int) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM visual_concept_embeddings WHERE concept_id = ?",
            (concept_id,),
        ).fetchone()[0]


# ── NEW: Dicionário de Rostos ──────────────────────────────────────────────────

class FaceDictDB:
    """
    Persiste as pessoas cadastradas pro reconhecimento facial: uma pessoa
    tem N embeddings de exemplo (um por rosto registrado — permite
    reconhecer o mesmo rosto em ângulos/luz diferentes).

    Os embeddings em si moram no LanceDB (`fd_index`); aqui só ficam o nome
    e o mapeamento embedding_id → person_id. `memory_id` é OPCIONAL (default
    NULL): só é preenchido quando o cadastro pede link_to_memory — reconhecimento
    facial não precisa, por padrão, de descrição gravada na memória geral, mas o
    link dá ao rosto um nó no grafo Hebbiano.
    """

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()  # ITEM 4 — ver comentário em MemoryDB
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS face_people (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name   TEXT    NOT NULL,
                person_key    TEXT    NOT NULL UNIQUE,   -- nome normalizado (lower/strip)
                description   TEXT    NOT NULL DEFAULT '',  -- quem é a pessoa (relação/contexto)
                source        TEXT    NOT NULL DEFAULT 'vision_pipeline',
                confidence    REAL    NOT NULL DEFAULT 1.0,
                created_at    REAL    NOT NULL,
                last_accessed REAL    NOT NULL,
                access_count  INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_fp_key ON face_people(person_key);

            CREATE TABLE IF NOT EXISTS face_embeddings (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id  INTEGER NOT NULL,
                created_at REAL    NOT NULL,
                FOREIGN KEY (person_id) REFERENCES face_people(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_fe_person ON face_embeddings(person_id);
        """)
        # Migração leve: bancos criados antes deste campo existir não têm a
        # coluna `description` — adiciona se estiver faltando, sem quebrar
        # instalações já em uso.
        cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(face_people)")}
        if "description" not in cols:
            self._conn.execute("ALTER TABLE face_people ADD COLUMN description TEXT NOT NULL DEFAULT ''")
        # NEW: link opcional com a memória de longo prazo (nó no grafo Hebbiano)
        if "memory_id" not in cols:
            self._conn.execute("ALTER TABLE face_people ADD COLUMN memory_id INTEGER DEFAULT NULL")

    @staticmethod
    def _normalize_key(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip().lower())

    # ── Person operations ──

    def insert_person(
        self, person_name: str, description: str, source: str, confidence: float,
        memory_id: Optional[int] = None,
    ) -> int:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO face_people "
                "(person_name, person_key, description, source, confidence, memory_id, "
                "created_at, last_accessed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (person_name, self._normalize_key(person_name), description, source,
                 confidence, memory_id, now, now),
            )
            return cur.lastrowid

    def set_memory_id(self, person_id: int, memory_id: Optional[int]):
        with self._lock:
            self._conn.execute(
                "UPDATE face_people SET memory_id = ? WHERE id = ?", (memory_id, person_id)
            )

    def get_by_name(self, person_name: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM face_people WHERE person_key = ?",
            (self._normalize_key(person_name),),
        ).fetchone()

    def get_person_by_id(self, person_id: int) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM face_people WHERE id = ?", (person_id,)
        ).fetchone()

    def update_description(self, person_id: int, description: str):
        """Atualiza/edita a descrição de uma pessoa já cadastrada. Chamado
        quando `person_name` já existe e um `description` não-vazio é
        enviado num novo /face-dict/write (ex.: cadastrando mais um
        exemplo de rosto e aproveitando pra corrigir/completar o texto)."""
        with self._lock:
            self._conn.execute(
                "UPDATE face_people SET description = ? WHERE id = ?",
                (description, person_id),
            )

    def update_access(self, person_id: int):
        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE face_people SET access_count = access_count + 1, "
                    "last_accessed = ? WHERE id = ?",
                    (time.time(), person_id),
                )
        except sqlite3.OperationalError:
            pass

    def delete_person(self, person_id: int) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM face_people WHERE id = ?", (person_id,))
            return cur.rowcount

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM face_people").fetchone()[0]

    def list_people(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT p.*, "
            "(SELECT COUNT(*) FROM face_embeddings e WHERE e.person_id = p.id) "
            "AS examples_count "
            "FROM face_people p ORDER BY p.last_accessed DESC"
        ).fetchall()

    # ── Embedding-row operations (mapeamento embedding_id → person_id) ──

    def insert_embedding(self, person_id: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO face_embeddings (person_id, created_at) VALUES (?, ?)",
                (person_id, time.time()),
            )
            return cur.lastrowid

    def get_person_id_by_embedding(self, embedding_id: int) -> Optional[int]:
        row = self._conn.execute(
            "SELECT person_id FROM face_embeddings WHERE id = ?",
            (embedding_id,),
        ).fetchone()
        return row["person_id"] if row else None

    def get_embedding_ids_by_person(self, person_id: int) -> list[int]:
        rows = self._conn.execute(
            "SELECT id FROM face_embeddings WHERE person_id = ?", (person_id,)
        ).fetchall()
        return [r["id"] for r in rows]

    def count_examples(self, person_id: int) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM face_embeddings WHERE person_id = ?",
            (person_id,),
        ).fetchone()[0]


# ── NEW: Dicionário de Voz ─────────────────────────────────────────────────────

class VoiceDictDB:
    """
    Persiste as pessoas cadastradas pela identificação de locutor: uma
    pessoa tem N embeddings de exemplo (um por frase falada registrada —
    permite reconhecer a mesma voz em condições de captação diferentes).

    Os embeddings em si moram no FAISS (`vo_index`); aqui ficam o nome, a
    descrição, o `memory_id` (link para a memória de longo prazo onde a
    descrição também foi gravada) e o mapeamento embedding_id → person_id.
    Estrutura idêntica à FaceDictDB, com a coluna extra `memory_id`.
    """

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()  # ITEM 4 — ver comentário em MemoryDB
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS voice_people (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name   TEXT    NOT NULL,
                person_key    TEXT    NOT NULL UNIQUE,   -- nome normalizado (lower/strip)
                description   TEXT    NOT NULL DEFAULT '',  -- quem é a pessoa (relação/contexto)
                source        TEXT    NOT NULL DEFAULT 'voice_pipeline',
                confidence    REAL    NOT NULL DEFAULT 1.0,
                memory_id     INTEGER,                    -- FK lógica p/ memories.id
                created_at    REAL    NOT NULL,
                last_accessed REAL    NOT NULL,
                access_count  INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_vp_key ON voice_people(person_key);

            CREATE TABLE IF NOT EXISTS voice_embeddings (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id  INTEGER NOT NULL,
                created_at REAL    NOT NULL,
                FOREIGN KEY (person_id) REFERENCES voice_people(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_ve_person ON voice_embeddings(person_id);
        """)

    @staticmethod
    def _normalize_key(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip().lower())

    # ── Person operations ──

    def insert_person(self, person_name: str, description: str, source: str,
                      confidence: float, memory_id: Optional[int]) -> int:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO voice_people "
                "(person_name, person_key, description, source, confidence, memory_id, "
                "created_at, last_accessed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (person_name, self._normalize_key(person_name), description, source,
                 confidence, memory_id, now, now),
            )
            return cur.lastrowid

    def get_by_name(self, person_name: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM voice_people WHERE person_key = ?",
            (self._normalize_key(person_name),),
        ).fetchone()

    def get_person_by_id(self, person_id: int) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM voice_people WHERE id = ?", (person_id,)
        ).fetchone()

    def update_description(self, person_id: int, description: str):
        """Atualiza/edita a descrição de uma pessoa já cadastrada."""
        with self._lock:
            self._conn.execute(
                "UPDATE voice_people SET description = ? WHERE id = ?",
                (description, person_id),
            )

    def update_access(self, person_id: int):
        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE voice_people SET access_count = access_count + 1, "
                    "last_accessed = ? WHERE id = ?",
                    (time.time(), person_id),
                )
        except sqlite3.OperationalError:
            pass

    def delete_person(self, person_id: int) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM voice_people WHERE id = ?", (person_id,))
            return cur.rowcount

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM voice_people").fetchone()[0]

    def list_people(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT p.*, "
            "(SELECT COUNT(*) FROM voice_embeddings e WHERE e.person_id = p.id) "
            "AS examples_count "
            "FROM voice_people p ORDER BY p.last_accessed DESC"
        ).fetchall()

    # ── Embedding-row operations (mapeamento embedding_id → person_id) ──

    def insert_embedding(self, person_id: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO voice_embeddings (person_id, created_at) VALUES (?, ?)",
                (person_id, time.time()),
            )
            return cur.lastrowid

    def get_person_id_by_embedding(self, embedding_id: int) -> Optional[int]:
        row = self._conn.execute(
            "SELECT person_id FROM voice_embeddings WHERE id = ?",
            (embedding_id,),
        ).fetchone()
        return row["person_id"] if row else None

    def get_embedding_ids_by_person(self, person_id: int) -> list[int]:
        rows = self._conn.execute(
            "SELECT id FROM voice_embeddings WHERE person_id = ?", (person_id,)
        ).fetchall()
        return [r["id"] for r in rows]

    def count_examples(self, person_id: int) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM voice_embeddings WHERE person_id = ?",
            (person_id,),
        ).fetchone()[0]


# ── NEW: Banco de tools (RAG de tool selection) ────────────────────────────────

class ToolsDB(EdgeGraphMixin):
    """
    Registro das tools expostas ao LLM. Cada tool tem N vetores no LanceDB
    (`tl_index`, memory_type="tool"): 1 do "documento" (nome + descrição), 1 por
    exemplo de query fornecido no cadastro (kind="example") e 1 por query real
    aprendida com o uso (kind="learned"). A busca faz max-pooling por tool.

    O grafo de co-uso vive aqui mesmo (tabela `tool_edges`, mesma lógica de
    arestas tipadas do grafo Hebbiano de memórias): temporal_precedence captura
    sequências (listar → ler → editar) e co_activation, tools usadas juntas.
    """
    EDGE_TABLE = "tool_edges"

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS tools (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                name_key    TEXT    NOT NULL UNIQUE,
                description TEXT    NOT NULL,
                is_core     INTEGER NOT NULL DEFAULT 0,
                doc_hash    TEXT    NOT NULL,
                created_at  REAL    NOT NULL,
                last_used   REAL,
                use_count   INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS tool_vectors (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_id    INTEGER NOT NULL,
                kind       TEXT    NOT NULL,      -- doc | example | learned
                text       TEXT    NOT NULL,
                created_at REAL    NOT NULL,
                FOREIGN KEY (tool_id) REFERENCES tools(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_tv_tool ON tool_vectors(tool_id);
        """)
        self._create_edge_table()

    @staticmethod
    def _normalize_key(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip().lower())

    # ── Tools ──
    def insert_tool(self, name: str, description: str, is_core: bool, doc_hash: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO tools (name, name_key, description, is_core, doc_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (name, self._normalize_key(name), description, int(is_core), doc_hash, time.time()),
            )
            return cur.lastrowid

    def update_tool(self, tool_id: int, description: str, is_core: bool, doc_hash: str):
        with self._lock:
            self._conn.execute(
                "UPDATE tools SET description = ?, is_core = ?, doc_hash = ? WHERE id = ?",
                (description, int(is_core), doc_hash, tool_id),
            )

    def set_core(self, tool_id: int, is_core: bool):
        with self._lock:
            self._conn.execute("UPDATE tools SET is_core = ? WHERE id = ?", (int(is_core), tool_id))

    def get_by_name(self, name: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM tools WHERE name_key = ?", (self._normalize_key(name),)
        ).fetchone()

    def get_by_ids(self, ids: list[int]) -> list[sqlite3.Row]:
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        return self._conn.execute(f"SELECT * FROM tools WHERE id IN ({ph})", list(ids)).fetchall()

    def list_tools(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT t.*, (SELECT COUNT(*) FROM tool_vectors v WHERE v.tool_id = t.id) "
            "AS vectors_count FROM tools t ORDER BY t.name"
        ).fetchall()

    def list_core(self) -> list[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM tools WHERE is_core = 1").fetchall()

    def mark_used(self, tool_ids: list[int]):
        if not tool_ids:
            return
        now = time.time()
        with self._lock:
            self._conn.executemany(
                "UPDATE tools SET use_count = use_count + 1, last_used = ? WHERE id = ?",
                [(now, tid) for tid in tool_ids],
            )

    def delete_tool(self, tool_id: int) -> int:
        """Apaga a tool (vetores em cascata) e todas as arestas dela."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM tools WHERE id = ?", (tool_id,))
            self._conn.execute(
                "DELETE FROM tool_edges WHERE memory_id_a = ? OR memory_id_b = ?",
                (tool_id, tool_id),
            )
            return cur.rowcount

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM tools").fetchone()[0]

    # ── Vetores (mapeamento vector_id → tool_id) ──
    def insert_vector(self, tool_id: int, kind: str, text: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO tool_vectors (tool_id, kind, text, created_at) VALUES (?, ?, ?, ?)",
                (tool_id, kind, text, time.time()),
            )
            return cur.lastrowid

    def get_vector_ids(self, tool_id: int, kinds: Optional[tuple[str, ...]] = None) -> list[int]:
        if kinds:
            ph = ",".join("?" * len(kinds))
            rows = self._conn.execute(
                f"SELECT id FROM tool_vectors WHERE tool_id = ? AND kind IN ({ph})",
                (tool_id, *kinds),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id FROM tool_vectors WHERE tool_id = ?", (tool_id,)
            ).fetchall()
        return [r["id"] for r in rows]

    def get_tool_ids_by_vectors(self, vector_ids: list[int]) -> dict[int, int]:
        if not vector_ids:
            return {}
        ph = ",".join("?" * len(vector_ids))
        rows = self._conn.execute(
            f"SELECT id, tool_id FROM tool_vectors WHERE id IN ({ph})", list(vector_ids)
        ).fetchall()
        return {r["id"]: r["tool_id"] for r in rows}

    def delete_vectors(self, vector_ids: list[int]):
        if not vector_ids:
            return
        ph = ",".join("?" * len(vector_ids))
        with self._lock:
            self._conn.execute(f"DELETE FROM tool_vectors WHERE id IN ({ph})", list(vector_ids))

    def count_learned(self, tool_id: int) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM tool_vectors WHERE tool_id = ? AND kind = 'learned'", (tool_id,)
        ).fetchone()[0]


# ── NEW: Chunking de texto ─────────────────────────────────────────────────────

def _chunk_text(
    text:         str,
    chunk_size:   int = CHUNK_SIZE,
    overlap:      int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Divide texto em chunks sobrepostos com metadados de posição.

    Estratégia de split (prioridade):
      1. Quebra de parágrafo (\\n\\n)
      2. Quebra de linha (\\n)
      3. Fim de sentença (. ! ?)
      4. Espaço (limite de palavra)
      5. Corte duro no chunk_size

    Retorna lista de dicts:
      {"text": str, "index": int, "char_start": int, "char_end": int}
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Tenta encontrar um ponto de quebra natural na segunda metade do chunk
            search_start = start + chunk_size // 2

            # 1. Parágrafo duplo
            split_pos = text.rfind('\n\n', search_start, end)
            if split_pos != -1:
                end = split_pos + 2  # inclui o \n\n
            else:
                # 2. Linha simples
                split_pos = text.rfind('\n', search_start, end)
                if split_pos != -1:
                    end = split_pos + 1
                else:
                    # 3. Fim de sentença
                    best_sep = -1
                    for sep in ('. ', '.\n', '! ', '? ', '。', '！', '？', '; ', ';\n'):
                        pos = text.rfind(sep, search_start, end)
                        if pos != -1 and pos > best_sep:
                            best_sep = pos + len(sep)
                    if best_sep != -1:
                        end = best_sep
                    else:
                        # 4. Espaço
                        split_pos = text.rfind(' ', search_start, end)
                        if split_pos != -1:
                            end = split_pos + 1
                        # 5. Corte duro — end já está setado

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text":       chunk_text,
                "index":      chunk_index,
                "char_start": start,
                "char_end":   end,
            })
            chunk_index += 1

        # Avança com sobreposição
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # garante progresso

        # Limite de chunks
        if chunk_index >= IF_MAX_CHUNKS:
            log.warning(f"Chunking atingiu limite de {IF_MAX_CHUNKS} — truncando")
            break

    return chunks


# ── Índice vetorial (LanceDB) ──────────────────────────────────────────────────

# Intervalo do job de manutenção periódica dos índices (ver index_flush_job).
# No LanceDB cada add() já é durável; o "flush" passou a ser a COMPACTAÇÃO dos
# fragmentos pequenos que os adds unitários geram (table.optimize()).
INDEX_FLUSH_INTERVAL_S = 60


class LanceStore:
    """Conexão única com o LanceDB + uma tabela por dimensão de embedding.

    Schema de cada tabela `vectors_<dim>`:
        memory_type : string          — qual store (long_term, short_term, tool…)
        id          : int64           — id do registro DENTRO do store
        vector      : fixed_size_list<float32>[dim]
    A chave lógica é (memory_type, id): os ids vêm de sequências SQLite
    independentes por store, então repetem entre stores."""

    def __init__(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self._path   = path
        self._db     = lancedb.connect(path)
        self.lock    = threading.RLock()
        self._tables: dict[int, object] = {}
        self._writes: dict[int, int]    = {}

    def schema(self, dim: int) -> "pa.Schema":
        return pa.schema([
            pa.field("memory_type", pa.string()),
            pa.field("id", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ])

    def table(self, dim: int):
        with self.lock:
            tbl = self._tables.get(dim)
            if tbl is None:
                tbl = self._db.create_table(
                    f"{LANCE_TABLE_PREFIX}_{dim}", schema=self.schema(dim), exist_ok=True
                )
                self._tables[dim] = tbl
            return tbl

    def note_write(self, dim: int, n: int = 1):
        with self.lock:
            self._writes[dim] = self._writes.get(dim, 0) + n

    def compact(self, dim: int):
        with self.lock:
            if not self._writes.get(dim):
                return
            try:
                self.table(dim).optimize()
            except Exception as e:
                log.error(f"LanceDB: falha ao compactar vectors_{dim}: {e}")
                return
            self._writes[dim] = 0


class MemoryIndex:
    """
    Visão de UM store (`memory_type`) sobre a tabela LanceDB da sua dimensão.
    Mantém a mesma interface do antigo índice FAISS (add, add_batch,
    remove_ids, reset, search, search_batch, search_similar, search_subset,
    flush, total), então o resto do arquivo não muda — e ganha:

      * sem .npy de id-map e sem IndexIDMap2: o id vive na própria linha;
      * remove_ids = DELETE com filtro (sem reconstruir nada);
      * search_subset com filtro nativo `id IN (...)` (antes: reconstruct em
        loop Python);
      * get_vectors(ids), usado pelo fallback de cosine do MMR.

    Métrica: cosine (score = 1 - distância cosine), equivalente ao produto
    interno do FAISS para vetores L2-normalizados — e correta se algum não for.

    Migração: se `legacy_index_path` existir e o store ainda estiver vazio no
    LanceDB, o índice FAISS antigo (IndexIDMap2, ou IndexFlatIP + .npy) é
    importado uma vez e renomeado para `<path>.migrated` (backup).
    """

    def __init__(
        self,
        store: LanceStore,
        memory_type: str,
        embed_dim: int = EMBED_DIM,
        legacy_index_path: Optional[str] = None,
        legacy_id_map_path: Optional[str] = None,
    ):
        self._store       = store
        self._memory_type = memory_type
        self._embed_dim   = embed_dim
        self._table       = store.table(embed_dim)
        self._filter      = f"memory_type = '{memory_type}'"
        if legacy_index_path:
            self._migrate_from_faiss(legacy_index_path, legacy_id_map_path)
        log.info(f"Índice LanceDB [{memory_type}] pronto — {self.total} vetores (dim={embed_dim})")

    # ── migração one-shot FAISS → LanceDB ──
    def _migrate_from_faiss(self, index_path: str, id_map_path: Optional[str]):
        p = Path(index_path)
        if not p.exists():
            return
        with self._store.lock:
            if self._table.count_rows(self._filter) > 0:
                log.warning(
                    f"[{self._memory_type}] {index_path} existe mas o LanceDB já tem dados "
                    f"desse store — migração ignorada (apague/renomeie o arquivo legado)"
                )
                return
        try:
            import faiss  # só necessário para migrar índices antigos
        except ImportError:
            log.warning(f"[{self._memory_type}] faiss não instalado — não foi possível migrar {index_path}")
            return
        loaded = faiss.read_index(str(p))
        n = loaded.ntotal
        if n > 0:
            if isinstance(loaded, faiss.IndexIDMap2) or hasattr(loaded, "id_map"):
                ids  = faiss.vector_to_array(loaded.id_map).astype(np.int64)
                vecs = faiss.downcast_index(loaded.index).reconstruct_n(0, n)
            else:
                # formato antigo: IndexFlatIP + .npy externo com os ids
                ids = np.load(id_map_path).astype(np.int64) if id_map_path and Path(id_map_path).exists() \
                      else np.arange(n, dtype=np.int64)
                if len(ids) != n:
                    log.warning(f"[{self._memory_type}] id_map legado com {len(ids)} ids p/ {n} vetores — usando posição")
                    ids = np.arange(n, dtype=np.int64)
                vecs = loaded.reconstruct_n(0, n)
            self.add_batch(np.asarray(vecs, dtype=np.float32), [int(i) for i in ids])
            if self.total != n:
                raise RuntimeError(
                    f"[{self._memory_type}] migração FAISS→LanceDB inconsistente "
                    f"({self.total} != {n}) — arquivo legado mantido"
                )
        p.rename(str(p) + ".migrated")
        log.info(f"[{self._memory_type}] {n} vetores migrados de {index_path} (backup: .migrated)")

    def _arrow(self, ids: list[int], embeddings: np.ndarray):
        emb = np.ascontiguousarray(embeddings, dtype=np.float32).reshape(len(ids), self._embed_dim)
        return pa.table(
            {
                "memory_type": pa.array([self._memory_type] * len(ids), pa.string()),
                "id":          pa.array(ids, pa.int64()),
                "vector":      pa.FixedSizeListArray.from_arrays(
                                   pa.array(emb.ravel(), pa.float32()), self._embed_dim),
            },
            schema=self._store.schema(self._embed_dim),
        )

    def add(self, embedding: np.ndarray, record_id: int):
        self.add_batch(np.asarray(embedding, dtype=np.float32).reshape(1, -1), [record_id])

    def add_batch(self, embeddings: np.ndarray, record_ids: list[int]):
        """Adiciona múltiplos vetores de uma vez — bem mais barato que add()
        individual (cada chamada gera um fragmento novo no LanceDB)."""
        if embeddings.shape[0] != len(record_ids):
            raise ValueError(
                f"embeddings ({embeddings.shape[0]}) e record_ids ({len(record_ids)}) "
                f"devem ter o mesmo tamanho"
            )
        if embeddings.shape[0] == 0:
            return
        data = self._arrow([int(i) for i in record_ids], embeddings)
        with self._store.lock:
            self._table.add(data)
            self._store.note_write(self._embed_dim, len(record_ids))

    def remove_ids(self, record_ids: set[int]):
        if not record_ids:
            return
        id_list = ",".join(str(int(i)) for i in record_ids)
        with self._store.lock:
            self._table.delete(f"{self._filter} AND id IN ({id_list})")
            self._store.note_write(self._embed_dim)
        log.info(f"LanceDB: remoção de até {len(record_ids)} vetores [{self._memory_type}]")

    def reset(self):
        with self._store.lock:
            self._table.delete(self._filter)
            self._store.note_write(self._embed_dim)

    def _query(self, q: np.ndarray, where: str, k: int) -> list[tuple[int, float]]:
        with self._store.lock:
            rows = (
                self._table.search(q)
                .distance_type("cosine")
                .where(where, prefilter=True)
                .select(["id", "_distance"])
                .limit(k)
                .to_list()
            )
        return [(int(r["id"]), 1.0 - float(r["_distance"])) for r in rows]

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        return self._query(q, self._filter, top_k)

    def search_batch(
        self, query_embeddings: np.ndarray, top_k: int
    ) -> list[list[tuple[int, float]]]:
        """N queries (ex.: os segmentos de uma pergunta composta). N é pequeno
        (SEGMENT_MAX_COUNT), então é um loop simples sobre search()."""
        n = query_embeddings.shape[0] if query_embeddings.ndim == 2 else 0
        return [self.search(query_embeddings[i], top_k) for i in range(n)]

    def search_similar(self, embedding: np.ndarray) -> float:
        results = self.search(embedding, top_k=1)
        return results[0][1] if results else 0.0

    def search_subset(
        self, query_embedding: np.ndarray, record_ids: list[int], top_k: int
    ) -> list[tuple[int, float]]:
        """Ranqueia `record_ids` por similaridade com `query_embedding`, SEM
        comparar com o restante do store (filtro nativo `id IN (...)`)."""
        if not record_ids or top_k <= 0:
            return []
        id_list = ",".join(str(int(i)) for i in record_ids)
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        return self._query(q, f"{self._filter} AND id IN ({id_list})", min(top_k, len(record_ids)))

    def get_vectors(self, record_ids: list[int]) -> dict[int, np.ndarray]:
        """Vetores armazenados dos ids pedidos (ids ausentes ficam de fora)."""
        if not record_ids:
            return {}
        id_list = ",".join(str(int(i)) for i in record_ids)
        with self._store.lock:
            rows = (
                self._table.search()
                .where(f"{self._filter} AND id IN ({id_list})")
                .select(["id", "vector"])
                .limit(len(record_ids) * 2)
                .to_list()
            )
        out: dict[int, np.ndarray] = {}
        for r in rows:
            out.setdefault(int(r["id"]), np.asarray(r["vector"], dtype=np.float32))
        return out

    def flush(self):
        """Compacta os fragmentos gerados por adds/deletes desde a última
        compactação (no-op se não houve escrita). Chamado pelo
        `index_flush_job` e no `shutdown()`. A durabilidade em si é imediata."""
        self._store.compact(self._embed_dim)

    @property
    def total(self) -> int:
        with self._store.lock:
            return int(self._table.count_rows(self._filter))


# ── Estado global ──────────────────────────────────────────────────────────────

@dataclass
class AppState:
    embed_engine: EmbeddingEngine = field(default=None)
    rerank_engine: RerankEngine   = field(default=None)
    lance:        LanceStore      = field(default=None)   # NEW: LanceDB (todos os índices vetoriais)
    lt_db:        MemoryDB        = field(default=None)
    lt_index:     MemoryIndex     = field(default=None)
    st_db:        ShortTermDB     = field(default=None)
    st_index:     MemoryIndex     = field(default=None)
    vs:           Optional[VectorStore] = field(default=None)
    # ── NEW: Indexed files ──
    if_db:        IndexedFilesDB  = field(default=None)
    if_index:     MemoryIndex     = field(default=None)
    # ── NEW: Dicionário Visual ──
    vd_db:        VisualDictDB    = field(default=None)
    vd_index:     MemoryIndex     = field(default=None)
    # ── NEW: Dicionário de Rostos ──
    fd_db:        FaceDictDB      = field(default=None)
    fd_index:     MemoryIndex     = field(default=None)
    # ── NEW: Dicionário de Voz ──
    vo_db:        VoiceDictDB     = field(default=None)
    vo_index:     MemoryIndex     = field(default=None)
    # ── NEW: RAG de tools ──
    tl_db:        ToolsDB         = field(default=None)
    tl_index:     MemoryIndex     = field(default=None)
    decay_task:   asyncio.Task    = field(default=None)
    cleanup_task: asyncio.Task    = field(default=None)
    flush_task:   asyncio.Task    = field(default=None)

state = AppState()


def _all_indices() -> list["MemoryIndex"]:
    """Todos os índices vetoriais (LanceDB) do processo — usado pelo job de
    manutenção periódica e pelo shutdown para compactar tudo de uma vez."""
    return [
        idx for idx in (
            state.lt_index, state.st_index,
            state.if_index, state.vd_index, state.fd_index, state.vo_index,
            state.tl_index,
        )
        if idx is not None
    ]


# ── Jobs em background ─────────────────────────────────────────────────────────

async def decay_job():
    while True:
        await asyncio.sleep(DECAY_JOB_INTERVAL_S)
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, state.lt_db.apply_decay, DECAY_HALF_LIFE_DAYS)
            # ── NEW (Hebbian graph): decay + poda das arestas no MESMO job —
            # sem criar segundo loop/task (reusa o decay_task existente).
            pruned = await loop.run_in_executor(
                None, state.lt_db.decay_and_prune_edges,
                EDGE_DECAY_HALF_LIFE_DAYS, EDGE_PRUNE_THRESHOLD,
            )
            if pruned:
                log.info(
                    f"Grafo Hebbiano: {pruned} arestas podadas "
                    f"(threshold={EDGE_PRUNE_THRESHOLD})"
                )
        except Exception as e:
            log.error(f"Erro no decay job: {e}")


async def index_flush_job():
    """Compacta os índices LanceDB que receberam escritas desde o último ciclo
    (cada add() já é durável; isto só junta os fragmentos pequenos). Substitui o antigo `_save()` a cada escrita (ver ITEM 1 da
    revisão) — o custo de I/O passa a ser amortizado em vez de pago a cada
    add()/remove_ids()."""
    while True:
        await asyncio.sleep(INDEX_FLUSH_INTERVAL_S)
        loop = asyncio.get_event_loop()
        for index in _all_indices():
            try:
                await loop.run_in_executor(None, index.flush)
            except Exception as e:
                log.error(f"Erro ao compactar índice LanceDB: {e}")


async def st_cleanup_job():
    while True:
        await asyncio.sleep(ST_CLEANUP_INTERVAL_S)
        try:
            cutoff = time.time() - ST_TTL_HOURS * 3600
            rows = state.st_db._conn.execute(
                "SELECT id FROM turn_groups WHERE last_accessed < ?", (cutoff,)
            ).fetchall()
            expired_ids = {row["id"] for row in rows}
            if expired_ids:
                state.st_db.expire_old(ST_TTL_HOURS)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, state.st_index.remove_ids, expired_ids)
        except Exception as e:
            log.error(f"Erro no cleanup de curto prazo: {e}")


# ── Lifespan ───────────────────────────────────────────────────────────────────

def _resolve_vs_paths() -> tuple[str, str, str]:
    try:
        from config import FAISS_INDEX_PATH as _cfg_idx, FAISS_META_PATH as _cfg_meta
        idx_path    = str(_cfg_idx)
        meta_str    = str(_cfg_meta)
        db_path     = meta_str.replace(".json", ".db") if meta_str.endswith(".json") else meta_str + ".db"
        id_map_path = idx_path.replace(".index", "_id_map.npy")
        log.info(f"VS paths from config: index={idx_path} db={db_path}")
        return idx_path, db_path, id_map_path
    except ImportError:
        log.info(f"VS paths from defaults: index={VS_FAISS_INDEX_PATH} db={VS_DB_PATH}")
        return VS_FAISS_INDEX_PATH, VS_DB_PATH, VS_FAISS_ID_MAP_PATH


async def startup():
    """Inicializa o estado global da memória (DBs, índices FAISS, engine de
    embeddings, jobs em background). Chamado pelo MCP server no startup do
    processo — substitui o antigo `lifespan` do FastAPI."""
    log.info("Iniciando AVA Memory (MCP)...")

    try:
        from onnx_client import check_health
        health = await check_health(ONNX_SERVING_URL)
        log.info(f"ONNX Serving API healthy: {health}")
    except Exception as e:
        log.warning(f"ONNX Serving API not reachable at {ONNX_SERVING_URL}: {e}")
        log.warning("Memory API will start but embedding calls will fail until ONNX serving is available.")

    state.embed_engine = EmbeddingEngine(base_url=ONNX_SERVING_URL)
    state.rerank_engine = RerankEngine(base_url=ONNX_SERVING_URL)

    # ── NEW: LanceDB — um diretório só; FAISS_*_PATH viram fontes de migração ──
    state.lance = LanceStore(LANCE_DIR)

    state.lt_db    = MemoryDB(DB_PATH)
    state.lt_index = MemoryIndex(state.lance, MT_LONG_TERM, EMBED_DIM,
                                 FAISS_INDEX_PATH, FAISS_ID_MAP_PATH)

    state.st_db    = ShortTermDB(ST_DB_PATH)
    state.st_index = MemoryIndex(state.lance, MT_SHORT_TERM, EMBED_DIM,
                                 ST_FAISS_INDEX_PATH, ST_FAISS_ID_MAP_PATH)

    # ── NEW: Indexed files ──
    state.if_db    = IndexedFilesDB(IF_DB_PATH)
    state.if_index = MemoryIndex(state.lance, MT_INDEXED_FILE, EMBED_DIM,
                                 IF_FAISS_INDEX_PATH, IF_FAISS_ID_MAP_PATH)

    # ── NEW: Dicionário Visual ──
    state.vd_db    = VisualDictDB(VD_DB_PATH)
    state.vd_index = MemoryIndex(state.lance, MT_VISUAL_DICT, VD_EMBED_DIM,
                                 VD_FAISS_INDEX_PATH, VD_FAISS_ID_MAP_PATH)

    # ── NEW: Dicionário de Rostos ──
    state.fd_db    = FaceDictDB(FD_DB_PATH)
    state.fd_index = MemoryIndex(state.lance, MT_FACE_DICT, FD_EMBED_DIM,
                                 FD_FAISS_INDEX_PATH, FD_FAISS_ID_MAP_PATH)

    # ── NEW: Dicionário de Voz ──
    state.vo_db    = VoiceDictDB(VO_DB_PATH)
    state.vo_index = MemoryIndex(state.lance, MT_VOICE_DICT, VO_EMBED_DIM,
                                 VO_FAISS_INDEX_PATH, VO_FAISS_ID_MAP_PATH)

    # ── NEW: RAG de tools (índice novo — sem legado FAISS) ──
    state.tl_db    = ToolsDB(TOOLS_DB_PATH)
    state.tl_index = MemoryIndex(state.lance, MT_TOOL, EMBED_DIM)

    if _VS_AVAILABLE:
        try:
            vs_idx, vs_db, vs_idmap = _resolve_vs_paths()
            state.vs = VectorStore(
                index_path  = vs_idx,
                db_path     = vs_db,
                id_map_path = vs_idmap,
                embed_dim   = EMBED_DIM,
            )
            log.info(f"VectorStore integrado — {state.vs.total} chunks de conhecimento")
        except Exception as e:
            log.error(f"VectorStore initialization failed: {e} — knowledge search disabled")
            state.vs = None
    else:
        state.vs = None
        log.warning("VectorStore not available — knowledge search disabled")

    state.decay_task   = asyncio.create_task(decay_job())
    state.cleanup_task = asyncio.create_task(st_cleanup_job())
    state.flush_task   = asyncio.create_task(index_flush_job())

    log.info(
        f"Pronto — {state.lt_db.count()} memórias LT | "
        f"{state.st_db.count()} grupos ST | "
        f"{state.vs.total if state.vs else 0} chunks de conhecimento | "
        f"{state.if_db.count_files()} arquivos indexados ({state.if_db.get_total_chunks()} chunks) | "
        f"{state.vd_db.count()} conceitos visuais ({state.vd_index.total} embeddings) | "
        f"{state.fd_db.count()} pessoas cadastradas ({state.fd_index.total} embeddings de rosto) | "
        f"{state.vo_db.count()} pessoas com voz cadastrada ({state.vo_index.total} embeddings de voz) | "
        f"{state.tl_db.count()} tools indexadas ({state.tl_index.total} vetores)"
    )


async def shutdown():
    """Encerra o estado global da memória de forma limpa. Chamado pelo MCP
    server no shutdown do processo — substitui a parte pós-`yield` do antigo
    `lifespan` do FastAPI."""
    await state.embed_engine.client.close()

    if state.decay_task is not None:
        state.decay_task.cancel()
    if state.cleanup_task is not None:
        state.cleanup_task.cancel()
    if state.flush_task is not None:
        state.flush_task.cancel()

    # Flush final e síncrono de todos os índices — garante que nada gravado
    # desde o último ciclo do index_flush_job seja perdido ao encerrar.
    for index in _all_indices():
        try:
            index.flush()
        except Exception as e:
            log.error(f"Erro ao compactar índice LanceDB no shutdown: {e}")

    log.info("AVA Memory (MCP) encerrada")


# ── Erros ──────────────────────────────────────────────────────────────────────

class MemoryToolError(Exception):
    """Erro de validação/negócio de uma tool de memória — equivalente ao
    antigo HTTPException do FastAPI, mas agnóstico de transporte (MCP não
    tem código de status HTTP). A mensagem vira o texto de erro da tool."""
    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


# ── Helpers de busca contextual ────────────────────────────────────────────────

def _truncate_text(text: str, max_chars: int) -> str:
    """
    Trunca texto preservando o início e quebrando em fronteira de palavra.
    Adiciona '...' quando truncado. Usado para reduzir tokens no /read.
    """
    if not text or len(text) <= max_chars:
        return text
    # Reserva 3 chars para '...'
    cut = text[:max(max_chars - 3, 1)]
    # Tenta cortar em fronteira de palavra para não cortar no meio de token
    last_space = cut.rfind(' ')
    if last_space > max_chars // 2:
        cut = cut[:last_space]
    return cut.rstrip() + "..."


def _apply_token_budget(
    entries: list[MemoryEntry],
    total_max_chars: int,
    top_k: int,
) -> list[MemoryEntry]:
    """
    Aplica orçamento global de caracteres no resultado final do /read.
    Corta entradas de menor score primeiro quando o total excede o teto.
    """
    if not entries:
        return entries
    # Primeiro garante o top_k
    selected = entries[:top_k]
    # Depois corta de trás pra frente (menor score) enquanto exceder o teto
    while len(selected) > 1:
        total = sum(len(e.text or "") for e in selected)
        if total <= total_max_chars:
            break
        selected.pop()  # remove o último (menor score, já ordenado)
    return selected


# ── NEW: Segmentação heurística de query composta (sem LLM) ───────────────────

# Fronteira antes de gerúndios ("usando", "utilizando", "aplicando", "seguindo"
# ...) — em português, costumam introduzir um novo sub-meio/estratégia dentro
# da mesma frase.
_GERUND_BOUNDARY_RE = re.compile(r'(?<=\s)(?=\w+(?:ando|endo|indo)\b)', re.IGNORECASE)

# Fronteira antes de "para <verbo no infinitivo>" — costuma introduzir um
# novo sub-objetivo/resultado dentro da mesma frase.
_PURPOSE_BOUNDARY_RE = re.compile(r'(?<=\s)(?=para\s+\w+(?:ar|er|ir)\b)', re.IGNORECASE)


def _split_query_segments(query: str) -> list[str]:
    """
    Decompõe a query em sub-tópicos SEM LLM, para busca por segmento.

    Regra: (1) separa por pontuação forte → sentenças; (2) dentro de
    sentenças longas, separa por vírgula e por fronteiras heurísticas
    (gerúndio, "para <verbo>") que tendem a introduzir um novo sub-pedido;
    (3) funde fragmentos curtos demais no vizinho; (4) deduplica e limita
    a quantidade de segmentos.

    Cada segmento resultante é embedado e buscado separadamente — isso é o
    que evita a "diluição de atenção semântica" de embedar a pergunta
    inteira de uma vez.
    """
    query = query.strip()
    if not query:
        return []

    sentences = [s.strip() for s in re.split(r'[.!?;]+', query) if s.strip()]

    raw_segments: list[str] = []
    for sent in sentences:
        if len(sent.split()) <= SEGMENT_TRIGGER_WORDS:
            raw_segments.append(sent)
            continue

        for part in re.split(r'\s*,\s*', sent):
            for sub in _GERUND_BOUNDARY_RE.split(part):
                raw_segments.extend(
                    p.strip() for p in _PURPOSE_BOUNDARY_RE.split(sub) if p.strip()
                )

    # funde fragmentos curtos demais no segmento anterior
    merged: list[str] = []
    for seg in raw_segments:
        if merged and len(seg.split()) < SEGMENT_MIN_WORDS:
            merged[-1] = f"{merged[-1]} {seg}"
        else:
            merged.append(seg)
    if len(merged) >= 2 and len(merged[0].split()) < SEGMENT_MIN_WORDS:
        merged[1] = f"{merged[0]} {merged[1]}"
        merged.pop(0)

    # dedup preservando ordem
    seen: set[str] = set()
    dedup: list[str] = []
    for seg in merged:
        key = seg.lower()
        if key not in seen:
            seen.add(key)
            dedup.append(seg)

    # teto de segmentos — mantém os mais "carregados" semanticamente
    if len(dedup) > SEGMENT_MAX_COUNT:
        dedup = sorted(dedup, key=lambda s: len(s.split()), reverse=True)[:SEGMENT_MAX_COUNT]

    return dedup


def _fuse_max(
    base: list[tuple[int, float]],
    extra: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Une dois conjuntos (id, score) mantendo o maior score por id. Usado
    para combinar a busca da query inteira com a busca por segmento, sem
    que um sub-tópico "afogue" o score de outro (diferente de uma média)."""
    scores: dict[int, float] = dict(base)
    for mid, score in extra:
        if score > scores.get(mid, -1.0):
            scores[mid] = score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


async def _search_segmented(
    segments: list[str],
    top_k: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """
    Embeda TODOS os segmentos em uma única chamada em lote à API de
    embeddings, busca todos de uma vez no FAISS (batch nativo) e funde por
    max-score. Determinístico, sem LLM — apenas cosine similarity aplicado
    a cada pedaço da pergunta em vez da pergunta inteira.
    """
    embeddings = await state.embed_engine.embed_queries(segments)  # shape (N, dim)
    loop = asyncio.get_event_loop()
    lt_batches, st_batches = await asyncio.gather(
        loop.run_in_executor(None, state.lt_index.search_batch, embeddings, top_k),
        loop.run_in_executor(None, state.st_index.search_batch, embeddings, top_k),
    )
    lt_fused: list[tuple[int, float]] = []
    st_fused: list[tuple[int, float]] = []
    for seg_results in lt_batches:
        lt_fused = _fuse_max(lt_fused, seg_results)
    for seg_results in st_batches:
        st_fused = _fuse_max(st_fused, seg_results)
    return lt_fused, st_fused


def _classify_query(query: str) -> str:
    tokens = re.findall(r"\w+", query.lower())
    if not tokens:
        return "expanded"
    n_words      = len(tokens)
    n_stopwords  = sum(1 for t in tokens if t in _STOP_WORDS)
    stop_ratio   = n_stopwords / n_words
    is_short     = n_words <= QUERY_SHORT_WORDS
    is_ambiguous = stop_ratio >= QUERY_AMBIGUOUS_RATIO
    strategy = "dual" if (is_short or is_ambiguous) else "expanded"
    log.debug(f"Query classify: words={n_words} stop_ratio={stop_ratio:.2f} → {strategy}")
    return strategy


def _build_context_block(session_id: str) -> str:
    lines = state.st_db.get_recent_turns_text(session_id, CONTEXT_TURNS_FETCH)
    if not lines:
        return ""
    block = "\n".join(lines)
    if len(block) > CONTEXT_MAX_CHARS:
        block = block[-CONTEXT_MAX_CHARS:]
        newline_pos = block.find("\n")
        if newline_pos != -1:
            block = block[newline_pos + 1:]
    return block


def _fuse_scores(
    q_results: list[tuple[int, float]],
    c_results: list[tuple[int, float]],
    context_weight: float,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for mid, score in q_results:
        scores[mid] = (1.0 - context_weight) * score
    for mid, score in c_results:
        scores[mid] = scores.get(mid, 0.0) + context_weight * score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


async def _search_expanded(
    query: str,
    context_block: str,
    top_k: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    expanded = f"{context_block}\n\nquery atual: {query}" if context_block else query
    emb = await state.embed_engine.embed_query_one(expanded)
    loop = asyncio.get_event_loop()
    lt_raw, st_raw = await asyncio.gather(
        loop.run_in_executor(None, state.lt_index.search, emb, top_k * 2),
        loop.run_in_executor(None, state.st_index.search, emb, top_k * 2),
    )
    return lt_raw, st_raw


async def _search_dual(
    query: str,
    context_block: str,
    top_k: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    emb_query, emb_ctx = await state.embed_engine.embed_query_two(query, context_block)
    loop = asyncio.get_event_loop()
    lt_q, st_q, lt_c, st_c = await asyncio.gather(
        loop.run_in_executor(None, state.lt_index.search, emb_query, top_k * 2),
        loop.run_in_executor(None, state.st_index.search, emb_query, top_k * 2),
        loop.run_in_executor(None, state.lt_index.search, emb_ctx,   top_k * 2),
        loop.run_in_executor(None, state.st_index.search, emb_ctx,   top_k * 2),
    )
    lt_fused = _fuse_scores(lt_q, lt_c, DUAL_CONTEXT_WEIGHT)
    st_fused = _fuse_scores(st_q, st_c, DUAL_CONTEXT_WEIGHT)
    return lt_fused, st_fused



# ── NEW (Hebbian graph): PPR + helpers ─────────────────────────────────────

def personalized_pagerank(
    adjacency: dict[int, dict[int, float]],
    personalization: dict[int, float],
    damping: float,
    max_iter: int,
    eps: float,
) -> dict[int, float]:
    """Personalized PageRank por power iteration VETORIZADA (scipy.sparse) —
    síncrono, rode via run_in_executor.

    `adjacency[a][b]` = peso da transição a → b (pode ser assimétrico: arestas
    dirigidas/tipadas já chegam aqui modulados por _build_adjacency). Cada linha
    é normalizada pela soma de saída; nó sem saída (dangling) simplesmente não
    propaga — mesma semântica da implementação anterior em dicts Python.
    A iteração é r ← (1-d)·p + d·Pᵀ·r, com Pᵀ em CSR."""
    # nós = origens ∪ DESTINOS ∪ sementes: com arestas dirigidas um destino pode
    # não aparecer como chave de `adjacency` (nó sem saída).
    node_set = set(adjacency) | set(personalization)
    for nbs in adjacency.values():
        node_set.update(nbs)
    nodes = sorted(node_set)
    if not nodes:
        return {}
    index = {n: i for i, n in enumerate(nodes)}
    size = len(nodes)
    p = np.fromiter((personalization.get(n, 0.0) for n in nodes), dtype=np.float64, count=size)

    rows, cols, vals = [], [], []
    for a, nbs in adjacency.items():
        ia = index[a]
        for b, w in nbs.items():
            if w > 0:
                rows.append(ia); cols.append(index[b]); vals.append(w)

    pt = None
    if vals:
        w_mat = sp.csr_matrix((vals, (rows, cols)), shape=(size, size), dtype=np.float64)
        out_sum = np.asarray(w_mat.sum(axis=1)).ravel()
        inv = np.divide(1.0, out_sum, out=np.zeros_like(out_sum), where=out_sum > 0)
        pt = (sp.diags(inv) @ w_mat).T.tocsr()

    rank = p.copy()
    for _ in range(max_iter):
        new_rank = (1.0 - damping) * p
        if pt is not None:
            new_rank = new_rank + damping * (pt @ rank)
        delta = float(np.abs(new_rank - rank).sum())
        rank = new_rank
        if delta < eps:
            break
    return {n: float(rank[i]) for n, i in index.items()}


def _edge_key(id_a: int, id_b: int) -> tuple[int, int]:
    """Chave canônica de um par não-direcionado (min, max)."""
    return (min(id_a, id_b), max(id_a, id_b))


def _edge_traversal_weights(row) -> tuple[float, float]:
    """(peso a→b, peso b→a) de uma linha de aresta, já com o fator do tipo
    (EDGE_PPR_FACTORS) aplicado."""
    f_ab, f_ba = EDGE_PPR_FACTORS.get(row["edge_type"], (1.0, 1.0))
    w = row["weight"]
    return w * f_ab, w * f_ba


def _build_adjacency(
    edge_rows, node_filter: Optional[set[int]] = None,
) -> dict[int, dict[int, float]]:
    """Adjacência dirigida p/ o PPR a partir das linhas de aresta tipadas.
    Várias arestas de tipos diferentes entre o mesmo par somam. Teto de
    vizinhos por nó (PPR_MIN_NEIGHBORS_PER_HOP) preserva os de maior peso."""
    adjacency: dict[int, dict[int, float]] = {}
    for row in edge_rows:
        a, b = row["memory_id_a"], row["memory_id_b"]
        if a == b or row["weight"] <= 0:
            continue
        if node_filter is not None and (a not in node_filter or b not in node_filter):
            continue
        w_ab, w_ba = _edge_traversal_weights(row)
        if w_ab > 0:
            nb = adjacency.setdefault(a, {})
            nb[b] = nb.get(b, 0.0) + w_ab
        if w_ba > 0:
            nb = adjacency.setdefault(b, {})
            nb[a] = nb.get(a, 0.0) + w_ba
    for node, nbs in adjacency.items():
        if len(nbs) > PPR_MIN_NEIGHBORS_PER_HOP:
            top = sorted(nbs.items(), key=lambda kv: kv[1], reverse=True)[:PPR_MIN_NEIGHBORS_PER_HOP]
            adjacency[node] = dict(top)
    return adjacency


def _expand_graph_sync(
    db, seed_ids: list[int], max_nodes: int, min_path_weight: float, neighbor_cap: int,
) -> tuple[list, set[int]]:
    """Expansão ADAPTATIVA do subgrafo: BFS priorizado por peso acumulado.

    O "peso acumulado" de um nó é o produto dos pesos efetivos de aresta (com
    fator de tipo/direção) ao longo do MELHOR caminho conhecido desde uma
    semente (sementes = 1.0). Uma fila de prioridade sempre expande o nó de
    maior peso primeiro; para quando o número de nós carregados chega a
    `max_nodes` (nós novos deixam de ser admitidos) ou quando o melhor nó
    restante tem peso < `min_path_weight`. Substitui o teto fixo de hops:
    memórias fortemente conectadas a 3+ hops entram; ramos fracos param cedo.

    Retorna (linhas de aresta coletadas, conjunto de nós carregados).
    Síncrono (uma query por nó expandido) — rode via run_in_executor."""
    best: dict[int, float] = {int(s): 1.0 for s in seed_ids}
    heap: list[tuple[float, int]] = [(-1.0, s) for s in best]
    heapq.heapify(heap)
    expanded: set[int] = set()
    edge_rows: dict[tuple[int, int, str], object] = {}

    while heap:
        neg_acc, node = heapq.heappop(heap)
        acc = -neg_acc
        if node in expanded or acc < best.get(node, 0.0) - 1e-12:
            continue                      # entrada obsoleta (achou-se caminho melhor)
        if acc < min_path_weight:
            break                         # heap é max-first: o resto é ainda mais fraco
        expanded.add(node)
        for row in db.get_neighbors(node, neighbor_cap):
            a, b = row["memory_id_a"], row["memory_id_b"]
            other = b if a == node else a
            w_ab, w_ba = _edge_traversal_weights(row)
            w_out = w_ab if node == a else w_ba
            edge_rows[(a, b, row["edge_type"])] = row
            if w_out <= 0 or other in expanded:
                continue
            new_acc = acc * w_out
            if new_acc < min_path_weight:
                continue                  # nunca seria expandido: nem admite no subgrafo
            known = best.get(other)
            if known is None:
                if len(best) >= max_nodes:
                    continue              # orçamento de nós esgotado
                best[other] = new_acc
                heapq.heappush(heap, (-new_acc, other))
            elif new_acc > known:
                best[other] = new_acc
                heapq.heappush(heap, (-new_acc, other))
    return list(edge_rows.values()), set(best)


def _log_activation_sync(seed_ids: list[int], rank: dict[int, float], elapsed_ms: float):
    """Grava um evento de ativação PPR no log JSONL, consumido por
    live_activation_server.py p/ visualização em tempo real. Best-effort:
    qualquer falha de IO aqui NUNCA deve derrubar uma leitura de memória."""
    try:
        top = sorted(rank.items(), key=lambda kv: kv[1], reverse=True)[:ACTIVATION_LOG_MAX_IDS]
        line = json.dumps({
            "ts": time.time(),
            "seeds": seed_ids,
            "rank": {str(k): round(v, 4) for k, v in top},
            "elapsed_ms": round(elapsed_ms, 2),
        }, ensure_ascii=False)
        Path(ACTIVATION_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(ACTIVATION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        log.debug("activation log: falha ao gravar (ignorado)", exc_info=True)


async def _ppr_spread_on(
    db,
    seeds: list[tuple[int, float]],
    loop: asyncio.AbstractEventLoop,
) -> dict:
    """Propagação PPR num grafo de arestas tipadas (`db` = MemoryDB ou ToolsDB),
    com sementes em `seeds` = [(id, score)]. Retorna:
      related_raw  — [(id, score)] dos ids alcançados SÓ pelo grafo
      ppr_rank     — {id: rank} completo (sementes + vizinhos), p/ Step 8
      adjacency    — {id: {vizinho: weight}} do subgrafo local, p/ Step 8
      edge_rows    — linhas brutas de aresta do subgrafo
      elapsed_ms   — tempo da propagação (profiling)
    """
    result: dict = {
        "related_raw": [], "ppr_rank": {}, "adjacency": {}, "edge_rows": [],
        "elapsed_ms": 0.0,
    }
    scores: dict[int, float] = {}
    for mid, score in seeds:
        scores[mid] = max(score, scores.get(mid, 0.0))
    seed_ids = list(scores)
    if not seed_ids:
        return result
    t0 = time.perf_counter()
    edges, discovered = await loop.run_in_executor(
        None, _expand_graph_sync, db, seed_ids,
        PPR_EXPAND_MAX_NODES, PPR_EXPAND_MIN_PATH_WEIGHT, PPR_MIN_NEIGHBORS_PER_HOP,
    )
    if not edges:
        return result
    adjacency = _build_adjacency(edges, discovered)
    if not adjacency:
        return result
    # Vetor de personalização: scores clipados (>= 0) e normalizados.
    clipped = {mid: max(score, 0.0) for mid, score in scores.items()}
    total = sum(clipped.values())
    if total <= 0:
        return result
    personalization = {mid: s / total for mid, s in clipped.items()}
    rank = await loop.run_in_executor(
        None, personalized_pagerank,
        adjacency, personalization, PPR_DAMPING, PPR_MAX_ITER, PPR_CONVERGENCE_EPS,
    )
    # ── NEW: dispara log de ativação p/ visualização em tempo real, sem bloquear ──
    asyncio.ensure_future(
        loop.run_in_executor(
            None, _log_activation_sync, seed_ids, rank, (time.perf_counter() - t0) * 1000.0
        )
    )
    seed_set = set(seed_ids)
    related_raw = [
        (mid, PPR_SPREAD_WEIGHT * r)
        for mid, r in rank.items()
        if mid not in seed_set and r >= PPR_MIN_ACTIVATION_REINFORCE
    ]
    related_raw.sort(key=lambda x: x[1], reverse=True)
    result.update(
        related_raw=related_raw, ppr_rank=rank,
        adjacency=adjacency, edge_rows=edges,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )
    return result


async def _ppr_spread(
    lt_raw: list[tuple[int, float]],
    loop: asyncio.AbstractEventLoop,
) -> dict:
    """Step 6 — propagação PPR no grafo Hebbiano de memórias LT, com sementes
    nos hits LT. Roda DEPOIS que `lt_raw` está finalizado e ANTES de converter
    em MemoryEntry."""
    return await _ppr_spread_on(state.lt_db, lt_raw, loop)


async def _dict_graph_related(
    seeds: list[tuple[Optional[int], float]],
    loop: asyncio.AbstractEventLoop,
) -> list[RelatedMemory]:
    """Busca por grafo a partir de hits de dicionário (objeto/rosto/voz): usa o
    `memory_id` de cada candidato como semente (score = similaridade do hit)
    e devolve as memórias LT alcançadas SÓ pelas arestas. Só leitura — não
    cria nem reforça arestas (leituras de dicionário são frequentes/por-frame
    e não devem fortalecer o grafo sozinhas). Candidatos sem memory_id não
    têm nó no grafo e são ignorados."""
    real = [(mid, s) for mid, s in seeds if mid is not None]
    if not real:
        return []
    ctx = await _ppr_spread(real, loop)
    related_raw = ctx["related_raw"][:DICT_GRAPH_MAX_RELATED]
    if not related_raw:
        return []
    rows = await loop.run_in_executor(
        None, state.lt_db.get_by_ids, [mid for mid, _ in related_raw]
    )
    by_id = {r["id"]: r for r in rows}
    return [
        RelatedMemory(
            memory_id=mid,
            text=_truncate_text(by_id[mid]["text"], READ_LT_MAX_CHARS),
            score=round(score, 4),
        )
        for mid, score in related_raw if mid in by_id
    ]


def _lateral_inhibition_filter(results: list[MemoryEntry]) -> list[MemoryEntry]:
    """Step 7 — MMR (inibição lateral) restrito às entradas "related".

    "Similaridade" entre dois candidatos:
      * se existe aresta entre eles → peso da aresta × EDGE_MMR_FACTORS[tipo]
        (o maior entre as arestas do par). Suprimir hub genérico é redundância
        GRÁFICA; contradicts/updates contam como redundância máxima (mesmo
        assunto, versões conflitantes); temporal_precedence conta pouco;
      * se NÃO existe aresta → cosine direta entre os embeddings (antes: 0.0,
        "totalmente diverso" mesmo p/ candidatos quase idênticos que nunca
        co-ativaram), descontada por MMR_COSINE_FLOOR.
    Entradas "primary" passam intactas."""
    primary = [r for r in results if r.match_type != "related"]
    related = [r for r in results if r.match_type == "related"]
    if len(related) <= 1:
        return results
    ids = [r.id for r in related]

    edge_sim: dict[tuple[int, int], float] = {}
    for row in state.lt_db.get_edges_for_ids(ids):
        sim = row["weight"] * EDGE_MMR_FACTORS.get(row["edge_type"], 1.0)
        key = _edge_key(row["memory_id_a"], row["memory_id_b"])
        edge_sim[key] = max(edge_sim.get(key, 0.0), sim)

    try:
        vecs = state.lt_index.get_vectors(ids)
    except Exception as e:
        log.warning(f"MMR: não foi possível buscar vetores p/ o fallback de cosine: {e}")
        vecs = {}
    unit: dict[int, np.ndarray] = {}
    for mid, v in vecs.items():
        n = float(np.linalg.norm(v))
        if n > 0:
            unit[mid] = v / n

    pair_cache: dict[tuple[int, int], float] = {}

    def pair_sim(i: int, j: int) -> float:
        key = _edge_key(i, j)
        cached = pair_cache.get(key)
        if cached is not None:
            return cached
        sim = edge_sim.get(key)
        if sim is None:                      # sem aresta → cosine direta
            vi, vj = unit.get(i), unit.get(j)
            if vi is None or vj is None:
                sim = 0.0
            else:
                cos = float(np.dot(vi, vj))
                sim = max(0.0, (cos - MMR_COSINE_FLOOR) / (1.0 - MMR_COSINE_FLOOR))
        pair_cache[key] = sim
        return sim

    selected: list[MemoryEntry] = []
    pool = list(related)
    while pool:
        def mmr_score(c: MemoryEntry) -> float:
            relevance = c.score * c.confidence
            if not selected:
                return relevance
            max_sim = max(pair_sim(c.id, s.id) for s in selected)
            return MMR_LAMBDA * relevance - (1.0 - MMR_LAMBDA) * max_sim

        best = max(pool, key=mmr_score)
        selected.append(best)
        pool.remove(best)

    # ATENÇÃO: NÃO reordenar por score aqui. O laço acima seleciona TODOS os
    # "related" (até esvaziar o pool), então um sort por relevância depois
    # descartava exatamente a ordem que o MMR acabou de calcular — o filtro
    # virava no-op e o corte do orçamento de tokens (entries[:top_k]) pegava os
    # mais relevantes, não os mais diversos. Os "related" sempre têm score PPR
    # (<= PPR_SPREAD_WEIGHT) abaixo dos "primary" (>= READ_MIN_SCORE_STRICT), então
    # primary (já ordenado por score) + related (ordem MMR) preserva a ordem
    # global e deixa o MMR decidir QUEM sobrevive ao corte.
    return primary + selected


def _upsert_edges_sync(pairs: list[tuple[int, int]], now: float) -> int:
    """Helper síncrono p/ executor: cria/reforça arestas em lote (Step 4)."""
    for id_a, id_b in pairs:
        state.lt_db.upsert_edge(id_a, id_b, EDGE_LEARNING_RATE, now)
    return len(pairs)


async def _hebbian_link_task(results: list[MemoryEntry]) -> None:
    """Step 4 (background) — regra hebbiana NO CONJUNTO FINAL do /read: todo
    par de memórias LT com score >= EDGE_MIN_SCORE_TO_LINK co-ativou, então
    cria/reforça a aresta. Fire-and-forget: loga falhas, nunca levanta —
    o cliente já recebeu a resposta antes disso começar."""
    try:
        scored = [
            (r.id, r.score)
            for r in results
            if r.memory_type == "long_term" and r.score >= EDGE_MIN_SCORE_TO_LINK
        ]
        if len(scored) < 2:
            return
        now = time.time()
        if len(scored) > HEBBIAN_LINK_MAX_IDS:
            # Safeguard: leitura multi-tópico não pode virar O(n²) de arestas
            # — liga cada id apenas ao de maior score e avisa.
            scored.sort(key=lambda x: x[1], reverse=True)
            hub_id = scored[0][0]
            pairs = [(hub_id, other) for other, _ in scored[1:]]
            log.warning(
                f"Grafo Hebbiano: {len(scored)} ids acima de {EDGE_MIN_SCORE_TO_LINK} "
                f"no /read — ligando cada um apenas ao top-score (anti-O(n²))"
            )
        else:
            ids = [mid for mid, _ in scored]
            pairs = [
                (ids[i], ids[j])
                for i in range(len(ids))
                for j in range(i + 1, len(ids))
            ]
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _upsert_edges_sync, pairs, now)
    except Exception as e:
        log.error(f"Grafo Hebbiano: falha ao criar arestas pós-/read: {e}")


async def _ppr_reinforce_task(ppr_ctx: dict) -> None:
    """Step 8 (background, Opção B) — reforço das arestas EXISTENTES do
    subgrafo PPR pela regra literal de co-ativação:
        reinforcement = rank_i * rank_j * w_ij
        w_new = w_old + lr * reinforcement * (1 - w_old)
    NÃO cria aresta nova (isso é exclusivamente do Step 4). Elegibilidade:
    rank_i e rank_j ambos > PPR_MIN_ACTIVATION_REINFORCE. Mais barato que
    rastrear caminhos exatos de propagação — e mais hebbiano de qualquer
    forma (co-ativação, não causalidade de caminho)."""
    try:
        rank = ppr_ctx.get("ppr_rank") or {}
        adjacency = ppr_ctx.get("adjacency") or {}
        if not rank or not adjacency:
            return
        updates: list[tuple[int, int, float]] = []
        seen: set[tuple[int, int]] = set()
        for a, neighbors in adjacency.items():
            rank_a = rank.get(a, 0.0)
            if rank_a <= PPR_MIN_ACTIVATION_REINFORCE:
                continue
            for b in neighbors:
                rank_b = rank.get(b, 0.0)
                if rank_b <= PPR_MIN_ACTIVATION_REINFORCE:
                    continue
                key = _edge_key(a, b)
                if key in seen:
                    continue  # adjacência é simétrica — não reforça 2x a mesma aresta
                seen.add(key)
                updates.append((key[0], key[1], rank_a * rank_b))
        if not updates:
            return
        now = time.time()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, state.lt_db.reinforce_edges_batch,
            updates, EDGE_LEARNING_RATE, PPR_MIN_ACTIVATION_REINFORCE, now,
        )
    except Exception as e:
        log.error(f"Grafo Hebbiano: falha no reforço por ativação PPR: {e}")


def _build_lt_entries(
    lt_raw: list[tuple[int, float]],
    min_score: float,
    loop: asyncio.AbstractEventLoop,
    max_chars: int = READ_LT_MAX_CHARS,
    related_ids: Optional[set[int]] = None,
) -> list[MemoryEntry]:
    ids_filtered = [mid for mid, score in lt_raw if score >= min_score]
    score_map    = {mid: score for mid, score in lt_raw}
    if not ids_filtered:
        return []
    related_ids = related_ids or set()
    entries = []
    for row in state.lt_db.get_by_ids(ids_filtered):
        entries.append(MemoryEntry(
            id           = row["id"],
            text         = _truncate_text(row["text"], max_chars),
            score        = round(score_map[row["id"]], 4),
            confidence   = round(row["confidence"], 4),
            created_at   = row["created_at"],
            access_count = row["access_count"],
            memory_type  = "long_term",
            source       = row["source"],
            forgettable  = bool(row["forgettable"]),
            ttl_days     = row["ttl_days"],
            # ── NEW (Hebbian graph): id alcançado só via PPR vira "related" ──
            match_type   = "related" if row["id"] in related_ids else "primary",
        ))
        loop.run_in_executor(None, state.lt_db.update_access, row["id"])
    return entries


def _build_st_entries(
    st_raw: list[tuple[int, float]],
    min_score: float,
    loop: asyncio.AbstractEventLoop,
    max_chars: int = READ_ST_MAX_CHARS,
) -> list[MemoryEntry]:
    ids_filtered = [mid for mid, score in st_raw if score >= min_score]
    score_map    = {mid: score for mid, score in st_raw}
    if not ids_filtered:
        return []
    entries = []
    for row in state.st_db.get_by_ids(ids_filtered):
        score      = score_map[row["id"]]
        turns_data = json.loads(row["turns_json"])
        turns      = [Turn(**t) for t in turns_data]
        # Representação compacta: prioriza último user turn, depois assistant
        # Antes: " | ".join(f"[role] content[:120]" for t in turns) — acumulava
        # Agora: apenas primeiro e último turn, cada um com 80 chars no máx
        if len(turns) <= 2:
            parts = [f"[{t.role}] {t.content[:100]}" for t in turns]
        else:
            first = turns[0]
            last  = turns[-1]
            parts = [
                f"[{first.role}] {first.content[:80]}",
                f"...(+{len(turns)-2} turns)...",
                f"[{last.role}] {last.content[:100]}",
            ]
        text_repr = " | ".join(parts)
        entries.append(MemoryEntry(
            id           = row["id"],
            text         = _truncate_text(text_repr, max_chars),
            score        = round(score, 4),
            confidence   = 1.0,
            created_at   = row["created_at"],
            access_count = row["access_count"],
            memory_type  = "short_term",
            session_id   = row["session_id"],
        ))
        loop.run_in_executor(None, state.st_db.update_access, row["id"])
    return entries


def _build_vs_entries(
    vs_results: list,
    min_score: float,
    max_chars: int = READ_VS_MAX_CHARS,
) -> list[MemoryEntry]:
    if not vs_results:
        return []
    entries = []
    for ventry, score in vs_results:
        if score < min_score:
            continue
        entries.append(MemoryEntry(
            id           = ventry.id,
            text         = _truncate_text(ventry.text, max_chars),
            score        = round(score, 4),
            confidence   = 1.0,
            created_at   = 0.0,
            access_count = 0,
            memory_type  = "knowledge",
            source       = ventry.source,
        ))
    return entries


# ── NEW: Build indexed file entries from FAISS search results ─────────────────

def _build_if_entries(
    if_raw: list[tuple[int, float]],
    min_score: float,
    loop: asyncio.AbstractEventLoop,
    return_full_content: bool = False,
    max_chars: int = READ_IF_MAX_CHARS,
) -> list[MemoryEntry]:
    """
    Converte resultados de busca FAISS de chunks em MemoryEntry.

    Se return_full_content=True, text contém o conteúdo completo do arquivo.
    Se False, text contém apenas o chunk que deu match (mais conciso para /read).
    Deduplica por file_id — mantém apenas o melhor score por arquivo.
    """
    if not if_raw:
        return []

    # Filtra por score mínimo
    filtered = [(cid, score) for cid, score in if_raw if score >= min_score]
    if not filtered:
        return []

    # Busca chunk records
    chunk_ids = [cid for cid, _ in filtered]
    chunk_rows = state.if_db.get_chunks_by_ids(chunk_ids)
    chunk_map = {row["id"]: row for row in chunk_rows}

    # Agrupa por file_id — mantém melhor score por arquivo
    file_best: dict[int, tuple[float, sqlite3.Row]] = {}
    for cid, score in filtered:
        chunk_row = chunk_map.get(cid)
        if chunk_row is None:
            continue
        fid = chunk_row["file_id"]
        if fid not in file_best or score > file_best[fid][0]:
            file_best[fid] = (score, chunk_row)

    if not file_best:
        return []

    # Busca file records
    file_ids = list(file_best.keys())
    file_rows = state.if_db._conn.execute(
        f"SELECT * FROM indexed_files WHERE id IN ({','.join('?' * len(file_ids))})",
        file_ids,
    ).fetchall()
    file_map = {row["id"]: row for row in file_rows}

    entries = []
    for fid, (score, chunk_row) in file_best.items():
        file_row = file_map.get(fid)
        if file_row is None:
            continue

        # No modo /read (return_full_content=False), sempre trunca o chunk
        # No modo full content (indexed-file/read), não trunca — preserva o original
        if return_full_content:
            text = file_row["content"]
        else:
            text = _truncate_text(chunk_row["chunk_text"], max_chars)

        entries.append(MemoryEntry(
            id           = fid,
            text         = text,
            score        = round(score, 4),
            confidence   = round(file_row["confidence"], 4),
            created_at   = file_row["created_at"],
            access_count = file_row["access_count"],
            memory_type  = "indexed_file",
            source       = file_row["source"],
            file_path    = file_row["file_path"],
            file_name    = file_row["file_name"],
            extension    = file_row["extension"],
            content_hash = file_row["content_hash"],
            file_hash    = file_row["file_hash"],
        ))
        loop.run_in_executor(None, state.if_db.update_access, fid)

    return entries


# ── POST /write ────────────────────────────────────────────────────────────────

# ── Locks de escrita (ITEM 5 da revisão) ───────────────────────────────────────
# Cada fluxo de escrita "check existe? → embed → dedup semântico → insert" tem
# um `await` (a chamada de embedding) no meio da checagem. Sem serializar essa
# sequência, duas escritas quase simultâneas do mesmo conteúdo podem passar as
# duas pela checagem de duplicata antes de qualquer uma delas inserir —
# resultando em duplicata (exact ou semântica) ou, no caso de `text_hash`/
# `person_key`/`concept_key` serem UNIQUE, uma exceção não tratada no insert.
# Um `asyncio.Lock` por fluxo elimina essa corrida sem serializar o processo
# inteiro (cada tipo de memória tem o seu).
_lt_write_lock = asyncio.Lock()
_st_write_lock = asyncio.Lock()
_vd_write_lock = asyncio.Lock()
_fd_write_lock = asyncio.Lock()
_vo_write_lock = asyncio.Lock()
_tl_write_lock = asyncio.Lock()


async def _store_long_term_text(
    text: str,
    source: str,
    confidence: float,
    forgettable: bool = True,
    # ── NEW (solução 3) ──
    ttl_days: Optional[float] = None,
    # ── NEW (solução 2) ──
    action: Literal["create", "update"] = "create",
    memory_id: Optional[int] = None,
    # ── NEW (arestas tipadas): True → o chamador já declarou (link_to +
    # link_type) que este texto é um fato DISTINTO relacionado a outro; a faixa
    # "possible_update" não bloqueia o write.
    allow_near: bool = False,
) -> tuple[bool, str, Optional[int], Optional[dict]]:
    """
    Lógica compartilhada de gravação em memória de longo prazo — usada tanto
    pelo endpoint /write (e /write_batch) quanto pelo /visual-dict/write
    (para persistir a descrição textual de um conceito visual como memória
    normal).
    Retorna (stored, reason, memory_id, candidate) — `candidate` só vem
    preenchido (dict com id/text/score) quando `reason` começa com
    "possible_update:".
    """
    text = text.strip()
    if len(text) < 10:
        return False, "too_short", None, None

    loop = asyncio.get_event_loop()

    # ── NEW (solução 2): fluxo de atualização/correção ─────────────────
    # Em vez de inserir um fato novo que conflita com um já existente, o
    # texto substitui o conteúdo de uma memória já gravada.
    if action == "update":
        async with _lt_write_lock:
            target_id = memory_id
            if target_id is None:
                # Sem memory_id explícito: acha a memória de LT mais
                # parecida semanticamente e usa ela como alvo, desde que
                # a similaridade seja alta o bastante para termos certeza
                # de que é "a mesma coisa, dita de outro jeito" — não
                # queremos "atualizar" um fato não relacionado por engano.
                #
                # O cosine do bi-encoder aqui é só RECALL (candidato a
                # checar); quem confirma é o reranker cross-encoder, que
                # não sofre do mesmo problema de anisotropia (ver
                # RERANK_UPDATE_SCORE / comentário perto de UPDATE_SIM_THRESHOLD).
                embedding = await state.embed_engine.embed_passage_one(text)
                hits = state.lt_index.search(embedding, top_k=1)
                if hits and hits[0][1] >= UPDATE_SIM_THRESHOLD:
                    candidate_id, candidate_sim = hits[0]
                    candidate_row = state.lt_db.get_by_id(candidate_id)
                    if candidate_row is not None:
                        cross_score = await state.rerank_engine.score_one(text, candidate_row["text"])
                        if cross_score >= RERANK_UPDATE_SCORE:
                            target_id = candidate_id
                        else:
                            log.info(
                                f"update sem memory_id: bi-encoder achou candidata #{candidate_id} "
                                f"(cosine={candidate_sim:.3f}) mas reranker discordou "
                                f"(cross_score={cross_score:.3f} < {RERANK_UPDATE_SCORE}) — ignorando"
                            )

            if target_id is None:
                return False, "update_target_not_found", None, None

            row = state.lt_db.get_by_id(target_id)
            if row is None:
                return False, "update_target_not_found", None, None

            ok = state.lt_db.update(
                target_id, text,
                source=source, confidence=confidence,
                forgettable=forgettable,
                ttl_days=ttl_days, ttl_days_set=True,
            )
            if not ok:
                return False, "update_conflict", None, None

            # O vetor antigo aponta pro texto anterior — remove e adiciona
            # de novo com o mesmo id, pra busca semântica continuar
            # refletindo o texto atual em vez do corrigido.
            new_embedding = await state.embed_engine.embed_passage_one(text)
            await loop.run_in_executor(None, state.lt_index.remove_ids, {target_id})
            await loop.run_in_executor(None, state.lt_index.add, new_embedding, target_id)

        log.info(f"LT #{target_id} atualizada: {text[:60]}")
        return True, "updated", target_id, None

    # ── Fluxo normal de criação ─────────────────────────────────────────
    async with _lt_write_lock:
        if state.lt_db.exists_exact(text):
            return False, "duplicate_exact", None, None

        embedding = await state.embed_engine.embed_passage_one(text)

        hits = state.lt_index.search(embedding, top_k=1)
        max_sim, nearest_id = (hits[0][1], hits[0][0]) if hits else (0.0, None)

        # ── Bi-encoder = recall (achar candidato), reranker = decisão ────────
        # Só vale a pena checar o cross-encoder se existe um candidato
        # plausível pelo cosine; abaixo de UPDATE_SIM_THRESHOLD nem chega
        # perto do piso de ruído do modelo, então é claramente um fato novo
        # e pulamos a chamada extra de rerank (mais rápido).
        if not allow_near and max_sim >= UPDATE_SIM_THRESHOLD and nearest_id is not None:
            candidate_row = state.lt_db.get_by_id(nearest_id)
            cross_score = None
            if candidate_row is not None:
                cross_score = await state.rerank_engine.score_one(text, candidate_row["text"])

            if cross_score is not None and cross_score >= RERANK_DUPLICATE_SCORE:
                return False, f"duplicate_semantic:{max_sim:.3f}:rerank={cross_score:.3f}", None, None

            if cross_score is not None and cross_score >= RERANK_UPDATE_SCORE:
                candidate = {
                    "id":    candidate_row["id"],
                    "text":  candidate_row["text"],
                    "score": round(max_sim, 4),
                    "rerank_score": round(cross_score, 4),
                }
                return False, f"possible_update:{max_sim:.3f}:rerank={cross_score:.3f}", None, candidate

            # cross_score baixo (ou candidate_row sumiu) → bi-encoder deu
            # falso positivo (anisotropia/domínio compartilhado); segue como
            # fato novo mesmo com cosine alto.
            if cross_score is not None:
                log.info(
                    f"LT: candidato #{nearest_id} descartado pelo reranker "
                    f"(cosine={max_sim:.3f} mas cross_score={cross_score:.3f} < {RERANK_UPDATE_SCORE}) "
                    f"— gravando como fato novo: {text[:60]}"
                )

        try:
            memory_id = state.lt_db.insert(text, source, confidence, forgettable, ttl_days)
        except sqlite3.IntegrityError:
            # Rede de segurança: mesmo com o lock, cobre o caso de outro
            # processo/writer ter inserido o mesmo texto entre o check e o
            # insert (ex.: dois workers do MCP).
            log.warning(f"LT: corrida de duplicata detectada no insert — '{text[:60]}'")
            return False, "duplicate_exact", None, None

        # add() faz uma cópia/realocação em C++ — pequena, mas offload pro
        # executor mantém o event loop livre mesmo sob concorrência alta.
        await loop.run_in_executor(None, state.lt_index.add, embedding, memory_id)

    log.info(f"LT #{memory_id} gravada: {text[:60]}")
    return True, "ok", memory_id, None


async def _link_new_memory(new_id: int, target_id: int, link_type: str) -> None:
    """Cria a aresta tipada entre uma memória recém-gravada e `target_id`.
    Direção: updates = nova → antiga; temporal_precedence = `target_id` (veio
    antes) → nova; contradicts/co_activation são simétricas."""
    loop = asyncio.get_event_loop()
    try:
        if await loop.run_in_executor(None, state.lt_db.get_by_id, target_id) is None:
            log.warning(f"link_to #{target_id} não existe — aresta {link_type} não criada")
            return
        src, dst = (target_id, new_id) if link_type == EDGE_TEMPORAL else (new_id, target_id)
        await loop.run_in_executor(
            None, state.lt_db.upsert_edge, src, dst, EDGE_LEARNING_RATE, time.time(), link_type,
        )
    except Exception as e:
        log.error(f"Grafo: falha ao criar aresta {link_type} #{new_id}→#{target_id}: {e}")


async def _process_write_request(req: WriteRequest) -> WriteResponse:
    """Ponto único usado tanto por /write quanto por /write_batch (solução 1),
    pra garantir que os dois caminhos tenham exatamente a mesma lógica de
    dedup/update/ttl."""
    if (req.link_to is None) != (req.link_type is None):
        log.warning("write: link_to e link_type devem vir juntos — ligação ignorada")
    linking = req.link_to is not None and req.link_type is not None
    stored, reason, memory_id, candidate = await _store_long_term_text(
        req.text, req.source, req.confidence, req.forgettable,
        ttl_days=req.ttl_days, action=req.action, memory_id=req.memory_id,
        allow_near=linking,
    )
    if linking and stored and reason == "ok" and memory_id is not None:
        await _link_new_memory(memory_id, req.link_to, req.link_type)
    resp = WriteResponse(stored=stored, reason=reason, memory_id=memory_id)
    if candidate is not None:
        resp.candidate_id    = candidate["id"]
        resp.candidate_text  = candidate["text"]
        resp.candidate_score = candidate["score"]
        resp.candidate_rerank_score = candidate.get("rerank_score")
    return resp


async def write_memory(req: WriteRequest):
    return await _process_write_request(req)


# ── POST /write_batch ──────────────────────────────────────────────────────
# NEW (solução 1): grava vários WriteRequest numa chamada só, em vez de N
# chamadas separadas ao /write. Cada item é processado com a mesma lógica de
# _process_write_request (dedup/update/ttl inclusos) — um item que falha
# (ex.: duplicata) não impede os demais de serem processados.

async def write_memory_batch(req: WriteBatchRequest):
    results: list[WriteResponse] = []
    loop = asyncio.get_event_loop()
    prev_id: Optional[int] = None
    for item in req.items:
        resp = await _process_write_request(item)
        results.append(resp)
        # ── NEW: batch sequencial → aresta temporal_precedence (anterior → seguinte)
        if req.sequential and resp.stored and resp.memory_id is not None:
            if prev_id is not None and prev_id != resp.memory_id:
                await loop.run_in_executor(
                    None, state.lt_db.upsert_edge, prev_id, resp.memory_id,
                    EDGE_LEARNING_RATE, time.time(), EDGE_TEMPORAL,
                )
            prev_id = resp.memory_id
    stored_count = sum(1 for r in results if r.stored)
    log.info(f"write_batch: {stored_count}/{len(results)} memórias gravadas")
    return WriteBatchResponse(
        results=results, stored_count=stored_count, total=len(results),
    )


# ── POST /write_st ─────────────────────────────────────────────────────────────

async def write_short_term(req: WriteSTRequest):
    if not req.turns:
        return WriteSTResponse(stored=False, reason="no_turns")

    all_text = " ".join(t.content.strip() for t in req.turns)
    if len(all_text) < 10:
        return WriteSTResponse(stored=False, reason="too_short")

    embed_text = "\n".join(f"{t.role}: {t.content}" for t in req.turns)
    loop = asyncio.get_event_loop()

    async with _st_write_lock:
        embedding = await state.embed_engine.embed_passage_one(embed_text)

        max_sim = state.st_index.search_similar(embedding)
        if max_sim >= DEDUP_THRESHOLD:
            return WriteSTResponse(stored=False, reason=f"duplicate_semantic:{max_sim:.3f}")

        group_id = state.st_db.insert(req.session_id, req.turns, embed_text)
        await loop.run_in_executor(None, state.st_index.add, embedding, group_id)

    log.info(f"ST #{group_id} gravado — session={req.session_id} turnos={len(req.turns)}: {embed_text[:80]}")
    return WriteSTResponse(stored=True, reason="ok", turn_ids=[group_id])


# ── POST /read_st ────────────────────────────────────────────────────────────
# Leitura crua do short-term: só as N duplas pergunta-resposta mais recentes
# de uma sessão, sem embeddings/scoring — pensado para o LLM montar o
# histórico de conversa como contexto (mais leve/rápido que /read).

async def read_short_term(req: ReadSTRequest):
    session_id = req.session_id.strip()
    if not session_id:
        raise MemoryToolError("session_id vazio")

    n_pairs = req.n_pairs if req.n_pairs > 0 else ST_READ_DEFAULT_PAIRS
    loop = asyncio.get_event_loop()
    raw_turns, groups_fetched = await loop.run_in_executor(
        None, state.st_db.get_recent_turn_groups, session_id, n_pairs
    )
    turns = [Turn(**t) for t in raw_turns]

    log.info(f"/read_st session={session_id} n_pairs={n_pairs} grupos={groups_fetched} turnos={len(turns)}")
    return ReadSTResponse(session_id=session_id, turns=turns, pairs_returned=groups_fetched)


# ── POST /read ─────────────────────────────────────────────────────────────────

async def read_memory(req: ReadRequest):
    query = req.query.strip()
    if not query:
        raise MemoryToolError("query vazia")

    loop = asyncio.get_event_loop()
    effective_strategy = "none"

    query_emb = await state.embed_engine.embed_query_one(query)

    # Start VS search in background
    vs_future = None
    if state.vs is not None and state.vs.total > 0:
        vs_min_score = min(req.min_score, VS_MIN_SCORE) if req.min_score < VS_MIN_SCORE else VS_MIN_SCORE
        vs_future = asyncio.ensure_future(
            loop.run_in_executor(
                None,
                state.vs.search,
                query_emb,
                req.top_k,
                vs_min_score,
            )
        )

    # ── NEW: Start indexed files search in background ──
    if_future = None
    if state.if_index.total > 0:
        if_future = asyncio.ensure_future(
            loop.run_in_executor(None, state.if_index.search, query_emb, req.top_k * 2)
        )

    if req.session_id and req.strategy != "none":
        context_block = await loop.run_in_executor(None, _build_context_block, req.session_id)

        if context_block:
            if req.strategy == "auto":
                effective_strategy = _classify_query(query)
            elif req.strategy in ("expanded", "dual"):
                effective_strategy = req.strategy
            else:
                log.warning(f"Estratégia desconhecida '{req.strategy}', usando 'auto'")
                effective_strategy = _classify_query(query)

            if effective_strategy == "expanded":
                lt_raw, st_raw = await _search_expanded(query, context_block, req.top_k)
            else:
                lt_raw, st_raw = await _search_dual(query, context_block, req.top_k)

            log.info(
                f"/read session={req.session_id} strategy={effective_strategy} "
                f"ctx_chars={len(context_block)} query='{query[:60]}'"
            )
        else:
            effective_strategy = "none"
            lt_raw, st_raw = await asyncio.gather(
                loop.run_in_executor(None, state.lt_index.search, query_emb, req.top_k * 2),
                loop.run_in_executor(None, state.st_index.search, query_emb, req.top_k * 2),
            )
    else:
        lt_raw, st_raw = await asyncio.gather(
            loop.run_in_executor(None, state.lt_index.search, query_emb, req.top_k * 2),
            loop.run_in_executor(None, state.st_index.search, query_emb, req.top_k * 2),
        )

    # ── NEW: query composta → busca adicional por segmento, sem LLM ───────────
    # Se a pergunta tem vários sub-pedidos numa frase só (ex.: "quero fazer X
    # usando Y para conseguir Z"), a busca acima (query inteira) tende a
    # trazer só o que é mais "central" na frase. Aqui cada sub-tópico é
    # embedado/buscado à parte e o resultado é fundido por max-score — sem
    # afogar um sub-tópico no outro nem gerar custo de LLM.
    query_segments = _split_query_segments(query)
    if len(query_segments) > 1:
        lt_seg, st_seg = await _search_segmented(query_segments, req.top_k)
        lt_raw = _fuse_max(lt_raw, lt_seg)
        st_raw = _fuse_max(st_raw, st_seg)
        log.info(f"/read query segmentada em {len(query_segments)} partes: {query_segments}")

    # ── NEW (Hebbian graph / Step 6): PPR no grafo, com sementes em lt_raw ──
    # lt_raw já está finalizado aqui (todas as estratégias convergem nele) e
    # a conversão pra MemoryEntry ainda não aconteceu — posição exata do Step 6.
    ppr_ctx = await _ppr_spread(lt_raw, loop)
    related_raw: list[tuple[int, float]] = ppr_ctx["related_raw"]
    related_ids = {mid for mid, _ in related_raw}

    # Await VS results
    vs_results = []
    if vs_future is not None:
        try:
            vs_results = await vs_future
        except Exception as e:
            log.error(f"VectorStore search failed: {e}")
            vs_results = []

    # ── NEW: Await indexed files results ──
    if_raw = []
    if if_future is not None:
        try:
            if_raw = await if_future
        except Exception as e:
            log.error(f"Indexed files search failed: {e}")
            if_raw = []

    # ── Otimização de tokens ───────────────────────────────────────────────────
    # 1. Threshold mais seletivo para /read (combinação de 4 fontes gera ruído)
    #    Usa max(req.min_score, READ_MIN_SCORE_STRICT) — sempre >= 0.85
    # 2. Corrige bug do IF_MIN_SCORE: usar max (mais seletivo) não min
    #    Antes: min(req.min_score, IF_MIN_SCORE) → retornava chunks com score 0.75
    #    Agora: max(req.min_score, IF_MIN_SCORE_READ) → >= 0.82
    strict_min_score = max(req.min_score, READ_MIN_SCORE_STRICT)
    if_strict_min_score = max(strict_min_score, IF_MIN_SCORE_READ)

    results: list[MemoryEntry] = (
        _build_lt_entries(lt_raw, strict_min_score, loop) +
        # "related" tem threshold próprio: score PPR é massa de probabilidade
        # (<= PPR_SPREAD_WEIGHT), jamais passaria no strict_min_score (0.85).
        # O piso real de elegibilidade é PPR_MIN_ACTIVATION_REINFORCE, já
        # aplicado dentro de _ppr_spread no próprio rank.
        _build_lt_entries(related_raw, 0.0, loop, related_ids=related_ids) +
        _build_st_entries(st_raw, strict_min_score, loop) +
        _build_vs_entries(vs_results, strict_min_score) +
        _build_if_entries(if_raw, if_strict_min_score, loop)  # NEW
    )
    results.sort(key=lambda r: r.score * r.confidence, reverse=True)

    # ── NEW (Step 7): inibição lateral (MMR) SÓ no subconjunto "related",
    # antes do orçamento de tokens — primary não é penalizado. ──
    results = await loop.run_in_executor(None, _lateral_inhibition_filter, results)

    # ── Orçamento global de tokens ─────────────────────────────────────────────
    # Limita o total de caracteres retornados, cortando entradas de menor score.
    # Sempre retorna pelo menos 1 entrada se existir.
    final_top_k = min(req.top_k, READ_TOP_K_FINAL) if req.top_k > 0 else READ_TOP_K_FINAL
    results = _apply_token_budget(results, READ_TOTAL_MAX_CHARS, final_top_k)

    # Log de diagnóstico (nível INFO para acompanhar redução no pipeline)
    total_chars = sum(len(r.text or "") for r in results)
    log.info(
        f"/read query='{query[:50]}' strategy={effective_strategy} "
        f"results={len(results)} total_chars={total_chars} "
        f"budget={READ_TOTAL_MAX_CHARS} strict_score={strict_min_score:.2f} "
        f"ppr_ms={ppr_ctx['elapsed_ms']:.1f} related={len(related_raw)}"
    )

    response = ReadResponse(results=results, query=query, strategy=effective_strategy)

    # ── NEW (Step 4, background): criação de arestas hebbianas entre as
    # memórias LT FINAIS com score >= EDGE_MIN_SCORE_TO_LINK. Fire-and-forget
    # — o caller já recebeu a resposta; falhas só logam. ──
    asyncio.ensure_future(_hebbian_link_task(results))
    # ── NEW (Step 8, background): reforço das arestas EXISTENTES do subgrafo
    # PPR pela regra de co-ativação (Opção B). Também fire-and-forget. ──
    asyncio.ensure_future(_ppr_reinforce_task(ppr_ctx))

    return response


# ── DELETE /session/{session_id} ───────────────────────────────────────────────

async def clear_session(session_id: str):
    ids_to_remove = set(state.st_db.get_ids_by_session(session_id))

    if not ids_to_remove:
        return {"cleared": 0, "session_id": session_id}

    state.st_db.delete_by_session(session_id)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, state.st_index.remove_ids, ids_to_remove)
    log.info(f"Sessão {session_id}: {len(ids_to_remove)} grupos removidos")
    return {"cleared": len(ids_to_remove), "session_id": session_id}


# ══════════════════════════════════════════════════════════════════════════════
# NEW: Arquivos Indexados — /indexed-file/*
# ══════════════════════════════════════════════════════════════════════════════

async def indexed_file_write(req: IndexedFileWriteRequest):
    """
    Armazena o conteúdo COMPLETO de um arquivo indexado.

    Fluxo:
      1. Verifica se o arquivo já está indexado (por file_path)
      2. Se existe e file_hash é igual → sem reindexação necessária
      3. Se existe e file_hash difere → remove chunks antigos, reindexa
      4. Se não existe → insere e indexa

    O conteúdo é dividido em chunks, cada chunk é embedado e
    adicionado ao FAISS para busca semântica. O hash do arquivo
    é armazenado para detectar mudanças futuras.
    """
    content = req.content
    if not content or not content.strip():
        return IndexedFileWriteResponse(stored=False, reason="content_empty")

    if len(content) > IF_MAX_CONTENT_SIZE:
        return IndexedFileWriteResponse(
            stored=False,
            reason=f"content_too_large:{len(content)}>{IF_MAX_CONTENT_SIZE}",
        )

    content_hash = hashlib.sha256(content.encode()).hexdigest()
    loop = asyncio.get_event_loop()

    # ── Check existing ──
    existing = state.if_db.get_by_path(req.file_path)

    if existing and not req.force_reindex:
        if existing["file_hash"] == req.file_hash and existing["content_hash"] == content_hash:
            # Arquivo inalterado — nada a fazer
            log.info(f"Indexed file unchanged: {req.file_path} (hash_match=True)")
            return IndexedFileWriteResponse(
                stored=False,
                reason="unchanged",
                file_id=existing["id"],
                chunks_created=0,
                was_reindexed=False,
                hash_match=True,
            )

    # ── Remove old chunks if re-indexing ──
    if existing:
        old_chunk_ids = state.if_db.get_chunk_ids_by_file(existing["id"])
        if old_chunk_ids:
            await loop.run_in_executor(None, state.if_index.remove_ids, set(old_chunk_ids))
            state.if_db.delete_chunks_by_file(existing["id"])
            log.info(f"Removed {len(old_chunk_ids)} old chunks for: {req.file_path}")

    # ── Chunk the content ──
    chunks = _chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        return IndexedFileWriteResponse(stored=False, reason="chunking_failed")

    # ── Embed chunks in batches ──
    chunk_texts = [c["text"] for c in chunks]
    all_embeddings = []

    for batch_start in range(0, len(chunk_texts), IF_EMBED_BATCH_SIZE):
        batch = chunk_texts[batch_start:batch_start + IF_EMBED_BATCH_SIZE]
        try:
            batch_embs = await state.embed_engine.embed_passages(batch)
            all_embeddings.append(batch_embs)
        except Exception as e:
            log.error(f"Embedding batch failed for {req.file_path}: {e}")
            return IndexedFileWriteResponse(
                stored=False,
                reason=f"embedding_failed:{e}",
            )

    embeddings = np.vstack(all_embeddings) if all_embeddings else np.empty((0, EMBED_DIM), dtype=np.float32)

    # ── Insert or update file record ──
    was_reindexed = existing is not None
    if existing:
        file_id = existing["id"]
        state.if_db.update_file(
            file_id=file_id,
            content=content,
            content_hash=content_hash,
            file_hash=req.file_hash,
            size=req.size or len(content),
            modified=req.modified,
        )
    else:
        file_id = state.if_db.insert_file(
            file_path=req.file_path,
            file_name=req.file_name,
            extension=req.extension,
            content=content,
            content_hash=content_hash,
            file_hash=req.file_hash,
            size=req.size or len(content),
            modified=req.modified,
            source=req.source,
            confidence=req.confidence,
        )

    # ── Insert chunks into DB ──
    state.if_db.insert_chunks(file_id, chunks)

    # ── Get chunk IDs (just inserted) ──
    chunk_rows = state.if_db.get_chunks_by_file(file_id)
    chunk_ids = [row["id"] for row in chunk_rows]

    if len(chunk_ids) != len(chunks):
        log.warning(
            f"Chunk ID mismatch: expected {len(chunks)}, got {len(chunk_ids)} "
            f"for {req.file_path}"
        )

    # ── Add embeddings to FAISS in batch ──
    if len(chunk_ids) == embeddings.shape[0]:
        await loop.run_in_executor(
            None,
            state.if_index.add_batch,
            embeddings,
            chunk_ids,
        )
    else:
        # Fallback: add one by one if sizes don't match
        log.warning("Chunk/embedding size mismatch — adding individually")
        for emb, cid in zip(embeddings, chunk_ids):
            await loop.run_in_executor(None, state.if_index.add, emb, cid)

    hash_match = (
        existing is not None
        and existing["file_hash"] == req.file_hash
        and existing["content_hash"] == content_hash
    )

    log.info(
        f"Indexed file: {req.file_path} → file_id={file_id} "
        f"chunks={len(chunks)} reindexed={was_reindexed} "
        f"hash_match={hash_match} content_size={len(content)}"
    )

    return IndexedFileWriteResponse(
        stored=True,
        reason="ok" if not was_reindexed else "reindexed",
        file_id=file_id,
        chunks_created=len(chunks),
        was_reindexed=was_reindexed,
        hash_match=hash_match,
    )


async def indexed_file_read(req: IndexedFileReadRequest):
    """
    Lê arquivos indexados. Três modos, mutuamente exclusivos:

      1. `file_path` sozinho → lookup EXATO por caminho absoluto (mesma
         chave usada em `/indexed-file/write`). Não passa pelo FAISS —
         é uma busca direta no SQLite por igualdade de `file_path`, o que
         garante que dois arquivos com o mesmo nome em pastas diferentes
         nunca sejam confundidos. Retorna o arquivo INTEIRO (1 resultado).

      2. `file_path` + `query` → mesmo lookup exato por caminho, mas em vez
         de devolver o arquivo inteiro de uma vez, ranqueia as chunks DESSE
         MESMO ARQUIVO (e só dele — não compara com outros arquivos do
         índice) contra `query` e devolve até `top_k` chunks mais
         relevantes, cada uma com seu `chunk_text` e `score`.

      3. Nem um nem outro, só `query` → busca semântica original entre
         TODOS os arquivos indexados, retornando até `top_k` arquivos
         cujos chunks tiveram similaridade acima do threshold.
    """
    file_path = (req.file_path or "").strip()
    query     = (req.query or "").strip()

    # ── Modo 1/2: caminho absoluto informado ──
    if file_path:
        row = state.if_db.get_by_path(file_path)
        if row is None:
            return IndexedFileReadResponse(results=[], file_path=file_path, query=query or None)

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, state.if_db.update_access, row["id"])

        # ── Modo 2: sem query → devolve o arquivo inteiro (comportamento original) ──
        if not query:
            entry = IndexedFileEntry(
                file_id      = row["id"],
                file_path    = row["file_path"],
                file_name    = row["file_name"],
                extension    = row["extension"],
                content      = row["content"],           # conteúdo COMPLETO
                file_hash    = row["file_hash"],
                content_hash = row["content_hash"],
                size         = row["size"],
                modified     = row["modified"],
                score        = 1.0,                       # lookup exato — sem score de similaridade
                confidence   = round(row["confidence"], 4),
                created_at   = row["created_at"],
                access_count = row["access_count"],
                source       = row["source"],
                chunk_text   = None,
                match_type   = "exact_path",
            )
            return IndexedFileReadResponse(results=[entry], file_path=file_path)

        # ── Modo 2b: com query → top_k chunks ranqueadas, restritas a este arquivo ──
        chunk_rows = state.if_db.get_chunks_by_file(row["id"])
        chunk_ids  = [c["id"] for c in chunk_rows]
        if not chunk_ids:
            return IndexedFileReadResponse(results=[], file_path=file_path, query=query)

        query_emb = await state.embed_engine.embed_query_one(query)
        ranked = await loop.run_in_executor(
            None, state.if_index.search_subset, query_emb, chunk_ids, req.top_k
        )
        ranked = [(cid, score) for cid, score in ranked if score >= req.min_score]
        if not ranked:
            return IndexedFileReadResponse(results=[], file_path=file_path, query=query)

        chunk_by_id = {c["id"]: c for c in chunk_rows}
        full_content = row["content"] if req.include_full_content else ""
        results = []
        for cid, score in ranked:
            chunk_row = chunk_by_id.get(cid)
            if chunk_row is None:
                continue
            results.append(IndexedFileEntry(
                file_id      = row["id"],
                file_path    = row["file_path"],
                file_name    = row["file_name"],
                extension    = row["extension"],
                content      = full_content,              # arquivo completo — vazio se include_full_content=False
                file_hash    = row["file_hash"],
                content_hash = row["content_hash"],
                size         = row["size"],
                modified     = row["modified"],
                score        = round(score, 4),
                confidence   = round(row["confidence"], 4),
                created_at   = row["created_at"],
                access_count = row["access_count"],
                source       = row["source"],
                chunk_text   = chunk_row["chunk_text"],   # a chunk específica ranqueada
                chunk_index  = chunk_row["chunk_index"],
                char_start   = chunk_row["char_start"],
                char_end     = chunk_row["char_end"],
                chunk_id = chunk_row["id"],
                match_type   = "exact_path_chunks",
            ))
        return IndexedFileReadResponse(results=results, file_path=file_path, query=query)

    # ── Modo 3: busca semântica entre todos os arquivos ──
    if not query:
        raise MemoryToolError("informe 'file_path' (leitura exata) ou 'query' (busca semântica)")

    if state.if_index.total == 0:
        return IndexedFileReadResponse(results=[], query=query)

    loop = asyncio.get_event_loop()
    query_emb = await state.embed_engine.embed_query_one(query)

    if_raw = await loop.run_in_executor(
        None, state.if_index.search, query_emb, req.top_k * 2
    )

    # Filtra por score
    filtered = [(cid, score) for cid, score in if_raw if score >= req.min_score]
    if not filtered:
        return IndexedFileReadResponse(results=[], query=query)

    # Busca chunk records
    chunk_ids = [cid for cid, _ in filtered]
    chunk_rows = state.if_db.get_chunks_by_ids(chunk_ids)
    chunk_map = {row["id"]: row for row in chunk_rows}

    # Agrupa por file_id — melhor score por arquivo
    file_best: dict[int, tuple[float, sqlite3.Row]] = {}
    for cid, score in filtered:
        chunk_row = chunk_map.get(cid)
        if chunk_row is None:
            continue
        fid = chunk_row["file_id"]
        if fid not in file_best or score > file_best[fid][0]:
            file_best[fid] = (score, chunk_row)

    if not file_best:
        return IndexedFileReadResponse(results=[], query=query)

    # Busca file records
    file_ids = list(file_best.keys())
    file_rows = state.if_db._conn.execute(
        f"SELECT * FROM indexed_files WHERE id IN ({','.join('?' * len(file_ids))})",
        file_ids,
    ).fetchall()
    file_map = {row["id"]: row for row in file_rows}

    results = []
    for fid, (score, chunk_row) in sorted(file_best.items(), key=lambda x: x[1][0], reverse=True):
        file_row = file_map.get(fid)
        if file_row is None:
            continue

        results.append(IndexedFileEntry(
            file_id      = fid,
            file_path    = file_row["file_path"],
            file_name    = file_row["file_name"],
            extension    = file_row["extension"],
            content      = file_row["content"],          # conteúdo COMPLETO
            file_hash    = file_row["file_hash"],
            content_hash = file_row["content_hash"],
            size         = file_row["size"],
            modified     = file_row["modified"],
            score        = round(score, 4),
            confidence   = round(file_row["confidence"], 4),
            created_at   = file_row["created_at"],
            access_count = file_row["access_count"],
            source       = file_row["source"],
            chunk_text   = chunk_row["chunk_text"],       # chunk que deu match
            chunk_index  = chunk_row["chunk_index"],
            char_start   = chunk_row["char_start"],
            char_end     = chunk_row["char_end"],
        ))
        loop.run_in_executor(None, state.if_db.update_access, fid)

    return IndexedFileReadResponse(results=results[:req.top_k], query=query)


async def indexed_file_check(file_path: str):
    """
    Verifica se um arquivo está indexado e se o hash bate.

    Usado pelo local-scraping para decidir se precisa reindexar:
      - indexed=False  → arquivo nunca foi indexado
      - hash_match=True → já indexado e inalterado
      - hash_match=False → indexado mas arquivo mudou → reindexar
    """
    if not file_path:
        raise MemoryToolError("file_path vazio")

    row = state.if_db.get_by_path(file_path)
    if row is None:
        return IndexedFileCheckResponse(indexed=False)

    chunks_count = state.if_db.count_chunks(row["id"])

    return IndexedFileCheckResponse(
        indexed            = True,
        file_id            = row["id"],
        stored_file_hash   = row["file_hash"],
        stored_content_hash = row["content_hash"],
        stored_modified    = row["modified"],
        chunks_count       = chunks_count,
        hash_match         = None,  # caller compara com o hash atual
    )


async def indexed_file_get(file_id: int):
    """
    Retorna o conteúdo completo de um arquivo indexado pelo seu ID.
    """
    row = state.if_db.get_by_id(file_id)
    if row is None:
        raise MemoryToolError(f"Arquivo indexado #{file_id} não encontrado")

    chunks = state.if_db.get_chunks_by_file(file_id)

    return {
        "file_id":      row["id"],
        "file_path":    row["file_path"],
        "file_name":    row["file_name"],
        "extension":    row["extension"],
        "content":      row["content"],
        "content_hash": row["content_hash"],
        "file_hash":    row["file_hash"],
        "size":         row["size"],
        "modified":     row["modified"],
        "source":       row["source"],
        "confidence":   row["confidence"],
        "created_at":   row["created_at"],
        "access_count": row["access_count"],
        "chunks_count": len(chunks),
    }



async def get_chunk_by_id(chunk_id: int):
    row = state.if_db._conn.execute(
        "SELECT c.*, f.file_path, f.file_hash, f.content "
        "FROM indexed_file_chunks c "
        "JOIN indexed_files f ON c.file_id = f.id "
        "WHERE c.id = ?", (chunk_id,)
    ).fetchone()
    if row is None:
        raise MemoryToolError(f"Chunk #{chunk_id} não encontrado")
    return {
        "chunk_id": row["id"],
        "file_path": row["file_path"],
        "file_hash": row["file_hash"],
        "char_start": row["char_start"],
        "char_end": row["char_end"],
        "chunk_text": row["chunk_text"],
        "file_content": row["content"],  # conteúdo completo para validação
    }


async def indexed_file_delete(file_id: int):
    """
    Remove um arquivo indexado e todos os seus chunks (DB + FAISS).
    """
    row = state.if_db.get_by_id(file_id)
    if row is None:
        raise MemoryToolError(f"Arquivo indexado #{file_id} não encontrado")

    # Remove FAISS vectors first
    chunk_ids = state.if_db.get_chunk_ids_by_file(file_id)
    if chunk_ids:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, state.if_index.remove_ids, set(chunk_ids))

    # Delete from DB (cascades to chunks)
    deleted = state.if_db.delete_file(file_id)

    log.info(f"Indexed file deleted: #{file_id} ({len(chunk_ids)} chunks removed)")
    return {"deleted": deleted, "file_id": file_id, "chunks_removed": len(chunk_ids)}


async def indexed_file_delete_by_path(file_path: str):
    """
    Remove um arquivo indexado pelo caminho.
    """
    if not file_path:
        raise MemoryToolError("file_path vazio")

    row = state.if_db.get_by_path(file_path)
    if row is None:
        return {"deleted": 0, "file_path": file_path, "message": "not indexed"}

    return await indexed_file_delete(row["id"])


async def indexed_file_list():
    """
    Lista todos os arquivos indexados com metadados.
    """
    rows = state.if_db.list_files()
    files = []
    for row in rows:
        files.append({
            "file_id":      row["id"],
            "file_path":    row["file_path"],
            "file_name":    row["file_name"],
            "extension":    row["extension"],
            "content_hash": row["content_hash"],
            "file_hash":    row["file_hash"],
            "size":         row["size"],
            "modified":     row["modified"],
            "created_at":   row["created_at"],
            "access_count": row["access_count"],
        })
    return {"total": len(files), "files": files}


# ── NEW: Dicionário Visual — endpoints usados pelo vision.py ──────────────────
#
# vision.py NÃO persiste nada localmente: ele roda o pipeline (depth →
# clustering → segmentação → embeddings) e manda cada embedding de crop pra
# cá. Este arquivo decide se é um conceito novo ou um exemplo a mais de um
# conceito já existente, e é quem guarda tudo (FAISS + SQLite + memória de
# longo prazo).

async def visual_dict_write(req: VisualDictWriteRequest):
    concept_name = req.concept_name.strip()
    description  = req.description.strip()
    if not concept_name:
        return VisualDictWriteResponse(stored=False, reason="empty_concept_name")
    if len(req.embedding) != VD_EMBED_DIM:
        raise MemoryToolError(f"embedding deve ter dimensão {VD_EMBED_DIM}, recebido {len(req.embedding)}")

    vec = np.asarray(req.embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm

    loop = asyncio.get_event_loop()

    async with _vd_write_lock:
        existing = state.vd_db.get_by_name(concept_name)
        new_concept = existing is None

        if existing is not None:
            concept_id = existing["id"]
            memory_id  = existing["memory_id"]
        else:
            memory_id = None
            if req.link_to_memory and description:
                _, _, memory_id, _ = await _store_long_term_text(
                    f"{concept_name}: {description}", "visual_dict", req.confidence,
                )
            try:
                concept_id = state.vd_db.insert_concept(
                    concept_name, description, req.source, req.confidence, memory_id,
                )
            except sqlite3.IntegrityError:
                # Corrida: outro write criou o mesmo concept_key nesse meio-tempo.
                existing = state.vd_db.get_by_name(concept_name)
                if existing is None:
                    raise
                concept_id, memory_id, new_concept = existing["id"], existing["memory_id"], False
            else:
                log.info(f"Visual-dict: novo conceito #{concept_id} criado — '{concept_name}'")

        embedding_id = state.vd_db.insert_embedding(concept_id)
        await loop.run_in_executor(None, state.vd_index.add, vec, embedding_id)

    log.info(
        f"Visual-dict: embedding #{embedding_id} gravado p/ conceito #{concept_id} "
        f"({'novo' if new_concept else 'exemplo adicional'})"
    )

    return VisualDictWriteResponse(
        stored=True, reason="ok", concept_id=concept_id,
        embedding_id=embedding_id, memory_id=memory_id, new_concept=new_concept,
    )


async def visual_dict_read(req: VisualDictReadRequest):
    if len(req.embedding) != VD_EMBED_DIM:
        raise MemoryToolError(f"embedding deve ter dimensão {VD_EMBED_DIM}, recebido {len(req.embedding)}")

    vec = np.asarray(req.embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm

    # sobre-amostra o kNN porque vários embeddings podem apontar pro mesmo
    # conceito (múltiplos exemplos) — precisamos deduplicar por concept_id
    hits = state.vd_index.search(vec, max(req.top_k * 4, 20))

    best_score_by_concept: dict[int, float] = {}
    for embedding_id, score in hits:
        concept_id = state.vd_db.get_concept_id_by_embedding(embedding_id)
        if concept_id is None:
            continue
        if concept_id not in best_score_by_concept or score > best_score_by_concept[concept_id]:
            best_score_by_concept[concept_id] = score

    ranked = sorted(best_score_by_concept.items(), key=lambda kv: kv[1], reverse=True)[:req.top_k]

    results: list[VisualDictCandidate] = []
    for concept_id, score in ranked:
        if score < req.min_score:
            continue
        row = state.vd_db.get_concept_by_id(concept_id)
        if row is None:
            continue
        state.vd_db.update_access(concept_id)
        results.append(VisualDictCandidate(
            concept_id=row["id"],
            concept_name=row["concept_name"],
            description=row["description"],
            score=score,
            confidence=row["confidence"],
            access_count=row["access_count"] + 1,
            memory_id=row["memory_id"],
        ))

    # ambíguo quando: nenhum resultado confiável, OU os dois melhores
    # candidatos estão muito próximos (o objeto pode ser qualquer um dos dois)
    ambiguous = (
        len(results) == 0
        or (len(results) > 1 and (results[0].score - results[1].score) < VD_AMBIGUOUS_MARGIN)
    )

    # ── NEW: busca por grafo a partir dos hits (memory_id = nó no grafo Hebbiano)
    related: list[RelatedMemory] = []
    if req.include_related:
        related = await _dict_graph_related(
            [(c.memory_id, c.score) for c in results], asyncio.get_event_loop()
        )

    return VisualDictReadResponse(results=results, ambiguous=ambiguous, related=related)


async def visual_dict_list():
    rows = state.vd_db.list_concepts()
    concepts = [
        VisualDictEntry(
            concept_id=row["id"],
            concept_name=row["concept_name"],
            description=row["description"],
            source=row["source"],
            confidence=row["confidence"],
            memory_id=row["memory_id"],
            examples_count=row["examples_count"],
            created_at=row["created_at"],
            access_count=row["access_count"],
        ).model_dump()
        for row in rows
    ]
    return {"total": len(concepts), "concepts": concepts}


async def visual_dict_get(concept_id: int):
    row = state.vd_db.get_concept_by_id(concept_id)
    if row is None:
        raise MemoryToolError(f"Conceito visual #{concept_id} não encontrado")
    return VisualDictEntry(
        concept_id=row["id"],
        concept_name=row["concept_name"],
        description=row["description"],
        source=row["source"],
        confidence=row["confidence"],
        memory_id=row["memory_id"],
        examples_count=state.vd_db.count_examples(concept_id),
        created_at=row["created_at"],
        access_count=row["access_count"],
    )


async def visual_dict_delete(concept_id: int):
    """Remove um conceito visual e todos os seus embeddings (DB + FAISS)."""
    row = state.vd_db.get_concept_by_id(concept_id)
    if row is None:
        raise MemoryToolError(f"Conceito visual #{concept_id} não encontrado")

    embedding_ids = state.vd_db.get_embedding_ids_by_concept(concept_id)
    if embedding_ids:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, state.vd_index.remove_ids, set(embedding_ids))

    deleted = state.vd_db.delete_concept(concept_id)
    log.info(f"Visual-dict: conceito #{concept_id} removido ({len(embedding_ids)} embeddings)")
    return {"deleted": deleted, "concept_id": concept_id, "embeddings_removed": len(embedding_ids)}


# ── NEW: Dicionário de Rostos — endpoints usados pelo vision.py ───────────────
#
# Mesma lógica do dicionário visual acima, adaptada pra reconhecimento
# facial: vision.py detecta+alinha o rosto, extrai o embedding (EdgeFace) e
# manda pra cá. Aqui decide-se se é uma pessoa nova ou mais um exemplo de
# uma pessoa já cadastrada.

async def face_dict_write(req: FaceDictWriteRequest):
    person_name = req.person_name.strip()
    if not person_name:
        return FaceDictWriteResponse(stored=False, reason="empty_person_name")

    if len(req.embedding) != FD_EMBED_DIM:
        raise MemoryToolError(f"embedding deve ter dimensão {FD_EMBED_DIM}, recebido {len(req.embedding)}")

    vec = np.asarray(req.embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm

    description = req.description.strip()
    loop = asyncio.get_event_loop()

    async with _fd_write_lock:
        existing = state.fd_db.get_by_name(person_name)
        new_person = existing is None

        memory_id: Optional[int] = None
        if existing is not None:
            person_id = existing["id"]
            memory_id = existing["memory_id"]
            # se veio uma descrição não-vazia num cadastro de exemplo adicional,
            # atualiza/completa a descrição já salva (permite corrigir depois)
            if description:
                state.fd_db.update_description(person_id, description)
                # NEW: pessoa já cadastrada SEM link + link pedido → "promove"
                if req.link_to_memory and memory_id is None:
                    stored, _, memory_id, _ = await _store_long_term_text(
                        f"{person_name}: {description}", "face_dict", req.confidence,
                    )
                    if stored:
                        state.fd_db.set_memory_id(person_id, memory_id)
                    else:
                        memory_id = None
        else:
            # NEW (opt-in): descrição também vira memória LT → nó no grafo Hebbiano
            if req.link_to_memory and description:
                stored, _, memory_id, _ = await _store_long_term_text(
                    f"{person_name}: {description}", "face_dict", req.confidence,
                )
                if not stored:
                    memory_id = None  # dedup/erro — pessoa cadastrada mesmo assim, sem link
            try:
                person_id = state.fd_db.insert_person(
                    person_name=person_name, description=description,
                    source=req.source, confidence=req.confidence, memory_id=memory_id,
                )
            except sqlite3.IntegrityError:
                # Corrida: outro write criou a mesma person_key nesse meio-tempo.
                existing = state.fd_db.get_by_name(person_name)
                if existing is None:
                    raise
                person_id, memory_id, new_person = existing["id"], existing["memory_id"], False

        embedding_id = state.fd_db.insert_embedding(person_id)
        await loop.run_in_executor(None, state.fd_index.add, vec, embedding_id)

    log.info(
        f"Face-dict: {'nova pessoa' if new_person else 'novo exemplo'} "
        f"'{person_name}' (person_id={person_id}, embedding_id={embedding_id})"
    )

    return FaceDictWriteResponse(
        stored=True, reason="ok", person_id=person_id,
        embedding_id=embedding_id, memory_id=memory_id, new_person=new_person,
    )


async def face_dict_read(req: FaceDictReadRequest):
    if len(req.embedding) != FD_EMBED_DIM:
        raise MemoryToolError(f"embedding deve ter dimensão {FD_EMBED_DIM}, recebido {len(req.embedding)}")

    vec = np.asarray(req.embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm

    # sobre-amostra o kNN porque vários embeddings podem apontar pra mesma
    # pessoa (múltiplos exemplos) — precisamos deduplicar por person_id
    hits = state.fd_index.search(vec, max(req.top_k * 4, 20))

    best_score_by_person: dict[int, float] = {}
    for embedding_id, score in hits:
        person_id = state.fd_db.get_person_id_by_embedding(embedding_id)
        if person_id is None:
            continue
        if person_id not in best_score_by_person or score > best_score_by_person[person_id]:
            best_score_by_person[person_id] = score

    ranked = sorted(best_score_by_person.items(), key=lambda kv: kv[1], reverse=True)[:req.top_k]

    results: list[FaceCandidate] = []
    for person_id, score in ranked:
        if score < req.min_score:
            continue
        row = state.fd_db.get_person_by_id(person_id)
        if row is None:
            continue
        state.fd_db.update_access(person_id)
        results.append(FaceCandidate(
            person_id=row["id"],
            person_name=row["person_name"],
            description=row["description"],
            score=score,
            confidence=row["confidence"],
            access_count=row["access_count"] + 1,
            memory_id=row["memory_id"],
        ))

    # ambíguo quando: ninguém bateu com confiança suficiente, OU os dois
    # melhores candidatos estão muito próximos (pode ser qualquer um dos dois)
    ambiguous = (
        len(results) == 0
        or (len(results) > 1 and (results[0].score - results[1].score) < FD_AMBIGUOUS_MARGIN)
    )

    # ── NEW: busca por grafo a partir dos hits (só rostos com memory_id)
    related: list[RelatedMemory] = []
    if req.include_related:
        related = await _dict_graph_related(
            [(c.memory_id, c.score) for c in results], asyncio.get_event_loop()
        )

    return FaceDictReadResponse(results=results, ambiguous=ambiguous, related=related)


async def face_dict_list():
    rows = state.fd_db.list_people()
    people = [
        FaceDictEntry(
            person_id=row["id"],
            person_name=row["person_name"],
            description=row["description"],
            source=row["source"],
            confidence=row["confidence"],
            memory_id=row["memory_id"],
            examples_count=row["examples_count"],
            created_at=row["created_at"],
            access_count=row["access_count"],
        ).model_dump()
        for row in rows
    ]
    return {"total": len(people), "people": people}


async def face_dict_get(person_id: int):
    row = state.fd_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")
    return FaceDictEntry(
        person_id=row["id"],
        person_name=row["person_name"],
        description=row["description"],
        source=row["source"],
        confidence=row["confidence"],
        memory_id=row["memory_id"],
        examples_count=state.fd_db.count_examples(person_id),
        created_at=row["created_at"],
        access_count=row["access_count"],
    )


class FaceDictUpdateRequest(BaseModel):
    description: str

async def face_dict_update(person_id: int, req: FaceDictUpdateRequest):
    """Edita só a descrição de uma pessoa já cadastrada, sem precisar
    mandar um novo embedding junto."""
    row = state.fd_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")
    state.fd_db.update_description(person_id, req.description.strip())
    row = state.fd_db.get_person_by_id(person_id)
    return FaceDictEntry(
        person_id=row["id"],
        person_name=row["person_name"],
        description=row["description"],
        source=row["source"],
        confidence=row["confidence"],
        memory_id=row["memory_id"],
        examples_count=state.fd_db.count_examples(person_id),
        created_at=row["created_at"],
        access_count=row["access_count"],
    )


async def face_dict_delete(person_id: int):
    """Remove uma pessoa e todos os seus embeddings de rosto (DB + FAISS)."""
    row = state.fd_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")

    embedding_ids = state.fd_db.get_embedding_ids_by_person(person_id)
    if embedding_ids:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, state.fd_index.remove_ids, set(embedding_ids))

    deleted = state.fd_db.delete_person(person_id)
    log.info(f"Face-dict: pessoa #{person_id} removida ({len(embedding_ids)} embeddings)")
    return {"deleted": deleted, "person_id": person_id, "embeddings_removed": len(embedding_ids)}


# ── NEW: Dicionário de Voz — endpoints usados pelo pipeline de áudio ────────────
#
# Mesma lógica do dicionário de rostos, adaptada pra identificação de locutor:
# o pipeline de áudio extrai o embedding da frase falada (CAM++ etc.) e manda
# pra cá. Aqui decide-se se é uma pessoa nova ou mais um exemplo de uma pessoa
# já cadastrada. DIFERENÇA em relação ao face-dict: a descrição de quem é a
# pessoa também é gravada na memória de longo prazo (source="voice_dict"),
# então ela é pesquisável pela leitura normal (/read) — igual ao link_to_memory
# do dicionário visual.

async def voice_dict_write(req: VoiceDictWriteRequest):
    person_name = req.person_name.strip()
    description = req.description.strip()
    if not person_name:
        return VoiceDictWriteResponse(stored=False, reason="empty_person_name")

    if len(req.embedding) != VO_EMBED_DIM:
        raise MemoryToolError(f"embedding deve ter dimensão {VO_EMBED_DIM}, recebido {len(req.embedding)}")

    vec = np.asarray(req.embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm

    loop = asyncio.get_event_loop()

    async with _vo_write_lock:
        existing = state.vo_db.get_by_name(person_name)
        new_person = existing is None

        if existing is not None:
            person_id = existing["id"]
            memory_id = existing["memory_id"]
            # se veio uma descrição não-vazia num cadastro de exemplo adicional,
            # atualiza/completa a descrição já salva
            if description:
                state.vo_db.update_description(person_id, description)
        else:
            memory_id = None
            if description:
                # ── descrição replicada na memória de longo prazo — mesma
                # política do visual-dict: só na criação do conceito/pessoa,
                # exemplos adicionais não duplicam a entrada de texto.
                stored, _, memory_id, _ = await _store_long_term_text(
                    f"{person_name}: {description}", "voice_dict", req.confidence,
                )
                if not stored:
                    memory_id = None  # dedup/erro — a pessoa ainda é cadastrada,
                                        # só sem o link pra memória
            try:
                person_id = state.vo_db.insert_person(
                    person_name=person_name, description=description,
                    source=req.source, confidence=req.confidence, memory_id=memory_id,
                )
            except sqlite3.IntegrityError:
                # Corrida: outro write criou a mesma person_key nesse meio-tempo.
                existing = state.vo_db.get_by_name(person_name)
                if existing is None:
                    raise
                person_id, memory_id, new_person = existing["id"], existing["memory_id"], False

        embedding_id = state.vo_db.insert_embedding(person_id)
        await loop.run_in_executor(None, state.vo_index.add, vec, embedding_id)

    log.info(
        f"Voice-dict: {'nova pessoa' if new_person else 'novo exemplo'} "
        f"'{person_name}' (person_id={person_id}, embedding_id={embedding_id}, memory_id={memory_id})"
    )

    return VoiceDictWriteResponse(
        stored=True, reason="ok", person_id=person_id,
        embedding_id=embedding_id, memory_id=memory_id, new_person=new_person,
    )


async def voice_dict_read(req: VoiceDictReadRequest):
    if len(req.embedding) != VO_EMBED_DIM:
        raise MemoryToolError(f"embedding deve ter dimensão {VO_EMBED_DIM}, recebido {len(req.embedding)}")

    vec = np.asarray(req.embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm

    # sobre-amostra o kNN porque vários embeddings podem apontar pra mesma
    # pessoa (múltiplos exemplos) — precisamos deduplicar por person_id
    hits = state.vo_index.search(vec, max(req.top_k * 4, 20))

    best_score_by_person: dict[int, float] = {}
    for embedding_id, score in hits:
        person_id = state.vo_db.get_person_id_by_embedding(embedding_id)
        if person_id is None:
            continue
        if person_id not in best_score_by_person or score > best_score_by_person[person_id]:
            best_score_by_person[person_id] = score

    ranked = sorted(best_score_by_person.items(), key=lambda kv: kv[1], reverse=True)[:req.top_k]

    results: list[VoiceCandidate] = []
    for person_id, score in ranked:
        if score < req.min_score:
            continue
        row = state.vo_db.get_person_by_id(person_id)
        if row is None:
            continue
        state.vo_db.update_access(person_id)
        results.append(VoiceCandidate(
            person_id=row["id"],
            person_name=row["person_name"],
            description=row["description"],
            score=score,
            confidence=row["confidence"],
            access_count=row["access_count"] + 1,
            memory_id=row["memory_id"],
        ))

    # ambíguo quando: ninguém bateu com confiança suficiente, OU os dois
    # melhores candidatos estão muito próximos (pode ser qualquer um dos dois)
    ambiguous = (
        len(results) == 0
        or (len(results) > 1 and (results[0].score - results[1].score) < VO_AMBIGUOUS_MARGIN)
    )

    # ── NEW: busca por grafo a partir dos hits
    related: list[RelatedMemory] = []
    if req.include_related:
        related = await _dict_graph_related(
            [(c.memory_id, c.score) for c in results], asyncio.get_event_loop()
        )

    return VoiceDictReadResponse(results=results, ambiguous=ambiguous, related=related)


async def voice_dict_list():
    rows = state.vo_db.list_people()
    people = [
        VoiceDictEntry(
            person_id=row["id"],
            person_name=row["person_name"],
            description=row["description"],
            source=row["source"],
            confidence=row["confidence"],
            memory_id=row["memory_id"],
            examples_count=row["examples_count"],
            created_at=row["created_at"],
            access_count=row["access_count"],
        ).model_dump()
        for row in rows
    ]
    return {"total": len(people), "people": people}


async def voice_dict_get(person_id: int):
    row = state.vo_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")
    return VoiceDictEntry(
        person_id=row["id"],
        person_name=row["person_name"],
        description=row["description"],
        source=row["source"],
        confidence=row["confidence"],
        memory_id=row["memory_id"],
        examples_count=state.vo_db.count_examples(person_id),
        created_at=row["created_at"],
        access_count=row["access_count"],
    )


async def voice_dict_update(person_id: int, req: VoiceDictUpdateRequest):
    """Edita só a descrição de uma pessoa já cadastrada, sem precisar
    mandar um novo embedding junto."""
    row = state.vo_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")
    state.vo_db.update_description(person_id, req.description.strip())
    row = state.vo_db.get_person_by_id(person_id)
    return VoiceDictEntry(
        person_id=row["id"],
        person_name=row["person_name"],
        description=row["description"],
        source=row["source"],
        confidence=row["confidence"],
        memory_id=row["memory_id"],
        examples_count=state.vo_db.count_examples(person_id),
        created_at=row["created_at"],
        access_count=row["access_count"],
    )


async def voice_dict_delete(person_id: int):
    """Remove uma pessoa e todos os seus embeddings de voz (DB + FAISS).

    A memória de longo prazo linkada (descrição gravada na criação) NÃO é
    removida — a remoção explícita lá continua sendo responsabilidade de
    quem gerencia `memories` (mesma política do visual-dict)."""
    row = state.vo_db.get_person_by_id(person_id)
    if row is None:
        raise MemoryToolError(f"Pessoa #{person_id} não encontrada")

    embedding_ids = state.vo_db.get_embedding_ids_by_person(person_id)
    if embedding_ids:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, state.vo_index.remove_ids, set(embedding_ids))

    deleted = state.vo_db.delete_person(person_id)
    log.info(f"Voice-dict: pessoa #{person_id} removida ({len(embedding_ids)} embeddings)")
    return {"deleted": deleted, "person_id": person_id, "embeddings_removed": len(embedding_ids)}


# ── NEW: RAG de tools — seleção dinâmica das tools expostas ao LLM ────────────
#
# Fluxo esperado no orquestrador:
#   1. no startup: tools_register_batch(todas as tools, prune_missing=True) —
#      idempotente (tool inalterada não é re-embedada);
#   2. a cada turno, ANTES de montar o campo `tools` da chamada ao LLM:
#      tools_select(query) → só as tools devolvidas vão no payload;
#   3. depois do turno: tools_record_usage(tools realmente chamadas, em ordem,
#      + a query) → alimenta o grafo de co-uso/sequência e os exemplos aprendidos.
#
# Seleção = kNN em vetores (doc + exemplos + queries aprendidas) com max-pooling
# por tool, threshold permissivo, expansão por PPR no grafo de co-uso, e tools
# "core" sempre presentes. Confiança baixa no top-1 → rede mais larga.

def _tool_doc_text(name: str, description: str) -> str:
    return f"{name.replace('_', ' ')}: {description.strip()}"


async def _delete_tool_row(row: sqlite3.Row) -> None:
    loop = asyncio.get_event_loop()
    vids = state.tl_db.get_vector_ids(row["id"])
    if vids:
        await loop.run_in_executor(None, state.tl_index.remove_ids, set(vids))
    await loop.run_in_executor(None, state.tl_db.delete_tool, row["id"])


async def _register_tool_locked(req: ToolRegisterRequest) -> ToolRegisterResponse:
    name        = req.name.strip()
    description = req.description.strip()
    if not name or not description:
        raise MemoryToolError("tool precisa de name e description não vazios")
    examples: list[str] = []
    for ex in req.examples:
        ex = ex.strip()
        if ex and ex not in examples:
            examples.append(ex)
    examples = examples[:TOOLS_MAX_EXAMPLES]

    doc_text = _tool_doc_text(name, description)
    doc_hash = hashlib.sha256("\n".join([doc_text, *examples]).encode()).hexdigest()
    loop = asyncio.get_event_loop()

    existing = state.tl_db.get_by_name(name)
    if existing is not None and existing["doc_hash"] == doc_hash:
        if bool(existing["is_core"]) != req.core:
            state.tl_db.set_core(existing["id"], req.core)
        return ToolRegisterResponse(
            name=name, tool_id=existing["id"], status="unchanged",
            vectors=len(state.tl_db.get_vector_ids(existing["id"])),
        )

    texts = [doc_text, *examples]
    embeddings = await state.embed_engine.embed_passages(texts)   # antes de mutar qualquer coisa

    if existing is not None:
        tool_id = existing["id"]
        # troca doc + exemplos; queries "learned" (uso real) são preservadas
        old_ids = state.tl_db.get_vector_ids(tool_id, kinds=("doc", "example"))
        if old_ids:
            await loop.run_in_executor(None, state.tl_index.remove_ids, set(old_ids))
            state.tl_db.delete_vectors(old_ids)
        state.tl_db.update_tool(tool_id, description, req.core, doc_hash)
        status = "updated"
    else:
        tool_id = state.tl_db.insert_tool(name, description, req.core, doc_hash)
        status = "created"

    vids = [
        state.tl_db.insert_vector(tool_id, "doc" if i == 0 else "example", t)
        for i, t in enumerate(texts)
    ]
    await loop.run_in_executor(None, state.tl_index.add_batch, embeddings, vids)
    log.info(f"Tools: '{name}' {status} ({len(vids)} vetores, core={req.core})")
    return ToolRegisterResponse(name=name, tool_id=tool_id, status=status, vectors=len(vids))


async def tools_register(req: ToolRegisterRequest):
    async with _tl_write_lock:
        return await _register_tool_locked(req)


async def tools_register_batch(req: ToolRegisterBatchRequest):
    results: list[ToolRegisterResponse] = []
    pruned: list[str] = []
    async with _tl_write_lock:
        for item in req.tools:
            results.append(await _register_tool_locked(item))
        if req.prune_missing:
            keep = {ToolsDB._normalize_key(t.name) for t in req.tools}
            for row in state.tl_db.list_tools():
                if row["name_key"] not in keep:
                    await _delete_tool_row(row)
                    pruned.append(row["name"])
    if pruned:
        log.info(f"Tools: {len(pruned)} tools removidas do índice (prune_missing): {pruned}")
    return ToolRegisterBatchResponse(results=results, pruned=pruned)


async def tools_select(req: ToolSelectRequest):
    query = req.query.strip()
    if not query:
        raise MemoryToolError("query vazia")
    loop = asyncio.get_event_loop()

    query_emb = await state.embed_engine.embed_query_one(query)
    # sobre-amostra: vários vetores (doc + exemplos) apontam pra mesma tool
    hits = await loop.run_in_executor(
        None, state.tl_index.search, query_emb, max(req.top_k * 6, 30)
    )
    vec_to_tool = state.tl_db.get_tool_ids_by_vectors([v for v, _ in hits])
    best: dict[int, float] = {}
    for vid, score in hits:
        tid = vec_to_tool.get(vid)
        if tid is not None and score > best.get(tid, -1.0):
            best[tid] = score
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)

    top1 = ranked[0][1] if ranked else 0.0
    low_confidence = top1 < TOOLS_LOW_CONFIDENCE_SCORE
    if low_confidence:
        # confiança baixa: recall > precisão — rede mais larga, sem threshold
        primary = ranked[:TOOLS_FALLBACK_TOP_K]
    else:
        primary = [(t, s) for t, s in ranked if s >= req.min_score][:req.top_k]

    # tools alcançadas SÓ pelo grafo de co-uso/sequência (listar → ler → editar)
    related: list[tuple[int, float]] = []
    if req.include_related and primary:
        ctx = await _ppr_spread_on(state.tl_db, primary, loop)
        taken = {t for t, _ in primary}
        related = [(t, s) for t, s in ctx["related_raw"] if t not in taken][:TOOLS_MAX_RELATED]

    core_rows = state.tl_db.list_core()
    core_ids = {r["id"] for r in core_rows}

    rows = {r["id"]: r for r in state.tl_db.get_by_ids(
        [t for t, _ in primary] + [t for t, _ in related] + list(core_ids)
    )}
    out: list[ToolMatch] = []
    seen: set[int] = set()

    def push(tid: int, score: float, match_type: str):
        row = rows.get(tid)
        if row is None or tid in seen:
            return
        seen.add(tid)
        out.append(ToolMatch(
            name=row["name"], description=row["description"],
            score=round(score, 4), match_type=match_type,
        ))

    for tid, s in primary:
        push(tid, s, "primary")
    for tid, s in related:
        push(tid, s, "related")
    for tid in core_ids:
        push(tid, best.get(tid, 0.0), "core")

    log.info(
        f"tools_select query='{query[:50]}' top1={top1:.3f} low_conf={low_confidence} "
        f"primary={len(primary)} related={len(related)} core={len(core_ids)} -> {len(out)} tools"
    )
    return ToolSelectResponse(tools=out, top1_score=round(top1, 4), low_confidence=low_confidence)


def _record_tool_edges_sync(tool_ids: list[int]) -> None:
    now = time.time()
    for i in range(len(tool_ids)):
        for j in range(i + 1, len(tool_ids)):
            state.tl_db.upsert_edge(tool_ids[i], tool_ids[j], EDGE_LEARNING_RATE, now, EDGE_CO_ACTIVATION)
    for prev, nxt in zip(tool_ids, tool_ids[1:]):
        state.tl_db.upsert_edge(prev, nxt, EDGE_LEARNING_RATE, now, EDGE_TEMPORAL)


async def tools_record_usage(req: ToolUsageRequest):
    """Registra o uso real: (1) arestas de co-uso e de sequência no grafo de
    tools, (2) a query vira exemplo aprendido das tools usadas."""
    loop = asyncio.get_event_loop()
    ordered: list[int] = []
    for name in req.tools_used[:TOOLS_USAGE_MAX_SEQUENCE]:
        row = state.tl_db.get_by_name(name)
        if row is not None and (not ordered or ordered[-1] != row["id"]):
            ordered.append(row["id"])
    if not ordered:
        return {"recorded": False, "reason": "no_known_tools", "tools": 0, "learned": 0}

    await loop.run_in_executor(None, state.tl_db.mark_used, ordered)
    if len(ordered) > 1:
        await loop.run_in_executor(None, _record_tool_edges_sync, ordered)

    learned = 0
    query = req.query.strip()
    if query:
        async with _tl_write_lock:
            # `query` aqui vira um EXEMPLO ARMAZENADO no índice de tools
            # (insert_vector abaixo + tl_index.add), pra ser encontrado
            # depois por outras queries em tools_select — por isso é
            # embedado como passage, não como query, apesar do nome da
            # variável.
            emb = await state.embed_engine.embed_passage_one(query)
            for tid in dict.fromkeys(ordered):
                if state.tl_db.count_learned(tid) >= TOOLS_MAX_LEARNED_EXAMPLES:
                    continue
                existing_vids = state.tl_db.get_vector_ids(tid)
                near = await loop.run_in_executor(
                    None, state.tl_index.search_subset, emb, existing_vids, 1
                )
                if near and near[0][1] >= TOOLS_LEARN_DEDUP_SCORE:
                    continue
                vid = state.tl_db.insert_vector(tid, "learned", query)
                await loop.run_in_executor(None, state.tl_index.add, emb, vid)
                learned += 1
    return {"recorded": True, "tools": len(ordered), "learned": learned}


async def tools_list():
    rows = state.tl_db.list_tools()
    return {
        "total": len(rows),
        "tools": [
            {
                "name": r["name"], "description": r["description"],
                "core": bool(r["is_core"]), "vectors": r["vectors_count"],
                "use_count": r["use_count"], "last_used": r["last_used"],
            }
            for r in rows
        ],
    }


async def tools_delete(name: str):
    row = state.tl_db.get_by_name(name)
    if row is None:
        raise MemoryToolError(f"Tool '{name}' não encontrada")
    async with _tl_write_lock:
        await _delete_tool_row(row)
    return {"deleted": True, "name": row["name"]}


# ── GET /status ────────────────────────────────────────────────────────────────

def _gather_status_sync() -> dict:
    """Coleta bloqueante (várias queries SQLite + state.vs.status()).
    Extraída para função síncrona própria para poder rodar via
    run_in_executor (ver status() abaixo) — igual ao padrão já usado no
    resto do arquivo para .search()/.apply_decay()/etc. Sem isso, essas
    ~8 queries síncronas rodavam direto no corpo da coroutine e, sob
    contenção com uma escrita concorrente seguranco o lock/transação
    SQLite (embedding batch, decay job, cleanup de short-term — todas via
    run_in_executor em outra thread), travavam o event loop inteiro
    enquanto esperavam — inclusive impedindo o servidor de responder ao
    handshake MCP de outros clientes (ver orchestrator._get_mcp_session)."""
    resp = {
        "long_term": {
            "memories_total":       state.lt_db.count(),
            "index_vectors":        state.lt_index.total,
            "decay_half_life_days": DECAY_HALF_LIFE_DAYS,
            "dedup_threshold":      DEDUP_THRESHOLD,
            "update_sim_threshold": UPDATE_SIM_THRESHOLD,
            "rerank_duplicate_score": RERANK_DUPLICATE_SCORE,
            "rerank_update_score":    RERANK_UPDATE_SCORE,
        },
        # ── NEW: grafo Hebbiano de co-ativação ──
        "hebbian_graph": {
            "edges_total":              state.lt_db.count_edges(),
            "edge_decay_half_life_d":   EDGE_DECAY_HALF_LIFE_DAYS,
            "edge_prune_threshold":     EDGE_PRUNE_THRESHOLD,
            "edge_learning_rate":       EDGE_LEARNING_RATE,
            "edge_min_score_to_link":   EDGE_MIN_SCORE_TO_LINK,
            "ltp_access_boost":         LTP_ACCESS_BOOST,
            "ppr_damping":              PPR_DAMPING,
            "ppr_spread_weight":        PPR_SPREAD_WEIGHT,
            "mmr_lambda":               MMR_LAMBDA,
            "ppr_expand_max_nodes":     PPR_EXPAND_MAX_NODES,
            "ppr_expand_min_path_w":    PPR_EXPAND_MIN_PATH_WEIGHT,
            "mmr_cosine_floor":         MMR_COSINE_FLOOR,
            "edge_types":               list(EDGE_TYPES),
        },
        # ── NEW: LanceDB ──
        "vector_store": {
            "backend": "lancedb",
            "path":    LANCE_DIR,
        },
        # ── NEW: RAG de tools ──
        "tools": {
            "tools_total":        state.tl_db.count(),
            "vectors_total":      state.tl_index.total,
            "edges_total":        state.tl_db.count_edges(),
            "top_k":              TOOLS_TOP_K,
            "min_score":          TOOLS_MIN_SCORE,
            "low_confidence":     TOOLS_LOW_CONFIDENCE_SCORE,
            "fallback_top_k":     TOOLS_FALLBACK_TOP_K,
        },
        "short_term": {
            "turn_groups_total":  state.st_db.count(),
            "index_vectors":      state.st_index.total,
            "ttl_hours":          ST_TTL_HOURS,
            "cleanup_interval_s": ST_CLEANUP_INTERVAL_S,
        },
        "contextual_search": {
            "query_short_words":     QUERY_SHORT_WORDS,
            "query_ambiguous_ratio": QUERY_AMBIGUOUS_RATIO,
            "context_max_chars":     CONTEXT_MAX_CHARS,
            "context_turns_fetch":   CONTEXT_TURNS_FETCH,
            "dual_context_weight":   DUAL_CONTEXT_WEIGHT,
        },
        # ── Otimização de tokens (anti 413 Payload Too Large) ──
        "token_optimization": {
            "read_top_k_final":       READ_TOP_K_FINAL,
            "read_total_max_chars":   READ_TOTAL_MAX_CHARS,
            "read_lt_max_chars":      READ_LT_MAX_CHARS,
            "read_st_max_chars":      READ_ST_MAX_CHARS,
            "read_vs_max_chars":      READ_VS_MAX_CHARS,
            "read_if_max_chars":      READ_IF_MAX_CHARS,
            "read_min_score_strict":  READ_MIN_SCORE_STRICT,
            "if_min_score_read":      IF_MIN_SCORE_READ,
        },
        "onnx_serving": {
            "url": ONNX_SERVING_URL,
            "mode": "remote_api",
        },
        # ── NEW: Indexed files status ──
        "indexed_files": {
            "files_total":    state.if_db.count_files(),
            "chunks_total":   state.if_db.get_total_chunks(),
            "index_vectors":  state.if_index.total,
            "min_score":      IF_MIN_SCORE,
            "chunk_size":     CHUNK_SIZE,
            "chunk_overlap":  CHUNK_OVERLAP,
            "max_content":    IF_MAX_CONTENT_SIZE,
            "max_chunks":     IF_MAX_CHUNKS,
        },
        # ── NEW: Visual dictionary status ──
        "visual_dict": {
            "concepts_total":  state.vd_db.count(),
            "embeddings_total": state.vd_index.total,
            "embed_dim":       VD_EMBED_DIM,
            "min_score":       VD_MIN_SCORE,
            "top_k":           VD_TOP_K,
            "ambiguous_margin": VD_AMBIGUOUS_MARGIN,
        },
        # ── NEW: Face dictionary status ──
        "face_dict": {
            "people_total":     state.fd_db.count(),
            "embeddings_total": state.fd_index.total,
            "embed_dim":        FD_EMBED_DIM,
            "min_score":        FD_MIN_SCORE,
            "top_k":            FD_TOP_K,
            "ambiguous_margin": FD_AMBIGUOUS_MARGIN,
        },
        # ── NEW: Voice dictionary status ──
        "voice_dict": {
            "people_total":     state.vo_db.count(),
            "embeddings_total": state.vo_index.total,
            "embed_dim":        VO_EMBED_DIM,
            "min_score":        VO_MIN_SCORE,
            "top_k":            VO_TOP_K,
            "ambiguous_margin": VO_AMBIGUOUS_MARGIN,
        },
    }
    if state.vs is not None:
        vs_status = state.vs.status()
        resp["knowledge"] = {
            "available":        True,
            "chunks_in_db":     vs_status["chunks_in_db"],
            "vectors_in_index": vs_status["vectors_in_index"],
            "embed_dim":        vs_status["embed_dim"],
            "dedup_threshold":  vs_status["dedup_threshold"],
            "vs_min_score":     VS_MIN_SCORE,
        }
    else:
        resp["knowledge"] = {
            "available":        False,
            "chunks_in_db":     0,
            "vectors_in_index": 0,
        }
    return resp


async def status():
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _gather_status_sync)