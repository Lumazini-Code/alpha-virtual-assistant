from __future__ import annotations

"""
memory_models.py — Modelos de request/response (Pydantic) + erros.

Contrato da API de memória: todos os modelos usados pelos handlers e pelo
memory_rest_api.py (mem.WriteRequest, mem.ReadResponse, …). Inclui
MemoryToolError, a exceção de negócio agnóstica de transporte.
"""

from typing import Literal, Optional

from pydantic import BaseModel

from memory_config import (
    FD_MIN_SCORE,
    FD_TOP_K,
    IF_MIN_SCORE,
    READ_MIN_SCORE,
    ST_READ_DEFAULT_PAIRS,
    TOOLS_MIN_SCORE,
    TOOLS_TOP_K,
    TOP_K_READ,
    VD_MIN_SCORE,
    VD_TOP_K,
    VO_MIN_SCORE,
    VO_TOP_K,
)

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
    # ── NEW: overrides de orçamento por request ──
    # O /read tem um corte final conservador (READ_TOP_K_FINAL=3 entradas /
    # READ_TOTAL_MAX_CHARS=2400 chars) pensado para prompts enxutos. Callers
    # que usam o resultado como recall de chat podem pedir mais contexto sem
    # tocar no default global — os valores são clampeados no servidor
    # (READ_TOP_K_FINAL_MAX / READ_TOTAL_MAX_CHARS_MAX). None = usa o default.
    top_k_final:     Optional[int] = None
    total_max_chars: Optional[int] = None

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


# ── Erros ──────────────────────────────────────────────────────────────────────

class MemoryToolError(Exception):
    """Erro de validação/negócio de uma tool de memória — equivalente ao
    antigo HTTPException do FastAPI, mas agnóstico de transporte (MCP não
    tem código de status HTTP). A mensagem vira o texto de erro da tool."""
    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class FaceDictUpdateRequest(BaseModel):
    description: str


__all__ = [
    'FaceCandidate',
    'FaceDictEntry',
    'FaceDictReadRequest',
    'FaceDictReadResponse',
    'FaceDictUpdateRequest',
    'FaceDictWriteRequest',
    'FaceDictWriteResponse',
    'IndexedFileCheckResponse',
    'IndexedFileEntry',
    'IndexedFileReadRequest',
    'IndexedFileReadResponse',
    'IndexedFileWriteRequest',
    'IndexedFileWriteResponse',
    'MemoryEntry',
    'MemoryToolError',
    'ReadRequest',
    'ReadResponse',
    'ReadSTRequest',
    'ReadSTResponse',
    'RelatedMemory',
    'ToolMatch',
    'ToolRegisterBatchRequest',
    'ToolRegisterBatchResponse',
    'ToolRegisterRequest',
    'ToolRegisterResponse',
    'ToolSelectRequest',
    'ToolSelectResponse',
    'ToolUsageRequest',
    'Turn',
    'VisualDictCandidate',
    'VisualDictEntry',
    'VisualDictReadRequest',
    'VisualDictReadResponse',
    'VisualDictWriteRequest',
    'VisualDictWriteResponse',
    'VoiceCandidate',
    'VoiceDictEntry',
    'VoiceDictReadRequest',
    'VoiceDictReadResponse',
    'VoiceDictUpdateRequest',
    'VoiceDictWriteRequest',
    'VoiceDictWriteResponse',
    'WriteBatchRequest',
    'WriteBatchResponse',
    'WriteRequest',
    'WriteResponse',
    'WriteSTRequest',
    'WriteSTResponse',
]
