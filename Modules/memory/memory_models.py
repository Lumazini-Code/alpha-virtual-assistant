from __future__ import annotations

"""
memory_models.py — Modelos de request/response (Pydantic) + erros.

Contrato da API de memória: todos os modelos usados pelos handlers e pelo
memory_rest_api.py (mem.WriteRequest, mem.ReadResponse, …). Inclui
MemoryToolError, a exceção de negócio agnóstica de transporte.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from memory_config import (
    PLANS_MAX_ACTIONS_MODEL,
    PLANS_MAX_STEPS,
    FD_MIN_SCORE,
    FD_TOP_K,
    IF_MIN_SCORE,
    READ_MIN_SCORE,
    SKILLS_MIN_SCORE,
    SKILLS_TOP_K,
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
    # ── NEW (people cards): vincula a memória a um card de pessoa (people.id).
    # A memória nova ganha uma aresta co_activation com o nó de identidade do
    # card no grafo Hebbiano — o PPR espalha ativação entre a pessoa e as
    # memórias sobre ela. person_id inválido → warning e grava sem vínculo.
    person_id:   Optional[int] = None

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
    # ── SESSÕES REMOVIDAS: session_id é só metadado de proveniência (fica na
    # coluna p/ o DELETE /session/{id} e diagnóstico); não participa da
    # leitura (global) nem do dedup (global).
    session_id: str = ""
    turns:      list[Turn]
    # ── NEW (people cards): vincula o grupo de turnos a um card de pessoa
    # (people.id) — FK direta, filtrável no /read_st. Invalido → warning.
    person_id:  Optional[int] = None

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
    # ── NEW (people cards): leitura centrada em uma pessoa. O nó de
    # identidade do card é injetado como semente extra no PPR — memórias
    # relacionadas à pessoa vêm como "related" mesmo sem match semântico.
    person_id:   Optional[int] = None

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
    # ── SESSÕES REMOVIDAS: a leitura do ST é GLOBAL (N duplas mais recentes,
    # independente de sessão). O campo permanece opcional por compatibilidade
    # de API — é aceito, ecoado na resposta e IGNORADO como filtro.
    session_id: str = ""
    n_pairs:    int = ST_READ_DEFAULT_PAIRS
    # ── NEW (people cards): filtra os grupos de turnos por card de pessoa.
    person_id:  Optional[int] = None

class ReadSTResponse(BaseModel):
    session_id:     str = ""   # eco do pedido (não filtra nada)
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
    # ── NEW (projetos): escopo dono da entrada na leitura mesclada —
    # "general" (vem com id composta, offset +10^12) ou o slug do projeto
    # ativo. None = modo geral puro (compatibilidade) ou entrada global (ST).
    scope:        Optional[str]        = None
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
    # ── NEW (projetos): escopo dono do arquivo na leitura mesclada —
    # "general" (file_id com offset +10^12) ou o slug do projeto ativo.
    # None = modo geral puro (compatibilidade).
    scope:        Optional[str] = None
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
    likes:           str = ""       # NEW (unificação): gostos/preferências
    religion:        str = ""       # NEW (unificação): crenças/religião
    additional_info: str = ""       # NEW (unificação): informações adicionais
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
    likes:          str = ""   # NEW (unificação): gostos/preferências
    religion:       str = ""   # NEW (unificação): crenças
    additional_info: str = ""  # NEW (unificação): informações adicionais
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
    likes:           str = ""       # NEW (unificação): gostos/preferências
    religion:        str = ""       # NEW (unificação): crenças/religião
    additional_info: str = ""       # NEW (unificação): informações adicionais
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
    likes:          str = ""   # NEW (unificação): gostos/preferências
    religion:       str = ""   # NEW (unificação): crenças
    additional_info: str = ""  # NEW (unificação): informações adicionais
    source:         str
    confidence:     float
    memory_id:      Optional[int] = None
    examples_count: int
    created_at:     float
    access_count:   int

class VoiceDictUpdateRequest(BaseModel):
    """Edita campos do card de pessoa. Campos não enviados (None) NÃO são
    alterados — só os campos presentes na requisição são atualizados."""
    description:     str = ""
    likes:           Optional[str] = None
    religion:        Optional[str] = None
    additional_info: Optional[str] = None


# ── NEW: Modelos — People Card (card textual de pessoa, sem embeddings) ───────
#
# O card unificado (PeopleDB) já é o modelo de dados das pessoas do sistema
# (face-dict e voice-dict usam a mesma tabela `people`). Esses modelos expõem
# o card PELA PRIMEIRA VEZ como endpoint textual puro — sem exigir embedding
# biométrico — para o MCP alpha-mcp criar/editar pessoas pela conversa.
# Metadados idênticos aos de FaceDict/VoiceDict: o mesmo person_id continua
# recebendo embeddings dos pipelines de visão/áudio normalmente.

class PeopleCardWriteRequest(BaseModel):
    person_name:     str   = Field(min_length=1)
    description:     str   = ""          # quem é a pessoa (relação, contexto)
    likes:           str   = ""          # gostos/preferências
    religion:        str   = ""          # crenças/religião
    additional_info: str   = ""          # informações adicionais
    # "criar" (default) ou "atualizar" um card existente. Em "update" os
    # campos enviados sobrescrevem os do card (texto vazio = apaga o campo;
    # não envie o campo para mantê-lo — mesmo contrato do UpdateRequest).
    # Em "create" com nome já existente → stored=False, reason="duplicate".
    action:          Literal["create", "update"] = "create"
    person_id:       Optional[int] = None  # só relevante em action="update"
    # replica o card na memória de longo prazo ("Nome: descrição",
    # source="people_card") → nó de identidade no grafo Hebbiano. Default True:
    # sem o nó, /read com person_id e a busca por grafo não funcionam.
    link_to_memory:  bool  = True

class PeopleCardWriteResponse(BaseModel):
    stored:       bool
    reason:       str
    person_id:    Optional[int] = None
    memory_id:    Optional[int] = None   # nó de identidade em `memories`, se linkado
    new_person:   bool = False

class PeopleCardEntry(BaseModel):
    person_id:      int
    person_name:    str
    description:    str
    likes:          str
    religion:       str
    additional_info: str
    source:         str
    confidence:     float
    memory_id:      Optional[int] = None
    face_examples_count: int = 0
    voice_examples_count: int = 0
    created_at:     float
    access_count:   int

class PeopleCardUpdateRequest(BaseModel):
    """Edita campos do card de pessoa SEM embedding. Campos não enviados
    (None) NÃO são alterados — só os campos presentes são atualizados.
    Campos enviados como string vazia limpam o campo no card."""
    person_name:     Optional[str] = None
    description:     Optional[str] = None
    likes:           Optional[str] = None
    religion:        Optional[str] = None
    additional_info: Optional[str] = None
    # renomear/atualizar a descrição também regenera a memória de identidade
    # LT quando o card já tem uma (textos LT antigos ficam como histórico).
    relink_memory:   bool = True


# ── NEW: Modelos — Identity Card (card singleton da própria assistente) ───────
#
# Mesmo desenho textual dos People Cards, aplicado à ALPHa: uma linha só
# (tabela `identity`, id=1) com quem é a assistente + memory_id apontando
# pra memória LT de identidade (source="identity") que ancora o nó no grafo
# Hebbiano. Semeado no boot a partir de resource/persona.json.

class IdentityEntry(BaseModel):
    name:            str
    description:     str
    roles:           str
    additional_info: str
    memory_id:       Optional[int] = None   # nó de identidade em `memories`
    source:          str
    created_at:      float
    updated_at:      float

class IdentityUpdateRequest(BaseModel):
    """Edita o card de identidade (singleton). Campos não enviados (None)
    NÃO são alterados; string vazia limpa o campo — mesmo contrato do
    PeopleCardUpdateRequest."""
    name:            Optional[str] = None
    description:     Optional[str] = None
    roles:           Optional[str] = None
    additional_info: Optional[str] = None
    # mudar name/description também regenera a memória de identidade LT
    # in-place (nó estável no grafo).
    relink_memory:   bool = True


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
    # "healed": doc_hash batia mas vetores estavam ausentes no LanceDB —
    # re-embedados a partir dos textos originais (self-heal de índice).
    status:  Literal["created", "updated", "unchanged", "healed"]
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


# ── NEW: Agent Skills (SKILL.md) — seleção no MESMO ToolsDB, namespace próprio ─
#
# Mesma mecânica do ToolSelect* (kNN + PPR + queries aprendidas), mas o
# endpoint só devolve linhas com o prefixo AGENT_SKILL_PREFIX (o seletor de
# MCP /tools/select exclui o prefixo — skills não competem por slots de
# servidores). `name` no match é o nome NU da skill (sem prefixo) — é o que a
# tool `use_skill` do orchestrator recebe.

class AgentSkillSelectRequest(BaseModel):
    query:           str
    top_k:           int   = SKILLS_TOP_K
    min_score:       float = SKILLS_MIN_SCORE
    include_related: bool  = True

class AgentSkillMatch(BaseModel):
    name:        str
    description: str
    score:       float
    match_type:  Literal["primary", "related", "core"]

class AgentSkillSelectResponse(BaseModel):
    skills:         list[AgentSkillMatch]
    top1_score:     float
    low_confidence: bool


# ── NEW: Modelos — planos de execução híbridos (macro + modelo) ────────────────
#
# Um plano é uma lista ordenada de passos, cada um marcado como "macro"
# (roda no Playwright sem LLM — seletores fixos gravados com codegen) ou
# "modelo" (o LLM lê/julga, com objetivo + escopo + schema de saída).
# Persistido na memory (PlansDB) e versionado a cada update.

class PlanStep(BaseModel):
    # executor: "macro" = Playwright direto (sem LLM); "modelo" = LLM lê/julga
    executor: Literal["macro", "modelo"]
    # macro: tipo de ação do Playwright MCP ("navigate", "click", "snapshot", ...)
    # modelo: ignorado (o modelo escolhe as ações)
    action: Optional[str] = None
    # macro: parâmetros fixos da ação (ex. {"url": "..."} / {"element": "...", "ref": "..."})
    params: dict = {}
    # macro: expectativa de sucesso — {"url_contains": "..."} e/ou {"selector_visible": "..."}
    expect: Optional[dict] = None
    # modelo: o que fazer neste passo, em linguagem natural
    objective: Optional[str] = None
    # modelo: limites do que pode ser feito (ex. "só a seção X, não navegar para fora")
    scope: Optional[str] = None
    # modelo: JSON Schema da saída estruturada esperada deste passo
    output_schema: Optional[dict] = None
    # modelo: limite de ações de browser neste passo
    max_actions: int = Field(default=PLANS_MAX_ACTIONS_MODEL, ge=1, le=50)
    # descrição humana do passo (log/plano legível)
    description: str = ""
    # ── NEW (tools ↔ planos): nome da tool MCP que este step executa.
    # macro: normalmente a tool Playwright; modelo: a tool que o LLM tende a
    # usar. Os tool_names distintos do plano são resolvidos para tool_ids na
    # ToolsDB e gravados em `tools_json` — os tools do plano ganham arestas
    # no grafo de co-uso (temporal_precedence na ordem dos steps + co_activation
    # entre todos), reforçando a seleção dinâmica de tools.
    tool_name: Optional[str] = None

class PlanCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    steps: list[PlanStep] = Field(min_length=1, max_length=PLANS_MAX_STEPS)
    # ── NEW (people cards): card de pessoa ao qual o plano se relaciona.
    # Invalido → warning e grava sem vínculo (não bloqueia).
    person_id: Optional[int] = None

class PlanCreateResponse(BaseModel):
    plan_id: int
    version: int
    created: bool

class PlanUpdateRequest(BaseModel):
    # identificação: por id OU por name (name_key)
    plan_id: Optional[int] = None
    name: Optional[str] = None
    # renomear: novo nome do plano (name_key é recalculado; falha se já
    # existir outro plano com o mesmo nome). None = mantém o nome atual.
    new_name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    description: Optional[str] = None
    steps: Optional[list[PlanStep]] = Field(default=None, max_length=PLANS_MAX_STEPS)
    # merge ou replace das correções acumuladas ({step_index: seletor corrigido})
    corrections: Optional[dict[str, str]] = None
    # ── NEW (people cards): novo vínculo com card de pessoa. None = mantém
    # o vínculo atual (o update_plan preserva o valor existente).
    person_id: Optional[int] = None

class PlanUpdateResponse(BaseModel):
    plan_id: int
    version: int
    updated: bool

class PlanEntry(BaseModel):
    plan_id:    int
    name:       str
    description: str
    version:    int
    steps:      list[PlanStep]
    corrections: dict[str, str] = {}
    # ── NEW (projetos): escopo dono do plano na listagem mesclada —
    # "general" (plan_id com offset +10^12) ou o slug do projeto ativo.
    # None = modo geral puro (compatibilidade).
    scope:      Optional[str] = None
    # ── NEW (people cards): card de pessoa ao qual este plano se relaciona
    # (ex.: "plano que paga a conta da pessoa X"). Opcional — None = sem vínculo.
    person_id:  Optional[int] = None
    # ── NEW (tools ↔ planos): tool_ids distintos usados pelos steps do plano,
    # resolvidos a partir de `tool_name` em PlanStep. Esse conjunto alimenta as
    # arestas de co-uso no grafo de tools (ver memory_ops_plans.py).
    tool_ids:   list[int] = []
    created_at: float
    updated_at: float

class PlanReadRequest(BaseModel):
    plan_id: Optional[int] = None
    name:    Optional[str] = None

class PlanReadResponse(BaseModel):
    plan: Optional[PlanEntry] = None

class PlanListResponse(BaseModel):
    total: int
    plans: list[PlanEntry] = []

class PlanDeleteRequest(BaseModel):
    plan_id: Optional[int] = None
    name:    Optional[str] = None

class PlanDeleteResponse(BaseModel):
    deleted: bool
    plan_id: Optional[int] = None


# ── Erros ──────────────────────────────────────────────────────────────────────

class MemoryToolError(Exception):
    """Erro de validação/negócio de uma tool de memória — equivalente ao
    antigo HTTPException do FastAPI, mas agnóstico de transporte (MCP não
    tem código de status HTTP). A mensagem vira o texto de erro da tool."""
    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


# ── NEW (projetos): Modelos de request/response — escopos ──────────────────────

class ScopeSwitchRequest(BaseModel):
    """POST /project/switch — troca o escopo ativo do processo de memória.
    "general" = modo geral (sem projeto). Slug de projeto = abre (se ainda
    não estiver) e ativa o conjunto de artefatos do projeto."""
    scope: str = Field(..., description='Escopo: "general" ou slug do projeto.')


class ScopeInfoResponse(BaseModel):
    """Estado do escopo ativo — consumido pelo orchestrator (re-assert a cada
    turno, proteção contra restart da memory) e pelo graph_activation.py."""
    scope:      str            # escopo ativo ("general" ou slug)
    db_dir:     str            # diretório dos artefatos do escopo ativo
    previous:   Optional[str]  = None  # escopo anterior (só no switch)
    opened:     Optional[bool] = None  # True se o escopo acabou de ser aberto


class ScopeDeleteResponse(BaseModel):
    """DELETE /project/{slug} — fecha o escopo (se aberto) e apaga o
    diretório db/<slug>/ INTEIRO (memórias LT, grafo, planos, arquivos
    indexados). "general" e o escopo ativo são recusados."""
    scope:   str
    deleted: bool


class FaceDictUpdateRequest(BaseModel):
    """Edita campos do card de pessoa. Campos não enviados (None) NÃO são
    alterados — só os campos presentes na requisição são atualizados."""
    description:     str = ""
    likes:           Optional[str] = None
    religion:        Optional[str] = None
    additional_info: Optional[str] = None


__all__ = [
    'FaceCandidate',
    'FaceDictEntry',
    'FaceDictReadRequest',
    'FaceDictReadResponse',
    'FaceDictUpdateRequest',
    'FaceDictWriteRequest',
    'FaceDictWriteResponse',
    'IdentityEntry',
    'IdentityUpdateRequest',
    'IndexedFileCheckResponse',
    'IndexedFileEntry',
    'IndexedFileReadRequest',
    'IndexedFileReadResponse',
    'IndexedFileWriteRequest',
    'IndexedFileWriteResponse',
    'MemoryEntry',
    'MemoryToolError',
    'PlanCreateRequest',
    'PlanCreateResponse',
    'PlanDeleteRequest',
    'PlanDeleteResponse',
    'PlanEntry',
    'PlanListResponse',
    'PeopleCardEntry',
    'PeopleCardUpdateRequest',
    'PeopleCardWriteRequest',
    'PeopleCardWriteResponse',
    'PlanReadRequest',
    'PlanReadResponse',
    'PlanStep',
    'PlanUpdateRequest',
    'PlanUpdateResponse',
    'ReadRequest',
    'ReadResponse',
    'ReadSTRequest',
    'ReadSTResponse',
    'RelatedMemory',
    'ScopeDeleteResponse',
    'ScopeInfoResponse',
    'ScopeSwitchRequest',
    'AgentSkillMatch',
    'AgentSkillSelectRequest',
    'AgentSkillSelectResponse',
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
