from __future__ import annotations

"""
memory_config.py — Configuração central do AVA Memory.

Todas as constantes de tuning (paths, thresholds, meias-vidas, parâmetros
de grafo/MMR/PPR, orçamento de tokens, segmentation) vivem aqui. Nenhum
outro módulo memory_* declara constantes de configuração — importe daqui.

Extraído do memory.py original sem alteração de valores.
"""

import logging
import os
import sys
from pathlib import Path

# ── Bootstrap de imports da pasta-mãe ─────────────────────────────────────────
# Os módulos memory_* vivem em Modules/memory/, mas dependem de recursos da
# pasta-mãe (Modules/): onnx_client.py, config.py e o pacote modules/.
# Como o servidor roda a partir do PRÓPRIO diretório (`cd Modules/memory &&
# python3 memory_api.py` — ver MEMORY_API_CWD no start.sh), a pasta-mãe não
# está no sys.path por padrão. Este é o ponto único de bootstrap: todo módulo
# memory_* importa memory_config antes de qualquer dependência da mãe.
_BASE_DIR    = Path(__file__).resolve().parent   # .../Modules/memory
_MODULES_DIR = str(_BASE_DIR.parent)             # .../Modules
if _MODULES_DIR not in sys.path:
    sys.path.insert(0, _MODULES_DIR)

# Diretório de dados do memory — ancorado no próprio pacote, independe do CWD
_DB_DIR = _BASE_DIR / "db"

# Logger compartilhado por todos os módulos memory_*
# ── Configuração ───────────────────────────────────────────────────────────────

ONNX_SERVING_URL = "http://localhost:2002"

# ── Prefixos E5 (multilingual-e5-small/base/large e primos) ─────────────────
# Modelos da família E5 são treinados com prefixos textuais fixos — sem eles,
# o embedding de QUALQUER texto colapsa para uma região estreita do espaço,
# e cosine similarity entre frases sem nenhuma relação semântica sai
# artificialmente alta (~0.85-0.93), quebrando dedup e busca por igual.
# Ref.: model card intfloat/multilingual-e5-small.
# Se um dia trocar pra um modelo que não é da família E5, ajuste isso pra "".
EMBED_QUERY_PREFIX   = "query: "
EMBED_PASSAGE_PREFIX = "passage: "

# Todos os paths abaixo ficam em Modules/memory/db/ e são ancorados via _DB_DIR
# (funcionam com qualquer CWD; antes eram "./memory/db/…" e só batiam rodando
# com CWD=Modules/).

# Longo prazo
DB_PATH           = str(_DB_DIR / "ava_memory.db")
FAISS_INDEX_PATH  = str(_DB_DIR / "ava_memory.index")
FAISS_ID_MAP_PATH = str(_DB_DIR / "ava_id_map.npy")

# Curto prazo
ST_DB_PATH           = str(_DB_DIR / "ava_short_term.db")
ST_FAISS_INDEX_PATH  = str(_DB_DIR / "ava_short_term.index")
ST_FAISS_ID_MAP_PATH = str(_DB_DIR / "ava_short_term_id_map.npy")

# Knowledge (Vector Store / KG-RAG)
VS_DB_PATH           = str(_DB_DIR / "ava_kg_chunks.db")
VS_FAISS_INDEX_PATH  = str(_DB_DIR / "ava_kg_vectors.index")
VS_FAISS_ID_MAP_PATH = str(_DB_DIR / "ava_kg_vectors_id_map.npy")
VS_MIN_SCORE         = 0.70

# ── NEW: Indexed Files (local-scraping) ────────────────────────────────────────
IF_DB_PATH           = str(_DB_DIR / "ava_indexed_files.db")
IF_FAISS_INDEX_PATH  = str(_DB_DIR / "ava_indexed_files.index")
IF_FAISS_ID_MAP_PATH = str(_DB_DIR / "ava_indexed_files_id_map.npy")
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
VD_DB_PATH           = str(_DB_DIR / "ava_visual_dict.db")
VD_FAISS_INDEX_PATH  = str(_DB_DIR / "ava_visual_dict.index")
VD_FAISS_ID_MAP_PATH = str(_DB_DIR / "ava_visual_dict_id_map.npy")
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
FD_DB_PATH           = str(_DB_DIR / "ava_face_dict.db")
FD_FAISS_INDEX_PATH  = str(_DB_DIR / "ava_face_dict.index")
FD_FAISS_ID_MAP_PATH = str(_DB_DIR / "ava_face_dict_id_map.npy")
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
VO_DB_PATH           = str(_DB_DIR / "ava_voice_dict.db")
VO_FAISS_INDEX_PATH  = str(_DB_DIR / "ava_voice_dict.index")
VO_FAISS_ID_MAP_PATH = str(_DB_DIR / "ava_voice_dict_id_map.npy")
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
LANCE_DIR          = str(_DB_DIR / "ava_lance")
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
TOOLS_DB_PATH               = str(_DB_DIR / "ava_tools.db")
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
ACTIVATION_LOG_PATH          = str(_DB_DIR / "activation_log.jsonl")
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

# ── Auto-link na ESCRITA (grafo Hebbiano) ─────────────────────────────────────
# Antes, arestas entre memórias só nasciam de co-ativação em LEITURAS (Step 4)
# ou de link_to explícito (que ninguém enviava) — uma memória gravada ficava
# isolada no grafo até que alguém pedisse as duas juntas. Agora cada memória LT
# recém-gravada é ligada aos seus vizinhos semânticos JÁ EXISTENTES.
#
# ── CALIBRAÇÃO (medida real do multilingual-e5-small neste deployment) ───────
#   * norma dos vetores: 1.0 (ok) — anisotropia extrema: pares COMPLETAMENTE
#     sem relação (idiomas/domínios distintos) saem com cosine MÉDIO 0.78
#     (min 0.71, max 0.86). O cross-encoder ms-marco, em contraste, é
#     discriminativo: ~0.000 p/ não-relacionados (visto em produção),
#     >=0.55 faixa update, >=0.90 duplicata.
#   * CONCLUSÃO: o cosine do bi-encoder NÃO separa relacionado de não-
#     relacionado — serve só como pré-filtro barato do custo do rerank. QUEM
#     DECIDE é o cross-encoder (AUTO_LINK_MIN_CROSS).
# Todos os valores são env-tunable p/ recalibração sem tocar código.
AUTO_LINK_ENABLED        = os.getenv("AUTO_LINK_ENABLED", "1") == "1"
AUTO_LINK_TOP_K          = int(os.getenv("AUTO_LINK_TOP_K", "5"))
AUTO_LINK_MIN_COSINE     = float(os.getenv("AUTO_LINK_MIN_COSINE", "0.80"))
                                  # só p/ podar candidatos do rerank: pouco acima
                                  # do piso de ruído medido (0.71-0.78)
AUTO_LINK_MIN_CROSS      = float(os.getenv("AUTO_LINK_MIN_CROSS", "0.25"))
                                  # GATE REAL: >> ruído do cross (~0.000 medido),
                                  # << faixa update (0.55) — banda "tópico
                                  # relacionado, não-paráfrase"
AUTO_LINK_INITIAL_WEIGHT = float(os.getenv("AUTO_LINK_INITIAL_WEIGHT", "0.5"))
AUTO_LINK_MAX_EDGES      = int(os.getenv("AUTO_LINK_MAX_EDGES", "3"))
                                  # teto de arestas criadas por escrita (anti-hub)

# ── Backfill no boot ("sono" do memory) ───────────────────────────────────────
# Na inicialização, consolida retroativamente o grafo: liga memórias que já
# estavam gravadas ANTES do auto-link existir (ou sob thresholds antigos) —
# analogia ao sono consolidando memórias do dia. Fire-and-forget, com delay
# para o ONNX serving terminar de subir. Roda a CADA boot (idempotente: arestas
# existentes são apenas reforçadas, com crescimento saturante até 1.0).
AUTO_LINK_BACKFILL          = os.getenv("AUTO_LINK_BACKFILL", "1") == "1"
AUTO_LINK_BACKFILL_DELAY_S  = float(os.getenv("AUTO_LINK_BACKFILL_DELAY_S", "10"))
AUTO_LINK_BACKFILL_MAX_NODES = int(os.getenv("AUTO_LINK_BACKFILL_MAX_NODES", "5000"))
AUTO_LINK_BACKFILL_BATCH    = int(os.getenv("AUTO_LINK_BACKFILL_BATCH", "64"))

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

# ── Rerank do /read com o cross-encoder (precisão do top-k final) ─────────────
# O ranking do /read é bi-encoder (cosine) + PPR + MMR — o cross-encoder
# (ms-marco-MiniLM, já carregado p/ dedup) só era usado na ESCRITA. Aqui ele
# reordena os candidatos "primary" (LT/ST/VS/IF) contra a query real ANTES do
# corte final: cosine de bi-encoder é rápido mas anisotrópico (ordenação
# imperfeita); o cross-encoder compara query×texto diretamente. Sendo apenas
# ~12 candidatos, o custo é 1 chamada /v1/score (~20-50ms). O score final é
# uma mistura (cross-encoder decide bem relevância; cosine preserva o sinal
# de "quão próximo do vetor gravado" e mantém a escala dos thresholds).
READ_RERANK_ENABLED       = True
READ_RERANK_TOP_N         = 12    # nº de candidatos primary a reranquear
READ_RERANK_CROSS_WEIGHT  = 0.65  # peso do cross-encoder no score final
READ_RERANK_COSINE_WEIGHT = 0.35  # peso do cosine original no score final

# ── Overrides por request (chamadores podem pedir recall mais rico) ───────────
# ReadRequest.top_k_final / ReadRequest.total_max_chars permitem que o caller
# (ex.: recall de chat) peça mais que o default conservador acima. Os teto
# abaixo evitam que um override descontrolado afogue o prompt.
READ_TOP_K_FINAL_MAX      = 12
READ_TOTAL_MAX_CHARS_MAX  = 12000

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

# Intervalo do job de manutenção periódica dos índices (ver index_flush_job).
# No LanceDB cada add() já é durável; o "flush" passou a ser a COMPACTAÇÃO dos
# fragmentos pequenos que os adds unitários geram (table.optimize()).
INDEX_FLUSH_INTERVAL_S = 60


__all__ = [
    'ACTIVATION_LOG_MAX_IDS',
    'ACTIVATION_LOG_PATH',
    'CHUNK_OVERLAP',
    'CHUNK_SIZE',
    'CONTEXT_MAX_CHARS',
    'CONTEXT_TURNS_FETCH',
    'DB_PATH',
    'DECAY_HALF_LIFE_DAYS',
    'DECAY_JOB_INTERVAL_S',
    'DEDUP_THRESHOLD',
    'DICT_GRAPH_MAX_RELATED',
    'DUAL_CONTEXT_WEIGHT',
    'EDGE_CONTRADICTS',
    'EDGE_CO_ACTIVATION',
    'EDGE_DECAY_HALF_LIFE_DAYS',
    'EDGE_DIRECTED_TYPES',
    'EDGE_EXPLICIT_LINK_WEIGHT',
    'EDGE_EXPLICIT_TYPES',
    'EDGE_LEARNING_RATE',
    'EDGE_MIN_SCORE_TO_LINK',
    'EDGE_MMR_FACTORS',
    'EDGE_PPR_FACTORS',
    'EDGE_PRUNE_THRESHOLD',
    'EDGE_TEMPORAL',
    'EDGE_TYPES',
    'EDGE_TYPES_NO_DECAY',
    'AUTO_LINK_ENABLED',
    'AUTO_LINK_TOP_K',
    'AUTO_LINK_MIN_COSINE',
    'AUTO_LINK_MIN_CROSS',
    'AUTO_LINK_INITIAL_WEIGHT',
    'AUTO_LINK_MAX_EDGES',
    'AUTO_LINK_BACKFILL',
    'AUTO_LINK_BACKFILL_DELAY_S',
    'AUTO_LINK_BACKFILL_MAX_NODES',
    'AUTO_LINK_BACKFILL_BATCH',
    'EDGE_UPDATES',
    'EMBED_DIM',
    'EMBED_PASSAGE_PREFIX',
    'EMBED_QUERY_PREFIX',
    'FAISS_ID_MAP_PATH',
    'FAISS_INDEX_PATH',
    'FD_AMBIGUOUS_MARGIN',
    'FD_DB_PATH',
    'FD_EMBED_DIM',
    'FD_FAISS_ID_MAP_PATH',
    'FD_FAISS_INDEX_PATH',
    'FD_MIN_SCORE',
    'FD_TOP_K',
    'HEBBIAN_LINK_MAX_IDS',
    'IF_DB_PATH',
    'IF_EMBED_BATCH_SIZE',
    'IF_FAISS_ID_MAP_PATH',
    'IF_FAISS_INDEX_PATH',
    'IF_MAX_CHUNKS',
    'IF_MAX_CONTENT_SIZE',
    'IF_MIN_SCORE',
    'IF_MIN_SCORE_READ',
    'INDEX_FLUSH_INTERVAL_S',
    'LANCE_DIR',
    'LANCE_TABLE_PREFIX',
    'LTP_ACCESS_BOOST',
    'MMR_COSINE_FLOOR',
    'MMR_LAMBDA',
    'MT_FACE_DICT',
    'MT_INDEXED_FILE',
    'MT_LONG_TERM',
    'MT_SHORT_TERM',
    'MT_TOOL',
    'MT_VISUAL_DICT',
    'MT_VOICE_DICT',
    'ONNX_SERVING_URL',
    'PPR_CONVERGENCE_EPS',
    'PPR_DAMPING',
    'PPR_EXPAND_MAX_NODES',
    'PPR_EXPAND_MIN_PATH_WEIGHT',
    'PPR_MAX_ITER',
    'PPR_MIN_ACTIVATION_REINFORCE',
    'PPR_MIN_NEIGHBORS_PER_HOP',
    'PPR_SPREAD_WEIGHT',
    'QUERY_AMBIGUOUS_RATIO',
    'QUERY_SHORT_WORDS',
    'READ_IF_MAX_CHARS',
    'READ_LT_MAX_CHARS',
    'READ_MIN_SCORE',
    'READ_MIN_SCORE_STRICT',
    'READ_RERANK_ENABLED',
    'READ_RERANK_TOP_N',
    'READ_RERANK_CROSS_WEIGHT',
    'READ_RERANK_COSINE_WEIGHT',
    'READ_TOP_K_FINAL_MAX',
    'READ_TOTAL_MAX_CHARS_MAX',
    'READ_ST_MAX_CHARS',
    'READ_TOP_K_FINAL',
    'READ_TOTAL_MAX_CHARS',
    'READ_VS_MAX_CHARS',
    'RERANK_DUPLICATE_SCORE',
    'RERANK_UPDATE_SCORE',
    'SEGMENT_MAX_COUNT',
    'SEGMENT_MIN_WORDS',
    'SEGMENT_TRIGGER_WORDS',
    'ST_CLEANUP_INTERVAL_S',
    'ST_DB_PATH',
    'ST_FAISS_ID_MAP_PATH',
    'ST_FAISS_INDEX_PATH',
    'ST_READ_DEFAULT_PAIRS',
    'ST_TTL_HOURS',
    'TOOLS_DB_PATH',
    'TOOLS_FALLBACK_TOP_K',
    'TOOLS_LEARN_DEDUP_SCORE',
    'TOOLS_LOW_CONFIDENCE_SCORE',
    'TOOLS_MAX_EXAMPLES',
    'TOOLS_MAX_LEARNED_EXAMPLES',
    'TOOLS_MAX_RELATED',
    'TOOLS_MIN_SCORE',
    'TOOLS_TOP_K',
    'TOOLS_USAGE_MAX_SEQUENCE',
    'TOP_K_READ',
    'UPDATE_SIM_THRESHOLD',
    'VD_AMBIGUOUS_MARGIN',
    'VD_DB_PATH',
    'VD_EMBED_DIM',
    'VD_FAISS_ID_MAP_PATH',
    'VD_FAISS_INDEX_PATH',
    'VD_MIN_SCORE',
    'VD_TOP_K',
    'VO_AMBIGUOUS_MARGIN',
    'VO_DB_PATH',
    'VO_EMBED_DIM',
    'VO_FAISS_ID_MAP_PATH',
    'VO_FAISS_INDEX_PATH',
    'VO_MIN_SCORE',
    'VO_TOP_K',
    'VS_DB_PATH',
    'VS_FAISS_ID_MAP_PATH',
    'VS_FAISS_INDEX_PATH',
    'VS_MIN_SCORE',
    '_STOP_WORDS',
    'log',
]
