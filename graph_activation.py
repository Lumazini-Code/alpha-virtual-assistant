"""graph_activation.py — Visualização em tempo real do grafo Hebbiano (memórias LT + tools).

Arquitetura (v2 — polling de diff):
  ══════════════════════════════════════════════════════════════════════════
  ANTES: snapshot estático carregado UMA vez no boot + SSE só de ativação PPR.
  Nós/arestas criados depois do boot nunca apareciam — com o auto-link na
  escrita, o grafo cresce e o visualizador ficava cego.

  AGORA: o GraphState faz polling do SQLite (~1s) e emite DELTA por SSE:
    - nós novos / atualizados / removidos (memória gravada, texto corrigido,
      decay podando memória esquecida);
    - arestas novas / reforçadas / removidas — captura QUALQUER origem
      automaticamente (auto-link na escrita, hebbiano em leitura, link_to
      explícito, cadeias temporais do batch, reforço PPR), sem o memory
      module precisar emitir evento nenhum.

  v3 — TOOLS: o mesmo polling cobre o grafo de tools (ToolsDB: ava_tools.db,
  tabelas `tools` + `tool_edges` — mesmo esquema de arestas tipadas). Tools
  (nativas + MCP registradas no mcp_manager) entram como nós HEXAGONAIS
  verdes com id prefixado ("t{id}") — os ids inteiros de tool colidem com os
  de memória, então o prefixo é obrigatório. Eventos de ativação PPR no
  JSONL carregam "graph" ("memory" | "tools", ver memory_graph._log_activation_sync)
  e a UI prefixa os ids de tools antes de acender.

  v4 — PLANS: os planos de execução híbridos (PlansDB: ava_plans.db) entram
  como nós DIAMANTE roxos com id prefixado ("p{id}") — sem arestas (nós
  soltos). Cobrem criação (POST /plans/write) e edição (PATCH /plans/update,
  que versiona — o diff acende "updated" a cada versão nova) feitas pela TUI
  ou pelo plan_recorder.

  v5 — MCP DB (fonte de verdade das tools MCP): o registry do mcp_manager
  (Rust no HOST: Modules/mcp_manager/db/ava_mcp_servers.db, tabelas
  `mcp_servers` + `mcp_tools`) entra no mesmo espaço visual. Como as ARESTAS
  de co-uso só existem na ToolsDB (ids inteiros), a fusão é por graph_name
  (`servidor__tool`):
    - tool presente nos DOIS: o nó `t{id}` MANTÉM o id (arestas intactas) e
      o CONTEÚDO passa a vir da MCP DB (snippet como descrição, core,
      transporte, enabled do servidor);
    - tool que só existe na MCP DB (sync pendente / memory offline): entra
      como nó extra id "m{servidor__tool}" (prefixo m, sem arestas);
    - servidor desabilitado aparece com badge "desabilitado" no tooltip/painel.

  v6 — MULTI-ESCOPO (projetos): o visualizador mostra TODOS os escopos de
  memória AO MESMO TEMPO, separados visualmente. O db raiz da memory
  (Modules/memory/db/) tem o escopo "geral" + um subdiretório por projeto
  (db/<slug>/ = LT + planos + grafo Hebbiano do projeto). A cada poll:
    - descoberta de escopos (os.scandir do db root — barato);
    - scan POR ESCOPO (ava_memory.db + ava_plans.db) com ids NAMESPACED:
      memória "{scope}:{id}" (ex. "geral:12", "meu-proj:3") e plano
      "{scope}:p{id}" — ids inteiros colidem ENTRE escopos;
    - tools/MCP/pessoas/identidade continuam GLOBAIS (ids t/m/u/a sem escopo);
    - cada nó/aresta carrega "scope" (null = global) — o cliente colore a
      borda pela cor do escopo, posiciona cada escopo num SETOR próprio e
      oferece liga/desliga por escopo na legenda;
    - arestas virtuais cruzadas: plan_tool/plan_person (plano do escopo →
      tool/pessoa global), person_identity (card global → memória de
      identidade que pode morar em QUALQUER escopo — resolvido sondando os
      ids de memória coletados por escopo no próprio scan) e
      identity_memory (card de identidade da Alpha "a1" → memória LT que
      representa a própria assistente no grafo Hebbiano);
    - resiliência: erro transitório de SQLite num escopo NÃO zera o delta
      (nós/arestas do escopo em falha ficam fora da passada de remoção —
      snapshot do escopo permanece); escopo cujo diretório desaparece tem
      seus nós removidos normalmente.
  A ativação PPR também é multi-escopo: UM ActivationTailer POR escopo
  (db/activation_log.jsonl + db/<slug>/activation_log.jsonl), cada evento
  publicado ganha "scope" e o /graph/replay funde todos os logs por
  timestamp. O escopo ATIVO da memory API (POST /project/switch) vira
  {"type":"scope_switch"} → badge "ativo" na legenda de escopos da UI.
  ══════════════════════════════════════════════════════════════════════════

  Eventos SSE (o cliente roteia por "type"):
    {"type": "activation",  "graph", "scope", "ts", "seeds", "rank",
                            "elapsed_ms", "query", "strategy", "session_id"}
                            ← PPR numa leitura; "scope" = escopo dono do log
    {"type": "scope_switch", "scope", "ts"}   ← escopo ativo mudou (badge UI)
    {"type": "graph_delta", "ts",
                            "nodes": {"added": [...], "updated": [...], "removed": [id,...]},
                            "edges": {"added": [...], "updated": [...],
                                      "removed": ["a|b|type", ...]}}
      — nós de tools têm id string "t{id}", kind="tool"; planos
        "{scope}:p{id}", kind="plan"; pessoas "u{id}", kind="person";
        identidade da Alpha "a{id}", kind="identity" (singleton "a1");
        memórias "{scope}:{id}" (string namespaced), kind="memory";
        nodes/edges carregam "scope" (null = global: tools/MCP/pessoas).

  Endpoints:
    GET /                  → UI (vis-network, overhaul: cores/setas/weights por
                             tipo de aresta, painel de inspeção de nó, filtros,
                             busca, decay slider, replay, legenda de ESCOPOS
                             com contagem/ativo/toggle)
    GET /graph             → snapshot atual enriquecido (boot do cliente;
                             + "scopes" e "active_scope")
    GET /events            → SSE (activation + graph_delta + scope_switch,
                             keepalive 15s)
    GET /graph/replay?n=N  → últimos N eventos de ativação FUNDIDOS de todos
                             os logs de escopo (ordenados por timestamp)

Rodar:
    python graph_activation.py                      (defaults abaixo)
    python graph_activation.py --port 8777 --poll 1.0
    python graph_activation.py --tools-db ./Modules/memory/db/ava_tools.db
"""

import argparse
import html
import json
import os
import queue
import sqlite3
import threading
import time
import urllib.request
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ── NEW (projetos): auth interna p/ consultar a memory API (/status). O
# pacote vive em Modules/internal_auth.py — o import depende de COMO o script
# é chamado (repo root × diretório próprio); env AVA_INTERNAL_TOKEN/AVA_BIND_HOST
# é o fallback (mesma política fail-open dos demais clientes internos).
try:
    from Modules.internal_auth import internal_bind_host, internal_token
except ImportError:  # script fora da raiz do repo
    try:
        from internal_auth import internal_bind_host, internal_token  # type: ignore
    except ImportError:
        def internal_bind_host() -> str:
            return os.getenv("AVA_BIND_HOST", "127.9.12.22")

        def internal_token() -> str | None:
            return os.getenv("AVA_INTERNAL_TOKEN")


# ─────────────────────────────────────────────────────────────────────────────
#                           SNAPSHOT + DIFF (GraphState)
# ─────────────────────────────────────────────────────────────────────────────

def _open_ro(db_path: str) -> sqlite3.Connection:
    """Conexão SQLite read-only por poll — nunca trava o memory writer
    (WAL permite leitor concorrente; mode=ro garante que ESTE processo não
    escreve nem cria arquivo novo por engano)."""
    p = Path(db_path).resolve()
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=1.0)
    conn.row_factory = sqlite3.Row
    return conn


def _node_public(row: sqlite3.Row, scope: str | None = None) -> dict:
    text = (row["text"] or "").strip().replace("\n", " ")
    return {
        # NEW (multi-escopo): id NAMESPACED "{scope}:{id}" — os inteiros
        # colidem ENTRE escopos (memória 1 do geral ≠ memória 1 do projeto).
        "id":          f"{scope}:{row['id']}" if scope else row["id"],
        "kind":        "memory",
        "label":       text[:40] + ("…" if len(text) > 40 else ""),
        "title":       html.escape(text[:400]),
        "text":        row["text"],
        "source":      row["source"],
        "confidence":  round(row["confidence"], 4),
        "created_at":  row["created_at"],
        "access_count": row["access_count"],
        "forgettable": bool(row["forgettable"]),
        # escopo dono da memória ("geral" | slug do projeto). Tools/MCP/
        # pessoas são GLOBAIS → scope=None.
        "scope":       scope,
    }


def _tool_node_public(row: sqlite3.Row) -> dict:
    """Nó público de uma tool (ToolsDB). Id STRING com prefixo "t" — os ids
    inteiros de tool colidem com os de memória no mesmo espaço visual. O
    servidor MCP é inferido do graph name (`servidor__tool`, gravado pelo
    mcp_manager); tools nativas não têm o separador."""
    name = (row["name"] or "").strip()
    desc = (row["description"] or "").strip().replace("\n", " ")
    server = name.split("__", 1)[0] if "__" in name else None
    return {
        "id":          f"t{row['id']}",
        "kind":        "tool",
        "label":       name[:40] + ("…" if len(name) > 40 else ""),
        "title":       html.escape(desc[:400]),
        "text":        row["description"],   # p/ busca (mesma chave das memórias)
        "name":        name,
        "server":      server,               # None = tool nativa
        "core":        bool(row["is_core"]),
        "use_count":   row["use_count"],
        "last_used":   row["last_used"],
        "created_at":  row["created_at"],
        "source":      f"tool:{server}" if server else "tool",
        "scope":       None,   # tools são GLOBAIS (compartilhadas entre escopos)
    }


def _mcp_tool_node_public(graph_name: str, srow: sqlite3.Row | None,
                          trow: sqlite3.Row) -> dict:
    """Nó público de tool MCP que existe SÓ na MCP registry do mcp_manager
    (Modules/mcp_manager/db/ava_mcp_servers.db) — ainda sem sync na ToolsDB
    (ex.: memory offline na hora do /sync). Id string com prefixo "m" +
    graph_name — nunca colide com t{N}/p{N}/ids inteiros. Sem arestas (o
    co-uso só existe na ToolsDB). O snippet (skill.md) é a descrição."""
    snippet = ((trow["snippet"] or "").strip()
               or (trow["description_raw"] or "").strip())
    snippet = snippet or f"Tool '{trow['name']}' do servidor MCP '{trow['server']}'."
    return {
        "id":          f"m{graph_name}",
        "kind":        "tool",
        "label":       graph_name[:40] + ("…" if len(graph_name) > 40 else ""),
        "title":       html.escape(snippet[:400]),
        "text":        snippet,              # p/ busca (mesma chave das demais)
        "name":        graph_name,
        "server":      trow["server"],
        "core":        bool(trow["core"]),
        "use_count":   0,
        "last_used":   None,
        "created_at":  None,
        "source":      f"tool:{trow['server']}",
        # metadata do registry (não existe na ToolsDB)
        "transport":   srow["transport"] if srow is not None else None,
        "enabled":     bool(srow["enabled"]) if srow is not None else True,
        "always_on":   bool(srow["always_on"]) if srow is not None else False,
        "scope":       None,   # registry MCP é GLOBAL (host, fora dos escopos)
    }


def _edge_public(row: sqlite3.Row, scope: str | None = None) -> dict:
    """Aresta do grafo Hebbiano de memórias. NEW (multi-escopo): os ids
    inteiros colidem ENTRE escopos — key/from/to ganham o prefixo
    "{scope}:" (ou ficam crus quando scope=None, formato antigo)."""
    p = f"{scope}:" if scope else ""
    return {
        "key":   f"{p}{row['memory_id_a']}|{p}{row['memory_id_b']}|{row['edge_type']}",
        "from":  f"{p}{row['memory_id_a']}",
        "to":    f"{p}{row['memory_id_b']}",
        "type":  row["edge_type"],
        "weight": round(row["weight"], 4),
        "count": row["coactivation_count"],
        "last":  row["last_coactivated"],
        "scope": scope,
    }


def _tool_edge_public(row: sqlite3.Row) -> dict:
    """Aresta do grafo de tools — mesmo formato das de memória, com o
    prefixo "t" nos ids (nós e key) pra não colidir no espaço visual."""
    return {
        "key":   f"t{row['memory_id_a']}|t{row['memory_id_b']}|{row['edge_type']}",
        "from":  f"t{row['memory_id_a']}",
        "to":    f"t{row['memory_id_b']}",
        "type":  row["edge_type"],
        "weight": round(row["weight"], 4),
        "count": row["coactivation_count"],
        "last":  row["last_coactivated"],
        "scope": None,   # tools são GLOBAIS
    }


def _plan_node_public(row: sqlite3.Row, scope: str | None = None) -> dict:
    """Nó público de um plano de execução (PlansDB: ava_plans.db). Id STRING
    com prefixo "p" — liga-se por arestas virtuais ao card de pessoa
    (plans.person_id) e às tools do plano (plans.tools_json). NEW
    (multi-escopo): prefixado com o escopo dono ("{scope}:p{id}")."""
    name = (row["name"] or "").strip()
    desc = (row["description"] or "").strip()
    steps_count = len(json.loads(row["steps_json"] or "[]"))
    try:
        tool_ids = json.loads(row["tools_json"] or "[]")
    except (json.JSONDecodeError, TypeError):
        tool_ids = []
    return {
        "id":          f"{scope}:p{row['id']}" if scope else f"p{row['id']}",
        "kind":        "plan",
        "label":       name[:40] + ("…" if len(name) > 40 else ""),
        "title":       html.escape(desc[:400]),
        "text":        row["description"],   # p/ busca (mesma chave das demais)
        "name":        name,
        "description": desc,
        "version":     row["version"],
        "steps_count": steps_count,
        # ── NEW (people cards / tools ↔ planos): vínculos persistidos
        "person_id":   row["person_id"],
        "tool_ids":    tool_ids,
        "created_at":  row["created_at"],
        "updated_at":  row["updated_at"],
        "source":      "plan",
        "scope":       scope,
    }


def _person_node_public(row: sqlite3.Row) -> dict:
    """Nó público de um card de pessoa (PeopleDB: ava_people.db). Id STRING
    com prefixo "u" — t/p/m já estão ocupados. O card liga-se por arestas
    virtuais à memória de identidade (people.memory_id) e aos planos que o
    citam (plans.person_id). voice_count/face_count vêm de subselects."""
    name = (row["person_name"] or "").strip()
    desc = (row["description"] or "").strip()
    return {
        "id":          f"u{row['id']}",
        "kind":        "person",
        "label":       name[:40] + ("…" if len(name) > 40 else ""),
        "title":       html.escape(desc[:400]),
        "text":        f"{name}. {desc}".strip(),   # p/ busca (nome + descrição)
        "name":        name,
        "description": desc,
        "likes":       row["likes"],
        "religion":    row["religion"],
        "additional_info": row["additional_info"],
        "memory_id":   row["memory_id"],     # nó de identidade no grafo de memórias
        "voice_count": row["voice_count"],   # exemplos de voz (CAM++)
        "face_count":  row["face_count"],    # exemplos de rosto (EdgeFace)
        "confidence":  round(row["confidence"], 4),
        "access_count": row["access_count"],
        "created_at":  row["created_at"],
        "last_accessed": row["last_accessed"],
        "source":      row["source"],
        "scope":       None,   # pessoas/cards são GLOBAIS (dicionário compartilhado)
    }


def _identity_node_public(row: sqlite3.Row) -> dict:
    """Nó público do card de IDENTIDADE da Alpha (PeopleDB: tabela `identity`,
    singleton id=1). Id STRING com prefixo "a" — t/p/m/u já estão ocupados.
    Liga-se por aresta virtual à memória de identidade LT (identity.memory_id),
    que pode morar em qualquer escopo (mesma sondagem do person_identity)."""
    name = (row["name"] or "").strip()
    desc = (row["description"] or "").strip()
    return {
        "id":          f"a{row['id']}",   # prefixo "a" (t/p/m/u já ocupados)
        "kind":        "identity",
        "label":       name[:40] + ("…" if len(name) > 40 else ""),
        "title":       html.escape(desc[:400]),
        "text":        f"{name}. {desc} {row['roles']}".strip(),  # p/ busca
        "name":        name,
        "description": desc,
        "roles":       row["roles"],
        "additional_info": row["additional_info"],
        "memory_id":   row["memory_id"],   # nó de identidade no grafo de memórias
        "source":      row["source"],
        "created_at":  row["created_at"],
        "updated_at":  row["updated_at"],
        "scope":       None,   # a identidade é GLOBAL (existe uma só, sem escopo)
    }


class GraphState:
    """Snapshot do grafo + diff por poll. Scan completo das tabelas a
    cada tick — com centenas/milhares de linhas custa poucos ms, e diff
    completo cobre criação, atualização E remoção (decay job) sem depender
    de surrogate keys.

    NEW (v6 — MULTI-ESCOPO): cobre TODOS os escopos de memória ao mesmo
    tempo. O db raiz da memory (pai de --db) é o escopo "geral"; cada
    subdiretório (db/<slug>/ = projeto criado via /projects) tem LT +
    planos do projeto. Cobre, no mesmo espaço visual:
      - memórias LT POR ESCOPO (ava_memory.db) — ids "{scope}:{id}"
        (ex. "geral:12", "meu-proj:3"): os ids inteiros colidem ENTRE
        escopos, então o namespacing é obrigatório;
      - planos POR ESCOPO (ava_plans.db) — ids "{scope}:p{id}";
      - tools (ava_tools.db) — GLOBAIS, ids "t{id}", kind="tool"; o DB é
        opcional (memory process pode não tê-lo criado ainda);
      - pessoas/cards (ava_people.db) — GLOBAIS, ids "u{id}", kind="person";
        opcional; liga-se às memórias pela memória de identidade;
      - identidade da Alpha (ava_people.db, tabela `identity` singleton) —
        GLOBAL, id "a{id}" (prefixo "a": t/p/m/u já ocupados),
        kind="identity"; liga-se à memória LT que a representa no grafo;
      - MCP registry (mcp_manager Rust no HOST: Modules/mcp_manager/db/
        ava_mcp_servers.db) — GLOBAIS, opcional; FUSÃO por graph_name com
        as tools da ToolsDB: conteúdo (snippet/core/enabled/transporte)
        vem daqui, ids t{N} e arestas ficam na ToolsDB; tools só-MCP
        entram como "m{servidor__tool}".

    Cada nó/aresta pública carrega "scope" ("geral" | slug | None p/ os
    globais) — o cliente separa os escopos por cor de borda, setor espacial
    e liga/desliga na legenda.

    Correlações ENTRE grafos (arestas virtuais derivadas dos metadados,
    recomputadas a cada scan):
      person_identity: card u{N} GLOBAL → memória de identidade — que pode
        morar em QUALQUER escopo; resolvido sondando os ids de memória
        coletados por escopo no próprio scan (geral primeiro, depois
        projetos em ordem alfabética);
      identity_memory: card de identidade a{N} GLOBAL → memória LT da
        própria Alpha (identity.memory_id) — mesma sondagem por escopo do
        person_identity;
      plan_person:     plano {scope}:p{N} ↔ card u{N} global (plans.person_id);
      plan_tool:       plano {scope}:p{N} ↔ tool t{N} global (plans.tools_json).

    Resiliência por escopo: erro transitório de SQLite num escopo (lock
    indisponível, bind mount lento) NÃO zera o delta — o escopo em falha é
    reportado em _scan() e seus nós/arestas ficam FORA da passada de
    remoção (snapshot do escopo permanece até a próxima leitura ok). Escopo
    cujo DIRETÓRIO desaparece (projeto apagado) sai da descoberta e seus
    nós são removidos normalmente."""

    TOOL_ID_PREFIX = "t"
    PLAN_ID_PREFIX = "p"
    MCP_ID_PREFIX = "m"
    PERSON_ID_PREFIX = "u"
    IDENTITY_ID_PREFIX = "a"   # card singleton da Alpha (tabela `identity`)
    VIRTUAL_EDGE_WEIGHT = 0.5   # peso fixo das arestas virtuais entre grafos
    GENERAL_SCOPE = "general"   # MESMO valor de memory_config.GENERAL_SCOPE —
                                # o badge da memory API tem que casar com o
                                # nome do escopo raiz do scan, senão nasce
                                # chip fantasma na UI

    def __init__(self, db_path: str, tools_db_path: str = "",
                 plans_db_path: str = "", mcp_db_path: str = "",
                 people_db_path: str = ""):
        self.db_path = db_path
        self.tools_db_path = tools_db_path
        self.plans_db_path = plans_db_path
        self.mcp_db_path = mcp_db_path
        self.people_db_path = people_db_path
        # NEW (multi-escopo): raiz dos escopos = diretório do db geral;
        # subdiretórios são os projetos (db/<slug>/ava_memory.db).
        self.db_root = Path(db_path).resolve().parent
        self.active_scope = self.GENERAL_SCOPE   # atualizado pelo ScopeWatcher
        self.lock = threading.Lock()
        self.nodes: dict = {}                                  # id -> node público ("{scope}:{id}" | "tN" | "pN" | "m..." | "uN")
        self.nodes_sig: dict = {}                              # id -> sig por tipo
        self.edges: dict[str, dict] = {}                       # key -> edge pública
        self.edges_sig: dict[str, tuple] = {}                  # key -> (weight, count)

    # ── NEW (multi-escopo): descoberta de escopos ────────────────────────
    def _discover_scopes(self) -> list[tuple[str, str, str]]:
        """Escopos visíveis agora: (nome, ava_memory.db, ava_plans.db).
        O db root é o escopo "geral" (usa os paths dos args); cada
        subdiretório é um projeto (slug = nome do diretório). Ordenado —
        determinístico p/ o cliente (cores/posições por ordem de aparição).
        os.scandir é barato (só readdir) — seguro a cada poll.
        Subdirs TÉCNICOS (convenção da memory: prefixo "ava_" — ex.
        ava_lance/, LanceDB de vetores do geral) NÃO são escopos."""
        scopes = [(self.GENERAL_SCOPE, self.db_path, self.plans_db_path)]
        try:
            for entry in sorted(os.scandir(self.db_root), key=lambda e: e.name):
                if entry.name.startswith("ava_"):
                    continue   # dir técnico (LanceDB etc.) — não é projeto
                if entry.is_dir(follow_symlinks=False):
                    d = Path(entry.path)
                    scopes.append((entry.name, str(d / "ava_memory.db"),
                                   str(d / "ava_plans.db")))
        except OSError:
            pass   # root sumiu (transitório) — só o geral, paths dos args
        return scopes

    def _resolve_identity_scope(self, mid: int,
                                mem_ids: dict[str, set]) -> str | None:
        """Escopo dono da memória de identidade `mid` (people.memory_id pode
        apontar pra qualquer escopo). Geral primeiro; senão o 1º projeto
        (ordem alfabética) que tiver a memória. None = memória não existe
        (ainda) em nenhum escopo → sem aresta."""
        if mid is None:
            return None
        if mid in mem_ids.get(self.GENERAL_SCOPE, ()):
            return self.GENERAL_SCOPE
        for scope in sorted(s for s in mem_ids if s != self.GENERAL_SCOPE):
            if mid in mem_ids[scope]:
                return scope
        return None

    def _scan(self) -> tuple[dict, dict, set]:
        """Scan completo (nodes, edges) — tudo público. NEW (multi-escopo):
        percorre TODOS os escopos descobertos (db root + subdiretórios),
        coleta os ids de memória por escopo (p/ resolver person_identity
        cruzada) e devolve também o conjunto de escopos EM FALHA transitória
        (o refresh os exclui da passada de remoção — snapshot permanece)."""
        nodes: dict = {}
        edges: dict[str, dict] = {}
        mem_ids: dict[str, set] = {}      # scope -> {id de memória} (identidade)
        failed_scopes: set[str] = set()

        # ── ESCOPOS: LT + planos de cada um (geral + projetos) ──────────
        for scope, mem_db, plans_db in self._discover_scopes():
            # Memórias + arestas hebbianas do escopo. Falha (lock/WAL/IO) →
            # marca o escopo como falho e segue pros próximos (os outros
            # escopos continuam atualizando; o falho mantém o snapshot).
            try:
                conn = _open_ro(mem_db)
                try:
                    snodes = {
                        f"{scope}:{r['id']}": _node_public(r, scope)
                        for r in conn.execute(
                            "SELECT id, text, source, confidence, created_at, "
                            "access_count, forgettable FROM memories"
                        )
                    }
                    sedges = {
                        _edge_public(r, scope)["key"]: _edge_public(r, scope)
                        for r in conn.execute(
                            "SELECT memory_id_a, memory_id_b, edge_type, weight, "
                            "coactivation_count, last_coactivated FROM memory_edges"
                        )
                    }
                finally:
                    conn.close()
            except (sqlite3.Error, FileNotFoundError, OSError):
                failed_scopes.add(scope)
                continue
            mem_ids[scope] = {int(nid.split(":", 1)[1]) for nid in snodes}
            nodes.update(snodes)
            edges.update(sedges)

            # Planos do escopo (opcional — DB pode não existir ainda)
            try:
                pconn = _open_ro(plans_db)
                try:
                    pnodes = {
                        f"{scope}:p{r['id']}": _plan_node_public(r, scope)
                        for r in pconn.execute(
                            "SELECT id, name, description, version, steps_json, "
                            "person_id, tools_json, created_at, updated_at FROM plans"
                        )
                    }
                finally:
                    pconn.close()
                nodes.update(pnodes)
            except (sqlite3.Error, FileNotFoundError, OSError):
                pass   # plans DB ainda não existe / sem tabela — ignora
        # ── Grafo de tools (opcional): mesmo esquema de arestas tipadas ──
        if self.tools_db_path:
            try:
                tconn = _open_ro(self.tools_db_path)
                try:
                    tnodes = {
                        f"{self.TOOL_ID_PREFIX}{r['id']}": _tool_node_public(r)
                        for r in tconn.execute(
                            "SELECT id, name, description, is_core, created_at, "
                            "last_used, use_count FROM tools"
                        )
                    }
                    tedges = {
                        _tool_edge_public(r)["key"]: _tool_edge_public(r)
                        for r in tconn.execute(
                            "SELECT memory_id_a, memory_id_b, edge_type, weight, "
                            "coactivation_count, last_coactivated FROM tool_edges"
                        )
                    }
                finally:
                    tconn.close()
                nodes.update(tnodes)
                edges.update(tedges)
            except (sqlite3.Error, FileNotFoundError):
                pass   # tools DB ainda não existe / sem tabelas — só memórias
        # ── NEW (people cards): grafo de pessoas/cards (opcional) ──
        if self.people_db_path:
            try:
                peconn = _open_ro(self.people_db_path)
                try:
                    unodes = {
                        f"{self.PERSON_ID_PREFIX}{r['id']}": _person_node_public(r)
                        for r in peconn.execute(
                            "SELECT p.id, p.person_name, p.description, p.likes, "
                            "p.religion, p.additional_info, p.source, p.confidence, "
                            "p.memory_id, p.created_at, p.last_accessed, p.access_count, "
                            "(SELECT COUNT(*) FROM voice_embeddings v "
                            " WHERE v.person_id = p.id) AS voice_count, "
                            "(SELECT COUNT(*) FROM face_embeddings f "
                            " WHERE f.person_id = p.id) AS face_count "
                            "FROM people p"
                        )
                    }
                finally:
                    peconn.close()
                nodes.update(unodes)
            except (sqlite3.Error, FileNotFoundError):
                pass   # people DB ainda não existe / sem tabela — ignora
            # NEW: card de IDENTIDADE da Alpha (tabela singleton `identity`).
            # Bloco SEPARADO de propósito: a tabela é nova (DBs antigos não a
            # têm) — erro aqui não pode derrubar os people cards de cima.
            try:
                iconn = _open_ro(self.people_db_path)
                try:
                    irow = iconn.execute(
                        "SELECT id, name, description, roles, additional_info, "
                        "memory_id, source, created_at, updated_at "
                        "FROM identity WHERE id = 1"
                    ).fetchone()
                finally:
                    iconn.close()
                if irow is not None:
                    nodes[f"{self.IDENTITY_ID_PREFIX}{irow['id']}"] = \
                        _identity_node_public(irow)
            except (sqlite3.Error, FileNotFoundError):
                pass   # tabela identity ainda não existe — ignora
        # ── MCP registry (opcional): fusão por graph_name ─────────────────
        # 1. Enriquece t-nodes da ToolsDB cujo name existe na MCP DB — o
        #    id t{N} MANTÉM (arestas de co-uso intactas) e o CONTEÚDO passa
        #    a vir daqui (snippet como descrição, core, enabled/transporte).
        # 2. Tools só na MCP DB (sync pendente / memory offline) entram
        #    como nós extras "m{graph_name}", sem arestas.
        if self.mcp_db_path:
            try:
                mconn = _open_ro(self.mcp_db_path)
                try:
                    servers = {
                        r["name"]: r
                        for r in mconn.execute(
                            "SELECT name, transport, enabled, always_on "
                            "FROM mcp_servers"
                        )
                    }
                    mcp_tools = {
                        f"{r['server']}__{r['name']}": (servers.get(r["server"]), r)
                        for r in mconn.execute(
                            "SELECT server, name, description_raw, snippet, "
                            "core FROM mcp_tools"
                        )
                    }
                finally:
                    mconn.close()

                for node in nodes.values():
                    if node.get("kind") != "tool" or node.get("server") is None:
                        continue
                    entry = mcp_tools.get(node["name"])
                    if entry is None:
                        continue
                    srow, trow = entry
                    snippet = ((trow["snippet"] or "").strip()
                               or (trow["description_raw"] or "").strip())
                    if snippet:
                        node["text"] = snippet
                        node["title"] = html.escape(snippet[:400])
                    node["core"] = bool(trow["core"])
                    node["transport"] = srow["transport"] if srow is not None else None
                    node["enabled"] = bool(srow["enabled"]) if srow is not None else True
                    node["always_on"] = bool(srow["always_on"]) if srow is not None else False

                known = {n["name"] for n in nodes.values() if n.get("kind") == "tool"}
                for gname, (srow, trow) in mcp_tools.items():
                    if gname in known:
                        continue
                    mnode = _mcp_tool_node_public(
                        gname, srow, trow
                    )
                    nodes[mnode["id"]] = mnode
            except (sqlite3.Error, FileNotFoundError):
                pass   # MCP DB ainda não existe / vazia — ToolsDB cobre
        # ── NEW (correlações entre grafos): arestas virtuais derivadas dos
        # metadados dos nós — não moram em tabela de arestas; são recomputadas
        # a cada scan e o diff detecta criação/remoção naturalmente (sem
        # back-edges fantasmas quando um plano/pessoa é apagado).
        #   person_identity: card u{N} GLOBAL → memória de identidade
        #     (people.memory_id) que pode morar em QUALQUER escopo —
        #     resolvido sondando mem_ids coletado por escopo acima;
        #   identity_memory: card de identidade a{N} GLOBAL → memória LT da
        #     própria Alpha (identity.memory_id) — mesma sondagem;
        #   plan_person:     plano {scope}:p{N} ↔ card u{N} GLOBAL
        #   plan_tool:       plano {scope}:p{N} ↔ tool t{N} GLOBAL
        # Aresta só nasce se AMBOS os nós existirem (identidade ainda sem
        # memória em nenhum escopo, plano sem card, tool não cadastrada →
        # sem aresta).
        virtual: dict[str, dict] = {}
        w = self.VIRTUAL_EDGE_WEIGHT
        for node in nodes.values():
            kind = node.get("kind")
            if kind == "person":
                mid = node.get("memory_id")
                scope_id = self._resolve_identity_scope(mid, mem_ids)
                if scope_id is not None:
                    target = f"{scope_id}:{mid}"
                    key = f"{node['id']}|{target}|person_identity"
                    virtual[key] = {
                        "key": key, "from": node["id"], "to": target,
                        "type": "person_identity", "weight": w,
                        "count": 1, "last": node.get("last_accessed") or time.time(),
                        "scope": scope_id,   # escopo da memória de identidade
                    }
            elif kind == "identity":
                # card único da Alpha → a memória LT que a representa
                mid = node.get("memory_id")
                scope_id = self._resolve_identity_scope(mid, mem_ids)
                if scope_id is not None:
                    target = f"{scope_id}:{mid}"
                    key = f"{node['id']}|{target}|identity_memory"
                    virtual[key] = {
                        "key": key, "from": node["id"], "to": target,
                        "type": "identity_memory", "weight": w,
                        "count": 1, "last": node.get("updated_at") or time.time(),
                        "scope": scope_id,   # escopo da memória de identidade
                    }
            elif kind == "plan":
                pscope = node.get("scope")   # escopo dono do plano
                pid = node.get("person_id")
                if pid is not None and f"{self.PERSON_ID_PREFIX}{pid}" in nodes:
                    puid = f"{self.PERSON_ID_PREFIX}{pid}"
                    key = f"{node['id']}|{puid}|plan_person"
                    virtual[key] = {
                        "key": key, "from": node["id"], "to": puid,
                        "type": "plan_person", "weight": w,
                        "count": 1, "last": node.get("updated_at") or time.time(),
                        "scope": pscope,
                    }
                for tid in node.get("tool_ids") or []:
                    tuid = f"{self.TOOL_ID_PREFIX}{tid}"
                    if tuid in nodes:
                        key = f"{node['id']}|{tuid}|plan_tool"
                        virtual[key] = {
                            "key": key, "from": node["id"], "to": tuid,
                            "type": "plan_tool", "weight": w,
                            "count": 1, "last": node.get("updated_at") or time.time(),
                            "scope": pscope,
                        }
        edges.update(virtual)
        return nodes, edges, failed_scopes

    def scope_log_paths(self) -> dict[str, str]:
        """NEW (multi-escopo): log de ativação POR escopo (geral + projetos) —
        fonte do /graph/replay (funde todos, ordenando por timestamp) e do
        ScopeWatcher (um ActivationTailer por escopo). Subdirs técnicos
        (prefixo "ava_") não são escopos."""
        logs = {self.GENERAL_SCOPE: str(self.db_root / "activation_log.jsonl")}
        try:
            for entry in sorted(os.scandir(self.db_root), key=lambda e: e.name):
                if entry.name.startswith("ava_"):
                    continue   # dir técnico (LanceDB etc.)
                if entry.is_dir(follow_symlinks=False):
                    logs[entry.name] = str(Path(entry.path) / "activation_log.jsonl")
        except OSError:
            pass
        return logs

    def refresh(self) -> dict:
        """Um poll: escaneia, dif contra o snapshot e devolve o delta.
        Retorna {"nodes": {"added","updated","removed"},
                 "edges": {"added","updated","removed"}} — sempre presente,
        com listas vazias quando nada mudou.
        NEW (multi-escopo): erro transitório num escopo (failed_scopes) NÃO
        zera o delta — os demais escopos atualizam normalmente e os nós/
        arestas do escopo em falha ficam FORA da passada de remoção (o
        snapshot dele permanece; o escopo volta a difar na leitura ok)."""
        try:
            db_nodes, db_edges, failed_scopes = self._scan()
        except sqlite3.Error as e:
            # Erro de leitura FORA do escopo (descoberta/abertura geral).
            # Sem backoff isso spamava o log a cada 1s: loga a 1ª falha e
            # depois 1 a cada 30 (~30s no interval default). O snapshot
            # anterior permanece — o grafo segue servindo.
            self._fail_n = getattr(self, "_fail_n", 0) + 1
            if self._fail_n == 1 or self._fail_n % 30 == 0:
                print(f"[graph] poll falhou ({e}) — mantendo snapshot anterior "
                      f"(falha nº {self._fail_n})")
            return {"nodes": {"added": [], "updated": [], "removed": []},
                    "edges": {"added": [], "updated": [], "removed": []}}
        self._fail_n = 0

        nodes_added, nodes_updated, nodes_removed = [], [], []
        edges_added, edges_updated, edges_removed = [], [], []

        with self.lock:
            for nid, node in db_nodes.items():
                # sig por tipo de nó — memórias, tools, planos e pessoas têm
                # campos distintos; mudança em person_id/tools_json do plano
                # vira update do nó (as arestas virtuais mudam junto)
                if node["kind"] == "tool":
                    # enabled = campo da MCP registry (t-nodes enriquecidos e
                    # m-nodes; t-nodes sem MCP DB usam o default True)
                    sig = (node["text"], node["name"], node["core"],
                           node["use_count"], node["last_used"],
                           node.get("enabled", True))
                elif node["kind"] == "plan":
                    sig = (node["name"], node["text"], node["version"],
                           node.get("person_id"), tuple(node.get("tool_ids") or []))
                elif node["kind"] == "person":
                    sig = (node["name"], node["text"], node["memory_id"],
                           node["voice_count"], node["face_count"], node["access_count"])
                elif node["kind"] == "identity":
                    # Card de identidade da Alpha (tabela `identity`): sem
                    # confidence/forgettable (NÃO cai no else das memórias —
                    # KeyError derrubava o refresh e o poller mantinha o
                    # snapshot sem o nó "a1"). memory_id/roles/updated_at
                    # cobrem o vínculo auto-curativo e a edição do card.
                    sig = (node["name"], node["text"], node.get("memory_id"),
                           node.get("roles"), node.get("updated_at"))
                else:
                    sig = (node["text"], node["source"], node["confidence"], node["forgettable"])
                old_sig = self.nodes_sig.get(nid)
                if old_sig is None:
                    self.nodes[nid] = node
                    self.nodes_sig[nid] = sig
                    nodes_added.append(node)
                elif old_sig != sig:
                    self.nodes[nid] = node
                    self.nodes_sig[nid] = sig
                    nodes_updated.append(node)
            for nid in list(self.nodes_sig):
                if nid in db_nodes:
                    continue
                # NEW (multi-escopo): escopo em falha transitória → mantém o
                # snapshot dele (remoção só quando a leitura confirmar).
                if (self.nodes.get(nid) or {}).get("scope") in failed_scopes:
                    continue
                self.nodes.pop(nid, None)
                self.nodes_sig.pop(nid, None)
                nodes_removed.append(nid)

            for key, edge in db_edges.items():
                sig = (edge["weight"], edge["count"])
                old_sig = self.edges_sig.get(key)
                if old_sig is None:
                    self.edges[key] = edge
                    self.edges_sig[key] = sig
                    edges_added.append(edge)
                elif old_sig != sig:
                    self.edges[key] = edge
                    self.edges_sig[key] = sig
                    edges_updated.append(edge)
            for key in list(self.edges_sig):
                if key in db_edges:
                    continue
                if self.edges.get(key, {}).get("scope") in failed_scopes:
                    continue   # idem: escopo em falha mantém snapshot
                self.edges.pop(key, None)
                self.edges_sig.pop(key, None)
                edges_removed.append(key)

        return {
            "nodes": {"added": nodes_added, "updated": nodes_updated, "removed": nodes_removed},
            "edges": {"added": edges_added, "updated": edges_updated, "removed": edges_removed},
        }

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "nodes": list(self.nodes.values()),
                "edges": list(self.edges.values()),
                # NEW (multi-escopo): lista de escopos p/ o cliente pré-atribuir
                # cores/setores na ordem do servidor + escopo ativo (badge).
                "scopes": [s for s, _, _ in self._discover_scopes()],
                "active_scope": self.active_scope,
                "ts": time.time(),
            }


# ─────────────────────────────────────────────────────────────────────────────
#                     BROADCAST (SSE) + TAILER + POLLER
# ─────────────────────────────────────────────────────────────────────────────

class Broadcaster:
    """Distribui eventos (dict) pra todas as filas SSE inscritas."""

    def __init__(self):
        self.subscribers: list[queue.Queue] = []
        self.lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=512)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def publish(self, event: dict):
        with self.lock:
            subs = list(self.subscribers)
        if not subs:
            return
        line = json.dumps(event, ensure_ascii=False)
        for q in subs:
            try:
                q.put_nowait(line)
            except queue.Full:
                pass  # cliente lento: descarta em vez de travar o poller


class ActivationTailer(threading.Thread):
    """Tail do activation_log.jsonl DE UM ESCOPO (eventos de PPR) →
    broadcaster. NEW (multi-escopo): existe UM TAILER POR ESCOPO (geral +
    cada db/<slug>/) — todos os escopos são seguidos SIMULTANEAMENTE (antes
    o tailer era único e fazia retarget quando o escopo ativo mudava). Cada
    evento publicado ganha "scope" (o cliente mapeia os ids p/ o espaço
    "{scope}:{id}"). Linhas antigas (sem "type", geradas antes do ctx) viram
    activation. Criado/removido pelo ScopeWatcher conforme o diretório do
    escopo aparece/desaparece (stop() encerra o thread)."""

    daemon = True

    def __init__(self, scope: str, log_path: str, broadcaster: Broadcaster):
        super().__init__()
        self.scope = scope
        self.log_path = log_path
        self.broadcaster = broadcaster
        self._stop_evt = threading.Event()

    def stop(self):
        """Para o tail (chamado pelo ScopeWatcher quando o escopo some)."""
        self._stop_evt.set()

    def run(self):
        # Espera o arquivo nascer (escopo novo ainda sem ativações) — até stop.
        pos = 0
        while not self._stop_evt.is_set():
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    f.seek(0, 2)
                    pos = f.tell()
                break
            except FileNotFoundError:
                self._stop_evt.wait(1.0)

        while not self._stop_evt.is_set():
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    f.seek(pos)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            evt = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        evt.setdefault("type", "activation")
                        evt["scope"] = self.scope   # multi-escopo: dono do log
                        self.broadcaster.publish(evt)
                    pos = f.tell()
            except FileNotFoundError:
                pass   # log removido/recriado — reabre no próximo tick
            self._stop_evt.wait(0.25)


class ScopeWatcher(threading.Thread):
    """V2 (multi-escopo): dois serviços leves.
      1. TAILERS — mantém um ActivationTailer POR escopo: db root (geral) +
         cada db/<slug>/ (projeto). Escopo novo → tailer nasce (e passa a
         publicar ativações com "scope"); diretório removido → tailer é
         parado e sai da lista.
      2. BADGE — consulta a memory API (GET /status → project.active_scope)
         e publica {"type":"scope_switch","scope"} quando muda, pra UI
         destacar o escopo ativo na legenda de escopos. NÃO retargeta paths:
         o grafo agora cobre TODOS os escopos ao mesmo tempo.
    Best-effort: memory offline → mantém o badge atual e tenta de novo."""

    daemon = True

    def __init__(self, state: GraphState, broadcaster: Broadcaster,
                 memory_api: str | None = None, interval: float = 2.0):
        super().__init__()
        self.state = state
        self.broadcaster = broadcaster
        self.memory_api = memory_api.rstrip("/") if memory_api else None
        self.interval = interval
        self.tailers: dict[str, ActivationTailer] = {}   # scope -> tailer
        self._last_badge: str | None = None

    def _fetch_active_scope(self) -> str | None:
        """GET /status na memory API → escopo ativo (best-effort). None =
        falha/offline (mantém o badge atual)."""
        req = urllib.request.Request(f"{self.memory_api}/status")
        tok = internal_token()
        if tok:
            req.add_header("Authorization", f"Bearer {tok}")
        try:
            with urllib.request.urlopen(req, timeout=3.0) as r:
                proj = json.loads(r.read().decode("utf-8")).get("project") or {}
                return proj.get("active_scope")
        except Exception:
            return None

    def _sync_tailers(self):
        """Cria tailer p/ escopo novo, para o de escopo removido."""
        logs = self.state.scope_log_paths()
        for scope, path in logs.items():
            if scope not in self.tailers:
                t = ActivationTailer(scope, path, self.broadcaster)
                self.tailers[scope] = t
                t.start()
                print(f"[graph] tailer de ativações: '{scope}' → {path}")
        for scope in list(self.tailers):
            if scope not in logs:
                self.tailers.pop(scope).stop()
                print(f"[graph] tailer removido: escopo '{scope}' desapareceu")

    def run(self):
        self._sync_tailers()   # tailers do boot (idempotente no loop)
        if self.memory_api:
            print(f"[graph] escopo ativo: seguindo {self.memory_api}/status")
            first = self._fetch_active_scope()
            if first:
                self.state.active_scope = first
                self._last_badge = first   # 1º fetch só registra — sem "troca"
        while True:
            time.sleep(self.interval)
            self._sync_tailers()
            if not self.memory_api:
                continue   # standalone: só tailers, sem badge
            scope = self._fetch_active_scope()
            if scope and scope != self._last_badge:
                self.state.active_scope = scope
                self._last_badge = scope
                self.broadcaster.publish({
                    "type": "scope_switch", "scope": scope, "ts": time.time(),
                })
                print(f"[graph] escopo ativo → '{scope}'")


class GraphPoller(threading.Thread):
    """Polling de diff no SQLite → graph_delta no broadcaster."""

    daemon = True

    def __init__(self, state: GraphState, broadcaster: Broadcaster, interval: float = 1.0):
        super().__init__()
        self.state = state
        self.broadcaster = broadcaster
        self.interval = interval

    def run(self):
        while True:
            delta = self.state.refresh()
            has_delta = any(
                delta[part][op]
                for part in ("nodes", "edges")
                for op in ("added", "updated", "removed")
            )
            if has_delta:
                self.broadcaster.publish({"type": "graph_delta", "ts": time.time(), **delta})
            time.sleep(self.interval)


# ─────────────────────────────────────────────────────────────────────────────
#                                HTTP HANDLER
# ─────────────────────────────────────────────────────────────────────────────

def make_handler(state: GraphState, broadcaster: Broadcaster):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # silencia log padrão

        def _json(self, payload: dict, status: int = 200):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                body = PAGE_TEMPLATE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/graph":
                self._json(state.snapshot())
            elif path == "/graph/replay":
                try:
                    n = int(self.path.split("n=")[1].split("&")[0])
                except (IndexError, ValueError):
                    n = 50
                n = max(1, min(n, 500))
                events: list[dict] = []
                # NEW (multi-escopo): funde TODOS os logs de escopo (geral +
                # projetos), taggeia cada evento com "scope" e ordena por
                # timestamp — o replay mostra ativações de qualquer projeto
                # em ordem cronológica real.
                for scope, log_path in state.scope_log_paths().items():
                    try:
                        with open(log_path, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    evt = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                evt.setdefault("type", "activation")
                                evt.setdefault("scope", scope)
                                events.append(evt)
                    except FileNotFoundError:
                        pass
                events.sort(key=lambda e: e.get("ts") or 0)
                self._json({"events": events[-n:], "ts": time.time()})
            elif path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                q = broadcaster.subscribe()
                try:
                    while True:
                        try:
                            line = q.get(timeout=15)
                            self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    broadcaster.unsubscribe(q)
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


# ─────────────────────────────────────────────────────────────────────────────
#                            UI (vis-network overhaul)
# ─────────────────────────────────────────────────────────────────────────────

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Grafo Hebbiano — AVA (tempo real)</title>
<script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  html, body { margin:0; padding:0; height:100%; background:#0b0d13; overflow:hidden;
               font-family: -apple-system, 'Segoe UI', sans-serif; color:#e8e8ea; }

  #network { width:100%; height:100vh; }

  /* ── HUD (topo-esquerda): contadores + log de eventos ── */
  #hud {
    position:absolute; top:12px; left:12px; z-index:10;
    background:rgba(16,18,26,0.92); padding:10px 14px; border-radius:12px;
    font-size:13px; line-height:1.5; max-width:340px;
    border:1px solid #232738; backdrop-filter: blur(6px);
  }
  #hud b { color:#fff; }
  #counts { font-size:12px; color:#aab; margin-top:2px; }
  #counts .t { padding:0 4px; border-radius:4px; margin-left:2px; }
  #log { max-height:150px; overflow-y:auto; font-size:11px; color:#9aa; margin-top:6px;
         border-top:1px solid #232738; padding-top:6px; }
  #log .q { color:#cdd3e0; }

  /* ── Status (topo-direita) ── */
  #status { position:absolute; top:14px; right:16px; z-index:10; font-size:12px; color:#8ecae6; }

  /* ── NEW (multi-escopo): legenda de escopos (topo-centro) — um chip por
     escopo de memória (geral + projetos), contagem, badge "ativo" e toggle */
  #scopeBar {
    position:absolute; top:12px; left:50%; transform:translateX(-50%); z-index:10;
    background:rgba(16,18,26,0.92); border:1px solid #232738; border-radius:12px;
    padding:7px 14px; font-size:11.5px; display:flex; gap:10px; align-items:center;
    flex-wrap:wrap; max-width:56vw;
  }
  #scopeBar .sbTitle { text-transform:uppercase; letter-spacing:.08em; color:#7a819a; font-size:10px; }
  #scopeChips { display:inline-flex; gap:6px; flex-wrap:wrap; }
  .scopeChip { display:inline-flex; align-items:center; gap:5px; cursor:pointer;
    padding:3px 10px; border-radius:10px; border:1px solid #33384a; background:#141724;
    color:#cdd3e0; user-select:none; white-space:nowrap; }
  .scopeChip:hover { background:#1c2032; }
  .scopeChip .dot { width:9px; height:9px; border-radius:50%; display:inline-block; }
  .scopeChip .cnt { color:#8a90a5; }
  .scopeChip.off { opacity:.35; text-decoration:line-through; }
  .scopeChip .act { font-size:9px; padding:0 6px; border-radius:6px;
    background:#51cf6622; color:#51cf66; text-transform:uppercase; letter-spacing:.05em; }

  /* ── Legenda (rodapé) ── */
  #legend {
    position:absolute; bottom:12px; left:50%; transform:translateX(-50%); z-index:10;
    background:rgba(16,18,26,0.92); border:1px solid #232738; border-radius:12px;
    padding:8px 16px; font-size:11.5px; display:flex; gap:14px; align-items:center;
    flex-wrap:wrap; max-width:92vw;
  }
  .chip { display:inline-block; width:22px; height:3px; border-radius:2px; margin-right:5px;
          vertical-align:middle; }
  .chip.node { width:10px; height:10px; border-radius:50%; }
  #legend span { white-space:nowrap; color:#b6bccb; }

  /* ── Controles (painel esquerdo retrátil) ── */
  #gear {
    position:absolute; top:12px; left:368px; z-index:11; cursor:pointer;
    background:rgba(16,18,26,0.92); border:1px solid #232738; border-radius:8px;
    color:#aab; padding:5px 9px; font-size:14px; user-select:none;
  }
  #gear:hover { color:#fff; }
  #controls {
    position:absolute; top:52px; left:12px; z-index:10; width:340px;
    background:rgba(16,18,26,0.94); border:1px solid #232738; border-radius:12px;
    padding:12px 14px; font-size:12.5px; display:none;
  }
  #controls.open { display:block; }
  #controls h4 { margin:8px 0 4px; font-size:11px; text-transform:uppercase;
                 letter-spacing:.08em; color:#7a819a; }
  #controls label { display:flex; align-items:center; gap:6px; margin:3px 0; cursor:pointer; }
  #controls input[type=text] { width:100%; box-sizing:border-box; background:#1a1d2a;
    border:1px solid #2a2e3d; border-radius:6px; color:#e8e8ea; padding:5px 8px; font-size:12px; }
  #controls .row { display:flex; align-items:center; gap:8px; margin:6px 0; }
  #controls input[type=range] { flex:1; }
  #controls button {
    background:#232738; color:#cdd3e0; border:1px solid #33384a; border-radius:6px;
    padding:5px 10px; cursor:pointer; font-size:12px;
  }
  #controls button:hover { background:#2d3247; }

  /* ── Painel de inspeção (direita) ── */
  #panel {
    position:absolute; top:0; right:-380px; width:360px; height:100vh; z-index:20;
    background:rgba(14,16,24,0.97); border-left:1px solid #232738;
    padding:18px 20px; box-sizing:border-box; overflow-y:auto;
    transition:right .25s ease;
  }
  #panel.open { right:0; }
  #panel h3 { margin:0 0 4px; font-size:15px; }
  #panel .meta { font-size:11.5px; color:#8a90a5; margin:2px 0; }
  #panel .text { font-size:12.5px; line-height:1.55; color:#dfe3ec; white-space:pre-wrap;
                 background:#141724; border:1px solid #232738; border-radius:8px;
                 padding:10px; margin:10px 0; }
  #panel .close { position:absolute; top:10px; right:12px; cursor:pointer; color:#7a819a;
                  font-size:18px; }
  #panel .close:hover { color:#fff; }
  #panel .nbr { display:flex; gap:6px; align-items:baseline; padding:5px 6px; margin:3px 0;
                background:#141724; border-radius:6px; cursor:pointer; font-size:12px; }
  #panel .nbr:hover { background:#1c2032; }
  #panel .nbr .t { font-size:10px; padding:1px 6px; border-radius:4px; white-space:nowrap; }
  .pill { display:inline-block; padding:1px 8px; border-radius:8px; font-size:10.5px; }
</style>
</head>
<body>

<div id="hud">
  <b>Grafo Hebbiano</b> — memórias LT + tools + planos em tempo real
  <div id="counts">carregando…</div>
  <div id="log"></div>
</div>

<div id="gear" title="filtros e controles">⚙</div>
<div id="controls">
  <h4>Filtros por tipo de aresta</h4>
  <div id="typeFilters"></div>
  <h4>Animação</h4>
  <div class="row">decay <input type="range" id="decaySlider" min="0.80" max="0.98" step="0.01" value="0.93"><span id="decayVal">0.93</span></div>
  <div class="row"><label style="margin:0"><input type="checkbox" id="autoFollow" checked> auto-follow em nós novos</label></div>
  <h4>Buscar memória ou tool</h4>
  <input type="text" id="searchBox" placeholder="texto / nome contém… (Enter)">
  <h4>Replay</h4>
  <div class="row">
    <input type="text" id="replayN" value="40" style="width:60px">
    <button id="replayBtn">▶ replay ativações</button>
  </div>
  <div class="row"><button id="resetBtn">reset view</button></div>
</div>

<div id="scopeBar"><span class="sbTitle">escopos</span><span id="scopeChips"></span></div>

<div id="status">conectando…</div>
<div id="network"></div>

<div id="legend">
  <span><i class="chip" style="background:#ffd166"></i>co-ativação</span>
  <span><i class="chip" style="background:#4dabf7"></i>temporal</span>
  <span><i class="chip" style="background:#ff922b"></i>updates →</span>
  <span><i class="chip" style="background:#ff6b6b"></i>contradicts (⌇)</span>
  <span><i class="chip node" style="background:#ff6b6b"></i>semente</span>
  <span><i class="chip node" style="background:#ffd166"></i>PPR</span>
  <span><i class="chip node" style="background:#51cf66"></i>novo</span>
  <span><i class="chip node" style="background:#38d9a9"></i>tool (⬡ MCP/nativa)</span>
  <span><i class="chip node" style="background:#b197fc"></i>plano (⬥ {escopo}:p{id})</span>
  <span><i class="chip node" style="background:#4dabf7"></i>borda do nó = escopo (geral/projeto)</span>
  <span style="opacity:.65">espessura = peso da aresta · tamanho = grau + ativação</span>
</div>

<div id="panel"><span class="close" id="panelClose">✕</span><div id="panelBody"></div></div>

<script>
// ═══════════════════════════════════════════════════════════════════════════
//  Estilo por tipo de aresta (fonte única da verdade p/ legenda e rede)
// ═══════════════════════════════════════════════════════════════════════════
const TYPE_STYLE = {
  co_activation:       { color: '#ffd166', directed: false, dashes: false, name: 'co-ativação' },
  temporal_precedence: { color: '#4dabf7', directed: true,  dashes: false, name: 'temporal' },
  updates:             { color: '#ff922b', directed: true,  dashes: false, name: 'updates' },
  contradicts:         { color: '#ff6b6b', directed: false, dashes: [6, 4], name: 'contradicts' },
  // ── NEW (people cards / correlações entre grafos): arestas virtuais
  // derivadas dos metadados — tracejadas pra diferenciar das arestas
  // "reais" do banco Hebbiano/ToolsDB.
  person_identity:     { color: '#63e6be', directed: true,  dashes: [2, 3], name: 'identidade' },
  identity_memory:     { color: '#ffd43b', directed: true,  dashes: [2, 3], name: 'identidade Alpha' },
  plan_person:         { color: '#f783ac', directed: false, dashes: [2, 3], name: 'plano↔pessoa' },
  plan_tool:           { color: '#74c0fc', directed: false, dashes: [2, 3], name: 'plano→tool' },
};
const BASE_NODE_BG = '#33384a', BASE_NODE_BORDER = '#54607a';
// Tools (nativas + MCP): hexágonos verdes — distinguem de memória à primeira vista
const TOOL_NODE_BG = '#1f3d3a', TOOL_NODE_BORDER = '#38d9a9', TOOL_FONT_COLOR = '#7fe3c8';
// Planos de execução: diamantes roxos, id "p{id}" — ligam-se por arestas
// virtuais ao card de pessoa e às tools do plano
const PLAN_NODE_BG = '#2a2340', PLAN_NODE_BORDER = '#b197fc', PLAN_FONT_COLOR = '#d0bfff';
// People cards: triângulos rosados, id "u{id}" — a aresta "identidade" liga
// o card à memória LT que representa a pessoa no grafo Hebbiano
const PERSON_NODE_BG = '#40232a', PERSON_NODE_BORDER = '#f783ac', PERSON_FONT_COLOR = '#ffdeeb';
// Identidade da Alpha: estrela dourada, id "a{id}" (singleton) — o ÚNICO nó
// que representa a própria assistente; a aresta "identidade Alpha" liga o
// card à memória LT que a representa no grafo Hebbiano
const IDENTITY_NODE_BG = '#3a3013', IDENTITY_NODE_BORDER = '#ffd43b', IDENTITY_FONT_COLOR = '#ffe8a3';
const FONT_COLOR   = '#9aa0b4';

// ═══════════════════════════════════════════════════════════════════════════
//  NEW (multi-escopo): paleta + SETORES por escopo — cada escopo de memória
//  ("geral" + db/<slug>/ do projeto) ganha COR de borda própria e um SETOR
//  espacial próprio; tools/MCP/pessoas são GLOBAIS (scope null → núcleo
//  central compartilhado, pois se ligam a todos os escopos).
// ═══════════════════════════════════════════════════════════════════════════
const SCOPE_PALETTE = ['#4dabf7', '#ffd166', '#ff922b', '#63e6be', '#f783ac',
                       '#b197fc', '#38d9a9', '#e599f7', '#a9e34b', '#74c0fc'];
const GOLDEN_ANGLE = 2.39996323;   // rad — distribui setores sem saber n total
const scopeOrder = [];             // escopos em ordem de aparição (geral primeiro)
const scopeColorMap = new Map();   // scope -> cor (atribuída por ordem de aparição)

function hashStr(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h;
}
function scopeIndex(scope) {
  if (!scopeOrder.includes(scope)) scopeOrder.push(scope);
  return scopeOrder.indexOf(scope);
}
function scopeColor(scope) {
  if (!scope) return BASE_NODE_BORDER;   // global: borda neutra original
  scopeIndex(scope);                     // garante registro na ordem
  if (!scopeColorMap.has(scope))
    scopeColorMap.set(scope, SCOPE_PALETTE[scopeColorMap.size % SCOPE_PALETTE.length]);
  return scopeColorMap.get(scope);
}
// Posição inicial (seed da física): setor do escopo, raio por tipo, jitter
// estável por id (mesmo nó sempre nasce no mesmo lugar do setor).
function scopePos(meta) {
  const j = (hashStr(String(meta.id)) % 160) - 80;
  if (!meta.scope) {   // global (tools/pessoas/MCP): núcleo central
    return { x: (hashStr('g' + String(meta.id)) % 200) - 100,
             y: (hashStr('h' + String(meta.id)) % 200) - 100 };
  }
  const ang = scopeIndex(meta.scope) * GOLDEN_ANGLE;
  const r = meta.kind === 'plan' ? 330 : 250;   // planos um anel além das memórias
  return { x: Math.cos(ang) * r + j, y: Math.sin(ang) * r + j };
}

const nodes = new vis.DataSet();
const edges = new vis.DataSet();

const nodeMeta = new Map();    // id -> node público (do servidor)
const edgeMeta = new Map();    // key -> edge pública
const edgeKeyToId = new Map(); // key -> id interno no DataSet
let edgeSeq = 0;

const activation = new Map();  // id -> 0..1
const isSeed = new Map();      // id -> ts
const bornTs = new Map();      // id -> ts de nascimento (pulso verde)
const bornEdges = new Map();   // key -> ts (pulso verde da aresta nova)

const enabledTypes = new Set(Object.keys(TYPE_STYLE));
let decayFactor = 0.93;
let autoFollow = true;

const adjByNode = new Map();   // id -> Set(edgeKey) — índice p/ acender arestas barato
function reindexAdjacency() {
  adjByNode.clear();
  for (const e of edgeMeta.values()) {
    if (!adjByNode.has(e.from)) adjByNode.set(e.from, new Set());
    if (!adjByNode.has(e.to))   adjByNode.set(e.to, new Set());
    adjByNode.get(e.from).add(e.key);
    adjByNode.get(e.to).add(e.key);
  }
}
function degreeOf(id) {
  const s = adjByNode.get(id);
  return s ? s.size : 0;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Rede vis-network
// ═══════════════════════════════════════════════════════════════════════════
const network = new vis.Network(
  document.getElementById('network'),
  { nodes, edges },
  {
    physics: {
      solver: 'forceAtlas2Based',
      forceAtlas2Based: { gravitationalConstant: -55, springLength: 95, springConstant: 0.05 },
      stabilization: { iterations: 200, fit: true },
    },
    interaction: { hover: true, tooltipDelay: 120 },
    edges: { smooth: { enabled: true, type: 'continuous', roundness: 0.35 } },
  }
);

function baseNodeOptions(meta) {
  const deg = degreeOf(meta.id);
  const isTool = meta.kind === 'tool';
  const isPlan = meta.kind === 'plan';
  const isPerson = meta.kind === 'person';
  const isIdentity = meta.kind === 'identity';
  return {
    id: meta.id, label: meta.label, title: nodeTitle(meta),
    shape: isTool ? 'hexagon' : isPlan ? 'diamond' : isPerson ? 'triangle'
                 : isIdentity ? 'star' : 'dot',
    size: (isTool ? 9 : isPlan ? 8 : isPerson ? 10 : isIdentity ? 12 : 5) + Math.min(deg, 12) * 1.3,
    // NEW (multi-escopo): borda na COR DO ESCOPO (memórias e planos);
    // tools/pessoas/identidade globais mantêm as cores próprias dos kinds.
    color: isTool ? { background: TOOL_NODE_BG, border: TOOL_NODE_BORDER }
                  : isPlan ? { background: PLAN_NODE_BG, border: scopeColor(meta.scope) }
                  : isPerson ? { background: PERSON_NODE_BG, border: PERSON_NODE_BORDER }
                  : isIdentity ? { background: IDENTITY_NODE_BG, border: IDENTITY_NODE_BORDER }
                             : { background: BASE_NODE_BG, border: scopeColor(meta.scope) },
    borderWidth: (isTool || isPerson) ? 1 : 2,   // escopo precisa ficar legível
    font: { color: isTool ? TOOL_FONT_COLOR : isPlan ? PLAN_FONT_COLOR : isPerson ? PERSON_FONT_COLOR
                  : isIdentity ? IDENTITY_FONT_COLOR : FONT_COLOR, size: 11 },
  };
}

function nodeTitle(meta) {
  if (meta.kind === 'plan') {
    const fmt = (t) => {
      const d = new Date((t || 0) * 1000);
      return isNaN(d) ? '?' : d.toLocaleDateString('pt-BR') + ' ' + d.toLocaleTimeString('pt-BR', {hour: '2-digit', minute: '2-digit'});
    };
    return `<b>${escapeHtml(meta.name || meta.label)}</b> <span style="color:${PLAN_FONT_COLOR}">v${meta.version ?? '?'}</span>` +
           ` <span style="color:${scopeColor(meta.scope)}">⬤ ${escapeHtml(meta.scope || 'global')}</span><br>` +
           `<i>${meta.title || '(sem descrição)'}</i><br>` +
           `${meta.steps_count ?? '?'} passo(s) · criado: ${fmt(meta.created_at)} · atualizado: ${fmt(meta.updated_at)}`;
  }
  if (meta.kind === 'identity') {
    const d = new Date((meta.created_at || 0) * 1000);
    const fmt = isNaN(d) ? '?' : d.toLocaleDateString('pt-BR') + ' ' + d.toLocaleTimeString('pt-BR', {hour: '2-digit', minute: '2-digit'});
    return `<b>${escapeHtml(meta.name || meta.label)}</b> <span style="color:${IDENTITY_FONT_COLOR}">identidade</span><br>` +
           `<i>${meta.title || '(sem descrição)'}</i><br>` +
           `${meta.roles ? `papéis: ${escapeHtml(meta.roles)}<br>` : ''}` +
           `criado: ${fmt} · fonte: ${escapeHtml(meta.source || '?')}`;
  }
  if (meta.kind === 'person') {
    const d = new Date((meta.created_at || 0) * 1000);
    const fmt = isNaN(d) ? '?' : d.toLocaleDateString('pt-BR') + ' ' + d.toLocaleTimeString('pt-BR', {hour: '2-digit', minute: '2-digit'});
    const extras = [meta.likes, meta.religion].filter(Boolean).map(escapeHtml).join(' · ');
    return `<b>${escapeHtml(meta.name || meta.label)}</b> <span style="color:${PERSON_FONT_COLOR}">card</span><br>` +
           `<i>${meta.title || '(sem descrição)'}</i><br>` +
           ` voz: ${meta.voice_count ?? 0} exemplo(s) · rosto: ${meta.face_count ?? 0} exemplo(s)` +
           `${extras ? `<br>${extras}` : ''}<br>` +
           `criado: ${fmt} · acessos: ${meta.access_count ?? '?'}`;
  }
  if (meta.kind === 'tool') {
    const lu = meta.last_used ? new Date(meta.last_used * 1000).toLocaleString('pt-BR') : 'nunca usada';
    const mcpInfo = meta.server
      ? `MCP${meta.transport ? ` · ${escapeHtml(meta.transport)}` : ''}${meta.enabled === false ? ' — <span style="color:#e0503c">desabilitado</span>' : ''}`
      : 'nativa';
    return `<b>${escapeHtml(meta.name || meta.label)}</b> ${meta.core ? '⭐ core' : ''}<br>` +
           `<i>${meta.title || '(sem snippet — rode /sync ou defina o skill.md no mcp_manager)'}</i><br>` +
           `${mcpInfo} · usos: ${meta.use_count ?? 0} · última: ${lu}`;
  }
  const d = new Date((meta.created_at || 0) * 1000);
  const dateStr = isNaN(d) ? '?' : d.toLocaleDateString('pt-BR') + ' ' + d.toLocaleTimeString('pt-BR', {hour: '2-digit', minute: '2-digit'});
  return `<b>#${meta.id}</b> <span style="color:${scopeColor(meta.scope)}">⬤ ${escapeHtml(meta.scope || 'global')}</span> (${(meta.source || '?')})<br>` +
         `<i>${meta.title || ''}</i><br>` +
         `confiança: ${Math.round((meta.confidence || 0) * 100)}% · ` +
         `acessos: ${meta.access_count ?? '?'} · criada: ${dateStr}`;
}

function edgeOptions(meta, isBirth) {
  const st = TYPE_STYLE[meta.type] || { color: '#888', directed: false, dashes: false };
  const w = meta.weight || 0;
  const opt = {
    id: edgeKeyToId.get(meta.key),
    from: meta.from, to: meta.to,
    color: { color: isBirth ? '#51cf66' : st.color, opacity: Math.min(0.8, 0.15 + w * 0.7) },
    width: 0.5 + w * 4,
    title: `<b>${st.name}</b> · peso ${w.toFixed(3)} · co-ativações: ${meta.count || 0}` +
           (meta.last ? `<br>última: ${new Date(meta.last * 1000).toLocaleString('pt-BR')}` : ''),
  };
  if (st.dashes) opt.dashes = st.dashes;
  if (st.directed) opt.arrows = { to: { enabled: true, scaleFactor: 0.5 } };
  return opt;
}

// ═══════════════════════════════════════════════════════════════════════════
//  NEW (multi-escopo): legenda de escopos — contagem, badge "ativo" e toggle
// ═══════════════════════════════════════════════════════════════════════════
let activeScope = 'general';      // escopo em uso na TUI (badge verde) —
                                  // memory_config.GENERAL_SCOPE = "general"
let serverScopes = [];            // lista de escopos do servidor (boot: /graph)
const hiddenScopes = new Set();   // escopos desligados (toggle do chip)

function renderScopeBar() {
  const counts = {};
  for (const m of nodeMeta.values()) {
    const s = m.scope || '';
    counts[s] = (counts[s] || 0) + 1;
  }
  // escopos nomeados: TODOS os do servidor (mesmo com 0 nós — projeto recém-
  // criado) + vistos em deltas/switches; globais (tools/pessoas/MCP) no fim.
  const named = [...serverScopes];
  for (const s of scopeOrder) if (!named.includes(s)) named.push(s);
  const extra = counts[''] ? [''] : [];
  document.getElementById('scopeChips').innerHTML =
    [...named, ...extra].map(s => {
      const c = scopeColor(s);
      const label = s || '🌐 global';
      const off = hiddenScopes.has(s) ? ' off' : '';
      const act = (s && s === activeScope) ? '<span class="act">ativo</span>' : '';
      const safe = escapeHtml(s);
      return `<span class="scopeChip${off}" data-scope="${safe}" ` +
             `title="clique: mostrar/esconder '${escapeHtml(label)}'">` +
             `<i class="dot" style="background:${c}"></i>${escapeHtml(label)} ` +
             `<span class="cnt">${counts[s] || 0}</span>${act}</span>`;
    }).join('');
}

document.getElementById('scopeChips').onclick = (e) => {
  const chip = e.target.closest('.scopeChip');
  if (!chip) return;
  const s = chip.dataset.scope;
  if (hiddenScopes.has(s)) hiddenScopes.delete(s); else hiddenScopes.add(s);
  applyScopeVisibility();
  renderScopeBar();
};

function applyScopeVisibility() {
  const nops = [];
  for (const [id, meta] of nodeMeta) {
    nops.push({ id, hidden: hiddenScopes.has(meta.scope || '') });
  }
  nodes.update(nops);
  const eops = [];
  for (const [key, meta] of edgeMeta) {
    const vid = edgeKeyToId.get(key);
    if (vid === undefined) continue;
    eops.push({ id: vid, hidden: hiddenScopes.has(meta.scope || '') });
  }
  edges.update(eops);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Aplicação de snapshot / delta
// ═══════════════════════════════════════════════════════════════════════════
function applyNodes(list, birth) {
  const now = Date.now();
  const updates = [];
  for (const meta of list) {
    nodeMeta.set(meta.id, meta);
    if (birth) bornTs.set(meta.id, now);
    if (nodes.get(meta.id)) updates.push(baseNodeOptions(meta));
    else {
      const opt = baseNodeOptions(meta);
      const pos = scopePos(meta);   // NEW (multi-escopo): nasce no setor do escopo
      opt.x = pos.x; opt.y = pos.y;
      if (birth) opt.color = { background: BASE_NODE_BG, border: '#51cf66' };
      nodes.add(opt);
    }
  }
  if (updates.length) nodes.update(updates);
  updateCounts();
}

function applyEdges(list, birth, forceBase) {
  const now = Date.now();
  const ops = [];
  for (const meta of list) {
    edgeMeta.set(meta.key, meta);
    if (!edgeKeyToId.has(meta.key)) {
      edgeKeyToId.set(meta.key, ++edgeSeq);
    }
    if (birth) bornEdges.set(meta.key, now);
    if (edges.get(edgeKeyToId.get(meta.key))) ops.push(edgeOptions(meta, birth && !forceBase));
    else ops.push(edgeOptions(meta, birth));
  }
  if (ops.length) {
    // add() e update() por id — vis aceita misturado via update? add para novos:
    for (const op of ops) {
      if (edges.get(op.id)) edges.update(op); else edges.add(op);
    }
  }
  reindexAdjacency();
  updateCounts();
}

function removeEdges(keys) {
  for (const key of keys) {
    const id = edgeKeyToId.get(key);
    if (id !== undefined) { edges.remove(id); edgeKeyToId.delete(key); }
    edgeMeta.delete(key);
  }
  reindexAdjacency();
}

function applyDelta(delta) {
  const e = delta.edges || {}, n = delta.nodes || {};
  if (n.removed && n.removed.length) {
    for (const id of n.removed) {
      nodes.remove(id); nodeMeta.delete(id);
      activation.delete(id); isSeed.delete(id); bornTs.delete(id);
    }
  }
  removeEdges(e.removed || []);
  applyNodes(n.added || [], true);
  applyNodes(n.updated || [], false);
  applyEdges(e.added || [], true);
  applyEdges(e.updated || [], false, true);

  const parts = [];
  const typesAdded = {};
  (e.added || []).forEach(x => typesAdded[x.type] = (typesAdded[x.type] || 0) + 1);
  const toolsAdded = (n.added || []).filter(x => x.kind === 'tool').length;
  if ((n.added || []).length) parts.push(`+${n.added.length} nó(s)` + (toolsAdded ? ` [${toolsAdded} tool 🔧]` : ''));
  if ((e.added || []).length) {
    const detail = Object.entries(typesAdded).map(([t, c]) => `${c}×${TYPE_STYLE[t]?.name || t}`).join(', ');
    parts.push(`+${e.added.length} aresta(s) [${detail}]`);
  }
  if ((e.updated || []).length) parts.push(`~${e.updated.length} reforçada(s)`);
  if (parts.length) pushLog('⟡ ' + parts.join(' · '));
  if (autoFollow && (n.added || []).length && (n.added || []).length <= 20) {
    network.fit({ animation: { duration: 500, easingFunction: 'easeOutQuad' } });
  }
}

function applyActivation(evt) {
  // Eventos PPR trazem "graph" ("memory" | "tools") e, NEW (multi-escopo),
  // "scope" (escopo dono do log). Ids de memória viram "{scope}:{id}" (os
  // inteiros colidem ENTRE escopos); ids de tool são globais → prefixo "t"
  // sem escopo. Linhas antigas do JSONL sem "graph"/"scope" eram do escopo
  // raiz ("general" — mesmo valor de memory_config.GENERAL_SCOPE).
  const scope = evt.scope || 'general';
  const mapId = (sid) => evt.graph === 'tools' ? 't' + sid : scope + ':' + sid;
  const now = Date.now();
  const seeds = (evt.seeds || []).map(mapId);
  for (const sid of seeds) {
    if (!nodeMeta.has(sid)) continue;   // semente de nó ainda não visto — será quando o delta chegar
    activation.set(sid, Math.max(activation.get(sid) || 0, 1.0));
    isSeed.set(sid, now);
  }
  for (const [k, v] of Object.entries(evt.rank || {})) {
    const nid = mapId(parseInt(k, 10));
    if (!nodeMeta.has(nid) || seeds.includes(nid)) continue;
    // rank PPR vem <= PPR_SPREAD_WEIGHT (0.4) — escala p/ brilho visível
    activation.set(nid, Math.max(activation.get(nid) || 0, Math.min(1, v * 2.5)));
  }
  const q = evt.query ? `"${String(evt.query).slice(0, 48)}"` : '(leitura)';
  const st = evt.strategy ? ` [${evt.strategy}]` : '';
  const g = evt.graph === 'tools' ? ' 🔧tools' : (scope !== 'general' ? ` @${scope}` : '');
  pushLog(`⚡${g} ${q}${st} → ${Object.keys(evt.rank || {}).length} nós em ${evt.elapsed_ms ?? '?'}ms`);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Loop de decay + repaint
// ═══════════════════════════════════════════════════════════════════════════
function lerp(a, b, t) { return a + (b - a) * t; }

function nodeColor(act, seedRecent, born, meta) {
  const isTool = meta && meta.kind === 'tool';
  const isPlan = meta && meta.kind === 'plan';
  const isIdentity = meta && meta.kind === 'identity';
  if (born) return isTool ? { background: TOOL_NODE_BG, border: '#51cf66' }
                          : isPlan ? { background: PLAN_NODE_BG, border: '#51cf66' }
                          : isIdentity ? { background: IDENTITY_NODE_BG, border: '#51cf66' }
                                       : { background: BASE_NODE_BG, border: '#51cf66' };
  if (seedRecent) {
    return { background: `rgb(${Math.round(lerp(51, 255, act))}, ${Math.round(lerp(56, 107, act))}, ${Math.round(lerp(74, 107, act))})`,
             border: '#ff3b3b' };
  }
  if (isTool) {
    // ativação de tool: verde-quente (teal → verde claro), borda teal acesa
    return { background: `rgb(${Math.round(lerp(31, 130, act))}, ${Math.round(lerp(61, 245, act))}, ${Math.round(lerp(58, 190, act))})`,
             border: act > 0.05 ? '#38d9a9' : TOOL_NODE_BORDER };
  }
  if (isPlan) {
    // ativação de plano (defensiva — PPR não semeia planos): lilás escuro → lilás claro
    return { background: `rgb(${Math.round(lerp(42, 224, act))}, ${Math.round(lerp(35, 219, act))}, ${Math.round(lerp(64, 255, act))})`,
             border: act > 0.05 ? '#e5dbff' : PLAN_NODE_BORDER };
  }
  if (isIdentity) {
    // ativação da identidade: dourado escuro → dourado aceso (a estrela da Alpha)
    return { background: `rgb(${Math.round(lerp(58, 255, act))}, ${Math.round(lerp(48, 212, act))}, ${Math.round(lerp(19, 59, act))})`,
             border: act > 0.05 ? '#ffd43b' : IDENTITY_NODE_BORDER };
  }
  return { background: `rgb(${Math.round(lerp(51, 255, act))}, ${Math.round(lerp(56, 209, act))}, ${Math.round(lerp(74, 102, act))})`,
           border: act > 0.05 ? '#ffd166' : BASE_NODE_BORDER };
}

function tick() {
  const now = Date.now();
  const nodeUpdates = [];
  const edgeUpdates = [];

  for (const [id, act] of Array.from(activation.entries())) {
    const newAct = act * decayFactor;
    if (newAct < 0.01) {
      activation.delete(id);
      isSeed.delete(id);
      const meta = nodeMeta.get(id);
      if (meta) nodeUpdates.push({ id, size: baseNodeOptions(meta).size, color: baseNodeOptions(meta).color });
      continue;
    }
    activation.set(id, newAct);
    const meta = nodeMeta.get(id);
    if (!meta) continue;
    const seedRecent = isSeed.has(id) && (now - isSeed.get(id)) < 1800;
    const born = bornTs.has(id) && (now - bornTs.get(id)) < 2500;
    const pulse = born ? 3 * Math.sin((now - bornTs.get(id)) / 120) : 0;
    nodeUpdates.push({
      id,
      size: baseNodeOptions(meta).size + newAct * 22 + Math.max(0, pulse),
      color: nodeColor(newAct, seedRecent, born, meta),
    });
  }
  if (nodeUpdates.length) nodes.update(nodeUpdates);

  // arestas adjacentes a nós ativos acendem na COR DO TIPO; nascidas pulsam verde
  const activeKeys = new Set();
  for (const id of activation.keys()) {
    const s = adjByNode.get(id);
    if (s) for (const k of s) activeKeys.add(k);
  }
  for (const key of activeKeys) {
    const meta = edgeMeta.get(key);
    if (!meta) continue;
    const id = edgeKeyToId.get(key);
    const a = activation.get(meta.from) || 0, b = activation.get(meta.to) || 0;
    const m = Math.min(a, b);
    if (m > 0.02) {
      const st = TYPE_STYLE[meta.type] || { color: '#888' };
      edgeUpdates.push({
        id,
        color: { color: st.color, opacity: Math.min(0.95, m + 0.25) },
        width: 0.5 + (meta.weight || 0) * 4 + m * 2.5,
      });
    }
  }
  for (const [key, ts] of Array.from(bornEdges.entries())) {
    if (now - ts > 2500) { bornEdges.delete(key); continue; }
    const meta = edgeMeta.get(key);
    const id = edgeKeyToId.get(key);
    if (meta && id !== undefined) edgeUpdates.push(edgeOptions(meta, true));
  }
  if (edgeUpdates.length) edges.update(edgeUpdates);

  requestAnimationFrame(() => setTimeout(tick, 60));
}

// ═══════════════════════════════════════════════════════════════════════════
//  HUD: contadores, log
// ═══════════════════════════════════════════════════════════════════════════
function updateCounts() {
  const byType = {};
  for (const e of edgeMeta.values()) byType[e.type] = (byType[e.type] || 0) + 1;
  const parts = Object.keys(TYPE_STYLE).map(t =>
    `<span class="t" style="background:${TYPE_STYLE[t].color}22;color:${TYPE_STYLE[t].color}">${TYPE_STYLE[t].name}: ${byType[t] || 0}</span>`
  ).join(' ');
  let tools = 0, plans = 0, people = 0, identity = 0;
  for (const m of nodeMeta.values()) {
    if (m.kind === 'tool') tools++;
    else if (m.kind === 'plan') plans++;
    else if (m.kind === 'person') people++;
    else if (m.kind === 'identity') identity++;
  }
  document.getElementById('counts').innerHTML =
    `memórias: ${nodeMeta.size - tools - plans - people - identity} · <span style="color:${TOOL_FONT_COLOR}">tools: ${tools}</span> · <span style="color:${PLAN_FONT_COLOR}">planos: ${plans}</span> · <span style="color:${PERSON_FONT_COLOR}">pessoas: ${people}</span> · <span style="color:${IDENTITY_FONT_COLOR}">identidade: ${identity}</span> · ${parts}`;
  renderScopeBar();   // NEW (multi-escopo): chips com contagem por escopo
}

function pushLog(msg) {
  const el = document.getElementById('log');
  const line = document.createElement('div');
  line.innerHTML = msg;
  el.prepend(line);
  while (el.childNodes.length > 8) el.removeChild(el.lastChild);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Painel de inspeção (click no nó)
// ═══════════════════════════════════════════════════════════════════════════
const panel = document.getElementById('panel');
network.on('click', (params) => {
  if (params.nodes.length) showPanel(params.nodes[0]);
  else if (params.edges.length) {
    const e = edges.get(params.edges[0]);
    if (e) pushLog(`aresta: ${TYPE_STYLE[edgeMeta.get(edgeKey(e))]?.name || '?'} · peso ${edgeMeta.get(edgeKey(e))?.weight.toFixed(3)}`);
  }
});
function edgeKey(visId) {
  for (const [k, id] of edgeKeyToId.entries()) if (id === visId) return k;
  return null;
}
function showPanel(id) {
  const meta = nodeMeta.get(id);
  if (!meta) return;
  const isTool = meta.kind === 'tool';
  const isPlan = meta.kind === 'plan';
  const isPerson = meta.kind === 'person';
  const isIdentity = meta.kind === 'identity';
  const d = new Date((meta.created_at || 0) * 1000);
  let head = '', body = '';
  if (isIdentity) {
    const fmt = isNaN(d) ? '?' : d.toLocaleString('pt-BR');
    const pills = meta.memory_id != null
      ? `<span class="pill" style="background:#ffd43b22;color:#ffd43b">memória #${meta.memory_id}</span>`
      : '<span class="pill" style="background:#e0503c22;color:#e0503c">sem memória de identidade</span>';
    head = `<h3 style="color:${IDENTITY_FONT_COLOR}">${escapeHtml(meta.name || meta.label)}</h3>
      <div class="meta">identidade da Alpha (★ ${escapeHtml(String(meta.id))}) · grau: ${degreeOf(id)}</div>
      <div class="meta">criado: ${fmt} · fonte: ${escapeHtml(meta.source || '?')}</div>
      <div class="meta">${pills}</div>`;
    const extras = [
      meta.description ? `<b>Quem é:</b> ${escapeHtml(meta.description)}` : '',
      meta.roles ? `<b>Papéis/valores:</b> ${escapeHtml(meta.roles)}` : '',
      meta.additional_info ? `<b>Mais:</b> ${escapeHtml(meta.additional_info)}` : '',
    ].filter(Boolean).map(s => `<div class="text">${s}</div>`).join('');
    body = extras || '<div class="meta">(card sem campos preenchidos)</div>';
  } else if (isPerson) {
    const fmt = isNaN(d) ? '?' : d.toLocaleString('pt-BR');
    const la = new Date((meta.last_accessed || 0) * 1000);
    const pills =
      `<span class="pill" style="background:#63e6be22;color:#63e6be">voz: ${meta.voice_count ?? 0}</span>` +
      `<span class="pill" style="background:#63e6be22;color:#63e6be">rosto: ${meta.face_count ?? 0}</span>` +
      (meta.memory_id != null ? `<span class="pill" style="background:#38d9a922;color:#38d9a9">identidade #${meta.memory_id}</span>` : '');
    head = `<h3 style="color:${PERSON_FONT_COLOR}">${escapeHtml(meta.name || meta.label)}</h3>
      <div class="meta">card de pessoa (▲ ${escapeHtml(String(meta.id))}) · grau: ${degreeOf(id)}</div>
      <div class="meta">criado: ${fmt} · acessos: ${meta.access_count ?? '?'} · última: ${isNaN(la) ? '?' : la.toLocaleString('pt-BR')}</div>
      <div class="meta">${pills}</div>`;
    const extras = [
      meta.description ? `<b>Quem é:</b> ${escapeHtml(meta.description)}` : '',
      meta.likes ? `<b>Gosta:</b> ${escapeHtml(meta.likes)}` : '',
      meta.religion ? `<b>Religião:</b> ${escapeHtml(meta.religion)}` : '',
      meta.additional_info ? `<b>Mais:</b> ${escapeHtml(meta.additional_info)}` : '',
    ].filter(Boolean).map(s => `<div class="text">${s}</div>`).join('');
    body = extras || '<div class="meta">(card sem campos preenchidos)</div>';
  } else if (isPlan) {
    const cu = new Date((meta.created_at || 0) * 1000);
    const uu = new Date((meta.updated_at || 0) * 1000);
    const fmt = (dd) => isNaN(dd) ? '?' : dd.toLocaleString('pt-BR');
    head = `<h3 style="color:${PLAN_FONT_COLOR}">${escapeHtml(meta.name || meta.label)}</h3>
      <div class="meta">plano de execução (⬥ ${escapeHtml(String(meta.id))}) · versão: <b>${meta.version ?? '?'}</b> · grau: ${degreeOf(id)}</div>
      <div class="meta">escopo: <span class="pill" style="background:${scopeColor(meta.scope)}22;color:${scopeColor(meta.scope)}">⬤ ${escapeHtml(meta.scope || 'global')}</span></div>
      <div class="meta">passos: ${meta.steps_count ?? '?'} · criado: ${fmt(cu)} · atualizado: ${fmt(uu)}</div>`;
    body = `<div class="text">${escapeHtml(meta.description || '(sem descrição — edite na TUI via Ctrl+S → planos)')}</div>`;
  } else if (isTool) {
    const lu = meta.last_used ? new Date(meta.last_used * 1000).toLocaleString('pt-BR') : 'nunca usada';
    const pills =
      (meta.core ? '<span class="pill" style="background:#38d9a922;color:#38d9a9">core ⭐</span>' : '') +
      (meta.always_on ? '<span class="pill" style="background:#ffd16622;color:#ffd166">always_on</span>' : '') +
      (meta.transport ? `<span class="pill" style="background:#54607a22;color:#9aa0b4">${escapeHtml(meta.transport)}</span>` : '') +
      (meta.enabled === false ? '<span class="pill" style="background:#e0503c22;color:#e0503c">desabilitado</span>' : '');
    head = `<h3 style="color:${TOOL_FONT_COLOR}">${escapeHtml(meta.name || meta.label)}</h3>
      <div class="meta">tool ${meta.server ? `MCP — servidor: <b>${escapeHtml(meta.server)}</b>` : 'nativa'} · grau: ${degreeOf(id)}</div>
      <div class="meta">usos: ${meta.use_count ?? 0} · última: ${lu}${pills ? ' · ' + pills : ''}</div>
      <div class="meta">registrada: ${isNaN(d) ? '?' : d.toLocaleString('pt-BR')}</div>
      <div class="text">${escapeHtml(meta.description || '(sem snippet — defina o skill.md via PUT /tools/{graph}/snippet no mcp_manager)')}</div>`;
  } else {
    const conf = Math.round((meta.confidence || 0) * 100);
    const confColor = conf >= 80 ? '#51cf66' : conf >= 55 ? '#ffd166' : '#ff6b6b';
    head = `<h3>#${escapeHtml(String(meta.id))} — memória LT</h3>
      <div class="meta">escopo: <span class="pill" style="background:${scopeColor(meta.scope)}22;color:${scopeColor(meta.scope)}">⬤ ${escapeHtml(meta.scope || 'global')}</span></div>
      <div class="meta">fonte: ${meta.source || '?'} · grau: ${degreeOf(id)}</div>
      <div class="meta">criada: ${isNaN(d) ? '?' : d.toLocaleString('pt-BR')} · acessos: ${meta.access_count ?? '?'}</div>
      <div class="meta">confiança: <span style="color:${confColor}"><b>${conf}%</b></span>
        ${meta.forgettable ? '<span class="pill" style="background:#ffd16622;color:#ffd166">esquecível</span>'
                            : '<span class="pill" style="background:#51cf6622;color:#51cf66">permanente</span>'}</div>`;
    body = `<div class="text">${escapeHtml(meta.text || '')}</div>`;
  }
  let nbrs = [];
  for (const key of (adjByNode.get(id) || [])) {
    const e = edgeMeta.get(key);
    if (!e) continue;
    const otherId = e.from === id ? e.to : e.from;
    nbrs.push({ other: nodeMeta.get(otherId), edge: e });
  }
  nbrs.sort((x, y) => (y.edge.weight || 0) - (x.edge.weight || 0));
  const nbrsHtml = nbrs.map(({ other, edge }) => {
    const st = TYPE_STYLE[edge.type] || { color: '#888', name: edge.type };
    // JSON.stringify preserva o TIPO do id: memória = número, tool = "tN"
    return `<div class="nbr" onclick="focusNode(${JSON.stringify(other?.id ?? null)})">
      <span class="t" style="background:${st.color}22;color:${st.color}">${st.name}</span>
      <span>${other ? escapeHtml(other.label) : '#?'} · w=${(edge.weight || 0).toFixed(2)}</span>
    </div>`;
  }).join('') || '<div class="meta">(sem vizinhos — nó isolado no grafo)</div>';

  document.getElementById('panelBody').innerHTML = `
    ${head}${body}
    <h4 style="margin:10px 0 4px;font-size:11px;color:#7a819a;text-transform:uppercase;letter-spacing:.08em">Vizinhos (${nbrs.length})</h4>
    ${nbrsHtml}`;
  panel.classList.add('open');
}
document.getElementById('panelClose').onclick = () => panel.classList.remove('open');
function focusNode(id) {
  if (!nodeMeta.has(id)) return;
  network.selectNodes([id]);
  network.focus(id, { scale: 1.1, animation: { duration: 400 } });
  showPanel(id);
}
function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ═══════════════════════════════════════════════════════════════════════════
//  Controles: filtros, decay, busca, replay, reset
// ═══════════════════════════════════════════════════════════════════════════
document.getElementById('gear').onclick = () => document.getElementById('controls').classList.toggle('open');

const typeFilters = document.getElementById('typeFilters');
for (const [t, st] of Object.entries(TYPE_STYLE)) {
  const label = document.createElement('label');
  label.innerHTML = `<input type="checkbox" checked data-type="${t}">
    <span class="chip" style="background:${st.color}"></span> ${st.name}${st.directed ? ' →' : ''}`;
  typeFilters.appendChild(label);
}
typeFilters.onchange = () => {
  for (const cb of typeFilters.querySelectorAll('input')) {
    cb.checked ? enabledTypes.add(cb.dataset.type) : enabledTypes.delete(cb.dataset.type);
  }
  // reconstrói o DataSet de arestas respeitando os filtros
  const ops = [];
  for (const [key, meta] of edgeMeta.entries()) {
    const id = edgeKeyToId.get(key);
    const visible = enabledTypes.has(meta.type);
    if (!visible && id !== undefined) { edges.remove(id); continue; }
    if (visible) {
      edgeKeyToId.set(key, id ?? ++edgeSeq);
      const opt = edgeOptions(meta, false);
      edges.get(opt.id) ? edges.update(opt) : edges.add(opt);
    }
  }
};

document.getElementById('decaySlider').oninput = (e) => {
  decayFactor = parseFloat(e.target.value);
  document.getElementById('decayVal').textContent = decayFactor.toFixed(2);
};
document.getElementById('autoFollow').onchange = (e) => { autoFollow = e.target.checked; };

let searchHighlight = [];
document.getElementById('searchBox').onkeydown = (e) => {
  if (e.key !== 'Enter') return;
  // limpa highlight anterior (restaura cores/shape base de memória OU tool)
  for (const id of searchHighlight) {
    const meta = nodeMeta.get(id);
    if (meta) {
      const base = baseNodeOptions(meta);
      nodes.update({ id, color: base.color, shape: base.shape, borderWidth: 1, font: base.font });
    }
  }
  searchHighlight = [];
  const q = e.target.value.trim().toLowerCase();
  if (!q) return;
  for (const [id, meta] of nodeMeta.entries()) {
    if ((meta.text || '').toLowerCase().includes(q)) {
      searchHighlight.push(id);
      nodes.update({ id, color: { background: meta.kind === 'tool' ? TOOL_NODE_BG : meta.kind === 'plan' ? PLAN_NODE_BG : meta.kind === 'person' ? PERSON_NODE_BG : meta.kind === 'identity' ? IDENTITY_NODE_BG : BASE_NODE_BG, border: '#00e5ff' }, borderWidth: 2.5 });
    }
  }
  pushLog(`🔍 busca "${q}": ${searchHighlight.length} nó(s)`);
  if (searchHighlight.length) {
    network.selectNodes(searchHighlight);
    network.fit({ nodes: searchHighlight, animation: { duration: 500 } });
  }
};

document.getElementById('resetBtn').onclick = () => network.fit({ animation: { duration: 600 } });

document.getElementById('replayBtn').onclick = async () => {
  const n = parseInt(document.getElementById('replayN').value, 10) || 40;
  try {
    const r = await fetch(`/graph/replay?n=${n}`);
    const data = await r.json();
    activation.clear(); isSeed.clear();
    pushLog(`⏪ replay: ${data.events.length} evento(s) de ativação`);
    let i = 0;
    const timer = setInterval(() => {
      if (i >= data.events.length) { clearInterval(timer); return; }
      applyActivation(data.events[i++]);
    }, 350);
  } catch (err) {
    pushLog(`⏪ replay falhou: ${err}`);
  }
};

// ═══════════════════════════════════════════════════════════════════════════
//  Boot: snapshot + SSE
// ═══════════════════════════════════════════════════════════════════════════
const statusEl = document.getElementById('status');
const es = new EventSource('/events');
es.onopen  = () => { statusEl.textContent = 'conectado';      statusEl.style.color = '#7CFC00'; };
es.onerror = () => { statusEl.textContent = 'reconectando…';  statusEl.style.color = '#ffb703'; };
es.onmessage = (ev) => {
  if (!ev.data) return;
  let evt;
  try { evt = JSON.parse(ev.data); } catch { return; }
  if (evt.type === 'graph_delta') applyDelta(evt);
  else if (evt.type === 'scope_switch') {
    // NEW (multi-escopo): badge "ativo" anda de escopo (POST /project/switch
    // na TUI) — o grafo continua mostrando TODOS os escopos.
    activeScope = evt.scope || 'general';
    if (evt.scope && !serverScopes.includes(evt.scope)) serverScopes.push(evt.scope);
    renderScopeBar();
    pushLog(`◆ escopo ativo → '<b>${escapeHtml(evt.scope ?? '?')}</b>' (projeto em uso na TUI)`);
  }
  else applyActivation(evt);          // "activation" ou linhas antigas sem type
};

(async () => {
  try {
    const g = await (await fetch('/graph')).json();
    (g.scopes || []).forEach(s => scopeColor(s));   // cores/setores na ordem do servidor
    serverScopes = g.scopes || [];
    activeScope = g.active_scope || 'general';
    applyNodes(g.nodes || [], false);
    applyEdges(g.edges || [], false);
    pushLog(`snapshot: ${nodeMeta.size} nós, ${edgeMeta.size} arestas · ${scopeOrder.length} escopo(s)`);
    tick();
  } catch (err) {
    pushLog(`falha no snapshot inicial: ${err}`);
    tick();
  }
})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="Visualização em tempo real do grafo Hebbiano (memórias + tools + planos + people cards)")
    ap.add_argument("--db", default="./Modules/memory/db/ava_memory.db")
    ap.add_argument("--tools-db", default="./Modules/memory/db/ava_tools.db",
                    help="grafo de tools (ToolsDB) — omitido com '' ; se o arquivo não existir, só memórias")
    ap.add_argument("--plans-db", default="./Modules/memory/db/ava_plans.db",
                    help="grafo de planos (PlansDB) — omitido com '' ; se o arquivo não existir, ignora")
    ap.add_argument("--people-db", default="./Modules/memory/db/ava_people.db",
                    help="grafo de people cards (PeopleDB) — omitido com '' ; se o arquivo não existir, ignora. "
                         "Correlações virtuais: card ↔ memória de identidade, plano ↔ card, plano ↔ tools")
    ap.add_argument("--mcp-db", default="./Modules/mcp_manager/db/ava_mcp_servers.db",
                    help="registry do mcp_manager (Rust, HOST) — opcional; fusão por "
                         "graph_name: conteúdo/snippet/enabled vêm daqui, arestas da ToolsDB")
    ap.add_argument("--memory-api", nargs="?", const="__default__", default=None,
                    help="NEW (multi-escopo): URL da memory API — lê o ESCOPO ATIVO "
                         "(GET /status → project.active_scope) e publica o badge "
                         "scope_switch; o grafo cobre TODOS os escopos ao mesmo tempo. "
                         "Flag sem valor → http://<bind>:3000; flag ausente → "
                         "standalone (tailers por escopo, sem badge)")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--poll", type=float, default=1.0, help="intervalo de polling do diff (s)")
    args = ap.parse_args()

    memory_api_url = args.memory_api
    if memory_api_url == "__default__":
        memory_api_url = f"http://{internal_bind_host()}:3000"

    state = GraphState(args.db, args.tools_db, args.plans_db, args.mcp_db,
                       args.people_db)

    # Snapshot inicial ANTES de servir — o cliente já nasce com o grafo atual.
    delta = state.refresh()
    snap = state.snapshot()
    if delta["edges"]["added"]:
        extra = f"+{len(delta['edges']['added'])} arestas desde o último poll"
    else:
        extra = "nada novo"
    # NEW (multi-escopo): contagens POR ESCOPO no boot (geral + projetos).
    by_scope: dict[str, int] = {}
    for n in snap["nodes"]:
        s = n.get("scope") or "(global)"
        by_scope[s] = by_scope.get(s, 0) + 1
    parts = [f"{s}: {c}" for s, c in by_scope.items()]
    print(f"Grafo carregado: {len(snap['nodes'])} nós ({', '.join(parts) or 'vazio'}), "
          f"{len(snap['edges'])} arestas ({extra})")

    broadcaster = Broadcaster()

    # NEW (multi-escopo): o ScopeWatcher mantém UM tailer POR escopo (geral +
    # projetos — todos simultâneos) e publica o badge do escopo ativo lido da
    # memory API. Standalone (sem --memory-api): só tailers, sem badge.
    watcher = ScopeWatcher(state, broadcaster, memory_api=memory_api_url)
    watcher.start()

    poller = GraphPoller(state, broadcaster, interval=args.poll)
    poller.start()

    handler = make_handler(state, broadcaster)
    server = ThreadingHTTPServer(("127.9.12.22", args.port), handler)
    print(f"Abra http://127.9.12.22:{args.port} no navegador.")
    print(f"Polling de diff: {args.poll}s · escopos: {', '.join(s for s, _, _ in state._discover_scopes())}"
          + (f" · badge do escopo ativo: {memory_api_url}" if memory_api_url else " · standalone"))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
