# Alpha AI (AVA) — Visualização em tempo real do grafo Hebbiano (memórias LT + tools)
# Copyright (C) 2026  Felipe lumazini
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
      o CONTEÚDO passa a vir da MCP DB (descrição real do MCP, core,
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

  Modo de execução (headless — NÃO há mais servidor HTTP nem página):
    O Manager (Rust) sobe este script via `docker exec` e lê stdout como
    JSON por linha. A primeira linha é sempre:
      {"type": "snapshot", "nodes": [...], "edges": [...],
       "scopes": [...], "active_scope": "..."}
    Depois vêm os mesmos eventos que antes iam por SSE:
      {"type": "graph_delta", ...}
      {"type": "activation", ...}
      {"type": "scope_switch", ...}

  A renderização (vis-network) vive no WebUI do Manager, não neste arquivo.

Rodar:
    python graph_activation.py --stdio --memory-api   (modo embutido)
    python graph_activation.py --stdio --poll 1.0
    python graph_activation.py --stdio --tools-db ./Modules/memory/db/ava_tools.db
"""

import argparse
import html
import json
import os
import queue
import sqlite3
import threading
import sys
import time
import urllib.request
from pathlib import Path

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
    co-uso só existe na ToolsDB). A descrição real do MCP (description_raw)
    é o texto — não há mais snippet/skill.md."""
    desc = (trow["description_raw"] or "").strip()
    desc = desc or f"Tool '{trow['name']}' do servidor MCP '{trow['server']}'."
    return {
        "id":          f"m{graph_name}",
        "kind":        "tool",
        "label":       graph_name[:40] + ("…" if len(graph_name) > 40 else ""),
        "title":       html.escape(desc[:400]),
        "text":        desc,                 # p/ busca (mesma chave das demais)
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
        as tools da ToolsDB: conteúdo (descrição/core/enabled/transporte)
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
        #    a vir daqui (descrição real do MCP, core, enabled/transporte).
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
                            "SELECT server, name, description_raw, "
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
                    desc = (trow["description_raw"] or "").strip()
                    if desc:
                        node["text"] = desc
                        node["title"] = html.escape(desc[:400])
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
#                                WORKER
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None):
    """Worker headless do grafo: imprime snapshot e eventos como JSON por linha.

    Não há mais servidor HTTP nem página standalone. O Manager (Rust) sobe
    este script via `docker exec` e retransmite os JSONs ao WebUI embutido.
    """
    ap = argparse.ArgumentParser(description="Worker headless do grafo Hebbiano (JSON por linha)")
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
                         "graph_name: conteúdo/descrição/enabled vêm daqui, arestas da ToolsDB")
    ap.add_argument("--memory-api", nargs="?", const="__default__", default=None,
                    help="URL da memory API — lê o ESCOPO ATIVO (GET /status → project.active_scope) "
                         "e publica o badge scope_switch. Flag sem valor → http://<bind>:3000; "
                         "flag ausente → worker sem badge de escopo ativo")
    ap.add_argument("--poll", type=float, default=1.0, help="intervalo de polling do diff (s)")
    ap.add_argument("--stdio", action="store_true",
                    help="emite JSON por linha em stdout (modo embutido no Manager)")
    ap.add_argument("--watch-stdin", action="store_true",
                    help="encerra quando o stdin fechar (evita worker órfão do Manager)")
    args = ap.parse_args(argv)

    if args.watch_stdin:
        def _watch_stdin() -> None:
            try:
                while sys.stdin.buffer.read(4096):
                    pass
            except Exception:
                pass
            os._exit(0)

        threading.Thread(target=_watch_stdin, daemon=True, name="graph-stdin-watch").start()

    memory_api_url = args.memory_api
    if memory_api_url == "__default__":
        memory_api_url = f"http://{internal_bind_host()}:3000"

    state = GraphState(args.db, args.tools_db, args.plans_db, args.mcp_db,
                       args.people_db)

    # Snapshot inicial ANTES de emitir eventos — o cliente já nasce completo.
    state.refresh()
    snap = state.snapshot()
    by_scope: dict[str, int] = {}
    for n in snap["nodes"]:
        s = n.get("scope") or "(global)"
        by_scope[s] = by_scope.get(s, 0) + 1
    parts = [f"{s}: {c}" for s, c in by_scope.items()]
    print(f"Grafo carregado: {len(snap['nodes'])} nós ({', '.join(parts) or 'vazio'}), "
          f"{len(snap['edges'])} arestas", file=sys.stderr, flush=True)

    broadcaster = Broadcaster()
    # Assina ANTES de iniciar os produtores — sem perder eventos do boot.
    queue_events = broadcaster.subscribe()
    watcher = ScopeWatcher(state, broadcaster, memory_api=memory_api_url)
    poller = GraphPoller(state, broadcaster, interval=args.poll)

    # O primeiro JSON é sempre o snapshot completo.
    print(json.dumps({"type": "snapshot", **snap}, ensure_ascii=False), flush=True)

    watcher.start()
    poller.start()

    try:
        while True:
            try:
                evt = queue_events.get(timeout=5)
            except queue.Empty:
                continue
            print(json.dumps(evt, ensure_ascii=False), flush=True)
    except (BrokenPipeError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
