"""
AVA KG-RAG — Knowledge Graph com persistência SQLite
Usa NetworkX em memória + SQLite para persistência entre sessões.
Padrão do AVA: SQLite com isolation_level=None (autocommit).
"""

import sqlite3
import networkx as nx
import json
import logging
from pathlib import Path
from typing import List, Tuple, Set, Optional

from config import GRAPH_DB_PATH

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Grafo de conhecimento dirigido (DiGraph) com persistência SQLite.
    Cada nó = entidade textual.
    Cada aresta = relação semântica (sujeito → objeto via predicado).
    """

    def __init__(self, db_path: str = str(GRAPH_DB_PATH)):
        self._db_path = db_path
        self._graph   = nx.DiGraph()
        self._conn    = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.isolation_level = None  # Autocommit — padrão AVA
        self._init_schema()
        self._load_from_db()
        logger.info(
            "KnowledgeGraph carregado — %d nós, %d arestas",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    # ─── Schema ──────────────────────────────────────────────────────────────

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS kg_nodes (
                id      TEXT PRIMARY KEY,
                label   TEXT NOT NULL,
                meta    TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS kg_edges (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source      TEXT NOT NULL,
                target      TEXT NOT NULL,
                relation    TEXT NOT NULL,
                source_doc  TEXT,
                FOREIGN KEY (source) REFERENCES kg_nodes(id),
                FOREIGN KEY (target) REFERENCES kg_nodes(id)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_source ON kg_edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON kg_edges(target);
        """)

    # ─── Load/Save ───────────────────────────────────────────────────────────

    def _load_from_db(self):
        """Reconstrói o grafo NetworkX a partir do SQLite."""
        cur = self._conn.cursor()
        for row in cur.execute("SELECT id, label, meta FROM kg_nodes"):
            node_id, label, meta_json = row
            self._graph.add_node(node_id, label=label, **json.loads(meta_json or "{}"))

        for row in cur.execute("SELECT source, target, relation, source_doc FROM kg_edges"):
            src, tgt, rel, doc = row
            self._graph.add_edge(src, tgt, relationship=rel, source_doc=doc or "")

    def _node_id(self, label: str) -> str:
        """ID canônico de nó: lowercase sem espaços extras."""
        return label.strip().lower().replace(" ", "_")[:128]

    # ─── API pública ─────────────────────────────────────────────────────────

    def add_triples(
        self,
        triples: List[Tuple[str, str, str]],
        source_doc: str = "",
    ) -> int:
        """
        Insere lista de triplas (sujeito, relação, objeto).
        Retorna número de novas arestas inseridas.
        """
        inserted = 0
        for subj, rel, obj in triples:
            subj_id = self._node_id(subj)
            obj_id  = self._node_id(obj)

            # Upsert nós
            for nid, nlabel in [(subj_id, subj.strip()), (obj_id, obj.strip())]:
                if nid not in self._graph:
                    self._graph.add_node(nid, label=nlabel)
                    self._conn.execute(
                        "INSERT OR IGNORE INTO kg_nodes (id, label) VALUES (?, ?)",
                        (nid, nlabel),
                    )

            # Evita arestas duplicadas exatas (mesmo source+target+relation)
            existing = [
                d for d in self._graph.get_edge_data(subj_id, obj_id, default={}).values()
                if isinstance(d, dict) and d.get("relationship") == rel
            ] if self._graph.has_edge(subj_id, obj_id) else []

            if not existing:
                self._graph.add_edge(subj_id, obj_id, relationship=rel, source_doc=source_doc)
                self._conn.execute(
                    "INSERT INTO kg_edges (source, target, relation, source_doc) VALUES (?,?,?,?)",
                    (subj_id, obj_id, rel.strip(), source_doc),
                )
                inserted += 1

        return inserted

    def get_neighborhood(
        self,
        entity_labels: List[str],
        hops: int = 1,
    ) -> List[str]:
        """
        Retorna descrições textuais das relações vizinhas (N hops).
        Usado para enriquecer o contexto de retrieval com fatos estruturais.
        """
        results: List[str] = []
        visited_nodes: Set[str] = set()

        seeds = [self._node_id(lbl) for lbl in entity_labels]
        frontier = {nid for nid in seeds if nid in self._graph}

        for _ in range(hops):
            next_frontier: Set[str] = set()
            for node in frontier:
                if node in visited_nodes:
                    continue
                visited_nodes.add(node)
                label = self._graph.nodes[node].get("label", node)

                # Arestas de saída
                for _, tgt, data in self._graph.out_edges(node, data=True):
                    tgt_label = self._graph.nodes[tgt].get("label", tgt)
                    results.append(f"{label} → [{data.get('relationship','?')}] → {tgt_label}")
                    next_frontier.add(tgt)

                # Arestas de entrada (contexto bidirecional)
                for src, _, data in self._graph.in_edges(node, data=True):
                    src_label = self._graph.nodes[src].get("label", src)
                    results.append(f"{src_label} → [{data.get('relationship','?')}] → {label}")
                    next_frontier.add(src)

            frontier = next_frontier - visited_nodes

        return results

    def search_nodes(self, query_text: str, limit: int = 10) -> List[str]:
        """
        Busca nós por substring no label.
        Usado para encontrar entidades relacionadas à query do usuário.
        """
        q = query_text.lower()
        matches = [
            data.get("label", nid)
            for nid, data in self._graph.nodes(data=True)
            if q in data.get("label", "").lower()
        ]
        return matches[:limit]

    @property
    def stats(self) -> dict:
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
        }

    def close(self):
        self._conn.close()
