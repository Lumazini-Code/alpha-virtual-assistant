from __future__ import annotations

"""
memory_graph.py — Grafo Hebbiano: storage (SQLite), PPR e expansão.

EdgeGraphMixin (tabela de arestas tipadas, reutilizado por MemoryDB e
ToolsDB), Personalized PageRank vetorizado (scipy) e os helpers puros de
travessia/expansão do subgrafo (_build_adjacency, _expand_graph_sync) e o
log de ativação JSONL (live_activation_server.py).

Nenhuma função aqui referencia `state` — é tudo lógica local do grafo.
"""

import heapq
import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

from memory_config import (
    ACTIVATION_LOG_MAX_IDS,
    ACTIVATION_LOG_PATH,
    EDGE_CO_ACTIVATION,
    EDGE_DIRECTED_TYPES,
    EDGE_EXPLICIT_LINK_WEIGHT,
    EDGE_EXPLICIT_TYPES,
    EDGE_PPR_FACTORS,
    EDGE_TYPES,
    EDGE_TYPES_NO_DECAY,
    PPR_MIN_NEIGHBORS_PER_HOP,
    log,
)

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


def _log_activation_sync(
    seed_ids: list[int], rank: dict[int, float], elapsed_ms: float,
    ctx: Optional[dict] = None,
):
    """Grava um evento de ativação PPR no log JSONL, consumido por
    graph_activation.py p/ visualização em tempo real. Best-effort:
    qualquer falha de IO aqui NUNCA deve derrubar uma leitura de memória.

    `ctx` (opcional) enriquece o evento com o CONTEXTO da leitura que o
    provocou (query, strategy, session_id) — a UI mostra POR QUÊ o grafo
    ativou, não só que ativou."""
    try:
        top = sorted(rank.items(), key=lambda kv: kv[1], reverse=True)[:ACTIVATION_LOG_MAX_IDS]
        line = json.dumps({
            "type": "activation",
            "ts": time.time(),
            "seeds": seed_ids,
            "rank": {str(k): round(v, 4) for k, v in top},
            "elapsed_ms": round(elapsed_ms, 2),
            "query": (ctx or {}).get("query"),
            "strategy": (ctx or {}).get("strategy"),
            "session_id": (ctx or {}).get("session_id"),
        }, ensure_ascii=False)
        Path(ACTIVATION_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(ACTIVATION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        log.debug("activation log: falha ao gravar (ignorado)", exc_info=True)


__all__ = [
    'EdgeGraphMixin',
    '_build_adjacency',
    '_edge_key',
    '_edge_traversal_weights',
    '_expand_graph_sync',
    '_log_activation_sync',
    'personalized_pagerank',
]
