"""graph_activation.py — Visualização em tempo real do grafo Hebbiano de memórias LT.

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
  ══════════════════════════════════════════════════════════════════════════

  Eventos SSE (o cliente roteia por "type"):
    {"type": "activation",  "ts", "seeds", "rank", "elapsed_ms",
                            "query", "strategy", "session_id"}   ← PPR numa leitura
    {"type": "graph_delta", "ts",
                            "nodes": {"added": [...], "updated": [...], "removed": [id,...]},
                            "edges": {"added": [...], "updated": [...],
                                      "removed": ["a|b|type", ...]}}

  Endpoints:
    GET /                  → UI (vis-network, overhaul: cores/setas/weights por
                             tipo de aresta, painel de inspeção de nó, filtros,
                             busca, decay slider, replay, legenda)
    GET /graph             → snapshot atual enriquecido (boot do cliente)
    GET /events            → SSE (activation + graph_delta, keepalive 15s)
    GET /graph/replay?n=N  → últimos N eventos de ativação do JSONL (animação)

Rodar:
    python graph_activation.py                      (defaults abaixo)
    python graph_activation.py --port 8777 --poll 1.0
"""

import argparse
import html
import json
import queue
import sqlite3
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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


def _node_public(row: sqlite3.Row) -> dict:
    text = (row["text"] or "").strip().replace("\n", " ")
    return {
        "id":          row["id"],
        "label":       text[:40] + ("…" if len(text) > 40 else ""),
        "title":       html.escape(text[:400]),
        "text":        row["text"],
        "source":      row["source"],
        "confidence":  round(row["confidence"], 4),
        "created_at":  row["created_at"],
        "access_count": row["access_count"],
        "forgettable": bool(row["forgettable"]),
    }


def _edge_public(row: sqlite3.Row) -> dict:
    return {
        "key":   f"{row['memory_id_a']}|{row['memory_id_b']}|{row['edge_type']}",
        "from":  row["memory_id_a"],
        "to":    row["memory_id_b"],
        "type":  row["edge_type"],
        "weight": round(row["weight"], 4),
        "count": row["coactivation_count"],
        "last":  row["last_coactivated"],
    }


class GraphState:
    """Snapshot do grafo + diff por poll. Scan completo das duas tabelas a
    cada tick — com centenas/milhares de linhas custa poucos ms, e diff
    completo cobre criação, atualização E remoção (decay job) sem depender
    de surrogate keys."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.nodes: dict[int, dict] = {}                       # id -> node público
        self.nodes_sig: dict[int, tuple] = {}                  # id -> (text, source, confidence, forgettable)
        self.edges: dict[str, dict] = {}                       # key -> edge pública
        self.edges_sig: dict[str, tuple] = {}                  # key -> (weight, count)

    def _scan(self) -> tuple[dict, dict]:
        """Scan completo (nodes, edges) — tudo público/serializável."""
        conn = _open_ro(self.db_path)
        try:
            nodes = {
                r["id"]: _node_public(r)
                for r in conn.execute(
                    "SELECT id, text, source, confidence, created_at, "
                    "access_count, forgettable FROM memories"
                )
            }
            edges = {
                _edge_public(r)["key"]: _edge_public(r)
                for r in conn.execute(
                    "SELECT memory_id_a, memory_id_b, edge_type, weight, "
                    "coactivation_count, last_coactivated FROM memory_edges"
                )
            }
        finally:
            conn.close()
        return nodes, edges

    def refresh(self) -> dict:
        """Um poll: escaneia, dif contra o snapshot e devolve o delta.
        Retorna {"nodes": {"added","updated","removed"},
                 "edges": {"added","updated","removed"}} — sempre presente,
        com listas vazias quando nada mudou."""
        try:
            db_nodes, db_edges = self._scan()
        except sqlite3.Error as e:
            print(f"[graph] poll falhou ({e}) — mantendo snapshot anterior")
            return {"nodes": {"added": [], "updated": [], "removed": []},
                    "edges": {"added": [], "updated": [], "removed": []}}

        nodes_added, nodes_updated, nodes_removed = [], [], []
        edges_added, edges_updated, edges_removed = [], [], []

        with self.lock:
            for nid, node in db_nodes.items():
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
                if nid not in db_nodes:
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
                if key not in db_edges:
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
    """Tail do activation_log.jsonl (eventos de PPR) → broadcaster.
    Linhas antigas (sem "type", geradas antes do ctx) viram activation."""

    daemon = True

    def __init__(self, log_path: str, broadcaster: Broadcaster):
        super().__init__()
        self.log_path = log_path
        self.broadcaster = broadcaster

    def run(self):
        pos = 0
        while True:
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    f.seek(0, 2)
                    pos = f.tell()
                break
            except FileNotFoundError:
                time.sleep(1.0)

        while True:
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
                        self.broadcaster.publish(evt)
                    pos = f.tell()
            except FileNotFoundError:
                pass
            time.sleep(0.25)


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

def make_handler(state: GraphState, broadcaster: Broadcaster, activation_log: str):
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
                try:
                    with open(activation_log, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                evt = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            evt.setdefault("type", "activation")
                            events.append(evt)
                except FileNotFoundError:
                    pass
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
  <b>Grafo Hebbiano</b> — memórias LT em tempo real
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
  <h4>Buscar memória</h4>
  <input type="text" id="searchBox" placeholder="texto contém… (Enter)">
  <h4>Replay</h4>
  <div class="row">
    <input type="text" id="replayN" value="40" style="width:60px">
    <button id="replayBtn">▶ replay ativações</button>
  </div>
  <div class="row"><button id="resetBtn">reset view</button></div>
</div>

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
};
const BASE_NODE_BG = '#33384a', BASE_NODE_BORDER = '#54607a';
const FONT_COLOR   = '#9aa0b4';

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
  return {
    id: meta.id, label: meta.label, title: nodeTitle(meta),
    shape: 'dot',
    size: 5 + Math.min(deg, 12) * 1.3,
    color: { background: BASE_NODE_BG, border: BASE_NODE_BORDER },
    font: { color: FONT_COLOR, size: 11 },
  };
}

function nodeTitle(meta) {
  const d = new Date((meta.created_at || 0) * 1000);
  const dateStr = isNaN(d) ? '?' : d.toLocaleDateString('pt-BR') + ' ' + d.toLocaleTimeString('pt-BR', {hour: '2-digit', minute: '2-digit'});
  return `<b>#${meta.id}</b> (${TYPE_STYLE ? '' : ''}${(meta.source || '?')})<br>` +
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
  if ((n.added || []).length) parts.push(`+${n.added.length} nó(s)`);
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
  const now = Date.now();
  const seeds = evt.seeds || [];
  for (const sid of seeds) {
    if (!nodeMeta.has(sid)) continue;   // semente de nó ainda não visto — será quando o delta chegar
    activation.set(sid, Math.max(activation.get(sid) || 0, 1.0));
    isSeed.set(sid, now);
  }
  for (const [k, v] of Object.entries(evt.rank || {})) {
    const id = parseInt(k, 10);
    if (!nodeMeta.has(id) || seeds.includes(id)) continue;
    // rank PPR vem <= PPR_SPREAD_WEIGHT (0.4) — escala p/ brilho visível
    activation.set(id, Math.max(activation.get(id) || 0, Math.min(1, v * 2.5)));
  }
  const q = evt.query ? `"${String(evt.query).slice(0, 48)}"` : '(leitura)';
  const st = evt.strategy ? ` [${evt.strategy}]` : '';
  pushLog(`⚡ ${q}${st} → ${Object.keys(evt.rank || {}).length} nós em ${evt.elapsed_ms ?? '?'}ms`);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Loop de decay + repaint
// ═══════════════════════════════════════════════════════════════════════════
function lerp(a, b, t) { return a + (b - a) * t; }

function nodeColor(act, seedRecent, born) {
  if (born) return { background: BASE_NODE_BG, border: '#51cf66' };
  if (seedRecent) {
    return { background: `rgb(${Math.round(lerp(51, 255, act))}, ${Math.round(lerp(56, 107, act))}, ${Math.round(lerp(74, 107, act))})`,
             border: '#ff3b3b' };
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
      if (meta) nodeUpdates.push({ id, size: baseNodeOptions(meta).size, color: { background: BASE_NODE_BG, border: BASE_NODE_BORDER } });
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
      color: nodeColor(newAct, seedRecent, born),
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
  document.getElementById('counts').innerHTML = `nós: ${nodeMeta.size} · ${parts}`;
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
  const d = new Date((meta.created_at || 0) * 1000);
  const conf = Math.round((meta.confidence || 0) * 100);
  const confColor = conf >= 80 ? '#51cf66' : conf >= 55 ? '#ffd166' : '#ff6b6b';
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
    return `<div class="nbr" onclick="focusNode(${other?.id})">
      <span class="t" style="background:${st.color}22;color:${st.color}">${st.name}</span>
      <span>${other ? other.label : '#' + '?'} · w=${(edge.weight || 0).toFixed(2)}</span>
    </div>`;
  }).join('') || '<div class="meta">(sem vizinhos — memória isolada no grafo)</div>';

  document.getElementById('panelBody').innerHTML = `
    <h3>#${meta.id} — memória LT</h3>
    <div class="meta">fonte: ${meta.source || '?'} · grau: ${degreeOf(id)}</div>
    <div class="meta">criada: ${isNaN(d) ? '?' : d.toLocaleString('pt-BR')} · acessos: ${meta.access_count ?? '?'}</div>
    <div class="meta">confiança: <span style="color:${confColor}"><b>${conf}%</b></span>
      ${meta.forgettable ? '<span class="pill" style="background:#ffd16622;color:#ffd166">esquecível</span>'
                          : '<span class="pill" style="background:#51cf6622;color:#51cf66">permanente</span>'}</div>
    <div class="text">${escapeHtml(meta.text || '')}</div>
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
  // limpa highlight anterior
  for (const id of searchHighlight) {
    const meta = nodeMeta.get(id);
    if (meta) nodes.update({ id, color: { background: BASE_NODE_BG, border: BASE_NODE_BORDER } });
  }
  searchHighlight = [];
  const q = e.target.value.trim().toLowerCase();
  if (!q) return;
  for (const [id, meta] of nodeMeta.entries()) {
    if ((meta.text || '').toLowerCase().includes(q)) {
      searchHighlight.push(id);
      nodes.update({ id, color: { background: BASE_NODE_BG, border: '#00e5ff' }, borderWidth: 2.5 });
    }
  }
  pushLog(`🔍 busca "${q}": ${searchHighlight.length} memória(s)`);
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
  else applyActivation(evt);          // "activation" ou linhas antigas sem type
};

(async () => {
  try {
    const g = await (await fetch('/graph')).json();
    applyNodes(g.nodes || [], false);
    applyEdges(g.edges || [], false);
    pushLog(`snapshot: ${nodeMeta.size} nós, ${edgeMeta.size} arestas`);
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
    ap = argparse.ArgumentParser(description="Visualização em tempo real do grafo Hebbiano de memórias")
    ap.add_argument("--db", default="./Modules/memory/db/ava_memory.db")
    ap.add_argument("--log", default="./Modules/memory/db/activation_log.jsonl")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--poll", type=float, default=1.0, help="intervalo de polling do diff (s)")
    args = ap.parse_args()

    state = GraphState(args.db)

    # Snapshot inicial ANTES de servir — o cliente já nasce com o grafo atual.
    delta = state.refresh()
    snap = state.snapshot()
    if delta["edges"]["added"]:
        extra = f"+{len(delta['edges']['added'])} arestas desde o último poll"
    else:
        extra = "nada novo"
    print(f"Grafo carregado: {len(snap['nodes'])} nós, {len(snap['edges'])} arestas ({extra})")

    broadcaster = Broadcaster()

    tailer = ActivationTailer(args.log, broadcaster)
    tailer.start()

    poller = GraphPoller(state, broadcaster, interval=args.poll)
    poller.start()

    handler = make_handler(state, broadcaster, args.log)
    server = ThreadingHTTPServer(("localhost", args.port), handler)
    print(f"Abra http://localhost:{args.port} no navegador.")
    print(f"Polling de diff: {args.poll}s · ativações: {args.log}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
