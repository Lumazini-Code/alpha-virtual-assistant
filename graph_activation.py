import argparse
import html
import json
import queue
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def load_static_graph(db_path: str, node_table: str, text_col: str, edge_table: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    nodes_rows = conn.execute(f"SELECT id, {text_col} AS text FROM {node_table}").fetchall()
    edges_rows = conn.execute(
        f"SELECT memory_id_a, memory_id_b, edge_type, weight FROM {edge_table}"
    ).fetchall()
    conn.close()
    nodes = []
    for r in nodes_rows:
        text = (r["text"] or "").strip().replace("\n", " ")
        nodes.append({
            "id": r["id"],
            "label": text[:40] + ("…" if len(text) > 40 else ""),
            "title": html.escape(text[:400]),
        })
    edges = [
        {"from": r["memory_id_a"], "to": r["memory_id_b"], "type": r["edge_type"], "weight": r["weight"]}
        for r in edges_rows
    ]
    return nodes, edges


class LogTailer(threading.Thread):
    """Lê linhas novas do JSONL e distribui pra todas as filas SSE inscritas."""
    daemon = True

    def __init__(self, log_path: str):
        super().__init__()
        self.log_path = log_path
        self.subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def run(self):
        pos = 0
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                f.seek(0, 2)
                pos = f.tell()
        except FileNotFoundError:
            pos = 0
        while True:
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    f.seek(pos)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        with self._lock:
                            subs = list(self.subscribers)
                        for q in subs:
                            q.put(line)
                    pos = f.tell()
            except FileNotFoundError:
                pass
            time.sleep(0.25)


def make_handler(nodes, edges, tailer: LogTailer):
    nodes_json = json.dumps(nodes, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)

    page = PAGE_TEMPLATE.replace("__NODES__", nodes_json).replace("__EDGES__", edges_json)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # silencia log padrão

        def do_GET(self):
            if self.path == "/":
                body = page.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                q = tailer.subscribe()
                try:
                    while True:
                        try:
                            line = q.get(timeout=15)
                            self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
                            self.wfile.flush()
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    tailer.unsubscribe(q)
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Ativação do grafo — ava (tempo real)</title>
<script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  html, body { margin:0; padding:0; height:100%; background:#0b0d13; font-family: -apple-system, sans-serif; }
  #network { width:100%; height:100vh; }
  #hud {
    position:absolute; top:12px; left:12px; z-index:10;
    background:rgba(20,22,30,0.88); color:#e8e8ea; padding:10px 14px;
    border-radius:10px; font-size:13px; line-height:1.5; max-width:280px;
  }
  #hud b { color:#fff; }
  #status { position:absolute; top:12px; right:16px; z-index:10; font-size:12px; color:#8ecae6; }
  #log { max-height:140px; overflow-y:auto; font-size:11px; color:#9aa; margin-top:6px; }
</style>
</head>
<body>
<div id="hud">
  <b>Ativação em tempo real</b><br>
  Sementes (leitura) em <span style="color:#ff6b6b">vermelho</span>,
  propagação PPR em <span style="color:#ffd166">amarelo</span>.<br>
  <div id="log"></div>
</div>
<div id="status">conectando…</div>
<div id="network"></div>
<script>
  const rawNodes = __NODES__;
  const rawEdges = __EDGES__;

  const nodeIndex = new Map(rawNodes.map(n => [n.id, n]));
  const activation = new Map();   // id -> valor atual (0..1)
  const isSeed = new Map();       // id -> timestamp em que foi semente (p/ cor diferente por um tempo)

  const nodes = new vis.DataSet(rawNodes.map(n => ({
    id: n.id, label: n.label, title: n.title,
    shape: 'dot', size: 10,
    color: { background: '#3a3f52', border: '#54607a' },
    font: { color: '#9aa0b4', size: 11 },
  })));

  const edges = new vis.DataSet(rawEdges.map((e, i) => ({
    id: i, from: e.from, to: e.to,
    color: { color: '#2a2e3d', opacity: 0.5 },
    width: 1,
  })));

  const container = document.getElementById('network');
  const network = new vis.Network(container, { nodes, edges }, {
    physics: {
      solver: 'forceAtlas2Based',
      forceAtlas2Based: { gravitationalConstant: -60, springLength: 100, springConstant: 0.05 },
      stabilization: { iterations: 150 },
    },
    interaction: { hover: true, tooltipDelay: 100 },
  });

  function lerp(a, b, t) { return a + (b - a) * t; }

  function colorForActivation(act, seedRecent) {
    // base cinza-azulado -> amarelo/vermelho conforme ativação
    if (seedRecent) {
      return `rgb(${Math.round(lerp(58,255,act))}, ${Math.round(lerp(63,107,act))}, ${Math.round(lerp(82,107,act))})`;
    }
    return `rgb(${Math.round(lerp(58,255,act))}, ${Math.round(lerp(63,209,act))}, ${Math.round(lerp(82,102,act))})`;
  }

  function tick() {
    const decay = 0.90;
    const nodeUpdates = [];
    const edgeUpdates = [];
    const now = Date.now();

    for (const [id, act] of Array.from(activation.entries())) {
      const newAct = act * decay;
      if (newAct < 0.01) {
        activation.delete(id);
        isSeed.delete(id);
        nodeUpdates.push({ id, size: 10, color: { background: '#3a3f52', border: '#54607a' } });
        continue;
      }
      activation.set(id, newAct);
      const seedTs = isSeed.get(id);
      const seedRecent = seedTs && (now - seedTs) < 1200;
      nodeUpdates.push({
        id,
        size: 10 + newAct * 26,
        color: {
          background: colorForActivation(newAct, seedRecent),
          border: seedRecent ? '#ff3b3b' : '#ffd166',
        },
      });
    }
    if (nodeUpdates.length) nodes.update(nodeUpdates);

    for (const e of rawEdges) {
      const a = activation.get(e.from) || 0;
      const b = activation.get(e.to) || 0;
      const m = Math.min(a, b);
      if (m > 0.02) {
        edgeUpdates.push({ id: rawEdges.indexOf(e), color: { color: '#ffd166', opacity: Math.min(0.9, m + 0.2) }, width: 1 + m * 5 });
      }
    }
    if (edgeUpdates.length) edges.update(edgeUpdates);

    requestAnimationFrame(() => setTimeout(tick, 60));
  }
  tick();

  function pushLog(msg) {
    const el = document.getElementById('log');
    const line = document.createElement('div');
    line.textContent = msg;
    el.prepend(line);
    while (el.childNodes.length > 6) el.removeChild(el.lastChild);
  }

  const statusEl = document.getElementById('status');
  const es = new EventSource('/events');
  es.onopen = () => { statusEl.textContent = 'conectado'; statusEl.style.color = '#7CFC00'; };
  es.onerror = () => { statusEl.textContent = 'reconectando…'; statusEl.style.color = '#ffb703'; };
  es.onmessage = (ev) => {
    if (!ev.data) return;
    let evt;
    try { evt = JSON.parse(ev.data); } catch (e) { return; }
    const seeds = evt.seeds || [];
    const rank = evt.rank || {};
    const now = Date.now();

    for (const sid of seeds) {
      activation.set(sid, Math.max(activation.get(sid) || 0, 1.0));
      isSeed.set(sid, now);
    }
    for (const [k, v] of Object.entries(rank)) {
      const id = parseInt(k, 10);
      if (seeds.includes(id)) continue;
      activation.set(id, Math.max(activation.get(id) || 0, v));
    }
    const seedLabels = seeds.map(id => (nodeIndex.get(id) || {}).label || id).join(', ');
    pushLog(`ativou a partir de: ${seedLabels} (${Object.keys(rank).length} nós, ${evt.elapsed_ms}ms)`);
  };
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./Modules/memory/ava_memory.db")
    ap.add_argument("--node-table", default="memories")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--edge-table", default="memory_edges")
    ap.add_argument("--log", default="./Modules/memory/activation_log.jsonl")
    ap.add_argument("--port", type=int, default=8777)
    args = ap.parse_args()

    nodes, edges = load_static_graph(args.db, args.node_table, args.text_col, args.edge_table)
    print(f"Grafo carregado: {len(nodes)} nós, {len(edges)} arestas.")

    tailer = LogTailer(args.log)
    tailer.start()

    handler = make_handler(nodes, edges, tailer)
    server = ThreadingHTTPServer(("localhost", args.port), handler)
    print(f"Abra http://localhost:{args.port} no navegador.")
    print(f"Aguardando eventos em: {args.log}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
