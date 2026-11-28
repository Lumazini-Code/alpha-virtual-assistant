"""Índices derivados do `Board`: classes de partes/redes, grade espacial, grafo."""
from __future__ import annotations

import math
from collections import defaultdict

from .gencad import Board, Part, Pin
from .semantics import (
    NetInfo, decode_value, is_ground_name, net_name_kind, net_state_hint,
    net_tags, net_voltage_hint, part_class,
)

CELL_MM = 5.0
GND_CAP_CLASSES = {"capacitor", "capacitor_polarized"}


class BoardIndex:
    def __init__(self, board: Board):
        self.b = board
        self.part_cls: dict[str, str] = {r: part_class(p) for r, p in board.parts.items()}
        self.net_info: dict[str, NetInfo] = {}
        self.vias_by_net: dict[str, list] = defaultdict(list)
        self.tps_by_net: dict[str, list] = defaultdict(list)
        for v in board.vias:
            self.vias_by_net[v.net].append(v)
        for t in board.testpoints:
            self.tps_by_net[t.net].append(t)

        # redes de terra: por nome, ou rede enorme com "GND" no nome
        self.ground_nets: set[str] = {
            n for n in board.nets
            if is_ground_name(n) and (len(board.nets[n]) >= 5 or n.upper() in ("GND", "AGND"))
        }
        self._classify_nets()
        self._grid: dict[tuple[int, int], list[str]] = defaultdict(list)
        for r, p in board.parts.items():
            cx, cy = p.center
            self._grid[(int(cx // CELL_MM), int(cy // CELL_MM))].append(r)

    # ------------------------------------------------------------------ #
    def _classify_nets(self) -> None:
        for name, pins in self.b.nets.items():
            kind = net_name_kind(name)
            conf = "name"
            if name in self.ground_nets:
                kind = "ground"
            elif kind == "ground":
                kind = "signal"       # nome com GND mas poucos pinos (ex.: FB_CAL_TERM_GND)
            else:
                ncaps = self.count_caps_to_ground(name)
                if kind == "signal" and ncaps >= 3 and "#" not in name:
                    kind, conf = "power", "name+topology"
                elif kind == "power" and ncaps == 0 and len(pins) <= 2:
                    kind, conf = "signal", "name+topology"
            self.net_info[name] = NetInfo(
                name=name,
                kind=kind,
                voltage_hint=net_voltage_hint(name) if kind == "power" else None,
                state_hint=net_state_hint(name) if kind == "power" else None,
                tags=net_tags(name),
                confidence=conf,
            )

    # ------------------------------------------------------------------ #
    def cls(self, refdes: str) -> str:
        return self.part_cls.get(refdes, "other")

    def is_ground(self, net: str | None) -> bool:
        return net is not None and net in self.ground_nets

    def other_nets(self, part: Part, net: str) -> list[str]:
        return sorted({q.net for q in part.pins if q.net and q.net != net})

    def count_caps_to_ground(self, net: str) -> int:
        n = 0
        for pin in self.b.nets.get(net, []):
            if self.cls(pin.part) in GND_CAP_CLASSES:
                part = self.b.parts[pin.part]
                if any(q is not pin and self.is_ground(q.net) for q in part.pins):
                    n += 1
        return n

    def value_info(self, refdes: str) -> dict:
        p = self.b.parts[refdes]
        return decode_value(p.value, self.cls(refdes))

    # ------------------------------------------------------------------ #
    # Resolução de nomes
    # ------------------------------------------------------------------ #
    def find_part(self, refdes: str) -> Part:
        p = self.b.parts.get(refdes) or self.b.parts.get(refdes.upper())
        if p is None:
            import difflib
            close = difflib.get_close_matches(refdes.upper(), self.b.parts.keys(), n=5, cutoff=0.6)
            raise KeyError(f"Componente '{refdes}' não existe." + (f" Parecidos: {close}" if close else ""))
        return p

    def find_net(self, name: str) -> str:
        if name in self.b.nets:
            return name
        up = {n.upper(): n for n in self.b.nets}
        if name.upper() in up:
            return up[name.upper()]
        import difflib
        close = difflib.get_close_matches(name.upper(), list(up), n=6, cutoff=0.6)
        raise KeyError(f"Rede '{name}' não existe." + (f" Parecidas: {[up[c] for c in close]}" if close else ""))

    # ------------------------------------------------------------------ #
    # Espacial
    # ------------------------------------------------------------------ #
    def parts_near(self, x: float, y: float, radius: float) -> list[str]:
        r = int(radius // CELL_MM) + 2
        cx, cy = int(x // CELL_MM), int(y // CELL_MM)
        out: list[str] = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                out += self._grid.get((cx + dx, cy + dy), [])
        return out

    @staticmethod
    def bbox_gap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        dx = max(b[0] - a[2], a[0] - b[2], 0.0)
        dy = max(b[1] - a[3], a[1] - b[3], 0.0)
        return math.hypot(dx, dy)

    def pin_dist(self, a: Pin, b: Pin) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    # ------------------------------------------------------------------ #
    # Grafo de redes para trace
    # ------------------------------------------------------------------ #
    def build_net_graph(self, series_classes: set[str], hub_limit: int = 400):
        """Aresta net_a --(parte)--> net_b para partes de 'série'.

        Redes de terra e redes com mais de `hub_limit` pinos não viram nós
        intermediários (explodiriam a busca)."""
        adj: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
        for refdes, p in self.b.parts.items():
            if self.cls(refdes) not in series_classes:
                continue
            by_net: dict[str, list[str]] = defaultdict(list)
            for q in p.pins:
                if q.net:
                    by_net[q.net].append(q.name)
            nets = list(by_net)
            for i in range(len(nets)):
                for j in range(i + 1, len(nets)):
                    a, c = nets[i], nets[j]
                    adj[a].append((c, refdes, ",".join(by_net[a]), ",".join(by_net[c])))
                    adj[c].append((a, refdes, ",".join(by_net[c]), ",".join(by_net[a])))
        self._hub_limit = hub_limit
        return adj
