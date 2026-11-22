"""Parser de GenCAD 1.4 (extensão .cad, o formato "CAD" lido pelo OpenBoardView).

Converte o arquivo em um modelo `Board` com componentes, pinos (com coordenadas
absolutas), redes, vias, testpoints e contorno da placa. Tudo em milímetros.

Convenções do GenCAD usadas aqui:
  * ``UNITS USER 1000`` -> 1000 unidades por polegada (1 unidade = 1 mil).
  * Um componente em ``LAYER BOTTOM`` tem a forma ``MIRRORY FLIP``: o pino é
    espelhado em X no sistema local, depois rotacionado (anti-horário) por
    ``ROTATION`` e por fim transladado para ``PLACE``.
"""
from __future__ import annotations

import math
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

MIL_TO_MM = 0.0254


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
@dataclass
class Pin:
    part: str
    name: str
    net: str | None
    x: float  # mm
    y: float  # mm
    w: float  # mm (tamanho do pad já rotacionado)
    h: float
    side: str  # TOP | BOTTOM
    thru: bool = False

    @property
    def ref(self) -> str:
        return f"{self.part}.{self.name}"


@dataclass
class Part:
    refdes: str
    x: float
    y: float
    side: str
    rotation: float
    shape: str
    device: str
    value: str
    package: str
    technology: str
    pins: list[Pin] = field(default_factory=list)

    @property
    def prefix(self) -> str:
        m = re.match(r"[A-Za-z]+", self.refdes)
        return m.group(0).upper() if m else ""

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        if not self.pins:
            return (self.x, self.y, self.x, self.y)
        xs0 = [p.x - p.w / 2 for p in self.pins]
        xs1 = [p.x + p.w / 2 for p in self.pins]
        ys0 = [p.y - p.h / 2 for p in self.pins]
        ys1 = [p.y + p.h / 2 for p in self.pins]
        return (min(xs0), min(ys0), max(xs1), max(ys1))

    @property
    def center(self) -> tuple[float, float]:
        if not self.pins:
            return (self.x, self.y)
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)


@dataclass
class Via:
    net: str
    x: float
    y: float
    padstack: str
    name: str


@dataclass
class TestPoint:
    name: str
    x: float
    y: float
    net: str
    side: str


@dataclass
class Board:
    source: str = ""
    drawing: str = ""
    revision: str = ""
    units_per_inch: float = 1000.0
    outline: list[tuple[float, float, float, float]] = field(default_factory=list)
    parts: dict[str, Part] = field(default_factory=dict)
    nets: dict[str, list[Pin]] = field(default_factory=dict)
    vias: list[Via] = field(default_factory=list)
    testpoints: list[TestPoint] = field(default_factory=list)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs, ys = [], []
        for x0, y0, x1, y1 in self.outline:
            xs += [x0, x1]
            ys += [y0, y1]
        if not xs:
            for p in self.parts.values():
                xs.append(p.x)
                ys.append(p.y)
        return (min(xs), min(ys), max(xs), max(ys))


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _tok(line: str) -> list[str]:
    try:
        return shlex.split(line, posix=True)
    except ValueError:
        return line.split()


def _rot(x: float, y: float, deg: float) -> tuple[float, float]:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return (x * c - y * s, x * s + y * c)


def _pad_bbox(items: list[list[str]]) -> tuple[float, float]:
    """(w, h) em unidades GenCAD a partir das primitivas de um PAD."""
    xs: list[float] = []
    ys: list[float] = []
    for t in items:
        k = t[0]
        try:
            if k == "CIRCLE":
                cx, cy, r = map(float, t[1:4])
                xs += [cx - r, cx + r]
                ys += [cy - r, cy + r]
            elif k == "RECTANGLE":
                x, y, w, h = map(float, t[1:5])
                xs += [x, x + w]
                ys += [y, y + h]
            elif k == "LINE":
                x0, y0, x1, y1 = map(float, t[1:5])
                xs += [x0, x1]
                ys += [y0, y1]
            elif k == "ARC":
                x0, y0, x1, y1 = map(float, t[1:5])
                xs += [x0, x1]
                ys += [y0, y1]
        except ValueError:
            continue
    if not xs:
        return (0.0, 0.0)
    return (max(xs) - min(xs), max(ys) - min(ys))


def parse_gencad(path: str | Path) -> Board:
    path = Path(path)
    text = path.read_text(encoding="latin-1", errors="replace")
    lines = [ln.rstrip("\r\n") for ln in text.split("\n")]

    b = Board(source=str(path))
    section = ""

    pads: dict[str, tuple[float, float]] = {}            # nome -> (w,h) unid.
    padstacks: dict[str, tuple[float, float]] = {}       # nome -> (w,h) unid.
    shapes: dict[str, dict] = {}                         # nome -> {pins, thru}
    devices: dict[str, str] = {}                         # DeviceN -> VALUE

    cur_pad: str | None = None
    cur_pad_items: list[list[str]] = []
    cur_ps: str | None = None
    cur_ps_best: tuple[int, tuple[float, float]] | None = None
    cur_shape: dict | None = None
    cur_comp: dict | None = None
    cur_dev: str | None = None
    cur_sig: str | None = None
    cur_route: str | None = None
    comps_raw: list[dict] = []
    sig_nodes: list[tuple[str, str, str]] = []
    outline_u: list[tuple[float, float, float, float]] = []

    def flush_pad():
        nonlocal cur_pad, cur_pad_items
        if cur_pad:
            pads[cur_pad] = _pad_bbox(cur_pad_items)
        cur_pad, cur_pad_items = None, []

    def flush_ps():
        nonlocal cur_ps, cur_ps_best
        if cur_ps and cur_ps_best:
            padstacks[cur_ps] = cur_ps_best[1]
        cur_ps, cur_ps_best = None, None

    for raw in lines:
        ln = raw.strip()
        if not ln:
            continue
        if ln.startswith("$"):
            if ln.startswith("$END"):
                if section == "PADS":
                    flush_pad()
                elif section == "PADSTACKS":
                    flush_ps()
                section = ""
            else:
                section = ln[1:]
            continue
        t = _tok(ln)
        k = t[0]

        if section == "HEADER":
            if k == "DRAWING" and len(t) > 1:
                b.drawing = t[1]
            elif k == "REVISION" and len(t) > 1:
                b.revision = t[1]
            elif k == "UNITS" and len(t) > 2:
                try:
                    b.units_per_inch = float(t[2])
                except ValueError:
                    pass

        elif section == "BOARD":
            if k == "LINE" and len(t) >= 5:
                outline_u.append(tuple(map(float, t[1:5])))  # type: ignore

        elif section == "PADS":
            if k == "PAD":
                flush_pad()
                cur_pad = t[1]
            elif cur_pad:
                cur_pad_items.append(t)

        elif section == "PADSTACKS":
            if k == "PADSTACK":
                flush_ps()
                cur_ps = t[1]
            elif k == "PAD" and cur_ps and len(t) >= 3:
                layer = t[2].upper()
                if layer in ("TOP", "BOTTOM", "INNER", "ALL"):
                    prio = {"TOP": 3, "BOTTOM": 3, "ALL": 2, "INNER": 1}[layer]
                    wh = pads.get(t[1], (0.0, 0.0))
                    if cur_ps_best is None or prio > cur_ps_best[0]:
                        cur_ps_best = (prio, wh)

        elif section == "SHAPES":
            if k == "SHAPE":
                cur_shape = {"pins": [], "thru": False}
                shapes[t[1]] = cur_shape
            elif cur_shape is not None:
                if k == "PIN" and len(t) >= 5:
                    cur_shape["pins"].append(
                        (t[1], t[2], float(t[3]), float(t[4]))
                    )
                elif k == "INSERT" and len(t) > 1:
                    cur_shape["thru"] = t[1].upper() == "TH"

        elif section == "COMPONENTS":
            if k == "COMPONENT":
                cur_comp = {"ref": t[1], "attrs": {}, "mirror": False}
                comps_raw.append(cur_comp)
            elif cur_comp is not None:
                if k == "PLACE":
                    cur_comp["x"], cur_comp["y"] = float(t[1]), float(t[2])
                elif k == "LAYER":
                    cur_comp["side"] = t[1].upper()
                elif k == "ROTATION":
                    cur_comp["rot"] = float(t[1])
                elif k == "SHAPE":
                    cur_comp["shape"] = t[1]
                    cur_comp["mirror"] = "MIRRORY" in [s.upper() for s in t[2:]]
                elif k == "DEVICE":
                    cur_comp["device"] = t[1]
                elif k == "ATTRIBUTE" and len(t) >= 4:
                    cur_comp["attrs"][t[2]] = t[3]

        elif section == "DEVICES":
            if k == "DEVICE":
                cur_dev = t[1]
            elif k == "VALUE" and cur_dev and len(t) > 1:
                devices[cur_dev] = t[1]

        elif section == "SIGNALS":
            if k == "SIGNAL":
                cur_sig = t[1]
                b.nets.setdefault(cur_sig, [])
            elif k == "NODE" and cur_sig and len(t) >= 3:
                sig_nodes.append((cur_sig, t[1], t[2]))

        elif section == "ROUTES":
            if k == "ROUTE":
                cur_route = t[1]
            elif k == "VIA" and cur_route and len(t) >= 5:
                b.vias.append(
                    Via(
                        net=cur_route,
                        x=float(t[2]) * MIL_TO_MM,
                        y=float(t[3]) * MIL_TO_MM,
                        padstack=t[1],
                        name=t[-1],
                    )
                )

        elif section == "TESTPINS":
            if k == "TESTPIN" and len(t) >= 5:
                side = t[-1].upper() if t[-1].upper() in ("TOP", "BOTTOM") else "?"
                b.testpoints.append(
                    TestPoint(
                        name=t[1],
                        x=float(t[2]) * MIL_TO_MM,
                        y=float(t[3]) * MIL_TO_MM,
                        net=t[4],
                        side=side,
                    )
                )

    k_mm = 25.4 / b.units_per_inch
    b.outline = [tuple(v * k_mm for v in seg) for seg in outline_u]  # type: ignore

    # --- componentes e pinos ------------------------------------------------
    for c in comps_raw:
        a = c["attrs"]
        shape = shapes.get(c.get("shape", ""), {"pins": [], "thru": False})
        side = c.get("side", "TOP")
        rot = c.get("rot", 0.0)
        dev = c.get("device", "")
        part = Part(
            refdes=c["ref"],
            x=c.get("x", 0.0) * k_mm,
            y=c.get("y", 0.0) * k_mm,
            side=side,
            rotation=rot,
            shape=c.get("shape", ""),
            device=dev,
            value=a.get("VALUE") or devices.get(dev, ""),
            package=a.get("COMP_PACKAGE", c.get("shape", "")),
            technology=a.get("TECHNOLOGY", "THRU" if shape["thru"] else "SMD"),
        )
        mirror = c["mirror"] or side == "BOTTOM"
        for pname, pstack, px, py in shape["pins"]:
            lx = -px if mirror else px
            rx, ry = _rot(lx, py, rot)
            w, h = padstacks.get(pstack, (0.0, 0.0))
            if int(round(rot)) % 180 == 90:
                w, h = h, w
            part.pins.append(
                Pin(
                    part=part.refdes,
                    name=pname,
                    net=None,
                    x=part.x + rx * k_mm,
                    y=part.y + ry * k_mm,
                    w=w * k_mm,
                    h=h * k_mm,
                    side=side,
                    thru=shape["thru"],
                )
            )
        b.parts[part.refdes] = part

    # --- nets -> pinos ------------------------------------------------------
    pin_index = {(p.part, p.name): p for part in b.parts.values() for p in part.pins}
    for net, ref, pn in sig_nodes:
        pin = pin_index.get((ref, pn))
        if pin is None:
            # pino citado no SIGNAL sem geometria na forma: cria pino sintético
            part = b.parts.get(ref)
            if part is None:
                continue
            pin = Pin(ref, pn, None, part.x, part.y, 0, 0, part.side)
            part.pins.append(pin)
            pin_index[(ref, pn)] = pin
        pin.net = net
        b.nets[net].append(pin)

    return b
