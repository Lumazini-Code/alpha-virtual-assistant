"""Implementação das consultas expostas como tools MCP (puras: dict in -> dict out)."""
from __future__ import annotations

import fnmatch
import math
import re
from collections import Counter, defaultdict, deque

from .gencad import Part, Pin
from .index import BoardIndex, GND_CAP_CLASSES
from .semantics import ACTIVE_SERIES, SERIES_CLASSES

R3 = lambda v: round(v, 3)  # noqa: E731

CLASS_ORDER = [
    "ic", "connector", "mosfet", "transistor", "inductor", "ferrite_bead", "fuse",
    "diode", "esd_protection", "resistor", "resistor_array", "jumper", "thermistor",
    "capacitor", "capacitor_polarized", "common_mode_filter", "crystal",
    "transformer", "testpoint", "mechanical", "passive_other", "other",
]
_ORD = {c: i for i, c in enumerate(CLASS_ORDER)}


def matcher(pattern: str):
    """Glob (case-insensitive) ou regex quando prefixado por 're:'."""
    if pattern.startswith("re:"):
        rx = re.compile(pattern[3:], re.I)
        return lambda s: bool(rx.search(s or ""))
    pat = pattern.upper()
    if not any(ch in pat for ch in "*?["):
        pat = f"*{pat}*"          # sem curinga = "contém"
    return lambda s: fnmatch.fnmatchcase((s or "").upper(), pat)


def _pin(ix: BoardIndex, q: Pin) -> dict:
    return {"pin": q.name, "net": q.net, "x": R3(q.x), "y": R3(q.y)}


def _part_brief(ix: BoardIndex, refdes: str) -> dict:
    p = ix.b.parts[refdes]
    d = {"refdes": refdes, "class": ix.cls(refdes), "value": p.value,
         "package": p.package, "side": p.side, "pins": len(p.pins)}
    dec = ix.value_info(refdes)
    if dec.get("pretty"):
        d["decoded"] = dec["pretty"]
    return d


# --------------------------------------------------------------------------- #
def board_info(ix: BoardIndex) -> dict:
    b = ix.b
    classes = Counter(ix.cls(r) for r in b.parts)
    sides = Counter(p.side for p in b.parts.values())
    kinds = Counter(i.kind for i in ix.net_info.values())
    big = sorted(
        ((r, len(p.pins)) for r, p in b.parts.items() if ix.cls(r) in ("ic", "connector") and len(p.pins) >= 20),
        key=lambda t: -t[1],
    )
    top_nets = sorted(((n, len(p)) for n, p in b.nets.items()), key=lambda t: -t[1])[:8]
    x0, y0, x1, y1 = b.bounds
    return {
        "source": b.source,
        "drawing": b.drawing,
        "revision": b.revision,
        "format": "GenCAD 1.4 (.cad)",
        "units": "todas as coordenadas e distâncias em mm; origem do arquivo; Y cresce para cima; lado BOTTOM é visto 'por baixo' só no render",
        "board_bounds_mm": {"x0": R3(x0), "y0": R3(y0), "x1": R3(x1), "y1": R3(y1),
                            "width": R3(x1 - x0), "height": R3(y1 - y0)},
        "counts": {
            "parts": len(b.parts), "nets": len(b.nets),
            "pins": sum(len(p.pins) for p in b.parts.values()),
            "pins_without_net": sum(1 for p in b.parts.values() for q in p.pins if q.net is None),
            "vias": len(b.vias), "testpoints": len(b.testpoints),
        },
        "parts_by_side": dict(sides),
        "parts_by_class": dict(classes.most_common()),
        "nets_by_kind_heuristic": dict(kinds),
        "ground_nets": sorted(ix.ground_nets),
        "largest_ics_and_connectors": [{"refdes": r, "pins": n, "value": b.parts[r].value, "side": b.parts[r].side}
                                       for r, n in big[:20]],
        "largest_nets": [{"net": n, "pins": c} for n, c in top_nets],
        "notes": [
            "Boardview não tem esquemático, valores reais, camadas internas nem trilhas: a análise usa conectividade + geometria.",
            "Classes de componentes e tipos de rede são heurísticas por nome/prefixo/valor.",
        ],
    }


# --------------------------------------------------------------------------- #
def search_parts(ix: BoardIndex, pattern: str, field: str = "any", side: str | None = None,
                 part_class: str | None = None, limit: int = 50) -> dict:
    m = matcher(pattern)
    out = []
    for r, p in ix.b.parts.items():
        if side and p.side != side.upper():
            continue
        if part_class and ix.cls(r) != part_class:
            continue
        ok = (
            (field in ("refdes", "any") and m(r))
            or (field in ("value", "any") and m(p.value))
            or (field in ("package", "any") and m(p.package))
            or (field in ("device", "any") and m(p.device))
        )
        if ok:
            out.append(r)
    out.sort(key=lambda r: (_ORD.get(ix.cls(r), 99), r))
    return {
        "pattern": pattern, "field": field, "total_matches": len(out),
        "truncated": len(out) > limit,
        "parts": [_part_brief(ix, r) for r in out[:limit]],
    }


def _role_hints(ix: BoardIndex, p: Part) -> list[str]:
    cls = ix.cls(p.refdes)
    nets = [q.net for q in p.pins]
    kinds = [ix.net_info[n].kind if n in ix.net_info else None for n in nets]
    hints: list[str] = []
    gnd = [n for n in nets if ix.is_ground(n)]
    other = [n for n in nets if n and not ix.is_ground(n)]
    if cls in GND_CAP_CLASSES and gnd and other:
        hints.append(f"capacitor de desacoplamento/filtro em {other[0]} (outro terminal no GND)")
    elif cls in GND_CAP_CLASSES and len(set(other)) == 2:
        hints.append("capacitor em série entre duas redes (acoplamento AC / bootstrap)")
    if cls in ("resistor", "resistor_array") and len(p.pins) == 2:
        if gnd and other:
            hints.append(f"resistor para GND em {other[0]} (pull-down, divisor, carga de sangria ou sensor)")
        elif len(set(other)) == 2:
            k = [ix.net_info[n].kind for n in set(other)]
            if k.count("power") == 1:
                hints.append("candidato a pull-up (um lado em rail, outro em sinal) ou resistor de série")
            elif k.count("power") == 2:
                hints.append("liga dois rails (pode ser jumper/zero-ohm ou resistor de sensing)")
            else:
                hints.append("resistor de série entre sinais (terminação/proteção)")
    if cls == "jumper":
        hints.append("jumper/zero-ohm/gap: liga as duas redes diretamente (continuidade esperada)")
    if cls == "inductor" and len(set(other)) == 2:
        hints.append("indutor em série: típico de saída de conversor buck/boost ou filtro de alimentação")
    if cls == "fuse":
        hints.append("fusível/PTC em série: confira continuidade se uma das redes estiver sem tensão")
    if cls in ("mosfet", "transistor"):
        hints.append("chave ativa: condução depende do gate (pino de controle); o boardview não informa a função de cada pino")
    if cls == "esd_protection":
        hints.append("proteção ESD/TVS: se estiver em curto, derruba a rede inteira (candidato clássico a curto)")
    if cls == "ic":
        npower = sum(1 for k in kinds if k == "power")
        hints.append(f"CI com {npower} pinos em redes de alimentação")
    return hints


def get_part(ix: BoardIndex, refdes: str, include_pins: bool = True, pin_limit: int = 100,
             pin_filter: str | None = None) -> dict:
    p = ix.find_part(refdes)
    r = p.refdes
    cls = ix.cls(r)
    x0, y0, x1, y1 = p.bbox
    pins = p.pins
    if pin_filter:
        m = matcher(pin_filter)
        pins = [q for q in pins if m(q.name) or m(q.net or "")]
    by_kind: dict[str, list] = defaultdict(list)
    for q in p.pins:
        k = "unconnected" if q.net is None else ix.net_info[q.net].kind
        by_kind[k].append(q)
    nets_summary: dict[str, list[str]] = defaultdict(list)
    for q in p.pins:
        if q.net:
            nets_summary[q.net].append(q.name)
    out = {
        "refdes": r, "class": cls, "value": p.value, "decoded_value": ix.value_info(r) or None,
        "package": p.package, "shape": p.shape, "technology": p.technology, "side": p.side,
        "x": R3(p.x), "y": R3(p.y), "rotation_deg": p.rotation,
        "bbox_mm": {"x0": R3(x0), "y0": R3(y0), "x1": R3(x1), "y1": R3(y1)},
        "pin_count": len(p.pins),
        "pins_by_net_kind": {k: len(v) for k, v in by_kind.items()},
        "role_hints": _role_hints(ix, p),
    }
    if len(p.pins) > 12:
        pw = {n: v for n, v in nets_summary.items() if ix.net_info[n].kind == "power"}
        out["power_nets_on_part"] = {n: {"pins": v[:12] + (["..."] if len(v) > 12 else []), "count": len(v),
                                         "voltage_hint": ix.net_info[n].voltage_hint,
                                         "state_hint": ix.net_info[n].state_hint}
                                     for n, v in sorted(pw.items(), key=lambda t: -len(t[1]))[:40]}
        out["ground_pin_count"] = len(by_kind.get("ground", []))
    if include_pins:
        lst = []
        for q in pins[:pin_limit]:
            d = _pin(ix, q)
            if q.net:
                ni = ix.net_info[q.net]
                d["net_kind"] = ni.kind
                d["net_pins"] = len(ix.b.nets[q.net])
            lst.append(d)
        out["pins"] = lst
        if len(pins) > pin_limit:
            out["pins_truncated"] = f"{len(pins) - pin_limit} pinos omitidos; use pin_filter (glob no nome do pino ou da rede) para filtrar"
    return out


# --------------------------------------------------------------------------- #
def search_nets(ix: BoardIndex, pattern: str, kind: str | None = None, min_pins: int = 0,
                limit: int = 60) -> dict:
    m = matcher(pattern)
    out = []
    for n, pins in ix.b.nets.items():
        info = ix.net_info[n]
        if kind and info.kind != kind:
            continue
        if len(pins) < min_pins or not m(n):
            continue
        out.append((n, info, len(pins)))
    out.sort(key=lambda t: (-t[2], t[0]))
    return {
        "pattern": pattern, "total_matches": len(out), "truncated": len(out) > limit,
        "nets": [{"net": n, "kind": i.kind, "pins": c, "voltage_hint": i.voltage_hint,
                  "state_hint": i.state_hint, "tags": i.tags} for n, i, c in out[:limit]],
    }


def get_net(ix: BoardIndex, name: str, limit: int = 60, include_passives: bool = True) -> dict:
    n = ix.find_net(name)
    info = ix.net_info[n]
    pins = ix.b.nets[n]
    by_part: dict[str, list[Pin]] = defaultdict(list)
    for q in pins:
        by_part[q.part].append(q)
    cls_count = Counter(ix.cls(r) for r in by_part)
    xs = [q.x for q in pins]
    ys = [q.y for q in pins]
    out: dict = {
        "net": n, "kind": info.kind, "confidence": info.confidence,
        "voltage_hint_V": info.voltage_hint, "power_state_hint": info.state_hint, "tags": info.tags,
        "pin_count": len(pins), "part_count": len(by_part),
        "parts_by_class": dict(cls_count.most_common()),
        "bbox_mm": {"x0": R3(min(xs)), "y0": R3(min(ys)), "x1": R3(max(xs)), "y1": R3(max(ys))} if pins else None,
        "vias": len(ix.vias_by_net.get(n, [])),
        "testpoints": [{"name": t.name, "x": R3(t.x), "y": R3(t.y), "side": t.side}
                       for t in ix.tps_by_net.get(n, [])][:20],
    }
    if info.kind == "ground" and len(by_part) > limit:
        out["note"] = "Rede de terra enorme: só resumo. Use get_part/neighbors para ver o GND de um componente."
        return out
    rows = []
    for r in sorted(by_part, key=lambda r: (_ORD.get(ix.cls(r), 99), r)):
        c = ix.cls(r)
        if not include_passives and c in ("resistor", "capacitor", "capacitor_polarized", "jumper"):
            continue
        d = _part_brief(ix, r)
        d["pins_on_net"] = [q.name for q in by_part[r]]
        part = ix.b.parts[r]
        if len(part.pins) <= 4:
            d["other_nets"] = ix.other_nets(part, n)
        rows.append(d)
    out["parts"] = rows[:limit]
    if len(rows) > limit:
        out["truncated"] = f"{len(rows) - limit} componentes omitidos; use include_passives=false ou aumente limit"
    return out


# --------------------------------------------------------------------------- #
def _resolve_endpoint(ix: BoardIndex, s: str) -> tuple[set[str], str]:
    """refdes | REF.PIN | rede -> (conjunto de redes, rótulo)."""
    if s in ix.b.nets:
        return {s}, f"rede {s}"
    if "." not in s and s.upper() not in ix.b.parts:
        up = {n.upper(): n for n in ix.b.nets}
        if s.upper() in up:
            return {up[s.upper()]}, f"rede {up[s.upper()]}"
    if "." in s:
        ref, pin = s.rsplit(".", 1)
        part = ix.find_part(ref)
        for q in part.pins:
            if q.name.upper() == pin.upper():
                if not q.net:
                    raise KeyError(f"Pino {part.refdes}.{q.name} não tem rede.")
                return {q.net}, f"pino {part.refdes}.{q.name}"
        raise KeyError(f"{part.refdes} não tem pino '{pin}'.")
    part = ix.find_part(s)
    nets = {q.net for q in part.pins if q.net and not ix.is_ground(q.net)}
    return nets, f"componente {part.refdes}"


def trace_connectivity(ix: BoardIndex, start: str, end: str, max_hops: int = 4,
                       include_active: bool = True, max_paths: int = 5, hub_limit: int = 400) -> dict:
    S, sl = _resolve_endpoint(ix, start)
    T, tl = _resolve_endpoint(ix, end)
    series = set(SERIES_CLASSES) | (set(ACTIVE_SERIES) if include_active else set())
    adj = ix.build_net_graph(series)

    def is_hub(net: str) -> bool:
        return ix.is_ground(net) or len(ix.b.nets[net]) > hub_limit

    # BFS multi-fonte guardando todos os pais de caminho mais curto
    dist: dict[str, int] = {}
    parents: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    dq: deque[str] = deque()
    for s in S:
        dist[s] = 0
        dq.append(s)
    found_at: int | None = None
    while dq:
        u = dq.popleft()
        if found_at is not None and dist[u] > found_at:
            break
        if u in T:
            found_at = dist[u] if found_at is None else found_at
            continue
        if dist[u] >= max_hops:
            continue
        for v, part, pa, pb in adj.get(u, []):
            if is_hub(v) and v not in T:
                continue
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
            if dist[v] == dist[u] + 1:
                parents[v].append((u, part, pa, pb))

    targets = [t for t in T if t in dist and (found_at is None or dist[t] == found_at)]
    if not targets or found_at is None:
        return {
            "start": sl, "end": tl, "connected": False,
            "message": (f"Nenhum caminho em até {max_hops} saltos via componentes de série "
                        f"({', '.join(sorted(series))}). Hubs ignorados: GND e redes com > {hub_limit} pinos. "
                        "Pode ainda haver ligação por CI/regulador (não rastreada) ou por mais saltos."),
        }

    if found_at == 0:
        shared = sorted(S & T)
        return {"start": sl, "end": tl, "connected": True, "hops": 0,
                "message": "Já estão na mesma rede (sem componentes no meio).", "shared_nets": shared,
                "paths": [{"text": n, "steps": [{"net": n}]} for n in shared]}

    paths: list[list[dict]] = []

    def walk(net: str, tail: list[dict]):
        if len(paths) >= max_paths:
            return
        if dist[net] == 0:
            paths.append([{"net": net}] + tail)
            return
        for u, part, pa, pb in parents[net]:
            step = {"via_part": part, "class": ix.cls(part), "value": ix.b.parts[part].value,
                    "from_pins": pa, "to_pins": pb, "net": net}
            walk(u, [step] + tail)

    for t in targets:
        walk(t, [])
        if len(paths) >= max_paths:
            break

    def text(path: list[dict]) -> str:
        s = path[0]["net"]
        for st in path[1:]:
            tag = " [ATIVO: depende do gate]" if st["class"] in ACTIVE_SERIES else ""
            s += f" --[{st['via_part']} {st['value']} pin {st['from_pins']}->{st['to_pins']}{tag}]--> {st['net']}"
        return s

    return {
        "start": sl, "end": tl, "connected": True, "hops": found_at,
        "paths": [{"text": text(p), "steps": p} for p in paths],
        "caveat": "Caminhos assumem condução através de R/L/FB/fusível/jumper (e FET/diodo se include_active). "
                  "Valores reais e estado de FETs devem ser confirmados por medição.",
    }


# --------------------------------------------------------------------------- #
def neighbors(ix: BoardIndex, target: str, radius_mm: float = 5.0, include_other_side: bool = True,
              part_class: str | None = None, limit: int = 40) -> dict:
    b = ix.b
    shared_nets: set[str] = set()
    if re.match(r"^\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?", target):
        bits = [t.strip() for t in target.split(",")]
        tx, ty = float(bits[0]), float(bits[1])
        tside = (bits[2].upper() if len(bits) > 2 else None)
        tb = (tx, ty, tx, ty)
        tref = None
        reach = radius_mm
    else:
        tp = ix.find_part(target)
        tx, ty = tp.center
        tb = tp.bbox
        tside = tp.side
        tref = tp.refdes
        shared_nets = {q.net for q in tp.pins if q.net and not ix.is_ground(q.net)}
        reach = radius_mm + max(tb[2] - tb[0], tb[3] - tb[1])
    rows = []
    for r in set(ix.parts_near(tx, ty, reach)):
        if r == tref:
            continue
        p = b.parts[r]
        if not include_other_side and tside and p.side != tside:
            continue
        if part_class and ix.cls(r) != part_class:
            continue
        gap = ix.bbox_gap(tb, p.bbox)
        if gap > radius_mm:
            continue
        d = _part_brief(ix, r)
        d["gap_mm"] = R3(gap)
        d["same_side"] = (tside is None) or p.side == tside
        d["dx"], d["dy"] = R3(p.center[0] - tx), R3(p.center[1] - ty)
        common = sorted({q.net for q in p.pins if q.net in shared_nets})
        if common:
            d["shares_nets_with_target"] = common
        if len(p.pins) <= 4:
            d["nets"] = sorted({q.net for q in p.pins if q.net})
        rows.append(d)
    rows.sort(key=lambda d: (d["gap_mm"], d["refdes"]))
    return {"target": target, "center_mm": {"x": R3(tx), "y": R3(ty)}, "radius_mm": radius_mm,
            "total": len(rows), "truncated": len(rows) > limit, "neighbors": rows[:limit]}


# --------------------------------------------------------------------------- #
def _cap_total(ix: BoardIndex, refs: list[str]) -> float:
    t = 0.0
    for r in refs:
        f = ix.value_info(r).get("farads")
        if f:
            t += f
    return t


def power_rail_analysis(ix: BoardIndex, net: str) -> dict:
    n = ix.find_net(net)
    info = ix.net_info[n]
    b = ix.b
    by_part: dict[str, list[Pin]] = defaultdict(list)
    for q in b.nets[n]:
        by_part[q.part].append(q)

    decoupling: list[str] = []
    to_gnd_res: list[dict] = []
    series: list[dict] = []
    esd: list[str] = []
    loads: list[dict] = []
    other_res: list[dict] = []
    tps: list[str] = []
    for r, pins in by_part.items():
        part = b.parts[r]
        cls = ix.cls(r)
        others = ix.other_nets(part, n)
        to_g = any(ix.is_ground(o) for o in others)
        if cls in GND_CAP_CLASSES and to_g:
            decoupling.append(r)
        elif cls == "esd_protection":
            esd.append(r)
        elif cls == "testpoint":
            tps.append(r)
        elif cls in ("resistor", "resistor_array") and to_g:
            to_gnd_res.append({**_part_brief(ix, r), "pins_on_net": [q.name for q in pins]})
        elif cls in SERIES_CLASSES | ACTIVE_SERIES | {"resistor", "resistor_array"}:
            far = [o for o in others if not ix.is_ground(o)]
            if far:
                series.append({**_part_brief(ix, r), "pins_on_net": [q.name for q in pins],
                               "other_nets": far})
        elif cls in ("ic", "connector", "common_mode_filter", "transformer", "crystal"):
            loads.append({**_part_brief(ix, r), "pins_on_net": [q.name for q in pins]})

    sources = [s for s in series if s["class"] in ("inductor", "mosfet", "fuse", "ferrite_bead", "jumper", "transistor", "diode")]
    pass_res = [s for s in series if s["class"] in ("resistor", "resistor_array")]
    caps_total = _cap_total(ix, decoupling)
    # --- suspeitos de curto (heurística ordenada por probabilidade usual) ---
    suspects = []
    for r in decoupling:
        pol = ix.cls(r) == "capacitor_polarized"
        score = 3 if pol else 5
        suspects.append({"refdes": r, "why": "capacitor ligado entre a rede e GND (causa nº 1 de curto em rail)", "score": score,
                         "value": b.parts[r].value, "side": b.parts[r].side})
    for r in esd:
        suspects.append({"refdes": r, "why": "TVS/ESD ligado à rede (falha em curto é comum após surto)", "score": 4,
                         "value": b.parts[r].value, "side": b.parts[r].side})
    for l in loads:
        if l["class"] == "ic":
            suspects.append({"refdes": l["refdes"], "why": "CI alimentado por esta rede (curto interno/carga excessiva)", "score": 2,
                             "value": l["value"], "side": l["side"]})
    for s in sources:
        if s["class"] in ("mosfet", "transistor"):
            suspects.append({"refdes": s["refdes"], "why": "FET/chave conectado à rede (dreno-fonte em curto?)", "score": 3,
                             "value": s["value"], "side": s["side"]})
    suspects.sort(key=lambda d: (-d["score"], d["refdes"]))

    return {
        "net": n, "kind": info.kind, "confidence": info.confidence,
        "voltage_hint_V": info.voltage_hint, "power_state_hint": info.state_hint,
        "pin_count": len(b.nets[n]),
        "decoupling_caps": {
            "count": len(decoupling),
            "estimated_total": (f"{caps_total * 1e6:.3g} uF" if caps_total else None),
            "parts": [{"refdes": r, "value": b.parts[r].value, "decoded": ix.value_info(r).get("pretty"),
                       "side": b.parts[r].side} for r in sorted(decoupling)][:60],
        },
        "possible_sources_or_series_elements": sources[:30],
        "series_resistors_to_other_nets": pass_res[:30],
        "resistors_to_ground": to_gnd_res[:30],
        "protection_esd": [{"refdes": r, "value": b.parts[r].value} for r in esd],
        "loads_ics_connectors": loads[:40],
        "testpoints": {"parts": tps[:20], "gencad_testpins": [{"name": t.name, "x": R3(t.x), "y": R3(t.y), "side": t.side}
                                                              for t in ix.tps_by_net.get(n, [])][:20]},
        "short_to_ground_suspects_ranked": suspects[:25],
        "method": "Heurística de ranking: capacitores > TVS/ESD > FETs > CIs. Confirme com medição de resistência/diodo ao GND, "
                  "injeção limitada de tensão e câmera térmica; o boardview não prova defeito.",
    }


def list_power_rails(ix: BoardIndex, state: str | None = None, min_v: float | None = None,
                     max_v: float | None = None, contains: str | None = None, limit: int = 150) -> dict:
    m = matcher(contains) if contains else None
    rows = []
    for n, info in ix.net_info.items():
        if info.kind != "power":
            continue
        if state and not (info.state_hint and state.upper() in info.state_hint.upper()):
            continue
        if min_v is not None and (info.voltage_hint is None or info.voltage_hint < min_v):
            continue
        if max_v is not None and (info.voltage_hint is None or info.voltage_hint > max_v):
            continue
        if m and not m(n):
            continue
        pins = ix.b.nets[n]
        caps = [q.part for q in pins if ix.cls(q.part) in GND_CAP_CLASSES]
        srcs = sorted({q.part for q in pins if ix.cls(q.part) in ("inductor", "fuse", "mosfet", "ferrite_bead")})
        rows.append({
            "net": n, "voltage_hint_V": info.voltage_hint, "power_state_hint": info.state_hint,
            "confidence": info.confidence, "pins": len(pins), "caps_on_net": len(set(caps)),
            "source_candidates": srcs[:6], "has_testpoint": bool(ix.tps_by_net.get(n)),
        })
    rows.sort(key=lambda d: (-(d["voltage_hint_V"] or -1), d["net"]))
    return {"total": len(rows), "truncated": len(rows) > limit, "rails": rows[:limit],
            "note": "Tensão e estado inferidos do NOME da rede (ex.: 3D3V_S5 ~ 3,3 V standby). Confirme no esquemático/medição."}
