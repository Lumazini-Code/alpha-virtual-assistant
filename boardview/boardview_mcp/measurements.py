"""Medições feitas pelo técnico, persistidas em JSON ao lado do estado do MCP.

Permite à IA cruzar o que foi medido (tensão, resistência/diodo ao GND,
continuidade) com a topologia da placa.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .index import BoardIndex

KINDS = ("voltage", "resistance_to_gnd", "diode_to_gnd", "continuity", "current", "frequency", "note")
STATES = ("unpowered", "adapter_only", "battery_only", "power_button_pressed", "running", "unknown")
UNIT_DEFAULT = {"voltage": "V", "resistance_to_gnd": "ohm", "diode_to_gnd": "V", "current": "A",
                "frequency": "Hz", "continuity": "", "note": ""}


def data_dir() -> Path:
    d = Path(os.environ.get("BOARDVIEW_MCP_DATA", Path.home() / ".boardview_mcp"))
    d.mkdir(parents=True, exist_ok=True)
    return d


class MeasurementStore:
    def __init__(self, board_source: str):
        h = hashlib.sha1(os.path.abspath(board_source).encode()).hexdigest()[:10]
        self.path = data_dir() / f"{Path(board_source).stem[:40]}.{h}.measurements.json"
        self.items: list[dict] = []
        if self.path.exists():
            try:
                self.items = json.loads(self.path.read_text())
            except Exception:
                self.items = []

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.items, ensure_ascii=False, indent=1))

    def add(self, target: str, kind: str, value, unit: str | None, state: str, note: str) -> dict:
        if kind not in KINDS:
            raise ValueError(f"kind inválido '{kind}'. Use: {', '.join(KINDS)}")
        if state not in STATES:
            raise ValueError(f"state inválido '{state}'. Use: {', '.join(STATES)}")
        nid = (max((i["id"] for i in self.items), default=0)) + 1
        item = {"id": nid, "target": target, "kind": kind, "value": value,
                "unit": unit if unit is not None else UNIT_DEFAULT[kind], "state": state,
                "note": note, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.items.append(item)
        self._save()
        return item

    def remove(self, ids: list[int] | None) -> int:
        before = len(self.items)
        self.items = [] if ids is None else [i for i in self.items if i["id"] not in ids]
        self._save()
        return before - len(self.items)


def _to_float(v) -> float | None:
    try:
        return float(str(v).replace(",", "."))
    except ValueError:
        return None


def _target_net(ix: BoardIndex, target: str) -> str | None:
    if target in ix.b.nets:
        return target
    if "." in target:
        ref, pin = target.rsplit(".", 1)
        p = ix.b.parts.get(ref.upper())
        if p:
            for q in p.pins:
                if q.name.upper() == pin.upper():
                    return q.net
    up = {n.upper(): n for n in ix.b.nets}
    return up.get(target.upper())


def check_measurements(ix: BoardIndex, items: list[dict]) -> list[dict]:
    """Cruza medições com a topologia. Heurístico — devolve flags, não diagnósticos."""
    out = []
    for it in items:
        net = _target_net(ix, it["target"])
        info = ix.net_info.get(net) if net else None
        flags: list[str] = []
        sev = "info"
        v = _to_float(it["value"])
        kind, state = it["kind"], it["state"]
        if info and info.kind == "ground" and kind in ("resistance_to_gnd", "diode_to_gnd"):
            flags.append("alvo é a própria rede de GND: a medição ao GND não tem significado (meça num pino/rede de sinal ou rail)")
            out.append({**it, "resolved_net": net, "severity": "warn", "flags": flags})
            continue
        if net is None and it["kind"] != "note":
            flags.append(f"alvo '{it['target']}' não resolve para uma rede conhecida")
            sev = "warn"
        if kind == "voltage" and v is not None and info:
            hint = info.voltage_hint
            st = (info.state_hint or "")
            on_states = {"adapter_only", "power_button_pressed", "running"}
            if info.kind == "ground" and abs(v) > 0.05:
                flags.append(f"{abs(v):g} V em rede de GND: referência de medição errada ou GND flutuando")
                sev = "warn"
            elif hint:
                standby = any(k in st for k in ("S5", "AUX", "ALW", "DSW", "RTC", "G3"))
                should_be_on = (standby and state in on_states) or (
                    not standby and state in ("running",)) or (state == "power_button_pressed" and "S3" in st)
                if should_be_on:
                    if v < 0.1 * hint:
                        flags.append(f"AUSENTE: esperado ~{hint:g} V (pelo nome) neste estado, medido {v:g} V")
                        sev = "fault"
                    elif v < 0.9 * hint:
                        flags.append(f"BAIXA: esperado ~{hint:g} V, medido {v:g} V (carga/curto parcial ou regulador em limite)")
                        sev = "warn"
                    elif v > 1.15 * hint:
                        flags.append(f"ALTA: esperado ~{hint:g} V, medido {v:g} V")
                        sev = "warn"
                    else:
                        flags.append(f"ok vs nome da rede (~{hint:g} V)")
                else:
                    flags.append(f"rede {info.state_hint or 'sem estado inferido'}: tensão {v:g} V no estado '{state}' pode ser normal")
        elif kind == "resistance_to_gnd" and v is not None:
            ohm = v * (1000 if it["unit"].lower().startswith("k") else 1e6 if it["unit"].lower().startswith("m") else 1)
            if state not in ("unpowered", "unknown"):
                flags.append("resistência deve ser medida com a placa DESLIGADA e sem bateria/adaptador")
                sev = "warn"
            if ohm < 5:
                flags.append("provável CURTO ao GND (<5 ohm)")
                sev = "fault"
            elif ohm < 50:
                flags.append("resistência muito baixa para o GND — suspeito (rails de baixa tensão/alta corrente podem ter poucos ohms normalmente)")
                sev = "warn" if sev != "fault" else sev
        elif kind == "diode_to_gnd":
            if isinstance(it["value"], str) and it["value"].strip().upper() in ("OL", "OPEN"):
                flags.append("OL: sem condução no modo diodo (aberto ou rail alto-Z)")
            elif v is not None and v < 0.05:
                flags.append("~0 V no modo diodo: CURTO ao GND")
                sev = "fault"
        elif kind == "continuity":
            flags.append("continuidade registrada" if str(it["value"]).lower() in ("1", "true", "yes", "sim") else "SEM continuidade registrada")
        if net and info and info.kind == "power" and not flags:
            flags.append(f"rail {net} (~{info.voltage_hint if info.voltage_hint else '?'} V, {info.state_hint or 'estado desconhecido'})")
        out.append({**it, "resolved_net": net, "severity": sev, "flags": flags})
    return out
