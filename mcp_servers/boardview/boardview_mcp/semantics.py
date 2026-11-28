"""Heurísticas sobre refdes, valores e nomes de rede.

Um boardview não traz esquemático nem valores confiáveis: tudo aqui é inferência
a partir de convenções de nomes (Dell/Wistron/Quanta no arquivo de exemplo, mas
as regras são genéricas). Toda inferência é marcada como "hint"/"heuristic".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .gencad import Part

# --------------------------------------------------------------------------- #
# Valores
# --------------------------------------------------------------------------- #
_R_RE = re.compile(r"^(?P<val>[0-9DKM]+?)R(?P<size>\d+)(?P<tol>[FJ])?")
_C_RE = re.compile(r"^S?C(?P<cap>[0-9DK]+[PNU])(?P<v>[0-9DK]+)V")
_TANT_RE = re.compile(r"^(?:S?E|ST)(?P<cap>[0-9D]+)U(?P<v>[0-9D]+)V")


def _num(s: str) -> float | None:
    """'24D9' -> 24.9, 'D33' -> 0.33, '3K3' -> 3300, '1K' -> 1000, '2M2' -> 2.2e6."""
    s = s.upper()
    mult = 1.0
    for ch, m in (("K", 1e3), ("M", 1e6)):
        if ch in s:
            mult = m
            s = s.replace(ch, ".", 1)
            break
    s = s.replace("D", ".")
    if s.endswith("."):
        s = s[:-1]
    if s.startswith("."):
        s = "0" + s
    try:
        return float(s) * mult
    except ValueError:
        return None


def _fmt_ohm(v: float) -> str:
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


def _fmt_cap(f: float) -> str:
    if f >= 1e-6:
        return f"{f * 1e6:g}uF"
    if f >= 1e-9:
        return f"{f * 1e9:g}nF"
    return f"{f * 1e12:g}pF"


def decode_value(value: str, cls: str) -> dict:
    """Decodifica o código de valor (best-effort). Devolve {} se não reconhecido."""
    v = (value or "").upper()
    out: dict = {}
    if cls in ("resistor", "jumper") and v:
        m = _R_RE.match(v)
        if m:
            ohm = _num(m.group("val"))
            if ohm is not None:
                out["ohms"] = ohm
                out["pretty"] = _fmt_ohm(ohm) + "ohm"
                out["size_code"] = m.group("size")
                tol = m.group("tol")
                if tol:
                    out["tolerance"] = {"F": "1%", "J": "5%"}[tol]
    elif cls in ("capacitor", "capacitor_polarized") and v:
        m = _C_RE.match(v)
        if m:
            capstr = m.group("cap")
            unit = capstr[-1]
            n = _num(capstr[:-1])
            if n is not None:
                f = n * {"P": 1e-12, "N": 1e-9, "U": 1e-6}[unit]
                out["farads"] = f
                out["pretty"] = _fmt_cap(f)
                vol = _num(m.group("v").replace("K", "")) if "K" not in m.group("v") else None
                if "K" in m.group("v"):
                    vol = (_num(m.group("v").replace("K", "")) or 0) * 1000
                if vol is not None:
                    out["rated_volts"] = vol
        else:
            m2 = _TANT_RE.match(v)
            if m2:
                n = _num(m2.group("cap"))
                vol = _num(m2.group("v"))
                if n is not None:
                    out["farads"] = n * 1e-6
                    out["pretty"] = _fmt_cap(n * 1e-6)
                if vol is not None:
                    out["rated_volts"] = vol
    return out


# --------------------------------------------------------------------------- #
# Classificação de partes
# --------------------------------------------------------------------------- #
SERIES_CLASSES = {
    "resistor", "jumper", "inductor", "ferrite_bead", "fuse", "thermistor",
}
ACTIVE_SERIES = {"mosfet", "transistor", "diode"}

_IC_PREFIXES = {
    "U", "PU", "CPU", "GPU", "VRAM", "KBC", "SCA", "EC", "PCH", "SIO", "IC",
}
_CONN_PREFIXES = {
    "CN", "J", "JP", "XDP", "DIMM", "NGFF", "USB", "SSUSB", "HDMI", "DCIN",
    "CON", "SKT", "LVDS", "EDP", "BAT", "SD", "CR", "SIM", "FPC",
}
_MOSFET_RE = re.compile(
    r"^(2N7002|AO\d|AON\d|SI\d|SIS|SIR|NTR|NTMF|BSS|BSC|RJK|FDMC|FDS|FDN|"
    r"IRF|PMV|DMN|DMP|TPCA|TPC|SSM|CSD|MTP|UPA|ME\d|AP\d|PJ\d|VTM)", re.I)


def classify_part(p: Part) -> str:
    pre = p.prefix
    val = (p.value or "").upper()
    pkg = (p.package or "").upper()
    # --- por valor / pacote (têm prioridade, pois prefixos variam por OEM) ---
    if val.startswith(("AZ", "ESD", "PESD", "TPD", "SRV05", "PRTR", "USBLC")):
        return "esd_protection"
    if val.startswith("NTC") or val.startswith("PTC"):
        return "thermistor"
    if val.startswith(("XTAL", "OSC")) or pre in ("X", "Y"):
        return "crystal"
    if val.startswith(("SKT-", "ACES-CON", "DC-JACK", "ARG-CON", "PAD-")) or pre.endswith("CON"):
        return "connector"
    if val.startswith("XFORM") or pre == "XF":
        return "transformer"
    if val.startswith("FILTER") or val.startswith(("DLM", "MCM")) or pre == "TR":
        return "common_mode_filter"
    if val.startswith(("GAP-CLOSE", "GAP_CLOSE")) or (pre in ("PG", "P") and "PAD" in val.split("-")[:3]):
        return "jumper"
    if val.startswith("FUSE") or val.startswith("POLYSW") or pre in ("F", "PF"):
        return "fuse"
    if val.startswith("IND-") or pre in ("L", "PL"):
        return "inductor"
    if pre in ("FB", "PFB") or val.startswith(("BEAD", "FBMH", "BLM", "MPZ", "HCB")):
        return "ferrite_bead"
    if pre in ("RN",) or val.startswith("SRN"):
        return "resistor_array"
    if pre in ("TP", "PAD", "TPAD") or val.startswith("TPAD"):
        return "testpoint"
    if pre in ("H", "HOLE", "MH") or (val.startswith(("GEN", "HOLE")) and len(p.pins) <= 1):
        return "mechanical"
    # --- por prefixo ---
    if pre in ("R", "PR", "PRN"):
        if val.startswith("0R") or val.startswith("0R0") or "PAD" in val and val.startswith("0"):
            return "jumper"
        return "resistor"
    if pre in ("C", "PC", "CE"):
        return "capacitor"
    if pre in ("TC", "PT", "EC", "TCP") or re.match(r"^(S?E|ST)\d", val):
        return "capacitor_polarized"
    if pre in ("D", "PD", "ZD", "LED", "DS"):
        return "diode"
    if pre in ("Q", "PQ", "QS"):
        if _MOSFET_RE.match(val):
            return "mosfet"
        return "transistor"
    if pre in _CONN_PREFIXES:
        return "connector"
    if pre in _IC_PREFIXES:
        return "ic"
    if pre == "P" and len(p.pins) == 2:
        return "jumper" if val.startswith("0R") else "passive_other"
    if pre == "PU":
        return "ic"
    # PU com pacote de MOSFET (AO4407 etc.) aparece como "ic" por prefixo; refina:
    if len(p.pins) >= 8:
        return "ic"
    return "other"


def refine_active(p: Part, cls: str) -> str:
    """PU/U com part-number de MOSFET (AO4407, AON6510...) vira 'mosfet'."""
    if cls == "ic" and _MOSFET_RE.match((p.value or "").upper()):
        return "mosfet"
    return cls


def part_class(p: Part) -> str:
    return refine_active(p, classify_part(p))


# --------------------------------------------------------------------------- #
# Redes
# --------------------------------------------------------------------------- #
_GND_RE = re.compile(r"(^|_)(A|P|S|C|D)?GND(A|D)?($|_)", re.I)
_CTRL_RE = re.compile(
    r"(PWRGD|PGOOD|_PG$|_PG_|POK|_EN$|_EN#|_EN_|ENABLE|_FB$|_FB_|_SS$|COMP|ILIM|"
    r"SW$|PHASE|_LX|BOOT|UGATE|LGATE|DRVH|DRVL|_CS\d|_REF$|#|"
    r"_EN\d*(_|$)|_FB\d*(_|$)|_PH\d*(_|$)|_SN$|_LDO\d|_BYP\d|_ACDET|_ACN$|_ACP$|_AD_OFF|VREF)", re.I)
_RAIL_RE = re.compile(
    r"(^\+?\d+D?\d*V\d*(_|$)|^\d+V\d+|^\+?\d+P\d+V|DCBATOUT|^DC_?IN|DC_\d+V|^VCC|^VDD|^VIN|^VBAT|"
    r"^BATT|^B\+|^ADAPTER|VCORE|^VBUS|^VSYS|3V3|5V0?_|^\+?V\d)", re.I)
_STATE_RE = re.compile(r"(?:^|_)(S0|S3|S4|S5|G3|AUX|DSW|RTC|ALW|RUN|PRIM)(?:$|_)", re.I)


def net_voltage_hint(name: str) -> float | None:
    n = name.upper()
    # ramos do barramento do adaptador/bateria e nós internos de reguladores não têm a tensão do nome
    if n.startswith(("DCBATOUT", "PWR_DCBATOUT")) or (n.startswith("PWR_") and not re.search(r"_V(OUT|IN)$", n)):
        return None
    m = re.match(r"^\+?(\d+)D(\d+)V", n)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m = re.match(r"^\+?(\d+)P(\d+)V", n)  # 3P3V_SPI
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m = re.match(r"^\+?(\d+)V(\d+)", n)  # 3V3AUX
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m = re.match(r"^\+?(\d+)V(?!\d)", n)
    if m:
        return float(m.group(1))
    m = re.match(r"^DC_(\d+)V", n)
    if m:
        return float(m.group(1))
    for rx in (r"(\d+)[DP](\d+)V", r"(?<!\d)(\d+)V(\d+)(?!\d)", r"_(\d+)[DP](\d+)(?:_|$)"):
        m = re.search(rx, n)
        if m:
            return float(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"(?:^|_)(\d+)V(?:_|$)", n)
    if m:
        return float(m.group(1))
    return None


def net_state_hint(name: str) -> str | None:
    if re.match(r"^(DCBATOUT|DC_?IN|ADAPTER|VBAT|BATT)", name, re.I):
        return "barramento do adaptador/bateria (tipicamente 7–20 V, presente antes de qualquer power-on)"
    m = _STATE_RE.search(name)
    if not m:
        return None
    s = m.group(1).upper()
    return {
        "S5": "S5 (standby: presente com o notebook 'desligado' + adaptador/bateria)",
        "S3": "S3 (suspensão)",
        "S0": "S0 (máquina ligada)",
        "S4": "S4 (hibernação)",
        "G3": "G3 (mecanicamente desligado / RTC)",
        "AUX": "AUX (alimentação auxiliar/standby)",
        "DSW": "DSW (deep sleep well do PCH)",
        "RTC": "RTC (bateria do relógio)",
        "ALW": "ALW (always-on)",
        "RUN": "RUN (só em operação)",
        "PRIM": "PRIM (primary well do PCH)",
    }[s]


def is_ground_name(name: str) -> bool:
    return bool(_GND_RE.search(name))


def net_name_kind(name: str) -> str:
    """'ground' | 'power' | 'signal' — só pelo nome (EN/PG/FB etc. contam como sinal)."""
    if is_ground_name(name):
        return "ground"
    if _CTRL_RE.search(name):
        return "signal"
    if _RAIL_RE.search(name):
        return "power"
    return "signal"


SIGNAL_TAGS = [
    ("i2c", re.compile(r"(SCL|SDA|I2C|SMB)", re.I)),
    ("usb", re.compile(r"USB", re.I)),
    ("pcie", re.compile(r"(PCIE|PEG|NGFF|CLKREQ|PERST)", re.I)),
    ("ddr", re.compile(r"(DDR|DQ|DQS|MA_|CKE|ODT|CLK_?DDR|M_A_)", re.I)),
    ("clock", re.compile(r"(CLK|XTAL|OSC|32K|24M|25M)", re.I)),
    ("reset", re.compile(r"(RST|RESET|PLTRST|RSMRST)", re.I)),
    ("power_good", re.compile(r"(PWRGD|PGOOD|_PG$|_PG_|POK|PG\b)", re.I)),
    ("enable", re.compile(r"(_EN$|_EN#|_EN_|ENABLE|^EN_|ON/OFF|SLP_S)", re.I)),
    ("spi", re.compile(r"(SPI|FLASH)", re.I)),
    ("lpc_espi", re.compile(r"(LPC|ESPI|LAD\d)", re.I)),
    ("display", re.compile(r"(HDMI|EDP|LVDS|DDI|TMDS|PANEL|BKLT|LCD)", re.I)),
    ("audio", re.compile(r"(HDA|AUD|SPK|HP_|MIC)", re.I)),
    ("debug_jtag", re.compile(r"(XDP|JTAG|TCK|TMS|TDI|TDO|TRST)", re.I)),
    ("battery_charger", re.compile(r"(BATT|CHG|ACOK|ADAPTER|DCIN|SMBC|SMBD)", re.I)),
    ("thermal", re.compile(r"(THERM|FAN|TEMP|NTC|PROCHOT)", re.I)),
    ("active_low", re.compile(r"#$")),
]


def net_tags(name: str) -> list[str]:
    return [t for t, rx in SIGNAL_TAGS if rx.search(name)]


@dataclass
class NetInfo:
    name: str
    kind: str                  # ground | power | control | signal
    voltage_hint: float | None
    state_hint: str | None
    tags: list[str]
    confidence: str            # name | name+topology
