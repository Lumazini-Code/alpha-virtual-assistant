"""Servidor MCP para análise de boardviews (GenCAD .cad) por IAs.

Stdio por padrão; `--transport streamable-http` expõe /mcp para o mcp_manager
do host (BOARDVIEW_MCP_HOST/BOARDVIEW_MCP_PORT controlam o bind).
"""
from __future__ import annotations

import functools
import glob as _glob
import json
import os
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP, Image

from . import analysis as A
from .gencad import parse_gencad
from .index import BoardIndex
from .measurements import KINDS, STATES, MeasurementStore, check_measurements
from .render import render_region as _render
from .render import resolve_center

mcp = FastMCP(
    "boardview",
    host=os.environ.get("BOARDVIEW_MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("BOARDVIEW_MCP_PORT", "8000")),
    streamable_http_path="/mcp",
    stateless_http=True,
    json_response=True,
    instructions=(
        "Análise de placas eletrônicas a partir de boardview GenCAD (.cad). Comece com load_board (se nenhuma placa "
        "estiver carregada) e board_info. Para rails use list_power_rails e power_rail_analysis; para ver um "
        "componente, get_part; para uma rede, get_net; para caminhos, trace_connectivity; para o que está ao redor, "
        "neighbors; para ver a região, render_region. Registre medições com add_measurement e cruze com "
        "check_measurements. O boardview não tem esquemático: tudo é conectividade e geometria; classes e tensões "
        "são heurísticas por nome."
    ),
)

_STATE: dict[str, Any] = {"ix": None, "store": None}


def _ix() -> BoardIndex:
    if _STATE["ix"] is None:
        env = os.environ.get("BOARDVIEW_FILE")
        if env:
            _load(env)
        else:
            raise ValueError("Nenhuma placa carregada. Chame load_board(path) primeiro.")
    return _STATE["ix"]


_BOARD_SUFFIXES = (".cad", ".gcd", ".gencad")


def _board_files(base: Path) -> list[Path]:
    """Arquivos de boardview sob `base` (diretos primeiro, recursivo como fallback)."""
    found = sorted(
        p for p in base.glob("*")
        if p.is_file() and p.suffix.lower() in _BOARD_SUFFIXES
    )
    if not found:
        found = sorted(
            p for p in base.rglob("*")
            if p.is_file() and p.suffix.lower() in _BOARD_SUFFIXES
        )
    return found


def _resolve_board_path(raw: str) -> Path:
    """Resolve um caminho de boardview aceitando arquivo, diretório ou glob.

    O cwd de um servidor stdio é a pasta de trabalho escolhida pelo usuário;
    aceitar `.` nessa pasta evita exigir que o modelo descubra o nome do `.cad`
    com uma ferramenta de listagem separada.
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Caminho vazio. Passe o arquivo .cad, a pasta que o contém ou um padrão como '*.cad'.")
    p = Path(raw).expanduser()

    if any(ch in raw for ch in "*?["):
        matches = sorted(
            Path(m) for m in _glob.glob(str(p), recursive=True)
            if Path(m).is_file() and Path(m).suffix.lower() in _BOARD_SUFFIXES
        )
        if not matches:
            raise ValueError(f"Nenhum boardview corresponde a '{raw}' (procurado a partir de {Path.cwd()}).")
        if len(matches) > 1:
            listing = ", ".join(str(m) for m in matches[:20])
            raise ValueError(f"O padrão '{raw}' encontrou {len(matches)} boardviews: {listing}. Passe o caminho de um deles.")
        return matches[0].resolve()

    if not p.exists():
        raise ValueError(f"Arquivo não encontrado: {p} (cwd atual: {Path.cwd()})")

    if p.is_dir():
        matches = _board_files(p)
        if not matches:
            raise ValueError(f"Nenhum arquivo .cad/.gcd/.gencad encontrado em {p}.")
        if len(matches) > 1:
            listing = ", ".join(str(m) for m in matches[:20])
            raise ValueError(f"{p} tem {len(matches)} boardviews: {listing}. Passe o caminho de um deles.")
        return matches[0].resolve()

    ext = p.suffix.lower()
    if ext not in _BOARD_SUFFIXES:
        raise ValueError(f"Formato '{ext}' não suportado ainda (só GenCAD .cad). Conversores: o OpenBoardView lê outros formatos, "
                         "mas este servidor implementa apenas o parser GenCAD.")
    return p.resolve()


def _load(path: str) -> BoardIndex:
    p = _resolve_board_path(path)
    ix = BoardIndex(parse_gencad(p))
    _STATE["ix"] = ix
    _STATE["store"] = MeasurementStore(str(p))
    return ix


def guard(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except KeyError as e:
            raise ValueError(e.args[0] if e.args else str(e)) from None
    return wrapper


# --------------------------------------------------------------------------- #
@mcp.tool()
@guard
def load_board(path: str) -> dict:
    """Carrega um boardview GenCAD (.cad) e o torna a placa ativa. Retorna o resumo (contagens, limites, ICs maiores).
    `path` aceita o arquivo `.cad`, a pasta que o contém (ex.: `.` = cwd do servidor) ou um glob (`*.cad`).
    Se houver mais de um boardview, a resposta lista os candidatos para você escolher. Chame uma vez por sessão;
    as medições registradas são lembradas por arquivo."""
    _load(path)
    return A.board_info(_STATE["ix"])


@mcp.tool()
@guard
def board_info() -> dict:
    """Visão geral da placa ativa: contagens, lados, classes de componentes, redes de GND, maiores CIs/conectores e
    maiores redes. Use no começo para se orientar."""
    return A.board_info(_ix())


@mcp.tool()
@guard
def list_power_rails(state: str | None = None, min_voltage: float | None = None,
                     max_voltage: float | None = None, contains: str | None = None,
                     limit: int = 150) -> dict:
    """Lista redes de alimentação (heurística por nome/topologia) com tensão e estado inferidos do nome
    (ex.: 3D3V_S5 ≈ 3,3 V standby), nº de capacitores, componentes-fonte candidatos (indutor/FET/fusível) e
    se há testpoint. `state`: 'S5', 'S3', 'S0', 'AUX'... `contains`: glob/regex ('re:') no nome. Use para montar
    a árvore de sequenciamento antes de medir."""
    return A.list_power_rails(_ix(), state, min_voltage, max_voltage, contains, limit)


@mcp.tool()
@guard
def search_parts(pattern: str, field: Literal["any", "refdes", "value", "package", "device"] = "any",
                 side: Literal["TOP", "BOTTOM"] | None = None, part_class: str | None = None,
                 limit: int = 50) -> dict:
    """Busca componentes. `pattern`: glob case-insensitive ('U1*', '*W25Q*'; sem curinga = 'contém') ou regex com
    prefixo 're:'. `part_class`: ic, connector, mosfet, transistor, inductor, ferrite_bead, fuse, diode,
    esd_protection, resistor, resistor_array, jumper, thermistor, capacitor, capacitor_polarized,
    common_mode_filter, crystal, transformer, testpoint, mechanical."""
    return A.search_parts(_ix(), pattern, field, side, part_class, limit)


@mcp.tool()
@guard
def get_part(refdes: str, include_pins: bool = True, pin_limit: int = 100,
             pin_filter: str | None = None) -> dict:
    """Detalhes de um componente: classe, valor (e valor decodificado: ex. SC10U6D3V → 10 uF/6,3 V), pacote, lado,
    posição, caixa, pinos com rede e coordenadas, dicas de papel (decoupling, pull-up, série...). Para CIs grandes
    (BGA) devolve resumo de redes de alimentação; use `pin_filter` (glob no nome do pino OU da rede) para filtrar."""
    return A.get_part(_ix(), refdes, include_pins, pin_limit, pin_filter)


@mcp.tool()
@guard
def search_nets(pattern: str, kind: Literal["power", "ground", "signal"] | None = None,
                min_pins: int = 0, limit: int = 60) -> dict:
    """Busca redes pelo nome (glob ou 're:regex'). Retorna tipo heurístico, nº de pinos, tensão/estado inferidos e tags
    (i2c, usb, pcie, ddr, clock, reset, power_good, enable, spi, display, audio, debug_jtag, battery_charger,
    thermal, active_low). Ex.: '*PWRGD*', 'PP3V3*', 're:VCC(IO|SA)'."""
    return A.search_nets(_ix(), pattern, kind, min_pins, limit)


@mcp.tool()
@guard
def get_net(name: str, limit: int = 60, include_passives: bool = True) -> dict:
    """Tudo que está numa rede: componentes agrupados por classe (com pinos e redes do outro lado dos passivos de 2
    pinos), vias, testpoints, caixa. Redes de GND gigantes retornam só resumo. Use include_passives=false para ver só
    ativos/indutores/FETs."""
    return A.get_net(_ix(), name, limit, include_passives)


@mcp.tool()
@guard
def trace_connectivity(start: str, end: str, max_hops: int = 4, include_active: bool = True,
                       max_paths: int = 5) -> dict:
    """Caminho(s) mais curto(s) entre dois pontos passando por componentes em série (R, L, ferrite, fusível,
    jumper/0 Ω, termistor; + FET/diodo se include_active). `start`/`end`: refdes ('U17'), pino ('U17.5') ou rede.
    GND e redes enormes não são atravessados. FETs são marcados como ATIVOS (condução depende do gate). Útil para
    seguir um rail da entrada até o consumidor ou achar o resistor entre dois sinais."""
    return A.trace_connectivity(_ix(), start, end, max_hops, include_active, max_paths)


@mcp.tool()
@guard
def neighbors(target: str, radius_mm: float = 5.0, include_other_side: bool = True,
              part_class: str | None = None, limit: int = 40) -> dict:
    """Componentes fisicamente próximos (distância entre caixas) a um refdes ou a um ponto 'x,y[,TOP|BOTTOM]' em mm.
    Retorna lado, deslocamento dx/dy, redes em comum com o alvo e as redes de passivos pequenos. Use para investigar
    dano térmico/líquido, capacitores sob BGA (lado oposto) e vizinhança de um componente suspeito."""
    return A.neighbors(_ix(), target, radius_mm, include_other_side, part_class, limit)


@mcp.tool()
@guard
def power_rail_analysis(net: str) -> dict:
    """Análise de uma rede de alimentação: capacitores de desacoplamento (com capacitância total estimada),
    fontes/elementos em série (indutores, FETs, fusíveis, ferrites), resistores para GND, TVS/ESD, cargas (CIs e
    conectores), testpoints e ranking de SUSPEITOS de curto ao GND. Ponto de partida para 'rail sem tensão' ou
    'curto no rail'."""
    return A.power_rail_analysis(_ix(), net)


@mcp.tool()
@guard
def render_region(center: str | None = None, x: float | None = None, y: float | None = None,
                  width_mm: float = 30.0, height_mm: float | None = None,
                  side: Literal["auto", "TOP", "BOTTOM"] = "auto",
                  highlight_parts: list[str] | None = None, highlight_nets: list[str] | None = None,
                  show_labels: bool = True, show_other_side: bool = False, max_px: int = 1000) -> list:
    """Imagem PNG de uma região da placa. Centre em um refdes/rede (`center`) ou em x,y (mm). `side='auto'` usa o lado
    do componente central; vista BOTTOM é espelhada (como virar a placa). Pinos: laranja=rail, cinza=GND, azul=sinal,
    magenta=rede destacada; contorno verde=componente destacado; círculos amarelos=vias, brancos=testpoints.
    Use largura 10–40 mm para ver refdes legíveis; >100 mm mostra só a visão geral."""
    ix = _ix()
    cx, cy, s, desc = resolve_center(ix, center, x, y, side)
    if center is None and x is None and y is None:
        x0, y0, x1, y1 = ix.b.bounds
        width_mm = max(width_mm, x1 - x0 + 4)
        height_mm = y1 - y0 + 4
    h = height_mm if height_mm else width_mm
    png, meta = _render(ix, cx, cy, width_mm, h, s, highlight_parts or [], highlight_nets or [],
                        show_labels, show_other_side, max_px)
    meta["centered_on"] = desc
    return [json.dumps(meta, ensure_ascii=False), Image(data=png, format="png")]


# --------------------------------------------------------------------------- #
def _store() -> MeasurementStore:
    _ix()
    return _STATE["store"]


@mcp.tool()
@guard
def add_measurement(target: str, kind: Literal["voltage", "resistance_to_gnd", "diode_to_gnd", "continuity",
                                               "current", "frequency", "note"],
                    value: str, unit: str | None = None,
                    state: Literal["unpowered", "adapter_only", "battery_only", "power_button_pressed",
                                   "running", "unknown"] = "unknown",
                    note: str = "") -> dict:
    """Registra uma medição do técnico. `target`: rede ('3D3V_S5'), pino ('U8.8') ou refdes. `state` = condição da
    placa na medição (resistência/diodo só valem 'unpowered'). `value` aceita 'OL' no modo diodo. Persistida por
    arquivo de placa."""
    item = _store().add(target, kind, value, unit, state, note)
    return {"saved": item, "total": len(_store().items)}


@mcp.tool()
@guard
def list_measurements() -> dict:
    """Lista as medições registradas para a placa ativa."""
    return {"measurements": _store().items}


@mcp.tool()
@guard
def check_measurements_vs_board(only_problems: bool = False) -> dict:
    """Cruza as medições com a topologia e o nome das redes (tensão esperada por estado de energia, curtos por
    resistência/diodo, alvos inexistentes). Devolve flags e severidade (info/warn/fault). Heurístico."""
    res = check_measurements(_ix(), _store().items)
    if only_problems:
        res = [r for r in res if r["severity"] != "info"]
    return {"results": res}


@mcp.tool()
@guard
def clear_measurements(ids: list[int] | None = None) -> dict:
    """Apaga medições por id (lista) ou TODAS se ids for omitido."""
    return {"removed": _store().remove(ids)}


@mcp.resource("boardview://skill")
def skill_resource() -> str:
    """O SKILL.md de uso deste servidor."""
    p = Path(__file__).resolve().parent.parent / "SKILL.md"
    return p.read_text(encoding="utf-8") if p.exists() else "SKILL.md não encontrado"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="boardview-mcp")
    ap.add_argument("--board", help="carrega este .cad ao iniciar (ou use BOARDVIEW_FILE)")
    ap.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    args = ap.parse_args()
    if args.board:
        os.environ["BOARDVIEW_FILE"] = args.board
        _load(args.board)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
