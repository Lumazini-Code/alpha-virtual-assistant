import asyncio
import glob
import json
import os
import pathlib
import platform
import re
import subprocess
import logging
import sys
import time
import uuid
from datetime import datetime
from typing import Optional
import httpx
import psutil
import pyfiglet
from rich.text import Text
from rich.markup import escape
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widget import Widget
from textual.widgets import Footer, Input, Static, RichLog, Markdown
import traceback
import requests
import tkinter.messagebox as messagebox
from Modules.local_scraping import status 


with open("ava_ui.log", "w") as f:
    f.write("")

# CORREÇÃO #6: arquivo dedicado para o log bruto do container Docker
# (stdout+stderr unificados). Usado por _read_docker_log() para que o
# usuário sempre tenha uma cópia completa e fácil de selecionar/copiar
# fora dos limites de renderização do widget RichLog da TUI.
with open("ava_docker.log", "w", encoding="utf-8") as f:
    f.write("")
_docker_log_file = open("ava_docker.log", "a", encoding="utf-8", buffering=1)

global chosen
API_BASE = "http://localhost:9001"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [UI] %(message)s",
    handlers=[
        logging.FileHandler("ava_ui.log"),
    ]
)
log = logging.getLogger("ava.ui")

# ─────────────────────────────────────────
# Redireciona stdout/stderr (nível Python) para o arquivo de log.
# IMPORTANTE: não usar os.dup2 aqui — isso reescreve o file
# descriptor 1/2 do processo inteiro e "rouba" o terminal antes do
# Textual conseguir desenhar a TUI (app.run() fica preso, tela não
# aparece). Trocar apenas sys.stdout/sys.stderr em nível Python já
# resolve prints/logging/requests vazando, sem afetar o terminal
# real que o Textual precisa para entrar em alternate screen.
# ─────────────────────────────────────────
sys.stdout = open("ava_ui.log", "a", buffering=1)
sys.stderr = open("ava_ui.log", "a", buffering=1)

# ─────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────
ORCHESTRATOR_URL = os.environ.get("AVA_ORCHESTRATOR_URL", "http://localhost:9000")
DEFAULT_VOICE    = "M1"
DEFAULT_LANG     = "pt"

# ─────────────────────────────────────────
# LOGO
# ─────────────────────────────────────────
LOGO_RAW    = pyfiglet.figlet_format("ALPHA  AI", font="banner3").rstrip("\n")
LOGO_LINES  = LOGO_RAW.split("\n")
LOGO_HEIGHT = len(LOGO_LINES)

PID_DIR = pathlib.Path(os.environ.get("LLAMA_PID_DIR", "/tmp/ava_llama_pids"))

# ─────────────────────────────────────────
# CORES
# ─────────────────────────────────────────
BLUE  = "#1243E4"
GRAY  = "#555555"
GREEN = "#4CAF7D"
AMBER = "#E4A012"
RED_C = "#E45012"
WHITE = "#CCCCCC"
DIM   = "#333333"


TAGLINE = "Any model. Every tool. Zero limits."

# ─────────────────────────────────────────
# NOTA: a remoção de sequências ANSI cruas do log do Docker agora é
# feita por Text.from_ansi() (ver DockerLog.write), que faz o parsing
# completo em vez de um regex parcial. Ver histórico da CORREÇÃO #4.
# ─────────────────────────────────────────


# ─────────────────────────────────────────
# GIT
# ─────────────────────────────────────────
def _git_info() -> tuple[str, str]:
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                    stderr=subprocess.DEVNULL).decode().strip()
        m = subprocess.check_output(["git", "log", "-1", "--format=%s"],
                                    stderr=subprocess.DEVNULL).decode().strip()
        return h, m
    except Exception:
        return "-------", "não é um repositório git"

GIT_HASH, GIT_MSG = _git_info()


def _safe_step(val):
    """Garante que o step seja sempre um int hashable."""
    if isinstance(val, dict):
        val = val.get("id") or val.get("step") or val.get("num") or 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0

def _safe_str(val):
    """Garante que o executor seja sempre uma str hashable."""
    if isinstance(val, dict):
        val = val.get("name") or val.get("executor") or str(val)
    return str(val) if not isinstance(val, str) else val


# ─────────────────────────────────────────
# PID / LLAMA
# ─────────────────────────────────────────
def _find_llama_pid() -> Optional[int]:
    if not PID_DIR.exists():
        return None
    for f in PID_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            pid  = int(data.get("pid", 0))
            if pid and psutil.pid_exists(pid):
                return pid
        except Exception:
            pass
    return None

# ─────────────────────────────────────────
# GPU — AMD (sysfs/rocm-smi) + NVIDIA (pynvml/nvidia-smi)
# ─────────────────────────────────────────
try:
    import pynvml as _nv
    _nv.nvmlInit()
    _NV_HANDLE  = _nv.nvmlDeviceGetHandleByIndex(0)
    _HAS_NVIDIA = True
except Exception:
    _HAS_NVIDIA = False

def _find_amd_sysfs() -> Optional[str]:
    for card in sorted(glob.glob("/sys/class/drm/card*/device")):
        driver = os.path.join(card, "driver")
        try:
            if "amdgpu" in os.readlink(driver):
                return card
        except OSError:
            pass
        if os.path.exists(os.path.join(card, "gpu_busy_percent")):
            return card
    return None

def _read_amd_sysfs(card_path: str) -> dict:
    def r(name):
        try: return int(open(os.path.join(card_path, name)).read().strip())
        except: return 0
    return dict(gpu_pct=float(r("gpu_busy_percent")),
                vram_used_mb=r("mem_info_vram_used")  / 1024**2,
                vram_tot_mb =r("mem_info_vram_total") / 1024**2,
                has_gpu=True, vendor="AMD")

def _read_nvidia() -> Optional[dict]:
    if _HAS_NVIDIA:
        try:
            util = _nv.nvmlDeviceGetUtilizationRates(_NV_HANDLE)
            mem  = _nv.nvmlDeviceGetMemoryInfo(_NV_HANDLE)
            return dict(gpu_pct=float(util.gpu),
                        vram_used_mb=mem.used  / 1024**2,
                        vram_tot_mb =mem.total / 1024**2,
                        has_gpu=True, vendor="NVIDIA")
        except Exception:
            pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi","--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=2).decode().strip()
        g, u, t = [float(x.strip()) for x in out.split(",")]
        return dict(gpu_pct=g, vram_used_mb=u, vram_tot_mb=t,
                    has_gpu=True, vendor="NVIDIA")
    except Exception:
        return None

def _read_amd_rocm() -> Optional[dict]:
    try:
        out  = subprocess.check_output(
            ["rocm-smi","--showuse","--showmeminfo","vram","--json"],
            stderr=subprocess.DEVNULL, timeout=2).decode()
        data = json.loads(out)
        card = next(iter(data))
        return dict(
            gpu_pct      = float(data[card].get("GPU use (%)", 0)),
            vram_used_mb = int(data[card].get("VRAM Total Used Memory (B)", 0)) / 1024**2,
            vram_tot_mb  = int(data[card].get("VRAM Total Memory (B)", 0))      / 1024**2,
            has_gpu=True, vendor="AMD")
    except Exception:
        return None

_AMD_SYSFS = _find_amd_sysfs()

def _detect_vendor() -> str:
    nv = _read_nvidia()
    if nv and nv["has_gpu"]:   return "nvidia"
    if _AMD_SYSFS:             return "amd"
    if _read_amd_rocm():       return "amd"
    return "none"

_GPU_VENDOR = _detect_vendor()

def read_gpu() -> dict:
    _empty = dict(gpu_pct=0.0, vram_used_mb=0.0, vram_tot_mb=0.0, has_gpu=False, vendor="N/A")
    if _GPU_VENDOR == "nvidia": return _read_nvidia() or _empty
    if _GPU_VENDOR == "amd":
        if _AMD_SYSFS: return _read_amd_sysfs(_AMD_SYSFS)
        return _read_amd_rocm() or _empty
    return _empty

# ─────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────
def _bar(pct: float, width: int = 8) -> str:
    return "█" * round(pct / 100 * width) + "░" * (width - round(pct / 100 * width))

def _color_pct(pct: float) -> str:
    if pct >= 85: return RED_C
    if pct >= 60: return AMBER
    return GREEN

def collect_metrics() -> dict:
    vm          = psutil.virtual_memory()
    sys_cpu_pct = psutil.cpu_percent(interval=None)
    llama_cpu   = 0.0
    llama_ram   = 0.0
    llama_pid   = _find_llama_pid()
    if llama_pid:
        try:
            lp = psutil.Process(llama_pid)
            llama_cpu = lp.cpu_percent(interval=None)
            llama_ram = lp.memory_info().rss / 1024**2
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            llama_pid = None
    g = read_gpu()
    return dict(
        total_cpu    = min(sys_cpu_pct + llama_cpu, 100.0),
        llama_cpu    = llama_cpu,
        sys_ram_pct  = vm.percent,
        sys_ram_gb   = vm.used  / 1024**3,
        sys_ram_tot  = vm.total / 1024**3,
        llama_ram_mb = llama_ram,
        llama_pid    = llama_pid,
        gpu_pct      = g["gpu_pct"],
        vram_used_mb = g["vram_used_mb"],
        vram_tot_mb  = g["vram_tot_mb"],
        vram_pct     = (g["vram_used_mb"] / g["vram_tot_mb"] * 100) if g["vram_tot_mb"] > 0 else 0.0,
        has_gpu      = g["has_gpu"],
    )

# ─────────────────────────────────────────
# RENDER: Markdown + LaTeX → Rich markup
# ─────────────────────────────────────────
def _render_rich(text: str) -> Text:
    result = Text()
    text = text.replace("\r\n", "\n")
    code_block_re = re.compile(r"```(\w*)\n?(.*?)```", re.DOTALL)
    parts = code_block_re.split(text)
    i = 0
    while i < len(parts):
        if i % 3 == 0:
            _append_markdown_lines(result, parts[i])
        elif i % 3 == 1:
            lang = parts[i] or "code"
            i += 1
            code_content = parts[i] if i < len(parts) else ""
            result.append(f"\n  [{lang}]\n", style=f"bold {BLUE}")
            for line in code_content.split("\n"):
                result.append(f"  {line}\n", style="bold #A8D8A8")
            result.append(f"  [/{lang}]\n", style=f"bold {BLUE}")
        i += 1
    return result


def _append_markdown_lines(result: Text, text: str):
    for line in text.split("\n"):
        stripped = line.rstrip()
        if re.match(r"^---+$", stripped):
            result.append("─" * 60 + "\n", style=DIM)
            continue
        h = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if h:
            level = len(h.group(1))
            sizes = {1: "bold #FFFFFF", 2: f"bold {WHITE}", 3: f"bold {BLUE}",
                     4: "bold", 5: "italic", 6: "italic dim"}
            result.append(h.group(2) + "\n", style=sizes.get(level, "bold"))
            continue
        if stripped.startswith("> "):
            result.append("│ ", style=BLUE)
            _append_inline(result, stripped[2:])
            result.append("\n")
            continue
        li = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)", stripped)
        if li:
            indent = "  " * (len(li.group(1)) // 2)
            bullet = "•" if not li.group(2)[0].isdigit() else li.group(2)
            result.append(f"{indent}{bullet} ", style=BLUE)
            _append_inline(result, li.group(3))
            result.append("\n")
            continue
        _append_inline(result, stripped)
        result.append("\n")


_INLINE_RE = re.compile(
    r"\$\$(.+?)\$\$"
    r"|\$([^$\n]+?)\$"
    r"|\*\*(.+?)\*\*"
    r"|__(.+?)__"
    r"|\*(.+?)\*"
    r"|_(.+?)_"
    r"|`([^`]+?)`"
    r"|\[([^\]]+)\]\(([^)]+)\)",
    re.DOTALL,
)

def _latex_to_plain(expr: str) -> str:
    expr = re.sub(r"\\text\{([^}]*)\}", r"\1", expr)
    expr = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1/\2)", expr)
    expr = re.sub(r"\\sqrt\{([^}]*)\}", r"√(\1)", expr)
    expr = expr.replace(r"\cdot", "·")
    expr = expr.replace(r"\times", "×")
    expr = expr.replace(r"\pm", "±")
    expr = expr.replace(r"\rightarrow", "→").replace(r"\to", "→")
    expr = expr.replace(r"\leftarrow", "←")
    expr = re.sub(r"\^\{([^}]*)\}", r"^\1", expr)
    expr = re.sub(r"\^(\w)", r"^\1", expr)
    expr = re.sub(r"_\{([^}]*)\}", r"_\1", expr)
    expr = re.sub(r"\\[a-zA-Z]+\*?", "", expr)
    expr = expr.replace("{", "").replace("}", "")
    return expr.strip()


def _append_inline(result: Text, text: str):
    last = 0
    for m in _INLINE_RE.finditer(text):
        if m.start() > last:
            result.append(text[last:m.start()], style=WHITE)
        latex_block  = m.group(1)
        latex_inline = m.group(2)
        bold1        = m.group(3)
        bold2        = m.group(4)
        italic1      = m.group(5)
        italic2      = m.group(6)
        code         = m.group(7)
        link_text    = m.group(8)
        link_url     = m.group(9)
        if latex_block:
            plain = _latex_to_plain(latex_block)
            result.append(f" {plain} ", style=f"italic {AMBER}")
        elif latex_inline:
            plain = _latex_to_plain(latex_inline)
            result.append(plain, style=f"italic {AMBER}")
        elif bold1 or bold2:
            result.append(bold1 or bold2, style="bold white")
        elif italic1 or italic2:
            result.append(italic1 or italic2, style="italic white")
        elif code:
            result.append(code, style="bold #A8D8A8")
        elif link_text and link_url:
            result.append(link_text, style=f"underline {BLUE}")
            result.append(f" ({link_url})", style=DIM)
        else:
            result.append(m.group(0), style=WHITE)
        last = m.end()
    if last < len(text):
        result.append(text[last:], style=WHITE)


# ─────────────────────────────────────────
# WIDGET: PIPELINE STEP (Tempo Real)
# ─────────────────────────────────────────
class PipelineStep(Widget):
    """Widget que mostra o progresso de um passo do CoT em tempo real."""

    DEFAULT_CSS = f"""
    PipelineStep {{
        height: 1;
        padding: 0 0 0 4;
        width: 100%;
        color: {AMBER};
    }}
    PipelineStep.-done {{ color: {GREEN}; }}
    PipelineStep.-error {{ color: {RED_C}; }}
    """

    def __init__(self, step_num: int, executor: str, action: str = "", **kwargs):
        super().__init__(**kwargs)
        self.step_num = step_num
        self.executor = executor

    def compose(self) -> ComposeResult:
        # Estado inicial: Aguardando
        yield Static(f"⏳ Step {self.step_num} [{self.executor}] aguardando...", id="step-text")

    def finish(self, success: bool, latency_ms: float, error: str = None):
        """Chamado quando o passo termina."""
        if not self.is_mounted:
            return
        icon = "✅" if success else "❌"
        err_str = f" — {error}" if error else ""
        try:
            self.query_one("#step-text", Static).update(
                f"{icon} Step {self.step_num} [{self.executor}] {latency_ms:.0f}ms{err_str}"
            )
        except Exception:
            return
        if success:
            self.set_class(True, "-done")
        else:
            self.set_class(True, "-error")



# ─────────────────────────────────────────
# WIDGET: CoT PLAN
# ─────────────────────────────────────────
class CoTPlanRow(Widget):
    """Uma linha do plano CoT: executor + action, atualiza com resultado."""

    DEFAULT_CSS = f"""
    CoTPlanRow {{
        height: auto;
        min-height: 2;
        padding: 0 0 0 6;
        border-bottom: solid #111111;
    }}
    .cpr-header {{ color: #888888; }}
    .cpr-action  {{ color: #BBBBBB;  padding: 0 0 0 2; }}
    CoTPlanRow.-pending .cpr-header {{ color: #777777; }}
    CoTPlanRow.-pending .cpr-action  {{ color: #666666; }}
    CoTPlanRow.-running .cpr-header {{ color: #E4A012; text-style: bold; }}
    CoTPlanRow.-running .cpr-action  {{ color: #FFD080; }}
    CoTPlanRow.-done    .cpr-header {{ color: #4CAF7D; }}
    CoTPlanRow.-done    .cpr-action  {{ color: #88BBAA; }}
    CoTPlanRow.-error   .cpr-header {{ color: #E45012; }}
    CoTPlanRow.-error   .cpr-action  {{ color: #FF8866; }}
    """

    def __init__(self, step_num: int, executor: str, action: str,
                 depends_on: list, **kwargs):
        super().__init__(**kwargs)
        self.step_num   = step_num
        self.executor   = executor
        self.action     = action
        self.depends_on = depends_on
        self.set_class(True, "-pending")

    def compose(self) -> ComposeResult:
        dep_str = f"  deps:{self.depends_on}" if self.depends_on else ""
        yield Static(
            f"\u25cb Step {self.step_num} [{self.executor}]{dep_str}",
            classes="cpr-header",
            markup=False,
        )
        # Sem truncamento: deixa o Static envolver (wrap) naturalmente.
        # Antes era self.action[:90] + "\u2026" — cortava descrições
        # comuns de 95-130 chars no meio, dando a impressão de que
        # o widget "não mostrava o que o step vai executar".
        yield Static(self.action, classes="cpr-action", markup=False)

    def set_running(self) -> None:
        # CORREÇÃO #1: blinda contra chamadas antes do widget estar
        # montado no DOM (ou depois de já ter sido removido). Sem
        # isso, query(...).first() levanta NoMatches e o Textual
        # imprime o traceback por cima da TUI no terminal.
        if not self.is_mounted:
            return
        dep_str = f"  deps:{self.depends_on}" if self.depends_on else ""
        try:
            self.query(".cpr-header", Static).first().update(
                f"\u23f3 Step {self.step_num} [{self.executor}]{dep_str}"
            )
        except Exception:
            return
        self.set_class(False, "-pending")
        self.set_class(True,  "-running")

    def set_done(self, success: bool, latency_ms: float, error: str = None) -> None:
        # CORREÇÃO: antes este método limpava a action (update("")) e
        # colapsava height=1. Isso apagava a descrição do step assim
        # que ele terminava — dando a impressão de que o widget "não
        # mostra se já foi concluído". Agora a action continua visível
        # (em verde/vermelho) e a altura é preservada.
        if not self.is_mounted:
            return
        icon    = "\u2713" if success else "\u2717"
        err_str = f"  \u2014 {error}" if error else ""
        dep_str = f"  deps:{self.depends_on}" if self.depends_on else ""
        try:
            self.query(".cpr-header", Static).first().update(
                f"{icon} Step {self.step_num} [{self.executor}] "
                f"{latency_ms:.0f}ms{err_str}{dep_str}",
                markup=False,
            )
            # Mantém a action visível — só reescreve para garantir refresh
            self.query(".cpr-action", Static).first().update(
                self.action, markup=False
            )
        except Exception:
            return
        # NÃO mexer em self.styles.height — preserva a altura
        self.set_class(False, "-running")
        self.set_class(False, "-pending")
        self.set_class(True,  "-done" if success else "-error")


class CoTPlan(Widget):
    """Container que mostra o plano CoT completo com status em tempo real."""

    DEFAULT_CSS = f"""
    CoTPlan {{
        height: auto;
        border-left: thick #E4A012;
        margin: 0 0 1 0;
        padding: 0;
        background: #0A0A0A;
    }}
    #cot-header {{
        height: 1;
        padding: 0 2;
        background: #111111;
        color: #E4A012;
        text-style: bold;
    }}
    """

    def __init__(self, steps: list, from_cache: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.steps      = steps
        self.from_cache = from_cache
        self._rows: dict[int, CoTPlanRow] = {}


    def compose(self) -> ComposeResult:
        cache_tag = "  [cache]" if self.from_cache else ""
        yield Static(
            f"🧠 CoT — {len(self.steps)} step(s){cache_tag}",
            id="cot-header",
        )
        for s in self.steps:
            # 👇 APLICA A DEFESA AQUI
            sn = _safe_step(s.get("step"))
            ex = _safe_str(s.get("executor"))
            
            row = CoTPlanRow(
                step_num   = sn,
                executor   = ex,
                action     = str(s.get("action", "")),
                depends_on = s.get("depends_on") or [],
            )
            self._rows[sn] = row  # 👇 Agora é 100% seguro usar como chave
            yield row

    def mark_running(self, step_num: int) -> None:
        if not self.is_mounted:
            return
        step_num = _safe_step(step_num) # 👇 Defesa aqui também
        if step_num in self._rows:
            self._rows[step_num].set_running()

    def mark_done(self, step_num: int, success: bool,
                  latency_ms: float, error: str = None) -> None:
        if not self.is_mounted:
            return
        step_num = _safe_step(step_num) # 👇 E aqui
        if step_num in self._rows:
            self._rows[step_num].set_done(success, latency_ms, error)

# ─────────────────────────────────────────
# WIDGET: DOCKER LOG PANEL
# ─────────────────────────────────────────
class DockerLog(Widget):
    """Widget do log do Docker com minimizar/expandir."""

    DEFAULT_CSS = f"""
    DockerLog {{
        width: 35;
        height: 1fr;
        background: #0A0A0A;
        border-left: tall #1E1E1E;
        padding: 0;
    }}
    #docker-header {{
        height: 1;
        background: #000000;
        border-bottom: tall #1E1E1E;
        padding: 0 1;
        layout: horizontal;
    }}
    #docker-title {{
        width: 1fr;
        height: 1;
        content-align: left middle;
        color: {BLUE};
        text-style: bold;
    }}
    #docker-toggle {{
        width: 3;
        height: 1;
        content-align: right middle;
        color: {GRAY};
    }}
    #docker-body {{
        height: 1fr;
        background: #0A0A0A;
    }}
    DockerLog.-collapsed {{
        width: 3;
    }}
    DockerLog.-collapsed #docker-title {{
        display: none;
    }}
    DockerLog.-collapsed #docker-body {{
        display: none;
    }}
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._collapsed = False
        self._toggle_icon = "▶"  # ▶ = colapsado (lateral), ◀ = expandido
        self._body: Optional[RichLog] = None

    def compose(self) -> ComposeResult:
        with Vertical(id="docker-header"):
            yield Static("● Docker", id="docker-title")
            yield Static(self._toggle_icon, id="docker-toggle")
        yield RichLog(id="docker-body", highlight=False, markup=False, max_lines=500)

    def on_mount(self) -> None:
        # CORREÇÃO #5: cacheia a referência do RichLog uma única vez.
        # Antes, cada linha de log disparava um query_one() novo no DOM
        # (custo desnecessário quando o container despeja dezenas de
        # linhas por segundo).
        self._body = self.query_one("#docker-body", RichLog)

    def on_click(self) -> None:
        """Toggle entre expandido e colapsado."""
        self._collapsed = not self._collapsed
        self.set_class(self._collapsed, "-collapsed")
        self._toggle_icon = "◀" if self._collapsed else "▶"
        self.query_one("#docker-toggle", Static).update(self._toggle_icon)

    def write(self, content) -> None:
        """Escreve uma linha no log.

        Aceita tanto `str` cru (saída do container, possivelmente com
        ANSI) quanto um `rich.text.Text` já estilizado (usado para
        mensagens de status internas, como "processo encerrado").
        """
        if not self.is_mounted or self._body is None:
            return
        # CORREÇÃO #4 (revisada): a versão anterior usava um regex que só
        # cobria sequências CSI simples (\x1b[...letra). Sequências que
        # ela NÃO pegava (OSC de título, save/restore de cursor, etc.,
        # comuns em saída de container Docker/build) passavam intactas
        # para o RichLog. Como esses bytes de escape sobreviviam até o
        # terminal real e eram interpretados por ELE, o texto "pulava"
        # para o meio da tela e sobrescrevia linhas já desenhadas
        # (o efeito de "partes do log somem").
        #
        # Text.from_ansi() faz o parsing correto: converte códigos de
        # cor/estilo (SGR) em spans de Style do Rich e descarta
        # sequências de controle não suportadas, sem nunca deixar um
        # \x1b cru dentro do texto final.
        try:
            if isinstance(content, Text):
                self._body.write(content)
            else:
                self._body.write(Text.from_ansi(str(content).rstrip("\n\r")))
        except Exception:
            return


# ─────────────────────────────────────────
# STREAMING AVA - SSE Parser
# ─────────────────────────────────────────
async def stream_ava(
    prompt: str,
    session_id: str,
    on_delta,
    on_meta,
    on_plan,
    on_step_start,
    on_step_done,
    on_result,
    on_error,
    on_done,
    on_reasoning=None,
):
    
    """
    POST /execute com stream=true e consome os SSE.
    """
    payload = {
        "input": prompt,
        "session_id": session_id,
        "voice": DEFAULT_VOICE,
        "lang": DEFAULT_LANG,
        "tts": False,
        "use_cache": True,
        "stream": True,
        "strategy": "parallel",
    }
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
        ) as client:
            async with client.stream(
                "POST",
                f"{ORCHESTRATOR_URL}/execute",
                json=payload,
            ) as resp:
                resp.raise_for_status()

                current_event = "message"
                current_data: list[str] = []

                async for line in resp.aiter_lines():
                    if not line:
                        if current_data:
                            raw = "\n".join(current_data)

                            if current_event in ("delta", "result", "reasoning"):
                                data = raw  # texto puro
                            else:
                                try:
                                    data = json.loads(raw)
                                except json.JSONDecodeError:
                                    data = raw

                            if current_event == "meta":
                                await on_meta(data)
                            elif current_event == "delta":
                                await on_delta(data)
                            elif current_event == "reasoning":
                                if on_reasoning:
                                    await on_reasoning(data)
                            elif current_event == "plan":
                                await on_plan(data)
                            elif current_event == "step_start":
                                await on_step_start(data)
                            elif current_event == "step_done":
                                await on_step_done(data)
                            elif current_event == "result":
                                await on_result(data)
                            elif current_event == "error":
                                err = data.get("error", str(data)) if isinstance(data, dict) else str(data)
                                await on_error(err)
                            elif current_event == "done":
                                await on_done(data)

                        current_event = "message"
                        current_data = []

                    elif line.startswith("event:"):
                        current_event = line[6:].strip()

                    elif line.startswith("data:"):
                        content = line[5:]
                        if content.startswith(" "):
                            content = content[1:]
                        current_data.append(content)

    except httpx.ConnectError:
        await on_error(f"Orchestrator offline — {ORCHESTRATOR_URL}")
    except httpx.HTTPStatusError as e:
        await on_error(f"HTTP {e.response.status_code} — {e.response.text[:200]}")
    except Exception as e:
        await on_error(f"Erro: {e}")


# ─────────────────────────────────────────
# WIDGET: METRICS BAR
# ─────────────────────────────────────────
class MetricsBar(Widget):

    DEFAULT_CSS = """
    MetricsBar {
        height: 3;
        background: #0A0A0A;
        border-top: tall #1E1E1E;
        padding: 0 4;
        layout: horizontal;
        align: center middle;
    }
    .mcell { width: 1fr; content-align: center middle; height: 3; }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="m-cpu",  classes="mcell")
        yield Static("", id="m-ram",  classes="mcell")
        yield Static("", id="m-gpu",  classes="mcell")
        yield Static("", id="m-vram", classes="mcell")

    def on_mount(self) -> None:
        psutil.cpu_percent(interval=None)
        self._tick()
        self.set_interval(2, self._tick)

    def _tick(self) -> None:
        if not self.is_mounted:
            return
        m = collect_metrics()

        lc = (f" [dim]+{m['llama_cpu']:.1f}%[/]"    if m["llama_pid"] else "")
        lr = (f" [dim]+{m['llama_ram_mb']:.0f}MB[/]" if m["llama_pid"] else "")

        cc = _color_pct(m["total_cpu"])
        rc = _color_pct(m["sys_ram_pct"])
        gc = _color_pct(m["gpu_pct"])
        vc = _color_pct(m["vram_pct"])

        gpu_str = (
            f"[{GRAY}]GPU [/][{gc}]{_bar(m['gpu_pct'])}[/] [{gc}]{m['gpu_pct']:.1f}%[/]"
            if m["has_gpu"] else f"[{GRAY}]GPU ——[/]"
        )
        vram_str = (
            f"[{GRAY}]VRAM[/][{vc}]{_bar(m['vram_pct'])}[/] "
            f"[{vc}]{m['vram_used_mb']/1024:.1f}[/][dim]/{m['vram_tot_mb']/1024:.1f}GB[/]"
            if m["has_gpu"] else f"[{GRAY}]VRAM ——[/]"
        )

        try:
            self.query_one("#m-cpu").update(
                f"[{GRAY}]CPU [/][{cc}]{_bar(m['total_cpu'])}[/] [{cc}]{m['total_cpu']:.1f}%[/]{lc}")
            self.query_one("#m-ram").update(
                f"[{GRAY}]RAM [/][{rc}]{_bar(m['sys_ram_pct'])}[/] "
                f"[{rc}]{m['sys_ram_gb']:.1f}[/][dim]/{m['sys_ram_tot']:.1f}GB[/]{lr}")
            self.query_one("#m-gpu").update(gpu_str)
            self.query_one("#m-vram").update(vram_str)
        except Exception:
            return


# ─────────────────────────────────────────
# WIDGET: SPLASH HEADER
# ─────────────────────────────────────────
model = os.environ.get("AVA_MODEL")
class SplashHeader(Widget):

    DEFAULT_CSS = f"""
    SplashHeader {{
        height: {LOGO_HEIGHT + 11};
        background: #0D0D0D;
        padding: 2 4;
    }}
    #logo    {{ color: {BLUE}; text-style: bold; height: {LOGO_HEIGHT}; }}
    #tagline {{ color: {GRAY}; padding: 0 0 1 0; }}
    #info-panel {{
        height: 7; background: #111111;
        border: tall #1E1E1E; padding: 0 2;
    }}
    #status-row  {{ height: 1; padding: 1 0 0 0; }}
    #status-left {{ width: 1fr; color: #AAAAAA; }}
    """

    def compose(self) -> ComposeResult:
        yield Static(LOGO_RAW, id="logo")
        yield Static(TAGLINE,  id="tagline")
        with Vertical(id="info-panel"):
            yield Static(f"[{GRAY}]OS      [/]  [{BLUE}]{platform.system()} {platform.release()}[/]")
            yield Static(f"[{GRAY}]Model   [/]  [{WHITE}]{model}[/]")
            yield Static(f"[{GRAY}]Commit  [/]  [{BLUE}]{GIT_HASH}[/]  [{GRAY}]{GIT_MSG}[/]")
        with Horizontal(id="status-row"):
            yield Static(
                f"[green]●[/green]  local    Ready — type [bold {BLUE}]/help[/] to begin",
                id="status-left",
            )


# ─────────────────────────────────────────
# WIDGET: CHAT MESSAGE (Markdown + LaTeX)
# ─────────────────────────────────────────
class ChatMessage(Widget):

    DEFAULT_CSS = f"""
    ChatMessage {{
        height: auto;
        padding: 0 0 1 0;
    }}
    #msg-header {{ height: 1; padding: 0 0 0 2; }}
    #thinking-box {{ margin: 0 2 0 2; }}
    #msg-body   {{ height: auto; padding: 0 0 0 4; }}
    """

    def __init__(self, role: str, content: str = "", msg_id: str = "", **kwargs):
        super().__init__(**kwargs)
        self.role    = role
        self.content = content
        self.msg_id  = msg_id or str(uuid.uuid4())[:8]
        # ── Throttle de render ───────────────────────────────────
        # Evita re-render O(n²): só reprocessa o Markdown/LaTeX no
        # máximo a cada 80 ms durante o streaming, e força um flush
        # final quando o stream termina (on_done / on_result).
        self._last_render       = 0.0
        self._dirty             = False
        self._render_scheduled  = False

    def compose(self) -> ComposeResult:
        role = getattr(self, "role", "assistant")
        
        if role == "user":
            header = f"[bold {BLUE}]▶ Você[/]"
        elif role == "system":
            header = f"[dim {GRAY}]● Sistema[/]"
        elif role == "step":
            header = f"[dim {AMBER}]⚙ Pipeline[/]"
        else:
            header = f"[bold {GREEN}]◆ ALPHA AI[/]"

        yield Static(header, id="msg-header")

        # ← NOVO: ThinkingBox para mensagens do assistant
        if role == "assistant":
            yield ThinkingBox(id="thinking-box")

        if role in ("step", "system"):
            yield Static(self.content, id="msg-body", markup=True)
        else:
            yield Static(_render_rich(self.content), id="msg-body")

    def append_delta(self, delta: str):
        """
        Adiciona um chunk ao conteúdo e re-renderiza com throttle.

        Antes: cada delta reprocessava TODO o conteúdo acumulado via
        _render_rich (regex com DOTALL sobre a string inteira), gerando
        um padrão O(n²). A partir de ~300-500 linhas o tempo de cada
        render ultrapassava o intervalo entre deltas, a fila de
        call_after_refresh crescia sem parar e a UI aparentava
        'congelar' — mesmo o stream ainda chegando pelo httpx.

        Agora: só re-renderiza no máximo a cada 80 ms. Um timer cuida
        do flush do último chunk pendente. Render final forçado via
        force_flush() no on_done / on_result.
        """
        if not self.is_mounted:
            return
        self.content += delta
        self._dirty  = True
        now = time.monotonic()
        elapsed = now - self._last_render
        if elapsed < 0.08:  # 80 ms throttle
            if not self._render_scheduled:
                self._render_scheduled = True
                self.set_timer(0.08 - elapsed, self._flush_render)
            return
        self._flush_render()

    def _flush_render(self):
        """Executa o render pendente (chamado direto ou via set_timer)."""
        self._render_scheduled = False
        if not self._dirty or not self.is_mounted:
            return
        self._dirty       = False
        self._last_render = time.monotonic()
        try:
            self.query_one("#msg-body", Static).update(_render_rich(self.content))
        except Exception:
            return

    def force_flush(self):
        """Força um render completo — chamar no on_done / on_result."""
        self._render_scheduled = False
        self._dirty            = True
        self._flush_render()
        # Força o Textual a recalcular o layout e altura do widget
        if self.is_mounted:
            self.refresh(layout=True)

    def append_reasoning(self, reasoning: str) -> None:
        """Adiciona reasoning ao ThinkingBox."""
        if not self.is_mounted:
            return
        try:
            box = self.query_one("#thinking-box", ThinkingBox)
            box.append_reasoning(reasoning)
        except Exception:
            pass  # ThinkingBox só existe para role="assistant"

    def finish_thinking(self) -> None:
        """Finaliza o thinking e auto-colapsa."""
        if not self.is_mounted:
            return
        try:
            box = self.query_one("#thinking-box", ThinkingBox)
            box.finish_thinking()
        except Exception:
            pass

# ─────────────────────────────────────────
# WIDGET: THINKING BOX (Reasoning)
# ─────────────────────────────────────────
class ThinkingBox(Widget):
    """Caixa colapsável que mostra o processo de raciocínio do LLM."""

    DEFAULT_CSS = f"""
    ThinkingBox {{
        height: auto;
        max-height: 10;
        margin: 0 2 0 4;
        padding: 0;
        background: #080808;
        border-left: tall {AMBER};
    }}
    ThinkingBox.-hidden {{
        display: none;
        height: 0;
        max-height: 0;
        margin: 0;
    }}
    #think-header {{
        height: 1;
        layout: horizontal;
        width: 1fr;
        padding: 0 1;
        background: #0A0A0A;
    }}
    #think-label {{
        width: 1fr;
        height: 1;
        color: {AMBER};
        text-style: italic dim;
        content-align: left middle;
    }}
    #think-toggle {{
        width: 3;
        height: 1;
        color: {GRAY};
        content-align: right middle;
    }}
    #think-content {{
        height: auto;
        max-height: 8;
        overflow-y: auto;
    }}
    ThinkingBox.-collapsed #think-content {{
        height: 0;
        max-height: 0;
        display: none;
    }}
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._collapsed = False
        self._char_count = 0

    def compose(self) -> ComposeResult:
        with Horizontal(id="think-header"):
            yield Static("💭 thinking...", id="think-label")
            yield Static("▼", id="think-toggle")
        yield RichLog(id="think-content", highlight=False, markup=False, max_lines=100)

    def on_mount(self) -> None:
        self.set_class(True, "-hidden")

    def on_click(self) -> None:
        """Toggle collapse — clique no header ou na caixa."""
        self._collapsed = not self._collapsed
        self.set_class(self._collapsed, "-collapsed")
        self.query_one("#think-toggle", Static).update("▶" if self._collapsed else "▼")

    def append_reasoning(self, text: str) -> None:
        """Adiciona texto de reasoning e torna a caixa visível."""
        if not self.is_mounted:
            return
        # Primeira vez: remove classe hidden
        if self.has_class("-hidden"):
            self.remove_class("-hidden")

        self._char_count += len(text)

        try:
            # Atualiza label com contagem
            self.query_one("#think-label", Static).update(
                f"💭 thinking... ({self._char_count} chars)"
            )

            # Escreve no log (scroll automático)
            self.query_one("#think-content", RichLog).write(text)
        except Exception:
            return

    def finish_thinking(self) -> None:
        """Chamado quando o reasoning termina — auto-colapsa."""
        if self._char_count == 0:
            return
        if not self.is_mounted:
            return

        try:
            # Atualiza label final
            self.query_one("#think-label", Static).update(
                f"💭 thought {self._char_count} chars"
            )
            self.query_one("#think-toggle", Static).update("▶")
        except Exception:
            return

        # Auto-colapsa após terminar
        self._collapsed = True
        self.set_class(True, "-collapsed")
        


# ─────────────────────────────────────────
# APP
# ─────────────────────────────────────────
class AlphaAI(App):

    CSS = f"""
    Screen {{
        layout: vertical;
        background: #0D0D0D;
    }}
    
    /* Header ocupa largura total, altura fixa */
    SplashHeader {{
        width: 100%;
    }}
    
    /* Área principal: Chat + Docker Log lado a lado */
    #main-area {{
        height: 1fr;
        layout: horizontal;
    }}
    
    /* Chat ocupa o espaço restante */
    #chat-wrap {{
        width: 1fr;
        height: 1fr;
        padding: 0 2;
    }}
    
    #chat-scroll {{
        height: 1fr;
        background: #0D0D0D;
        border-top: tall #1E1E1E;
        padding: 0 2;
    }}
    
    ChatMessage {{
        border-bottom: solid #151515;
        margin: 0;
    }}
    
    #route-bar {{
        height: 1;
        background: #0A0A0A;
        padding: 0 4;
        color: {GRAY};
    }}
    
    /* Input area */
    #input-wrap {{
        height: auto;
        padding: 1 4 1 4;
        background: #0D0D0D;
        border-top: tall #1E1E1E;
    }}
    
    Input {{
        background: #111111;
        border: tall #1E1E1E;
        color: {WHITE};
        padding: 0 2;
    }}
    Input:focus {{ border: tall {BLUE}; }}
    
    Footer {{
        background: #0D0D0D;
        color: #444444;
        border-top: tall #1A1A1A;
    }}
    """

    BINDINGS = [
        ("ctrl+q", "quit",        "Sair"),
        ("ctrl+l", "clear_chat",  "Limpar"),
        ("ctrl+n", "new_session", "Nova sessão"),
        ("ctrl+d", "toggle_docker", "Docker Log"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.session_id     = str(uuid.uuid4())
        self._streaming_msg : Optional[ChatMessage] = None
        self._is_streaming  = False
        self._proc_docker: Optional[subprocess.Popen] = None

    def compose(self) -> ComposeResult:
        # Header no topo (largura total)
        yield SplashHeader()
        
        # Área principal: Chat (esquerda) + Docker Log (direita)
        with Horizontal(id="main-area"):
            with Vertical(id="chat-wrap"):
                yield Static("", id="route-bar")
                yield ScrollableContainer(id="chat-scroll")
            yield DockerLog()
        
        # Input area
        with Vertical(id="input-wrap"):
            yield Input(
                placeholder="> Digite seu prompt e pressione Enter...",
                id="prompt-input",
            )
        
        # Metrics and footer
        yield MetricsBar()
        yield Footer()

    def action_toggle_docker(self) -> None:
        """Toggle do Docker Log via atalho Ctrl+D."""
        docker_log = self.query_one(DockerLog)
        docker_log.on_click()
    

    def on_mount(self) -> None:
        psutil.cpu_percent(interval=None)
        self.call_after_refresh(self._focus)
        # Inicia leitura do log do Docker se o processo existir
        if self._proc_docker is not None:
            asyncio.create_task(self._read_docker_log())

    async def _read_docker_log(self) -> None:
        """
        Lê stdout do proc_docker linha a linha em background
        e exibe no painel DockerLog sem bloquear o event loop.
        """
        proc = self._proc_docker
        if proc is None or proc.stdout is None:
            log.warning("DockerLog: proc_docker ou stdout é None")
            return

        # CORREÇÃO #3: garante que o widget DockerLog já está no DOM
        # antes de começar a escrever nele. Se a task disparar antes
        # do compose() inicial da tela terminar, query_one levanta
        # NoMatches e o traceback aparece por cima da TUI.
        try:
            docker_log = self.query_one(DockerLog)
        except Exception:
            log.warning("DockerLog ainda não montado, abortando leitura")
            return

        # CORREÇÃO #6: além de mostrar no widget (que não suporta
        # seleção de texto nativa do Textual — só a do terminal, com
        # tecla modificadora), grava cada linha também em
        # ava_docker.log. Isso dá um jeito confiável de copiar o log
        # inteiro: basta abrir o arquivo em qualquer editor de texto
        # e selecionar tudo. Sem isso, o conteúdo do container fica
        # preso só dentro do widget da TUI.
        global _docker_log_file
        loop = asyncio.get_event_loop()

        try:
            while True:
                line = await loop.run_in_executor(None, proc.stdout.readline)
                if not line:
                    if self.is_running:
                        docker_log.write(Text("--- processo encerrado ---", style="dim"))
                    break
                # ✅ CORRIGIDO: line já é string (text=True), não precisa de .decode()
                try:
                    _docker_log_file.write(line if line.endswith("\n") else line + "\n")
                except Exception:
                    pass
                if self.is_running:
                    docker_log.write(line)
        except Exception as e:
            log.error(f"Erro na leitura do Docker log: {e}")
            if self.is_running:
                try:
                    docker_log.write(Text(f"Erro: {e}", style="red"))
                except Exception:
                    pass

    def _focus(self) -> None:
        try:
            self.query_one("#prompt-input", Input).focus()
        except Exception:
            return

    def _add_message(self, role: str, content: str) -> ChatMessage:
        scroll = self.query_one("#chat-scroll", ScrollableContainer)
        msg = ChatMessage(role=role, content=content)
        scroll.mount(msg)
        self.call_after_refresh(lambda: scroll.scroll_end(animate=False))
        return msg

    def _set_route_bar(self, route: str, conf: float, method: str, direct: bool):
        via = "direto" if direct else "CoT"
        color = BLUE if direct else AMBER
        try:
            self.query_one("#route-bar").update(
                f"  [{color}]▸ {route}[/] [{GRAY}]{conf:.0%} via {method} • {via}[/]"
            )
        except Exception:
            return

    def action_clear_chat(self) -> None:
        scroll = self.query_one("#chat-scroll", ScrollableContainer)
        scroll.remove_children()
        self._add_message("system", "Chat limpo.")

    def action_new_session(self) -> None:
        self.session_id = str(uuid.uuid4())
        scroll = self.query_one("#chat-scroll", ScrollableContainer)
        scroll.remove_children()
        self._add_message("system", f"Nova sessão iniciada: `{self.session_id[:8]}`")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "prompt-input" or self._is_streaming:
            return
        if chosen:
            await start_processes(chosen)
            
        prompt = event.value.strip()
        if not prompt:
            return
        event.input.value    = ""
        event.input.disabled = True
        self._is_streaming   = True
        self._add_message("user", prompt)
        ai_msg = self._add_message("assistant", "")
        self._streaming_msg = ai_msg
        asyncio.create_task(self._run_stream(prompt, ai_msg))

    async def _run_stream(self, prompt: str, ai_msg: ChatMessage) -> None:
        scroll = self.query_one("#chat-scroll", ScrollableContainer)
        
        async def on_delta(delta: str):
            # CORREÇÃO: NÃO chamar finish_thinking() aqui — era redundante
            # (chamado a cada chunk) e somava trabalho ao render O(n²).
            # finish_thinking() é delegado ao on_done / on_result.
            try:
                ai_msg.append_delta(delta)
                
                self.call_after_refresh(lambda: scroll.scroll_end(animate=False))
            except Exception as e:
                log.error(f"Erro em on_delta: {e}\n{traceback.format_exc()}")

        async def on_reasoning(reasoning: str):
            try:
                ai_msg.append_reasoning(reasoning)
                self.call_after_refresh(lambda: scroll.scroll_end(animate=False))
            except Exception as e:
                log.error(f"Erro em on_reasoning: {e}\n{traceback.format_exc()}")

        async def on_meta(event: dict):
            try:
                self._set_route_bar(
                    event.get("route", "?"),
                    event.get("route_confidence", 0.0),
                    event.get("route_method", "?"),
                    event.get("routed_directly", False),
                )
            except Exception as e:
                log.error(f"Erro em on_meta: {e}\n{traceback.format_exc()}")

        _cot_plan_widget: Optional[CoTPlan] = None

        async def on_plan(event):
            nonlocal _cot_plan_widget
            try:
                if not isinstance(event, dict):
                    log.warning(f"Evento 'plan' recebido não é dict: {type(event)}")
                    return
                
                steps = event.get("steps", [])
                if not isinstance(steps, list):
                    steps = []
                    
                _cot_plan_widget = CoTPlan(
                    steps      = steps,
                    from_cache = event.get("from_cache", False),
                )
                # CORREÇÃO #2: await no mount garante que o widget e
                # seus filhos (CoTPlanRow) já estão de fato no DOM
                # quando on_plan retorna — eliminando a corrida com
                # on_step_start/on_step_done que chegam logo em
                # seguida via SSE.
                await scroll.mount(_cot_plan_widget)
                self.call_after_refresh(lambda: scroll.scroll_end(animate=False))
                log.info(f"Plano CoT recebido com {len(steps)} step(s).")
            except Exception as e:
                log.error(f"Erro em on_plan: {e}\n{traceback.format_exc()}")

        async def on_step_start(event):
            try:
                if not isinstance(event, dict): return
                step_num = _safe_step(event.get("step", 0))
                if _cot_plan_widget:
                    _cot_plan_widget.mark_running(step_num)
                self.call_after_refresh(lambda: scroll.scroll_end(animate=False))
            except Exception as e:
                log.error(f"Erro em on_step_start: {e}\n{traceback.format_exc()}")

        async def on_step_done(event):
            try:
                if not isinstance(event, dict): return
                step_num = _safe_step(event.get("step", 0))
                success  = event.get("success", False)
                latency  = event.get("latency_ms", 0.0)
                error    = event.get("error")
                if _cot_plan_widget:
                    _cot_plan_widget.mark_done(step_num, success, latency, error)
                self.call_after_refresh(lambda: scroll.scroll_end(animate=False))
                log.debug(f"Step {step_num} concluído: success={success}")
            except Exception as e:
                log.error(f"Erro em on_step_done: {e}\n{traceback.format_exc()}")

        async def on_result(text: str):
            try:
                ai_msg.finish_thinking()
                ai_msg.append_delta(text)
                ai_msg.force_flush()  # render final completo
                self.call_after_refresh(lambda: scroll.scroll_end(animate=False))
            except Exception as e:
                log.error(f"Erro em on_result: {e}\n{traceback.format_exc()}")

        async def on_error(error: str):
            try:
                ai_msg.finish_thinking()
                self._add_message("system", f"**Erro:** {error}")
                log.error(f"Erro recebido do Orchestrator: {error}")
            except Exception as e:
                log.error(f"Erro em on_error: {e}\n{traceback.format_exc()}")

        async def on_done(event: dict):
            try:
                ai_msg.finish_thinking()
                ai_msg.force_flush()  # garante texto 100% renderizado
                lat    = event.get("total_latency_ms", 0)
                errs   = event.get("errors", [])
                route  = event.get("route", "?")
                conf   = event.get("route_confidence", 0.0)
                meth   = event.get("route_method", "?")
                direct = event.get("routed_directly", False)
                via    = "direto" if direct else "CoT"
                color  = BLUE if direct else AMBER
                self.query_one("#route-bar").update(
                    f"  [{color}]▸ {route}[/] [{GRAY}]{conf:.0%} via {meth}"
                    f" • {via} • {lat:.0f}ms[/]"
                    + (f"  [{RED_C}]{len(errs)} erro(s)[/]" if errs else "")
                )
                log.info("Execução finalizada (evento 'done' recebido).")
            except Exception as e:
                log.error(f"Erro em on_done: {e}\n{traceback.format_exc()}")

        try:
            await stream_ava(
                prompt, self.session_id,
                on_delta, on_meta, on_plan, on_step_start, on_step_done,
                on_result, on_error, on_done,
                on_reasoning=on_reasoning,
            )
        except Exception as e:
            log.error(f"Erro fatal no stream_ava: {e}\n{traceback.format_exc()}")

        self._is_streaming  = False
        self._streaming_msg = None
        inp = self.query_one("#prompt-input", Input)
        inp.disabled = False
        self.call_after_refresh(self._focus)


async def start_processes(info):
    try:
        async with httpx.AsyncClient() as client:
            # Inicia o processo
            resp = await client.post(
                f"{API_BASE}/llama/start",
                json={
                    "model": info["model"],
                    "mmproj_used": bool(info.get("mmproj"))
                },
                timeout=15.0
            )
            resp.raise_for_status()

            # Espera até o processo estar pronto

        while True:
            try:
                r = requests.get(f"http://localhost:2001/health")
            except requests.RequestException as e:
                log.error(f"Erro ao verificar status do ava_tray: {e}")
                await asyncio.sleep(0.5)
                continue

            status = r.json().get("status")
            if status == "ok":
                break   

            await asyncio.sleep(0.5)

    except httpx.ConnectError:
        err_msg = "Falha ao conectar ao ava_tray."
        log.error(err_msg)
        messagebox.showerror("Erro de Conexão", err_msg)
        raise SystemExit(1)

    except Exception as e:
        err_msg = f"Falha ao iniciar o llama-server: {e}"
        log.error(err_msg)
        messagebox.showerror("Erro na API", err_msg)
        raise SystemExit(1)
    # ════════════════════════════════════════════════════════════════════
    # 2. Inicia o Docker via API
    # ════════════════════════════════════════════════════════════════════
    try:
        resp = requests.post(
            f"{API_BASE}/docker/start",
            timeout=30.0 
        )
        resp.raise_for_status()
        data = resp.json()
        log.info(f"Docker: {data.get('message')}")
    except requests.exceptions.RequestException as e:
        # Apenas avisa no log, não encerra o programa por causa do Docker
        log.warning(f"Aviso ao contatar a API do Docker: {e}")
        
# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────


log = logging.getLogger(__name__)


if __name__ == "__main__":
    from model_selector import run_model_selector

    chosen = run_model_selector()

    if chosen is None:
        raise SystemExit(0)

    mmproj_flag = "true" if chosen["mmproj"] else "false"
    log.info(f"Modelo escolhido: {chosen['model']} (mmproj={mmproj_flag})")
    
    # Inicia os processos necessários (Llama e Docker)
    asyncio.run(start_processes(chosen))

    # ════════════════════════════════════════════════════════════════════
    # 3. "Espia" os logs do container Docker
    # ════════════════════════════════════════════════════════════════════
    proc_docker = None
    container_name = "ava-vulkan"

    def _wait_for_container(timeout: float = 30.0, interval: float = 1.0) -> bool:
        """
        Espera o container aparecer no `docker ps` antes de tentar ler
        os logs dele.

        CORREÇÃO: antes, se o container ainda não tivesse subido no
        momento em que a UI chamava _try_docker_logs() (ex: o endpoint
        /docker/start já respondeu 200, mas o container ainda está
        sendo criado), a leitura de logs falhava uma única vez com
        "no such container" e NUNCA mais era tentada — o painel de
        logs ficava vazio pelo resto da sessão. Agora esperamos até
        `timeout` segundos, checando a cada `interval`.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                result = subprocess.run(
                    ["docker", "ps", "-q", "-f", f"name={container_name}"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.stdout.strip():
                    return True
            except Exception as e:
                log.warning(f"Erro ao checar se o container subiu: {e}")
            time.sleep(interval)
        return False

    def _try_docker_logs() -> Optional[subprocess.Popen]:
        """Tenta docker logs com o ambiente atual do Python."""
        # Debug: mostra qual docker o Python está encontrando
        try:
            which = subprocess.run(
                ["which", "docker"], capture_output=True, text=True, timeout=5
            )
            log.info(f"Python vê docker em: {which.stdout.strip()}")
        except Exception:
            pass

        # Debug: mostra o context
        try:
            ctx = subprocess.run(
                ["docker", "context", "ls"],
                capture_output=True, text=True, timeout=5
            )
            log.info(f"Docker contexts:\n{ctx.stdout}")
            if ctx.stderr:
                log.warning(f"Docker context stderr: {ctx.stderr}")
        except Exception as e:
            log.warning(f"Não consegui listar docker context: {e}")

        # CORREÇÃO: espera o container existir antes de tentar ler
        # logs, em vez de tentar uma única vez e desistir de vez.
        log.info(f"Aguardando container '{container_name}' subir...")
        if not _wait_for_container():
            log.error(f"Container '{container_name}' não apareceu em 30s.")
            return None

        # Tenta o docker logs
        try:
            proc = subprocess.Popen(
                ["docker", "logs", "-f", container_name],
                stdout=subprocess.PIPE,
                # CORREÇÃO: stderr=STDOUT em vez de PIPE separado.
                # A maior parte da saída dos serviços (logging/uvicorn
                # em Python vai para stderr por padrão) — com um pipe
                # separado que só era drenado para o arquivo de log
                # (nunca para o widget da UI), o painel de logs
                # mostrava só uma fração mínima do que `docker logs`
                # realmente exibe. Unificando os streams,
                # _read_docker_log() (que já lê stdout) passa a
                # receber TUDO.
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            return proc

        except FileNotFoundError:
            log.error("Comando 'docker' não encontrado no PATH do Python")
            return None
        except Exception as e:
            log.error(f"Erro ao iniciar docker logs: {e}")
            return None

    def _try_tail_log_file() -> Optional[subprocess.Popen]:
        """Fallback: encontra o arquivo de log do container e usa tail."""
        try:
            # Pega o container ID
            result = subprocess.run(
                ["docker", "ps", "-q", "-f", f"name={container_name}"],
                capture_output=True, text=True, timeout=5
            )
            container_id = result.stdout.strip()
            
            if not container_id:
                log.warning(f"Não encontrou ID do container '{container_name}'")
                return None
                
            log.info(f"Container ID: {container_id}")
            
            # Tenta encontrar o arquivo de log
            # Docker Desktop (Mac/Windows) ou Linux
            possible_paths = [
                f"/var/lib/docker/containers/{container_id}/{container_id}-json.log",
            ]
            
            # No Docker Desktop, tenta via docker inspect
            try:
                inspect = subprocess.run(
                    ["docker", "inspect", container_id],
                    capture_output=True, text=True, timeout=5
                )
                import json
                data = json.loads(inspect.stdout)
                log_path = data[0].get("LogPath", "")
                if log_path:
                    possible_paths.insert(0, log_path)
                    log.info(f"LogPath do inspect: {log_path}")
            except Exception as e:
                log.warning(f"Não consegui fazer docker inspect: {e}")

            for path in possible_paths:
                if os.path.exists(path):
                    log.info(f"Usando tail -f {path}")
                    return subprocess.Popen(
                        ["tail", "-f", "-n", "50", path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
            
            log.warning(f"Nenhum arquivo de log encontrado em: {possible_paths}")
            return None
            
        except Exception as e:
            log.error(f"Erro no fallback tail: {e}")
            return None

    # Tenta na ordem: docker logs → tail do arquivo de log
    proc_docker = _try_docker_logs()
    
    if proc_docker is None:
        log.info("Tentando fallback com tail do arquivo de log...")
        proc_docker = _try_tail_log_file()

    if proc_docker is None:
        log.warning(
            "Nenhum método de leitura de logs funcionou. "
            "O widget de logs ficará vazio. "
            "Verifique o arquivo ava_ui.log para detalhes."
        )

    # ════════════════════════════════════════════════════════════════════
    # 4. Inicia a interface principal
    # ════════════════════════════════════════════════════════════════════
    os.environ["AVA_MODEL"] = pathlib.Path(chosen["model"]).stem
    
    app = AlphaAI()
    
    # ✅ CORRIGIDO: Injeta o processo de tail para a UI poder ler os logs
    app._proc_docker = proc_docker 
    
    app.run()