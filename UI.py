import asyncio
import glob
import json
import os
import pathlib
import platform
import re
import subprocess
import logging
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [UI] %(message)s",
    handlers=[
        logging.FileHandler("ava_ui.log"),
    ]
)
log = logging.getLogger("ava.ui")

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

MODEL   = os.environ.get("AVA_MODEL", "")
TAGLINE = "Any model. Every tool. Zero limits."

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
        icon = "✅" if success else "❌"
        err_str = f" — {error}" if error else ""
        self.query_one("#step-text", Static).update(
            f"{icon} Step {self.step_num} [{self.executor}] {latency_ms:.0f}ms{err_str}"
        )
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
        height: 2;
        padding: 0 0 0 6;
        border-bottom: solid #111111;
    }}
    .cpr-header {{ color: #555555; }}
    .cpr-action  {{ color: #333333;  padding: 0 0 0 2; }}
    CoTPlanRow.-running  .cpr-header {{ color: #E4A012; }}
    CoTPlanRow.-done     .cpr-header {{ color: #4CAF7D; }}
    CoTPlanRow.-error    .cpr-header {{ color: #E45012; }}
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
            f"○ Step {self.step_num} [{self.executor}]{dep_str}",
            classes="cpr-header"
        )
        preview = self.action[:90] + ("…" if len(self.action) > 90 else "")
        yield Static(preview, classes="cpr-action")

    def set_running(self) -> None:
        dep_str = f"  deps:{self.depends_on}" if self.depends_on else ""
        self.query(".cpr-header", Static).first().update(
            f"⏳ Step {self.step_num} [{self.executor}]{dep_str}"
        )
        self.set_class(False, "-pending")
        self.set_class(True,  "-running")

    def set_done(self, success: bool, latency_ms: float, error: str = None) -> None:
        icon    = "✓" if success else "✗"
        err_str = f" — {error}" if error else ""
        dep_str = f"  deps:{self.depends_on}" if self.depends_on else ""
        self.query(".cpr-header", Static).first().update(
            f"{icon} Step {self.step_num} [{self.executor}] {latency_ms:.0f}ms{err_str}{dep_str}"
        )
        self.query(".cpr-action", Static).first().update("")
        self.styles.height = 1
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
        step_num = _safe_step(step_num) # 👇 Defesa aqui também
        if step_num in self._rows:
            self._rows[step_num].set_running()

    def mark_done(self, step_num: int, success: bool,
                  latency_ms: float, error: str = None) -> None:
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

    def compose(self) -> ComposeResult:
        with Vertical(id="docker-header"):
            yield Static("● Docker", id="docker-title")
            yield Static(self._toggle_icon, id="docker-toggle")
        yield RichLog(id="docker-body", highlight=False, markup=False, max_lines=500)

    def on_click(self) -> None:
        """Toggle entre expandido e colapsado."""
        self._collapsed = not self._collapsed
        self.set_class(self._collapsed, "-collapsed")
        self._toggle_icon = "◀" if self._collapsed else "▶"
        self.query_one("#docker-toggle", Static).update(self._toggle_icon)

    def write(self, line: str) -> None:
        """Escreve uma linha no log."""
        self.query_one("#docker-body", RichLog).write(line.rstrip())


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

        self.query_one("#m-cpu").update(
            f"[{GRAY}]CPU [/][{cc}]{_bar(m['total_cpu'])}[/] [{cc}]{m['total_cpu']:.1f}%[/]{lc}")
        self.query_one("#m-ram").update(
            f"[{GRAY}]RAM [/][{rc}]{_bar(m['sys_ram_pct'])}[/] "
            f"[{rc}]{m['sys_ram_gb']:.1f}[/][dim]/{m['sys_ram_tot']:.1f}GB[/]{lr}")
        self.query_one("#m-gpu").update(gpu_str)
        self.query_one("#m-vram").update(vram_str)


# ─────────────────────────────────────────
# WIDGET: SPLASH HEADER
# ─────────────────────────────────────────
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
            yield Static(f"[{GRAY}]Model   [/]  [{WHITE}]{MODEL}[/]")
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
        self.content += delta
        self.query_one("#msg-body", Static).update(_render_rich(self.content))

    def append_reasoning(self, reasoning: str) -> None:
        """Adiciona reasoning ao ThinkingBox."""
        try:
            box = self.query_one("#thinking-box", ThinkingBox)
            box.append_reasoning(reasoning)
        except Exception:
            pass  # ThinkingBox só existe para role="assistant"

    def finish_thinking(self) -> None:
        """Finaliza o thinking e auto-colapsa."""
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
        # Primeira vez: remove classe hidden
        if self.has_class("-hidden"):
            self.remove_class("-hidden")

        self._char_count += len(text)
        
        # Atualiza label com contagem
        self.query_one("#think-label", Static).update(
            f"💭 thinking... ({self._char_count} chars)"
        )
        
        # Escreve no log (scroll automático)
        self.query_one("#think-content", RichLog).write(text)

    def finish_thinking(self) -> None:
        """Chamado quando o reasoning termina — auto-colapsa."""
        if self._char_count == 0:
            return
        
        # Atualiza label final
        self.query_one("#think-label", Static).update(
            f"💭 thought {self._char_count} chars"
        )
        
        # Auto-colapsa após terminar
        self._collapsed = True
        self.set_class(True, "-collapsed")
        self.query_one("#think-toggle", Static).update("▶")
        


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
            return

        docker_log = self.query_one(DockerLog)
        loop = asyncio.get_event_loop()

        while True:
            # run_in_executor evita bloquear o event loop do Textual
            # durante o readline() que pode ficar esperando dados
            line = await loop.run_in_executor(None, proc.stdout.readline)
            if not line:
                # Processo encerrou ou pipe fechou
                docker_log.write("[processo encerrado]")
                break
            docker_log.write(line.decode(errors="replace"))

    def _focus(self) -> None:
        self.query_one("#prompt-input", Input).focus()

    def _add_message(self, role: str, content: str) -> ChatMessage:
        scroll = self.query_one("#chat-scroll", ScrollableContainer)
        msg = ChatMessage(role=role, content=content)
        scroll.mount(msg)
        self.call_after_refresh(lambda: scroll.scroll_end(animate=False))
        return msg

    def _set_route_bar(self, route: str, conf: float, method: str, direct: bool):
        via = "direto" if direct else "CoT"
        color = BLUE if direct else AMBER
        self.query_one("#route-bar").update(
            f"  [{color}]▸ {route}[/] [{GRAY}]{conf:.0%} via {method} • {via}[/]"
        )

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
            try:
                ai_msg.finish_thinking()
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
                scroll.mount(_cot_plan_widget)
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



# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    from model_selector import run_model_selector

    chosen = run_model_selector()

    if chosen is None:
        raise SystemExit(0)

    mmproj_flag = "true" if chosen["mmproj"] else "false"

    # Inicia o llama manager
    subprocess.Popen([
        "python",
        "./Modules/llamaManager.py",
        "start",
        chosen["model"],
        f"--mmproj-used {mmproj_flag}",
    ])

    # Inicia o Docker (redireciona saída para captura)
    if platform.system() == "Windows":
        proc_docker = subprocess.Popen(
            ["./docker-start.bat", "--profile", "vulkan", "up"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # junta stderr no stdout
        )
    else:
        proc_docker = subprocess.Popen(
            ["bash", "./docker-start.sh", "--profile", "vulkan", "up"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    os.environ["AVA_MODEL"] = pathlib.Path(chosen["model"]).stem

    # Injeta o processo no app ANTES de rodar
    # (on_mount lê self._proc_docker para iniciar a leitura do log)
    app = AlphaAI()
    app._proc_docker = proc_docker
    app.run()