import asyncio
import glob
import json
import os
import pathlib
import platform
import re
import subprocess
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

MODEL   = os.environ.get("AVA_MODEL", "llama-local")
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
    """
    Converte Markdown e LaTeX inline para Rich Text renderizável no RichLog.
    Suporta: **bold**, *italic*, `code`, ```blocos```, # headers,
             > blockquote, $LaTeX$, $$LaTeX$$, listas, ---
    """
    result = Text()

    # Normaliza quebras de linha
    text = text.replace("\r\n", "\n")

    # Divide em blocos de código e texto normal para não processar o interior
    code_block_re = re.compile(r"```(\w*)\n?(.*?)```", re.DOTALL)
    parts = code_block_re.split(text)

    # parts alterna: texto_normal, lang, código, texto_normal, lang, código ...
    i = 0
    while i < len(parts):
        if i % 3 == 0:
            # texto normal — processa markdown/latex linha a linha
            _append_markdown_lines(result, parts[i])
        elif i % 3 == 1:
            lang = parts[i] or "code"
            i += 1  # pula para o código
            code_content = parts[i] if i < len(parts) else ""
            result.append(f"\n  [{lang}]\n", style=f"bold {BLUE}")
            for line in code_content.split("\n"):
                result.append(f"  {line}\n", style="bold #A8D8A8")  # verde claro
            result.append(f"  [/{lang}]\n", style=f"bold {BLUE}")
        i += 1

    return result


def _append_markdown_lines(result: Text, text: str):
    """Processa texto linha a linha aplicando estilos Markdown/LaTeX."""
    for line in text.split("\n"):
        stripped = line.rstrip()

        # Linha horizontal
        if re.match(r"^---+$", stripped):
            result.append("─" * 60 + "\n", style=DIM)
            continue

        # Headers
        h = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if h:
            level = len(h.group(1))
            sizes = {1: "bold #FFFFFF", 2: f"bold {WHITE}", 3: f"bold {BLUE}",
                     4: "bold", 5: "italic", 6: "italic dim"}
            result.append(h.group(2) + "\n", style=sizes.get(level, "bold"))
            continue

        # Blockquote
        if stripped.startswith("> "):
            result.append("│ ", style=BLUE)
            _append_inline(result, stripped[2:])
            result.append("\n")
            continue

        # Listas
        li = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)", stripped)
        if li:
            indent = "  " * (len(li.group(1)) // 2)
            bullet = "•" if not li.group(2)[0].isdigit() else li.group(2)
            result.append(f"{indent}{bullet} ", style=BLUE)
            _append_inline(result, li.group(3))
            result.append("\n")
            continue

        # Linha normal
        _append_inline(result, stripped)
        result.append("\n")


_INLINE_RE = re.compile(
    r"\$\$(.+?)\$\$"       # $$LaTeX$$     r"|\$([^$\n]+?)\$"     # $LaTeX$     r"|\*\*(.+?)\*\*"      # **bold**
    r"|__(.+?)__"          # __bold__
    r"|\*(.+?)\*"          # *italic*
    r"|_(.+?)_"            # _italic_
    r"|`([^`]+?)`"         # `code`
    r"|\[([^\]]+)\]\(([^)]+)\)",  # [link](url)
    re.DOTALL,
)


def _append_inline(result: Text, text: str):
    """Aplica estilos inline (bold, italic, code, LaTeX, link) em um trecho."""
    last = 0
    for m in _INLINE_RE.finditer(text):
        # texto antes do match
        if m.start() > last:
            result.append(text[last:m.start()], style=WHITE)

        latex_block, latex_inline = m.group(1), m.group(2)
        bold1, bold2 = m.group(3), m.group(4)
        italic1, italic2 = m.group(5), m.group(6)
        code = m.group(7)
        link_text, link_url = m.group(8), m.group(9)

        if latex_block:
            result.append(f" [{latex_block}] ", style=f"italic {AMBER}")
        elif latex_inline:
            result.append(f"[{latex_inline}]", style=f"italic {AMBER}")
        elif bold1 or bold2:
            result.append(bold1 or bold2, style="bold white")
        elif italic1 or italic2:
            result.append(italic1 or italic2, style="italic white")
        elif code:
            result.append(code, style="bold #A8D8A8")
        elif link_text:
            result.append(link_text, style=f"underline {BLUE}")
            result.append(f" ({link_url})", style=DIM)

        last = m.end()

    if last < len(text):
        result.append(text[last:], style=WHITE)


# ─────────────────────────────────────────
# STREAMING AVA - SSE Parser Atualizado
# ─────────────────────────────────────────
async def stream_ava(prompt: str, session_id: str,
                     on_delta, on_meta, on_step, on_result, on_error, on_done):
    """
    POST /execute com stream=true e consome os SSE.
    Agora suporta o padrão SSE nativo com 'event:' e 'data:' multilinha.
    - Eventos 'delta' e 'result' recebem TEXTO PURO (preserva Markdown/LaTeX).
    - Eventos 'meta', 'step', 'error', 'done' recebem JSON.
    """
    payload = {
        "input":      prompt,
        "session_id": session_id,
        "voice":      DEFAULT_VOICE,
        "lang":       DEFAULT_LANG,
        "tts":        False,
        "use_cache":  True,
        "stream":     True,
        "strategy":   "parallel",
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            async with client.stream("POST", f"{ORCHESTRATOR_URL}/execute",
                                     json=payload) as resp:
                resp.raise_for_status()
                
                current_event = "message"
                current_data = []
                
                async for line in resp.aiter_lines():
                    if not line:
                        # Linha vazia indica o fim de um bloco de evento SSE
                        if current_data:
                            raw = "\n".join(current_data)
                            
                            if current_event in ("delta", "result"):
                                # Eventos de texto puro (Markdown/LaTeX intactos)
                                data = raw
                            else:
                                # Eventos estruturados (JSON)
                                try:
                                    data = json.loads(raw)
                                except json.JSONDecodeError:
                                    data = raw
                            
                            # Dispara o callback correspondente
                            if current_event == "meta":
                                await on_meta(data)
                            elif current_event == "delta":
                                await on_delta(data)
                            elif current_event == "step":
                                await on_step(data)
                            elif current_event == "result":
                                await on_result(data)
                            elif current_event == "error":
                                err = data.get("error", str(data)) if isinstance(data, dict) else str(data)
                                await on_error(err)
                            elif current_event == "done":
                                await on_done(data)
                                
                        # Reseta para o próximo evento
                        current_event = "message"
                        current_data = []
                        
                    elif line.startswith("event:"):
                        current_event = line[6:].strip()
                        
                    elif line.startswith("data:"):
                        content = line[5:]
                        # Segundo a spec SSE, se começar com espaço, remove o primeiro espaço
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
    #logo   {{ color: {BLUE}; text-style: bold; height: {LOGO_HEIGHT}; }}
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
    """Mensagem individual do chat com suporte a Markdown/LaTeX."""

    DEFAULT_CSS = f"""
    ChatMessage {{
        height: auto;
        padding: 0 0 1 0;
    }}
    #msg-header {{ height: 1; padding: 0 0 0 2; }}
    #msg-body   {{ height: auto; padding: 0 0 0 4; }}
    """

    def __init__(self, role: str, content: str = "", msg_id: str = "", **kwargs):
        super().__init__(**kwargs)
        self.role    = role
        self.content = content
        self.msg_id  = msg_id or str(uuid.uuid4())[:8]

    def compose(self) -> ComposeResult:
        if self.role == "user":
            header = f"[bold {BLUE}]▶ Você[/]"
        elif self.role == "system":
            header = f"[dim {GRAY}]● Sistema[/]"
        elif self.role == "step":
            header = f"[dim {AMBER}]⚙ Pipeline[/]"
        else:
            header = f"[bold {GREEN}]◆ ALPHA AI[/]"

        yield Static(header, id="msg-header")
        yield Static(_render_rich(self.content), id="msg-body")

    def append_delta(self, delta: str):
        """Adiciona delta de streaming ao conteúdo e re-renderiza."""
        self.content += delta
        self.query_one("#msg-body", Static).update(_render_rich(self.content))


# ─────────────────────────────────────────
# APP
# ─────────────────────────────────────────
class AlphaAI(App):

    CSS = f"""
    Screen {{
        layout: vertical;
        background: #0D0D0D;
    }}
    #chat-wrap {{
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
    ]

    def __init__(self):
        super().__init__()
        self.session_id     = str(uuid.uuid4())
        self._streaming_msg : Optional[ChatMessage] = None
        self._is_streaming  = False

    def compose(self) -> ComposeResult:
        yield SplashHeader()
        with Vertical(id="chat-wrap"):
            yield Static("", id="route-bar")
            yield ScrollableContainer(id="chat-scroll")
        with Vertical(id="input-wrap"):
            yield Input(
                placeholder="> Digite seu prompt e pressione Enter...",
                id="prompt-input",
            )
        yield MetricsBar()
        yield Footer()

    def on_mount(self) -> None:
        psutil.cpu_percent(interval=None)
        self.call_after_refresh(self._focus)
    def _focus(self) -> None:
        self.query_one("#prompt-input", Input).focus()

    # ── helpers ─────────────────────────────────────────────────────────
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

    # ── actions ─────────────────────────────────────────────────────────
    def action_clear_chat(self) -> None:
        scroll = self.query_one("#chat-scroll", ScrollableContainer)
        scroll.remove_children()
        self._add_message("system", "Chat limpo.")

    def action_new_session(self) -> None:
        self.session_id = str(uuid.uuid4())
        scroll = self.query_one("#chat-scroll", ScrollableContainer)
        scroll.remove_children()
        self._add_message("system",
            f"Nova sessão iniciada: `{self.session_id[:8]}`")

    # ── input ────────────────────────────────────────────────────────────
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "prompt-input" or self._is_streaming:
            return

        prompt = event.value.strip()
        if not prompt:
            return

        event.input.value = ""
        event.input.disabled = True
        self._is_streaming   = True

        self._add_message("user", prompt)
        ai_msg = self._add_message("assistant", "")
        self._streaming_msg = ai_msg

        asyncio.create_task(self._run_stream(prompt, ai_msg))

    async def _run_stream(self, prompt: str, ai_msg: ChatMessage) -> None:
        scroll = self.query_one("#chat-scroll", ScrollableContainer)

        async def on_delta(delta: str):
            ai_msg.append_delta(delta)
            self.call_after_refresh(lambda: scroll.scroll_end(animate=False))

        async def on_meta(event: dict):
            self._set_route_bar(
                event.get("route", "?"),
                event.get("route_confidence", 0.0),
                event.get("route_method", "?"),
                event.get("routed_directly", False),
            )

        async def on_step(event: dict):
            step_n   = event.get("step", "?")
            executor = event.get("executor", "?")
            ok       = event.get("success", False)
            lat      = event.get("latency_ms", 0)
            icon     = "✓" if ok else "✗"
            color    = GREEN if ok else RED_C
            step_text = (f"\n[{GRAY}]  {icon} step {step_n} "
                         f"[{executor}] {lat:.0f}ms[/]\n")
            ai_msg.content += step_text
            ai_msg.query_one("#msg-body", Static).update(_render_rich(ai_msg.content))

        async def on_result(text: str):
            ai_msg.append_delta(text)
            self.call_after_refresh(lambda: scroll.scroll_end(animate=False))

        async def on_error(error: str):
            self._add_message("system", f"**Erro:** {error}")

        async def on_done(event: dict):
            lat  = event.get("total_latency_ms", 0)
            errs = event.get("errors", [])
            route = event.get("route", "?")
            conf  = event.get("route_confidence", 0.0)
            meth  = event.get("route_method", "?")
            direct = event.get("routed_directly", False)
            via   = "direto" if direct else "CoT"
            color = BLUE if direct else AMBER
            self.query_one("#route-bar").update(
                f"  [{color}]▸ {route}[/] [{GRAY}]{conf:.0%} via {meth} "
                f"• {via} • {lat:.0f}ms[/]"
                + (f"  [{RED_C}]{len(errs)} erro(s)[/]" if errs else "")
            )

        await stream_ava(
            prompt, self.session_id,
            on_delta, on_meta, on_step, on_result, on_error, on_done,
        )

        self._is_streaming = False
        self._streaming_msg = None
        inp = self.query_one("#prompt-input", Input)
        inp.disabled = False
        self.call_after_refresh(self._focus)


if __name__ == "__main__":
    AlphaAI().run()