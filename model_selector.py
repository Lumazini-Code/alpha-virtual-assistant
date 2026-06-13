"""
model_selector.py — Tela de seleção/download de modelos para o AlphaAI.

Fluxo:
  1. ModelSelectScreen  — lista GGUFs locais, detecta mmproj par a par.
  2. HFDownloadScreen   — busca e baixa modelos do Hugging Face (via API pública).

Uso (no __main__ do app):
    from model_selector import run_model_selector
    chosen = run_model_selector()          # retorna {"model": path, "mmproj": path|None}
    if chosen:
        os.environ["AVA_MODEL_PATH"] = chosen["model"]
        if chosen["mmproj"]:
            os.environ["AVA_MMPROJ_PATH"] = chosen["mmproj"]
    AlphaAI().run()
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx
from rich.text import Text
from textual.app import App, ComposeResult
from textual.message import Message
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widget import Widget
from textual.widgets import Button, Footer, Input, Label, Static

# ─────────────────────────────────────────
# CONFIG — diretórios onde buscar GGUFs
# ─────────────────────────────────────────
SEARCH_DIRS: list[pathlib.Path] = [
    pathlib.Path(os.environ.get("LLAMA_MODEL_DIR", "./Modules/Models")).expanduser().resolve(),
    pathlib.Path("/mnt/models"),
    pathlib.Path("/opt/models"),
    (pathlib.Path.home() / ".cache" / "llama").resolve(),
    (pathlib.Path.home() / ".local" / "share" / "llama").resolve(),
]

# Criadores curados no Hugging Face
HF_CREATORS = [
    "bartowski",
    "unsloth",
    "TheBloke",
    "lmstudio-community",
    "MaziyarPanahi",
]

HF_API = "https://huggingface.co/api"

# ─────────────────────────────────────────
# CORES (mesmas do app principal)
# ─────────────────────────────────────────
BLUE  = "#1243E4"
GREEN = "#4CAF7D"
AMBER = "#E4A012"
RED_C = "#E45012"
WHITE = "#CCCCCC"
GRAY  = "#555555"
DIM   = "#333333"


# ─────────────────────────────────────────
# DATA
# ─────────────────────────────────────────
@dataclass
class LocalModel:
    path: pathlib.Path
    mmproj: Optional[pathlib.Path] = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def size_gb(self) -> float:
        try:
            return self.path.stat().st_size / 1024**3
        except OSError:
            return 0.0


@dataclass
class HFModel:
    repo_id: str
    filename: str
    creator: str
    size_bytes: int = 0
    url: str = ""

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1024**3

    @property
    def name(self) -> str:
        return f"{self.repo_id.split('/')[-1]}/{self.filename}"


# ─────────────────────────────────────────
# SCANNER LOCAL
# ─────────────────────────────────────────
def scan_local_models() -> list[LocalModel]:
    """Varre SEARCH_DIRS em busca de .gguf, detecta mmproj adjacente."""
    found: list[LocalModel] = []
    seen: set[pathlib.Path] = set()

    for base in SEARCH_DIRS:
        if not base.exists():
            continue
        for gguf in sorted(base.rglob("*.gguf")):
            gguf = gguf.resolve()          # garante caminho absoluto
            if gguf in seen:
                continue
            # Ignora arquivos que já são mmproj
            if "mmproj" in gguf.name.lower():
                continue
            seen.add(gguf)
            mmproj = _find_mmproj(gguf)
            found.append(LocalModel(path=gguf, mmproj=mmproj))

    return found


def _find_mmproj(gguf: pathlib.Path) -> Optional[pathlib.Path]:
    """Tenta encontrar arquivo mmproj correspondente ao modelo."""
    parent = gguf.parent
    stem = gguf.stem  # e.g. "llava-v1.5-7b.Q4_K_M"

    # Estratégia 1 — mesmo diretório, padrão *mmproj*
    for candidate in parent.glob("*mmproj*.gguf"):
        return candidate.resolve()

    # Estratégia 2 — prefixo comum
    prefix = re.split(r"[.\-_][qQ]\d", stem)[0]  # antes do quantizer
    for candidate in parent.glob(f"{prefix}*mmproj*.gguf"):
        return candidate.resolve()

    return None


# ─────────────────────────────────────────
# HF API
# ─────────────────────────────────────────
async def hf_search(query: str, creators: list[str], limit: int = 40) -> list[HFModel]:
    """Busca modelos GGUF no HF Hub dos criadores curados."""
    results: list[HFModel] = []

    async with httpx.AsyncClient(timeout=15.0) as client:
        for creator in creators:
            try:
                resp = await client.get(
                    f"{HF_API}/models",
                    params={
                        "author":   creator,
                        "search":   query,
                        "filter":   "gguf",
                        "sort":     "downloads",
                        "direction": "-1",
                        "limit":    limit // len(creators) + 2,
                    },
                )
                resp.raise_for_status()
                for model in resp.json():
                    repo_id  = model.get("modelId", "")
                    siblings = model.get("siblings", [])
                    for sib in siblings:
                        fname = sib.get("rfilename", "")
                        if not fname.endswith(".gguf"):
                            continue
                        if "mmproj" in fname.lower():
                            continue
                        results.append(HFModel(
                            repo_id=repo_id,
                            filename=fname,
                            creator=creator,
                            size_bytes=sib.get("size", 0),
                            url=f"https://huggingface.co/{repo_id}/resolve/main/{fname}",
                        ))
            except Exception:
                pass  # silencia falhas individuais por criador

    return results


async def hf_download(model: HFModel, dest_dir: pathlib.Path,
                      on_progress) -> pathlib.Path:
    """Baixa arquivo GGUF com callback de progresso (bytes_done, total)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / model.filename
    tmp_file  = dest_file.with_suffix(".tmp")

    async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
        async with client.stream("GET", model.url,
                                 follow_redirects=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            done  = 0
            with tmp_file.open("wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=1024 * 512):
                    f.write(chunk)
                    done += len(chunk)
                    await on_progress(done, total)

    tmp_file.rename(dest_file)
    return dest_file


# ─────────────────────────────────────────
# CSS COMPARTILHADO
# ─────────────────────────────────────────
_SHARED_CSS = f"""
Screen {{
    background: #0D0D0D;
    layout: vertical;
}}
#header {{
    height: 4;
    padding: 1 4;
    background: #0A0A0A;
    border-bottom: tall #1E1E1E;
}}
#header-title {{
    color: {BLUE};
    text-style: bold;
}}
#header-sub {{
    color: {GRAY};
}}
#content {{
    height: 1fr;
    padding: 1 4;
}}
#footer-bar {{
    height: 3;
    padding: 0 4;
    background: #0A0A0A;
    border-top: tall #1E1E1E;
    align: center middle;
}}
Button {{
    background: #111111;
    border: tall #1E1E1E;
    color: {WHITE};
    height: 3;
}}
Button:focus {{ border: tall {BLUE}; }}
Button.-primary {{
    background: {BLUE};
    color: #FFFFFF;
    border: tall {BLUE};
}}
Button.-success {{
    background: #0D2E1A;
    color: {GREEN};
    border: tall {GREEN};
}}
Button.-danger {{
    background: #2E0D0D;
    color: {RED_C};
    border: tall {RED_C};
}}
ScrollableContainer {{
    background: #0D0D0D;
}}
"""


# ═══════════════════════════════════════════════════════════
# WIDGET: ModelCard (local)
# ═══════════════════════════════════════════════════════════
class ModelCard(Widget):
    """Linha clicável representando um modelo local."""

    DEFAULT_CSS = f"""
    ModelCard {{
        height: 4;
        padding: 0 2;
        border-bottom: solid #151515;
        layout: horizontal;
        align: center middle;
    }}
    ModelCard:hover {{ background: #111111; }}
    ModelCard.-selected {{
        background: #0D1A3A;
        border-left: thick {BLUE};
    }}
    #mc-info  {{ width: 1fr; height: 4; content-align: left middle; }}
    #mc-badge {{ width: 16; height: 4; content-align: right middle; }}
    """

    def __init__(self, model: LocalModel, **kwargs):
        super().__init__(**kwargs)
        self.model    = model
        self.selected = False

    def compose(self) -> ComposeResult:
        m = self.model
        name_text = Text(m.name, style=f"bold {WHITE}")
        meta_parts = [f"{m.size_gb:.2f} GB"]
        if m.mmproj:
            meta_parts.append(f"mmproj: {m.mmproj.name}")
        meta_text = "  " + "  │  ".join(meta_parts)

        with Vertical(id="mc-info"):
            yield Static(name_text)
            yield Static(meta_text, markup=False)

        badge = f"[{GREEN}]✦ mmproj[/]" if m.mmproj else f"[{DIM}]·[/]"
        yield Static(badge, id="mc-badge")

    def on_click(self) -> None:
        self.post_message(ModelCard.Selected(self))

    def toggle_selected(self, value: bool) -> None:
        self.selected = value
        self.set_class(value, "-selected")

    class Selected(Message):
        def __init__(self, card: "ModelCard") -> None:
            super().__init__()
            self.card = card


# ═══════════════════════════════════════════════════════════
# SCREEN 1: ModelSelectScreen
# ═══════════════════════════════════════════════════════════
class ModelSelectScreen(App):
    """Tela 1 — seleção de modelo local."""

    CSS = _SHARED_CSS + f"""
    #no-models {{
        padding: 2 4;
        color: {GRAY};
    }}
    #select-btn {{ width: 22; }}
    #download-btn {{ width: 28; }}
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Sair"),
        Binding("enter",  "confirm", "Selecionar"),
        Binding("ctrl+d", "go_download", "Baixar modelos"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._models: list[LocalModel] = []
        self._selected_card: Optional[ModelCard] = None
        self._result: Optional[dict] = None    # {"model": str, "mmproj": str|None}
        self._go_download = False

    # ── resultado público ──────────────────────────────────
    @property
    def result(self) -> Optional[dict]:
        return self._result

    @property
    def wants_download(self) -> bool:
        return self._go_download

    # ── compose ───────────────────────────────────────────
    def compose(self) -> ComposeResult:
        with Vertical(id="header"):
            yield Static("ALPHA AI  —  seleção de modelo", id="header-title")
            yield Static("escolha um modelo GGUF local ou baixe do Hugging Face",
                         id="header-sub")

        with ScrollableContainer(id="content"):
            pass  # preenchido no on_mount

        with Horizontal(id="footer-bar"):
            yield Button("Selecionar  ↵", id="select-btn",   classes="-primary")
            yield Button("Baixar mais modelos  HF ↗", id="download-btn", classes="-success")
            yield Button("Sair  ^Q",       id="quit-btn")

        yield Footer()

    def on_mount(self) -> None:
        self._models = scan_local_models()
        content = self.query_one("#content", ScrollableContainer)

        if not self._models:
            content.mount(
                Static(
                    f"[{GRAY}]Nenhum modelo .gguf encontrado em:[/]\n"
                    + "\n".join(f"  [{DIM}]{d}[/]" for d in SEARCH_DIRS)
                    + f"\n\n[{AMBER}]Use Ctrl+D para baixar um modelo do Hugging Face.[/]",
                    id="no-models",
                )
            )
        else:
            for m in self._models:
                content.mount(ModelCard(m))

    # ── eventos ───────────────────────────────────────────
    def on_model_card_selected(self, event: ModelCard.Selected) -> None:
        if self._selected_card:
            self._selected_card.toggle_selected(False)
        self._selected_card = event.card
        event.card.toggle_selected(True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "select-btn":
            self.action_confirm()
        elif event.button.id == "download-btn":
            self.action_go_download()
        elif event.button.id == "quit-btn":
            self.action_quit()

    def action_confirm(self) -> None:
        if not self._selected_card:
            return
        m = self._selected_card.model
        self._result = {
            "model":  str(m.path),
            "mmproj": str(m.mmproj) if m.mmproj else None,
        }
        self.exit()

    def action_go_download(self) -> None:
        self._go_download = True
        self.exit()

    def action_quit(self) -> None:
        self.exit()


# ═══════════════════════════════════════════════════════════
# WIDGET: HFModelRow
# ═══════════════════════════════════════════════════════════
class HFModelRow(Widget):
    DEFAULT_CSS = f"""
    HFModelRow {{
        height: 5;
        padding: 0 2;
        border-bottom: solid #151515;
        layout: horizontal;
        align: center middle;
    }}
    HFModelRow:hover {{ background: #111111; }}
    #hf-info    {{ width: 1fr; height: 5; content-align: left middle; }}
    #hf-actions {{ width: 18; height: 5; content-align: right middle; }}
    """

    def __init__(self, model: HFModel, **kwargs):
        super().__init__(**kwargs)
        self.hf_model  = model
        self._progress = -1.0   # -1 = não iniciado

    def compose(self) -> ComposeResult:
        m = self.hf_model
        size_str = f"{m.size_gb:.2f} GB" if m.size_bytes else "tamanho desconhecido"
        with Vertical(id="hf-info"):
            yield Static(Text(m.filename, style=f"bold {WHITE}"))
            yield Static(
                f"  [{GRAY}]{m.creator}[/]  [{DIM}]│[/]  [{AMBER}]{size_str}[/]",
                markup=True,
            )
            yield Static("", id=f"prog-{id(self)}")
        yield Button("↓ baixar", id="dl-btn", classes="-success")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "dl-btn":
            event.stop()
            self.post_message(HFModelRow.DownloadRequested(self))

    def set_progress(self, done: int, total: int) -> None:
        pct = done / total * 100 if total else 0
        bar_w = 30
        filled = round(pct / 100 * bar_w)
        bar = "█" * filled + "░" * (bar_w - filled)
        label = self.query_one(f"#prog-{id(self)}", Static)
        label.update(
            f"  [{GREEN}]{bar}[/] [{WHITE}]{pct:.0f}%[/]  "
            f"[{GRAY}]{done/1024**2:.0f}/{total/1024**2:.0f} MB[/]"
        )

    def mark_done(self, dest: pathlib.Path) -> None:
        label = self.query_one(f"#prog-{id(self)}", Static)
        label.update(f"  [{GREEN}]✓ salvo em {dest}[/]")
        self.query_one("#dl-btn", Button).disabled = True

    class DownloadRequested(Message):
        def __init__(self, row: "HFModelRow") -> None:
            super().__init__()
            self.row = row


# ═══════════════════════════════════════════════════════════
# SCREEN 2: HFDownloadScreen
# ═══════════════════════════════════════════════════════════
class HFDownloadScreen(App):

    CSS = _SHARED_CSS + f"""
    #search-row {{
        height: 5;
        padding: 1 0;
        layout: horizontal;
    }}
    #search-input {{
        width: 1fr;
        margin-right: 2;
        background: #111111;
        border: tall #1E1E1E;
        color: {WHITE};
        padding: 0 2;
    }}
    #search-input:focus {{ border: tall {BLUE}; }}
    #search-btn {{ width: 14; }}
    #creator-row {{
        height: 3;
        layout: horizontal;
        padding: 0 0 1 0;
    }}
    .creator-btn {{
        width: auto;
        min-width: 14;
        margin-right: 1;
        height: 3;
    }}
    .creator-btn.-active {{
        background: {BLUE};
        color: #FFFFFF;
        border: tall {BLUE};
    }}
    #status-bar {{
        height: 1;
        color: {GRAY};
        padding: 0 0 1 0;
    }}
    #back-btn {{ width: 16; }}
    """

    BINDINGS = [
        Binding("ctrl+q", "quit",   "Sair"),
        Binding("ctrl+b", "go_back", "Voltar"),
        Binding("enter",  "search",  "Buscar"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._results: list[HFModel]         = []
        self._active_creators: set[str]      = set(HF_CREATORS)
        self._downloading: set[str]          = set()
        self._dest_dir = pathlib.Path(
            os.environ.get("LLAMA_MODEL_DIR", "~/models")
        ).expanduser().resolve()

    def compose(self) -> ComposeResult:
        with Vertical(id="header"):
            yield Static("ALPHA AI  —  baixar modelo do Hugging Face", id="header-title")
            yield Static(
                f"criadores curados: {', '.join(HF_CREATORS)}",
                id="header-sub",
            )

        with Vertical(id="content"):
            with Horizontal(id="search-row"):
                yield Input(placeholder="buscar modelo (ex: llama 3.2, mistral 7b)...",
                            id="search-input")
                yield Button("Buscar ↵", id="search-btn", classes="-primary")

            with Horizontal(id="creator-row"):
                for c in HF_CREATORS:
                    yield Button(c, id=f"cr-{c}", classes="creator-btn -active")

            yield Static("", id="status-bar")
            yield ScrollableContainer(id="results-scroll")

        with Horizontal(id="footer-bar"):
            yield Button("← voltar", id="back-btn")
            yield Button("Sair  ^Q", id="quit-btn")

        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#search-input", Input).focus()

    # ── busca ─────────────────────────────────────────────
    def action_search(self) -> None:
        q = self.query_one("#search-input", Input).value.strip()
        self._do_search(q)

    def _do_search(self, query: str) -> None:
        self.query_one("#status-bar").update(f"[{AMBER}]buscando...[/]")
        asyncio.create_task(self._async_search(query))

    async def _async_search(self, query: str) -> None:
        creators = list(self._active_creators)
        results  = await hf_search(query, creators, limit=60)
        self._results = results

        scroll = self.query_one("#results-scroll", ScrollableContainer)
        await scroll.remove_children()

        if not results:
            await scroll.mount(
                Static(f"[{GRAY}]Nenhum resultado para '{query}'.[/]",
                       markup=True)
            )
        else:
            for m in results[:30]:
                await scroll.mount(HFModelRow(m))

        self.query_one("#status-bar").update(
            f"[{GREEN}]{len(results)} resultado(s)[/]"
            + (f"  [{AMBER}](mostrando 30)[/]" if len(results) > 30 else "")
        )

    # ── download ──────────────────────────────────────────
    def on_hf_model_row_download_requested(self, event: HFModelRow.DownloadRequested) -> None:
        m = event.row.hf_model
        if m.url in self._downloading:
            return
        self._downloading.add(m.url)
        asyncio.create_task(self._async_download(event.row, m))

    async def _async_download(self, row: HFModelRow, m: HFModel) -> None:
        async def on_progress(done: int, total: int):
            row.set_progress(done, total)

        try:
            dest = await hf_download(m, self._dest_dir, on_progress)
            row.mark_done(dest)
            self.query_one("#status-bar").update(
                f"[{GREEN}]✓ {m.filename} salvo em {dest}[/]"
            )
        except Exception as e:
            self.query_one("#status-bar").update(
                f"[{RED_C}]Erro no download: {e}[/]"
            )
            self._downloading.discard(m.url)

    # ── filtro de criador ─────────────────────────────────
    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""

        if bid.startswith("cr-"):
            creator = bid[3:]
            if creator in self._active_creators:
                self._active_creators.discard(creator)
                event.button.remove_class("-active")
            else:
                self._active_creators.add(creator)
                event.button.add_class("-active")
            return

        if bid == "search-btn":
            self.action_search()
        elif bid == "back-btn":
            self.action_go_back()
        elif bid == "quit-btn":
            self.exit()

    def action_go_back(self) -> None:
        self.exit()

    def action_quit(self) -> None:
        self.exit()


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
def run_model_selector() -> Optional[dict]:
    """
    Executa o fluxo completo de seleção de modelo.
    Retorna {"model": str, "mmproj": str|None} ou None (saiu sem selecionar).
    """
    while True:
        sel_app = ModelSelectScreen()
        sel_app.run()

        if sel_app.wants_download:
            dl_app = HFDownloadScreen()
            dl_app.run()
            # Após download, volta para seleção (o loop reinicia)
            continue

        return sel_app.result  # None ou {"model": ..., "mmproj": ...}