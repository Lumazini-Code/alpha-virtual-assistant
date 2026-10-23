#!/usr/bin/env python3
"""
plan_recorder.py — CLI de gravação de planos híbridos (macro + modelo)
═══════════════════════════════════════════════════════════════════════

Grava planos para o executor do orchestrator (run_plan) usando o recurso
de RECORDING do Playwright MCP (browser_start_recording /
browser_stop_recording), via mcp_manager (porta 4008).

Fluxo (gravação segmentada — resolve a ordem entre macro e modelo):

  1. CLI abre a URL inicial no navegador do servidor "playwright"
  2. browser_start_recording → você opera o navegador (parte macro)
  3. No CLI, digite uma instrução em linguagem natural → CLI faz:
       - browser_stop_recording → código Playwright do segmento → steps macro
       - cria 1 step "modelo" com a instrução (na posição correta)
       - browser_start_recording de novo → continue no navegador
  4. `done` (ou Ctrl+D) → grava o último segmento e mostra o preview
  5. Confirmado → POST /plans/write na memory (porta 3000)

Comandos do CLI (durante a gravação):
  <texto livre>            → step "modelo" com essa instrução
  expect url_contains X    → anexa expect ao último step macro
  expect selector_visible X → idem
  desc <texto>             → descrição do último step
  undo                     → descarta o último segmento gravado
  done / Ctrl+D            → finalizar e mostrar preview
  cancel / Ctrl+C          → abortar sem gravar nada

Uso:
  python scripts/plan_recorder.py [url_inicial] [--server playwright]
         [--memory http://localhost:3000] [--mcp http://localhost:4008]

Pré-requisitos:
  - mcp_manager (4008) com servidor "playwright" registrado e habilitado
  - memory (3000) no ar (só para o --push / gravação final)
  - o snippet do servidor playwright precisa incluir as tools de recording
    (browser_start_recording / browser_stop_recording — capability
    "devtools" do @playwright/mcp ≥ versão com recording; se o registro
    do seu servidor filtrar capabilities, habilite devtools)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from urllib.parse import urlparse

import httpx

# ── Config default ────────────────────────────────────────────────────────────
MCP_MANAGER_URL = "http://127.9.12.22:4008"
MEMORY_API_URL = "http://127.9.12.22:3000"
PLAYWRIGHT_SERVER = "playwright"
CALL_TIMEOUT = 120.0

# ── Parser: código Playwright (recorder) → steps macro ───────────────────────
#
# O browser_stop_recording devolve o código no codegenLanguage do servidor
# (default "typescript"). Padrões cobertos:
#
#   await page.goto('https://...')                     → browser_navigate
#   await page.getByRole('button', { name: 'X' }).click()  → browser_click
#   await page.getByLabel('X').fill('texto')           → browser_type
#   await page.getByRole('textbox', ...).pressSequentially('txt')
#   await page.getByRole('combobox', ...).selectOption('a')
#   await page.getByLabel('X').check() / uncheck()     → browser_check/uncheck
#   await page.keyboard.press('Enter')                 → browser_press_key
#
# IMPORTANTE — `target` vs `element`: o schema do Playwright MCP exige
# `target` (string OBRIGATÓRIA) em click/type/check/etc. O `target` aceita
# tanto um ref do snapshot (f18e215, volátil) quanto uma expressão locator
# JS (`getByRole('link', { name: 'New issue' })`) — o MCP converte via
# locatorOrSelectorAsSelector. Aqui gravamos a EXPRESSÃO do locator (sem o
# `page.`) em `target` — estável entre execuções — e a descrição humana em
# `element` (só permissão/legibilidade).
#
# Linhas não reconhecidas viram step macro com action=null (aviso no preview;
# editável via PATCH /plans/update depois).

_JS_STRING = r"(?:'[^']*'|\"[^\"]*\"|`[^`]*`)"


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"`":
        return s[1:-1]
    return s


def _locator_expr(line: str) -> str:
    """Extrai a EXPRESSÃO locator JS da linha (p/ params.target) — o formato
    que o Playwright MCP aceita como alvo estável. Ex.:
    page.getByRole('link', { name: 'New issue' }).click() →
    getByRole('link', { name: 'New issue' })
    Suporta encadeamento (.locator(...).getByRole(...)) e frames."""
    m = re.search(r"page\.(getBy\w+\(.*)", line)
    if not m:
        m = re.search(r"page\.(locator\(.*)", line)
        if not m:
            m = re.search(r"page\.(frameLocator\(.*)", line)
    if not m:
        return ""
    expr = m.group(1)
    # corta na chamada de ação (.click( / .fill( / .check( etc.)
    expr = re.split(r"\.(?:click|dblclick|fill|pressSequentially|selectOption|check|uncheck|press|type)\(", expr)[0]
    expr = expr.strip()
    # fecha parênteses/chaves balanceando (a ação pode estar em outra linha)
    if expr.count("(") > expr.count(")"):
        expr += ")" * (expr.count("(") - expr.count(")"))
    return expr


def _locator_text(line: str) -> str:
    """Extrai uma descrição humana do locator da linha (p/ params.element).
    Ex.: page.getByRole('button', { name: 'Faturas' }) → 'button "Faturas"'."""
    m = re.search(r"getByRole\(\s*(" + _JS_STRING + r")", line)
    if m:
        role = _unquote(m.group(1))
        m2 = re.search(r"name:\s*(" + _JS_STRING + r")", line)
        name = _unquote(m2.group(1)) if m2 else ""
        return f'{role} "{name}"' if name else role
    m = re.search(r"getByLabel\(\s*(" + _JS_STRING + r")", line)
    if m:
        return f'label "{_unquote(m.group(1))}"'
    m = re.search(r"getByText\(\s*(" + _JS_STRING + r")", line)
    if m:
        return f'text "{_unquote(m.group(1))}"'
    m = re.search(r"getByPlaceholder\(\s*(" + _JS_STRING + r")", line)
    if m:
        return f'placeholder "{_unquote(m.group(1))}"'
    m = re.search(r"getByTestId\(\s*(" + _JS_STRING + r")", line)
    if m:
        return f'testid "{_unquote(m.group(1))}"'
    m = re.search(r"locator\(\s*(" + _JS_STRING + r")", line)
    if m:
        return _unquote(m.group(1))
    # frame/nested locators: pega o último .getByXxx(...) como texto solto
    m = re.search(r"(getBy\w+\(.*)$", line)
    if m:
        return m.group(1).split(".").pop().strip().rstrip(".")
    return ""


def parse_recorded_code(code: str) -> list[dict]:
    """Converte o código gerado pelo recorder em steps macro. Cada linha
    executável vira um step; linhas de comentário/blank são ignoradas."""
    steps: list[dict] = []
    for raw in code.splitlines():
        line = raw.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        step = _parse_line(line)
        if step is not None:
            steps.append(step)
        else:
            steps.append({
                "executor": "macro", "action": None,
                "params": {"raw": line},
                "description": "LINHA NÃO RECONHECIDA — edite depois",
            })
    return steps


def _parse_line(line: str) -> dict | None:
    def mk(action: str, params: dict) -> dict:
        return {"executor": "macro", "action": action, "params": params, "description": ""}

    m = re.search(r"page\.goto\(\s*(" + _JS_STRING + r")", line)
    if m and "click" not in line:
        return mk("browser_navigate", {"url": _unquote(m.group(1))})

    # ações de ELEMENTO: target = expressão locator JS (obrigatório no schema
    # do Playwright MCP), element = descrição humana (opcional, legibilidade)
    target = _locator_expr(line)

    if re.search(r"\.click\(\s*\)", line):
        return mk("browser_click", {"element": _locator_text(line), "target": target})

    if re.search(r"\.dblclick\(\s*\)", line):
        return mk("browser_click", {"element": _locator_text(line), "target": target,
                                    "doubleClick": True})

    m = re.search(r"\.fill\(\s*(" + _JS_STRING + r")\s*\)", line)
    if m:
        return mk("browser_type", {"element": _locator_text(line), "target": target,
                                   "text": _unquote(m.group(1))})

    m = re.search(r"\.pressSequentially\(\s*(" + _JS_STRING + r")", line)
    if m:
        return mk("browser_type", {"element": _locator_text(line), "target": target,
                                   "text": _unquote(m.group(1))})

    m = re.search(r"\.selectOption\(\s*(" + _JS_STRING + r")\s*\)", line)
    if m:
        return mk("browser_select_option", {"element": _locator_text(line), "target": target,
                                            "values": [_unquote(m.group(1))]})

    if re.search(r"\.check\(\s*\)", line):
        return mk("browser_check", {"element": _locator_text(line), "target": target})
    if re.search(r"\.uncheck\(\s*\)", line):
        return mk("browser_uncheck", {"element": _locator_text(line), "target": target})

    m = re.search(r"page\.keyboard\.press\(\s*(" + _JS_STRING + r")\s*\)", line)
    if m:
        return mk("browser_press_key", {"key": _unquote(m.group(1))})

    m = re.search(r"\.press\(\s*(" + _JS_STRING + r")\s*\)", line)
    if m:
        return mk("browser_press_key", {"key": _unquote(m.group(1))})

    m = re.search(r"page\.waitForURL\(\s*(" + _JS_STRING + r")", line)
    if m:
        return mk("browser_navigate", {"url": _unquote(m.group(1)),
                                       "description": "aguardar URL (gravação)"})
    return None


# ── Cliente MCP (via mcp_manager) e memory ────────────────────────────────────

class Recorder:
    def __init__(self, http: httpx.AsyncClient, server: str):
        self.http = http
        self.server = server

    async def call(self, tool: str, arguments: dict | None = None) -> str:
        r = await self.http.post(
            f"/servers/{self.server}/call",
            json={"tool": tool, "arguments": arguments or {}, "timeout": CALL_TIMEOUT},
        )
        r.raise_for_status()
        data = r.json()
        if data.get("is_error"):
            raise RuntimeError(f"{tool} reportou erro: {data.get('result')}")
        return data.get("result") or ""

    async def start_recording(self) -> None:
        await self.call("browser_start_recording")

    async def stop_recording(self) -> str:
        raw = await self.call("browser_stop_recording")
        return _strip_codeframe(raw)

    async def navigate(self, url: str) -> str:
        return await self.call("browser_navigate", {"url": url})


def _strip_codeframe(text: str) -> str:
    """O stop_recording devolve: 'Recording stopped... ```ts\\n<código>\\n```'.
    Extrai só o código; se não houver codeframe, devolve o texto integral."""
    m = re.search(r"```[a-z]*\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


async def push_plan(memory: httpx.AsyncClient, name: str, description: str,
                    steps: list[dict]) -> dict:
    r = await memory.post("/plans/write", json={
        "name": name, "description": description, "steps": steps,
    })
    if r.status_code != 200:
        try:
            detail = r.json().get("error", "")
        except Exception:
            detail = r.text[:200]
        raise RuntimeError(f"memory recusou o plano (HTTP {r.status_code}): {detail}")
    return r.json()


# ── Preview / helpers de exibição ─────────────────────────────────────────────

def step_brief(i: int, step: dict) -> str:
    ex = step.get("executor", "?")
    if ex == "modelo":
        obj = step.get("objective") or ""
        extra = []
        if step.get("scope"):
            extra.append("escopo")
        if step.get("output_schema"):
            extra.append("schema")
        tag = f" [{'+'.join(extra)}]" if extra else ""
        return f"{i:>3}. MODELO  {obj[:90]!r}{tag}"
    action = (step.get("action") or "??").replace("browser_", "")
    params = step.get("params") or {}
    if action == "navigate":
        detail = params.get("url", "")
    elif action == "click":
        detail = params.get("element", "")
    elif action == "type":
        detail = f"{params.get('element', '')} ← {params.get('text', '')!r}"
    elif action == "select_option":
        detail = f"{params.get('element', '')} ← {params.get('values')}"
    elif action == "press_key":
        detail = params.get("key", "")
    elif "raw" in params:
        detail = params["raw"][:80]
    else:
        detail = json.dumps(params, ensure_ascii=False)[:80]
    expect = step.get("expect")
    ex_tag = f"  (expect: {json.dumps(expect, ensure_ascii=False)})" if expect else ""
    return f"{i:>3}. MACRO   {action:<14} {detail}{ex_tag}"


def show_preview(steps: list[dict]) -> None:
    print("\n── Preview do plano (só o que executa) ──")
    for i, s in enumerate(steps, 1):
        print(step_brief(i, s))
    n_modelo = sum(1 for s in steps if s.get("executor") == "modelo")
    print(f"total: {len(steps)} passos ({len(steps) - n_modelo} macro, {n_modelo} modelo)")


# ── Loop principal ────────────────────────────────────────────────────────────

HELP_TEXT = """
Comandos:
  <texto livre>             instrução p/ o modelo (vira step "modelo" aqui)
  expect url_contains X     expect no último step macro
  expect selector_visible X idem
  desc <texto>              descrição do último step
  undo                      descarta o último segmento gravado
  done  (ou Ctrl+D)         finalizar e mostrar preview
  cancel (ou Ctrl+C)        abortar sem gravar
  help                      mostrar esta ajuda
"""


async def main() -> int:
    ap = argparse.ArgumentParser(description="Grava planos híbridos (macro+modelo) via Playwright MCP recording")
    ap.add_argument("url", nargs="?", help="URL inicial para abrir no navegador")
    ap.add_argument("--server", default=PLAYWRIGHT_SERVER, help="nome do servidor MCP no mcp_manager")
    ap.add_argument("--mcp", default=MCP_MANAGER_URL, help="URL do mcp_manager")
    ap.add_argument("--memory", default=MEMORY_API_URL, help="URL da memory API")
    ap.add_argument("--name", help="nome do plano (pergunta no fim se omitido)")
    ap.add_argument("--description", default="", help="descrição do plano")
    args = ap.parse_args()

    recorder = Recorder(
        httpx.AsyncClient(base_url=args.mcp.rstrip("/"), timeout=CALL_TIMEOUT + 10.0),
        args.server,
    )
    memory = httpx.AsyncClient(base_url=args.memory.rstrip("/"), timeout=30.0)

    # health-check rápido dos dois serviços
    for label, client, path in (("mcp_manager", recorder.http, "/health"),
                                ("memory", memory, "/status")):
        try:
            r = await client.get(path)
            r.raise_for_status()
        except Exception as e:
            print(f"✗ {label} indisponível em {client.base_url} ({type(e).__name__}: {e})")
            await recorder.http.aclose()
            await memory.aclose()
            return 1

    if args.url:
        print(f"→ abrindo {args.url} ...")
        await recorder.navigate(args.url)

    segments: list[list[dict]] = []   # cada segmento = lista de steps macro
    model_steps: list[dict] = []      # 1:1 com as "divisórias" entre segmentos
    # layout final: seg[0], modelo[0], seg[1], modelo[1], ..., seg[n]
    # model_steps[i] vem DEPOIS de segments[i].

    print(__doc__.strip().splitlines()[-1] if False else "")
    print("Gravação interativa de plano. Digite instruções p/ o modelo,")
    print("'done' para finalizar, 'help' para comandos.\n")

    recording = False
    async def _start():
        nonlocal recording
        if not recording:
            await recorder.start_recording()
            recording = True
            print("  ● gravando... (opere o navegador; volte ao CLI para inserir steps de modelo)")

    async def _stop_segment() -> list[dict]:
        nonlocal recording
        if not recording:
            return []
        code = await recorder.stop_recording()
        recording = False
        return parse_recorded_code(code)

    await _start()

    pending_instruction: str | None = None
    try:
        while True:
            try:
                line = input("\n[plano] > ").strip()
            except EOFError:
                line = "done"
            except KeyboardInterrupt:
                print("\ncancelado — nada foi gravado")
                return 130

            if not line:
                continue

            low = line.lower()

            if low in ("done", "fim"):
                seg = await _stop_segment()
                if seg:
                    segments.append(seg)
                break

            if low in ("cancel", "sair"):
                print("cancelado — nada foi gravado")
                return 130

            if low == "help":
                print(HELP_TEXT)
                continue

            if low == "undo":
                seg = await _stop_segment()
                if segments:
                    discarded = segments.pop()
                    print(f"  ↩ segmento anterior descartado ({len(discarded)} steps)"
                          + (" — segmento atual ainda não gravado foi descartado junto" if seg else ""))
                elif seg:
                    print(f"  ↩ segmento atual descartado ({len(seg)} steps)")
                else:
                    print("  nada para desfazer")
                await _start()
                continue

            if low.startswith("expect "):
                rest = line[len("expect "):].strip()
                m = re.match(r"(url_contains|selector_visible)\s+(.+)", rest, re.DOTALL)
                if not m:
                    print("  uso: expect url_contains X | expect selector_visible X")
                    continue
                key, value = m.group(1), m.group(2).strip()
                seg = await _stop_segment()
                if seg:
                    segments.append(seg)
                # aplica no último step macro do último segmento
                if segments:
                    target = segments[-1][-1]
                    target.setdefault("expect", {})[key] = value
                    print(f"  ✓ expect anexado ao step: {step_brief(0, target).split('. ', 1)[-1]}")
                else:
                    print("  ✗ nenhum step macro ainda — nada para anexar")
                await _start()
                continue

            if low.startswith("desc "):
                text = line[len("desc "):].strip()
                if segments and segments[-1]:
                    segments[-1][-1]["description"] = text
                    print("  ✓ descrição anexada ao último step macro")
                else:
                    print("  ✗ nenhum step macro ainda")
                continue

            # qualquer outro texto = instrução para o modelo
            seg = await _stop_segment()
            if seg:
                segments.append(seg)
            model_steps.append({
                "executor": "modelo",
                "objective": line,
                "max_actions": 15,
                "description": "",
            })
            print(f"  ✓ step MODELO inserido: {line[:80]!r}")
            await _start()

    except RuntimeError as e:
        print(f"\n✗ erro do MCP recording: {e}")
        return 1

    # ── monta a lista final: seg[0], modelo[0], seg[1], modelo[1], ... ──
    steps: list[dict] = []
    for i, seg in enumerate(segments):
        steps.extend(seg)
        if i < len(model_steps):
            steps.append(model_steps[i])

    # steps modelo pendentes sem segmento anterior (ex.: instrução antes
    # de qualquer ação gravada) — model_steps sempre tem 1:1 com segments
    # exceto se o usuário der done logo após instrução sem novo segmento;
    # nesse caso o último modelo fica de fora do zip — corrige:
    while len(model_steps) > len(segments):
        steps.append(model_steps[len(segments)])
        # insere no lugar certo: após o último segmento correspondente é
        # irrelevante para o caso simples (fim do plano)
        segments.append([])  # mantém o invariante só para o loop de contagem

    if not steps:
        print("nenhum passo gravado — nada a salvar")
        return 1

    show_preview(steps)

    confirm = input("\nConfirmar e gravar? [S/n] ").strip().lower()
    if confirm in ("n", "no", "não"):
        print("descartado — nada foi gravado")
        return 130

    name = args.name
    while not name:
        name = input("Nome do plano: ").strip() or ""

    description = args.description
    if not description:
        description = input("Descrição (opcional, Enter pula): ").strip()

    try:
        res = await push_plan(memory, name, description, steps)
    except Exception as e:
        print(f"\n✗ falha ao gravar na memory: {e}")
        print("  (o JSON do plano está abaixo — você pode gravar manualmente)")
        print(json.dumps({"name": name, "description": description, "steps": steps},
                         ensure_ascii=False, indent=2))
        return 1

    print(f"✓ plano gravado: id={res.get('plan_id')}, versão {res.get('version')}")
    print(f"  execute com run_plan (plan_name={name!r}) no orchestrator")
    await recorder.http.aclose()
    await memory.aclose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
