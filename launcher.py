# Alpha AI (AVA) — launcher de processo único (orchestrator + memory + onnx)
# Copyright (C) 2026  Felipe lumazini
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
launcher.py — serviços "puros" do AVA em UM processo (uma thread cada)
═══════════════════════════════════════════════════════════════════════════

Sucessor do papel do start.sh p/ orchestrator/memory/onnx/LLM/graph: cada
serviço roda como uvicorn.Server em UMA thread do mesmo interpreter — uma
só cópia de libs/interpreter (~200-300MB de RAM a menos) SEM mudar o
contrato: portas e endereços continuam idênticos (TUI/tray/healthchecks
não percebem nada). Usa app OBJECT direto (nunca a forma string do
uvicorn.run) — elimina o double-import que o `python modulo.py` + string
provocava (o módulo era executado como __main__ e re-importado pelo
uvicorn, duplicando singletons/estado).

Serviços (ordem de boot importa: memory fala com onnx no startup):
  :2002 ONNX Serving   (gate AVA_MERGED_ONNX,   default 1)
  :3000 Memory API     (gate AVA_MERGED_MEMORY, default 1)
  :9000 orchestrator   (+ LLM :4003 e grafo :8777 via orchestrator
                        ._start_merged_services — gates AVA_MERGED_SERVICES
                        e AVA_GRAPH_UI, mesmos contratos do modo standalone)

TTS (:3004) e Vision (:4002) NÃO entram: nativos pesados (Supertonic/ORT)
— segfault neles derrubaria o processo único inteiro. Seguem como processos
standalone no start.sh, atrás dos mesmos gates (AVA_TTS_ENABLED,
VISION_OBJECTS_ENABLED).

Contratos preservados (validados contra o comportamento REAL do uvicorn):

rc=78 (EX_CONFIG — ex. modelo ONNX corrompido): o serviço fica FORA e os
  demais continuam — o mesmo contrato do restart_on_fail do start.sh. O
  uvicorn engole SystemExit levantada DENTRO do lifespan (testado: ele
  captura BaseException no main() do LifespanOn e sai com SystemExit(3)
  DELE, mascarando o código) — por isso _fatal_exit no módulo marca a flag
  _UNIFIED_FATAL (AVA_UNIFIED=1) e o launcher checa a flag após run().

SIGTERM/SIGINT: uvicorn só instala handlers na main thread — aqui a main
  thread é o dono dos sinais e propaga should_exit para TODOS os servers:
  shutdown graceful de uma vez (substitui o trap kill -TERM -- -$$ do
  start.sh, eliminando os sockets órfãos/TIME_WAIT que motivavam o trap).

orchestrator caiu (e não foi shutdown): launcher encerra graceful os
  demais e sai rc=1 — o restart_on_fail do start.sh recomeça o launcher
  INTEIRO. Semântica igual à de hoje (restart de processo = estado fresco:
  import de tudo de novo), ao custo de um reset dos 3 serviços juntos.

Restart de memory/onnx (exceção Python): thread re-roda o lifespan no
  MESMO interpreter — memory.startup() reatribui state.* do zero (DBs,
  engines, índices) e é idempotente; onnx recarrega os modelos. Estado de
  módulo imutável (configs) persiste — sem risco.
"""

from __future__ import annotations

import os
import signal
import socket
import sys
import threading
import time
import traceback

# ── paths — o start.sh fazia via cd; aqui é sys.path ─────────────────────────
_APP = os.path.dirname(os.path.abspath(__file__))          # /app
for _p in (_APP, os.path.join(_APP, "Modules"),
           os.path.join(_APP, "Modules", "memory")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── defaults de runtime que o start.sh exportava ─────────────────────────────
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("SUPERTONIC_INTRA_OP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")
# Canal p/ os módulos saberem que estão no processo único (onnxManager:
# _fatal_exit levanta SystemExit + flag em vez de os._exit; orchestrator:
# _fatal_merged idem). O launcher é o único que seta isto.
os.environ["AVA_UNIFIED"] = "1"

import uvicorn  # noqa: E402  (após os paths)

EXIT_CONFIG = 78          # EX_CONFIG — mesma semântica do restart_on_fail
_RESTART_DELAY = 5        # "reiniciando em 5s..." — mesma cadência do start.sh
_WAIT_ONNX_PORT = 30      # memory fala com onnx no startup (check_health)

_servers: dict[str, uvicorn.Server] = {}
_shutting_down = False
_orch_dead = threading.Event()


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "0", "false", "no", "off")


def _shutdown_all() -> None:
    """should_exit em todos os servers — chamado pelo handler de sinal (main
    thread) e pelo caminho de morte do orchestrator."""
    global _shutting_down
    _shutting_down = True
    for srv in list(_servers.values()):
        try:
            srv.should_exit = True
        except Exception:
            pass


def _signal_handler(signum, frame) -> None:  # noqa: ARG001 (assinatura API)
    try:
        sig = signal.Signals(signum).name
    except Exception:
        sig = str(signum)
    print(f"🛑 {sig} — shutdown graceful de todos os servers...", flush=True)
    _shutdown_all()


def _wait_port(host: str, port: int, timeout: float) -> None:
    """Espera o serviço abrir a porta (boot-order: memory → onnx). Cede cedo
    se o worker do onnx morreu (78) — memory sobe degradada por design dela
    (check_health só loga warning)."""
    deadline = time.time() + timeout
    while time.time() < deadline and not _shutting_down:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.25)


def _worker(nome: str, factory, fatal_module=None) -> None:
    """Thread de serviço com loop de restart. factory() cria um Server NOVO a
    cada iteração (o objeto Server não é reusável pós-run). fatal_module é o
    módulo dono do serviço quando ele tem a semântica rc=78 (_UNIFIED_FATAL)."""
    global _shutting_down
    while not _shutting_down:
        if fatal_module is not None and hasattr(fatal_module, "_UNIFIED_FATAL"):
            fatal_module._UNIFIED_FATAL = False        # reset p/ esta tentativa
        try:
            server = factory()
            _servers[nome] = server
            server.run()
            if _shutting_down:
                return
            print(f"  [WARN] {nome} saiu — reiniciando em {_RESTART_DELAY}s...",
                  flush=True)
        except SystemExit as e:
            # uvicorn normaliza QUALQUER falha de lifespan p/ SystemExit(3)
            # dele — o verdadeiro 78 só se distingue pela flag do módulo.
            if _shutting_down:
                return
            traceback.print_exc()
            print(f"  [WARN] {nome} caiu (rc={e.code}) — reiniciando em "
                  f"{_RESTART_DELAY}s...", flush=True)
        except BaseException:
            if _shutting_down:
                return
            traceback.print_exc()
            print(f"  [WARN] {nome} caiu — reiniciando em {_RESTART_DELAY}s...",
                  flush=True)
        # rc=78 (flag marcada por _fatal_exit do módulo): fica FORA, os demais
        # continuam — reiniciar em loop não resolve config/dados quebrados.
        if (fatal_module is not None
                and getattr(fatal_module, "_UNIFIED_FATAL", False)):
            print(f"  [FATAL] {nome}: config/dados inválidos (78) — SEM "
                  "auto-restart. Rode scripts/verify_models.py e reinicie o "
                  "container. Demais serviços continuam de pé.", flush=True)
            return
        _servers.pop(nome, None)
        time.sleep(_RESTART_DELAY)


# ── factories — app OBJECT direto de cada módulo (sem double-import) ────────

def _onnx_module():
    import onnxManager
    return onnxManager


def _onnx_factory():
    m = _onnx_module()
    return uvicorn.Server(uvicorn.Config(
        m.app, host=m.SERVER_HOST, port=m.SERVER_PORT, log_level="info"))


def _memory_factory():
    import memory_api
    return uvicorn.Server(uvicorn.Config(
        memory_api.app, host=memory_api.REST_HOST, port=memory_api.REST_PORT,
        log_level="info"))


def _orchestrator_factory():
    import orchestrator
    from Modules.internal_auth import internal_bind_host
    return uvicorn.Server(uvicorn.Config(
        orchestrator.app, host=internal_bind_host(), port=9000,
        log_level="info"))


def _orchestrator_worker() -> None:
    """Orchestrator NÃO tem loop de restart: qualquer saída (que não seja o
    shutdown de sinal) encerra o processo único inteiro — restart de processo
    implica estado fresco, e orchestrator quebrado em loop dentro de um
    processo vivo deixaria memory/onnx presos num processo sem dono."""
    try:
        server = _orchestrator_factory()
        _servers["Orchestrator"] = server
        server.run()
        if not _shutting_down:
            print("  [FATAL] Orchestrator saiu — encerrando o processo único "
                  "(restart_on_fail do start.sh recomeça tudo, estado fresco).",
                  flush=True)
            _orch_dead.set()
    except BaseException:
        if not _shutting_down:
            traceback.print_exc()
            print("  [FATAL] Orchestrator caiu — encerrando o processo único.",
                  flush=True)
            _orch_dead.set()


def main() -> None:
    global _shutting_down
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    print("🚀 Iniciando AVA (processo único — launcher.py)...", flush=True)
    threads: list[threading.Thread] = []

    # ── ONNX primeiro (memory fala com ele no startup: check_health/embeds) ──
    if _env_flag("AVA_MERGED_ONNX"):
        onnx_mod = _onnx_module()
        t = threading.Thread(target=_worker, name="uvicorn-onnx",
                             args=("onnx Manager", _onnx_factory, onnx_mod),
                             daemon=True)
        t.start()
        threads.append(t)
        _wait_port(onnx_mod.SERVER_HOST, onnx_mod.SERVER_PORT, _WAIT_ONNX_PORT)
    else:
        print("  • onnx Manager standalone (AVA_MERGED_ONNX=0) — start.sh sobe")

    # ── Memory (lifespan → mem.startup(): SQLite/LanceDB/engines/jobs) ───────
    if _env_flag("AVA_MERGED_MEMORY"):
        t = threading.Thread(target=_worker, name="uvicorn-memory",
                             args=("Memory Server", _memory_factory, None),
                             daemon=True)
        t.start()
        threads.append(t)
    else:
        print("  • Memory Server standalone (AVA_MERGED_MEMORY=0) — start.sh sobe")

    # ── LLM (:4003) + grafo (:8777) mesclados pelo PRÓPRIO orchestrator ─────
    # (_start_merged_services respeita AVA_MERGED_SERVICES/AVA_GRAPH_UI e é o
    # código que já roda em produção no modo standalone do orchestrator.)
    try:
        import orchestrator
    except BaseException:
        traceback.print_exc()
        print("  [FATAL] orchestrator não importou — nada sobe; start.sh "
              "vai reiniciar o launcher.", flush=True)
        raise

    if orchestrator._merged_enabled():
        try:
            orchestrator._start_merged_services()
        except SystemExit as e:
            # _fatal_merged → rc=78: LLM/grafo ficam FORA (sem auto-restart,
            # contrato restart_on_fail) — orchestrator/memory/onnx continuam.
            print(f"  [FATAL] serviços mesclados: config/dados inválidos "
                  f"(rc={e.code}) — sem LLM/grafo, demais serviços continuam.",
                  flush=True)
    else:
        print("  • LLM/grafo standalone (AVA_MERGED_SERVICES=0) — start.sh sobe")

    # ── Orchestrator — por último (a TUI espera por ele) ─────────────────────
    t = threading.Thread(target=_orchestrator_worker, name="uvicorn-orchestrator",
                         daemon=True)
    t.start()
    threads.append(t)

    print("✓ Todos os serviços iniciados (processo único)", flush=True)

    # ── Main thread: dona dos sinais — espera shutdown ou morte do orchestrator
    while not _shutting_down and not _orch_dead.is_set():
        time.sleep(0.5)
    if not _shutting_down:
        _shutdown_all()
    for t in threads:
        t.join()
    if _orch_dead.is_set() and not _shutting_down:
        sys.exit(1)   # rc≠0 → restart_on_fail recomeça o launcher (estado fresco)
    print("✓ Encerrado.", flush=True)


if __name__ == "__main__":
    main()
