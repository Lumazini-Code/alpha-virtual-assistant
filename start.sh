#!/bin/bash
# Alpha AI (AVA) — PID 1 do container — sobe todos os serviços do AVA
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
# Remove o set -e — não queremos que um serviço derrube os outros
# set -e  ← REMOVIDO

# ─────────────────────────────────────────
# CORREÇÃO: propaga SIGTERM/SIGINT do Docker para todo o grupo de
# processos. Sem isso, este script (PID 1) morre e deixa os filhos
# órfãos até o Docker mandar SIGKILL — o que costuma deixar sockets
# em TIME_WAIT/portas presas, causando falhas em cascata no boot
# seguinte (visto no log: vários serviços reiniciando juntos).
# ─────────────────────────────────────────
_shutting_down=0
export ALPHA_ENABLE_GIT=0

# ── Otimização de memória/CPU (onnxManager, TTS, vision) ──────────────────────
# MALLOC_ARENA_MAX=2 — glibc aloca 1 arena heap por thread; serviços ONNX têm
#   dezenas de threads e isso inflava o RSS (regiões anônimas fantasma).
# SUPERTONIC_INTRA_OP_THREADS=2 — o default do ORT usa TODOS os cores por
#   sessão; com workers + warmup isso causava spike de CPU no boot.
# OMP_NUM_THREADS — teto de segurança para libs que consultam OpenMP.
export MALLOC_ARENA_MAX=2
export SUPERTONIC_INTRA_OP_THREADS=2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# cwd fixo do Memory Server — isolado do resto do script (nenhum outro `cd`
# depende dele nem é afetado por ele; ver uso em `restart_on_fail` abaixo).
MEMORY_API_CWD="/app/Modules/memory"

_cleanup() {
    if [ "$_shutting_down" -eq 1 ]; then return; fi
    _shutting_down=1
    echo "🛑 Sinal de parada recebido — encerrando processos filhos..."
    kill -TERM -- -$$ 2>/dev/null
    wait
    exit 0
}
trap _cleanup SIGTERM SIGINT

# ── TTS é opt-in (AVA_TTS_ENABLED, default 0) ────────────────────────────────
# O Supertonic carrega ~470MB de pesos+arena por worker na RAM. Com a flag em
# 0 (default), o TTS.py não sobe e o pre-download é pulado — economia de
# ~500MB. Quem usa fala liga AVA_TTS_ENABLED=1 no Modules/.env.
if [ "${AVA_TTS_ENABLED:-0}" = "1" ]; then
    # Baixa Supertonic se não estiver em cache
    python3 -c "
import os; os.environ['HF_HUB_TIMEOUT'] = '600'
try:
    from supertonic import TTS; TTS(auto_download=True); print('✓ Supertonic pronto')
except Exception as e: print(f'⚠ Supertonic: {e}')
"
else
    echo "  • TTS desligado (AVA_TTS_ENABLED=0) — economia de ~500MB de RAM"
fi
echo "🚀 Iniciando AVA..."

# Função que reinicia serviços que caem.
# Exit code 78 (EX_CONFIG) = falha irrecuperável de config/dados (ex.:
# modelo ONNX corrompido detectado pelo onnxManager) — reiniciar em loop
# NÃO resolve; para e grita no log. Correção: rodar
# `python3 scripts/verify_models.py` e reiniciar o container.
restart_on_fail() {
    local name="$1"
    local cmd="$2"
    while true; do
        echo "  [START] $name"
        eval "$cmd"
        local rc=$?
        if [ "$rc" -eq 78 ]; then
            echo "  [FATAL] $name saiu com código 78 (config/dados inválidos) — SEM auto-restart."
            echo "  [FATAL] Rode python3 scripts/verify_models.py dentro do container e reinicie."
            return 78
        fi
        echo "  [WARN] $name caiu (rc=$rc) — reiniciando em 5s..."
        sleep 5
    done
}

echo "  • Orchestrator..."
# CORREÇÃO: agora protegido por restart_on_fail como todo o resto —
# antes, se caísse uma vez, nunca mais voltava.
restart_on_fail "Orchestrator" "python3 /app/orchestrator.py" &
sleep 1

# ── Modo mesclado (AVA_MERGED_SERVICES, default 1) ───────────────────────────
# O orchestrator sobe LLM (:4003) e o visualizador do grafo (:8777) em
# threads do PRÓPRIO processo — um único interpreter Python (~120-150MB
# economizados; as portas continuam as mesmas). Com a flag em 0, os dois
# voltam a ser processos standalone (abaixo) — fallback em caso de problema
# de import/config no modo mesclado.
# Graph UI continua opt-in (AVA_GRAPH_UI, default 0): é o visualizador do
# grafo (SSE + polling de 1s no SQLite), útil para debug, ~80-100MB.
if [ "${AVA_MERGED_SERVICES:-1}" = "1" ]; then
    echo "  • LLM + Graph UI mesclados no processo do orchestrator (AVA_MERGED_SERVICES=1)"
else
    # NEW (multi-escopo): --memory-api lê o escopo ativo (GET /status) e
    # publica o badge scope_switch — SEM a flag o watcher não sobe e os
    # tailers por projeto (db/<slug>/activation_log.jsonl) nunca publicam
    # ativação (bug runtime).
    if [ "${AVA_GRAPH_UI:-0}" = "1" ]; then
        restart_on_fail "Graph Activation Server" "python3 /app/graph_activation.py --memory-api" &
    else
        echo "  • Graph UI desligado (AVA_GRAPH_UI=0) — visualização em :8777 não sobe"
    fi
fi
sleep 1



cd /app/Modules

echo "  • onnx Manager..."
restart_on_fail "onnx Manager" "python3 onnxManager.py" &

# Deep Search removido — busca web agora é um servidor MCP próprio
# (mcp_servers/alpha-mcp, serviço alpha-mcp no docker-compose).
# LLM não sobe aqui no modo mesclado (default): ele roda dentro do processo
# do orchestrator — ver bloco AVA_MERGED_SERVICES acima.
if [ "${AVA_MERGED_SERVICES:-1}" = "0" ]; then
    echo "  • LLM (standalone)..."
    restart_on_fail "LLM" "python3 LLM.py" &
fi

# ── TTS (ver gate AVA_TTS_ENABLED no topo do script) ─────────────────────────
# O Supertonic carrega ~470MB/worker na RAM — por default (AVA_TTS_ENABLED=0)
# o serviço não sobe. A forma como ele foi projetado (agente totalmente
# autônomo, com fala constante) ainda não é suportada; quem usa fala liga a
# flag no Modules/.env.
if [ "${AVA_TTS_ENABLED:-0}" = "1" ]; then
    echo "  • TTS API..."
    restart_on_fail "TTS" "python3 TTS.py" &
fi

# ── Vision (gate VISION_OBJECTS_ENABLED, default 0) ──────────────────────────
# Pipeline pesado (Depth Anything V2 + EdgeSAM + DINOv3). O vision.py só faz
# sentido com VISION_OBJECTS_ENABLED=1 no orchestrator (que então oferece a
# tool vision_objects); a forma autônoma ainda não é suportada.
if [ "${VISION_OBJECTS_ENABLED:-0}" = "1" ]; then
    echo "  • Vision API..."
    restart_on_fail "Vision" "python3 vision.py" &
fi





echo "  • Memory Server..."
restart_on_fail "Memory Server" "cd '$MEMORY_API_CWD' && python3 memory_api.py" &


echo "✓ Todos os serviços iniciados"
echo ""
echo "Aguardando processos..."

wait -n
echo "⚠️  Um serviço saiu (normal se foi reiniciado pelo restart_on_fail)"
wait