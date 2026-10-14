#!/bin/bash
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

# Baixa Supertonic se não estiver em cache
python3 -c "
import os; os.environ['HF_HUB_TIMEOUT'] = '600'
try:
    from supertonic import TTS; TTS(auto_download=True); print('✓ Supertonic pronto')
except Exception as e: print(f'⚠ Supertonic: {e}')
"
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

restart_on_fail "Graph Activation Server" "python3 /app/graph_activation.py" &
sleep 1



cd /app/Modules

echo "  • onnx Manager..."
restart_on_fail "onnx Manager" "python3 onnxManager.py" &

# Deep Search removido — busca web agora é um servidor MCP próprio
# (mcp_servers/websearch, serviço websearch-mcp no docker-compose).

echo "  • LLM..."
restart_on_fail "LLM" "python3 LLM.py" &

# Desativando o TTS por enquanto — a forma com que ele foi projetado para ser usado (agente totalmente outônomo, com fala constante) ainda não é suportada.

#echo "  • TTS API..."
#restart_on_fail "TTS" "python3 TTS.py" &

# Desativando o Vision API por enquanto — A forma com que ele foi projetado para ser usado (agente totalmente outônomo, com visão constante) ainda não é suportada.

#echo "  • Vision API..."
#restart_on_fail "Vision" "python3 vision.py" &




echo "  • Memory Server..."
restart_on_fail "Memory Server" "cd '$MEMORY_API_CWD' && python3 memory_api.py" &


echo "✓ Todos os serviços iniciados"
echo ""
echo "Aguardando processos..."

wait -n
echo "⚠️  Um serviço saiu (normal se foi reiniciado pelo restart_on_fail)"
wait