#!/bin/bash
# Remove o set -e — não queremos que um serviço derrube os outros
# set -e  ← REMOVIDO


# Baixa Supertonic se não estiver em cache
python3 -c "
import os; os.environ['HF_HUB_TIMEOUT'] = '600'
try:
    from supertonic import TTS; TTS(auto_download=True); print('✓ Supertonic pronto')
except Exception as e: print(f'⚠ Supertonic: {e}')
"
echo "🚀 Iniciando AVA..."

echo "  • Orchestrator..."
python3 /app/orchestrator.py &
sleep 1

cd /app/Modules

# Função que reinicia serviços que caem
restart_on_fail() {
    local name="$1"
    local cmd="$2"
    while true; do
        echo "  [START] $name"
        eval "$cmd"
        echo "  [WARN] $name caiu — reiniciando em 5s..."
        sleep 5
    done
}

echo "  • onnx Manager..."
restart_on_fail "onnx Manager" "python3 onnxManager.py" &

echo "  • CoT Generator..."
restart_on_fail "CoT" "python3 'CoT generator.py'" &

echo "  • Deep Search..."
restart_on_fail "Deep Search" "python3 deep_search.py" &

echo "  • LLM..."
restart_on_fail "LLM" "python3 LLM.py" &

echo "  • Memory API..."
restart_on_fail "Memory" "python3 memory.py" &

echo "  • Search API..."
restart_on_fail "Search" "python3 Search_api.py" &

echo "  • TTS API..."
restart_on_fail "TTS" "python3 TTS.py" &

echo "  • VQA..."
restart_on_fail "VQA" "python3 VQA.py" &

echo "✓ Todos os serviços iniciados"
echo ""
echo "Aguardando processos..."

wait -n  # Aguarda qualquer processo — sem set -e não derruba os outros
echo "⚠️  Um serviço saiu (normal se foi reiniciado pelo restart_on_fail)"
wait     # Aguarda os demais