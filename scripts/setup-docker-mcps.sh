#!/usr/bin/env bash
# Sobe os MCPs locais (CadQuery 3D e Boardview) em containers, no mesmo
# esquema do alpha-mcp: transporte streamable-http registrado no mcp_manager.
#
# Uso:
#   ./scripts/setup-docker-mcps.sh [--install-dir DIR] [--build] [--register]
#                                  [--skip-boardview] [--skip-cadquery]
#                                  [--restart] [--yes]
#
# Sem flags, faz as duas coisas (build das imagens + registro/sync no Manager).
# --restart remove os containers atuais antes de registrar (use após rebuild).
# O Manager inicia os containers sob demanda via `autostart` — não precisa
# mantê-los no docker compose. As imagens são multi-stage (sem pip/cache) e os
# processos usam HTTP stateless e carregamento lazy do CadQuery/OCP.
#
# O $HOME do host é montado no MESMO caminho dentro do container, então os
# paths absolutos que o modelo vê no WebUI continuam válidos (e os arquivos
# exportados aparecem na aba Artefatos). Nada de camada de tradução de paths.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
INSTALL_DIR="$ROOT"
MANAGER_URL="http://127.9.12.22:9001"
HOME_DIR="${HOME:-/home/felipe}"
# Exportações feitas pelo container não devem ficar root-owned no host.
HOST_UID_GID="$(id -u):$(id -g)"
BUILD=0
REGISTER=0
DO_CADQUERY=1
DO_BOARDVIEW=1
RESTART=0
YES=0

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[erro]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"
}

while (($#)); do
    case "$1" in
        --install-dir)   INSTALL_DIR="${2:?--install-dir requer um valor}"; shift 2 ;;
        --manager)       MANAGER_URL="${2:?--manager requer um valor}"; shift 2 ;;
        --home)          HOME_DIR="${2:?--home requer um valor}"; shift 2 ;;
        --build)         BUILD=1; shift ;;
        --register)      REGISTER=1; shift ;;
        --skip-boardview) DO_BOARDVIEW=0; shift ;;
        --skip-cadquery)  DO_CADQUERY=0; shift ;;
        --restart)       RESTART=1; shift ;;
        --yes|-y)        YES=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *) die "argumento desconhecido: $1 (use --help)" ;;
    esac
done

if (( ! BUILD && ! REGISTER )); then BUILD=1; REGISTER=1; fi
INSTALL_DIR="$(cd -- "$INSTALL_DIR" && pwd)"
ENV_FILE="$INSTALL_DIR/Modules/.env"
[[ -f "$ENV_FILE" ]] || die "sem $ENV_FILE — passe --install-dir apontando para a instalação."

TOKEN="$(awk -F= '/^AVA_INTERNAL_TOKEN=/{sub(/^AVA_INTERNAL_TOKEN=/,""); print; exit}' "$ENV_FILE")"
[[ -n "$TOKEN" ]] || die "AVA_INTERNAL_TOKEN ausente em $ENV_FILE"

CADQUERY_IMAGE="ava-cadquery-mcp:latest"
BOARDVIEW_IMAGE="ava-boardview-mcp:latest"
CADQUERY_URL="http://127.9.12.22:4011/mcp"
BOARDVIEW_URL="http://127.9.12.22:4012/mcp"
SCRATCH="$INSTALL_DIR/cache/mcp-autostart-data"

if (( ! YES )); then
    printf 'Build/registrar MCPs Docker (CadQuery e Boardview) no manager %s? [s/N] ' "$MANAGER_URL"
    read -r ans || ans=""
    case "$ans" in s*|S*) ;; *) info "Cancelado."; exit 0 ;; esac
fi

# ── Build ─────────────────────────────────────────────────────────────────────
if (( BUILD )); then
    command -v docker >/dev/null 2>&1 || die "docker não encontrado no PATH."
    mkdir -p "$SCRATCH"
    if (( DO_CADQUERY )); then
        info "Build da imagem $CADQUERY_IMAGE …"
        docker build -t "$CADQUERY_IMAGE" "$INSTALL_DIR/mcp_servers/3d_modeling"
        ok "Imagem CadQuery pronta."
    fi
    if (( DO_BOARDVIEW )); then
        info "Build da imagem $BOARDVIEW_IMAGE …"
        docker build -t "$BOARDVIEW_IMAGE" "$INSTALL_DIR/mcp_servers/boardview"
        ok "Imagem Boardview pronta."
    fi
fi

# ── Registro + sync no mcp_manager ────────────────────────────────────────────
if (( REGISTER )); then
    mkdir -p "$SCRATCH"
    if (( RESTART )); then
        # Força o autostart a usar a imagem nova e os limites atuais.
        for c in ava-cadquery-mcp ava-boardview-mcp; do
            if docker ps -a --format '{{.Names}}' | grep -qx -- "$c"; then
                info "Removendo container antigo $c …"
                docker rm -f "$c" >/dev/null
            fi
        done
    fi

    register_one() {
        local name="$1" payload="$2"
        info "Registrando $name no mcp_manager …"
        curl -fsS --max-time 20 -X POST "$MANAGER_URL/servers" \
            -H 'Content-Type: application/json' \
            -H "Authorization: Bearer $TOKEN" \
            -d "$payload" >/dev/null || die "falha ao registrar $name"
        # O autostart espera a TCP porta, não o handshake MCP: na primeira
        # chamada o sync pode chegar antes do /mcp aceitar requests (502).
        info "Sincronizando tools de $name …"
        local i synced=0
        for i in 1 2 3 4 5; do
            if curl -fsS --max-time 240 -X POST "$MANAGER_URL/servers/$name/sync" \
                    -H "Authorization: Bearer $TOKEN" >/dev/null 2>&1; then
                synced=1
                break
            fi
            sleep 5
        done
        (( synced )) || die "falha no sync de $name (container subiu? veja os logs do Manager)"
        ok "$name registrado e sincronizado."
    }

    if (( DO_CADQUERY )); then
        payload="$(HOME_DIR="$HOME_DIR" IMAGE="$CADQUERY_IMAGE" URL="$CADQUERY_URL" SCRATCH="$SCRATCH" HOST_UID_GID="$HOST_UID_GID" python3 - <<'PY'
import json, os
home, scratch = os.environ["HOME_DIR"], os.environ["SCRATCH"]
print(json.dumps({
    "name": "cadquery-3d-printing",
    "skill_summary": (
        "CAD paramétrico e impressão 3D (CadQuery): crie, edite, meça, valide e exporte "
        "peças em milímetros para STEP/STL/3MF com histórico reproduzível"
    ),
    "transport": "http",
    "url": os.environ["URL"],
    "autostart": {
        "enabled": True,
        "image": os.environ["IMAGE"],
        "container": "ava-cadquery-mcp",
        "ports": "127.9.12.22:4011:4011",
        "volume": scratch,
        "args": ["-v", f"{home}:{home}", "-u", os.environ["HOST_UID_GID"],
                 "--memory=768m", "--cpus=1"],
        "ready_timeout_s": 120,
    },
    "enabled": True,
    "always_on": False,
    "examples": {
        "create_primitive": ["crie uma caixa 3d", "modele um cilindro", "faça uma peça paramétrica"],
        "make_hole": ["faça um furo na peça", "furos de montagem", "fure a placa com 4 mm"],
        "boolean_operation": ["una os dois sólidos", "subtraia um bloco da peça", "corte o cilindro da caixa"],
        "export_object": ["exporte para STL", "gere um arquivo STEP", "salve a peça em 3MF", "exporte para impressão 3d"],
        "validate_printability": ["valide a peça para impressão 3d", "verifique se é imprimível"],
        "measure_objects": ["meça a peça", "confira o encaixe das peças"],
        "execute_cadquery_script": ["rode um script cadquery", "modele com código cadquery"],
    },
}))
PY
)"
        register_one cadquery-3d-printing "$payload"
    fi

    if (( DO_BOARDVIEW )); then
        payload="$(HOME_DIR="$HOME_DIR" IMAGE="$BOARDVIEW_IMAGE" URL="$BOARDVIEW_URL" SCRATCH="$SCRATCH" HOST_UID_GID="$HOST_UID_GID" python3 - <<'PY'
import json, os
home, scratch = os.environ["HOME_DIR"], os.environ["SCRATCH"]
print(json.dumps({
    "name": "boardview",
    "skill_summary": (
        "Boardview GenCAD (.cad): conectividade e geometria da placa para reparo "
        "(rails, curtos, caminhos de alimentação, vizinhança e medições do técnico)"
    ),
    "transport": "http",
    "url": os.environ["URL"],
    "autostart": {
        "enabled": True,
        "image": os.environ["IMAGE"],
        "container": "ava-boardview-mcp",
        "ports": "127.9.12.22:4012:4012",
        "volume": scratch,
        "args": ["-v", f"{home}:{home}", "-u", os.environ["HOST_UID_GID"],
                 "--memory=192m", "--cpus=1"],
        "ready_timeout_s": 120,
    },
    "enabled": True,
    "always_on": False,
    "examples": {
        "load_board": [
            "analise esse boardview", "abre essa placa no boardview",
            "diagnostica essa placa pelo .cad", "carregue o .cad da pasta",
            "analise o boardview da pasta selecionada",
        ],
        "power_rail_analysis": [
            "esse rail está sem tensão", "a placa não liga",
            "tem um curto nessa linha de alimentação",
        ],
    },
}))
PY
)"
        register_one boardview "$payload"
    fi
fi

info "MCPs Docker prontos. Os containers sobem sob demanda no primeiro uso."
