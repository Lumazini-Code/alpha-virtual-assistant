#!/usr/bin/env bash
# install.sh — Instalador Alpha AI (Linux)
#
# Fluxo:
#   1. Valida prereqs (Docker + compose v2, curl, unzip, x86_64)
#   2. Baixa o asset `*linux*.zip` do último release no GitHub
#   3. Extrai em ~/.local/opt/alpha-ai (override: ALPHA_INSTALL_DIR ou --install-dir)
#   4. Extrai Modules/Models.zip em Modules/Models (e apaga o zip, salvo --keep-models-zip)
#   5. Pede a OPENROUTER_API_KEY e grava em Modules/.env (chmod 600) — ANTES do up,
#      porque Modules/LLM.py lê a chave no import
#   6. Build + up do stack (docker-start.sh) e espera o /status responder
#   7. Symlink do comando `alpha-ai` em ~/.local/bin
#   8. Auto-start do ava-tray via XDG autostart (+ inicia agora se houver GUI)
#   9. Inicia a TUI (se o terminal for interativo)
#
# Reinstalação é idempotente: sobrescreve arquivos, preserva Modules/.env com
# chave válida e `up -d` não derruba containers já sadios.

set -euo pipefail

REPO="Lumazini-Code/alpha-virtual-assistant"
DEFAULT_INSTALL_DIR="$HOME/.local/opt/alpha-ai"

# ── Re-execução a partir de cópia em /tmp ─────────────────────────────────────
# O unzip abaixo sobrescreve install.sh no diretório de destino; se o usuário
# rodou o script a partir de lá, o bash leria um arquivo mudando sob os pés.
# Copiamos o script para /tmp e re-executamos antes de qualquer coisa.
if [ -z "${_ALPHA_INSTALL_REEXEC:-}" ]; then
  _SELF_TMP="$(mktemp -t alpha-install-XXXXXX.sh)"
  cp -- "$0" "$_SELF_TMP"
  chmod +x "$_SELF_TMP"
  # export: o processo re-executado precisa herdá-la (o trap do EXIT a usa)
  _SELF_TMP="$_SELF_TMP" _ALPHA_INSTALL_REEXEC=1 export _SELF_TMP _ALPHA_INSTALL_REEXEC
  exec bash "$_SELF_TMP" "$@"
fi

# ── Flags ─────────────────────────────────────────────────────────────────────
KEEP_MODELS_ZIP=0
SKIP_BUILD=0
NO_LAUNCH=0
INSTALL_DIR="${ALPHA_INSTALL_DIR:-$DEFAULT_INSTALL_DIR}"

usage() {
  cat <<EOF
Uso: install.sh [opções]

  --install-dir DIR   Pasta de instalação (padrão: $DEFAULT_INSTALL_DIR)
  --keep-models-zip   Não apaga Modules/Models.zip após extrair
  --skip-build        Pula o build/up do Docker (útil para reinstalar)
  --no-launch         Não abre a TUI ao final
  -h, --help          Esta ajuda
EOF
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --install-dir)   INSTALL_DIR="$2"; shift 2 ;;
    --keep-models-zip) KEEP_MODELS_ZIP=1; shift ;;
    --skip-build)    SKIP_BUILD=1; shift ;;
    --no-launch)     NO_LAUNCH=1; shift ;;
    -h|--help)       usage ;;
    *) echo "Opção desconhecida: $1"; usage ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
info()  { printf '[INFO] %s\n' "$*"; }
ok()    { printf '[ OK ] %s\n' "$*"; }
warn()  { printf '[AVISO] %s\n' "$*"; }
die()   { printf '[ERRO] %s\n' "$*" >&2; exit 1; }

# ── 1. Prereqs ────────────────────────────────────────────────────────────────
info "Validando pré-requisitos..."

[ "$(uname -m)" = "x86_64" ] || die "Arquitetura $(uname -m) não suportada (binários são x86_64)."

for cmd in curl unzip docker; do
  command -v "$cmd" >/dev/null 2>&1 || die "'$cmd' não encontrado. Instale-o e rode novamente."
done

if ! docker compose version >/dev/null 2>&1; then
  die "Plugin 'docker compose' (v2) ausente.
       Instale com: sudo apt install docker-compose-plugin
       (ou https://docs.docker.com/engine/install/ubuntu/#install-the-plugin-manually)"
fi

if ! docker info >/dev/null 2>&1; then
  die "Docker indisponível (daemon parado ou sem permissão).
       - Daemon parado:      sudo systemctl start docker
       - Permissão:          sudo usermod -aG docker \$USER  (depois: newgrp docker ou relogin)"
fi

ok "Pré-requisitos validados."

# ── 2. Download do último release ─────────────────────────────────────────────
TMP_DIR="$(mktemp -d -t alpha-install-XXXXXX)"
trap 'rm -rf "$TMP_DIR" "$_SELF_TMP"' EXIT

info "Consultando último release de https://github.com/$REPO..."

API_URL="https://api.github.com/repos/$REPO/releases/latest"
RELEASE_JSON="$(curl -fsSL --max-time 30 "$API_URL" 2>/dev/null)" || \
  die "Falha ao consultar a API do GitHub (sem releases publicados? sem internet?)"

TAG="$(printf '%s' "$RELEASE_JSON" | grep -oE '"tag_name": *"[^"]+"' | head -1 | cut -d'"' -f4)" || true
[ -n "$TAG" ] || die "Não encontrei tag_name no payload do release."

ASSET_URL="$(printf '%s' "$RELEASE_JSON" | grep -oE '"browser_download_url": *"[^"]+"' | cut -d'"' -f4 | grep -i 'linux' | grep -i '\.zip$' | head -1)" || true
[ -n "$ASSET_URL" ] || die "Release $TAG não tem asset *linux*.zip. Verifique: https://github.com/$REPO/releases"

info "Release $TAG — baixando $(basename "$ASSET_URL")..."

curl -fL --connect-timeout 15 --progress-bar -o "$TMP_DIR/alpha-ai.zip" "$ASSET_URL" || \
  die "Download falhou. Verifique a conexão e tente de novo."

ok "Download concluído."

# ── 3. Extração ───────────────────────────────────────────────────────────────
info "Extraindo em $INSTALL_DIR ..."
mkdir -p "$INSTALL_DIR"
unzip -oq "$TMP_DIR/alpha-ai.zip" -d "$INSTALL_DIR"

[ -f "$INSTALL_DIR/Modules/Models.zip" ] || \
  die "Modules/Models.zip não veio no zip do release — asset incompatível com esta versão do instalador."

cd "$INSTALL_DIR"

chmod +x tui/alpha-ai Manager/ava-tray docker-start.sh 2>/dev/null || true
ok "Arquivos extraídos."

# ── 4. Modelos ────────────────────────────────────────────────────────────────
info "Extraindo Models.zip (~1,4 GB) para Modules/Models ..."
unzip -oq Modules/Models.zip -d Modules/Models && \
  ok "Modelos extraídos."

if [ "$KEEP_MODELS_ZIP" -eq 0 ]; then
  rm -f Modules/Models.zip
  ok "Modules/Models.zip removido (economia de ~450 MB)."
fi

# ── 4b. Verificação de modelos ───────────────────────────────────────────────
# Modelo truncado (download/extracão interrompida) deixa o onnxManager fora
# do ar (INVALID_PROTOBUF). O verify confere tamanho mínimo e re-baixa o que
# estiver corrompido. Best-effort: sem python no host, a validação acontece
# dentro do container no boot (exit 78 + instrução no log).
VERIFY="scripts/verify_models.py"
if [ -f "$VERIFY" ] && command -v python3 >/dev/null 2>&1; then
  info "Verificando integridade dos modelos (verify_models)..."
  if python3 "$VERIFY" --quiet; then
    ok "Modelos íntegros."
  else
    warn "Modelo inválido detectado — tentando re-download..."
    if python3 "$VERIFY" --download; then
      ok "Modelos reparados."
    else
      warn "Não foi possível reparar os modelos automaticamente — o boot do container vai indicar o arquivo exato."
    fi
  fi
else
  warn "python3/scripts indisponíveis no host — verificação de modelos será feita dentro do container no boot."
fi

# ── 5. API key (antes do up: LLM.py lê no import) ─────────────────────────────
ENV_FILE="Modules/.env"

existing_key() {
  [ -f "$ENV_FILE" ] && grep -E '^OPENROUTER_API_KEY=.+' "$ENV_FILE" >/dev/null 2>&1
}

write_key() {
  local key="$1"
  printf 'OPENROUTER_API_KEY=%s\n' "$key" > "$ENV_FILE"
  chmod 600 "$ENV_FILE"
}

prompt_key() {
  local key
  while :; do
    printf 'Cole sua OPENROUTER_API_KEY (https://openrouter.ai/keys): '
    if ! read -rs key; then
      die "Entrada encerrada (EOF) — instalação abortada na etapa da API key."
    fi
    printf '\n'
    # trim
    key="$(printf '%s' "$key" | tr -d '[:space:]')"
    [ -n "$key" ] || { warn "Chave vazia — tente de novo."; continue; }
    case "$key" in
      sk-or-*) ;;
      *) warn "Chaves do OpenRouter normalmente começam com 'sk-or-'. Prosseguindo mesmo assim." ;;
    esac
    write_key "$key"
    return 0
  done
}

if existing_key; then
  printf 'Já existe uma OPENROUTER_API_KEY em Modules/.env.\n'
  printf 'Manter a chave existente? [S/n] '
  read -r ans || ans=""
  case "$ans" in
    n|N*)
      prompt_key || true
      ;;
  esac
else
  prompt_key || true
fi

ok "OPENROUTER_API_KEY configurada em $ENV_FILE (chmod 600)."

# ── 5b. Registro do Playwright MCP (dependência direta do sistema) ────────────
# Cadastra o @playwright/mcp no mcp_manager (:4008) logo após o `up` do stack.
# O POST /servers é upsert idempotente e o /sync tem no-op por doc_hash —
# seguro re-executar. Tolerante a falha: se o manager ainda estiver bootando,
# retenta; se falhar de vez, só avisa (não aborta a instalação).
MCP_MANAGER_URL="http://localhost:4008"

register_playwright_mcp() {
  local payload
  payload='{
  "name": "playwright",
  "skill_summary": "Automação de navegador (navegar, clicar, extrair conteúdo)",
  "transport": "stdio",
  "command": "npx",
  "args": ["-y", "@playwright/mcp@latest", "--cdp-endpoint", "http://localhost:9222"],
  "env": {"DISPLAY": ":0"}
}'

  info "Registrando Playwright MCP no mcp_manager ($MCP_MANAGER_URL)..."
  local registered=0 i
  for i in $(seq 1 24); do
    if curl -fsS --max-time 10 -X POST "$MCP_MANAGER_URL/servers" \
         -H 'Content-Type: application/json' -d "$payload" >/dev/null 2>&1; then
      registered=1
      break
    fi
    sleep 5
  done
  if [ "$registered" -ne 1 ]; then
    warn "Não consegui cadastrar o Playwright MCP — manager não respondeu em 2 min."
    warn "Cadastre manualmente depois (dois comandos):"
    printf '  curl -X POST %s/servers -H '"'"'Content-Type: application/json'"'"' -d '"'"'%s'"'"'\n' "$MCP_MANAGER_URL" "$payload"
    printf '  curl -X POST %s/servers/playwright/sync\n' "$MCP_MANAGER_URL"
    return 1
  fi

  # /sync: exige a memory_api (:3000) de pé; o 1º sync baixa o pacote npm.
  local synced=0
  for i in $(seq 1 12); do
    if curl -fsS --max-time 240 -X POST "$MCP_MANAGER_URL/servers/playwright/sync" >/dev/null 2>&1; then
      synced=1
      break
    fi
    sleep 10
  done
  if [ "$synced" -ne 1 ]; then
    warn "Playwright MCP cadastrado, mas o sync falhou (memory_api ainda bootando?)."
    warn "Rode manualmente: curl -X POST $MCP_MANAGER_URL/servers/playwright/sync"
    return 1
  fi
  ok "Playwright MCP registrado e sincronizado."
}

# ── 6. Build + first up ───────────────────────────────────────────────────────
if [ "$SKIP_BUILD" -eq 0 ]; then
  info "Build inicial das imagens Docker (pode levar 5-15 min na primeira vez)..."
  ./docker-start.sh build || die "docker compose build falhou — rode ./docker-start.sh build para ver o erro."

  info "Subindo o stack..."
  ./docker-start.sh up -d || die "docker compose up falhou."

  # Registro do Playwright MCP — logo após o up, quando o stack já está carregado.
  register_playwright_mcp || true

  # Espera o /status do orchestrator (1º boot baixa Supertonic do HF)
  STATUS_URL="http://localhost:9000/status"
  MAX_WAIT=1200   # 20 min
  ELAPSED=0
  info "Aguardando orchestrator em $STATUS_URL (primeiro boot pode demorar)..."
  while [ "$ELAPSED" -lt "$MAX_WAIT" ]; do
    if curl -fsS --max-time 5 "$STATUS_URL" >/dev/null 2>&1; then
      ok "Orchestrator respondeu após $((ELAPSED / 60)) min."
      break
    fi
    if [ $((ELAPSED % 60)) -eq 0 ] && [ "$ELAPSED" -gt 0 ]; then
      printf '  ...aguardando há %s min (logs: ./docker-start.sh logs -f ava)\n' "$((ELAPSED / 60))"
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
  done

  if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    warn "Orchestrator não respondeu em 20 min. Veja os logs: ./docker-start.sh logs -f ava"
  fi

  # Detecta chave ausente no /status (LLM.py expõe openrouter_key: "missing")
  if curl -fsS --max-time 5 "$STATUS_URL" 2>/dev/null | grep -q '"openrouter_key": *"missing"'; then
    warn "O orchestrator reportou openrouter_key: missing."
    printf 'Deseja re-informar a chave agora? [S/n] '
    read -r ans || ans=""
    case "$ans" in
      n|N*) ;;
      *)
        prompt_key || true
        info "Reiniciando container ava para reler a chave..."
        ./docker-start.sh restart ava || warn "restart falhou — rode manualmente: ./docker-start.sh restart ava"
        ;;
    esac
  fi
else
  info "--skip-build: pulei o build/up do Docker."
fi

# ── 7. Comando do terminal ────────────────────────────────────────────────────
mkdir -p "$HOME/.local/bin"
ln -sf "$INSTALL_DIR/tui/alpha-ai" "$HOME/.local/bin/alpha-ai"
ok "Comando instalado: alpha-ai -> $INSTALL_DIR/tui/alpha-ai"

case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *)
    warn "~/.local/bin não está no PATH."
    printf 'Adicioná-lo ao ~/.bashrc? [S/n] '
    read -r ans || ans=""
    case "$ans" in
      n|N*) ;;
      *)
        {
          printf '\n# Alpha AI (instalador)\n'
          printf 'export PATH="$HOME/.local/bin:$PATH"\n'
        } >> "$HOME/.bashrc"
        ok "Adicionado ao ~/.bashrc — reabra o terminal ou rode: export PATH=\"\$HOME/.local/bin:\$PATH\""
        ;;
    esac
    ;;
esac

# ── 8. Auto-start do tray ─────────────────────────────────────────────────────
AUTOSTART_DIR="$HOME/.config/autostart"
mkdir -p "$AUTOSTART_DIR"
cat > "$AUTOSTART_DIR/alpha-ai-tray.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Alpha AI Tray
Comment=Gerenciador Docker do Alpha AI (bandeja do sistema)
Exec=$INSTALL_DIR/Manager/ava-tray
Terminal=false
StartupNotify=false
X-GNOME-Autostart-enabled=true
EOF
ok "Auto-start configurado: $AUTOSTART_DIR/alpha-ai-tray.desktop"

if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
  nohup "$INSTALL_DIR/Manager/ava-tray" >/dev/null 2>&1 &
  disown || true
  ok "ava-tray iniciado agora (ícone na bandeja)."
else
  info "Sem sessão gráfica — o tray vai subir no próximo login."
fi

# ── 9. Encerramento ───────────────────────────────────────────────────────────
echo
ok "Instalação concluída ($INSTALL_DIR)"
info "Comandos úteis:"
info "  alpha-ai                          # abre a TUI"
info "  $INSTALL_DIR/docker-start.sh logs -f ava   # logs do backend"
info "  $INSTALL_DIR/uninstall.sh         # desinstalação"

if [ "$NO_LAUNCH" -eq 0 ] && [ -t 0 ]; then
  info "Abrindo a TUI..."
  exec "$HOME/.local/bin/alpha-ai"
fi
