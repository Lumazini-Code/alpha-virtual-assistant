#!/usr/bin/env bash
# install.sh — Instalador Alpha AI (Linux)
#
# Fluxo:
#   1. Valida/instala prereqs (Node 24, Docker Engine + compose v2, curl, unzip, x86_64)
#   2. Baixa o asset `*linux*.zip` do último release no GitHub
#   3. Extrai em ~/.local/opt/alpha-ai (override: ALPHA_INSTALL_DIR ou --install-dir)
#   4. Extrai Modules/Models.zip em Modules/Models (e apaga o zip, salvo --keep-models-zip)
#   5. Pede a OPENROUTER_API_KEY e grava em Modules/.env (chmod 600) — ANTES do up,
#      porque Modules/LLM.py lê a chave no import
#   6. Build das imagens; inicia o Manager (ava-tray, API :9001) e pede a ele o
#      start do Docker (POST /docker/start) + garante o mcp_manager (/mcp/start);
#      registra os MCPs e espera o /status do orchestrator responder
#      (sem GUI/Manager: cai para docker-start.sh up -d)
#   7. Symlink do comando `alpha-ai` em ~/.local/bin
#   8. Auto-start do ava-tray via XDG autostart (+ inicia agora se ainda não estiver no ar)
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

# Gerenciador de pacotes da distro (usado para instalar o que faltar)
PKG=""
for c in apt-get dnf dnf5 yum pacman; do
  if command -v "$c" >/dev/null 2>&1; then PKG="$c"; break; fi
done

# Executa um comando como root (via sudo, salvo se já for root)
as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    die "Preciso de root para '$*' e não há sudo. Rode o instalador como root (ou instale o sudo)."
  fi
}

# 1a. curl e unzip (necessários para baixar/instalar o resto)
for cmd in curl unzip; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    info "'$cmd' não encontrado — instalando..."
    [ -n "$PKG" ] || die "'$cmd' não encontrado e não identifiquei o gerenciador de pacotes. Instale-o manualmente e rode novamente."
    case "$PKG" in
      apt-get) as_root apt-get update -qq && as_root apt-get install -y -qq "$cmd" ;;
      dnf|dnf5) as_root "$PKG" install -y "$cmd" ;;
      yum)      as_root yum install -y "$cmd" ;;
      pacman)   as_root pacman -S --noconfirm --needed "$cmd" ;;
    esac
    command -v "$cmd" >/dev/null 2>&1 || die "Falha ao instalar '$cmd'. Instale-o manualmente e rode novamente."
  fi
done
ok "curl e unzip disponíveis."

# 1b. Docker Engine (+ compose v2) — instala automaticamente se ausente
# Testa o daemon: direto (usuário já no grupo docker) ou via sg (grupo
# recém-adicionado, ainda não ativo na sessão atual).
docker_alive() {
  docker info >/dev/null 2>&1 && return 0
  [ -S /var/run/docker.sock ] && command -v sg >/dev/null 2>&1 && \
    sg docker -c "docker info" >/dev/null 2>&1
}

install_docker() {
  info "Docker não encontrado — instalando Docker Engine automaticamente..."
  info "(vai pedir a senha do sudo; pode levar alguns minutos)"
  case "$PKG" in
    apt-get|dnf|dnf5|yum)
      # get.docker.com adiciona o repositório oficial da distro e instala
      # docker-ce + plugin compose v2 — funciona em Debian/Ubuntu/Fedora/RHEL.
      curl -fsSL https://get.docker.com | as_root sh || \
        die "Instalação do Docker Engine falhou. Instale manualmente: https://docs.docker.com/engine/install/"
      ;;
    pacman)
      as_root pacman -Sy --noconfirm --needed docker docker-compose || \
        die "Instalação do Docker Engine falhou. Instale manualmente: sudo pacman -S docker docker-compose"
      ;;
    *)
      die "Docker não encontrado e não identifiquei o gerenciador de pacotes.
           Instale o Docker Engine manualmente: https://docs.docker.com/engine/install/"
      ;;
  esac
  # Serviço no boot + usuário atual no grupo docker (sem exigir relogin:
  # docker_alive cobre via sg)
  if command -v systemctl >/dev/null 2>&1; then
    as_root systemctl enable --now docker 2>/dev/null || true
  else
    as_root service docker start 2>/dev/null || true
  fi
  if [ "$(id -u)" -ne 0 ]; then
    as_root usermod -aG docker "$(id -un)" 2>/dev/null || \
      warn "Não consegui adicionar $(id -un) ao grupo docker (rode: sudo usermod -aG docker \$USER)."
  fi
  ok "Docker Engine instalado."
}

if command -v docker >/dev/null 2>&1; then
  # Docker já existe — só garante o plugin compose v2
  if ! docker compose version >/dev/null 2>&1; then
    info "Plugin 'docker compose' (v2) ausente — instalando..."
    case "$PKG" in
      apt-get|dnf|dnf5|yum)
        curl -fsSL https://get.docker.com | as_root sh || \
          die "Falha ao instalar docker-compose-plugin.
               Manual: sudo apt install docker-compose-plugin (Ubuntu/Debian)"
        ;;
      pacman)
        as_root pacman -S --noconfirm --needed docker-compose || \
          die "Falha ao instalar docker-compose."
        ;;
      *)
        die "Plugin 'docker compose' (v2) ausente.
             Instale com: sudo apt install docker-compose-plugin
             (ou https://docs.docker.com/engine/install/ubuntu/#install-the-plugin-manually)"
        ;;
    esac
    docker compose version >/dev/null 2>&1 || die "Plugin 'docker compose' (v2) continua ausente."
  fi
  ok "Docker + compose v2 disponíveis."
else
  install_docker
fi

if ! docker_alive; then
  die "Docker indisponível (daemon parado ou sem permissão).
       - Daemon parado:      sudo systemctl start docker
       - Permissão:          sudo usermod -aG docker \$USER  (depois: relogin)"
fi
ok "Daemon Docker respondendo."

# 1c. Node.js no host — os MCPs Playwright/Desktop Commander são spawnados
# pelo mcp_manager do host via `npx`; sem node/npm o registro falha.
node_major() {
  node --version 2>/dev/null | sed 's/^v//' | cut -d. -f1
}

if command -v node >/dev/null 2>&1; then
  if [ "$(node_major)" -ge 24 ] 2>/dev/null; then
    ok "Node $(node --version) disponível no host."
  else
    warn "Node $(node --version) é mais antigo que a v24 (recomendado para os MCPs)."
  fi
else
  info "Node.js não encontrado no host — instalando Node 24 (NodeSource)..."
  case "$PKG" in
    apt-get)
      curl -fsSL https://deb.nodesource.com/setup_24.x | as_root bash - && \
        as_root apt-get install -y -qq nodejs || \
        die "Falha ao instalar o Node.js. Instale manualmente: https://nodejs.org (v24+)"
      ;;
    dnf|dnf5)
      curl -fsSL https://rpm.nodesource.com/setup_24.x | as_root bash - && \
        as_root "$PKG" install -y nodejs || \
        die "Falha ao instalar o Node.js. Instale manualmente: https://nodejs.org (v24+)"
      ;;
    yum)
      curl -fsSL https://rpm.nodesource.com/setup_24.x | as_root bash - && \
        as_root yum install -y nodejs || \
        die "Falha ao instalar o Node.js. Instale manualmente: https://nodejs.org (v24+)"
      ;;
    pacman)
      # Arch não tem NodeSource — o repo oficial traz a versão mais recente
      as_root pacman -S --noconfirm --needed nodejs npm || \
        die "Falha ao instalar o Node.js. Instale manualmente: sudo pacman -S nodejs npm"
      ;;
    *)
      die "Node.js não encontrado e não identifiquei o gerenciador de pacotes.
           Instale manualmente: https://nodejs.org (v24+)"
      ;;
  esac
  command -v node >/dev/null 2>&1 || \
    die "Node.js não ficou disponível após a instalação. Instale manualmente: https://nodejs.org (v24+)"
  ok "Node $(node --version) instalado no host."
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

# Lê o último valor de uma variável do .env (vazio se ausente)
get_env_var() {
  [ -f "$ENV_FILE" ] || return 0
  grep -E "^${1}=" "$ENV_FILE" | tail -1 | cut -d= -f2- || true
}

# Define/substitui uma variável no .env sem duplicar nem apagar as demais
set_env_var() {
  local k="$1" v="$2" tmp
  tmp="$(mktemp)"
  { [ -f "$ENV_FILE" ] && grep -vE "^${k}=" "$ENV_FILE"; } > "$tmp" || true
  printf '%s=%s\n' "$k" "$v" >> "$tmp"
  cat "$tmp" > "$ENV_FILE"
  rm -f "$tmp"
  chmod 600 "$ENV_FILE"
}

existing_key() {
  [ -f "$ENV_FILE" ] && grep -E '^OPENROUTER_API_KEY=.+' "$ENV_FILE" >/dev/null 2>&1
}

write_key() {
  local key="$1"
  set_env_var OPENROUTER_API_KEY "$key"
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

# ── 5a. Token de autenticação interna (AVA_INTERNAL_TOKEN) ───────────────────
# Todos os serviços internos (orchestrator, LLM, TTS, vision, memory, mcp_manager, Manager)
# escutam em 127.9.12.22 e exigem `Authorization: Bearer <AVA_INTERNAL_TOKEN>`.

existing_token() {
  [ -f "$ENV_FILE" ] && grep -E '^AVA_INTERNAL_TOKEN=.+' "$ENV_FILE" >/dev/null 2>&1
}

gen_token() {
  openssl rand -hex 32 2>/dev/null || python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
}

if existing_token; then
  printf 'Já existe um AVA_INTERNAL_TOKEN em Modules/.env.\n'
  printf 'Manter o token existente? [S/n] '
  read -r ans || ans=""
  case "$ans" in
    n|N*) set_env_var AVA_INTERNAL_TOKEN "$(gen_token)" ;;
  esac
else
  set_env_var AVA_INTERNAL_TOKEN "$(gen_token)"
fi

# Token usado nas chamadas ao mcp_manager (Authorization: Bearer)
AVA_TOKEN="$(get_env_var AVA_INTERNAL_TOKEN)"
[ -n "$AVA_TOKEN" ] || die "AVA_INTERNAL_TOKEN não encontrado em $ENV_FILE."

ok "AVA_INTERNAL_TOKEN configurado em $ENV_FILE (chmod 600)."

# ── 5b. Token do crawl4ai (CRAWL4AI_API_TOKEN) ───────────────────────────────
# O compose exige esta variável (falha de boot se ausente). O websearch-mcp
# e o crawl4ai precisam do MESMO valor para o pipeline de busca funcionar.

if ! grep -qE '^CRAWL4AI_API_TOKEN=.+' "$ENV_FILE" 2>/dev/null; then
  set_env_var CRAWL4AI_API_TOKEN "$(gen_token)"
  ok "CRAWL4AI_API_TOKEN gerado em $ENV_FILE."
else
  ok "CRAWL4AI_API_TOKEN já existe em $ENV_FILE."
fi

# ── 5b-2. Token da extensão do Playwright (PLAYWRIGHT_MCP_EXTENSION_TOKEN) ───
# O @playwright/mcp roda com --extension e pareia com a extensão oficial do
# Chrome via este token. O token NÃO é gerado pelo instalador: a própria
# extensão o cria (base64url de 32 bytes) e persiste em localStorage
# ('auth-token') na origem chrome-extension://mmlmfjhmonkocbjadbfplnigmagldckm
# quando a página status.html é aberta. Extraímos o valor do LevelDB de
# Local Storage do perfil do navegador. Se a extensão ainda não estiver
# instalada, a extração é refeita na etapa 5c, após abrir a Chrome Web Store.

PLAYWRIGHT_EXTENSION_URL="https://chromewebstore.google.com/detail/playwright-extension/mmlmfjhmonkocbjadbfplnigmagldckm"
PLAYWRIGHT_EXTENSION_STATUS_URL="chrome-extension://mmlmfjhmonkocbjadbfplnigmagldckm/status.html"

# Procura o 'auth-token' da extensão no LevelDB de Local Storage dos perfis
# dos navegadores Chromium instalados e imprime o token (43 chars base64url).
# A página status.html renderiza o token via React — o valor confiável está no
# localStorage persistido, não no HTML estático.
extract_playwright_token() {
  command -v python3 >/dev/null 2>&1 || return 1
  python3 - <<'PY'
import glob, os, re, sys

home = os.path.expanduser('~')
leveldb_globs = [
    home + '/.config/google-chrome/*/Local Storage/leveldb',
    home + '/.config/chromium/*/Local Storage/leveldb',
    home + '/.config/microsoft-edge/*/Local Storage/leveldb',
    home + '/.config/BraveSoftware/Brave-Browser/*/Local Storage/leveldb',
    home + '/snap/chromium/common/chromium/*/Local Storage/leveldb',
    home + '/.var/app/org.chromium.Chromium/config/chromium/*/Local Storage/leveldb',
]
# Token: 32 bytes em base64url sem padding = 43 chars de [A-Za-z0-9_-]
token_re = re.compile(r'(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{43}(?![A-Za-z0-9_-])')
needle_utf8 = b'auth-token'
needle_utf16 = 'auth-token'.encode('utf-16-le')

files = []
for pattern in leveldb_globs:
    for d in glob.glob(pattern):
        files.extend(
            os.path.join(d, name) for name in os.listdir(d)
            if name.endswith(('.ldb', '.log'))
        )
# Arquivo mais recente primeiro: se o usuário regenerou o token, o valor
# atual costuma estar no WAL (.log) ou no .ldb mais novo.
files.sort(key=os.path.getmtime, reverse=True)

for path in files:
    try:
        with open(path, 'rb') as fh:
            raw = fh.read()
    except OSError:
        continue
    if needle_utf8 not in raw and needle_utf16 not in raw:
        continue
    # O valor pode estar gravado em UTF-8 ou UTF-16LE no LevelDB —
    # decodifica o arquivo nas duas visões e procura o padrão.
    for view in (raw.decode('utf-8', errors='ignore'),
                 raw.decode('utf-16-le', errors='ignore')):
        match = token_re.search(view)
        if match:
            print(match.group(0))
            sys.exit(0)
sys.exit(1)
PY
}

open_url() {
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$1" >/dev/null 2>&1 || true
  elif command -v sensible-browser >/dev/null 2>&1; then
    sensible-browser "$1" >/dev/null 2>&1 || true
  else
    return 1
  fi
}

if ! grep -qE '^PLAYWRIGHT_MCP_EXTENSION_TOKEN=.+' "$ENV_FILE" 2>/dev/null; then
  info "Procurando token da extensão Playwright no perfil do navegador..."
  if token="$(extract_playwright_token)"; then
    set_env_var PLAYWRIGHT_MCP_EXTENSION_TOKEN "$token"
    ok "PLAYWRIGHT_MCP_EXTENSION_TOKEN extraído do status.html da extensão → $ENV_FILE."
  else
    warn "Extensão Playwright ainda não instalada (token não encontrado no perfil)."
    warn "A instalação abrirá a Chrome Web Store na etapa 5c e extrairá o token após a instalação."
  fi
else
  ok "PLAYWRIGHT_MCP_EXTENSION_TOKEN já existe em $ENV_FILE."
fi

# ── 5c. Registro do Playwright MCP (dependência direta do sistema) ────────────
# Cadastra o @playwright/mcp no mcp_manager (:4008) logo após o `up` do stack.
# O POST /servers é upsert idempotente e o /sync tem no-op por doc_hash —
# seguro re-executar. Tolerante a falha: se o manager ainda estiver bootando,
# retenta; se falhar de vez, só avisa (não aborta a instalação).
MCP_MANAGER_URL="http://127.9.12.22:4008"

register_playwright_mcp() {
  local payload
  payload='{
  "name": "playwright",
  "skill_summary": "Automação de navegador (navegar, clicar, extrair conteúdo)",
  "transport": "stdio",
  "command": "npx",
  "args": ["-y", "@playwright/mcp@latest", "--extension"],
  "env": {"DISPLAY": ":0"}
}'

  # A extensão precisa ser instalada manualmente na Chrome Web Store —
  # o Playwright MCP no modo --extension conecta a ela via bridge.
  info "Abrindo a Chrome Web Store para instalar a extensão do Playwright..."
  info "$PLAYWRIGHT_EXTENSION_URL"
  open_url "$PLAYWRIGHT_EXTENSION_URL" || {
    warn "Não encontrei um opener de URLs — instale a extensão manualmente:"
    printf '  %s\n' "$PLAYWRIGHT_EXTENSION_URL"
  }

  # Se o token ainda não foi extraído (extensão instalada agora), abre a
  # status.html — a extensão então grava o token no localStorage do perfil —
  # e extrai o valor do LevelDB de Local Storage. Pausa para dar tempo do
  # usuário concluir a instalação na loja e o Chrome gravar em disco.
  if ! grep -qE '^PLAYWRIGHT_MCP_EXTENSION_TOKEN=.+' "$ENV_FILE" 2>/dev/null; then
    printf 'Instale a extensão na janela aberta e pressione ENTER para continuar... '
    read -r || true
    info "Abrindo a página de status da extensão (${PLAYWRIGHT_EXTENSION_STATUS_URL})..."
    open_url "$PLAYWRIGHT_EXTENSION_STATUS_URL" || true
    info "Aguardando o Chrome persistir o token (30 s)... "
    sleep 30
    if token="$(extract_playwright_token)"; then
      set_env_var PLAYWRIGHT_MCP_EXTENSION_TOKEN "$token"
      ok "PLAYWRIGHT_MCP_EXTENSION_TOKEN extraído do status.html da extensão → $ENV_FILE."
    else
      warn "Não consegui extrair o token automaticamente."
      warn "Abra a página de status da extensão, copie o token e adicione ao $ENV_FILE:"
      printf '  %s\n' "$PLAYWRIGHT_EXTENSION_STATUS_URL"
      printf '  PLAYWRIGHT_MCP_EXTENSION_TOKEN=<token>\n'
    fi
  fi

  info "Registrando Playwright MCP no mcp_manager ($MCP_MANAGER_URL)..."
  local registered=0 i
  for i in $(seq 1 24); do
    if curl -fsS --max-time 10 -X POST "$MCP_MANAGER_URL/servers" \
         -H 'Content-Type: application/json' \
         -H "Authorization: Bearer $AVA_TOKEN" -d "$payload" >/dev/null 2>&1; then
      registered=1
      break
    fi
    sleep 5
  done
  if [ "$registered" -ne 1 ]; then
    warn "Não consegui cadastrar o Playwright MCP — manager não respondeu em 2 min."
    warn "Cadastre manualmente depois (dois comandos):"
    printf '  curl -X POST %s/servers -H '"'"'Content-Type: application/json'"'"' -H "Authorization: Bearer $AVA_INTERNAL_TOKEN" -d '"'"'%s'"'"'\n' "$MCP_MANAGER_URL" "$payload"
    printf '  curl -X POST %s/servers/playwright/sync -H "Authorization: Bearer $AVA_INTERNAL_TOKEN"\n' "$MCP_MANAGER_URL"
    return 1
  fi

  # /sync: exige a memory_api (:3000) de pé; o 1º sync baixa o pacote npm.
  local synced=0
  for i in $(seq 1 12); do
    if curl -fsS --max-time 240 -X POST "$MCP_MANAGER_URL/servers/playwright/sync" \
         -H "Authorization: Bearer $AVA_TOKEN" >/dev/null 2>&1; then
      synced=1
      break
    fi
    sleep 10
  done
  if [ "$synced" -ne 1 ]; then
    warn "Playwright MCP cadastrado, mas o sync falhou (memory_api ainda bootando?)."
    warn "Rode manualmente: curl -X POST $MCP_MANAGER_URL/servers/playwright/sync -H \"Authorization: Bearer \$AVA_INTERNAL_TOKEN\""
    return 1
  fi
  ok "Playwright MCP registrado e sincronizado."
}

register_websearch_mcp() {
  # URL do MCP do websearch (server.py). Se o manager rodar dentro do Docker,
  # use o nome do serviço (http://websearch:4010/mcp) ou host.docker.internal.
  local ws_url="${WEBSEARCH_MCP_URL:-http://localhost:4010/mcp}"
  local payload
  payload=$(cat <<EOF
{
  "name": "websearch",
  "skill_summary": "Busca na web: pesquisa fatos atuais e retorna trechos relevantes das melhores páginas, com título e URL para citação.",
  "transport": "http",
  "url": "$ws_url",
  "enabled": true,
  "always_on": false,
  "examples": {
    "web_search": [
      "pesquise na internet sobre",
      "qual é a notícia mais recente sobre",
      "busque informações atuais de",
      "procure na web"
    ],
    "memory_write_batch": [
      "lembre que",
      "guarde isso na memória",
      "anote que",
      "salve essa informação",
      "lembre-se de que",
      "memorize os fatos deste documento"
    ]
  }
}
EOF
)

  info "Registrando websearch MCP no mcp_manager ($MCP_MANAGER_URL)..."
  local registered=0 i
  for i in $(seq 1 24); do
    if curl -fsS --max-time 10 -X POST "$MCP_MANAGER_URL/servers" \
         -H 'Content-Type: application/json' \
         -H "Authorization: Bearer $AVA_TOKEN" -d "$payload" >/dev/null 2>&1; then
      registered=1
      break
    fi
    sleep 5
  done
  if [ "$registered" -ne 1 ]; then
    warn "Não consegui cadastrar o websearch MCP — manager não respondeu em 2 min."
    warn "Cadastre manualmente depois (dois comandos):"
    printf '  curl -X POST %s/servers -H '"'"'Content-Type: application/json'"'"' -H "Authorization: Bearer $AVA_INTERNAL_TOKEN" -d '"'"'%s'"'"'\n' "$MCP_MANAGER_URL" "$payload"
    printf '  curl -X POST %s/servers/websearch/sync -H "Authorization: Bearer $AVA_INTERNAL_TOKEN"\n' "$MCP_MANAGER_URL"
    return 1
  fi

  # /sync: exige a memory_api (:3000) de pé E o server.py do websearch
  # respondendo em $ws_url (o cadastro é lazy; só o sync conecta de fato).
  local synced=0
  for i in $(seq 1 12); do
    if curl -fsS --max-time 120 -X POST "$MCP_MANAGER_URL/servers/websearch/sync" \
         -H "Authorization: Bearer $AVA_TOKEN" >/dev/null 2>&1; then
      synced=1
      break
    fi
    sleep 10
  done
  if [ "$synced" -ne 1 ]; then
    warn "websearch MCP cadastrado, mas o sync falhou (memory_api ou websearch ainda bootando?)."
    warn "Rode manualmente: curl -X POST $MCP_MANAGER_URL/servers/websearch/sync -H \"Authorization: Bearer \$AVA_INTERNAL_TOKEN\""
    return 1
  fi
  ok "websearch MCP registrado e sincronizado."
}

register_desktop_commander_mcp() {
  local payload
  payload='{
  "name": "desktop-commander",
  "skill_summary": "Arquivos e terminal do computador (ler/escrever/editar arquivos, listar diretórios, buscar, executar comandos e processos)",
  "transport": "stdio",
  "command": "npx",
  "args": ["-y", "@wonderwhy-er/desktop-commander@latest"],
  "enabled": true,
  "always_on": false
}'

  info "Registrando Desktop Commander MCP no mcp_manager ($MCP_MANAGER_URL)..."
  local registered=0 i
  for i in $(seq 1 24); do
    if curl -fsS --max-time 10 -X POST "$MCP_MANAGER_URL/servers" \
         -H 'Content-Type: application/json' \
         -H "Authorization: Bearer $AVA_TOKEN" -d "$payload" >/dev/null 2>&1; then
      registered=1
      break
    fi
    sleep 5
  done
  if [ "$registered" -ne 1 ]; then
    warn "Não consegui cadastrar o Desktop Commander MCP — manager não respondeu em 2 min."
    warn "Cadastre manualmente depois (dois comandos):"
    printf '  curl -X POST %s/servers -H '"'"'Content-Type: application/json'"'"' -H "Authorization: Bearer $AVA_INTERNAL_TOKEN" -d '"'"'%s'"'"'\n' "$MCP_MANAGER_URL" "$payload"
    printf '  curl -X POST %s/servers/desktop-commander/sync -H "Authorization: Bearer $AVA_INTERNAL_TOKEN"\n' "$MCP_MANAGER_URL"
    return 1
  fi

  # /sync: exige a memory_api (:3000) de pé; o 1º sync baixa o pacote npm.
  local synced=0
  for i in $(seq 1 12); do
    if curl -fsS --max-time 240 -X POST "$MCP_MANAGER_URL/servers/desktop-commander/sync" \
         -H "Authorization: Bearer $AVA_TOKEN" >/dev/null 2>&1; then
      synced=1
      break
    fi
    sleep 10
  done
  if [ "$synced" -ne 1 ]; then
    warn "Desktop Commander MCP cadastrado, mas o sync falhou (memory_api ainda bootando?)."
    warn "Rode manualmente: curl -X POST $MCP_MANAGER_URL/servers/desktop-commander/sync -H \"Authorization: Bearer \$AVA_INTERNAL_TOKEN\""
    return 1
  fi
  ok "Desktop Commander MCP registrado e sincronizado."
}


# ── 5d. Manager (ava-tray) — API REST em :9001 ───────────────────────────────
# O Manager sobe/derruba o stack Docker e supervisiona o mcp_manager do host.
# O instalador o inicia primeiro e pede o start do Docker pela API dele.
MANAGER_URL="http://localhost:9001"

manager_up() {
  curl -fsS --max-time 5 -H "Authorization: Bearer $AVA_TOKEN" "$MANAGER_URL/status" >/dev/null 2>&1
}

has_gui() { [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; }

start_manager() {
  if manager_up; then
    ok "Manager já está respondendo em $MANAGER_URL."
    return 0
  fi
  local bin="$INSTALL_DIR/Manager/ava-tray"
  [ -x "$bin" ] || { warn "Manager não encontrado/executável em $bin."; return 1; }
  has_gui || { warn "Sem sessão gráfica — o Manager (tray) não pode iniciar agora."; return 1; }

  info "Iniciando o Manager (ava-tray)..."
  nohup "$bin" >/dev/null 2>&1 &
  disown || true

  local i
  for i in $(seq 1 30); do
    if manager_up; then
      ok "Manager respondeu em $MANAGER_URL."
      return 0
    fi
    sleep 2
  done
  warn "Manager não respondeu em 60s."
  return 1
}

# POST autenticado na API do Manager. $1 = rota, $2 = timeout (s)
manager_post() {
  curl -fsS --max-time "$2" -X POST -H "Authorization: Bearer $AVA_TOKEN" "$MANAGER_URL$1"
}

docker_up_via_manager() {
  local resp
  info "Pedindo ao Manager para iniciar o Docker (POST /docker/start)..."
  if resp="$(manager_post /docker/start 900 2>&1)" && printf '%s' "$resp" | grep -qE '"ok": *true'; then
    ok "Docker iniciado pelo Manager."
    return 0
  fi
  warn "Manager não iniciou o Docker: ${resp:-sem resposta}"
  return 1
}

# O mcp_manager (:4008) roda no host, supervisionado pelo Manager.
ensure_mcp_manager() {
  manager_up || { warn "Manager indisponível — não consegui garantir o mcp_manager."; return 1; }
  local out i
  out="$(curl -fsS --max-time 5 -H "Authorization: Bearer $AVA_TOKEN" "$MANAGER_URL/mcp/status" 2>/dev/null)" || out=""
  if printf '%s' "$out" | grep -qE '"healthy": *true'; then
    ok "mcp_manager já está ativo."
    return 0
  fi
  info "Pedindo ao Manager para iniciar o mcp_manager (POST /mcp/start)..."
  manager_post /mcp/start 60 >/dev/null 2>&1 || warn "POST /mcp/start falhou — vou aguardar mesmo assim."
  for i in $(seq 1 12); do
    out="$(curl -fsS --max-time 5 -H "Authorization: Bearer $AVA_TOKEN" "$MANAGER_URL/mcp/status" 2>/dev/null)" || out=""
    if printf '%s' "$out" | grep -qE '"healthy": *true'; then
      ok "mcp_manager ativo."
      return 0
    fi
    sleep 5
  done
  warn "mcp_manager não ficou saudável em 1 min (veja: curl -H 'Authorization: Bearer \$AVA_INTERNAL_TOKEN' $MANAGER_URL/mcp/status)."
  return 1
}

# ── 6. Build + first up ───────────────────────────────────────────────────────
if [ "$SKIP_BUILD" -eq 0 ]; then
  info "Build inicial das imagens Docker (pode levar 5-15 min na primeira vez)..."
  ./docker-start.sh build || die "docker compose build falhou — rode ./docker-start.sh build para ver o erro."

  # Manager primeiro: é ele quem sobe o stack Docker (POST /docker/start).
  # Sem GUI ou se o Manager falhar, cai para o docker-start.sh.
  if ! { start_manager && docker_up_via_manager; }; then
    warn "Usando docker-start.sh como alternativa."
    info "Subindo o stack..."
    ./docker-start.sh up -d || die "docker compose up falhou."
  fi

  # mcp_manager (host) ativo antes do cadastro dos MCPs
  ensure_mcp_manager || true

  # Registro dos MCPs — logo após o up, quando o stack já está carregado.
  register_playwright_mcp || true
  register_websearch_mcp || true
  register_desktop_commander_mcp || true



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

if manager_up; then
  ok "ava-tray já está em execução."
elif has_gui; then
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
