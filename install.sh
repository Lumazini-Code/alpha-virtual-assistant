#!/usr/bin/env bash
# install.sh — Instalador Alpha AI (Linux)
#
# Fluxo:
#   1. Valida/instala prereqs (Node 24, Docker Engine + compose v2, curl, unzip, x86_64)
#   2. Baixa o asset `*linux*.zip` do último release no GitHub — MODO TESTE:
#      ALPHA_RELEASE_ZIP=/caminho.zip (exportada) usa um zip local e pula o
#      GitHub inteiro (para testar antes do release oficial)
#   3. Extrai em ~/.local/opt/alpha-ai (override: ALPHA_INSTALL_DIR ou --install-dir)
#   3b. Agent Skills: npx skills add heygen-com/hyperframes → .agents/skills
#       (gitignored; flags -a universal -y; falha = aviso, não morte)
#   3c. OfficeCLI: npm install -g @officecli/officecli + skill
#       officecli.ai/SKILL.md → .agents/skills/officecli (falha = aviso)
#   4. Extrai Modules/Models.zip em Modules/Models (e apaga o zip, salvo --keep-models-zip)
#   5. Pede o provider de inferência: LLM_BASE_URL (Enter = OpenRouter) e, se
#      o provider NÃO for local, a LLM_API_KEY — grava em Modules/.env
#      (chmod 600) ANTES do up, porque Modules/LLM.py lê no import
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

Env:
  ALPHA_RELEASE_ZIP=ARQUIVO.zip  Usa um zip local em vez do release do GitHub
                                 (modo teste antes do release oficial)
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

# ALPHA_RELEASE_ZIP (MODO TESTE, antes do release oficial): quando a env var
# aponta para um zip local, o GitHub é pulado inteiro e esse zip é o que
# instala. O arquivo é COPIADO para o TMP_DIR (o unzip extraí dele) — o
# original nunca é apagado pelo trap. Uso:
#   ALPHA_RELEASE_ZIP=/caminho/alpha-ai-linux.zip ./install.sh
# (a var precisa estar EXPORTADA para sobreviver ao re-exec via /tmp.)
if [ -n "${ALPHA_RELEASE_ZIP:-}" ]; then
  [ -f "$ALPHA_RELEASE_ZIP" ] || \
    die "ALPHA_RELEASE_ZIP aponta para arquivo inexistente: $ALPHA_RELEASE_ZIP"
  info "MODO TESTE (ALPHA_RELEASE_ZIP): zip local '$ALPHA_RELEASE_ZIP' — pulando o GitHub."
  cp -f -- "$ALPHA_RELEASE_ZIP" "$TMP_DIR/alpha-ai.zip"
else
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
fi

# ── 3. Extração ───────────────────────────────────────────────────────────────
info "Extraindo em $INSTALL_DIR ..."
mkdir -p "$INSTALL_DIR"
unzip -oq "$TMP_DIR/alpha-ai.zip" -d "$INSTALL_DIR"

[ -f "$INSTALL_DIR/Modules/Models.zip" ] || \
  die "Modules/Models.zip não veio no zip do release — asset incompatível com esta versão do instalador."

cd "$INSTALL_DIR"

chmod +x tui/alpha-ai Manager/ava-tray docker-start.sh 2>/dev/null || true
ok "Arquivos extraídos."

# ── 3b. Agent Skills (SKILL.md — hyperframes) ─────────────────────────────────
# `.agents/` e `skills-lock.json` são gitignored: um install novo NÃO traz as
# skills. O CLI `skills` roda via npx (sem install global — o --yes do NPX
# baixa o pacote no cache). Flags testadas contra o CLI:
#   -a universal → grava SÓ em .agents/skills (store canônico; sem linkar em
#                  .claude/.cursor/etc do usuário)
#   -y           → pula os prompts (escopo = projeto; -y = non-interactive)
# Idempotente (re-run re-sincroniza pelo lock). Falha (rede, p.ex.) é AVISO,
# não morte — o sistema sobe sem skills.
if command -v npx >/dev/null 2>&1; then
  info "Agent Skills: baixando heygen-com/hyperframes (npx skills add)..."
  if npx --yes skills add heygen-com/hyperframes -a universal -y; then
    ok "Agent Skills instaladas em .agents/skills ($(ls .agents/skills 2>/dev/null | wc -l) skills)."
  else
    warn "npx skills add falhou (sem rede?) — o sistema sobe sem as skills."
    warn "Para instalar depois, na pasta $INSTALL_DIR: npx skills add heygen-com/hyperframes"
  fi
else
  warn "npx ausente no host — pulei as Agent Skills (rode: npx skills add heygen-com/hyperframes)."
fi

# ── 3c. OfficeCLI (binário npm -g + skill SKILL.md) ───────────────────────────
# `officecli` = leitura/edição de .docx/.xlsx/.pptx (o binário nativo vem no
# postinstall do pacote). npm -g com prefix do usuário (nvm) não pede root;
# com node de apt/pacman o prefix não é escrevível → as_root. A SKILL.md do
# site oficial vai direto para o store canônico .agents/skills/officecli/
# (gitignored; o mount ro entrega ao container e o scanner do orchestrator
# descobre na próxima pergunta, sem restart). Download em arquivo temporário +
# mv — SKILL.md truncado por rede não entra no índice. Falha é AVISO, não
# morte — o sistema sobe sem o OfficeCLI/skill.
OFFICE_FAIL=0
if command -v npm >/dev/null 2>&1; then
  info "OfficeCLI: npm install -g @officecli/officecli ..."
  GPREFIX="$(npm prefix -g 2>/dev/null || echo /usr/lib)"
  if [ -w "$GPREFIX" ]; then
    npm install -g @officecli/officecli || OFFICE_FAIL=1
  else
    as_root npm install -g @officecli/officecli || OFFICE_FAIL=1
  fi
else
  warn "npm ausente no host — pulei o OfficeCLI (rode: npm install -g @officecli/officecli)."
fi
if [ "$OFFICE_FAIL" -eq 0 ] && command -v npm >/dev/null 2>&1; then
  ok "OfficeCLI instalado ($(command -v officecli 2>/dev/null || echo 'via npm -g'))."
elif [ "$OFFICE_FAIL" -ne 0 ]; then
  warn "npm install -g @officecli/officecli falhou — o sistema sobe sem o OfficeCLI."
  warn "Para instalar depois: npm install -g @officecli/officecli"
fi

mkdir -p .agents/skills/officecli
if curl -fsSL https://officecli.ai/SKILL.md -o .agents/skills/officecli/.SKILL.md.tmp; then
  mv .agents/skills/officecli/.SKILL.md.tmp .agents/skills/officecli/SKILL.md
  ok "Skill officecli instalada (.agents/skills/officecli/SKILL.md)."
else
  rm -f .agents/skills/officecli/.SKILL.md.tmp
  warn "curl https://officecli.ai/SKILL.md falhou — skill ausente (o CLI, se instalado, funciona)."
fi

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
  # Aceita a key nova (LLM_API_KEY) ou a de instalações antigas (OPENROUTER_API_KEY)
  [ -f "$ENV_FILE" ] && grep -qE '^(LLM_API_KEY|OPENROUTER_API_KEY)=.+' "$ENV_FILE" 2>/dev/null
}

# ── Provider de inferência genérico (LLM_*) ───────────────────────────────────
# Qualquer API OpenAI-compatible é configurada em Modules/.env
# (LLM_BASE_URL/LLM_API_KEY/LLM_MODEL; default = OpenRouter). Fluxo do install:
#   (1) pede o LLM_BASE_URL (Enter = OpenRouter);
#   (2) pede a LLM_API_KEY SÓ se o provider NÃO for local
#       (localhost/127.0.0.1/127.9.12.22/*.local);
#   (3) os dois são escritos no .env só DEPOIS de coletados.

llm_base_url() {
  [ -f "$ENV_FILE" ] || return 0
  grep -E '^LLM_BASE_URL=' "$ENV_FILE" 2>/dev/null | tail -1 \
    | cut -d= -f2- | tr -d "\"'"
}

# LLM_PROVIDER_LOCAL=1 aceita qualquer url como "local" (ex.: IP na LAN).
llm_local_flag() {
  [ "${LLM_PROVIDER_LOCAL:-0}" = "1" ] && return 0
  [ -f "$ENV_FILE" ] && grep -qE '^LLM_PROVIDER_LOCAL=1' "$ENV_FILE" 2>/dev/null
}

# Local para uma url JÁ DIGITADA (avaliada antes de gravar no .env).
# Nota: `\[` é obrigatório — sem o escape, `[::1]` vira classe de caracteres
# no glob do case e casaria qualquer url começando com "1" (ex.: 192.168...).
llm_url_is_local() {
  llm_local_flag && return 0
  case "$1" in
    http://localhost*|https://localhost*|\
    http://127.0.0.1*|https://127.0.0.1*|\
    http://127.9.12.22*|https://127.9.12.22*|\
    http://\[::1\]*|https://\[::1\]*|\
    http://*.local*|https://*.local*) return 0 ;;
  esac
  return 1
}

# Local segundo o .env (usado no health check). Sem LLM_BASE_URL no .env o
# LLM.py usa OpenRouter → key p/ valer.
llm_is_local() {
  llm_local_flag && return 0
  local base
  base="$(llm_base_url)"
  [ -n "$base" ] || return 1
  llm_url_is_local "$base"
}

# Pede a API key do provider ($1 = base url) e devolve em $LLM_PROMPT_KEY.
# Nada é gravado aqui — o caller escreve no .env depois de coletar tudo.
prompt_key() {
  local base="$1" key is_or=""
  case "$base" in *openrouter.ai*) is_or=1 ;; esac
  LLM_PROMPT_KEY=""
  while :; do
    if [ -n "$is_or" ]; then
      printf 'Cole sua OPENROUTER_API_KEY (https://openrouter.ai/keys): '
    else
      printf 'Cole a API key do provider (%s) — Enter = sem key: ' "$base"
    fi
    if ! read -rs key; then
      die "Entrada encerrada (EOF) — instalação abortada na etapa da API key."
    fi
    printf '\n'
    key="$(printf '%s' "$key" | tr -d '[:space:]')"
    if [ -n "$key" ]; then
      if [ -n "$is_or" ]; then
        case "$key" in
          sk-or-*) ;;
          *) warn "Chaves do OpenRouter normalmente começam com 'sk-or-'. Prosseguindo mesmo assim." ;;
        esac
      fi
      LLM_PROMPT_KEY="$key"
      return 0
    fi
    # Enter vazio: OpenRouter exige key; provider genérico aceita sem key
    # (o caller oferece marcar como local / segue e o health check avisa).
    if [ -n "$is_or" ]; then
      warn "Chave vazia — tente de novo."
      continue
    fi
    return 0
  done
}

prompt_llm_config() {
  local base="" key="" local_flag="" ans

  # ── (1) endereço base da API ──
  while :; do
    printf 'Endereço base da API de inferência (LLM_BASE_URL) [Enter = OpenRouter]: '
    if ! read -r base; then
      die "Entrada encerrada (EOF) — instalação abortada na etapa do provider de inferência."
    fi
    base="$(printf '%s' "$base" | tr -d '[:space:]')"
    [ -z "$base" ] && base="https://openrouter.ai/api/v1"
    base="${base%/}"
    case "$base" in
      http://*|https://*) break ;;
      *) warn "O endereço deve começar com http:// ou https:// — tente de novo." ;;
    esac
  done

  # ── (2) API key — só quando o provider NÃO é local ──
  if llm_url_is_local "$base"; then
    ok "Provider local detectado ($base) — API key não é necessária."
  else
    while :; do
      prompt_key "$base"
      key="$LLM_PROMPT_KEY"
      [ -n "$key" ] && break
      printf 'Sem API key — em provider remoto a inferência vai falhar no runtime.\n'
      printf 'Marcar como LOCAL (LLM_PROVIDER_LOCAL=1 — ex.: servidor na LAN) e seguir sem key? [S/n] '
      read -r ans || ans=""
      case "$ans" in
        n|N*) ;;                      # não: pede a key de novo
        *) local_flag=1; break ;;     # sim: segue sem key, marcado como local
      esac
    done
  fi

  # ── (3) coletado → "ai sim" escritos no .env ──
  set_env_var LLM_BASE_URL "$base"
  [ -n "$key" ] && set_env_var LLM_API_KEY "$key"
  [ -n "$local_flag" ] && set_env_var LLM_PROVIDER_LOCAL 1
  ok "LLM_BASE_URL=$base gravada em $ENV_FILE (chmod 600)."
  [ -n "$key" ] && ok "LLM_API_KEY gravada em $ENV_FILE (chmod 600)."
  [ -n "$local_flag" ] && ok "LLM_PROVIDER_LOCAL=1 — provider marcado como local, sem key."
}

# Já existe configuração de provider (base e/ou key) → manter ou reconfigurar.
# Instalação antiga só com OPENROUTER_API_KEY e sem LLM_BASE_URL mantém o
# comportamento original (default = OpenRouter + aquela key).
if [ -n "$(get_env_var LLM_BASE_URL)" ] || existing_key; then
  printf 'Já existe configuração de provider em Modules/.env (LLM_BASE_URL=%s).\n' "$(get_env_var LLM_BASE_URL)"
  printf 'Manter a configuração existente? [S/n] '
  read -r ans || ans=""
  case "$ans" in
    n|N*)
      prompt_llm_config
      ;;
    *)
      ok "Configuração de provider mantida em $ENV_FILE."
      ;;
  esac
else
  prompt_llm_config
fi

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
# O compose exige esta variável (falha de boot se ausente). O alpha-mcp
# e o crawl4ai precisam do MESMO valor para o pipeline de busca funcionar.

if ! grep -qE '^CRAWL4AI_API_TOKEN=.+' "$ENV_FILE" 2>/dev/null; then
  set_env_var CRAWL4AI_API_TOKEN "$(gen_token)"
  ok "CRAWL4AI_API_TOKEN gerado em $ENV_FILE."
else
  ok "CRAWL4AI_API_TOKEN já existe em $ENV_FILE."
fi

# ── 5b-1. Perfil de recursos (RAM) ───────────────────────────────────────────
# Escreve as flags de consumo no Modules/.env. Econômico (default): TTS
# (~470MB), visualizador de grafo (~80MB) e voz do host (~600MB) desligados —
# tudo sobe on-demand conforme o uso. Completo: tudo ligado desde o boot.
# AVA_DOCKER_IDLE_MINUTES=15 vale para os dois: o stack dorme cedo e a
# pergunta seguinte o reergue sozinho.
printf 'Perfil de recursos? [1]=Econômico (~500MB, recomendado p/ PC com outras apps) [2]=Completo (voz+TTS+grafo, ~1.4GB) [1/2, default 1] '
read -r ans || ans=""
case "$ans" in
  2)
    set_env_var AVA_TTS_ENABLED 1
    set_env_var AVA_GRAPH_UI 1
    set_env_var AVA_VOICE 1
    ok "Perfil completo: voz + TTS + visualizador de grafo ligados."
    ;;
  *)
    set_env_var AVA_TTS_ENABLED 0
    set_env_var AVA_GRAPH_UI 0
    set_env_var AVA_VOICE 0
    ok "Perfil econômico: TTS, visualizador de grafo e voz desligados (ligue as flags em Modules/.env quando quiser)."
    ;;
esac
set_env_var AVA_DOCKER_IDLE_MINUTES 15
ok "Docker dorme após 15 min de inatividade (AVA_DOCKER_IDLE_MINUTES=15)."

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
EXT_ID = b'mmlmfjhmonkocbjadbfplnigmagldckm'
# Chave completa do localStorage da extensão no LevelDB (Meta key:
# _<origem>\x00\x01<chave>). Ancorar a busca na origem evita FALSO POSITIVO:
# outros sites gravam valores de 43 chars [A-Za-z0-9_-] no Local Storage
# (caso real: token extraído errado vindo de mega.nz).
KEY_NEEDLE = EXT_ID + b'\x00\x01auth-token'
# Após a chave vem a parte interna do registro LevelDB (seq~type, 8 bytes
# nos .ldb; varints nos .log) e então o valor. O token (43 chars base64url)
# aparece na janela logo à frente — sem parse estrutural do SST, que varia
# entre WAL e SST compactado (blocos snappy + prefix-compression).
WINDOW = 64
token_re = re.compile(rb'[A-Za-z0-9_-]{43}')

leveldb_globs = [
    home + '/.config/google-chrome/*/Local Storage/leveldb',
    home + '/.config/chromium/*/Local Storage/leveldb',
    home + '/.config/microsoft-edge/*/Local Storage/leveldb',
    home + '/.config/BraveSoftware/Brave-Browser/*/Local Storage/leveldb',
    home + '/snap/chromium/common/chromium/*/Local Storage/leveldb',
    home + '/.var/app/org.chromium.Chromium/config/chromium/*/Local Storage/leveldb',
]

files = []
for pattern in leveldb_globs:
    for d in glob.glob(pattern):
        files.extend(
            os.path.join(d, name) for name in os.listdir(d)
            if name.endswith(('.ldb', '.log')))
# Ordem de geração do LevelDB: número do arquivo maior = write mais recente.
def file_num(path):
    try:
        return int(os.path.basename(path).split('.')[0])
    except ValueError:
        return -1
files.sort(key=file_num, reverse=True)

for path in files:
    try:
        with open(path, 'rb') as fh:
            raw = fh.read()
    except OSError:
        continue
    idx = 0
    while True:
        i = raw.find(KEY_NEEDLE, idx)
        if i < 0:
            break
        m = token_re.search(raw, i + len(KEY_NEEDLE), i + len(KEY_NEEDLE) + WINDOW)
        if m:
            print(m.group(0).decode('ascii'))
            sys.exit(0)
        idx = i + 1
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
  # Verificação de drift — o mecanismo de autenticação da extensão: a
  # connect.html compara o 'token' da URL com o 'auth-token' do localStorage;
  # divergência = "Invalid token provided" em TODA conexão (o servidor manda o
  # token velho do env). Se o token do .env não é o valor atual da extensão
  # (regenerado na status.html), atualiza o .env; o re-registro adiante derruba
  # a sessão do @playwright/mcp no mcp_manager, que respawna com o token novo.
  if token="$(extract_playwright_token)" && [ -n "$token" ]; then
    current="$(get_env_var PLAYWRIGHT_MCP_EXTENSION_TOKEN)"
    if [ "$current" != "$token" ]; then
      warn "PLAYWRIGHT_MCP_EXTENSION_TOKEN divergente: a extensão está usando outro token (regenerado?)."
      set_env_var PLAYWRIGHT_MCP_EXTENSION_TOKEN "$token"
      ok "PLAYWRIGHT_MCP_EXTENSION_TOKEN atualizado → $ENV_FILE (o re-registro adiante respawna o servidor)."
    fi
  fi
fi

# ── 5c. Registro do Playwright MCP (dependência direta do sistema) ────────────
# Cadastra o @playwright/mcp no mcp_manager (:4008) logo após o `up` do stack.
# O POST /servers é upsert idempotente e o /sync tem no-op por doc_hash —
# seguro re-executar. Tolerante a falha: se o manager ainda estiver bootando,
# retenta; se falhar de vez, só avisa (não aborta a instalação).
MCP_MANAGER_URL="http://127.9.12.22:4008"

register_playwright_mcp() {
  # O env do registro é autocontido (o mcp_manager mescla server.env sobre o
  # ambiente herdado — a linha vale mesmo sem o tray injetando o env):
  #   • PLAYWRIGHT_MCP_EXTENSION_TOKEN — a connect.html da extensão compara o
  #     'token' da URL com o 'auth-token' do localStorage; igual → a conexão é
  #     aprovada sozinha (sem diálogo manual por chamada).
  #   • PLAYWRIGHT_EXTENSION_PROTOCOL=2 — a extensão exige exatamente a
  #     protocolVersion 2 (v<2 → "unsupported protocol version", v>2 →
  #     versionMismatch); forçar 2 mantém o par testado com a extensão 0.4.x
  #     da loja mesmo se o @playwright/mcp@latest subir o default do relay
  #     (o env é lido pelo relay no spawn: process.env.PLAYWRIGHT_EXTENSION_PROTOCOL).
  local payload token
  token="$(get_env_var PLAYWRIGHT_MCP_EXTENSION_TOKEN)"
  local env_json='{"DISPLAY": ":0", "PLAYWRIGHT_EXTENSION_PROTOCOL": "2"}'
  if [ -n "$token" ]; then
    env_json="$(printf '{"DISPLAY": ":0", "PLAYWRIGHT_EXTENSION_PROTOCOL": "2", "PLAYWRIGHT_MCP_EXTENSION_TOKEN": "%s"}' "$token")"
  fi
  payload="$(printf '{
  "name": "playwright",
  "skill_summary": "Automação de navegador (navegar, clicar, extrair conteúdo)",
  "transport": "stdio",
  "command": "npx",
  "args": ["-y", "@playwright/mcp@latest", "--extension"],
  "env": %s
}' "$env_json")"

  # A extensão precisa ser instalada manualmente na Chrome Web Store —
  # o Playwright MCP no modo --extension conecta a ela via bridge. Com o
  # token já no .env (extensão instalada em execução anterior) a abertura é
  # pulada: re-executar o instalador não deve reabrir a loja.
  if ! grep -qE '^PLAYWRIGHT_MCP_EXTENSION_TOKEN=.+' "$ENV_FILE" 2>/dev/null; then
    info "Abrindo a Chrome Web Store para instalar a extensão do Playwright..."
    info "$PLAYWRIGHT_EXTENSION_URL"
    open_url "$PLAYWRIGHT_EXTENSION_URL" || {
      warn "Não encontrei um opener de URLs — instale a extensão manualmente:"
      printf '  %s\n' "$PLAYWRIGHT_EXTENSION_URL"
    }
  fi

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

register_alpha_mcp() {
  # URL do MCP do alpha-mcp (server.py). Se o manager rodar dentro do Docker,
  # use o nome do serviço (http://alpha-mcp:4010/mcp) ou host.docker.internal.
  local ws_url="${ALPHA_MCP_URL:-http://localhost:4010/mcp}"
  local payload
  payload=$(cat <<EOF
{
  "name": "alpha-mcp",
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

  info "Registrando alpha-mcp no mcp_manager ($MCP_MANAGER_URL)..."
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
    warn "Não consegui cadastrar o alpha-mcp — manager não respondeu em 2 min."
    warn "Cadastre manualmente depois (dois comandos):"
    printf '  curl -X POST %s/servers -H '"'"'Content-Type: application/json'"'"' -H "Authorization: Bearer $AVA_INTERNAL_TOKEN" -d '"'"'%s'"'"'\n' "$MCP_MANAGER_URL" "$payload"
    printf '  curl -X POST %s/servers/alpha-mcp/sync -H "Authorization: Bearer $AVA_INTERNAL_TOKEN"\n' "$MCP_MANAGER_URL"
    return 1
  fi

  # /sync: exige a memory_api (:3000) de pé E o server.py do alpha-mcp
  # respondendo em $ws_url (o cadastro é lazy; só o sync conecta de fato).
  local synced=0
  for i in $(seq 1 12); do
    if curl -fsS --max-time 120 -X POST "$MCP_MANAGER_URL/servers/alpha-mcp/sync" \
         -H "Authorization: Bearer $AVA_TOKEN" >/dev/null 2>&1; then
      synced=1
      break
    fi
    sleep 10
  done
  if [ "$synced" -ne 1 ]; then
    warn "alpha-mcp cadastrado, mas o sync falhou (memory_api ou alpha-mcp ainda bootando?)."
    warn "Rode manualmente: curl -X POST $MCP_MANAGER_URL/servers/alpha-mcp/sync -H \"Authorization: Bearer \$AVA_INTERNAL_TOKEN\""
    return 1
  fi
  ok "alpha-mcp registrado e sincronizado."
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
  register_alpha_mcp || true
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

  # Detecta API key ausente no /health do LLM.py (llm_key: "missing" =
  # provider remoto sem key; "not-required" = provider local, ok — não reclama)
  if ! llm_is_local && curl -fsS --max-time 5 "http://127.9.12.22:4003/health" 2>/dev/null \
       | grep -q '"llm_key": *"missing"'; then
    warn "O LLM.py reportou llm_key: missing (provider remoto sem API key)."
    printf 'Deseja re-informar a chave agora? [S/n] '
    read -r ans || ans=""
    case "$ans" in
      n|N*) ;;
      *)
        prompt_key "$(llm_base_url)"
        [ -n "$LLM_PROMPT_KEY" ] && set_env_var LLM_API_KEY "$LLM_PROMPT_KEY"
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
