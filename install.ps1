#!/usr/bin/env bash
# install.sh — Instalador Alpha AI (Linux)
#
# Espelha o fluxo do install.sh:
#   1. Valida prereqs (Docker + compose v2)
#   2. Baixa o asset *windows*.zip do ultimo release no GitHub
#   3. Extrai em %LOCALAPPDATA%\Programs\AlphaAI (override: -InstallDir)
#   4. Extrai Modules/Models.zip em Modules/Models
#   5. Pede a OPENROUTER_API_KEY e grava em Modules\.env - ANTES do up,
#      porque Modules/LLM.py le a chave no import
#   6. Build; inicia o Manager (ava-tray.exe, API :9001), pede a ele o start do
#      Docker (POST /docker/start) e garante o mcp_manager (/mcp/start); registra
#      os MCPs e espera o /status responder (falha do Manager: docker-start.bat up -d)
#   7. Adiciona tui\ ao PATH do usuario (comando `alpha-ai`)
#   8. Auto-start do ava-tray.exe via HKCU\...\Run (+ inicia agora)
#   9. Inicia a TUI

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

# -- 5. API key (antes do up) --------------------------------------------------
$EnvFile = Join-Path $InstallDir 'Modules\.env'
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

# Le o ultimo valor de uma variavel do .env ($null se ausente)
function Get-EnvVar {
  param([string]$Name)
  if (-not (Test-Path $EnvFile)) { return $null }
  $m = [System.IO.File]::ReadAllLines($EnvFile) | Where-Object { $_ -match "^$([regex]::Escape($Name))=" } | Select-Object -Last 1
  if (-not $m) { return $null }
  return $m.Substring($Name.Length + 1)
}

# Define/substitui uma variavel no .env sem duplicar nem apagar as demais
# (UTF-8 sem BOM - dotenv nao aceita BOM)
function Set-EnvVar {
  param([string]$Name, [string]$Value)
  $lines = @()
  if (Test-Path $EnvFile) {
    $lines = @([System.IO.File]::ReadAllLines($EnvFile) | Where-Object { $_ -notmatch "^$([regex]::Escape($Name))=" })
  }
  $lines += "$Name=$Value"
  [System.IO.File]::WriteAllText($EnvFile, (($lines -join "`n") + "`n"), $Utf8NoBom)
}

# Token aleatorio de 32 bytes em hex (equivalente a `openssl rand -hex 32`)
function New-RandomToken {
  $bytes = New-Object byte[] 32
  $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
  $rng.GetBytes($bytes)
  $rng.Dispose()
  return (($bytes | ForEach-Object { $_.ToString('x2') }) -join '')
}

function Test-ExistingKey {
  if (-not (Test-Path $EnvFile)) { return $false }
  $line = Select-String -Path $EnvFile -Pattern '^OPENROUTER_API_KEY=.+' -Quiet
  return [bool]$line
}

function Write-Key {
  param([string]$Key)
  Set-EnvVar 'OPENROUTER_API_KEY' $Key

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

# ── 5c. Registro do Playwright MCP (dependência direta do sistema) ────────────
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

# -- 5a. Token de autenticacao interna (AVA_INTERNAL_TOKEN) -------------------
# Todos os servicos internos (orchestrator, LLM, TTS, vision, memory, mcp_manager,
# Manager) exigem `Authorization: Bearer <AVA_INTERNAL_TOKEN>`.
if (Get-EnvVar 'AVA_INTERNAL_TOKEN') {
  $ans = Read-Host 'Ja existe um AVA_INTERNAL_TOKEN em Modules\.env. Manter o token existente? [S/n]'
  if ($ans -match '^[nN]') { Set-EnvVar 'AVA_INTERNAL_TOKEN' (New-RandomToken) }
} else {
  Set-EnvVar 'AVA_INTERNAL_TOKEN' (New-RandomToken)
}
Write-Ok "AVA_INTERNAL_TOKEN configurado em $EnvFile"

# Token usado nas chamadas ao mcp_manager (Authorization: Bearer)
$AvaToken = Get-EnvVar 'AVA_INTERNAL_TOKEN'
if (-not $AvaToken) { Die "AVA_INTERNAL_TOKEN nao encontrado em $EnvFile." }
$McpHeaders = @{ Authorization = "Bearer $AvaToken" }

# -- 5b'. Token do crawl4ai (CRAWL4AI_API_TOKEN) ------------------------------
# O compose exige esta variavel (falha de boot se ausente). O websearch-mcp e o
# crawl4ai precisam do MESMO valor para o pipeline de busca funcionar.
if (-not (Get-EnvVar 'CRAWL4AI_API_TOKEN')) {
  Set-EnvVar 'CRAWL4AI_API_TOKEN' (New-RandomToken)
  Write-Ok "CRAWL4AI_API_TOKEN gerado em $EnvFile"
} else {
  Write-Ok "CRAWL4AI_API_TOKEN ja existe em $EnvFile"
}

# -- 5b. Registro do Playwright MCP (dependencia direta do sistema) -----------
# Cadastra o @playwright/mcp no mcp_manager (:4008) logo apos o `up` do stack.
# O POST /servers e upsert idempotente e o /sync tem no-op por doc_hash -
# seguro re-executar. Tolerante a falha: se o manager ainda estiver bootando,
# retenta; se falhar de vez, so avisa (nao aborta a instalacao).
$McpManagerUrl = 'http://localhost:4008'

function Register-PlaywrightMcp {
  # No Windows nao ha X server (DISPLAY) e o Chrome CDP roda no host
  # via host.docker.internal (mesmo endereco do AVA_CDP_ENDPOINT).
  $payload = @{
    name           = 'playwright'
    skill_summary  = 'Automação de navegador (navegar, clicar, extrair conteúdo)'
    transport      = 'stdio'
    command        = 'npx'
    args           = @('-y', '@playwright/mcp@latest', '--cdp-endpoint', 'http://host.docker.internal:9222')
  } | ConvertTo-Json -Depth 3

  Write-Info "Registrando Playwright MCP no mcp_manager ($McpManagerUrl)..."
  $registered = $false
  for ($i = 0; $i -lt 24; $i++) {
    try {
      Invoke-RestMethod -Uri "$McpManagerUrl/servers" -Method Post `
        -ContentType 'application/json' -Body $payload -TimeoutSec 10 -Headers $McpHeaders | Out-Null
      $registered = $true
      break
    } catch {
      Start-Sleep -Seconds 5
    }
  }
  if (-not $registered) {
    Write-Warn2 'Nao consegui cadastrar o Playwright MCP - manager nao respondeu em 2 min.'
    Write-Warn2 "Cadastre manualmente depois (header Authorization: Bearer <AVA_INTERNAL_TOKEN>): POST $McpManagerUrl/servers + POST $McpManagerUrl/servers/playwright/sync"
    return $false
  }

  # /sync: exige a memory_api (:3000) de pe; o 1o sync baixa o pacote npm.
  $synced = $false
  for ($i = 0; $i -lt 12; $i++) {
    try {
      Invoke-RestMethod -Uri "$McpManagerUrl/servers/playwright/sync" -Method Post -TimeoutSec 240 -Headers $McpHeaders | Out-Null
      $synced = $true
      break
    } catch {
      Start-Sleep -Seconds 10
    }
  }
  if (-not $synced) {
    Write-Warn2 'Playwright MCP cadastrado, mas o sync falhou (memory_api ainda bootando?).'
    Write-Warn2 "Rode manualmente: curl -X POST $McpManagerUrl/servers/playwright/sync -H `"Authorization: Bearer <AVA_INTERNAL_TOKEN de Modules\.env>`""
    return $false
  }
  Write-Ok 'Playwright MCP registrado e sincronizado.'
  return $true
}

# -- 5c. Registro dos demais MCPs (websearch e Desktop Commander) -------------
# Mesmo fluxo do Playwright: POST /servers (upsert idempotente) + POST /sync.
# Helper generico para nao triplicar o loop de retry. O body vai em bytes UTF-8
# (Invoke-RestMethod com string usa ISO-8859-1 no Windows PowerShell 5.1 e
# corromperia os acentos do skill_summary).
function Register-McpServer {
  param(
    [Parameter(Mandatory)][string]$Label,
    [Parameter(Mandatory)][string]$Name,
    [Parameter(Mandatory)][hashtable]$Spec,
    [int]$SyncTimeoutSec = 240
  )
  $payload = $Spec | ConvertTo-Json -Depth 6
  $bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)

  Write-Info "Registrando $Label MCP no mcp_manager ($McpManagerUrl)..."
  $registered = $false
  for ($i = 0; $i -lt 24; $i++) {
    try {
      Invoke-RestMethod -Uri "$McpManagerUrl/servers" -Method Post `
        -ContentType 'application/json; charset=utf-8' -Body $bytes -TimeoutSec 10 -Headers $McpHeaders | Out-Null
      $registered = $true
      break
    } catch {
      Start-Sleep -Seconds 5
    }
  }
  if (-not $registered) {
    Write-Warn2 "Nao consegui cadastrar o $Label MCP - manager nao respondeu em 2 min."
    Write-Warn2 "Cadastre manualmente depois (header Authorization: Bearer <AVA_INTERNAL_TOKEN>): POST $McpManagerUrl/servers + POST $McpManagerUrl/servers/$Name/sync"
    return $false
  }

  # /sync: exige a memory_api (:3000) de pe.
  $synced = $false
  for ($i = 0; $i -lt 12; $i++) {
    try {
      Invoke-RestMethod -Uri "$McpManagerUrl/servers/$Name/sync" -Method Post -TimeoutSec $SyncTimeoutSec -Headers $McpHeaders | Out-Null
      $synced = $true
      break
    } catch {
      Start-Sleep -Seconds 10
    }
  }
  if (-not $synced) {
    Write-Warn2 "$Label MCP cadastrado, mas o sync falhou (memory_api ou o proprio servidor ainda bootando?)."
    Write-Warn2 "Rode manualmente: curl -X POST $McpManagerUrl/servers/$Name/sync -H `"Authorization: Bearer <AVA_INTERNAL_TOKEN de Modules\.env>`""
    return $false
  }
  Write-Ok "$Label MCP registrado e sincronizado."
  return $true

}


# ── 6. Build + first up ───────────────────────────────────────────────────────
if [ "$SKIP_BUILD" -eq 0 ]; then
  info "Build inicial das imagens Docker (pode levar 5-15 min na primeira vez)..."
  ./docker-start.sh build || die "docker compose build falhou — rode ./docker-start.sh build para ver o erro."

# -- 5d. Manager (ava-tray.exe) - API REST em :9001 --------------------------
# O Manager sobe/derruba o stack Docker e supervisiona o mcp_manager do host.
# O instalador o inicia primeiro e pede o start do Docker pela API dele.
$ManagerUrl = 'http://localhost:9001'

function Test-ManagerUp {
  try {
    Invoke-RestMethod -Uri "$ManagerUrl/status" -TimeoutSec 5 -Headers $McpHeaders | Out-Null
    return $true
  } catch { return $false }
}

function Start-Manager {
  if (Test-ManagerUp) { Write-Ok "Manager ja esta respondendo em $ManagerUrl."; return $true }
  $exe = Join-Path $InstallDir 'Manager\ava-tray.exe'
  if (-not (Test-Path $exe)) { Write-Warn2 "Manager nao encontrado em $exe."; return $false }

  Write-Info 'Iniciando o Manager (ava-tray)...'
  Start-Process -FilePath $exe -WorkingDirectory $InstallDir
  for ($i = 0; $i -lt 30; $i++) {
    if (Test-ManagerUp) { Write-Ok "Manager respondeu em $ManagerUrl."; return $true }
    Start-Sleep -Seconds 2
  }
  Write-Warn2 'Manager nao respondeu em 60s.'
  return $false
}

function Start-DockerViaManager {
  Write-Info 'Pedindo ao Manager para iniciar o Docker (POST /docker/start)...'
  try {
    $r = Invoke-RestMethod -Uri "$ManagerUrl/docker/start" -Method Post -TimeoutSec 900 -Headers $McpHeaders
    if ($r.ok) { Write-Ok 'Docker iniciado pelo Manager.'; return $true }
    Write-Warn2 "Manager nao iniciou o Docker: $($r.message)"
    return $false
  } catch {
    Write-Warn2 "Manager nao iniciou o Docker: $($_.Exception.Message)"
    return $false
  }
}

# O mcp_manager (:4008) roda no host, supervisionado pelo Manager.
function Ensure-McpManager {
  if (-not (Test-ManagerUp)) { Write-Warn2 'Manager indisponivel - nao consegui garantir o mcp_manager.'; return $false }
  try {
    $s = Invoke-RestMethod -Uri "$ManagerUrl/mcp/status" -TimeoutSec 5 -Headers $McpHeaders
    if ($s.healthy) { Write-Ok 'mcp_manager ja esta ativo.'; return $true }
  } catch { }

  Write-Info 'Pedindo ao Manager para iniciar o mcp_manager (POST /mcp/start)...'
  try {
    Invoke-RestMethod -Uri "$ManagerUrl/mcp/start" -Method Post -TimeoutSec 60 -Headers $McpHeaders | Out-Null
  } catch {
    Write-Warn2 "POST /mcp/start falhou: $($_.Exception.Message) - vou aguardar mesmo assim."
  }
  for ($i = 0; $i -lt 12; $i++) {
    try {
      $s = Invoke-RestMethod -Uri "$ManagerUrl/mcp/status" -TimeoutSec 5 -Headers $McpHeaders
      if ($s.healthy) { Write-Ok 'mcp_manager ativo.'; return $true }
    } catch { }
    Start-Sleep -Seconds 5
  }
  Write-Warn2 "mcp_manager nao ficou saudavel em 1 min (veja GET $ManagerUrl/mcp/status)."
  return $false
}

# -- 6. Build + first up -------------------------------------------------------
if (-not $SkipBuild) {
  Write-Info 'Build inicial das imagens Docker (pode levar 5-15 min na primeira vez)...'
  & .\docker-start.bat build
  if ($LASTEXITCODE -ne 0) { Die 'docker compose build falhou - rode .\docker-start.bat build para ver o erro.' }

  # Manager primeiro: e ele quem sobe o stack Docker (POST /docker/start).
  # Se o Manager falhar, cai para o docker-start.bat.
  $viaManager = (Start-Manager) -and (Start-DockerViaManager)
  if (-not $viaManager) {
    Write-Warn2 'Usando docker-start.bat como alternativa.'
    Write-Info 'Subindo o stack...'
    & .\docker-start.bat up -d
    if ($LASTEXITCODE -ne 0) { Die 'docker compose up falhou.' }
  }

  # mcp_manager (host) ativo antes do cadastro dos MCPs
  Ensure-McpManager | Out-Null

  # Registro dos MCPs - logo apos o up, quando o stack ja esta carregado.
  Register-PlaywrightMcp | Out-Null
  Register-WebsearchMcp | Out-Null
  Register-DesktopCommanderMcp | Out-Null

  $StatusUrl = 'http://localhost:9000/status'
  $MaxWait = 1200  # 20 min
  $Elapsed = 0
  Write-Info "Aguardando orchestrator em $StatusUrl (primeiro boot baixa Supertonic do HF)..."
  while ($Elapsed -lt $MaxWait) {
    try {
      Invoke-WebRequest -Uri $StatusUrl -UseBasicParsing -TimeoutSec 5 *> $null
      Write-Ok "Orchestrator respondeu apos $([int]($Elapsed / 60)) min."
      break
    } catch {
      if (($Elapsed % 60) -eq 0 -and $Elapsed -gt 0) {
        Write-Info "  ...aguardando ha $([int]($Elapsed / 60)) min (logs: .\docker-start.bat logs -f ava)"
      }
      Start-Sleep -Seconds 5
      $Elapsed += 5
    }
  }
  if ($Elapsed -ge $MaxWait) {
    Write-Warn2 'Orchestrator nao respondeu em 20 min. Veja: .\docker-start.bat logs -f ava'
  }

  # Registro do Playwright MCP — logo após o up, quando o stack já está carregado.
  register_playwright_mcp || true
  register_websearch_mcp || true
  register_desktop_commander_mcp || true


# -- 8. Auto-start do tray -----------------------------------------------------
$TrayExe = Join-Path $InstallDir 'Manager\ava-tray.exe'
$RunKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'
if (Test-Path $TrayExe) {
  # Aspas: caminho com espaco quebraria o auto-start no login
  Set-ItemProperty -Path $RunKey -Name 'AlphaAITray' -Value "`"$TrayExe`"" -Type String
  Write-Ok 'Auto-start configurado: HKCU\...\Run\AlphaAITray'
  if (Test-ManagerUp) {
    Write-Ok 'ava-tray ja esta em execucao.'
  } else {
    Start-Process -FilePath $TrayExe -WorkingDirectory $InstallDir
    Write-Ok 'ava-tray iniciado agora.'
  }
} else {
  Write-Warn2 "ava-tray.exe nao encontrado em $TrayExe - auto-start pulado."
}
main

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
