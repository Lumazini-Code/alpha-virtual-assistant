# install.ps1 - Instalador Alpha AI (Windows)
#
# Espelha o fluxo do install.sh:
#   1. Valida prereqs (Node 24 no host; Docker Desktop OU Docker Engine no WSL)
#   2. Baixa o asset *windows*.zip do ultimo release no GitHub
#   3. Extrai em %LOCALAPPDATA%\Programs\AlphaAI (override: -InstallDir)
#   4. Extrai Modules/Models.zip em Modules/Models
#   5. Pede a OPENROUTER_API_KEY e grava em Modules\.env - ANTES do up,
#      porque Modules/LLM.py le a chave no import
#   6. Build; inicia o Manager (ava-tray.exe, API :9001), pede a ele o start do
#      Docker (POST /docker/start) e garante o mcp_manager (/mcp/start); registra
#      os MCPs e espera o /status responder (falha do Manager: compose up -d direto)
#   7. Adiciona tui\ ao PATH do usuario (comando `alpha-ai`)
#   8. Auto-start do ava-tray.exe via HKCU\...\Run (+ inicia agora)
#   9. Inicia a TUI
#
# Uso: powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 [-InstallDir DIR] [-SkipBuild] [-NoLaunch]

param(
  [string]$InstallDir = $env:ALPHA_INSTALL_DIR,
  [switch]$KeepModelsZip,
  [switch]$SkipBuild,
  [switch]$NoLaunch
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # Invoke-WebRequest ~10x mais rapido

$Repo = 'Lumazini-Code/alpha-virtual-assistant'
if (-not $InstallDir) { $InstallDir = Join-Path $env:LOCALAPPDATA 'Programs\AlphaAI' }

function Write-Info { param($m) Write-Host "[INFO] $m" }
function Write-Ok   { param($m) Write-Host "[ OK ] $m" }
function Write-Warn2 { param($m) Write-Host "[AVISO] $m" -ForegroundColor Yellow }
function Die {
  param($m)
  Write-Host "[ERRO] $m" -ForegroundColor Red
  Read-Host 'Pressione Enter para sair'
  exit 1
}

# -- 0. Feature do Windows: Virtual Machine Platform ----------------------------
# WSL2 roda sobre o Hyper-V (VirtualMachinePlatform). Sem essa feature o
# `wsl --import` da rota B falha. O `dism` exige elevation (UAC).
Write-Info 'Verificando feature do Windows: VirtualMachinePlatform...'
$VmpState = $null
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$DismQuery = (dism.exe /online /get-featureinfo /featurename:VirtualMachinePlatform 2>&1 | Out-String)
$ErrorActionPreference = $prevEAP
if ($DismQuery -match 'State\s*:\s*(\S+)') { $VmpState = $Matches[1] }

if ($VmpState -eq 'Enabled') {
  Write-Ok 'VirtualMachinePlatform ja esta habilitada.'
} else {
  Write-Warn2 'VirtualMachinePlatform desabilitada (ou sem permissao para consultar) - habilitando...'
  $prevEAP = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
  $dismCode = $LASTEXITCODE
  $ErrorActionPreference = $prevEAP
  if ($dismCode -ne 0) {
    Die "Falha ao habilitar VirtualMachinePlatform (dism saiu com $dismCode). Rode o install.ps1 em um PowerShell como Administrador."
  }
  Write-Ok 'VirtualMachinePlatform habilitada (sem reinicio automatico).'
  Write-Host ''
  Write-Warn2 '>> REINICIE O COMPUTADOR para a feature ficar ativa. <<'
  Write-Host '    Depois de reiniciar, execute o install.ps1 novamente - a instalacao'
  Write-Host '    continua normalmente a partir daqui (essa etapa sera pulada).'
  $ans = Read-Host 'Reiniciar agora? [s/N]'
  if ($ans -match '^[sS]') {
    Write-Info 'Reiniciando em 10 segundos (Ctrl+C para cancelar)...'
    shutdown.exe /r /t 10
  }
  exit 0
}

# -- 1. Prereqs ----------------------------------------------------------------
Write-Info 'Validando pre-requisitos...'

# Helper: executa docker (via Docker Desktop ou via WSL Debian-slim)
$script:UseWSLDocker = $false
function Invoke-Docker {
  if ($script:UseWSLDocker) { wsl -d Debian-slim -e docker @args; return $LASTEXITCODE }
  docker @args
  return $LASTEXITCODE
}

# Verifica se o daemon Docker responde (Docker Desktop ou WSL)
function Test-DockerAlive {
  $prevEAP = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  $code = Invoke-Docker info *> $null
  $ErrorActionPreference = $prevEAP
  return ($code -eq 0)
}

# Verifica o plugin 'docker compose' (v2)
function Test-DockerCompose {
  $prevEAP = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  Invoke-Docker compose version *> $null
  $ok = ($LASTEXITCODE -eq 0)
  $ErrorActionPreference = $prevEAP
  return $ok
}

# Testa o Docker Desktop (instalado no Windows)
$DesktopDockerOk = $false
if (Get-Command docker -ErrorAction SilentlyContinue) {
  if (Test-DockerAlive) { $DesktopDockerOk = $true }
}

# Testa o Docker no WSL (Debian-slim)
$WslDockerOk = $false
if (-not $DesktopDockerOk) {
  $wslCmd = Get-Command wsl -ErrorAction SilentlyContinue
  if ($wslCmd) {
    # Verifica se a distro Debian-slim ja existe
    $distros = (wsl --list --quiet 2>$null) -replace "`0", '' | Where-Object { $_ }
    if ($distros -contains 'Debian-slim') {
      # Verifica se o docker responde dentro dela
      if ((Invoke-Docker info *> $null) -eq 0) { $WslDockerOk = $true }
    }
  }
}

if ($DesktopDockerOk) {
  Write-Ok 'Docker Desktop detectado.'
} elseif ($WslDockerOk) {
  $script:UseWSLDocker = $true
  Write-Ok 'Docker no WSL (Debian-slim) detectado.'
} else {
  # Configura o Docker no WSL (Debian-slim) automaticamente
  Write-Info 'Docker nao encontrado no Windows/WSL. Configurando WSL + Docker automaticamente...'
  $wslCmd = Get-Command wsl -ErrorAction SilentlyContinue
  if (-not $wslCmd) {
    Die "WSL nao encontrado. Instale o WSL (ab PowerShell como Admin: 'wsl --install') e rode novamente."
  }
  $distros = (wsl --list --quiet 2>$null) -replace "`0", '' | Where-Object { $_ }
  if ($distros -notcontains 'Debian-slim') {
    Write-Info 'Importando Debian-slim no WSL...'
    $RootFs = "$env:TEMP\debian-slim.tar.xz"
    Invoke-WebRequest -Uri 'https://github.com/debuerreotype/docker-debian-artifacts/raw/dist-amd64/trixie/slim/rootfs.tar.xz' -OutFile $RootFs -UseBasicParsing
    New-Item -ItemType Directory -Force -Path 'C:\WSL\Debian-slim' | Out-Null
    wsl --import Debian-slim C:\WSL\Debian-slim $RootFs --version 2
    if ($LASTEXITCODE -ne 0) { Die 'Falha ao importar Debian-slim no WSL.' }
    Remove-Item $RootFs -Force
    Write-Ok 'Debian-slim importado.'
  }
  # Configura systemd, usuario e sudo
  Write-Info 'Configurando systemd/usuario no Debian-slim...'
  wsl -d Debian-slim -u root -- bash -c "apt update && apt install -y systemd systemd-sysv sudo curl ca-certificates && adduser --disabled-password --gecos '' alpha-ai && usermod -aG sudo alpha-ai && printf '[boot]\nsystemd=true\n[user]\ndefault=alpha-ai\n' > /etc/wsl.conf"
  if ($LASTEXITCODE -ne 0) { Die 'Falha ao configurar systemd no Debian-slim.' }
  wsl --shutdown
  Start-Sleep -Seconds 5
  # Instala o Docker Engine
  Write-Info 'Instalando Docker Engine no Debian-slim...'
  wsl -d Debian-slim -u root -- bash -c "curl -fsSL https://get.docker.com | sh && usermod -aG docker alpha-ai && systemctl enable --now docker"
  if ($LASTEXITCODE -ne 0) { Die 'Falha ao instalar o Docker Engine no Debian-slim.' }
  $script:UseWSLDocker = $true
  if (-not (Test-DockerAlive)) { Die 'Docker no WSL nao respondeu apos instalacao.' }
  Write-Ok 'Docker Engine instalado no WSL (Debian-slim).'
}

if (-not (Test-DockerCompose)) {
  Die "Plugin 'docker compose' ausente - atualize o Docker (Desktop ou Engine no WSL)."
}

Write-Ok 'Pre-requisitos validados.'

# -- 1b. Node 24 no host -------------------------------------------------------
# Os MCPs Playwright e Desktop Commander sao spawnados pelo mcp_manager do host
# via `npx` — sem Node 24 no Windows o registro falha na primeira pergunta.
function Test-Node24 {
  $node = Get-Command node -ErrorAction SilentlyContinue
  if (-not $node) { return $false }
  try {
    $major = [int]((& node --version) -replace '^v', '').Split('.')[0]
    return ($major -ge 24)
  } catch { return $false }
}

if (Test-Node24) {
  Write-Ok "Node 24+ ja presente no host ($(& node --version))."
} else {
  Write-Info 'Node 24 ausente no host - instalando automaticamente...'
  $installed = $false

  # Tentativa 1: winget (LTS atual = v24.x)
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    Write-Info '  via winget (OpenJS.NodeJS.LTS)...'
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    winget install --id OpenJS.NodeJS.LTS --silent --accept-package-agreements --accept-source-agreements --disable-interactivity *> $null
    $ErrorActionPreference = $prevEAP
    if ($LASTEXITCODE -eq 0) { $installed = $true }
  }

  # Tentativa 2: MSI oficial do nodejs.org (latest v24.x) com elevacao UAC
  if (-not $installed) {
    Write-Info '  winget indisponivel ou falhou - fallback: MSI oficial do nodejs.org...'
    try {
      $html = (Invoke-WebRequest -Uri 'https://nodejs.org/dist/latest-v24.x/' -UseBasicParsing).Content
      $MsiName = [regex]::Match($html, 'node-v[\d\.]+-x64\.msi').Value
      if (-not $MsiName) { throw 'MSI do Node 24 nao listado no indice do nodejs.org.' }
      $MsiPath = Join-Path $env:TEMP $MsiName
      Write-Info "  baixando $MsiName (~30 MB)..."
      Invoke-WebRequest -Uri "https://nodejs.org/dist/latest-v24.x/$MsiName" -OutFile $MsiPath -UseBasicParsing
      Write-Info '  instalando silenciosamente (confirme o UAC se pedir)...'
      $msiProc = Start-Process msiexec -ArgumentList "/i `"$MsiPath`" /qn /norestart" -Verb RunAs -Wait -PassThru
      if ($msiProc.ExitCode -eq 0) { $installed = $true }
      else { Write-Warn2 "  msiexec saiu com codigo $($msiProc.ExitCode)." }
      Remove-Item $MsiPath -Force -ErrorAction SilentlyContinue
    } catch {
      Write-Warn2 "  fallback MSI falhou: $($_.Exception.Message)"
    }
  }

  # Novos terminais pegam o PATH atualizado pelo MSI; a sessao atual nao — injeta.
  if (Test-Path 'C:\Program Files\nodejs') {
    $env:Path = "$env:Path;C:\Program Files\nodejs"
  }

  if (-not (Test-Node24)) {
    Die 'Node 24 nao ficou disponivel apos a instalacao. Instale manualmente (https://nodejs.org) e rode o instalador de novo.'
  }
  Write-Ok "Node 24 instalado no host ($(& node --version))."
}

# -- 2. Download do ultimo release ---------------------------------------------
Write-Info "Consultando ultimo release de https://github.com/$Repo..."

try {
  $Release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" -UseBasicParsing
} catch {
  Die "Falha ao consultar a API do GitHub (sem releases? sem internet?): $($_.Exception.Message)"
}

$Asset = $Release.assets | Where-Object { $_.name -match 'windows' -and $_.name -match '\.zip$' } | Select-Object -First 1
if (-not $Asset) {
  Die "Release $($Release.tag_name) nao tem asset *windows*.zip. Veja: https://github.com/$Repo/releases"
}

$Tag = $Release.tag_name
Write-Info "Release $Tag - baixando $($Asset.name) (~700 MB)..."

$TmpZip = Join-Path $env:TEMP "alpha-ai-$Tag.zip"
Invoke-WebRequest -Uri $Asset.browser_download_url -OutFile $TmpZip -UseBasicParsing
Write-Ok 'Download concluido.'

# -- 3. Extracao ---------------------------------------------------------------
Write-Info "Extraindo em $InstallDir ..."
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Expand-Archive -Path $TmpZip -DestinationPath $InstallDir -Force
Remove-Item -Force $TmpZip

$ModelsZip = Join-Path $InstallDir 'Modules\Models.zip'
if (-not (Test-Path $ModelsZip)) {
  Die "Modules/Models.zip nao veio no zip do release - asset incompativel com esta versao do instalador."
}
Set-Location $InstallDir
Write-Ok 'Arquivos extraidos.'

# -- 4. Modelos ----------------------------------------------------------------
Write-Info 'Extraindo Models.zip (~1,4 GB) para Modules/Models...'
Expand-Archive -Path $ModelsZip -DestinationPath (Join-Path $InstallDir 'Modules\Models') -Force
Write-Ok 'Modelos extraidos.'
if (-not $KeepModelsZip) {
  Remove-Item -Force $ModelsZip
  Write-Ok 'Modules/Models.zip removido (economia de ~450 MB).'
}

# -- 4b. Verificacao de modelos ------------------------------------------------
# Modelo truncado (download/extracao interrompida) deixa o onnxManager fora
# do ar (INVALID_PROTOBUF). O verify confere tamanho minimo e re-baixa o que
# estiver corrompido. Best-effort: sem python no host, a validacao acontece
# dentro do container no boot (exit 78 + instrucao no log).
$VerifyScript = Join-Path $InstallDir 'scripts\verify_models.py'
$Py = Get-Command python -ErrorAction SilentlyContinue
if (-not $Py) { $Py = Get-Command python3 -ErrorAction SilentlyContinue }
if ((Test-Path $VerifyScript) -and $Py) {
  Write-Info 'Verificando integridade dos modelos (verify_models)...'
  & $Py.Source $VerifyScript --quiet
  if ($LASTEXITCODE -ne 0) {
    Write-Warn2 'Modelo invalido detectado - tentando re-download...'
    & $Py.Source $VerifyScript --download
    if ($LASTEXITCODE -eq 0) { Write-Ok 'Modelos reparados.' }
    else { Write-Warn2 'Nao foi possivel reparar os modelos automaticamente - o boot do container vai indicar o arquivo exato.' }
  } else {
    Write-Ok 'Modelos integros.'
  }
} else {
  Write-Warn2 'python/scripts indisponiveis no host - verificacao de modelos sera feita dentro do container no boot.'
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

function Prompt-Key {
  while ($true) {
    $secure = Read-Host -Prompt 'Cole sua OPENROUTER_API_KEY (https://openrouter.ai/keys)' -AsSecureString
    $plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
    $Key = $plain.Trim()
    if (-not $Key) { Write-Warn2 'Chave vazia - tente de novo.'; continue }
    if ($Key -notlike 'sk-or-*') { Write-Warn2 "Chaves do OpenRouter normalmente comecam com 'sk-or-'. Prosseguindo mesmo assim." }
    Write-Key $Key
    return
  }
}

if (Test-ExistingKey) {
  $ans = Read-Host 'Ja existe uma OPENROUTER_API_KEY em Modules\.env. Manter a chave existente? [S/n]'
  if ($ans -match '^[nN]') { Prompt-Key }
} else {
  Prompt-Key
}
Write-Ok "OPENROUTER_API_KEY configurada em $EnvFile"

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

# -- 5b''. Token da extensão do Playwright (PLAYWRIGHT_MCP_EXTENSION_TOKEN) ---
# O @playwright/mcp roda com --extension e pareia com a extensão oficial do
# Chrome via este token. O token NAO e gerado pelo instalador: a propria
# extensao o cria (base64url de 32 bytes) e persiste em localStorage
# ('auth-token') na origem chrome-extension://mmlmfjhmonkocbjadbfplnigmagldckm
# quando a pagina status.html e aberta. Extraimos o valor do LevelDB de
# Local Storage do perfil do navegador. Se a extensao ainda nao estiver
# instalada, a extracao e refeita na etapa 5b, apos abrir a Chrome Web Store.

$PlaywrightExtensionUrl = 'https://chromewebstore.google.com/detail/playwright-extension/mmlmfjhmonkocbjadbfplnigmagldckm'
$PlaywrightExtensionStatusUrl = 'chrome-extension://mmlmfjhmonkocbjadbfplnigmagldckm/status.html'

# Procura o 'auth-token' da extensao no LevelDB de Local Storage dos perfis
# dos navegadores Chromium instalados e imprime o token (43 chars base64url).
# A pagina status.html renderiza o token via React - o valor confiavel esta no
# localStorage persistido, nao no HTML estatico.
function Get-PlaywrightExtensionToken {
  # Extracao ANCORADA na origem da extensao. Um regex solto de 43 chars gera
  # FALSO POSITIVO: outros sites gravam valores de 43 chars no Local Storage
  # (caso real: token extraido errado vindo de mega.nz). A chave completa no
  # LevelDB e '_chrome-extension://<id>\x00\x01auth-token' e o token (43
  # chars base64url) aparece na janela de 64B logo apos a chave.
  $extId = 'mmlmfjhmonkocbjadbfplnigmagldckm'
  $keyNeedle = [System.Text.Encoding]::UTF8.GetBytes("chrome-extension://$extId`0`1auth-token")
  $tokenRegex = [regex]'[A-Za-z0-9_-]{43}'
  $window = 64 + 43

  $leveldbRoots = @(
    (Join-Path $env:LOCALAPPDATA 'Google\Chrome\User Data'),
    (Join-Path $env:LOCALAPPDATA 'Chromium\User Data'),
    (Join-Path $env:LOCALAPPDATA 'Microsoft\Edge\User Data'),
    (Join-Path $env:LOCALAPPDATA 'BraveSoftware\Brave-Browser\User Data')
  )

  $files = @()
  foreach ($root in $leveldbRoots) {
    if (-not (Test-Path $root)) { continue }
    $files += Get-ChildItem -Path $root -Filter '*' -Directory -ErrorAction SilentlyContinue |
      ForEach-Object { Join-Path $_.FullName 'Local Storage\leveldb' } |
      Where-Object { Test-Path $_ } |
      ForEach-Object { Get-ChildItem -Path $_ -File -ErrorAction SilentlyContinue } |
      Where-Object { $_.Extension -in @('.ldb', '.log') }
  }
  # Ordem de geracao do LevelDB: numero do arquivo maior = write mais recente.
  $files = $files | Sort-Object { [int]($_.BaseName -replace '\D', '') } -Descending

  foreach ($file in $files) {
    try { $raw = [System.IO.File]::ReadAllBytes($file.FullName) } catch { continue }
    # Varre todas as ocorrencias da chave exata; a primeira neste arquivo
    # (mais novo) ja e o write mais recente do token.
    $pos = 0
    while ($true) {
      $i = Find-Bytes $raw $keyNeedle $pos
      if ($i -lt 0) { break }
      $start = $i + $keyNeedle.Length
      $end = [Math]::Min($start + $window, $raw.Length)
      $seg = New-Object byte[] ($end - $start)
      [Array]::Copy($raw, $start, $seg, 0, $end - $start)
      $m = $tokenRegex.Match([System.Text.Encoding]::UTF8.GetString($seg))
      if ($m.Success) { return $m.Value }
      $pos = $i + 1
    }
  }
  return $null
}

# Posicao da primeira ocorrencia de $Needle em $Haystack a partir de $Start
function Find-Bytes {
  param([byte[]]$Haystack, [byte[]]$Needle, [int]$Start = 0)
  if ($Needle.Length -eq 0 -or $Haystack.Length -lt $Needle.Length) { return -1 }
  for ($i = $Start; $i -le $Haystack.Length - $Needle.Length; $i++) {
    $match = $true
    for ($j = 0; $j -lt $Needle.Length; $j++) {
      if ($Haystack[$i + $j] -ne $Needle[$j]) { $match = $false; break }
    }
    if ($match) { return $i }
  }
  return -1
}

if (-not (Get-EnvVar 'PLAYWRIGHT_MCP_EXTENSION_TOKEN')) {
  Write-Info 'Procurando token da extensao Playwright no perfil do navegador...'
  $token = Get-PlaywrightExtensionToken
  if ($token) {
    Set-EnvVar 'PLAYWRIGHT_MCP_EXTENSION_TOKEN' $token
    Write-Ok "PLAYWRIGHT_MCP_EXTENSION_TOKEN extraido do status.html da extensao -> $EnvFile"
  } else {
    Write-Warn2 'Extensao Playwright ainda nao instalada (token nao encontrado no perfil).'
    Write-Warn2 'A instalacao abrira a Chrome Web Store na etapa 5b e extraira o token apos a instalacao.'
  }
} else {
  Write-Ok "PLAYWRIGHT_MCP_EXTENSION_TOKEN ja existe em $EnvFile"
  # Verificacao de drift — o mecanismo de autenticacao da extensao: a
  # connect.html compara o 'token' da URL com o 'auth-token' do localStorage;
  # divergencia = "Invalid token provided" em TODA conexao (o servidor manda o
  # token velho do env). Se o token do .env nao e o valor atual da extensao
  # (regenerado na status.html), atualiza o .env; o re-registro adiante derruba
  # a sessao do @playwright/mcp no mcp_manager, que respawna com o token novo.
  $fresh = Get-PlaywrightExtensionToken
  if ($fresh -and $fresh -ne (Get-EnvVar 'PLAYWRIGHT_MCP_EXTENSION_TOKEN')) {
    Write-Warn2 'PLAYWRIGHT_MCP_EXTENSION_TOKEN divergente: a extensao esta usando outro token (regenerado?).'
    Set-EnvVar 'PLAYWRIGHT_MCP_EXTENSION_TOKEN' $fresh
    Write-Ok "PLAYWRIGHT_MCP_EXTENSION_TOKEN atualizado -> $EnvFile (o re-registro adiante respawna o servidor)."
  }
}

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
$McpManagerUrl = 'http://127.9.12.22:4008'

function Register-PlaywrightMcp {
  # O env do registro e autocontido (o mcp_manager mescla server.env sobre o
  # ambiente herdado — a linha vale mesmo sem o tray injetando o env):
  #   - PLAYWRIGHT_MCP_EXTENSION_TOKEN: a connect.html da extensao compara o
  #     'token' da URL com o 'auth-token' do localStorage; igual -> a conexao e
  #     aprovada sozinha (sem dialogo manual por chamada).
  #   - PLAYWRIGHT_EXTENSION_PROTOCOL=2: a extensao exige exatamente a
  #     protocolVersion 2 (v<2 -> "unsupported protocol version", v>2 ->
  #     versionMismatch); forcar 2 mantem o par testado com a extensao 0.4.x
  #     da loja mesmo se o @playwright/mcp@latest subir o default do relay
  #     (o env e lido pelo relay no spawn: process.env.PLAYWRIGHT_EXTENSION_PROTOCOL).
  $token = Get-EnvVar 'PLAYWRIGHT_MCP_EXTENSION_TOKEN'
  $envMap = @{ PLAYWRIGHT_EXTENSION_PROTOCOL = '2' }
  if ($token) { $envMap['PLAYWRIGHT_MCP_EXTENSION_TOKEN'] = $token }
  $payload = @{
    name           = 'playwright'
    skill_summary  = 'Automação de navegador (navegar, clicar, extrair conteúdo)'
    transport      = 'stdio'
    command        = 'npx'
    args           = @('-y', '@playwright/mcp@latest', '--extension')
    env            = $envMap
  } | ConvertTo-Json -Depth 3

  # A extensao precisa ser instalada manualmente na Chrome Web Store -
  # o Playwright MCP no modo --extension conecta a ela via bridge. Com o
  # token ja no .env (extensao instalada em execucao anterior) a abertura e
  # pulada: re-executar o instalador nao deve reabrir a loja.
  if (-not (Get-EnvVar 'PLAYWRIGHT_MCP_EXTENSION_TOKEN')) {
    Write-Info 'Abrindo a Chrome Web Store para instalar a extensao do Playwright...'
    Write-Info $PlaywrightExtensionUrl
    try { Start-Process $PlaywrightExtensionUrl } catch {
      Write-Warn2 'Nao consegui abrir o navegador - instale a extensao manualmente:'
      Write-Warn2 $PlaywrightExtensionUrl
    }
  }

  # Se o token ainda nao foi extraido (extensao instalada agora), abre a
  # status.html - a extensao entao grava o token no localStorage do perfil -
  # e extrai o valor do LevelDB de Local Storage. Pausa para dar tempo do
  # usuario concluir a instalacao na loja e o Chrome gravar em disco.
  if (-not (Get-EnvVar 'PLAYWRIGHT_MCP_EXTENSION_TOKEN')) {
    Read-Host 'Instale a extensao na janela aberta e pressione ENTER para continuar' | Out-Null
    Write-Info "Abrindo a pagina de status da extensao ($PlaywrightExtensionStatusUrl)..."
    try { Start-Process $PlaywrightExtensionStatusUrl } catch {}
    Write-Info 'Aguardando o Chrome persistir o token (30 s)...'
    Start-Sleep -Seconds 30
    $token = Get-PlaywrightExtensionToken
    if ($token) {
      Set-EnvVar 'PLAYWRIGHT_MCP_EXTENSION_TOKEN' $token
      Write-Ok "PLAYWRIGHT_MCP_EXTENSION_TOKEN extraido do status.html da extensao -> $EnvFile"
    } else {
      Write-Warn2 'Nao consegui extrair o token automaticamente.'
      Write-Warn2 'Abra a pagina de status da extensao, copie o token e adicione ao Modules\.env:'
      Write-Warn2 "  $PlaywrightExtensionStatusUrl"
      Write-Warn2 '  PLAYWRIGHT_MCP_EXTENSION_TOKEN=<token>'
    }
  }

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

function Register-WebsearchMcp {
  # URL do MCP do websearch (server.py, streamable-http em :4010/mcp). O manager
  # roda DENTRO do container: localhost seria ele mesmo, entao o default usa
  # host.docker.internal (porta 4010 publicada no host). Se o websearch for um
  # servico do mesmo compose, sobrescreva: $env:WEBSEARCH_MCP_URL = 'http://websearch:4010/mcp'
  $wsUrl = if ($env:WEBSEARCH_MCP_URL) { $env:WEBSEARCH_MCP_URL } else { 'http://host.docker.internal:4010/mcp' }
  return (Register-McpServer -Label 'websearch' -Name 'websearch' -SyncTimeoutSec 120 -Spec @{
    name          = 'websearch'
    skill_summary = 'Busca na web: pesquisa fatos atuais e retorna trechos relevantes das melhores páginas, com título e URL para citação.'
    transport     = 'http'
    url           = $wsUrl
    enabled       = $true
    always_on     = $false
    examples      = @{
      web_search = @(
        'pesquise na internet sobre',
        'qual é a notícia mais recente sobre',
        'busque informações atuais de',
        'procure na web'
      )
      memory_write_batch = @(
        'lembre que',
        'guarde isso na memoria',
        'anote que',
        'salve essa informacao',
        'lembre-se de que',
        'memorize os fatos deste documento'
      )
    }
  })
}

function Register-DesktopCommanderMcp {
  return (Register-McpServer -Label 'Desktop Commander' -Name 'desktop-commander' -SyncTimeoutSec 240 -Spec @{
    name          = 'desktop-commander'
    skill_summary = 'Arquivos e terminal do computador (ler/escrever/editar arquivos, listar diretórios, buscar, executar comandos e processos)'
    transport     = 'stdio'
    command       = 'npx'
    args          = @('-y', '@wonderwhy-er/desktop-commander@latest')
    enabled       = $true
    always_on     = $false
  })
}

# -- 5d. Manager (ava-tray.exe) - API REST em :9001 --------------------------
# O Manager sobe/derruba o stack Docker e supervisiona o mcp_manager do host.
# O instalador o inicia primeiro e pede o start do Docker pela API dele.
$ManagerUrl = 'http://127.9.12.22:9001'

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

# -- 5e. Rede da rota B: relays socat na VM + portproxy no Windows -------------
# Rota B = Docker Engine dentro do Debian-slim. O compose publica as portas em
# 127.9.12.22 DENTRO da VM — o Windows nao enxerga esse loopback (o auto-forward
# do WSL2 cobre apenas 127.0.0.1, issue microsoft/WSL#11034). Ponte em 3 camadas:
#   (1) VM:      socat 0.0.0.0:PORTA -> 127.9.12.22:PORTA (stack, 16 portas)
#                socat 0.0.0.0:PORTA -> GW_WSL:PORTA      (servicos do HOST)
#   (2) Windows: portproxy 127.9.12.22:PORTA -> IP_VM:PORTA  (host -> containers)
#   (3) Windows: portproxy GW_WSL:9001/9222/9100 -> loopback (containers -> host)
# A porta 4008 fica FORA: o mcp_manager roda no HOST e publicar a porta colidiria.
# Rota A (Docker Desktop) nao precisa de nada disso: ele publica no loopback do
# Windows e host.docker.internal ja aponta para o host.
if ($script:UseWSLDocker) {
  Write-Info 'Rota B: instalando relays socat no Debian-slim...'

  # (1a) socat dentro da VM
  wsl -d Debian-slim -u root -- bash -c "command -v socat >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq socat; }"
  if ($LASTEXITCODE -ne 0) { Die 'Falha ao instalar o socat no Debian-slim.' }

  # Script de relay: alvos do stack sao fixos (127.9.12.22 na VM); os do HOST
  # dependem do gateway do WSL (vEthernet do Windows) — resolvido a cada boot.
  $RelaySh = @'
#!/bin/bash
# Alpha AI - relays socat da rota WSL (gerado pelo install.ps1)
# (a) stack: 0.0.0.0:PORTA -> 127.9.12.22:PORTA (publicacao do docker na VM)
# (b) host:  0.0.0.0:PORTA -> GW:PORTA          (tray/CDP/transcribe no Windows)
GW="$(ip route show default | awk '{print $3}' | head -n1)"
APP_PORTS="3000 4005 4003 3003 3001 2002 3002 3004 4002 9000 3005 2001 8081 6379 11235 4010"
HOST_PORTS="9001 9222 9100"
for p in $APP_PORTS; do
  socat "TCP-LISTEN:$p,bind=0.0.0.0,fork,reuseaddr" "TCP:127.9.12.22:$p" &
done
for p in $HOST_PORTS; do
  socat "TCP-LISTEN:$p,bind=0.0.0.0,fork,reuseaddr" "TCP:$GW:$p" &
done
wait -n
'@
  $RelaySh = $RelaySh -replace "`r`n", "`n"
  $RelayTmp = Join-Path $env:TEMP 'alpha-ai-relay.sh'
  [System.IO.File]::WriteAllText($RelayTmp, $RelaySh, (New-Object System.Text.UTF8Encoding($false)))
  $RelayWsl = ((wsl -d Debian-slim -- wslpath -a $RelayTmp | Out-String) -replace "`0", '').Trim()
  if (-not $RelayWsl) { Die 'Nao consegui converter o caminho do relay para o WSL (wslpath).' }

  $SvcTmp = Join-Path $env:TEMP 'alpha-ai-relay.service'
  if ((wsl -d Debian-slim -u root -- bash -c "test -d /run/systemd/system && echo yes || echo no").Trim() -eq 'yes') {
    # Com systemd: servico garante os relays a cada boot do WSL
    $RelaySvc = @'
[Unit]
Description=Alpha AI relay (socat) - rota WSL
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/alpha-ai-relay.sh
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
'@
    $RelaySvc = $RelaySvc -replace "`r`n", "`n"
    [System.IO.File]::WriteAllText($SvcTmp, $RelaySvc, (New-Object System.Text.UTF8Encoding($false)))
    $SvcWsl = ((wsl -d Debian-slim -- wslpath -a $SvcTmp | Out-String) -replace "`0", '').Trim()
    wsl -d Debian-slim -u root -- bash -c "cp '$RelayWsl' /usr/local/bin/alpha-ai-relay.sh && chmod 755 /usr/local/bin/alpha-ai-relay.sh && cp '$SvcWsl' /etc/systemd/system/alpha-ai-relay.service && systemctl daemon-reload && systemctl enable --now alpha-ai-relay.service"
    if ($LASTEXITCODE -ne 0) { Die 'Falha ao registrar o servico alpha-ai-relay no Debian-slim.' }
    Write-Ok 'Relays socat ativos (systemd: alpha-ai-relay.service).'
  } else {
    # Sem systemd: relays soltos via nohup (morrem em wsl --shutdown)
    Write-Warn2 'systemd inativo no Debian-slim - relays via nohup (morrem em wsl --shutdown).'
    wsl -d Debian-slim -u root -- bash -c "pkill -f alpha-ai-relay.sh 2>/dev/null; cp '$RelayWsl' /usr/local/bin/alpha-ai-relay.sh && chmod 755 /usr/local/bin/alpha-ai-relay.sh && nohup /usr/local/bin/alpha-ai-relay.sh >/var/log/alpha-ai-relay.log 2>&1 & sleep 1; pgrep -f alpha-ai-relay.sh >/dev/null"
    if ($LASTEXITCODE -ne 0) { Die 'Falha ao iniciar os relays socat no Debian-slim.' }
  }
  Remove-Item $RelayTmp, $SvcTmp -Force -ErrorAction SilentlyContinue

  # (2)+(3) portproxy no Windows — exige admin; eleva via UAC quando necessario
  Write-Info 'Rota B: criando regras netsh portproxy no Windows (pode pedir UAC)...'
  $VmIp = ((wsl -d Debian-slim -- hostname -I | Out-String) -replace "`0", '').Trim().Split(' ')[0]
  if (-not $VmIp) { Die 'Nao consegui obter o IP da VM Debian-slim (hostname -I).' }
  $GwIp = [regex]::Match(((wsl -d Debian-slim -- ip route show default | Out-String) -replace "`0", ''), 'default via (\S+)').Groups[1].Value
  if (-not $GwIp) { Die 'Nao consegui obter o gateway do WSL (IP do vEthernet).' }

  $PortProxyPs1 = @'
# Regras netsh portproxy da rota B do Alpha AI (gerado pelo install.ps1)
# (2) host -> containers: 127.9.12.22 (loopback Windows) -> IP da VM (socat)
$appPorts = @(3000,4005,4003,3003,3001,2002,3002,3004,4002,9000,3005,2001,8081,6379,11235,4010)
foreach ($p in $appPorts) {
  netsh interface portproxy delete v4tov4 listenaddress=127.9.12.22 listenport=$p | Out-Null
  netsh interface portproxy add v4tov4 listenaddress=127.9.12.22 listenport=$p connectaddress=VMIP connectport=$p | Out-Null
}
# (3) containers -> HOST: tray :9001 (bind 127.9.12.22), Chrome CDP :9222 e
# transcribe :9100 (bind 127.0.0.1). O container chega pelo gateway do WSL.
$gw = 'GWIP'
foreach ($r in @(@(9001,'127.9.12.22'), @(9222,'127.0.0.1'), @(9100,'127.0.0.1'))) {
  netsh interface portproxy delete v4tov4 listenaddress=$gw listenport=$($r[0]) | Out-Null
  netsh interface portproxy add v4tov4 listenaddress=$gw listenport=$($r[0]) connectaddress=$($r[1]) connectport=$($r[0]) | Out-Null
}
# Firewall: libera a VM alcancar as portas de servicos do host via vEthernet
netsh advfirewall firewall delete rule name="Alpha AI WSL relay" | Out-Null
netsh advfirewall firewall add rule name="Alpha AI WSL relay" dir=in action=allow protocol=TCP localip=$gw localport=9001,9222,9100 | Out-Null
'@
  $PortProxyPs1 = $PortProxyPs1.Replace('VMIP', $VmIp).Replace('GWIP', $GwIp)
  $PpTmp = Join-Path $env:TEMP 'alpha-ai-portproxy.ps1'
  [System.IO.File]::WriteAllText($PpTmp, $PortProxyPs1, (New-Object System.Text.UTF8Encoding($false)))

  $IsAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
  try {
    if ($IsAdmin) {
      & $PpTmp
    } else {
      Start-Process powershell -Verb RunAs -Wait -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PpTmp`"")
    }
  } catch {
    Die "Falha ao criar as regras de portproxy (UAC recusado?): $($_.Exception.Message)"
  }

  $ppShow = (netsh interface portproxy show v4tov4 | Out-String)
  if ($ppShow -notmatch '127\.9\.12\.22') {
    Die 'As regras de portproxy nao foram criadas (precisa de admin). Rode o install.ps1 como Administrador.'
  }
  Remove-Item $PpTmp -Force -ErrorAction SilentlyContinue
  Write-Ok 'Portproxy criado: 127.9.12.22 -> VM (stack) e gateway WSL -> loopback (host).'
  Write-Warn2 'Nota: o IP da VM pode mudar apos reiniciar o WSL/PC - se os servicos nao responderem, rode o install.ps1 novamente.'
  Write-Info 'Nota (rota B): o tray supervisiona o docker via CLI do host - nessa rota ele nao vera o stack; o instalador sobe o stack direto via compose.'
}

# Wrapper de compose com suporte as duas rotas:
#   A (Docker Desktop): docker-start.bat no host, como sempre.
#   B (Docker Engine no Debian-slim): compose roda DENTRO da VM — nao existe
#      CLI docker no host. O caminho do projeto e traduzido via wslpath.
function Invoke-Compose {
  if ($script:UseWSLDocker) {
    $wslDir = ((wsl -d Debian-slim -- wslpath -a $InstallDir | Out-String) -replace "`0", '').Trim()
    $envArgs = @()
    if (Test-Path (Join-Path $InstallDir 'Modules\.env')) { $envArgs = @('--env-file', 'Modules/.env') }
    wsl -d Debian-slim --cd $wslDir -- docker compose @envArgs -f docker-compose.yml -f docker-compose.windows.yml @args
    return $LASTEXITCODE
  }
  & .\docker-start.bat @args
  return $LASTEXITCODE
}
$script:LogsHint = if ($script:UseWSLDocker) {
  "wsl -d Debian-slim --cd `"$InstallDir`" -- docker compose logs -f ava"
} else {
  "$InstallDir\docker-start.bat logs -f ava"
}

# -- 6. Build + first up -------------------------------------------------------
if (-not $SkipBuild) {
  Write-Info 'Build inicial das imagens Docker (pode levar 5-15 min na primeira vez)...'
  Invoke-Compose build
  if ($LASTEXITCODE -ne 0) { Die 'docker compose build falhou - rode o build novamente para ver o erro.' }

  # Manager primeiro: e ele quem sobe o stack Docker (POST /docker/start).
  # Se o Manager falhar, cai para o compose direto.
  $viaManager = (Start-Manager) -and (Start-DockerViaManager)
  if (-not $viaManager) {
    Write-Warn2 'Subindo o stack diretamente via compose...'
    Invoke-Compose up -d
    if ($LASTEXITCODE -ne 0) { Die 'docker compose up falhou.' }
  }

  # mcp_manager (host) ativo antes do cadastro dos MCPs
  Ensure-McpManager | Out-Null

  # Registro dos MCPs - logo apos o up, quando o stack ja esta carregado.
  Register-PlaywrightMcp | Out-Null
  Register-WebsearchMcp | Out-Null
  Register-DesktopCommanderMcp | Out-Null

  $StatusUrl = 'http://127.9.12.22:9000/status'
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
        Write-Info "  ...aguardando ha $([int]($Elapsed / 60)) min (logs: $script:LogsHint)"
      }
      Start-Sleep -Seconds 5
      $Elapsed += 5
    }
  }
  if ($Elapsed -ge $MaxWait) {
    Write-Warn2 "Orchestrator nao respondeu em 20 min. Veja: $script:LogsHint"
  }

  try {
    $Status = Invoke-WebRequest -Uri $StatusUrl -UseBasicParsing -TimeoutSec 5
    if ($Status.Content -match '"openrouter_key":\s*"missing"') {
      Write-Warn2 'O orchestrator reportou openrouter_key: missing.'
      $ans = Read-Host 'Deseja re-informar a chave agora? [S/n]'
      if ($ans -notmatch '^[nN]') {
        Prompt-Key
        Write-Info 'Reiniciando container ava para reler a chave...'
        Invoke-Compose restart ava
      }
    }
  } catch { }
} else {
  Write-Info '-SkipBuild: pulei o build/up do Docker.'
}

# -- 7. PATH do usuario --------------------------------------------------------
$TuiDir = Join-Path $InstallDir 'tui'
$UserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (-not $UserPath) { $UserPath = '' }
if ($UserPath -notlike "*$TuiDir*") {
  $NewPath = if ($UserPath) { "$UserPath;$TuiDir" } else { $TuiDir }
  [Environment]::SetEnvironmentVariable('Path', $NewPath, 'User')
  Write-Ok "Comando instalado: alpha-ai (em $TuiDir). Reabra o terminal para valer."
} else {
  Write-Ok "Comando alpha-ai ja no PATH."
}

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

# -- 9. Encerramento -----------------------------------------------------------
Write-Host ''
Write-Ok "Instalacao concluida ($InstallDir)"
Write-Info 'Comandos uteis:'
Write-Info '  alpha-ai                          # abre a TUI (novo terminal)'
Write-Info "  $script:LogsHint   # logs do backend"

if (-not $NoLaunch) {
  Write-Info 'Abrindo a TUI...'
  Start-Process -FilePath (Join-Path $TuiDir 'alpha-ai.exe')
}
