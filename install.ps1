# install.ps1 - Instalador Alpha AI (Windows)
#
# Espelha o fluxo do install.sh:
#   1. Valida prereqs (Docker + compose v2)
#   2. Baixa o asset *windows*.zip do ultimo release no GitHub
#   3. Extrai em %LOCALAPPDATA%\Programs\AlphaAI (override: -InstallDir)
#   4. Extrai Modules/Models.zip em Modules/Models
#   5. Pede a OPENROUTER_API_KEY e grava em Modules\.env - ANTES do up,
#      porque Modules/LLM.py le a chave no import
#   6. Build + up via docker-start.bat e espera o /status responder
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

# -- 1. Prereqs ----------------------------------------------------------------
Write-Info 'Validando pre-requisitos...'

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Die "Docker nao encontrado. Instale o Docker Desktop (com WSL2) e rode novamente."
}
try { docker compose version | Out-Null } catch {
  Die "Plugin 'docker compose' ausente - atualize o Docker Desktop."
}
# EAP=Stop + stderr de comando nativo redirecionado dispara NativeCommandError
# (derruba o script exatamente quando o daemon esta desligado) - relaxa aqui.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
docker info *> $null
$dockerOk = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $prevEAP
if (-not $dockerOk) {
  Die "Docker indisponivel. Inicie o Docker Desktop e aguarde ele ficar 'running'."
}

Write-Ok 'Pre-requisitos validados.'

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

function Test-ExistingKey {
  if (-not (Test-Path $EnvFile)) { return $false }
  $line = Select-String -Path $EnvFile -Pattern '^OPENROUTER_API_KEY=.+' -Quiet
  return [bool]$line
}

function Write-Key {
  param([string]$Key)
  # UTF-8 sem BOM (dotenv nao aceita BOM)
  [System.IO.File]::WriteAllText($EnvFile, "OPENROUTER_API_KEY=$Key`n", (New-Object System.Text.UTF8Encoding($false)))
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
        -ContentType 'application/json' -Body $payload -TimeoutSec 10 | Out-Null
      $registered = $true
      break
    } catch {
      Start-Sleep -Seconds 5
    }
  }
  if (-not $registered) {
    Write-Warn2 'Nao consegui cadastrar o Playwright MCP - manager nao respondeu em 2 min.'
    Write-Warn2 "Cadastre manualmente depois: POST $McpManagerUrl/servers + POST $McpManagerUrl/servers/playwright/sync"
    return $false
  }

  # /sync: exige a memory_api (:3000) de pe; o 1o sync baixa o pacote npm.
  $synced = $false
  for ($i = 0; $i -lt 12; $i++) {
    try {
      Invoke-RestMethod -Uri "$McpManagerUrl/servers/playwright/sync" -Method Post -TimeoutSec 240 | Out-Null
      $synced = $true
      break
    } catch {
      Start-Sleep -Seconds 10
    }
  }
  if (-not $synced) {
    Write-Warn2 'Playwright MCP cadastrado, mas o sync falhou (memory_api ainda bootando?).'
    Write-Warn2 "Rode manualmente: curl -X POST $McpManagerUrl/servers/playwright/sync"
    return $false
  }
  Write-Ok 'Playwright MCP registrado e sincronizado.'
  return $true
}

# -- 6. Build + first up -------------------------------------------------------
if (-not $SkipBuild) {
  Write-Info 'Build inicial das imagens Docker (pode levar 5-15 min na primeira vez)...'
  & .\docker-start.bat build
  if ($LASTEXITCODE -ne 0) { Die 'docker compose build falhou - rode .\docker-start.bat build para ver o erro.' }

  Write-Info 'Subindo o stack...'
  & .\docker-start.bat up -d
  if ($LASTEXITCODE -ne 0) { Die 'docker compose up falhou.' }

  # Registro do Playwright MCP - logo apos o up, quando o stack ja esta carregado.
  Register-PlaywrightMcp | Out-Null

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

  try {
    $Status = Invoke-WebRequest -Uri $StatusUrl -UseBasicParsing -TimeoutSec 5
    if ($Status.Content -match '"openrouter_key":\s*"missing"') {
      Write-Warn2 'O orchestrator reportou openrouter_key: missing.'
      $ans = Read-Host 'Deseja re-informar a chave agora? [S/n]'
      if ($ans -notmatch '^[nN]') {
        Prompt-Key
        Write-Info 'Reiniciando container ava para reler a chave...'
        & .\docker-start.bat restart ava
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
  Set-ItemProperty -Path $RunKey -Name 'AlphaAITray' -Value $TrayExe -Type String
  Write-Ok 'Auto-start configurado: HKCU\...\Run\AlphaAITray'
  Start-Process -FilePath $TrayExe
  Write-Ok 'ava-tray iniciado agora.'
} else {
  Write-Warn2 "ava-tray.exe nao encontrado em $TrayExe - auto-start pulado."
}

# -- 9. Encerramento -----------------------------------------------------------
Write-Host ''
Write-Ok "Instalacao concluida ($InstallDir)"
Write-Info 'Comandos uteis:'
Write-Info '  alpha-ai                          # abre a TUI (novo terminal)'
Write-Info "  $InstallDir\docker-start.bat logs -f ava   # logs do backend"

if (-not $NoLaunch) {
  Write-Info 'Abrindo a TUI...'
  Start-Process -FilePath (Join-Path $TuiDir 'alpha-ai.exe')
}