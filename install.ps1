# install.ps1 - Instalador Alpha AI (Windows)
#
# Espelha o fluxo do install.sh:
#   1. Valida prereqs (Node 24 no host; Docker Desktop OU Docker Engine no WSL)
#   2. Baixa o asset *windows*.zip do ultimo release no GitHub - MODO TESTE:
#      $env:ALPHA_RELEASE_ZIP = 'caminho.zip' usa um zip local e pula o
#      GitHub inteiro (para testar antes do release oficial)
#   3. Extrai em %LOCALAPPDATA%\Programs\AlphaAI (override: -InstallDir)
#   3b. Agent Skills: npx skills add heygen-com/hyperframes -> .agents\skills
#       (gitignored; flags -a universal -y; falha = aviso, nao morte)
#   3c. OfficeCLI: npm install -g @officecli/officecli + skill
#       officecli.ai/SKILL.md -> .agents\skills\officecli (falha = aviso)
#   4. Extrai Modules/Models.zip em Modules/Models
#   5. Pede o provider de inferencia: LLM_BASE_URL (Enter = OpenRouter) e, se
#      o provider NAO for local, a LLM_API_KEY - grava em Modules\.env ANTES
#      do up, porque Modules/LLM.py le no import
#   6. Build; inicia o Manager (ava-tray.exe, API :9001), pede a ele o start do
#      Docker (POST /docker/start) e garante o mcp_manager (/mcp/start); registra
#      os MCPs e espera o /status responder (falha do Manager: compose up -d direto)
#   7. Adiciona Manager\ ao PATH + wrapper `alpha-ai.cmd` -> ava-web.exe
#   8. Auto-start do ava-tray.exe via HKCU\...\Run (+ inicia agora)
#   9. Inicia o WebUI (Manager\ava-web.exe)
#
# Uso: powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 [-InstallDir DIR] [-SkipBuild] [-NoLaunch]
# Env: $env:ALPHA_RELEASE_ZIP = 'caminho.zip' -> usa zip local em vez do
#      release do GitHub (modo teste antes do release oficial)

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
# `wsl --import` da rota B falha.
#
# A deteccao NAO pode depender do texto do `dism` (em Windows pt-BR a saida e
# "Estado : Habilitado", entao o regex 'State : Enabled' nunca casava e o
# script tentava habilitar a feature a cada execucao) nem de elevation (sem
# admin o dism consulta com erro 740 e tambem nao devolve o estado). Por isso
# a consulta usa fontes independentes de idioma, em ordem:
#   1. Win32_ComputerSystem.HypervisorPresent -> hypervisor rodando (feature
#      ativa e PC ja reiniciado)
#   2. Win32_OptionalFeature (CIM/WMI, funciona sem admin; InstallState 1 =
#      habilitada)
#   3. Get-WindowsOptionalFeature (enum 'Enabled', exige admin)
function Test-IsAdmin {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  return (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Retorna: 'Active' (hypervisor rodando) | 'Enabled' (habilitada, sem hypervisor
# ativo ainda) | 'Disabled' | 'Unknown' (nao foi possivel consultar)
function Get-VmpStatus {
  try {
    if ((Get-CimInstance -ClassName Win32_ComputerSystem -ErrorAction Stop).HypervisorPresent) { return 'Active' }
  } catch {}
  try {
    $f = Get-CimInstance -ClassName Win32_OptionalFeature -Filter "Name='VirtualMachinePlatform'" -ErrorAction Stop
    if ($f) { if ($f.InstallState -eq 1) { return 'Enabled' } else { return 'Disabled' } }
  } catch {}
  try {
    $f = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -ErrorAction Stop
    if ($f) { if ("$($f.State)" -eq 'Enabled') { return 'Enabled' } else { return 'Disabled' } }
  } catch {}
  return 'Unknown'
}

Write-Info 'Verificando feature do Windows: VirtualMachinePlatform...'
$VmpStatus = Get-VmpStatus

if ($VmpStatus -eq 'Active') {
  Write-Ok 'VirtualMachinePlatform habilitada e ativa (hypervisor em execucao).'
} elseif ($VmpStatus -eq 'Enabled') {
  Write-Ok 'VirtualMachinePlatform ja esta habilitada.'
  Write-Warn2 'Hypervisor ainda nao esta em execucao - se acabou de habilitar a feature, reinicie o PC antes de usar o WSL2.'
} elseif ($VmpStatus -eq 'Unknown') {
  # Nao deu para consultar: nao tenta instalar as cegas (era isso que travava).
  # Segue; se faltar mesmo, o Docker/WSL adiante acusa com mensagem clara.
  Write-Warn2 'Nao foi possivel verificar a VirtualMachinePlatform - seguindo sem habilitar.'
  Write-Warn2 'Se o WSL falhar adiante, rode: dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart (como Administrador) e reinicie.'
} else {
  # Disabled: so aqui habilita (o dism exige elevation/UAC)
  if (-not (Test-IsAdmin)) {
    Die 'VirtualMachinePlatform esta desabilitada e habilitar exige permissao de Administrador. Abra o PowerShell como Administrador e rode o install.ps1 de novo.'
  }
  Write-Warn2 'VirtualMachinePlatform desabilitada - habilitando...'
  $prevEAP = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
  $dismCode = $LASTEXITCODE
  $ErrorActionPreference = $prevEAP
  # 3010 = sucesso, reinicio necessario
  if ($dismCode -ne 0 -and $dismCode -ne 3010) {
    Die "Falha ao habilitar VirtualMachinePlatform (dism saiu com $dismCode)."
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
  # NAO capturar o retorno de Invoke-Docker: o `*> $null` descartava tambem o
  # codigo devolvido por `return $LASTEXITCODE` ($code ficava vazio e o teste
  # dava sempre falso). O codigo de saida real vem de $LASTEXITCODE.
  $null = Invoke-Docker info *> $null
  $ok = ($LASTEXITCODE -eq 0)
  $ErrorActionPreference = $prevEAP
  return $ok
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

# Helpers WSL. O wsl.exe existe no Windows mesmo SEM o WSL instalado (e so um
# stub que imprime "O Subsistema do Windows para Linux nao esta instalado"). Com
# $ErrorActionPreference='Stop', o stderr de um comando nativo vira excecao no
# PS 5.1 (NativeCommandError) - por isso as chamadas aqui rodam em 'Continue'.
function Test-WslInstalled {
  if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) { return $false }
  $prevEAP = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { $null = & wsl.exe --status 2>&1; $ok = ($LASTEXITCODE -eq 0) } catch { $ok = $false }
  $ErrorActionPreference = $prevEAP
  return $ok
}

# Lista as distros (saida do wsl e UTF-16: remove os NULs). Vazio se nao houver.
function Get-WslDistros {
  if (-not (Test-WslInstalled)) { return @() }
  $prevEAP = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { $out = (& wsl.exe --list --quiet 2>$null) } catch { $out = @() }
  $ErrorActionPreference = $prevEAP
  return @($out | ForEach-Object { ($_ -replace "`0", '').Trim() } | Where-Object { $_ })
}

# Teto de RAM da VM do WSL2 (.wslconfig), calculado pela memoria LIVRE agora.
# Sem isso o WSL2 tenta reservar 50% da RAM total e a criacao da VM aborta
# (Wsl/Service/RegisterDistro/CreateVm/E_ABORT, 0x8007000E) quando a livre e menor.
# Cria o arquivo se nao existir; se existir, so reduz a linha memory= quando ela
# excede a memoria livre (guarda backup em .wslconfig.bak).
function Set-WslMemoryCap {
  $cfg = Join-Path $env:USERPROFILE '.wslconfig'
  try {
    $freeGb  = [math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)
    $totalGb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    $cap = [int][math]::Max(2, [math]::Min(8, [math]::Min([math]::Floor($totalGb / 2), [math]::Floor($freeGb * 0.7))))
    if (-not (Test-Path $cfg)) {
      @('[wsl2]', "memory=${cap}GB", 'swap=2GB', 'vmIdleTimeout=-1', '', '[general]', 'instanceIdleTimeout=-1') | Set-Content -Path $cfg -Encoding ASCII
      Write-Ok ".wslconfig criado: WSL2 limitado a ${cap}GB (RAM total ${totalGb}GB, livre ${freeGb}GB)."
      return
    }
    $txt = Get-Content -Raw -Path $cfg
    if ($txt -match '(?im)^\s*memory\s*=\s*(\d+)\s*GB') {
      $cur = [int]$Matches[1]
      if ($cur -gt $freeGb) {
        Copy-Item -Force $cfg "$cfg.bak"
        $txt = [regex]::Replace($txt, '(?im)^(\s*memory\s*=\s*)\d+\s*GB', "`${1}${cap}GB")
        Set-Content -Path $cfg -Value $txt -Encoding ASCII
        Write-Warn2 ".wslconfig tinha memory=${cur}GB (livre: ${freeGb}GB) - reduzido para ${cap}GB (backup em .wslconfig.bak)."
      } else {
        Write-Ok ".wslconfig existente mantido (memory=${cur}GB)."
      }
    } else {
      Write-Ok '.wslconfig existente mantido.'
    }
    Set-WslKeepAlive $cfg
  } catch {
    Write-Warn2 "Nao foi possivel ajustar o .wslconfig: $($_.Exception.Message)"
  }
}

# WSL 3.x desliga a DISTRO ociosa em 15s ([general] instanceIdleTimeout),
# parando dockerd e TODOS os containers ~15-20s apos o up. O vmIdleTimeout
# (WSL2) e o timeout da VM. Desliga os dois no .wslconfig existente, sem
# sobrescrever tuning do usuario (so acrescenta as chaves se faltarem).
function Set-WslKeepAlive {
  param([string]$Path)
  try {
    if (-not (Test-Path $Path)) { return }
    $txt = Get-Content -Raw -Path $Path
    $orig = $txt
    if ($txt -notmatch '(?im)^\s*vmIdleTimeout\s*=') {
      if ($txt -match '(?im)^\[wsl2\]') {
        $txt = [regex]::Replace($txt, '(?im)^(\[wsl2\][^\r\n]*)', "`$1`r`nvmIdleTimeout=-1", 1)
      } else {
        $txt = $txt.TrimEnd() + "`r`n`r`n[wsl2]`r`nvmIdleTimeout=-1`r`n"
      }
    }
    if ($txt -notmatch '(?im)^\s*instanceIdleTimeout\s*=') {
      if ($txt -match '(?im)^\[general\]') {
        $txt = [regex]::Replace($txt, '(?im)^(\[general\][^\r\n]*)', "`$1`r`ninstanceIdleTimeout=-1", 1)
      } else {
        $txt = $txt.TrimEnd() + "`r`n`r`n[general]`r`ninstanceIdleTimeout=-1`r`n"
      }
    }
    if ($txt -ne $orig) {
      Copy-Item -Force $Path "$Path.bak"
      Set-Content -Path $Path -Value $txt -Encoding ASCII
      Write-Ok '.wslconfig: timeouts de inatividade desligados (vmIdleTimeout=-1, instanceIdleTimeout=-1).'
    }
  } catch {
    Write-Warn2 "Nao foi possivel ajustar os timeouts do .wslconfig: $($_.Exception.Message)"
  }
}

# Converte caminho do Windows (C:\Users\...) para o do WSL (/mnt/c/Users/...)
# direto no PowerShell. NAO usar `wsl -- wslpath -a <caminho>`: o argumento passa
# por um shell Linux que consome as barras invertidas (C:UsersWILLIA~1...).
function ConvertTo-WslPath {
  param([Parameter(Mandatory)][string]$Path)
  $full = [System.IO.Path]::GetFullPath($Path)
  if ($full -notmatch '^[A-Za-z]:') { return $null }
  return ('/mnt/' + $full.Substring(0, 1).ToLower() + ($full.Substring(2) -replace '\\', '/')).TrimEnd('/')
}

# Testa o Docker no WSL (Debian-slim)
$WslDockerOk = $false
if (-not $DesktopDockerOk) {
  # Verifica se o WSL esta instalado e a distro Debian-slim ja existe
  $distros = Get-WslDistros
  if ($distros -contains 'Debian-slim') {
    # Verifica se o docker responde DENTRO dela (Invoke-Docker usa o WSL so com a flag)
    $script:UseWSLDocker = $true
    if (Test-DockerAlive) { $WslDockerOk = $true } else { $script:UseWSLDocker = $false }
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
  if (-not (Test-WslInstalled)) {
    if (-not (Test-IsAdmin)) {
      Die "WSL nao esta instalado. Abra o PowerShell como Administrador e rode o install.ps1 de novo (ele instala o WSL), ou rode manualmente: wsl --install --no-distribution"
    }
    Write-Info 'WSL nao instalado - instalando (wsl --install --no-distribution)...'
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    wsl.exe --install --no-distribution
    $wslCode = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP
    if ($wslCode -ne 0) { Die "Falha ao instalar o WSL (codigo $wslCode). Rode 'wsl --install --no-distribution' manualmente." }
    Write-Ok 'WSL instalado.'
    Write-Warn2 '>> REINICIE O COMPUTADOR para concluir a instalacao do WSL. <<'
    Write-Host '    Depois de reiniciar, execute o install.ps1 novamente - a instalacao continua daqui.'
    $ans = Read-Host 'Reiniciar agora? [s/N]'
    if ($ans -match '^[sS]') {
      Write-Info 'Reiniciando em 10 segundos (Ctrl+C para cancelar)...'
      shutdown.exe /r /t 10
    }
    exit 0
  }
  Set-WslMemoryCap
  Write-Info 'Atualizando o WSL (wsl --update)...'
  $prevEAP = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  wsl.exe --update 2>&1 | Out-Null
  wsl.exe --shutdown 2>&1 | Out-Null
  $ErrorActionPreference = $prevEAP
  Start-Sleep -Seconds 3
  $distros = Get-WslDistros
  if ($distros -notcontains 'Debian-slim') {
    Write-Info 'Importando Debian-slim no WSL...'
    # O repo do Debian parou de publicar rootfs.tar.xz; o rootfs agora esta no
    # blob OCI (tar.gz, ~28 MB), que o `wsl --import` aceita normalmente.
    $RootFs = "$env:TEMP\debian-slim.tar.gz"
    $RootFsUrl = 'https://raw.githubusercontent.com/debuerreotype/docker-debian-artifacts/dist-amd64/trixie/slim/oci/blobs/rootfs.tar.gz'
    try {
      Invoke-WebRequest -Uri $RootFsUrl -OutFile $RootFs -UseBasicParsing
    } catch {
      Die "Falha ao baixar o rootfs do Debian ($RootFsUrl): $($_.Exception.Message)"
    }
    New-Item -ItemType Directory -Force -Path 'C:\WSL\Debian-slim' | Out-Null
    $imported = $false
    for ($try = 1; $try -le 2 -and -not $imported; $try++) {
      wsl --import Debian-slim C:\WSL\Debian-slim $RootFs --version 2
      if ($LASTEXITCODE -eq 0) { $imported = $true }
      elseif ($try -lt 2) {
        Write-Warn2 'Falha ao importar - reiniciando o WSL e tentando de novo...'
        wsl --shutdown
        Start-Sleep -Seconds 8
      }
    }
    if (-not $imported) {
      Die "Falha ao importar Debian-slim no WSL (CreateVm). Feche apps pesados (libere RAM), confira o memory= em $env:USERPROFILE\.wslconfig, rode 'wsl --update' e reinicie o PC se a VirtualMachinePlatform foi habilitada ha pouco."
    }
    Remove-Item $RootFs -Force
    Write-Ok 'Debian-slim importado.'
  }
  # Configura systemd, usuario e sudo
  Write-Info 'Configurando systemd/usuario no Debian-slim...'
  wsl -d Debian-slim -u root -- bash -c "export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y systemd systemd-sysv sudo curl ca-certificates && (id alpha-ai >/dev/null 2>&1 || useradd -m -s /bin/bash -G sudo alpha-ai) && printf '[boot]\nsystemd=true\n[user]\ndefault=alpha-ai\n' > /etc/wsl.conf"
  if ($LASTEXITCODE -ne 0) { Die 'Falha ao configurar systemd no Debian-slim.' }
  wsl --shutdown
  Start-Sleep -Seconds 5
  # Instala o Docker Engine
  Write-Info 'Instalando Docker Engine no Debian-slim...'
  wsl -d Debian-slim -u root -- bash -c "curl -fsSL https://get.docker.com | sh && usermod -aG docker alpha-ai && systemctl enable --now docker"
  if ($LASTEXITCODE -ne 0) { Die 'Falha ao instalar o Docker Engine no Debian-slim.' }
  $script:UseWSLDocker = $true
  # O daemon (systemd) pode levar alguns segundos para responder apos a instalacao
  $alive = $false
  for ($i = 0; $i -lt 12; $i++) {
    if (Test-DockerAlive) { $alive = $true; break }
    Start-Sleep -Seconds 5
  }
  if (-not $alive) { Die 'Docker no WSL nao respondeu apos instalacao (aguardei 1 min).' }
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

# -- 1c. .wslconfig: teto de RAM do WSL2 ---------------------------------------
# Sem cap, o vmmem (Docker Desktop / WSL2) engole até 50-80% da RAM livre e
# sufoca o resto da máquina. Só grava se o arquivo NÃO existe — nunca
# sobrescreve tuning do usuário. Cap = metade da RAM, entre 4 e 8 GB.
$WslConfigPath = Join-Path $env:USERPROFILE '.wslconfig'
if (Test-Path $WslConfigPath) {
  Write-Ok ".wslconfig existente mantido ($WslConfigPath)."
  Set-WslKeepAlive $WslConfigPath
} else {
  try {
    $TotalGb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    $Cap = [math]::Max(4, [math]::Min(8, [math]::Floor($TotalGb / 2)))
    @"
[wsl2]
memory=${Cap}GB
swap=2GB
vmIdleTimeout=-1

[general]
instanceIdleTimeout=-1
"@ | Set-Content -Path $WslConfigPath -Encoding ASCII
    Write-Ok ".wslconfig criado: WSL2 limitado a ${Cap}GB de RAM (maquina tem ${TotalGb}GB)."
    Write-Warn2 'O teto vale apos `wsl --shutdown` (ou reiniciar o PC).'
  } catch {
    Write-Warn2 "Nao foi possivel criar .wslconfig ($($_.Exception.Message)) — Docker pode consumir RAM sem controle."
  }
}

# -- 2. Download do ultimo release ---------------------------------------------
# ALPHA_RELEASE_ZIP (MODO TESTE, antes do release oficial): quando a env var
# aponta para um zip local, o GitHub e pulado inteiro e esse zip e o que
# instala. O arquivo e COPIADO para %TEMP% (o Remove-Item apos o
# Expand-Archive apaga a copia, nunca o original). Uso:
#   $env:ALPHA_RELEASE_ZIP = 'C:\caminho\alpha-ai-windows.zip'; .\install.ps1
if ($env:ALPHA_RELEASE_ZIP) {
  if (-not (Test-Path $env:ALPHA_RELEASE_ZIP -PathType Leaf)) {
    Die "ALPHA_RELEASE_ZIP aponta para arquivo inexistente: $($env:ALPHA_RELEASE_ZIP)"
  }
  Write-Info "MODO TESTE (ALPHA_RELEASE_ZIP): zip local '$($env:ALPHA_RELEASE_ZIP)' - pulando o GitHub."
  $Tag = 'local-test'
  $TmpZip = Join-Path $env:TEMP "alpha-ai-$Tag.zip"
  Copy-Item -Force $env:ALPHA_RELEASE_ZIP $TmpZip
} else {
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
}

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

# -- 3a. Runtime Windows: separa BIND de CLIENT e tolera probe ----------------
# AVA_BIND_HOST=0.0.0.0 e necessario no bridge do Windows, mas 0.0.0.0 nao e
# destino valido para httpx. Esses patches sao idempotentes: viram no-op quando
# o release ja vier corrigido.
function Patch-WindowsRuntime {
  param([Parameter(Mandatory)][string]$Root)
  $utf8 = New-Object System.Text.UTF8Encoding($false)

  $llm = Join-Path $Root 'Modules\LLM.py'
  if (Test-Path $llm) {
    $t = ([IO.File]::ReadAllText($llm)) -replace "`r`n", "`n"
    if ($t -notmatch 'AVA_CLIENT_HOST') {
      $old = @'
HOST_BIND = os.getenv("AVA_BIND_HOST", "127.9.12.22")

MEMORY_API_URL = os.getenv("MEMORY_API_URL", f"http://{HOST_BIND}:3000")
TTS_URL    = f"http://{HOST_BIND}:3004"
'@ -replace "`r`n", "`n"
      $new = @'
HOST_BIND = os.getenv("AVA_BIND_HOST", "127.9.12.22")
CLIENT_HOST = os.getenv("AVA_CLIENT_HOST") or ("127.0.0.1" if HOST_BIND in ("0.0.0.0", "::") else HOST_BIND)
MEMORY_API_URL = os.getenv("MEMORY_API_URL", f"http://{CLIENT_HOST}:3000")
TTS_URL    = os.getenv("TTS_URL", f"http://{CLIENT_HOST}:3004")
'@ -replace "`r`n", "`n"
      if ($t.Contains($old)) {
        $t = $t.Replace($old, $new)
        [IO.File]::WriteAllText($llm, $t, $utf8)
        Write-Ok 'LLM.py: client host separado do bind (0.0.0.0 -> 127.0.0.1).'
      } else {
        Write-Warn2 'LLM.py: bloco HOST_BIND nao encontrado - patch manual necessario.'
      }
    }
  }

  $graph = Join-Path $Root 'graph_activation.py'
  if (Test-Path $graph) {
    $t = ([IO.File]::ReadAllText($graph)) -replace "`r`n", "`n"
    $oldGraph = '        memory_api_url = f"http://{internal_bind_host()}:3000"'
    if ($t.Contains($oldGraph)) {
      $newGraph = @(
        '        _bind = internal_bind_host()',
        '        _client = "127.0.0.1" if _bind in ("0.0.0.0", "::") else _bind',
        '        memory_api_url = f"http://{_client}:3000"'
      ) -join "`n"
      $t = $t.Replace($oldGraph, $newGraph)
      [IO.File]::WriteAllText($graph, $t, $utf8)
      Write-Ok 'graph_activation.py: memory client host 127.0.0.1.'
    }
  }

  $orch = Join-Path $Root 'orchestrator.py'
  if (Test-Path $orch) {
    $t = ([IO.File]::ReadAllText($orch)) -replace "`r`n", "`n"
    $before = $t
    $t = [regex]::Replace($t, '(?m)^(\s*)except httpx\.ConnectError:(\s*\n\s*log\.warning\(f"[^\n]*OFFLINE"\))', '$1except httpx.HTTPError:$2', 1)
    if ($t -ne $before) {
      [IO.File]::WriteAllText($orch, $t, $utf8)
      Write-Ok 'orchestrator.py: probe inicial tolera HTTPError (memory subindo).'
    }
  }
}

# -- 3a. Compose Windows: bind 0.0.0.0 nos servicos internos -------------------
# No bridge do Windows/WSL2 o docker-proxy encaminha 127.9.12.22:9000 para o
# IP do container. Os servicos internos fazem bind em 127.9.12.22 (loopback
# DENTRO do container), onde o proxy nao chega -> "Empty reply"/SSE offline.
# AVA_BIND_HOST=0.0.0.0 faz o bind em todas as interfaces do container; o
# healthcheck em 127.9.12.22 continua valendo. Idempotente: nao duplica.
$WinCompose = Join-Path $InstallDir 'docker-compose.windows.yml'
if (Test-Path $WinCompose) {
  try {
    $composeTxt = Get-Content -Raw -Path $WinCompose
    if ($composeTxt -notmatch '(?m)^\s*AVA_BIND_HOST\s*:') {
      $composeTxt = [regex]::Replace($composeTxt, '(?m)^(\s*environment:\s*)$', "`$1`r`n      AVA_BIND_HOST: `"0.0.0.0`"", 1)
      Copy-Item -Force $WinCompose "$WinCompose.bak"
      [System.IO.File]::WriteAllText($WinCompose, $composeTxt, (New-Object System.Text.UTF8Encoding($false)))
      Write-Ok 'docker-compose.windows.yml: AVA_BIND_HOST=0.0.0.0 aplicado.'
    }
  } catch {
    Write-Warn2 "Nao foi possivel ajustar o compose Windows: $($_.Exception.Message)"
  }
}
Patch-WindowsRuntime -Root $InstallDir

# -- 3b. Agent Skills (SKILL.md - hyperframes) ---------------------------------
# .agents\ e skills-lock.json sao gitignored: um install novo NAO traz as
# skills. O CLI `skills` roda via npx (sem install global - o --yes do NPX
# baixa o pacote no cache). Flags testadas contra o CLI:
#   -a universal -> grava SO em .agents\skills (store canonico; sem linkar em
#                   .claude/.cursor/etc do usuario)
#   -y           -> pula os prompts (escopo = projeto; -y = non-interactive)
# Idempotente (re-run re-sincroniza pelo lock). Falha (rede, p.ex.) e aviso,
# nao morte - o sistema sobe sem skills.
# O `skills add` clona o repo via `git` (sem git: "spawn git ENOENT"). Instala o
# Git via winget se faltar; falha = aviso (as skills sao opcionais).
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    Write-Info 'Git ausente no host - instalando via winget (Git.Git)...'
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    winget install --id Git.Git --silent --accept-package-agreements --accept-source-agreements --disable-interactivity *> $null
    $ErrorActionPreference = $prevEAP
    foreach ($gd in @('C:\Program Files\Git\cmd', 'C:\Program Files (x86)\Git\cmd')) {
      if ((Test-Path $gd) -and ($env:Path -notlike "*$gd*")) { $env:Path = "$env:Path;$gd" }
    }
  }
  if (Get-Command git -ErrorAction SilentlyContinue) { Write-Ok 'Git instalado no host.' }
  else { Write-Warn2 'Nao consegui instalar o Git (instale em https://git-scm.com) - as Agent Skills podem falhar.' }
}

$Npx = Get-Command npx -ErrorAction SilentlyContinue
if ($Npx) {
  Write-Info 'Agent Skills: baixando heygen-com/hyperframes (npx skills add)...'
  & npx --yes skills add heygen-com/hyperframes -a universal -y
  if ($LASTEXITCODE -eq 0) {
    $NSkills = @(Get-ChildItem (Join-Path $InstallDir '.agents\skills') -Directory -ErrorAction SilentlyContinue).Count
    Write-Ok "Agent Skills instaladas em .agents\skills ($NSkills skills)."
  } else {
    Write-Warn2 'npx skills add falhou (sem rede?) - o sistema sobe sem as skills.'
    Write-Warn2 "Para instalar depois, em $InstallDir`: npx skills add heygen-com/hyperframes"
  }
} else {
  Write-Warn2 'npx ausente no host - pulei as Agent Skills (rode: npx skills add heygen-com/hyperframes).'
}

# -- 3c. OfficeCLI (binario npm -g + skill SKILL.md) ---------------------------
# `officecli` = leitura/edicao de .docx/.xlsx/.pptx (o binario nativo vem no
# postinstall do pacote). npm -g usa o prefix do usuario (%APPDATA%\npm, sem
# admin). A SKILL.md do site oficial vai direto para o store canonico
# .agents\skills\officecli\ (gitignored; o mount ro entrega ao container e o
# scanner do orchestrator descobre na proxima pergunta, sem restart).
# Download em arquivo temporario + Move-Item — SKILL.md truncado por rede nao
# entra no indice. Obs.: no PS 5.1 `curl` e alias de Invoke-WebRequest, entao
# o cmdlet e chamado explicitamente. Falha e aviso, nao morte.
$NpmCmd = Get-Command npm -ErrorAction SilentlyContinue
$OfficeFail = $false
if ($NpmCmd) {
  Write-Info 'OfficeCLI: npm install -g @officecli/officecli...'
  & npm install -g '@officecli/officecli'
  if ($LASTEXITCODE -ne 0) { $OfficeFail = $true }
} else {
  Write-Warn2 'npm ausente no host - pulei o OfficeCLI (rode: npm install -g @officecli/officecli).'
}
if ($OfficeFail) {
  Write-Warn2 'npm install -g @officecli/officecli falhou - o sistema sobe sem o OfficeCLI.'
  Write-Warn2 'Para instalar depois: npm install -g @officecli/officecli'
} elseif ($NpmCmd) {
  $Ocli = Get-Command officecli -ErrorAction SilentlyContinue
  $OcliLabel = 'via npm -g'
  if ($Ocli) { $OcliLabel = $Ocli.Source }
  Write-Ok "OfficeCLI instalado ($OcliLabel)."
}

$OfficeSkillDir = Join-Path $InstallDir '.agents\skills\officecli'
New-Item -ItemType Directory -Force -Path $OfficeSkillDir | Out-Null
try {
  Invoke-WebRequest -UseBasicParsing -Uri 'https://officecli.ai/SKILL.md' -OutFile (Join-Path $OfficeSkillDir 'SKILL.md.tmp') -ErrorAction Stop
  Move-Item -Force (Join-Path $OfficeSkillDir 'SKILL.md.tmp') (Join-Path $OfficeSkillDir 'SKILL.md')
  Write-Ok 'Skill officecli instalada (.agents\skills\officecli\SKILL.md).'
} catch {
  Remove-Item (Join-Path $OfficeSkillDir 'SKILL.md.tmp') -Force -ErrorAction SilentlyContinue
  Write-Warn2 "Download do SKILL.md (officecli.ai) falhou ($($_.Exception.Message)) - skill ausente (o CLI, se instalado, funciona)."
}

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
  # Aceita a key nova (LLM_API_KEY) ou a de instalacoes antigas (OPENROUTER_API_KEY)
  if (-not (Test-Path $EnvFile)) { return $false }
  $line = Select-String -Path $EnvFile -Pattern '^(LLM_API_KEY|OPENROUTER_API_KEY)=.+' -Quiet
  return [bool]$line
}

# LLM_PROVIDER_LOCAL=1 aceita qualquer url como "local" (ex.: IP na LAN)
function Test-LlmLocalFlag {
  if ($env:LLM_PROVIDER_LOCAL -eq '1') { return $true }
  if ((Get-EnvVar 'LLM_PROVIDER_LOCAL') -eq '1') { return $true }
  return $false
}

# Local para uma url ja digitada (avaliada antes de gravar no .env)
function Test-LlmUrlLocal {
  param([string]$Base)
  if (Test-LlmLocalFlag) { return $true }
  if (-not $Base) { return $false }
  if ($Base -match '^https?://(localhost|127\.0\.0\.1|127\.9\.12\.22|\[::1\])([:/]|$)') { return $true }
  if ($Base -match '^https?://[^/]+\.local([:/]|$)') { return $true }
  return $false
}

# Local segundo o .env (health check). Sem LLM_BASE_URL o LLM.py usa OpenRouter.
function Test-LlmIsLocal {
  if (Test-LlmLocalFlag) { return $true }
  return (Test-LlmUrlLocal (Get-EnvVar 'LLM_BASE_URL'))
}

# Pede a API key do provider ($Base) e devolve em $Script:PromptKey.
# Nada e gravado aqui - o caller escreve no .env depois de coletar tudo.
function Prompt-Key {
  param([string]$Base)
  $isOr = ($Base -like '*openrouter.ai*')
  $Script:PromptKey = ''
  while ($true) {
    if ($isOr) {
      $secure = Read-Host -Prompt 'Cole sua OPENROUTER_API_KEY (https://openrouter.ai/keys)' -AsSecureString
    } else {
      $secure = Read-Host -Prompt "Cole a API key do provider ($Base) - Enter = sem key" -AsSecureString
    }
    $plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
    $Key = $plain.Trim()
    if ($Key) {
      if ($isOr -and ($Key -notlike 'sk-or-*')) { Write-Warn2 "Chaves do OpenRouter normalmente comecam com 'sk-or-'. Prosseguindo mesmo assim." }
      $Script:PromptKey = $Key
      return
    }
    # Enter vazio: OpenRouter exige key; provider generico aceita sem key
    # (o caller oferece marcar como local / segue e o health check avisa)
    if ($isOr) { Write-Warn2 'Chave vazia - tente de novo.'; continue }
    return
  }
}

function Prompt-LlmConfig {
  # (1) endereco base da API
  $base = ''
  while ($true) {
    $base = Read-Host 'Endereco base da API de inferencia (LLM_BASE_URL) [Enter = OpenRouter]'
    $base = ($base.Trim() -replace '/+$', '')
    if (-not $base) { $base = 'https://openrouter.ai/api/v1' }
    if ($base -match '^https?://') { break }
    Write-Warn2 'O endereco deve comecar com http:// ou https:// - tente de novo.'
  }

  # (2) API key - so quando o provider NAO e local
  $key = ''
  $localFlag = $false
  if (Test-LlmUrlLocal $base) {
    Write-Ok "Provider local detectado ($base) - API key nao e necessaria."
  } else {
    while ($true) {
      Prompt-Key $base
      $key = $Script:PromptKey
      if ($key) { break }
      Write-Host 'Sem API key - em provider remoto a inferencia vai falhar no runtime.'
      $ans = Read-Host 'Marcar como LOCAL (LLM_PROVIDER_LOCAL=1 - ex.: servidor na LAN) e seguir sem key? [S/n]'
      if ($ans -notmatch '^[nN]') { $localFlag = $true; break }
    }
  }

  # (3) coletado -> "ai sim" escritos no .env
  Set-EnvVar 'LLM_BASE_URL' $base
  if ($key) { Set-EnvVar 'LLM_API_KEY' $key }
  if ($localFlag) { Set-EnvVar 'LLM_PROVIDER_LOCAL' '1' }
  Write-Ok "LLM_BASE_URL=$base gravada em $EnvFile"
  if ($key) { Write-Ok "LLM_API_KEY gravada em $EnvFile" }
  if ($localFlag) { Write-Ok 'LLM_PROVIDER_LOCAL=1 - provider marcado como local, sem key.' }
}

# Ja existe configuracao de provider (base e/ou key) -> manter ou reconfigurar.
# Instalacao antiga so com OPENROUTER_API_KEY e sem LLM_BASE_URL mantem o
# comportamento original (default = OpenRouter + aquela key).
if ((Get-EnvVar 'LLM_BASE_URL') -or (Test-ExistingKey)) {
  $ans = Read-Host "Ja existe configuracao de provider em Modules\.env (LLM_BASE_URL=$(Get-EnvVar 'LLM_BASE_URL')). Manter? [S/n]"
  if ($ans -match '^[nN]') { Prompt-LlmConfig }
  else { Write-Ok 'Configuracao de provider mantida.' }
} else {
  Prompt-LlmConfig
}

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

# chrome-extension:// NAO e um protocolo registrado no Windows: Start-Process
# com essa URL da "computador nao tem aplicativo para abrir". So o proprio
# Chrome entende - entao chama o chrome.exe explicitamente com a URL.
function Open-InChrome {
  param([Parameter(Mandatory)][string]$Url)
  $candidates = @()
  foreach ($hive in @('HKCU', 'HKLM')) {
    try {
      $v = (Get-ItemProperty -Path "${hive}:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe" -ErrorAction Stop).'(default)'
      if ($v) { $candidates += $v }
    } catch {}
  }
  $candidates += @(
    (Join-Path $env:ProgramFiles 'Google\Chrome\Application\chrome.exe'),
    (Join-Path ${env:ProgramFiles(x86)} 'Google\Chrome\Application\chrome.exe'),
    (Join-Path $env:LOCALAPPDATA 'Google\Chrome\Application\chrome.exe')
  )
  foreach ($c in $candidates) {
    if ($c -and (Test-Path $c)) {
      try { Start-Process -FilePath $c -ArgumentList $Url; return $true } catch {}
    }
  }
  return $false
}

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
  # NUL e SOH precisam ser bytes explicitos: em PowerShell `1 NAO e escape
  # (so `0 e). Com "`0`1" o needle virava "...\x00" + "1auth-token" e nunca
  # casava com a chave real "...\x00\x01auth-token".
  $keyNeedle = [System.Text.Encoding]::UTF8.GetBytes("chrome-extension://$extId" + [char]0 + [char]1 + "auth-token")
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
    # FileShare.ReadWrite: o Chrome em execucao mantem o .log aberto.
    # O -bor precisa ser atribuido ANTES do New-Object: dentro da lista de
    # argumentos o PowerShell trata a expressao como array e lanca
    # "op_BitwiseOr" -> o catch ignorava TODOS os arquivos.
    try {
      $share = [System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete
      $fs = New-Object System.IO.FileStream($file.FullName, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, $share)
      try { $ms = New-Object System.IO.MemoryStream; $fs.CopyTo($ms); $raw = $ms.ToArray() } finally { $fs.Dispose() }
    } catch { continue }
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
# O compose exige esta variavel (falha de boot se ausente). O alpha-mcp e o
# crawl4ai precisam do MESMO valor para o pipeline de busca funcionar.
if (-not (Get-EnvVar 'CRAWL4AI_API_TOKEN')) {
  Set-EnvVar 'CRAWL4AI_API_TOKEN' (New-RandomToken)
  Write-Ok "CRAWL4AI_API_TOKEN gerado em $EnvFile"
} else {
  Write-Ok "CRAWL4AI_API_TOKEN ja existe em $EnvFile"
}

# -- 5b''. Perfil de recursos (RAM) --------------------------------------------
# Escreve as flags de consumo no Modules\.env. Economico (default): TTS
# (~470MB) e voz do host (~600MB) desligados —
# tudo sobe on-demand conforme o uso. Completo: tudo ligado desde o boot.
# AVA_DOCKER_IDLE_MINUTES=15 vale para os dois: o stack dorme cedo e a
# pergunta seguinte o reergue sozinho.
$ans = Read-Host 'Perfil de recursos? [1]=Economico (~500MB, recomendado p/ PC com outras apps) [2]=Completo (voz+TTS, ~1.3GB) [1/2, default 1]'
if ($ans -match '^[2]') {
  Set-EnvVar 'AVA_TTS_ENABLED' '1'
  Set-EnvVar 'AVA_VOICE' '1'
  Write-Ok 'Perfil completo: voz + TTS ligados.'
} else {
  Set-EnvVar 'AVA_TTS_ENABLED' '0'
  Set-EnvVar 'AVA_VOICE' '0'
  Write-Ok 'Perfil economico: TTS e voz desligados (ligue flags em Modules\.env quando quiser).'
}
Set-EnvVar 'AVA_DOCKER_IDLE_MINUTES' '15'
Write-Ok 'Docker dorme apos 15 min de inatividade (AVA_DOCKER_IDLE_MINUTES=15).'

# -- 5b. Registro do Playwright MCP (dependencia direta do sistema) -----------
# Cadastra o @playwright/mcp no mcp_manager (:9001) logo apos o `up` do stack.
# O POST /servers e upsert idempotente e o /sync tem no-op por doc_hash -
# seguro re-executar. Tolerante a falha: se o manager ainda estiver bootando,
# retenta; se falhar de vez, so avisa (nao aborta a instalacao).
$McpManagerUrl = 'http://127.9.12.22:9001'

# Instrucao de sync manual para PowerShell. O `curl` no Windows PowerShell 5.1
# e alias de Invoke-WebRequest e nao aceita -H; o comando abaixo usa o cmdlet
# nativo e le o token do proprio Modules\.env gerado pelo instalador.
function Show-ManualSyncHint {
  param(
    [Parameter(Mandatory)][string]$Name,
    [int]$TimeoutSec = 300
  )
  Write-Warn2 'Para sincronizar manualmente no PowerShell:'
  Write-Warn2 "  `$envFile = `"$EnvFile`""
  Write-Warn2 "  `$token = (Get-Content `$envFile | Where-Object { `$_ -match '^AVA_INTERNAL_TOKEN=' } | Select-Object -Last 1) -replace '^AVA_INTERNAL_TOKEN=',''"
  Write-Warn2 "  Invoke-RestMethod -Uri '$McpManagerUrl/servers/$Name/sync' -Method Post -Headers @{ Authorization = `"Bearer `$token`" } -TimeoutSec $TimeoutSec"
}

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
  # Corpo em bytes UTF-8: no Windows PowerShell 5.1 o Invoke-RestMethod com
  # string usa ISO-8859-1 e os acentos do skill_summary viram bytes inválidos
  # em UTF-8 (o servidor responde 422 e o Playwright nunca é cadastrado).
  $payloadBytes = [System.Text.Encoding]::UTF8.GetBytes($payload)

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
    if (-not (Open-InChrome $PlaywrightExtensionStatusUrl)) {
      Write-Warn2 'Nao consegui abrir o Chrome automaticamente. Abra o Chrome, cole este endereco na barra e pressione Enter:'
      Write-Warn2 "  $PlaywrightExtensionStatusUrl"
      Read-Host 'Depois de abrir a pagina, pressione ENTER para continuar' | Out-Null
    }
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
        -ContentType 'application/json; charset=utf-8' -Body $payloadBytes -TimeoutSec 10 -Headers $McpHeaders | Out-Null
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
    Show-ManualSyncHint -Name 'playwright' -TimeoutSec 300
    return $false
  }
  Write-Ok 'Playwright MCP registrado e sincronizado.'
  return $true
}

# -- 5c. Registro dos demais MCPs (alpha-mcp e Desktop Commander) -------------
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
    Show-ManualSyncHint -Name $Name -TimeoutSec $SyncTimeoutSec
    return $false
  }
  Write-Ok "$Label MCP registrado e sincronizado."
  return $true
}

function Register-AlphaMcp {
  # URL do MCP do alpha-mcp (server.py, streamable-http em :4010/mcp). O
  # mcp_manager roda NO HOST (embutido no tray), nao dentro do container:
  # host.docker.internal so resolve em Docker Desktop/rota A. Na rota B ele
  # falha sempre. 127.9.12.22:4010 e a porta publicada no host (no Windows via
  # portproxy) e funciona nas duas rotas.
  $wsUrl = if ($env:ALPHA_MCP_URL) { $env:ALPHA_MCP_URL } else { 'http://127.9.12.22:4010/mcp' }
  return (Register-McpServer -Label 'Alpha' -Name 'alpha-mcp' -SyncTimeoutSec 120 -Spec @{
    name          = 'alpha-mcp'
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
  # Captura do stderr do tray (panics, saida pre-init de logging) — o
  # tracing do tray ja vai direto para logs\ava-tray.log por conta dele.
  New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir 'logs') | Out-Null
  Start-Process -FilePath $exe -WorkingDirectory $InstallDir -RedirectStandardError (Join-Path $InstallDir 'logs\ava-tray.stderr.log')
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

# O mcp_manager (:9001) roda embutido no Manager, no host.
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
# A porta 9001 fica FORA: o mcp_manager roda no HOST e publicar a porta colidiria.
# Rota A (Docker Desktop) nao precisa de nada disso: ele publica no loopback do
# Windows e host.docker.internal ja aponta para o host.
if ($script:UseWSLDocker) {
  Write-Info 'Rota B: instalando relays socat no Debian-slim...'

  # (1a) socat dentro da VM
  wsl -d Debian-slim -u root -- bash -c "export DEBIAN_FRONTEND=noninteractive; { command -v socat >/dev/null 2>&1 && command -v ip >/dev/null 2>&1; } || { apt-get update -qq && apt-get install -y -qq socat iproute2; }"
  if ($LASTEXITCODE -ne 0) { Die 'Falha ao instalar o socat no Debian-slim.' }

  # Script de relay: alvos do stack sao fixos (127.9.12.22 na VM); os do HOST
  # dependem do gateway do WSL (vEthernet do Windows) — resolvido a cada boot.
  # IMPORTANTE: os relays do stack escutam no IP DA VM (eth0), NAO em 0.0.0.0.
  # Docker publica em 127.9.12.22:PORTA; se o socat escutar em 0.0.0.0:PORTA e
  # o container ainda nao subiu, o connect para 127.9.12.22:PORTA cai no
  # proprio socat (loopback) -> fork() por conexao em loop -> pids esgotados,
  # WSL/dockerd derrubados ("fork(): Resource temporarily unavailable").
  $RelaySh = @'
#!/bin/bash
# Alpha AI - relays socat da rota WSL (gerado pelo install.ps1)
# (a) stack: VMIP:PORTA -> 127.9.12.22:PORTA (publicacao do docker na VM)
# (b) host:  0.0.0.0:PORTA -> GW:PORTA        (tray/CDP/transcribe no Windows)
GW="$(ip route show default | awk '{print $3}' | head -n1)"
VMIP="$(hostname -I | awk '{print $1}')"
APP_PORTS="3000 4005 4003 3003 3001 2002 3002 3004 4002 9000 3005 2001 8081 6379 11235 4010"
HOST_PORTS="9001 9222 9100"
for p in $APP_PORTS; do
  socat "TCP-LISTEN:$p,bind=$VMIP,fork,reuseaddr" "TCP:127.9.12.22:$p" &
done
for p in $HOST_PORTS; do
  socat "TCP-LISTEN:$p,bind=0.0.0.0,fork,reuseaddr" "TCP:$GW:$p" &
done
wait -n
'@
  $RelaySh = $RelaySh -replace "`r`n", "`n"
  $RelayTmp = Join-Path $env:TEMP 'alpha-ai-relay.sh'
  [System.IO.File]::WriteAllText($RelayTmp, $RelaySh, (New-Object System.Text.UTF8Encoding($false)))
  $RelayWsl = (ConvertTo-WslPath $RelayTmp)
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
    $SvcWsl = (ConvertTo-WslPath $SvcTmp)
    wsl -d Debian-slim -u root -- bash -c "cp '$RelayWsl' /usr/local/bin/alpha-ai-relay.sh && chmod 755 /usr/local/bin/alpha-ai-relay.sh && cp '$SvcWsl' /etc/systemd/system/alpha-ai-relay.service && systemctl daemon-reload && systemctl enable alpha-ai-relay.service && systemctl restart alpha-ai-relay.service"
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
    $wslDir = (ConvertTo-WslPath $InstallDir)
    # Skills globais do WINDOWS (%USERPROFILE%\.agents\skills) no formato da
    # VM — mesmo esquema do docker-start.bat. O docker-compose.windows.yml
    # interpola AVA_WIN_GLOBAL_SKILLS no mount /root/.agents-win/skills
    # (fallback /tmp quando vazia — nunca derruba o stack).
    $winSkills = (ConvertTo-WslPath "$env:USERPROFILE\.agents\skills")
    if ($winSkills) { $env:AVA_WIN_GLOBAL_SKILLS = $winSkills }
    else { Remove-Item Env:\AVA_WIN_GLOBAL_SKILLS -ErrorAction SilentlyContinue }
    # WSLENV: o wsl.exe so exporta env do Windows p/ o Linux se a var estiver
    # listada aqui (flag /w = Win32 -> WSL). Sem isso o compose dentro da VM
    # nao ve a var e o override cai no fallback /tmp. Guarda p/ nao duplicar
    # a entrada quando a funcao roda varias vezes no mesmo install.
    if ($env:WSLENV -notlike '*AVA_WIN_GLOBAL_SKILLS*') {
      if ($env:WSLENV) { $env:WSLENV = "AVA_WIN_GLOBAL_SKILLS/w:$env:WSLENV" }
      else { $env:WSLENV = 'AVA_WIN_GLOBAL_SKILLS/w' }
    }
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

# -- 5f. Rebind dos containers (reinstall sobre stack rodando) -----------------
# O Expand-Archive recria subarvores do diretorio de instalacao; containers de
# um stack anterior ficam com bind nos handles VELHOS do filesystem — veem uma
# arvore fantasma (codigo novo invisivel, db que "sumiu": manifest LanceDB em
# cache apontando para fragmentos inexistentes -> /status 500 em loop, container
# unhealthy). `up -d` simples nao recria container saudavel; so o
# --force-recreate refaz os binds. Primeiro install: o ps nao lista containers
# e o bloco e pulado. Falha aqui e aviso — a etapa 6 segue o fluxo normal.
$rawPs = ((Invoke-Compose ps --quiet 2>$null | Out-String) -replace "`0", '')
# O wrapper devolve o exit code no fim do pipeline (0) e o docker-start.bat ecoa
# a rota usada ("Rota A/B...") — o filtro mantem SO linhas de container ID (hex
# de 12-64 chars), imune a esse ruido.
$ids = @($rawPs -split '\r?\n' | Where-Object { $_ -match '^[0-9a-f]{12,64}\s*$' })
if ($LASTEXITCODE -eq 0 -and $ids.Count -gt 0) {
  Write-Info 'Stack ja existia para este diretorio - recriando containers (rebind dos mounts)...'
  Invoke-Compose up -d --force-recreate
  if ($LASTEXITCODE -eq 0) { Write-Ok 'Containers recriados sobre a arvore nova.' }
  else { Write-Warn2 'docker compose up -d --force-recreate falhou (o fluxo da etapa 6 segue normal).' }
}

# -- 6. Build + first up -------------------------------------------------------
if (-not $SkipBuild) {
  Write-Info 'Build inicial das imagens Docker (pode levar 5-15 min na primeira vez)...'
  Invoke-Compose build
  if ($LASTEXITCODE -ne 0) { Die 'docker compose build falhou - rode o build novamente para ver o erro.' }

  # Baixa as imagens de terceiros ANTES do primeiro up via Manager.
  # O /docker/start do Manager sobe o stack; se as imagens ainda nao estiverem
  # no cache, o compose baixa tudo dentro do request e o Manager pode estourar
  # o timeout. Lista explicita: ava e alpha-mcp sao build local e NAO entram no
  # pull (evita tentar sobrescrever com tag do registry).
  Write-Info 'Baixando imagens Docker de terceiros (searxng, valkey, crawl4ai, docker-socket-proxy)...'
  Invoke-Compose pull searxng valkey crawl4ai docker-socket-proxy | Out-Null
  if ($LASTEXITCODE -ne 0) {
    Write-Warn2 'Falha ao baixar alguma imagem - o up tentara baixar novamente.'
  }

  # Rota A (Docker Desktop): Manager sobe o stack (POST /docker/start).
  # Rota B (Docker no WSL): o tray roda no Windows SEM CLI docker. O health
  # check dele (`docker compose ps`) falha, ele conclui que os containers
  # pararam e o auto-stop derruba o stack. Nessa rota o instalador sobe o
  # compose direto e o Manager sobe depois apenas para API/MCP/WebUI.
  $viaManager = $false
  if (-not $script:UseWSLDocker) {
    $viaManager = (Start-Manager) -and (Start-DockerViaManager)
  }
  if (-not $viaManager) {
    Write-Info 'Subindo o stack diretamente via compose...'
    Invoke-Compose up -d
    if ($LASTEXITCODE -ne 0) { Die 'docker compose up falhou.' }
  }
  # Rota B: Manager de pe para a API :9001 + mcp_manager, sem /docker/start.
  if (-not (Test-ManagerUp)) { Start-Manager | Out-Null }

  # mcp_manager (host) ativo antes do cadastro dos MCPs
  Ensure-McpManager | Out-Null

  # Registro dos MCPs - logo apos o up, quando o stack ja esta carregado.
  Register-PlaywrightMcp | Out-Null
  Register-AlphaMcp | Out-Null
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
    # Detecta API key ausente no /health do LLM.py (llm_key: "missing" =
    # provider remoto sem key; "not-required" = provider local, ok - nao reclama)
    if (-not (Test-LlmIsLocal)) {
      $h = Invoke-WebRequest -Uri 'http://127.9.12.22:4003/health' -UseBasicParsing -TimeoutSec 5
      if ($h.Content -match '"llm_key":\s*"missing"') {
        Write-Warn2 'O LLM.py reportou llm_key: missing (provider remoto sem API key).'
        $ans = Read-Host 'Deseja re-informar a chave agora? [S/n]'
        if ($ans -notmatch '^[nN]') {
          $b = Get-EnvVar 'LLM_BASE_URL'
          if (-not $b) { $b = 'https://openrouter.ai/api/v1' }
          Prompt-Key $b
          if ($Script:PromptKey) { Set-EnvVar 'LLM_API_KEY' $Script:PromptKey }
          Write-Info 'Reiniciando container ava para reler a chave...'
          Invoke-Compose restart ava
        }
      }
    }
  } catch { }
} else {
  Write-Info '-SkipBuild: pulei o build/up do Docker.'
}

# -- 7. PATH do usuario --------------------------------------------------------
# `alpha-ai` abre o WebUI nativo (Manager\ava-web.exe), nao mais a TUI antiga.
$ManagerDir = Join-Path $InstallDir 'Manager'
$WebExe = Join-Path $ManagerDir 'ava-web.exe'
if (Test-Path $WebExe) {
  # O .cmd fica ao lado do ava-web.exe; %~dp0 resolve o caminho com espacos.
  $Cmd = Join-Path $ManagerDir 'alpha-ai.cmd'
  Set-Content -Path $Cmd -Encoding ASCII -Value '@echo off', 'start "" "%~dp0ava-web.exe" %*'

  $TuiDir = Join-Path $InstallDir 'tui'
  $UserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
  if (-not $UserPath) { $UserPath = '' }
  $Parts = @($UserPath -split ';' | Where-Object { $_ -and $_ -ne $TuiDir })
  if ($Parts -notcontains $ManagerDir) { $Parts += $ManagerDir }
  $NewPath = ($Parts -join ';')
  if ($NewPath -ne $UserPath) {
    [Environment]::SetEnvironmentVariable('Path', $NewPath, 'User')
    Write-Ok "Comando instalado: alpha-ai -> $WebExe. Reabra o terminal para valer."
  } else {
    Write-Ok "Comando alpha-ai ja no PATH ($ManagerDir)."
  }
} else {
  Write-Warn2 "ava-web.exe nao encontrado em $WebExe - comando alpha-ai nao instalado."
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
    # Mesma captura de stderr do Start-Manager (secao 5d)
    New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir 'logs') | Out-Null
    Start-Process -FilePath $TrayExe -WorkingDirectory $InstallDir -RedirectStandardError (Join-Path $InstallDir 'logs\ava-tray.stderr.log')
    Write-Ok 'ava-tray iniciado agora.'
  }
} else {
  Write-Warn2 "ava-tray.exe nao encontrado em $TrayExe - auto-start pulado."
}

# -- 9. Encerramento -----------------------------------------------------------
Write-Host ''
Write-Ok "Instalacao concluida ($InstallDir)"
Write-Info 'Comandos uteis:'
Write-Info '  alpha-ai                          # abre o WebUI (Manager\ava-web.exe)'
Write-Info '  Manager\ava-web.exe               # mesma janela, caminho direto'
Write-Info "  $script:LogsHint   # logs do backend"
Write-Info "  $InstallDir\logs    # logs do tray (ava-tray.log) e do mcp_manager (mcp_manager.log)"

# -- 9b. Patch OPCIONAL da extensao Playwright (Windows e manual) --------------
# O Chrome no Windows nao aceita instalar um CRX local automaticamente nem
# --load-extension (removido no Chrome 137). O patch fica documentado aqui:
# a extensao patcheada (aba ativa real, sem tab group) e carregada como
# "unpacked" uma unica vez, pelo usuario.
Write-Host ''
Write-Info '--- Patch opcional da extensao Playwright (aba ativa / sem tab group) ---'
Write-Info 'O @playwright/mcp ja funciona com a extensao da Chrome Web Store.'
Write-Info 'Para controlar a ULTIMA ABA ATIVA real (e nao abrir uma aba nova), faca:'
Write-Info ''
Write-Info '  1. Gere a copia patcheada da extensao (requer Python 3 no host):'
Write-Info ('       python "' + $InstallDir + '\scripts\patch_playwright_extension.py"')
Write-Info '     Saida: $env:LOCALAPPDATA\alpha-ai\playwright-extension'
Write-Info ''
Write-Info '  2. Abra chrome://extensions, ative o Developer mode,'
Write-Info '     REMOVA a "Playwright Extension" da Chrome Web Store e clique em'
Write-Info '     "Load unpacked" na pasta acima.'
Write-Info ''
Write-Info '  3. Abra chrome-extension://mmlmfjhmonkocbjadbfplnigmagldckm/status.html'
Write-Info '     e copie o token para PLAYWRIGHT_MCP_EXTENSION_TOKEN em Modules\.env.'
Write-Info '     (Ou rode novamente: install.ps1 -SkipBuild -NoLaunch, que extrai o token.)'
Write-Info ''
Write-Info '  4. Reinicie o ava-tray e execute o sync do Playwright (PowerShell):'
Show-ManualSyncHint -Name 'playwright' -TimeoutSec 300
Write-Info ''
Write-Info 'O patch NAO e instalado automaticamente no Windows: o Chrome bloqueia'
Write-Info 'CRX local/registro externo fora da Chrome Web Store. No Linux o install.sh'
Write-Info 'faz esse fluxo sozinho via external extension + CRX proprio.'

if (-not $NoLaunch) {
  $LaunchExe = Join-Path $InstallDir 'Manager\ava-web.exe'
  if (Test-Path $LaunchExe) {
    Write-Info 'Abrindo o WebUI...'
    Start-Process -FilePath $LaunchExe -WorkingDirectory $InstallDir
  } else {
    Write-Warn2 "ava-web.exe nao encontrado em $LaunchExe - WebUI nao iniciado."
  }
}
