@echo off
REM docker-start.bat — Windows (duas rotas)
REM
REM Rota A (Docker Desktop): roda no host, como sempre.
REM Rota B (Docker Engine no WSL Debian-slim): repassa o comando para
REM   wsl -d Debian-slim --cd <projeto> -- docker compose ...
REM
REM A deteccao replica o install.ps1: primeiro testa o docker do host;
REM se nao responder, testa o docker dentro do Debian-slim. Nao altera
REM a distro padrao do WSL — sempre -d Debian-slim explicito.
REM Uso: docker-start.bat build | up [-d] | logs -f ava | restart ava | ...
REM
REM O compose resolve a interpolacao (${CRAWL4AI_API_TOKEN:?...}) a partir
REM do --env-file — os tokens vivem em Modules\.env (gerado pelo
REM install.ps1 no Windows / install.sh no Linux). Sem o arquivo, o proprio
REM erro ":?" do compose explica qual variavel falta.

setlocal enabledelayedexpansion

set "ENV_ARGS="
if exist "Modules\.env" set "ENV_ARGS=--env-file Modules\.env"

REM ── Agent Skills GLOBAIS do Windows ────────────────────────────────────
REM `npx skills add -g` instala em %USERPROFILE%\.agents\skills. Na rota A
REM (compose no Windows) o caminho literal serve; na rota B (compose dentro
REM do WSL) ele e traduzido via wslpath abaixo — mesmo esquema do diretorio
REM do projeto. O docker-compose.windows.yml interpola essa var no mount
REM /root/.agents-win/skills (fallback /tmp quando vazia — nunca derruba).
set "AVA_WIN_GLOBAL_SKILLS=%USERPROFILE%\.agents\skills"
REM O wsl.exe SO exporta env do Windows p/ o Linux se a var estiver listada
REM no WSLENV (flag /w = Win32 -> WSL). Sem isso a rota B perde a var e o
REM override cai no fallback /tmp. Rota A nao atravessa o wsl.exe (compose
REM roda no host) — declarar aqui e inofensivo.
if defined WSLENV (set "WSLENV=AVA_WIN_GLOBAL_SKILLS/w:%WSLENV%") else set "WSLENV=AVA_WIN_GLOBAL_SKILLS/w"

REM ── Rota A: docker.exe responde no host? (Docker Desktop) ──────────
where docker >nul 2>&1
if not errorlevel 1 (
  docker info >nul 2>&1
  if not errorlevel 1 (
    echo Rota A: Docker Desktop.
    docker compose !ENV_ARGS! -f docker-compose.yml -f docker-compose.windows.yml %*
    endlocal & exit /b %ERRORLEVEL%
  )
)

REM ── Rota B: Docker Engine no WSL Debian-slim ───────────────────────
where wsl >nul 2>&1
if errorlevel 1 (
  echo [X] Docker nao responde no Windows e o WSL nao esta disponivel.
  echo     Rode install.ps1 para configurar ^(Docker Desktop ou rota WSL^).
  endlocal & exit /b 1
)

wsl -d Debian-slim -e docker info >nul 2>&1
if errorlevel 1 (
  echo [X] Docker nao responde no Windows nem no WSL Debian-slim.
  echo     Rode install.ps1 para configurar e/ou iniciar o Docker.
  endlocal & exit /b 1
)

echo Rota B: Docker Engine no WSL Debian-slim.

REM Converte o diretorio do projeto para o path da VM e executa o compose
REM dentro dela — o build/up acontece na distro, os relays socat/portproxy
REM do install.ps1 fazem a ponte host ^<-^> VM.
for /f "usebackq delims=" %%i in (`wsl -d Debian-slim -- wslpath -a "%CD%"`) do set "WSL_DIR=%%i"

REM Skills globais do Windows no formato da VM (wslpath). Comeca VAZIA: se o
REM wslpath falhar, a var fica vazia e o override usa o fallback /tmp — em
REM vez de passar um caminho Windows-style que o Docker da VM nao monta.
set "AVA_WIN_GLOBAL_SKILLS="
for /f "usebackq delims=" %%i in (`wsl -d Debian-slim -- wslpath -a "%USERPROFILE%\.agents\skills" 2^>nul`) do set "AVA_WIN_GLOBAL_SKILLS=%%i"

wsl -d Debian-slim --cd "!WSL_DIR!" -- docker compose !ENV_ARGS! -f docker-compose.yml -f docker-compose.windows.yml %*
endlocal & exit /b %ERRORLEVEL%
