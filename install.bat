@echo off
REM install.bat - Instalador Alpha AI (Windows)
REM Toda a logica esta em install.ps1 (PowerShell 5.1+, embutido no Windows 10+).
REM Uso: install.bat [-InstallDir DIR] [-KeepModelsZip] [-SkipBuild] [-NoLaunch]

setlocal
set "SCRIPT_DIR=%~dp0"

where powershell >nul 2>&1
if errorlevel 1 (
  echo [ERRO] PowerShell nao encontrado - instale o Windows Management Framework 5.1.
  exit /b 1
)

if not exist "%SCRIPT_DIR%install.ps1" (
  echo [ERRO] install.ps1 nao encontrado em %SCRIPT_DIR%
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install.ps1" %*
set "RC=%errorlevel%"

REM Se foi aberto com duplo-clique (cmd /c), pausa para o usuario ler o resultado.
echo %cmdcmdline% | find /i "%~0" >nul 2>&1
if not errorlevel 1 pause

endlocal & exit /b %RC%