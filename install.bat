@echo off
REM install.bat — Instalador Alpha AI (Windows)
REM Toda a logica esta em install.ps1 (PowerShell 5.1+, embutido no Windows 10+).
REM Uso: install.bat [-InstallDir DIR] [-KeepModelsZip] [-SkipBuild] [-NoLaunch]

setlocal
set "SCRIPT_DIR=%~dp0"

where powershell >nul 2>&1
if errorlevel 1 (
  echo [ERRO] PowerShell nao encontrado — instale o Windows Management Framework 5.1.
  exit /b 1
)

[IO.File]::WriteAllText($p, [IO.File]::ReadAllText($p, [Text.Encoding]::UTF8), (New-Object Text.UTF8Encoding $true))
powershell -NoProfile -ExecutionPolicy Bypass -File $p

REM Se foi aberto com duplo-clique (cmd /c), pausa para o usuario ler o resultado.
echo %cmdcmdline% | find /i "%~0" >nul 2>&1
if not errorlevel 1 pause

endlocal
