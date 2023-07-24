@echo off
echo.
title Windows Repair By @Lumazini Play
color B
echo Bem Vindo(a)...
:m
mode 110,40
:m
echo.
echo Bem Vindo(a)...
echo.
echo.
echo.
echo        -------------------------------
echo       ! Digite (1) Procurar por erros !
echo       !                               !
echo       ! Digite (2) Cancelar           !
echo        -------------------------------
echo.
echo.
echo _____________________________________
set /p op=
if %op% equ 1 goto procurar
if %op% equ 2 goto Sair
goto m
:procurar
cls
echo  -----------------------------------------------
echo ! Aperte qualquer tecla para procurar por erros !
echo  -----------------------------------------------
echo.
pause
echo on
sfc /scannow
@echo off
echo.
echo     -----------------------------------------------------------------------------------
echo    ! Se apareceu uma mensagem dizendo: "A Protecao de recursos do Windows encontrou    !
echo    ! encontrou arquivos corrompidos e os reparou com exito" aperte (1) para recomecar  !
echo    ! a reparacao, se nao apareceu aperte (2) para reiniciar o computador para aplicar  !
echo    ! as configurações.                                                                 !
echo     -----------------------------------------------------------------------------------
echo.
echo.
echo.
echo _____________________________________
set /p op=
if %op% equ 1 goto sim
if %op% equ 2 goto nao
goto procurar
:sim
goto procurar

:nao
goto Sair
cls

:Sair
echo.
echo seu computador sera reiniciado para aplicar as correcoes
shutdown -f -r -t 10