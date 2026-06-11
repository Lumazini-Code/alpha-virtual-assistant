@echo off
REM start.bat — Windows
REM Uso: start.bat --profile <cpu^|cuda^|vulkan> up [-d]
REM Exemplo: start.bat --profile vulkan up -d

docker compose ^
  -f docker-compose.yml ^
  -f docker-compose.windows.yml ^
  %*
