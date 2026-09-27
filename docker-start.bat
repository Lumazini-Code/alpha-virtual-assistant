@echo off
REM start.bat — Windows
REM Uso: start.bat up [-d]

docker compose ^
  -f docker-compose.yml ^
  -f docker-compose.windows.yml ^
  %*
