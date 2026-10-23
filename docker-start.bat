@echo off
REM start.bat — Windows
REM Uso: start.bat up [-d]
REM
REM O compose resolve a interpolação (${CRAWL4AI_API_TOKEN:?...}) a partir
REM do shell e do `.env` da RAIZ do projeto — MAS os tokens vivem em
REM Modules\.env (gerado pelo install.sh), que ele NÃO lê automaticamente.
REM Passamos --env-file explicitamente. Sem o arquivo, o próprio erro ":?"
REM do compose explica qual variável falta.

set "ENV_FILE=Modules\.env"
set "ENV_ARGS="
if exist "%ENV_FILE%" set "ENV_ARGS=--env-file %ENV_FILE%"

docker compose %ENV_ARGS% ^
  -f docker-compose.yml ^
  -f docker-compose.windows.yml ^
  %*
