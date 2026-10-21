#!/bin/bash
# start.sh — Linux
# Uso: ./start.sh up [-d]

# O compose resolve a interpolação (${CRAWL4AI_API_TOKEN:?...}) a partir
# do shell e do `.env` da RAIZ do projeto — MAS os tokens vivem em
# Modules/.env (gerado pelo install.sh), que ele NÃO lê automaticamente.
# Passamos --env-file explicitamente. Sem o arquivo, o próprio erro ":?"
# do compose explica qual variável falta.
ENV_FILE="Modules/.env"
ENV_ARGS=()
if [ -f "$ENV_FILE" ]; then
  ENV_ARGS=(--env-file "$ENV_FILE")
fi

docker compose "${ENV_ARGS[@]}" \
  -f docker-compose.yml \
  -f docker-compose.linux.yml \
  "$@"
