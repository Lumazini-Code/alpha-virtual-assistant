#!/bin/bash
# Alpha AI (AVA) — wrapper do docker compose — build e subida do stack
# Copyright (C) 2026  Felipe lumazini
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
