#!/bin/bash
# start.sh — Linux
# Uso: ./start.sh --profile <cpu|cuda|vulkan> up [-d]
# Exemplo: ./start.sh --profile vulkan up -d

docker compose \
  -f docker-compose.yml \
  -f docker-compose.linux.yml \
  "$@"
