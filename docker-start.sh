#!/bin/bash
# start.sh — Linux
# Uso: ./start.sh up [-d]

docker compose \
  -f docker-compose.yml \
  -f docker-compose.linux.yml \
  "$@"
