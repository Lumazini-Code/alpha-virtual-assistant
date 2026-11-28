# Boardview MCP — streamable-http para o mcp_manager do AVA.
# Alpine + multi-stage: só runtime Python + mcp/pillow na imagem final
# (~100 MB em vez de ~265 MB no slim). mcp e pillow têm wheels musllinux.
FROM python:3.12-alpine AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m pip install --no-cache-dir --target /deps \
        "mcp>=1.10,<2" "pillow>=10.1"

FROM python:3.12-alpine
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/deps:/app \
    BOARDVIEW_MCP_HOST=0.0.0.0 \
    BOARDVIEW_MCP_PORT=4012

COPY --from=deps /deps /deps
WORKDIR /app
COPY boardview_mcp/ /app/boardview_mcp/
COPY SKILL.md /app/SKILL.md

# $HOME do host é montado neste mesmo caminho; paths relativos (".") e globs
# resolvem como no WebUI em vez de resolverem para /app.
WORKDIR /home/felipe
EXPOSE 4012
CMD ["python", "-m", "boardview_mcp.server", "--transport", "streamable-http"]
