FROM python:3.11-slim

WORKDIR /app

# uv — instalador Python rápido (downloads paralelos, cache mínimo).
# O binário vem da imagem oficial (não polui o venv com o próprio uv).
# Para reprodutibilidade total, fixe a versão (ex.: ghcr.io/astral-sh/uv:0.9.x).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY requirements.txt .

# apt consolidado em UMA camada: cada RUN cria uma camada no cache de build —
# separar "apt update" do install assava as listas de pacotes em camada própria.
# --no-install-recommends evita pacotes sugeridos; as listas são removidas no
# MESMO RUN (senão ficam baked na camada, ~100-200MB por build).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        pulseaudio-utils \
        alsa-utils \
        portaudio19-dev \
        libc++1 \
        libvulkan1 \
        mesa-vulkan-drivers \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Dependências Python via uv:
#   --system          → instala no Python da própria imagem (sem venv extra;
#                       start.sh chama python3 direto)
#   --no-cache        → não persiste cache de wheels em camada
#   --compile-bytecode → pré-compila .pyc (boot dos serviços mais rápido)
RUN uv pip install --system --no-cache --compile-bytecode -r requirements.txt

CMD ["/bin/bash", "/app/start.sh"]
