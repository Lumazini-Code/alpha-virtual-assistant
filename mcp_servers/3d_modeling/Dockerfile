# CadQuery MCP — streamable-http para o mcp_manager do AVA.
# Build multi-stage (pip/setuptools ficam fora da imagem final) e deps mínimas:
# CadQuery 2.4 + OCP 7.7 (sem VTK/CasADi/Numba/trame da 2.8, que somavam
# ~1,9 GB). O solver de Assembly é stubado — nenhuma tool MCP usa constraints.
# OCP é carregado sob demanda, na primeira tool que modela.
FROM python:3.12-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m pip install --no-cache-dir --target /deps --no-deps \
        cadquery==2.4.0 cadquery-ocp==7.7.2 \
 && python -m pip install --no-cache-dir --target /deps \
        "numpy==1.26.4" ezdxf multimethod typish nptyping pyparsing \
        typing_extensions nlopt "mcp>=1.10,<2"
COPY casadi_stub.py /deps/casadi.py

FROM python:3.12-slim
# OCP 7.7 precisa de libGL.so.1 e símbolos X11, mas não do driver Gallium
# (~180 MB) — este server só faz geometria/STEP/STL, sem renderização.
# Instalamos só os .deb necessários (1,8 MB) com --force-depends, sem mesa-dri.
RUN apt-get update -qq \
 && mkdir -p /tmp/ava-gl && cd /tmp/ava-gl \
 && apt-get download -qq libgl1 libglvnd0 libglx0 libglx-mesa0 libx11-6 \
        libexpat1 libxcb1 libxcb-dri3-0 libdrm2 libxext6 libxau6 \
        libxdmcp6 libbsd0 libmd0 \
 && (dpkg -i --force-depends ./*.deb >/dev/null 2>&1 || true) \
 && ldconfig \
 && test -e /usr/lib/x86_64-linux-gnu/libGL.so.1 \
 && cd / && rm -rf /tmp/ava-gl /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/deps:/app \
    CQ_MCP_HOST=0.0.0.0 \
    CQ_MCP_PORT=4011 \
    CQ_MCP_TRANSPORT=streamable-http \
    CQ_WORKDIR=/home/felipe

COPY --from=deps /deps /deps
WORKDIR /app
COPY cadquery_mcp_server.py /app/
COPY SKILL.md /app/SKILL.md

# O host monta $HOME neste mesmo caminho: os paths que o modelo vê no WebUI
# continuam válidos dentro do container, sem camada de tradução.
WORKDIR /home/felipe
EXPOSE 4011
CMD ["python", "-m", "cadquery_mcp_server"]
