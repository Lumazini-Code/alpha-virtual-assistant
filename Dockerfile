FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt update

RUN apt install -y pulseaudio-utils alsa-utils 

RUN apt-get install -y ripgrep


RUN apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

COPY package.json package-lock.json* ./

RUN npm ci

CMD ["/bin/bash", "/app/start.sh"]
