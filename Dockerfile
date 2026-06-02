FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x /app/start.sh

RUN apt update

RUN apt install -y pulseaudio-utils alsa-utils

CMD ["/bin/bash", "/app/start.sh"]
