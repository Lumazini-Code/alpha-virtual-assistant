FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt update

RUN apt install -y pulseaudio-utils alsa-utils portaudio19-dev libc++1 libvulkan1 mesa-vulkan-drivers

RUN apt-get install -y ripgrep

CMD ["/bin/bash", "/app/start.sh"]
