padrão de uso das portas de API rest localHost (0.0.0.0):

0-1024: não é permitido o uso (Linux bloqueia)

2000-2999: conexões Llama server

    2001 = LLM
    2002 = code generator
    2003 = CoT generator


3000-3999: Módulos Auxiliares (CoT generator, Search API, text Classification, memory, TTS, etc)

    3000 = CoT generator
    3001 = memory
    3002 = Search API
    3003 = text classification
    3004 = TTS


4000-4999: Módulos principais (VQA, LLM, code generator, etc)
    
    4000 = LLM
    4001 = Code generator
    4002 = VQA

9000-9999: conexão entre backend e frontend






Uso dos módulos Auxiliares:

------------------------------------------------------------------------------------

API de memória

import httpx

result = httpx.post("http://localhost:6000/read", json={
    "query": "preferências de comunicação do usuário",
    "top_k": 5,
    "min_score": 0.0
})
for r in result.json()["results"]:
    print(f"{r['score']:.4f} — {r['text']}")
    
    
    
------------------------------------------------------------------------------------
    
gerador de CoT
    
import httpx

result = httpx.post("http://localhost:6001/plan", json={
    "input": "qual a melhor GPU pra rodar modelos de 256B?",
    "context": "Felipe tem orçamento de R$2000 e usa Ubuntu",
    "use_cache": True
}, timeout=30)

plan = result.json()
for step in plan["steps"]:
    print(f"[{step['executor']}] {step['step']}. {step['action']}")

------------------------------------------------------------------------------------
    
API de busca na internet
    
    
import httpx
import json

BASE_URL = "http://localhost:6002"

def print_results(response: dict):
    print(f"\n📋 Query     : {response['query']}")
    print(f"⏱  Latência  : {response['latency_ms']}ms")
    print(f"💾 Cache     : {'sim' if response['from_cache'] else 'não'}")
    print(f"📦 Resultados: {len(response['results'])}\n")

    for i, r in enumerate(response["results"], 1):
        print(f"  [{i}] Score: {r['score']:.4f}")
        print(f"       Título : {r['title'][:70]}")
        print(f"       Fonte  : {r['source'][:70]}")
        print(f"       Trecho : {r['text'][:200]}...")
        print()

def test_status():
    print("=" * 60)
    print("🔍 GET /status")
    r = httpx.get(f"{BASE_URL}/status")
    print(json.dumps(r.json(), indent=2))

def test_search(query: str, use_cache: bool = True):
    print("=" * 60)
    print(f"🔍 POST /search — '{query}'")
    r = httpx.post(f"{BASE_URL}/search", json={
        "query":      query,
        "max_results": 3,
        "use_cache":  use_cache,
    }, timeout=30.0)
    if r.status_code == 200:
        print_results(r.json())
    else:
        print(f"❌ Erro {r.status_code}: {r.text}")

def test_cache(query: str):
    print("=" * 60)
    print(f"💾 Testando cache com: '{query}'")

    print("  1ª chamada (sem cache)...")
    r1 = httpx.post(f"{BASE_URL}/search", json={
        "query": query, "use_cache": True
    }, timeout=30.0).json()
    print(f"  Latência: {r1['latency_ms']}ms | Cache: {r1['from_cache']}")

    print("  2ª chamada (deve usar cache)...")
    r2 = httpx.post(f"{BASE_URL}/search", json={
        "query": query, "use_cache": True
    }, timeout=30.0).json()
    print(f"  Latência: {r2['latency_ms']}ms | Cache: {r2['from_cache']}")

    speedup = r1["latency_ms"] / max(r2["latency_ms"], 1)
    print(f"  ⚡ Speedup do cache: {speedup:.1f}x\n")

if __name__ == "__main__":
    test_status()
    test_search("qual a melhor GPU para rodar modelos LLM localmente em 2026?")
    test_search("como instalar o llama.cpp no Ubuntu com suporte CUDA?")
    test_cache("preço RTX 3060 no Brasil")
    
    
------------------------------------------------------------------------------------


API de Text-to-Speech
    
import base64
import io
import time
import wave
import requests
import numpy as np
import sounddevice as sd


API_URL = "http://127.0.0.1:6003"

VOICE = "M1"
LANG = "pt"

TEST_TEXT = "Olá, este é um teste da API de Text-to-Speech. Estamos testando a geração de áudio a partir de texto, utilizando a voz M1 e o idioma português. Esta API é capaz de gerar áudio de alta qualidade com baixa latência, ideal para aplicações em tempo real. Obrigado por testar!"

PLAY_AUDIO = True


def decode_wav_base64(audio_b64: str):

    wav_bytes = base64.b64decode(audio_b64)

    wav_file = wave.open(io.BytesIO(wav_bytes), "rb")

    sample_rate = wav_file.getframerate()

    frames = wav_file.readframes(wav_file.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16)

    audio = audio.astype(np.float32) / 32767.0

    return audio, sample_rate


def save_wav(audio_b64: str, filename: str):

    wav_bytes = base64.b64decode(audio_b64)

    with open(filename, "wb") as f:
        f.write(wav_bytes)


def play_audio(audio, sample_rate):

    print(f"[AUDIO] Playing at {sample_rate} Hz")

    sd.play(audio, samplerate=sample_rate, blocking=True)

def test_stream():

    print("\n================ STREAM ================\n")

    payload = {
        "text": TEST_TEXT,
        "voice": VOICE,
        "lang": LANG
    }

    t0 = time.perf_counter()

    r = requests.post(
        f"{API_URL}/stream",
        json=payload
    )

    elapsed = (time.perf_counter() - t0) * 1000

    print("HTTP:", r.status_code)

    if r.status_code != 200:
        print(r.text)
        return

    data = r.json()

    print(f"Latency total: {elapsed:.2f} ms")
    print(f"Latency API:   {data['latency_ms']} ms")
    print(f"Chunks:        {len(data['chunks'])}")
    print(f"Total audio:   {data['total_duration_s']} s")

    for chunk in data["chunks"]:

        idx = chunk["index"]

        print(
            f"\nChunk {idx}"
            f"\nDuration: {chunk['duration_s']}s"
            f"\nText: {chunk['text']}"
        )

        filename = f"stream_chunk_{idx}.wav"

        save_wav(
            chunk["audio_b64"],
            filename
        )

        print(f"Saved: {filename}")

        if PLAY_AUDIO:

            audio, sr = decode_wav_base64(
                chunk["audio_b64"]
            )

            play_audio(audio, sr)



if __name__ == "__main__":

    print("\n===================================")
    print("AVA TTS API TEST")
    print("===================================\n")

    try:
        test_stream()


        print("\n===================================")
        print("TESTS FINISHED")
        print("===================================\n")

    except requests.exceptions.ConnectionError:

        print(
            "\n[ERROR] Não foi possível conectar na API.\n"
            "Inicie primeiro:\n\n"
            "uvicorn tts_api:app --host 0.0.0.0 --port 6003\n"
        )

        
