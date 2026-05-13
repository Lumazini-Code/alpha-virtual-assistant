padrão de uso das portas de API rest localHost (0.0.0.0):

0-1024: não é permitido o uso (Linux bloqueia)

2000-2999: conexões Llama server

    2001 = LLM
    2002 = code generator
    2003 = CoT generator
    2004 = VQA


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


busca na memória:

result = httpx.post("http://localhost:3001/read", json={
    "query": "preferências de comunicação do usuário",
    "top_k": 5,
    "min_score": 0.0
})
for r in result.json()["results"]:
    print(f"{r['score']:.4f} — {r['text']}")
    

-----------------


escrita na memória

result = httpx.post("http://localhost:3001/write", json={
    "text": "eu gosto de maracujá",
    "source": "user_explicit", # chat, user_explicit ou system
    "confidence": 1.0
})
for r in result.json()["results"]:
    print(f"{r['stored']} — {r['reason']}")
    
    
------------------------------------------------------------------------------------
    
gerador de CoT
    
import httpx

result = httpx.post("http://localhost:3000/plan", json={
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

BASE_URL = "http://localhost:3002"

def print_results(response: dict):
    for i, r in enumerate(response["results"], 1):
        print(f"       Fonte  : {r['source'][:100]}")
        print(f"       Trecho : {r['text'][:1000]}...")


def test_search(query: str, use_cache: bool = True):
    r = httpx.post(f"{BASE_URL}/search", json={
        "query":      query,
        "max_results": 3,
        "use_cache":  use_cache,
    }, timeout=30.0)
    if r.status_code == 200:
        print_results(r.json())
    else:
        print(f"❌ Erro {r.status_code}: {r.text}")

if __name__ == "__main__":
    test_search("como instalar o llama.cpp no Ubuntu com suporte CUDA?")
    
    
------------------------------------------------------------------------------------


API de Text-to-Speech
    

import httpx


API_URL = "http://127.0.0.1:3004"

VOICE = "M1"
LANG = "pt"

TEST_TEXT = "Olá, este é um teste da API de Text-to-Speech. Estamos testando a geração de áudio a partir de texto, utilizando a voz M1 e o idioma português. Esta API é capaz de gerar áudio de alta qualidade com baixa latência, ideal para aplicações em tempo real. Obrigado por testar!"

# geração de audio a aprtir de texto completo
r = httpx.post(f"{API_URL}/stream", json={
    "text":  TEST_TEXT,
    "voice": VOICE,
    "lang":  LANG   
}, timeout=30.0)

if r.status_code == 200:
    print_results(r.json())
else:
    print(f"❌ Erro {r.status_code}: {r.text}")

# geração de audio para texto gerado via streaming

r = httpx.post(f"{API_URL}/speak", json={
    "text":  TEST_TEXT,
    "voice": VOICE,
    "lang":  LANG   
}, timeout=30.0)

if r.status_code == 200:
    print_results(r.json())
else:
    print(f"❌ Erro {r.status_code}: {r.text}")


--------------------------------------------------------------------------------------------


## Módulos principais




API LLM principal


import httpx

r = httpx.post("0.0.0.0:4003/chat", {
  "message": "Qual é meu nome?",
  "voice": "F1",
  "lang": null,
  "max_turns": 10,
  "tts": true
}, timeout=30.0)

if r.status_code == 200:
    print_results(r.json())
else:
    print(f"❌ Erro {r.status_code}: {r.text}")



r = httpx.post("0.0.0.0:4003/chat/stream", {
  "message": "Qual é meu nome?",
  "voice": "F1",
  "lang": null,
  "max_turns": 10,
  "tts": true
}, timeout=30.0)

if r.status_code == 200:
    print_results(r.json())
else:
    print(f"❌ Erro {r.status_code}: {r.text}")



