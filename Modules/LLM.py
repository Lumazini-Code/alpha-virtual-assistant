"""
AVA — LLM Inference API
========================
REST API para inferência conversacional com:
  - Gerenciamento do llama-server (subida/desligamento)
  - Integração com API de memória (localhost:3001)
  - Integração com API de TTS (localhost:3004)
  - Streaming de texto + disparo paralelo de áudio
  - Histórico de chat persistido em JSON
  - Detecção de idioma para resposta automática

Porta: 0.0.0.0:4003
"""

import sys
import os
import json
import datetime
import asyncio
import subprocess
import time
from pathlib import Path
from threading import Event

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langdetect import detect

# ─────────────────────────────────────────────────────────────
#                          CONFIG
# ─────────────────────────────────────────────────────────────

BASEFOLDER = Path(__file__).parent.parent

# URLs das APIs satélite
MEMORY_URL = "http://localhost:3001"
TTS_URL    = "http://0.0.0.0:3004"

# llama-server
LLAMA_SERVER_PATH = r".\llama-cpp\llama-server"
LLAMA_HOST        = "0.0.0.0"
LLAMA_PORT        = 2001
LLAMA_URL         = f"http://{LLAMA_HOST}:{LLAMA_PORT}"

# Histórico
CHAT_HISTORY_PATH = BASEFOLDER / "chat_history.json"

# ─────────────────────────────────────────────────────────────
#                       LOGGING DUPLO
# ─────────────────────────────────────────────────────────────

def _setup_logging():
    log_dir = BASEFOLDER / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"LLM_api_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

    class LogDuplicado:
        def __init__(self, terminal, path):
            self.terminal = terminal
            self.log = open(path, "w", encoding="utf-8")

        def write(self, msg):
            try:
                self.terminal.write(msg)
            except Exception:
                pass
            self.log.write(msg)

        def flush(self):
            try:
                self.terminal.flush()
            except Exception:
                pass
            self.log.flush()

        def isatty(self):
            return False

    sys.stdout = LogDuplicado(sys.__stdout__, log_path)
    sys.stderr = LogDuplicado(sys.__stderr__, log_path)

_setup_logging()

# ─────────────────────────────────────────────────────────────
#                     LEITURA DE CONFIGS
# ─────────────────────────────────────────────────────────────

def _read(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

username   = _read(BASEFOLDER / r"resource\username.dll")
voiceModel = _read(BASEFOLDER / r"resource\VoiceModel.dll") or "F1"
ctxUsed    = _read(BASEFOLDER / r"resource\ctxConfig.dll")
context    = _read(BASEFOLDER / f"Ctxbin/{ctxUsed}.bin")
searchCfg  = _read(BASEFOLDER / r"resource\SearchCfg.dll")

model_raw = _read(BASEFOLDER / "resource/AiConfig.dll").replace("on-", "")

with open(BASEFOLDER / f"CfgModels/{model_raw}.json", "r", encoding="utf-8") as f:
    MODELCFG = json.load(f)

MODEL_NAME      = model_raw
THREADS         = (max(4, os.cpu_count() - 2)
                   if MODELCFG["threads"] == "max"
                   else int(MODELCFG["threads"]))
KV_CACHE_QUANT  = MODELCFG.get("kv_cache", "q8_0")
MODEL_PATH      = MODELCFG.get("model_path", "")
CTX_SIZE        = int(MODELCFG.get("ctx_size", 8192))
GPU_LAYERS      = MODELCFG.get("gpu_layers", "all")

LLAMA_ARGS = [
    "--model",         MODEL_PATH,
    "--host",          LLAMA_HOST,
    "--port",          str(LLAMA_PORT),
    "--n-gpu-layers",  str(GPU_LAYERS),
    "--threads",       str(THREADS),
    "--threads-batch", str(min(THREADS + 2, os.cpu_count())),
    "--batch-size",    "2048",
    "--ubatch-size",   "512",
    "--flash-attn",    "on",
    "--cache-type-k",  KV_CACHE_QUANT if KV_CACHE_QUANT in ("q4_0", "q8_0") else "f16",
    "--cache-type-v",  "q8_0",
    "--ctx-size",      str(CTX_SIZE),
    "--parallel",      "1",
    "--cont-batching",
    "--mmap",
    "--cache-reuse",   "256",
    "--slot-prompt-similarity", "0.5",
]

# ─────────────────────────────────────────────────────────────
#                    GERENCIAMENTO DO SERVIDOR
# ─────────────────────────────────────────────────────────────

llama_proc: subprocess.Popen | None = None


def _start_llama():
    global llama_proc
    print(f"[SERVER] Iniciando llama-server com modelo: {MODEL_NAME}")
    llama_proc = subprocess.Popen(
        [LLAMA_SERVER_PATH] + LLAMA_ARGS,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def _wait_llama(timeout: int = 90):
    print("[SERVER] Aguardando llama-server...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{LLAMA_URL}/health", timeout=2)
            if r.status_code == 200:
                print("[SERVER] llama-server pronto!\n")
                return
        except httpx.ConnectError:
            pass
        time.sleep(1)
    raise TimeoutError("[SERVER] llama-server não respondeu a tempo.")


def _stop_llama():
    if llama_proc:
        print("[SERVER] Encerrando llama-server...")
        llama_proc.terminate()
        try:
            llama_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            llama_proc.kill()


def _warmup():
    """Requisição mínima para pré-aquecer KV cache e GPU."""
    print("[MAIN] Warmup do modelo...")
    try:
        httpx.post(
            f"{LLAMA_URL}/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "ok"}],
                "max_tokens": 1,
            },
            timeout=30,
        )
        print("[MAIN] Warmup concluído.\n")
    except Exception as e:
        print(f"[MAIN] Warmup falhou (não crítico): {e}")

# ─────────────────────────────────────────────────────────────
#                      HISTÓRICO DE CHAT
# ─────────────────────────────────────────────────────────────

def _load_history() -> list:
    try:
        with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]
    except FileNotFoundError:
        return []


def _save_history(history: list):
    with open(CHAT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


chat_history: list = _load_history()

# ─────────────────────────────────────────────────────────────
#                     INTEGRAÇÃO: MEMÓRIA
# ─────────────────────────────────────────────────────────────

async def memory_read(query: str, top_k: int = 5) -> list[dict]:
    """Busca memórias relevantes para o contexto da conversa."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.post(
                f"{MEMORY_URL}/read",
                json={"query": query, "top_k": top_k, "min_score": 0.3},
            )
            return r.json().get("results", [])
    except Exception as e:
        print(f"[MEMORY] Falha na leitura: {e}")
        return []


async def memory_write(text: str, source: str = "chat", confidence: float = 0.8):
    """Grava informação relevante na memória em background."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{MEMORY_URL}/write",
                json={"text": text, "source": source, "confidence": confidence},
            )
    except Exception as e:
        print(f"[MEMORY] Falha na escrita: {e}")


# ─────────────────────────────────────────────────────────────
#                      INTEGRAÇÃO: TTS
# ─────────────────────────────────────────────────────────────

async def tts_stream_chunk(text: str, voice: str, lang: str):
    """Envia um delta de texto diretamente ao endpoint de streaming do TTS."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"{TTS_URL}/stream",
                json={"text": text, "voice": voice, "lang": lang},
            )
    except Exception as e:
        print(f"[TTS] Falha no chunk: {e}")


async def tts_speak(text: str, voice: str, lang: str):
    """
    Fallback para texto completo (usado pelo /chat síncrono).
    """
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(
                f"{TTS_URL}/speak",
                json={"text": text, "voice": voice, "lang": lang},
            )
    except Exception as e:
        print(f"[TTS] Falha ao disparar áudio: {e}")




# ─────────────────────────────────────────────────────────────
#                       CONSTRUÇÃO DO PROMPT
# ─────────────────────────────────────────────────────────────

def _build_messages(
    user_input: str,
    lang: str,
    memories: list[dict],
    max_turns: int = 10,
) -> list[dict]:
    """
    Monta a lista de mensagens para o llama-server.
    Injeta memórias relevantes no system prompt sem poluir o histórico.
    """
    memory_block = ""
    if memories:
        mem_lines = "\n".join(
            f"- {m['text']}" for m in memories if m.get("text")
        )
        memory_block = f"\n\n[Memórias relevantes sobre o usuário]\n{mem_lines}"

    system_content = (
        f"{context}{memory_block}\n\n"
        f"O nome do usuário é {username}. "
        f"Data de hoje: {datetime.datetime.now().strftime('%d/%m/%Y')}. "
        f"Responda sempre em {lang}."
    )

    messages = [{"role": "system", "content": system_content}]

    # Histórico recente (últimas max_turns rodadas)
    for turn in chat_history[-(max_turns * 2):]:
        if isinstance(turn, dict) and "role" in turn and "content" in turn:
            messages.append({
                "role":    turn["role"],
                "content": turn["content"].strip()[:600],  # trunca turns longos
            })

    messages.append({"role": "user", "content": user_input})
    return messages

# ─────────────────────────────────────────────────────────────
#                           FASTAPI
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="AVA — LLM API", version="1.0.0")


# ── Schemas ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    voice:   str  = Field(default=None, description="Voz TTS. None = sem áudio.")
    lang:    str  = Field(default=None, description="Idioma forçado. None = detectado.")
    max_turns: int = Field(default=10,  ge=1, le=40)
    tts: bool = Field(default=True, description="Dispara TTS após gerar resposta.")


class ClearRequest(BaseModel):
    confirm: bool = False


# ── Ciclo de vida ─────────────────────────────────────────────

@app.on_event("startup")
def startup():
    _start_llama()
    _wait_llama()
    _warmup()


@app.on_event("shutdown")
def shutdown():
    _stop_llama()


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
def health():
    """Verifica se a API e o llama-server estão no ar."""
    try:
        r = httpx.get(f"{LLAMA_URL}/health", timeout=2)
        llama_ok = r.status_code == 200
    except Exception:
        llama_ok = False
    return {"api": "ok", "llama_server": "ok" if llama_ok else "down"}


@app.post("/chat")
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    """
    Inferência síncrona — retorna a resposta completa em JSON.
    TTS é disparado em background após a geração.

    Body:
        message   : mensagem do usuário
        voice     : voz TTS (ex: "F1", "M1") — None desativa TTS
        lang      : idioma forçado — None detecta automaticamente
        max_turns : quantas rodadas do histórico incluir (padrão 10)
        tts       : ativa/desativa disparo de áudio (padrão true)
    """
    user_input = req.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Mensagem vazia.")

    # 1. Detectar idioma
    lang = req.lang or _safe_detect(user_input)

    # 2. Buscar memórias relevantes
    memories = await memory_read(user_input)

    # 3. Montar prompt
    messages = _build_messages(user_input, lang, memories, req.max_turns)

    # 4. Inferência
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{LLAMA_URL}/v1/chat/completions",
                json={
                    "model":       MODEL_NAME,
                    "messages":    messages,
                    "max_tokens":  1024,
                    "temperature": 0.7,
                    "stream":      False,
                },
            )
            r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"llama-server error: {e}")

    response_text = r.json()["choices"][0]["message"]["content"]
    elapsed = time.perf_counter() - t0
    print(f"[CHAT] Concluído em {elapsed:.2f}s | {len(response_text)} chars")

    # 5. Persistir histórico
    chat_history.append({"role": "user",      "content": user_input})
    chat_history.append({"role": "assistant", "content": response_text})
    _save_history(chat_history)

    # 6. Gravar memória relevante em background
    background_tasks.add_task(
        memory_write,
        f"Usuário disse: {user_input[:200]}",
        "chat",
        0.7,
    )

    # 7. Disparar TTS em background
    voice = req.voice or voiceModel
    if req.tts and voice:
        background_tasks.add_task(tts_speak, response_text, voice, lang)

    return {
        "response": response_text,
        "lang":     lang,
        "elapsed":  round(elapsed, 3),
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Inferência com streaming — retorna Server-Sent Events (SSE).
    Cada evento tem o formato: data: {"delta": "..."}\n\n
    Após [DONE], o TTS é disparado com o texto completo acumulado.

    Ideal para interface com typewriter effect.
    """
    user_input = req.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Mensagem vazia.")

    lang      = req.lang or _safe_detect(user_input)
    memories  = await memory_read(user_input)
    messages  = _build_messages(user_input, lang, memories, req.max_turns)
    voice     = req.voice or voiceModel

    async def generator():
        full_response = ""
        t0 = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    f"{LLAMA_URL}/v1/chat/completions",
                    json={
                        "model":       MODEL_NAME,
                        "messages":    messages,
                        "max_tokens":  1024,
                        "temperature": 0.7,
                        "stream":      True,
                    },
                ) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if not delta:
                                continue

                            full_response += delta

                            # Repassa delta para o cliente (UI)
                            yield f"data: {json.dumps({'delta': delta})}\n\n"

                            # Dispara delta pro TTS imediatamente, sem buffer
                            if req.tts and voice:
                                asyncio.create_task(tts_stream_chunk(delta, voice, lang))

                        except (json.JSONDecodeError, KeyError):
                            continue

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        elapsed = time.perf_counter() - t0
        print(f"[STREAM] Concluído em {elapsed:.2f}s | {len(full_response)} chars")

        # Sinaliza fim ao cliente
        yield f"data: {json.dumps({'done': True, 'elapsed': round(elapsed, 3)})}\n\n"

        # Persistir histórico
        chat_history.append({"role": "user",      "content": user_input})
        chat_history.append({"role": "assistant", "content": full_response})
        _save_history(chat_history)

        # Gravar memória
        asyncio.create_task(
            memory_write(f"Usuário disse: {user_input[:200]}", "chat", 0.7)
        )

        # Disparar TTS com texto completo
        if req.tts and voice and full_response:
            asyncio.create_task(tts_speak(full_response, voice, lang))

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.delete("/history")
def clear_history(req: ClearRequest):
    """Limpa o histórico de chat. Requer confirm=true."""
    if not req.confirm:
        raise HTTPException(status_code=400, detail="Envie confirm=true para confirmar.")
    chat_history.clear()
    _save_history(chat_history)
    return {"cleared": True}


@app.get("/history")
def get_history(last_n: int = 20):
    """Retorna as últimas N mensagens do histórico."""
    return {"history": chat_history[-(last_n * 2):], "total": len(chat_history)}


# ─────────────────────────────────────────────────────────────
#                         UTILITÁRIOS
# ─────────────────────────────────────────────────────────────

def _safe_detect(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "pt"


# ─────────────────────────────────────────────────────────────
#                         ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4003, log_level="info")