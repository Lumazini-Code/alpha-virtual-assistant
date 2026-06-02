"""
AVA — LLM Inference API
========================
REST API para inferência conversacional com:
  - Gerenciamento do llama-server (subida/desligamento)
  - Integração correta com API de Memória (LT + ST via session_id)
  - Integração com API de TTS (localhost:3004)
  - Streaming de texto + disparo paralelo de áudio
  - Histórico de chat persistido via módulo de memória externo
  - Detecção de idioma para resposta automática

Porta: localhost:4003
"""
import sys
import json
import datetime
import asyncio
import time
from pathlib import Path
from typing import Optional
import re
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langdetect import detect

# ─────────────────────────────────────────────────────────────
#                          CONFIG
# ─────────────────────────────────────────────────────────────

BASEFOLDER = Path(__file__).parent.parent

# URLs das APIs satélite
MEMORY_URL = "http://localhost:3001"
TTS_URL    = "http://localhost:3004"

# ── Persistent TTS client (reused across chunks) ──
_tts_http: httpx.AsyncClient | None = None

def _get_tts_client() -> httpx.AsyncClient:
    global _tts_http
    if _tts_http is None or _tts_http.is_closed:
        _tts_http = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )
    return _tts_http

# llama-server
LLAMA_SERVER_PATH = r".\llama-cpp\llama-server"
LLAMA_HOST        = "localhost"
LLAMA_PORT        = 2001
LLAMA_URL         = f"http://{LLAMA_HOST}:{LLAMA_PORT}"

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
            try: self.terminal.write(msg)
            except Exception: pass
            self.log.write(msg)

        def flush(self):
            try: self.terminal.flush()
            except Exception: pass
            self.log.flush()

        def isatty(self): return False

    sys.stdout = LogDuplicado(sys.__stdout__, log_path)
    sys.stderr = LogDuplicado(sys.__stderr__, log_path)

_setup_logging()

# ─────────────────────────────────────────────────────────────
#                     LEITURA DE CONFIGS
# ─────────────────────────────────────────────────────────────

def _read(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

username   = _read(BASEFOLDER / r"resource/username.dll")
voiceModel = _read(BASEFOLDER / r"resource/VoiceModel.dll") or "F1"
ctxUsed    = _read(BASEFOLDER / r"resource/ctxConfig.dll")
context    = _read(BASEFOLDER / f"ctxBin/{ctxUsed}.bin")
searchCfg  = _read(BASEFOLDER / r"resource/SearchCfg.dll")

model_raw = _read(BASEFOLDER / r"resource/Aiconfig.dll")
model_path = model_raw.split("/") or model_raw.split("\\")
MODEL_PATH = model_path[-1]
MODEL_NAME = model_raw # Normalmente o nome completo passado pro llama

try:
    with open(BASEFOLDER / f"CfgModels/{model_raw}.json", "r", encoding="utf-8") as f:
        MODELCFG = json.load(f)
except FileNotFoundError:
    MODELCFG = {}

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
#                     INTEGRAÇÃO: MEMÓRIA
# ─────────────────────────────────────────────────────────────

async def memory_read(query: str, session_id: Optional[str] = None, top_k: int = 10) -> list[dict]:
    """
    Busca memórias relevantes (Long-Term e Short-Term) para o contexto da conversa.
    Passar o session_id permite que o módulo de memória retorne o histórico recente.
    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.post(
                f"{MEMORY_URL}/read",
                json={
                    "query": query, 
                    "top_k": top_k, 
                    "min_score": 0.3,
                    "session_id": session_id,
                    "strategy": "auto"  # Permite ao módulo buscar LT e ST
                },
            )
            return r.json().get("results", [])
    except Exception as e:
        print(f"[MEMORY] Falha na leitura: {e}")
        return []


async def memory_save_turn(session_id: str, user_input: str, assistant_response: str):
    """Grava o par de turnos (user + assistant) na memória de curto prazo (/write_st)."""
    if not session_id: 
        return
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{MEMORY_URL}/write_st",
                json={
                    "session_id": session_id,
                    "turns": [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": assistant_response}
                    ]
                },
            )
    except Exception as e:
        print(f"[MEMORY] Falha ao salvar turno ST: {e}")


async def memory_write_fact(text: str, source: str = "chat", confidence: float = 0.7):
    """Extrai e grava informações relevantes na memória de longo prazo (/write)."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{MEMORY_URL}/write",
                json={"text": text, "source": source, "confidence": confidence},
            )
    except Exception as e:
        print(f"[MEMORY] Falha na escrita LT: {e}")


# ─────────────────────────────────────────────────────────────
#                      INTEGRAÇÃO: TTS
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
#                     INTEGRAÇÃO: TTS
# ─────────────────────────────────────────────────────────────

_tts_queue: Optional[asyncio.Queue] = None
_tts_http: Optional[httpx.AsyncClient] = None

async def _tts_sender_worker():
    """
    Worker em background que envia sentenças para o TTS /stream SEQUENCIALMENTE.
    Garante que o TTS receba e processe os chunks na ordem correta,
    sem intercalar sequências e sem causar pulos no HeapPlayer.
    """
    global _tts_http
    _tts_http = httpx.AsyncClient(
        base_url=TTS_URL,
        timeout=httpx.Timeout(15.0),
        limits=httpx.Limits(max_keepalive_connections=2, max_connections=4),
    )
    while True:
        text, voice, lang = await _tts_queue.get()
        try:
            r = await _tts_http.post(
                "/stream",
                json={"text": text, "voice": voice, "lang": lang},
            )
            if r.status_code >= 400:
                print(f"[TTS] Chunk rejeitado ({r.status_code}): '{text[:60]}'")
        except Exception as e:
            print(f"[TTS] Falha no chunk: {type(e).__name__}: {e}")
        finally:
            _tts_queue.task_done()

def _ensure_tts_queue():
    global _tts_queue
    if _tts_queue is None:
        _tts_queue = asyncio.Queue()
        asyncio.create_task(_tts_sender_worker())

async def tts_speak(text: str, voice: str, lang: str):
    """Usado pelo endpoint síncrono /chat para enfileirar fala."""
    _ensure_tts_queue()
    await _tts_queue.put((text[:2000], voice, lang))

# ─────────────────────────────────────────────────────────────
#                       CONSTRUÇÃO DO PROMPT
# ─────────────────────────────────────────────────────────────

def _build_messages(
    user_input: str,
    lang: str,
    memories: list[dict],
) -> list[dict]:
    """
    Monta a lista de mensagens para o llama-server.
    O histórico de curto prazo já vem ordenado dentro de `memories` 
    quando usamos session_id no memory_read.
    """
    memory_block = ""
    if memories:
        mem_lines = "\n".join(
            f"- {m.get('text', m.get('content', ''))}" 
            for m in memories if m.get("text") or m.get("content")
        )
        if mem_lines:
            memory_block = f"\n\n[Contexto e Memórias Relevantes]\n{mem_lines}"

    system_content = (
        f"{context}{memory_block}\n\n"
        f"O nome do usuário é {username}. "
        f"Data de hoje: {datetime.datetime.now().strftime('%d/%m/%Y')}. "
        f"Responda sempre em {lang}."
    )

    messages = [{"role": "system", "content": system_content}]

    # O histórico recente já foi injetado via memória de curto prazo (ST).
    # Apenas adicionamos a mensagem atual do usuário.
    messages.append({"role": "user", "content": user_input})
    
    return messages

# ─────────────────────────────────────────────────────────────
#                           FASTAPI
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="AVA — LLM API", version="2.0.0")

# ── Schemas ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = Field(default="default", description="ID da sessão para memória de curto prazo")
    voice:   Optional[str] = Field(default=None, description="Voz TTS. None = usa padrão.")
    lang:    Optional[str] = Field(default=None, description="Idioma forçado. None = detectado.")
    max_turns: int = Field(default=10,  ge=1, le=40, description="Limite de contexto recuperado")
    tts: bool = Field(default=True, description="Dispara TTS após gerar resposta.")

class ClearRequest(BaseModel):
    confirm: bool = False
    session_id: Optional[str] = "default"

# ── Ciclo de vida ─────────────────────────────────────────────

@app.on_event("startup")
def startup():
    _warmup()

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
async def chat(req: ChatRequest):
    """
    Inferência síncrona — retorna a resposta completa em JSON.
    Memória e TTS são disparados em background.
    """
    user_input = req.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Mensagem vazia.")

    # 1. Detectar idioma
    lang = req.lang or _safe_detect(user_input)

    # 2. Buscar memórias (inclui histórico ST via session_id)
    memories = await memory_read(user_input, session_id=req.session_id, top_k=req.max_turns)

    # 3. Montar prompt
    messages = _build_messages(user_input, lang, memories)

    # 4. Inferência
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{LLAMA_URL}/v1/chat/completions",
                json={
                    "model":       MODEL_NAME,
                    "messages":    messages,
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

    # 5. Gravar turno na memória de curto prazo (fire-and-forget)
    asyncio.create_task(
        memory_save_turn(req.session_id, user_input, response_text)
    )

    # 6. Gravar fato relevante na memória de longo prazo (fire-and-forget)
    asyncio.create_task(
        memory_write_fact(f"Usuário disse: {user_input[:300]}", "chat", 0.7)
    )

    # 7. Disparar TTS em background
    voice = req.voice or voiceModel
    if req.tts and voice:
        asyncio.create_task(tts_speak(response_text, voice, lang))

    return {
        "response": response_text,
        "session_id": req.session_id,
        "lang":     lang,
        "elapsed":  round(elapsed, 3),
    }
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Inferência com streaming — retorna Server-Sent Events (SSE).
    TTS é disparado por SENTENÇA COMPLETA via fila sequencial.
    Memória é salva ao final.
    """
    user_input = req.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Mensagem vazia.")

    lang      = req.lang or _safe_detect(user_input)
    memories  = await memory_read(user_input, session_id=req.session_id, top_k=req.max_turns)
    messages  = _build_messages(user_input, lang, memories)
    voice     = req.voice or voiceModel
    
    _ensure_tts_queue()
    _tts_buf = ""

    def _clean_for_tts(text: str) -> str:
        """Remove formatação Markdown que o TTS não consegue ler."""
        t = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)   # bold/italic
        t = re.sub(r'`{1,3}[^`]*`{1,3}', '', t)            # inline code
        t = re.sub(r'#{1,6}\s+', '', t)                      # headers
        t = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', t)       # links
        t = re.sub(r'^\s*[-*]\s+', '', t, flags=re.M)        # list bullets
        t = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', t)           # underline
        return t.strip()

    def _flush_tts_buf():
        nonlocal _tts_buf
        cleaned = _clean_for_tts(_tts_buf)
        if len(cleaned) >= 3:  # Ignora fragmentos minúsculos
            _tts_queue.put_nowait((cleaned, voice, lang))
        _tts_buf = ""

    async def generator():
        nonlocal _tts_buf
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
                            # Envia o delta para o cliente IMEDIATAMENTE
                            yield f"data: {json.dumps({'delta': delta})}\n\n"

                            # Buffer para o TTS — só envia quando a sentença termina
                            if req.tts and voice:
                                _tts_buf += delta
                                buf_rstrip = _tts_buf.rstrip()
                                # Flush ao final de pontuação ou se ficar muito longo
                                if (buf_rstrip and buf_rstrip[-1] in '.!?\n。') \
                                   or len(_tts_buf) > 150:
                                    _flush_tts_buf()

                        except (json.JSONDecodeError, KeyError):
                            continue

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        # Faz flush do que sobrou no buffer de texto
        if _tts_buf.strip():
            _flush_tts_buf()

        elapsed = time.perf_counter() - t0
        print(f"[STREAM] Concluído em {elapsed:.2f}s | {len(full_response)} chars")

        # Sinaliza fim ao cliente
        yield f"data: {json.dumps({'done': True, 'elapsed': round(elapsed, 3)})}\n\n"

        # Salva memória (fire-and-forget)
        if full_response:
            asyncio.create_task(
                memory_save_turn(req.session_id, user_input, full_response)
            )
            asyncio.create_task(
                memory_write_fact(f"Usuário disse: {user_input[:300]}", "chat", 0.7)
            )
            # O TTS já foi disparado aos poucos pelo buffer durante a geração.
            # NÃO chamar tts_speak() aqui para não repetir o áudio inteiro.

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.delete("/history")
async def clear_history(req: ClearRequest):
    """Limpa o histórico de chat da sessão no servidor de memória. Requer confirm=true."""
    if not req.confirm:
        raise HTTPException(status_code=400, detail="Envie confirm=true para confirmar.")
    
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.delete(f"{MEMORY_URL}/session/{req.session_id}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Falha ao limpar sessão: {e}")
    
    return {"cleared": True, "session_id": req.session_id}


@app.get("/history")
async def get_history(session_id: str = "default", last_n: int = 20):
    """Retorna as últimas N mensagens do histórico via módulo de memória."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.post(
                f"{MEMORY_URL}/read",
                json={"query": "histórico recente", "session_id": session_id, "top_k": last_n}
            )
            results = r.json().get("results", [])
            return {"history": results, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Falha ao buscar histórico: {e}")

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