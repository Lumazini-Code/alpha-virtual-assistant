"""
AVA TTS API — Síntese de Voz via Supertonic
============================================
Modos  : POST /speak   — sintetiza texto e retorna áudio WAV em base64
         POST /stream  — sintetiza múltiplas sentenças em sequência
         GET  /voices  — lista vozes disponíveis
         GET  /status  — status do servidor

Dependências:
  pip install fastapi uvicorn supertonic numpy sounddevice pydantic

Uso:
  uvicorn tts_api:app --host 0.0.0.0 --port 6003 --log-level info
"""

from __future__ import annotations

import re
import time
import base64
import queue
import threading
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import asyncio
import io

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supertonic import TTS

# ── Configuração ───────────────────────────────────────────────────────────────

DEFAULT_VOICE   = "M1"
DEFAULT_LANG    = "pt"
SAMPLE_RATE     = 48000
NUM_WORKERS     = 2
WORKER_TIMEOUT  = 30.0   # segundos máximos por síntese

AVAILABLE_VOICES = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ava.tts")

# ── Modelos de request/response ────────────────────────────────────────────────

class SpeakRequest(BaseModel):
    text:   str
    voice:  str   = DEFAULT_VOICE
    lang:   str   = DEFAULT_LANG
    speed:  float = 1.0             # >1.0 acelera, <1.0 desacelera

class SpeakResponse(BaseModel):
    audio_b64:   str     # WAV em base64
    duration_s:  float
    latency_ms:  float
    text:        str

class StreamRequest(BaseModel):
    text:  str
    voice: str  = DEFAULT_VOICE
    lang:  str  = DEFAULT_LANG

class StreamChunk(BaseModel):
    index:      int
    audio_b64:  str
    duration_s: float
    text:       str

class StreamResponse(BaseModel):
    chunks:     list[StreamChunk]
    total_duration_s: float
    latency_ms: float

class VoicesResponse(BaseModel):
    voices: list[str]
    default: str

# ── Worker TTS ─────────────────────────────────────────────────────────────────

class TTSWorker(threading.Thread):
    """
    Worker dedicado com instância própria do modelo Supertonic.
    Recebe tarefas via task_queue, devolve resultados via result_queue.
    """

    def __init__(self, worker_id: int, task_queue: queue.Queue, result_queue: queue.Queue):
        super().__init__(daemon=True, name=f"tts-worker-{worker_id}")
        self.worker_id    = worker_id
        self.task_queue   = task_queue
        self.result_queue = result_queue
        self._tts         = None
        self._voice_cache: dict[str, object] = {}

    def _load(self):
        log.info(f"[Worker {self.worker_id}] Carregando modelo Supertonic...")
        self._tts = TTS(auto_download=True)
        log.info(f"[Worker {self.worker_id}] Pronto")

    def _get_style(self, voice: str) -> object:
        if voice not in self._voice_cache:
            self._voice_cache[voice] = self._tts.get_voice_style(voice_name=voice)
        return self._voice_cache[voice]

    def run(self):
        self._load()

        while True:
            item = self.task_queue.get()
            if item is None:
                break

            task_id, text, voice, lang, speed, result_event, result_holder = item

            try:
                style = self._get_style(voice)

                # Supertonic aceita speed como parâmetro
                kwargs = {"text": text, "voice_style": style, "lang": lang}
                if speed != 1.0:
                    kwargs["speed"] = speed

                wav, duration = self._tts.synthesize(**kwargs)

                # Normaliza para numpy float32
                if hasattr(wav, "cpu"):
                    wav = wav.cpu().numpy()
                wav = np.asarray(wav, dtype=np.float32).squeeze()

                result_holder["wav"]      = wav
                # Corrige duration vindo como tensor/list/array
                if hasattr(duration, "cpu"):
                    duration = duration.cpu().numpy()

                duration = np.asarray(duration).reshape(-1)

                if len(duration) > 0:
                    duration = float(duration[0])
                else:
                    duration = 0.0

                result_holder["duration"] = duration
                result_holder["error"]    = None

            except Exception as e:
                log.error(f"[Worker {self.worker_id}] Erro: {e}")
                result_holder["error"] = str(e)

            finally:
                result_event.set()
                self.task_queue.task_done()


# ── Pool de workers ────────────────────────────────────────────────────────────

class TTSPool:
    def __init__(self, num_workers: int = NUM_WORKERS):
        self._task_queue   = queue.Queue()
        self._result_queue = queue.Queue()
        self._workers: list[TTSWorker] = []
        self._task_counter = 0
        self._lock = threading.Lock()
        self._num_workers = num_workers

    def start(self):
        for i in range(self._num_workers):
            w = TTSWorker(i, self._task_queue, self._result_queue)
            w.start()
            self._workers.append(w)
        log.info(f"TTSPool iniciado com {self._num_workers} workers")

    def stop(self):
        for _ in self._workers:
            self._task_queue.put(None)

    def synthesize(self, text: str, voice: str = DEFAULT_VOICE,
                   lang: str = DEFAULT_LANG, speed: float = 1.0,
                   timeout: float = WORKER_TIMEOUT) -> tuple[np.ndarray, float]:
        """
        Envia tarefa para o pool e aguarda resultado de forma síncrona.
        Thread-safe — pode ser chamado de múltiplas corrotinas via run_in_executor.
        """
        with self._lock:
            task_id = self._task_counter
            self._task_counter += 1

        result_event  = threading.Event()
        result_holder = {"wav": None, "duration": 0.0, "error": None}

        self._task_queue.put((task_id, text, voice, lang, speed, result_event, result_holder))

        if not result_event.wait(timeout=timeout):
            raise TimeoutError(f"TTS timeout após {timeout}s para: {text[:40]}")

        if result_holder["error"]:
            raise RuntimeError(result_holder["error"])

        return result_holder["wav"], result_holder["duration"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def wav_to_b64(wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Converte numpy float32 para WAV PCM 16-bit em base64."""
    import struct

    pcm = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()

    # Escreve header WAV
    data_size   = pcm.nbytes
    num_ch      = 1
    bits        = 16
    byte_rate   = sample_rate * num_ch * bits // 8
    block_align = num_ch * bits // 8

    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, num_ch, sample_rate,
                          byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())

    return base64.b64encode(buf.getvalue()).decode()


def split_sentences(text: str) -> list[str]:
    """Divide texto em sentenças para síntese em stream."""
    sentences = re.split(r'(?<=[.!?\n])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ── Estado global ──────────────────────────────────────────────────────────────

@dataclass
class AppState:
    pool: TTSPool = field(default=None)

state = AppState()


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Iniciando AVA TTS API...")

    state.pool = TTSPool(num_workers=NUM_WORKERS)
    state.pool.start()

    # Warmup — primeira síntese é mais lenta por carregar pesos
    log.info("Warmup TTS...")
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, state.pool.synthesize, ".", DEFAULT_VOICE, DEFAULT_LANG)
        log.info("TTS API pronta")
    except Exception as e:
        log.warning(f"Warmup falhou (não crítico): {e}")

    yield

    state.pool.stop()
    log.info("AVA TTS API encerrada")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="AVA TTS API", lifespan=lifespan)


# ── POST /speak ────────────────────────────────────────────────────────────────

@app.post("/speak", response_model=SpeakResponse)
async def speak(req: SpeakRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="texto vazio")

    if req.voice not in AVAILABLE_VOICES:
        raise HTTPException(status_code=400,
                            detail=f"voz inválida — use uma de: {AVAILABLE_VOICES}")

    t0   = time.perf_counter()
    loop = asyncio.get_event_loop()

    try:
        wav, duration = await loop.run_in_executor(
            None,
            lambda: state.pool.synthesize(text, req.voice, req.lang, req.speed)
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="TTS timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    audio_b64  = wav_to_b64(wav)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    log.info(f"Sintetizado em {latency_ms}ms: {text[:60]}")

    return SpeakResponse(
        audio_b64  = audio_b64,
        duration_s = round(duration, 3),
        latency_ms = latency_ms,
        text       = text,
    )


# ── POST /stream ───────────────────────────────────────────────────────────────

@app.post("/stream", response_model=StreamResponse)
async def stream(req: StreamRequest):
    """
    Divide o texto em sentenças e sintetiza cada uma em paralelo
    mantendo ordem de índice. Útil para textos longos.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="texto vazio")

    if req.voice not in AVAILABLE_VOICES:
        raise HTTPException(status_code=400,
                            detail=f"voz inválida — use uma de: {AVAILABLE_VOICES}")

    sentences = split_sentences(text)
    if not sentences:
        raise HTTPException(status_code=400, detail="nenhuma sentença encontrada")

    t0   = time.perf_counter()
    loop = asyncio.get_event_loop()

    # Sintetiza todas as sentenças em paralelo
    async def synth_one(idx: int, sentence: str) -> StreamChunk:
        wav, duration = await loop.run_in_executor(
            None,
            lambda: state.pool.synthesize(sentence, req.voice, req.lang)
        )
        return StreamChunk(
            index      = idx,
            audio_b64  = wav_to_b64(wav),
            duration_s = round(duration, 3),
            text       = sentence,
        )

    try:
        tasks  = [synth_one(i, s) for i, s in enumerate(sentences)]
        chunks = await asyncio.gather(*tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Garante ordem por índice
    chunks = sorted(chunks, key=lambda c: c.index)
    total  = sum(c.duration_s for c in chunks)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    log.info(f"Stream de {len(chunks)} sentenças em {latency_ms}ms")

    return StreamResponse(
        chunks           = chunks,
        total_duration_s = round(total, 3),
        latency_ms       = latency_ms,
    )


# ── GET /voices ────────────────────────────────────────────────────────────────

@app.get("/voices", response_model=VoicesResponse)
async def voices():
    return VoicesResponse(voices=AVAILABLE_VOICES, default=DEFAULT_VOICE)


# ── GET /status ────────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    return {
        "workers":      NUM_WORKERS,
        "default_voice": DEFAULT_VOICE,
        "sample_rate":  SAMPLE_RATE,
        "voices":       AVAILABLE_VOICES,
    }


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3004, log_level="info")