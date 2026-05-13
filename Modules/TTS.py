"""
AVA TTS API — Supertonic + Arquitetura Piper Battle-Tested
===========================================================
Engine   : Supertonic (mantido do módulo novo)
Playback : miniaudio 40ms, heap ordenado por seq_idx (do módulo Piper)
Pipeline : FastAPI REST + worker pool + fila de texto + heap de áudio

Endpoints:
  POST /speak   — sintetiza texto e toca (blocking por sentença)
  POST /stream  — divide em chunks, sintetiza em paralelo, toca em ordem
  POST /cancel  — interrompe fala e limpa filas
  GET  /voices  — lista vozes disponíveis
  GET  /status  — status do servidor

Dependências:
  pip install fastapi uvicorn supertonic numpy miniaudio pydantic

Uso:
  uvicorn tts_api:app --host 0.0.0.0 --port 3004 --log-level info
"""

from __future__ import annotations

import io
import re
import sys
import time
import heapq
import queue
import threading
import datetime
import logging
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import miniaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supertonic import TTS

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_VOICE    = "M1"
DEFAULT_LANG     = "pt"
SAMPLE_RATE      = 48000
NUM_WORKERS      = 3
WORKER_TIMEOUT   = 30.0

# Playback
BUFFER_MS        = 40       # buffer miniaudio — 40ms = mínima latência audível
SKIP_TIMEOUT_MS  = 100      # ms máximos esperando chunk atrasado antes de pular

# Chunking para /stream
MIN_PALAVRAS     = 4        # mínimo de palavras por chunk

AVAILABLE_VOICES = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]

# ════════════════════════════════════════════════════════════════════════════
# LOGGER DUAL (terminal + arquivo)
# ════════════════════════════════════════════════════════════════════════════

BASEFOLDER = Path(__file__).parent
log_dir    = BASEFOLDER / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_path   = log_dir / f"tts_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

class _LogDuplicado:
    def __init__(self, terminal, caminho: Path):
        self.terminal = terminal
        self._f = open(caminho, "w", encoding="utf-8")
    def write(self, msg: str):
        self.terminal.write(msg)
        self._f.write(msg)
    def flush(self):
        self.terminal.flush()
        self._f.flush()
    def isatty(self) -> bool:
        return self.terminal.isatty()

sys.stdout = _LogDuplicado(sys.stdout, log_path)
sys.stderr = _LogDuplicado(sys.stderr, log_path)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ava.tts")

# ════════════════════════════════════════════════════════════════════════════
# NORMALIZAÇÃO DE TEXTO PT-BR (portado do módulo Piper)
# ════════════════════════════════════════════════════════════════════════════

_RE_MD_CODE_BLOCK  = re.compile(r"```[\s\S]*?```")
_RE_MD_CODE_INLIN  = re.compile(r"`([^`\n]+)`")
_RE_MD_BOLD3       = re.compile(r"\*{3}(.+?)\*{3}", re.DOTALL)
_RE_MD_UNDER3      = re.compile(r"_{3}(.+?)_{3}",     re.DOTALL)
_RE_MD_BOLD        = re.compile(r"\*{2}(.+?)\*{2}", re.DOTALL)
_RE_MD_UNDER2      = re.compile(r"_{2}(.+?)_{2}",     re.DOTALL)
_RE_MD_ITALIC_S    = re.compile(r"\*(.+?)\*",       re.DOTALL)
_RE_MD_ITALIC_U    = re.compile(r"_(.+?)_",           re.DOTALL)
_RE_MD_HEADER      = re.compile(r"^\s*#{1,6}\s+",   re.MULTILINE)
_RE_MD_HR          = re.compile(r"^\s*[-*_]{3,}\s*$", re.MULTILINE)
_RE_MD_BLOCKQUOTE  = re.compile(r"^\s*>+\s?",       re.MULTILINE)
_RE_MD_TABLE_SEP   = re.compile(r"^\s*\|[\s\-:|]+\|\s*$", re.MULTILINE)
_RE_MD_TABLE_ROW   = re.compile(r"^\s*\|(.+)\|\s*$", re.MULTILINE)
_RE_MD_IMAGE       = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_RE_MD_LINK        = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_RE_MD_LIST_UL     = re.compile(r"^\s*[-*+]\s+",    re.MULTILINE)
_RE_MD_LIST_OL     = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_RE_MD_STRIKE      = re.compile(r"~~(.+?)~~")
_RE_MD_ESCAPE      = re.compile(r"\\([*_#\[\]()!`>~|])")
_RE_HTML_BR        = re.compile(r"<br\s*/?>", re.IGNORECASE)
_RE_HTML_INLINE    = re.compile(
    r"<(b|strong|em|i|u|s|del|ins|mark|small|sub|sup)>(.+?)</\1>",
    re.IGNORECASE | re.DOTALL,
)
_RE_HTML_TAG       = re.compile(r"<[^>]+>")
_RE_DIMENSAO       = re.compile(r"(\d+)[xX×](\d+)")
_RE_MULTI_EXCL     = re.compile(r"[!]{2,}")
_RE_MULTI_QUEST    = re.compile(r"[?]{2,}")
_RE_ESPACO_MULTI   = re.compile(r" {2,}")
_RE_CHARS_ILEGAIS  = re.compile(r"[<>|\\^~@#]")
_RE_RETICENCIAS    = re.compile(r"\.{2,}")
_RE_LATEX_BLOCK    = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
_RE_LATEX_INLINE   = re.compile(r'\$([^$\n]+?)\$')
_RE_LATEX_CMD_BARE = re.compile(
    r'\\(?:frac|sqrt|sum|prod|int|lim|infty?|alpha|beta|gamma|delta|theta|lambda|mu|'
    r'pi|sigma|omega|sin|cos|tan|log|ln|exp|partial|nabla|forall|exists|in|notin|'
    r'subset|supset|cup|cap|vec|hat|bar|dot|ddot|tilde|overline|underline|'
    r'mathbb|mathbf|mathrm|left|right|begin|end|text|operatorname)\b'
)
_RE_EMOJIS = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF\uFE0F\u200D]+",
    flags=re.UNICODE,
)
_RE_SPLIT_SENTENCA = re.compile(
    r'(?<!\w\.\w)'
    r'(?<![A-Z][a-z]\.)'
    r'(?<![A-Z]\.)'
    r'(?<=\.|\!|\?)'
    r'(?!\.)'
    r'\s+'
)
_RE_SPLIT_SUBPARTE = re.compile(r'(?<=[,;:])\s+')


def _latex_para_fala(expr: str) -> str:
    expr = re.sub(r'\\text\{([^}]+)\}',             r'\1',           expr)
    expr = re.sub(r'\\math\w+\{([^}]+)\}',          r'\1',           expr)
    expr = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1 sobre \2', expr)
    expr = re.sub(r'\\sqrt\{([^}]+)\}',              r'raiz de \1',   expr)
    expr = re.sub(r'_\{([^}]+)\}',                  r'\1',           expr)
    expr = re.sub(r'\^\{([^}]+)\}',                 r'\1',           expr)
    expr = re.sub(r'_(\w)',                           r'\1',           expr)
    expr = re.sub(r'\^(\w)',                          r'\1',           expr)
    expr = expr.replace('\\times',  ' vezes ')
    expr = expr.replace('\\cdot',   ' vezes ')
    expr = expr.replace('\\div',    ' dividido por ')
    expr = expr.replace('\\pm',     ' mais ou menos ')
    expr = expr.replace('\\geq',    ' maior ou igual a ')
    expr = expr.replace('\\leq',    ' menor ou igual a ')
    expr = expr.replace('\\neq',    ' diferente de ')
    expr = expr.replace('\\approx', ' aproximadamente ')
    expr = expr.replace('\\infty',  ' infinito ')
    expr = expr.replace('=',        ' igual a ')
    expr = expr.replace('+',        ' mais ')
    expr = expr.replace('-',        ' menos ')
    expr = re.sub(r'\\[a-zA-Z]+',  '', expr)
    expr = re.sub(r'[{}$]',        '', expr)
    return re.sub(r' {2,}', ' ', expr).strip()


def _tem_bloco_aberto(texto: str) -> bool:
    return texto.count("```") % 2 != 0


def normalizar_texto(texto: str) -> str:
    """Limpa Markdown, LaTeX, HTML e normaliza PT-BR para síntese de voz."""
    if not texto:
        return ""

    # Fecha ``` incompleto antes de qualquer regex (flush no meio de bloco)
    if _tem_bloco_aberto(texto):
        texto += "```"

    texto = _RE_MD_CODE_BLOCK.sub("", texto)
    texto = _RE_MD_CODE_INLIN.sub(r"\1", texto)

    # LaTeX delimitado
    texto = _RE_LATEX_BLOCK.sub(lambda m: _latex_para_fala(m.group(1)), texto)
    texto = _RE_LATEX_INLINE.sub(lambda m: _latex_para_fala(m.group(1)), texto)

    # LaTeX bare (sem $) — converte linha inteira se detectar comando
    def _converter_linha_latex(linha: str) -> str:
        return _latex_para_fala(linha) if _RE_LATEX_CMD_BARE.search(linha) else linha

    texto = "\n".join(_converter_linha_latex(l) for l in texto.split("\n"))

    texto = _RE_MD_ESCAPE.sub(r"\1", texto)
    texto = _RE_HTML_BR.sub(" ", texto)
    texto = _RE_HTML_INLINE.sub(r"\2", texto)
    texto = _RE_HTML_TAG.sub("", texto)
    texto = _RE_MD_HEADER.sub("", texto)
    texto = _RE_MD_HR.sub("", texto)
    texto = _RE_MD_BLOCKQUOTE.sub("", texto)
    texto = _RE_MD_TABLE_SEP.sub("", texto)
    texto = _RE_MD_TABLE_ROW.sub(
        lambda m: "  ".join(c.strip() for c in m.group(1).split("|") if c.strip()),
        texto,
    )
    texto = _RE_MD_IMAGE.sub(r"\1", texto)
    texto = _RE_MD_LINK.sub(r"\1", texto)
    texto = _RE_MD_STRIKE.sub(r"\1", texto)
    texto = _RE_MD_BOLD3.sub(r"\1",    texto)
    texto = _RE_MD_UNDER3.sub(r"\1",   texto)
    texto = _RE_MD_BOLD.sub(r"\1",     texto)
    texto = _RE_MD_UNDER2.sub(r"\1",   texto)
    texto = _RE_MD_ITALIC_S.sub(r"\1", texto)
    texto = _RE_MD_ITALIC_U.sub(r"\1", texto)
    texto = _RE_MD_LIST_UL.sub("", texto)
    texto = _RE_MD_LIST_OL.sub("", texto)
    texto = texto.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    texto = texto.replace("R$",     "reais ")
    texto = texto.replace("US$",    "dolares ")
    texto = texto.replace("\u20ac", "euros ")
    texto = texto.replace("%",      " por cento")
    texto = texto.replace("&",      " e ")
    texto = texto.replace("\u2192", ", ")
    texto = texto.replace("\u2022", ", ")
    texto = texto.replace("\u2026", "...")
    texto = _RE_DIMENSAO.sub(r"\1 por \2", texto)
    texto = _RE_MULTI_EXCL.sub("!", texto)
    texto = _RE_MULTI_QUEST.sub("?", texto)
    texto = _RE_CHARS_ILEGAIS.sub("", texto)
    texto = _RE_EMOJIS.sub("", texto)
    return _RE_ESPACO_MULTI.sub(" ", texto).strip()


def split_chunks(texto: str) -> list[str]:
    """
    Divide texto em chunks para TTS com mínimo delay.
    Respeita abreviações (Dr., Sr., e.g.) e reticências.
    """
    if not texto:
        return []

    # Protege reticências durante split
    texto_prot = _RE_RETICENCIAS.sub("\u2026", texto)
    partes     = _RE_SPLIT_SENTENCA.split(texto_prot)
    resultado  = []

    for parte in partes:
        parte = parte.replace("\u2026", "...").strip()
        if not parte:
            continue
        for sub in _RE_SPLIT_SUBPARTE.split(parte):
            sub = sub.strip()
            if sub:
                resultado.append(sub)

    return resultado or ([texto.strip()] if texto.strip() else [])


# ════════════════════════════════════════════════════════════════════════════
# SENTINELAS
# ════════════════════════════════════════════════════════════════════════════

SENTINEL_AUD = object()   # sinaliza fim de fala no heap de áudio

# ════════════════════════════════════════════════════════════════════════════
# PLAYBACK — heap ordenado + miniaudio 40ms (portado do módulo Piper)
# ════════════════════════════════════════════════════════════════════════════


from dataclasses import dataclass as _dc

@_dc(order=False)
class _Chunk:
    seq:    int
    wav:    object   # np.ndarray | None
    is_end: bool = False

    def __lt__(self, o): return self.seq < o.seq
    def __le__(self, o): return self.seq <= o.seq
    def __gt__(self, o): return self.seq > o.seq
    def __ge__(self, o): return self.seq >= o.seq
    def __eq__(self, o): return self.seq == o.seq

class HeapPlayer:
    def __init__(self):
        self._heap:       list              = []
        self._heap_lock                     = threading.Lock()
        self._cancel_flag                   = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="tts-play"
        )

    def start(self):
        self._thread.start()


    def push(self, seq_idx: int, wav: np.ndarray):
        with self._heap_lock:
            heapq.heappush(self._heap, _Chunk(seq=seq_idx, wav=wav))

    def push_sentinel(self, sentinel_seq: int):
        """sentinel_seq deve ser base + n_chunks, calculado ANTES da síntese."""
        with self._heap_lock:
            heapq.heappush(self._heap, _Chunk(seq=sentinel_seq, wav=None, is_end=True))


    def cancel(self):
        self._cancel_flag.set()
        time.sleep(0.06)  # aguarda o generator perceber (> 1 ciclo de 40ms)
        with self._heap_lock:
            self._heap.clear()
        self._cancel_flag.clear()
        log.info("[PLAY] Cancelado e heap limpo")

    def reset_seq(self):
        pass  # seq é gerenciado pelo AppState; HeapPlayer só ordena

    def _loop(self):
        heap      = self._heap
        heap_lock = self._heap_lock
        cancel    = self._cancel_flag

        def _stream_gen():
            prox_seq    = 0
            audio_buf   = None   # np.ndarray atual sendo reproduzido
            pos         = 0
            ultimo_av   = time.monotonic()

            required_frames = yield b""  # handshake miniaudio

            while True:
                # ── Cancel ──────────────────────────────────────────────
                if cancel.is_set():
                    audio_buf = None
                    pos       = 0
                    prox_seq  = 0
                    required_frames = yield np.zeros(required_frames, dtype=np.int16).tobytes()
                    continue

                output = np.zeros(required_frames, dtype=np.int16)
                out_pos = 0

                while out_pos < required_frames:
                    # Consome buffer atual
                    if audio_buf is not None:
                        disp   = len(audio_buf) - pos
                        copiar = min(disp, required_frames - out_pos)
                        output[out_pos:out_pos + copiar] = audio_buf[pos:pos + copiar]
                        out_pos += copiar
                        pos     += copiar
                        if pos >= len(audio_buf):
                            audio_buf = None
                            pos       = 0
                        continue

                    # Tenta pegar próximo chunk
                    with heap_lock:
                        if not heap:
                            break
                        topo = heap[0]

                                        
                    if topo.seq == prox_seq:
                        with heap_lock:
                            chunk = heapq.heappop(heap)

                        if chunk.is_end:                          # ← era: chunk is SENTINEL_AUD
                            log.info("[PLAY] Fala completa")
                            prox_seq  = 0
                            ultimo_av = time.monotonic()
                            break

                        # Conversão float32→int16 (estava faltando no path de playback!)
                        pcm = (np.clip(chunk.wav, -1.0, 1.0) * 32767).astype(np.int16)
                        audio_buf = pcm   # ← mesmo nome que o loop lê
                        pos       = 0     # ← mesmo nome que o loop lê
                        ultimo_av = time.monotonic()
                        log.info(f"[PLAY] seq={chunk.seq} ({len(pcm)} samples)")
                        prox_seq += 1
                        continue

                    elif topo.seq > prox_seq:
                        elapsed_ms = (time.monotonic() - ultimo_av) * 1000
                        if elapsed_ms > SKIP_TIMEOUT_MS:
                            log.warning(
                                f"[PLAY] seq={prox_seq} atrasado {elapsed_ms:.0f}ms"
                                f" — pula para seq={topo.seq}"
                            )
                            prox_seq  = topo.seq
                            ultimo_av = time.monotonic()
                        break

                    else:
                        # Chunk obsoleto — descarta
                        with heap_lock:
                            if heap and heap[0].seq < prox_seq:
                                heapq.heappop(heap)

                required_frames = yield output.tobytes()

        try:
            with miniaudio.PlaybackDevice(
                output_format   = miniaudio.SampleFormat.SIGNED16,
                nchannels       = 1,
                sample_rate     = SAMPLE_RATE,
                buffersize_msec = BUFFER_MS,
            ) as device:
                gen = _stream_gen()
                next(gen)
                device.start(gen)
                # Mantém a thread (e o `with`) vivos enquanto o daemon rodar
                threading.Event().wait()
        except Exception as e:
            log.error(f"[PLAY] Erro fatal no playback: {e}", exc_info=True)


# ════════════════════════════════════════════════════════════════════════════
# WORKER TTS (Supertonic)
# ════════════════════════════════════════════════════════════════════════════

class TTSWorker(threading.Thread):
    """
    Worker dedicado com instância própria do modelo Supertonic.
    Recebe (task_id, seq_idx, text, voice, lang, speed, event, holder) da task_queue.
    """

    def __init__(self, worker_id: int, task_queue: queue.Queue):
        super().__init__(daemon=True, name=f"tts-worker-{worker_id}")
        self.worker_id  = worker_id
        self._task_q    = task_queue
        self._tts: TTS  = None
        self._vc_cache: dict[str, object] = {}

    def _load(self):
        log.info(f"[W{self.worker_id}] Carregando Supertonic...")
        self._tts = TTS(auto_download=True)

    def _style(self, voice: str) -> object:
        if voice not in self._vc_cache:
            self._vc_cache[voice] = self._tts.get_voice_style(voice_name=voice)
        return self._vc_cache[voice]

    def _synth(self, text: str, voice: str, lang: str, speed: float) -> tuple[np.ndarray, float]:
        style  = self._style(voice)
        kwargs = {"text": text, "voice_style": style, "lang": lang}
        if speed != 1.0:
            kwargs["speed"] = speed

        wav, duration = self._tts.synthesize(**kwargs)

        if hasattr(wav, "cpu"):
            wav = wav.cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32).squeeze()

        if hasattr(duration, "cpu"):
            duration = duration.cpu().numpy()
        dur_arr  = np.asarray(duration).reshape(-1)
        duration = float(dur_arr[0]) if len(dur_arr) > 0 else float(len(wav) / SAMPLE_RATE)

        return wav, duration

    def run(self):
        self._load()

        # Pre-warm: elimina ~300-500ms de lag ORT na 1ª síntese real
        try:
            self._synth(".", DEFAULT_VOICE, DEFAULT_LANG, 1.0)
            log.info(f"[W{self.worker_id}] Pronto (pré-aquecido)")
        except Exception as e:
            log.warning(f"[W{self.worker_id}] Warmup falhou: {e}")

        while True:
            item = self._task_q.get()
            if item is None:
                break

            task_id, seq_idx, text, voice, lang, speed, done_event, holder = item

            try:
                wav, duration = self._synth(text, voice, lang, speed)
                holder["wav"]      = wav
                holder["duration"] = duration
                holder["error"]    = None
                log.info(f"[W{self.worker_id}] seq={seq_idx} ok ({duration:.2f}s) '{text[:50]}'")
            except Exception as e:
                log.error(f"[W{self.worker_id}] seq={seq_idx} erro: {e}")
                holder["error"] = str(e)
            finally:
                done_event.set()
                self._task_q.task_done()


# ════════════════════════════════════════════════════════════════════════════
# POOL TTS
# ════════════════════════════════════════════════════════════════════════════

class TTSPool:
    """Pool de workers Supertonic com fila compartilhada."""

    def __init__(self, num_workers: int = NUM_WORKERS):
        self._q:        queue.Queue  = queue.Queue()
        self._workers:  list[TTSWorker] = []
        self._counter:  int          = 0
        self._lock      = threading.Lock()
        self._n         = num_workers

    def start(self):
        for i in range(self._n):
            w = TTSWorker(i, self._q)
            w.start()
            self._workers.append(w)
        log.info(f"TTSPool iniciado com {self._n} workers")

    def stop(self):
        for _ in self._workers:
            self._q.put(None)

    def synthesize(
        self,
        text:    str,
        voice:   str   = DEFAULT_VOICE,
        lang:    str   = DEFAULT_LANG,
        speed:   float = 1.0,
        seq_idx: int   = 0,
        timeout: float = WORKER_TIMEOUT,
    ) -> tuple[np.ndarray, float]:
        """Bloqueia até o worker retornar. Thread-safe via run_in_executor."""
        with self._lock:
            task_id = self._counter
            self._counter += 1

        done   = threading.Event()
        holder = {"wav": None, "duration": 0.0, "error": None}

        self._q.put((task_id, seq_idx, text, voice, lang, speed, done, holder))

        if not done.wait(timeout=timeout):
            raise TimeoutError(f"TTS timeout após {timeout}s: '{text[:40]}'")
        if holder["error"]:
            raise RuntimeError(holder["error"])

        return holder["wav"], holder["duration"]


# ════════════════════════════════════════════════════════════════════════════
# ESTADO GLOBAL + CANCEL
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class AppState:
    pool:   TTSPool    = field(default=None)
    player: HeapPlayer = field(default=None)

    # Contador de sequência global — compartilhado entre /speak e /stream
    _seq:      int             = field(default=0, init=False)
    _seq_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def next_seq(self) -> int:
        with self._seq_lock:
            v = self._seq
            self._seq += 1
            return v

    def reset_seq(self):
        with self._seq_lock:
            self._seq = 0


state = AppState()


def cancelar_fala():
    """
    Cancela fala em andamento: limpa heap, reseta sequência.
    Thread-safe — pode ser chamado de qualquer contexto.
    """
    state.reset_seq()
    if state.player:
        state.player.cancel()
    log.info("[CANCEL] Fala cancelada")


# ════════════════════════════════════════════════════════════════════════════
# MODELOS PYDANTIC
# ════════════════════════════════════════════════════════════════════════════

class SpeakRequest(BaseModel):
    text:  str
    voice: str   = DEFAULT_VOICE
    lang:  str   = DEFAULT_LANG
    speed: float = 1.0

class SpeakResponse(BaseModel):
    duration_s: float
    latency_ms: float
    text:       str
    chunks:     int

class StreamRequest(BaseModel):
    text:  str
    voice: str = DEFAULT_VOICE
    lang:  str = DEFAULT_LANG

class StreamResponse(BaseModel):
    sentences:        int
    total_duration_s: float
    latency_ms:       float

class VoicesResponse(BaseModel):
    voices:  list[str]
    default: str


# ════════════════════════════════════════════════════════════════════════════
# LIFESPAN
# ════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Iniciando AVA TTS API...")

    state.player = HeapPlayer()
    state.player.start()

    state.pool = TTSPool(num_workers=NUM_WORKERS)
    state.pool.start()

    # Workers já fazem warmup internamente — aguardamos apenas o pool estar pronto
    log.info("TTS API pronta")

    yield

    state.pool.stop()
    log.info("AVA TTS API encerrada")


# ════════════════════════════════════════════════════════════════════════════
# APP
# ════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="AVA TTS API", lifespan=lifespan)


# ════════════════════════════════════════════════════════════════════════════
# POST /speak
# Sintetiza o texto, normaliza, divide em chunks se necessário e
# enfileira no heap em ordem correta. Resposta retorna imediatamente
# (não bloqueia até o áudio terminar de tocar).
# ════════════════════════════════════════════════════════════════════════════

@app.post("/speak", response_model=SpeakResponse)
async def speak(req: SpeakRequest):
    texto = normalizar_texto(req.text.strip())
    if not texto:
        raise HTTPException(status_code=400, detail="texto vazio após normalização")
    if req.voice not in AVAILABLE_VOICES:
        raise HTTPException(status_code=400, detail=f"voz inválida — use: {AVAILABLE_VOICES}")

    chunks = split_chunks(texto)
    if not chunks:
        chunks = [texto]

    t0   = time.perf_counter()
    loop = asyncio.get_event_loop()

    total_dur = 0.0
    # Atribui seq_idx antes de disparar para garantir ordem mesmo em paralelo
    seq_base = state.next_seq() if len(chunks) == 1 else None

    async def _synth_and_push(seq_idx: int, chunk: str):
        nonlocal total_dur
        try:
            wav, dur = await loop.run_in_executor(
                None,
                lambda: state.pool.synthesize(chunk, req.voice, req.lang, req.speed, seq_idx)
            )
            state.player.push(seq_idx, wav)
            total_dur += dur
        except TimeoutError:
            log.error(f"[SPEAK] Timeout seq={seq_idx}")
        except Exception as e:
            log.error(f"[SPEAK] Erro seq={seq_idx}: {e}")

    if len(chunks) == 1:
        sentinel_seq = seq_base + 1
        await _synth_and_push(seq_base, chunks[0])
    else:
        with state._seq_lock:
            base = state._seq
            state._seq += len(chunks)
        sentinel_seq = base + len(chunks)   # ← fixo, antes de qualquer síntese

        tasks = [_synth_and_push(base + i, c) for i, c in enumerate(chunks)]
        await asyncio.gather(*tasks)

    state.player.push_sentinel(sentinel_seq)   # ← agora com seq correto

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    log.info(f"[SPEAK] {len(chunks)} chunk(s) enfileirados em {latency_ms}ms")

    return SpeakResponse(
        duration_s = round(total_dur, 3),
        latency_ms = latency_ms,
        text       = texto,
        chunks     = len(chunks),
    )


# ════════════════════════════════════════════════════════════════════════════
# POST /stream
# Mesmo pipeline do /speak mas otimizado para textos longos:
# sintetiza sentenças em paralelo e as enfileira no heap assim que
# cada uma fica pronta — sem esperar todas terminarem.
# Sentença 0 começa a tocar assim que sintetizada, não importa se
# sentença 1 ainda está sendo processada.
# ════════════════════════════════════════════════════════════════════════════

@app.post("/stream", response_model=StreamResponse)
async def stream(req: StreamRequest):
    texto = normalizar_texto(req.text.strip())
    if not texto:
        raise HTTPException(status_code=400, detail="texto vazio após normalização")
    if req.voice not in AVAILABLE_VOICES:
        raise HTTPException(status_code=400, detail=f"voz inválida — use: {AVAILABLE_VOICES}")

    sentencas = split_chunks(texto)
    if not sentencas:
        raise HTTPException(status_code=400, detail="nenhum chunk encontrado")

    t0   = time.perf_counter()
    loop = asyncio.get_event_loop()
    n    = len(sentencas)

    # Reserva bloco contíguo de seq_idx — garante ordem global no heap
    with state._seq_lock:
        base       = state._seq
        state._seq += n

    sentinel_seq = base + n 

    durations: list[float]       = [0.0] * n
    events:    list[asyncio.Event] = [asyncio.Event() for _ in range(n)]

    async def _synth_and_signal(i: int, sentenca: str):
        try:
            wav, dur = await loop.run_in_executor(
                None,
                lambda: state.pool.synthesize(sentenca, req.voice, req.lang, seq_idx=base + i)
            )
            durations[i] = dur
            state.player.push(base + i, wav)
        except Exception as e:
            log.error(f"[STREAM] Erro seq={base + i}: {e}")
        finally:
            events[i].set()   # libera o enqueuer ordenado

    # Enqueuer: aguarda slot i estar pronto antes de avançar
    # (o push já acontece dentro de _synth_and_signal, mas o sentinel
    #  só é empurrado depois que todos os slots confirmaram)
    async def _aguardar_todos():
        for i in range(n):
            await events[i].wait()
        state.player.push_sentinel(sentinel_seq) 

    synth_tasks   = [asyncio.create_task(_synth_and_signal(i, s)) for i, s in enumerate(sentencas)]
    sentinel_task = asyncio.create_task(_aguardar_todos())

    try:
        await asyncio.gather(*synth_tasks, sentinel_task)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    log.info(f"[STREAM] {n} sentenças enfileiradas em {latency_ms}ms")

    return StreamResponse(
        sentences        = n,
        total_duration_s = round(sum(durations), 3),
        latency_ms       = latency_ms,
    )


# ════════════════════════════════════════════════════════════════════════════
# POST /cancel
# ════════════════════════════════════════════════════════════════════════════

@app.post("/cancel")
async def cancel():
    cancelar_fala()
    return {"cancelled": True}


# ════════════════════════════════════════════════════════════════════════════
# GET /voices  |  GET /status
# ════════════════════════════════════════════════════════════════════════════

@app.get("/voices", response_model=VoicesResponse)
async def voices():
    return VoicesResponse(voices=AVAILABLE_VOICES, default=DEFAULT_VOICE)


@app.get("/status")
async def status():
    return {
        "workers":       NUM_WORKERS,
        "default_voice": DEFAULT_VOICE,
        "sample_rate":   SAMPLE_RATE,
        "buffer_ms":     BUFFER_MS,
        "skip_timeout_ms": SKIP_TIMEOUT_MS,
        "voices":        AVAILABLE_VOICES,
    }


# ════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS (linha de comando / testes)
# ════════════════════════════════════════════════════════════════════════════

def tts_to_bytes(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """Sintetiza para WAV em memória — útil para testes unitários."""
    import struct

    text = normalizar_texto(text)
    pool = TTSPool(num_workers=1)
    pool.start()

    try:
        wav, _ = pool.synthesize(text, voice=voice)
    finally:
        pool.stop()

    pcm         = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
    buf         = io.BytesIO()
    data_size   = pcm.nbytes
    byte_rate   = SAMPLE_RATE * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, byte_rate, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())
    return buf.getvalue()


# ════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3004, log_level="info")