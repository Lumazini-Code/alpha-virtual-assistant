import sys
import os
from pathlib import Path
import datetime
import time
import json
import requests
import base64
import subprocess
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────────

LLAMA_SERVER_PATH = r".//llama-cpp//llama-server"
LLAMA_ARGS = [
    "--model",        r"Models/Qwen3VL-2B-Instruct-Q4_K_M.gguf",   # ← barras corrigidas p/ Linux
    "--mmproj",       r"Models/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
    "--host",         "0.0.0.0",
    "--port",         "2004",
    "--n-gpu-layers", "all",
    "--threads",      "4",
    "--threads-batch","6",
    "--batch-size",   "2048",
    "--ubatch-size",  "512",
    "--flash-attn",   "on",
    "--cache-type-k", "q4_0",
    "--cache-type-v", "q8_0",
    "--ctx-size",     "8192",
    "--parallel",     "1",
    "--cont-batching",
    "--mmap",
    "--poll",         "50",
    "--prio",         "2",
    "--mmproj-offload",
    "--cache-reuse",  "256",
    "--slot-prompt-similarity", "0.1",
]

QWEN_URL = "http://0.0.0.0:2004/v1/chat/completions"  # ← corrigido # ← 127.0.0.1 em vez de 0.0.0.0
QWEN_MODEL = "Qwen3VL-2B-Instruct-Q4_K_M"

# ── Logging ───────────────────────────────────────────────────────────────────

BASEFOLDER = Path(__file__).parent.parent
log_dir    = BASEFOLDER / "logs"
log_dir.mkdir(exist_ok=True)
log_path   = log_dir / f"Florence_log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"


class LogDuplicado:
    def __init__(self, terminal, caminho_log):
        self.terminal = terminal
        self.log = open(caminho_log, "w", encoding="utf-8")

    def write(self, mensagem):
        self.terminal.write(mensagem)
        self.log.write(mensagem)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()


sys.stdout = LogDuplicado(sys.__stdout__, log_path)
sys.stderr = LogDuplicado(sys.__stderr__, log_path)

# ── Llama server ──────────────────────────────────────────────────────────────

llama_proc: subprocess.Popen | None = None
llama_log_path = log_dir / f"llama_server_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"



def wait_for_server(url: str, timeout: int = 120):   # ← timeout aumentado para 120s
    print(f"[SERVER] Aguardando servidor em {url}/health ...")
    start = time.time()
    while time.time() - start < timeout:
        # Verifica se o processo ainda está vivo
        if llama_proc and llama_proc.poll() is not None:
            raise RuntimeError(
                f"llama-server morreu durante inicialização (código {llama_proc.returncode}). "
                f"Veja: {llama_log_path}"
            )
        try:
            res = requests.get(f"{url}/health", timeout=2)
            if res.status_code == 200:
                print("[SERVER] ✅ Servidor pronto!\n")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    raise TimeoutError(
        f"[SERVER] Servidor não respondeu em {timeout}s. Veja o log: {llama_log_path}"
    )


def stop_llama_server(proc: subprocess.Popen):
    print("[SERVER] Encerrando llama-server...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_mime(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    return "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

# ── Lifespan (substitui on_event depreciado) ──────────────────────────────────

class DescribeRequest(BaseModel):
    img_path: str
    prompt: str = "Descreva a imagem detalhadamente."



@asynccontextmanager
async def lifespan(app: FastAPI):
    global llama_proc
    wait_for_server("http://0.0.0.0:2004")
    _warmup()
    yield
    if llama_proc:
        stop_llama_server(llama_proc)


app = FastAPI(title="Florence / Qwen3VL", lifespan=lifespan)


def _warmup():
    print("[MAIN] Aquecendo Qwen3VL...")
    try:
        payload = {
            "model": QWEN_MODEL,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "ok"}]}],
            "max_tokens": 1,
        }
        requests.post(QWEN_URL, json=payload, timeout=30)
        print("[MAIN] ✅ Pronto!\n")
    except Exception as e:
        print(f"[MAIN] Warmup falhou: {e}")

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/describe")
def describe(req: DescribeRequest):
    if not Path(req.img_path).exists():
        raise HTTPException(status_code=404, detail=f"Imagem não encontrada: {req.img_path}")

    img_b64 = encode_image(req.img_path)
    mime    = get_mime(req.img_path)

    payload = {
        "model": QWEN_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                {"type": "text",      "text": req.prompt},
            ],
        }],
        "stream": False,
    }

    try:
        t0  = time.perf_counter()
        res = requests.post(QWEN_URL, json=payload, timeout=60)
        res.raise_for_status()
        result = res.json()["choices"][0]["message"]["content"]
        print(f"[MAIN] Concluído em {time.perf_counter()-t0:.3f}s")
        return {"result": result}
    except Exception as e:
        import traceback
        print(f"[ERRO] {e}")
        print(traceback.format_exc())
        # Mostra também o que o llama-server respondeu, se houver
        try:
            print(f"[ERRO] Resposta do llama-server: {res.text}")
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/describe/stream")
def describe_stream(req: DescribeRequest):
    if not Path(req.img_path).exists():
        raise HTTPException(status_code=404, detail=f"Imagem não encontrada: {req.img_path}")

    img_b64 = encode_image(req.img_path)
    mime    = get_mime(req.img_path)

    payload = {
        "model": QWEN_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                {"type": "text",      "text": req.prompt},
            ],
        }],
        "stream": True,
    }

    def generator():
        with requests.post(QWEN_URL, json=payload, stream=True, timeout=60) as res:
            res.raise_for_status()
            for line in res.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        print(delta, end="", flush=True)
                        yield f"data: {json.dumps({'delta': delta})}\n\n"
                except (json.JSONDecodeError, KeyError):
                    continue

    return StreamingResponse(generator(), media_type="text/event-stream")




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4002)