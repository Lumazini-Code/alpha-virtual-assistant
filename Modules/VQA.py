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
import time
import uvicorn
import httpx
qwen_adress = ""
# ── Detecta suporte a visão de forma síncrona ─────────────────────────────────
def is_vision_supported(base_url: str = "http://localhost:2001") -> bool:
    try:
        r = httpx.get(f"{base_url}/props", timeout=5)
        data = r.json()
        return data.get("modalities", {}).get("vision", False)
    except Exception:
        return False

time.sleep(10)
is_vision = is_vision_supported()

if is_vision:
    # Modelo principal (porta 2001) suporta visão
    config_path = Path(__file__).parent.parent / "resource" / "Aiconfig.dll"
    QWEN_MODEL  = config_path.read_text().strip()
    qwen_adress = "localhost:2001"
else:
    # Fallback para modelo VL dedicado na porta 2004
    qwen_adress = "localhost:2004"
    QWEN_MODEL  = "Qwen3VL-2B-Instruct-Q4_K_M"

qwen_url = f"http://{qwen_adress}/v1/chat/completions"
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
            res = requests.get(f"http://{url}/health", timeout=2)
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
    try:
        wait_for_server(qwen_adress, timeout=30)
        _warmup()
    except (TimeoutError, RuntimeError) as e:
        print(f"[WARN] VQA iniciou sem servidor disponível: {e}")
    yield


app = FastAPI(title="Florence / Qwen3VL", lifespan=lifespan)


def _warmup():
    print("[MAIN] Aquecendo Qwen3VL...")
    try:
        payload = {
            "model": QWEN_MODEL,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "ok"}]}],
            "max_tokens": 1,
        }
        requests.post(qwen_url, json=payload, timeout=30)
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
        res = requests.post(qwen_url, json=payload, timeout=60)
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
        with requests.post(qwen_url, json=payload, stream=True, timeout=60) as res:
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