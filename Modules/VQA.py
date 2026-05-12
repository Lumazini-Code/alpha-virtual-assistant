import sys
import os
from pathlib import Path
import datetime
import socket
import time
import json
import requests
import base64
import subprocess

LLAMA_SERVER_PATH = r".\llama-cpp\llama-server"
LLAMA_ARGS = [
    "--model",        r"Models\Qwen3VL-2B-Instruct-Q4_K_M.gguf",
    "--mmproj",       r"Models\mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
    "--host",         "127.0.0.2",
    "--port",         "1",
    "--n-gpu-layers", "all",
    "--threads",      "4",
    "--threads-batch","6",
    "--batch-size",   "2048",
    "--ubatch-size",  "512",
    "--flash-attn",   "on",
    "--cache-type-k", "q4_0",
    "--cache-type-v", "q8_0",
    "--ctx-size",     "16384",
    "--parallel",     "1",
    "--cont-batching",
    "--mmap",
    "--mlock",
    "--poll",         "50",
    "--prio",         "2",
    "--mmproj-offload",
    "--cache-reuse",  "256",
    "--slot-prompt-similarity", "0.1",
]

def start_llama_server() -> subprocess.Popen:
    print("[SERVER] Iniciando llama-server...")
    proc = subprocess.Popen(
        [LLAMA_SERVER_PATH] + LLAMA_ARGS,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW  # sem janela no Windows
    )
    return proc

def wait_for_server(url: str, timeout: int = 60):
    """Aguarda o servidor ficar disponível."""
    print("[SERVER] Aguardando servidor ficar pronto...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            res = requests.get(f"{url}/health", timeout=2)
            if res.status_code == 200:
                print("[SERVER] Servidor pronto!\n")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    raise TimeoutError("[SERVER] Servidor não respondeu a tempo.")

def stop_llama_server(proc: subprocess.Popen):
    print("[SERVER] Encerrando llama-server...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


BASEFOLDER = Path(__file__).parent.parent
log_dir = BASEFOLDER / "logs"
log_dir.mkdir(exist_ok=True)
log_filename = f"Florence_log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
log_path = os.path.join(log_dir, log_filename)


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

QWEN_URL   = "http://127.0.0.2:1/v1/chat/completions"
QWEN_MODEL = "Qwen3VL-2B-Instruct-Q4_K_M"

# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def describe_image_stream(img_path: str, prompt) -> str:
    img_b64 = encode_image(img_path)
    ext = Path(img_path).suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{img_b64}"}
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 512,
        "stream": True
    }

    full_response = ""

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
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    full_response += delta
                    print(delta, end="", flush=True)
                    # envia cada chunk para o commander via socket
                    conn.sendall(delta.encode("utf-8"))
            except (json.JSONDecodeError, KeyError):
                continue

    print()
    return full_response

# ── Socket ────────────────────────────────────────────────────────────────────

def connect():
    HOST = '127.0.0.1'
    PORT = 2
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            client_socket.connect((HOST, PORT))
            print(f"[SOCKET] Conectado em {HOST}:{PORT}")
            return client_socket
        except Exception as e:
            print(f"[SOCKET] Erro ao conectar: {e}, tentando novamente...")
            time.sleep(1)

# ── Warmup ────────────────────────────────────────────────────────────────────

def warmup():
    """Faz uma requisição dummy para aquecer o servidor llama.cpp."""
    print("[MAIN] Aquecendo Qwen3VL...")
    try:
        payload = {
            "model": QWEN_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "ok"}]
                }
            ],
            "max_tokens": 1
        }
        requests.post(QWEN_URL, json=payload, timeout=30)
        print("[MAIN] Pronto!\n")
    except Exception as e:
        print(f"[MAIN] Warmup falhou: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────

llama_proc = start_llama_server()
wait_for_server("http://127.0.0.2:8080")

conn = connect()
warmup()



if __name__ == "__main__":
    while True:
        try:
            print("[MAIN] Aguardando mensagem...")
            user_input = ""
            while True:
                user_input = conn.recv(4096).decode("utf-8").strip()
                if user_input:
                    break

            print(f"[MAIN] Recebido: {user_input}")

            try:
                imgPathRaw, _ = user_input.split("</img>")
                imgPath = imgPathRaw.replace("<img>", "").strip()
            except ValueError:
                print("[MAIN] Formato inválido. Esperado: <img>caminho</img>pergunta")
                continue

            if not Path(imgPath).exists():
                print(f"[MAIN] Imagem não encontrada: {imgPath}")
                conn.sendall(json.dumps({"result": "Imagem não encontrada."}).encode("utf-8"))
                conn.sendall(b"{end}")
                continue

            try:
                t0 = time.perf_counter()
                result = describe_image_stream(imgPath)
                print(f"[MAIN] Concluído em {time.perf_counter()-t0:.3f}s")
                print(result)

                conn.sendall(json.dumps({"result": result}).encode("utf-8"))
                conn.sendall(b"{end}")
                print("[MAIN] Resultado enviado")

            except Exception as e:
                print(f"[MAIN] Erro na inferência: {e}")
                conn.sendall(json.dumps({"result": f"Erro: {e}"}).encode("utf-8"))
                conn.sendall(b"{end}")
        except Exception as e:
            print(f"[MAIN] Erro geral: {e}")
            stop_llama_server(llama_proc)
            break
            