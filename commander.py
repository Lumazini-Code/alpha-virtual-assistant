import socket
import os
import signal
import sys
import json
import datetime
import threading
import unicodedata
import requests
from pathlib import Path
from langdetect import detect

BASEFOLDER = Path(__file__).parent.parent
log_dir = BASEFOLDER / "logs"
log_dir.mkdir(exist_ok=True)
log_filename = f"commander_log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
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

    def close(self):
        self.log.close()


sys.stdout = LogDuplicado(sys.stdout, log_path)

with open(Path(__file__).parent / "resource" / "agentCfg.json", "r", encoding="utf-8") as agentFile:
    agentCfg = json.load(agentFile)

MODULE_PORTS = {
    "csh": 9999,
    "llm": 5,
    "img": 2,
    "translate": 3,
    "sum": 6,
    "tts": 10,
    "search": 1,
}

MODULES = {
    1: "image_processing",
    2: "main_llm",
    3: "summarizer",
}

ROUTER_URL = "http://127.0.0.1:7070/classify"

ROUTE_LABELS = {
    "processar imagem":   1,
    "responder pergunta": 2,
    "explicar conteúdo":  2,
    "resumir texto":      3,
}

# ── Router ────────────────────────────────────────────────────────────────────

def rotear(pergunta: str, tem_imagem: bool = False) -> list[int]:
    try:
        response = requests.post(
            ROUTER_URL,
            json={"text": pergunta, "labels": list(ROUTE_LABELS.keys())},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()

        best_label = data["best"]["label"]
        best_score = data["best"]["score"]
        module_id  = ROUTE_LABELS[best_label]

        print(f"[ROUTER] '{best_label}' (score={best_score:.4f}) -> módulo {module_id}")

        sequencia = []
        if tem_imagem:
            sequencia.append(1)
        if module_id != 1:
            sequencia.append(module_id)

        return sequencia if sequencia else [2]

    except requests.exceptions.ConnectionError:
        print("[WARN] Router API offline, defaulting to [2]")
        return [2]
    except Exception as e:
        print(f"[WARN] Router error: {e}, defaulting to [2]")
        return [2]

# ── Socket helpers ────────────────────────────────────────────────────────────

def start_server(port, name=""):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', port))
    srv.listen(1)
    print(f"[INFO] Waiting for {name or 'client'} on port {port}...")
    conn, addr = srv.accept()
    print(f"[INFO] {name or 'Client'} connected: {addr}")
    srv.close()
    return conn


def accept_connections():
    threads = []
    holders = {}

    def accept(name, port):
        holders[name] = start_server(port, name)

    required = [("csh", MODULE_PORTS["csh"]), ("llm", MODULE_PORTS["llm"])]
    conditional = []
    if agentCfg.get("Img") == "on":
        conditional.append(("img", MODULE_PORTS["img"]))
    if agentCfg.get("Traductor") == "on":
        conditional.append(("translate", MODULE_PORTS["translate"]))
    if agentCfg.get("Sum") == "on":
        conditional.append(("sum", MODULE_PORTS["sum"]))
    if agentCfg.get("TTS") == "on":
        conditional.append(("tts", MODULE_PORTS["tts"]))

    for name, port in required + conditional:
        t = threading.Thread(target=accept, args=(name, port))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return holders

# ── TTS helpers ───────────────────────────────────────────────────────────────

def eh_enviavel_tts(texto: str) -> bool:
    return any(unicodedata.category(c).startswith(('L', 'N')) for c in texto)

def send_tts(conn_tts, text):
    encoded = text.encode("utf-8")
    conn_tts.sendall(len(encoded).to_bytes(4, byteorder='big'))
    conn_tts.sendall(encoded)

def send_tts_end(conn_tts):
    conn_tts.sendall((0).to_bytes(4, byteorder='big'))

# ── Translation ───────────────────────────────────────────────────────────────

def translate(conn_translate, text, source_lang, target_lang):
    conn_translate.sendall(f"<//{source_lang}-{target_lang}//>{text}".encode("utf-8"))
    while True:
        data = conn_translate.recv(4096).decode("utf-8").strip()
        if data:
            break
    for lang_token in ["por_Latn", "eng_Latn", "spa_Latn", "fra_Latn", "deu_Latn"]:
        data = data.replace(lang_token, "")
    return data

# ── Accept connections ────────────────────────────────────────────────────────

holders      = accept_connections()
conn_csh     = holders["csh"]
conn_llm     = holders["llm"]
conn_img     = holders.get("img")
conn_translate = holders.get("translate")
conn_sum     = holders.get("sum")
conn_tts     = holders.get("tts")

# ── Module handlers ───────────────────────────────────────────────────────────

def handle_image(pergunta, imgPath, sequencia):
    if not imgPath or conn_img is None:
        return ""

    conn_img.sendall(f"<img>{imgPath}</img>".encode("utf-8"))
    chunks = []
    while True:
        chunk = conn_img.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        if b"{end}" in chunk:
            break

    raw = b"".join(chunks).replace(b"{end}", b"").decode("utf-8").strip()
    if not raw:
        return ""

    try:
        result = json.loads(raw)
        first_key = next(iter(result))
        imgDesc = str(result[first_key])
        print(f"Resposta do módulo de imagem: {imgDesc}")

        if 2 not in sequencia and 3 not in sequencia:
            conn_csh.sendall(imgDesc.encode("utf-8"))
            conn_csh.sendall("{end}".encode("utf-8"))

        return imgDesc
    except (json.JSONDecodeError, StopIteration) as e:
        print(f"[WARN] Failed to parse image module response: {e}")
        return ""


def handle_llm(pergunta, imgDesc, voice):
    if imgDesc:
        pergunta = f"{pergunta}\n\n Image description: {imgDesc}"

    conn_llm.sendall(pergunta.encode("utf-8"))
    print("[LLM] Recebendo resposta: ", end="", flush=True)

    response = ""
    while True:
        chunk = conn_llm.recv(4096).decode("utf-8")
        if not chunk:
            continue
        if chunk == "{end}":
            break
        if chunk == "<error>":
            print("\n[LLM] Error in generation")
            break

        print(chunk, end="", flush=True)
        response += chunk
        conn_csh.sendall(chunk.encode("utf-8"))

        if voice and conn_tts is not None:
            send_tts(conn_tts, chunk)

    if voice and conn_tts is not None:
        send_tts_end(conn_tts)

    print()
    conn_csh.sendall("{end}".encode("utf-8"))
    return response


def handle_summarize(pergunta, askLang):
    if askLang != "en" and agentCfg.get("Traductor") == "on" and conn_translate is not None:
        print("traduzindo frase para inglês...")
        pergunta = translate(conn_translate, pergunta, askLang, "en")
        print(f"frase traduzida: {pergunta}")

    if conn_sum is None:
        print("[WARN] Summarizer not connected, falling back to LLM")
        return None

    conn_sum.sendall(pergunta.encode("utf-8"))
    while True:
        data = conn_sum.recv(4096).decode("utf-8").strip()
        if data:
            break

    print(f"Resposta do módulo de sumarização: {data}")
    conn_csh.sendall(data.encode("utf-8"))
    conn_csh.sendall("{end}".encode("utf-8"))
    return data

# ── Graceful shutdown ─────────────────────────────────────────────────────────

def shutdown(signum=None, frame=None):
    print("\n[INFO] Shutting down commander...")
    for name, conn in [("csh", conn_csh), ("llm", conn_llm),
                        ("img", conn_img), ("translate", conn_translate),
                        ("sum", conn_sum), ("tts", conn_tts)]:
        if conn:
            try:
                conn.close()
            except OSError:
                pass
    if isinstance(sys.stdout, LogDuplicado):
        sys.stdout.close()
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# ── Main loop ─────────────────────────────────────────────────────────────────

while True:
    try:
        pergunta = conn_csh.recv(4096).decode("utf-8").strip()
        if not pergunta:
            print("[WARN] C# client disconnected")
            break
    except (ConnectionResetError, OSError) as e:
        print(f"[ERROR] Connection lost: {e}")
        break

    askLang = detect(pergunta)
    voice   = False
    img     = False
    imgPath = ""
    imgDesc = ""

    if "<voice>" in pergunta:
        pergunta = pergunta.replace("<voice>", "")
        voice = True

    if "<img>" in pergunta:
        modPergunta = pergunta.replace("<img>", "")
        imgPath, ask = modPergunta.split("</img>")
        img = True
        pergunta = ask.strip()

    originalAsk = pergunta

    if askLang != "en" and agentCfg.get("Traductor") == "on" and conn_translate is not None:
        print("traduzindo frase para inglês...")
        pergunta = translate(conn_translate, pergunta, askLang, "en")
        print(f"frase traduzida: {pergunta}")

    sequencia = rotear(pergunta, tem_imagem=img)
    print([MODULES.get(m, f"unknown({m})") for m in sequencia])

    for m in sequencia:
        if m == 1:
            imgDesc = handle_image(pergunta, imgPath, sequencia)
        elif m == 2:
            pergunta = handle_llm(pergunta, imgDesc, voice)
            imgDesc = ""
        elif m == 3:
            result = handle_summarize(pergunta, askLang)
            if result is None:
                pergunta = handle_llm(pergunta, imgDesc, voice)
            else:
                pergunta = result
            imgDesc = ""